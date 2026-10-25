"""EnclosureSpec: the declarative description of a printable enclosure (pure, no FreeCAD).

This module is the single source of truth for what the kit builds and checks.
Everything here is plain Python on dicts and dataclasses so the agent thread,
the tools, the tests and the headless rebuild can all use it:

* ``EnclosureSpec`` + nested ``StandoffSpec``, ``LidSpec``, ``HookSpec``,
  ``VentSpec``, ``PortSpec`` mirror the enclosure-spec JSON key for key;
  ``default_spec``, ``from_dict``, ``to_dict``, ``merge`` convert and combine them;
* ``validate`` applies every rule of the validation table (errors fail, warnings do not)
  and reports ``"<path>: <problem>; <what to do>"`` strings the model can act on
  (the rules live in the private module ``_validate``);
* ``spec_hash`` fingerprints a stage's OWN fields plus the hashes of the stages
  it depends on (``STAGE_FIELDS`` / ``STAGE_DEPS``), so an edit reaches exactly
  the stages that really depend on it;
* ``derived`` evaluates every derived parameter of the ``Params`` sheet with
  the SAME formulas the sheet holds (``DERIVED_FORMULAS``), so Python and
  FreeCAD cannot disagree; ``input_params`` / ``param_rows`` give the rows;
* kit-internal helpers shared by the checks and the geometry:
  ``active_connectors``, ``port_openings``, ``wall_openings``, ``hook_positions``,
  ``hook_strain``, ``catch_pocket_depth``, ``vent_area``.

The board is a dict
of the ``refparts/boards/*.json`` schema; when none is passed it is loaded
through ``refparts.catalog.load_board`` (guarded import: before that module
exists, board-dependent rules are skipped with one warning).

Note: ``kit.geometry.hook_positions`` and ``kit.build.STAGES`` are the
contract names; they re-export ``hook_positions`` and ``STAGES`` from here (the
validator needs the placement rule and must stay importable without FreeCAD).
"""

import ast
import copy
import dataclasses
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import materials as _materials

SCHEMA_VERSION = 1

FACES = ("-x", "+x", "-y", "+y", "top", "bottom")
WALL_FACES = ("-x", "+x", "-y", "+y")
STAGES = ("params", "reference", "shell", "standoffs", "ports", "lid", "vents", "hooks", "assembly")
PARTS = ("shell", "lid")
LID_STYLES = ("snap_fit", "screw", "friction")
SHRINK_MODES = ("none", "clearance", "scale")
VENT_PATTERNS = ("slots", "holes")
SHELL_ORIENTATIONS = ("floor_down", "as_modeled")
LID_ORIENTATIONS = ("top_down", "as_modeled")

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,23}$")
KIT_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")

POCKET_SEAT = 0.1       # mm of extra catch-pocket depth beyond the engaged barb
HOOK_WIDEN = 1.0        # mm kept clear beside a wall opening, plus hooks.w / 2
MIN_HOOK_GAP = 1.0      # mm of clear space kept between two neighbouring hooks
STRAIN_WARN_FRACTION = 0.8
MAX_HOOKS = 64          # hook_positions refuses larger counts (the validator allows 2..8)
TOL = 1e-9             # mm: tolerance for 'at least the minimum' comparisons (3.3 - 2.1 is 1.1999999999999997 in floats)

# Private copy of refparts.catalog.PLUG_DEFAULTS: (plug width, plug height) of the CABLE PLUG per
# connector kind. The catalog's table wins once refparts exists (see _plug_defaults).
_PLUG_DEFAULTS = {"usb_c": (12.0, 6.5), "micro_hdmi": (11.0, 7.0), "audio_jack": (9.0, 9.0)}


def _n(x: float) -> str:
    """A millimetre quantity for a message: at most 6 significant digits, at least one decimal."""
    s = f"{x:.6g}"
    return s if ("." in s or "e" in s or "n" in s) else s + ".0"


# --------------------------------------------------------------------------------------------------------------------
# the dataclasses (field order = key order of the spec JSON)
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class StandoffSpec:
    """Board support posts: ``h`` height under the PCB, ``d`` ring diameter, ``pilot_d`` screw pilot hole."""

    h: float = 4.0
    d: float = 6.0
    pilot_d: float = 2.2


@dataclass
class LidSpec:
    """Cap lid: ``style`` snap_fit|screw|friction, ``gap`` above the tallest part, skirt ``lip_h`` x ``lip_t``, ``fit`` clearance."""

    style: str = "snap_fit"
    gap: float = 1.0
    lip_h: float = 3.0
    lip_t: float = 1.2
    fit: float = 0.25


@dataclass
class HookSpec:
    """Snap-fit hook beams; ``count`` is the total over the two long walls."""

    count: int = 4
    w: float = 6.0
    t: float = 1.2
    overhang: float = 0.8
    length: float = 12.0
    barb_h: float = 1.0
    ramp_deg: float = 30
    retention_deg: float = 45
    window_margin: float = 0.3


@dataclass
class VentSpec:
    """One vent array (``add_vents`` writes here): ``count`` slots ``slot_w`` x ``slot_l`` at ``pitch``, optional ``center`` [u, v]."""

    enabled: bool = True
    part: str = "lid"
    face: str = "top"
    pattern: str = "slots"
    count: int = 9
    slot_w: float = 2.0
    slot_l: float = 24.0
    pitch: float = 4.0
    center: "list[float] | None" = None
    angle_deg: float = 0


@dataclass
class PortSpec:
    """Connector openings: ``auto`` opens every ``cutout`` connector, ``include`` / ``exclude`` adjust, ``margin`` / ``gap`` / plug sizing."""

    auto: bool = True
    include: list = field(default_factory=list)
    exclude: list = field(default_factory=list)
    margin: float = 0.4
    gap: float = 0.3
    plug_clearance: bool = True
    overrides: list = field(default_factory=list)


@dataclass
class EnclosureSpec:
    """The whole specification; ``to_dict()`` equals ``default_spec('rpi4b', 'ABS')`` JSON for the defaults."""

    version: int = 1
    kit_id: str = "enc1"
    name: str = "PiCase"
    board: str = "rpi4b"
    material: str = "ABS"
    wall_t: float = 2.0
    floor_t: float = 2.0
    lid_t: float = 2.0
    clearance: float = 0.3
    shrink_comp: str = "clearance"
    corner_r: float = 3.0
    standoff: StandoffSpec = field(default_factory=StandoffSpec)
    lid: LidSpec = field(default_factory=LidSpec)
    hooks: HookSpec = field(default_factory=HookSpec)
    vents: VentSpec = field(default_factory=VentSpec)
    ports: PortSpec = field(default_factory=PortSpec)
    cutouts: list = field(default_factory=list)
    lid_lift: float = 0.0
    print_orientation: dict = field(default_factory=lambda: {"shell": "floor_down", "lid": "top_down"})
    colors: dict = field(default_factory=lambda: {"shell": [0.18, 0.45, 0.75], "lid": [0.85, 0.85, 0.85],
                                                  "ref": [0.10, 0.55, 0.20]})

    def to_dict(self) -> dict:
        """Plain JSON-able dict (a deep copy), key order as in the spec JSON."""
        return to_dict(self)


_GROUP_TYPES = {"standoff": StandoffSpec, "lid": LidSpec, "hooks": HookSpec, "vents": VentSpec, "ports": PortSpec}

# dotted leaf path -> value kind. Kinds: num (float), deg (number, integral values stay int), int, bool, str,
# enum (lower-cased text), id (connector id), idlist, center, rgb, objlist.
_LEAF_KINDS = {
    "version": "int", "kit_id": "str", "name": "str", "board": "id", "material": "material",
    "wall_t": "num", "floor_t": "num", "lid_t": "num", "clearance": "num", "shrink_comp": "enum", "corner_r": "num",
    "standoff.h": "num", "standoff.d": "num", "standoff.pilot_d": "num",
    "lid.style": "enum", "lid.gap": "num", "lid.lip_h": "num", "lid.lip_t": "num", "lid.fit": "num",
    "hooks.count": "int", "hooks.w": "num", "hooks.t": "num", "hooks.overhang": "num", "hooks.length": "num",
    "hooks.barb_h": "num", "hooks.ramp_deg": "deg", "hooks.retention_deg": "deg", "hooks.window_margin": "num",
    "vents.enabled": "bool", "vents.part": "enum", "vents.face": "enum", "vents.pattern": "enum", "vents.count": "int",
    "vents.slot_w": "num", "vents.slot_l": "num", "vents.pitch": "num", "vents.center": "center", "vents.angle_deg": "deg",
    "ports.auto": "bool", "ports.include": "idlist", "ports.exclude": "idlist", "ports.margin": "num",
    "ports.gap": "num", "ports.plug_clearance": "bool", "ports.overrides": "objlist",
    "cutouts": "objlist",
    "lid_lift": "num",
    "print_orientation.shell": "enum", "print_orientation.lid": "enum",
    "colors.shell": "rgb", "colors.lid": "rgb", "colors.ref": "rgb",
}
_DICT_GROUPS = ("print_orientation", "colors")
# keys (and their kinds) accepted inside the entries of the two list-of-objects fields
_ENTRY_KINDS = {
    "cutouts": {"id": "str", "part": "enum", "face": "enum", "u": "num", "v": "num", "w": "num", "h": "num",
                "r": "num", "margin": "num", "through": "bool"},
    "ports.overrides": {"id": "id", "margin": "num"},
}
_KIND_WORDS = {"num": "a number (millimetres, for example 2.4)", "deg": "a number (degrees, for example 30)",
               "int": "a whole number", "bool": "true or false", "str": "text", "enum": "text", "id": "text",
               "material": "text", "idlist": "a list of connector ids", "center": "null or a [u, v] pair of numbers",
               "rgb": "three numbers between 0 and 1 ([r, g, b])", "objlist": "a list of objects"}

_NUM_RE = re.compile(r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:mm|deg|°)?\s*$")


# --------------------------------------------------------------------------------------------------------------------
# coercion helpers (lenient: a value that cannot be coerced is kept as is, validate() reports it)
# --------------------------------------------------------------------------------------------------------------------

def _as_number(v: Any) -> "float | None":
    """A finite float from an int/float or a numeric string ('2.4', '2.4 mm', '45 deg'); None otherwise (bools too)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, str):
        m = _NUM_RE.match(v)
        if m:
            x = float(m.group(1))
            return x if math.isfinite(x) else None
    return None


def _as_int(v: Any) -> "int | None":
    x = _as_number(v)
    return int(x) if x is not None and float(x).is_integer() else None


def _as_bool(v: Any) -> "bool | None":
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "yes", "on", "1", "false", "no", "off", "0"):
        return v.strip().lower() in ("true", "yes", "on", "1")
    return None


def _coerce(kind: str, v: Any) -> Any:
    """Coerce ``v`` to ``kind``; return ``v`` unchanged when that is impossible."""
    if kind == "num":
        x = _as_number(v)
        return v if x is None else x
    if kind == "deg":
        x = _as_number(v)
        return v if x is None else (int(x) if x.is_integer() else x)
    if kind == "int":
        x = _as_int(v)
        return v if x is None else x
    if kind == "bool":
        b = _as_bool(v)
        return v if b is None else b
    if kind == "str":
        return v.strip() if isinstance(v, str) else v
    if kind == "enum":
        return v.strip().lower() if isinstance(v, str) else v
    if kind == "id":
        return v.strip().lower() if isinstance(v, str) else v
    if kind == "material":
        return _materials.normalize(v)
    if kind == "idlist":
        if isinstance(v, str):
            v = [v]
        if isinstance(v, (list, tuple)) and all(isinstance(i, str) for i in v):
            return [i.strip().lower() for i in v if i.strip()]
        return v
    if kind == "center":
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and len(v) == 2:
            nums = [_as_number(i) for i in v]
            if all(n is not None for n in nums):
                return nums
        return v
    if kind == "rgb":
        if isinstance(v, (list, tuple)) and len(v) == 3:
            nums = [_as_number(i) for i in v]
            if all(n is not None for n in nums):
                return nums
        return v
    return v


def _coerce_entries(path: str, v: Any) -> Any:
    """Coerce the known keys of every object in a list-of-objects field; unknown keys are dropped."""
    if not isinstance(v, (list, tuple)):
        return v
    kinds = _ENTRY_KINDS[path]
    out = []
    for entry in v:
        if not isinstance(entry, dict):
            out.append(entry)
            continue
        out.append({k: _coerce(kinds[k], val) for k, val in entry.items() if k in kinds})
    return out


def _leaf(path: str, v: Any) -> Any:
    kind = _LEAF_KINDS[path]
    return _coerce_entries(path, v) if kind == "objlist" else _coerce(kind, v)


# --------------------------------------------------------------------------------------------------------------------
# construction and conversion
# --------------------------------------------------------------------------------------------------------------------

def default_spec(board_id: str = "rpi4b", material: str = "ABS") -> EnclosureSpec:
    """The default spec for ``board_id`` and ``material``.

    The default numbers are tuned for a Pi 4 in ABS. They are kept for every board they validate on; for a board they
    do not fit (Zero 2 W, Pico, Nano, ESP32 dev kit) ``_defaults.board_sized`` shrinks the vent array and the hooks (or turns the
    lid into a friction lid) so that the default spec of EVERY catalogue board validates. ``board_id`` is stored lower-cased and
    ``material`` upper-cased; an unknown board id (or a missing board library) still yields the plain spec.
    """
    spec = EnclosureSpec()
    spec.board = _coerce("id", board_id)
    spec.material = _coerce("material", material)
    from ._defaults import board_sized    # lazy: _defaults imports this module
    return board_sized(spec)


def to_dict(spec: "EnclosureSpec | dict") -> dict:
    """``spec`` as a plain dict (deep copy; a dict argument is copied as is)."""
    if isinstance(spec, dict):
        return copy.deepcopy(spec)
    return dataclasses.asdict(spec)


def merge(spec: dict, patch: dict) -> dict:
    """Deep merge: nested dicts are merged key by key, everything else (lists, scalars, null) is replaced.

    Neither argument is modified; the result shares no mutable object with them.
    """
    out = copy.deepcopy(spec)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _template() -> dict:
    return default_spec().to_dict()


def _strip_unknown(raw: dict, template: dict) -> dict:
    """``raw`` without the keys the schema does not know (one level per nested group)."""
    out = {}
    for key, value in raw.items():
        if key not in template:
            continue
        if isinstance(template[key], dict):      # a group: keep its known keys; a non-dict keeps the default (validate reports it)
            if isinstance(value, dict):
                out[key] = {k: v for k, v in value.items() if k in template[key]}
            continue
        out[key] = value
    return out


def _build(m: dict) -> EnclosureSpec:
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(EnclosureSpec):
        name = f.name
        value = m.get(name)
        if name in _GROUP_TYPES:
            cls = _GROUP_TYPES[name]
            group = value if isinstance(value, dict) else {}
            kwargs[name] = cls(**{g.name: _leaf(f"{name}.{g.name}", group[g.name])
                                  for g in dataclasses.fields(cls) if g.name in group})
        elif name in _DICT_GROUPS:
            group = value if isinstance(value, dict) else {}
            kwargs[name] = {k: _leaf(f"{name}.{k}", v) for k, v in group.items()}
        elif name in m:
            kwargs[name] = _leaf(name, value)
    return EnclosureSpec(**kwargs)


def from_dict(d: dict) -> EnclosureSpec:
    """Merge ``d`` onto the defaults of ``d['board']`` / ``d['material']`` and coerce the values.

    Lenient by design: unknown keys are ignored, numeric strings ('2.4 mm') become numbers, ``material`` is
    upper-cased, ids and enum words lower-cased. A value that cannot be coerced is kept as given, and
    ``validate`` reports it; this function never raises for bad values.
    """
    raw = d if isinstance(d, dict) else {}
    board = raw.get("board", "rpi4b")
    material = raw.get("material", "ABS")
    base = default_spec(board if isinstance(board, str) and board.strip() else "rpi4b",
                        material if isinstance(material, str) and material.strip() else "ABS").to_dict()
    return _build(merge(base, _strip_unknown(raw, base)))


def _as_spec(spec: "EnclosureSpec | dict") -> EnclosureSpec:
    return spec if isinstance(spec, EnclosureSpec) else from_dict(spec)


# --------------------------------------------------------------------------------------------------------------------
# board access
# --------------------------------------------------------------------------------------------------------------------

def _catalog():
    """``oscar.refparts.catalog`` or None while that module does not exist (guarded import)."""
    try:
        from ..refparts import catalog
        return catalog
    except ModuleNotFoundError as exc:
        if (exc.name or "").endswith(("refparts", "refparts.catalog")):
            return None
        raise


def _plug_defaults() -> dict:
    cat = _catalog()
    table = getattr(cat, "PLUG_DEFAULTS", None) if cat is not None else None
    return table if isinstance(table, dict) and table else _PLUG_DEFAULTS


def load_board(board_id: str) -> dict:
    """The board dict for ``board_id`` through ``refparts.catalog.load_board``.

    Raises ``KeyError`` (message lists the known ids) for an unknown id and ``LookupError`` when the board
    library is not installed in this build.
    """
    cat = _catalog()
    if cat is None:
        raise LookupError("the board library (refparts.catalog) is not available in this build")
    return cat.load_board(board_id)


def _known_board_ids() -> list[str]:
    cat = _catalog()
    if cat is None:
        return []
    try:
        return [str(b["id"]) for b in cat.list_boards() if isinstance(b, dict) and "id" in b]
    except Exception:    # noqa: BLE001 - a broken catalog must not break validation
        return []


# --------------------------------------------------------------------------------------------------------------------
# derived parameters: the formulas live HERE and are evaluated by a tiny safe evaluator
# --------------------------------------------------------------------------------------------------------------------

# (name, formula, unit): the derived rows, in order; formulas use bare aliases and ';' in functions.
DERIVED_FORMULAS: tuple[tuple[str, str, str], ...] = (
    ("clearance_eff", "=clearance + shrink_pct/100*board_len/2", "mm"),
    ("margin_nx", "=max(clearance_eff; pr_nx + port_gap)", "mm"),
    ("margin_px", "=max(clearance_eff; pr_px + port_gap)", "mm"),
    ("margin_ny", "=max(clearance_eff; pr_ny + port_gap)", "mm"),
    ("margin_py", "=max(clearance_eff; pr_py + port_gap)", "mm"),
    ("cav_x0", "=-margin_nx", "mm"),
    ("cav_x1", "=board_len + margin_px", "mm"),
    ("cav_y0", "=-margin_ny", "mm"),
    ("cav_y1", "=board_w + margin_py", "mm"),
    ("cav_len", "=cav_x1 - cav_x0", "mm"),
    ("cav_wid", "=cav_y1 - cav_y0", "mm"),
    ("shell_len", "=cav_len + 2*wall_t", "mm"),
    ("shell_wid", "=cav_wid + 2*wall_t", "mm"),
    ("z_floor_top", "=-standoff_h", "mm"),
    ("z_floor_bot", "=-(standoff_h + floor_t)", "mm"),
    ("z_cavity_top", "=tallest_z + lid_gap", "mm"),
    ("shell_h", "=z_cavity_top - z_floor_bot", "mm"),
    ("lid_len", "=shell_len + 2*(lid_fit + lip_t)", "mm"),
    ("lid_wid", "=shell_wid + 2*(lid_fit + lip_t)", "mm"),
)

_FUNCTIONS = {"max": max, "min": min, "abs": abs, "ceil": math.ceil, "floor": math.floor, "round": round,
              "sqrt": math.sqrt}


def eval_formula(formula: str, env: dict) -> float:
    """Evaluate a sheet formula (``=a + b*2``, ``=max(a; b)``) over ``env`` {alias: number}.

    Supports ``+ - * / ** %``, unary signs, parentheses, numbers, aliases and ``max min abs ceil floor round sqrt``
    (function arguments are separated by ``;`` as in the sheet). Raises ``ValueError`` (only) for anything else:
    a name missing from ``env``, a syntax error, division by zero, a domain or overflow error, a wrong argument
    count, or a result that is not finite. Used to keep ``derived`` and the Params sheet on one set of formulas.
    """
    text = formula.strip()
    if text.startswith("="):
        text = text[1:]
    try:
        tree = ast.parse(text.replace(";", ","), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"formula {formula!r} is not valid: {exc.msg}") from None
    except (RecursionError, MemoryError):
        raise ValueError(f"formula {formula!r} is too deeply nested") from None

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"formula {formula!r}: '{node.id}' is not a known parameter")
            return float(env[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                if b == 0:
                    raise ValueError(f"formula {formula!r}: division by zero")
                return a / b
            if isinstance(node.op, ast.Pow):
                return a ** b
            if isinstance(node.op, ast.Mod):
                return a % b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
            return float(_FUNCTIONS[node.func.id](*[ev(a) for a in node.args]))
        raise ValueError(f"formula {formula!r}: unsupported expression")

    try:
        result = ev(tree)
    except ValueError:
        raise
    except (ArithmeticError, TypeError, RecursionError) as exc:
        raise ValueError(f"formula {formula!r}: cannot be evaluated ({exc})") from None
    if not math.isfinite(result):
        raise ValueError(f"formula {formula!r}: the result is not a finite number")
    return result


def _confidence_source(confidence: Any) -> str:
    """Board confidence -> parameter source: measured -> step_measurement, datasheet -> datasheet, else assumption."""
    return {"measured": "step_measurement", "datasheet": "datasheet"}.get(confidence, "assumption")


def _fields_of(board: dict, *path: str) -> Any:
    cur: Any = board
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise ValueError(f"board {board.get('id', '?')!r}: incomplete board data (missing {'.'.join(path)!r})")
        cur = cur[key]
    return cur


def _mat(spec: EnclosureSpec) -> dict:
    try:
        return _materials.get(spec.material)
    except KeyError as exc:
        raise ValueError(str(exc.args[0])) from None


def _row(name: str, value: float, unit: str, source: str, note: str = "", formula: "str | None" = None) -> dict:
    return {"name": name, "value": value, "unit": unit, "source": source, "note": note, "formula": formula}


# parameter name -> (dotted spec path, unit, note): the DESIGN rows (every one `assumption` unless the user states it)
DESIGN_PARAMS: dict[str, tuple[str, str, str]] = {
    "wall_t": ("wall_t", "mm", "wall thickness"),
    "floor_t": ("floor_t", "mm", "floor thickness"),
    "lid_t": ("lid_t", "mm", "lid plate thickness"),
    "clearance": ("clearance", "mm", "gap between board edge and cavity wall, before shrink compensation"),
    "corner_r": ("corner_r", "mm", "outer corner radius of the shell"),
    "standoff_h": ("standoff.h", "mm", "standoff height under the PCB"),
    "standoff_d": ("standoff.d", "mm", "standoff ring diameter"),
    "standoff_pilot_d": ("standoff.pilot_d", "mm", "standoff pilot hole diameter"),
    "lid_gap": ("lid.gap", "mm", "gap between the tallest component and the lid underside"),
    "lip_h": ("lid.lip_h", "mm", "lid skirt height"),
    "lip_t": ("lid.lip_t", "mm", "lid skirt thickness"),
    "lid_fit": ("lid.fit", "mm", "clearance between lid skirt and shell wall"),
    "hook_count": ("hooks.count", "", "number of snap hooks (both long walls)"),
    "hook_w": ("hooks.w", "mm", "snap hook width"),
    "hook_t": ("hooks.t", "mm", "snap hook beam thickness"),
    "hook_overhang": ("hooks.overhang", "mm", "barb protrusion past the beam"),
    "hook_len": ("hooks.length", "mm", "snap hook beam length"),
    "hook_barb_h": ("hooks.barb_h", "mm", "barb height"),
    "hook_ramp_deg": ("hooks.ramp_deg", "deg", "barb lead-in ramp angle"),
    "hook_retention_deg": ("hooks.retention_deg", "deg", "barb retention face angle"),
    "window_margin": ("hooks.window_margin", "mm", "barb top clearance below the catch pocket ceiling"),
    "vent_count": ("vents.count", "", "number of vent slots"),
    "vent_slot_w": ("vents.slot_w", "mm", "vent slot width"),
    "vent_slot_l": ("vents.slot_l", "mm", "vent slot length"),
    "vent_pitch": ("vents.pitch", "mm", "vent slot centre-to-centre distance"),
    "port_margin": ("ports.margin", "mm", "clearance added around every connector opening"),
    "port_gap": ("ports.gap", "mm", "space between a protruding connector and the wall"),
    "lid_lift": ("lid_lift", "mm", "assembly pose: how far the lid is lifted off the shell"),
}
PARAM_OF_FIELD: dict[str, str] = {path: name for name, (path, _u, _n_) in DESIGN_PARAMS.items()}
FIELD_OF_PARAM: dict[str, str] = {name: path for name, (path, _u, _n_) in DESIGN_PARAMS.items()}
BOARD_PARAMS = ("board_len", "board_w", "pcb_t", "hole_off", "hole_dx", "hole_dy", "hole_d", "tallest_z", "bottom_z",
                "pr_nx", "pr_px", "pr_ny", "pr_py")
DERIVED_PARAMS = tuple(name for name, _f, _u in DERIVED_FORMULAS)
# the order of every table: board rows, then design rows (with shrink_pct after clearance), then derived rows
INPUT_PARAM_ORDER = BOARD_PARAMS + tuple(
    n for name in DESIGN_PARAMS for n in ((name, "shrink_pct") if name == "clearance" else (name,)))


def _get_path(d: dict, path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        cur = cur[key]
    return cur


def _protrusion(connector: dict, board: dict) -> float:
    """How far a connector box extends past the PCB edge on its own face (7.1 ``opening``): 0 for top / bottom."""
    box = connector["box"]
    face = connector["face"]
    length = _fields_of(board, "outline", "length")
    width = _fields_of(board, "outline", "width")
    if face == "-y":
        return max(0.0, -box["y"][0])
    if face == "+y":
        return max(0.0, box["y"][1] - width)
    if face == "-x":
        return max(0.0, -box["x"][0])
    if face == "+x":
        return max(0.0, box["x"][1] - length)
    return 0.0


def active_connectors(spec: "EnclosureSpec | dict", board: dict) -> list[dict]:
    """The board connectors that get an opening, in board order.

    ``ports.auto`` opens every connector whose ``cutout`` is true; ``ports.include`` forces a connector (any
    face, also ``cutout: false`` ones) and works with ``auto`` off; ``ports.exclude`` always wins.
    """
    s = _as_spec(spec)
    include, exclude = set(s.ports.include), set(s.ports.exclude)
    out = []
    for c in board.get("connectors", []):
        cid = c.get("id")
        if cid in exclude:
            continue
        if cid in include or (s.ports.auto and c.get("cutout")):
            out.append(c)
    return out


def _board_param_rows(s: EnclosureSpec, board: dict) -> list[dict]:
    outline = _fields_of(board, "outline")
    holes = _fields_of(board, "mounting_holes")
    pcb = _fields_of(board, "pcb")
    out_src = _confidence_source(outline.get("confidence"))
    hole_src = _confidence_source(holes.get("confidence"))
    pattern = holes.get("pattern")
    positions = holes.get("positions") or []
    if isinstance(pattern, dict) and "offset" in pattern:
        hole_off, hole_dx, hole_dy = float(pattern["offset"][0]), float(pattern["dx"]), float(pattern["dy"])
    elif positions:
        xs, ys = [p[0] for p in positions], [p[1] for p in positions]
        hole_off, hole_dx, hole_dy = float(min(xs)), float(max(xs) - min(xs)), float(max(ys) - min(ys))
    else:
        raise ValueError(f"board {board.get('id', '?')!r}: incomplete board data (no mounting hole positions)")
    connectors = board.get("connectors", [])
    height = board.get("height") if isinstance(board.get("height"), dict) else {}
    tops = [(c["box"]["z"][1], c) for c in connectors]
    bottoms = [(c["box"]["z"][0], c) for c in connectors]
    top_c = max(tops, key=lambda t: t[0])[1] if tops else None
    bot_c = min(bottoms, key=lambda t: t[0])[1] if bottoms else None
    tallest = float(height["max_top_z"]) if "max_top_z" in height else (max(t[0] for t in tops) if tops else float(pcb["thickness"]))
    bottom = float(height["min_bottom_z"]) if "min_bottom_z" in height else (min(b[0] for b in bottoms) if bottoms else 0.0)
    pr: dict[str, tuple[float, str]] = {}
    active = active_connectors(s, board)
    for face, key in (("-x", "pr_nx"), ("+x", "pr_px"), ("-y", "pr_ny"), ("+y", "pr_py")):
        best, src = 0.0, "assumption"
        for c in active:
            if c.get("face") == face:
                p = _protrusion(c, board)
                if p > best:
                    best, src = p, _confidence_source(c.get("confidence"))
        pr[key] = (best, src)
    rows = [
        _row("board_len", float(outline["length"]), "mm", out_src, "board length (long edge, +x)"),
        _row("board_w", float(outline["width"]), "mm", out_src, "board width (short edge, +y)"),
        _row("pcb_t", float(pcb["thickness"]), "mm", _confidence_source(pcb.get("confidence")), "PCB thickness"),
        _row("hole_off", hole_off, "mm", hole_src, "first mounting hole offset from the board corner"),
        _row("hole_dx", hole_dx, "mm", hole_src, "mounting hole spacing along x"),
        _row("hole_dy", hole_dy, "mm", hole_src, "mounting hole spacing along y"),
        _row("hole_d", float(holes["diameter"]), "mm", hole_src, "mounting hole diameter"),
        _row("tallest_z", tallest, "mm", _confidence_source(top_c.get("confidence")) if top_c else "assumption",
             "top of the tallest component, measured from the PCB bottom"),
        _row("bottom_z", bottom, "mm", _confidence_source(bot_c.get("confidence")) if bot_c else "assumption",
             "lowest point of the board underside (negative = below the PCB bottom)"),
    ]
    for key, label in (("pr_nx", "-x"), ("pr_px", "+x"), ("pr_ny", "-y"), ("pr_py", "+y")):
        rows.append(_row(key, pr[key][0], "mm", pr[key][1], f"largest connector protrusion past the board edge on the {label} face"))
    return rows


def input_params(spec: "EnclosureSpec | dict", board: dict) -> list[dict]:
    """The INPUT rows of the ``Params`` sheet for ``spec`` on ``board``, in ``DEFAULT_PARAMS`` order.

    Each row is ``{"name", "value", "unit", "source", "note", "formula": None}``. Board rows come from the board
    data (source from its confidence; the connector protrusions ``pr_*`` count only the connectors
    that get an opening), design rows from the spec (source ``assumption``), ``shrink_pct`` is the material's
    shrinkage in percent when ``shrink_comp == 'clearance'`` and 0 otherwise. Raises ``ValueError`` for an unknown
    material or incomplete board data.
    """
    s = _as_spec(spec)
    mat = _mat(s)
    d = to_dict(s)
    rows = _board_param_rows(s, board)
    for name, (path, unit, note) in DESIGN_PARAMS.items():
        value = _get_path(d, path)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{path}: expected a number, got {value!r}")
        rows.append(_row(name, float(value), unit, "assumption", note))
        if name == "clearance":
            pct = round(mat["shrink"] * 100.0, 6) if s.shrink_comp == "clearance" else 0.0
            rows.append(_row("shrink_pct", pct, "", "assumption",
                             f"{s.material} shrinkage in percent" if pct else "shrinkage is not compensated"))
    return rows


def param_rows(spec: "EnclosureSpec | dict", board: dict) -> list[dict]:
    """Every row of the ``Params`` sheet for ``spec``: ``input_params`` followed by the derived rows.

    Derived rows carry their formula (``"=cav_len + 2*wall_t"``), the evaluated ``value``, ``source == 'derived'``.
    """
    rows = input_params(spec, board)
    env = {r["name"]: r["value"] for r in rows}
    for name, formula, unit in DERIVED_FORMULAS:
        env[name] = eval_formula(formula, env)
        rows.append(_row(name, round(env[name], 9), unit, "derived", "", formula))
    return rows


def derived(spec: "EnclosureSpec | dict", board: dict) -> dict:
    """Every derived parameter: ``{name: {"value": float, "expr": "=<formula>", "unit": "mm"}}``.

    Evaluates ``DERIVED_FORMULAS`` (the formulas of the sheet itself) over ``input_params``. Pi 4 / ABS defaults:
    ``clearance_eff 0.64``, cavity 90.1 x 57.94 (x -2.8..87.3, y -1.3..56.64), shell 94.1 x 61.94 x 24.0 (z -6..18),
    lid 97.0 x 64.84. Raises ``ValueError`` for an unknown material or incomplete board data.
    """
    env = {r["name"]: r["value"] for r in input_params(spec, board)}
    out: dict[str, dict] = {}
    for name, formula, unit in DERIVED_FORMULAS:
        env[name] = eval_formula(formula, env)
        out[name] = {"value": round(env[name], 9), "expr": formula, "unit": unit}
    return out


# --------------------------------------------------------------------------------------------------------------------
# connector openings, hook placement, snap-fit numbers (shared by validate, checks and the geometry)
# --------------------------------------------------------------------------------------------------------------------

def connector_margin(spec: "EnclosureSpec | dict", connector: dict) -> float:
    """Margin the kit cuts around one connector: an ``ports.overrides[]`` entry's explicit ``margin``, else
    ``max(spec.ports.margin, connector.margin_hint)`` (7.1a: the spec's value is a floor, not an override)."""
    s = _as_spec(spec)
    for entry in s.ports.overrides:
        if isinstance(entry, dict) and entry.get("id") == connector.get("id") and isinstance(entry.get("margin"), (int, float)):
            return float(entry["margin"])
    return max(float(s.ports.margin), float(connector.get("margin_hint", 0.0) or 0.0))


def connector_opening(connector: dict, board: dict, margin: "float | None" = None, plug: bool = True,
                      dz: float = 0.0) -> dict:
    """The opening geometry of one connector (the semantics of ``refparts.catalog.opening``, 7.1).

    Returns ``{"id", "kind", "face", "u", "v", "w", "h", "protrusion"}`` in absolute world coordinates.
    With ``margin=None`` it is the RECEPTACLE (box extent); with a margin ``w = max(w_receptacle + 2*margin,
    plug_w)`` and likewise ``h`` when ``plug`` is true (the cable plug, from the connector's ``plug_w`` /
    ``plug_h`` or the kind's default). ``dz`` shifts every box z at or above the PCB top (the PCB thickness rule).
    """
    box = connector["box"]
    t_board = float(_fields_of(board, "pcb", "thickness"))
    z0, z1 = (z + dz if z >= t_board else z for z in box["z"])
    x0, x1 = box["x"]
    y0, y1 = box["y"]
    face = connector["face"]
    if face in ("-y", "+y"):
        u, v, w, h = (x0 + x1) / 2, (z0 + z1) / 2, x1 - x0, z1 - z0
    elif face in ("-x", "+x"):
        u, v, w, h = (y0 + y1) / 2, (z0 + z1) / 2, y1 - y0, z1 - z0
    else:
        u, v, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    if margin is not None:
        w, h = w + 2 * margin, h + 2 * margin
        if plug:
            dw, dh = _plug_defaults().get(connector.get("kind"), (0.0, 0.0))
            pw = connector.get("plug_w", dw) or 0.0
            ph = connector.get("plug_h", dh) or 0.0
            w, h = max(w, float(pw)), max(h, float(ph))
    return {"id": connector.get("id"), "kind": connector.get("kind"), "face": face, "u": u, "v": v, "w": w, "h": h,
            "protrusion": _protrusion(connector, board)}


def port_openings(spec: "EnclosureSpec | dict", board: dict, *, pcb_t: "float | None" = None) -> list[dict]:
    """One opening per active connector (``active_connectors``) as the kit cuts it, in board order.

    Each dict is ``connector_opening`` with ``margin = connector_margin`` and ``plug = spec.ports.plug_clearance``,
    plus the ``margin`` used. ``pcb_t`` (the sheet's measured PCB thickness) shifts the z of every connector box at
    or above the PCB top by ``pcb_t - board.pcb.thickness``.
    """
    s = _as_spec(spec)
    dz = 0.0 if pcb_t is None else float(pcb_t) - float(_fields_of(board, "pcb", "thickness"))
    out = []
    for c in active_connectors(s, board):
        m = connector_margin(s, c)
        o = connector_opening(c, board, margin=m, plug=bool(s.ports.plug_clearance), dz=dz)
        o["margin"] = m
        out.append(o)
    return out


def shell_cutout_openings(spec: "EnclosureSpec | dict") -> list[dict]:
    """The user cut-outs (``spec.cutouts``) that belong to the shell, as openings ``{"id","face","u","v","w","h"}``
    with ``margin`` added around w x h. Entries that are malformed or not on the shell are skipped."""
    s = _as_spec(spec)
    out = []
    for i, e in enumerate(s.cutouts):
        if not isinstance(e, dict) or e.get("part") != "shell" or e.get("face") not in FACES:
            continue
        nums = [e.get(k) for k in ("u", "v", "w", "h")]
        if not all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in nums):
            continue
        m = e.get("margin") if isinstance(e.get("margin"), (int, float)) and not isinstance(e.get("margin"), bool) else 0.0
        out.append({"id": e.get("id") or f"cutout{i + 1}", "kind": "cutout", "face": e["face"], "u": float(nums[0]),
                    "v": float(nums[1]), "w": float(nums[2]) + 2 * m, "h": float(nums[3]) + 2 * m, "protrusion": 0.0,
                    "margin": float(m)})
    return out


def wall_openings(spec: "EnclosureSpec | dict", board: dict, wall: "str | None" = None, *,
                  pcb_t: "float | None" = None) -> list[dict]:
    """Every opening in the shell walls (port openings plus the user's shell cut-outs); ``wall`` filters by face.

    Top / bottom openings are included only when ``wall`` is None or names that face.
    """
    s = _as_spec(spec)
    rows = port_openings(s, board, pcb_t=pcb_t) + shell_cutout_openings(s)
    return [o for o in rows if wall is None or o["face"] == wall]


def catch_pocket_depth(spec: "EnclosureSpec | dict") -> float:
    """Depth of the blind catch pocket in the shell's outer wall: ``hooks.overhang - lid.fit + 0.1``."""
    s = _as_spec(spec)
    return s.hooks.overhang - s.lid.fit + POCKET_SEAT


def hook_strain(spec: "EnclosureSpec | dict") -> float:
    """Bending strain of a hook beam while the lid is pushed on: ``1.5 * t * (overhang - fit) / length**2``.

    Default 1.5 * 1.2 * (0.8 - 0.25) / 12**2 = 0.006875. Returns ``inf`` for a beam of non-positive length.
    """
    s = _as_spec(spec)
    if s.hooks.length <= 0:
        return math.inf
    try:
        return 1.5 * s.hooks.t * (s.hooks.overhang - s.lid.fit) / s.hooks.length ** 2
    except OverflowError:       # an absurd length (1e308): the strain is effectively zero
        return 0.0


def hook_strain_limit(spec: "EnclosureSpec | dict", *, molded: bool = False) -> float:
    """The allowed strain for ``spec.material``: ``max_strain * FDM_DERATE`` (ABS 0.010), or the raw limit when ``molded``."""
    return _materials.hook_strain_limit(_as_spec(spec).material, molded=molded)


def _free_spans(lo: float, hi: float, blocked: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The parts of [lo, hi] not covered by the ``blocked`` open intervals (spans shorter than 1e-6 are dropped)."""
    spans: list[tuple[float, float]] = []
    cursor = lo
    for b0, b1 in sorted(blocked):
        if b0 > cursor and min(b0, hi) - cursor > 1e-6:
            spans.append((cursor, min(b0, hi)))
        cursor = max(cursor, b1)
    if hi - cursor > 1e-6:
        spans.append((cursor, hi))
    return spans


def _place(candidates: list[float], spans: list[tuple[float, float]], width: float) -> list[float]:
    """Keep candidates that lie in a free span; move blocked ones into the nearest span; drop what does not fit."""
    if not spans:
        return []
    assigned: dict[int, list[float]] = {i: [] for i in range(len(spans))}
    moved: set[int] = set()
    for c in candidates:
        idx = next((i for i, (a, b) in enumerate(spans) if a <= c <= b), None)
        if idx is None:
            idx = min(range(len(spans)), key=lambda i: min(abs(c - spans[i][0]), abs(c - spans[i][1])))
            moved.add(idx)
        assigned[idx].append(c)
    out: list[float] = []
    for i, (a, b) in enumerate(spans):
        cs = sorted(assigned[i])
        if not cs:
            continue
        spaced = all(q - p >= width + MIN_HOOK_GAP for p, q in zip(cs, cs[1:]))
        if i not in moved and spaced:
            out.extend(cs)
            continue
        n = min(len(cs), max(1, int((b - a) // (width + MIN_HOOK_GAP))))
        out.extend(a + (j + 0.5) / n * (b - a) for j in range(n))
    return sorted(out)


def hook_positions(spec: "EnclosureSpec | dict", board: dict, *, pcb_t: "float | None" = None) -> list[dict]:
    """Where the snap hooks go: ``[{"wall", "u", "w"}]``, ``u`` an absolute x coordinate on a long wall.

    ``hooks.count // 2`` hooks per long wall, ``+y`` first (then ``-y``), ascending ``u``. The ideal positions
    split the cavity length into equal parts (two per wall: 25 % and 75 %, x = 19.7 and 64.8 on the Pi 4). A hook
    centre must keep ``1 mm + hooks.w / 2`` from every opening on its wall (``wall_openings``) and stay on the
    straight part of the wall (``corner_r - wall_t + hooks.w / 2`` from the cavity ends). A candidate that falls in
    a blocked zone moves into the nearest free span; all hooks of a span are then spread evenly over it (Pi 4 ``-y``:
    the free span starts after the audio opening, hooks at about x = 67 and 78); hooks that do not fit are dropped,
    so the result can be shorter than ``hooks.count`` (the validator reports that). Returns ``[]`` unless
    ``lid.style == 'snap_fit'`` and ``2 <= hooks.count <= 64``. Raises ``ValueError`` for an unknown material or incomplete board data.
    """
    s = _as_spec(spec)
    if s.lid.style != "snap_fit" or not isinstance(s.hooks.count, int) or not 2 <= s.hooks.count <= MAX_HOOKS:
        return []
    dv = derived(s, board)
    cx0, cx1 = dv["cav_x0"]["value"], dv["cav_x1"]["value"]
    w = float(s.hooks.w)
    widen = HOOK_WIDEN + w / 2
    edge = max(s.corner_r - s.wall_t, 0.0) + w / 2
    lo, hi = cx0 + edge, cx1 - edge
    per_wall = s.hooks.count // 2
    candidates = [cx0 + (i + 0.5) / per_wall * (cx1 - cx0) for i in range(per_wall)]
    out: list[dict] = []
    for wall in ("+y", "-y"):
        blocked = [(o["u"] - o["w"] / 2 - widen, o["u"] + o["w"] / 2 + widen) for o in wall_openings(s, board, wall, pcb_t=pcb_t)]
        for u in _place(candidates, _free_spans(lo, hi, blocked), w):
            out.append({"wall": wall, "u": round(u, 6), "w": w})
    return out


def _face_rect(dv: dict, face: str) -> "tuple[float, float, float, float]":
    """(u0, u1, v0, v1) of a face over the cavity, in world coordinates (7.2: wall u = x or y, v = z; top/bottom u = x, v = y)."""
    x0, x1, y0, y1 = (dv[k]["value"] for k in ("cav_x0", "cav_x1", "cav_y0", "cav_y1"))
    z0, z1 = dv["z_floor_top"]["value"], dv["z_cavity_top"]["value"]
    if face in ("top", "bottom"):
        return x0, x1, y0, y1
    if face in ("-y", "+y"):
        return x0, x1, z0, z1
    return y0, y1, z0, z1


def vent_area(spec: "EnclosureSpec | dict", board: dict, part: "str | None" = None, face: "str | None" = None) -> dict:
    """The area a vent array may occupy on ``face`` of ``part`` (default: the spec's vents).

    ``{"u0","u1","v0","v1","cu","cv","part","face"}``: the cavity footprint of the face shrunk by the material's
    ``min_web`` on every side, and the default array centre (the centre of the unshrunk face over the cavity).
    """
    s = _as_spec(spec)
    part = part or s.vents.part
    face = face or s.vents.face
    dv = derived(s, board)
    web = _mat(s)["min_web"]
    u0, u1, v0, v1 = _face_rect(dv, face)
    return {"u0": u0 + web, "u1": u1 - web, "v0": v0 + web, "v1": v1 - web, "cu": (u0 + u1) / 2, "cv": (v0 + v1) / 2,
            "part": part, "face": face}


def vent_extent(spec: "EnclosureSpec | dict") -> "tuple[float, float, float]":
    """``(span, length, count)``: the vent array's size across the slots, along the slots (before rotation) and the slot count.

    ``span = (count - 1) * pitch + slot_w``; ``length = slot_l`` for slots, ``slot_w`` for round holes.
    """
    v = _as_spec(spec).vents
    return (v.count - 1) * v.pitch + v.slot_w, (v.slot_l if v.pattern == "slots" else v.slot_w), v.count


# --------------------------------------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------------------------------------

def validate(spec: "EnclosureSpec | dict", board: "dict | None" = None) -> "tuple[bool, list[str], list[str]]":
    """Check ``spec`` against every enclosure-spec rule. Returns ``(ok, errors, warnings)``.

    ``spec`` may be an ``EnclosureSpec`` or a (partial) dict; a dict is merged onto the defaults like ``from_dict``,
    and its unknown keys are reported as errors with ``difflib`` suggestions. ``board`` is the board data of
    ``spec.board``; when None it is loaded through ``refparts.catalog.load_board``. An unknown board id is an
    error that lists the known ids; while the board library does not exist in the build the board-dependent rules
    (corner, standoff, hooks, vents, ports) are skipped with one warning. Messages are
    ``"<path>: <problem>; <what to do>"``. Never raises for bad values. ``ok`` is ``not errors``.
    """
    from ._validate import validate as _validate    # lazy: _validate imports this module
    return _validate(spec, board)


# --------------------------------------------------------------------------------------------------------------------
# stage hashes
# --------------------------------------------------------------------------------------------------------------------

# own fields per stage. Paths are dotted; a whole group ('standoff', 'lid', 'ports', 'hooks', 'vents') means the
# entire sub-dict; 'cutouts:shell' / 'cutouts:lid' are the cut-out entries of that part.
STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "params": ("version", "kit_id", "board", "material", "wall_t", "floor_t", "lid_t", "clearance", "shrink_comp",
               "corner_r", "standoff", "lid", "hooks", "vents", "ports", "cutouts", "lid_lift"),
    "reference": ("board",),
    # ports.auto is NOT in the validation table, but it decides which connectors protrude into the cavity margins, so a
    # change must reach the shell (reported as a deviation: the table is otherwise unchanged).
    "shell": ("board", "material", "wall_t", "floor_t", "clearance", "shrink_comp", "corner_r", "standoff.h", "lid.gap",
              "ports.auto", "ports.include", "ports.exclude", "ports.gap", "ports.margin", "ports.overrides",
              "ports.plug_clearance"),
    "standoffs": ("standoff",),
    "ports": ("ports", "cutouts:shell", "wall_t"),
    "lid": ("lid", "lid_t", "wall_t", "clearance", "shrink_comp"),
    "vents": ("vents", "cutouts:lid"),
    "hooks": ("hooks", "lid.fit", "lid.lip_t", "lid.lip_h", "wall_t", "ports"),
    "assembly": ("lid_lift", "colors"),
}
STAGE_DEPS: dict[str, tuple[str, ...]] = {
    "params": (), "reference": (), "shell": (), "standoffs": ("shell",), "ports": ("shell",), "lid": ("shell",),
    "vents": ("shell", "lid"), "hooks": ("shell", "lid", "ports"), "assembly": ("shell", "lid"),
}


def _canon(v: Any) -> Any:
    """Canonical JSON-able form: floats rounded to 9 decimals, integral floats become ints, dict keys sorted at dump."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        r = round(v, 9)
        return int(r) if r.is_integer() else r
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    return str(v)


def _stage_view(d: dict, paths: Iterable[str]) -> dict:
    out: dict[str, Any] = {}
    for p in paths:
        if p.startswith("cutouts:"):
            part = p.split(":", 1)[1]
            cuts = d.get("cutouts", [])
            # a malformed value (not a list) is hashed as it is: an invalid spec is never built, but it must not crash
            out[p] = ([e for e in cuts if isinstance(e, dict) and e.get("part") == part]
                      if isinstance(cuts, (list, tuple)) else cuts)
        else:
            try:
                out[p] = _get_path(d, p)
            except (KeyError, TypeError):
                out[p] = None
    return out


def _digest(obj: Any) -> str:
    text = json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def spec_hash(spec: "EnclosureSpec | dict", stage: "str | None" = None) -> str:
    """12 hex characters identifying ``spec`` (``stage=None``: the whole spec) or one stage of it.

    A stage hash covers the stage's OWN fields (``STAGE_FIELDS``) plus the hashes of the stages it depends on
    (``STAGE_DEPS``, transitively), so editing a field changes exactly the hashes of the stages that depend on it.
    Numbers are canonicalised (``2`` == ``2.0``, float noise below 1e-9 ignored) and the result does not depend on key
    order, so it is stable across runs and machines. Raises ``ValueError`` for an unknown stage.
    """
    d = to_dict(_as_spec(spec))
    if stage is None:
        return _digest({"spec": d})
    if stage not in STAGES:
        raise ValueError(f"stage {stage!r} is not a kit stage; use one of {', '.join(STAGES)}")
    return _digest({"stage": stage, "own": _stage_view(d, STAGE_FIELDS[stage]),
                    "deps": {dep: spec_hash(d, dep) for dep in STAGE_DEPS[stage]}})


def stages_affected_by(path: str) -> list[str]:
    """The stages whose hash changes when the spec field ``path`` (dotted, for example ``hooks.length``) changes.

    Derived from ``STAGE_FIELDS`` / ``STAGE_DEPS``; ``cutouts`` counts for both ``cutouts:shell`` and ``cutouts:lid``.
    """
    def owns(stage: str) -> bool:
        for f in STAGE_FIELDS[stage]:
            base = f.split(":", 1)[0]
            if path == base or path.startswith(base + ".") or base.startswith(path + "."):
                return True
        return False

    out = []
    for stage in STAGES:
        seen: set[str] = set()
        todo = [stage]
        while todo:
            cur = todo.pop()
            if cur in seen:
                continue
            seen.add(cur)
            todo.extend(STAGE_DEPS[cur])
        if any(owns(s) for s in seen):
            out.append(stage)
    return out
