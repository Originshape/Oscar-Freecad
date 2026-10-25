"""The ``Params`` spreadsheet: one table of every driving dimension.

Layout: sheet ``Params`` (object Name and Label), row 1 the header ``name | value | unit | source | note``, rows from 2;
column A the name, column B the VALUE aliased to the name, C the unit text, D the source, E the note. Objects bind to
a row with ``obj.setExpression("Length", "Params.wall_t*2")`` (``Params.bind`` prefixes the sheet name for you).

Rules this module enforces:

* alias: ``^[a-z][a-z0-9_]{2,31}$``, not cell-address-like, not a reserved name; the sheet's own ``setAlias`` is the
  AUTHORITATIVE judge (it rejects unit names such as ``min``, ``psi``, ``rad``) and is probed on the real sheet;
* numbers are written as FORMULAS (``=2.4 mm``, ``=-1.5 mm``, ``=4``): FreeCAD stores a cell whose text starts with ``-``
  or ``+`` as a STRING, and ``2,5 mm`` parses locale-dependently (both verified); every write is read back with
  ``sheet.get`` and a string result raises ``ParamsError``;
* reads go through ``float(getattr(v, "Value", v))``: unit cells give a ``Quantity``, unitless cells a plain number;
* ``ensure_sheet`` is the ONE producer: it gets or creates the object whose ``Name`` is ``Params`` (a second sheet created
  under that name would become ``Params001`` and every ``Params.<alias>`` expression would silently bind to the first).

``Params`` is GUI-thread-only (it touches an ``App.Document``); this module imports no FreeCAD (everything goes through
the document object you pass), so it is importable anywhere and its pure helpers (``valid_alias``, ``infer_unit``,
``DEFAULT_PARAMS``) work on any thread.
"""

import difflib
import math
import re
from typing import Any, Iterable

from . import spec as _spec

SHEET_NAME = "Params"
HEADER = ("name", "value", "unit", "source", "note")
SOURCES = ("user_requirement", "datasheet", "step_measurement", "standard", "derived", "assumption")
UNITS = ("mm", "deg", "")

# Extra guard only: the sheet's own setAlias is authoritative.
RESERVED = frozenset({"min", "max", "abs", "sin", "cos", "tan", "exp", "log", "sqrt", "pi", "deg", "rad", "mod", "pow",
                      "round", "ceil", "floor", "true", "false", "not", "and", "or", "if", "mm", "cm", "psi", "bar",
                      "ohm", "day", "sec"})
# Names of 3+ lower-case characters that FreeCAD 1.1.3's Spreadsheet.setAlias rejects (measured by brute force over all
# 3-character and 4-letter names; they are unit names or literals). A fast pre-filter for the thread-safe check; the
# probe on the sheet still decides, and test_kit_params re-checks every entry against the running FreeCAD.
SHEET_REJECTED = frozenset({"bar", "cal", "cft", "cwt", "deg", "gon", "ksi", "lbf", "lbm", "mil", "min", "mol", "mph",
                            "psi", "rad", "kcal", "mbar", "mmol", "sqft", "thou", "true"})

ALIAS_RE = re.compile(r"^[a-z][a-z0-9_]{2,31}$")
ADDRESS_RE = re.compile(r"^[a-z]{1,2}[0-9]+$")
_SCRATCH_CELL = "ZZ16384"          # alias probe cell: far outside the table (the sheet reaches column ZZ, row 16384)
_VALUE_RE = re.compile(r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(mm|deg)?\s*$")
_LITERAL_FORMULA_RE = re.compile(r"^=\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*(?:mm|deg)?\s*$")


class ParamsError(ValueError):
    """A Params operation was refused or failed; the message follows the model-facing error style."""


# --------------------------------------------------------------------------------------------------------------------
# names and units (pure)
# --------------------------------------------------------------------------------------------------------------------

def suggest_alias(name: str) -> str:
    """The closest valid alias for ``name`` ('Wall T' -> 'wall_t'), or '' when nothing sensible exists."""
    s = re.sub(r"[^a-z0-9_]+", "_", str(name).strip().lower()).strip("_")
    s = re.sub(r"_+", "_", s)
    if s and not s[0].isalpha():
        s = "p_" + s
    s = s[:32].rstrip("_")
    if len(s) < 3:
        return ""
    return s if _alias_problem(s) == "" else ""


def _alias_problem(name: Any) -> str:
    """'' when ``name`` passes the local alias rules, else the reason (without the 'Params:' prefix)."""
    if not isinstance(name, str) or not ALIAS_RE.match(name):
        hint = suggest_alias(name) if isinstance(name, str) else ""
        return (f"{name!r} is not a valid parameter name; use lower_snake_case, 3 to 32 characters, starting with a "
                f"letter (for example wall_t)" + (f"; try {hint!r}" if hint else ""))
    if ADDRESS_RE.match(name):
        return f"{name!r} looks like a spreadsheet cell address; choose a descriptive snake_case name (for example wall_t)"
    if name in RESERVED or name in SHEET_REJECTED:
        return f"{name!r} is reserved (a unit, function or constant name); choose another name (for example {name}_val)"
    return ""


def _probe_alias(sheet: Any, name: str) -> str:
    """Ask the sheet itself whether it accepts ``name`` as an alias: '' when it does, FreeCAD's message when it does not.

    Uses a scratch cell far from the table and cleans it up; no row is created. GUI-thread only (touches the sheet).
    """
    try:
        sheet.setAlias(_SCRATCH_CELL, name)
    except Exception as exc:    # noqa: BLE001 - FreeCAD raises ValueError('Invalid alias') / RuntimeError; all mean "no"
        return str(exc) or "Invalid alias"
    for cleanup in (lambda: sheet.setAlias(_SCRATCH_CELL, ""), lambda: sheet.clear(_SCRATCH_CELL)):
        try:
            cleanup()
        except Exception:    # noqa: BLE001, S110 - best effort
            pass
    return ""


def valid_alias(name: str, *, sheet: Any = None) -> "tuple[bool, str]":
    """Is ``name`` acceptable as a ``Params`` alias? Returns ``(ok, reason)``; ``reason`` is '' when ok.

    Always applies the local rules (``ALIAS_RE``, not cell-address-like, ``RESERVED`` and the measured
    ``SHEET_REJECTED`` set). Without ``sheet`` that is the whole answer (thread-safe, no FreeCAD). With a Spreadsheet
    object the sheet's own ``setAlias`` is asked as well and decides: its ``Invalid alias`` is reported verbatim; a name
    the sheet already uses for one of its cells is accepted (it is its own alias).
    """
    reason = _alias_problem(name)
    if reason:
        return False, reason
    if sheet is not None:
        try:
            if sheet.getCellFromAlias(name):
                return True, ""
        except Exception:    # noqa: BLE001 - treat as "not in use"
            pass
        probe = _probe_alias(sheet, name)
        if probe:
            return False, f"{name!r} is not accepted as an alias by the spreadsheet ({probe}); choose another name (for example {name}_val)"
    return True, ""


_UNITLESS_END = ("_count", "_pct", "_ratio", "_factor")
_UNITLESS_EQ = {"shrink_pct"}
_DEG_END = ("_deg", "angle")
_MM_END = ("_t", "_h", "_w", "_l", "_d", "_r", "_len", "_wid", "_off", "_gap", "_fit", "_dx", "_dy", "_z",
           "_pitch", "_overhang", "_margin", "_x", "_y", "_width", "_height", "_length", "_thickness", "_depth",
           "_radius", "_diameter", "_dia", "_lift", "_clearance")
_MM_EQ = {"clearance", "pitch", "overhang", "lid_lift", "corner_r"}


def infer_unit(name: str) -> str:
    """The unit implied by a parameter name when none is given: 'mm', 'deg' or '' (a count or ratio).

    Unitless: ``*_count``, ``*_pct``, ``*_ratio``, ``*_factor``, ``shrink_pct``. Degrees: ``*_deg``, ``*angle``.
    Millimetres: ``*_t *_h *_w *_l *_d *_r *_len *_wid *_off *_gap *_fit *_dx *_dy *_z`` (and the longer words
    ``*_pitch *_overhang *_margin *_x *_y *_width *_height *_length *_thickness *_depth *_radius *_diameter *_dia *_lift
    *_clearance``) or ``clearance``, ``pitch``, ``overhang``, ``lid_lift``, ``corner_r``. Anything else is unitless.
    """
    if name in _UNITLESS_EQ or name.endswith(_UNITLESS_END):
        return ""
    if name.endswith(_DEG_END):
        return "deg"
    if name in _MM_EQ or name.endswith(_MM_END):
        return "mm"
    return ""


def _fmt(x: float) -> str:
    return format(x, ".12g")


def _cell_value(name: str, value: Any, unit: "str | None", formula: "str | None",
                default_unit: "str | None" = None) -> "tuple[str, str]":
    """``(cell text, unit text)`` for one row; always a formula text (leading '='). Raises ``ParamsError``.

    The unit is, in order: the explicit ``unit``, the one written in ``value`` ('2.4 mm'), ``default_unit`` (what the
    existing row already uses), the one implied by the name (``infer_unit``).
    """
    if unit is not None and unit not in UNITS:
        raise ParamsError(f"Params: '{name}': unit {unit!r} is not supported; use 'mm', 'deg' or '' (a count).")
    fallback = default_unit if default_unit is not None else infer_unit(name)
    if formula is not None:
        if not isinstance(formula, str) or not formula.strip().lstrip("="):
            raise ParamsError(f"Params: '{name}': the formula is empty; write it like '=wall_t*2'.")
        text = formula.strip()
        return (text if text.startswith("=") else "=" + text), (unit if unit is not None else fallback)
    if isinstance(value, bool) or value is None:
        raise ParamsError(f"Params: '{name}': {value!r} is not a number; write 2.4, '2.4 mm' or a formula starting with '='.")
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ParamsError(f"Params: '{name}': {value!r} is not a finite number.")
        unit_txt = unit if unit is not None else fallback
        return "=" + _fmt(float(value)) + (f" {unit_txt}" if unit_txt else ""), unit_txt
    if isinstance(value, str):
        v = value.strip()
        if v.startswith("="):
            return v, (unit if unit is not None else fallback)
        m = _VALUE_RE.match(v)
        if not m:
            hint = "; use a dot as the decimal separator" if re.match(r"^\s*[-+]?\d+,\d", v) else ""
            raise ParamsError(f"Params: '{name}': cannot read {value!r} as a number{hint}; write 2.4, '2.4 mm' or a formula starting with '='.")
        number, written_unit = float(m.group(1)), m.group(2) or ""
        if written_unit and unit is not None and unit != written_unit:
            raise ParamsError(f"Params: '{name}': the value {value!r} is in '{written_unit}' but unit={unit!r} was given; pass one of them.")
        unit_txt = written_unit or (unit if unit is not None else fallback)
        return "=" + _fmt(number) + (f" {unit_txt}" if unit_txt else ""), unit_txt
    raise ParamsError(f"Params: '{name}': {value!r} is not a number; write 2.4, '2.4 mm' or a formula starting with '='.")


def _safe_text(s: str) -> str:
    """Text for a note / unit / source cell: a leading '=' or apostrophe must not become a formula or be swallowed."""
    return "'" + s if s[:1] in ("=", "'") else s


def _strip_quote(contents: str) -> str:
    return contents[1:] if contents.startswith("'") else contents


def _prefix_aliases(expr: str, names: "set[str]", sheet_name: str) -> str:
    """Prefix every bare reference to a parameter in ``expr`` with the sheet name ('wall_t*2' -> 'Params.wall_t*2').

    Identifiers that follow a '.', are followed by '(' (functions) or '.' (object references), or are not parameter
    names (units, constants) are left alone.
    """
    def repl(m: "re.Match[str]") -> str:
        return f"{sheet_name}.{m.group(1)}" if m.group(1) in names else m.group(1)

    return re.sub(r"(?<![\w.])([a-z][a-z0-9_]*)(?![\w(.])", repl, expr)


# --------------------------------------------------------------------------------------------------------------------
# the table of defaults and the stage map (pure data)
# --------------------------------------------------------------------------------------------------------------------

# parameter -> the first stage that must be rebuilt when its value changes (count / shape parameters are not
# expression-bound: editing the cell alone changes nothing until the stage is rebuilt; 7.4, 2.6)
STAGE_OF_PARAM: dict[str, str] = {
    "vent_count": "vents", "vent_slot_w": "vents", "vent_slot_l": "vents", "vent_pitch": "vents",
    "hook_count": "hooks", "hook_w": "hooks", "hook_t": "hooks", "hook_overhang": "hooks", "hook_len": "hooks",
    "hook_barb_h": "hooks", "hook_ramp_deg": "hooks", "hook_retention_deg": "hooks", "window_margin": "hooks",
    "port_margin": "ports",
}

# parameter -> the first stage to REBUILD although expressions already follow the cell. These values decide WHICH features exist (a
# skirt notch for a connector that now reaches into the skirt, an opening that now merges with the rim) or feed numbers frozen at
# build time (the barb profile and the catch pocket height depend on ``lid_fit``; the connector windows on ``pcb_t``), so a plain
# recompute leaves a model that differs from a fresh build with the same value (e.g. an Ethernet jack blocked by a taller skirt).
# ``pcb_t`` also reshapes the nominal reference (``reference`` is rebuilt on its own: nothing is built on it). Not part of
# ``STAGE_OF_PARAM``: these rows stay bound, so ``needs_rebuild`` and the "(rebuild: set_parameter)" marks do not apply to them.
STRUCTURAL_STAGE_OF_PARAM: dict[str, str] = {
    "pcb_t": "ports", "tallest_z": "ports", "lid_gap": "ports", "lip_h": "lid", "lid_fit": "hooks",
}

_BOARD_DEFAULTS = (
    ("board_len", 85.0, "mm", "datasheet", "board length (long edge, +x)"),
    ("board_w", 56.0, "mm", "datasheet", "board width (short edge, +y)"),
    ("pcb_t", 1.4, "mm", "assumption", "PCB thickness"),
    ("hole_off", 3.5, "mm", "datasheet", "first mounting hole offset from the board corner"),
    ("hole_dx", 58.0, "mm", "datasheet", "mounting hole spacing along x"),
    ("hole_dy", 49.0, "mm", "datasheet", "mounting hole spacing along y"),
    ("hole_d", 2.75, "mm", "datasheet", "mounting hole diameter"),
    ("tallest_z", 17.0, "mm", "assumption", "top of the tallest component, measured from the PCB bottom"),
    ("bottom_z", -1.5, "mm", "assumption", "lowest point of the board underside (negative = below the PCB bottom)"),
    ("pr_nx", 2.5, "mm", "assumption", "largest connector protrusion past the board edge on the -x face"),
    ("pr_px", 2.0, "mm", "assumption", "largest connector protrusion past the board edge on the +x face"),
    ("pr_ny", 1.0, "mm", "assumption", "largest connector protrusion past the board edge on the -y face"),
    ("pr_py", 0.0, "mm", "assumption", "largest connector protrusion past the board edge on the +y face"),
)
_DESIGN_DEFAULTS = {
    "wall_t": 2.0, "floor_t": 2.0, "lid_t": 2.0, "clearance": 0.3, "corner_r": 3.0, "standoff_h": 4.0, "standoff_d": 6.0,
    "standoff_pilot_d": 2.2, "lid_gap": 1.0, "lip_h": 3.0, "lip_t": 1.2, "lid_fit": 0.25, "hook_count": 4, "hook_w": 6.0,
    "hook_t": 1.2, "hook_overhang": 0.8, "hook_len": 12.0, "hook_barb_h": 1.0, "hook_ramp_deg": 30,
    "hook_retention_deg": 45, "window_margin": 0.3, "vent_count": 9, "vent_slot_w": 2.0, "vent_slot_l": 24.0,
    "vent_pitch": 4.0, "port_margin": 0.4, "port_gap": 0.3, "lid_lift": 0.0,
}


def _default_rows() -> list[dict]:
    rows = [{"name": n, "value": v, "unit": u, "source": s, "note": note, "formula": None}
            for n, v, u, s, note in _BOARD_DEFAULTS]
    for name, (_path, unit, note) in _spec.DESIGN_PARAMS.items():
        rows.append({"name": name, "value": _DESIGN_DEFAULTS[name], "unit": unit, "source": "assumption",
                     "note": note, "formula": None})
        if name == "clearance":
            rows.append({"name": "shrink_pct", "value": 0.8, "unit": "", "source": "assumption",
                         "note": "ABS shrinkage in percent", "formula": None})
    env = {r["name"]: r["value"] for r in rows}
    for name, formula, unit in _spec.DERIVED_FORMULAS:
        env[name] = _spec.eval_formula(formula, env)
        rows.append({"name": name, "value": round(env[name], 9), "unit": unit, "source": "derived", "note": "",
                     "formula": formula})
    return rows


# Every row of the default enclosure table (the Pi 4 / ABS numbers), in dependency order: board rows,
# design rows, derived rows. Keys: name, value (the default's number), unit, source, note, formula (None for inputs, the
# '=...' text for derived rows). ``spec.param_rows(spec, board)`` builds the same table from a live spec and board.
DEFAULT_PARAMS: list[dict] = _default_rows()


# --------------------------------------------------------------------------------------------------------------------
# the sheet
# --------------------------------------------------------------------------------------------------------------------

def _resolve_doc(doc: Any) -> Any:
    if isinstance(doc, str):
        try:
            import FreeCAD as App   # noqa: PLC0415 - lazy: this module must import without FreeCAD
        except ImportError:
            raise ParamsError(f"Params: cannot open document {doc!r}: FreeCAD is not available.") from None
        try:
            return App.getDocument(doc)
        except Exception:    # noqa: BLE001
            raise ParamsError(f"Params: document {doc!r} is not open.") from None
    if not hasattr(doc, "getObject"):
        raise ParamsError(f"Params: expected a FreeCAD document or its name, got {type(doc).__name__}.")
    return doc


class Params:
    """The ``Params`` spreadsheet of one document. GUI-thread only.

    ``Params(doc)`` opens an existing sheet (``ParamsError`` when missing); ``Params(doc, create=True)`` and
    ``ensure_sheet(doc)`` get-or-create it. ``doc`` is an ``App.Document`` (or its name). Rows are found by the alias
    on column B, so a sheet written by the feature-plan builder (name | value, no header) is readable too.
    """

    def __init__(self, doc: Any, sheet_name: str = SHEET_NAME, create: bool = False) -> None:
        self._doc = _resolve_doc(doc)
        self.sheet_name = sheet_name
        sheet = self._doc.getObject(sheet_name)
        if sheet is None:
            if not create:
                raise ParamsError(f"Params: document '{self._doc.Name}' has no sheet named '{sheet_name}'; create it with ensure_sheet or Params(doc, create=True).")
            sheet = self._create_sheet(sheet_name)
        elif getattr(sheet, "TypeId", "") != "Spreadsheet::Sheet":
            raise ParamsError(f"Params: object '{sheet_name}' in document '{self._doc.Name}' is a {getattr(sheet, 'TypeId', '?')}, not a Spreadsheet::Sheet; rename or delete it.")
        self._sheet = sheet
        if create:
            self._ensure_header()

    # --- plumbing ---------------------------------------------------------------------------------------------------
    @property
    def doc(self) -> Any:
        """The document this sheet lives in."""
        return self._doc

    @property
    def sheet(self) -> Any:
        """The ``Spreadsheet::Sheet`` object."""
        return self._sheet

    def _create_sheet(self, name: str) -> Any:
        sheet = self._doc.addObject("Spreadsheet::Sheet", name)
        if sheet.Name != name:
            actual = sheet.Name
            self._doc.removeObject(actual)
            raise ParamsError(f"Params: FreeCAD named the new sheet '{actual}' instead of '{name}' (another object already uses that name, so every Params.<alias> expression would bind to the wrong sheet); rename or delete the other object.")
        try:
            sheet.Label = name
        except Exception:    # noqa: BLE001, S110 - a Label clash is cosmetic; the Name is what expressions use
            pass
        return sheet

    def _tag(self, kit_id: str) -> None:
        """Give the sheet the kit's tag properties unless it already carries them."""
        wanted = {"OscarKit": kit_id, "OscarStage": "params", "OscarRole": "params", "OscarPart": "", "OscarSpecHash": ""}
        for prop, value in wanted.items():
            if prop not in self._sheet.PropertiesList:
                self._sheet.addProperty("App::PropertyString", prop, "Oscar", f"kit tag: {prop}")
                setattr(self._sheet, prop, value)
            elif prop == "OscarKit" and not getattr(self._sheet, prop):
                setattr(self._sheet, prop, value)

    def _cells(self) -> list[str]:
        try:
            return list(self._sheet.getNonEmptyCells())
        except Exception:    # noqa: BLE001
            return []

    def _rows(self) -> "dict[str, int]":
        """alias -> row number of every parameter row (a cell on column B that carries an alias)."""
        rows: dict[str, int] = {}
        for addr in self._cells():
            m = re.match(r"^B(\d+)$", addr)
            if m:
                alias = self._sheet.getAlias(addr)
                if alias:
                    rows[alias] = int(m.group(1))
        return rows

    def _current_unit(self, name: str, row: int) -> "str | None":
        """The unit an existing row uses, or None when it is unknown (then the name decides).

        The unit column wins when it names a unit; an empty column falls back to the type of the value now in the
        cell (a Length quantity is 'mm', an Angle 'deg', a plain number ''), because a sheet written by other code may
        have no unit column and silently turning '2 mm' into a bare '3' would break every Length binding.
        """
        column = self._text(f"C{row}").strip()
        if column in ("mm", "deg"):
            return column
        try:
            v = self._sheet.get(name)
        except Exception:    # noqa: BLE001
            return None
        if isinstance(v, str):
            return None
        if not hasattr(v, "Unit"):
            return column if column in UNITS else ""
        kind = str(getattr(v.Unit, "Type", ""))
        if kind == "Length":
            return "mm"
        if kind == "Angle":
            return "deg"
        return "" if not kind else None

    def _text(self, cell: str) -> str:
        return _strip_quote(self._sheet.getContents(cell) or "")

    def _write_text(self, cell: str, text: str) -> None:
        if text:
            self._sheet.set(cell, _safe_text(text))
        else:
            self._sheet.clear(cell)

    def _ensure_header(self) -> None:
        if not self._cells():
            for col, label in zip("ABCDE", HEADER):
                self._sheet.set(f"{col}1", label)

    def _next_row(self) -> int:
        last = 1
        for addr in self._cells():
            m = re.match(r"^[A-Z]+(\d+)$", addr)
            if m:
                last = max(last, int(m.group(1)))
        return last + 1

    def _snapshot(self, row: int) -> "dict[str, str]":
        return {col: self._sheet.getContents(f"{col}{row}") or "" for col in "ABCDE"}

    def _restore(self, row: int, prev: "dict[str, str] | None") -> None:
        """Undo a failed write: put the previous cell contents back, or remove the row when it was new.

        Every step is attempted on its own (the alias may never have been set); the caller re-raises the original error.
        """
        def attempt(fn: Any, *args: Any) -> None:
            try:
                fn(*args)
            except Exception:    # noqa: BLE001, S110 - best effort
                pass

        if prev is None:
            attempt(self._sheet.setAlias, f"B{row}", "")
        for col in "ABCDE":
            cell = f"{col}{row}"
            if prev is None or not prev[col]:
                attempt(self._sheet.clear, cell)
            else:
                attempt(self._sheet.set, cell, prev[col])
        attempt(self._sheet.recompute)

    def _read(self, name: str) -> float:
        """The value of ``name``: raises ``ParamsError`` for text results and unevaluable cells; may be inf."""
        if getattr(self._sheet, "MustExecute", False):
            self._sheet.recompute()
        try:
            v = self._sheet.get(name)
        except Exception as exc:    # noqa: BLE001
            raise ParamsError(f"Params: '{name}' could not be evaluated ({exc}); define the parameters it refers to first.") from None
        if isinstance(v, str):
            raise ParamsError(f"Params: '{name}' was stored as text, not a number; write it as '=<value> <unit>'")
        return float(getattr(v, "Value", v))

    # --- public API ---------------------------------------------------------------------------------------------------
    def names(self) -> list[str]:
        """The parameter names (aliases), in row order."""
        return [n for n, _r in sorted(self._rows().items(), key=lambda kv: kv[1])]

    def values(self) -> "dict[str, float]":
        """``{alias: value}`` for every parameter row that evaluates to a finite number, in ONE pass over the sheet.

        Rows whose cell is text or cannot be evaluated are left out (``get`` reports those one by one). Used by the
        kit builders, which read the whole table once per stage instead of calling ``get`` for each of ~60 names.
        """
        if getattr(self._sheet, "MustExecute", False):
            self._sheet.recompute()
        out: dict[str, float] = {}
        for name in self._rows():
            try:
                v = self._sheet.get(name)
            except Exception:    # noqa: BLE001 - an unevaluable cell is simply not a value
                continue
            if isinstance(v, str):
                continue
            x = float(getattr(v, "Value", v))
            if math.isfinite(x):
                out[name] = x
        return out

    def sources(self) -> "dict[str, str]":
        """``{alias: source}`` (column D text) for every parameter row; '' where the cell is empty."""
        return {name: self._text(f"D{row}") for name, row in self._rows().items()}

    def get(self, name: str) -> float:
        """The numeric value of ``name`` as a float (``float(getattr(v, 'Value', v))``: unit cells give a Quantity, unitless a number).

        Raises ``ParamsError`` for an unknown name (with 'similar' suggestions), for a cell stored as text and for a
        cell that cannot be evaluated. A formula that divides by zero reads as ``inf`` (FreeCAD does not raise).
        """
        rows = self._rows()
        if name not in rows:
            similar = difflib.get_close_matches(str(name), list(rows), n=3, cutoff=0.6)
            hint = f" (similar: {', '.join(similar)})" if similar else ""
            raise ParamsError(f"Params: {name!r} is not a parameter in sheet '{self.sheet_name}'{hint}; parameters: {', '.join(sorted(rows)[:40])}{' ...' if len(rows) > 40 else ''}.")
        return self._read(name)

    def set(self, name: str, value: Any, *, unit: "str | None" = None, source: str = "assumption", note: str = "",
            formula: "str | None" = None) -> None:
        """Create or update the row ``name``.

        ``value`` is a number, a string with an optional unit ('2.4 mm', '45 deg', '4') or a formula starting with '='
        (kept as given). ``formula`` (or a value starting with '=') makes a derived row. With ``unit`` omitted the unit
        comes from the text, else from the row's current unit, else from the name (``infer_unit``). Numeric cells are written as formulas (``=2.4 mm``) and
        read back; a string result raises ``ParamsError`` (``Params: '<name>' was stored as text, not a number; write it
        as '=<value> <unit>'``) and the row is restored. A bad alias, source or unit raises ``ParamsError`` before
        anything is written. The sheet is recomputed (not the whole document; the caller recomputes dependents).
        """
        ok, reason = valid_alias(name, sheet=self._sheet)
        if not ok:
            raise ParamsError(f"Params: {reason}.")
        if source not in SOURCES:
            raise ParamsError(f"Params: '{name}': source {source!r} is not valid; use one of {', '.join(SOURCES)}.")
        row = self._rows().get(name)
        text, unit_txt = _cell_value(name, value, unit, formula, None if row is None else self._current_unit(name, row))
        prev = None if row is None else self._snapshot(row)
        if row is None:
            self._ensure_header()
            row = self._next_row()
        sheet = self._sheet
        try:
            sheet.set(f"B{row}", text)
            if prev is None:
                sheet.setAlias(f"B{row}", name)
            sheet.set(f"A{row}", name)
            self._write_text(f"C{row}", unit_txt)
            self._write_text(f"D{row}", source)
            self._write_text(f"E{row}", note)
            sheet.recompute()
            x = self._read(name)
            if not math.isfinite(x):
                raise ParamsError(f"Params: '{name}' evaluates to {x}; check the value {text!r} (division by zero or an overflow).")
        except Exception:
            self._restore(row, prev)
            raise

    def bind(self, obj: Any, prop: str, expr: str) -> None:
        """Bind ``obj.prop`` to a Params expression: ``bind(obj, 'Length', 'wall_t*2')`` sets ``Params.wall_t*2``.

        Bare parameter names in ``expr`` get the sheet name as prefix (``Params.`` ...); anything already qualified, any
        function call and any unit stays as written. A leading '=' is ignored. Raises ``ParamsError`` when FreeCAD
        rejects the binding; the caller validates it by recomputing (``bind_expression`` does).
        """
        text = str(expr).strip()
        if text.startswith("="):
            text = text[1:].strip()
        if not text:
            raise ParamsError(f"Params: cannot bind {getattr(obj, 'Name', '?')}.{prop} to an empty expression; write it like 'wall_t*2'.")
        final = _prefix_aliases(text, set(self._rows()), self._sheet.Name)
        try:
            obj.setExpression(prop, final)
        except Exception as exc:    # noqa: BLE001
            raise ParamsError(f"Params: cannot bind {getattr(obj, 'Name', '?')}.{prop} to '{final}': {exc}") from None

    def _bound_counts(self, names: Iterable[str]) -> "dict[str, int]":
        """For each alias, how many ExpressionEngine entries in the document contain ``<sheet>.<alias>`` (word boundary)."""
        exprs: list[str] = []
        for obj in self._doc.Objects:
            if obj is self._sheet:
                continue
            for entry in getattr(obj, "ExpressionEngine", None) or []:
                try:
                    exprs.append(str(entry[1]))
                except (IndexError, TypeError):
                    continue
        sheet = re.escape(self._sheet.Name)
        out = {}
        for n in names:
            pat = re.compile(rf"\b{sheet}\.{re.escape(n)}\b")
            out[n] = sum(1 for e in exprs if pat.search(e))
        return out

    def table(self) -> list[dict]:
        """One dict per row, the shape of ``cad.health.read_param_table`` rows.

        ``{"name", "value" (float or None for a text / unevaluable cell), "unit", "expr" ('' for a plain number, the
        '=...' text for a real formula), "source", "note", "cell" ('B14'), "bound" (ExpressionEngine entries that use
        it), "needs_rebuild" (a count / shape parameter: editing it needs ``STAGE_OF_PARAM`` rebuilt)}``.
        """
        rows = self._rows()
        bound = self._bound_counts(rows)
        out = []
        for name, row in sorted(rows.items(), key=lambda kv: kv[1]):
            contents = self._sheet.getContents(f"B{row}") or ""
            try:
                value: "float | None" = self._read(name)
            except ParamsError:
                value = None
            expr = contents if contents.startswith("=") and not _LITERAL_FORMULA_RE.match(contents) else ""
            out.append({"name": name, "value": value, "unit": self._text(f"C{row}"), "expr": expr,
                        "source": self._text(f"D{row}"), "note": self._text(f"E{row}"), "cell": f"B{row}",
                        "bound": bound[name], "needs_rebuild": name in STAGE_OF_PARAM})
        return out

    @staticmethod
    def valid_alias(name: str, *, sheet: Any = None) -> "tuple[bool, str]":
        """See the module function ``valid_alias``; callable on the class (no sheet: local rules) or with ``sheet=``."""
        return valid_alias(name, sheet=sheet)


def ensure_sheet(doc: Any, name: str = SHEET_NAME, *, kit_id: str = "enc1") -> Params:
    """Get or create THE ``Params`` sheet of ``doc`` and return its ``Params``. GUI-thread only.

    The one producer every writer of the sheet goes through (``set_parameters``, ``measure_reference_part``, the
    ``params`` stage, ``set_parameter``): it finds the object whose ``Name`` is ``name`` or creates it, writes the header
    row, tags it (``OscarKit`` = ``kit_id``, ``OscarStage``/``OscarRole`` = ``params``; existing tags are kept) and
    verifies ``sheet.Name == name``. Raises ``ParamsError`` when FreeCAD would name the new sheet differently (``Params001``,
    after which every ``Params.<alias>`` expression would bind to the wrong sheet) or when ``name`` is not a sheet.
    """
    params = Params(doc, name, create=True)
    params._tag(kit_id)
    return params
