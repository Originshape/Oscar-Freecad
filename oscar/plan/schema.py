"""FeaturePlan schema v1 and v2 — the CAD-neutral contract between Oscar and the builder.

A FeaturePlan is what Oscar writes for a structured part (from the user's request,
a datasheet or a standard) and what a builder translates into a native editable CAD model. The vocabulary is
deliberately generic parametric-CAD (sketch / pad / pocket / revolution /
hole over rectangle / circle / slot / segment-arc profiles) — nothing in a
plan is FreeCAD-specific. ``target`` names the backend the plan is meant
for ("freecad" in v1); a Fusion/SolidWorks/Onshape builder would consume
the identical schema, which is the architectural seam that lets Oscar
drive other CAD kernels later.

Rules enforced by ``validate`` (all errors are actionable strings — the
model gets them fed back for repair):

* every dimensional value is a parameter NAME (string), never a literal;
  literals are reserved for non-dimensional data (directions, flags);
* every name a feature uses exists in ``parameters`` (reference closure),
  every op carries its required fields, dimensions that are lengths,
  diameters, depths or radii are positive, angles are in range, and a
  key the schema does not know is an error with the nearest known key
  suggested ("depht" → "did you mean 'depth'?");
* every parameter carries ``evidence`` (where the number comes from + a real
  detail) or an explicit ``assumption`` note — unevidenced numbers do not
  validate; a derived parameter instead carries ``expr`` (a formula over
  other parameters, dependency-ordered and cycle-checked);
* feature ids are unique identifiers and ordered; extrusion ops must
  reference an earlier sketch;
* the op/profile vocabulary is a closed set — anything else is rejected
  by name.

The plan is a plain dict throughout (JSON-serializable); this module validates dicts and offers small accessors, but does
not impose dataclasses on consumers.

Schema v2 (``schema_version`` 2) is a strict superset: every v1 plan keeps
validating and building unchanged. Ops live in ONE registry table (``OPS``:
op -> keys, required fields, dimensional fields with their range rule,
references to earlier features, extra validator, build kind) that the
validator, ``dimension_refs`` and the builder all read; v2 adds fillet,
chamfer, thickness (shell), linear / polar pattern, mirror, boss, sketches
with several ``profiles``, ``parts`` (one PartDesign Body each, never fused)
with an ``assembly`` of named placements and informational ``references``,
and a ``source`` ({kind, ref}) on parameters. The v2 vocabulary and its
validators are in ``_schema_v2`` (registered at the end of this module).
"""

import difflib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

SCHEMA_VERSION = 1             # the original vocabulary
SCHEMA_VERSION_V2 = 2          # op registry, parts, patterns, dressing, sources
SUPPORTED_VERSIONS = (SCHEMA_VERSION, SCHEMA_VERSION_V2)

VALID_TARGETS = ("freecad",)   # other CAD backends register here
VALID_UNITS = ("mm",)          # the builder interprets every number as millimetres

SKETCH_SHAPES = ("rectangle", "circle", "slot", "profile")
FEATURE_OPS = ("sketch", "pad", "pocket", "revolution", "hole")   # the v1 vocabulary

# keys each level of a plan may carry; anything else is an error (features,
# profiles, planes, segments, parameters) or a warning (top level, frame)
PLAN_KEYS = ("schema_version", "id", "part_class", "units", "target", "frame",
             "parameters", "features", "source", "notes", "patch_log")
PLAN_KEYS_V2 = PLAN_KEYS + ("parts", "assembly", "references")
PARAM_KEYS = ("value", "expr", "evidence", "assumption", "note", "source")
# where a parameter's number comes from (v2 ``source`` = {kind, ref})
SOURCE_KINDS = ("user_requirement", "datasheet", "step_measurement",
                "standard", "derived", "assumption")
SOURCE_KEYS = ("kind", "ref", "note")
FRAME_KEYS = ("origin", "planes", "axes", "note")
PLANE_KEYS = ("origin", "normal", "x_dir", "frame_plane", "note")
AXIS_KEYS = ("direction", "point", "note")
FEATURE_KEYS = {
    "sketch": ("id", "op", "note", "plane", "profile"),
    "pad": ("id", "op", "note", "sketch", "length", "midplane", "invert"),
    "pocket": ("id", "op", "note", "sketch", "depth", "through", "midplane",
               "invert"),
    "revolution": ("id", "op", "note", "sketch", "axis", "angle", "midplane",
                   "invert"),
    "hole": ("id", "op", "note", "plane", "position", "diameter", "through",
             "depth", "exact", "counterbore", "countersink",
             "cb_diameter", "cb_depth", "cs_diameter", "cs_angle"),
}
PROFILE_KEYS = {
    "rectangle": ("shape", "width", "height", "center", "note"),
    "circle": ("shape", "diameter", "center", "note"),
    "slot": ("shape", "length", "width", "center", "angle", "angle_deg",
             "note"),
    "profile": ("shape", "start", "segments", "note"),
}
SEGMENT_KEYS = ("type", "to", "center", "ccw", "radius", "length", "diameter",
                "note")

# fields of each op whose values must be parameter names
_DIMENSIONAL_FIELDS = {
    "sketch": (),  # validated per-profile below
    "pad": ("length",),
    "pocket": ("depth",),
    "revolution": ("angle",),
    "hole": ("diameter", "depth", "x", "y"),
}
_NON_DIMENSIONAL = {"plane", "sketch", "id", "op", "direction", "through",
                    "midplane", "exact", "counterbore",
                    "countersink", "cb_diameter", "cb_depth", "cs_diameter",
                    "cs_angle", "axis", "profile", "invert", "note"}
_FLAG_FIELDS = ("through", "midplane", "invert", "exact")


# ---------------------------------------------------------------------------
# the op registry: ONE table the validator, dimension_refs and the builder read


@dataclass
class OpSpec:
    """Everything the toolchain knows about one feature op.

    ``kind``: ``"sketch"`` (makes a profile, no solid) or ``"solid"``.
    ``post``: how the builder judges the Body after the feature -
    ``"additive"`` (volume must grow), ``"subtractive"`` (must shrink),
    ``"dress"`` (volume must change, the bounding box must not grow),
    ``"any"`` (volume must change, no box rule), ``"follow"`` (a pattern or
    mirror: judged like its original, ``of``), ``None`` for a sketch.
    ``keys``: every key the feature may carry (an unknown one is an error with
    the nearest known key suggested). ``required``: fields that must be
    present. ``dims``: field -> range rule (``positive``, ``angle360``,
    ``angle180``, ``count``) of the fields that must name a parameter.
    ``refs``: field -> ``"sketch"`` / ``"solid"`` / ``"any"``: the field must
    name an EARLIER feature of that kind in the same part. ``nondim``: keys
    that legitimately hold something other than a dimension. ``check``: extra
    ``(feat, fid, vc)`` validator appending to ``vc.errors`` / ``vc.warnings``.
    ``builder``: ``(doc, body, feat, plan, built, ctx) -> (object, info)``,
    attached by the builder modules at import. ``since``: first schema
    version that knows the op."""

    name: str
    kind: str
    post: str | None
    keys: tuple
    required: tuple = ()
    dims: dict = field(default_factory=dict)
    refs: dict = field(default_factory=dict)
    nondim: tuple = ()
    check: Callable | None = None
    builder: Callable | None = None
    since: int = 1


OPS: dict[str, OpSpec] = {}


def register_op(spec: OpSpec) -> OpSpec:
    """Add (or replace) an op of the registry; returns it."""
    OPS[spec.name] = spec
    return spec


def attach_builder(op: str, fn: Callable) -> None:
    """Attach the translator of ``op`` (called by the builder modules)."""
    OPS[op].builder = fn


def ops_for(version: Any) -> tuple[str, ...]:
    """Names of the ops a plan of ``version`` may use (registry order)."""
    try:
        v = int(version)
    except (TypeError, ValueError):
        v = SCHEMA_VERSION
    return tuple(n for n, o in OPS.items() if o.since <= v)


@dataclass
class VCtx:
    """What an op validator sees: the plan pieces and the sink for messages.
    ``earlier`` maps the ids of the features before this one IN THE SAME PART
    to their ops; ``version`` is the plan's schema_version."""

    plan: dict
    params: dict
    frame: dict
    axes: dict
    errors: list
    warnings: list
    earlier: dict = field(default_factory=dict)
    version: int = 1

    def sketches(self) -> set:
        return {i for i, o in self.earlier.items() if o in OPS and OPS[o].kind == "sketch"}

MAX_PATTERN_COUNT = 500        # occurrences of a linear / polar pattern

# smallest number of characters an evidence detail / assumption note needs
_MIN_NOTE_CHARS = 8
_TRIVIAL_NOTES = {"n/a", "na", "none", "null", "nil", "tbd", "todo", "unknown",
                  "measured", "see above", "yes", "ok",
                  "assumed", "assumption", "estimate", "estimated", "guess"}

_FEATURE_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")



def _is_param_ref(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0


def _num(value: Any) -> bool:
    """A finite real number — NaN/inf pass isinstance but hang or crash the
    CAD kernel (an int too large for a float counts as non-finite)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _norm3(v: list) -> float:
    return math.sqrt(sum(float(c) * float(c) for c in v))


def suggest(key: str, allowed) -> str:
    """`` — did you mean 'x'?`` for the nearest allowed key, else the empty
    string (used by every unknown-key error)."""
    hit = difflib.get_close_matches(str(key), [str(a) for a in allowed],
                                    n=1, cutoff=0.6)
    return f"; did you mean {hit[0]!r}?" if hit else ""


def _unknown_keys(obj: dict, allowed, what: str, errors: list[str],
                  warnings: list[str] | None = None) -> None:
    """Report every key of ``obj`` outside ``allowed`` (an error, or a
    warning when ``warnings`` is given)."""
    for key in obj:
        if key in allowed:
            continue
        msg = (f"{what}: unknown key {key!r}{suggest(key, allowed)}"
               f" (known: {', '.join(sorted(map(str, allowed)))})")
        (warnings if warnings is not None else errors).append(msg)


def _vec3(value: Any, what: str, errors: list[str]) -> bool:
    if not isinstance(value, list) or len(value) != 3 or not all(_num(v) for v in value):
        errors.append(f"{what}: expected a 3-number vector, got {value!r}")
        return False
    return True


def _plane_origin_vec(value: Any, what: str, errors: list[str],
                      params: dict) -> bool:
    """Plane origins additionally accept parameter names per coordinate so
    sketch placements can ride parameter changes (holes on a grown pad)."""
    if not isinstance(value, list) or len(value) != 3:
        errors.append(f"{what}: expected a 3-coordinate vector, got {value!r}")
        return False
    ok = True
    for i, c in enumerate(value):
        if not (_num(c) or (_is_param_ref(c) and c in params)):
            errors.append(f"{what}[{i}]: must be a number or known parameter "
                          f"name, got {c!r}")
            ok = False
    return ok


# names FreeCAD's spreadsheet rejects as aliases because they parse as unit
# symbols or constants (measured on FreeCAD 1.1.3: the sheet raises "Invalid
# alias"); compared lower-case. The builder still tries the real setAlias and
# reports FreeCAD's own verdict, this list only fails fast, in the plan.
_UNIT_ALIASES = {
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "mol", "cd", "ohm", "deg", "rad", "sr", "lm", "lx", "pa", "lb", "oz",
    "ft", "in", "mi", "yd", "psi", "kt", "ct",
    "mm", "cm", "km", "um", "nm", "kg", "mg", "min", "gon", "ma", "kn", "mpa",
    "gpa", "kpa", "ksi", "bar", "kw", "kj", "thou", "mil", "ml", "pi",
    "true", "false", "uf", "hz", "cal", "cft", "cwt", "lbf", "lbm", "mph",
    "kcal", "sqrt", "abs",
}

# unit spellings the builder reads as millimetres (compared lower-case)
_MM_UNITS = {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}


_PARAM_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_CELL_ADDRESS = re.compile(r"^[A-Z]{1,2}[0-9]{1,5}$")


def _valid_param_name(name: str) -> bool:
    """True for a name FreeCAD accepts as a spreadsheet alias (measured on
    1.1.3): ASCII letters / digits / underscore, starting with a letter, not
    a unit symbol or constant and not an upper-case cell address (``A1``)."""
    if not (isinstance(name, str) and _PARAM_NAME.match(name)):
        return False
    return name.lower() not in _UNIT_ALIASES and not _CELL_ADDRESS.match(name)


def _bad_name_hint(name: str) -> str:
    if isinstance(name, str) and _PARAM_NAME.match(name):
        if name.lower() in _UNIT_ALIASES:
            return (f"{name!r} is a unit symbol or constant FreeCAD rejects as "
                    f"a spreadsheet alias; use a descriptive snake_case name "
                    f"such as '{name.lower()}_len' or '{name.lower()}_dim'")
        return (f"{name!r} is a spreadsheet cell address, which FreeCAD "
                f"rejects as an alias; use a descriptive snake_case name such "
                f"as '{name.lower()}_dim'")
    return (f"names must be ASCII identifiers (letters, digits, underscore; "
            f"starting with a letter), got {name!r}")


from ._exprs import (ExprError, _FUNCTIONS, _eval_params, _eval_tree,  # noqa: E402,F401
                     _tokenize, derived_order, is_derived, parameter_values,
                     parse_expr)


def _check_note(text: Any, what: str, errors: list[str],
                other: str = "") -> None:
    """Evidence detail / assumption note must say something real."""
    if not isinstance(text, str):
        errors.append(f"{what} must be text, got {text!r}")
        return
    clean = text.strip()
    if (len(clean) < _MIN_NOTE_CHARS or clean.lower() in _TRIVIAL_NOTES
            or not any(ch.isalnum() for ch in clean)
            or (other and clean.lower() == other.lower())):
        errors.append(f"{what} is too thin ({text!r}): say where the number "
                      f"comes from (at least {_MIN_NOTE_CHARS} characters, e.g. "
                      f"'the user asked for an 8 mm plate' or 'ISO 273 clearance "
                      f"hole for M4')")


def _check_source(name: str, src: Any, errors: list[str]) -> None:
    """A parameter ``source`` ({kind, ref}): the kind is one of
    ``SOURCE_KINDS`` and ``ref`` says where the number comes from (the user's
    sentence, the datasheet page, the measured STEP feature, the standard's
    clause, the formula's inputs, why it is assumed)."""
    what = f"parameter {name!r}: source"
    if not isinstance(src, dict):
        errors.append(f"{what} must be {{'kind': one of {SOURCE_KINDS}, "
                      f"'ref': where it comes from}}, got {src!r}")
        return
    _unknown_keys(src, SOURCE_KEYS, what, errors)
    kind = src.get("kind")
    if kind not in SOURCE_KINDS:
        errors.append(f"{what}.kind {kind!r} not in {list(SOURCE_KINDS)}"
                      f"{suggest(kind, SOURCE_KINDS)}")
    if kind == "derived" and src.get("ref") is None:
        return                      # a formula parameter needs no further note
    if "ref" not in src:
        errors.append(f"{what}.ref is required: say where the number comes "
                      f"from (e.g. 'user: width 60 mm', 'datasheet p.3')")
    else:
        _check_note(src["ref"], f"{what}.ref", errors)


def _check_parameters(params: dict, errors: list[str],
                      warnings: list[str],
                      version: int = SCHEMA_VERSION) -> dict[str, float]:
    """Validate every parameter spec → numeric values of the usable ones.
    A plain parameter needs ``evidence`` or ``assumption`` or (accepted in
    both versions) a ``source`` {kind, ref}."""
    for name, spec in params.items():
        if not _valid_param_name(name):
            errors.append(f"parameter {name!r}: {_bad_name_hint(name)}")
            continue
        if not isinstance(spec, dict):
            errors.append(f"parameter {name!r}: must be an object with a value")
            continue
        _unknown_keys(spec, PARAM_KEYS, f"parameter {name!r}", errors)
        derived = spec.get("expr") is not None
        value = spec.get("value")
        if derived:
            if "value" in spec and not _num(value):
                warnings.append(f"parameter {name!r}: value is ignored "
                                f"because expr is set")
        elif not _num(value):
            errors.append(f"parameter {name!r}: value must be a finite number, "
                          f"got {value!r}")
        if not derived and not (spec.get("evidence") or spec.get("assumption")
                                or spec.get("source")):
            errors.append(f"parameter {name!r}: needs evidence "
                          f"({{op, detail}}), an explicit assumption note or "
                          f"a source ({{kind, ref}}) — unevidenced numbers "
                          f"are rejected")
        if spec.get("source") is not None:
            _check_source(name, spec["source"], errors)
            if (isinstance(spec["source"], dict)
                    and spec["source"].get("kind") == "derived" and not derived):
                errors.append(f"parameter {name!r}: source kind 'derived' "
                              f"needs an 'expr' formula over other parameters")
        ev = spec.get("evidence")
        if ev is not None:
            if not (isinstance(ev, dict) and ev.get("op") and ev.get("detail")):
                errors.append(f"parameter {name!r}: evidence must be "
                              f"{{'op': …, 'detail': …}}")
            else:
                _check_note(ev["detail"], f"parameter {name!r}: evidence.detail",
                            errors, other=str(ev["op"]))
        if spec.get("assumption") is not None:
            _check_note(spec["assumption"], f"parameter {name!r}: assumption",
                        errors)
    values, problems, _order = _eval_params(params)
    errors.extend(problems)
    # a plan value that disagrees with its own formula is not an error (the
    # formula wins) but the planner should know
    for name, spec in params.items():
        if is_derived(spec) and _num(spec.get("value")) and name in values:
            if abs(values[name] - float(spec["value"])) > 1e-6 * max(
                    1.0, abs(values[name])):
                warnings.append(f"parameter {name!r}: value {spec['value']:g} "
                                f"disagrees with its expr ({values[name]:g}); "
                                f"the formula wins")
    return values


def _check_profile(profile: dict, fid: str, errors: list[str],
                   warnings: list[str], params: dict | None = None) -> None:
    shape = profile.get("shape")
    if shape not in SKETCH_SHAPES:
        errors.append(f"feature {fid}: profile shape {shape!r} not in "
                      f"{SKETCH_SHAPES} (bounded v1 set)")
        return
    params_known = set(params or {})
    _unknown_keys(profile, PROFILE_KEYS[shape], f"feature {fid}: profile",
                  errors)

    def coord_ok(v):
        return _num(v) or (_is_param_ref(v) and v in params_known)

    def known(v, what):
        if _is_param_ref(v) and v not in params_known:
            errors.append(f"feature {fid}: {what} references unknown "
                          f"parameter {v!r}{suggest(v, params_known)}")
    dim = {
        "rectangle": ("width", "height"),
        "circle": ("diameter",),
        "slot": ("length", "width"),
    }.get(shape, ())
    for field in dim:
        v = profile.get(field)
        if not _is_param_ref(v):
            errors.append(f"feature {fid}: profile.{field} must be a parameter "
                          f"name (string), got {v!r}")
        else:
            known(v, f"profile.{field}")
    if shape in ("rectangle", "circle", "slot"):
        center = profile.get("center", [0, 0])
        if not isinstance(center, list) or len(center) != 2:
            errors.append(f"feature {fid}: profile.center must be [x, y]")
        else:
            for i, c in enumerate(center):
                if not (_is_param_ref(c) or _num(c)):
                    errors.append(f"feature {fid}: profile.center[{i}] must be "
                                  f"a parameter name or number, got {c!r}")
                else:
                    known(c, f"profile.center[{i}]")
    if shape == "slot":
        for key in ("angle", "angle_deg"):
            if key in profile and not _num(profile[key]) \
                    and not _is_param_ref(profile[key]):
                errors.append(f"feature {fid}: slot.{key} must be a number or "
                              f"param")
            else:
                known(profile.get(key), f"slot.{key}")
        if "angle" in profile and "angle_deg" in profile:
            errors.append(f"feature {fid}: slot gives both 'angle' and "
                          f"'angle_deg'; keep one (degrees)")
    if shape == "profile":
        start = profile.get("start")
        if not isinstance(start, list) or len(start) != 2:
            errors.append(f"feature {fid}: profile.start must be [x, y]")
        elif not all(coord_ok(c) for c in start):
            errors.append(f"feature {fid}: profile.start coords must be "
                          f"numbers or known parameter names")
        segments = profile.get("segments")
        if not isinstance(segments, list) or not segments:
            errors.append(f"feature {fid}: profile.segments must be a "
                          f"non-empty list")
            return
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                errors.append(f"feature {fid}: segment {i} must be an object")
                continue
            _unknown_keys(seg, SEGMENT_KEYS, f"feature {fid}: segment {i}",
                          errors)
            if seg.get("type") not in ("line", "arc"):
                errors.append(f"feature {fid}: segment {i} type must be "
                              f"line/arc, got {seg.get('type')!r}")
            to = seg.get("to")
            if not isinstance(to, list) or len(to) != 2:
                errors.append(f"feature {fid}: segment {i}.to must be [x, y]")
            elif not all(coord_ok(c) for c in to):
                bad = [c for c in to if not coord_ok(c)]
                errors.append(f"feature {fid}: segment {i}.to coords must be "
                              f"numbers or known parameter names, got {bad}")
            if seg.get("type") == "arc":
                ctr = seg.get("center")
                if not isinstance(ctr, list) or len(ctr) != 2:
                    errors.append(f"feature {fid}: arc segment {i} needs "
                                  f"center [x, y]")
                elif not all(coord_ok(c) for c in ctr):
                    errors.append(f"feature {fid}: arc segment {i}.center "
                                  f"coords must be numbers or known parameter "
                                  f"names, got {ctr}")
                if not isinstance(seg.get("ccw"), bool):
                    errors.append(f"feature {fid}: arc segment {i} needs "
                                  f"ccw true/false")
            for key in ("length", "radius", "diameter"):
                if key in seg:
                    if not (_is_param_ref(seg[key]) or _num(seg[key])):
                        errors.append(f"feature {fid}: segment {i}.{key} must "
                                      f"be param name or number")
                    else:
                        known(seg[key], f"segment {i}.{key}")


def _check_plane(plane: Any, fid: str, frame: dict, errors: list[str],
                 params: dict | None = None) -> None:
    if not isinstance(plane, dict):
        errors.append(f"feature {fid}: plane must be an object "
                      f"({{'frame_plane': <name>}} or {{normal, origin, "
                      f"x_dir}})")
        return
    _unknown_keys(plane, PLANE_KEYS, f"feature {fid}: plane", errors)
    if "frame_plane" in plane:
        name = plane["frame_plane"]
        planes = frame.get("planes") if isinstance(frame, dict) else None
        if not isinstance(planes, dict):
            planes = {}
        if not isinstance(name, str) or name not in planes:
            errors.append(f"feature {fid}: plane references unknown frame "
                          f"plane {name!r} (have: {sorted(map(str, planes))})")
        return
    if not _vec3(plane.get("normal"), f"feature {fid}: plane.normal", errors):
        return
    normal = plane["normal"]
    n = _norm3(normal)
    if n < 1e-9:
        errors.append(f"feature {fid}: plane.normal must not be the zero "
                      f"vector")
    if "origin" in plane:
        _plane_origin_vec(plane.get("origin"), f"feature {fid}: plane.origin",
                          errors, params or {})
    if "x_dir" in plane and _vec3(plane.get("x_dir"),
                                  f"feature {fid}: plane.x_dir", errors) \
            and n >= 1e-9:
        x = plane["x_dir"]
        dot = sum(a * b for a, b in zip(x, normal)) / (n * n)
        perp = [a - dot * b for a, b in zip(x, normal)]
        if _norm3(perp) <= 1e-6 * max(_norm3(x), 1e-12):
            errors.append(f"feature {fid}: plane.x_dir must not be zero or "
                          f"parallel to plane.normal")


# ---------------------------------------------------------------------------
# hole counterbore / countersink: flat and nested spellings, one reading


def hole_cut(feat: dict) -> dict | None:
    """The counterbore / countersink of a hole feature, normalised.

    Accepts both spellings the planner produces — flat (``counterbore:
    true`` with ``cb_diameter`` / ``cb_depth`` on the feature) and nested
    (``counterbore: {cb_diameter, cb_depth}``, also ``{diameter, depth}``;
    ``countersink: {cs_diameter, cs_angle}`` / ``{diameter, angle}``).
    Returns ``{"type": "counterbore" | "countersink", "diameter": v,
    "depth": v | None, "angle": v | None}`` (values are parameter names or
    numbers) or None for a plain hole. Raises ValueError, phrased for the
    planner, when the spec is contradictory or incomplete."""
    cb, cs = feat.get("counterbore"), feat.get("countersink")
    if cb and cs:
        raise ValueError("counterbore and countersink are mutually exclusive; "
                         "give one")

    def nested(spec, *keys):
        if isinstance(spec, dict):
            for key in keys:
                if spec.get(key) is not None:
                    return spec[key]
        return None

    if cb:
        diameter = nested(cb, "cb_diameter", "diameter")
        depth = nested(cb, "cb_depth", "depth")
        diameter = feat.get("cb_diameter") if diameter is None else diameter
        depth = feat.get("cb_depth") if depth is None else depth
        if diameter is None or depth is None:
            raise ValueError("a counterbore needs both cb_diameter and "
                             "cb_depth (parameter names)")
        return {"type": "counterbore", "diameter": diameter, "depth": depth,
                "angle": None}
    if cs:
        diameter = nested(cs, "cs_diameter", "diameter")
        angle = nested(cs, "cs_angle", "angle")
        diameter = feat.get("cs_diameter") if diameter is None else diameter
        angle = feat.get("cs_angle") if angle is None else angle
        if diameter is None:
            raise ValueError("a countersink needs cs_diameter (parameter "
                             "name); cs_angle is optional (default 90)")
        return {"type": "countersink", "diameter": diameter, "depth": None,
                "angle": angle}
    stray = [k for k in ("cb_diameter", "cb_depth", "cs_diameter", "cs_angle")
             if feat.get(k) is not None]
    if stray:
        raise ValueError(f"{', '.join(stray)} given without counterbore: true "
                         f"/ countersink: true")
    return None


# ---------------------------------------------------------------------------
# dimension roles: which parameter drives which dimension, and its range


def all_features(plan: dict) -> list[dict]:
    """Every feature of the plan in build order: the top-level ``features``
    list, or - for a v2 plan with ``parts`` - the features of each part one
    after the other. Non-dict entries are skipped (validate reports them)."""
    if not isinstance(plan, dict):
        return []
    parts = plan.get("parts")
    if isinstance(parts, list) and parts:
        out: list[dict] = []
        for part in parts:
            feats = part.get("features") if isinstance(part, dict) else None
            out.extend(f for f in (feats or []) if isinstance(f, dict))
        return out
    return [f for f in (plan.get("features") or []) if isinstance(f, dict)]


def plan_units(plan: dict) -> list[dict]:
    """The Bodies the plan builds, in order: ``[{"part": id | None,
    "features": [...], "placement": dict | None, "body": explicit name |
    None}]``. A plan without ``parts`` (every v1 plan) is ONE unit whose part
    is None."""
    parts = plan.get("parts") if isinstance(plan, dict) else None
    if isinstance(parts, list) and parts:
        return [{"part": p.get("id"), "features": [f for f in (p.get("features") or [])
                                                   if isinstance(f, dict)],
                 "placement": p.get("placement") if isinstance(p.get("placement"), dict) else None,
                 "body": p.get("body")}
                for p in parts if isinstance(p, dict)]
    return [{"part": None, "features": [f for f in (plan.get("features") or [])
                                        if isinstance(f, dict)] if isinstance(plan, dict) else [],
             "placement": None, "body": None}]


def feature_profiles(feat: dict) -> list[dict]:
    """The profile dicts of a sketch / boss feature: ``profiles`` (v2, several
    shapes in one sketch) or the single ``profile``."""
    many = feat.get("profiles")
    if isinstance(many, list):
        return [p for p in many if isinstance(p, dict)]
    one = feat.get("profile")
    return [one] if isinstance(one, dict) else []


def dimension_refs(plan: dict) -> list[tuple[str, str, Any, str]]:
    """Every dimensional reference of the plan's features as ``(feature_id,
    role, ref, rule)``.

    ``ref`` is a parameter name or a literal number; ``role`` names the
    place (``pad.length``, ``hole.diameter``, ``sketch.rectangle.width`` ...);
    ``rule`` is ``"positive"`` (> 0), ``"angle360"`` (0 < v <= 360),
    ``"angle180"`` (0 < v < 180) or ``"count"`` (a whole number >= 2).
    Malformed features are skipped (validate reports them); the builder and
    the editability test read the same list. The v2 ops contribute through
    their registry ``dims``; sketches and bosses through every profile."""
    out: list[tuple[str, str, Any, str]] = []
    for feat in all_features(plan):
        if not isinstance(feat.get("id"), str):
            continue
        fid, op = feat["id"], feat.get("op")

        def add(role, ref, rule="positive", fid=fid):
            if isinstance(ref, str) or _num(ref):
                out.append((fid, role, ref, rule))

        def add_profile(prof, prefix, add=add):
            shape = prof.get("shape")
            for fld in {"rectangle": ("width", "height"),
                        "circle": ("diameter",),
                        "slot": ("length", "width")}.get(shape, ()):
                add(f"{prefix}.{shape}.{fld}", prof.get(fld))
            if shape == "profile" and isinstance(prof.get("segments"), list):
                for seg in prof["segments"]:
                    if isinstance(seg, dict):
                        for key in ("length", "radius", "diameter"):
                            add(f"{prefix}.profile.{key}", seg.get(key))
        if op in ("sketch", "boss"):
            for prof in feature_profiles(feat):
                add_profile(prof, "sketch" if op == "sketch" else op)
        if op == "pad":
            add("pad.length", feat.get("length"))
        elif op == "pocket" and not feat.get("through"):
            add("pocket.depth", feat.get("depth"))
        elif op == "revolution":
            add("revolution.angle", feat.get("angle"), "angle360")
        elif op == "hole":
            add("hole.diameter", feat.get("diameter"))
            if not feat.get("through"):
                add("hole.depth", feat.get("depth"))
            try:
                cut = hole_cut(feat)
            except ValueError:
                cut = None
            if cut:
                add(f"hole.{cut['type']}.diameter", cut["diameter"])
                add(f"hole.{cut['type']}.depth", cut["depth"])
                if cut["angle"] is not None:
                    add("hole.countersink.angle", cut["angle"], "angle180")
        elif op in OPS and OPS[op].since > 1:
            for fld, rule in OPS[op].dims.items():
                add(f"{op}.{fld}", feat.get(fld), rule)
    return out


def _check_ranges(plan: dict, values: dict[str, float],
                  errors: list[str]) -> None:
    """Positivity / range by role, for parameters and literal numbers."""
    reported: set[tuple[str, str]] = set()
    for fid, role, ref, rule in dimension_refs(plan):
        value = values.get(ref) if isinstance(ref, str) else ref
        if value is None:
            continue
        if rule == "positive":
            bad, need = value <= 0, "positive (> 0)"
        elif rule == "angle360":
            bad, need = not 0 < value <= 360, "an angle in (0, 360] degrees"
        elif rule == "count":
            bad = not (2 <= value <= MAX_PATTERN_COUNT
                       and abs(value - round(value)) < 1e-9)
            need = f"a whole number from 2 to {MAX_PATTERN_COUNT}"
        else:
            bad, need = not 0 < value < 180, "an angle in (0, 180) degrees"
        if bad and (fid, role) not in reported:
            reported.add((fid, role))
            who = f"parameter {ref!r} ({value:g})" if isinstance(ref, str) \
                else f"value {value:g}"
            errors.append(f"feature {fid}: {who} is used as {role} and must be "
                          f"{need}")


def _check_relations(plan: dict, values: dict[str, float],
                     errors: list[str], warnings: list[str]) -> None:
    """Cross-field relations that make a feature geometrically impossible."""
    def val(ref):
        return values.get(ref) if isinstance(ref, str) else (
            float(ref) if _num(ref) else None)
    for feat in all_features(plan):
        fid = feat.get("id")
        profs = feature_profiles(feat) if feat.get("op") in ("sketch", "boss") else []
        for prof in profs:
            if prof.get("shape") != "slot":
                continue
            length, width = val(prof.get("length")), val(prof.get("width"))
            if length is not None and width is not None and length <= width:
                errors.append(f"feature {fid}: slot length ({length:g}) must "
                              f"exceed its width ({width:g}); a slot as wide "
                              f"as it is long is a circle (use shape "
                              f"'circle')")
        if feat.get("op") != "hole":
            continue
        try:
            cut = hole_cut(feat)
        except ValueError:
            continue
        if not cut:
            continue
        hole_d, cut_d = val(feat.get("diameter")), val(cut["diameter"])
        if hole_d is not None and cut_d is not None and cut_d <= hole_d:
            errors.append(f"feature {fid}: {cut['type']} diameter ({cut_d:g}) "
                          f"must exceed the hole diameter ({hole_d:g})")
        depth, cut_depth = val(feat.get("depth")), val(cut["depth"])
        if depth is not None and cut_depth is not None and cut_depth >= depth:
            warnings.append(f"feature {fid}: counterbore depth ({cut_depth:g})"
                            f" is not smaller than the hole depth ({depth:g})")


def _profile_gap(prof: dict, values: dict[str, float]) -> float | None:
    """Distance between the end of a polyline profile and its start, or
    None when a coordinate cannot be resolved (checked by the builder)."""
    def coord(v):
        return values.get(v) if isinstance(v, str) else (
            float(v) if _num(v) else None)
    start = prof.get("start")
    segs = prof.get("segments")
    if not (isinstance(start, list) and len(start) == 2
            and isinstance(segs, list) and segs
            and isinstance(segs[-1], dict)):
        return None
    end = segs[-1].get("to")
    if not (isinstance(end, list) and len(end) == 2):
        return None
    pts = [coord(start[0]), coord(start[1]), coord(end[0]), coord(end[1])]
    if any(p is None for p in pts):
        return None
    return math.hypot(pts[2] - pts[0], pts[3] - pts[1])


def _check_profiles(feat: dict, fid: str, vc: VCtx) -> None:
    """The ``profile`` / (v2) ``profiles`` of a sketch or boss."""
    errors = vc.errors
    if "profiles" in feat:
        many = feat["profiles"]
        if vc.version < SCHEMA_VERSION_V2:
            errors.append(f"feature {fid}: 'profiles' needs schema_version 2 "
                          f"(this plan is version 1); use one 'profile'")
        elif "profile" in feat:
            errors.append(f"feature {fid}: gives both 'profile' and "
                          f"'profiles'; keep one")
        elif not isinstance(many, list) or not many:
            errors.append(f"feature {fid}: profiles must be a non-empty list "
                          f"of profile objects, got {many!r}")
        elif len(many) > MAX_PROFILES:
            errors.append(f"feature {fid}: {len(many)} profiles in one sketch "
                          f"(at most {MAX_PROFILES})")
        else:
            for i, prof in enumerate(many):
                if not isinstance(prof, dict):
                    errors.append(f"feature {fid}: profiles[{i}] must be a "
                                  f"profile object, got {prof!r}")
                else:
                    _check_profile(prof, f"{fid} (profiles[{i}])", errors,
                                   vc.warnings, vc.params)
    else:
        profile = feat.get("profile")
        if not isinstance(profile, dict):
            errors.append(f"feature {fid}: sketch needs a profile object")
        else:
            _check_profile(profile, fid, errors, vc.warnings, vc.params)


def _check_sketch_op(feat: dict, fid: str, vc: VCtx) -> None:
    """``sketch``: the profile(s) plus the plane."""
    _check_profiles(feat, fid, vc)
    _check_plane(feat.get("plane"), fid, vc.frame, vc.errors, vc.params)


def _check_pocket_op(feat: dict, fid: str, vc: VCtx) -> None:
    if feat.get("through") and "depth" in feat:
        vc.errors.append(f"feature {fid}: through-pocket must not also set "
                         f"depth")
    elif not feat.get("through") and "depth" not in feat:
        vc.errors.append(f"feature {fid}: pocket requires 'depth' (a "
                         f"parameter name) or 'through': true")


def _check_hole_op(feat: dict, fid: str, vc: VCtx) -> None:
    errors, warnings, params = vc.errors, vc.warnings, vc.params
    pos = feat.get("position")
    if not isinstance(pos, dict) or "x" not in pos or "y" not in pos:
        errors.append(f"feature {fid}: hole needs position {{x, y}}")
    else:
        _unknown_keys(pos, ("x", "y", "note"), f"feature {fid}: position",
                      errors)
        for key in ("x", "y"):
            v = pos.get(key)
            if not (_is_param_ref(v) or _num(v)):
                errors.append(f"feature {fid}: position.{key} must be a "
                              f"parameter name or number, got {v!r}")
            elif _is_param_ref(v) and v not in params:
                errors.append(f"feature {fid}: position.{key} "
                              f"references unknown parameter {v!r}"
                              f"{suggest(v, params)}")
    if feat.get("diameter") and not _is_param_ref(feat["diameter"]):
        errors.append(f"feature {fid}: hole.diameter must be a "
                      f"parameter name")
    _check_plane(feat.get("plane"), fid, vc.frame, errors, params)
    if feat.get("through") and "depth" in feat:
        errors.append(f"feature {fid}: through-hole must not also set "
                      f"depth")
    elif not feat.get("through") and "depth" not in feat:
        errors.append(f"feature {fid}: hole requires 'through': true "
                      f"or a 'depth' parameter")
    try:
        cut = hole_cut(feat)
    except ValueError as e:
        errors.append(f"feature {fid}: {e}")
        cut = None
    if cut:
        for key in ("diameter", "depth", "angle"):
            ref = cut[key]
            if ref is None:
                continue
            if _is_param_ref(ref):
                if ref not in params:
                    errors.append(f"feature {fid}: {cut['type']} "
                                  f"{key} references unknown parameter "
                                  f"{ref!r}{suggest(ref, params)}")
            elif not _num(ref):
                errors.append(f"feature {fid}: {cut['type']} {key} must "
                              f"be a parameter name, got {ref!r}")
            elif key != "angle":
                warnings.append(f"feature {fid}: literal {cut['type']} "
                                f"{key} {ref:g} — dimensional values "
                                f"should bind a parameter name")


def _check_revolution_op(feat: dict, fid: str, vc: VCtx) -> None:
    errors, axes = vc.errors, vc.axes
    axis = feat.get("axis")
    if not isinstance(axis, dict) or not (
            "direction" in axis or "frame_axis" in axis):
        errors.append(f"feature {fid}: revolution needs axis "
                      f"{{direction, point}} or axis.frame_axis")
    elif "frame_axis" in axis:
        name = axis["frame_axis"]
        if not isinstance(name, str) or name not in axes:
            errors.append(f"feature {fid}: axis.frame_axis "
                          f"{name!r} is not in frame.axes "
                          f"(have: {sorted(map(str, axes))})")
    else:
        _unknown_keys(axis, ("direction", "point", "note"),
                      f"feature {fid}: axis", errors)
        if _vec3(axis.get("direction"), f"feature {fid}: axis.direction",
                 errors) and _norm3(axis["direction"]) < 1e-9:
            errors.append(f"feature {fid}: axis.direction must not be "
                          f"the zero vector")
        _vec3(axis.get("point", [0, 0, 0]),
              f"feature {fid}: axis.point", errors)


MAX_PROFILES = 24              # profiles in one sketch


def _register_v1_ops() -> None:
    """The v1 vocabulary as registry entries (the keys, dimensional fields
    and validators the per-op branches of ``_validate_plan`` used to hold)."""
    register_op(OpSpec("sketch", "sketch", None,
                       FEATURE_KEYS["sketch"] + ("profiles",),
                       check=_check_sketch_op,
                       nondim=("profile", "profiles")))
    register_op(OpSpec("pad", "solid", "additive", FEATURE_KEYS["pad"],
                       required=("length",), dims={"length": "positive"},
                       refs={"sketch": "sketch"}))
    register_op(OpSpec("pocket", "solid", "subtractive", FEATURE_KEYS["pocket"],
                       dims={"depth": "positive"}, refs={"sketch": "sketch"},
                       check=_check_pocket_op))
    register_op(OpSpec("revolution", "solid", "additive",
                       FEATURE_KEYS["revolution"], dims={"angle": "angle360"},
                       refs={"sketch": "sketch"}, check=_check_revolution_op))
    register_op(OpSpec("hole", "solid", "subtractive", FEATURE_KEYS["hole"],
                       required=("diameter",),
                       dims={"diameter": "positive", "depth": "positive"},
                       check=_check_hole_op))


def validate(plan: dict) -> tuple[bool, list[str], list[str]]:
    """Validate a FeaturePlan dict (v1 or v2) → ``(ok, errors, warnings)``.

    Never raises: a structure the checks do not understand becomes an
    error string. Errors are phrased for the planner ("what failed: why;
    what is accepted") and feed its repair loop; warnings are allowed."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(plan, dict):
        return False, [f"plan must be a JSON object (the FeaturePlan itself), "
                       f"got {type(plan).__name__}"], []
    if not isinstance(plan.get("parameters"), dict):
        return False, [f"parameters must be an object name -> {{value, "
                       f"evidence|assumption}}, got "
                       f"{type(plan.get('parameters')).__name__} — pass the "
                       f"WHOLE plan object, not a fragment"], []
    if not isinstance(plan.get("features"), list) \
            and not isinstance(plan.get("parts"), list):
        return False, [f"features must be an ordered list of feature "
                       f"objects, got "
                       f"{type(plan.get('features')).__name__} — pass the "
                       f"WHOLE plan object, not a fragment"
                       f"{' (schema 2: or parts[] each with its features)' if plan.get('schema_version') == 2 else ''}"], []
    try:
        _validate_plan(plan, errors, warnings)
    except Exception as e:  # noqa: BLE001 — a validator must not crash the caller
        errors.append(f"plan structure not understood ({type(e).__name__}: "
                      f"{e}); check every feature, profile and plane is an "
                      f"object of the documented shape")
    return (not errors), errors, warnings


def _validate_features(features: list, plan: dict, params: dict, frame: dict,
                       axes: dict, version: int, errors: list[str],
                       warnings: list[str], seen_ids: set[str]) -> None:
    """Validate ONE ordered feature list (the plan's, or one part's): ids,
    op vocabulary, keys, flags and - through the registry - required fields,
    dimensional fields, earlier-sketch references and the op validator."""
    vc = VCtx(plan, params, frame, axes, errors, warnings, {}, version)
    known = ops_for(version)
    for i, feat in enumerate(features):
        fid = feat.get("id") if isinstance(feat, dict) else None
        label = fid if isinstance(fid, str) else f"#{i}"
        if not isinstance(feat, dict):
            errors.append(f"feature {label}: must be an object")
            continue
        if not isinstance(fid, str) or not fid:
            errors.append(f"feature #{i}: id must be a non-empty string")
            continue
        if not _FEATURE_ID.match(fid):
            errors.append(f"feature {fid}: id must be an identifier (letters, "
                          f"digits, underscore; not starting with a digit) "
                          f"because it becomes the FreeCAD object name")
        if fid in seen_ids:
            errors.append(f"feature {fid}: duplicate id")
        seen_ids.add(fid)
        op = feat.get("op")
        if op not in OPS:
            errors.append(f"feature {fid}: op {op!r} not in bounded set "
                          f"{known}{suggest(op, known)}")
            continue
        spec = OPS[op]
        if spec.since > version:
            errors.append(f"feature {fid}: op {op!r} needs schema_version "
                          f"{spec.since} (this plan is version {version}); "
                          f"the version-{version} ops are {known}")
            continue
        _unknown_keys(feat, spec.keys, f"feature {fid} ({op})", errors)
        for flag in _FLAG_FIELDS:
            if flag in feat and not isinstance(feat[flag], bool):
                errors.append(f"feature {fid}: {flag} must be true or false, "
                              f"got {feat[flag]!r}")
        if spec.kind == "sketch":
            if spec.check:
                spec.check(feat, fid, vc)
            vc.earlier[fid] = op
            continue
        if not vc.sketches():
            errors.append(f"feature {fid}: {op} before any sketch — the first "
                          f"feature must be a sketch")
        for fld, kind in spec.refs.items():
            ref = feat.get(fld)
            if kind == "sketch" and (not isinstance(ref, str)
                                     or ref not in vc.sketches()):
                errors.append(f"feature {fid}: references sketch {ref!r} "
                              f"which does not precede it")
        for fld in spec.required:
            if fld not in feat:
                errors.append(f"feature {fid}: {op} requires '{fld}' (a "
                              f"parameter name)")
        # dimensional fields must be param refs
        for fld in spec.dims:
            if fld not in feat:
                continue
            if not _is_param_ref(feat[fld]):
                errors.append(f"feature {fid}: {op}.{fld} must be a parameter "
                              f"name (string), got {feat[fld]!r}")
            elif feat[fld] not in params:
                errors.append(f"feature {fid}: {op}.{fld} references unknown "
                              f"parameter {feat[fld]!r}"
                              f"{suggest(feat[fld], params)}")
        if spec.check:
            spec.check(feat, fid, vc)
        for key in feat:
            if key not in _NON_DIMENSIONAL and key not in spec.dims \
                    and key not in spec.nondim and key not in ("id", "op"):
                if isinstance(feat[key], (int, float)) \
                        and not isinstance(feat[key], bool):
                    warnings.append(f"feature {fid}: literal numeric field "
                                    f"{key!r}={feat[key]} — dimensional values "
                                    f"should bind a parameter name")
        vc.earlier[fid] = op


def _validate_plan(plan: dict, errors: list[str], warnings: list[str]) -> None:
    version = plan.get("schema_version")
    if isinstance(version, bool) or version not in SUPPORTED_VERSIONS:
        errors.append(f"schema_version must be {SCHEMA_VERSION} or "
                      f"{SCHEMA_VERSION_V2}")
    v = SCHEMA_VERSION_V2 if version == SCHEMA_VERSION_V2 and not isinstance(
        version, bool) else SCHEMA_VERSION
    for field_ in ("id", "part_class", "units"):
        if not isinstance(plan.get(field_), str) or not plan[field_]:
            errors.append(f"{field_} must be a non-empty string")
    units = plan.get("units")
    if isinstance(units, str) and units and units.strip().lower() not in _MM_UNITS:
        errors.append(f"units {units!r} not supported (have {VALID_UNITS}): "
                      f"the builder reads every number as millimetres, so "
                      f"convert the values and set units to 'mm'")
    target = plan.get("target", "freecad")
    if target not in VALID_TARGETS:
        errors.append(f"target {target!r} not supported (have {VALID_TARGETS})")
    _unknown_keys(plan, PLAN_KEYS_V2, "plan", errors, warnings)
    for key in ("parts", "assembly", "references"):
        if key in plan and v < SCHEMA_VERSION_V2:
            errors.append(f"plan: {key!r} needs schema_version 2 (this plan "
                          f"is version 1); put the features in 'features'")

    # -- parameters ---------------------------------------------------------
    params = plan.get("parameters")
    if not params:
        errors.append("parameters must be a non-empty object "
                      "(name → {value, evidence|assumption})")
        params = {}
    used: set[str] = set()
    values = _check_parameters(params, errors, warnings, v)
    for spec in params.values():
        if is_derived(spec):
            try:
                used |= set(parse_expr(spec["expr"])[1])
            except ExprError:
                pass

    # -- frame ---------------------------------------------------------------
    frame = plan.get("frame", {})
    if not isinstance(frame, dict):
        errors.append("frame must be an object")
        frame = {}
    else:
        _unknown_keys(frame, FRAME_KEYS, "frame", errors, warnings)
    planes = frame.get("planes") or {}
    axes = frame.get("axes") or {}
    if not isinstance(planes, dict):
        errors.append("frame.planes must be an object name -> plane")
        planes = {}
    if not isinstance(axes, dict):
        errors.append("frame.axes must be an object name -> axis")
        axes = {}
    for name, pl in planes.items():
        _check_plane(pl, f"frame.planes.{name}", {"planes": {}}, errors, params)
        if isinstance(pl, dict):
            for c in pl.get("origin") or []:
                if _is_param_ref(c) and c in params:
                    used.add(c)
    for name, ax in axes.items():
        if not isinstance(ax, dict):
            errors.append(f"frame.axes.{name}: must be an object")
            continue
        _unknown_keys(ax, AXIS_KEYS, f"frame.axes.{name}", errors)
        if _vec3(ax.get("direction"), f"frame.axes.{name}.direction", errors) \
                and _norm3(ax["direction"]) < 1e-9:
            errors.append(f"frame.axes.{name}.direction must not be the zero "
                          f"vector")
        _vec3(ax.get("point", [0, 0, 0]), f"frame.axes.{name}.point", errors)

    # -- features (one list, or one list per part) -----------------------------
    seen_ids: set[str] = set()
    if v >= SCHEMA_VERSION_V2 and plan.get("parts") is not None:
        if plan.get("features"):
            errors.append("plan gives both 'features' and 'parts'; put the "
                          "features inside the parts")
        _schema_v2.check_parts(plan, params, frame, axes, errors, warnings,
                               seen_ids, _validate_features)
    else:
        features = plan.get("features")
        if not features:
            errors.append("features must be a non-empty ordered list")
            features = []
        _validate_features(features, plan, params, frame, axes, v, errors,
                           warnings, seen_ids)
    if v >= SCHEMA_VERSION_V2:
        _schema_v2.check_assembly(plan, params, errors, warnings)
    features = all_features(plan)

    # -- cross checks over the whole plan ---------------------------------------
    _check_ranges(plan, values, errors)
    _check_relations(plan, values, errors, warnings)
    for feat in features:
        if feat.get("op") not in ("sketch", "boss"):
            continue
        for prof in feature_profiles(feat):
            if prof.get("shape") != "profile":
                continue
            gap = _profile_gap(prof, values)
            if gap is not None and gap > 1e-3:
                errors.append(f"feature {feat.get('id')}: profile is not "
                              f"closed — the last segment ends {gap:g} mm "
                              f"from the start; end the last segment at the "
                              f"start point")
    sketch_owner = {f"{f['id']}_sketch": f["op"] for f in features
                    if f.get("op") in ("hole", "boss")
                    and isinstance(f.get("id"), str)}
    for clash in sorted(set(sketch_owner) & seen_ids):
        errors.append(f"feature id {clash!r} collides with the sketch the "
                      f"builder creates for the {sketch_owner[clash]} "
                      f"{clash[:-7]!r}; rename it")

    # -- binding closure ------------------------------------------------------
    _META_KEYS = {"normal", "origin", "x_dir", "direction", "point",
                  "evidence", "detail", "op", "note", "id", "plane",
                  "frame_plane", "frame_axis", "sketch", "type", "shape",
                  "of", "edges", "open_faces", "face", "axis", "part", "kind",
                  "name", "ref", "body"}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, val in node.items():
                if k in _META_KEYS:
                    continue
                if _is_param_ref(val) and val in params:
                    used.add(val)
                walk(val)
        elif isinstance(node, list):
            for val in node:
                # coordinate lists ("to": ["profile_len", 0]) carry refs
                if _is_param_ref(val) and val in params:
                    used.add(val)
                walk(val)

    walk(features)
    if v >= SCHEMA_VERSION_V2:
        for part in plan.get("parts") or []:
            if isinstance(part, dict):
                walk(part.get("placement"))
        walk(plan.get("assembly"))
        walk(plan.get("references"))
    for feat in features:           # explicit plane origins ride parameters too
        pl = feat.get("plane")
        if isinstance(pl, dict):
            for c in pl.get("origin") or []:
                if _is_param_ref(c) and c in params:
                    used.add(c)
    unused = set(params) - used
    if unused:
        warnings.append(f"parameters never referenced by features: "
                        f"{sorted(map(str, unused))}")


def resolve(plan: dict, name: str) -> float:
    """Numeric value of parameter ``name`` (builders resolve bindings).

    A derived parameter is evaluated from its ``expr``. Raises KeyError for
    an unknown name and ValueError when the parameter has no usable number
    (bad formula, cycle)."""
    params = plan["parameters"]
    spec = params[name]            # KeyError for an unknown name
    if not is_derived(spec):
        return float(spec["value"])
    values, problems, _order = _eval_params(params)
    if name not in values:
        raise ValueError(f"parameter {name!r} cannot be evaluated: "
                         f"{'; '.join(problems) or 'bad expr'}")
    return values[name]


def sketch_basis(plan: dict, plane: dict) -> dict:
    """Resolve a sketch plane to origin / normal / x_dir (orthonormalized).

    Accepts either an explicit plane or a frame_plane reference.
    """
    if "frame_plane" in plane:
        plane = plan["frame"]["planes"][plane["frame_plane"]]

    normal = [float(v) for v in plane["normal"]]
    n = math.sqrt(sum(c * c for c in normal))
    normal = [c / n for c in normal]
    if plane.get("x_dir"):
        x = [float(v) for v in plane["x_dir"]]
        dot = sum(a * b for a, b in zip(x, normal))
        x = [a - dot * b for a, b in zip(x, normal)]
        m = math.sqrt(sum(c * c for c in x))
        x = [c / m for c in x]
    else:
        # stable default: least-aligned world axis
        pick = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]
        dot = sum(a * b for a, b in zip(pick, normal))
        x = [a - dot * b for a, b in zip(pick, normal)]
        m = math.sqrt(sum(c * c for c in x))
        x = [c / m for c in x]
    y = [normal[1] * x[2] - normal[2] * x[1],
         normal[2] * x[0] - normal[0] * x[2],
         normal[0] * x[1] - normal[1] * x[0]]

    def coord(v):
        if isinstance(v, str):
            return resolve(plan, v)
        return float(v)

    return {"origin": [coord(v) for v in plane.get("origin", [0, 0, 0])],
            "normal": normal, "x_dir": x, "y_dir": y,
            "origin_refs": [v if isinstance(v, str) else None
                            for v in plane.get("origin", [0, 0, 0])]}


# ---------------------------------------------------------------------------
# registration: the v1 ops, then the v2 vocabulary (imported last: it reads
# the helpers defined above)

FEATURE_OPS_V2 = ("fillet", "chamfer", "thickness", "linear_pattern",   # the ops only schema 2 knows
                  "polar_pattern", "mirror", "boss")                    # (_schema_v2 registers exactly these)
ALL_FEATURE_OPS = FEATURE_OPS + FEATURE_OPS_V2

_register_v1_ops()
from . import _schema_v2  # noqa: E402,F401  (registers the v2 ops; see its docstring)
