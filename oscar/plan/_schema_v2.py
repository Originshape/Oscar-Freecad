"""FeaturePlan schema v2 vocabulary and validators (private to oscar.plan).

Registers the v2 ops in the ``schema.OPS`` registry and validates the v2
plan-level blocks. ``schema`` imports this module last (it reads helpers
defined there), so import ``schema``, not this module.

Ops (every dimension is a parameter NAME, every edge / face choice a robust
RULE - never an edge or face index, which would not survive a parameter
change):

* ``fillet`` / ``chamfer``: ``size`` (+ ``size2`` for a chamfer with two
  distances) and ``edges`` = ``all_outer`` | ``vertical`` | ``top_outer`` |
  ``bottom_outer`` | ``edges_of_feature:<id>`` (see ``_edgerules``);
* ``thickness`` (shell): ``wall``, ``open_faces`` = ``top`` | ``bottom``
  (the face that is removed; a sealed hollow, ``none``, is refused because the
  FreeCAD shell tool needs an opening), ``direction`` inward (default) | outward;
* ``linear_pattern``: ``of`` (an earlier pad / pocket / hole / revolution /
  boss), ``count`` and ``spacing`` parameters, ``direction`` x | y | z (a
  leading ``-`` reverses); ``polar_pattern``: ``of``, ``count``, ``angle``
  (the total sweep, 360 = a full circle) and ``axis`` x | y | z through the
  Body origin; ``mirror``: ``of`` and ``plane`` xy | xz | yz (through the
  Body origin);
* ``boss``: a pad of ``profile`` / ``profiles`` of ``length`` on a face
  chosen by rule (``top`` | ``bottom`` | ``front`` | ``back`` | ``left`` |
  ``right`` = the planar face of the Body facing that way; sketch x / y run
  along the world axes, see ``_dressbuild``);
* ``sketch`` gains ``profiles`` (several shapes in ONE sketch).

Plan level: ``parts`` (each ``{id, features, placement, body}``: its own
Body, never fused), ``assembly`` (named placements driven by parameters),
``references`` (mating faces, validated and reported, informational).
"""

import re

from . import schema as S

EDGE_RULES = ("all_outer", "vertical", "top_outer", "bottom_outer")
EDGE_OF_FEATURE = "edges_of_feature:"
FACE_RULES = ("top", "bottom", "front", "back", "left", "right")
OPEN_FACE_RULES = ("top", "bottom")
PATTERN_DIRECTIONS = ("x", "y", "z", "-x", "-y", "-z")
AXIS_NAMES = ("x", "y", "z")
MIRROR_PLANES = ("xy", "xz", "yz")
THICKNESS_DIRECTIONS = ("inward", "outward")
PATTERN_ORIGINALS = ("pad", "pocket", "hole", "revolution", "boss")

PART_KEYS = ("id", "note", "features", "placement", "body")
PLACEMENT_KEYS = ("x", "y", "z", "rx", "ry", "rz", "note")
ASSEMBLY_KEYS = ("placements", "note")
ASSEMBLY_PLACEMENT_KEYS = ("name", "part", "x", "y", "z", "rx", "ry", "rz", "note")
REFERENCE_KEYS = ("id", "kind", "a", "b", "gap", "note")
REFERENCE_KINDS = ("mating_faces",)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _enum(feat: dict, fid: str, op: str, key: str, allowed, vc, default=None,
          extra: str = "") -> object:
    """Check ``feat[key]`` is one of ``allowed`` (a missing key reads
    ``default``); the error lists the vocabulary and the nearest spelling."""
    val = feat.get(key, default)
    if val not in allowed:
        vc.errors.append(f"feature {fid}: {op}.{key} {val!r} not in "
                         f"{list(allowed)}{extra}{S.suggest(val, allowed)}")
    return val


def edge_rule_parts(rule) -> tuple[str, str]:
    """``("all_outer", "")`` or ``("edges_of_feature", "<id>")`` of an edge
    rule string (the kind is "" when it is not a rule at all)."""
    if isinstance(rule, str) and rule.startswith(EDGE_OF_FEATURE):
        return "edges_of_feature", rule[len(EDGE_OF_FEATURE):]
    if rule in EDGE_RULES:
        return rule, ""
    return "", ""


def _check_edges(feat: dict, fid: str, op: str, vc) -> None:
    rule = feat.get("edges", "all_outer")
    kind, target = edge_rule_parts(rule)
    listed = list(EDGE_RULES) + [EDGE_OF_FEATURE + "<id>"]
    if not isinstance(rule, str) or not kind:
        vc.errors.append(f"feature {fid}: {op}.edges {rule!r} not in {listed}"
                         f"{S.suggest(rule, EDGE_RULES)}")
    elif kind == "edges_of_feature" and target not in vc.earlier:
        vc.errors.append(f"feature {fid}: {op}.edges {rule!r} names feature "
                         f"{target!r}, which does not precede it in this "
                         f"part{S.suggest(target, vc.earlier)} (earlier: "
                         f"{list(vc.earlier)})")


def _check_dress(op: str):
    def check(feat: dict, fid: str, vc) -> None:
        _check_edges(feat, fid, op, vc)
    return check


def _check_thickness(feat: dict, fid: str, vc) -> None:
    rule = feat.get("open_faces", "top")
    if rule == "none":
        vc.errors.append(f"feature {fid}: thickness.open_faces 'none' (a "
                         f"sealed hollow) is not supported: FreeCAD's shell "
                         f"tool needs one face to open; use 'top' or "
                         f"'bottom' and close it with a second part (a lid)")
    else:
        _enum(feat, fid, "thickness", "open_faces", OPEN_FACE_RULES, vc, "top")
    _enum(feat, fid, "thickness", "direction", THICKNESS_DIRECTIONS, vc,
          "inward")


def _check_of(feat: dict, fid: str, op: str, vc) -> None:
    ref = feat.get("of")
    if not isinstance(ref, str) or ref not in vc.earlier:
        vc.errors.append(f"feature {fid}: {op}.of {ref!r} is not an earlier "
                         f"feature of this part{S.suggest(ref, vc.earlier)} "
                         f"(earlier: {list(vc.earlier)})")
    elif vc.earlier[ref] not in PATTERN_ORIGINALS:
        vc.errors.append(f"feature {fid}: {op}.of {ref!r} is a "
                         f"{vc.earlier[ref]!r} feature; a {op} repeats one "
                         f"of {list(PATTERN_ORIGINALS)}")


def _check_linear(feat: dict, fid: str, vc) -> None:
    _check_of(feat, fid, "linear_pattern", vc)
    _enum(feat, fid, "linear_pattern", "direction", PATTERN_DIRECTIONS, vc, "x")


def _check_polar(feat: dict, fid: str, vc) -> None:
    _check_of(feat, fid, "polar_pattern", vc)
    _enum(feat, fid, "polar_pattern", "axis", AXIS_NAMES, vc, "z")


def _check_mirror(feat: dict, fid: str, vc) -> None:
    _check_of(feat, fid, "mirror", vc)
    _enum(feat, fid, "mirror", "plane", MIRROR_PLANES, vc)


def _check_boss(feat: dict, fid: str, vc) -> None:
    _enum(feat, fid, "boss", "face", FACE_RULES, vc)
    S._check_profiles(feat, fid, vc)


def _register() -> None:
    S.register_op(S.OpSpec(
        "fillet", "solid", "dress", ("id", "op", "note", "size", "edges"),
        required=("size",), dims={"size": "positive"}, since=2,
        check=_check_dress("fillet"), nondim=("edges",)))
    S.register_op(S.OpSpec(
        "chamfer", "solid", "dress",
        ("id", "op", "note", "size", "size2", "edges"),
        required=("size",), dims={"size": "positive", "size2": "positive"},
        since=2, check=_check_dress("chamfer"), nondim=("edges",)))
    S.register_op(S.OpSpec(
        "thickness", "solid", "any",
        ("id", "op", "note", "wall", "open_faces", "direction"),
        required=("wall",), dims={"wall": "positive"}, since=2,
        check=_check_thickness, nondim=("open_faces",)))
    S.register_op(S.OpSpec(
        "linear_pattern", "solid", "follow",
        ("id", "op", "note", "of", "count", "spacing", "direction"),
        required=("of", "count", "spacing"),
        dims={"count": "count", "spacing": "positive"}, since=2,
        check=_check_linear, nondim=("of",)))
    S.register_op(S.OpSpec(
        "polar_pattern", "solid", "follow",
        ("id", "op", "note", "of", "count", "angle", "axis"),
        required=("of", "count", "angle"),
        dims={"count": "count", "angle": "angle360"}, since=2,
        check=_check_polar, nondim=("of",)))
    S.register_op(S.OpSpec(
        "mirror", "solid", "follow", ("id", "op", "note", "of", "plane"),
        required=("of", "plane"), since=2, check=_check_mirror,
        nondim=("of",)))
    S.register_op(S.OpSpec(
        "boss", "solid", "additive",
        ("id", "op", "note", "face", "profile", "profiles", "length",
         "invert"),
        required=("face", "length"), dims={"length": "positive"}, since=2,
        check=_check_boss, nondim=("face", "profile", "profiles")))


_register()
FEATURE_OPS_V2 = tuple(n for n, o in S.OPS.items() if o.since >= 2)
assert FEATURE_OPS_V2 == S.FEATURE_OPS_V2, "schema.FEATURE_OPS_V2 and the registered v2 ops differ"


# ---------------------------------------------------------------------------
# plan-level blocks


def check_placement(spec, what: str, params: dict, errors: list,
                    warnings: list) -> None:
    """A placement block: offsets ``x`` / ``y`` / ``z`` (mm) and rotations
    ``rx`` / ``ry`` / ``rz`` (degrees about the world axes), each a parameter
    name (or a number, which draws a warning unless it is 0)."""
    if not isinstance(spec, dict):
        errors.append(f"{what}: placement must be an object {{x, y, z, rx, "
                      f"ry, rz}} of parameter names, got {spec!r}")
        return
    S._unknown_keys(spec, PLACEMENT_KEYS, f"{what} placement", errors)
    for key in PLACEMENT_KEYS[:6]:
        if key not in spec:
            continue
        v = spec[key]
        if S._is_param_ref(v):
            if v not in params:
                errors.append(f"{what}: placement.{key} references unknown "
                              f"parameter {v!r}{S.suggest(v, params)}")
        elif not S._num(v):
            errors.append(f"{what}: placement.{key} must be a parameter name "
                          f"(or a number), got {v!r}")
        elif v != 0:
            warnings.append(f"{what}: literal placement.{key} {v:g} — "
                            f"offsets and rotations should bind a parameter "
                            f"name so the assembly stays editable")


def check_parts(plan: dict, params: dict, frame: dict, axes: dict,
                errors: list, warnings: list, seen_ids: set,
                validate_features) -> None:
    """Validate ``plan['parts']`` and every part's feature list (through
    ``validate_features``, which keeps feature ids unique ACROSS parts)."""
    parts = plan.get("parts")
    if not isinstance(parts, list) or not parts:
        errors.append("parts must be a non-empty list of {id, features, "
                      "placement} objects")
        return
    part_ids: set[str] = set()
    bodies: set[str] = set()
    for i, part in enumerate(parts):
        pid = part.get("id") if isinstance(part, dict) else None
        label = pid if isinstance(pid, str) and pid else f"#{i}"
        if not isinstance(part, dict):
            errors.append(f"part {label}: must be an object")
            continue
        S._unknown_keys(part, PART_KEYS, f"part {label}", errors)
        if not isinstance(pid, str) or not _IDENT.match(pid):
            errors.append(f"part #{i}: id must be an identifier (letters, "
                          f"digits, underscore; not starting with a digit), "
                          f"got {pid!r}")
        elif pid in part_ids:
            errors.append(f"part {pid}: duplicate id")
        else:
            part_ids.add(pid)
        body = part.get("body")
        if body is not None:
            if not isinstance(body, str) or not _IDENT.match(body):
                errors.append(f"part {label}: body must be an object name "
                              f"(letters, digits, underscore), got {body!r}")
            elif body in bodies:
                errors.append(f"part {label}: body name {body!r} is used by "
                              f"another part")
            else:
                bodies.add(body)
        feats = part.get("features")
        if not isinstance(feats, list) or not feats:
            errors.append(f"part {label}: features must be a non-empty "
                          f"ordered list")
            feats = []
        validate_features(feats, plan, params, frame, axes, 2, errors,
                          warnings, seen_ids)
        if part.get("placement") is not None:
            check_placement(part["placement"], f"part {label}", params,
                            errors, warnings)


def check_assembly(plan: dict, params: dict, errors: list,
                   warnings: list) -> None:
    """``assembly`` (named placements of parts, parameter driven) and
    ``references`` (mating faces). Both need ``parts``."""
    parts = [p for p in (plan.get("parts") or []) if isinstance(p, dict)]
    part_ids = {p.get("id") for p in parts if isinstance(p.get("id"), str)}
    placed = {p.get("id") for p in parts if p.get("placement") is not None}
    asm, refs = plan.get("assembly"), plan.get("references")
    if (asm is not None or refs is not None) and not parts:
        errors.append("assembly / references need parts[] (schema 2): there "
                      "is nothing to place")
        return
    if asm is not None:
        _check_assembly_block(asm, params, part_ids, placed, errors, warnings)
    if refs is not None:
        _check_references(refs, params, part_ids, errors)


def _check_assembly_block(asm, params, part_ids, placed, errors, warnings):
    if not isinstance(asm, dict):
        errors.append(f"assembly must be an object {{placements: [...]}}, "
                      f"got {type(asm).__name__}")
        return
    S._unknown_keys(asm, ASSEMBLY_KEYS, "assembly", errors)
    plist = asm.get("placements")
    if not isinstance(plist, list) or not plist:
        errors.append("assembly.placements must be a non-empty list of "
                      "{name, part, x, y, z, rx, ry, rz}")
        return
    names: set[str] = set()
    targets: set[str] = set()
    for i, pl in enumerate(plist):
        if not isinstance(pl, dict):
            errors.append(f"assembly.placements[{i}]: must be an object")
            continue
        name = pl.get("name")
        label = name if isinstance(name, str) and name else f"#{i}"
        S._unknown_keys(pl, ASSEMBLY_PLACEMENT_KEYS,
                        f"assembly placement {label}", errors)
        if not isinstance(name, str) or not _IDENT.match(name):
            errors.append(f"assembly.placements[{i}]: name must be an "
                          f"identifier, got {name!r}")
        elif name in names:
            errors.append(f"assembly placement {name}: duplicate name")
        else:
            names.add(name)
        part = pl.get("part")
        if part not in part_ids:
            errors.append(f"assembly placement {label}: part {part!r} is not "
                          f"in parts{S.suggest(part, part_ids)} "
                          f"(have: {sorted(part_ids)})")
        elif part in targets:
            errors.append(f"assembly placement {label}: part {part!r} is "
                          f"already placed by another assembly placement")
        elif part in placed:
            errors.append(f"assembly placement {label}: part {part!r} has its "
                          f"own 'placement'; keep one of the two")
        else:
            targets.add(part)
        check_placement({k: v for k, v in pl.items()
                         if k in PLACEMENT_KEYS}, f"assembly placement {label}",
                        params, errors, warnings)


def _check_references(refs, params, part_ids, errors) -> None:
    if not isinstance(refs, list) or not refs:
        errors.append("references must be a non-empty list of {id, kind, a, "
                      "b} objects")
        return
    for i, ref in enumerate(refs):
        rid = ref.get("id") if isinstance(ref, dict) else None
        label = rid if isinstance(rid, str) and rid else f"#{i}"
        if not isinstance(ref, dict):
            errors.append(f"reference {label}: must be an object")
            continue
        S._unknown_keys(ref, REFERENCE_KEYS, f"reference {label}", errors)
        if ref.get("kind") not in REFERENCE_KINDS:
            errors.append(f"reference {label}: kind {ref.get('kind')!r} not "
                          f"in {list(REFERENCE_KINDS)}")
        sides = []
        for side in ("a", "b"):
            spec = ref.get(side)
            if not isinstance(spec, dict) or set(spec) - {"part", "face"}:
                errors.append(f"reference {label}: {side} must be {{'part': "
                              f"<part id>, 'face': one of {list(FACE_RULES)}}}"
                              f", got {spec!r}")
                continue
            if spec.get("part") not in part_ids:
                errors.append(f"reference {label}: {side}.part "
                              f"{spec.get('part')!r} is not in parts"
                              f"{S.suggest(spec.get('part'), part_ids)}")
            if spec.get("face") not in FACE_RULES:
                errors.append(f"reference {label}: {side}.face "
                              f"{spec.get('face')!r} not in "
                              f"{list(FACE_RULES)}"
                              f"{S.suggest(spec.get('face'), FACE_RULES)}")
            sides.append(spec.get("part"))
        if len(sides) == 2 and sides[0] == sides[1]:
            errors.append(f"reference {label}: a and b are the same part "
                          f"({sides[0]!r}); a mating pair joins two parts")
        gap = ref.get("gap")
        if gap is not None and not (S._is_param_ref(gap) and gap in params):
            errors.append(f"reference {label}: gap must be a known parameter "
                          f"name, got {gap!r}{S.suggest(gap, params)}")
