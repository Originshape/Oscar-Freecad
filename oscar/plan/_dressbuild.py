"""Schema-v2 solid features of the FeaturePlan → FreeCAD builder (private).

fillet / chamfer / thickness (shell) dress the Body's current tip, the
patterns and the mirror repeat an earlier feature, a boss pads a profile on a
face. Edge and face choices are RULES resolved against the real shape at
build time (``_edgerules``), never indices. Every dimension that names a plan
parameter is bound by expression to ``<sheet>.<alias>``, so editing a cell
re-drives the feature (the pattern ``Occurrences`` included).

Each ``build_*`` takes ``(doc, body, feat, plan, built, ctx)`` and returns
``(object, info)``; the caller recomputes, checks the post-conditions and
rolls back on failure.

FreeCAD 1.1 and the first recompute: an edge / face link made by index (what a
rule resolves to) can turn Invalid the first time an upstream feature
recomputes - editing ANY dimension afterwards would break the fillet. Every
such feature is therefore "settled" once its transaction is committed (the
builder calls ``settle``): the Body's features are touched and recomputed until
the feature stays valid, re-selecting its edges from the rule when it does not.
The rule is also kept on the object (``OscarPlanRule``) so ``refresh_edge_rules``
can repair it later.

Boss sketch frame (the sketch is attached to the face, then normalised so the
coordinates mean the same thing whatever FreeCAD's default would be): on
``top`` x = world X, y = world Y; on ``bottom`` x = X, y = -Y (the sketch
looks at the face from outside); on ``front`` / ``back`` / ``left`` / ``right``
y = world Z and x runs X, -X, -Y, +Y respectively. The origin is the
projection of the Body origin onto the face plane.
"""

from . import _edgerules, _featurebuild, _sketchbuild, schema
from ._schema_v2 import OPEN_FACE_RULES
from ._buildcore import BuildError, Ctx, feature_of, ref_expr, resolve
from ._featurebuild import _bind, _length, _new

# sketch x / y axes (local frame) per boss face rule; the normal is x cross y
_BOSS_FRAME = {
    "top": ((1, 0, 0), (0, 1, 0)), "bottom": ((1, 0, 0), (0, -1, 0)),
    "front": ((1, 0, 0), (0, 0, 1)), "back": ((-1, 0, 0), (0, 0, 1)),
    "left": ((0, -1, 0), (0, 0, 1)), "right": ((0, 1, 0), (0, 0, 1)),
}
_FACE_AXIS = {"top": (2, "ZMax"), "bottom": (2, "ZMin"), "front": (1, "YMin"),
              "back": (1, "YMax"), "left": (0, "XMin"), "right": (0, "XMax")}
RULE_PROP = "OscarPlanRule"        # the edge / open-face rule a dressing feature was built from
_AXIS_ROLES = {"x": "X_Axis", "y": "Y_Axis", "z": "Z_Axis"}
_PLANE_ROLES = {"xy": "XY_Plane", "xz": "XZ_Plane", "yz": "YZ_Plane"}


def _tip(body, feat: dict):
    """The Body's current tip feature (what a dressing feature rounds)."""
    tip = getattr(body, "Tip", None)
    if tip is None:
        raise BuildError(feat["id"], f"{feat['op']} needs an existing solid: "
                                     f"the Body has no feature yet")
    return tip


def _origin(body, role: str):
    for o in body.Origin.OriginFeatures:
        if o.Role == role:
            return o
    raise BuildError("", f"the Body has no origin feature {role!r}")


def _feature_shape(built: dict, fid: str, target: str):
    """The geometry the feature ``target`` added or removed (a PartDesign
    feature's ``AddSubShape``, else the object's own shape, e.g. a sketch's
    wires), in the Body-local frame."""
    if target not in built:
        raise BuildError(fid, f"edges_of_feature:{target}: feature {target!r} "
                              f"is not built yet")
    obj = built[target][0]
    shape = getattr(obj, "AddSubShape", None)
    if shape is None or shape.isNull():
        shape = obj.Shape
    return _edgerules.local_copy(shape)


def _edges(feat: dict, tip, built: dict) -> list[str]:
    rule = feat.get("edges", "all_outer")
    target = rule.split(":", 1)[1] if rule.startswith("edges_of_feature:") else None
    fshape = _feature_shape(built, feat["id"], target) if target else None
    try:
        return _edgerules.select_edges(rule, _edgerules.local_copy(tip.Shape), fshape)
    except BuildError as e:
        raise BuildError(feat["id"], f"{feat['op']}.edges: {e.message}") from None


def _keep_rule(obj, rule: str) -> None:
    if RULE_PROP not in obj.PropertiesList:
        obj.addProperty("App::PropertyString", RULE_PROP, "Oscar",
                        "Edge / face rule this feature was built from")
    setattr(obj, RULE_PROP, rule)


def _reselect(obj, built: dict) -> None:
    """Assign ``obj.Base`` again from the rule stored on it (edges for a
    fillet / chamfer, planar faces for a thickness)."""
    rule = getattr(obj, RULE_PROP)
    if rule in OPEN_FACE_RULES:
        base = obj.Base[0]
        obj.Base = (base, _edgerules.select_faces(rule, _edgerules.local_copy(base.Shape)))
    else:
        _reselect_edges(obj, rule, built)


SETTLE_CYCLES = 3      # measured on FreeCAD 1.1.3: one cycle is not enough, three keep every later edit valid


def settle(doc, body, obj, built: dict) -> None:
    """Touch every feature of the Body and recompute ``SETTLE_CYCLES`` times
    so the dressing feature ``obj`` keeps a valid edge / face link through every
    later edit (see the module docstring), re-selecting its edges from the
    stored rule whenever the link goes stale. A no-op for any other object.
    Raises BuildError when the feature cannot be made valid."""
    if RULE_PROP not in getattr(obj, "PropertiesList", []) or "Invalid" in obj.State:
        return
    for _ in range(SETTLE_CYCLES):
        for o in list(body.Group):
            if o.isDerivedFrom("PartDesign::Feature") or o.isDerivedFrom("Sketcher::SketchObject"):
                o.touch()
        doc.recompute()
        if "Invalid" in obj.State:
            _reselect(obj, built)
            doc.recompute()
        if "Invalid" in obj.State:
            raise BuildError("", f"{obj.Name}: the edge / face link does not survive a recompute "
                                 f"({obj.getStatusString()})")


def _reselect_edges(obj, rule: str, built: dict) -> None:
    base = obj.Base[0]
    target = rule.split(":", 1)[1] if rule.startswith("edges_of_feature:") else None
    fshape = _feature_shape(built, "", target) if target else None
    obj.Base = (base, _edgerules.select_edges(rule, _edgerules.local_copy(base.Shape), fshape))


def refresh_edge_rules(doc) -> list[str]:
    """Repair every dressing feature (fillet / chamfer) the plan builder made
    that FreeCAD has marked Invalid because its edge link went stale: the edges
    are chosen again from the rule stored on the object. Returns the names of
    the features that were repaired (empty when nothing needed it)."""
    repaired: list[str] = []
    for obj in list(doc.Objects):
        if RULE_PROP not in getattr(obj, "PropertiesList", []) or "Invalid" not in obj.State:
            continue
        rule = getattr(obj, RULE_PROP)
        if not rule.startswith("edges_of_feature:") and rule not in ("all_outer", "vertical", "top_outer", "bottom_outer"):
            continue
        members = {feature_of(o): (o, None) for o in doc.Objects if feature_of(o)}
        try:
            _reselect_edges(obj, rule, members)
            doc.recompute()
        except Exception:  # noqa: BLE001 - a repair that cannot be done leaves the feature as it was
            continue
        if "Invalid" not in obj.State:
            repaired.append(obj.Name)
    return repaired


def build_fillet(doc, body, feat: dict, plan: dict, built: dict, ctx: Ctx):
    """PartDesign::Fillet of the tip's edges chosen by rule; ``size`` bound to
    ``Radius``."""
    tip = _tip(body, feat)
    names = _edges(feat, tip, built)
    obj = _new(body, "PartDesign::Fillet", feat, ctx)
    obj.Base = (tip, names)
    obj.Radius = _length(plan, feat, "size", "size")
    _bind(obj, "Radius", feat["size"], ctx)
    _keep_rule(obj, feat.get("edges", "all_outer"))
    doc.recompute()
    return obj, {"edges": len(names)}


def build_chamfer(doc, body, feat: dict, plan: dict, built: dict, ctx: Ctx):
    """PartDesign::Chamfer: ``size`` (equal distance) or ``size`` + ``size2``
    (two distances), bound to ``Size`` / ``Size2``."""
    tip = _tip(body, feat)
    names = _edges(feat, tip, built)
    obj = _new(body, "PartDesign::Chamfer", feat, ctx)
    obj.Base = (tip, names)
    obj.Size = _length(plan, feat, "size", "size")
    _bind(obj, "Size", feat["size"], ctx)
    if feat.get("size2") is not None:
        obj.ChamferType = "Two distances"
        obj.Size2 = _length(plan, feat, "size2", "size2")
        _bind(obj, "Size2", feat["size2"], ctx)
    _keep_rule(obj, feat.get("edges", "all_outer"))
    doc.recompute()
    return obj, {"edges": len(names)}


def build_thickness(doc, body, feat: dict, plan: dict, built: dict, ctx: Ctx):
    """PartDesign::Thickness (shell): the planar face(s) selected by
    ``open_faces`` are removed and the walls grow inward (default) or outward
    by ``wall``."""
    tip = _tip(body, feat)
    rule = feat.get("open_faces", "top")
    try:
        faces = _edgerules.select_faces(rule, _edgerules.local_copy(tip.Shape))
    except BuildError as e:
        raise BuildError(feat["id"], f"thickness.open_faces: {e.message}") from None
    obj = _new(body, "PartDesign::Thickness", feat, ctx)
    obj.Base = (tip, faces)
    obj.Value = _length(plan, feat, "wall", "wall")
    _bind(obj, "Value", feat["wall"], ctx)
    obj.Mode = "Skin"
    obj.Join = "Intersection"
    obj.Reversed = feat.get("direction", "inward") == "inward"
    _keep_rule(obj, rule)
    doc.recompute()
    return obj, {"open_faces": faces}


def _transformed(doc, body, feat: dict, ctx: Ctx, type_id: str, original):
    """A PartDesign pattern / mirror of ``original``, linked into the Body's
    chain. FreeCAD does not treat a Transformed feature as a solid feature until
    its Originals are set, so ``body.newObject`` leaves the Tip (and the
    feature's BaseFeature) where they were: both are set by hand."""
    before = getattr(body, "Tip", None)
    obj = _new(body, type_id, feat, ctx)
    obj.Originals = [original]
    if before is not None and obj.BaseFeature is None:
        obj.BaseFeature = before
    body.Tip = obj
    return obj


def _original(feat: dict, built: dict):
    ref = feat["of"]
    if ref not in built:
        raise BuildError(feat["id"], f"{feat['op']}.of: feature {ref!r} is not "
                                     f"built yet")
    return built[ref][0]


def _count(doc, obj, feat: dict, plan: dict, ctx: Ctx) -> int:
    """Set ``Occurrences`` from the ``count`` parameter (bound by expression
    so editing the cell re-patterns)."""
    n = resolve(plan, feat["count"], f"{feat['id']} count", feat["id"])
    if abs(n - round(n)) > 1e-9 or n < 2:
        raise BuildError(feat["id"], f"count {feat['count']!r} = {n:g} must be a "
                                     f"whole number >= 2")
    obj.Occurrences = int(round(n))
    _bind(obj, "Occurrences", feat["count"], ctx)
    return int(round(n))


def build_linear_pattern(doc, body, feat: dict, plan: dict, built: dict, ctx: Ctx):
    """PartDesign::LinearPattern of an earlier feature along a Body axis:
    ``count`` occurrences ``spacing`` apart (``Length`` = spacing * (count-1),
    bound as that expression)."""
    original = _original(feat, built)
    direction = feat.get("direction", "x")
    axis = _origin(body, _AXIS_ROLES[direction.lstrip("-")])
    obj = _transformed(doc, body, feat, ctx, "PartDesign::LinearPattern", original)
    obj.Direction = (axis, [""])
    obj.Reversed = direction.startswith("-")
    n = _count(doc, obj, feat, plan, ctx)
    spacing = _length(plan, feat, "spacing", "spacing")
    obj.Length = spacing * (n - 1)
    obj.setExpression("Length", f"{ref_expr(ctx.sheet, feat['spacing'])} * "
                                f"({ref_expr(ctx.sheet, feat['count'])} - 1)")
    doc.recompute()
    return obj, {"occurrences": n}


def build_polar_pattern(doc, body, feat: dict, plan: dict, built: dict, ctx: Ctx):
    """PartDesign::PolarPattern of an earlier feature about a Body axis:
    ``count`` occurrences over ``angle`` degrees (360 = a full circle, evenly
    spaced)."""
    original = _original(feat, built)
    axis = _origin(body, _AXIS_ROLES[feat.get("axis", "z")])
    obj = _transformed(doc, body, feat, ctx, "PartDesign::PolarPattern", original)
    obj.Axis = (axis, [""])
    n = _count(doc, obj, feat, plan, ctx)
    obj.Angle = _length(plan, feat, "angle", "angle")
    _bind(obj, "Angle", feat["angle"], ctx)
    doc.recompute()
    return obj, {"occurrences": n}


def build_mirror(doc, body, feat: dict, plan: dict, built: dict, ctx: Ctx):
    """PartDesign::Mirrored of an earlier feature across a Body origin plane."""
    original = _original(feat, built)
    plane = _origin(body, _PLANE_ROLES[feat["plane"]])
    obj = _transformed(doc, body, feat, ctx, "PartDesign::Mirrored", original)
    obj.MirrorPlane = (plane, [""])
    doc.recompute()
    return obj, {}


def _normalise_attachment(sketch, body, face: str, rule: str, tip_shape) -> None:
    """Orient and place an attached sketch per ``_BOSS_FRAME``: origin at the
    projection of the Body origin onto the face plane, x / y along the rule's
    axes. Works through ``AttachmentOffset``; raises BuildError when the
    result does not match."""
    import FreeCAD as App

    axis, side = _FACE_AXIS[rule]
    xs, ys = (App.Vector(*v) for v in _BOSS_FRAME[rule])
    zs = xs.cross(ys)
    bb = tip_shape.BoundBox
    plane_at = getattr(bb, side)
    origin = [0.0, 0.0, 0.0]
    origin[axis] = plane_at
    m = App.Matrix(xs.x, ys.x, zs.x, 0, xs.y, ys.y, zs.y, 0,
                   xs.z, ys.z, zs.z, 0, 0, 0, 0, 1)
    want = body.Placement.multiply(App.Placement(App.Vector(*origin),
                                                 App.Placement(m).Rotation))
    have = sketch.Placement
    off = have.inverse().multiply(want)
    sketch.AttachmentOffset = off
    sketch.recompute()
    got = sketch.Placement
    if (got.Base - want.Base).Length > 1e-6 or not got.Rotation.isSame(want.Rotation, 1e-9):
        raise BuildError("", f"could not orient the sketch on the {rule} face "
                             f"(placement {got} instead of {want})")


def build_boss(doc, body, feat: dict, plan: dict, built: dict, ctx: Ctx):
    """A pad of ``profile`` / ``profiles`` of ``length`` on the ``face`` (a
    rule), through a sketch ATTACHED to that face (so the boss follows the
    face when a parameter moves it). The sketch is named ``<id>_sketch``."""
    fid = feat["id"]
    rule = feat["face"]
    tip = _tip(body, feat)
    local = _edgerules.local_copy(tip.Shape)
    try:
        face = _edgerules.select_faces(rule, local, many=False)[0]
    except BuildError as e:
        raise BuildError(fid, f"boss.face: {e.message}") from None
    sketch = _new(body, "Sketcher::SketchObject", feat, ctx, "_sketch")
    sketch.AttachmentSupport = (tip, [face])
    sketch.MapMode = "FlatFace"
    doc.recompute()
    try:
        _normalise_attachment(sketch, body, face, rule, local)
    except BuildError as e:
        raise BuildError(fid, e.message) from None
    info = _sketchbuild.fill_sketch(doc, sketch, schema.feature_profiles(feat),
                                    plan, fid, ctx)
    pad = _new(body, "PartDesign::Pad", feat, ctx)
    pad.Profile = sketch
    pad.Length = _length(plan, feat, "length", "length")
    _bind(pad, "Length", feat["length"], ctx)
    if feat.get("invert"):
        pad.Reversed = True
    doc.recompute()
    info["sketch"] = sketch.Name
    info["face"] = face
    return pad, info


for _op, _fn in (("fillet", build_fillet), ("chamfer", build_chamfer),
                 ("thickness", build_thickness),
                 ("linear_pattern", build_linear_pattern),
                 ("polar_pattern", build_polar_pattern),
                 ("mirror", build_mirror), ("boss", build_boss)):
    _featurebuild.register_solid(_op, _fn)
