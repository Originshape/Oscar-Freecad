"""Solid-feature translation of the FeaturePlan → FreeCAD builder (private to oscar.plan).

Pad / pocket / revolution / hole, each created inside the Body from a sketch
the sketch pass built, every dimension that names a plan parameter bound by
expression to ``<sheet>.<alias>``. FreeCAD property names are the ones of
1.1 (verified on the installed app): a pocket's depth is ``Length`` (there is
no ``Depth``), a counterbore / countersink lives in ``HoleCutType`` /
``HoleCutDiameter`` / ``HoleCutDepth`` / ``HoleCutCountersinkAngle``.

Each ``build_*`` takes ``(doc, body, feat, plan, built, ctx)`` and returns
``(object, info)``; the caller recomputes, checks the post-conditions and
rolls back on failure.
"""

import math

from . import _sketchbuild, schema
from ._buildcore import (BuildError, Ctx, orthonormal_placement, ref_expr,
                         resolve, tag_object)


def _length(plan: dict, feat: dict, key: str, what: str) -> float:
    """The number behind one dimensional field of a feature."""
    return resolve(plan, feat[key], f"{feat['id']} {what}", feat["id"])


def _new(body, type_id: str, feat: dict, ctx: Ctx, suffix: str = ""):
    """A feature object inside the Body, named, labelled and tagged."""
    name = ctx.name(feat["id"]) + suffix
    obj = body.newObject(type_id, name)
    obj.Label = name
    tag_object(obj, ctx.plan_id, feat["id"] + suffix)
    return obj


def _bind(obj, prop: str, ref, ctx: Ctx) -> None:
    expr = ref_expr(ctx.sheet, ref)
    if expr:
        obj.setExpression(prop, expr)


def build_pad(doc, body, feat: dict, plan: dict, built: dict, ctx: Ctx):
    """PartDesign::Pad of an earlier sketch; ``length`` bound to its
    parameter, ``midplane`` / ``invert`` honoured."""
    sketch = built[feat["sketch"]][0]
    pad = _new(body, "PartDesign::Pad", feat, ctx)
    pad.Profile = sketch
    pad.Length = _length(plan, feat, "length", "length")
    _bind(pad, "Length", feat["length"], ctx)
    if feat.get("midplane"):
        pad.Midplane = True
    if feat.get("invert"):
        pad.Reversed = True
    doc.recompute()
    return pad, {}


def build_pocket(doc, body, feat: dict, plan: dict, built: dict, ctx: Ctx):
    """PartDesign::Pocket of an earlier sketch: through-all, or a blind
    pocket whose ``depth`` is bound to ``Length`` (the FreeCAD property)."""
    sketch = built[feat["sketch"]][0]
    pocket = _new(body, "PartDesign::Pocket", feat, ctx)
    pocket.Profile = sketch
    if feat.get("through"):
        pocket.Type = "ThroughAll"
    else:
        pocket.Type = "Length"
        pocket.Length = _length(plan, feat, "depth", "depth")
        _bind(pocket, "Length", feat["depth"], ctx)
    if feat.get("midplane"):
        pocket.Midplane = True
    if feat.get("invert"):
        pocket.Reversed = True
    doc.recompute()
    return pocket, {}


def build_revolution(doc, body, feat: dict, plan: dict, built: dict, ctx: Ctx):
    """PartDesign::Revolution of an earlier sketch about an in-plane axis
    (frame axis or explicit direction); ``angle`` defaults to 360."""
    sketch = built[feat["sketch"]][0]
    rev = _new(body, "PartDesign::Revolution", feat, ctx)
    rev.Profile = sketch
    axis = feat.get("axis", {})
    if "frame_axis" in axis:
        _align_revolution_axis(rev, sketch, plan, feat,
                               plan["frame"]["axes"][axis["frame_axis"]])
    elif "direction" in axis:
        _align_revolution_axis(rev, sketch, plan, feat, axis)
    else:
        rev.ReferenceAxis = (sketch, ["V_Axis"])
    angle = feat.get("angle")
    if angle is not None:
        rev.Angle = _length(plan, feat, "angle", "angle")
        _bind(rev, "Angle", angle, ctx)
    if feat.get("midplane"):
        rev.Midplane = True
    if feat.get("invert"):
        rev.Reversed = not rev.Reversed   # on top of a negative axis direction
    doc.recompute()
    return rev, {}


def _align_revolution_axis(rev, sketch, plan: dict, feat: dict, spec: dict) -> None:
    """Set ReferenceAxis (and the sweep direction) for an axis spec relative
    to the sketch plane.

    v1 supports the clean cases: the world axis lies IN the sketch plane,
    along its x or y direction and through its origin (→ the sketch's H/V
    axis); a negative direction reverses the sweep. Axes skew to the plane,
    or running parallel but offset from the sketch's own axis, are rejected
    with an actionable error rather than silently revolved about the wrong
    line."""
    feat_by_id = {f["id"]: f for f in plan["features"]}
    # rev.Profile reads back as a (obj, subs) link: use the sketch we hold
    sketch_feat = feat_by_id.get(feat["sketch"])
    basis = schema.sketch_basis(plan, sketch_feat.get("plane", {}))
    d = [float(v) for v in spec["direction"]]
    norm = math.sqrt(sum(c * c for c in d))
    d = [c / norm for c in d]
    point = [float(v) for v in spec.get("point", [0, 0, 0])]
    off = [p - o for p, o in zip(point, basis["origin"])]
    for vec, cross, ref in ((basis["x_dir"], basis["y_dir"], "H_Axis"),
                            (basis["y_dir"], basis["x_dir"], "V_Axis")):
        dot = sum(a * b for a, b in zip(d, vec))
        if abs(abs(dot) - 1.0) < 1e-6:
            # parallel is not enough: the axis must pass through the sketch's
            # H/V axis (no offset in the plane across it, none along the normal)
            miss = max(abs(sum(a * b for a, b in zip(off, cross))),
                       abs(sum(a * b for a, b in zip(off, basis["normal"]))))
            if miss > 1e-6:
                raise BuildError(
                    feat["id"],
                    f"revolution axis does not pass through the sketch's "
                    f"{ref[0]} axis (off by {miss:.3g} mm) — move the sketch "
                    f"plane origin onto the axis")
            rev.ReferenceAxis = (sketch, [ref])
            rev.Reversed = dot < 0
            return
    raise BuildError(feat["id"],
                     "revolution axis is not in the sketch plane — align the "
                     "sketch frame or use an explicit in-plane axis (v1)")


def build_hole(doc, body, feat: dict, plan: dict, built: dict, ctx: Ctx):
    """A one-circle sketch + PartDesign::Hole.

    Through-all or blind (flat bottom, so the depth is the cylinder depth);
    a counterbore / countersink — flat or nested spelling — sets the
    ``HoleCut*`` properties, each bound to its parameter."""
    import FreeCAD as App
    import Part
    import Sketcher

    fid = feat["id"]
    try:
        cut = schema.hole_cut(feat)
    except ValueError as e:
        raise BuildError(fid, str(e))
    basis = schema.sketch_basis(plan, feat.get("plane", {}))
    sketch = _new(body, "Sketcher::SketchObject", feat, ctx, "_sketch")
    sketch.Placement = orthonormal_placement(basis)
    _sketchbuild.bind_placement(sketch, basis, ctx)

    x, y = feat["position"]["x"], feat["position"]["y"]
    cx = resolve(plan, x, f"{fid} x", fid)
    cy = resolve(plan, y, f"{fid} y", fid)
    dia = resolve(plan, feat["diameter"], f"{fid} diameter", fid)
    idx = sketch.addGeometry(
        Part.Circle(App.Vector(cx, cy, 0), App.Vector(0, 0, 1), dia / 2), False)

    bound = 0

    def named(constraint, name, ref):
        nonlocal bound
        i = sketch.addConstraint(constraint)
        sketch.renameConstraint(i, name)
        expr = ref_expr(ctx.sheet, ref)
        if expr:
            sketch.setExpression(f".Constraints.{name}", expr)
            bound += 1
    named(Sketcher.Constraint("Diameter", idx, dia), "diameter", feat["diameter"])
    named(Sketcher.Constraint("DistanceX", -1, 1, idx, 3, cx), "cx", x)
    named(Sketcher.Constraint("DistanceY", -1, 1, idx, 3, cy), "cy", y)
    doc.recompute()
    info = _sketchbuild.sketch_report(sketch, "circle", bound, fid)

    hole = _new(body, "PartDesign::Hole", feat, ctx)
    hole.Profile = sketch
    hole.Diameter = dia
    _bind(hole, "Diameter", feat["diameter"], ctx)
    if feat.get("through"):
        hole.DepthType = "ThroughAll"
    else:
        hole.DepthType = "Dimension"
        hole.DrillPoint = "Flat"
        hole.Depth = _length(plan, feat, "depth", "depth")
        _bind(hole, "Depth", feat["depth"], ctx)
    if cut:
        _apply_hole_cut(hole, cut, plan, fid, ctx)
    doc.recompute()
    info["sketch"] = sketch.Name
    return hole, info


def _apply_hole_cut(hole, cut: dict, plan: dict, fid: str, ctx: Ctx) -> None:
    """Counterbore / countersink properties of a Hole (``schema.hole_cut``
    already normalised the flat and nested spellings)."""
    kind = cut["type"]
    hole.HoleCutType = "Counterbore" if kind == "counterbore" else "Countersink"
    hole.HoleCutDiameter = resolve(plan, cut["diameter"], f"{fid} {kind} diameter", fid)
    _bind(hole, "HoleCutDiameter", cut["diameter"], ctx)
    if kind == "counterbore":
        hole.HoleCutDepth = resolve(plan, cut["depth"], f"{fid} counterbore depth", fid)
        _bind(hole, "HoleCutDepth", cut["depth"], ctx)
    elif cut["angle"] is not None:
        hole.HoleCutCountersinkAngle = resolve(plan, cut["angle"],
                                               f"{fid} countersink angle", fid)
        _bind(hole, "HoleCutCountersinkAngle", cut["angle"], ctx)


# solid-feature translators by op ("sketch" is handled by the entry points):
# a plain dict the builder dispatches through. ``register_solid`` records a
# translator both here and on the op registry (``schema.OPS[op].builder``), so
# the table and the registry cannot disagree; ``_dressbuild`` registers the
# schema-v2 ops the same way.
SOLID_BUILDERS: dict = {}


def register_solid(op: str, fn) -> None:
    """Attach ``fn`` as the translator of the registered solid op ``op``."""
    schema.attach_builder(op, fn)
    SOLID_BUILDERS[op] = fn


for _op, _fn in (("pad", build_pad), ("pocket", build_pocket),
                 ("revolution", build_revolution), ("hole", build_hole)):
    register_solid(_op, _fn)
