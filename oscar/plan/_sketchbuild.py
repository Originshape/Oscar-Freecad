"""Sketch translation of the FeaturePlan → FreeCAD builder (private to oscar.plan).

Turns a plan ``sketch`` feature (rectangle / circle / slot / polyline profile)
into a real ``Sketcher::SketchObject``: exact geometry, geometric and
dimensional constraints, and every dimension that names a plan parameter bound
by expression to the spreadsheet (``<sheet>.<alias>``), so editing a cell and
recomputing re-drives the profile.

Constraint names must not be unit symbols (``h``, ``m``, ``s`` …): FreeCAD
parses ``.Constraints.h`` as an expression with a unit and fails, so every
name below is a descriptive word or ``<word>_<n>``.
"""

import math

from ._buildcore import (BuildError, Ctx, center_expr, orthonormal_placement,
                         ref_expr, resolve, tag_object, term)
from . import schema


# ---------------------------------------------------------------------------
# geometry


def _angle_ref(profile: dict):
    """The slot angle in degrees as the planner spelled it (``angle_deg``
    or ``angle``; a number or a parameter name; 0 when absent)."""
    if "angle_deg" in profile:
        return profile["angle_deg"]
    return profile.get("angle", 0)


def add_geometry(sketch, profile: dict, plan: dict, feat_id: str) -> dict:
    """Create the geometry of one bounded profile; return what the
    constraint pass needs (geometry indices, resolved numbers).

    Rectangle / slot / circle build exact geometry; constraints come
    separately so the tree keeps a clean geometry → constraints story."""
    import FreeCAD as App
    import Part

    shape = profile.get("shape")
    nz = App.Vector(0, 0, 1)

    def num(value, what):
        return resolve(plan, value, f"{feat_id} {what}", feat_id)

    def pt(coord, what):
        return (num(coord[0], f"{what} x"), num(coord[1], f"{what} y"))

    if shape == "rectangle":
        w = num(profile["width"], "width")
        h = num(profile["height"], "height")
        cx, cy = pt(profile.get("center", [0, 0]), "center")
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        ids = []
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            ids.append(sketch.addGeometry(
                Part.LineSegment(App.Vector(a[0], a[1], 0),
                                 App.Vector(b[0], b[1], 0)), False))
        return {"shape": "rectangle", "lines": ids, "corners": corners,
                "width": w, "height": h, "center": (cx, cy)}

    if shape == "circle":
        d = num(profile["diameter"], "diameter")
        cx, cy = pt(profile.get("center", [0, 0]), "center")
        idx = sketch.addGeometry(
            Part.Circle(App.Vector(cx, cy, 0), nz, d / 2), False)
        return {"shape": "circle", "circle": idx, "diameter": d,
                "center": (cx, cy)}

    if shape == "slot":
        length = num(profile["length"], "length")
        width = num(profile["width"], "width")
        cx, cy = pt(profile.get("center", [0, 0]), "center")
        angle = math.radians(num(_angle_ref(profile), "angle"))
        r = width / 2
        flat = length - width
        ca, sa = math.cos(angle), math.sin(angle)

        def world(lx, ly):
            return (cx + lx * ca - ly * sa, cy + lx * sa + ly * ca)

        def vec(p):
            return App.Vector(p[0], p[1], 0)

        def cap(center, start, end):
            return Part.ArcOfCircle(Part.Circle(vec(center), nz, r),
                                    start + angle, end + angle)

        # obround, length along the slot's own x axis: bottom line, right cap,
        # top line, left cap (counter-clockwise, every end meets the next start)
        ids = [
            sketch.addGeometry(Part.LineSegment(
                vec(world(-flat / 2, -r)), vec(world(flat / 2, -r))), False),
            sketch.addGeometry(cap(world(flat / 2, 0), -math.pi / 2,
                                   math.pi / 2), False),
            sketch.addGeometry(Part.LineSegment(
                vec(world(flat / 2, r)), vec(world(-flat / 2, r))), False),
            sketch.addGeometry(cap(world(-flat / 2, 0), math.pi / 2,
                                   3 * math.pi / 2), False),
        ]
        # a construction point at the slot centre carries the position
        # dimensions (the cap centres are symmetric about it)
        centre_pt = sketch.addGeometry(
            Part.Point(App.Vector(cx, cy, 0)), True)
        return {"shape": "slot", "elements": ids, "center_point": centre_pt,
                "length": length, "width": width, "center": (cx, cy),
                "angle": angle}

    if shape == "profile":
        segs = profile.get("segments", [])
        cursor = pt(profile.get("start", [0, 0]), "start")
        start = cursor
        ids, meta = [], []
        for i, seg in enumerate(segs):
            to = pt(seg["to"], f"segment {i} to")
            if seg.get("type") == "line":
                idx = sketch.addGeometry(
                    Part.LineSegment(App.Vector(cursor[0], cursor[1], 0),
                                     App.Vector(to[0], to[1], 0)), False)
                ids.append(("line", idx))
                meta.append({"from": cursor, "to": to, "index": idx,
                             "ref_to": seg.get("to")})
            elif seg.get("type") == "arc":
                ctr = pt(seg["center"], f"segment {i} center")
                a0 = math.atan2(cursor[1] - ctr[1], cursor[0] - ctr[0])
                a1 = math.atan2(to[1] - ctr[1], to[0] - ctr[0])
                if not seg.get("ccw", True):
                    a0, a1 = a1, a0
                radius = math.hypot(cursor[0] - ctr[0], cursor[1] - ctr[1])
                idx = sketch.addGeometry(Part.ArcOfCircle(
                    Part.Circle(App.Vector(ctr[0], ctr[1], 0), nz, radius),
                    a0, a1), False)
                ids.append(("arc", idx))
                # FreeCAD arcs run counter-clockwise start -> end; a cw
                # segment has its angles swapped above, so its sketch point 1
                # is the segment's `to` and point 2 the cursor ("rev")
                meta.append({"from": cursor, "to": to, "center": ctr,
                             "ref_to": seg.get("to"), "index": idx,
                             "radius_ref": seg.get("radius"),
                             "rev": not seg.get("ccw", True)})
            else:
                raise BuildError(feat_id, f"segment {i}: unknown type "
                                          f"{seg.get('type')!r}")
            cursor = to
        closed = math.hypot(cursor[0] - start[0], cursor[1] - start[1]) < 1e-6
        return {"shape": "profile", "ids": ids, "meta": meta,
                "closed": closed, "start": start}

    raise BuildError(feat_id, f"profile shape {shape!r} not supported")


# ---------------------------------------------------------------------------
# constraints


def constrain(sketch, geom: dict, profile: dict, plan: dict, feat_id: str,
              ctx: Ctx, prefix: str = "") -> int:
    """Geometric + dimensional constraints, dimensions bound to the
    spreadsheet; returns the number of dimensional expressions bound.

    Every closed profile ends fully constrained: the rectangle is placed by
    its lower-left corner (``center - size / 2`` as an expression, so the
    centre — literal or parameter — is honoured), the circle and the slot by
    their centre, the polyline by its start point and per-segment deltas.
    ``prefix`` goes in front of every constraint name (the second profile of a
    multi-profile sketch uses ``p1_`` so the names stay unique)."""
    import Sketcher

    sheet = ctx.sheet
    bound = 0

    def named(constraint, name, expr=None):
        nonlocal bound
        name = prefix + name
        i = sketch.addConstraint(constraint)
        sketch.renameConstraint(i, name)
        if expr:
            sketch.setExpression(f".Constraints.{name}", expr)
            bound += 1
        return i

    def when_ref(expr, *values):
        """``expr`` only when at least one of ``values`` is a parameter name
        (an all-literal dimension stays a plain number)."""
        return expr if any(isinstance(v, str) for v in values) else None

    shape = geom["shape"]
    if shape == "rectangle":
        lines = geom["lines"]
        w_ref, h_ref = profile["width"], profile["height"]
        c_ref = profile.get("center", [0, 0])
        for i in range(4):
            named(Sketcher.Constraint("Coincident", lines[i], 2,
                                      lines[(i + 1) % 4], 1), f"join_{i}")
        named(Sketcher.Constraint("Horizontal", lines[0]), "side_bottom")
        named(Sketcher.Constraint("Horizontal", lines[2]), "side_top")
        named(Sketcher.Constraint("Vertical", lines[1]), "side_right")
        named(Sketcher.Constraint("Vertical", lines[3]), "side_left")
        named(Sketcher.Constraint("DistanceX", lines[0], 1, lines[0], 2,
                                  geom["width"]), "width", ref_expr(sheet, w_ref))
        named(Sketcher.Constraint("DistanceY", lines[1], 1, lines[1], 2,
                                  geom["height"]), "height", ref_expr(sheet, h_ref))
        # place the lower-left corner: centre - size / 2 per axis
        x0, y0 = geom["corners"][0]
        named(Sketcher.Constraint("DistanceX", -1, 1, lines[0], 1, x0), "corner_x",
              when_ref(center_expr(sheet, c_ref[0], w_ref), c_ref[0], w_ref))
        named(Sketcher.Constraint("DistanceY", -1, 1, lines[0], 1, y0), "corner_y",
              when_ref(center_expr(sheet, c_ref[1], h_ref), c_ref[1], h_ref))

    elif shape == "circle":
        idx = geom["circle"]
        c_ref = profile.get("center", [0, 0])
        named(Sketcher.Constraint("Diameter", idx, geom["diameter"]),
              "diameter", ref_expr(sheet, profile["diameter"]))
        named(Sketcher.Constraint("DistanceX", -1, 1, idx, 3, geom["center"][0]),
              "cx", ref_expr(sheet, c_ref[0]))
        named(Sketcher.Constraint("DistanceY", -1, 1, idx, 3, geom["center"][1]),
              "cy", ref_expr(sheet, c_ref[1]))

    elif shape == "slot":
        line_a, cap_a, line_b, cap_b = geom["elements"]
        point = geom["center_point"]
        c_ref = profile.get("center", [0, 0])
        l_ref, w_ref = profile["length"], profile["width"]
        # end-to-end tangency also makes the ends coincident (a separate
        # Coincident would be a redundant constraint)
        named(Sketcher.Constraint("Tangent", line_a, 2, cap_a, 1), "tan_0")
        named(Sketcher.Constraint("Tangent", cap_a, 2, line_b, 1), "tan_1")
        named(Sketcher.Constraint("Tangent", line_b, 2, cap_b, 1), "tan_2")
        named(Sketcher.Constraint("Tangent", cap_b, 2, line_a, 1), "tan_3")
        named(Sketcher.Constraint("Equal", cap_a, cap_b), "equal_r")
        named(Sketcher.Constraint("Diameter", cap_a, geom["width"]),
              "width", ref_expr(sheet, w_ref))
        named(Sketcher.Constraint("Distance", cap_a, 3, cap_b, 3,
                                  geom["length"] - geom["width"]), "flat",
              when_ref(f"{term(sheet, l_ref)} - {term(sheet, w_ref)}",
                       l_ref, w_ref))
        named(Sketcher.Constraint("Symmetric", cap_a, 3, cap_b, 3, point, 1),
              "centred")
        named(Sketcher.Constraint("DistanceX", -1, 1, point, 1,
                                  geom["center"][0]), "cx",
              ref_expr(sheet, c_ref[0]))
        named(Sketcher.Constraint("DistanceY", -1, 1, point, 1,
                                  geom["center"][1]), "cy",
              ref_expr(sheet, c_ref[1]))
        a_ref = _angle_ref(profile)
        if isinstance(a_ref, str) or abs(geom["angle"]) > 1e-9:
            named(Sketcher.Constraint("Angle", line_a, geom["angle"]), "angle",
                  f"{sheet}.{a_ref} * 1 deg" if isinstance(a_ref, str) else None)
        else:
            named(Sketcher.Constraint("Horizontal", line_a), "side_bottom")

    elif shape == "profile":
        _constrain_profile(sketch, geom, profile, ctx, named)

    return bound


def _constrain_profile(sketch, geom: dict, profile: dict, ctx: Ctx, named) -> None:
    """Polyline / arc profile: coincident chain, horizontal / vertical
    detection, per-segment DistanceX/Y expressions, radius and start pin
    (``named`` adds each constraint and counts the bound expressions)."""
    import Sketcher

    sheet = ctx.sheet
    metas = geom["meta"]
    n = len(metas)

    def end_pt(m):      # sketch point id of the segment's `to` end
        return 1 if m.get("rev") else 2

    def start_pt(m):    # ... and of its `from` end
        return 2 if m.get("rev") else 1

    for i, m in enumerate(metas):
        nxt = metas[(i + 1) % n] if geom["closed"] else (
            metas[i + 1] if i + 1 < n else None)
        if nxt is not None:
            named(Sketcher.Constraint("Coincident", m["index"], end_pt(m),
                                      nxt["index"], start_pt(nxt)), f"join_{i}")
        dx = abs(m["to"][0] - m["from"][0])
        dy = abs(m["to"][1] - m["from"][1])
        # an arc has no direction, so it aligns its two ENDPOINTS (the
        # single-geometry form is malformed on an arc)
        ends = ((m["index"], start_pt(m), m["index"], end_pt(m))
                if "center" in m else (m["index"],))
        if dx < 1e-9 and dy > 1e-9:
            named(Sketcher.Constraint("Vertical", *ends), f"vert_{i}")
        elif dy < 1e-9 and dx > 1e-9:
            named(Sketcher.Constraint("Horizontal", *ends), f"horz_{i}")

    def coord_expr(v):
        if isinstance(v, str):
            return f"{sheet}.{v}"
        if isinstance(v, (int, float)):
            return repr(float(v))
        return None

    # In a closed loop the per-axis deltas sum to zero, so the last nonzero
    # delta per axis is redundant — leave those undimensioned (the
    # coincidence chain already pins them).
    skip_dims: dict[int, tuple] = {}
    if geom["closed"]:
        for axis in (0, 1):
            nonzero = [i for i, m in enumerate(metas)
                       if abs(m["to"][axis] - m["from"][axis]) > 1e-9]
            if len(nonzero) > 1:
                last = nonzero[-1]
                skip_dims[last] = skip_dims.get(last, ()) + (axis,)

    for i, m in enumerate(metas):
        if m.get("radius_ref"):
            r_val = math.hypot(m["from"][0] - m["center"][0],
                               m["from"][1] - m["center"][1])
            named(Sketcher.Constraint("Radius", m["index"], r_val), f"rad_{i}",
                  ref_expr(sheet, m["radius_ref"]))
        from_ref = metas[i - 1]["ref_to"] if i > 0 else profile.get("start")
        to_ref = m.get("ref_to")
        # each vertex is bound relative to the previous one; the constraint
        # value is a number, the expression recomputes it
        for axis in (0, 1):
            value = m["to"][axis] - m["from"][axis]
            kind = "DistanceX" if axis == 0 else "DistanceY"
            if axis in skip_dims.get(i, ()) or abs(value) <= 1e-9:
                continue
            tr, fr = coord_expr(to_ref[axis]), coord_expr(from_ref[axis])
            if tr and fr:
                expr = f"{tr} - ({fr})"
            elif tr:
                expr = tr
            elif fr:
                expr = f"-({fr})"
            else:
                expr = None
            named(Sketcher.Constraint(kind, m["index"], start_pt(m),
                                      m["index"], end_pt(m), value),
                  f"seg_{i}_{kind.lower()}", expr)
    # pin the start point
    sref = profile.get("start", [0, 0])
    sx, sy = geom["start"]
    named(Sketcher.Constraint("DistanceX", -1, 1, metas[0]["index"],
                              start_pt(metas[0]), sx),
          "start_x", ref_expr(sheet, sref[0]))
    named(Sketcher.Constraint("DistanceY", -1, 1, metas[0]["index"],
                              start_pt(metas[0]), sy),
          "start_y", ref_expr(sheet, sref[1]))


# ---------------------------------------------------------------------------
# the sketch object


def bind_placement(sketch, basis: dict, ctx: Ctx) -> None:
    """Expression-bind the sketch placement origin to the parameters its
    coordinates reference — so hole sketches ride the geometry when a
    thickness / dimension parameter changes."""
    for axis, ref in zip(("x", "y", "z"), basis.get("origin_refs", [])):
        if ref:
            sketch.setExpression(f"Placement.Base.{axis}", f"{ctx.sheet}.{ref}")


def fill_sketch(doc, sketch, profiles: list, plan: dict, fid: str,
                ctx: Ctx) -> dict:
    """Add the geometry and constraints of ``profiles`` (one, or several shapes
    in ONE sketch) to ``sketch``, recompute and return the solver report.

    The first profile uses the plain constraint names, the next ones ``p1_``,
    ``p2_`` ... Raises BuildError for a closed profile that produced no wire
    or constraints that conflict."""
    geoms, bound = [], 0
    for i, prof in enumerate(profiles):
        geom = add_geometry(sketch, prof, plan, fid)
        bound += constrain(sketch, geom, prof, plan, fid, ctx,
                           "" if i == 0 else f"p{i}_")
        geoms.append(geom)
    doc.recompute()
    kind = geoms[0]["shape"] if len(geoms) == 1 else "multi:" + "+".join(g["shape"] for g in geoms)
    info = sketch_report(sketch, kind, bound, fid)
    if len(geoms) > 1:
        info["profiles"] = len(geoms)
    # a closed profile must produce a wire before any feature consumes it
    if all(g["shape"] in ("rectangle", "circle", "slot") or g.get("closed")
           for g in geoms):
        shape = getattr(sketch, "Shape", None)
        if shape is not None and not shape.Wires:
            hint = ""
            if info.get("redundant"):
                hint = (f"; the solver found {info['redundant']} redundant "
                        f"constraint(s) — an arc that is exactly a semicircle "
                        f"over its chord is singular for a radius dimension: "
                        f"use a 'slot' profile or two quarter arcs")
            raise BuildError(fid, "sketch produced no wires — profile not "
                                  "closed" + hint)
        if shape is not None and len(shape.Wires) < len(geoms):
            raise BuildError(fid, f"sketch has {len(shape.Wires)} closed "
                                  f"wire(s) for {len(geoms)} profiles; two "
                                  f"profiles touch or cross - keep them apart "
                                  f"or nest one inside the other")
    return info


def build_sketch(doc, body, feat: dict, plan: dict, ctx: Ctx):
    """Create the Sketcher object inside the Body.

    Returns ``(sketch, info)``; ``info`` carries the geometry kind, the number
    of bound dimension expressions and the solver verdict (``solver`` code,
    ``fully_constrained``, ``constraint_count``, ``redundant`` when the
    solver found redundant constraints). A v2 sketch may carry several
    ``profiles`` (see ``fill_sketch``). Raises BuildError for a profile that
    produced no closed wire or constraints that conflict."""
    fid = feat["id"]
    basis = schema.sketch_basis(plan, feat.get("plane", {}))
    sketch = body.newObject("Sketcher::SketchObject", ctx.name(fid))
    sketch.Label = ctx.name(fid)
    tag_object(sketch, ctx.plan_id, fid)
    sketch.Placement = orthonormal_placement(basis)
    bind_placement(sketch, basis, ctx)
    info = fill_sketch(doc, sketch, schema.feature_profiles(feat), plan, fid, ctx)
    return sketch, info


def sketch_report(sketch, kind: str, bound: int, feature_id: str) -> dict:
    """The solver's verdict on a built sketch. Raises BuildError when the
    constraints conflict or are malformed (the shape would be invalid)."""
    info = {"geometry": kind, "dimensional_expressions": bound}
    try:
        info["solver"] = int(sketch.solve())
        info["fully_constrained"] = bool(sketch.FullyConstrained)
        info["constraint_count"] = len(sketch.Constraints)
    except Exception as e:  # noqa: BLE001 - a solver failure is reportable
        info["solver_error"] = str(e)
    conflicting = list(getattr(sketch, "ConflictingConstraints", []) or [])
    malformed = list(getattr(sketch, "MalformedConstraints", []) or [])
    if conflicting or malformed:
        names = []
        for idx in conflicting + malformed:
            try:
                names.append(sketch.Constraints[idx - 1].Name or f"#{idx}")
            except Exception:  # noqa: BLE001
                names.append(f"#{idx}")
        raise BuildError(feature_id,
                         f"sketch constraints {'conflict' if conflicting else 'are malformed'}"
                         f" ({', '.join(names)}); check that the dimensions "
                         f"describe one consistent closed profile")
    redundant = list(getattr(sketch, "RedundantConstraints", []) or [])
    if redundant:
        info["redundant"] = len(redundant)
    return info
