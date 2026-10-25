"""Robust edge and face selection for the v2 builder (private to oscar.plan).

A plan never names an edge or face by index (``Edge7`` changes when a
parameter does); it names a RULE and this module turns the rule into the
sub-element names of the shape at build time.

Edge rules (``edges`` of ``fillet`` / ``chamfer``)::

    all_outer              every CONVEX edge that is not part of a hole or pocket
                           wall (the silhouette of the part)
    vertical               every straight edge parallel to the local Z axis
                           that is not part of a hole wall (concave inside
                           corners included)
    top_outer              the all_outer edges lying in the top plane (Z max)
    bottom_outer           the all_outer edges lying in the bottom plane (Z min)
    edges_of_feature:<id>  every edge of the shape that lies on the geometry
                           the feature <id> added or removed (a hole's rim, a
                           pad's outline, a sketch's profile)

Seam edges (one face on both sides) and tangent edges (faces meeting
smoothly, such as the join of a slot's flat and its cap) are never selected:
they cannot be rounded. Face rules (``top``, ``bottom``, ``front``, ``back``,
``left``, ``right``) pick the planar faces whose outward normal is +Z, -Z, -Y,
+Y, -X, +X at the extreme of the shape along that axis. All of it runs in the
shape's own (Body-local) frame.
"""

import math

from ._buildcore import BuildError

_TANGENT_COS = math.cos(math.radians(1.0))
_FACE_NORMALS = {"top": (0, 0, 1), "bottom": (0, 0, -1), "front": (0, -1, 0),
                 "back": (0, 1, 0), "left": (-1, 0, 0), "right": (1, 0, 0)}
_AXIS_OF = {"top": (2, True), "bottom": (2, False), "front": (1, False),
            "back": (1, True), "left": (0, False), "right": (0, True)}


def local_copy(shape):
    """``shape`` as an independent copy with an identity placement (the
    Body-local frame the rules reason in; element ORDER is unchanged)."""
    import FreeCAD as App

    out = shape.copy()
    out.Placement = App.Placement()
    return out


def face_normal(face, point=None):
    """Outward unit normal of ``face`` at ``point`` (default: its centre of
    mass projected onto the face)."""
    p = point if point is not None else face.CenterOfMass
    u, v = face.Surface.parameter(p)
    n = face.normalAt(u, v)
    n.normalize()
    return n


def _index_in(items, sub) -> int:
    for i, it in enumerate(items):
        if it.isSame(sub):
            return i
    return -1


def _mid(edge):
    t = 0.5 * (edge.FirstParameter + edge.LastParameter)
    return t, edge.valueAt(t)


def _into_face(edge, p, t_dir, face):
    """Unit vector at ``p`` lying in the tangent plane of ``face``, perpendicular
    to the edge, pointing INTO the face."""
    n = face_normal(face, p)
    cand = t_dir.cross(n)
    if cand.Length < 1e-12:
        return None
    cand.normalize()
    import Part

    eps = 1e-2
    plus = Part.Vertex(p + cand * eps).distToShape(face)[0]
    minus = Part.Vertex(p - cand * eps).distToShape(face)[0]
    return cand if plus <= minus else cand * -1


def _is_convex(edge, f1, f2) -> bool | None:
    """True when material wraps less than 180 degrees round the edge; None
    when it cannot be decided."""
    try:
        t, p = _mid(edge)
        tangent = edge.tangentAt(t)
        n1 = face_normal(f1, p)
        e2 = _into_face(edge, p, tangent, f2)
        if e2 is None:
            return None
        return n1.dot(e2) < -1e-9
    except Exception:  # noqa: BLE001 - an undecidable edge is simply not selected
        return None


def _is_smooth(edge, f1, f2) -> bool:
    try:
        _t, p = _mid(edge)
        return face_normal(f1, p).dot(face_normal(f2, p)) > _TANGENT_COS
    except Exception:  # noqa: BLE001
        return False


def _concave_wall(face) -> bool:
    """A cylinder whose material lies OUTSIDE it - the wall of a hole."""
    try:
        surf = face.Surface
        if surf.__class__.__name__ != "Cylinder":
            return False
        u0, u1, v0, v1 = face.ParameterRange   # the centre of mass of a full cylinder lies ON its axis
        u, v = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        p = face.valueAt(u, v)
        axis = surf.Axis
        rad = p - surf.Center
        rad = rad - axis * rad.dot(axis)
        if rad.Length < 1e-9:
            return False
        return face.normalAt(u, v).dot(rad) < 0
    except Exception:  # noqa: BLE001
        return False


def _inner_edge_ids(shape) -> set[int]:
    """Indices of the edges that sit in an inner wire of some face (the rim
    of a hole or pocket)."""
    edges = shape.Edges
    inner: set[int] = set()
    for face in shape.Faces:
        outer = face.OuterWire
        for wire in face.Wires:
            if wire.isSame(outer):
                continue
            for e in wire.Edges:
                i = _index_in(edges, e)
                if i >= 0:
                    inner.add(i)
    return inner


def _edge_table(shape) -> list[dict]:
    """One record per edge that can be rounded: ``{"i", "edge", "convex",
    "inner", "vertical", "z0", "z1"}`` (seams, tangent edges and edges that
    do not border exactly two faces are left out)."""
    import FreeCAD as App
    import Part

    faces = shape.Faces
    inner_ids = _inner_edge_ids(shape)
    walls = {i for i, f in enumerate(faces) if _concave_wall(f)}
    table = []
    for i, e in enumerate(shape.Edges):
        anc = shape.ancestorsOfType(e, Part.Face)
        if len(anc) != 2:
            continue
        f1, f2 = anc
        if _is_smooth(e, f1, f2):
            continue
        fi = (_index_in(faces, f1), _index_in(faces, f2))
        inner = i in inner_ids or bool(walls & set(fi))
        straight = e.Curve.__class__.__name__ in ("Line", "LineSegment")
        vertical = False
        if straight:
            d = e.Vertexes[-1].Point - e.Vertexes[0].Point
            vertical = d.Length > 1e-9 and abs(d.normalize().dot(App.Vector(0, 0, 1))) > 1 - 1e-9
        bb = e.BoundBox
        table.append({"i": i, "edge": e, "convex": _is_convex(e, f1, f2),
                      "inner": inner, "vertical": vertical,
                      "z0": bb.ZMin, "z1": bb.ZMax})
    return table


def _near(shape, edge, tol: float) -> bool:
    import Part

    _t, p = _mid(edge)
    ends = [v.Point for v in edge.Vertexes] + [p]
    return all(Part.Vertex(q).distToShape(shape)[0] <= tol for q in ends)


def select_edges(rule: str, shape, feature_shape=None) -> list[str]:
    """``EdgeN`` names of ``shape`` matching the edge ``rule`` (see the module
    docstring). ``feature_shape`` is the added / removed geometry of the
    feature for ``edges_of_feature:<id>``. Raises BuildError (no feature id;
    the caller names it) when nothing matches."""
    table = _edge_table(shape)
    zmax, zmin = shape.BoundBox.ZMax, shape.BoundBox.ZMin
    tol = 1e-6 + 1e-9 * max(abs(zmax), abs(zmin), 1.0)

    def outer(row):
        return row["convex"] is True and not row["inner"]

    if rule == "all_outer":
        chosen = [r for r in table if outer(r)]
    elif rule == "vertical":
        chosen = [r for r in table if r["vertical"] and not r["inner"]]
    elif rule == "top_outer":
        chosen = [r for r in table if outer(r) and r["z0"] >= zmax - tol]
    elif rule == "bottom_outer":
        chosen = [r for r in table if outer(r) and r["z1"] <= zmin + tol]
    elif rule.startswith("edges_of_feature:") and feature_shape is not None:
        size = max(shape.BoundBox.DiagonalLength, 1.0)
        chosen = [r for r in table if _near(feature_shape, r["edge"], 1e-4 * size)]
    else:
        raise BuildError("", f"edge rule {rule!r} is not known")
    if not chosen:
        raise BuildError("", f"edge rule {rule!r} matches no edge that can be "
                             f"rounded on the part (it has {len(shape.Edges)} "
                             f"edges; seam and tangent edges are never "
                             f"selected, 'all_outer' needs convex edges). "
                             f"Try another rule, e.g. 'vertical' or "
                             f"'edges_of_feature:<id>'")
    return [f"Edge{r['i'] + 1}" for r in chosen]


def select_faces(rule: str, shape, many: bool = True) -> list[str]:
    """``FaceN`` names of the planar faces facing ``rule`` (``top`` ...) at
    the extreme of ``shape`` along that axis; ``many=False`` keeps only the
    largest one. Raises BuildError when there is none."""
    import FreeCAD as App

    want = App.Vector(*_FACE_NORMALS[rule])
    axis, positive = _AXIS_OF[rule]
    bb = shape.BoundBox
    extreme = (bb.ZMax, bb.YMax, bb.XMax)[2 - axis] if positive else (bb.ZMin, bb.YMin, bb.XMin)[2 - axis]
    tol = 1e-6 + 1e-9 * max(abs(extreme), 1.0)
    hits = []
    for i, f in enumerate(shape.Faces):
        if f.Surface.__class__.__name__ != "Plane":
            continue
        if (face_normal(f) - want).Length > 1e-6:
            continue
        fb = f.BoundBox
        lo, hi = (fb.XMin, fb.XMax) if axis == 0 else (fb.YMin, fb.YMax) if axis == 1 else (fb.ZMin, fb.ZMax)
        if abs((hi if positive else lo) - extreme) <= tol and hi - lo <= tol * 4:
            hits.append((f.Area, i))
    if not hits:
        raise BuildError("", f"no planar face faces {rule!r} at the "
                             f"{'top' if positive else 'bottom'} extreme of "
                             f"the part along the {'XYZ'[axis]} axis; use a "
                             f"face the part really has")
    hits.sort(reverse=True)
    pick = hits if many else hits[:1]
    return [f"Face{i + 1}" for _a, i in pick]
