"""What an imported file brought in (``CadEngine.import_file``): each object measured, and meshes made into solids.

A mesh (STL / OBJ / 3MF / PLY) imports as a ``Mesh::Feature``: triangles that can be looked at and measured but not
cut, filleted or fused. ``solid_from_mesh`` makes a Part solid of a closed mesh: one face per triangle, sewn into a
shell and closed into a solid, then the triangles of each flat region merged into one face. Each separate piece of the
mesh becomes its own solid, and the pieces are fused (exported models are often several overlapping bodies in one
file). Sewing gets slow fast on the GUI thread (measured: about 6 s at 7,000 triangles, a minute at 24,000, three and
a half at 51,000), so a mesh over ``MAX_SOLID_TRIANGLES`` stays a mesh and the merge is skipped over
``MAX_REFINE_TRIANGLES``.

``facts`` / ``lines`` say what each imported object is and where it sits, in the words the model reads.

FreeCAD module, GUI-thread only (document access); importable without FreeCAD (``oscar.tools`` reads its limits).
"""

from typing import Any

MAX_SOLID_TRIANGLES = 20_000
MAX_REFINE_TRIANGLES = 8_000
SEW_TOLERANCE = 0.05          # mm: mesh points closer than this become one vertex of the shell
MAX_LINES = 12                # objects listed one by one; the rest are counted
MESH_TYPE = "Mesh::Feature"
MESH_EXTENSIONS = frozenset({"stl", "obj", "3mf", "ply", "ast"})


def solid_from_mesh(mesh) -> tuple[Any, str]:
    """``(shape, "")`` of a closed ``Mesh.Mesh``, or ``(None, why)`` when it cannot be one (never raises).

    One solid per separate piece of the mesh (an exported model is often several closed bodies in one file, and one
    solid sewn from all of them is invalid), fused when there are several: a solid, or a compound of solids when the
    pieces do not touch."""
    count = int(mesh.CountFacets)
    if count == 0:
        return None, "the mesh has no triangles"
    if count > MAX_SOLID_TRIANGLES:
        return None, (f"it has {count:,} triangles; Oscar makes solids of meshes up to {MAX_SOLID_TRIANGLES:,} "
                      "(more would hold FreeCAD for minutes)")
    if not mesh.isSolid():
        return None, "the mesh is not closed (it has holes or open edges), so it does not enclose a volume"
    import Part

    refine = count <= MAX_REFINE_TRIANGLES
    try:
        pieces = mesh.getSeparateComponents() or [mesh]
        solids = [_piece_solid(Part, piece, refine) for piece in pieces]
        if len(solids) == 1:
            return solids[0], ""
        fused = solids[0].fuse(solids[1:])
        if refine:
            fused = _refined(fused)
    except Exception as e:  # noqa: BLE001 - OCC raises its own types
        return None, f"sewing its triangles into a solid failed ({type(e).__name__}: {e})"
    return (fused.Solids[0] if len(fused.Solids) == 1 else Part.makeCompound(fused.Solids)), ""


def _piece_solid(Part, mesh, refine: bool):
    """The solid of one closed piece of a mesh: one face per triangle sewn shut, flat regions merged (``refine``),
    small defects repaired."""
    shape = Part.Shape()
    try:
        shape.makeShapeFromMesh(mesh.Topology, SEW_TOLERANCE, False)
    except TypeError:   # FreeCAD before 0.20 takes no sew flag
        shape.makeShapeFromMesh(mesh.Topology, SEW_TOLERANCE)
    solid = Part.Solid(Part.Shell(shape.Faces))
    if solid.Volume < 0:   # triangles wound inside out
        solid.reverse()
    if refine:
        solid = _refined(solid)
        if solid.ShapeType != "Solid" and len(solid.Solids) == 1:
            solid = solid.Solids[0]
    if not solid.isValid():
        # degenerate or self-touching triangles (common in exported meshes) leave small defects that ShapeFix
        # clears; it hands the solid back inside a compound
        fixed = solid.copy()
        try:
            fixed.fix(1e-2, 1e-3, 1e-1)
            if len(fixed.Solids) == 1 and fixed.Solids[0].isValid():
                solid = fixed.Solids[0]
        except Exception:  # noqa: BLE001 - the unrepaired solid is still a solid
            pass
    return solid


def _refined(shape):
    """``shape`` with the faces of each flat region merged into one; ``shape`` itself when that fails or changes
    its volume."""
    try:
        refined = shape.removeSplitter()
        if refined.Solids and abs(refined.Volume - shape.Volume) <= 1e-6 * max(1.0, abs(shape.Volume)):
            return refined
    except Exception:  # noqa: BLE001
        pass
    return shape


def facts(obj) -> "dict[str, Any] | None":
    """What ``obj`` is and where: ``name``, ``label``, ``kind`` ("mesh" | "solid" | "surface" | "drawing"), ``min`` /
    ``max`` (global, mm), and ``triangles`` + ``closed`` (meshes), ``volume`` (closed meshes, solids), ``faces`` and
    ``solids`` (shapes). None for an object with no geometry (a group, a layer, an origin)."""
    from .serialize import ORIGIN_TYPES, global_shape

    base = {"name": obj.Name, "label": str(getattr(obj, "Label", "") or obj.Name)}
    if getattr(obj, "TypeId", "") == MESH_TYPE:
        mesh = obj.Mesh
        count = int(mesh.CountFacets)
        if count == 0:
            return None
        closed = bool(mesh.isSolid())
        out = {**base, "kind": "mesh", **_box(mesh.BoundBox), "triangles": count, "closed": closed}
        if closed:
            out["volume"] = abs(float(mesh.Volume))
        return out
    if getattr(obj, "TypeId", "") in ORIGIN_TYPES or getattr(obj, "Shape", None) is None:
        return None
    try:
        shape = global_shape(obj)
        if shape.isNull():
            return None
        out = {**base, **_box(shape.BoundBox), "faces": len(shape.Faces), "solids": len(shape.Solids)}
        if shape.Solids:
            out.update(kind="solid", volume=abs(float(shape.Volume)))
        else:
            flat = out["max"][2] - out["min"][2] <= 1e-6
            out["kind"] = "surface" if shape.Faces and not flat else "drawing"
        return out
    except Exception:  # noqa: BLE001 - a broken imported object is listed by name only
        return None


def _box(box) -> dict[str, list[float]]:
    return {"min": [box.XMin, box.YMin, box.ZMin], "max": [box.XMax, box.YMax, box.ZMax]}


def _fmt(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _where(low: list[float], high: list[float], axes: int = 3) -> str:
    size = " × ".join(_fmt(high[i] - low[i]) for i in range(axes))
    span = ", ".join(f"{axis} {_fmt(low[i])}..{_fmt(high[i])}" for i, axis in enumerate("xyz"[:axes]))
    return f"size {size} mm ({span})"


def line(part: dict[str, Any]) -> str:
    """One line for the model: ``- X_tile: mesh of 3052 triangles, closed; size 37.6 × 39.19 × 8.9 mm (x …); volume …``."""
    kind = part["kind"]
    if kind == "mesh":
        what = f"mesh of {part['triangles']} triangles, " + ("closed" if part["closed"] else "NOT closed (it has holes)")
    elif kind == "solid":
        made = f" made from a mesh of {part['from_triangles']} triangles" if part.get("from_triangles") else ""
        solids = f", {part['solids']} solids" if part["solids"] != 1 else ""
        valid = "" if part.get("valid") is None else ("; valid solid" if part["valid"] else "; NOT a valid solid")
        what = f"solid{made}, {part['faces']} faces{solids}{valid}"
    elif kind == "surface":
        what = f"surfaces ({part['faces']} faces), no closed solid"
    else:
        what = "2D drawing"
    flat = kind == "drawing" and part["max"][2] - part["min"][2] <= 1e-6
    volume = f"; volume {_fmt(part['volume'])} mm³" if part.get("volume") is not None else ""
    return f"- {part['name']}: {what}; {_where(part['min'], part['max'], 2 if flat else 3)}{volume}"


def lines(parts: list[dict[str, Any]]) -> list[str]:
    """The lines of every imported object: at most ``MAX_LINES``, then a count of the rest and their overall extent."""
    out = [line(part) for part in parts[:MAX_LINES]]
    if len(parts) > MAX_LINES:
        low = [min(p["min"][i] for p in parts) for i in range(3)]
        high = [max(p["max"][i] for p in parts) for i in range(3)]
        out.append(f"- … and {len(parts) - MAX_LINES} more; all {len(parts)} together: {_where(low, high)}")
    return out
