"""Measuring a MESH reference (an STL): the part of ``measure_reference_part`` that has no exact B-rep to work with.

FreeCAD module, GUI-thread only (document access). A mesh is an approximation: the result is the exact bounding box,
volume and facet count of FreeCAD's own mesh; it never claims PCB length / width, mounting holes or connector
positions (the board outline is not separable from the connectors in one lump of triangles), so ``write_params``
writes nothing for a mesh and the report says why.
"""

from . import measure


def collect_meshes(root) -> list:
    """``[(label, mesh_copy_in_global_frame)]`` of the ``Mesh::Feature`` members of ``root`` (GUI-thread only)."""
    out = []
    for obj in measure._members(root):
        if getattr(obj, "TypeId", "") != "Mesh::Feature" or int(obj.Mesh.CountFacets) <= 0:
            continue
        mesh = obj.Mesh.copy()
        mesh.transform(obj.getGlobalPlacement().toMatrix())
        out.append((str(getattr(obj, "Label", "") or obj.Name), mesh))
    return out


def analyze_mesh(root, *, top_n: int = 8, object_name: str = "") -> "measure.Measurement":
    """Measure the meshes of ``root``: bounding box, volume and facets per mesh. Never raises on an empty reference."""
    m = measure.Measurement(kind="mesh", object=object_name or str(getattr(root, "Name", "")))
    meshes = collect_meshes(root)
    m.solid_count = len(meshes)
    if not meshes:
        m.warnings.append("the reference holds no mesh with triangles; nothing to measure")
        return m
    boxes = []
    for label, mesh in meshes:
        b = mesh.BoundBox
        box = (b.XMin, b.YMin, b.ZMin, b.XMax, b.YMax, b.ZMax)
        boxes.append(box)
        m.solids.append({"name": label, "size": [round(box[i + 3] - box[i], 3) for i in range(3)], "volume": round(float(mesh.Volume), 2),
                         "z": [round(box[2], 3), round(box[5], 3)], "facets": int(mesh.CountFacets)})
    m.solids = sorted(m.solids, key=lambda s: -s["volume"])[:max(1, top_n)]
    m.bbox = measure._bbox_dict(boxes)
    m.frame = {"found": False, "applied": False, "description": "a mesh keeps its imported frame (no exact PCB planes to normalize with)"}
    m.mesh = {"facets": sum(int(mesh.CountFacets) for _l, mesh in meshes)}
    return m


def render_text(m: "measure.Measurement") -> str:
    """The bounded model-facing report of a mesh measurement."""
    box = m.bbox
    lines = [f"Measured {m.object} (mesh reference, {m.mesh['facets'] if m.mesh else 0} triangles, units mm; a mesh is approximate)."]
    if box:
        lo, hi, size = box["min"], box["max"], box["size"]
        lines.append(f"Bounding box: x {measure._mm(lo[0])}..{measure._mm(hi[0])}, y {measure._mm(lo[1])}..{measure._mm(hi[1])}, "
                     f"z {measure._mm(lo[2])}..{measure._mm(hi[2])} (size {measure._mm(size[0])} x {measure._mm(size[1])} x {measure._mm(size[2])} mm).")
    lines.append("Board length / width, mounting holes and connector positions are not measured on a mesh; Params were not changed. "
                 "Ask the user for the manufacturer's STEP file for exact dimensions.")
    for warning in m.warnings:
        lines.append(f"Warning: {warning}.")
    return "\n".join(lines)
