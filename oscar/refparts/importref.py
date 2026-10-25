"""Bringing a reference file into a document as ONE hidden ``Ref_<ID>`` ``App::Part``.

FreeCAD module, GUI-thread only (``import_reference_part`` calls it inside a ``cad.dispatch`` task).

``Import.insert`` makes its own container (``Open_CASCADE_STEP_translator_7_8_1`` with ``Part__Feature*`` children and a fresh
Origin) and object Names are immutable, so this module creates the ``Ref_<ID>`` ``App::Part`` ITSELF, moves the imported
solids into it (keeping their global placement), deletes the importer container together with its Origin objects, hides the
reference and its own helper Origin objects, tags everything (``OscarRole = "ref"``; NO ``OscarKit`` so no kit stage ever
deletes it), and finally normalizes the frame: the part's ``Placement`` is set so that the PCB bottom lower-left corner is the
origin and the long edge is +X (``measure.fit_frame``). Without that every interference / clearance / containment check
against a vendor STEP (arbitrary origin, sometimes Y-up) would be meaningless.

The whole import is ONE undo transaction: any failure leaves the document exactly as it was (``UndoMode`` is set to 1 around
the transaction because ``freecadcmd`` documents have ``UndoMode 0`` and ``abortTransaction`` would then undo nothing).
STL meshes are imported the same way (a ``Mesh::Feature`` inside the part) but have no PCB planes to normalize with: they
keep their imported frame and the result says so.
"""

import os
import re

from . import fetch, identity, measure

TAG_PROPS = ("OscarKit", "OscarStage", "OscarRole", "OscarPart", "OscarSpecHash")
_ORIGIN_TYPES = frozenset({"App::Origin", "App::Line", "App::Plane", "App::Point"})
_CONTAINERS = ("App::Part", "App::DocumentObjectGroup")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class ReferenceImportError(ValueError):
    """The import was refused or failed; the message is model-facing (``<what failed>: <why>; <what to do>``) and the document is unchanged."""


def _set_string(obj, prop: str, value: str, doc_text: str = "") -> None:
    if prop not in obj.PropertiesList:
        obj.addProperty("App::PropertyString", prop, "Oscar", doc_text or f"reference tag: {prop}")
    setattr(obj, prop, value)


def tag(obj, *, stage: str = "reference", extra: "dict | None" = None) -> None:
    """Give ``obj`` the kit tag properties of a reference (``OscarRole`` ``ref``, ``OscarPart`` ``Ref``, empty ``OscarKit``) plus ``extra``."""
    for prop, value in (("OscarKit", ""), ("OscarStage", stage), ("OscarRole", "ref"), ("OscarPart", "Ref"), ("OscarSpecHash", "")):
        _set_string(obj, prop, value)
    for prop, value in (extra or {}).items():
        _set_string(obj, prop, str(value))


def _hide(obj) -> None:
    try:
        obj.Visibility = False
    except Exception:  # noqa: BLE001 - a headless object may have no visibility; harmless
        pass


def _is_leaf(obj, kind: str) -> bool:
    tid = getattr(obj, "TypeId", "")
    if tid in _ORIGIN_TYPES or tid in _CONTAINERS or hasattr(obj, "Group") and tid != "PartDesign::Body":
        return False
    if kind == "mesh":
        return tid == "Mesh::Feature" and int(obj.Mesh.CountFacets) > 0
    shape = getattr(obj, "Shape", None)
    return shape is not None and not shape.isNull() and len(shape.Faces) > 0


def _mesh_bbox(leaves) -> dict:
    boxes = []
    for obj in leaves:
        mesh = obj.Mesh.copy()
        mesh.transform(obj.getGlobalPlacement().toMatrix())
        b = mesh.BoundBox
        boxes.append((b.XMin, b.YMin, b.ZMin, b.XMax, b.YMax, b.ZMax))
    return measure._bbox_dict(boxes)


def _move(doc, leaves, part) -> list:
    """Move ``leaves`` into ``part`` keeping their GLOBAL placement; returns the importer containers they came out of."""
    emptied = []
    placements = {leaf.Name: leaf.getGlobalPlacement() for leaf in leaves}
    for leaf in leaves:
        parent = leaf.getParentGeoFeatureGroup() or leaf.getParentGroup()
        if parent is not None and parent.Name != part.Name:
            parent.removeObject(leaf)
            emptied.append(parent)
        part.addObject(leaf)
        leaf.Placement = placements[leaf.Name]
    return emptied


def _purge(doc, created_names: list, keep: set) -> None:
    """Delete what the importer created besides the leaves: its containers (their Origin objects go with them) and stray Origin features.

    Works by NAME: removing a container deletes its Origin objects, and a Python wrapper of a deleted object raises on any access.
    """
    for wanted in (_CONTAINERS, tuple(_ORIGIN_TYPES)):
        for name in created_names:
            obj = doc.getObject(name)
            if obj is not None and name not in keep and getattr(obj, "TypeId", "") in wanted:
                doc.removeObject(name)


def import_reference(doc, path: str, *, ref_id: str, name: str = "", board: "dict | None" = None, expected_bbox=None,
                     sha256: str = "", normalize: bool = True, strict_identity: bool = False, trust_label: str = "") -> dict:
    """Import ``path`` into ``doc`` as the hidden ``App::Part`` ``name`` (default ``Ref_<ID>``) and return a summary dict.

    ``board`` (a catalog board dict) and ``expected_bbox`` feed the frame orientation and the identity check. Raises
    ``ReferenceImportError`` (document unchanged) for a name that is taken or invalid, a file the importer cannot read and a
    file that holds no solids. The summary: ``{"object", "kind", "leaves", "solids", "bbox", "largest", "frame", "identity",
    "warnings", "facets", "pin_check"}`` with every length in mm in the (normalized) global frame. GUI-thread only.

    ``strict_identity`` is the hash policy of a catalog file whose sha256 differs from the pinned one (``identity.pin_verdict``): the
    bounding box (against ``expected_bbox``) and the PCB footprint (against ``board``) must still match, else ``ReferenceImportError``
    (document unchanged); a pass adds a warning and ``summary["pin_check"] = {"verdict": "accepted_with_warning", "text": ...}``.
    ``trust_label`` (``catalog.trust_label``) is stored on the part as ``OscarRefTrust`` so later tools state the trust level too.
    """
    ext = os.path.splitext(path)[1].lower()
    kind = "mesh" if ext in fetch.MESH_EXTENSIONS else "brep"
    obj_name = (name or fetch.ref_object_name(ref_id)).strip()
    if not _NAME_RE.match(obj_name):
        raise ReferenceImportError(f"import_reference_part: invalid object name {obj_name!r}; use letters, digits and underscores, "
                                   "starting with a letter (for example 'Ref_PI4')")
    if doc.getObject(obj_name) is not None:
        raise ReferenceImportError(f"import_reference_part: '{obj_name}' already exists in '{doc.Name}'; pass name='Ref_<other>' to import "
                                   "a second reference, or delete the existing object first")
    previous_mode = doc.UndoMode
    doc.UndoMode = 1
    doc.openTransaction("Oscar: import reference")
    committed = False
    try:
        summary = _import(doc, path, kind, obj_name, ref_id, board, expected_bbox, sha256, normalize, strict_identity, trust_label)
        committed = True
        return summary
    finally:
        if committed:
            doc.commitTransaction()
        else:
            doc.abortTransaction()
        doc.UndoMode = previous_mode


def _strict_checks(summary: dict, kind: str, board, expected_bbox, m) -> list:
    """The identity checks of the hash policy: bounding box against ``expected_bbox`` and, for a B-rep with a PCB, footprint against the board."""
    checks = []
    if expected_bbox:
        checks.append(identity.bbox_identity((summary.get("bbox") or {}).get("size"), expected_bbox))
    if kind == "brep" and board and m is not None and m.pcb:
        checks.append(identity.outline_identity((m.pcb["length"], m.pcb["width"], m.pcb["thickness"]), board))
    return checks


def _apply_policy(summary: dict, kind: str, board, expected_bbox, m, base: str) -> None:
    verdict, text = identity.pin_verdict("mismatch", _strict_checks(summary, kind, board, expected_bbox, m))
    if verdict == "rejected":
        raise ReferenceImportError(f"import_reference_part: {base!r} was not imported: {text}; the catalog's verified copy is not this file. "
                                   "Ask the user to attach the right STEP file or continue with the nominal board data")
    summary["pin_check"] = {"verdict": verdict, "text": text}
    summary["warnings"] = list(summary.get("warnings") or []) + [text]


def _import(doc, path, kind, obj_name, ref_id, board, expected_bbox, sha256, normalize, strict=False, trust_label="") -> dict:
    before = {o.Name for o in doc.Objects}
    try:
        if kind == "brep":
            import Import

            Import.insert(path, doc.Name)
        else:
            import Mesh

            Mesh.insert(path, doc.Name)
        doc.recompute()
    except Exception as exc:  # noqa: BLE001 - the importers raise assorted exception types
        raise ReferenceImportError(f"import_reference_part: the importer could not read {os.path.basename(path)!r}: "
                                   f"{type(exc).__name__}: {exc}; check that the file is a valid STEP/STL file") from exc
    created = [o for o in doc.Objects if o.Name not in before]
    leaves = [o for o in created if _is_leaf(o, kind)]
    if not leaves:
        raise ReferenceImportError(f"import_reference_part: {os.path.basename(path)!r} produced no geometry; the file is empty or holds only "
                                   "curves; ask the user for a solid STEP or STL model")
    if kind == "brep" and not any(len(o.Shape.Solids) > 0 for o in leaves):
        raise ReferenceImportError(f"import_reference_part: {os.path.basename(path)!r} holds surfaces but no solid bodies; ask the user for a "
                                   "solid STEP model (export the part, not the sheet)")
    created_names = {o.Name for o in created}
    part = doc.addObject("App::Part", obj_name)
    keep = {o.Name for o in doc.Objects} - before - created_names           # the new part and its own Origin objects
    _move(doc, leaves, part)
    keep |= {leaf.Name for leaf in leaves}
    _purge(doc, sorted(created_names), keep)
    tag(part, extra={"OscarRefId": ref_id, "OscarRefFile": os.path.basename(path), "OscarRefSha256": sha256,
                     **({"OscarRefTrust": trust_label} if trust_label else {})})
    for leaf in leaves:
        tag(leaf)
    for obj in [part, part.Origin, *part.Origin.OriginFeatures, *leaves]:
        _hide(obj)
    warnings: list = []
    frame_info: dict = {"found": False, "applied": False, "description": ""}
    if kind == "brep" and normalize:
        frame = measure.fit_frame(measure.collect_solids(part), board)
        frame_info = {"found": frame.found, "applied": frame.found and not frame.is_identity, "axes": frame.axes,
                      "translation": [round(v, 4) for v in frame.translation], "description": frame.describe()}
        if frame.found and not frame.is_identity:
            part.Placement = frame.placement
        elif not frame.found:
            warnings.append(frame.note)
        doc.recompute()
    elif kind == "mesh":
        frame_info["description"] = "mesh reference: the frame is kept as imported (a mesh has no exact PCB planes to normalize with)"
        warnings.append("an STL mesh cannot be measured exactly or normalized into the board frame; a STEP file gives exact "
                        "dimensions (ask the user for the manufacturer's STEP file)")
    summary: dict = {"object": part.Name, "kind": kind, "leaves": [leaf.Name for leaf in leaves], "frame": frame_info,
                     "warnings": warnings, "identity": None, "largest": [], "facets": 0, "solids": 0, "bbox": {}}
    if kind == "mesh":
        summary["facets"] = sum(int(leaf.Mesh.CountFacets) for leaf in leaves)
        summary["bbox"] = _mesh_bbox(leaves)
        summary["solids"] = len(leaves)
        if strict:
            _apply_policy(summary, kind, board, expected_bbox, None, os.path.basename(path))
        return summary
    solids = measure.collect_solids(part)
    m = measure.analyze(solids, board=board, expected_bbox=expected_bbox, top_n=5, object_name=part.Name)
    summary.update(solids=m.solid_count, bbox=m.bbox, largest=m.solids, identity=m.identity)
    summary["warnings"] = warnings + [w for w in m.warnings if w not in warnings]
    if strict:
        _apply_policy(summary, kind, board, expected_bbox, m, os.path.basename(path))
    return summary
