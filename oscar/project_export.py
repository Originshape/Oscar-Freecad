"""Model export of :mod:`oscar.project` (private): the code behind the ``export_model`` tool AND the
auto-export of ``Project.finalize``, so both produce the same files.

One file per part (or one combined file), in STEP / STL / 3MF / BREP, optionally re-oriented for
printing from the project's ``spec.json`` (the kit spec keys ``print_orientation.shell`` /
``print_orientation.lid``; no spec, or no such key, means "as modelled"). Shapes are exported in the
GLOBAL frame (``global_shape``), never the container-local one. All FreeCAD work runs in ONE
GUI-thread task that writes into a private temp directory next to the target; the files are then
moved into place atomically (``os.replace``) with collision-free names from the calling thread.
"""

import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable
from . import fsretry as _fsretry

from . import project_gui as gui

FORMATS = ("step", "stl", "3mf", "brep")
EXPORT_TIMEOUT_S = 120.0
AS_MODELED = "as_modeled"
# The orientations the kit spec accepts per part role (``kit.spec.SHELL_ORIENTATIONS`` / ``LID_ORIENTATIONS``);
# used only when the kit cannot be imported.
_DEFAULT_ORIENTATIONS = {"shell": ("floor_down", AS_MODELED), "lid": ("top_down", AS_MODELED)}


def known_orientations() -> dict[str, tuple[str, ...]]:
    """``{"shell": (...), "lid": (...)}``: the print orientations the kit spec accepts per part role, read from
    ``oscar.kit.spec`` through a guarded import (the defaults above when the kit is absent)."""
    try:
        from .kit import spec as kit_spec  # noqa: PLC0415 - lazy by design
    except ModuleNotFoundError as e:
        if e.name not in ("oscar.kit", "oscar.kit.spec"):
            raise
        return dict(_DEFAULT_ORIENTATIONS)
    shell = tuple(getattr(kit_spec, "SHELL_ORIENTATIONS", ()) or _DEFAULT_ORIENTATIONS["shell"])
    lid = tuple(getattr(kit_spec, "LID_ORIENTATIONS", ()) or _DEFAULT_ORIENTATIONS["lid"])
    return {"shell": shell, "lid": lid}


def plan_orientation(spec: dict | None, role: str, *, enabled: bool = True) -> tuple[str, str]:
    """``(mode, note)``: how a part of ``role`` (``"shell"`` / ``"lid"`` / ``""``) is oriented in the export.

    ``mode`` is ``floor_down`` / ``top_down`` / ``as_modeled``. Only the two kit roles are ever re-oriented, and
    only when ``enabled`` and the spec's ``print_orientation`` names a value the kit knows for that role. A
    value the kit does not know (a hand-edited spec.json) exports AS MODELLED and says so in ``note``;
    a missing spec or key is silently as modelled."""
    if not enabled or role not in ("shell", "lid"):
        return AS_MODELED, ""
    config = (spec or {}).get("print_orientation") if isinstance(spec, dict) else None
    if not isinstance(config, dict) or config.get(role) in (None, ""):
        return AS_MODELED, ""
    value = str(config[role])
    accepted = known_orientations()[role]
    if value not in accepted:
        return AS_MODELED, (f"print_orientation.{role} '{value}' is not one of {', '.join(accepted)}; "
                            f"the {role} was exported as modelled")
    return value, ""


@dataclass
class ExportedFile:
    """One written file. ``rel`` is relative to the project root when the file is inside it, else absolute."""

    path: str
    rel: str
    format: str
    bytes: int
    sha256: str
    label: str
    solids: int
    orientation: str = AS_MODELED   # floor_down | top_down | as_modeled (how the part sits in the file)


@dataclass
class ExportResult:
    """What ``export_tips`` produced: the document name, the files written, one text per file that failed and
    ``notes`` (things the user should know that are not failures, e.g. an unknown print orientation)."""

    doc: str = ""
    files: list[ExportedFile] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def safe_stem(text: str, fallback: str = "export") -> str:
    """A file-name stem: ``[A-Za-z0-9._-]`` only, never empty, never starting with a dot."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text or "")).strip("-.")
    return stem[:80] or fallback


def unique_path(directory: str, stem: str, ext: str, taken: Any = ()) -> str:
    """``<directory>/<stem>.<ext>``, or ``<stem>-2.<ext>``, ``-3`` ... on a clash with an existing file or a
    path in ``taken``. ``ext`` has no leading dot."""
    ext = ext.lstrip(".").lower()
    candidate = os.path.join(directory, f"{stem}.{ext}")
    n = 1
    while os.path.exists(candidate) or candidate in taken:
        n += 1
        candidate = os.path.join(directory, f"{stem}-{n}.{ext}")
    return candidate


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role_of(obj) -> str:
    role = str(getattr(obj, "OscarRole", "") or "").lower()
    if role in ("shell", "lid"):
        return role
    label = str(getattr(obj, "Label", "") or "").lower()
    return label if label in ("shell", "lid") else ""


def _orient(shape, mode: str):
    """The shape re-oriented for printing: ``top_down`` flips it 180 degrees about X through its centre (a lid:
    plate on the bed, skirt up), ``floor_down`` keeps the modelled pose (a shell: floor on the bed); either way
    the part is then dropped onto z = 0. ``as_modeled`` leaves the shape untouched."""
    import FreeCAD as App

    if mode not in ("floor_down", "top_down"):
        return shape
    out = shape.copy()
    if mode == "top_down":
        centre = out.BoundBox.Center
        out.rotate(App.Vector(centre.x, centre.y, centre.z), App.Vector(1, 0, 0), 180)
    out.translate(App.Vector(0, 0, -out.BoundBox.ZMin))
    return out


def _write_one(shape, fmt: str, path: str, tolerance: float) -> None:
    """Write ``shape`` as ``fmt`` to ``path`` (the extension of ``path`` is the format). Raises on failure."""
    if fmt == "step":
        shape.exportStep(path)
    elif fmt == "brep":
        shape.exportBrep(path)
    else:
        try:
            import MeshPart

            mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=tolerance, AngularDeflection=0.5236,
                                          Relative=False)
            mesh.write(path)
        except ImportError:
            if fmt != "stl":
                raise
            shape.exportStl(path)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"the {fmt} writer produced no file")


def export_tips(project, doc_name: str, *, objects: list[str] | None = None,
                formats: tuple[str, ...] | list[str] = ("step", "stl"), per_part: bool = True,
                print_orientation: bool = True, stl_tolerance: float = 0.05, directory: str | None = None,
                should_stop: Callable[[], bool] | None = None) -> ExportResult:
    """Export ``doc_name``'s model to files (any thread; the FreeCAD work is one GUI-thread task).

    ``objects`` names the objects to export (each must hold a solid); the default is every visible solid TIP
    (``project_gui.visible_tips``: boolean results and Bodies, not their inputs, never the hidden ``Ref_*``
    reference). ``per_part=False`` writes ONE file with all of them. The files go to ``directory`` when
    given, else to the project's ``exports/`` folder (``project`` is required then); names are
    ``<spec name or document>-<part label>.<format>`` with a ``-2``, ``-3`` suffix on a clash, written
    atomically. ``print_orientation`` applies the spec's ``print_orientation.shell|lid`` (``plan_orientation``);
    it is NOT applied to a combined file (``per_part=False``): the parts would land on top of each other, so the
    combined file keeps the assembled pose and ``ExportResult.notes`` says so. After a successful export into the
    project's own ``exports/`` folder ``project.mark_exported()`` records which model state the files show.

    Failures of single files land in ``ExportResult.errors`` (the other files are still written); a problem
    with the arguments or the document raises ``ProjectError``.
    """
    from .project import ProjectError

    wanted = [str(f).lower().lstrip(".") for f in (formats or ())] or ["step", "stl"]
    unknown = [f for f in wanted if f not in FORMATS]
    if unknown:
        raise ProjectError(f"export: unknown format(s) {', '.join(repr(f) for f in unknown)}; "
                           f"accepted: {', '.join(FORMATS)}.")
    wanted = list(dict.fromkeys(wanted))
    try:
        tolerance = float(stl_tolerance)
    except (TypeError, ValueError):
        raise ProjectError("export: 'stl_tolerance' must be a number (mm); use 0.05.") from None
    if not 0.001 <= tolerance <= 5.0:
        raise ProjectError(f"export: 'stl_tolerance' {tolerance:g} is outside 0.001..5 mm; use 0.05.")
    if not directory and project is None:
        raise ProjectError("export: no project is bound to this chat; pass 'directory' (an absolute folder).")
    if directory:
        if not os.path.isabs(directory):
            raise ProjectError(f"export: 'directory' must be an absolute path, got '{directory}'.")
        target = directory
    else:
        project.ensure_dir()
        target = project.path("exports")
    os.makedirs(target, exist_ok=True)
    spec = project.read_spec() if project is not None else None
    use_orientation = bool(print_orientation) and bool(per_part)
    notes: list[str] = []
    if print_orientation and not per_part and isinstance((spec or {}).get("print_orientation"), dict):
        notes.append("print orientation is not applied to a combined file (per_part=false): the parts stay in "
                     "the assembled pose")
    base = safe_stem((spec or {}).get("name") or doc_name, "model").lower()
    work = tempfile.mkdtemp(prefix=".oscar-export-", dir=target)

    def task() -> dict:
        doc = gui.resolve_doc(doc_name)
        if doc is None:
            return {"error": f"document '{doc_name}' is not open"}
        if objects:
            chosen, missing = [], []
            for name in objects:
                obj = doc.getObject(name)
                if obj is None:
                    missing.append(name)
                elif gui._solid_count(obj) == 0:
                    return {"error": f"object '{name}' has no solid to export"}
                else:
                    chosen.append(obj)
            if missing:
                have = ", ".join(o.Name for o in gui.visible_tips(doc)) or "none"
                return {"error": f"no such object(s) {', '.join(repr(m) for m in missing)} in '{doc.Name}'; "
                                 f"exportable parts: {have}"}
        else:
            chosen = gui.visible_tips(doc)
        if not chosen:
            return {"error": f"'{doc.Name}' has no visible solid part to export; build the model first"}
        import Part

        parts = []
        task_notes: list[str] = []
        for obj in chosen:
            mode, note = plan_orientation(spec, _role_of(obj), enabled=use_orientation)
            if note and note not in task_notes:
                task_notes.append(note)
            parts.append((str(getattr(obj, "Label", obj.Name) or obj.Name), _orient(gui.global_shape(obj), mode),
                          gui._solid_count(obj), mode))
        groups = parts if per_part else [("model", Part.makeCompound([p[1] for p in parts]),
                                          sum(p[2] for p in parts), AS_MODELED)]
        produced, errors = [], []
        for index, (label, shape, solids, mode) in enumerate(groups):
            for fmt in wanted:
                tmp = os.path.join(work, f"{index:02d}-{safe_stem(label, 'part')}.{fmt}")
                try:
                    _write_one(shape, fmt, tmp, tolerance)
                    produced.append({"label": label, "format": fmt, "tmp": tmp, "solids": solids, "orientation": mode})
                except Exception as e:  # noqa: BLE001 - one bad format must not lose the others
                    errors.append(f"{label}.{fmt}: {type(e).__name__}: {e}")
        return {"doc": doc.Name, "produced": produced, "errors": errors, "notes": task_notes}

    result = ExportResult()
    try:
        data, error = gui.gui_call(task, EXPORT_TIMEOUT_S, "project_export", should_stop)
        if error:
            raise ProjectError(f"export: {error}")
        if data.get("error"):
            raise ProjectError(f"export: {data['error']}.")
        result.doc = data.get("doc", doc_name)
        result.errors = list(data.get("errors") or [])
        result.notes = [*notes, *(data.get("notes") or [])]
        taken: set[str] = set()
        for item in data["produced"]:
            stem = f"{base}-{safe_stem(item['label'], 'part').lower()}" if per_part else base
            if project is not None and not directory:
                final = project.export_path(stem, item["format"])
            else:
                final = unique_path(target, stem, item["format"], taken)
            taken.add(final)
            try:
                _fsretry.replace(item["tmp"], final)
                size = os.path.getsize(final)
                digest = _sha256(final)
            except OSError as e:
                result.errors.append(f"{os.path.basename(final)}: {e}")
                continue
            rel = project.rel(final) if project is not None and _inside(project.root, final) else final
            result.files.append(ExportedFile(final, rel, item["format"], size, digest, item["label"], item["solids"],
                                             item.get("orientation", AS_MODELED)))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if result.files and project is not None and not directory:
        mark = getattr(project, "mark_exported", None)
        if mark is not None:
            mark()
    return result


def _inside(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(root), os.path.abspath(path)]) == os.path.abspath(root)
    except ValueError:
        return False
