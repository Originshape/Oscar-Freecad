"""Project tools: ``record_note``, ``save_document``, ``export_model``, ``export_rebuild_script``.

The model never writes into the Project directly: these tools call the Project API of the project bound to
the chat (``current_context().project``). Handlers run on the agent thread; every document / shape access
goes through ``oscar.project_gui.gui_call`` (``cad.dispatch.dispatch``) with NO lock held, and the context is
captured BEFORE dispatching (a ContextVar does not cross to the GUI thread).

Matrix flags: ``record_note`` plan / not mutating / all phases; ``save_document`` deliver / mutating /
BUILD..DELIVER; ``export_model`` and ``export_rebuild_script`` deliver / mutating / VERIFY, DELIVER. None is a
geometry tool, so no health, snapshot or checkpoint runs for them, but each call is a step record.
"""

import os
import tempfile
import time
import zipfile
from typing import Any

from . import project_gui as gui
from .project import NOTE_KINDS, Project, ProjectError
from .project_export import AS_MODELED, FORMATS, export_tips
from .project_replay import REBUILD_TOLERANCE, compare_rebuild, parse_rebuild_output
from .tools import ToolError, current_context, tool

_NOTE_LIMIT = 2000
_SAVE_TIMEOUT_S = 60.0
_RUN_CHECK_TIMEOUT_S = 300.0

_BUILD_TO_DELIVER = ("BUILD", "VERIFY", "DELIVER")
_VERIFY_TO_DELIVER = ("VERIFY", "DELIVER")


def _no_project(name: str) -> ToolError:
    return ToolError(f"{name}: no project is bound to this chat (open a chat in the Oscar panel)")


def _bound_project(name: str) -> Project:
    project = current_context().project
    if project is None:
        raise _no_project(name)
    return project


def _fmt_size(size: int) -> str:
    """``48.2 KB`` / ``1.2 MB`` / ``312 B``."""
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


# ---------------------------------------------------------------------------
# write-location guard (save_document 'path' / export_model 'directory')

_SYSTEM_PREFIXES = ("/System", "/usr", "/bin", "/sbin", "/etc", "/var", "/Library", "/Applications", "/opt", "/dev", "/proc",
                    "/sys", "/boot", "/private/etc", "/private/var", "/cores")
_TEMP_PREFIXES = ("/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp", "/var/folders", "/private/var/folders")


def _under(path: str, base: str) -> bool:
    base = base.rstrip(os.sep)
    return path == base or path.startswith(base + os.sep)


def _real(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def _write_location_problem(path: str, project: "Project | None", *, directory: bool, existing_fcstd_only: bool = False) -> str:
    """Why Oscar may not write to ``path`` ('' when it may). Symlinks are resolved first.

    Rules: never a system directory (``/System``, ``/usr``, ``/etc``, ``/var``, ``/Library``, ``/Applications``... ; temp
    folders are fine), never ``~/Library`` or a hidden folder of the home (``~/.ssh``, ``~/.config``), never the file system
    root; the project folder is always fine. An EXISTING file is only replaced when it is an .FCStd document (a zip) that lies in
    the project, the home or a temp folder: anything else (a text file, a script, a file outside the user's home) is refused.
    """
    real = _real(path)
    if real == os.path.dirname(real):
        return "the file system root"
    if project is not None and _under(real, _real(project.root)):
        return ""
    temps = {_real(t) for t in _TEMP_PREFIXES} | {_real(tempfile.gettempdir())}
    in_temp = any(_under(real, t) for t in temps)
    home = _real("~")
    if not in_temp:
        if any(_under(real, _real(prefix)) for prefix in _SYSTEM_PREFIXES):
            return "a system directory"
        if _under(real, home):
            first = os.path.relpath(real, home).split(os.sep)[0]
            if first.startswith(".") or first == "Library":
                return f"a protected folder of your home ({first})"
    if os.path.isdir(real) != directory and os.path.lexists(real):
        return "a folder where a file is expected" if not directory else "an existing file where a folder is expected"
    if not directory and os.path.isfile(real):
        if existing_fcstd_only and not zipfile.is_zipfile(real):
            return "an existing file that is not an .FCStd document"
        if not in_temp and not _under(real, home):
            return "an existing file outside your home folder"
    return ""


# ---------------------------------------------------------------------------
# record_note

@tool(
    "record_note",
    "Write a note into the project's NOTES.md: an assumption you made, a decision, a provenance fact, a "
    "validation result, or something unresolved. Use it for anything the user must know that is not in the "
    "geometry.",
    {"type": "object", "properties": {
        "kind": {"type": "string", "enum": list(NOTE_KINDS)},
        "text": {"type": "string", "description": "At most 2000 characters."},
        "refs": {"type": "array", "items": {"type": "string"},
                 "description": "Related step or requirement ids, e.g. ['S5','R2']"}},
     "required": ["kind", "text"]},
    category="plan",
)
def _record_note(kind: str, text: str, refs: list | None = None):
    """Add one note to the bound project (``Project.note``); the text is clipped to 2000 characters."""
    project = _bound_project("record_note")
    body = str(text or "").strip()
    problems = []
    if kind not in NOTE_KINDS:
        problems.append(f"'kind' must be one of {', '.join(NOTE_KINDS)}, got {kind!r}")
    if not body:
        problems.append("'text' is empty; say what the user must know")
    if problems:
        raise ToolError(f"record_note: {'; '.join(problems)}.")
    clean_refs = [str(r).strip() for r in (refs or []) if str(r).strip()]
    try:
        note_id = project.note(kind, body, refs=clean_refs)
    except ProjectError as e:
        raise ToolError(str(e)) from e
    except OSError as e:
        raise ToolError(f"record_note: the note could not be written: {e}; try again") from e
    clipped = f" The text was clipped to {_NOTE_LIMIT} characters." if len(body) > _NOTE_LIMIT else ""
    return f"Note {note_id} recorded ({kind}) in NOTES.md.{clipped}"


# ---------------------------------------------------------------------------
# save_document

@tool(
    "save_document",
    "Save a document to disk as an .FCStd file. Without path it saves to the project folder (model.FCStd) so "
    "the work can be reopened. The app also autosaves after every successful geometry step; use this for an "
    "explicit save-as. 'path' must not be in a system or hidden/protected folder, and an existing file there is "
    "only replaced when it already is an .FCStd document inside your home or the project.",
    {"type": "object", "properties": {
        "doc_name": {"type": "string"},
        "path": {"type": "string", "description": "Absolute target path ending in .FCStd (optional)."}},
     "required": ["doc_name"]},
    category="deliver", mutating=True, phases=_BUILD_TO_DELIVER,
)
def _save_document(doc_name: str, path: str = ""):
    """Save ``doc_name`` as an FCStd: into the project folder, or to ``path`` (atomic copy; the open document
    keeps its own file association)."""
    ctx = current_context()
    target = str(path or "").strip()
    if target:
        problems = []
        if not os.path.isabs(target):
            problems.append(f"'path' must be absolute, got '{target}'")
        if not target.endswith(".FCStd"):
            problems.append("'path' must end in .FCStd (exact case)")
        if not problems:
            why = _write_location_problem(target, ctx.project, directory=False, existing_fcstd_only=True)
            if why:
                problems.append(f"'path' is {why}; nothing was written. Save inside the project folder (omit 'path') or into your "
                                "Documents folder")
        if problems:
            raise ToolError(f"save_document: {'; '.join(problems)}; for example "
                            f"'{os.path.join(os.path.expanduser('~'), 'Documents', doc_name + '.FCStd')}'.")

        def task() -> dict:
            doc = gui.resolve_doc(doc_name)
            if doc is None:
                import FreeCAD as App

                return {"error": f"document '{doc_name}' is not open; open documents: "
                                 f"{', '.join(App.listDocuments()) or 'none'}"}
            saved, error = gui.save_copy(doc, target)
            return {"ok": saved, "error": error, "doc": doc.Name}

        result, error = gui.gui_call(task, _SAVE_TIMEOUT_S, "save_document", ctx.should_stop)
        if error:
            raise ToolError(f"save_document: {error}")
        if result.get("error"):
            raise ToolError(f"save_document: {result['error']}; nothing was written.")
        return f"Saved '{result['doc']}' to {target} ({_fmt_size(os.path.getsize(target))})."
    project = _bound_project("save_document")
    try:
        rel, note = project.save_now(doc_name)
    except ProjectError as e:
        raise ToolError(str(e)) from e
    if not rel:
        raise ToolError(f"save_document: '{doc_name}' was not saved ({note}); check the document name with "
                        "list_documents, or pass 'path' to save somewhere else.")
    abs_path = project.path(rel)
    return f"Saved '{doc_name}' to {abs_path} ({_fmt_size(os.path.getsize(abs_path))})."


# ---------------------------------------------------------------------------
# export_model

@tool(
    "export_model",
    "Export the model for manufacturing: STEP / STL / 3MF / BREP, one file per part (shell and lid "
    "separately), optionally re-oriented for printing. Files go to the project's exports/ folder. Returns each "
    "file with size and sha256.",
    {"type": "object", "properties": {
        "doc_name": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"},
                    "description": "Object names to export; default = every visible solid tip."},
        "formats": {"type": "array", "items": {"type": "string", "enum": list(FORMATS)},
                    "default": ["step", "stl"]},
        "per_part": {"type": "boolean", "default": True},
        "print_orientation": {"type": "boolean", "default": True,
                              "description": "Re-orient each part for printing (shell floor down, lid top down) "
                                             "using the spec."},
        "stl_tolerance": {"type": "number", "default": 0.05, "minimum": 0.001,
                          "description": "Mesh deviation in mm."},
        "directory": {"type": "string", "description": "Override the output folder (absolute). Never a system "
                                                                "or hidden/protected folder; existing files are not overwritten."}},
     "required": ["doc_name"]},
    category="deliver", mutating=True, phases=_VERIFY_TO_DELIVER,
    title=lambda a: "Export " + "+".join(str(f) for f in (a.get("formats") or ["step", "stl"])),
)
def _export_model(doc_name: str, objects: list | None = None, formats: list | None = None,
                  per_part: bool = True, print_orientation: bool = True, stl_tolerance: float = 0.05,
                  directory: str = ""):
    """Export STEP / STL / 3MF / BREP files of ``doc_name`` (``project_export.export_tips``, the same code as the
    auto-export of ``Project.finalize``): one line per file, then the print orientation that was applied from the
    project's spec.json (``print_orientation.shell|lid``) and the folder."""
    ctx = current_context()
    project = ctx.project
    wanted = [str(f).lower() for f in (formats or ["step", "stl"])] or ["step", "stl"]
    problems = []
    if project is None and not directory:
        problems.append("no project is bound to this chat, so there is no exports/ folder; pass 'directory' "
                        "(an absolute folder)")
    if directory and not os.path.isabs(directory):
        problems.append(f"'directory' must be an absolute path, got '{directory}'")
    if directory and os.path.isabs(directory):
        why = _write_location_problem(directory, project, directory=True)
        if why:
            problems.append(f"'directory' is {why}; export into the project folder (omit 'directory') or a folder in your Documents")
    unknown = [f for f in wanted if f not in FORMATS]
    if unknown:
        problems.append(f"unknown format(s) {', '.join(repr(f) for f in unknown)}; use {', '.join(FORMATS)}")
    try:
        tolerance = float(stl_tolerance)
    except (TypeError, ValueError):
        tolerance = None
        problems.append(f"'stl_tolerance' must be a number in mm, got {stl_tolerance!r}; use 0.05")
    if tolerance is not None and not 0.001 <= tolerance <= 5.0:   # NaN fails the comparison too
        problems.append(f"'stl_tolerance' {tolerance:g} is outside 0.001..5 mm; use 0.05")
    if problems:
        raise ToolError(f"export_model: {'; '.join(problems)}.")
    try:
        result = export_tips(project, doc_name, objects=[str(o) for o in objects] if objects else None,
                             formats=wanted, per_part=bool(per_part), print_orientation=bool(print_orientation),
                             stl_tolerance=tolerance, directory=directory or None,
                             should_stop=ctx.should_stop)
    except ProjectError as e:
        message = str(e)
        if message.startswith("export: "):
            message = message[len("export: "):]
        raise ToolError(f"export_model: {message}") from e
    except OSError as e:
        raise ToolError(f"export_model: the export folder is not writable: {e}; pass another 'directory'") from e
    if not result.files:
        detail = "; ".join(result.errors[:4]) or "nothing was produced"
        raise ToolError(f"export_model: no file was written ({detail}); check the document with get_objects.")
    lines = []
    for item in result.files:
        solids = f"{item.solids} solid" + ("" if item.solids == 1 else "s")
        lines.append(f"{item.rel}  {_fmt_size(item.bytes)}  sha256 {item.sha256[:12]}  ({item.label}, {solids})")
    if result.errors:
        lines.append("Some files were NOT written: " + "; ".join(result.errors[:6]))
    oriented = list(dict.fromkeys(f"{item.label} {item.orientation}" for item in result.files
                                  if item.orientation != AS_MODELED))
    if oriented:
        lines.append("Print orientation (print_orientation in spec.json): " + ", ".join(oriented) + "; the other parts "
                     "are as modelled.")
    lines += [f"Note: {note}." for note in result.notes[:4]]
    folder = os.path.dirname(result.files[0].path)
    lines.append(f"Folder: {folder}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# export_rebuild_script

def _step_counts(project: Project) -> tuple[int, int]:
    """``(replayable, not replayed)`` among the non-superseded step records."""
    steps = project.read_steps()
    replayable = sum(1 for s in steps if s.ok and (s.replay or {}).get("kind") in ("tool", "script"))
    return replayable, len(steps) - replayable


def _run_headless(code: str, timeout: float) -> dict[str, Any]:
    """``cad.headless.run`` (a function of its own so tests can replace the child process)."""
    from .cad import headless

    return headless.run(code, timeout=timeout)


def _live_summary(project: Project, doc_name: str, should_stop) -> dict:
    """``{doc: {object: volume | None}}`` of the live ``is_geometry`` objects of the project's documents (or of
    ``doc_name`` alone). Raises ``ToolError`` when a named document is not open."""
    names = [doc_name] if doc_name else list(dict.fromkeys(
        [*(project.manifest.get("docs") or []), *([project.manifest["active_doc"]] if project.manifest.get("active_doc") else [])]))

    def task() -> dict:
        import FreeCAD as App

        out: dict[str, Any] = {}
        missing = []
        for name in names:
            doc = gui.resolve_doc(name)
            if doc is None:
                missing.append(name)
            else:
                out[doc.Name] = gui.live_volumes(doc)
        return {"docs": out, "missing": missing, "open": list(App.listDocuments())}

    result, error = gui.gui_call(task, _SAVE_TIMEOUT_S, "rebuild_live_summary", should_stop)
    if error:
        raise ToolError(f"export_rebuild_script: could not read the live model ({error})")
    if doc_name and result["missing"]:
        raise ToolError(f"export_rebuild_script: document '{doc_name}' is not open; open documents: "
                        f"{', '.join(result['open']) or 'none'}.")
    return result["docs"]


@tool(
    "export_rebuild_script",
    "Write rebuild.py into the project: a script that replays every recorded step headless to rebuild the model "
    "from scratch. With run_check=true it also runs the script headless and reports whether the rebuilt model "
    "matches.",
    {"type": "object", "properties": {
        "doc_name": {"type": "string"},
        "run_check": {"type": "boolean", "default": False}}},
    category="deliver", mutating=True, phases=_VERIFY_TO_DELIVER,
)
def _export_rebuild_script(doc_name: str = "", run_check: bool = False):
    """Regenerate ``rebuild.py``; with ``run_check`` run it in a fresh headless FreeCAD and compare the rebuilt
    model with the live one (REBUILD_OK marker, object names, volumes within 1e-6)."""
    ctx = current_context()
    project = _bound_project("export_rebuild_script")
    try:
        rel = project.write_rebuild_py()
    except ProjectError as e:
        raise ToolError(str(e)) from e
    except OSError as e:
        raise ToolError(f"export_rebuild_script: rebuild.py could not be written: {e}") from e
    script = project.path(rel)
    replayable, skipped = _step_counts(project)
    lines = [f"rebuild.py written: {script} ({replayable} replayable step(s)"
             + (f", {skipped} recorded step(s) not replayed" if skipped else "") + ")."]
    if not run_check:
        return "\n".join(lines)
    if replayable == 0:
        raise ToolError("export_rebuild_script: there are no replayable steps yet, so there is nothing to check; "
                        "build the model first, then call this tool with run_check=true.")
    live = _live_summary(project, str(doc_name or "").strip(), ctx.should_stop)
    started = time.monotonic()
    child = _run_headless(f"exec(open({script!r}, encoding='utf-8').read())", _RUN_CHECK_TIMEOUT_S)
    output = str(child.get("output") or "")
    marker, summary = parse_rebuild_output(output)
    failures: list[str] = []
    if not marker:
        fails = [ln.strip() for ln in output.splitlines() if ln.startswith("FAIL ")]
        tail = output.strip().splitlines()[-1] if output.strip() else ""
        detail = child.get("error") or (fails[0] if fails else "") or tail or "no output"
        failures.append("the headless run did not print REBUILD_OK: " + str(detail)[:400])
    if summary is None:
        if marker:
            failures.append("the headless run printed no REBUILD_RESULT line to compare")
    else:
        failures += compare_rebuild(live, summary["docs"], REBUILD_TOLERANCE)
    seconds = time.monotonic() - started
    if failures:
        try:
            project.record_rebuild_check(False, "; ".join(failures)[:400])
        except (ProjectError, OSError):
            pass
        lines.append(f"Rebuild check: FAIL ({seconds:.0f} s). The rebuilt model does NOT match the live one:")
        lines += [f"- {f}" for f in failures[:8]]
        lines.append("Fix the steps that failed (see steps.jsonl / NOTES.md), or tell the user rebuild.py is not "
                     "reliable for this model.")
        return "\n".join(lines)
    objects = sum(len(v) for v in live.values())
    try:
        project.record_rebuild_check(True, f"{objects} objects matched")
    except (ProjectError, OSError):
        pass
    lines.append(f"Rebuild check: PASS ({seconds:.0f} s): REBUILD_OK, {objects} object(s) in {len(live)} "
                 f"document(s) match the live model (names and volumes within {REBUILD_TOLERANCE:g}).")
    return "\n".join(lines)
