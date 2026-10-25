"""Replay of recorded steps (private; the public names are re-exported by :mod:`oscar.project`).

``replay_steps`` re-runs the ``replay`` entries of step records through the SAME code path the app used
(``oscar.tools.run_tool`` for tool steps, ``oscar.cad.scripting.run_script`` / ``cad.headless.run`` for
saved scripts), into fresh documents, which is what ``rebuild.py`` and the headless rebuild check do.
"""

import copy
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from . import project_gui as gui

_TEXT_LIMIT = 1500
_CREATED = re.compile(r"Document '([^']+)' created")
RunTool = Callable[[str, dict], "tuple[str, str | None, bool]"]


@dataclass
class ReplayResult:
    """Outcome of ``replay_steps``: ``results`` rows are ``{"index", "tool", "ok", "text"}`` (override rows carry
    ``"override": True``); ``name_map`` maps each recorded document name to the document the replay used;
    ``doc_name`` is the replayed primary document; ``error`` is set when the replay refused to start."""

    ok: bool
    results: list = field(default_factory=list)
    doc_name: str = ""
    name_map: dict = field(default_factory=dict)
    error: str = ""


def _clip(text: Any) -> str:
    s = str(text or "")
    return s if len(s) <= _TEXT_LIMIT else s[:_TEXT_LIMIT] + f"...[{len(s) - _TEXT_LIMIT} chars omitted]"


def _app(fn, op: str = "replay"):
    """Run a small FreeCAD read/activation on the GUI thread (inline in console runs); raises on failure."""
    result, error = gui.gui_call(fn, 60.0, op)
    if error:
        raise RuntimeError(error)
    return result


def _open_documents() -> list[str]:
    import FreeCAD as App

    return list(App.listDocuments().keys())


def steps_from_literal(steps: list[dict]) -> list:
    """Turn the ``STEPS`` literal of a generated rebuild.py back into ``StepRecord`` objects.

    Literal entries are ``{"index", "tool", "title", "kind", "args" | "file", "headless", "gui_only", "doc"}``;
    fields the literal does not carry get defaults (``ok`` True: only successful steps are written).
    """
    from .project import StepRecord

    out = []
    for lit in steps or []:
        kind = lit.get("kind", "none")
        if kind == "script":
            replay = {"kind": "script", "file": lit.get("file", ""), "headless": bool(lit.get("headless", False)),
                      "gui_only": bool(lit.get("gui_only", False))}
        elif kind == "tool":
            replay = {"kind": "tool", "name": lit.get("tool", ""), "args": copy.deepcopy(lit.get("args") or {})}
        else:
            replay = {"kind": "none", "why": str(lit.get("why", ""))}
        out.append(StepRecord(
            index=int(lit.get("index", 0)), ts="", turn_id="", round=0, call_id="", tool=str(lit.get("tool", "")),
            title=str(lit.get("title", "")), args=copy.deepcopy(lit.get("args") or {}), ok=True, result_head="",
            result_chars=0, dur_s=0.0, doc=str(lit.get("doc", "")), delta={}, health=None, snapshot="",
            snapshot_note="", checkpoint="", checkpoint_note="", script=str(lit.get("file", "")), replay=replay))
    return out


def _created_name(text: str, before: list[str]) -> str:
    match = _CREATED.search(text or "")
    if match:
        return match.group(1)
    fresh = [n for n in _open_documents() if n not in before]
    return fresh[-1] if fresh else ""


def _run_script_step(step, project_root: str, name_map: dict[str, str]) -> tuple[bool, str]:
    replay = step.replay
    rel = replay.get("file") or step.script
    path = os.path.join(project_root, rel) if rel else ""
    if replay.get("gui_only"):
        return False, "not headless-replayable (uses ViewObject/Gui)"
    try:
        with open(path, encoding="utf-8") as f:
            code = f.read()
    except OSError as e:
        return False, f"script file '{rel}' cannot be read: {e}"
    doc = name_map.get(step.doc, step.doc)
    if doc:
        def activate():
            import FreeCAD as App

            if doc in App.listDocuments():
                App.setActiveDocument(doc)

        try:
            _app(activate, "replay_activate")
        except RuntimeError:
            pass
    if replay.get("headless"):
        from .cad import headless

        result = headless.run(code)
        text = result.get("output") or result.get("error") or ""
        return bool(result.get("success")), text
    from .cad import scripting

    result = scripting.run_script(code)
    return bool(result.get("success")), str(result.get("message") or result.get("error") or "")


def replay_steps(steps: list, *, project_root: str, doc_name: str = "", overrides: dict[str, str] | None = None,
                 run_tool: RunTool | None = None, continue_on_error: bool = False) -> ReplayResult:
    """Replay recorded ``steps`` into fresh documents.

    Rules: (0) it REFUSES (``ok=False``, ``error`` ``document '<name>' is open in this session``) when a target
    document is already open in a GUI session, or in any session when a recorded SCRIPT step is replayed:
    scripts that call ``App.getDocument("PiCase")`` would otherwise modify the ORIGINAL model while the replay
    builds ``PiCase_rebuild``; a tool-only replay in ``freecadcmd`` resolves a name clash with ``<name>_rebuild``. (1) Each ``create_document`` step creates its document (name
    clash -> ``<name>_rebuild``); ``name_map`` maps recorded -> actual names and every ``doc_name`` argument
    is rewritten through it; EVERY document of the project is replayed, in creation order; the function
    checks that ``create_document`` returned (and FreeCAD really has) the name it expected instead of
    relying on prefix resolution. (2) A ``tool`` step runs ``run_tool(name, args)`` (default
    ``oscar.tools.run_tool``, no ToolContext, so no hooks run). (3) A ``script`` step runs the saved file with
    ``cad.scripting.run_script`` (or ``cad.headless.run`` when ``headless``); a ``gui_only`` script fails
    with ``not headless-replayable (uses ViewObject/Gui)``. (4) After each ``set_parameters`` /
    ``enclosure_spec`` step the ``overrides`` (``{name: value}``) are applied through
    ``run_tool("set_parameter", ...)``; an override that cannot be applied yet (the sheet does not exist) is
    retried after later steps and once more at the end, and a recorded ``set_parameter`` of an overridden
    name replays with the override value. (5) The first failure stops the replay unless
    ``continue_on_error``. Steps that are superseded, failed or have ``replay.kind == "none"`` are skipped.
    Never raises for a failing step: it becomes a result row.
    """
    steps = sorted((s for s in steps if not getattr(s, "superseded", False)), key=lambda s: s.index)
    todo = [s for s in steps if s.ok and (s.replay or {}).get("kind") in ("tool", "script")]
    result = ReplayResult(ok=True)
    pending = dict(overrides or {})
    if run_tool is None:
        from .tools import run_tool as _default_run_tool

        run_tool = _default_run_tool

    import FreeCAD as App

    targets = [s.doc for s in todo if s.tool == "create_document" and s.doc]
    if doc_name and doc_name not in targets:
        targets.append(doc_name)
    scripted = any(s.replay.get("kind") == "script" for s in todo)
    if getattr(App, "GuiUp", False) or scripted:
        clash = [t for t in targets if t in _open_documents()]
        if clash:
            # A recorded script addresses its document BY NAME (App.getDocument("PiCase")): with the original open it
            # would modify the original while the replay builds PiCase_rebuild. Tool steps are rewritten through
            # name_map, scripts cannot be, so any session (GUI or console) refuses when a script is involved.
            result.ok = False
            result.error = f"document '{clash[0]}' is open in this session"
            if not getattr(App, "GuiUp", False):
                result.error += "; recorded scripts address documents by name, close it first or replay headless"
            result.results.append({"index": 0, "tool": "replay", "ok": False, "text": result.error})
            return result

    def row(step, ok: bool, text: Any, **extra) -> dict:
        r = {"index": step.index, "tool": step.tool, "ok": bool(ok), "text": _clip(text)}
        r.update(extra)
        result.results.append(r)
        if not ok:
            result.ok = False
        return r

    def apply_overrides(step) -> bool:
        """Try every pending override; True when none of them hit a real failure that must stop the replay."""
        for name in list(pending):
            target = next(iter(result.name_map.values()), "") if not doc_name else result.name_map.get(doc_name, doc_name)
            text, _img, is_error = run_tool("set_parameter", {"doc_name": target, "name": name, "value": pending[name]})
            if not is_error:
                row(step, True, text, override=True)
                result.results[-1]["tool"] = "set_parameter"
                del pending[name]
        return True

    for step in todo:
        replay = step.replay
        if replay["kind"] == "tool":
            args = copy.deepcopy(replay.get("args") or {})
            if isinstance(args.get("doc_name"), str) and args["doc_name"] in result.name_map:
                args["doc_name"] = result.name_map[args["doc_name"]]
            if step.tool == "set_parameter" and args.get("name") in pending:
                args["value"] = pending.pop(args["name"])
            if step.tool == "create_document":
                recorded = step.doc or str(args.get("name", ""))
                before = _open_documents()
                clash = recorded in before or str(args.get("name", "")) in before
                if clash:
                    args["name"] = f"{args.get('name') or recorded}_rebuild"
                text, _img, is_error = run_tool("create_document", args)
                actual = "" if is_error else _created_name(text, before)
                if not is_error and (not actual or actual not in _open_documents()):
                    is_error, text = True, f"create_document returned no open document (got '{actual}'): {text}"
                elif not is_error and not clash and actual != recorded:
                    is_error = True
                    text = f"create_document returned '{actual}' but the recorded model used '{recorded}': {text}"
                if not is_error:
                    result.name_map[recorded] = actual
                    result.doc_name = result.doc_name or actual
                row(step, not is_error, text)
            else:
                text, _img, is_error = run_tool(step.tool, args)
                row(step, not is_error, text)
        else:
            ok, text = _run_script_step(step, project_root, result.name_map)
            row(step, ok, text)
        if not result.ok and not continue_on_error:
            return result
        if pending and step.tool in ("set_parameters", "enclosure_spec") and result.name_map:
            apply_overrides(step)
    if pending and result.name_map and (result.ok or continue_on_error):
        last = todo[-1] if todo else None
        if last is not None:
            apply_overrides(last)
        for name, value in pending.items():
            result.ok = False
            result.results.append({"index": last.index if last else 0, "tool": "set_parameter", "ok": False,
                                   "text": f"override '{name}={value}' was not applied: no such parameter in the "
                                           "replayed document (check the name against params.json)",
                                   "override": True})
    elif pending and result.ok:
        result.ok = False
        result.results.append({"index": 0, "tool": "set_parameter", "ok": False,
                               "text": "overrides were not applied: the replay created no document", "override": True})
    if not result.doc_name and doc_name:
        result.doc_name = result.name_map.get(doc_name, "")
    return result


def replay_summary(result: ReplayResult) -> dict:
    """``{"docs": {doc_name: {object_name: volume_mm3 | None}}}`` for the ``is_geometry`` objects that hold solids
    in every replayed document (None = no solid; Origin features and containers are excluded because their
    ``Volume`` is uninitialised garbage). Keys are the ACTUAL names of the replayed documents (the values
    of ``name_map``). Printed by rebuild.py as ``REBUILD_RESULT <json>``."""
    names = list(dict.fromkeys(result.name_map.values())) or ([result.doc_name] if result.doc_name else [])

    def collect() -> dict:
        import FreeCAD as App

        docs = App.listDocuments()
        return {n: gui.live_volumes(docs[n]) for n in names if n in docs}

    try:
        return {"docs": _app(collect, "replay_summary")}
    except RuntimeError:
        return {"docs": {}}


# ---------------------------------------------------------------------------
# the headless rebuild check (export_rebuild_script run_check=true)

REBUILD_TOLERANCE = 1e-6   # relative volume tolerance of the rebuild check


def parse_rebuild_output(output: str) -> tuple[bool, dict | None]:
    """``(marker, summary)`` of the output of a headless ``rebuild.py`` run.

    ``marker`` is True when a line equal to ``REBUILD_OK`` is present (the success signal: the child may exit
    134 at teardown, a platform quirk, so the exit status is NOT used); ``summary`` is the parsed
    ``REBUILD_RESULT <json>`` object (``{"docs": {...}}``) or None when the line is missing or malformed."""
    marker, summary = False, None
    for line in str(output or "").splitlines():
        line = line.strip()
        if line == "REBUILD_OK":
            marker = True
        elif line.startswith("REBUILD_RESULT "):
            try:
                parsed = json.loads(line[len("REBUILD_RESULT "):])
            except ValueError:
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("docs"), dict):
                summary = parsed
    return marker, summary


def _same_volume(a: Any, b: Any, tolerance: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= tolerance * max(abs(a), abs(b), 1e-9)


def compare_rebuild(live: dict, rebuilt: dict, tolerance: float = REBUILD_TOLERANCE) -> list[str]:
    """What differs between the live model and a rebuilt one: ``live`` and ``rebuilt`` are the ``docs`` mapping of
    ``replay_summary`` (``{doc: {object: volume | None}}``, ``is_geometry`` objects only). Returns one text per
    problem (empty list = equal): a missing document, an object-name set that differs, a volume that differs by
    more than ``tolerance`` relative. Pure."""
    problems: list[str] = []
    for doc, objects in sorted((live or {}).items()):
        other = (rebuilt or {}).get(doc)
        if other is None:
            problems.append(f"document '{doc}' was not rebuilt")
            continue
        missing, extra = sorted(set(objects) - set(other)), sorted(set(other) - set(objects))
        if missing or extra:
            problems.append(f"object names differ in '{doc}' (missing after rebuild: {missing[:6] or 'none'}; "
                            f"extra: {extra[:6] or 'none'})")
        for name in sorted(set(objects) & set(other)):
            if not _same_volume(objects[name], other[name], tolerance):
                problems.append(f"volume of '{doc}.{name}' differs: live {objects[name]} vs rebuilt {other[name]}")
    return problems
