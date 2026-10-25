"""Scripted execution.

Two execution modes share one seed namespace (``FreeCAD``/``App``, ``FreeCADGui``/``Gui``, ``Part``,
``Sketcher``, ``Spreadsheet``, ``math``, ``Mesh``, ``Import`` pre-imported; helpers ``qv``, ``param``,
``set_param``; the persistent ``state`` dict; whatever ``register_helpers`` added):

- ``run_script`` executes on the GUI thread — the safe default, because document and view APIs may only run
  there. Every run gets a FRESH namespace (``__name__ == "__main__"``, a synthetic ``__file__``): variables of an
  earlier script are gone (carry data in the document or in ``state``), a shadowed builtin cannot leak into the
  next script, and a ``if __name__ == "__main__":`` guard builds something instead of silently doing nothing.
  The run is ATOMIC: on any exception every change it made is rolled back (see ``run_script``).
- ``start_job`` executes on a worker thread for heavy geometry work that touches neither documents nor views.
  Such scripts apply their document writes through the injected ``commit(fn)`` helper, which bounces ``fn``
  onto the GUI thread and waits for it.

Job results land in a registry (state/error/traceback) the chat can poll with the async-status tool, so a failed
background script is always observable.
"""

import ast
import contextlib
import io
import itertools
import keyword
import math
import os
import re
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from typing import Any

import FreeCAD
import FreeCADGui

from .. import netguard
from ..toolctx import current_context
from . import dispatch
from .health_eval import snapshot_state
from .health_types import ObjectDelta, StateSnapshot, delta_between

# ---------------------------------------------------------------------------
# seed namespace

# The default seed. Runs copy it into a fresh dict, so changing a name here (or adding one) is visible to later
# runs, while nothing a script does to ITS namespace can leak back.
NAMESPACE: dict[str, Any] = {
    # Scripts run like a macro: without this ``__name__`` resolves to
    # "builtins" and the ubiquitous ``if __name__ == "__main__": main()``
    # idiom silently does nothing.
    "__name__": "__main__",
    "FreeCAD": FreeCAD,
    "App": FreeCAD,
    "FreeCADGui": FreeCADGui,
    "Gui": FreeCADGui,
    "math": math,
}
for _mod in ("Part", "Sketcher", "Spreadsheet", "Mesh", "Import"):
    try:  # a module that is missing in this build is simply not pre-imported
        NAMESPACE[_mod] = __import__(_mod)
    except Exception:  # noqa: BLE001
        pass
del _mod
# Model scripts often use FreeCAD.__file__ (it does not exist for
# builtin modules) — point it at the FreeCAD library so those scripts
# run instead of dying with AttributeError.
try:
    if not hasattr(FreeCAD, "__file__"):
        FreeCAD.__file__ = os.path.join(FreeCAD.getHomePath(), "lib",
                                        "FreeCAD.so")
except Exception:
    pass

# The ONLY thing that persists between scripts besides the documents themselves.
STATE: dict[str, Any] = {}

_in_worker = threading.local()

JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_KEEP_FINISHED = 20


def _record(job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        job = JOBS.pop(job_id, {"id": job_id})
        job.update(fields)
        JOBS[job_id] = job
        finished = [k for k, v in JOBS.items() if v.get("state") in {"done", "failed"}]
        for key in finished[: max(0, len(finished) - _KEEP_FINISHED)]:
            del JOBS[key]


def commit(fn: Callable[[], Any], timeout: float = 120.0) -> Any:
    """Apply a document/view write from a worker script on the GUI thread."""
    if not getattr(_in_worker, "active", False):
        raise RuntimeError("commit() is only available inside background scripts")

    def guarded():
        # Same transaction as run_script (and engine.txn_task): without it
        # the write never enters the undo stack and "revert turn" skips it.
        doc = FreeCAD.ActiveDocument
        if doc is not None:
            doc.openTransaction("Oscar: background job")
        try:
            # The callback is the script's code running on the GUI thread: it gets the script's network guard.
            with netguard.script_guard():
                result = fn()
                _recompute_touched()
            return result
        except SystemExit as e:
            # Defence in depth: dispatch's drain loop also converts a
            # SystemExit, but only into "SystemExit: n"; say what happened.
            raise RuntimeError(f"script called exit({e.code!r}) inside commit()") from None
        finally:
            try:
                if doc is not None and doc.Name in FreeCAD.listDocuments():
                    doc.commitTransaction()
            except Exception:
                pass  # the callback closed/replaced the document itself

    result, failed = dispatch.dispatch(guarded, timeout=timeout, op="commit")
    if failed:
        raise RuntimeError(f"commit failed: {result}")
    return result


NAMESPACE["commit"] = commit


# ---------------------------------------------------------------------------
# helpers injected into every script


def qv(value: Any) -> Any:
    """The plain number of a spreadsheet / property value: ``Quantity.Value`` for a quantity, else the value itself
    (``Spreadsheet.get`` returns a Quantity for cells with a unit and a bare int / float for unitless cells, so
    ``.Value`` on the result is a classic crash). ``qv(sheet.get("wall_t"))`` always works."""
    return getattr(value, "Value", value)


_SHEET_NAMES = ("Params", "Parameters")


def _param_sheet(doc: Any = None) -> tuple[Any, Any]:
    doc = doc if doc is not None else FreeCAD.ActiveDocument
    if doc is None:
        raise RuntimeError("no document is open")
    for name in _SHEET_NAMES:
        sheet = doc.getObject(name)
        if sheet is not None and sheet.TypeId == "Spreadsheet::Sheet":
            return doc, sheet
    raise ValueError(f"document '{doc.Name}' has no 'Params' spreadsheet; create it with set_parameters first")


def _known_aliases(sheet: Any) -> list[str]:
    out = []
    try:
        for addr in sorted(sheet.getUsedCells()):
            alias = sheet.getAlias(addr)
            if alias:
                out.append(alias)
    except Exception:
        pass
    return out


def param(name: str, doc: Any = None) -> float:
    """The number stored under ``name`` in the document's ``Params`` (or ``Parameters``) spreadsheet, in the
    sheet's own unit (mm for lengths). Raises ``ValueError`` naming the known parameters when ``name`` is missing
    or its cell holds text."""
    doc, sheet = _param_sheet(doc)
    if not sheet.getCellFromAlias(name):
        known = ", ".join(_known_aliases(sheet)[:20]) or "none"
        raise ValueError(f"param '{name}' is not in sheet '{sheet.Name}'; known parameters: {known}")
    value = sheet.get(name)
    if isinstance(value, str):
        raise ValueError(f"param '{name}' holds the text '{value}', not a number; "
                         "rewrite the cell as a formula such as '=-1.5 mm'")
    return float(qv(value))


def set_param(name: str, value: Any, unit: str = "", doc: Any = None) -> None:
    """Set an EXISTING parameter of the ``Params`` / ``Parameters`` sheet and recompute (dependents follow because
    objects are expression-bound). A number is written as a formula (``=2.4 mm``) so a negative value is never
    stored as text; a string starting with ``=`` is written as given. ``unit`` defaults to the cell's current unit.
    Raises ``ValueError`` when the parameter does not exist (create rows with set_parameters)."""
    doc, sheet = _param_sheet(doc)
    cell = sheet.getCellFromAlias(name)
    if not cell:
        known = ", ".join(_known_aliases(sheet)[:20]) or "none"
        raise ValueError(f"param '{name}' is not in sheet '{sheet.Name}'; known parameters: {known} "
                         "(set_parameters creates new rows)")
    if isinstance(value, str):
        if not value.startswith("="):
            raise ValueError(f"set_param('{name}', {value!r}): pass a number, or a formula starting with '='")
        text = value
    else:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"set_param('{name}'): {value!r} is not a finite number")
        if not unit:
            old = sheet.get(name)
            try:
                unit = str(old).split(" ", 1)[1] if hasattr(old, "Value") else ""
            except Exception:
                unit = ""
        text = f"={format(number, '.15g')}{' ' + unit if unit else ''}"
    sheet.set(cell, text)
    doc.recompute()
    if isinstance(sheet.get(name), str):
        raise ValueError(f"set_param('{name}'): the cell was stored as text; use the form '=<number> <unit>'")


NAMESPACE["qv"] = qv
NAMESPACE["param"] = param
NAMESPACE["set_param"] = set_param

_RESERVED_HELPERS = frozenset(set(NAMESPACE) | {"state", "FreeCAD", "App", "Gui", "FreeCADGui", "__name__", "__file__"})
_HELPERS: dict[str, Any] = {}
_HELPERS_LOCK = threading.Lock()


def register_helpers(name: str, obj: Any) -> None:
    """Make ``obj`` available as the global ``name`` in every script that starts after this call (``execute_code``
    and ``execute_code_async``). The kit registers ``kit`` and ``P`` this way. Registering the same name again
    replaces it (a re-import is harmless). Raises ``ValueError`` for a name that is not a plain identifier, is a
    keyword, is dunder, or would shadow a built-in seed name (``App``, ``Part``, ``qv``, ``state``, ...)."""
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) or name.startswith("__"):
        raise ValueError(f"register_helpers: {name!r} is not a usable global name; use a plain identifier "
                         "that does not start with '__'")
    if name in _RESERVED_HELPERS:
        raise ValueError(f"register_helpers: '{name}' is already a built-in name of the script namespace; "
                         "pick another name")
    with _HELPERS_LOCK:
        _HELPERS[name] = obj


def unregister_helpers(name: str) -> None:
    """Remove a helper registered with ``register_helpers`` (no error when it is not registered)."""
    with _HELPERS_LOCK:
        _HELPERS.pop(name, None)


def registered_helpers() -> dict[str, Any]:
    """A copy of the registered helpers."""
    with _HELPERS_LOCK:
        return dict(_HELPERS)


def reset_state() -> None:
    """Empty the persistent ``state`` dict."""
    STATE.clear()


def _synthetic_file() -> str:
    """A believable ``__file__`` for scripts (many use ``os.path.dirname(__file__)``); the file need not exist."""
    try:
        from ..paths import cache_dir

        return os.path.join(cache_dir("scripts"), "execute_code.py")
    except Exception:
        import tempfile

        return os.path.join(tempfile.gettempdir(), "execute_code.py")


def new_namespace() -> dict[str, Any]:
    """A fresh script namespace: the seed, the registered helpers, ``state``, ``__name__ == "__main__"`` and a
    synthetic ``__file__``. Nothing a script does to it can leak into the next run."""
    ns = dict(NAMESPACE)
    ns.update(registered_helpers())
    ns.update({"__name__": "__main__", "__file__": _synthetic_file(), "__doc__": None, "state": STATE})
    return ns


# ---------------------------------------------------------------------------
# error formatting

_SCRIPT_FILENAME = "<execute_code>"
_OUTPUT_LIMIT = 8000  # characters of printed output handed back

# Every in-process run compiles under its OWN filename ("<execute_code:7>"). A function one run stored (``state``,
# a registered helper, a document-level callback) and a later run then calls reports ITS line in ITS source, not
# whatever the later script has on that line number. The sources of recent runs are kept (bounded) to look the
# failing line up.
_SOURCES_KEPT = 50
_script_ids = itertools.count(1)
_script_sources: dict[str, list[str]] = {}

_PREIMPORTED = ("Part", "Sketcher", "Spreadsheet", "math", "Mesh", "Import", "App", "FreeCAD", "Gui", "FreeCADGui")


def _register_script(code: str) -> str:
    """The compile filename of one run (``<execute_code:N>``); its source is remembered for later error lookups."""
    filename = f"{_SCRIPT_FILENAME[:-1]}:{next(_script_ids)}>"
    _script_sources[filename] = code.splitlines()
    while len(_script_sources) > _SOURCES_KEPT:
        del _script_sources[next(iter(_script_sources))]
    return filename


def _is_script_file(name: str) -> bool:
    """True for the compile filename of an in-process run (the generic one or a per-run ``<execute_code:N>``)."""
    return name == _SCRIPT_FILENAME or (name.startswith(_SCRIPT_FILENAME[:-1] + ":") and name.endswith(">"))


def _clip(text: str) -> str:
    if len(text) <= _OUTPUT_LIMIT:
        return text
    return text[: _OUTPUT_LIMIT // 2] + "\n…(output truncated)…\n" + text[-_OUTPUT_LIMIT // 2:]


def _hints_for(type_name: str, mro: "tuple[str, ...] | list[str]", text: str, missing: str | None = None) -> list[str]:
    """Short, concrete fixes for the errors models hit again and again (at most two). Works from names and text only, so
    the headless helper (whose exception only reaches us as a report) gets the same hints as an in-process script."""
    classes = set(mro) | {type_name}
    out: list[str] = []
    if "AttributeError" in classes and re.search(r"'(int|float)' object has no attribute 'Value'", text):
        out.append("Spreadsheet.get() returns a plain number for a unitless cell and a Quantity only for cells with a "
                   "unit; read values with qv(x) (same as getattr(x, 'Value', x)) or param('name').")
    elif "NameError" in classes:
        missing = missing or (re.search(r"name '(\w+)' is not defined", text) or [None, None])[1]
        if missing in _PREIMPORTED:
            out.append(f"'{missing}' is pre-imported in execute_code but not in other processes "
                       f"(execute_code_headless needs its own 'import {missing}').")
        elif missing in ("os", "sys", "re", "json", "random", "time", "itertools", "collections", "pathlib"):
            out.append(f"add 'import {missing}' at the top of the script.")
        else:
            out.append("every execute_code run starts with a fresh namespace: variables and functions of an earlier "
                       "script no longer exist. Re-create them, or look objects up with App.getDocument(...).getObject(...); "
                       "use the 'state' dict to carry plain data between runs.")
    elif type_name == "OCCError" or "Standard_" in text or "BRep_API" in text:
        out.append("OpenCASCADE rejected the operation: typical causes are a zero or negative dimension, a profile that "
                   "is not closed or self-intersects, a fillet / chamfer larger than the edge allows, or a boolean on "
                   "an invalid shape. Check the numbers and shape.isValid(), then retry with a simpler operation.")
    elif "ImportError" in classes:
        what = f"'{missing}'" if missing else "that module"
        out.append(f"{what} is not installed in the app's Python and pip packages cannot be added: only the standard "
                   "library, FreeCAD's own modules and the packages the app bundles (numpy) can be imported; rewrite "
                   "the step without it.")
    elif "AttributeError" in classes and "'NoneType' object has no attribute" in text:
        out.append("a lookup returned None: doc.getObject('Name') is None when the object's internal Name does not exist "
                   "(Name is not Label); list the objects with get_objects first.")
    elif "AttributeError" in classes and re.search(r"'(App\.)?Document' object has no attribute", text):
        out.append("a document exposes an object as an attribute only under its internal Name; use "
                   "doc.getObject('Name') and check the Names with get_objects.")
    elif any(t in text.lower() for t in ("null shape", "shape is null", "shape is invalid")):
        out.append("the shape is null or invalid: the object failed to recompute or has no geometry yet. Check "
                   "obj.State and obj.getStatusString() after doc.recompute() before reading obj.Shape.")
    return out[:2]


def _missing_name(exc: BaseException) -> str | None:
    """The name a NameError / ImportError is about (``exc.name``), else None."""
    return getattr(exc, "name", None) if isinstance(exc, (NameError, ImportError)) else None


def _hints(exc: BaseException) -> list[str]:
    """``_hints_for`` of a live exception."""
    return _hints_for(type(exc).__name__, [c.__name__ for c in type(exc).__mro__], str(exc), _missing_name(exc))


def format_script_error(code: str, type_name: str, message: str, lines: "list[int] | tuple[int, ...]", *,
                        mro: "tuple[str, ...] | list[str]" = (), missing: str | None = None,
                        names: "tuple[str, ...] | list[str]" = (),
                        last_source: "list[str] | None" = None) -> str:
    """The ONE model-facing error format of every script runner: ``Type: message``, the failing script line and its
    source, the call chain (the other script frames: ``lines`` is the line of each script frame, outermost first,
    ``names`` optionally the function name of each) and, for a known trap, a ``hint:`` line. ``execute_code`` and
    ``execute_code_headless`` both produce it. ``last_source`` (source lines of ANOTHER, earlier run) says the failing
    frame was defined by that run: its line is looked up there and labelled, never in ``code`` (an empty list: the
    earlier source is gone, so no line is shown rather than a wrong one)."""
    earlier = last_source is not None
    src = last_source if earlier else code.splitlines()
    line_no = lines[-1] if lines else None
    text = f"{type_name}: {message}"
    if line_no and 0 < line_no <= len(src):
        where = " (an earlier execute_code call)" if earlier else ""
        text += f"\n  at script line {line_no}{where}: {src[line_no - 1].strip()}"
    chain = [f"  line {n}" + (f" in {names[i]}" if i < len(names) and names[i] not in ("", "<module>") else "")
             for i, n in enumerate(list(lines)[:-1])]
    if chain:
        text += "\n  call chain:\n" + "\n".join(chain)
    for hint in _hints_for(type_name, mro, message, missing):
        text += f"\n  hint: {hint}"
    return text


def _script_error(code: str, exc: BaseException, filename: str = _SCRIPT_FILENAME) -> str:
    """Model-facing error of an in-process script: exception, the failing script line, the traceback frames that belong
    to the script (not the engine's) and, when the error is a known trap, a one-line hint."""
    lines: list[int] = []
    if isinstance(exc, SyntaxError) and exc.filename == filename and exc.lineno:
        lines = [exc.lineno]
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if _is_script_file(f.filename)]
    if frames:
        lines = [f.lineno for f in frames]
    earlier = None
    if frames and frames[-1].filename != filename:     # the failing function was defined by an earlier run
        earlier = _script_sources.get(frames[-1].filename, [])
    text = format_script_error(code, type(exc).__name__, str(exc), lines,
                               mro=[c.__name__ for c in type(exc).__mro__],
                               missing=_missing_name(exc), names=[f.name for f in frames], last_source=earlier)
    return text


# ---------------------------------------------------------------------------
# AST pre-flight


def _is_fresh_feature_call(node: ast.AST) -> bool:
    """``<doc>.addObject("Part::Feature" | "Part::FeaturePython", ...)``: a plain object that keeps an assigned Shape."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "addObject"
            and bool(node.args) and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in ("Part::Feature", "Part::FeaturePython"))


def preflight_warnings(code: str) -> list[str]:
    """Static warnings about a script that will probably not do what its author expects (never blocks it).

    Currently: ``X.Shape = ...`` where X is not a plain ``Part::Feature`` created by the same script. A parametric
    feature (Part::Box, Part::Cut, Pad, ...) rebuilds its own Shape on every recompute, so the assignment is silently
    discarded and the model keeps the old geometry. Returns ``[]`` for code that does not parse."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    fresh: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_fresh_feature_call(node.value):
            fresh.update(t.id for t in node.targets if isinstance(t, ast.Name))
    warnings: list[str] = []
    seen: set[int] = set()
    for node in ast.walk(tree):
        targets = (node.targets if isinstance(node, ast.Assign) else
                   [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
        for t in targets:
            if not (isinstance(t, ast.Attribute) and t.attr == "Shape"):
                continue
            base = t.value
            if (isinstance(base, ast.Name) and base.id in fresh) or _is_fresh_feature_call(base):
                continue
            if node.lineno in seen:
                continue
            seen.add(node.lineno)
            try:
                text = ast.unparse(t)
            except Exception:
                text = "<object>.Shape"
            warnings.append(
                f"line {node.lineno}: '{text} = ...' is discarded when that object recomputes (parametric features such "
                "as Part::Box, Part::Cut or Pad rebuild their own Shape). To show a custom shape add a plain "
                "doc.addObject('Part::Feature', 'Name') and assign its Shape, or change the feature's inputs instead.")
    return warnings


# ---------------------------------------------------------------------------
# transactions and rollback

_MAX_DOCS = 64            # documents that get a transaction / snapshot per run, the active one first
_SNAPSHOT_TOTAL_S = 3.0   # wall-clock budget of ALL the before (or all the after) snapshots of one run


class _Txn:
    """One document's transaction around a script run."""

    __slots__ = ("name", "uid", "doc", "prev_mode", "mode_changed", "opened", "names_before")

    def __init__(self, doc: Any):
        self.doc = doc
        self.name = doc.Name
        self.uid = getattr(doc, "Uid", "")
        self.prev_mode = None
        self.mode_changed = False
        self.opened = False
        self.names_before: set[str] = set()


def _live_docs(limit: int) -> list[Any]:
    """Open documents, the active one first, at most ``limit``."""
    docs = list(FreeCAD.listDocuments().values())
    active = FreeCAD.ActiveDocument
    docs.sort(key=lambda d: 0 if (active is not None and d.Name == active.Name) else 1)
    return docs[:limit]


def _open_transactions(label: str) -> list[_Txn]:
    """Open a transaction on every live document. ``freecadcmd`` documents have ``UndoMode == 0`` and then
    ``abortTransaction`` undoes NOTHING (verified), so the mode is set to 1 for the run and restored afterwards."""
    txns: list[_Txn] = []
    for doc in _live_docs(_MAX_DOCS):
        t = _Txn(doc)
        try:
            t.names_before = {o.Name for o in doc.Objects}
            t.prev_mode = doc.UndoMode
            if t.prev_mode != 1:
                doc.UndoMode = 1
                t.mode_changed = True
            doc.openTransaction(label)
            t.opened = True
        except Exception:  # noqa: BLE001 - a document that cannot take a transaction is just not rollback-able
            pass
        txns.append(t)
    return txns


def _same_doc(t: _Txn) -> Any:
    """The live document of ``t`` (None when the script closed or replaced it)."""
    doc = FreeCAD.listDocuments().get(t.name)
    if doc is None or (t.uid and getattr(doc, "Uid", t.uid) != t.uid):
        return None
    return doc


def _close_transactions(txns: list[_Txn], commit_changes: bool) -> None:
    for t in txns:
        doc = _same_doc(t)
        if doc is None:
            continue
        try:
            if t.opened:
                if commit_changes:
                    doc.commitTransaction()
                else:
                    doc.abortTransaction()
        except Exception:  # noqa: BLE001
            pass
        try:
            if t.mode_changed:
                doc.UndoMode = t.prev_mode
        except Exception:  # noqa: BLE001
            pass


def _discard_created(doc: Any, names: list[str]) -> list[str]:
    """Fallback rollback: remove objects the failed script created (newest first). Returns the names still present."""
    for name in reversed(names):
        try:
            if doc.getObject(name) is not None:
                doc.removeObject(name)
        except Exception:  # noqa: BLE001
            pass
    return [n for n in names if doc.getObject(n) is not None]


def _rollback(txns: list[_Txn], docs_before: set[str], active_before: str | None) -> dict[str, Any]:
    """Undo everything a failed script did and report it from a before / after diff.

    Order: record what changed, ``abortTransaction`` on every document (it removes created objects, restores deleted
    ones and reverts property / expression edits), then verify; objects that survived the abort are removed by name
    (the diff is only a fallback: it cannot restore a deleted sheet or an edited value). Documents the script CREATED
    are closed (a retry would otherwise create ``Doc1``)."""
    report: dict[str, Any] = {"removed": {}, "restored": {}, "leftover": {}, "closed_docs": [], "closed_by_script": []}
    created_names: dict[str, list[str]] = {}
    deleted_names: dict[str, list[str]] = {}
    for t in txns:
        doc = _same_doc(t)
        if doc is None:
            report["closed_by_script"].append(t.name)
            continue
        now = [o.Name for o in doc.Objects]
        created_names[t.name] = [n for n in now if n not in t.names_before]
        deleted_names[t.name] = sorted(t.names_before - set(now))
    _close_transactions(txns, commit_changes=False)
    for t in txns:
        doc = _same_doc(t)
        if doc is None:
            continue
        created = created_names.get(t.name, [])
        left = _discard_created(doc, [n for n in created if doc.getObject(n) is not None]) if created else []
        gone = [n for n in created if n not in left]
        if gone:
            report["removed"][t.name] = gone
        if left:
            report["leftover"][t.name] = left
        restored = [n for n in deleted_names.get(t.name, []) if doc.getObject(n) is not None]
        if restored:
            report["restored"][t.name] = restored
        _recompute_touched((doc,))
    for name in sorted(set(FreeCAD.listDocuments()) - docs_before):
        try:
            FreeCAD.closeDocument(name)
            report["closed_docs"].append(name)
        except Exception:  # noqa: BLE001
            pass
    if active_before and active_before in FreeCAD.listDocuments():
        try:
            FreeCAD.setActiveDocument(active_before)
            if getattr(FreeCAD, "GuiUp", 0):
                FreeCADGui.setActiveDocument(active_before)
        except Exception:  # noqa: BLE001
            pass
    return report


def _rollback_text(report: dict[str, Any]) -> str:
    """One model-facing paragraph about what the rollback did."""
    def names(d: dict[str, list[str]]) -> str:
        flat = [n for v in d.values() for n in v]
        return ", ".join(flat[:6]) + (f" (+{len(flat) - 6} more)" if len(flat) > 6 else "")

    parts = []
    if report["removed"]:
        parts.append(f"removed {sum(len(v) for v in report['removed'].values())} created object(s): {names(report['removed'])}")
    if report["restored"]:
        parts.append(f"restored {sum(len(v) for v in report['restored'].values())} deleted object(s): {names(report['restored'])}")
    if report["closed_docs"]:
        parts.append("closed the document(s) the script created: " + ", ".join(report["closed_docs"]))
    text = ("ROLLED BACK: the failed script's changes were undone"
            + (f" ({'; '.join(parts)})" if parts else "")
            + ". The model is as it was before the script ran, so fix the script and run it again IN FULL "
              "(do not continue from the middle).")
    if report["leftover"]:
        text += (f"\nROLLBACK INCOMPLETE: could not remove {names(report['leftover'])}; delete them with delete_object "
                 "before retrying (check get_objects).")
    if report["closed_by_script"]:
        text += ("\nWARNING: the script closed document(s) " + ", ".join(report["closed_by_script"])
                 + "; closed documents cannot be restored.")
    return text


def _recompute_touched(docs: "list[Any] | tuple[Any, ...] | None" = None) -> None:
    """Scripts often forget doc.recompute(); geometry then never appears.
    Recompute every document left with pending (touched) objects. An object that already failed to recompute stays
    ['Touched', 'Invalid'] forever, so only an object that is Touched and NOT Invalid triggers a recompute (a broken
    object must not be re-run, and re-reported, by every later script)."""
    for doc in list(docs if docs is not None else FreeCAD.listDocuments().values()):
        try:
            if any("Touched" in (o.State or []) and "Invalid" not in (o.State or []) for o in doc.Objects):
                doc.recompute()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# run_script


def _snapshot_docs(docs: list[Any]) -> dict[str, StateSnapshot]:
    """``snapshot_state`` of each document within ``_SNAPSHOT_TOTAL_S`` in total (a document that did not get its turn
    has no entry: its delta is simply not reported). Never raises: a delta is a nicety, not a reason to fail a script."""
    out: dict[str, StateSnapshot] = {}
    deadline = time.monotonic() + _SNAPSHOT_TOTAL_S
    for doc in docs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            out[doc.Name] = snapshot_state(doc, prev_findings=[], budget_s=remaining)
        except Exception:  # noqa: BLE001
            pass
    return out


def _deltas(before: dict[str, StateSnapshot],
            docs_before: set[str]) -> tuple[dict[str, ObjectDelta], dict[str, list[str]]]:
    """Per-document object delta of the run, and the geometry objects that disappeared (``lost_geometry``).
    Documents that existed before the run but were not snapshotted (over the snapshot budget) are skipped."""
    deltas: dict[str, ObjectDelta] = {}
    lost: dict[str, list[str]] = {}
    deadline = time.monotonic() + _SNAPSHOT_TOTAL_S
    docs = list(FreeCAD.listDocuments().items())
    docs.sort(key=lambda item: 0 if item[0] in before else 1)     # the documents we can compare first
    for name, doc in docs:
        old = before.get(name)
        if old is None and name in docs_before:
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            now = snapshot_state(doc, prev_findings=[], budget_s=remaining)
        except Exception:  # noqa: BLE001
            continue
        if old is None:  # a document the script created: everything in it is new
            if now.objects:
                deltas[name] = ObjectDelta(added=list(now.objects))
            continue
        d = delta_between(old, now)
        if not d.is_empty():
            deltas[name] = d
        gone = [n for n in d.removed if str((old.objects.get(n) or {}).get("key", ("",))[0]) in ("g", "m")]
        if gone:
            lost[name] = gone
    return deltas, lost


def _delta_line(deltas: dict[str, ObjectDelta]) -> str:
    if not deltas:
        return ""
    if len(deltas) == 1:
        return next(iter(deltas.values())).line(max_names=6)
    return " ".join(f"[{name}] {d.line(max_names=4)}" for name, d in deltas.items())


def run_script(code: str, timeout: float = 90.0, *,
               should_stop: "Callable[[], bool] | None" = None) -> dict[str, Any]:
    """Execute ``code`` on the GUI thread, ATOMICALLY; returns success + captured stdout / stderr.

    The script runs in a fresh namespace (``new_namespace``) inside a transaction on every open document
    (``UndoMode`` is forced to 1 for the run and restored). On any exception (including ``sys.exit`` with a non-zero
    status) every change is rolled back: created objects disappear (so a retry does not produce ``Shell001``),
    deleted ones and edited values come back, documents the script created are closed, and the error names the
    failing script line, the output printed before it, a hint for known traps and what the rollback did. On success
    documents left with pending changes are recomputed and the result reports an object delta (``deltas``,
    ``lost_geometry`` for geometry that disappeared) plus pre-flight warnings. A timeout / Stop (``should_stop``,
    default: the bound tool context's) returns the dispatcher's result; a script that has not started by then never
    starts."""
    buffer = io.StringIO()
    if should_stop is None:
        should_stop = current_context().should_stop
    warnings = preflight_warnings(code)
    info: dict[str, Any] = {}      # filled by the GUI-thread task: deltas, lost geometry, rollback report

    def task():
        filename = _register_script(code)
        try:
            compiled = compile(code, filename, "exec")
        except (SyntaxError, ValueError) as e:
            return {"success": False, "error": _script_error(code, e, filename)}
        docs_before = set(FreeCAD.listDocuments())
        active = FreeCAD.ActiveDocument
        active_before = active.Name if active is not None else None
        snaps = _snapshot_docs(_live_docs(_MAX_DOCS))
        txns = _open_transactions("Oscar: execute_code")
        error = None
        early_exit = False
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer), netguard.script_guard():
                exec(compiled, new_namespace())  # noqa: S102 - the whole point of the tool
            # the recompute runs the execute() of FeaturePython objects the script defined: script code, so guarded
            with netguard.script_guard():
                _recompute_touched()
        except SystemExit as e:
            # sys.exit()/exit()/quit() must never unwind into the GUI-thread slot (it would take the application
            # down): a non-zero exit fails the (rolled back) step, a clean one just ends the script early.
            if e.code not in (None, 0):
                error = f"SystemExit: the script called sys.exit({e.code!r}); a script must not exit, it just ends"
            else:
                early_exit = True
                try:
                    with netguard.script_guard():
                        _recompute_touched()
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001 - reported to the model
            error = _script_error(code, e, filename)
        if error:
            with netguard.script_guard():     # the rollback recomputes the script's objects too
                report = _rollback(txns, docs_before, active_before)
            info["rollback"] = report
            return {"success": False, "error": error + "\n" + _rollback_text(report)}
        _close_transactions(txns, commit_changes=True)
        info["deltas"], info["lost"] = _deltas(snaps, docs_before)
        return {"success": True, "early_exit": early_exit}

    result, failed = dispatch.dispatch(task, timeout=timeout, op="execute_code", should_stop=should_stop)
    output = _clip(buffer.getvalue())
    if isinstance(result, dict) and result.get("success"):
        deltas = info.get("deltas", {})
        message = "Script executed.\nOutput: " + output.rstrip()
        if result.get("early_exit"):
            message += "\n(the script called exit(); treated as finishing early)"
        line = _delta_line(deltas)
        if line:
            message += "\nObjects: " + line
        if warnings:
            message += "\nWARNING (pre-flight): " + "\nWARNING (pre-flight): ".join(warnings)
        out: dict[str, Any] = {"success": True, "message": message,
                               "deltas": {k: v.to_dict() for k, v in deltas.items()}}
        if info.get("lost"):
            out["lost_geometry"] = info["lost"]
        if warnings:
            out["warnings"] = warnings
        return out
    if isinstance(result, dict) and result.get("error"):
        error = result["error"]
    else:
        error = str(result)
    if output.strip():
        error += "\nOutput printed before the error:\n" + output
    if warnings:
        error += "\nWARNING (pre-flight): " + "\nWARNING (pre-flight): ".join(warnings)
    out = {"success": False, "error": error}
    for key in ("timed_out", "still_running", "stopped"):
        if isinstance(result, dict) and key in result:
            out[key] = result[key]
    if "rollback" in info:
        out["rolled_back"] = info["rollback"]
    return out


# Direct document/GUI access from a background thread wedges the app; these
# patterns in async code are almost always a mistake (writes belong in a
# commit(fn) callback, which runs on the GUI thread).
_ASYNC_DANGER_PATTERNS = (
    "ActiveDocument", "FreeCADGui", "newDocument", "openDocument",
    "closeDocument", "Gui.getDocument", "Gui.activeDocument",
)


def async_code_warnings(code: str) -> list[str]:
    """Patterns in async code that indicate direct document/GUI access."""
    return [pattern for pattern in _ASYNC_DANGER_PATTERNS if pattern in code]


_JOB_FILENAME = "<execute_code_async>"


def start_job(code: str) -> dict[str, Any]:
    """Start ``code`` on a worker thread; returns the job id immediately. The job gets a fresh namespace
    (``new_namespace``) with the ``commit(fn)`` helper."""
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    _record(job_id, state="running", started=time.time(),
            code=code if len(code) <= 200 else code[:200] + "…")

    def worker():
        _in_worker.active = True
        outcome: dict[str, Any] = {"state": "done"}
        try:
            with netguard.script_guard():
                exec(compile(code, _JOB_FILENAME, "exec"), new_namespace())  # noqa: S102 - the whole point of the tool
        except BaseException as e:  # SystemExit must still finish the record
            outcome = {
                "state": "failed",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc().rstrip(),
            }
        finally:
            del _in_worker.active
            _record(job_id, finished=time.time(), **outcome)
            FreeCAD.Console.PrintMessage(
                f"Oscar: background job {job_id} {outcome['state']}\n"
            )

    try:
        threading.Thread(target=worker, daemon=True, name=f"oscar-job-{job_id}").start()
    except Exception as e:
        _record(job_id, finished=time.time(), state="failed",
                error=f"{type(e).__name__}: {e}")
        return {"success": False, "job_id": job_id, "error": str(e)}
    message = (
        "Job STARTED in the background — it is NOT complete yet. Build "
        "shapes only; apply every document/view write through commit(fn). "
        "Poll get_async_status(job_id) until state is done/failed BEFORE "
        "reporting results."
    )
    warnings = async_code_warnings(code)
    if warnings:
        message += (
            " WARNING: this code appears to touch documents/GUI directly "
            f"({', '.join(warnings[:4])}) from the background thread — that "
            "wedges the application. Use execute_code for document work."
        )
    return {"success": True, "job_id": job_id, "message": message}


def job_status(job_id: str = "") -> dict[str, Any]:
    """State of one background job (``{"success": True, "job": {...}}``, or an error for an unknown id) or, with no id,
    of every remembered job (``{"success": True, "jobs": [...]}``). Never blocks."""
    with _JOBS_LOCK:
        if job_id:
            job = JOBS.get(job_id)
            if job is None:
                return {"success": False, "error": f"unknown job: {job_id}"}
            return {"success": True, "job": dict(job)}
        return {"success": True, "jobs": [dict(j) for j in JOBS.values()]}
