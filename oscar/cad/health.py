"""Model health: "is the model in a sound state, and what changed?".

This is the public facade. The types and renderings live in ``health_types`` (pure Python), the FreeCAD
evaluation in ``health_eval``; import everything from HERE.

Two groups of functions, and the difference matters:

- GUI-THREAD ONLY (they touch ``App.Document`` objects and shapes): ``is_geometry``, ``global_shape``,
  ``snapshot_state``, ``model_health``, ``read_param_table``. Call them only inside ``dispatch.dispatch`` tasks.
- ANY-THREAD wrappers that dispatch to the GUI thread themselves: ``snapshot_via_dispatch``,
  ``snapshot_all_via_dispatch``, ``health_via_dispatch``, ``param_table_via_dispatch``. The agent loop, the tool
  handlers, ``run_tool`` and the Project call ONLY these. They never raise: a dispatch timeout, a Stop or a missing
  document gives ``None`` (``{}`` / an empty table for the dict-returning ones) and logs ``health_error``. In
  console runs dispatch is inline, so they work in tests. When a tool context is bound they pass its
  ``should_stop`` to the dispatcher, so Stop abandons a queued health check instead of waiting for it.
"""

from typing import Any

import FreeCAD as App

from ..toolctx import current_context
from . import dispatch as _dispatch
from .health_eval import (  # noqa: F401  (re-exported public API)
    DEFAULT_HEALTH_BUDGET_S,
    SNAPSHOT_BUDGET_S,
    clear_cache,
    default_budget_s,
    global_shape,
    is_geometry,
    model_health,
    read_param_table,
    snapshot_state,
)
from .health_types import (  # noqa: F401  (re-exported public API)
    FINDING_CODES,
    SEVERITIES,
    Finding,
    HealthReport,
    ObjectDelta,
    ObjRow,
    StateSnapshot,
    delta_between,
)

__all__ = [
    "FINDING_CODES", "SEVERITIES", "Finding", "ObjRow", "ObjectDelta", "StateSnapshot", "HealthReport",
    "delta_between", "is_geometry", "global_shape", "snapshot_state", "model_health", "read_param_table",
    "snapshot_via_dispatch", "snapshot_all_via_dispatch", "health_via_dispatch", "param_table_via_dispatch",
]


def _find_doc(name: str | None) -> Any:
    """The document called ``name`` (exact Name, exact Label, then a unique case-insensitive match); None = the
    active document. Raises LookupError when there is none. GUI-thread only."""
    if not name:
        doc = App.ActiveDocument
        if doc is None:
            raise LookupError("no document is open")
        return doc
    docs = App.listDocuments()
    if name in docs:
        return docs[name]
    for doc in docs.values():
        if getattr(doc, "Label", None) == name:
            return doc
    lowered = [d for d in docs.values()
               if d.Name.lower() == name.lower() or str(getattr(d, "Label", "")).lower() == name.lower()]
    if len(lowered) == 1:
        return lowered[0]
    raise LookupError(f"document '{name}' is not open")


def _log_error(op: str, doc_name: str | None, error: Any) -> None:
    try:
        from ..turnlog import log_event

        log_event("health_error", op=op, doc=doc_name or "", error=str(error)[:200])
    except Exception:
        pass


class _Missing:
    """Returned by a GUI task when the document does not exist: an expected outcome, not a crash (the dispatcher
    prints a traceback for every exception a task raises, so a missing document must not be an exception)."""

    def __init__(self, error: str):
        self.error = error


def _on_doc(doc_name: str | None, fn):
    """A GUI task that resolves ``doc_name`` and applies ``fn(doc)``; a missing document gives ``_Missing``."""

    def task():
        try:
            doc = _find_doc(doc_name)
        except LookupError as e:
            return _Missing(str(e))
        return fn(doc)

    return task


def _run(op: str, doc_name: str | None, task, timeout: float) -> Any:
    """Dispatch ``task`` to the GUI thread. Returns its value, or None on a timeout / Stop / failure / missing
    document (logged as ``health_error``)."""
    try:
        ctx = current_context()
        result, failed = _dispatch.dispatch(task, timeout=timeout, op=op, should_stop=ctx.should_stop)
    except Exception as e:  # a wrapper never raises
        _log_error(op, doc_name, e)
        return None
    if failed:
        _log_error(op, doc_name, result.get("error") if isinstance(result, dict) else result)
        return None
    if isinstance(result, _Missing):
        _log_error(op, doc_name, result.error)
        return None
    return result


def snapshot_via_dispatch(doc_name: str | None, *, timeout: float = 30.0) -> StateSnapshot | None:
    """ANY-THREAD: ``snapshot_state`` of ``doc_name`` (None = the active document). None on timeout / Stop / no such document."""
    return _run("health_snapshot", doc_name, _on_doc(doc_name, snapshot_state), timeout)


def snapshot_all_via_dispatch(max_docs: int = 5, *, timeout: float = 30.0) -> dict[str, StateSnapshot]:
    """ANY-THREAD: ``{document name: StateSnapshot}`` for up to ``max_docs`` open documents, the active one first.
    ``{}`` on timeout / Stop; a document that fails to snapshot is left out (the others are still returned)."""

    def task() -> dict[str, StateSnapshot]:
        docs = list(App.listDocuments().values())
        active = App.ActiveDocument
        docs.sort(key=lambda d: 0 if (active is not None and d.Name == active.Name) else 1)
        out: dict[str, StateSnapshot] = {}
        for doc in docs[:max(0, int(max_docs))]:
            try:
                out[doc.Name] = snapshot_state(doc)
            except Exception as e:
                _log_error("health_snapshot", doc.Name, e)
        return out

    result = _run("health_snapshot_all", None, task, timeout)
    return result if isinstance(result, dict) else {}


def health_via_dispatch(doc_name: str | None, prev: StateSnapshot | None = None, *, deep: bool = False,
                        timeout: float = 60.0, strict: bool = False) -> HealthReport | None:
    """ANY-THREAD: ``model_health`` of ``doc_name`` (None = the active document at dispatch time) against the
    snapshot ``prev`` taken before the call; non-strict (the per-call mode) unless ``strict`` is set
    (validate_model / run_validation). None on timeout / Stop / no such document."""
    return _run("model_health", doc_name,
                _on_doc(doc_name, lambda doc: model_health(doc, prev=prev, deep=deep, strict=strict)), timeout)


def param_table_via_dispatch(doc_name: str | None, *, timeout: float = 20.0) -> dict:
    """ANY-THREAD: ``read_param_table`` of ``doc_name``; ``{"sheet": "", "rows": []}`` on timeout / Stop / failure."""
    result = _run("param_table", doc_name, _on_doc(doc_name, read_param_table), timeout)
    return result if isinstance(result, dict) else {"sheet": "", "rows": []}
