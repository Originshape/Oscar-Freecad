"""The per-chat Project: the dossier the APP writes from tool events.

A Project is the single place where "every step is a real artifact and can be rebuilt" is made true. The
model never writes into it directly (it uses tools that call the Project). One chat has one Project, a
folder under ``paths.projects_dir()``::

    project.json  steps.jsonl  ledger.json  params.json  provenance.json  notes.json  spec.json
    validation.json  NOTES.md  rebuild.py  model.FCStd  docs/  scripts/  snapshots/  checkpoints/
    exports/  refs/  ledgers/

PURE module: importable with no FreeCAD / Qt (FreeCAD is touched only inside the GUI-thread tasks of the
methods that take a document name, through ``oscar.project_gui`` and ``oscar.cad.dispatch``). Private
helpers live in ``project_util`` (errors, records, atomic writes), ``project_gui`` (GUI-thread tasks; the
geometry predicates and the parameter-sheet reader are those of ``oscar.cad.health``), ``project_docs``
(NOTES / Build Summary / rebuild.py text), ``project_summary`` (the mixin that generates and checks those
files), ``project_validation`` (the ``validation.json`` record: normalising a ``ValidationResult`` and merging
partial records), ``project_checkpoints`` (the keep rules of ``checkpoints/``), ``project_replay`` and
``project_export``.

Threading rules (binding):

* ``Project._lock`` guards counters, the manifest, ``steps.jsonl`` appends and every generated-file
  rewrite. It is NEVER held while waiting on the GUI thread: GUI work first, with no lock, then the lock
  only to publish.
* Every ``read_*``, ``artifact_list``, ``last_step_index``, ``last_geometry_step`` and ``reference_status``
  is lock-free (it parses files or reads immutable values), so a GUI-thread caller can never wait on a
  worker that is waiting on the GUI thread.
* No method raises into the agent loop except ``ProjectError`` for invalid arguments; the tool hook
  wraps everything else, logs ``project_error`` and still returns a record.
"""

import contextvars
import copy
import functools
import json
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from typing import Any, Callable, Iterable
from . import fsretry as _fsretry

from . import paths
from . import project_checkpoints as ckpts
from . import project_docs as _docs
from . import project_gui as gui
from . import project_validation as vrec
from .project_export import export_tips, safe_stem, unique_path
from .project_replay import ReplayResult, replay_steps, replay_summary, steps_from_literal  # noqa: F401 (public re-exports)
from .project_rewind import RewindMixin
from .project_summary import SummaryMixin
from .project_util import (AUTO_EXPORT_INTENTS, BuildSummary, ProjectError, ProjectStorageError, RestoreResult,  # noqa: F401
                           StepRecord, _NOT_REPLAYED, _Saved, _app_label, _as_dict, _atomic_write_bytes,
                           _atomic_write_json, _atomic_write_text, _clip_str, _clip_title, _copy_file_atomic,
                           _fmt_index, _json_safe, _lint_gui_only, _read_json, _sha256_file, _ts)
from .toolctx import current_context
from .turnlog import log_event

REFERENCE_STATUSES = ("none", "catalog-nominal", "step-imported", "step-measured", "user-attached")
PROVENANCE_STATUSES = ("obtained", "declined", "failed", "nominal", "attached")
CONFIDENCE_LEVELS = ("measured", "datasheet", "community", "approximate", "assumed")
NOTE_KINDS = ("plan", "assumption", "provenance", "validation", "unresolved", "decision")
PROVENANCE_KINDS = ("reference", "datasheet", "download", "attachment", "measurement", "assumption")
DETECTED_TYPES = ("step", "stl", "zip", "pdf", "brep", "iges", "none")
VALIDATION_CATALOG = _docs.VALIDATION_CATALOG

# Tools whose successful calls can be replayed as tool calls.
REPLAYABLE = frozenset({
    "create_document", "create_object", "edit_object", "delete_object", "import_file", "insert_part_from_library",
    "apply_feature_plan", "reapply_plan_feature", "build_enclosure", "add_cutout", "add_vents", "set_parameters",
    "set_parameter", "bind_expression", "import_reference_part", "measure_reference_part", "enclosure_spec",
    "reload_document"})
SCRIPT_TOOLS = ("execute_code", "execute_code_headless")
# Non-geometry tools that can still change the Params sheet (everything geometry syncs it too).
_PARAM_AFFECTING = frozenset({"measure_reference_part", "enclosure_spec", "reload_document", "import_reference_part"})

_RESULT_HEAD = 600
_NOTE_LIMIT = 2000
_PENDING_CAP = 500
_PROVENANCE_FIELDS = {"kind", "claim", "source", "url", "local_path", "sha256", "bytes", "detected_type",
                      "status", "confidence", "reference_status", "note"}
_rnd = round   # a hook parameter is called `round`, which shadows the builtin


def _lifecycle(where: str, default: Any = None):
    """Decorator of the methods the agent loop calls directly (``begin_turn``, ``end_turn``, ``save_ledger`` ...):
    a storage failure (``OSError``, ``ProjectStorageError``) or any unexpected exception is logged as
    ``project_error`` and ``default`` is returned; only ``ProjectError`` for an invalid ARGUMENT propagates."""

    def decorate(fn):
        @functools.wraps(fn)
        def inner(self, *args, **kwargs):
            try:
                return fn(self, *args, **kwargs)
            except ProjectStorageError as exc:
                log_event("project_error", where=where, error=f"{type(exc).__name__}: {exc}"[:200])
                return default
            except ProjectError:
                raise
            except Exception as exc:  # noqa: BLE001 - the agent turn must never die because the dossier failed
                log_event("project_error", where=where, error=f"{type(exc).__name__}: {exc}"[:200])
                return default

        return inner

    return decorate


class _Budget:
    """The time one tool hook may still spend on the GUI thread (``Project.HOOK_BUDGET_S`` in total)."""

    def __init__(self, seconds: float):
        self._deadline = time.monotonic() + seconds

    def remaining(self) -> float:
        """Seconds left (negative once the budget is spent)."""
        return self._deadline - time.monotonic()


_BUDGET: "contextvars.ContextVar[_Budget | None]" = contextvars.ContextVar("oscar_project_budget", default=None)


# ---------------------------------------------------------------------------
# small helpers


def _skip_note_of(error: str) -> str:
    """The ``snapshot_note`` / ``checkpoint_note`` text for a failed GUI call."""
    if "did not complete within" in error:
        return "skipped (GUI busy)"
    if error.strip() == "stopped":
        return "skipped (stopped)"
    return f"failed: {error}"[:200]


def _timeout_for(own: float) -> float:
    """``own`` bounded by what is left of the running hook's shared GUI budget."""
    budget = _BUDGET.get()
    return own if budget is None else max(0.0, min(own, budget.remaining()))


# ---------------------------------------------------------------------------
# the Project

class Project(SummaryMixin, RewindMixin):
    """The per-chat dossier (see the module docstring)."""

    SCHEMA = 1
    SNAPSHOT_PX = (480, 360)
    MAX_CHECKPOINTS = 60
    CHECKPOINT_MAX_BYTES = 64 * 1024 * 1024       # per file; above it the checkpoint is skipped with a visible checkpoint_note
    CHECKPOINT_BUDGET_BYTES = 600 * 1024 * 1024   # per project; oldest-first pruning keeps the total under it
    HOOK_BUDGET_S = 20.0                          # total GUI time one on_tool_result may spend
    TURN_CHECKPOINTS_KEEP = 10                    # newest turn checkpoints kept
    GEOMETRY_KEEP_NEWEST = 20                     # newest geometry checkpoints never pruned
    GEOMETRY_KEEP_EVERY = 5                       # every 5th geometry checkpoint is never pruned (index % 5 == 0)
    AUTOSAVE_TIMEOUT_S = 20.0
    TURN_CKPT_RETRIES = 3                         # transient capture failures per (turn, document) before giving up for the turn
    SNAPSHOT_TIMEOUT_S = 15.0
    PARAMS_TIMEOUT_S = 10.0
    RESTORE_TIMEOUT_S = 120.0

    def __init__(self, root: str, *, chat_id: str = "", title: str = "",
                 clock: Callable[[], float] = time.time,
                 snapshot_fn: Callable[[str, str, int, int], tuple[bool, str]] | None = None,
                 create: bool = False):
        """A Project for the folder ``root``. Nothing is written until ``ensure_dir()`` (the first write method)
        unless ``create=True``; an existing ``project.json`` is read. ``snapshot_fn(doc_name, abs_path, w, h)``
        runs on the GUI thread and returns ``(ok, note)`` (default: ``project_gui.capture_snapshot``);
        ``clock`` returns epoch seconds (injectable for tests)."""
        self.root = os.path.abspath(root)
        self._clock = clock
        self._snapshot_fn = snapshot_fn or gui.capture_snapshot
        self._lock = threading.RLock()
        self._export_lock = threading.Lock()          # tiny: only the in-memory export-name reservation
        self._reserved_exports: set[str] = set()
        self._pending: list[dict] = []                # journal lines buffered until the directory exists
        self._materialized = False
        self._tail_checked = False
        self._max_index = 0
        self._scanned = False
        self._jcache: tuple | None = None             # (signature, steps, requests) of the parsed journal
        self._turn_id = ""
        self._turn_ckpts: dict[tuple[str, str], str] = {}
        self._turn_ckpt_fails: dict[tuple[str, str], int] = {}   # transient capture failures per (turn, doc)
        self.manifest: dict = self._default_manifest(chat_id, title)
        existing = _read_json(self.path("project.json"), None)
        if isinstance(existing, dict):
            self._adopt_manifest(existing)
            self._materialized = True
        if create:
            self.ensure_dir()

    # -- construction / binding ------------------------------------------------------------------

    @classmethod
    def for_chat(cls, chat_id: str, title: str = "", *, projects_root: str | None = None,
                 snapshot_fn=None, clock=time.time, create: bool = False) -> "Project":
        """The Project of one chat. Idempotent.

        The folder is ``<projects_root or paths.projects_dir()>/<paths.project_dirname(chat_id, title)>``; when
        that folder does not exist but the root already holds a project of the SAME chat under another title
        (the chat was renamed) that folder is reused, so one chat never splits into two projects. An existing
        ``project.json`` is re-read (a restored chat). With ``create=False`` (the default) NOTHING is written
        and no disk probe happens: the directory and project.json are materialized lazily by ``ensure_dir()``
        on the first write (the first mutating tool call, or ``begin_turn(..., work=True)``), on the worker
        thread; a pure Q&A chat never creates a folder.
        """
        base = projects_root or paths.projects_dir(probe=False)
        root = os.path.join(base, paths.project_dirname(chat_id, title))
        if not os.path.isdir(root):
            tail = paths.project_dirname(chat_id, "").rsplit("-", 1)[-1]
            # a chat whose project had to move to the fallback folder (unwritable ~/Documents) is found there too
            bases = [base] + ([paths.fallback_projects_dir()] if not projects_root else [])
            for where in bases:
                dirname = paths.project_dirname(chat_id, title)
                if where is not base and os.path.isfile(os.path.join(where, dirname, "project.json")):
                    root = os.path.join(where, dirname)
                    break
                found_root = ""
                try:
                    for name in sorted(os.listdir(where)):
                        candidate = os.path.join(where, name)
                        if name.endswith("-" + tail):
                            found = _read_json(os.path.join(candidate, "project.json"), None)
                            if isinstance(found, dict) and found.get("chat_id") == chat_id:
                                found_root = candidate
                                break
                except OSError:
                    pass
                if found_root:
                    root = found_root
                    break
        return cls(root, chat_id=chat_id, title=title, snapshot_fn=snapshot_fn, clock=clock, create=create)

    @classmethod
    def open(cls, root: str, **kw) -> "Project":
        """The Project in ``root``; raises ``ProjectError`` when the folder has no ``project.json``."""
        if not os.path.isfile(os.path.join(root, "project.json")):
            raise ProjectError(f"no project in '{root}': project.json is missing; open a project folder "
                               "created by Oscar.")
        return cls(root, **kw)

    def _default_manifest(self, chat_id: str, title: str) -> dict:
        now = self._now()
        return {"schema": self.SCHEMA, "id": os.path.basename(self.root), "chat_id": chat_id, "title": title,
                "created": now, "updated": now, "app": _app_label(), "active_doc": "", "docs": [],
                "counters": {"step": 0, "export": 0}, "storage_note": "",
                "reference": {"id": "", "name": "", "status": "none", "entry": ""},
                "last_geometry_step": 0, "finalized": None}

    def _adopt_manifest(self, loaded: dict) -> None:
        manifest = self._default_manifest(loaded.get("chat_id", ""), loaded.get("title", ""))
        manifest.update(loaded)
        counters = {"step": 0, "export": 0}
        counters.update(loaded.get("counters") or {})
        manifest["counters"] = counters
        manifest["docs"] = list(loaded.get("docs") or [])
        manifest["reference"] = {**manifest["reference"], **(loaded.get("reference") or {})}
        self.manifest = manifest

    def _now(self) -> str:
        return _ts(self._clock())

    def path(self, *parts: str) -> str:
        """Absolute path under the project root."""
        return os.path.join(self.root, *parts)

    def rel(self, abs_path: str) -> str:
        """``abs_path`` relative to the project root with ``/`` separators; ``ProjectError`` outside the root."""
        root = os.path.abspath(self.root)
        target = os.path.abspath(abs_path)
        try:
            inside = os.path.commonpath([root, target]) == root
        except ValueError:
            inside = False
        if not inside:
            raise ProjectError(f"'{abs_path}' is outside the project folder '{self.root}'.")
        return os.path.relpath(target, root).replace(os.sep, "/")

    # -- directory materialization ----------------------------------------------------------------

    def ensure_dir(self) -> None:
        """Materialize the project folder + project.json (+ the buffered turn_start line) once; idempotent.

        Every write method calls it first. It runs on the worker thread (never at send time on the GUI
        thread). On an OSError (macOS TCC denial of ~/Documents, a read-only or offline synced folder) the
        project is re-routed to ``paths.fallback_projects_dir()``, ``paths.PROJECTS_FALLBACK`` is set and
        ``manifest["storage_note"]`` says so (the panel shows it); a fallback that fails too raises ``ProjectError``.
        """
        if self._materialized:
            return
        with self._lock:
            if self._materialized:
                return
            try:
                self._materialize()
            except OSError as exc:
                self._reroute(exc)

    def _materialize(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        if not os.path.isfile(self.path("project.json")):
            _atomic_write_json(self.path("project.json"), self.manifest)
        while self._pending:
            self._append_raw(self._pending[0])
            self._pending.pop(0)
        self._materialized = True

    def _reroute(self, cause: OSError) -> None:
        fallback = paths.fallback_projects_dir()
        new_root = os.path.join(fallback, os.path.basename(self.root))
        old = self.root
        try:
            self.root = new_root
            self._tail_checked = False
            existing = _read_json(self.path("project.json"), None)
            if isinstance(existing, dict):
                # the project already lives in the fallback folder (a restored chat): keep its counters, documents,
                # reference and finalized record instead of overwriting them with a fresh manifest
                self._adopt_manifest(existing)
            self._materialize()
        except OSError as exc2:
            self.root = old
            raise ProjectStorageError(f"cannot create the project folder '{old}' ({cause}) or the fallback "
                                      f"'{new_root}' ({exc2}); set OSCAR_PROJECTS_DIR to a writable folder.") from exc2
        paths.PROJECTS_FALLBACK = True
        self.manifest["storage_note"] = (f"The projects folder '{os.path.dirname(old)}' is not writable "
                                         f"({type(cause).__name__}: {cause}); this project is stored in '{fallback}'.")
        self._save_manifest()
        log_event("project_storage_fallback", root=new_root)

    @property
    def storage_note(self) -> str:
        """Non-empty when the project had to move to the fallback folder (show it to the user once)."""
        return str(self.manifest.get("storage_note") or "")

    def _save_manifest(self) -> None:
        """Write project.json (caller holds the lock; the directory exists)."""
        self.manifest["updated"] = self._now()
        _atomic_write_json(self.path("project.json"), self.manifest)

    # -- the journal (steps.jsonl) ----------------------------------------------------------------

    def _append_raw(self, obj: dict) -> None:
        """Append ONE complete line with ONE ``os.write`` on an ``O_APPEND`` descriptor (caller holds the lock).

        On the first append after the Project opens, a file that does not end in a newline (a crash left a
        torn last line) gets a ``\\n`` first, so one corrupt record never swallows the next good one."""
        try:
            payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
            data = (payload + "\n").encode("utf-8")
        except UnicodeEncodeError:   # lone surrogates in model text
            data = (json.dumps(obj, ensure_ascii=True, separators=(",", ":"), default=str) + "\n").encode("utf-8")
        path = self.path("steps.jsonl")
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            if not self._tail_checked:
                self._tail_checked = True
                size = os.fstat(fd).st_size
                if size > 0:
                    with open(path, "rb") as probe:
                        probe.seek(size - 1)
                        if probe.read(1) != b"\n":
                            data = b"\n" + data
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
        finally:
            os.close(fd)

    def _journal_line(self, obj: dict) -> None:
        """Append a journal line, or buffer it while the directory does not exist yet (caller holds the lock)."""
        if self._materialized:
            self._append_raw(obj)
        else:
            self._pending.append(obj)
            del self._pending[:-_PENDING_CAP]

    def _journal(self) -> tuple[list[StepRecord], list[dict]]:
        """``(step records incl. superseded, turn_start lines)`` of steps.jsonl (lock-free; cached by file signature).

        Lines that fail ``json.loads`` are skipped; ``superseded`` is True for every step whose index is greater
        than the ``to_step`` of a LATER rewind line."""
        path = self.path("steps.jsonl")
        try:
            st = os.stat(path)
        except OSError:
            return [], []
        signature = (st.st_size, st.st_mtime_ns)
        cached = self._jcache
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        try:
            with open(path, "rb") as f:
                raw = f.read().decode("utf-8", errors="replace")
        except OSError:
            return [], []
        steps: list[StepRecord] = []
        starts: list[dict] = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            kind = obj.get("type")
            if kind == "step":
                try:
                    steps.append(StepRecord.from_dict(obj))
                except (TypeError, ValueError):
                    continue
            elif kind == "rewind":
                try:
                    to_step = int(obj.get("to_step", 0))
                except (TypeError, ValueError):
                    continue
                for s in steps:
                    if s.index > to_step:
                        s.superseded = True
            elif kind == "turn_start":
                starts.append(obj)
        self._jcache = (signature, steps, starts)
        return steps, starts

    def read_steps(self, include_superseded: bool = False) -> list[StepRecord]:
        """The step records in file order (lock-free; shallow copies, treat as read-only).

        Superseded steps (rewound past) are excluded unless ``include_superseded``."""
        steps, _ = self._journal()
        return [copy.copy(s) for s in steps if include_superseded or not s.superseded]

    def last_step_index(self) -> int:
        """The newest recorded step index (superseded ones included, indexes are never reused); 0 when none."""
        steps, _ = self._journal()
        return max((s.index for s in steps), default=0)

    def last_geometry_step(self) -> int:
        """Index of the newest NON-superseded successful geometry step; 0 when there is none."""
        steps, _ = self._journal()
        return max((s.index for s in steps if s.ok and s.geometry and not s.superseded), default=0)

    def _reserve_index(self) -> int:
        """Next step index = ``max(counters.step, last index in steps.jsonl) + 1`` (caller holds the lock): a crash
        between a journal append and the manifest write must never reuse an index."""
        if not self._scanned:
            self._max_index = max((s.index for s in self._journal()[0]), default=0)
            self._scanned = True
        index = max(int(self.manifest["counters"].get("step", 0)), self._max_index) + 1
        self.manifest["counters"]["step"] = index
        return index

    # -- reading ------------------------------------------------------------------------------------

    def read_ledger(self) -> dict | None:
        """``ledger.json`` as a dict, or None (lock-free)."""
        data = _read_json(self.path("ledger.json"), None)
        return data if isinstance(data, dict) else None

    def read_params(self) -> dict:
        """``params.json`` (``{"schema", "doc", "sheet", "step", "updated", "params": [...]}``) or ``{"params": []}``."""
        data = _read_json(self.path("params.json"), None)
        if not isinstance(data, dict):
            return {"params": []}
        data.setdefault("params", [])
        return data

    def read_provenance(self) -> dict:
        """``provenance.json`` or the empty skeleton ``{"schema": 1, "reference": {...}, "entries": []}``."""
        data = _read_json(self.path("provenance.json"), None)
        if not isinstance(data, dict):
            return {"schema": 1, "reference": {"id": "", "name": "", "status": "none", "entry": ""}, "entries": []}
        data.setdefault("reference", {"id": "", "name": "", "status": "none", "entry": ""})
        data.setdefault("entries", [])
        return data

    def read_notes(self) -> list[dict]:
        """The ``record_note`` entries of ``notes.json`` (``[]`` when there are none)."""
        data = _read_json(self.path("notes.json"), None)
        entries = data.get("entries") if isinstance(data, dict) else None
        return list(entries) if isinstance(entries, list) else []

    def read_spec(self) -> dict | None:
        """``spec.json`` (the EnclosureSpec dict) or None."""
        data = _read_json(self.path("spec.json"), None)
        return data if isinstance(data, dict) else None

    def read_validation(self) -> dict | None:
        """The latest ``validation.json`` or None."""
        data = _read_json(self.path("validation.json"), None)
        return data if isinstance(data, dict) else None

    @property
    def reference_status(self) -> str:
        """One of ``REFERENCE_STATUSES`` (``none`` when unset)."""
        status = (self.manifest.get("reference") or {}).get("status", "none")
        return status if status in REFERENCE_STATUSES else "none"

    def artifact_list(self) -> list[dict]:
        """The file entries of the ``artifacts_changed`` signal: ``{"kind", "name", "path"}`` with ``kind`` one of
        ``fcstd`` (model.FCStd, docs/*.FCStd), ``notes``, ``rebuild``, ``provenance``, ``export`` (exports/*) and
        ``snapshot`` (the newest six); ``path`` is absolute. ``*.FCBak`` and temp files are ignored; ``[]`` when
        the project is not materialized. Lock-free."""
        if not os.path.isdir(self.root):
            return []
        out: list[dict] = []

        def add(kind: str, rel: str) -> None:
            if os.path.isfile(self.path(rel)):
                out.append({"kind": kind, "name": rel, "path": self.path(rel)})

        add("fcstd", "model.FCStd")
        for name in self._listdir("docs"):
            if name.endswith(".FCStd"):
                add("fcstd", f"docs/{name}")
        add("notes", "NOTES.md")
        add("rebuild", "rebuild.py")
        add("provenance", "provenance.json")
        for name in self._listdir("exports"):
            add("export", f"exports/{name}")
        for name in self._listdir("snapshots")[-6:]:
            add("snapshot", f"snapshots/{name}")
        return out

    def _listdir(self, sub: str) -> list[str]:
        """Sorted visible file names of a project sub folder (no dot files, ``.tmp-`` files or ``*.FCBak``)."""
        try:
            names = os.listdir(self.path(sub))
        except OSError:
            return []
        return sorted(n for n in names
                      if not n.startswith(".") and ".tmp-" not in n and not n.endswith(".FCBak"))

    # -- turn lifecycle (called by AgentWorker) ---------------------------------------------------------

    @_lifecycle("begin_turn")
    def begin_turn(self, turn_id: str, request: str, *, doc_names: Iterable[str] = (), work: bool = False) -> None:
        """Start a turn: appends ``{"type": "turn_start", ...}`` (buffered in memory until the directory exists).

        ``work=True`` (the agent passes ``ledger.is_work_turn()``) materializes the directory now. It takes NO
        checkpoint: the turn checkpoint is created just in time by ``ensure_turn_checkpoint``, only for the
        documents the turn actually changes."""
        line = {"type": "turn_start", "turn_id": turn_id, "ts": self._now(),
                "request": _clip_str(request or "", 20_000), "docs": [str(d) for d in doc_names]}
        with self._lock:
            self._turn_id = turn_id
            self._journal_line(line)
        if work:
            self.ensure_dir()

    @_lifecycle("end_turn")
    def end_turn(self, turn_id: str, outcome: str, *, rounds: int = 0) -> None:
        """End a turn: appends ``{"type": "turn_end", ...}`` and prunes turn checkpoints beyond the newest 10."""
        line = {"type": "turn_end", "turn_id": turn_id, "ts": self._now(), "outcome": outcome, "rounds": int(rounds)}
        with self._lock:
            self._journal_line(line)
            if self._materialized:
                self._prune_checkpoints()

    # -- THE HOOK -----------------------------------------------------------------------------------------

    def on_tool_result(self, *, turn_id: str, round: int, call_id: str, tool: str, title: str,
                       args: dict, text: str, ok: bool, dur_s: float, doc: str, delta: dict,
                       health: dict | None, mutating: bool, geometry: bool) -> "StepRecord | None":
        """Record one tool result (called by ``tools.run_tool`` through ``ToolContext.project``).

        Returns None when ``mutating`` is False (read tools are not recorded). ``ok`` is the HANDLER result: a
        call that ran but left new health errors is still ``ok=True`` and gets autosave, checkpoint, snapshot
        and replay like any other step (``health["valid"]`` is False in its record). Order:
        reserve the index (lock), capture the script, do ALL GUI-thread work with no lock held (autosave,
        checkpoint copy, snapshot, params sync; skipped after a dispatch timeout or Stop, and bounded by
        ``HOOK_BUDGET_S`` in total), build the replay entry, then publish (lock): append the journal line,
        update the manifest, regenerate NOTES.md and rebuild.py. Never raises: on an unexpected failure it logs
        ``project_error`` and returns a record whose ``snapshot_note`` says so (``index`` -1 when nothing could
        be reserved)."""
        if not mutating:
            return None
        try:
            return self._record(turn_id=turn_id, round_=round, call_id=call_id, tool=tool, title=title, args=args,
                                text=text, ok=ok, dur_s=dur_s, doc=doc, delta=delta, health=health,
                                geometry=geometry)
        except Exception as exc:  # noqa: BLE001 - the agent turn must never die because the dossier failed
            message = f"{type(exc).__name__}: {exc}"[:200]
            log_event("project_error", where="on_tool_result", tool=tool, error=message)
            return StepRecord(index=-1, ts=self._now(), turn_id=turn_id, round=int(round or 0), call_id=call_id,
                              tool=tool, title=_clip_title(title) or tool, args={}, ok=bool(ok), result_head="",
                              result_chars=0, dur_s=0.0, doc=doc or "", delta={"added": [], "removed": [], "changed": []},
                              health=None, snapshot="", snapshot_note=f"project error: {message}", checkpoint="",
                              checkpoint_note="", script="", replay={"kind": "none", "why": "project error"},
                              geometry=bool(geometry))

    def _record(self, *, turn_id: str, round_: int, call_id: str, tool: str, title: str, args: dict, text: str,
                ok: bool, dur_s: float, doc: str, delta: dict, health: dict | None, geometry: bool) -> StepRecord:
        self.ensure_dir()
        ctx = current_context()
        should_stop = ctx.should_stop
        text = text if isinstance(text, str) else str(text or "")
        args = args if isinstance(args, dict) else {}
        title = _clip_title(title) or tool
        doc = doc or ""
        if tool == "create_document" and ok:
            created = re.search(r"Document '([^']+)' created", text)
            if created:
                doc = created.group(1)
        with self._lock:
            index = self._reserve_index()
        rec = StepRecord(
            index=index, ts=self._now(), turn_id=turn_id, round=int(round_ or 0), call_id=call_id, tool=tool,
            title=title, args=_json_safe(args), ok=bool(ok), result_head=text[:_RESULT_HEAD], result_chars=len(text),
            dur_s=_rnd(float(dur_s or 0.0), 3), doc=doc,
            delta=self._norm_delta(delta), health=self._norm_health(health), snapshot="", snapshot_note="",
            checkpoint="", checkpoint_note="", script="", replay={}, geometry=bool(geometry))
        # -- script capture
        code = args.get("code")
        headless = tool == "execute_code_headless"
        gui_only = False
        if tool in SCRIPT_TOOLS and ok and isinstance(code, str) and code.strip():
            rec.script = self._write_script(rec, tool, code)
            gui_only = (not headless) and _lint_gui_only(code)
        # -- GUI work, no lock held
        budget = _Budget(self.HOOK_BUDGET_S)
        token = _BUDGET.set(budget)
        try:
            self._gui_hooks(rec, text=text, delta=delta, ok=bool(ok), geometry=bool(geometry),
                            should_stop=should_stop, budget=budget)
        finally:
            _BUDGET.reset(token)
        rec.replay = self._replay_entry(rec, tool, args, bool(ok), headless, gui_only)
        # -- publish
        with self._lock:
            resolved = rec.doc
            if resolved:
                self._note_doc(resolved)
            if geometry and ok:
                if resolved and not self.manifest.get("active_doc"):
                    self.manifest["active_doc"] = resolved
                self.manifest["last_geometry_step"] = index
            self._append_raw(rec.to_line())
            self._max_index = max(self._max_index, index)
            self._save_manifest()
            self._regenerate_docs(rebuild=True)
        return rec

    @staticmethod
    def _norm_delta(delta: Any) -> dict:
        data = _json_safe(delta) if isinstance(delta, dict) else {}
        return {"added": list(data.get("added") or []), "removed": list(data.get("removed") or []),
                "changed": list(data.get("changed") or [])}

    @staticmethod
    def _norm_health(health: Any) -> dict | None:
        if not isinstance(health, dict):
            return None
        data = _json_safe(health)
        if isinstance(data.get("findings"), list):
            data["findings"] = data["findings"][:5]
        return data

    def _note_doc(self, name: str) -> None:
        docs = self.manifest.setdefault("docs", [])
        if name and name not in docs:
            docs.append(name)

    def _write_script(self, rec: StepRecord, tool: str, code: str) -> str:
        rel = f"scripts/{rec.index:03d}_{paths.slugify(rec.title, 40) if rec.title else 'step'}.py"
        header = (f"# Oscar step {rec.index:03d}: {rec.title}\n"
                  f"# turn {rec.turn_id}  tool {tool}  doc {rec.doc}  {rec.ts}\n"
                  "# Run through oscar.cad.scripting.run_script (see rebuild.py) or paste into the Python console.\n")
        with self._lock:
            os.makedirs(self.path("scripts"), exist_ok=True)
            _atomic_write_text(self.path(rel), header + code + ("" if code.endswith("\n") else "\n"))
        return rel

    def _gui_hooks(self, rec: StepRecord, *, text: str, delta: Any, ok: bool, geometry: bool,
                   should_stop: Callable[[], bool], budget: _Budget) -> None:
        """Autosave, checkpoint, snapshot and params sync of one recorded call (no lock held; sets the notes)."""
        initial = ("GUI busy" if "did not complete within" in text
                   else "stopped" if should_stop() else "")

        def gate(hook: str) -> str:
            reason = initial
            if not reason and should_stop():
                reason = "stopped"
            if not reason and budget.remaining() < 0.5:
                reason = "GUI busy"
            if reason:
                log_event("hook_skipped", hook=hook, why=reason, tool=rec.tool)
            return reason

        if geometry and not ok:
            rec.snapshot_note = "step failed"
        elif geometry:
            if not rec.doc:
                rec.snapshot_note = rec.checkpoint_note = "no document"
            else:
                known = isinstance(delta, dict) and any(k in delta for k in ("added", "removed", "changed"))
                unchanged = known and not (delta.get("added") or delta.get("removed") or delta.get("changed"))
                reason = gate("autosave")
                if reason:
                    rec.checkpoint_note = f"skipped ({reason})"
                    saved = _Saved(note=rec.checkpoint_note)
                else:
                    saved = self._autosave(rec.doc, skip_if_unchanged=unchanged)
                    if saved.doc:
                        rec.doc = saved.doc
                        with self._lock:
                            if saved.rel and not self.manifest.get("active_doc"):
                                self.manifest["active_doc"] = saved.doc
                if saved.note == "unchanged":
                    rec.checkpoint_note = rec.snapshot_note = "unchanged"
                elif not saved.rel:
                    rec.checkpoint_note = saved.note or "autosave failed"
                else:
                    reason = gate("checkpoint")
                    if reason:
                        rec.checkpoint_note = f"skipped ({reason})"
                    else:
                        rec.checkpoint, rec.checkpoint_note = self._checkpoint_from_file(
                            _fmt_index(rec.index), saved, rec.index)
                if rec.snapshot_note != "unchanged":
                    reason = gate("snapshot")
                    if reason:
                        rec.snapshot_note = f"skipped ({reason})"
                    else:
                        rec.snapshot, rec.snapshot_note = self.snapshot(rec.doc, rec.title, rec.index)
        elif ok:
            rec.snapshot_note = "not a geometry tool"
        if ok and rec.doc and (geometry or rec.tool in _PARAM_AFFECTING):
            reason = gate("sync_params")
            if not reason:
                self._sync_params(rec.doc, rec.index)

    def _replay_entry(self, rec: StepRecord, tool: str, args: dict, ok: bool, headless: bool, gui_only: bool) -> dict:
        if not ok:
            return {"kind": "none", "why": "the call failed"}
        if tool in SCRIPT_TOOLS:
            if rec.script:
                return {"kind": "script", "file": rec.script, "headless": headless, "gui_only": gui_only}
            return {"kind": "none", "why": "the call carried no script text"}
        if tool in REPLAYABLE:
            return {"kind": "tool", "name": tool, "args": rec.args}
        return {"kind": "none", "why": _NOT_REPLAYED.get(tool, f"'{tool}' is not replayable")}

    # -- artifacts: autosave, checkpoints, snapshot -------------------------------------------------------

    def _autosave(self, doc_name: str, *, skip_if_unchanged: bool = False) -> _Saved:
        """``saveCopy`` of ``doc_name`` into ``model.FCStd`` (the manifest's ``active_doc``, or the first document
        saved) or ``docs/<DocName>.FCStd``; atomic, temp name ending in ``.FCStd``. With ``skip_if_unchanged`` and
        an existing file nothing is written and ``note == "unchanged"``."""
        self.ensure_dir()
        ctx = current_context()
        active = self.manifest.get("active_doc") or ""
        timeout = _timeout_for(self.AUTOSAVE_TIMEOUT_S)
        if timeout < 0.5:
            return _Saved(note="skipped (GUI busy)")

        def task() -> dict:
            doc = gui.resolve_doc(doc_name)
            if doc is None:
                return {"note": "no document"}
            rel = "model.FCStd" if (not active or doc.Name == active) else f"docs/{safe_stem(doc.Name, 'doc')}.FCStd"
            dest = self.path(rel)
            if skip_if_unchanged and os.path.isfile(dest):
                return {"note": "unchanged", "rel": rel, "abs": dest, "doc": doc.Name}
            saved_ok, error = gui.save_copy(doc, dest)
            if not saved_ok:
                return {"note": f"autosave failed: {error}", "doc": doc.Name}
            primary, every = gui.object_names(doc)
            invalid = [o.Name for o in doc.Objects if "Invalid" in (getattr(o, "State", None) or [])]
            return {"rel": rel, "abs": dest, "doc": doc.Name, "names": primary, "all": every, "invalid": invalid}

        result, error = gui.gui_call(task, timeout, "project_autosave", ctx.should_stop)
        if error:
            return _Saved(note=_skip_note_of(error))
        return _Saved(rel=result.get("rel", "") if result.get("note") in (None, "") else "",
                      abs=result.get("abs", ""), doc=result.get("doc", ""), names=result.get("names", []),
                      all_names=result.get("all", []), invalid=result.get("invalid", []),
                      note=result.get("note", ""))

    def autosave(self, doc_name: str) -> str:
        """Save ``doc_name`` now: relative path of ``model.FCStd`` (the active document) or
        ``docs/<DocName>.FCStd``, or ``""`` (the reason is logged). Runs the ``saveCopy`` on the GUI thread
        (20 s, bounded by the hook budget); the first document saved becomes the manifest's ``active_doc``."""
        if not doc_name:
            return ""
        rel, note = self.save_now(doc_name)
        if not rel:
            log_event("autosave_skipped", doc=doc_name, why=note)
        return rel

    def save_now(self, doc_name: str) -> tuple[str, str]:
        """``autosave`` that says WHY it failed: ``(relative path, "")`` on success, ``("", reason)`` otherwise
        (``no document``, ``skipped (GUI busy)``, ``autosave failed: ...``). ``doc_name == ""`` means the active
        document. The ``save_document`` tool uses it; like ``autosave`` it never raises."""
        saved = self._autosave(doc_name)
        if not saved.rel:
            return "", saved.note or "autosave failed"
        with self._lock:
            self._note_doc(saved.doc)
            if not self.manifest.get("active_doc"):
                self.manifest["active_doc"] = saved.doc
            self._save_manifest()
        return saved.rel, ""

    def _checkpoint_dir(self) -> str:
        return self.path("checkpoints")

    def _copy_into_checkpoints(self, src: str, dest: str) -> None:
        """Copy ``src`` to ``dest`` through a temp name that ends in ``.FCStd`` + ``os.replace``."""
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest[: -len(".FCStd")] + f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}.FCStd"
        try:
            shutil.copyfile(src, tmp)
            _fsretry.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _checkpoint_from_file(self, ckpt_id: str, saved: _Saved, step: int) -> tuple[str, str]:
        """Checkpoint = copy of the just-written autosave + sidecar ``<id>.json``; returns ``(id, note)``."""
        try:
            size = os.path.getsize(saved.abs)
            if size > self.CHECKPOINT_MAX_BYTES:
                return "", f"skipped: {size / (1024 * 1024):.0f} MB > per-file limit"
            dest = os.path.join(self._checkpoint_dir(), f"{ckpt_id}.FCStd")
            self._copy_into_checkpoints(saved.abs, dest)
            sidecar = {"step": step, "doc": saved.doc, "objects": saved.names, "all_objects": saved.all_names,
                       "invalid": saved.invalid, "ts": self._now(), "bytes": size}
            with self._lock:
                _atomic_write_json(os.path.join(self._checkpoint_dir(), f"{ckpt_id}.json"), sidecar)
                self._prune_checkpoints()
            return ckpt_id, ""
        except OSError as e:
            return "", f"checkpoint failed: {e}"[:200]

    def checkpoint(self, doc_name: str, label: str = "") -> str:
        """A restorable copy of ``doc_name`` now (its OWN ``saveCopy``): the checkpoint id or ``""``.

        ``label`` becomes the id (letters, digits, ``_`` and ``-``); without one the id is the newest step index
        (``007``). The sidecar ``<id>.json`` records the step, the document and its object names. Skipped
        (``""``, logged) when the file exceeds ``CHECKPOINT_MAX_BYTES``."""
        if not doc_name:
            return ""
        self.ensure_dir()
        ckpt_id = safe_stem(label) if label else _fmt_index(self.last_step_index())
        ctx = current_context()
        dest = os.path.join(self._checkpoint_dir(), f"{ckpt_id}.FCStd")
        timeout = _timeout_for(self.AUTOSAVE_TIMEOUT_S)
        if timeout < 0.5:
            return ""

        def task() -> dict:
            doc = gui.resolve_doc(doc_name)
            if doc is None:
                return {"note": "no document"}
            saved_ok, error = gui.save_copy(doc, dest)
            if not saved_ok:
                return {"note": error, "doc": doc.Name}
            primary, every = gui.object_names(doc)
            return {"doc": doc.Name, "names": primary, "all": every,
                    "invalid": [o.Name for o in doc.Objects if "Invalid" in (getattr(o, "State", None) or [])]}

        result, error = gui.gui_call(task, timeout, "project_checkpoint", ctx.should_stop)
        if error or result.get("note"):
            log_event("checkpoint_skipped", doc=doc_name, why=error or result.get("note"))
            return ""
        size = os.path.getsize(dest)
        if size > self.CHECKPOINT_MAX_BYTES:
            os.remove(dest)
            log_event("checkpoint_skipped", doc=doc_name, why="per-file limit", bytes=size)
            return ""
        sidecar = {"step": self.last_step_index(), "doc": result["doc"], "objects": result["names"],
                   "all_objects": result["all"], "invalid": result["invalid"], "ts": self._now(), "bytes": size}
        with self._lock:
            _atomic_write_json(os.path.join(self._checkpoint_dir(), f"{ckpt_id}.json"), sidecar)
            self._prune_checkpoints()
        return ckpt_id

    @_lifecycle("ensure_turn_checkpoint", "")
    def ensure_turn_checkpoint(self, doc_name: str) -> str:
        """Save the document as it is just BEFORE the turn's first geometry step: ``checkpoints/T<last 4 of the
        turn id>.FCStd`` (+ sidecar); idempotent per (turn, document); returns the id (``T9f3a``) or ``""``.

        Called by ``run_tool`` before the first geometry handler runs on ``doc_name`` (``""`` = the active
        document). Never touches documents the turn does not change: the user's unrelated open documents are
        never saved or restored. A second document of the same turn is stored as ``T<suffix>.<DocName>.FCStd``
        (the id stays ``T<suffix>``; ``restore(doc, id)`` picks the file of that document). The document is
        registered in ``manifest["docs"]``.

        A TRANSIENT failure (the GUI thread was busy and the capture timed out, the turn was stopped, the save
        failed, no document yet) is NOT cached: the next mutating call of the turn tries again, up to
        ``TURN_CKPT_RETRIES`` times, so one busy moment never silently removes the turn's rewind safety net. A
        permanent outcome (a document over ``CHECKPOINT_MAX_BYTES``) is cached for the rest of the turn."""
        ctx = current_context()
        turn = self._turn_id or ctx.turn_id
        if not turn:
            return ""
        suffix = "T" + turn[-4:]
        key = (turn, doc_name or "")
        with self._lock:
            if key in self._turn_ckpts:
                return self._turn_ckpts[key]
        self.ensure_dir()
        step = self.last_step_index()
        directory = self._checkpoint_dir()

        def task() -> dict:
            doc = gui.resolve_doc(doc_name)
            if doc is None:
                return {"note": "no document"}
            name = doc.Name
            primary = os.path.join(directory, f"{suffix}.FCStd")
            side = _read_json(os.path.join(directory, f"{suffix}.json"), {}) if os.path.isfile(primary) else {}
            dest, stem = primary, suffix
            if side and side.get("turn") == turn:
                if side.get("doc") == name:
                    return {"id": suffix, "doc": name, "existing": True}
                stem = f"{suffix}.{safe_stem(name, 'doc')}"
                dest = os.path.join(directory, f"{stem}.FCStd")
                again = _read_json(os.path.join(directory, f"{stem}.json"), {})
                if again.get("turn") == turn and again.get("doc") == name:
                    return {"id": suffix, "doc": name, "existing": True}
            saved_ok, error = gui.save_copy(doc, dest)
            if not saved_ok:
                return {"note": error, "doc": name}
            names, every = gui.object_names(doc)
            return {"id": suffix, "doc": name, "stem": stem, "dest": dest, "names": names, "all": every,
                    "invalid": [o.Name for o in doc.Objects if "Invalid" in (getattr(o, "State", None) or [])]}

        result, error = gui.gui_call(task, self.AUTOSAVE_TIMEOUT_S, "project_turn_checkpoint", ctx.should_stop)
        ckpt = ""
        transient = False
        if error:
            transient = True
            log_event("hook_skipped", hook="turn_checkpoint", why=error[:120])
        elif result.get("id"):
            ckpt = result["id"]
            with self._lock:
                self._note_doc(result["doc"])
                if not result.get("existing"):
                    size = os.path.getsize(result["dest"])
                    if size > self.CHECKPOINT_MAX_BYTES:
                        os.remove(result["dest"])
                        ckpt = ""
                    else:
                        _atomic_write_json(os.path.join(directory, f"{result['stem']}.json"), {
                            "turn": turn, "doc": result["doc"], "after_step": step, "objects": result["names"],
                            "all_objects": result["all"], "invalid": result["invalid"], "ts": self._now(),
                            "bytes": size})
                        self._prune_checkpoints()
                self._save_manifest()
        else:
            transient = True   # "no document" / the save failed: the document or the disk may be fine next time
            log_event("hook_skipped", hook="turn_checkpoint", why=str((result or {}).get("note", "no checkpoint"))[:120])
        if transient and not ckpt:
            with self._lock:
                fails = self._turn_ckpt_fails.get(key, 0) + 1
                self._turn_ckpt_fails[key] = fails
                if fails < self.TURN_CKPT_RETRIES:
                    return ""   # not cached: the next call for this (turn, document) captures again
        self._turn_ckpts[key] = ckpt
        return ckpt

    def _checkpoint_entries(self) -> list[dict]:
        """Every checkpoint of the project (``project_checkpoints.list_entries``; lock-free; ``_rescue`` and temp
        files excluded)."""
        return ckpts.list_entries(self._checkpoint_dir())

    def _prune_checkpoints(self) -> None:
        """Keep ``checkpoints/`` bounded (caller holds the lock; file deletes only): the keep rules of
        ``project_checkpoints.prune`` with this class's ``TURN_CHECKPOINTS_KEEP`` / ``GEOMETRY_KEEP_*`` /
        ``MAX_CHECKPOINTS`` / ``CHECKPOINT_BUDGET_BYTES``."""
        ckpts.prune(self._checkpoint_entries(), keep_turns=self.TURN_CHECKPOINTS_KEEP,
                    keep_newest=self.GEOMETRY_KEEP_NEWEST, keep_every=self.GEOMETRY_KEEP_EVERY,
                    max_geometry=self.MAX_CHECKPOINTS, budget_bytes=self.CHECKPOINT_BUDGET_BYTES)

    def snapshot(self, doc_name: str, title: str, index: int) -> tuple[str, str]:
        """A picture for humans of ``doc_name`` after step ``index``: ``("snapshots/007_<slug>.png", "")`` or
        ``("", note)`` with the reason (``no document``, ``empty document``, ``no 3D view (headless session)``,
        ``blank frame``, ``skipped (GUI busy)``, ...). The capture runs on the GUI thread through ``snapshot_fn``
        at ``SNAPSHOT_PX`` (15 s, bounded by the hook budget) and is written atomically."""
        if not doc_name:
            return "", "no document"
        self.ensure_dir()
        ctx = current_context()
        rel = f"snapshots/{int(index):03d}_{paths.slugify(title, 40) if title else 'step'}.png"
        final = self.path(rel)
        tmp = final[: -len(".png")] + f".tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}.png"
        width, height = self.SNAPSHOT_PX
        fn = self._snapshot_fn
        timeout = _timeout_for(self.SNAPSHOT_TIMEOUT_S)
        if timeout < 0.5:
            return "", "skipped (GUI busy)"
        os.makedirs(os.path.dirname(final), exist_ok=True)

        def task() -> tuple[str, str]:
            doc = gui.resolve_doc(doc_name)
            if doc is None:
                return "", "no document"
            if not gui.has_geometry(doc):
                return "", "empty document"
            outcome = fn(doc.Name, tmp, width, height)
            ok, note = outcome if isinstance(outcome, tuple) else (bool(outcome), "")
            if not ok:
                return "", str(note or "snapshot failed")
            if not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
                return "", "snapshot produced no image"
            _fsretry.replace(tmp, final)
            return rel, ""

        try:
            result, error = gui.gui_call(task, timeout, "project_snapshot", ctx.should_stop)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        if error:
            return "", _skip_note_of(error).replace("failed: ", "snapshot failed: ", 1)
        return result

    # -- restore ---------------------------------------------------------------------------------------------

    def _find_checkpoint(self, doc_name: str, ckpt_id: str) -> tuple[str, dict] | None:
        """``(file, sidecar)`` of checkpoint ``ckpt_id`` for ``doc_name`` (a turn checkpoint has one file per document)."""
        wanted = str(ckpt_id or "").strip()
        if wanted.isdigit():
            wanted = _fmt_index(int(wanted))
        matches = [e for e in self._checkpoint_entries() if e["id"] == wanted]
        if not matches:
            return None
        if len(matches) > 1 and doc_name:
            lowered = doc_name.lower()
            for entry in matches:
                if str(entry["side"].get("doc", "")).lower() == lowered:
                    return entry["path"], entry["side"]
        entry = matches[0]
        return entry["path"], entry["side"]

    def restore(self, doc_name: str, checkpoint_id: str) -> RestoreResult:
        """Replace the content of the LIVE document ``doc_name`` with checkpoint ``checkpoint_id`` (``007`` or
        ``T9f3a``), as ONE named undo transaction (``Oscar restore <id>``): a single Ctrl+Z reverts it.

        The whole call is ONE GUI-thread dispatch task (120 s): (1) the file exists and is a zip (checked before
        dispatching); (2) a rescue copy ``checkpoints/_rescue.FCStd`` of the live document; (3) the checkpoint is
        opened HIDDEN, the live objects are removed and ``copyObject`` brings the checkpoint's over; (4) the
        temp document is closed and the live document RE-ACTIVATED (App and Gui: after the temp document
        closes both active documents are None, which would break the next script, snapshot and fit-view);
        (5) recompute and validation: the object-name set equals the sidecar's, no object became ``Invalid``
        that was not already, and the live document is active; (6) on ANY failure the transaction is aborted
        or, after a commit, the rescue copy is restored, and ``ok`` is False with the error. The live
        ``doc.Name`` never changes; unrelated open documents are never touched. Never raises."""
        found = self._find_checkpoint(doc_name, checkpoint_id)
        if found is None:
            known = ", ".join(sorted({e["id"] for e in self._checkpoint_entries()})) or "none"
            return RestoreResult(False, f"checkpoint '{checkpoint_id}' does not exist; available: {known}.",
                                 doc_name, str(checkpoint_id))
        path, side = found
        ckpt = os.path.basename(path)[: -len(".FCStd")].split(".", 1)[0]
        if not zipfile.is_zipfile(path):
            return RestoreResult(False, f"checkpoint '{ckpt}' is not a valid FCStd file.", doc_name, ckpt)
        rescue = os.path.join(self._checkpoint_dir(), "_rescue.FCStd")

        def task() -> dict:
            doc = gui.resolve_doc(doc_name)
            if doc is None:
                return {"error": f"document '{doc_name}' is not open", "doc": doc_name}
            if side.get("doc") and side["doc"] != doc.Name:
                return {"error": f"checkpoint '{ckpt}' belongs to document '{side['doc']}', not '{doc.Name}'",
                        "doc": doc.Name}
            before = [o.Name for o in doc.Objects]
            rescue_ok, _ = gui.save_copy(doc, rescue)
            res = gui.restore_from_file(doc, path, ckpt)
            problems = []
            if res["committed"]:
                expected = side.get("all_objects")
                if expected is not None:
                    names = set(res["names_after"])
                    if names != set(expected):
                        missing, extra = sorted(set(expected) - names), sorted(names - set(expected))
                        problems.append(f"object names differ from the checkpoint (missing {missing[:5]}, extra {extra[:5]})")
                new_invalid = [n for n in res["invalid"] if n not in set(side.get("invalid") or [])]
                if new_invalid:
                    problems.append(f"objects invalid after the restore: {', '.join(new_invalid[:5])}")
            if not res["reactivated"]:
                problems.append("the live document could not be re-activated")
            error = "; ".join(p for p in [res["error"], *problems] if p)
            ok = bool(res["ok"]) and not problems
            reactivated = res["reactivated"]
            after = res["names_after"]
            if not ok and res["committed"] and rescue_ok:
                back = gui.restore_from_file(doc, rescue, "rescue")
                error += ("; the previous state was restored from the rescue copy" if back["ok"]
                          else "; ALSO the rescue copy could not be restored: " + back["error"])
                after, reactivated = back["names_after"], back["reactivated"]
            return {"ok": ok, "error": error, "doc": doc.Name, "before": before, "after": after,
                    "reactivated": reactivated}

        result, error = gui.gui_call(task, self.RESTORE_TIMEOUT_S, "project_restore")
        if error:
            return RestoreResult(False, f"restore did not run: {error}", doc_name, ckpt)
        if result.get("error") and "ok" not in result:
            return RestoreResult(False, result["error"], result.get("doc", doc_name), ckpt)
        return RestoreResult(bool(result["ok"]), result["error"], result["doc"], ckpt, result["before"],
                             result["after"], bool(result["reactivated"]))

    def rewind_to(self, step_index: int, doc_name: str | None = None) -> RestoreResult:
        """Restore the model as it was after step ``step_index`` and append a ``rewind`` journal line.

        The checkpoint used is the newest geometry checkpoint at or before ``step_index`` (of ``doc_name`` when
        given, else of the document that checkpoint belongs to). Steps after ``step_index`` are NOT deleted:
        they become ``superseded`` (excluded from rebuild.py, struck through in NOTES). A rewind happens
        between turns, never inside one."""
        try:
            wanted = int(step_index)
        except (TypeError, ValueError):
            return RestoreResult(False, f"step index must be an integer, got {step_index!r}.")
        if wanted < 0 or wanted > self.last_step_index():
            return RestoreResult(False, f"there is no step {wanted}; steps run 1..{self.last_step_index()}.")
        candidates = [e for e in self._checkpoint_entries() if e["kind"] == "geometry" and e["index"] <= wanted
                      and (not doc_name or str(e["side"].get("doc", "")).lower() == doc_name.lower())]
        if not candidates:
            return RestoreResult(False, f"no checkpoint at or before step {wanted}"
                                        + (f" for document '{doc_name}'" if doc_name else "") + ".", doc_name or "")
        entry = max(candidates, key=lambda e: e["index"])
        result = self.restore(doc_name or str(entry["side"].get("doc", "")), entry["id"])
        if result.ok:
            with self._lock:
                self._journal_line({"type": "rewind", "ts": self._now(), "to_step": wanted,
                                    "restored_checkpoint": entry["id"]})
                self._jcache = None
                self.manifest["last_geometry_step"] = self.last_geometry_step()
                self._save_manifest()
                self._regenerate_docs(rebuild=True)
            # the sheet went back with the document: refresh the params.json mirror (a GUI call, so no lock held)
            self._sync_params(result.doc or doc_name or "", wanted)
        return result

    @_lifecycle("note_reverted", 0)
    def note_reverted(self, from_step: int, via: str = "undo", checkpoint: str = "") -> int:
        """The user reverted the live model to just before step ``from_step`` ("revert changes" on a turn card: the
        undo stack is popped, no checkpoint is restored). Journals a ``rewind`` line (``to_step = from_step - 1``,
        ``via = "undo"``, or ``via="checkpoint"`` with the restored ``checkpoint`` id) so the steps from there on
        become ``superseded``: rebuild.py and NOTES.md stop describing geometry that is gone, and a validation recorded for them counts as stale. The steps stay in steps.jsonl.

        Returns how many steps were superseded (0 when the project holds nothing yet or nothing at or after
        ``from_step``). Does not touch the document; NOTES.md reads the model tree through ``gui_call`` (inline
        when called on the GUI thread, as the panel does)."""
        try:
            first = int(from_step)
        except (TypeError, ValueError):
            raise ProjectError(f"step index must be an integer, got {from_step!r}.") from None
        if first < 1 or not self._materialized:
            return 0
        with self._lock:
            steps, _ = self._journal()
            affected = [s for s in steps if s.index >= first and not s.superseded]
            if not affected:
                return 0
            self._journal_line({"type": "rewind", "ts": self._now(), "to_step": first - 1,
                                "restored_checkpoint": str(checkpoint or ""), "via": str(via or "undo")})
            self._jcache = None
            self.manifest["last_geometry_step"] = self.last_geometry_step()
            self._save_manifest()
            self._regenerate_docs(rebuild=True)
            doc_name = next((s.doc for s in affected if s.doc), "")
        # the sheet went back with the model: refresh the params.json mirror (a GUI call, so no lock held)
        self._sync_params(doc_name, first - 1, clear_stale=True)
        return len(affected)

    def export_path(self, stem: str, ext: str) -> str:
        """An absolute, unique path under ``exports/`` for ``<stem>.<ext>`` (``-2``, ``-3`` ... on a clash with a
        file or a path handed out earlier); the folder is created. Thread-safe; nothing is created at the
        path itself (the caller writes the file, ideally atomically)."""
        self.ensure_dir()
        directory = self.path("exports")
        os.makedirs(directory, exist_ok=True)
        with self._export_lock:
            path = unique_path(directory, safe_stem(stem), str(ext), self._reserved_exports)
            self._reserved_exports.add(path)
        with self._lock:
            self.manifest["counters"]["export"] = int(self.manifest["counters"].get("export", 0)) + 1
            self._save_manifest()
        return path

    # -- parameters --------------------------------------------------------------------------------------------

    def sync_params(self, doc_name: str) -> list[dict]:
        """Mirror the ``Params`` / ``Parameters`` sheet of ``doc_name`` into ``params.json`` and return its rows
        (``{"name","value","unit","expr","source","note","cell","bound","needs_rebuild"}``; ``[]`` and no file
        when the document has no such sheet). The sheet stays the source of truth; the read runs on the GUI
        thread (10 s)."""
        return self._sync_params(doc_name, self.last_step_index())

    def _sync_params(self, doc_name: str, step: int, clear_stale: bool = False) -> list[dict]:
        if not doc_name:
            return []
        ctx = current_context()
        timeout = _timeout_for(self.PARAMS_TIMEOUT_S)
        if timeout < 0.5:
            return []

        def task() -> dict:
            doc = gui.resolve_doc(doc_name)
            if doc is None:
                return {"sheet": "", "rows": []}
            table = gui.param_table(doc)
            return {"doc": doc.Name, "sheet": table["sheet"], "rows": table["rows"], "found": True}

        result, error = gui.gui_call(task, timeout, "project_sync_params", ctx.should_stop)
        if error:
            log_event("hook_skipped", hook="sync_params", why=error[:120])
            return []
        if not result or not result.get("sheet"):
            if clear_stale and result and result.get("found"):   # the sheet is gone from the live document
                try:
                    os.remove(self.path("params.json"))
                except OSError:
                    pass
            return []
        rows = _json_safe(result["rows"])
        self.ensure_dir()
        with self._lock:
            _atomic_write_json(self.path("params.json"), {"schema": 1, "doc": result["doc"], "sheet": result["sheet"],
                                                          "step": step, "updated": self._now(), "params": rows})
        return rows

    # -- provenance ---------------------------------------------------------------------------------------------

    def provenance_add(self, **fields) -> str:
        """Record one source / claim in ``provenance.json`` and return its id (``E1``, ``E2``, ...).

        Accepts exactly ``kind`` (``reference|datasheet|download|attachment|measurement|assumption``), ``claim``
        (non-empty: what the source proves), ``source`` (non-empty), ``url``, ``local_path`` (an absolute file: it is COPIED to
        ``refs/<sha256[:12]>_<basename>`` and the stored ``local_path`` is the relative copy), ``sha256``
        (computed from the file when ``local_path`` is given; a given value must match), ``bytes``,
        ``detected_type`` (``step|stl|zip|pdf|brep|iges|none``), ``status`` (``PROVENANCE_STATUSES``),
        ``confidence`` (``CONFIDENCE_LEVELS``, default ``assumed``), ``reference_status`` and ``note``. A
        missing or invalid enum value raises ``ProjectError`` listing the accepted values. When
        ``reference_status`` is passed the call also updates ``reference`` (like ``set_reference``)."""
        unknown = sorted(set(fields) - _PROVENANCE_FIELDS)
        if unknown:
            raise ProjectError(f"provenance_add: unknown field(s) {', '.join(repr(u) for u in unknown)}; "
                               f"accepted: {', '.join(sorted(_PROVENANCE_FIELDS))}.")

        def enum(name: str, accepted: tuple[str, ...], default: str | None = None) -> str:
            value = fields.get(name, default)
            if value not in accepted:
                raise ProjectError(f"provenance_add: '{name}' must be one of {', '.join(accepted)}, got {value!r}.")
            return value

        kind = enum("kind", PROVENANCE_KINDS)
        status = enum("status", PROVENANCE_STATUSES)
        confidence = enum("confidence", CONFIDENCE_LEVELS, "assumed")
        detected = enum("detected_type", DETECTED_TYPES, "none")
        ref_status = fields.get("reference_status")
        if ref_status is not None and ref_status not in REFERENCE_STATUSES:
            raise ProjectError("provenance_add: 'reference_status' must be one of "
                               f"{', '.join(REFERENCE_STATUSES)}, got {ref_status!r}.")
        source = fields.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ProjectError("provenance_add: 'source' must be a non-empty string (where the claim comes from).")
        claim = fields.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise ProjectError("provenance_add: 'claim' must be a non-empty string (the fact the source backs, "
                               "for example 'Board outline 85 x 56 mm'); an entry without a claim proves nothing.")
        self.ensure_dir()
        try:
            size = int(fields.get("bytes") or 0)
        except (TypeError, ValueError, OverflowError):
            raise ProjectError(f"provenance_add: 'bytes' must be a non-negative integer, got {fields.get('bytes')!r}.") from None
        if size < 0:
            raise ProjectError(f"provenance_add: 'bytes' must be a non-negative integer, got {size}.")
        local_rel, digest = "", str(fields.get("sha256") or "")
        local = fields.get("local_path")
        if local:
            if not os.path.isabs(str(local)) or not os.path.isfile(str(local)):
                raise ProjectError(f"provenance_add: 'local_path' must be an existing absolute file, got {local!r}.")
            computed = _sha256_file(str(local))
            if digest and digest.lower() != computed:
                raise ProjectError(f"provenance_add: sha256 {digest} does not match the file ({computed}).")
            digest, size = computed, os.path.getsize(str(local))
            local_rel = f"refs/{digest[:12]}_{safe_stem(os.path.basename(str(local)), 'file')}"
            os.makedirs(self.path("refs"), exist_ok=True)
            if not os.path.isfile(self.path(local_rel)):
                _copy_file_atomic(str(local), self.path(local_rel))
        with self._lock:
            data = self.read_provenance()
            numbers = [int(m.group(1)) for e in data["entries"] if (m := re.fullmatch(r"E(\d+)", str(e.get("id", ""))))]
            entry_id = f"E{max(numbers, default=0) + 1}"
            data["entries"].append({
                "id": entry_id, "ts": self._now(), "kind": kind, "claim": claim.strip()[:_NOTE_LIMIT],
                "source": source.strip(), "url": str(fields.get("url") or ""), "local_path": local_rel,
                "sha256": digest, "bytes": size, "detected_type": detected, "status": status,
                "confidence": confidence, "reference_status": ref_status or "",
                "note": str(fields.get("note") or "")[:_NOTE_LIMIT]})
            data["schema"] = 1
            if ref_status is not None:
                data["reference"] = {**data["reference"], "status": ref_status, "entry": entry_id}
                self.manifest["reference"] = dict(data["reference"])
            _atomic_write_json(self.path("provenance.json"), data)
            if ref_status is not None:
                self._save_manifest()
            self._regenerate_docs(rebuild=False)
        return entry_id

    def set_reference(self, ref_id: str, name: str, status: str, entry_id: str) -> None:
        """Set the project's reference part (``reference`` in project.json and provenance.json).
        ``status`` must be one of ``REFERENCE_STATUSES``."""
        if status not in REFERENCE_STATUSES:
            raise ProjectError(f"set_reference: 'status' must be one of {', '.join(REFERENCE_STATUSES)}, got {status!r}.")
        self.ensure_dir()
        reference = {"id": str(ref_id or ""), "name": str(name or ""), "status": status, "entry": str(entry_id or "")}
        with self._lock:
            data = self.read_provenance()
            data["schema"] = 1
            data["reference"] = reference
            _atomic_write_json(self.path("provenance.json"), data)
            self.manifest["reference"] = dict(reference)
            self._save_manifest()
            self._regenerate_docs(rebuild=False)

    # -- notes / ledger / spec / validation -----------------------------------------------------------------------

    def note(self, kind: str, text: str, *, refs: Iterable[str] = ()) -> str:
        """Add a ``record_note`` entry (``kind`` in ``NOTE_KINDS``, ``text`` clipped to 2000 characters) to
        ``notes.json`` and regenerate NOTES.md; returns the id (``N1``, ``N2``, ...). ``refs`` are related step or
        requirement ids (``S5``, ``R2``)."""
        if kind not in NOTE_KINDS:
            raise ProjectError(f"note: 'kind' must be one of {', '.join(NOTE_KINDS)}, got {kind!r}.")
        body = str(text or "").strip()
        if not body:
            raise ProjectError("note: 'text' must not be empty; say what the user must know.")
        self.ensure_dir()
        with self._lock:
            entries = self.read_notes()
            numbers = [int(m.group(1)) for e in entries if (m := re.fullmatch(r"N(\d+)", str(e.get("id", ""))))]
            note_id = f"N{max(numbers, default=0) + 1}"
            entries.append({"id": note_id, "ts": self._now(), "kind": kind, "text": body[:_NOTE_LIMIT],
                            "refs": [str(r) for r in refs], "step": self.last_step_index()})
            _atomic_write_json(self.path("notes.json"), {"schema": 1, "entries": entries})
            self._regenerate_docs(rebuild=False)
        return note_id

    @_lifecycle("save_ledger")
    def save_ledger(self, ledger: dict) -> None:
        """Write ``ledger.json`` atomically (a Ledger object with ``to_dict()`` is accepted; anything else raises
        ``ProjectError``; a disk failure is logged, not raised)."""
        data = _as_dict(ledger)
        if data is None:
            raise ProjectError("save_ledger: the ledger must be a dict (Ledger.to_dict()).")
        self.ensure_dir()
        with self._lock:
            _atomic_write_json(self.path("ledger.json"), data)

    @_lifecycle("archive_ledger")
    def archive_ledger(self, turn_id: str) -> None:
        """Move ``ledger.json`` to ``ledgers/<turn_id>.json`` (a finished task's ledger); no-op without one; a disk
        failure is logged, not raised."""
        if not os.path.isfile(self.path("ledger.json")):
            return
        with self._lock:
            os.makedirs(self.path("ledgers"), exist_ok=True)
            _fsretry.replace(self.path("ledger.json"), self.path("ledgers", f"{safe_stem(turn_id, 'task')}.json"))

    def save_spec(self, spec: dict) -> None:
        """Write ``spec.json`` (the EnclosureSpec dict) atomically and refresh NOTES.md."""
        data = _as_dict(spec)
        if data is None:
            raise ProjectError("save_spec: the spec must be a dict (EnclosureSpec.to_dict()).")
        self.ensure_dir()
        with self._lock:
            _atomic_write_json(self.path("spec.json"), data)
            self._regenerate_docs(rebuild=False)

    @_lifecycle("record_validation")
    def record_validation(self, validation: dict) -> None:
        """Write ``validation.json`` (the latest validation record) and refresh NOTES.md (a disk failure is logged,
        not raised).

        Accepts the ``tools_verify.ValidationResult`` (dataclass or dict) or any dict of the
        validation.json shape; missing fields get defaults (``schema``, ``ts``, ``after_step`` = the newest
        geometry step, empty lists) and ``text`` / ``fit_ran`` / ``fail_rows`` are not stored. A PARTIAL record
        (``validate_model`` alone: ``ran == ["health"]``; ``check_fit`` alone: only fit rows) is merged into the
        stored record of the SAME model state (same document and ``after_step``) so it never wipes the evidence
        of a wider run; a record of another model state is replaced (``project_validation.merge``)."""
        self._store_validation(validation)

    def _store_validation(self, validation: dict) -> dict:
        """``record_validation`` that returns the record as stored (raises on a disk failure; ``ProjectError`` for a
        wrongly typed argument)."""
        data = _as_dict(validation)
        if data is None:
            raise ProjectError("record_validation: expected a validation dict (ValidationResult), got None.")
        self.ensure_dir()
        with self._lock:
            record = vrec.normalize(data, now=self._now(), default_after_step=self.last_geometry_step())
            record = _json_safe(vrec.merge(self.read_validation(), record))
            _atomic_write_json(self.path("validation.json"), record)
            self._regenerate_docs(rebuild=False)
        return record

    def mark_exported(self) -> None:
        """Remember that ``exports/`` now reflects the model as of the newest geometry step (called by the export code);
        the Build Summary lists exports older than the model as an open item. Never raises."""
        try:
            self.ensure_dir()
            with self._lock:
                self.manifest["exported_after_step"] = self.last_geometry_step()
                self._save_manifest()
        except Exception as exc:  # noqa: BLE001
            log_event("project_error", where="mark_exported", error=f"{type(exc).__name__}: {exc}"[:200])

    def record_rebuild_check(self, ok: bool, detail: str = "") -> None:
        """Remember the outcome of the headless rebuild check (``export_rebuild_script(run_check=true)``).

        The Build Summary lists ``rebuild.py reproduces the model headless`` as validated only while the last
        passing check is not older than the last geometry step."""
        self.ensure_dir()
        with self._lock:
            self.manifest["rebuild_check"] = {"ok": bool(ok), "after_step": self.last_step_index(),
                                              "ts": self._now(), "detail": _clip_str(detail, 400)}
            self._save_manifest()
            self._regenerate_docs(rebuild=False)

    def _rebuild_ok(self) -> bool:
        check = self.manifest.get("rebuild_check") or {}
        return bool(check.get("ok")) and int(check.get("after_step", -1)) >= self.last_geometry_step()

    # -- generation ------------------------------------------------------------------------------------------------


    def finalize(self, *, ledger: dict | None, outcome: str, validation: dict | None,
                 export: bool | None = None) -> BuildSummary:
        """End of a WORK turn: autosave, write NOTES.md + rebuild.py, record ``manifest["finalized"]`` and return
        the Build Summary. A non-work turn never calls this (it must not rewrite NOTES.md / rebuild.py).

        ``validation`` (a ``ValidationResult`` or its dict) is stored as ``validation.json`` first (merged with a
        partial record of the same model state), so the Summary and NOTES.md agree; without one the stored record
        is used (the Summary never claims what no check recorded). ``export=None`` means True for the ledger
        intents enclosure / fixture / bracket / generic_part / deliver: when ``exports/`` holds no file yet, STEP +
        STL of the visible solid tips are exported with the spec's print orientation (one file per part, the SAME
        code as the ``export_model`` tool); a failure becomes a NOTES "Unresolved" note and never an exception. ``export=True`` forces the export, ``False`` skips it. A disk failure while writing
        the files is logged and the Summary is still returned (the turn must not die because the dossier failed);
        only a wrongly typed ``ledger`` / ``validation`` raises ``ProjectError``."""
        ledger_dict = _as_dict(ledger)
        validation_dict = _as_dict(validation)   # a wrongly typed argument is the caller's bug: raise before anything is written
        effective = validation_dict
        try:
            if validation_dict is not None:
                # validation.json first: NOTES.md and the Summary then read the same (merged) record
                effective = self._store_validation(validation_dict)
            self._finalize_files(ledger_dict, outcome, export)
        except (OSError, ProjectStorageError) as exc:
            log_event("project_error", where="finalize", error=f"{type(exc).__name__}: {exc}"[:200])
        except ProjectError:
            raise
        except Exception as exc:  # noqa: BLE001 - the Summary must still be returned
            log_event("project_error", where="finalize", error=f"{type(exc).__name__}: {exc}"[:200])
        return self.build_summary(ledger_dict, outcome, effective)

    def _finalize_files(self, ledger_dict: dict | None, outcome: str, export: bool | None) -> None:
        """The writing half of ``finalize`` (may raise ``OSError`` / ``ProjectStorageError``)."""
        self.ensure_dir()
        if ledger_dict is not None:
            self.save_ledger(ledger_dict)
        doc = self.manifest.get("active_doc") or ""
        if doc:
            self.autosave(doc)
        wants = export if export is not None else ((ledger_dict or {}).get("intent") in AUTO_EXPORT_INTENTS)
        if wants and doc and (export is True or not self._listdir("exports")):
            try:
                result = export_tips(self, doc, formats=("step", "stl"))
                for problem in result.errors:
                    self.note("unresolved", f"auto-export problem: {problem}")
            except Exception as exc:  # noqa: BLE001 - the Summary must still be written
                self.note("unresolved", f"auto-export failed: {exc}")
                log_event("project_error", where="finalize_export", error=f"{type(exc).__name__}: {exc}"[:200])
        self.write_notes_md(ledger_dict)
        self.write_rebuild_py()
        with self._lock:
            self.manifest["finalized"] = {"turn_id": self._turn_id, "outcome": outcome, "ts": self._now()}
            self._save_manifest()

