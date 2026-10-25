"""Types, errors and small helpers of :mod:`oscar.project` (private, PURE: no FreeCAD, no Qt).

``ProjectError`` / ``ProjectStorageError``, the ``StepRecord`` / ``RestoreResult`` / ``BuildSummary`` records, the
atomic-write helpers (``<name>.tmp-<pid>-<rand>`` in the same folder + ``os.replace``: a reader never sees a partial
file), JSON helpers and the AST lint that marks a script as not headless-replayable. ``oscar.project`` re-exports
every public name, so callers import from there.
"""

import ast
import dataclasses
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from . import fsretry as _fsretry

_ARG_LIMIT = 200_000   # longest string kept in a recorded argument (longer ones are truncated with a marker)
# Why a successful call is not part of rebuild.py (the "why" of its replay entry).
_NOT_REPLAYED = {
    "execute_code_async": "async jobs cannot be replayed deterministically; use execute_code for rebuildable work",
    "source_reference_part": "the download is recorded in provenance; its file is replayed by import_reference_part",
    "run_fem_analysis": "FEM analysis results are not replayed",
    "save_document": "an output file, not model state",
    "export_model": "an output file, not model state",
    "export_rebuild_script": "an output file, not model state",
}
# Ledger intents whose finished work turn auto-exports STEP + STL when exports/ is empty (Project.finalize).
AUTO_EXPORT_INTENTS = ("enclosure", "fixture", "bracket", "generic_part", "deliver")


class ProjectError(Exception):
    """An invalid argument or an unusable project folder; the message says what is accepted."""


class ProjectStorageError(ProjectError):
    """The project folder (and its fallback) cannot be written. A ``ProjectError`` for callers that only care about
    "the dossier failed"; the agent-facing lifecycle methods swallow it (the turn must never die because the
    dossier failed), argument errors (plain ``ProjectError``) still propagate."""


@dataclass
class StepRecord:
    """One ``steps.jsonl`` line for one mutating tool call."""

    index: int                 # 1-based, monotonic per project, never reused
    ts: str                    # ISO seconds
    turn_id: str
    round: int
    call_id: str
    tool: str
    title: str                 # <= 80 chars: step_title argument or tools.derive_title(...)
    args: dict                 # full JSON-safe arguments (any string > 200_000 chars truncated with "...[truncated N chars]")
    ok: bool
    result_head: str           # first 600 chars of the result text
    result_chars: int
    dur_s: float
    doc: str                   # document name or ""
    delta: dict                # {"added": [...], "removed": [...], "changed": [{"name","props":[...],"volume_before","volume_after"}]}
    health: dict | None        # {"valid", "errors", "warnings", "line", "findings": [<=5 dicts]}
    snapshot: str              # relative path "snapshots/007_shell.png" or ""
    snapshot_note: str         # reason when snapshot == ""
    checkpoint: str            # checkpoint id "007" or ""
    checkpoint_note: str       # reason when checkpoint == "" for a successful geometry step
    script: str                # relative path "scripts/007_shell.py" or ""
    replay: dict               # {"kind":"script","file","headless","gui_only"} | {"kind":"tool","name","args"} | {"kind":"none","why"}
    superseded: bool = False   # derived from rewind lines when reading; never stored as True
    geometry: bool = False     # the tool was a geometry tool (stored so last_geometry_step() needs no tool registry)

    def to_line(self) -> dict:
        """The ``{"type": "step", ...}`` journal object (``superseded`` is derived on read, never stored)."""
        data = {"type": "step"}
        for f in dataclasses.fields(self):
            if f.name != "superseded":
                data[f.name] = getattr(self, f.name)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "StepRecord":
        """A record from a journal object; unknown keys are ignored, missing ones get defaults."""
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name in data:
                kwargs[f.name] = data[f.name]
            elif f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            else:
                kwargs[f.name] = {} if f.name in ("args", "delta", "replay") else (
                    None if f.name == "health" else (0 if f.name in ("index", "round", "result_chars") else (
                        0.0 if f.name == "dur_s" else (False if f.name == "ok" else ""))))
        kwargs["index"] = int(kwargs["index"])
        kwargs["superseded"] = False
        return cls(**kwargs)


@dataclass
class RestoreResult:
    """Outcome of ``Project.restore`` / ``rewind_to``: ``ok`` False carries the reason in ``error`` and the live
    document is then unchanged (or put back from the rescue copy); ``objects_before`` / ``objects_after`` are the
    object names of the live document around the restore; ``reactivated`` says the live document is the active one
    again (App and Gui) after the temporary document was closed."""

    ok: bool
    error: str = ""
    doc: str = ""
    checkpoint: str = ""
    objects_before: list = field(default_factory=list)
    objects_after: list = field(default_factory=list)
    reactivated: bool = False  # the live document was re-activated (App + Gui) after the temp document was closed


@dataclass
class BuildSummary:
    """The Build Summary of a finished work turn: ``markdown`` is the ``## Build Summary`` block (at most 70 lines /
    7000 characters), ``data`` the same content as JSON-safe data."""

    markdown: str              # the "## Build Summary" block
    data: dict                 # same content as JSON: tree, parameters, provenance, validated, not_validated, open_items, files


@dataclass
class _Saved:
    """Result of one autosave (private)."""

    rel: str = ""
    abs: str = ""
    doc: str = ""
    names: list = field(default_factory=list)
    all_names: list = field(default_factory=list)
    invalid: list = field(default_factory=list)
    note: str = ""


def _ts(clock_value: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(clock_value))


def _clip_str(value: Any, limit: int) -> str:
    s = value if isinstance(value, str) else str(value)
    return s if len(s) <= limit else s[:limit] + f"...[truncated {len(s) - limit} chars]"


def _json_safe(value: Any, limit: int = _ARG_LIMIT, _depth: int = 0) -> Any:
    """``value`` as plain JSON data: dict keys become strings, tuples / sets lists, non-finite floats and unknown
    objects strings; any string over ``limit`` characters is truncated with ``...[truncated N chars]``."""
    if _depth > 24:
        return _clip_str(str(value), 200)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _clip_str(value, limit)
    if isinstance(value, dict):
        return {str(k): _json_safe(v, limit, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, limit, _depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(v, limit, _depth + 1) for v in sorted(value, key=str)]
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return _clip_str(str(value), limit)


def _as_dict(obj: Any) -> dict | None:
    """``obj`` as a plain dict: a dict is returned as is, objects with ``to_dict()`` and dataclasses are converted."""
    if obj is None or isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise ProjectError(f"expected a dict, got {type(obj).__name__}")


def _unique_tmp(path: str) -> str:
    return f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _atomic_write_bytes(path: str, data: bytes) -> None:
    """``<name>.tmp-<pid>-<rand>`` in the same directory, flush, ``os.replace``: a reader never sees a partial file."""
    tmp = _unique_tmp(path)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
        _fsretry.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _atomic_write_text(path: str, text: str) -> None:
    """Atomically write ``text`` as UTF-8. A lone surrogate (a model can emit ``\\ud83d`` on its own) cannot be encoded
    and would otherwise make every regeneration of the file fail forever: it becomes ``?``."""
    _atomic_write_bytes(path, text.encode("utf-8", "replace"))


def _atomic_write_json(path: str, obj: Any) -> None:
    """Atomically write ``obj`` as indented JSON. Lone surrogates fall back to ``\\u`` escapes (they round-trip
    through ``json.loads``) instead of failing the write."""
    try:
        text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = json.dumps(obj, ensure_ascii=True, indent=2, default=str)
    _atomic_write_text(path, text + "\n")


def _copy_file_atomic(src: str, dest: str) -> None:
    """Copy ``src`` to ``dest`` in 1 MiB chunks through a unique temp name + ``os.replace`` (a 200 MB reference file
    is never held in memory and a reader never sees a partial copy)."""
    tmp = _unique_tmp(dest)
    try:
        with open(src, "rb") as reader, open(tmp, "wb") as writer:
            for chunk in iter(lambda: reader.read(1 << 20), b""):
                writer.write(chunk)
            writer.flush()
        _fsretry.replace(tmp, dest)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _read_json(path: str, default: Any = None) -> Any:
    """The parsed JSON file, or ``default`` when it is missing or unreadable (never raises)."""
    try:
        with open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8"))
    except (OSError, ValueError):
        return default


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt_index(index: int) -> str:
    return f"{int(index):03d}"


def _clip_title(title: Any) -> str:
    return " ".join(str(title or "").split())[:80]


def _app_label() -> str:
    try:
        import FreeCAD

        version = FreeCAD.Version()
        return f"Oscar {version[0]}.{version[1]}"
    except Exception:
        return "Oscar"


def _lint_gui_only(code: str) -> bool:
    """True when the script's AST touches ``ViewObject`` / ``Gui`` / ``FreeCADGui`` UNGUARDED.

    Guarded means: inside an ``if`` whose test mentions ``GuiUp`` or ``ViewObject``, or inside the body of a
    ``try``; ``getattr(obj, "ViewObject", None)`` (a string, not an attribute access) is never flagged.
    Such a script cannot replay in ``freecadcmd`` (``ViewObject`` is None, ``Gui`` has no document API), so
    rebuild.py reports the step as not headless-replayable. A script that does not parse is not flagged.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False

    class Lint(ast.NodeVisitor):
        def __init__(self) -> None:
            self.hits = 0
            self.guard = 0

        def _guarded(self, nodes) -> None:
            self.guard += 1
            for node in nodes:
                self.visit(node)
            self.guard -= 1

        def visit_If(self, node: ast.If) -> None:
            names = {n.attr for n in ast.walk(node.test) if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            names |= {n.value for n in ast.walk(node.test) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            if {"GuiUp", "ViewObject"} & names:
                self._guarded([node.test, *node.body, *node.orelse])
            else:
                self.generic_visit(node)

        def visit_Try(self, node: ast.Try) -> None:
            if node.handlers:
                self._guarded(node.body)
                for part in (*node.handlers, *node.orelse, *node.finalbody):
                    self.visit(part)
            else:
                self.generic_visit(node)

        visit_TryStar = visit_Try

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if node.attr == "ViewObject" and not self.guard:
                self.hits += 1
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if node.id in ("FreeCADGui", "Gui") and isinstance(node.ctx, ast.Load) and not self.guard:
                self.hits += 1

        def visit_Import(self, node: ast.Import) -> None:
            if any(alias.name.split(".")[0] == "FreeCADGui" for alias in node.names) and not self.guard:
                self.hits += 1

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if (node.module or "").split(".")[0] == "FreeCADGui" and not self.guard:
                self.hits += 1

    lint = Lint()
    lint.visit(tree)
    return lint.hits > 0
