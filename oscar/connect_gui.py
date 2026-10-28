"""Connect mode inside FreeCAD: the workspace's turns run in the Oscar panel like the user's own.

``oscar.connect`` (pure) polls the OriginShape workspace and reports; this module is its FreeCAD side:

* ``WorkspaceConnector`` (one per panel) starts and stops the ``connect.Loop`` from the setting
  (``connect.PREF``, off by default) and the sign-in, and starts each job on the GUI thread: it opens the task's
  chat (``Workspace: <title>``, remembered per task in ``workspace-tasks.json`` next to the chats) and sends the
  message through the panel's own ``_send``, on the OriginShape account with the job's model alias, with the files
  the loop downloaded (``workspace-files/<task>/<message>/`` next to the chats folder) as its attachments. A computer
  that has not seen the task before seeds the chat with the task's earlier messages.
  Each task builds in its own FreeCAD document (``workspace-docs.json`` next to the chats): a new task opens the
  Oscar workbench and a new document named after it; a later message activates that document again (reopening its
  file when it was saved and closed), so one task never builds into another task's model.
* ``GuiTurn`` turns the worker's signals into workspace events (steps as ``connect.step_event`` describes them, the
  reply as it is written, account problems) and makes the preview mesh, the feature tree and the exports when the
  loop asks. The preview is a GLB with one node per visible part, named by its FreeCAD object name like the tree
  (``connect.preview_glb``; an STL of the whole model if that fails). The tree carries the document's parameter
  sheet (``project_gui.param_table``). Exports are one STEP and STL per part plus the whole assembly as STL and the
  FreeCAD file (``project_export.export_tips``, which does its FreeCAD work on the GUI thread itself).

A model the user sends from the browser (``connect.OpenModel``) opens as a new document named
after its task (``open_model_document``): the STEP file is imported with its parts' names and colours, and the view
is framed on it. The connector hands the loop ``open_model``, which is what makes ``hello`` say this Oscar can.

Nothing here changes what a turn may do: scripts and network access keep their approvals. Each question is one
step in the workspace that changes in place: "Waiting for your approval in FreeCAD", then approved, declined or no
answer (``netguard.on_approval``). With *Approve from the workspace* on (``connect.REMOTE_PREF``, on by default once connected),
the step offers the dialog's answers in the browser too, and an answer from there clicks that button in the dialog
(``netguard.answer_remote``); the dialog in FreeCAD stays open and answerable meanwhile.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import tempfile
import threading
import time
from typing import Any

from PySide import QtCore

from . import chats as _chats
from . import connect
from . import params as _params
from .auth import account as _account
from .providers import Message

MAP_FILE = "workspace-tasks.json"
OUTBOX_FILE = "workspace-outbox.json"   # answers not yet taken by the server (``connect.Outbox``)
RECOVER_TIMEOUT_S = 10.0
DOCS_FILE = "workspace-docs.json"
WORKBENCH = "OscarWorkbench"
FILES_DIR = "workspace-files"
START_TIMEOUT_S = 30.0
OPEN_TIMEOUT_S = 120.0       # importing a model the browser sent
REVERT_TIMEOUT_S = 600.0     # undoing turns: restoring checkpoints takes up to 2 min a document
MODELS_REFETCH_S = 300.0     # a job's model the server's list doesn't name: ask for the list again at most this often
REVERT_RESTARTED = ("FreeCAD was restarted since Oscar ran these prompts, so it can no longer undo them from the "
                    "browser. Undo them in FreeCAD (Edit > Undo, or reopen the file).")
REVERT_LATER_TURN = ("A turn made in FreeCAD after these prompts has to be reverted first. Revert it in Oscar’s "
                     "panel in FreeCAD, then try again.")
REVERT_BUSY = "Oscar was busy in FreeCAD. Try the revert again when it’s done."
OPENED_STATUS_MS = 10_000
PREVIEW_TOLERANCE_MM = 0.2
PREVIEW_TIMEOUT_S = 60.0
PARAM_ROWS = 500             # parameter rows sent with a tree
PARAM_FIELDS = ("name", "value", "unit", "expr", "source", "note")
TREE_LIMIT = 1500            # objects sent in a feature tree (a bigger document is cut, ``truncated``)
TREE_BUDGET_S = 3.0          # volumes are measured for this long; later objects are sent without one
TREE_TIMEOUT_S = 20.0
BLOCKER_TEXT = {
    "signed_out": "Oscar in FreeCAD is signed out. Sign in again in Oscar’s settings (⚙).",
    "waitlisted": "Your OriginShape account is on the waitlist, so Oscar can’t build yet.",
    "server": "The OriginShape server set in Oscar’s settings can’t be used.",
    "no_profile": "Oscar in FreeCAD has no chat profile.",
    "outdated": "Oscar in FreeCAD is older than OriginShape supports. Update Oscar, then restart FreeCAD.",
}

_current: WorkspaceConnector | None = None


def current() -> WorkspaceConnector | None:
    """The panel's connector (the settings dialog applies the setting through it)."""
    return _current


def setting_changed() -> None:
    """The user switched connect mode on or off (settings dialog)."""
    if _current is not None:
        _current.apply_setting()


def _log(kind: str, **fields: Any) -> None:
    try:
        from . import turnlog

        turnlog.log_event(kind, **fields)
    except Exception:  # noqa: BLE001 - logging never breaks connect mode
        pass


# --- task -> chat ---------------------------------------------------------------------------------------------


def _map_path() -> str:
    return os.path.join(_chats.chats_dir(), MAP_FILE)


def outbox() -> connect.Outbox:
    return connect.Outbox(os.path.join(_chats.chats_dir(), OUTBOX_FILE))


def _redact(text: str) -> str:
    from . import project_paths

    return project_paths.redact(text)


def _message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
    if isinstance(content, list):
        content = "\n".join(str(part.get("text") or "") for part in content
                            if isinstance(part, dict) and part.get("type", "text") == "text")
    return str(content or "")


def _role(message: Any) -> str:
    return str(message.get("role") if isinstance(message, dict) else getattr(message, "role", ""))


def files_dir() -> str:
    """Where the workspace's attachments are downloaded: beside the chats folder (``connect.Loop``)."""
    return os.path.join(os.path.dirname(os.path.abspath(_chats.chats_dir())), FILES_DIR)


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path: str, data: dict) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    os.replace(tmp, path)


def _load_map() -> dict[str, str]:
    return {str(k): str(v) for k, v in _read_json(_map_path()).items()}


def _save_map(mapping: dict[str, str]) -> None:
    _write_json(_map_path(), mapping)


# --- task -> FreeCAD document -----------------------------------------------------------------------------------


def _docs_path() -> str:
    return os.path.join(_chats.chats_dir(), DOCS_FILE)


def _load_docs() -> dict[str, dict]:
    """``task id -> {"doc": document name, "file": its saved path or ""}``."""
    return {str(k): v for k, v in _read_json(_docs_path()).items() if isinstance(v, dict)}


_NAME_SKIP = frozenset({"and", "for", "the", "with", "from", "into", "make", "build", "create", "design"})


def _doc_name_for(title: str) -> str:
    """A FreeCAD document name from the task's title: its first three real words in CamelCase ("An 80 mm flange
    with bolt holes" -> "FlangeBoltHoles"), else "Task". The document's label carries the whole title."""
    words = [w for w in re.findall(r"[A-Za-z]{3,}", title or "") if w.lower() not in _NAME_SKIP][:3]
    return "".join(w[:1].upper() + w[1:].lower() for w in words)[:40] or "Task"


def remember_document(task_id: str, doc_name: str) -> None:
    """Record ``doc_name`` (and its file, once saved) as the task's document (GUI thread)."""
    if not task_id or not doc_name:
        return
    try:
        import FreeCAD as App

        doc = App.listDocuments().get(doc_name)
        if doc is None:   # not an open document (closed meanwhile): keep what is known
            return
        file_name = str(getattr(doc, "FileName", "") or "")
        docs = _load_docs()
        previous = docs.get(task_id) or {}
        docs[task_id] = {"doc": doc_name, "file": file_name or (previous.get("file", "") if previous.get("doc") ==
                                                                 doc_name else "")}
        _write_json(_docs_path(), docs)
    except Exception as e:  # noqa: BLE001 - losing the link costs a new document next time, never the turn
        _log("connect_doc_map_failed", task=task_id, error=f"{type(e).__name__}: {e}"[:200])


def _gui(name: str, *args: Any) -> None:
    """``FreeCADGui.<name>(*args)`` when there is a GUI; a failure is logged, never raised."""
    try:
        import FreeCADGui as Gui

        call = getattr(Gui, name, None)
        if call is not None:
            call(*args)
    except Exception as e:  # noqa: BLE001 - the panel is open anyway; the view and workbench are niceties
        _log("connect_gui_call_failed", call=name, error=f"{type(e).__name__}: {e}"[:200])


def open_task_document(task_id: str, title: str, new_task: bool) -> str:
    """Make the task's own document the active one before its turn (GUI thread); returns its name.

    A new task opens the Oscar workbench and a new document. A known task gets its document back: activated when
    open, reopened from its file when it was saved and closed, else a new one. A task from before documents were
    remembered keeps building in the active document (its earlier geometry is there)."""
    import FreeCAD as App

    from .cad.engine import get_engine

    entry = _load_docs().get(task_id) or {}
    open_docs = App.listDocuments()
    doc = open_docs.get(entry.get("doc", ""))
    if doc is None and entry.get("file") and os.path.isfile(entry["file"]):
        try:
            doc = App.openDocument(entry["file"])
        except Exception as e:  # noqa: BLE001 - an unreadable file: start over in a new document
            _log("connect_doc_reopen_failed", task=task_id, error=f"{type(e).__name__}: {e}"[:200])
            doc = None
    if doc is None and not new_task and not entry and App.ActiveDocument is not None:
        doc = App.ActiveDocument
    if doc is None:
        if new_task:
            _gui("activateWorkbench", WORKBENCH)
        created = get_engine().create_document(_doc_name_for(title))
        doc = App.listDocuments().get(str(created.get("document_name") or ""))
        if doc is None:
            raise connect.TurnRefused("Oscar couldn’t create a FreeCAD document for this task.")
        if title:
            doc.Label = " ".join(title.split())[:60]
    App.setActiveDocument(doc.Name)
    _gui("setActiveDocument", doc.Name)   # the 3D view follows (FreeCADGui has neither without a GUI)
    remember_document(task_id, doc.Name)
    _log("connect_doc", task=task_id, doc=doc.Name, new_task=new_task)
    return doc.Name


def _import_step(path: str, doc_name: str) -> None:
    """Import a STEP file into the document: with its colours when FreeCAD has its GUI importer."""
    try:
        import ImportGui

        ImportGui.insert(path, doc_name)
    except ImportError:
        import Import

        Import.insert(path, doc_name)


def open_model_document(title: str, path: str) -> str:
    """Open a model the browser sent (the STEP file ``path``) as a new document named after its task, show it and
    frame the view; returns the document's label (GUI thread). A file that holds no solid leaves no document."""
    import FreeCAD as App

    from .cad.engine import get_engine

    created = get_engine().create_document(_doc_name_for(title))
    doc = App.listDocuments().get(str(created.get("document_name") or ""))
    if doc is None:
        raise RuntimeError("Oscar couldn’t create a FreeCAD document for the model.")
    label = " ".join((title or "Model").split())[:60] or "Model"
    doc.Label = label
    try:
        _import_step(path, doc.Name)
        doc.recompute()
        if not any(len(getattr(getattr(obj, "Shape", None), "Solids", None) or ()) for obj in doc.Objects):
            raise ValueError("the file holds no solid")
    except Exception:
        App.closeDocument(doc.Name)
        raise
    App.setActiveDocument(doc.Name)
    _gui("setActiveDocument", doc.Name)
    try:
        import FreeCADGui as Gui

        from .cad.views import fit_view

        view = Gui.activeDocument().activeView() if Gui.activeDocument() else None
        if view is not None and hasattr(view, "viewIsometric"):
            view.viewIsometric()
        fit_view()
        window = Gui.getMainWindow()
        window.statusBar().showMessage(f"Opened “{label}” from the OriginShape studio", OPENED_STATUS_MS)
        window.raise_()
        window.activateWindow()
    except Exception as e:  # noqa: BLE001 - the model is open; framing it and the notice are niceties
        _log("connect_open_view_failed", error=f"{type(e).__name__}: {e}"[:200])
    _log("connect_model_document", doc=doc.Name, objects=len(doc.Objects))
    return label


# --- one turn ---------------------------------------------------------------------------------------------------


class GuiTurn:
    """A workspace turn in the panel (``connect.Turn``). The signal slots run on the GUI thread; ``wait``, ``stop``,
    ``result``, ``take_preview`` and ``exports`` are called from the loop's thread."""

    def __init__(self, job: connect.Job, doc: str = "") -> None:
        self.job = job
        self.events = connect.EventBatch()
        self.refusal = ""
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._worker: Any = None
        self._reply = ""
        self._ok, self._text = False, ""
        self._doc = doc   # the task's document (open_task_document); a step's active document replaces it
        self._preview_due: int | None = None
        self._geometry = False
        self._tree_due = False
        self._tree_final = False
        self._offered: set[int] = set()   # approvals offered to the browser that are still waiting
        self.on_end: Any = None   # called on the GUI thread when the turn ends (the panel's model label goes back)

    # GUI thread ---------------------------------------------------------------------------------------------------

    def attach(self, worker: Any) -> None:
        """Called by the panel's ``_send`` right after it created the worker, before it starts."""
        self._worker = worker
        worker.text_delta.connect(self._on_text)
        worker.turn_finished.connect(self._on_finished)
        for name, slot in (("step_finished", self._on_step_finished), ("account_problem", self._on_account_problem)):
            signal = getattr(worker, name, None)
            if signal is not None and hasattr(signal, "connect"):
                signal.connect(slot)

    def refuse(self, reason: str) -> None:
        """The panel did not start the turn (``_send_blocker``)."""
        self.refusal = BLOCKER_TEXT.get(reason, reason or "Oscar couldn’t start this task in FreeCAD.")

    def _on_text(self, chunk: str) -> None:
        self._reply += chunk
        self.events.reply(self._reply)

    def _on_step_finished(self, payload: str) -> None:
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            return
        if not isinstance(data, dict) or data.get("skipped"):
            return
        self.events.step(connect.step_event(data))
        if data.get("ok") and isinstance(data.get("step_index"), int) and data["step_index"] >= 0:
            doc = _active_doc_name()
            with self._lock:
                self._preview_due = data["step_index"]
                self._doc = doc or self._doc
                self._geometry = True
                self._tree_due = True

    def on_approval(self, info: dict) -> None:
        """A question asked at this computer (``netguard.on_approval``; any thread). Offered to the browser too
        when the setting allows it."""
        state = str(info.get("state") or "")
        remote = state == "waiting" and remote_approvals() and bool(info.get("choices"))
        approval_id = info.get("id")
        with self._lock:
            if remote and isinstance(approval_id, int):
                self._offered.add(approval_id)
            elif state != "waiting":
                self._offered.discard(approval_id)
        self.events.step(connect.approval_event(
            approval_id, state, _approval_what(info),
            choices=info.get("choices") if remote else (),
            script=str(info.get("detailed") or "") if remote and info.get("op") == "code_permission" else "",
            by=str(info.get("by") or "")))

    def _on_account_problem(self, payload: str) -> None:
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            data = {}
        message = str((data or {}).get("message") or "Oscar couldn’t use your OriginShape account.")
        self.events.error(message)

    def _on_finished(self, ok: bool, text: str) -> None:
        self._ok = bool(ok)
        self._text = text or self._reply
        with self._lock:
            doc = self._doc
        remember_document(self.job.task_id, doc or _active_doc_name())   # also picks up a file saved meanwhile
        if self.on_end is not None:
            try:
                self.on_end()
            except Exception as e:  # noqa: BLE001 - a label never fails the turn
                _log("connect_label_failed", error=f"{type(e).__name__}: {e}"[:200])
        self._done.set()

    # any thread (connect.Turn) ------------------------------------------------------------------------------------

    def wait(self, timeout: float) -> bool:
        return self._done.wait(timeout)

    def awaiting_answer(self) -> bool:
        """A question offered to the browser is waiting for its answer."""
        with self._lock:
            return bool(self._offered)

    def answer(self, key: str, choice: str) -> None:
        """The browser answered approval ``key`` ("approval-<id>"): give that answer to the open dialog. Ignored
        unless this turn offered that question and the setting still allows it."""
        match = re.fullmatch(r"approval-(\d+)", key or "")
        if match is None or not remote_approvals():
            return
        approval_id = int(match.group(1))
        with self._lock:
            if approval_id not in self._offered:
                return
        from . import netguard

        taken = netguard.answer_remote(approval_id, choice)
        _log("connect_remote_answer", task=self.job.task_id, approval=approval_id, choice=choice[:40], taken=taken)

    def stop(self) -> None:
        worker = self._worker
        if worker is None:
            return
        try:
            resume = getattr(worker, "resume", None)
            if resume is not None:
                resume()   # a paused turn only sees the stop flag once released
            worker.stop()
        except RuntimeError:   # the C++ thread object is gone: the turn has ended anyway
            pass

    def result(self) -> tuple[bool, str]:
        return self._ok, self._text

    def take_preview(self) -> tuple[int, bytes, str] | None:
        """The model after a geometry step: a GLB with one node per part, else an STL of the whole model."""
        with self._lock:
            due, doc = self._preview_due, self._doc
            self._preview_due = None
        if due is None or not doc:
            return None
        glb = _glb_preview(doc)
        if glb is not None:
            return due, glb, "glb"
        files = _export(doc, ("stl",), PREVIEW_TOLERANCE_MM)
        return (due, files[0][2], "stl") if files else None

    def take_tree(self) -> dict | None:
        """The document's feature tree after a geometry change, and once more when the turn ends."""
        with self._lock:
            doc, due = self._doc, self._tree_due
            final = self._done.is_set() and self._geometry and not self._tree_final
            self._tree_due = False
            if final:
                self._tree_final = True
        if not doc or not (due or final):
            return None
        return _feature_tree(doc)

    def exports(self) -> list[tuple[str, str, bytes]]:
        with self._lock:
            doc, geometry = self._doc, self._geometry
        if not geometry or not doc:
            return []
        parts = _export(doc, ("step", "stl"), 0.05, per_part=True)
        assembly = []
        if sum(1 for _name, fmt, _data in parts if fmt == "stl") > 1:   # several parts: the whole model too
            assembly = [(f"{doc}-assembly.stl", fmt, data) for _name, fmt, data in _export(doc, ("stl",), 0.05)]
        fcstd = _fcstd(doc)
        return parts + assembly + ([fcstd] if fcstd else [])


def remote_approvals() -> bool:
    """The person at this computer lets the workspace answer its approvals (``connect.REMOTE_PREF``)."""
    return _params.get_bool(connect.REMOTE_PREF, connect.REMOTE_DEFAULT)


def _approval_what(info: dict) -> str:
    """What a question at this computer was about, in one line."""
    text = " ".join(str(info.get("text") or "").split())
    first = text.split(". ")[0].rstrip(".") if text else ""
    if info.get("op") == "code_permission":
        return f"Run a script: {first}" if first else "Run a script"
    if info.get("op") == "net_permission":
        return f"Network access: {first}" if first else "Network access"
    return first or str(info.get("title") or "A question in FreeCAD")


def _active_doc_name() -> str:
    try:
        import FreeCAD as App

        doc = App.ActiveDocument
        return doc.Name if doc is not None else ""
    except Exception:  # noqa: BLE001
        return ""


def _export(doc: str, formats: tuple[str, ...], tolerance: float, *, per_part: bool = False
            ) -> list[tuple[str, str, bytes]]:
    """``(name, format, bytes)`` of the document's visible solids: one combined file per format, or with ``per_part``
    one per part (in the assembled pose, so the parts line up in a viewer)."""
    from .project_export import export_tips

    directory = tempfile.mkdtemp(prefix="oscar-connect-")
    try:
        result = export_tips(None, doc, formats=formats, per_part=per_part, print_orientation=False,
                             stl_tolerance=tolerance, directory=directory)
        out = []
        for exported in result.files:
            with open(exported.path, "rb") as handle:
                out.append((os.path.basename(exported.path), exported.format, handle.read()))
        return out
    except Exception as e:  # noqa: BLE001 - a failed export costs the web its file, never the turn
        _log("connect_export_failed", formats=list(formats), error=f"{type(e).__name__}: {e}"[:200])
        return []
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _glb_preview(doc: str) -> bytes | None:
    """The document's visible parts as a GLB, one node per part (``connect.preview_glb``); None when there is
    nothing to show or the parts could not be meshed (the caller then sends an STL)."""
    from .project_gui import gui_call, resolve_doc

    parts, error = gui_call(lambda: _preview_parts(resolve_doc(doc)), PREVIEW_TIMEOUT_S, "connect_preview")
    if error or not parts:
        if error:
            _log("connect_preview_failed", doc=doc, error=str(error)[:200])
        return None
    try:
        return connect.preview_glb(parts, doc)
    except Exception as e:  # noqa: BLE001 - the STL is the fallback
        _log("connect_preview_failed", doc=doc, error=f"{type(e).__name__}: {e}"[:200])
        return None


def _preview_parts(document) -> list | None:
    """``connect.PreviewPart`` for each visible part of ``document`` (GUI thread): the solids a viewer shows, i.e.
    ``project_gui.visible_tips`` (what the per-part exports are made of: no boolean inputs, no hidden reference),
    meshed face by face in the global frame (a part inside an App::Part is placed as in FreeCAD), with its colour."""
    if document is None:
        return None
    from .project_gui import global_shape, visible_tips

    parts = []
    for obj in visible_tips(document):
        positions: list[float] = []
        indices: list[int] = []
        for face in global_shape(obj).Faces:
            points, triangles = face.tessellate(PREVIEW_TOLERANCE_MM)
            base = len(positions) // 3
            for point in points:
                positions.extend((point.x, point.y, point.z))
            for a, b, c in triangles:
                indices.extend((base + a, base + b, base + c))
        if indices:
            parts.append(connect.PreviewPart(name=obj.Name, label=str(getattr(obj, "Label", obj.Name)),
                                             positions=positions, indices=indices, color=_shape_color(obj)))
    return parts


def _shape_color(obj) -> tuple[float, float, float] | None:
    """The object's colour in FreeCAD's 3D view, as linear RGB (None without a GUI)."""
    view = getattr(obj, "ViewObject", None)
    color = getattr(view, "ShapeColor", None) if view is not None else None
    try:
        return tuple(_linear(float(c)) for c in tuple(color)[:3]) if color is not None else None
    except (TypeError, ValueError):
        return None


def _linear(channel: float) -> float:
    channel = max(0.0, min(1.0, channel))
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _feature_tree(doc: str) -> dict | None:
    """The document's tree as FreeCAD's tree view shows it (GUI thread): ``{"document", "label", "roots",
    "objects": [{"name", "label", "type", "children", "visible", "error", "solids", "volume", "bbox"}],
    "truncated"}``; ``bbox`` is the size ``[x, y, z]`` in mm. None when the document is gone."""
    from .project_gui import gui_call, resolve_doc

    result, error = gui_call(lambda: _tree_of(resolve_doc(doc)), TREE_TIMEOUT_S, "connect_tree")
    if error or not isinstance(result, dict):
        _log("connect_tree_failed", doc=doc, error=str(error)[:200])
        return None
    return result


def _children(obj) -> list:
    """The objects FreeCAD's tree view shows under ``obj``: its view provider's ``claimChildren()``. Without a GUI
    the same is rebuilt from the document: an App::Part or Body shows its Origin and the members that no other
    member is built from (as ``ViewProviderGeoFeatureGroupExtension`` does), a feature the objects it is built
    from, an Origin its axes and planes."""
    view = getattr(obj, "ViewObject", None)
    if view is not None and hasattr(view, "claimChildren"):
        try:
            return list(view.claimChildren() or [])
        except Exception:  # noqa: BLE001 - a view provider without children
            return []
    kids = list(getattr(obj, "OriginFeatures", None) or [])
    group = list(getattr(obj, "Group", None) or [])
    if group:
        inner = {getattr(kid, "Name", None) for member in group for kid in _inputs(member)}
        origin = getattr(obj, "Origin", None)
        kids.extend([origin] if origin is not None and hasattr(origin, "Name") else [])
        kids.extend(member for member in group if getattr(member, "Name", None) not in inner)
    return kids + _inputs(obj)


def _inputs(obj) -> list:
    """What a feature is built from: a boolean's base and tool, a fusion's shapes, an extrusion's or a PartDesign
    feature's profile."""
    kids = []
    for name in ("Base", "Tool", "Profile"):
        value = getattr(obj, name, None)
        if isinstance(value, tuple) and value:   # a link with sub-elements: (object, [subs])
            value = value[0]
        if value is not None and hasattr(value, "Name"):
            kids.append(value)
    kids.extend(getattr(obj, "Shapes", None) or [])
    return kids


def _model_first(document) -> list:
    """The document's objects with the model before the hidden reference parts (``Ref_*``): a cut tree, and the
    volume budget, then never lose the user's parts to a big imported reference."""
    from .project_gui import is_ref

    objects = list(document.Objects)
    try:
        refs = {obj.Name for obj in objects if is_ref(obj)}
    except Exception:  # noqa: BLE001 - document order then
        return objects
    return [obj for obj in objects if obj.Name not in refs] + [obj for obj in objects if obj.Name in refs]


def _params_of(document) -> dict | None:
    """The document's parameter sheet (``Params``) as ``{"sheet", "rows": [{"name", "value", "unit", "expr",
    "source", "note"}]}``; None without one."""
    try:
        from .project_gui import param_table

        table = param_table(document)
    except Exception as e:  # noqa: BLE001 - the tree goes without parameters
        _log("connect_params_failed", error=f"{type(e).__name__}: {e}"[:200])
        return None
    if not table.get("sheet"):
        return None
    rows = [{key: row.get(key) for key in PARAM_FIELDS} for row in table.get("rows") or [] if isinstance(row, dict)]
    return {"sheet": str(table["sheet"])[:120], "rows": rows[:PARAM_ROWS]}


def _tree_of(document) -> dict | None:
    if document is None:
        return None
    every = _model_first(document)
    objects = every[:TREE_LIMIT]
    names = {obj.Name for obj in objects}
    claimed: set[str] = set()
    nodes = []
    deadline = time.monotonic() + TREE_BUDGET_S
    for obj in objects:
        children = []
        for child in _children(obj):
            name = getattr(child, "Name", None)
            if name in names and name != obj.Name and name not in children:
                children.append(name)
        claimed.update(children)
        view = getattr(obj, "ViewObject", None)
        visible = getattr(view, "Visibility", None) if view is not None else getattr(obj, "Visibility", None)
        node: dict[str, Any] = {"name": obj.Name, "label": str(getattr(obj, "Label", obj.Name))[:200],
                                "type": str(getattr(obj, "TypeId", ""))[:80], "children": children[:500],
                                "visible": bool(visible) if visible is not None else True,
                                "error": "Invalid" in list(getattr(obj, "State", []) or [])}
        shape = getattr(obj, "Shape", None)
        try:
            if shape is not None and not shape.isNull():
                box = shape.BoundBox
                node["bbox"] = [round(box.XLength, 3), round(box.YLength, 3), round(box.ZLength, 3)]
                node["solids"] = len(shape.Solids)
                if node["solids"] and time.monotonic() < deadline:
                    node["volume"] = round(float(shape.Volume), 3)
        except Exception:  # noqa: BLE001 - an object without a usable shape is listed without its facts
            pass
        nodes.append(node)
    tree = {"document": document.Name, "label": str(getattr(document, "Label", document.Name))[:200],
            "roots": [node["name"] for node in nodes if node["name"] not in claimed], "objects": nodes,
            "truncated": len(every) > len(objects)}
    params = _params_of(document)
    if params is not None:
        tree["params"] = params
    return tree


def _fcstd(doc: str) -> tuple[str, str, bytes] | None:
    """A copy of the FreeCAD document itself (``model.FCStd``)."""
    from .project_gui import gui_call, resolve_doc, save_copy

    directory = tempfile.mkdtemp(prefix="oscar-connect-")
    try:
        path = os.path.join(directory, "model.FCStd")
        result, error = gui_call(lambda: save_copy(resolve_doc(doc), path), 60.0, "connect_fcstd")
        if error or not result or not result[0]:
            return None
        with open(path, "rb") as handle:
            return "model.FCStd", "fcstd", handle.read()
    except Exception as e:  # noqa: BLE001
        _log("connect_export_failed", formats=["fcstd"], error=f"{type(e).__name__}: {e}"[:200])
        return None
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# --- the connector ----------------------------------------------------------------------------------------------


class _StartRequest:
    def __init__(self, job: connect.Job, files: list[str]) -> None:
        self.job = job
        self.files = files
        self.ready = threading.Event()
        self.turn: GuiTurn | None = None
        self.error = ""


class WorkspaceConnector(QtCore.QObject):
    """Connect mode for one panel. ``state_changed(state, message)`` follows the loop (``connect.Loop``); ``off``
    while the setting is off."""

    state_changed = QtCore.Signal(str, str)
    _start_requested = QtCore.Signal(object)
    _state_reported = QtCore.Signal(str, str)

    def __init__(self, panel: Any) -> None:
        super().__init__(panel)
        global _current
        self._panel = panel
        self._loop: connect.Loop | None = None
        self._turn: GuiTurn | None = None
        self._models_asked: dict[str, float] = {}   # api_url -> when /v1/models was last fetched for a job
        self.state: tuple[str, str] = ("off", "")
        self._start_requested.connect(self._on_start_requested, QtCore.Qt.QueuedConnection)
        self._state_reported.connect(self._on_state, QtCore.Qt.QueuedConnection)
        self._wrap_approvals()
        _current = self
        self.apply_setting()

    # setting and lifecycle ----------------------------------------------------------------------------------------

    def apply_setting(self) -> None:
        on = _params.get_bool(connect.PREF, False)
        if on and self._loop is None:
            self._loop = connect.Loop(self._session, start_turn=self._start_turn, is_idle=self._is_idle,
                                      on_state=self._state_reported.emit, versions=connect.versions(_freecad_version()),
                                      log=_log, files_dir=files_dir, outbox=outbox(), recover=self._recover,
                                      redact=_redact, open_model=self._open_model, revert=self._revert)
            self._loop.start()
            _log("connect_on")
        elif not on and self._loop is not None:
            self._loop.stop()
            self._loop = None
            self._on_state("off", "")
            _log("connect_off")

    def shutdown(self) -> None:
        global _current
        if self._loop is not None:
            self._loop.stop(join_s=2.0)
            self._loop = None
        if _current is self:
            _current = None

    def _on_state(self, state: str, message: str) -> None:
        self.state = (state, message)
        self.state_changed.emit(state, message)

    # loop callbacks (the loop's thread) --------------------------------------------------------------------------

    @staticmethod
    def _session() -> Any:
        from .auth import config, session

        try:
            return session.get_session(config.api_url())
        except Exception:  # noqa: BLE001 - an unusable server URL: nothing to connect to
            return None

    def _is_idle(self) -> bool:
        panel = self._panel
        return getattr(panel, "worker", None) is None and not getattr(panel, "_building", False)

    def _start_turn(self, job: connect.Job, files: list[str]) -> GuiTurn:
        session = self._session()
        api_url = getattr(session, "api_url", None)
        asked = self._models_asked.get(api_url or "", float("-inf"))
        if job.model and api_url and _account.cached_model(job.model, api_url) is None \
                and time.monotonic() - asked >= MODELS_REFETCH_S:
            # A model this Oscar hasn't seen listed: its name and capabilities come with the server's list. Asked
            # once per MODELS_REFETCH_S, so a model the list never names (or a list that can't be fetched) doesn't
            # hold up every job.
            self._models_asked[api_url] = time.monotonic()
            try:
                _account.fetch_models(api_url)
            except Exception:  # noqa: BLE001 - the turn runs on what is known (the built-in table, the defaults)
                pass
        request = _StartRequest(job, files)
        self._start_requested.emit(request)
        if not request.ready.wait(START_TIMEOUT_S):
            raise connect.TurnRefused("FreeCAD didn’t respond in time. Try again.")
        if request.turn is None:
            raise connect.TurnRefused(request.error or "Oscar couldn’t start this task in FreeCAD.")
        return request.turn

    @staticmethod
    def _open_model(model: connect.OpenModel, path: str) -> tuple[bool, str, str]:
        """Open a model the browser sent, on the GUI thread: ``(ok, message for the browser, document label)``."""
        from .project_gui import gui_call

        label, error = gui_call(lambda: open_model_document(model.title, path), OPEN_TIMEOUT_S, "connect_open_model")
        if error:
            return False, _redact(f"FreeCAD couldn’t open the model: {error}")[:500], ""
        return True, "", str(label or "")

    def _revert(self, job: connect.RevertJob) -> connect.RevertOutcome:
        """Undo a revert job's turns (on the GUI thread), then read the model after them: the preview and the
        feature tree of the task's document."""
        from .project_gui import gui_call

        outcome, error = gui_call(lambda: self._revert_on_gui(job), REVERT_TIMEOUT_S, "connect_revert")
        if error or not isinstance(outcome, connect.RevertOutcome):
            _log("connect_revert_failed", task=job.task_id, error=str(error)[:200])
            return connect.RevertOutcome((), "FreeCAD didn’t finish the revert in time. Check the model in FreeCAD.")
        doc = str((_load_docs().get(job.task_id) or {}).get("doc") or "")
        if not outcome.reverted or not doc:
            return outcome
        glb = _glb_preview(doc)
        return dataclasses.replace(outcome, preview=("glb", glb) if glb else None, tree=_feature_tree(doc))

    def _revert_on_gui(self, job: connect.RevertJob) -> connect.RevertOutcome:
        """(GUI thread) Undo the job's turns with the panel's own revert, newest first, in the task's chat (made
        current, as a turn does). Only turns this FreeCAD run holds can be undone (their cards and the undo stack
        go with a restart), and only the newest undoable turn at a time: a turn typed in FreeCAD after them stops
        it. Nothing is asked here (the browser asked); a revert that would throw away later changes stops instead."""
        from .panel_blocks import Block

        panel = self._panel
        if panel._busy():
            return connect.RevertOutcome((), REVERT_BUSY)
        chat_id = _load_map().get(job.task_id)
        session = next((s for s in panel.sessions if s.get("id") == chat_id), None) if chat_id else None
        held = (session.get("_workspace_turns") or set()) if session is not None else set()
        if session is None or session.get("blocks") is None or job.messages[0] not in held:
            return connect.RevertOutcome((), REVERT_RESTARTED)
        panel._persist_current()
        panel._activate_session(session, restore=False)
        cards = {block.workspace_message: block for block in panel.blocks
                 if block.kind == Block.TOOL and block.workspace_message}
        reverted: list[str] = []
        for message_id in job.messages:
            if message_id not in held:
                return connect.RevertOutcome(tuple(reverted), REVERT_RESTARTED)
            card = cards.get(message_id)
            if card is None or card.reverted or not _changed_model(card):
                reverted.append(message_id)   # nothing (left) of this turn to undo
                continue
            if not card.revertible:
                return connect.RevertOutcome(tuple(reverted), "Oscar can no longer undo one of these turns in "
                                             "FreeCAD. Undo it in FreeCAD (Edit > Undo, or reopen the file).")
            if panel._latest_revertible() is not card:
                return connect.RevertOutcome(tuple(reverted), REVERT_LATER_TURN)
            if not panel._revert_turn(card, remote=True):
                note = panel.blocks[-1].text if panel.blocks and panel.blocks[-1].kind == Block.TOOL else ""
                return connect.RevertOutcome(tuple(reverted), _redact(note or "FreeCAD couldn’t undo it.")[:1000])
            reverted.append(message_id)
        return connect.RevertOutcome(tuple(reverted))

    # GUI thread ---------------------------------------------------------------------------------------------------

    def _on_start_requested(self, request: _StartRequest) -> None:
        try:
            request.turn = self._begin(request.job, request.files)
        except connect.TurnRefused as e:
            request.error = str(e)
        except Exception as e:  # noqa: BLE001 - the web user gets a message, the log the reason
            _log("connect_start_failed", task=request.job.task_id, error=f"{type(e).__name__}: {e}"[:200])
            request.error = "Oscar couldn’t start this task in FreeCAD."
        finally:
            request.ready.set()

    def _begin(self, job: connect.Job, files: list[str] = ()) -> GuiTurn:
        panel = self._panel
        if panel._busy():
            raise connect.TurnRefused("Oscar was busy with another chat in FreeCAD. Send the message again.")
        new_task = job.task_id not in _load_map()
        session = self._chat_for(job)
        panel._persist_current()
        panel._activate_session(session, restore=False)
        doc = open_task_document(job.task_id, job.title, new_task)
        profile = _params.account_profile()
        # The task's model as the browser picked it (the server checked it is one of its aliases, including models
        # added since this Oscar was released): the gateway is asked for it. A retired alias (a task made before the
        # rename) runs on its tier's current one, which /v1/models lists with the model's real capabilities.
        model = _params.account_model_alias(job.model) if _account.is_model_id(job.model) else profile.model
        profile = dataclasses.replace(profile, model=model)
        turn = GuiTurn(job, doc)
        panel._send(workspace=WorkspaceSend(text=job.text, profile=profile, turn=turn, attachments=list(files)))
        if turn.refusal or turn._worker is None:
            raise connect.TurnRefused(turn.refusal or "Oscar couldn’t start this task in FreeCAD.")
        # While the turn runs, the panel names the model it runs on (the task's, e.g. Origin1 Super), not the model
        # of the profile picked in FreeCAD; afterwards the label shows that profile's model again.
        _show_model(panel, profile)
        turn.on_end = lambda: _show_model(panel, _params.active_profile())
        self._turn = turn
        return turn

    def _recover(self, task_id: str, message_id: str) -> tuple[str, bool, str]:
        """The answer FreeCAD has for a turn the website closed (loop thread): the last message of the task's
        chat when it is Oscar's (``"found"``), ``"busy"`` while that chat still runs a turn, else ``"none"``."""
        from .project_gui import gui_call

        chat_id = _load_map().get(task_id)
        if not chat_id:
            return "none", False, ""

        def look() -> tuple[str, bool, str]:
            panel = self._panel
            session = next((s for s in panel.sessions if s.get("id") == chat_id), None)
            if session is not None and panel._busy() and panel._current_session() is session:
                return "busy", False, ""
            history = (session or _chats.load(chat_id) or {}).get("history") or []
            last = history[-1] if history else None
            if last is None or _role(last) != "assistant" or not _message_text(last).strip():
                return "none", False, ""
            return "found", True, _message_text(last)

        result, error = gui_call(look, RECOVER_TIMEOUT_S, "connect_recover")
        if error or not isinstance(result, tuple):
            return "busy", False, ""   # the GUI is busy: ask again next round
        return result

    def _chat_for(self, job: connect.Job) -> dict:
        """The task's chat: an open one, a stored one, or a new one seeded with the task's earlier messages. A chat
        the user didn't name in FreeCAD follows the task's title (renamed in the workspace)."""
        panel = self._panel
        mapping = _load_map()
        chat_id = mapping.get(job.task_id)
        wanted = f"Workspace: {job.title}"[:80]
        if chat_id:
            found = next((session for session in panel.sessions if session.get("id") == chat_id), None)
            if found is None:
                found = _chats.load(chat_id)
                if found is not None:
                    found["blocks"] = None
                    panel.sessions.append(found)
            if found is not None:
                if job.title and not found.get("named") and found.get("title") != wanted:
                    found["title"] = wanted   # the picker shows it when the chat is activated
                return found
        session = _chats.new_session(title=wanted)
        session["blocks"] = None
        session["history"].extend(Message(role=role, content=text) for role, text in job.history)
        panel.sessions.insert(0, session)
        _chats.save(session)
        mapping[job.task_id] = session["id"]
        _save_map(mapping)
        return session

    def _wrap_approvals(self) -> None:
        """A script or network question at this computer shows in the workspace while its turn runs: waiting, then
        approved, declined or no answer."""
        try:
            from . import netguard
        except Exception:  # noqa: BLE001
            return
        previous = netguard.on_approval

        def hook(info: dict) -> None:
            if previous is not None:
                previous(info)
            turn = self._turn
            if turn is not None and not turn.wait(0):
                turn.on_approval(info)

        netguard.on_approval = hook


def _changed_model(card: Any) -> bool:
    """A turn card that changed the model: revertible, or with a step that changed something (a card a failed
    revert left not revertible still did)."""
    if card.revertible:
        return True
    return any(getattr(step.view, "mutating", False) or (getattr(step.view, "step_index", 0) or 0) >= 1
               for step in card.steps if getattr(step, "view", None) is not None)


def _show_model(panel: Any, profile: Any) -> None:
    """Show ``profile``'s model ("Origin1 Super") in the panel's model label."""
    label = getattr(panel, "model_label", None)
    if label is not None:
        label.setText(_params.model_display_name(profile))


@dataclasses.dataclass
class WorkspaceSend:
    """What the panel's ``_send`` needs for a workspace turn instead of the input box."""

    text: str
    profile: Any
    turn: GuiTurn
    attachments: list[str] = dataclasses.field(default_factory=list)   # local paths of the downloaded files


def _freecad_version() -> str:
    try:
        import FreeCAD as App

        return ".".join(str(part) for part in App.Version()[:3])
    except Exception:  # noqa: BLE001
        return ""
