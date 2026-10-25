"""Connect mode: Oscar runs the tasks a user starts in the OriginShape workspace (``/studio`` on the web app).

This module is pure python (no FreeCAD, no Qt): the protocol client, the batch of events waiting to be posted and the
polling ``Loop``. The FreeCAD side (``connect_gui``) starts each turn in the panel and gives the loop a ``Turn`` to
report on.

The loop is one daemon thread:

* before each (re)connect it reads the account (``/v1/me``): an Oscar older than the server's ``min_version`` stays
  ``blocked`` with a message saying to update it (``account.require_supported``), and asks again every minute;
* it asks ``POST /v1/connect/next`` for work only while Oscar is idle: a turn the user started in FreeCAD always
  comes first. While Oscar is busy, ``hello`` every 20 s keeps this computer online for the website;
* during a workspace turn it posts the batched events every half second. An empty post every 10 s is the keep-alive,
  and its answer says whether the user pressed Stop in the browser;
* after a geometry change it uploads a preview mesh (at most one every 3 s); when the turn ends, the exports, then
  the reply (``finish``). The preview is a GLB with one named node per part (``preview_glb``: the browser shows,
  hides and selects each part by its FreeCAD object name, as the feature tree names it), or an STL of the whole
  model when the parts could not be meshed one by one.

Files the user attached in the browser are downloaded before the turn starts, each checked
against the size and sha256 the job announced, into ``<files_dir>/<task>/<message>/``. The turn gets them as local
paths, like files attached in FreeCAD: images (and the page images that come with a PDF) go to the model as
images, PDFs and text files as text, CAD files by path for the ``import_file`` tool. A file that can't be downloaded
is reported to the browser and the turn goes on without it.

Everything Oscar does in a workspace turn stays behind the same approvals as in FreeCAD: scripts need the codeguard
approval and network access its own. The browser sees each approval as one step that changes in place
(``approval_event``: waiting, then approved, declined or no answer). With ``REMOTE_PREF`` on (the default once the
person at this computer turned connect mode on; they can switch it off to answer here only), the step also offers
the dialog's answers in the browser, with the script it is about, and the browser's answer
(``Client.post_events_reply``) clicks that button in the open dialog. A lasting grant ("Always allow") is never
offered there.

What the browser shows of a step is decided here (``step_event``): plan bookkeeping is left out, a failed check is
summarised in plain words with the kind of problem it is (``issue``: ``design`` = the geometry needs a change,
``tool`` = Oscar's own call failed and it tries again, ``permission`` = something was declined in FreeCAD), and the
model's feature tree goes along after geometry changes (``Turn.take_tree``, a ``tree`` event), with the document's
parameter sheet (``params``).

A model built in the browser (the OriginShape engine) can be opened here: when the loop
was given ``open_model``, ``hello`` says so (the ``open_model`` capability) and ``/next`` may answer with an ``open``
job instead of a task message. The loop downloads that STEP file (checked against its size and sha256) into
``<files_dir>/models/<delivery>/``, has FreeCAD open it as a new document, and reports how that went. An Oscar without
``open_model`` never says it can, so the server never sends it one.

The browser can revert a task's turns: when the loop was given ``revert``, ``hello`` says
so (the ``revert`` capability) and ``/next`` may answer with a ``revert`` job: the task's prompts whose turns to undo,
newest first. FreeCAD undoes them with the panel's own revert (``revert(job)``), the loop uploads the model's preview
after it and reports which prompts' turns were undone, with the feature tree, and why not all of them if not.

A turn's answer always reaches the workspace. The loop never abandons a turn over one failed
call; the answer is written to an ``Outbox`` on disk before it is sent and removed only once the server has it. While
idle, the loop delivers what is left there and asks ``GET /v1/connect/unanswered`` which of its turns the website
closed without an answer (FreeCAD stopped reporting, or quit); it answers each from the outbox or, failing that, from
the task's chat in FreeCAD (``recover``). Text that leaves this computer has its local paths redacted (``redact``:
"Oscar project" instead of the project folder).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform as _platform
import re
import shutil
import struct
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlencode

from . import __version__
from .auth import account as _account
from .auth import transport
from .auth.transport import AuthError, AuthUnavailable, SignedOut

PREF = "ConnectWorkspace"          # params boolean: connect mode is on (off by default)
NEXT_WAIT_S = 25.0                 # the server holds /next at most 25 s
HEARTBEAT_S = 20.0                 # hello while busy, so the device stays online (the server's window is 60 s)
FLUSH_S = 0.5                      # how often a turn's events are posted
KEEPALIVE_S = 10.0                 # an empty events post: keeps the device online and asks about Stop
ANSWER_POLL_S = 1.0                # while a question waits for the browser: ask for its answer this often
PREVIEW_INTERVAL_S = 3.0           # at most one preview upload every 3 s
BACKOFF_S = (1.0, 2.0, 5.0, 10.0, 30.0)
BLOCKED_RETRY_S = 60.0             # waitlisted / suspended: ask again in a minute
SIGNED_OUT_RETRY_S = 5.0
FINISH_ATTEMPTS = 5
SYNC_S = 60.0                      # while idle: deliver the outbox and ask for unanswered turns this often
OUTBOX_KEPT_S = 7 * 86400          # an answer the server never took is dropped after a week
NO_ANSWER_TEXT = ("Oscar in FreeCAD has no answer for this message (the turn ended before it finished). "
                  "Send the message again to retry.")
TITLE_LIMIT, DETAIL_LIMIT, MESSAGE_LIMIT, REPLY_LIMIT = 300, 2000, 2000, 100_000
SCRIPT_LIMIT = 20_000              # the script an approval shows in the browser (the server's limit)
REMOTE_PREF = "ConnectApprovals"   # params boolean: the browser may answer approvals (on by default in connect mode)
REMOTE_DEFAULT = True
MAX_ATTACHMENTS = 50               # files in one job (10 per message, a PDF brings up to 4 pages)
DOWNLOAD_ATTEMPTS = 3
FILES_KEPT_S = 30 * 86400          # downloaded attachments older than this are removed when the loop starts
APPROVAL_WAIT_TEXT = "Waiting for your approval in FreeCAD"
STOPPED_TEXT = "Stopped."
HIDDEN_TOOLS = frozenset({"plan_set", "plan_update", "record_note"})   # plan bookkeeping: not a step of the build
CHECK_TOOLS = frozenset({"check_fit", "validate_model"})
MAX_PROBLEMS = 8
OPEN_MODEL = "open_model"          # the capability: this Oscar opens models the browser sends (section 12.2)
MODELS_DIR = "models"              # under files_dir: the models the browser sent, one folder per delivery
MODEL_FORMATS = frozenset({"step"})
OPEN_REPORT_ATTEMPTS = 3
REVERT = "revert"                  # the capability: this Oscar reverts turns the browser asks it to (section 14)
REVERT_REPORT_ATTEMPTS = 3
APPROVAL_TITLES = {
    "waiting": APPROVAL_WAIT_TEXT,
    "approved": "Approved in FreeCAD",
    "declined": "Declined in FreeCAD",
    "no_answer": "No answer in FreeCAD",
}
APPROVAL_REMOTE_WAIT_TEXT = "Waiting for your approval"   # answerable in the browser and in FreeCAD
APPROVAL_BROWSER_TITLES = {"approved": "Approved in the browser", "declined": "Declined in the browser"}
APPROVE_ROLES = frozenset({"YesRole", "AcceptRole"})


# --- errors ---------------------------------------------------------------------------------------------------


class ConnectError(Exception):
    """A connect call failed in a way the loop reports and retries (bad answer from the server)."""

    def __init__(self, message: str, *, code: str = "", status: int = 0) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class TurnGone(ConnectError):
    """The server no longer runs this turn: the task was deleted, or reset after a disconnect."""


class UploadRefused(ConnectError):
    """One upload was refused (too big, storage limit reached); the turn goes on without it."""


class TurnRefused(Exception):
    """FreeCAD could not start the turn (Oscar is busy, or the account can't send right now); the message is
    shown to the web user."""


# --- the job ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class JobFile:
    """A file attached to the job's message (``message.attachments``): download it with ``Client.download``."""

    id: str
    name: str
    format: str
    kind: str        # image | pdf | cad | text
    size: int
    sha256: str
    page: int = 0    # a page image of the PDF before it (1-based); 0 for the user's own files

    @classmethod
    def parse(cls, data: Any) -> JobFile:
        size, sha256 = int(data["size"]), str(data["sha256"]).lower()
        if size < 0 or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("size or sha256")
        return cls(id=str(data["id"]), name=str(data.get("name") or "attachment"), format=str(data.get("format") or ""),
                   kind=str(data.get("kind") or ""), size=size, sha256=sha256, page=int(data.get("page") or 0))


@dataclass(frozen=True)
class Job:
    """One user message to answer, as ``/v1/connect/next`` hands it out."""

    task_id: str
    title: str
    model: str
    message_id: str
    text: str
    history: tuple[tuple[str, str], ...]   # (role, text) of the task's earlier messages, oldest first
    turn: int
    attachments: tuple[JobFile, ...] = ()

    @classmethod
    def parse(cls, data: Any) -> Job:
        try:
            message = data["message"]
            history = tuple(
                (str(item["role"]), _with_names(str(item["text"]), item.get("attachments")))
                for item in data.get("history") or ()
                if item.get("role") in ("user", "assistant")
            )
            job = cls(
                task_id=str(data["task_id"]),
                title=str(data.get("title") or ""),
                model=str(data.get("model") or ""),
                message_id=str(message["id"]),
                text=str(message.get("text") or ""),
                history=history,
                turn=int(data.get("turn") or 1),
                attachments=tuple(JobFile.parse(item) for item in message.get("attachments") or ())[:MAX_ATTACHMENTS],
            )
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise ConnectError(f"The OriginShape workspace sent a job Oscar can't read ({type(e).__name__}).",
                               code="bad_job") from None
        if not job.task_id or not (job.text.strip() or job.attachments):
            raise ConnectError("The OriginShape workspace sent an empty job.", code="bad_job")
        return job


@dataclass(frozen=True)
class OpenModel:
    """A model the user sent from the browser to open in this FreeCAD (the ``open`` job of ``/v1/connect/next``):
    download it with ``Client.download_model``, open it, then say how that went with ``Client.model_opened``."""

    id: str
    task_id: str
    title: str
    name: str
    format: str
    size: int
    sha256: str

    @classmethod
    def parse(cls, data: Any) -> OpenModel:
        try:
            size, sha256 = int(data["size"]), str(data["sha256"]).lower()
            model = cls(id=str(data["id"]), task_id=str(data.get("task_id") or ""), title=str(data.get("title") or ""),
                        name=str(data.get("name") or "model.step"), format=str(data.get("format") or ""), size=size,
                        sha256=sha256)
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise ConnectError(f"The OriginShape workspace sent a model Oscar can't read ({type(e).__name__}).",
                               code="bad_job") from None
        if not model.id or size <= 0 or not re.fullmatch(r"[0-9a-f]{64}", sha256) or model.format not in MODEL_FORMATS:
            raise ConnectError("The OriginShape workspace sent a model Oscar can't read.", code="bad_job")
        return model


@dataclass(frozen=True)
class RevertJob:
    """The user taking a task back to just before one of their prompts (the ``revert`` job of
    ``/v1/connect/next``): undo the turns of ``messages`` (message ids, newest first) in
    FreeCAD, then say which were with ``Client.revert_done``."""

    id: str
    task_id: str
    title: str
    messages: tuple[str, ...]

    @classmethod
    def parse(cls, data: Any) -> RevertJob:
        try:
            job = cls(id=str(data["id"]), task_id=str(data["task_id"]), title=str(data.get("title") or ""),
                      messages=tuple(str(item["id"]) for item in data["messages"]))
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise ConnectError(f"The OriginShape workspace sent a revert Oscar can't read ({type(e).__name__}).",
                               code="bad_job") from None
        if not job.id or not job.task_id or not job.messages:
            raise ConnectError("The OriginShape workspace sent a revert Oscar can't read.", code="bad_job")
        return job


@dataclass(frozen=True)
class RevertOutcome:
    """What a revert did in FreeCAD: the prompts whose turns are undone (message ids), and why not all of them if
    not (``message``). ``preview`` (``(format, bytes)``) and ``tree`` are the model after it, when there is one."""

    reverted: tuple[str, ...]
    message: str = ""
    preview: tuple[str, bytes] | None = None
    tree: dict[str, Any] | None = None


def _with_names(text: str, names: Any) -> str:
    """An earlier message's text with the names of the files it carried (the files themselves are not resent)."""
    if not isinstance(names, list) or not names:
        return text
    note = "[attached: " + ", ".join(_clip(name, 200) for name in names[:20]) + "]"
    return f"{text}\n\n{note}" if text.strip() else note


# --- the protocol client ----------------------------------------------------------------------------------------


class Client:
    """``/v1/connect/*`` over one signed-in ``AuthSession`` (``oscar.auth.session``). ``SignedOut`` propagates;
    ``AuthUnavailable`` means try again later; other refusals are ``AuthError`` (``code`` from the server)."""

    WHAT = "the OriginShape workspace"

    def __init__(self, session: Any, redact: Callable[[str], str] | None = None) -> None:
        self.session = session
        self._redact = redact

    def _out(self, value: Any) -> Any:
        """``value`` as it may leave this computer: every string redacted (``redact``)."""
        if self._redact is None:
            return value
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, list):
            return [self._out(item) for item in value]
        if isinstance(value, dict):
            return {key: self._out(item) for key, item in value.items()}
        return value

    def _call(self, method: str, path: str, *, timeout: transport.TimeoutSpec = (5.0, 20.0), **body: Any
              ) -> transport.Reply:
        reply = self.session.authorized_request(method, path, timeout=timeout, **body)
        if 200 <= reply.status < 300:
            return reply
        if reply.status in (404, 409) and path.startswith("/v1/connect/tasks/"):
            raise TurnGone("This task is no longer running in the OriginShape workspace.", status=reply.status)
        raise transport.api_error(reply, self.WHAT)

    def hello(self, versions: dict[str, str]) -> None:
        self._call("POST", "/v1/connect/hello", json_body=versions)

    def next_job(self, wait: float = NEXT_WAIT_S) -> Job | OpenModel | RevertJob | None:
        """The next task message to answer, a revert or a model to open (only for an Oscar that said it can), or
        None."""
        seconds = max(0, min(int(wait), int(NEXT_WAIT_S)))
        reply = self._call("POST", f"/v1/connect/next?wait={seconds}", timeout=(5.0, seconds + 15.0))
        if reply.status == 204:
            return None
        data = reply.json()
        if isinstance(data, dict) and isinstance(data.get("revert"), dict):
            return RevertJob.parse(data["revert"])
        if isinstance(data, dict) and isinstance(data.get("open"), dict):
            return OpenModel.parse(data["open"])
        if not isinstance(data, dict) or not isinstance(data.get("job"), dict):
            raise ConnectError("The OriginShape workspace answered with something Oscar can't read.", code="bad_job")
        return Job.parse(data["job"])

    def post_events(self, task_id: str, events: list[dict[str, Any]]) -> bool:
        """Post ``events`` (possibly none: the keep-alive); returns whether the browser asked the turn to stop."""
        return self.post_events_reply(task_id, events)[0]

    def post_events_reply(self, task_id: str, events: list[dict[str, Any]]) -> tuple[bool, list[tuple[str, str]]]:
        """``post_events``, with the browser's answers to approvals as well: ``(cancel, [(key, choice), ...])``. An
        answer comes with every reply until the approval's outcome is posted."""
        reply = self._call("POST", f"/v1/connect/tasks/{task_id}/events", json_body={"events": self._out(events)})
        data = reply.json()
        if not isinstance(data, dict):
            return False, []
        answers = [(str(item["key"]), str(item["choice"])) for item in data.get("answers") or ()
                   if isinstance(item, dict) and isinstance(item.get("key"), str) and isinstance(item.get("choice"), str)]
        return bool(data.get("cancel")), answers

    def upload(self, task_id: str, *, kind: str, fmt: str, name: str, version: int, data: bytes) -> None:
        query = urlencode({"kind": kind, "format": fmt, "name": name, "version": max(0, int(version))})
        path = f"/v1/connect/tasks/{task_id}/files?{query}"
        reply = self.session.authorized_request("PUT", path, timeout=(5.0, 60.0), content=data)
        if 200 <= reply.status < 300:
            return
        if reply.status == 404:
            raise TurnGone("This task is no longer running in the OriginShape workspace.", status=404)
        if reply.status in (400, 409, 413):   # too big, wrong format, storage limit: this file only
            code, description = transport.error_details(reply.body)
            raise UploadRefused(description or f"The upload was refused (HTTP {reply.status}).", code=code,
                                status=reply.status)
        raise transport.api_error(reply, self.WHAT)

    def finish(self, task_id: str, text: str, ok: bool, message_id: str = "") -> None:
        """The turn's answer. With ``message_id`` the server also takes it after it closed the turn (section 11)."""
        body: dict[str, Any] = {"text": self._out(text)[:REPLY_LIMIT], "ok": ok}
        if message_id:
            body["message_id"] = message_id
        self._call("POST", f"/v1/connect/tasks/{task_id}/finish", json_body=body)

    def unanswered(self) -> list[tuple[str, str]]:
        """``(task_id, message_id)`` of this computer's turns the website closed without an answer; empty from a
        server that doesn't know the call."""
        try:
            reply = self._call("GET", "/v1/connect/unanswered")
        except (AuthUnavailable, SignedOut):
            raise
        except AuthError as e:
            if e.status in (404, 405):
                return []
            raise
        data = reply.json()
        turns = data.get("turns") if isinstance(data, dict) else None
        return [(str(t["task_id"]), str(t["message_id"])) for t in turns or ()
                if isinstance(t, dict) and t.get("task_id") and t.get("message_id")]

    def download(self, item: JobFile, dest: str) -> None:
        """The attachment ``item`` into the file ``dest``, checked against its announced size and sha256. Raises
        ``ConnectError`` when it is gone or not what was announced; ``AuthUnavailable`` means try again."""
        self._fetch(f"/v1/connect/attachments/{item.id}", dest, item.size, item.sha256, "the attached file")

    def download_model(self, item: OpenModel, dest: str) -> None:
        """The model of an ``open`` job into the file ``dest``, checked like an attachment."""
        self._fetch(f"/v1/connect/deliveries/{item.id}/file", dest, item.size, item.sha256, "the model")

    def model_opened(self, delivery_id: str, ok: bool, message: str = "", document: str = "") -> bool:
        """Report on a model this FreeCAD was sent: opened (as ``document``) or not (``message``). False when the
        server no longer waits for this report (it gave up on it, or it was already made)."""
        body = {"ok": bool(ok), "message": self._out(message)[:2000], "document": document[:200]}
        reply = self.session.authorized_request("POST", f"/v1/connect/deliveries/{delivery_id}/done",
                                                timeout=(5.0, 20.0), json_body=body)
        if 200 <= reply.status < 300:
            return True
        if reply.status == 404:
            return False
        raise transport.api_error(reply, self.WHAT)

    def upload_revert_preview(self, task_id: str, revert_id: str, fmt: str, data: bytes) -> None:
        """The model's preview after a revert (it goes with the revert, not with a turn)."""
        query = urlencode({"kind": "preview", "format": fmt, "name": f"preview-revert.{fmt}", "revert": revert_id})
        reply = self.session.authorized_request("PUT", f"/v1/connect/tasks/{task_id}/files?{query}",
                                                timeout=(5.0, 60.0), content=data)
        if not 200 <= reply.status < 300:
            raise transport.api_error(reply, self.WHAT)

    def revert_done(self, revert_id: str, ok: bool, reverted: list[str], message: str = "",
                    tree: dict[str, Any] | None = None) -> bool:
        """Report on a revert this FreeCAD took: the prompts whose turns it undid (newest first), why not all of
        them if not, and the model's tree now. False when the server no longer waits for this report."""
        body: dict[str, Any] = {"ok": bool(ok), "reverted": list(reverted), "message": self._out(message)[:2000]}
        if tree is not None:
            body["tree"] = self._out(tree)
        reply = self.session.authorized_request("POST", f"/v1/connect/reverts/{revert_id}/done",
                                                timeout=(5.0, 20.0), json_body=body)
        if 200 <= reply.status < 300:
            return True
        if reply.status in (404, 409):
            return False
        raise transport.api_error(reply, self.WHAT)

    def _fetch(self, path: str, dest: str, size: int, sha256: str, what: str) -> None:
        try:
            reply, _saved = self.session.authorized_download(path, dest, max_bytes=size, sha256=sha256)
        except (AuthUnavailable, SignedOut):
            raise
        except AuthError as e:   # too large or damaged: downloading it again gives the same file
            raise ConnectError(str(e), code=e.code, status=e.status) from None
        if reply.status == 200:
            return
        if reply.status == 404:
            raise ConnectError("It is no longer in the OriginShape workspace.", code="not_found", status=404)
        raise transport.api_error(reply, what)


# --- answers on their way --------------------------------------------------------------------------------------


class Outbox:
    """Turn answers not yet taken by the server, in a JSON file (atomic writes; thread-safe): ``{task_id,
    message_id, text, ok, at}``. Written when a turn ends, removed when ``finish`` succeeds or the server has no
    use for it; what is left is delivered later, after a reconnect or a restart of FreeCAD."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> list[dict[str, Any]]:
        try:
            with open(self.path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return []
        return [e for e in data if isinstance(e, dict) and e.get("task_id") and e.get("message_id")] \
            if isinstance(data, list) else []

    def _write(self, entries: list[dict[str, Any]]) -> None:
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(entries, handle)
            os.replace(tmp, self.path)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def add(self, task_id: str, message_id: str, text: str, ok: bool) -> None:
        with self._lock:
            entries = [e for e in self._read() if (e["task_id"], e["message_id"]) != (task_id, message_id)]
            entries.append({"task_id": task_id, "message_id": message_id, "text": text[:REPLY_LIMIT], "ok": bool(ok),
                            "at": time.time()})
            self._write(entries)

    def remove(self, task_id: str, message_id: str) -> None:
        with self._lock:
            entries = self._read()
            kept = [e for e in entries if (e["task_id"], e["message_id"]) != (task_id, message_id)]
            if len(kept) != len(entries):
                self._write(kept)

    def get(self, task_id: str, message_id: str) -> dict[str, Any] | None:
        with self._lock:
            return next((e for e in self._read() if (e["task_id"], e["message_id"]) == (task_id, message_id)), None)

    def entries(self, *, now: float | None = None) -> list[dict[str, Any]]:
        """What is waiting, oldest first; entries older than ``OUTBOX_KEPT_S`` are dropped."""
        now = time.time() if now is None else now
        with self._lock:
            entries = self._read()
            kept = [e for e in entries if now - float(e.get("at") or 0) <= OUTBOX_KEPT_S]
            if len(kept) != len(entries):
                self._write(kept)
            return kept


# --- events -----------------------------------------------------------------------------------------------------


class EventBatch:
    """Events waiting to be posted, in order. Thread-safe. The reply is the whole text so far: a newer one replaces
    the older (the server keeps only the newest too), and it is posted after the other events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._reply: str | None = None

    def progress(self, title: str, *, detail: str = "", phase: str = "", ok: bool | None = None,
                 **extra: Any) -> None:
        """A step. ``extra`` carries the optional fields of ``step_event`` / ``approval_event`` (``kind``, ``key``,
        ``state``, ``issue``, ``items``, ``counts``)."""
        data: dict[str, Any] = {"title": _clip(title, TITLE_LIMIT) or "Working"}
        if detail:
            data["detail"] = _clip(detail, DETAIL_LIMIT)
        if phase:
            data["phase"] = _clip(phase, 40)
        if ok is not None:
            data["ok"] = bool(ok)
        data.update({key: value for key, value in extra.items() if value not in (None, "", [], {})})
        self._add({"kind": "progress", "data": data})

    def step(self, event: dict[str, Any] | None) -> None:
        """A ``step_event`` / ``approval_event`` (None: nothing to show)."""
        if event:
            fields = dict(event)
            self.progress(fields.pop("title", ""), detail=fields.pop("detail", ""), phase=fields.pop("phase", ""),
                          ok=fields.pop("ok", None), **fields)

    def reply(self, text: str) -> None:
        with self._lock:
            self._reply = text[:REPLY_LIMIT]

    def title(self, text: str) -> None:
        if text.strip():
            self._add({"kind": "title", "data": {"title": _clip(text, 200)}})

    def error(self, message: str) -> None:
        self._add({"kind": "error", "data": {"message": _clip(message, MESSAGE_LIMIT) or "Something went wrong."}})

    def _add(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)

    def drain(self) -> list[dict[str, Any]]:
        """Everything waiting, oldest first (the reply last), at most 50 (the server's limit); the rest stays."""
        with self._lock:
            events = self._events[:49]
            del self._events[:49]
            if self._reply is not None and not self._events:
                events.append({"kind": "reply", "data": {"text": self._reply}})
                self._reply = None
            return events

    def requeue(self, events: list[dict[str, Any]]) -> None:
        """Put back events whose post failed, in front of the newer ones."""
        with self._lock:
            replies = [e for e in events if e["kind"] == "reply"]
            self._events[:0] = [e for e in events if e["kind"] != "reply"]
            if replies and self._reply is None:
                self._reply = replies[-1]["data"]["text"]


def _clip(text: Any, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


# --- what the browser shows of a step ---------------------------------------------------------------------------

_ROW = re.compile(r"^(PASS|WARN|FAIL|SKIP)\s+([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)\s+(.*)$")   # dotted check ids
_FINDING = re.compile(r"^ERROR\s+\[([A-Z0-9_]+)\]\s+(.*)$")


def _humanize(name: str) -> str:
    text = " ".join(str(name or "").replace("_", " ").replace(".", " ").split())
    return text[:1].upper() + text[1:] if text else "Step"


def _check_name(rid: str) -> str:
    """``interference.reference_shell`` -> ``Interference (reference shell)``."""
    head, _, rest = rid.partition(".")
    return f"{_humanize(head)} ({_humanize(rest).lower()})" if rest else _humanize(head)


def _sentence(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:]


def _units(text: str) -> str:
    return re.sub(r"\bmm3\b", "mm³", re.sub(r"\bmm2\b", "mm²", text))


def check_results(text: str) -> tuple[dict[str, int], list[str]]:
    """``(counts, problems)`` of a ``check_fit`` / ``validate_model`` report: how many rows passed, warned, failed
    or were skipped, and each failure as one plain sentence (``Interference, reference shell: the reference
    overlaps Shell by 1.38 mm³``)."""
    counts = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
    problems: list[str] = []
    for line in str(text or "").splitlines():
        line = line.strip()
        row = _ROW.match(line)
        if row:
            status, rid, rest = row.groups()
            counts[status.lower()] += 1
            if status == "FAIL":
                head, _, why = rest.partition(" - ")
                problems.append(f"{_check_name(rid)}: {_units(why or head).strip()}")
            continue
        finding = _FINDING.match(line)
        if finding:
            counts["fail"] += 1
            code, message = finding.groups()
            problems.append(f"{_humanize(code.lower())}: {_units(message).strip()}")
    return ({k: v for k, v in counts.items() if v} if any(counts.values()) else {}), \
        [_clip(problem, 300) for problem in problems]


def _first_sentence(text: str, name: str) -> str:
    line = next((part.strip() for part in str(text or "").splitlines() if part.strip()), "")
    if line.lower().startswith(f"{name.lower()}:"):
        line = line[len(name) + 1:].strip()
    end = re.search(r"(?<=[.!?])\s", line)
    return _clip(line[: end.start()] if end else line, 240)


def step_event(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The progress event the browser shows for a finished step (the worker's ``step_finished`` payload), or None
    for one it doesn't show (skipped steps, plan bookkeeping)."""
    if not isinstance(payload, dict) or payload.get("skipped"):
        return None
    name = str(payload.get("name") or "")
    if name in HIDDEN_TOOLS:
        return None
    ok = bool(payload.get("ok"))
    title = str(payload.get("title") or "").strip()
    if not title or title == name:
        title = _humanize(name)
    text = str(payload.get("text") or "")
    event: dict[str, Any] = {"title": title, "ok": ok}
    if name in CHECK_TOOLS:
        counts, problems = check_results(text)
        event["kind"] = "check"
        if counts:
            event["counts"] = counts
        if not ok and problems:
            n = len(problems)
            event.update(title=f"{title}: {n} problem{'' if n == 1 else 's'} found", issue="design",
                         items=problems[:MAX_PROBLEMS],
                         detail="The design needs a change. Oscar adjusts the model and checks again.")
            return event
    if ok:
        return event
    lowered = text.lower()
    if "declined" in lowered or "nobody answered" in lowered or "denied" in lowered:
        event.update(issue="permission", detail="Not allowed in FreeCAD, so Oscar continues without it.")
    else:
        event.update(issue="tool", detail=_sentence(_first_sentence(text, name)) or "This step didn’t work.")
    return event


def approval_event(approval_id: Any, state: str, what: str, *, choices: Any = (), script: str = "",
                   by: str = "") -> dict[str, Any]:
    """One approval asked at this computer, as a step the browser updates in place (same ``key``): ``state`` is
    ``waiting``, ``approved``, ``declined`` or ``no_answer``; ``what`` says what was asked.

    ``choices`` (``[{"key", "label", "role"}]``, given only when the browser may answer) offer the question in the
    browser too, with the ``script`` it is about; ``by`` ("browser" or "freecad") says who settled it."""
    state = state if state in APPROVAL_TITLES else "no_answer"
    offered = [{"key": str(item["key"]), "label": _clip(item.get("label") or item["key"], 120),
                "tone": "approve" if item.get("role") in APPROVE_ROLES else "decline"}
               for item in choices or () if isinstance(item, dict) and item.get("key")][:4]
    remote = state == "waiting" and bool(offered)
    if remote:
        title = APPROVAL_REMOTE_WAIT_TEXT
    elif by == "browser" and state in APPROVAL_BROWSER_TITLES:
        title = APPROVAL_BROWSER_TITLES[state]
    else:
        title = APPROVAL_TITLES[state]
    event: dict[str, Any] = {"title": title, "detail": _clip(what, DETAIL_LIMIT), "kind": "approval",
                             "key": f"approval-{approval_id}", "state": state,
                             **({} if state == "waiting" else {"ok": state == "approved"})}
    if remote:
        event["choices"] = offered
        if script:
            event["script"] = script[:SCRIPT_LIMIT]
    if state != "waiting" and by in ("browser", "freecad"):
        event["by"] = by
    return event


def _unique(name: str, taken: set[str]) -> str:
    """``name``, or ``stem (2).ext`` ... when another file of the message already has it (case-insensitively)."""
    stem, ext = os.path.splitext(name)
    candidate, n = name, 1
    while candidate.lower() in taken:
        n += 1
        candidate = f"{stem} ({n}){ext}"
    taken.add(candidate.lower())
    return candidate


# --- the FreeCAD side's part --------------------------------------------------------------------------------------


class Turn(Protocol):
    """A workspace turn running in FreeCAD (``connect_gui`` implements it). Every method may be called from the
    loop's thread."""

    events: EventBatch

    def wait(self, timeout: float) -> bool:
        """Block up to ``timeout`` seconds; True once the turn has ended."""

    def stop(self) -> None:
        """Ask the turn to stop (the browser pressed Stop, or the task is gone)."""

    def result(self) -> tuple[bool, str]:
        """``(ok, reply text)`` once ended."""

    def take_preview(self) -> tuple[int, bytes] | tuple[int, bytes, str] | None:
        """The model when it changed since the last call: ``(step number, bytes)`` of a binary STL, or ``(step
        number, bytes, format)`` with ``format`` ``"glb"`` or ``"stl"``; else None."""

    def exports(self) -> list[tuple[str, str, bytes]]:
        """``(file name, format, bytes)`` of the finished model (STEP, STL, FCStd); empty without geometry."""

    # Optional: ``take_tree() -> dict | None``, the document's feature tree when it changed since the last call
    # (``{"document", "objects": [...]}``).
    # Optional: ``awaiting_answer() -> bool``, a question offered to the browser is waiting (the loop then asks for
    # answers every ``ANSWER_POLL_S``), and ``answer(key, choice)``, the browser's answer to it.


# --- the preview mesh -------------------------------------------------------------------------------------------

PREVIEW_FORMATS = ("glb", "stl")
# FreeCAD is Z up, glTF is Y up: the root node turns the model a quarter turn about X. Units stay millimetres.
_Z_UP_TO_Y_UP = [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
_FLOAT, _UINT32 = 5126, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963


@dataclass(frozen=True)
class PreviewPart:
    """One part of the preview: ``name`` is the FreeCAD object name (what the tree calls it), ``positions`` the
    vertices as flat x, y, z in mm (global frame, Z up) and ``indices`` three per triangle."""

    name: str
    label: str
    positions: tuple[float, ...] | list[float]
    indices: tuple[int, ...] | list[int]
    color: tuple[float, float, float] | None = None   # linear RGB 0-1; None: the viewer's own colour


def preview_glb(parts: list[PreviewPart], document: str = "") -> bytes:
    """A binary glTF 2.0 (GLB) with one node per part, named by its FreeCAD object name (``extras`` carry the name
    and label too). Every part is one mesh with one triangle primitive: positions and uint32 indices, no normals
    (the browser computes them). Parts without triangles are left out. Pure python: no FreeCAD needed."""
    nodes: list[dict[str, Any]] = [{"name": _clip(document, 120) or "Model", "rotation": _Z_UP_TO_Y_UP,
                                    "children": []}]
    meshes: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    blob = bytearray()

    def view(data: bytes, target: int) -> int:
        blob.extend(b"\0" * (-len(blob) % 4))
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data)
        return len(views) - 1

    for part in parts:
        positions, indices = list(part.positions), list(part.indices)
        count = len(positions) // 3
        if count == 0 or len(indices) < 3 or len(indices) % 3 or any(not 0 <= i < count for i in indices):
            continue
        packed = struct.pack(f"<{count * 3}f", *positions[:count * 3])
        stored = struct.unpack(f"<{count * 3}f", packed)   # min / max of the float32 values, as glTF requires
        xs, ys, zs = stored[0::3], stored[1::3], stored[2::3]
        position = len(accessors)
        accessors.append({"bufferView": view(packed, _ARRAY_BUFFER),
                          "componentType": _FLOAT, "count": count, "type": "VEC3",
                          "min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]})
        accessors.append({"bufferView": view(struct.pack(f"<{len(indices)}I", *indices), _ELEMENT_ARRAY_BUFFER),
                          "componentType": _UINT32, "count": len(indices), "type": "SCALAR"})
        primitive: dict[str, Any] = {"attributes": {"POSITION": position}, "indices": position + 1, "mode": 4}
        if part.color is not None:
            primitive["material"] = len(materials)
            materials.append({"name": _clip(part.label, 120) or part.name, "pbrMetallicRoughness": {
                "baseColorFactor": [*(max(0.0, min(1.0, float(c))) for c in part.color), 1.0],
                "metallicFactor": 0.08, "roughnessFactor": 0.62}})
        meshes.append({"name": part.name, "primitives": [primitive]})
        nodes[0]["children"].append(len(nodes))
        nodes.append({"name": part.name, "mesh": len(meshes) - 1,
                      "extras": {"name": part.name, "label": _clip(part.label, 200) or part.name}})

    if not nodes[0]["children"]:
        del nodes[0]["children"]
    gltf: dict[str, Any] = {"asset": {"version": "2.0", "generator": f"Oscar for FreeCAD {__version__}"},
                            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes}
    if meshes:
        blob.extend(b"\0" * (-len(blob) % 4))
        gltf.update(meshes=meshes, accessors=accessors, bufferViews=views, buffers=[{"byteLength": len(blob)}])
    if materials:
        gltf["materials"] = materials
    text = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    text += b" " * (-len(text) % 4)
    body = struct.pack("<I4s", len(text), b"JSON") + text
    if blob:
        body += struct.pack("<I4s", len(blob), b"BIN\0") + bytes(blob)
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


# --- the loop -----------------------------------------------------------------------------------------------------


_UNSAFE_NAME = re.compile(r'[\x00-\x1f<>:"/\\|?*]')
_RESERVED_NAMES = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def safe_filename(name: str) -> str:
    """``name`` usable as a file name on Windows, macOS and Linux: no folders, reserved characters or device names,
    no leading dot, at most 120 characters (the extension kept)."""
    base = _UNSAFE_NAME.sub("_", re.split(r"[\\/]", name or "")[-1]).strip().strip(".").strip()
    stem, ext = os.path.splitext(base)
    if stem.lower() in _RESERVED_NAMES or not stem:
        stem = f"file-{stem}" if stem else "attachment"
    ext = ext[:16]
    return stem[: 120 - len(ext)] + ext


def _path_part(value: str) -> str:
    """A server id as one folder name (ids are UUIDs; anything else is hashed)."""
    if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        return value
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def default_files_dir() -> str:
    return os.path.join(tempfile.gettempdir(), "oscar-workspace-files")


def prune_files(base: str, max_age_s: float = FILES_KEPT_S, *, now: float | None = None) -> int:
    """Remove the task folders under ``base`` that nothing was written to for ``max_age_s``; returns how many."""
    now = time.time() if now is None else now
    removed = 0
    try:
        entries = list(os.scandir(base))
    except OSError:
        return 0
    for entry in entries:
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
            newest = max([entry.stat().st_mtime] + [os.path.getmtime(os.path.join(root, name))
                                                    for root, dirs, files in os.walk(entry.path)
                                                    for name in dirs + files])
            if now - newest > max_age_s:
                shutil.rmtree(entry.path, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def platform_name() -> str:
    system = _platform.system()
    return {"Darwin": "macOS"}.get(system, system or "unknown")


def versions(freecad_version: str) -> dict[str, str]:
    """What ``hello`` reports: this Oscar, the FreeCAD it runs in, the operating system."""
    return {"oscar_version": __version__, "freecad_version": freecad_version[:40], "platform": platform_name()}


class Loop:
    """The connect-mode thread. ``session_for()`` returns the signed-in ``AuthSession`` (or None), ``is_idle()``
    tells whether Oscar may take work now, ``start_turn(job, files)`` starts it in FreeCAD with the paths of the
    downloaded attachments (raising ``TurnRefused``) and ``on_state(state, message)`` reports ``connecting`` /
    ``connected`` / ``working`` / ``offline`` / ``signed_out`` / ``blocked`` (called on the loop's thread).
    ``files_dir()`` is where attachments are downloaded to. ``open_model(model, path)``, when given, opens a model
    the browser sent (a downloaded STEP file) in FreeCAD and returns ``(ok, message, document)``; with it, ``hello``
    tells the server this Oscar can. So does ``revert(job)``, which undoes a revert job's turns in FreeCAD and
    returns a ``RevertOutcome``."""

    def __init__(self, session_for: Callable[[], Any], *, start_turn: Callable[[Job, list[str]], Turn],
                 is_idle: Callable[[], bool], on_state: Callable[[str, str], None], versions: dict[str, str],
                 log: Callable[..., None] | None = None, files_dir: Callable[[], str] = default_files_dir,
                 outbox: Outbox | None = None,
                 recover: Callable[[str, str], tuple[str, bool, str]] | None = None,
                 redact: Callable[[str], str] | None = None,
                 open_model: Callable[[OpenModel, str], tuple[bool, str, str]] | None = None,
                 revert: Callable[[RevertJob], RevertOutcome] | None = None) -> None:
        """``outbox`` keeps answers until the server has them; ``recover(task_id, message_id)`` answers a turn the
        website closed from FreeCAD's own chat: ``("found", ok, text)``, ``("busy", ...)`` while FreeCAD still
        works on it, or ``("none", ...)``; ``redact`` cleans text that leaves this computer."""
        self._outbox = outbox
        self._recover = recover
        self._redact = redact
        self._session_for = session_for
        self._start_turn = start_turn
        self._files_dir = files_dir
        self._is_idle = is_idle
        self._on_state = on_state
        self._open_model = open_model
        self._revert_turns = revert
        self._versions: dict[str, Any] = dict(versions)
        capabilities = [name for name, handler in ((OPEN_MODEL, open_model), (REVERT, revert)) if handler is not None]
        if capabilities:
            self._versions["capabilities"] = capabilities
        self._log = log or (lambda *args, **kwargs: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = ""

    # lifecycle ----------------------------------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="oscar-connect", daemon=True)
        self._thread.start()

    def stop(self, join_s: float = 0.0) -> None:
        """Stop polling. A running workspace turn is NOT stopped here (the panel owns it); its report ends early."""
        self._stop.set()
        thread = self._thread
        if join_s > 0 and thread is not None and thread is not threading.current_thread():
            thread.join(join_s)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    # the thread ---------------------------------------------------------------------------------------------------

    def _state_is(self, state: str, message: str = "") -> None:
        if state != self._state or message:
            self._state = state
            try:
                self._on_state(state, message)
            except Exception as e:  # noqa: BLE001 - a failing listener must not end the loop
                self._log("connect_state_error", error=type(e).__name__)

    def _run(self) -> None:
        self._state_is("connecting")
        try:
            pruned = prune_files(self._files_dir())
            if pruned:
                self._log("connect_files_pruned", folders=pruned)
        except Exception as e:  # noqa: BLE001 - housekeeping never stops the loop
            self._log("connect_prune_error", error=type(e).__name__)
        failures = 0
        greeted = False
        last_hello = 0.0
        last_sync = float("-inf")
        while not self._stop.is_set():
            try:
                session = self._session_for()
                if session is None or not session.has_session() or session.kind != "oauth":
                    greeted = False
                    self._state_is("signed_out", "Sign in with your OriginShape account to connect to the workspace.")
                    self._stop.wait(SIGNED_OUT_RETRY_S)
                    continue
                client = Client(session, redact=self._redact)
                idle = self._is_idle()
                if not greeted or (not idle and time.monotonic() - last_hello >= HEARTBEAT_S):
                    if not greeted:
                        last_sync = float("-inf")   # a (re)connect: catch up at once
                        # Too old for this server: blocked. Read afresh: the server may have changed its minimum.
                        _account.require_supported(session.account(force=True))
                    client.hello(self._versions)
                    greeted, last_hello = True, time.monotonic()
                    self._state_is("connected")
                if not idle:
                    self._stop.wait(1.0)
                    continue
                if time.monotonic() - last_sync >= SYNC_S:
                    last_sync = time.monotonic()
                    self._sync(client)
                job = client.next_job(NEXT_WAIT_S)
                last_hello = time.monotonic()   # /next counts as being seen
                failures = 0
                self._state_is("connected")
                if isinstance(job, RevertJob) and not self._stop.is_set():
                    self._revert(client, job)
                elif isinstance(job, OpenModel) and not self._stop.is_set():
                    self._open(client, job)
                elif job is not None and not self._stop.is_set():
                    self._run_job(client, job)
            except SignedOut as e:
                greeted = False
                self._state_is("signed_out", str(e))
                self._stop.wait(SIGNED_OUT_RETRY_S)
            except AuthUnavailable as e:
                greeted = False
                failures += 1
                self._state_is("offline", str(e))
                self._stop.wait(BACKOFF_S[min(failures, len(BACKOFF_S)) - 1])
            except AuthError as e:
                greeted = False
                if getattr(e, "code", "") in ("waitlisted", "suspended", "forbidden", _account.PLUGIN_OUTDATED):
                    self._state_is("blocked", str(e))
                    self._stop.wait(BLOCKED_RETRY_S)
                else:
                    failures += 1
                    self._state_is("offline", str(e))
                    self._stop.wait(BACKOFF_S[min(failures, len(BACKOFF_S)) - 1])
            except Exception as e:  # noqa: BLE001 - the loop outlives any single failure
                greeted = False
                failures += 1
                self._log("connect_error", error=type(e).__name__, message=str(e)[:200])
                self._state_is("offline", f"{type(e).__name__}: {e}")
                self._stop.wait(BACKOFF_S[min(failures, len(BACKOFF_S)) - 1])
        self._state_is("stopped")

    def _run_job(self, client: Client, job: Job) -> None:
        self._log("connect_job", task=job.task_id, turn=job.turn, files=len(job.attachments))
        try:
            files, problems = self._download(client, job)
        except TurnGone:
            return
        problem_events = [{"kind": "error", "data": {"message": _clip(text, MESSAGE_LIMIT)}} for text in problems]
        if job.attachments and not files and not job.text.strip():
            self._finish(client, job.task_id, "Oscar couldn’t download the attached files, so there was nothing to "
                         "work on. Attach them again.", ok=False, events=problem_events)
            return
        try:
            turn = self._start_turn(job, files)
        except TurnRefused as e:
            self._finish(client, job.task_id, str(e) or "Oscar couldn't start this task.", ok=False,
                         events=[*problem_events, {"kind": "error", "data": {"message": _clip(e, MESSAGE_LIMIT)}}])
            return
        for event in problem_events:
            turn.events.error(event["data"]["message"])
        self._state_is("working")
        try:
            self._report(client, job, turn)
        except TurnGone:
            turn.stop()
            turn.wait(30.0)
            self._log("connect_turn_gone", task=job.task_id)
        except SignedOut:
            self._keep_answer(job, turn)   # delivered after the next sign-in
            raise
        except Exception as e:  # noqa: BLE001 - a bug while reporting must never cost the turn its answer
            self._log("connect_report_error", task=job.task_id, error=f"{type(e).__name__}: {e}"[:200])
            self._keep_answer(job, turn)
            entry = self._outbox.get(job.task_id, job.message_id) if self._outbox is not None else None
            if entry is not None:
                self._finish(client, job.task_id, entry["text"], ok=entry["ok"], events=[],
                             message_id=job.message_id)
        finally:
            self._state_is("connected")

    def _open(self, client: Client, model: OpenModel) -> None:
        """Download a model the browser sent, have FreeCAD open it, and say how that went. ``SignedOut``
        propagates (the server gives up on the model after a while and tells the browser)."""
        self._log("connect_open_model", delivery=model.id, task=model.task_id, size=model.size)
        folder = os.path.join(self._files_dir(), MODELS_DIR, _path_part(model.id))
        dest = os.path.join(folder, safe_filename(model.name) or "model.step")
        problem = ""
        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                os.makedirs(folder, exist_ok=True)
                client.download_model(model, dest)
                problem = ""
                break
            except SignedOut:
                raise
            except OSError as e:
                problem = f"Oscar couldn’t save the model on this computer ({type(e).__name__})."
                break
            except (ConnectError, AuthError) as e:
                problem = f"Oscar couldn’t download the model: {e}"
                if not isinstance(e, AuthUnavailable) or attempt == DOWNLOAD_ATTEMPTS - 1:
                    break
                if self._stop.wait(BACKOFF_S[attempt]):
                    break
        ok, message, document = False, problem, ""
        if not problem:
            if self._open_model is None:
                message = "This Oscar for FreeCAD can’t open models from the browser."
            else:
                self._state_is("working")
                try:
                    ok, message, document = self._open_model(model, dest)
                except Exception as e:  # noqa: BLE001 - the browser gets a message, the log the reason
                    self._log("connect_open_model_error", delivery=model.id, error=f"{type(e).__name__}: {e}"[:200])
                    ok, message, document = False, f"FreeCAD couldn’t open the model ({type(e).__name__}).", ""
                finally:
                    self._state_is("connected")
        self._log("connect_model_opened" if ok else "connect_model_not_opened", delivery=model.id,
                  message=_clip(message, 200))
        for attempt in range(OPEN_REPORT_ATTEMPTS):
            try:
                client.model_opened(model.id, ok, _clip(message, MESSAGE_LIMIT), _clip(document, 200))
                return
            except AuthUnavailable:
                if attempt == OPEN_REPORT_ATTEMPTS - 1 or self._stop.wait(BACKOFF_S[attempt]):
                    return
            except AuthError as e:
                if isinstance(e, SignedOut):
                    raise
                self._log("connect_open_report_error", delivery=model.id, error=str(e)[:200])
                return

    def _revert(self, client: Client, job: RevertJob) -> None:
        """Undo a revert job's turns in FreeCAD, upload the model's preview after it, and say which were undone.
        ``SignedOut`` propagates (the server gives up on the revert after a while and tells the browser)."""
        self._log("connect_revert", revert=job.id, task=job.task_id, turns=len(job.messages))
        if self._revert_turns is None:
            outcome = RevertOutcome((), "This Oscar for FreeCAD can’t revert from the browser.")
        else:
            self._state_is("working")
            try:
                outcome = self._revert_turns(job)
            except Exception as e:  # noqa: BLE001 - the browser gets a message, the log the reason
                self._log("connect_revert_error", revert=job.id, error=f"{type(e).__name__}: {e}"[:200])
                outcome = RevertOutcome((), f"FreeCAD couldn’t revert ({type(e).__name__}).")
            finally:
                self._state_is("connected")
        done = set(outcome.reverted)
        reverted = [message for message in job.messages if message in done]
        ok = len(reverted) == len(job.messages)
        if outcome.preview is not None and reverted:
            fmt, data = outcome.preview
            try:
                client.upload_revert_preview(job.task_id, job.id, fmt, data)
            except SignedOut:
                raise
            except (AuthError, ConnectError) as e:   # the revert counts without its preview
                self._log("connect_revert_preview_error", revert=job.id, error=str(e)[:200])
        self._log("connect_reverted" if ok else "connect_revert_partial", revert=job.id, reverted=len(reverted),
                  message=_clip(outcome.message, 200))
        tree = outcome.tree
        for attempt in range(REVERT_REPORT_ATTEMPTS):
            try:
                client.revert_done(job.id, ok, reverted, _clip(outcome.message, MESSAGE_LIMIT), tree)
                return
            except AuthUnavailable:
                if attempt == REVERT_REPORT_ATTEMPTS - 1 or self._stop.wait(BACKOFF_S[attempt]):
                    return
            except AuthError as e:
                if isinstance(e, SignedOut):
                    raise
                self._log("connect_revert_report_error", revert=job.id, error=str(e)[:200])
                if tree is None:
                    return
                tree = None   # a tree the server refused must not lose the report: once more without it

    def _download(self, client: Client, job: Job) -> tuple[list[str], list[str]]:
        """The job's attachments as local paths (in the job's order), and a message for each that couldn't be
        downloaded. ``SignedOut`` and ``TurnGone`` propagate."""
        if not job.attachments:
            return [], []
        folder = os.path.join(self._files_dir(), _path_part(job.task_id), _path_part(job.message_id))
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            return [], [f"Oscar couldn’t save the attached files on this computer ({type(e).__name__})."]
        names = ", ".join(item.name for item in job.attachments if not item.page)
        count = sum(1 for item in job.attachments if not item.page)
        try:
            client.post_events(job.task_id, [{"kind": "progress", "data": {
                "title": _clip(f"Reading {count} attached file{'' if count == 1 else 's'}", TITLE_LIMIT),
                "detail": _clip(names, DETAIL_LIMIT), "phase": "FILES"}}])
        except AuthUnavailable:
            pass
        files: list[str] = []
        problems: list[str] = []
        taken: set[str] = set()
        for item in job.attachments:
            dest = os.path.join(folder, _unique(safe_filename(item.name), taken))
            for attempt in range(DOWNLOAD_ATTEMPTS):
                try:
                    client.download(item, dest)
                    files.append(dest)
                    break
                except ConnectError as e:
                    problems.append(f"Couldn’t download {item.name}: {e}")
                    break
                except AuthUnavailable as e:
                    if attempt == DOWNLOAD_ATTEMPTS - 1 or self._stop.wait(BACKOFF_S[attempt]):
                        problems.append(f"Couldn’t download {item.name}: {e}")
                        break
                except AuthError as e:
                    problems.append(f"Couldn’t download {item.name}: {e}")
                    break
        self._log("connect_files", task=job.task_id, downloaded=len(files), failed=len(problems))
        return files, problems

    def _guarded(self, job: Job, what: str, call: Callable[[], Any]) -> Any:
        """``call()``; any failure but the ones that end the turn (``TurnGone``, ``SignedOut``) is logged and
        skipped, so one bad step never keeps the turn from its answer."""
        try:
            return call()
        except (TurnGone, SignedOut):
            raise
        except Exception as e:  # noqa: BLE001
            self._log("connect_step_failed", task=job.task_id, what=what, error=f"{type(e).__name__}: {e}"[:200])
            return None

    def _keep_answer(self, job: Job, turn: Turn, cancelled: bool = False) -> None:
        """Write the turn's answer to the outbox once it has ended (waits for the end up to a minute)."""
        if self._outbox is None or not turn.wait(60.0):
            return
        try:
            ok, text = turn.result()
        except Exception:  # noqa: BLE001
            ok, text = False, ""
        self._outbox.add(job.task_id, job.message_id, text or (STOPPED_TEXT if cancelled else "Done."),
                         ok and not cancelled)

    def _report(self, client: Client, job: Job, turn: Turn) -> None:
        last_post = time.monotonic()
        last_preview = float("-inf")
        version = 0
        cancelled = False
        awaiting = getattr(turn, "awaiting_answer", None)
        answer = getattr(turn, "answer", None)
        while True:
            done = turn.wait(FLUSH_S)
            now = time.monotonic()
            events = turn.events.drain()
            polling = awaiting is not None and now - last_post >= ANSWER_POLL_S and bool(
                self._guarded(job, "awaiting_answer", awaiting))
            if events or polling or now - last_post >= KEEPALIVE_S:
                try:
                    stop, answers = client.post_events_reply(job.task_id, events)
                    if stop and not cancelled:
                        cancelled = True
                        turn.stop()
                    last_post = now
                    for key, choice in answers if answer is not None else ():
                        self._guarded(job, "answer", lambda k=key, c=choice: answer(k, c))
                except AuthUnavailable:
                    turn.events.requeue(events)
                except (TurnGone, SignedOut):
                    raise
                except AuthError as e:   # a refused batch: drop it, keep reporting
                    self._log("connect_events_refused", task=job.task_id, error=str(e)[:200])
                    last_post = now
            if done or now - last_preview >= PREVIEW_INTERVAL_S:
                preview = self._guarded(job, "preview", turn.take_preview)
                if preview is not None:
                    version, data = preview[0], preview[1]
                    fmt = preview[2] if len(preview) > 2 and preview[2] in PREVIEW_FORMATS else "stl"
                    self._guarded(job, "preview_upload", lambda: self._upload(
                        client, job, turn, "preview", fmt, f"preview-{version}.{fmt}", version, data))
                    last_preview = now
                if preview is not None or done:
                    self._guarded(job, "tree", lambda: self._send_tree(client, job, turn))
            if done:
                break
        ok, text = self._guarded(job, "result", turn.result) or (False, "")
        answer, ok = text or (STOPPED_TEXT if cancelled else "Done."), ok and not cancelled
        if self._outbox is not None:
            self._outbox.add(job.task_id, job.message_id, answer, ok)   # before anything else can fail
        if ok:
            for name, fmt, data in self._guarded(job, "exports", turn.exports) or ():
                self._guarded(job, "export_upload", lambda n=name, f=fmt, d=data: self._upload(
                    client, job, turn, "export", f, n, version, d))
        self._finish(client, job.task_id, answer, ok=ok, events=turn.events.drain(), message_id=job.message_id)

    def _send_tree(self, client: Client, job: Job, turn: Turn) -> None:
        take_tree = getattr(turn, "take_tree", None)
        if take_tree is None:
            return
        try:
            tree = take_tree()
        except Exception as e:  # noqa: BLE001 - the tree is a view, never a reason to fail the turn
            self._log("connect_tree_failed", task=job.task_id, error=f"{type(e).__name__}: {e}"[:200])
            return
        if not tree:
            return
        try:
            client.post_events(job.task_id, [{"kind": "tree", "data": tree}])
        except AuthUnavailable as e:
            self._log("connect_tree_failed", task=job.task_id, error=str(e)[:200])
        except SignedOut:
            raise
        except AuthError as e:   # a server that doesn't know trees yet: the turn goes on without one
            self._log("connect_tree_refused", task=job.task_id, error=str(e)[:200])

    def _upload(self, client: Client, job: Job, turn: Turn, kind: str, fmt: str, name: str, version: int,
                data: bytes) -> None:
        try:
            client.upload(job.task_id, kind=kind, fmt=fmt, name=name, version=version, data=data)
        except UploadRefused as e:
            turn.events.error(f"Couldn’t upload {name}: {e}")
        except AuthUnavailable as e:
            self._log("connect_upload_failed", task=job.task_id, file_kind=kind, error=str(e)[:200])

    def _finish(self, client: Client, task_id: str, text: str, *, ok: bool, events: list[dict[str, Any]],
                message_id: str = "") -> None:
        """The last events and the reply. Retried while the server is unreachable; what still didn't arrive stays
        in the outbox and is delivered by ``_sync`` later."""
        for attempt in range(FINISH_ATTEMPTS):
            try:
                if events:
                    try:
                        client.post_events(task_id, events)
                    except TurnGone:
                        pass   # the turn was closed meanwhile: the answer can still go (message_id)
                    except AuthError as e:
                        if isinstance(e, (AuthUnavailable, SignedOut)):
                            raise
                        self._log("connect_events_refused", task=task_id, error=str(e)[:200])
                    events = []
                client.finish(task_id, text, ok, message_id)
                self._delivered(task_id, message_id)
                return
            except TurnGone:
                self._delivered(task_id, message_id)   # the server has no use for it
                return
            except AuthUnavailable:
                if self._stop.wait(BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]):
                    return
            except SignedOut:
                raise
            except AuthError as e:
                self._log("connect_finish_refused", task=task_id, error=str(e)[:200])
                return
        self._log("connect_finish_failed", task=task_id)

    def _delivered(self, task_id: str, message_id: str) -> None:
        if self._outbox is not None and message_id:
            self._outbox.remove(task_id, message_id)

    def _sync(self, client: Client) -> None:
        """While idle: deliver the answers left in the outbox, then answer the turns the website closed without
        one (from the outbox, else from FreeCAD's chat). ``SignedOut`` propagates; anything else waits for the
        next round."""
        try:
            outbox = self._outbox.entries() if self._outbox is not None else []
            for entry in outbox:
                self._deliver(client, entry["task_id"], entry["message_id"], entry["text"], bool(entry["ok"]))
            delivered = {(e["task_id"], e["message_id"]) for e in outbox}
            for task_id, message_id in client.unanswered():
                if (task_id, message_id) in delivered or self._stop.is_set():
                    continue
                state, ok, text = self._recover(task_id, message_id) if self._recover else ("none", False, "")
                if state == "busy":
                    continue
                if state != "found":
                    ok, text = False, NO_ANSWER_TEXT
                self._deliver(client, task_id, message_id, text, ok)
                self._log("connect_resynced", task=task_id, source=state)
        except SignedOut:
            raise
        except Exception as e:  # noqa: BLE001 - catching up is retried at the next round
            self._log("connect_sync_failed", error=f"{type(e).__name__}: {e}"[:200])

    def _deliver(self, client: Client, task_id: str, message_id: str, text: str, ok: bool) -> None:
        try:
            client.finish(task_id, text, ok, message_id)
        except TurnGone:
            pass   # answered meanwhile, or no longer this computer's task
        self._delivered(task_id, message_id)
