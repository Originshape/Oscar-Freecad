# SPDX-License-Identifier: Apache-2.0
"""FreeCAD chat records: what OriginShape keeps of a chat that runs on the account (pure: no Qt, no FreeCAD).

A turn is recorded when it ran on the OriginShape account, was typed in FreeCAD (a workspace turn is already the
workspace's own record) and the account accepted the current Terms (``consent.recorded``): ``applies``. Then, once
the turn has ended (answered, failed or stopped), ``record_turn`` hands it to the ``Recorder``, which sends:

* the turn: ``PUT /v1/freecad/chats/{chat_id}/turns/{turn}`` (``turn_body``): the prompt as the user typed it,
  Oscar's reply, the steps (tool, title, ok, the script of a step that ran code, the result or error text), the model
  alias, the start and end times, the project folder's and the chat's names and ``client``
  (``oscar-freecad/<version>``). ``chat_id`` is the chat's own id (``chats``), ``turn`` its index in the chat (``chats.next_turn``). The server
  replaces a record it already has, so sending one again is safe.
* the files: ``PUT /v1/freecad/chats/{chat_id}/files?name=&kind=&turn=``, the raw bytes with their
  ``Content-Length``: the files the user attached to the message (``attachment``) and the STEP, STL and FreeCAD files
  Oscar wrote in the project folder during the turn (``output``, ``output_files``). Each file goes once per account
  (``SentFiles``, by sha256); a file over ``MAX_FILE_BYTES`` (an STL over ``MAX_STL_BYTES``) is left out.

Every text is cleaned before it leaves (``clean``): the credentials this process handled and the usual key and token
shapes are redacted (``context_guard.redact``, the same rule as for the model, so a script that printed a settings file
uploads ``[redacted credential]``), then project folders read "Oscar project" and the home folder ``~``
(``project_paths.redact``), and only then is the text cut to the server's limits.

The ``Recorder`` sends on ONE daemon thread, from a queue of at most ``QUEUE_JOBS`` turns and ``QUEUE_CHARS``
characters: a long offline session drops its oldest records instead of growing. It never reaches the chat: network
errors, 429 (``Retry-After`` honoured up to ``MAX_WAIT_S``) and 5xx are tried again after ``RETRY_S``; any other answer
drops what was being sent; every exception is caught. ``AuthSession.authorized_request`` renews the sign-in after a
401 (a second 401 is ``SignedOut``: dropped). A 403 ``terms_required`` drops the account's queued turns and calls
``on_terms_required(api_url)`` (by default ``consent.mark_needed``: the panel then asks for consent again). A turn
carries the account it was recorded for: after a sign-out, or a sign-in as someone else, it is dropped unsent.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import stat
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable
from urllib.parse import quote, urlencode

from . import __version__, context_guard, project_paths
from .auth import consent as _consent
from .auth.transport import AuthError, AuthUnavailable, SignedOut, error_details, is_unavailable
from .provider_transport import retry_after_seconds
from .turnlog import log_event

CLIENT = f"oscar-freecad/{__version__}"
CHATS_PATH = "/v1/freecad/chats"
# The server's limits (characters, unless said otherwise).
PROMPT_LIMIT, REPLY_LIMIT = 100_000, 400_000
PROJECT_LIMIT, TITLE_LIMIT, MODEL_LIMIT, CLIENT_LIMIT = 200, 300, 100, 100
STEPS_LIMIT, STEPS_BYTES = 500, 2_000_000          # steps per turn, and their JSON in all
SHORT_DETAIL = 500                                  # a step's detail when the steps are over STEPS_BYTES
TOOL_LIMIT, STEP_TITLE_LIMIT, SCRIPT_LIMIT, DETAIL_LIMIT = 80, 300, 100_000, 4_000
MAX_TURN = 100_000
MAX_FILE_BYTES = 50_000_000                         # "50 MB", read as the smaller of 50 MB and 50 MiB
MAX_STL_BYTES = 100_000_000                         # an .stl may be twice that
NAME_LIMIT = 255
OUTPUT_EXTENSIONS = frozenset({".step", ".stp", ".stl", ".fcstd"})
MTIME_SLACK_S = 2.0                                 # a file system's coarse clock: written "just before" the turn began
# The queue and the retries.
QUEUE_JOBS = 50
QUEUE_CHARS = 8_000_000
RETRY_S = (2.0, 10.0, 30.0)                         # waits before the 2nd, 3rd and 4th attempt
MAX_WAIT_S = 120.0                                  # a longer Retry-After (the daily upload limit) drops the file
SENT_KEPT = 5_000                                   # sha256s remembered per account
TURN_TIMEOUT: tuple[float, float] = (5.0, 30.0)
FILE_TIMEOUT: tuple[float, float] = (5.0, 120.0)

_CHAT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


# -- what a turn sends ---------------------------------------------------------------------------------------

@dataclass
class Step:
    """One step of a turn as the panel saw it."""

    tool: str
    title: str = ""
    ok: bool | None = None
    script: str = ""        # the code a script step ran (the panel has at most 12000 characters of it)
    script_path: str = ""   # the whole script, as the project recorded it: read when the record is sent
    detail: str = ""        # the step's result or error text


@dataclass
class TurnRecord:
    """One finished turn of a chat. Times are epoch seconds; ``attachments`` and ``outputs`` are file paths."""

    chat_id: str
    turn: int
    prompt: str = ""
    reply: str = ""
    steps: list[Step] = field(default_factory=list)
    model: str = ""
    project: str = ""
    title: str = ""
    ok: bool | None = None
    stopped: bool = False
    started_at: float = 0.0
    finished_at: float = 0.0
    attachments: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


def clean(text: Any, limit: int) -> str:
    """``text`` as it may leave this computer: credentials redacted, project folders as "Oscar project" and the home
    folder as ``~``, NUL characters dropped, and only then cut to ``limit`` characters (so a cut never leaves half a
    key behind)."""
    if not isinstance(text, str) or not text:
        return ""
    out = context_guard.redact(text.replace("\x00", ""))
    try:
        out = project_paths.redact(out)
    except Exception:  # noqa: BLE001 - an odd home folder: the credentials are gone already
        pass
    return out[:limit]


def iso_time(epoch: Any) -> str | None:
    """``2026-11-02T10:00:00Z`` (UTC), or None for no time."""
    if isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or epoch <= 0:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _read_script(path: str) -> str:
    """The recorded script at ``path`` (a ``.py`` file), at most twice ``SCRIPT_LIMIT`` characters; "" when there is
    none or it can't be read."""
    if not path or not path.lower().endswith(".py") or not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(SCRIPT_LIMIT * 2)
    except OSError:
        return ""


def step_entry(step: Step) -> dict[str, Any]:
    """One step of ``turn_body``: ``tool``, ``title``, ``ok``, ``script`` (only for a step that ran code: the whole
    recorded script when there is one) and ``detail``."""
    script = (_read_script(step.script_path) or step.script) if (step.script or step.script_path) else ""
    tool = clean(step.tool, TOOL_LIMIT) or "tool"
    return {"tool": tool, "title": clean(step.title, STEP_TITLE_LIMIT) or tool, "ok": step.ok is True,
            "script": clean(script, SCRIPT_LIMIT) or None, "detail": clean(step.detail, DETAIL_LIMIT) or None}


def _json_size(value: Any) -> int:
    return len(json.dumps(value).encode("utf-8"))


def _fit_steps(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``entries`` within ``STEPS_BYTES`` of JSON, every step kept: each starts without its script and with at most
    ``SHORT_DETAIL`` characters of detail, then the whole details and then the scripts are given back, first steps
    first, while they fit."""
    small = [{**e, "script": None, "detail": e["detail"][:SHORT_DETAIL] if e["detail"] else None} for e in entries]
    total = 2 + sum(_json_size(e) + 2 for e in small)   # "[" "]" and ", " between the steps
    while small and total > STEPS_BYTES:                 # titles alone too long: the last steps go
        total -= _json_size(small.pop()) + 2
    for name in ("detail", "script"):
        for index, entry in enumerate(small):
            better = {**entry, name: entries[index][name]}
            delta = _json_size(better) - _json_size(entry)
            if delta and total + delta <= STEPS_BYTES:
                small[index] = better
                total += delta
    return small


def turn_body(record: TurnRecord) -> dict[str, Any]:
    """The ``PUT .../turns/{turn}`` body of ``record``: every text cleaned and cut to the server's limits, at most
    ``STEPS_LIMIT`` steps and ``STEPS_BYTES`` of step JSON (``_fit_steps``: over that, the later steps go without
    their scripts and long details, never without the steps themselves)."""
    entries = [step_entry(step) for step in record.steps[:STEPS_LIMIT]]
    steps = entries if 2 + sum(_json_size(e) + 2 for e in entries) <= STEPS_BYTES else _fit_steps(entries)
    return {
        "project": clean(record.project, PROJECT_LIMIT) or None,
        "title": clean(record.title, TITLE_LIMIT) or None,
        "model": clean(record.model, MODEL_LIMIT) or None,
        "prompt": clean(record.prompt, PROMPT_LIMIT),
        "reply": clean(record.reply, REPLY_LIMIT),
        "ok": record.ok if isinstance(record.ok, bool) else None,
        "stopped": bool(record.stopped),
        "steps": steps,
        "started_at": iso_time(record.started_at),
        "finished_at": iso_time(record.finished_at),
        "client": CLIENT[:CLIENT_LIMIT],
    }


def _bounded(record: TurnRecord) -> TurnRecord:
    """``record`` with its texts cut to twice the server's limits, so a queued turn holds a bounded amount of text.
    (Cleaning, then the final cut, happens when it is sent; a key cut here is past the final cut.)"""
    def cut(text: Any, limit: int) -> str:
        return text[:limit * 2] if isinstance(text, str) else ""

    steps = [Step(cut(s.tool, TOOL_LIMIT), cut(s.title, STEP_TITLE_LIMIT), s.ok, cut(s.script, SCRIPT_LIMIT),
                  s.script_path if isinstance(s.script_path, str) else "", cut(s.detail, DETAIL_LIMIT))
             for s in record.steps[:STEPS_LIMIT] if isinstance(s, Step)]
    return replace(record, prompt=cut(record.prompt, PROMPT_LIMIT), reply=cut(record.reply, REPLY_LIMIT),
                   title=cut(record.title, TITLE_LIMIT), project=cut(record.project, PROJECT_LIMIT),
                   model=cut(record.model, MODEL_LIMIT), steps=steps,
                   attachments=[p for p in record.attachments if isinstance(p, str) and p],
                   outputs=[p for p in record.outputs if isinstance(p, str) and p])


def _chars(record: TurnRecord) -> int:
    return (len(record.prompt) + len(record.reply) + len(record.title) + len(record.project)
            + sum(len(s.tool) + len(s.title) + len(s.script) + len(s.detail) for s in record.steps))


# -- the files -----------------------------------------------------------------------------------------------

def output_files(project_dir: str, since: float) -> list[str]:
    """The STEP, STL and FreeCAD files in the project folder ``project_dir`` written at or after ``since`` (epoch
    seconds, less ``MTIME_SLACK_S``): ``model.FCStd``, ``docs/*.FCStd`` and ``exports/*``. Local: a few ``stat``s."""
    if not project_dir or not os.path.isdir(project_dir):
        return []
    candidates = [os.path.join(project_dir, "model.FCStd")]
    for sub in ("docs", "exports"):
        try:
            names = sorted(os.listdir(os.path.join(project_dir, sub)))
        except OSError:
            continue
        candidates += [os.path.join(project_dir, sub, n) for n in names if not n.startswith(".") and ".tmp-" not in n]
    found: list[str] = []
    for path in candidates:
        if os.path.splitext(path)[1].lower() not in OUTPUT_EXTENSIONS:
            continue
        try:
            info = os.stat(path)
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode) and info.st_mtime >= since - MTIME_SLACK_S:
            found.append(path)
    return found


def file_allowed(name: str, size: int) -> bool:
    """A file the server takes: not empty, at most ``MAX_FILE_BYTES`` (``MAX_STL_BYTES`` for an ``.stl``)."""
    limit = MAX_STL_BYTES if str(name).lower().endswith(".stl") else MAX_FILE_BYTES
    return 0 < size <= limit


def upload_name(path: str) -> str:
    """The name a file is sent under: its own name (no folders), one line, credentials redacted, at most
    ``NAME_LIMIT`` characters with its extension kept."""
    name = _CONTROL.sub(" ", os.path.basename(str(path or "")).strip()).strip()
    name = context_guard.redact(name)
    if len(name) > NAME_LIMIT:
        stem, ext = os.path.splitext(name)
        ext = ext[:16]
        name = stem[:NAME_LIMIT - len(ext)] + ext
    return name


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def account_key(api_url: str, email: str) -> str:
    """Who a sent file is remembered for: a hash of the server and the account's email (never the email itself)."""
    return hashlib.sha256(f"{api_url}\n{str(email).strip().lower()}".encode("utf-8")).hexdigest()[:24]


class SentFiles:
    """The sha256 of every file sent, per account (``account_key``), so each file goes once. Kept in ``path`` (JSON,
    the newest ``limit`` per account); in memory only without a path, or when the file can't be read or written.
    Thread-safe."""

    def __init__(self, path: str = "", limit: int = SENT_KEPT):
        self.path = path
        self.limit = max(1, int(limit))
        self._lock = threading.Lock()
        self._sent: dict[str, dict[str, None]] | None = None

    def _load(self) -> dict[str, dict[str, None]]:
        """Under ``_lock``."""
        if self._sent is None:
            self._sent = {}
            try:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                data = None
            if isinstance(data, dict):
                for key, hashes in data.items():
                    if isinstance(key, str) and isinstance(hashes, list):
                        self._sent[key] = {h: None for h in hashes[-self.limit:] if isinstance(h, str)}
        return self._sent

    def has(self, account: str, sha256: str) -> bool:
        with self._lock:
            return sha256 in self._load().get(account, {})

    def add(self, account: str, sha256: str) -> None:
        with self._lock:
            hashes = self._load().setdefault(account, {})
            hashes.pop(sha256, None)
            hashes[sha256] = None
            while len(hashes) > self.limit:
                del hashes[next(iter(hashes))]
            self._save()

    def _save(self) -> None:
        """Under ``_lock``: an atomic replace of the file."""
        if not self.path:
            return
        data = {key: list(hashes) for key, hashes in (self._sent or {}).items()}
        try:
            folder = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(folder, exist_ok=True)
            handle, temp = tempfile.mkstemp(prefix=".sent-", dir=folder)
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(temp, self.path)
            except BaseException:
                if os.path.exists(temp):
                    os.remove(temp)
                raise
        except OSError:
            pass   # remembered for this run; a file sent again is answered "already have it" by the server


# -- sending -------------------------------------------------------------------------------------------------

def _log(event: str, **fields: Any) -> None:
    log_event("records", event=event, **fields)


@dataclass
class _Job:
    record: TurnRecord
    api_url: str
    account: str
    chars: int


def _default_session(api_url: str):
    from .auth.session import get_session

    return get_session(api_url)


def _default_terms_required(api_url: str) -> None:
    _consent.mark_needed(_default_session(api_url))


class Recorder:
    """The queue and the thread that send turn records (see the module docstring). ``session_for(api_url)`` gives
    the ``AuthSession``; ``wait(seconds)`` waits before a retry and returns True when the recorder is closing;
    ``autostart=False`` keeps the thread from starting until ``start()`` (tests). ``stats`` counts what happened."""

    def __init__(self, *, session_for: Callable[[str], Any] | None = None, sent: SentFiles | None = None,
                 on_terms_required: Callable[[str], Any] | None = None, retry_s: tuple[float, ...] = RETRY_S,
                 max_wait_s: float = MAX_WAIT_S, queue_jobs: int = QUEUE_JOBS, queue_chars: int = QUEUE_CHARS,
                 wait: Callable[[float], bool] | None = None, autostart: bool = True):
        self._session_for = session_for or _default_session
        self.sent = sent if sent is not None else SentFiles()
        self._on_terms = on_terms_required or _default_terms_required
        self._retry_s = tuple(retry_s)
        self._max_wait_s = float(max_wait_s)
        self._max_jobs = max(1, int(queue_jobs))
        self._max_chars = max(1, int(queue_chars))
        self._closed = threading.Event()
        self._wait = wait or self._closed.wait
        self._autostart = autostart
        self._cond = threading.Condition()
        self._queue: collections.deque[_Job] = collections.deque()
        self._chars = 0
        self._busy = False
        self._thread: threading.Thread | None = None
        self.stats: collections.Counter = collections.Counter()

    # -- the queue ---------------------------------------------------------------------------------------------

    def submit(self, record: TurnRecord, *, api_url: str, account: str) -> bool:
        """Queue ``record`` for the account ``account`` (its email) of ``api_url``. Returns at once; False when the
        record can't be sent (no chat id, a turn index out of range, closed). A full queue drops its oldest turns."""
        if (not isinstance(record, TurnRecord) or not isinstance(record.chat_id, str)
                or not _CHAT_ID.fullmatch(record.chat_id) or isinstance(record.turn, bool)
                or not isinstance(record.turn, int) or not 0 <= record.turn <= MAX_TURN or not api_url or not account):
            return False
        record = _bounded(record)
        job = _Job(record, api_url, account, _chars(record))
        with self._cond:
            if self._closed.is_set():
                return False
            self._queue.append(job)
            self._chars += job.chars
            while len(self._queue) > 1 and (len(self._queue) > self._max_jobs or self._chars > self._max_chars):
                old = self._queue.popleft()
                self._chars -= old.chars
                self.stats["dropped_full"] += 1
            self._cond.notify_all()
        if self._autostart:
            self.start()
        return True

    def pending(self) -> int:
        """Turns waiting to be sent (not the one being sent)."""
        with self._cond:
            return len(self._queue)

    def start(self) -> None:
        """Start the sending thread (once)."""
        with self._cond:
            if self._closed.is_set() or (self._thread is not None and self._thread.is_alive()):
                return
            self._thread = threading.Thread(target=self._loop, name="oscar-records", daemon=True)
            self._thread.start()

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Block until nothing is queued or being sent (tests); False on timeout."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._queue or self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def close(self) -> None:
        """Stop after the current request; what is queued is dropped."""
        self._closed.set()
        with self._cond:
            self._cond.notify_all()

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._closed.is_set():
                    self._cond.wait()
                if self._closed.is_set():
                    return
                job = self._queue.popleft()
                self._chars -= job.chars
                self._busy = True
            try:
                self._run(job)
            except Exception as e:  # noqa: BLE001 - a bug here must never end the thread, nor reach the chat
                self.stats["errors"] += 1
                _log("error", error=type(e).__name__)
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()

    # -- one turn ----------------------------------------------------------------------------------------------

    def _run(self, job: _Job) -> None:
        session = self._session_for(job.api_url)
        if not self._same_account(session, job.account):
            self.stats["dropped_account"] += 1
            _log("dropped", why="account")
            return
        record = job.record
        base = f"{CHATS_PATH}/{quote(record.chat_id, safe=':')}"
        body = turn_body(record)
        outcome = self._send(lambda: session.authorized_request("PUT", f"{base}/turns/{record.turn}", json_body=body,
                                                                timeout=TURN_TIMEOUT))
        self.stats[f"turn_{outcome}"] += 1
        _log("turn", outcome=outcome, steps=len(body["steps"]))
        if outcome in ("terms", "signed_out", "closed"):
            if outcome == "terms":
                self._terms_required(job)
            return
        key = account_key(job.api_url, job.account)
        files = [(p, "attachment") for p in record.attachments] + [(p, "output") for p in record.outputs]
        for path, kind in files:
            if self._closed.is_set():
                return
            outcome = self._send_file(session, base, record.turn, path, kind, key)
            self.stats[f"file_{outcome}"] += 1
            if outcome in ("terms", "signed_out", "closed"):
                if outcome == "terms":
                    self._terms_required(job)
                return

    @staticmethod
    def _same_account(session: Any, account: str) -> bool:
        """The session is still signed in as the account the turn was recorded for."""
        try:
            return bool(session.has_session()) and str(session.email).strip().lower() == account.strip().lower()
        except Exception:  # noqa: BLE001 - an unreadable sign-in: not the same account
            return False

    def _send_file(self, session: Any, base: str, turn: int, path: str, kind: str, key: str) -> str:
        """One file: ``"ok"``, ``"known"`` (sent before), ``"skipped"`` (too big, empty, unreadable) or what
        ``_send`` says."""
        name = upload_name(path)
        try:
            if not name or not file_allowed(name, os.path.getsize(path)):
                return "skipped"
            if self.sent.has(key, _sha256_file(path)):
                return "known"
            with open(path, "rb") as f:
                data = f.read(MAX_STL_BYTES + 1)
        except OSError:
            return "skipped"
        if not file_allowed(name, len(data)):   # it changed meanwhile
            return "skipped"
        digest = hashlib.sha256(data).hexdigest()
        if self.sent.has(key, digest):
            return "known"
        query = urlencode({"name": name, "kind": kind, "turn": turn})
        outcome = self._send(lambda: session.authorized_request("PUT", f"{base}/files?{query}", content=data,
                                                                timeout=FILE_TIMEOUT))
        if outcome == "ok":
            self.sent.add(key, digest)
        return outcome

    def _send(self, request: Callable[[], Any]) -> str:
        """``request()`` until it is answered: ``"ok"`` (2xx), ``"terms"`` (403 ``terms_required``), ``"signed_out"``,
        ``"dropped"`` (any other answer, or still failing after the retries) or ``"closed"``. Network errors, 429 and
        5xx are tried again after ``RETRY_S`` (a ``Retry-After`` is waited for instead, when it is at most
        ``MAX_WAIT_S``)."""
        for attempt in range(len(self._retry_s) + 1):
            last = attempt >= len(self._retry_s)
            try:
                reply = request()
            except SignedOut:
                return "signed_out"
            except AuthUnavailable:   # the network: no answer at all
                if last:
                    return "dropped"
                delay = self._retry_s[attempt]
            except AuthError:         # an oversized or unreadable answer: asking again changes nothing
                return "dropped"
            else:
                status = reply.status
                if 200 <= status < 300:
                    return "ok"
                if status == 403 and error_details(reply.body)[0] == "terms_required":
                    return "terms"
                if not is_unavailable(status) or last:
                    _log("dropped", status=status)
                    return "dropped"
                delay = self._retry_s[attempt]
                after = retry_after_seconds(reply.headers)
                if after is not None:
                    if after > self._max_wait_s:
                        _log("dropped", status=status, why="retry_after")
                        return "dropped"
                    delay = after
            self.stats["retries"] += 1
            if self._wait(delay):
                return "closed"
        return "dropped"

    def _terms_required(self, job: _Job) -> None:
        """403 ``terms_required``: the account's queued turns are dropped (the server refuses them too), and
        ``on_terms_required`` is told."""
        with self._cond:
            kept = collections.deque(j for j in self._queue if (j.api_url, j.account) != (job.api_url, job.account))
            self.stats["dropped_terms"] += len(self._queue) - len(kept)
            self._queue = kept
            self._chars = sum(j.chars for j in kept)
        _log("terms_required")
        try:
            self._on_terms(job.api_url)
        except Exception:  # noqa: BLE001 - the panel asks again on its next account refresh
            pass


# -- the panel's entry ---------------------------------------------------------------------------------------

def applies(*, account_profile: bool, workspace: bool, state: Any) -> bool:
    """Whether a finished turn is recorded: it ran on the OriginShape account (``account_profile``), it was typed in
    FreeCAD (not a ``workspace`` turn), and ``state`` (the panel's account state) is signed in to an account that
    accepted the current Terms."""
    if not account_profile or workspace or not isinstance(state, dict) or not state.get("signed_in"):
        return False
    return _consent.recorded(state.get("snapshot"))


def record_turn(record: TurnRecord, *, account_profile: bool, workspace: bool, state: Any,
                into: Recorder | None = None) -> bool:
    """Queue ``record`` (into ``into``, default the process's ``recorder()``) when ``applies``. True when queued."""
    if not applies(account_profile=account_profile, workspace=workspace, state=state):
        return False
    snapshot = state.get("snapshot") if isinstance(state.get("snapshot"), dict) else {}
    account = str(snapshot.get("email") or state.get("email") or "")
    return (into or recorder()).submit(record, api_url=str(state.get("api_url") or ""), account=account)


_RECORDER: Recorder | None = None
_RECORDER_LOCK = threading.Lock()


def recorder() -> Recorder:
    """The process's ``Recorder`` (created on first use; files sent are remembered under the cache folder)."""
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is None:
            try:
                from . import paths

                where = os.path.join(paths.cache_dir("records"), "sent-files.json")
            except Exception:  # noqa: BLE001 - no cache folder: remembered for this run only
                where = ""
            _RECORDER = Recorder(sent=SentFiles(where))
        return _RECORDER
