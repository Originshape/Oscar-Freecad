"""The fake OriginShape API plus the workspace's connect endpoints (``/v1/connect/*``), for connect-mode tests.

Scripting:
    fake.add_job(text="A flange", task_id="t1", history=[("user", "..."), ("assistant", "...")])
    fake.cancel["t1"] = True          # the next events post answers {"cancel": true}
    fake.answers["t1"] = [{"key": "approval-3", "choice": "run"}]   # sent with every events reply
    fake.gone.add("t1")               # the task was deleted: its connect calls answer 404
    fake.refuse_uploads = (409, "conflict", "storage limit")
    fake.add_job("", attachments=[("plan.pdf", "pdf", b"%PDF-1.4 ..."), ("plan-page-1.png", "image", png, 1)])
    fake.damaged.add(attachment_id)   # served with other bytes than its sha256 says
    fake.oversized.add(attachment_id) # served with more bytes than its size says
    fake.unanswered = [("t1", "m-1")]  # GET /v1/connect/unanswered lists these; a finish with that message_id is taken
    fake.add_model(step_bytes, title="A flange")   # an ``open`` job from /next, for an Oscar that said it can
    fake.damaged_models.add(delivery_id)           # served with other bytes than its sha256 says
    fake.add_revert(["m-2", "m-1"], task_id="t1")  # a ``revert`` job from /next, for an Oscar that said it can
State: fake.hellos, fake.events[task] (posted events), fake.uploads (dicts), fake.finished[task] = (text, ok),
fake.finish_bodies (every accepted finish body, in order),
fake.attachment_ids (in the order added), fake.downloads (attachment ids, in the order asked),
fake.model_reports[delivery_id] (the body of its done call), fake.model_downloads (delivery ids, in the order asked),
fake.revert_reports[revert_id] (the body of its done call), fake.revert_uploads (previews sent with a revert).
"""

from __future__ import annotations

import collections
import hashlib
import re
from typing import Any

from support.fake_originshape import FakeOriginShape, RecordedCall, envelope

_TASK_PATH = re.compile(r"^/v1/connect/tasks/(?P<task>[^/]+)/(?P<what>events|files|finish)$")
_ATTACHMENT_PATH = re.compile(r"^/v1/connect/attachments/(?P<id>[^/]+)$")
_DELIVERY_PATH = re.compile(r"^/v1/connect/deliveries/(?P<id>[^/]+)/(?P<what>file|done)$")
_REVERT_PATH = re.compile(r"^/v1/connect/reverts/(?P<id>[^/]+)/done$")


class FakeWorkspace(FakeOriginShape):
    def __init__(self, **kwargs: Any) -> None:
        self.jobs: collections.deque[dict] = collections.deque()
        self.running: set[str] = set()
        self.hellos: list[dict] = []
        self.next_calls = 0
        self.events: dict[str, list[dict]] = collections.defaultdict(list)
        self.event_posts: dict[str, int] = collections.Counter()
        self.uploads: list[dict] = []
        self.finished: dict[str, tuple[str, bool]] = {}
        self.cancel: dict[str, bool] = {}
        self.answers: dict[str, list[dict]] = {}
        self.gone: set[str] = set()
        self.refuse_uploads: tuple[int, str, str] | None = None
        self.refuse_trees = False          # an older server: events of kind "tree" answer 400
        self.unanswered: list[tuple[str, str]] = []
        self.finish_bodies: list[dict] = []
        self.files: dict[str, bytes] = {}
        self.attachment_ids: list[str] = []
        self.downloads: list[str] = []
        self.damaged: set[str] = set()
        self.oversized: set[str] = set()
        self.model_jobs: collections.deque[dict] = collections.deque()
        self.model_files: dict[str, bytes] = {}
        self.model_reports: dict[str, dict] = {}
        self.model_downloads: list[str] = []
        self.damaged_models: set[str] = set()
        self.revert_jobs: collections.deque[dict] = collections.deque()
        self.sent_reverts: dict[str, str] = {}      # revert id -> task id, once /next handed it over
        self.revert_reports: dict[str, dict] = {}
        self.revert_uploads: list[dict] = []
        self.refuse_revert_trees = False           # done calls with a tree answer 400 (a tree the server refuses)
        self._message_seq = 0
        super().__init__(**kwargs)

    def add_revert(self, messages: list[str], *, task_id: str = "task-1", title: str = "A flange") -> str:
        """Queue a ``revert`` job: undo the turns of these message ids (newest first). Returns its id."""
        with self.lock:
            revert_id = f"r-{len(self.revert_jobs) + len(self.sent_reverts) + 1}"
            self.revert_jobs.append({"id": revert_id, "task_id": task_id, "title": title,
                                     "messages": [{"id": message, "text": f"prompt {message}"} for message in messages]})
            return revert_id

    def can_revert(self) -> bool:
        """As the server decides: the latest hello said ``revert``."""
        return bool(self.hellos) and "revert" in (self.hellos[-1].get("capabilities") or [])

    def add_model(self, data: bytes, *, title: str = "An 80 mm flange", name: str = "", task_id: str = "t-engine",
                  fmt: str = "step") -> str:
        """Queue an ``open`` job: a model the browser sent. Returns its delivery id."""
        with self.lock:
            delivery_id = f"d-{len(self.model_files) + 1}"
            self.model_files[delivery_id] = data
            self.model_jobs.append({"id": delivery_id, "task_id": task_id, "title": title,
                                    "name": name or f"{title}.step", "format": fmt, "size": len(data),
                                    "sha256": hashlib.sha256(data).hexdigest()})
            return delivery_id

    def can_open_models(self) -> bool:
        """As the server decides: the latest hello said ``open_model``."""
        return bool(self.hellos) and "open_model" in (self.hellos[-1].get("capabilities") or [])

    def add_job(self, text: str, *, task_id: str = "task-1", title: str = "", model: str = "origin1",
                history: list[tuple] | None = None, turn: int = 1, attachments: list[tuple] | None = None) -> None:
        """``history`` items are ``(role, text)`` or ``(role, text, [attachment names])``; ``attachments`` items
        ``(name, kind, data)`` or ``(name, kind, data, page)``."""
        with self.lock:
            files = []
            for name, kind, data, *page in attachments or ():
                attachment_id = f"a-{len(self.files) + 1}"
                self.files[attachment_id] = data
                self.attachment_ids.append(attachment_id)
                files.append({"id": attachment_id, "name": name, "format": name.rsplit(".", 1)[-1], "kind": kind,
                              "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                              "page": page[0] if page else None})
            self.jobs.append({
                "task_id": task_id,
                "title": title or text[:80],
                "model": model,
                "message": {"id": self._message_id(), "text": text, "attachments": files},
                "history": [{"role": item[0], "text": item[1], **({"attachments": item[2]} if len(item) > 2 else {})}
                            for item in (history or [])],
                "turn": turn,
            })

    def _message_id(self) -> str:
        self._message_seq += 1   # unique per fake, also after jobs were taken
        return f"m-{self._message_seq}"

    def route_for(self, method: str, path: str) -> str | None:
        if (method, path) == ("POST", "/v1/connect/hello"):
            return "route_connect_hello"
        if (method, path) == ("POST", "/v1/connect/next"):
            return "route_connect_next"
        if (method, path) == ("GET", "/v1/connect/unanswered"):
            return "route_connect_unanswered"
        if method == "GET" and _ATTACHMENT_PATH.match(path):
            return "route_connect_attachment"
        if method == "POST" and _REVERT_PATH.match(path):
            return "route_connect_revert_done"
        delivery = _DELIVERY_PATH.match(path)
        if delivery is not None:
            return {("GET", "file"): "route_connect_model_file",
                    ("POST", "done"): "route_connect_model_done"}.get((method, delivery["what"]))
        match = _TASK_PATH.match(path)
        if match is None:
            return None
        return {("POST", "events"): "route_connect_events", ("PUT", "files"): "route_connect_upload",
                ("POST", "finish"): "route_connect_finish"}.get((method, match["what"]))

    # routes ---------------------------------------------------------------------------------------------------

    def _task(self, call: RecordedCall) -> str:
        return _TASK_PATH.match(call.path)["task"]

    def route_connect_hello(self, call: RecordedCall) -> tuple:
        if not self._bearer(call.headers):
            return self._unauthorized()
        self.hellos.append(dict(call.json or {}))
        return self._json(200, {"poll_wait": 25})

    def route_connect_next(self, call: RecordedCall) -> tuple:
        if not self._bearer(call.headers):
            return self._unauthorized()
        self.next_calls += 1
        if self.revert_jobs and self.can_revert():
            job = self.revert_jobs.popleft()
            self.sent_reverts[job["id"]] = job["task_id"]
            return self._json(200, {"revert": job})
        if self.model_jobs and self.can_open_models():
            return self._json(200, {"open": self.model_jobs.popleft()})
        if not self.jobs:
            return 204, {}, b"", "application/json"
        job = self.jobs.popleft()
        self.running.add(job["task_id"])
        return self._json(200, {"job": job})

    def route_connect_events(self, call: RecordedCall) -> tuple:
        if not self._bearer(call.headers):
            return self._unauthorized()
        task = self._task(call)
        if task in self.gone or task not in self.running:
            return self._json(404, envelope("not_found", "Task not found."))
        posted = (call.json or {}).get("events", [])
        if self.refuse_trees and any(event.get("kind") == "tree" for event in posted):
            return self._json(400, envelope("invalid_request", "Unknown event kind."))
        self.event_posts[task] += 1
        self.events[task].extend(posted)
        return self._json(200, {"cancel": bool(self.cancel.get(task)), "answers": list(self.answers.get(task, ()))})

    def route_connect_upload(self, call: RecordedCall) -> tuple:
        if not self._bearer(call.headers):
            return self._unauthorized()
        task = self._task(call)
        revert = call.query.get("revert")
        if revert is not None:   # the model's preview after a revert this FreeCAD took
            if self.sent_reverts.get(revert) != task or revert in self.revert_reports:
                return self._json(404, envelope("not_found", "Revert not found."))
            self.revert_uploads.append({"task": task, **call.query, "data": call.raw})
            return self._json(201, {"id": f"f-revert-{len(self.revert_uploads)}"})
        if task in self.gone or task not in self.running:
            return self._json(404, envelope("not_found", "Task not found."))
        if self.refuse_uploads is not None:
            status, code, message = self.refuse_uploads
            return self._json(status, envelope(code, message))
        self.uploads.append({"task": task, **call.query, "data": call.raw,
                             "content_type": call.headers.get("content-type", "")})
        return self._json(201, {"id": f"f-{len(self.uploads)}"})

    def route_connect_finish(self, call: RecordedCall) -> tuple:
        if not self._bearer(call.headers):
            return self._unauthorized()
        task = self._task(call)
        body = call.json or {}
        late = (task, str(body.get("message_id") or "")) in self.unanswered
        if task in self.gone or (task not in self.running and not late):
            return self._json(409 if task not in self.gone else 404, envelope("conflict", "No turn is running."))
        self.finished[task] = (str(body.get("text", "")), bool(body.get("ok")))
        self.finish_bodies.append(dict(body))
        self.running.discard(task)
        if late:
            self.unanswered.remove((task, str(body.get("message_id"))))
        return self._json(200, {"ok": True})

    def route_connect_unanswered(self, call: RecordedCall) -> tuple:
        if not self._bearer(call.headers):
            return self._unauthorized()
        return self._json(200, {"turns": [{"task_id": t, "message_id": m} for t, m in self.unanswered]})

    def route_connect_attachment(self, call: RecordedCall) -> tuple:
        if not self._bearer(call.headers):
            return self._unauthorized()
        attachment_id = _ATTACHMENT_PATH.match(call.path)["id"]
        self.downloads.append(attachment_id)
        data = self.files.get(attachment_id)
        if data is None:
            return self._json(404, envelope("not_found", "Attachment not found."))
        if attachment_id in self.damaged:
            data = bytes(len(data))
        if attachment_id in self.oversized:
            data += b"more"
        return 200, {"Content-Disposition": "attachment"}, data, "application/octet-stream"

    def route_connect_model_file(self, call: RecordedCall) -> tuple:
        if not self._bearer(call.headers):
            return self._unauthorized()
        delivery_id = _DELIVERY_PATH.match(call.path)["id"]
        self.model_downloads.append(delivery_id)
        data = self.model_files.get(delivery_id)
        if data is None or delivery_id in self.model_reports:
            return self._json(404, envelope("not_found", "Delivery not found."))
        if delivery_id in self.damaged_models:
            data = bytes(len(data))
        return 200, {"Content-Disposition": "attachment"}, data, "model/step"

    def route_connect_model_done(self, call: RecordedCall) -> tuple:
        if not self._bearer(call.headers):
            return self._unauthorized()
        delivery_id = _DELIVERY_PATH.match(call.path)["id"]
        if delivery_id not in self.model_files or delivery_id in self.model_reports:
            return self._json(404, envelope("not_found", "Delivery not found."))
        self.model_reports[delivery_id] = dict(call.json or {})
        return self._json(200, {"ok": True})

    def route_connect_revert_done(self, call: RecordedCall) -> tuple:
        if not self._bearer(call.headers):
            return self._unauthorized()
        revert_id = _REVERT_PATH.match(call.path)["id"]
        if revert_id not in self.sent_reverts or revert_id in self.revert_reports:
            return self._json(404, envelope("not_found", "Revert not found."))
        body = dict(call.json or {})
        if self.refuse_revert_trees and "tree" in body:
            return self._json(400, envelope("invalid_request", "A tree has a list of at most 2000 objects."))
        self.revert_reports[revert_id] = body
        return self._json(200, {"ok": True})
