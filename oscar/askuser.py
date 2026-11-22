# SPDX-License-Identifier: Apache-2.0
"""Interactive decision requests for the ``ask_user`` tool: payload validation, the pending-request
registry and the answer protocol.

Pure Python — no FreeCAD, no Qt. The tool validates the model's request here, opens a
``PendingRequest`` bound to (conversation_id, run_id) and blocks its worker thread in
``RequestRegistry.wait`` until the panel resolves the matching id — or Stop / timeout / shutdown
releases it. Registry discipline: one answer resolves exactly ONE request of the SAME conversation
and run; everything else (duplicate, wrong chat, wrong run) is a stale rejection that never settles
the live request. Malformed payloads are rejected, never repaired or guessed into shape.

Request lifecycle (``open`` is the wire name of PENDING; the end states are final)::

    open → answering → answered        a submission is being validated, then accepted
    open → answering → open            validation failed: retryable, the question reopens
    open → cancelled / timeout / rejected

Every transition happens under the registry lock, so ``answered`` can never become anything
else: the deadline, Stop and duplicate submissions all bounce off the same lock that guards
the accepting transition. Lifecycle events (register/wait/submit/accept/resolve/return/
timeout/reject — ids and states only, never answer content) go to the turn log.
"""

import threading
import time
import uuid

from .turnlog import log_event

MIN_OPTIONS = 2
MAX_OPTIONS = 4
MIN_WIZARD_QUESTIONS = 2
MAX_WIZARD_QUESTIONS = 6
DEFAULT_TIMEOUT_S = 1800.0        # a decision mid-reconstruction may wait until the user comes back
WAIT_POLL_S = 0.25                # the same cooperative slice cad/dispatch waits in

OPEN, ANSWERING, ANSWERED, CANCELLED, TIMEOUT, REJECTED = (
    "open", "answering", "answered", "cancelled", "timeout", "rejected")

# The legal lifecycle edges. ``answered``/``cancelled``/``timeout``/``rejected`` are terminal:
# an accepted answer can never be overwritten by the deadline, a Stop or a duplicate submission.
_ALLOWED_FROM = {
    OPEN: (ANSWERING, ANSWERED, CANCELLED, TIMEOUT, REJECTED),
    ANSWERING: (OPEN, ANSWERED),
    ANSWERED: (), CANCELLED: (), TIMEOUT: (), REJECTED: (),
}

# The deterministic rejection reason for submitting against a request in each end state.
_REASON_FOR_STATE = {ANSWERED: "already_answered", TIMEOUT: "timed_out",
                     CANCELLED: "cancelled", REJECTED: "rejected"}


def _nonempty_str(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _option(raw) -> tuple[dict | None, str]:
    if not isinstance(raw, dict):
        return None, "an option must be an object"
    if not _nonempty_str(raw.get("label")):
        return None, "every option needs a non-empty label"
    label = raw["label"].strip()
    description = raw.get("description", "")
    likeliest = raw.get("likeliest", False)
    if not isinstance(description, str) or not isinstance(likeliest, bool):
        return None, f"option '{label}': description must be text and likeliest a boolean"
    return {"label": label, "description": description, "likeliest": likeliest}, ""


def validate_question(payload: object) -> tuple[dict | None, str]:
    """``(normalized, "")`` when a question is well formed, else ``(None, reason)``.

    Rules (master spec): required title; optional subtitle / monospace subtitle; 2 to 4 mutually
    exclusive options; EXACTLY one likeliest; no duplicate labels; optional custom-value field;
    optional free-text flag. Normalization only adds defaults — it never reorders, invents or
    repairs content."""
    if not isinstance(payload, dict):
        return None, "the question must be an object"
    if not _nonempty_str(payload.get("title")):
        return None, "the question needs a non-empty title"
    raw_options = payload.get("options")
    if not isinstance(raw_options, list) or not (MIN_OPTIONS <= len(raw_options) <= MAX_OPTIONS):
        return None, f"a question needs {MIN_OPTIONS} to {MAX_OPTIONS} options"
    options: list[dict] = []
    for raw in raw_options:
        option, err = _option(raw)
        if option is None:
            return None, err
        options.append(option)
    marked = [o for o in options if o["likeliest"]]
    if len(marked) != 1:
        return None, "exactly one option must be the likeliest (the evidence-supported default)"
    labels = [o["label"] for o in options]
    if len(set(labels)) != len(labels):
        return None, "duplicate option labels are not allowed"
    subtitle = payload.get("subtitle", "")
    mono = payload.get("mono_subtitle", False)
    free_text = payload.get("allow_free_text", False)
    if not isinstance(subtitle, str) or not isinstance(mono, bool) or not isinstance(free_text, bool):
        return None, "subtitle must be text; mono_subtitle and allow_free_text must be booleans"
    custom = payload.get("custom_value")
    if custom is not None:
        if not isinstance(custom, dict) or not _nonempty_str(custom.get("label")):
            return None, "custom_value needs a non-empty label"
        unit = custom.get("unit", "")
        if not isinstance(unit, str):
            return None, "custom_value unit must be text"
        custom = {"label": custom["label"].strip(), "unit": unit}
    return {"title": payload["title"].strip(), "subtitle": subtitle, "mono_subtitle": mono,
            "options": options, "custom_value": custom, "allow_free_text": free_text}, ""


def validate_wizard(payload: object) -> tuple[dict | None, str]:
    """``(normalized, "")`` when a wizard request is well formed, else ``(None, reason)``.

    A wizard is one logical pending interaction of 2 to 6 questions, each validated by the same
    rules as an individual question; the card updates in place as the user answers each one."""
    if not isinstance(payload, dict) or not isinstance(payload.get("wizard"), list):
        return None, "a wizard needs a 'wizard' list of questions"
    questions = payload["wizard"]
    if not (MIN_WIZARD_QUESTIONS <= len(questions) <= MAX_WIZARD_QUESTIONS):
        return None, f"a wizard needs {MIN_WIZARD_QUESTIONS} to {MAX_WIZARD_QUESTIONS} questions"
    validated: list[dict] = []
    for i, question in enumerate(questions):
        normalized, err = validate_question(question)
        if normalized is None:
            return None, f"wizard question {i + 1}: {err}"
        validated.append(normalized)
    return {"wizard": validated}, ""


class PendingRequest:
    """One open (or settled) decision request. The waiter blocks on ``event``; the resolver mutates
    ``state``/``answer`` under the registry lock and then sets ``event`` exactly once."""

    __slots__ = ("id", "conversation_id", "run_id", "payload", "wizard", "created_ts", "timeout_s",
                 "event", "state", "answer", "resolved_ts")

    def __init__(self, *, payload: dict, conversation_id: str, run_id: str, timeout_s: float):
        self.id = uuid.uuid4().hex[:12]
        self.conversation_id = conversation_id
        self.run_id = run_id
        self.payload = payload
        self.wizard = "wizard" in payload
        self.created_ts = time.time()
        self.timeout_s = timeout_s
        self.event = threading.Event()
        self.state = OPEN
        self.answer: dict | None = None
        self.resolved_ts: float | None = None

    def record(self) -> dict:
        """The serializable form for transcript persistence (never a live Event/thread)."""
        answers = self.answer.get("answers") if self.wizard else [self.answer] if self.answer else []
        return {"request_id": self.id, "conversation_id": self.conversation_id, "run_id": self.run_id,
                "state": self.state, "created_ts": self.created_ts, "resolved_ts": self.resolved_ts,
                "payload": self.payload, "answers": answers}


class RequestRegistry:
    """The pending-request table. Settled requests stay registered so a duplicate answer is
    recognisable; ``pending()`` lists only open ones, and ``shutdown()`` releases every waiter."""

    def __init__(self):
        self._lock = threading.Lock()
        self._requests: dict[str, PendingRequest] = {}

    def open(self, payload: dict, *, conversation_id: str, run_id: str,
             timeout_s: float = DEFAULT_TIMEOUT_S) -> PendingRequest:
        request = PendingRequest(payload=payload, conversation_id=conversation_id, run_id=run_id,
                                 timeout_s=timeout_s)
        with self._lock:
            self._requests[request.id] = request
        log_event("ask_register", request_id=request.id, conversation_id=conversation_id,
                  run_id=run_id, timeout_s=timeout_s, wizard=request.wizard)
        return request

    def submit_answer(self, request_id: str, answer: dict, *, conversation_id: str,
                      run_id: str) -> tuple[bool, str]:
        """Submit the user's answer to ``request_id`` — the ONE answer path every surface (card
        anchor, QuestionsBar, remote) funnels into. Atomically, under the registry lock: locate
        the request, verify conversation/run ownership, verify it is PENDING, move it through
        ANSWERING (validation; a malformed answer reopens the question for retry), store the
        validated answer and resolve exactly the waiter the worker blocks on.

        Returns ``(True, "")`` on acceptance. Every refusal is deterministic and never pretends
        the answer was taken: the reason codes are ``unknown_request`` / ``stale_request`` /
        ``already_answered`` / ``timed_out`` / ``cancelled`` / ``invalid_answer``."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return self._rejected(request_id, "unknown_request", f"unknown request {request_id}")
            if (request.conversation_id, request.run_id) != (conversation_id, run_id):
                return self._rejected(request_id, "stale_request",
                                      f"stale answer: request {request_id} belongs to conversation "
                                      f"{request.conversation_id}, run {request.run_id}")
            if request.state != OPEN:
                reason = _REASON_FOR_STATE.get(request.state, "not_pending")
                return self._rejected(request_id, reason,
                                      f"request {request_id} is already {request.state}")
            log_event("ask_submit", request_id=request_id)
            if not self._transition(request, ANSWERING):     # atomic with the PENDING check above
                return self._rejected(request_id, "illegal_transition",
                                      f"request {request_id} could not enter answering")
            checked, err = self._check_answer(request, answer)
            if checked is None:
                self._transition(request, OPEN)              # retryable: the question reopens
                return self._rejected(request_id, "invalid_answer", err)
            request.answer = checked
            log_event("ask_accept", request_id=request_id)
            # state → resolved_ts → event, all under this lock: the waiter can only wake to ANSWERED
            if not self._transition(request, ANSWERED):
                request.answer = None
                self._transition(request, OPEN)
                return self._rejected(request_id, "illegal_transition",
                                      f"request {request_id} could not be answered")
            return True, ""

    def resolve(self, request_id: str, answer: dict, *, conversation_id: str, run_id: str) -> tuple[bool, str]:
        """Backwards-compatible name for :meth:`submit_answer` (kept for the panel and older tests)."""
        return self.submit_answer(request_id, answer, conversation_id=conversation_id, run_id=run_id)

    @staticmethod
    def _rejected(request_id: str, reason: str, message: str) -> tuple[bool, str]:
        """The one refusal shape: logged, machine-readable reason first, human detail after it."""
        log_event("ask_reject", request_id=request_id, reason=reason)
        return False, f"{reason}: {message}"

    @staticmethod
    def _check_answer(request: PendingRequest, answer: dict) -> tuple[dict | None, str]:
        """Validate the answer SHAPE against the question and enrich it with the protocol fields."""
        if not isinstance(answer, dict):
            return None, "the answer must be an object"
        questions = request.payload.get("wizard", []) if request.wizard else [request.payload]
        raw = answer.get("answers") if request.wizard else [answer]

        def one(question: dict, part: dict, position: int) -> tuple[dict | None, str]:
            if not isinstance(part, dict) or not _nonempty_str(part.get("answer")):
                return None, f"answer {position} needs a non-empty 'answer' text"
            index = part.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                return None, f"answer {position} needs an integer option index"
            options = question["options"]
            accepts_custom = bool(question.get("allow_free_text") or question.get("custom_value"))
            if index == -1:
                if not accepts_custom:
                    return None, f"answer {position}: this question accepts no custom value"
            elif not 0 <= index < len(options):
                return None, f"answer {position}: index {index} is outside the {len(options)} options"
            entry = {"answer": part["answer"], "index": index}
            if "value" in part:
                entry["value"] = part["value"]
            return entry, ""

        entries: list[dict] = []
        if request.wizard:
            if not isinstance(raw, list):
                return None, "a wizard answer needs an 'answers' list"
            if len(raw) != len(questions):
                return None, f"the wizard has {len(questions)} questions but {len(raw)} answers came back"
            for i, (question, part) in enumerate(zip(questions, raw)):
                entry, err = one(question, part, i)
                if entry is None:
                    return None, err
                entries.append(entry)
        else:
            entry, err = one(questions[0], raw[0], 0)
            if entry is None:
                return None, err
            entries = [entry]
        now = time.time()
        protocol = {"request_id": request.id, "conversation_id": request.conversation_id,
                    "run_id": request.run_id, "answered_ts": now}
        if request.wizard:
            return {"answers": entries, **protocol}, ""
        return {**entries[0], **protocol}, ""

    def cancel_run(self, run_id: str) -> int:
        """Release every OPEN request of ``run_id`` (Stop pressed / turn ended). Returns the count."""
        with self._lock:
            open_now = [r for r in self._requests.values() if r.state == OPEN and r.run_id == run_id]
            for request in open_now:
                self._transition(request, CANCELLED)
            return len(open_now)

    def shutdown(self) -> int:
        """Release every OPEN waiter (application close) and empty the registry."""
        with self._lock:
            open_now = [r for r in self._requests.values() if r.state == OPEN]
            for request in open_now:
                self._transition(request, CANCELLED)
            self._requests.clear()
            return len(open_now)

    @staticmethod
    def _transition(request: PendingRequest, state: str) -> bool:
        """Move ``request`` to ``state`` iff the lifecycle allows it (the caller holds the lock).
        A refused move is logged and leaves the request untouched — a terminal state is final,
        so the deadline, a Stop or a duplicate can never overwrite an accepted answer."""
        if state not in _ALLOWED_FROM.get(request.state, ()):
            log_event("ask_reject", request_id=request.id, reason="illegal_transition",
                      from_state=request.state, to_state=state)
            return False
        request.state = state
        if state in (ANSWERED, CANCELLED, TIMEOUT, REJECTED):
            request.resolved_ts = time.time()
            log_event("ask_resolve", request_id=request.id, state=state)
            request.event.set()
        return True

    @staticmethod
    def _settle(request: PendingRequest, state: str) -> bool:
        """Compatibility name for the guarded transition into an end state."""
        return RequestRegistry._transition(request, state)

    def pending(self, conversation_id: str | None = None) -> list[PendingRequest]:
        """The OPEN requests, oldest first, optionally of one conversation."""
        with self._lock:
            return [r for r in self._requests.values()
                    if r.state == OPEN and (conversation_id is None or r.conversation_id == conversation_id)]

    def wait(self, request: PendingRequest, timeout_s: float | None = None,
             should_stop=None) -> PendingRequest:
        """Block the worker thread until the request settles, Stop fires or the deadline passes.

        Cooperative 0.25 s slices (the cad/dispatch cadence) — never a hard thread kill. The
        deadline only settles a request that is still PENDING by its own lock turn: an answer
        accepted (or being accepted) first always wins."""
        log_event("ask_wait", request_id=request.id)
        deadline = time.monotonic() + (request.timeout_s if timeout_s is None else timeout_s)
        while True:
            if request.event.wait(WAIT_POLL_S):
                return request
            if should_stop is not None and should_stop():
                with self._lock:
                    if request.state == OPEN:
                        self._transition(request, CANCELLED)
                return request
            if time.monotonic() >= deadline:
                with self._lock:
                    if request.state == OPEN:
                        log_event("ask_timeout", request_id=request.id)
                        self._transition(request, TIMEOUT)
                return request


# The one live registry: the ask_user tool opens here, the panel resolves here, AgentWorker.stop
# cancels its run here and the panel's shutdown drains it. Tests build their own RequestRegistry.
REGISTRY = RequestRegistry()


def cancel_run(run_id: str) -> None:
    """Release every OPEN request of ``run_id`` (Stop pressed); NEVER raises — AgentWorker.stop
    calls this on its own failure path and must not fail (the loop modules stay lean: the
    swallow lives here, agent.py keeps a one-line call)."""
    try:
        REGISTRY.cancel_run(run_id)
    except Exception:  # noqa: BLE001 - Stop itself must never fail
        pass
