"""The ``chat()`` generator machinery of the providers: reader thread, ticks,
retries with backoff, request adaptation. ``oscar.providers.BaseProvider`` inherits ``ChatLoop`` and supplies
the wire-specific parts (``_body``, ``_parse_stream``, ``_adapt_to_rejection``, ``_set_vision`` ...); a provider that
signs in also refines ``_reauthenticate`` and ``_http_error`` (no-ops here).

No FreeCAD, no Qt. ``requests.post`` is looked up as a module attribute at call time on purpose: tests patch it.
"""

import queue
import time
from typing import Callable, Generator

import requests

from . import context_guard as _context_guard
from .provider_stream import LIVE, StreamEvent, looks_like_image_rejection
from .provider_transport import (
    EARLY_EOF_BILLED_CHARS, MAX_ATTEMPTS, RETRY_AFTER_CAP_S, RETRY_SLEEP_BUDGET_S, AttemptRunner, ProviderError,
    StopRequested, close_quietly, http_error, is_transport_error, retry_delay,
)
from .turnlog import log_event

_TICK_S = 1.0          # at least one tick per second while a call is open
_POLL_S = 0.2          # Stop is polled every 200 ms
_MAX_ADAPTATIONS = 6   # 400-driven request adaptations per chat() call (each flips a one-way flag)
_READER_JOIN_S = 1.0


def _never() -> bool:
    return False


class CallState:
    """Bookkeeping of one ``chat()`` call: tick counters of the current attempt and the retry budget."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.last_tick = self.t0
        self.slept = 0.0          # total retry sleep so far (budget: 90 s)
        self.begin_attempt()

    def begin_attempt(self) -> None:
        """Reset the per-attempt counters (output sizes, phase, liveness) at the start of every attempt."""
        now = time.monotonic()
        self.reasoning_chars = 0
        self.text_chars = 0
        self.tool_chars: dict[int, int] = {}
        self.phase = "connecting"
        self.tool: str | None = None
        self.last_event = now
        self.last_alive = now
        self.emitted = False      # partial output was delivered to the consumer during this attempt

    def output_chars(self) -> int:
        """Characters of reasoning, text and tool arguments the current attempt has delivered so far."""
        return self.reasoning_chars + self.text_chars + sum(self.tool_chars.values())

    def note(self, kind: str, payload, now: float) -> None:
        """Account for one event that is delivered to the consumer."""
        self.last_event = now
        if kind == "text":
            self.text_chars += len(payload or "")
            self.phase, self.emitted = "writing", True
        elif kind == "reasoning":
            self.reasoning_chars += len(payload or "")
            self.phase, self.emitted = "thinking", True
        elif kind == "tool_progress" and isinstance(payload, dict):
            self.tool_chars[int(payload.get("index") or 0)] = int(payload.get("chars") or 0)
            self.tool = payload.get("name") or self.tool
            self.phase, self.emitted = "tool_args", True

    def tick(self, now: float, *, phase: str | None = None, retry: list[int] | None = None) -> StreamEvent:
        """The ``("tick", {...})`` liveness event for time ``now``; ``phase`` / ``retry`` override the current ones."""
        shown = phase or self.phase
        return ("tick", {
            "elapsed": round(now - self.t0, 2),
            "reasoning_chars": self.reasoning_chars,
            "text_chars": self.text_chars,
            "tool_chars": sum(self.tool_chars.values()),
            "out_tokens": None,
            "phase": shown,
            "tool": self.tool if shown == "tool_args" else None,
            "retry": list(retry) if retry else None,
            "idle_s": round(max(0.0, now - max(self.last_event, self.last_alive)), 2),
        })


class ChatLoop:
    """Mixin: ``chat()`` and everything it needs around the wire-specific ``_body`` / ``_parse_stream``."""

    max_retries = MAX_ATTEMPTS
    connect_timeout = 15
    read_timeout = 600
    backoff_scale = 1.0           # multiplies every retry sleep (tests set 0); the 90 s budget is counted unscaled

    # -- public -----------------------------------------------------------

    def chat(
        self,
        system: str,
        messages: list,
        tools: list,
        should_stop: Callable[[], bool] | None = None,
        *,
        tool_choice: str | None = None,
        reasoning: str | None = None,
        max_tokens: int | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> Generator[StreamEvent, None, None]:
        """Stream one model reply as event tuples (the events are documented in ``oscar.providers``).

        ``tool_choice``: None | "auto" | "required" | "none" | "<tool name>". ``reasoning``: per-call effort
        ``off|low|medium|high|auto`` overriding ``Profile.reasoning``. ``max_tokens``: per-call output cap
        (still bounded by the learned cap). ``parallel_tool_calls=False`` asks for at most one call per reply.

        Raises ``StopRequested`` when ``should_stop()`` turns true (the reader is cancelled first). Closing
        the generator cancels the in-flight request too. Transport failures are retried up to
        ``max_retries`` attempts with jittered backoff / ``Retry-After``; an HTTP 400 that names an optional
        or unsupported parameter adapts the request and retries without counting as an attempt. A first HTTP 401
        that ``_reauthenticate`` can fix (a signed-in provider renews its token) is sent once more, not counted as
        an attempt either; every other HTTP failure becomes ``_http_error(status, text, headers)``.
        """
        should_stop = should_stop or _never
        # What the model may see: no credential, wherever it came from (context_guard). Copies; the history stays.
        _context_guard.remember(getattr(getattr(self, "profile", None), "api_key", ""))
        system, messages = _context_guard.guard_request(system, messages)
        opts = {"tool_choice": tool_choice, "reasoning": reasoning, "max_tokens": max_tokens,
                "parallel_tool_calls": parallel_tool_calls}
        call = CallState()
        failed = 0           # attempts that failed so far
        adaptations = 0
        reauthed = False     # the one resend after a 401 was used
        billed_drops = 0     # early EOFs after a long generation (each retry replays it, billed)
        vision_dropped = ""  # the 400 body that made this call retry without images (learned once the retry works)
        while True:
            if should_stop():
                raise StopRequested()
            call.begin_attempt()
            body = self._body(system, messages, tools, **opts)
            run = AttemptRunner(lambda r, b=body: self._reader(r, b), name=f"oscar-reader-{self.kind or self.name}")
            run.start()
            try:
                outcome = yield from self._drain(run, call, should_stop)
            finally:
                self._reap(run)
            kind, data = outcome
            if kind == "ok":
                if vision_dropped:
                    # The text-only retry was accepted, so the model really cannot take images: remember it for later
                    # turns (a retry that fails could have been something else entirely, and learns nothing).
                    self._learn(vision=False, evidence=vision_dropped)
                return
            reason = "retry"
            headers = None
            if kind == "http":
                status, text, headers = data["status"], data["text"], data["headers"]
                if status == 401 and not reauthed and self._reauthenticate(text):
                    # The token was refused (expired early, revoked server-side): the next attempt reads a renewed
                    # one in ``_headers`` (reader thread). Once only, and not an attempt: nothing was retried.
                    reauthed = True
                    continue
                if status == 400:
                    if self._vision_ok and self._body_has_images(body) and looks_like_image_rejection(text):
                        # Text-only model: degrade this session to text and
                        # retry — the note in the rebuilt body tells the
                        # model what it is missing. Only an UNAMBIGUOUS
                        # rejection is remembered across sessions (a size or
                        # base64 complaint is about this image, not the model),
                        # and only once the text-only retry has worked.
                        self._set_vision(False, remember=False)
                        vision_dropped = text
                        continue
                    if adaptations < _MAX_ADAPTATIONS and self._adapt_to_rejection(text, body):
                        adaptations += 1
                        continue
                error = self._http_error(status, text, headers)
                advised = getattr(error, "retry_after", None)
                if error.retriable and advised is not None \
                        and advised > min(RETRY_AFTER_CAP_S, RETRY_SLEEP_BUDGET_S - call.slept):
                    # The server asks for a longer wait than this call may still sleep: retrying early would only be
                    # refused again, so the user hears it now (``http_error`` never sets ``retry_after``).
                    error.retriable = False
                if not error.retriable:
                    yield ("error", error)
                    return
            elif kind == "exc":
                if not is_transport_error(data):
                    yield ("error", RuntimeError(f"{type(data).__name__}: {data}"))
                    return
                error = data
                if call.output_chars() > EARLY_EOF_BILLED_CHARS:
                    # a connection dropped in the middle of a long generation is billed like an early EOF
                    billed_drops += 1
                    if billed_drops > 1:
                        yield ("error", RuntimeError(f"request failed after a long generation was cut off: {error}"))
                        return
            elif kind == "error":
                error = data
                if not getattr(error, "retriable", False):
                    yield ("error", error)
                    return
            elif kind == "incomplete":
                reason = "stream_dropped"
                error = RuntimeError("stream ended early (no terminal marker)")
                if call.output_chars() > EARLY_EOF_BILLED_CHARS:
                    billed_drops += 1
                    if billed_drops > 1:
                        yield ("error", error)
                        return
            else:  # anything unexpected: terminal
                yield ("error", data if isinstance(data, Exception) else RuntimeError(str(data)))
                return
            failed += 1
            budget = RETRY_SLEEP_BUDGET_S - call.slept
            if failed >= self.max_retries or budget <= 0:
                yield ("error", self._exhausted(kind, error, failed))
                return
            delay = min(retry_delay(failed - 1, headers), budget)
            call.slept += delay
            delay *= self.backoff_scale
            log_event("provider_retry", reason=reason, attempt=failed, delay_s=round(delay, 1),
                      status=getattr(error, "status", 0), model=getattr(self.profile, "model", ""))
            if call.emitted:
                yield ("reset", reason)   # whatever this attempt streamed is void
                call.begin_attempt()
            yield from self._sleep(delay, call, should_stop, [failed + 1, self.max_retries])

    @staticmethod
    def _exhausted(kind: str, error, attempts: int) -> Exception:
        """The final error after the last attempt failed."""
        if kind == "exc":
            return RuntimeError(f"request failed after {attempts} attempts: {error}")
        return error

    # -- hooks for providers that sign in (no-ops here: a key in the profile is not renewed) ----------------------

    def _reauthenticate(self, text: str) -> bool:
        """After the first HTTP 401 of a call (``text`` = its body): make the next attempt carry fresh credentials.
        True sends the same request once more (not counted as an attempt). Runs on the consumer thread, so it must
        not block: renewing belongs in ``_headers``, which runs on the reader thread."""
        return False

    def _http_error(self, status: int, text: str, headers) -> ProviderError:
        """The error for an HTTP failure (``text`` = the first 2000 characters of the body, ``headers`` the response's).
        Its ``retriable`` decides whether the request is sent again; a ``retry_after`` longer than the call may still
        wait makes it final."""
        return http_error(status, text)

    def _drain(self, run: AttemptRunner, call: CallState, should_stop) -> Generator[StreamEvent, None, tuple]:
        """Forward the reader's events to the caller (with ticks) and return the attempt's outcome.

        Outcome is ``("ok", None)`` (terminal marker seen), ``("incomplete", None)`` (stream ended early),
        ``("http", {status, text, headers})``, ``("exc", exception)`` or ``("error", ProviderError)`` (an
        in-stream error event). Raises ``StopRequested`` when Stop is pressed.
        """
        outcome: tuple | None = None
        while True:
            try:
                item = run.events.get(timeout=_POLL_S)
            except queue.Empty:
                item = None
            if should_stop():
                raise StopRequested()
            now = time.monotonic()
            if item is not None:
                kind, payload = item
                if kind == "_end":
                    break
                if kind == "_alive":
                    call.last_alive = now
                elif kind == "_status":
                    if call.phase == "connecting":
                        call.phase = "waiting"
                elif kind == "stream_end":
                    if outcome is None:
                        outcome = ("ok" if (payload or {}).get("complete") else "incomplete", None)
                elif kind == "error":
                    if outcome is None:
                        outcome = ("error", payload)
                elif kind == "_http":
                    outcome = ("http", payload)
                elif kind == "_exc":
                    if outcome is None and not run.abort.is_set():
                        outcome = ("exc", payload)
                else:
                    call.note(kind, payload, now)
                    yield item
            if now - call.last_tick >= _TICK_S:
                call.last_tick = now
                yield call.tick(now)
        return outcome or ("exc", RuntimeError("the provider attempt ended without a result"))

    def _sleep(self, delay: float, call: CallState, should_stop, retry: list[int]) -> Generator[StreamEvent, None, None]:
        """Backoff sleep that keeps ticking (phase ``retrying``) and polls Stop every 200 ms."""
        if delay <= 0:
            return
        deadline = time.monotonic() + delay
        call.last_tick = time.monotonic()
        yield call.tick(call.last_tick, phase="retrying", retry=retry)
        while True:
            if should_stop():
                raise StopRequested()
            now = time.monotonic()
            if now >= deadline:
                return
            if now - call.last_tick >= _TICK_S:
                call.last_tick = now
                yield call.tick(now, phase="retrying", retry=retry)
            time.sleep(min(_POLL_S, deadline - now))

    def _reap(self, run: AttemptRunner) -> None:
        """Cancel the attempt (a no-op after a normal end) and log a reader that outlives its join."""
        if not run.cancel(_READER_JOIN_S):
            log_event("reader_leak", n=AttemptRunner.live_reader_count())

    def _reader(self, run: AttemptRunner, body: dict) -> None:
        """Reader thread: send the request, then push every parsed event into ``run.events``.

        Internal events (never delivered to callers): ``_status`` (headers received), ``_http`` (an error
        status with its body), ``_alive`` (SSE comment line), ``_exc`` / ``_end`` (added by the runner). A
        ``ProviderError`` from ``_endpoint`` / ``_headers`` (no usable sign-in) becomes an ``error`` event.
        """
        LIVE.callback = lambda: run.put("_alive")
        response = None
        try:
            if run.abort.is_set():
                return
            try:
                endpoint, headers = self._endpoint(), self._headers()
            except ProviderError as e:   # a signed-in provider without usable credentials: nothing is sent
                run.put("error", e)
                return
            if run.abort.is_set():       # Stop arrived while the headers were made (a token renewal can take a while)
                return
            response = requests.post(
                endpoint,
                headers=headers,
                json=body,
                stream=True,
                timeout=(self.connect_timeout, self.read_timeout),
            )
            run.response = response
            if run.abort.is_set():
                return
            status = getattr(response, "status_code", 200)
            run.put("_status", {"status": status})
            if status >= 400:
                try:
                    text = str(getattr(response, "text", "") or "")[:2000]
                except Exception:  # noqa: BLE001 - undecodable body
                    text = ""
                run.put("_http", {"status": status, "text": text,
                                  "headers": getattr(response, "headers", None) or {}})
                return
            try:
                for event in self._parse_stream(response, run.abort.is_set):
                    run.put(*event)
            except ProviderError as e:    # e.g. a 200 answer that is not an event stream at all
                run.put("error", e)
        finally:
            LIVE.callback = None
            if response is not None:
                close_quietly(response)
