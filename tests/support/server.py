"""tests/support/server.py - ScriptedServer: a local HTTP server that speaks a provider's SSE dialect.

It is driven by the same ``Round`` scripts as ``support.scripted_llm.ScriptedProvider`` (``say``, ``calls``,
``think``, ``bad_json``, ...), so one scenario can run against the fake provider (fast, no HTTP) and against
the REAL providers over a real socket (transport, parsing, retries, the reader thread).

* dialects: ``"openai"`` (chat completions, also Ollama / LM Studio / OpenRouter shaped) and ``"anthropic"``
  (messages API). Any POST path is accepted; ``GET .../models`` answers a model list.
* script: a list consumed one item per request, or a REACTIVE callable ``script(request_json) -> item`` that
  decides from the request body (see ``last_event``), which is what a loop with nudges, gates and retries needs:
  answering by request ORDINAL desynchronises as soon as one extra request is made.
* items: a ``Round`` (rendered as a stream), ``http_error(...)`` (a 429 with ``Retry-After``, a 5xx, ...),
  ``drop_after`` / ``stall_after`` / ``slow`` (a stream that is cut, goes silent, or trickles), ``raw(...)`` (a
  pre-rendered body, what ``mock_llm_server`` used to hand out) or a plain ``str`` (= ``say(str)``).
* state is per instance (no class-level globals); every request is recorded in ``server.requests``.

Nothing here touches the user's data: the server binds 127.0.0.1 only, and handler threads are daemons.
"""

import json
import select
import socket
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from oscar.stopreason import normalize_stop_reason
from support.scripted_llm import Round, say

SPLIT_CHARS = 2000        # reasoning / text events longer than this are streamed in several chunks
ARG_CHUNK = 40            # tool-call arguments are streamed in pieces of this size (exercises accumulation)

_OPENAI_FINISH = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length",
                  "refusal": "content_filter", "content_filter": "content_filter"}
_ANTHROPIC_STOP = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens",
                   "refusal": "refusal", "content_filter": "refusal"}
_REASON = {400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found", 408: "Request Timeout",
           409: "Conflict", 425: "Too Early", 429: "Too Many Requests", 500: "Internal Server Error",
           502: "Bad Gateway", 503: "Service Unavailable", 504: "Gateway Timeout", 529: "Overloaded"}


# -- script items -----------------------------------------------------------------------------------------

@dataclass
class HttpFault:
    """Answer with an HTTP error status instead of a stream."""

    status: int = 500
    body: str = ""
    headers: dict = field(default_factory=dict)
    delay: float = 0.0


@dataclass
class StreamFault:
    """A normal ``Round`` delivered with a transport fault.

    ``mode``: ``"drop"`` ends the stream after ``after`` SSE chunks WITHOUT the terminal marker (``style``
    ``"eof"``: the HTTP body ends cleanly, ``"cut"``: the connection closes mid-body (``IncompleteRead``),
    ``"reset"``: TCP reset); ``"stall"`` sends ``after`` chunks, then goes silent for
    ``seconds`` (or until the client disconnects); ``"slow"`` waits ``gap`` seconds between chunks."""

    round: Round
    mode: str = "drop"
    after: int = 1
    style: str = "eof"
    seconds: float = 30.0
    gap: float = 0.0


@dataclass
class RawReply:
    """A pre-rendered response body (the bytes ``mock_llm_server`` builds), sent as is."""

    body: bytes
    status: int = 200
    content_type: str = "text/event-stream"
    headers: dict = field(default_factory=dict)


def http_error(status: int = 500, body: str = "", *, retry_after: float | str | None = None,
               headers: dict | None = None, delay: float = 0.0) -> HttpFault:
    """An HTTP error answer. ``retry_after`` sets the ``Retry-After`` header (seconds or an HTTP date string);
    an empty ``body`` becomes a small JSON error object in the shape providers return."""
    hdrs = dict(headers or {})
    if retry_after is not None:
        hdrs["Retry-After"] = str(retry_after)
    if not body:
        kind = {429: "rate_limit_error", 529: "overloaded_error", 401: "authentication_error"}.get(status, "api_error")
        body = json.dumps({"error": {"type": kind, "message": _REASON.get(status, "error").lower()}})
    return HttpFault(status, body, hdrs, delay)


def drop_after(rnd: Round, chunks: int = 1, *, style: str = "eof") -> StreamFault:
    """``rnd`` cut after ``chunks`` SSE chunks, no terminal marker (a mid-stream drop / early EOF)."""
    return StreamFault(rnd, "drop", chunks, style=style)


def stall_after(rnd: Round, chunks: int = 1, *, seconds: float = 30.0) -> StreamFault:
    """``rnd`` stalled mid-stream: ``chunks`` chunks, then silence until the client leaves or ``seconds`` pass."""
    return StreamFault(rnd, "stall", chunks, seconds=seconds)


def slow(rnd: Round, gap: float = 0.2) -> StreamFault:
    """``rnd`` delivered with ``gap`` seconds between chunks."""
    return StreamFault(rnd, "slow", gap=gap)


def raw(body: bytes | str, *, status: int = 200, content_type: str = "text/event-stream",
        headers: dict | None = None) -> RawReply:
    """A pre-rendered body (e.g. an HTML login page answered with 200, or SSE without a charset)."""
    return RawReply(body.encode("utf-8") if isinstance(body, str) else body, status, content_type, dict(headers or {}))


# -- rendering a Round as SSE -----------------------------------------------------------------------------

def _tool_args(call) -> str:
    """The raw argument text a model would stream for ``call`` (truncated text for a ``bad_json`` call)."""
    if getattr(call, "args_error", ""):
        return getattr(call, "raw_args", "") or (call.arguments or {}).get("_raw", "")
    return json.dumps(call.arguments or {})


def _pieces(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def _event_stop(events) -> str:
    return next((str(p) for k, p in events if k == "stop_reason"), "")


def _error_kind(message: str) -> tuple[str, str]:
    low = message.lower()
    if "overload" in low:
        return "overloaded_error", message
    if "rate" in low:
        return "rate_limit_error", message
    return "api_error", message


def _sse(obj: Any, event: str = "") -> bytes:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(obj)}\n\n".encode("utf-8")


def render_openai(rnd: Round, model: str = "scripted-model") -> tuple[list[bytes], list[bytes]]:
    """``(body_chunks, terminal_chunks)`` of an OpenAI chat-completions stream for ``rnd``. The terminal chunks
    (finish reason, usage, ``[DONE]``) are what a dropped stream lacks."""
    cid = "chatcmpl-" + uuid.uuid4().hex[:8]

    def chunk(delta=None, finish=None, **extra):
        choice: dict = {"index": 0, "delta": delta or {}}
        if finish:
            choice["finish_reason"] = finish
        return _sse({"id": cid, "object": "chat.completion.chunk", "model": model, "choices": [choice], **extra})

    body: list[bytes] = []
    usage = None
    has_calls = False
    first = True
    error: tuple[str, str] | None = None
    for kind, payload in rnd.events:
        if kind == "text" and payload:
            for part in _pieces(str(payload), SPLIT_CHARS):
                body.append(chunk({"role": "assistant", "content": part} if first else {"content": part}))
                first = False
        elif kind == "reasoning" and payload:
            for part in _pieces(str(payload), SPLIT_CHARS):
                body.append(chunk({"role": "assistant", "reasoning_content": part} if first else {"reasoning_content": part}))
                first = False
        elif kind == "tool_calls":
            for index, call in enumerate(payload):
                has_calls = True
                body.append(chunk({"tool_calls": [{"index": index, "id": call.id, "type": "function",
                                                   "function": {"name": call.name, "arguments": ""}}]}))
                for part in _pieces(_tool_args(call), ARG_CHUNK):
                    if part:
                        body.append(chunk({"tool_calls": [{"index": index, "function": {"arguments": part}}]}))
        elif kind == "usage" and isinstance(payload, dict):
            usage = payload
        elif kind == "error":
            error = _error_kind(str(payload))
    if error is not None:
        body.append(_sse({"error": {"type": error[0], "message": error[1]}}))
        return body, []
    raw_stop = _event_stop(rnd.events)
    normal = normalize_stop_reason(raw_stop) if raw_stop else ("tool_use" if has_calls else "end_turn")
    finish = _OPENAI_FINISH.get(normal, raw_stop or "stop")
    terminal = [chunk(finish=finish)]
    if usage:
        terminal.append(_sse({"id": cid, "object": "chat.completion.chunk", "model": model, "choices": [], "usage": usage}))
    terminal.append(b"data: [DONE]\n\n")
    return body, terminal


def render_anthropic(rnd: Round, model: str = "scripted-model") -> tuple[list[bytes], list[bytes]]:
    """``(body_chunks, terminal_chunks)`` of an Anthropic messages stream for ``rnd`` (thinking, text and
    tool_use blocks in event order; consecutive reasoning / text events share one block)."""
    usage = next((p for k, p in rnd.events if k == "usage" and isinstance(p, dict)), None) or {}
    start_usage = {k: v for k, v in usage.items() if k != "output_tokens"} or {"input_tokens": 1}
    start_usage.setdefault("output_tokens", 1)
    body: list[bytes] = [_sse({"type": "message_start", "message": {
        "id": "msg_" + uuid.uuid4().hex[:8], "type": "message", "role": "assistant", "model": model,
        "content": [], "usage": start_usage}}, "message_start")]
    index = -1
    open_kind = ""
    has_calls = False
    error: tuple[str, str] | None = None

    def close():
        nonlocal open_kind
        if open_kind == "reasoning":   # a thinking block is signed when it closes
            body.append(_sse({"type": "content_block_delta", "index": index,
                              "delta": {"type": "signature_delta", "signature": "scripted-signature"}},
                             "content_block_delta"))
        if open_kind:
            body.append(_sse({"type": "content_block_stop", "index": index}, "content_block_stop"))
            open_kind = ""

    def open_block(kind: str, block: dict):
        nonlocal index, open_kind
        close()
        index += 1
        open_kind = kind
        body.append(_sse({"type": "content_block_start", "index": index, "content_block": block}, "content_block_start"))

    for kind, payload in rnd.events:
        if kind in ("text", "reasoning") and payload:
            if open_kind != kind:
                if kind == "text":
                    open_block("text", {"type": "text", "text": ""})
                else:
                    open_block("reasoning", {"type": "thinking", "thinking": "", "signature": ""})
            for part in _pieces(str(payload), SPLIT_CHARS):
                delta = ({"type": "text_delta", "text": part} if kind == "text"
                         else {"type": "thinking_delta", "thinking": part})
                body.append(_sse({"type": "content_block_delta", "index": index, "delta": delta}, "content_block_delta"))
        elif kind == "tool_calls":
            for call in payload:
                has_calls = True
                open_block("tool", {"type": "tool_use", "id": call.id, "name": call.name, "input": {}})
                for part in _pieces(_tool_args(call), ARG_CHUNK):
                    if part:
                        body.append(_sse({"type": "content_block_delta", "index": index,
                                          "delta": {"type": "input_json_delta", "partial_json": part}},
                                         "content_block_delta"))
        elif kind == "error":
            error = _error_kind(str(payload))
    if error is not None:
        close()
        body.append(_sse({"type": "error", "error": {"type": error[0], "message": error[1]}}, "error"))
        return body, []
    close()
    raw_stop = _event_stop(rnd.events)
    normal = normalize_stop_reason(raw_stop) if raw_stop else ("tool_use" if has_calls else "end_turn")
    stop = _ANTHROPIC_STOP.get(normal, raw_stop or "end_turn")
    terminal = [_sse({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                      "usage": {"output_tokens": usage.get("output_tokens", 1)}}, "message_delta"),
                _sse({"type": "message_stop"}, "message_stop")]
    return body, terminal


RENDERERS = {"openai": render_openai, "anthropic": render_anthropic}


# -- request helpers --------------------------------------------------------------------------------------

def _text_of(content) -> str:
    """Plain text of an OpenAI / Anthropic message ``content`` (string, or a list of parts / blocks)."""
    if isinstance(content, str):
        return content
    out = []
    for part in content or []:
        if isinstance(part, dict):
            if part.get("type") in ("text", "input_text", "output_text"):
                out.append(str(part.get("text", "")))
            elif part.get("type") == "tool_result":
                out.append(_text_of(part.get("content")))
    return "\n".join(x for x in out if x)


def _is_screenshot_caption(content, text: str) -> bool:
    """True for the message the agent appends after a tool that captured a view: an image plus a caption such as
    ``(screenshot after get_view)``. It is the picture belonging to the tool result before it, never a new user turn."""
    if not isinstance(content, list) or not text.strip().startswith("(screenshot"):
        return False
    return any(isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image") for p in content)


def last_event(request_json: dict) -> tuple[str, str, str]:
    """What the model is answering: ``("tool", tool_call_id, result_text)`` when the conversation ends with a tool
    result, ``("user", "", text)`` when it ends with user text (a prompt, a nudge, a steer), else ``("none", "", "")``.

    Handles the OpenAI (role ``tool`` / ``user``) and the Anthropic (``tool_result`` blocks inside a user message)
    wire formats; a user message that only carries an image, or an image with the agent's ``(screenshot ...)`` caption,
    is skipped, so the answer to "last event" is stable whether or not the agent attached a picture."""
    for msg in reversed((request_json or {}).get("messages") or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "tool":
            return "tool", str(msg.get("tool_call_id", "")), _text_of(msg.get("content"))
        if role == "user":
            content = msg.get("content")
            if isinstance(content, list):
                results = [p for p in content if isinstance(p, dict) and p.get("type") == "tool_result"]
                if results:
                    last = results[-1]
                    return "tool", str(last.get("tool_use_id", "")), _text_of(last.get("content"))
            text = _text_of(content)
            if _is_screenshot_caption(content, text):
                continue
            if text.strip():
                return "user", "", text
    return "none", "", ""


@dataclass
class RecordedRequest:
    """One request the server received."""

    n: int
    method: str
    path: str
    headers: dict
    raw: bytes
    json: Any
    t: float
    kind: str = ""                 # what was served: round | http_fault | stream_fault | raw | exhausted
    status: int = 0
    closed_by_client: bool = False
    _closed: threading.Event = field(default_factory=threading.Event, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def finished(self) -> bool:
        """True once the server wrote the whole reply (a client can finish reading a moment BEFORE this turns true;
        use ``wait_finished`` right after a call returns)."""
        return self._done.is_set()

    @finished.setter
    def finished(self, value: bool) -> None:
        if value:
            self._done.set()

    def wait_finished(self, timeout: float = 2.0) -> bool:
        """Block until the whole reply was written; True when it was within ``timeout``."""
        return self._done.wait(timeout)

    @property
    def messages(self) -> list:
        body = self.json if isinstance(self.json, dict) else {}
        return body.get("messages") or []

    @property
    def tool_names(self) -> list[str]:
        """Names of the tools offered in this request (OpenAI ``function.name`` or Anthropic ``name``)."""
        body = self.json if isinstance(self.json, dict) else {}
        names = []
        for t in body.get("tools") or []:
            if isinstance(t, dict):
                names.append((t.get("function") or {}).get("name") or t.get("name") or "")
        return names

    @property
    def last(self) -> tuple[str, str, str]:
        """``last_event`` of this request."""
        return last_event(self.json if isinstance(self.json, dict) else {})

    def wait_closed(self, timeout: float = 2.0) -> bool:
        """Block until the client was seen to close this connection; True when it did within ``timeout``."""
        return self._closed.wait(timeout)

    def _mark_closed(self) -> None:
        self.closed_by_client = True
        self._closed.set()


# -- the server -------------------------------------------------------------------------------------------

class ScriptedServer:
    """A scripted LLM endpoint on 127.0.0.1 (ephemeral port by default). Use as a context manager or call
    ``shutdown()``. ``script`` is a list of items (consumed in order) or a callable ``script(request_json)``
    returning an item (reactive mode); a ``str`` item is ``say(str)``, ``bytes`` is ``raw(bytes)``.

    Exhaustion: with ``strict`` (default) a request beyond the list is answered with HTTP 500 and recorded in
    ``unexpected`` (so a loop that asks for more rounds than intended fails loudly in ``assert_exhausted`` /
    ``assert_no_unexpected``); with ``repeat_last`` the last item answers every further request (the old mock's
    behaviour)."""

    def __init__(self, script=None, *, dialect: str = "openai", host: str = "127.0.0.1", port: int = 0,
                 repeat_last: bool = False, strict: bool = True, model: str = "scripted-model",
                 chunk_delay: float = 0.0, start: bool = True):
        if dialect not in RENDERERS:
            raise ValueError(f"unknown dialect {dialect!r}; use one of {sorted(RENDERERS)}")
        self.dialect = dialect
        self.model = model
        self.repeat_last = repeat_last
        self.strict = strict
        self.chunk_delay = chunk_delay
        self.requests: list[RecordedRequest] = []
        self.unexpected: list[RecordedRequest] = []
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._fn: Callable[[Any], Any] | None = script if callable(script) else None
        self._items: list = [] if callable(script) or script is None else list(script)
        self._cursor = 0
        self._host, self._port = host, port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        if start:
            self.start()

    # lifecycle ------------------------------------------------------------------------------------------

    def start(self) -> "ScriptedServer":
        """Bind and serve on a daemon thread (idempotent)."""
        if self._httpd is not None:
            return self
        self._closing.clear()
        owner = self

        class Handler(_Handler):
            server_owner = owner

        httpd = ThreadingHTTPServer((self._host, self._port), Handler)
        httpd.daemon_threads = True
        self._port = httpd.server_address[1]     # remembered: ``url`` stays valid (and a restart reuses it) after shutdown
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True,
                                        name="scripted-server")
        self._thread.start()
        return self

    def shutdown(self) -> None:
        """Stop serving, release stalled handlers and close the socket (idempotent)."""
        self._closing.set()
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            try:
                httpd.shutdown()
            finally:
                httpd.server_close()
        if self._thread is not None:
            self._thread.join(2.0)
            self._thread = None

    close = server_close = shutdown

    def __enter__(self) -> "ScriptedServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.shutdown()

    # addressing -----------------------------------------------------------------------------------------

    @property
    def port(self) -> int:
        """The bound port."""
        return self._httpd.server_address[1] if self._httpd is not None else self._port

    @property
    def url(self) -> str:
        """``http://127.0.0.1:<port>``."""
        return f"http://{self._host}:{self.port}"

    @property
    def base_url(self) -> str:
        """The ``Profile.base_url`` to use: ``<url>/v1`` for the OpenAI dialect, ``<url>`` for Anthropic."""
        return self.url + ("/v1" if self.dialect == "openai" else "")

    @property
    def server_address(self) -> tuple:
        """``(host, port)`` like ``http.server`` objects (the old ``make_server`` returned one)."""
        return (self._host, self.port)

    # script ---------------------------------------------------------------------------------------------

    @property
    def request_count(self) -> int:
        """Requests received so far."""
        with self._lock:
            return len(self.requests)

    @property
    def remaining(self) -> int:
        """Scripted items not yet served (0 for a callable script)."""
        with self._lock:
            return max(len(self._items) - self._cursor, 0)

    def add(self, *items) -> "ScriptedServer":
        """Append items to a list script."""
        with self._lock:
            self._items.extend(items)
        return self

    def reset(self) -> None:
        """Forget the recorded requests and rewind a list script (per-turn restarts in one test)."""
        with self._lock:
            self.requests.clear()
            self.unexpected.clear()
            self._cursor = 0

    def assert_exhausted(self) -> None:
        """Fail when scripted items were never requested or an extra request arrived."""
        self.assert_no_unexpected()
        assert self.remaining == 0, f"{self.remaining} scripted item(s) were never requested"

    def assert_no_unexpected(self) -> None:
        """Fail when a request arrived after a strict list script ran out."""
        if self.unexpected:
            first = self.unexpected[0]
            raise AssertionError(f"unexpected extra provider request #{first.n}; last event: {first.last!r}")

    def wait_for_requests(self, n: int, timeout: float = 5.0) -> bool:
        """Block until ``n`` requests were received; True when they were within ``timeout``."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.request_count >= n:
                return True
            time.sleep(0.01)
        return self.request_count >= n

    def _pick(self, rec: RecordedRequest):
        """The item that answers ``rec`` (None = exhausted)."""
        if self._fn is not None:
            return self._fn(rec.json if rec.json is not None else {})
        with self._lock:
            if self._cursor < len(self._items):
                item = self._items[self._cursor]
                self._cursor += 1
                return item
            if self.repeat_last and self._items:
                return self._items[-1]
        return None


def _coerce(item):
    if isinstance(item, str):
        return say(item)
    if isinstance(item, (bytes, bytearray)):
        return RawReply(bytes(item))
    return item


class _Handler(BaseHTTPRequestHandler):
    """One request. Streams use ``Transfer-Encoding: chunked`` like every real SSE server: ``requests`` reads a
    close-delimited body in 512-byte blocks and would hold a small event back until the stream ends, which
    would make every stall and timing test lie. Errors and raw bodies use ``Content-Length``."""

    server_owner: ScriptedServer
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence http.server's stderr logging
        pass

    # GET: a model list so ``list_models`` / test-connection code has something to read
    def do_GET(self):  # noqa: N802 - http.server API
        owner = self.server_owner
        if self.path.rstrip("/").endswith("/models"):
            body = json.dumps({"object": "list", "data": [{"id": owner.model, "object": "model"}]}).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b'{"error": "not found"}', "application/json")

    def do_POST(self):  # noqa: N802 - http.server API
        owner = self.server_owner
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            length = 0
        raw_body = self.rfile.read(length) if length else b""
        try:
            parsed = json.loads(raw_body.decode("utf-8")) if raw_body else None
        except ValueError:
            parsed = None
        with owner._lock:
            rec = RecordedRequest(len(owner.requests) + 1, "POST", self.path,
                                  {k.lower(): v for k, v in self.headers.items()}, raw_body, parsed, time.monotonic())
            owner.requests.append(rec)
        try:
            item = _coerce(owner._pick(rec))
        except Exception as e:  # noqa: BLE001 - a reactive script that crashes is a test bug: say so on the wire
            rec.kind, rec.status = "script_error", 500
            self._send(500, json.dumps({"error": {"type": "script_error", "message": f"{type(e).__name__}: {e}"}}).encode(),
                       "application/json")
            return
        if item is None:
            with owner._lock:
                owner.unexpected.append(rec)
            rec.kind, rec.status = "exhausted", 500
            self._send(500, json.dumps({"error": {"type": "script_exhausted",
                                                  "message": f"unexpected extra request #{rec.n}"}}).encode(),
                       "application/json")
            return
        self._streaming = False
        try:
            self._serve(item, rec)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            rec._mark_closed()
        except Exception as e:  # noqa: BLE001 - an unserveable item is a test bug: answer 500, never a stderr traceback
            rec.kind, rec.status = "serve_error", 500
            if not self._streaming:
                self._send(500, json.dumps({"error": {"type": "serve_error", "message": f"{type(e).__name__}: {e}"}}).encode(),
                           "application/json")

    # serving --------------------------------------------------------------------------------------------

    def _serve(self, item, rec: RecordedRequest) -> None:
        owner = self.server_owner
        if isinstance(item, HttpFault):
            rec.kind, rec.status = "http_fault", item.status
            self._sleep(item.delay)
            self._send(item.status, item.body.encode("utf-8"), "application/json", item.headers)
            rec.finished = True
            return
        if isinstance(item, RawReply):
            rec.kind, rec.status = "raw", item.status
            self._send(item.status, item.body, item.content_type, item.headers)
            rec.finished = True
            return
        fault = item if isinstance(item, StreamFault) else None
        rnd = fault.round if fault is not None else item
        if not isinstance(rnd, Round):
            raise TypeError(f"cannot serve a {type(item).__name__}; use a Round, http_error(), drop_after(), raw() or a str")
        rec.kind, rec.status = ("stream_fault" if fault else "round"), 200
        if rnd.raise_exc is not None:   # a provider that raises has no wire equivalent; the closest is a 500
            rec.kind, rec.status = "http_fault", 500
            self._send(500, json.dumps({"error": {"type": "api_error", "message": str(rnd.raise_exc)}}).encode(),
                       "application/json")
            rec.finished = True
            return
        self._sleep(rnd.delay)
        body, terminal = RENDERERS[owner.dialect](rnd, owner.model)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")   # no charset on purpose: the client must cope
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        self._streaming = True
        self.wfile.flush()
        gap = (fault.gap if fault is not None and fault.mode == "slow" else owner.chunk_delay)
        cut = None
        if fault is not None and fault.mode in ("drop", "stall"):
            cut = max(0, min(fault.after, len(body)))
            chunks = body[:cut]
        else:
            chunks = body + terminal
        for i, chunk in enumerate(chunks):
            if owner._closing.is_set():
                return
            if i and gap:
                self._sleep(gap)
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
        if fault is not None and fault.mode == "stall":
            self._hold(fault.seconds, rec)
            return
        if fault is not None and fault.mode == "drop" and fault.style == "reset":
            try:   # SO_LINGER 0 -> close() sends a RST instead of a FIN
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                self.connection.close()
            except OSError:
                pass
            return
        if fault is not None and fault.mode == "drop" and fault.style == "cut":
            return   # connection closed without the chunked terminator: the client sees IncompleteRead
        self.wfile.write(b"0\r\n\r\n")   # a well-formed end of the HTTP body (with or without the SSE terminal marker)
        self.wfile.flush()
        rec.finished = fault is None or fault.mode == "slow"

    def _sleep(self, seconds: float) -> None:
        if seconds and seconds > 0:
            self.server_owner._closing.wait(seconds)

    def _hold(self, seconds: float, rec: RecordedRequest) -> None:
        """Keep the connection open and silent until the client leaves, the server closes or ``seconds`` pass."""
        sock = self.connection
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self.server_owner._closing.is_set():
            try:
                ready, _, _ = select.select([sock], [], [], 0.05)
                if ready and not sock.recv(1, socket.MSG_PEEK):
                    rec._mark_closed()
                    return
            except (OSError, ValueError):
                rec._mark_closed()
                return

    def _send(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
        self.send_response(status, _REASON.get(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
