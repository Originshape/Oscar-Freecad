"""Regression tests for the llm-providers review fixes.

Covered: SSE decoding, Stop while blocked on the socket, a mid-stream drop replays the reply, in-stream error
chunks, the retry policy, model-list parsing and base URLs, chat-completions wire order and empty assistant turns,
the vision downgrade surviving across turns, waking a blocked TLS read without SIGPIPE, and learning the vision
downgrade only from model-capability errors.

No real network: the SSE decoding tests talk to a throw-away loopback server,
everything else uses scripted stand-ins for ``requests``.

The tests pin behaviour through the provider layer's own entry points (reader thread + tick / reset events,
``provider_transport``, ``provider_chat``).
"""

import contextlib
import json
import os
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (suite guard)

import requests  # noqa: E402

from oscar import models as _models, params, provider_chat, provider_transport, providers  # noqa: E402
from oscar.providers import Message, StopRequested, ToolCall, ToolResult  # noqa: E402


def _profile(provider="openai_compat", model="fix-model", base_url="https://example.test/v1"):
    return params.Profile(name="fix", provider=provider, base_url=base_url, model=model,
                          api_key="k", temperature=0.2, max_tokens=256)


class _Resp:
    """requests.Response stand-in: scripted lines, optional mid-stream error."""

    def __init__(self, lines=(), status=200, text="", headers=None, error=None):
        self._lines = list(lines)
        self.status_code = status
        self.text = text
        self.headers = headers or {}
        self._error = error
        self.closed = False

    def iter_lines(self, decode_unicode=True):
        yield from self._lines
        if self._error is not None:
            raise self._error

    def close(self):
        self.closed = True


class _FakePost:
    """requests.post stand-in answering with scripted responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.bodies = []
        self.headers = []

    def __call__(self, url, headers=None, json=None, stream=True, timeout=None):
        self.calls += 1
        self.bodies.append(json)
        self.headers.append(dict(headers or {}))
        item = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item


def _oa(*chunks):
    """chat-completions SSE data lines for the given chunk dicts."""
    return ["data: " + json.dumps(c) for c in chunks] + ["data: [DONE]"]


def _text_chunk(text):
    return {"choices": [{"delta": {"content": text}}]}


def _stop_chunk():
    return {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def _by_kind(events):
    out: dict = {}
    for kind, payload in events:
        out.setdefault(kind, []).append(payload)
    return out


def _user(text="hi"):
    return Message(role="user", content=text)


class _Base(unittest.TestCase):
    def setUp(self):
        # _LEARNED is process-global and keyed by (class, base_url, model);
        # tests share base URLs, so never inherit another test's adaptations.
        # What a provider learns about a model (no vision, ...) goes through
        # oscar.models (an in-process mirror plus the settings file): both are
        # private to each test too.
        providers._LEARNED.clear()
        self.addCleanup(providers._LEARNED.clear)
        tmp = tempfile.mkdtemp(prefix="oscar_fix_providers_cfg_")
        self.addCleanup(shutil.rmtree, tmp, True)
        env = mock.patch.dict(os.environ, {
            "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
            "OSCAR_AI_SETTINGS": os.path.join(tmp, "settings.json")})
        env.start()
        self.addCleanup(env.stop)
        migrated = params._MIGRATED
        params._MIGRATED = True  # skip the one-time import of old user parameters
        self.addCleanup(setattr, params, "_MIGRATED", migrated)
        for mirror in (_models._LEARNED, _models._RUNTIME, _models._PROBE_FAILED):
            mirror.clear()
            self.addCleanup(mirror.clear)


class _IsolatedSettings(_Base):
    """A private settings file, whatever other modules did to the env var (every test case has one now)."""


# ---------------------------------------------------------------------------
# providers-01: SSE bodies are UTF-8 whatever the Content-Type says

_TEXT = ("Hole Ø10 mm × 5 ± 0.1 — “quoted” "
         "Å 孔 Привет")


class _SSEHandler(BaseHTTPRequestHandler):
    body = b""
    content_type = "text/event-stream"

    def do_POST(self):  # noqa: N802 - http.server API
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        if self.content_type:
            self.send_header("Content-Type", self.content_type)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        pass


class SseDecodingTest(_Base):
    def _chat(self, body, content_type):
        handler = type("H", (_SSEHandler,), {"body": body, "content_type": content_type})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        provider = providers.OpenAICompatProvider(
            _profile(base_url=f"http://127.0.0.1:{server.server_address[1]}/v1"))
        return list(provider.chat("s", [_user()], []))

    @staticmethod
    def _sse_body(*texts):
        out = ""
        for text in texts:
            out += "data: " + json.dumps(_text_chunk(text), ensure_ascii=False) + "\n\n"
        out += "data: " + json.dumps(_stop_chunk()) + "\n\ndata: [DONE]\n\n"
        return out.encode("utf-8")

    def test_utf8_is_not_decoded_as_latin1(self):
        # Local servers (Ollama, LM Studio, llama.cpp) send a bare
        # "text/event-stream"; requests then decodes as ISO-8859-1, which
        # garbles every non-ASCII character and splits a line at U+0085.
        long_tail = "孔" * 400  # multi-byte characters across read chunks
        body = self._sse_body(_TEXT[:12], _TEXT[12:], long_tail)
        out = _by_kind(self._chat(body, "text/event-stream"))
        self.assertNotIn("error", out)
        self.assertEqual("".join(out["text"]), _TEXT + long_tail)

    def test_missing_content_type_does_not_raise(self):
        out = _by_kind(self._chat(self._sse_body("plain ascii"), None))
        self.assertNotIn("error", out)
        self.assertEqual("".join(out["text"]), "plain ascii")


def _sse(*events):
    return ["data: " + json.dumps(e) for e in events]


# ---------------------------------------------------------------------------
# agent-loop-01 / providers-11: chat-completions wire order


class OpenAIWireOrderTest(_Base):
    def setUp(self):
        super().setUp()
        self.provider = providers.OpenAICompatProvider(_profile())

    def assert_tool_messages_adjacent(self, wire):
        for n, message in enumerate(wire):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                ids = [c["id"] for c in message["tool_calls"]]
                following = wire[n + 1:n + 1 + len(ids)]
                self.assertEqual([m["role"] for m in following], ["tool"] * len(ids), wire)
                self.assertEqual([m["tool_call_id"] for m in following], ids)

    def _round(self, n, with_image=(True, False)):
        ids = [f"c{n}a", f"c{n}b"]
        return [
            Message(role="assistant", tool_calls=[ToolCall(i, "get_view") for i in ids]),
            Message(role="user", tool_results=[
                ToolResult(i, "get_view", "ok", image_b64="QUJD" if shot else "")
                for i, shot in zip(ids, with_image)]),
        ]

    def test_screenshot_turn_never_splits_parallel_results(self):
        wire = self.provider._body("s", [_user("go")] + self._round(1), [])["messages"]
        self.assertEqual([m["role"] for m in wire],
                         ["system", "user", "assistant", "tool", "tool", "user"])
        self.assert_tool_messages_adjacent(wire)
        self.assertIn("data:image/png;base64,QUJD",
                      wire[-1]["content"][1]["image_url"]["url"])

    def test_followup_note_comes_after_the_screenshots(self):
        history = self._round(1)
        history[-1].content = "note"
        wire = self.provider._body("s", history, [])["messages"]
        self.assertEqual([m["role"] for m in wire],
                         ["system", "assistant", "tool", "tool", "user", "user"])
        self.assertEqual(wire[-1]["content"], "note")

    def test_old_screenshot_rewriting_keeps_results_adjacent(self):
        history = [_user("go")]
        for n in range(providers.MAX_WIRE_IMAGES + 3):
            history += self._round(n, with_image=(n % 2 == 0, True))
        wire = self.provider._body("s", history, [])["messages"]
        self.assert_tool_messages_adjacent(wire)
        images = sum(1 for m in wire if isinstance(m.get("content"), list)
                     for p in m["content"] if p.get("type") == "image_url")
        self.assertEqual(images, providers.MAX_WIRE_IMAGES)

    def test_empty_assistant_turn_has_non_null_content(self):
        for text in ("", "   ", None):
            wire = self.provider._to_wire(Message(role="assistant", content=text or ""))
            self.assertIsInstance(wire[0]["content"], str)
            self.assertTrue(wire[0]["content"].strip())
        calls_only = self.provider._to_wire(
            Message(role="assistant", tool_calls=[ToolCall("c", "get_view")]))
        self.assertIsNone(calls_only[0]["content"])  # allowed alongside tool_calls


# ---------------------------------------------------------------------------
# providers-07: error chunks that also carry choices


class InStreamErrorTest(_Base):
    def setUp(self):
        super().setUp()
        self.provider = providers.OpenAICompatProvider(_profile())

    def _parse(self, *chunks):
        return _by_kind(self.provider._parse_stream(_Resp(_oa(*chunks)), lambda: False))

    def test_openrouter_midstream_error_is_an_error(self):
        out = self._parse(
            _text_chunk("partial"),
            {"error": {"code": 502, "message": "Provider disconnected"},
             "choices": [{"delta": {"content": ""}, "finish_reason": "error"}]})
        self.assertIn("Provider disconnected", str(out["error"][0]))
        self.assertNotIn("done", out)

    def test_finish_reason_error_alone_is_an_error(self):
        out = self._parse(_text_chunk("partial"),
                          {"choices": [{"delta": {}, "finish_reason": "error"}]})
        self.assertIn("error", out)
        self.assertNotIn("done", out)

    def test_null_error_field_is_ignored(self):
        out = self._parse({"choices": [{"delta": {"content": "x"}}], "error": None},
                          _stop_chunk())
        self.assertNotIn("error", out)
        self.assertEqual(out["done"], ["stop"])


# ---------------------------------------------------------------------------
# providers-05 / providers-09 / providers-12: BaseProvider.chat


class _ChatBase(_Base):
    """chat() with the backoff sleeps recorded instead of slept.

    The retry sleep of this layer is ChatLoop._sleep (a generator that keeps
    ticking) and its delay comes from provider_chat.retry_delay (Retry-After,
    else jittered exponential backoff); the jitter is pinned to 0 here so the
    recorded delays are exact.
    """

    def setUp(self):
        super().setUp()
        self.delays = []
        self.provider = providers.OpenAICompatProvider(_profile())
        self.record_sleeps(self.provider)
        patcher = mock.patch.object(
            provider_chat, "retry_delay",
            lambda attempt, headers=None, **_: provider_transport.retry_delay(
                attempt, headers, jitter=lambda low, high: 0.0))
        patcher.start()
        self.addCleanup(patcher.stop)

    def record_sleeps(self, provider):
        def record(delay, call, should_stop, retry):
            self.delays.append(delay)
            return iter(())

        patcher = mock.patch.object(provider, "_sleep", side_effect=record)
        patcher.start()
        self.addCleanup(patcher.stop)

    def chat_events(self, *responses, messages=None, provider=None):
        post = _FakePost(responses)
        with mock.patch.object(providers.requests, "post", post):
            events = list((provider or self.provider).chat(
                "s", messages or [_user()], []))
        return post, events

    def chat(self, *responses, messages=None, provider=None):
        post, events = self.chat_events(*responses, messages=messages, provider=provider)
        return post, _by_kind(events)


class MidStreamDropTest(_ChatBase):
    def test_a_drop_after_output_is_not_duplicated(self):
        # The other implementation refused to retry once text was shown. This
        # layer retries and tells the consumer that what the failed attempt
        # streamed is void ("reset"; the agent and the panel drop it), so what
        # the consumer ends up holding is the second attempt's reply alone.
        part = _Resp(_oa(_text_chunk("I will build the housing. "))[:-1],
                     error=requests.exceptions.ChunkedEncodingError("connection reset"))
        full = _Resp(_oa(_text_chunk("I will build the housing. Done."), _stop_chunk()))
        post, events = self.chat_events(part, full)
        self.assertEqual(post.calls, 2)
        held = ""
        for kind, payload in events:
            if kind == "reset":
                held = ""
            elif kind == "text":
                held += payload
        self.assertEqual(held, "I will build the housing. Done.")
        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds.count("reset"), 1)
        self.assertLess(kinds.index("reset"), len(kinds) - 1 - kinds[::-1].index("text"))  # before the replay
        self.assertNotIn("error", kinds)

    def test_a_drop_before_any_output_still_retries(self):
        dropped = _Resp(error=requests.exceptions.ConnectionError("reset"))
        full = _Resp(_oa(_text_chunk("Hi"), _stop_chunk()))
        post, out = self.chat(dropped, full)
        self.assertEqual(post.calls, 2)
        self.assertEqual(out["text"], ["Hi"])
        self.assertNotIn("error", out)


class RetryPolicyTest(_ChatBase):
    def test_retry_after_is_honoured(self):
        limited = _Resp(status=429, text="slow down", headers={"Retry-After": "7"})
        post, out = self.chat(limited, _Resp(_oa(_text_chunk("Hi"), _stop_chunk())))
        self.assertEqual(post.calls, 2)
        self.assertEqual(self.delays, [7.0])
        self.assertNotIn("error", out)

    def test_no_sleep_after_the_last_attempt_and_the_body_is_reported(self):
        self.provider.max_retries = 3  # this layer allows 5 attempts by default
        busy = _Resp(status=503, text='{"error":"upstream busy"}')
        post, out = self.chat(busy)
        self.assertEqual(post.calls, 3)
        self.assertEqual(self.delays, [1.0, 2.0])
        message = str(out["error"][0])
        self.assertIn("upstream busy", message)
        self.assertIn("503", message)

    def test_exhausted_quota_is_not_retried(self):
        quota = _Resp(status=429, headers={"Retry-After": "30"},
                      text='{"error":{"code":"insufficient_quota","message":"You exceeded your current quota"}}')
        post, out = self.chat(quota)
        self.assertEqual(post.calls, 1)
        self.assertEqual(self.delays, [])
        self.assertIn("insufficient_quota", str(out["error"][0]))

    def test_a_malformed_base_url_fails_fast(self):
        provider = providers.OpenAICompatProvider(_profile(base_url="localhost:11434/v1"))
        self.record_sleeps(provider)
        out = _by_kind(provider.chat("s", [_user()], []))  # real requests.post, no network
        self.assertEqual(self.delays, [])
        self.assertIn("error", out)


class VisionLearnedTest(_ChatBase):
    def _rejecting(self):
        return _Resp(status=400, text='{"error":{"message":"messages.content.type is invalid"}}')

    def _ok(self):
        return _Resp(_oa(_text_chunk("Hi"), _stop_chunk()))

    @staticmethod
    def _anthropic_ok():
        return _Resp(_sse(
            {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
            {"type": "message_stop"}))

    def test_a_later_turn_starts_text_only(self):
        messages = [Message(role="user", content="look", images=["QUJD"])]
        post, _out = self.chat(self._rejecting(), self._ok(), messages=messages)
        self.assertEqual(post.calls, 2)
        second = providers.OpenAICompatProvider(_profile())  # next turn's worker
        self.assertFalse(second.vision_enabled)
        post, out = self.chat(self._ok(), messages=messages, provider=second)
        self.assertEqual(post.calls, 1)
        self.assertNotIn("image_url", json.dumps(post.bodies[0]))
        self.assertEqual(out["text"], ["Hi"])

    def test_not_learned_when_the_text_only_retry_fails_too(self):
        messages = [Message(role="user", content="look", images=["QUJD"])]
        failing = _Resp(status=400, text="bad request: something else entirely")
        _post, out = self.chat(self._rejecting(), failing, messages=messages)
        self.assertIn("error", out)
        self.assertTrue(providers.OpenAICompatProvider(_profile()).vision_enabled)

    def test_a_size_error_is_never_learned_as_no_vision(self):
        # llm-03: a 400 about one oversized image says nothing about the
        # model; the retry only worked because the image was gone. The other
        # implementation surfaced the error; this layer degrades THIS
        # provider (the turn goes on without the image, test_ollama_provider /
        # test_providers pin it) but never remembers it: the next turn tries
        # images again.
        envelope = ('{"type":"error","error":{"type":"invalid_request_error","message":'
                    '"messages.0.content.1.image.source.base64: image exceeds 5 MB maximum: '
                    '6291456 bytes > 5242880 bytes"}}')
        for provider_name, cls, ok in (("openai_compat", providers.OpenAICompatProvider, self._ok),
                                       ("anthropic", providers.AnthropicProvider, self._anthropic_ok)):
            with self.subTest(provider=provider_name):
                providers._LEARNED.clear()
                provider = cls(_profile(provider=provider_name))
                messages = [Message(role="user", content="look", images=["QUJD"])]
                post, out = self.chat(_Resp(status=400, text=envelope), ok(),
                                      messages=messages, provider=provider)
                self.assertEqual(post.calls, 2)
                self.assertNotIn("error", out)
                self.assertEqual(out["text"], ["Hi"])
                self.assertNotIn("QUJD", json.dumps(post.bodies[1]))  # the retry went without the image
                self.assertFalse(provider.vision_enabled)              # this provider degraded ...
                self.assertTrue(cls(_profile(provider=provider_name)).vision_enabled)  # ... the next turn does not


class ImageRejectionHeuristicTest(unittest.TestCase):
    """llm-03: what may be REMEMBERED as "this model cannot take images" must
    recognise a model that cannot, not any 400 that happens to mention one.

    That is ``models.unambiguous_image_rejection`` here (it gates
    ``models.learn(vision=False)``). ``providers._looks_like_image_rejection``
    is the deliberately broader test that only degrades the current provider
    (test_providers / test_ollama_provider pin it): a size complaint matches it
    but is never persisted, so the capability phrasings below must match both
    and the data complaints only the broad one."""

    @staticmethod
    def _envelope(message, kind="invalid_request_error"):
        return json.dumps({"type": "error", "error": {"type": kind, "message": message}})

    @staticmethod
    def _model_cannot(text):
        return _models.unambiguous_image_rejection(text)

    def test_model_capability_phrasings_match(self):
        for message in (
            "This model does not support image input.",
            "Model does not support images. Please use a model that does.",
            "image content is not supported by this model",
            "vision is not supported for this model",
            "This is not a multimodal model",
            "messages.content.type is invalid, allowed values: ['text']",
            "Invalid content type. image_url is only supported by certain models.",
        ):
            with self.subTest(message=message):
                self.assertTrue(self._model_cannot(message))
                self.assertTrue(self._model_cannot(self._envelope(message)))
                self.assertTrue(providers._looks_like_image_rejection(message))
                self.assertTrue(providers._looks_like_image_rejection(self._envelope(message)))

    def test_size_count_and_bad_data_errors_do_not_match(self):
        for message in (
            "messages.0.content.1.image.source.base64: image exceeds 5 MB maximum: "
            "6291456 bytes > 5242880 bytes",
            "Invalid image data",
            "image dimensions exceed the allowed limit of 8000 pixels",
            "Too many images in one request: 25 (maximum 20)",
            "image too large",
            "Could not process image",
            "You uploaded an unsupported image. Please make sure your image is below 20 MB "
            "in size and is of one the following formats: 'png', 'jpeg', 'gif', or 'webp'.",
        ):
            with self.subTest(message=message):
                self.assertFalse(self._model_cannot(message))
                self.assertFalse(self._model_cannot(self._envelope(message)))

    def test_the_generic_error_type_alone_is_not_evidence(self):
        self.assertFalse(self._model_cannot(
            self._envelope("image.source.media_type: unexpected value")))
        self.assertFalse(self._model_cannot(""))
        self.assertFalse(self._model_cannot(None))
        self.assertFalse(providers._looks_like_image_rejection(""))
        self.assertFalse(providers._looks_like_image_rejection(None))


# ---------------------------------------------------------------------------
# providers-04: Stop must not wait for the next byte


class _BlockedResp(_Resp):
    """A server that went quiet: iter_lines blocks until the socket is shut
    down. Like http.client, close() waits for the blocked reader (it needs the
    lock the reader holds), so only a socket shutdown ends the wait."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        sock = types.SimpleNamespace(shutdown=lambda how: self.release.set())
        self.raw = types.SimpleNamespace(_connection=types.SimpleNamespace(sock=sock))

    def iter_lines(self, decode_unicode=True):
        self.release.wait(3)
        yield from ()

    def close(self):
        self.release.wait(3)
        super().close()


class StopResponsivenessTest(_Base):
    """In this layer a stream is parsed on the attempt's reader thread and
    chat() polls Stop every 200 ms, then shuts the socket down to free the
    blocked reader; so Stop is exercised through chat() (the parsers are not
    polled for Stop themselves)."""

    def _stop_after(self, seconds):
        t0 = time.monotonic()
        return t0, lambda: time.monotonic() - t0 > seconds


    def test_keep_alive_comments_do_not_defer_stop(self):
        class Comments(_Resp):
            def iter_lines(self, decode_unicode=True):
                for _ in range(300):
                    time.sleep(0.01)
                    yield ": PROCESSING"

        t0, stop = self._stop_after(0.2)
        with mock.patch.object(providers.requests, "post", _FakePost([Comments()])):
            with self.assertRaises(StopRequested):
                list(providers.OpenAICompatProvider(_profile()).chat("s", [_user()], [], stop))
        self.assertLess(time.monotonic() - t0, 2.5)   # (was 1.5: 1.6 s measured on a loaded CI runner; a stop that is not prompt waits >= 3 s)

    def test_stop_interrupts_a_request_the_server_never_answers(self):
        release = threading.Event()
        late = _Resp(_oa(_stop_chunk()))
        self.addCleanup(release.set)

        def silent_post(url, headers=None, json=None, stream=True, timeout=None):
            release.wait(3)
            return late

        provider = providers.OpenAICompatProvider(_profile())
        t0, stop = self._stop_after(0.3)
        with mock.patch.object(providers.requests, "post", silent_post):
            with self.assertRaises(StopRequested):
                list(provider.chat("s", [_user()], [], stop))
            self.assertLess(time.monotonic() - t0, 2.5)   # (was 1.5: 1.6 s measured on a loaded CI runner; a stop that is not prompt waits >= 3 s)
            release.set()
            deadline = time.monotonic() + 1.5
            while not late.closed and time.monotonic() < deadline:
                time.sleep(0.02)
        self.assertTrue(late.closed, "the abandoned response must not pin a connection")


class _QuietHandler(BaseHTTPRequestHandler):
    """An SSE server that goes quiet. mode "silent": headers and one chunk,
    then nothing (it records when the client hangs up); "noheaders": accepts
    the request and never answers."""

    protocol_version = "HTTP/1.1"
    mode = "silent"
    release = None
    hung_up = None

    def do_POST(self):  # noqa: N802 - http.server API
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.mode == "silent":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            data = ("data: " + json.dumps(_text_chunk("Hi")) + "\n\n").encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()
            self.connection.settimeout(3)
            try:
                if self.connection.recv(1) == b"":  # EOF: the client closed its end
                    self.hung_up.set()
            except OSError:
                pass
        else:
            self.release.wait(3)
        self.close_connection = True

    def log_message(self, *args):
        pass


class StopOnRealSocketsTest(_Base):
    def test_stop_is_prompt_when_the_server_goes_quiet(self):
        for mode in ("silent", "noheaders"):
            with self.subTest(mode=mode):
                release, hung_up = threading.Event(), threading.Event()
                self.addCleanup(release.set)
                handler = type("H", (_QuietHandler,), {
                    "mode": mode, "release": release, "hung_up": hung_up})
                server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
                threading.Thread(target=server.serve_forever, daemon=True).start()
                self.addCleanup(server.server_close)
                self.addCleanup(server.shutdown)
                provider = providers.OpenAICompatProvider(
                    _profile(base_url=f"http://127.0.0.1:{server.server_address[1]}/v1"))
                t0 = time.monotonic()
                events = []
                with self.assertRaises(StopRequested):
                    for event in provider.chat("s", [_user()], [],
                                               lambda: time.monotonic() - t0 > 0.4):
                        events.append(event)
                self.assertLess(time.monotonic() - t0, 2.5)   # (was 1.5: 1.6 s measured on a loaded CI runner; a stop that is not prompt waits >= 3 s)
                self.assertEqual(events, [("text", "Hi")] if mode == "silent" else [])
                if mode == "silent":
                    # the connection is released too, not left pinned by a
                    # reader still waiting for the next byte
                    self.assertTrue(hung_up.wait(1.5), "client never closed the connection")


# ---------------------------------------------------------------------------
# llm-01: waking a blocked TLS read must not raise SIGPIPE


class InterruptReadTest(unittest.TestCase):
    """The socket shutdown that wakes a blocked read is
    provider_transport.shutdown_socket (AttemptRunner.cancel calls it)."""

    def test_a_posix_socket_is_shut_down_for_reading_only(self):
        # SHUT_RDWR makes OpenSSL's alert write hit EPIPE, and SIGPIPE kills a
        # process that never ignored it (FreeCAD's embedded Python). Windows
        # needs the full shutdown for the blocked recv to return.
        calls = []
        sock = types.SimpleNamespace(shutdown=calls.append)
        response = types.SimpleNamespace(
            raw=types.SimpleNamespace(_connection=types.SimpleNamespace(sock=sock)))
        self.assertTrue(provider_transport.shutdown_socket(response))
        self.assertEqual(calls, [socket.SHUT_RDWR if os.name == "nt" else socket.SHUT_RD])

    def test_a_tls_socket_is_never_shut_down_for_writing_on_posix(self):
        calls = []

        class Tls(ssl.SSLSocket):  # never connected: only the type matters
            def __init__(self):
                pass

            def shutdown(self, how):
                calls.append(how)

        response = types.SimpleNamespace(
            raw=types.SimpleNamespace(_connection=types.SimpleNamespace(sock=Tls())))
        provider_transport.shutdown_socket(response)
        self.assertEqual(calls, [socket.SHUT_RDWR if os.name == "nt" else socket.SHUT_RD])

    def test_a_response_without_a_socket_is_ignored(self):
        self.assertFalse(provider_transport.shutdown_socket(types.SimpleNamespace()))
        self.assertFalse(provider_transport.shutdown_socket(_Resp()))


class _TlsChunkHandler(BaseHTTPRequestHandler):
    """An HTTPS SSE endpoint that stays quiet after its one chunk until the
    client hangs up (it never writes again: with SIGPIPE at SIG_DFL, a write to
    a reset connection would kill the test process from the server side).
    mode "silent": the chunk holds one text delta; "unterminated": it holds the
    whole reply, [DONE] included, but the chunked terminator never follows, so
    the client's reader is still inside SSL_read when its consumer is done."""

    protocol_version = "HTTP/1.1"
    mode = "silent"

    def do_POST(self):  # noqa: N802 - http.server API
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        lines = _oa(_text_chunk("Hi"), _stop_chunk()) if self.mode == "unterminated" else \
            ["data: " + json.dumps(_text_chunk("Hi"))]
        data = ("\n\n".join(lines) + "\n\n").encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()
        self.connection.settimeout(3)
        try:
            self.connection.recv(1)  # returns once the client closed its end
        except OSError:
            pass
        self.close_connection = True

    def log_message(self, *args):
        pass


def _throwaway_certificate(directory):
    """(cert, key) of a self-signed localhost certificate, minted with the
    openssl CLI; the stdlib cannot make one. Skips when that is unavailable."""
    exe = shutil.which("openssl")
    if exe is None:
        raise unittest.SkipTest("no openssl CLI to mint a throwaway certificate")
    cert, key = os.path.join(directory, "cert.pem"), os.path.join(directory, "key.pem")
    try:
        subprocess.run(
            [exe, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key,
             "-out", cert, "-days", "2", "-subj", "/CN=localhost",
             "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1"],
            check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise unittest.SkipTest(f"openssl could not mint a certificate: {e}")
    return cert, key


@contextlib.contextmanager
def _sigpipe_watch():
    """Count SIGPIPEs instead of dying from them (SIG_DFL, as in freecadcmd, is
    fatal, so a regression would otherwise take the whole test run down)."""
    hits = []
    if not hasattr(signal, "SIGPIPE") or threading.current_thread() is not threading.main_thread():
        yield hits
        return
    previous = signal.signal(signal.SIGPIPE, lambda *_: hits.append(1))
    try:
        yield hits
    finally:
        signal.signal(signal.SIGPIPE, previous if previous is not None else signal.SIG_DFL)


class TlsInterruptTest(_Base):
    """Stop, and a stream whose consumer finished before the chunk terminator
    arrived, both wake the reader thread through shutdown_socket."""

    def _serve(self, mode):
        folder = tempfile.mkdtemp(prefix="oscar_tls_")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        cert, key = _throwaway_certificate(folder)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), type("H", (_TlsChunkHandler,), {"mode": mode}))
        server.socket = context.wrap_socket(server.socket, server_side=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        env = mock.patch.dict(os.environ, {"REQUESTS_CA_BUNDLE": cert})
        env.start()
        self.addCleanup(env.stop)
        return providers.OpenAICompatProvider(
            _profile(base_url=f"https://127.0.0.1:{server.server_address[1]}/v1"))

    def test_stop_on_a_tls_stream_does_not_raise_sigpipe(self):
        provider = self._serve("silent")
        t0 = time.monotonic()
        events = []
        with _sigpipe_watch() as hits:
            with self.assertRaises(StopRequested):
                for event in provider.chat("s", [_user()], [],
                                           lambda: time.monotonic() - t0 > 0.4):
                    events.append(event)
            time.sleep(0.3)  # let the reader thread finish unwinding
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(events, [("text", "Hi")])
        self.assertEqual(hits, [], "the interrupted TLS read raised SIGPIPE")

    def test_finishing_before_the_chunk_terminator_does_not_raise_sigpipe(self):
        provider = self._serve("unterminated")
        with _sigpipe_watch() as hits:
            out = _by_kind(provider.chat("s", [_user()], [], lambda: False))
            time.sleep(0.5)
        self.assertEqual(out["text"], ["Hi"])
        self.assertNotIn("error", out)
        self.assertEqual(hits, [], "closing the finished TLS stream raised SIGPIPE")


# ---------------------------------------------------------------------------
# providers-10 / providers-13: model listing and base URLs


class _Json:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class ListModelsTest(_Base):
    def _list(self, profile, payload):
        urls = []

        def fake_get(url, headers=None, timeout=None):
            urls.append(url)
            return _Json(payload)

        with mock.patch.object(providers.requests, "get", fake_get):
            result = providers.list_models(profile)
        return result, urls

    def test_odd_payloads_never_raise(self):
        profile = _profile()
        self.assertEqual(self._list(profile, {"data": [{"id": "b"}, {"id": "a"}]})[0],
                         (["a", "b"], ""))
        self.assertEqual(self._list(profile, [{"id": "m1"}, {"name": "m2"}, "m3"])[0],
                         (["m1", "m2", "m3"], ""))
        self.assertEqual(self._list(profile, {"models": [{"name": "m1"}]})[0], (["m1"], ""))
        for payload in (None, "oops", 7, {"data": None}, {"data": "x"}):
            models, error = self._list(profile, payload)[0]
            self.assertEqual(models, [], payload)
            self.assertTrue(error, payload)

    def test_chat_completions_url_is_reduced_to_the_api_base(self):
        profile = _profile(base_url="https://api.openai.com/v1/chat/completions")
        _result, urls = self._list(profile, {"data": [{"id": "m"}]})
        self.assertEqual(urls, ["https://api.openai.com/v1/models"])

    def test_anthropic_base_with_and_without_v1(self):
        for base in ("https://gw.example.com", "https://gw.example.com/",
                     "https://gw.example.com/v1", "https://gw.example.com/v1/",
                     "https://gw.example.com/v1/messages"):
            profile = _profile("anthropic", "claude-x", base)
            self.assertEqual(providers.AnthropicProvider(profile)._endpoint(),
                             "https://gw.example.com/v1/messages", base)
            _result, urls = self._list(profile, {"data": [{"id": "m"}]})
            self.assertEqual(urls, ["https://gw.example.com/v1/models"], base)


if __name__ == "__main__":
    unittest.main()
