"""Provider fault matrix: retries, backoff, Retry-After, early EOF, Stop,
ticks, the reader thread and its cancellation against a real local HTTP server.

``requests.post`` is patched with ``fake_post(url, headers=None, json=None, stream=True, timeout=None)`` for the
matrix; the cancellation and encoding tests talk to a ``ThreadingHTTPServer`` on 127.0.0.1 with the real ``requests``.
"""

import datetime
import email.utils
import http.client
import http.server
import json
import os
import select
import socket
import sys
import threading
import time
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import requests  # noqa: E402

from oscar import models, params, provider_chat, provider_transport, providers  # noqa: E402
from test_providers import anthropic_reply, forget_learned, openai_reply, sse  # noqa: E402

_REAL_SLEEP = provider_chat.ChatLoop.__dict__["_sleep"]    # FaultCase replaces ChatLoop._sleep; some tests want the real one

Message = providers.Message
USER = [Message(role="user", content="hi")]
TOOL = providers.ToolSpec("create_object", "Create an object.", {"type": "object", "properties": {}})

OK_OPENAI = openai_reply(text="Hello")
OK_ANTHROPIC = anthropic_reply(blocks=[("text", "Hello")])


class Resp:
    """A ``requests.Response`` look-alike: status, text, headers, ``iter_lines``; ``fail`` is raised after the lines."""

    def __init__(self, status=200, lines=(), text="", headers=None, fail=None, block=None):
        self.status_code = status
        self.text = text
        self.headers = headers or {}
        self._lines = list(lines)
        self._fail = fail
        self._block = block          # threading.Event the stream waits on after its lines (released by close())
        self.closed = threading.Event()

    def iter_lines(self, decode_unicode=True):
        yield from self._lines
        if self._block is not None:
            self._block.wait(10)
            if self.closed.is_set():
                raise requests.exceptions.ConnectionError("closed")
        if self._fail is not None:
            raise self._fail

    def close(self):
        self.closed.set()
        if self._block is not None:
            self._block.set()


def ok(lines=OK_OPENAI):
    return Resp(200, lines)


def http_status(status, text="", headers=None):
    return Resp(status, text=text, headers=headers)


class Script:
    """Scripted ``requests.post``: each call consumes the next item (a Resp, an exception to raise, or a callable)."""

    def __init__(self, *items):
        self.items = list(items)
        self.calls = []

    def __call__(self, url, headers=None, json=None, stream=True, timeout=None):
        self.calls.append({"url": url, "headers": dict(headers or {}), "json": json, "stream": stream, "timeout": timeout})
        if not self.items:
            raise AssertionError("more requests than scripted")
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item()
        return item


def collect(events):
    """What a consumer ends with: text after the last ``reset``, the kinds seen and the terminal event."""
    state = {"text": "", "kinds": [], "resets": [], "error": None, "done": None, "calls": None, "ticks": []}
    for kind, payload in events:
        state["kinds"].append(kind)
        if kind == "text":
            state["text"] += payload
        elif kind == "reset":
            state["text"] = ""
            state["resets"].append(payload)
        elif kind == "error":
            state["error"] = payload
        elif kind == "done":
            state["done"] = payload
        elif kind == "tool_calls":
            state["calls"] = payload
        elif kind == "tick":
            state["ticks"].append(payload)
    return state


class FaultCase(unittest.TestCase):
    """Builds providers with no real sleeping; records the retry sleeps instead."""

    def setUp(self):
        self._profiles = []
        self.addCleanup(lambda: forget_learned(*self._profiles))
        self.sleeps = []

        def fake_sleep(loop, delay, call, should_stop, retry):
            self.sleeps.append((round(delay, 3), tuple(retry)))
            return iter(())

        patcher = mock.patch.object(provider_chat.ChatLoop, "_sleep", fake_sleep)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.logged = []
        patcher = mock.patch.object(provider_chat, "log_event", lambda name, **kw: self.logged.append((name, kw)))
        patcher.start()
        self.addCleanup(patcher.stop)

    def prof(self, provider, base_url, model, **extra):
        p = params.Profile(name=f"p-{uuid.uuid4().hex[:6]}", provider=provider, base_url=base_url, model=model,
                           api_key="key", max_tokens=1024, **extra)
        self._profiles.append(p)
        forget_learned(p)
        return p

    def openai(self, model="gpt-4o", base_url="https://llm.example.com/v1", **extra):
        return providers.OpenAICompatProvider(self.prof("openai_compat", base_url, model, **extra))

    def anthropic(self, model="claude-sonnet-4-5", **extra):
        return providers.AnthropicProvider(self.prof("anthropic", "https://api.anthropic.com", model, **extra))

    def run_chat(self, provider, script, *, should_stop=None, tools=(), **opts):
        with mock.patch.object(providers.requests, "post", script):
            return collect(provider.chat("sys", USER, list(tools), should_stop, **opts))


class PostCallShapeTest(FaultCase):
    def test_requests_post_is_called_with_exactly_the_pinned_keyword_shape(self):
        script = Script(ok())
        state = self.run_chat(self.openai(), script)
        self.assertEqual(state["text"], "Hello")
        call = script.calls[0]
        self.assertEqual(call["url"], "https://llm.example.com/v1/chat/completions")
        self.assertTrue(call["stream"])
        self.assertEqual(call["timeout"], (15, 600))
        self.assertEqual(call["headers"]["Authorization"], "Bearer key")
        self.assertEqual(call["json"]["model"], "gpt-4o")

    def test_event_order_and_internal_events_never_reach_the_caller(self):
        state = self.run_chat(self.openai(), Script(ok()))
        self.assertEqual(state["kinds"], ["text", "stop_reason", "stop_info", "done"])
        for internal in ("stream_end", "_status", "_http", "_alive", "_end", "_exc"):
            self.assertNotIn(internal, state["kinds"])
        self.assertEqual(state["done"], "stop")

    def test_per_call_options_reach_the_request_body(self):
        script = Script(ok())
        self.run_chat(self.openai(), script, tools=[TOOL], tool_choice="required", max_tokens=321,
                      parallel_tool_calls=False)
        body = script.calls[0]["json"]
        self.assertEqual((body["tool_choice"], body["max_tokens"]), ("required", 321))

    def test_a_tool_call_reply(self):
        lines = openai_reply(calls=[("c1", "create_object", '{"a": 1}')], finish="tool_calls")
        state = self.run_chat(self.openai(), Script(ok(lines)), tools=[TOOL])
        self.assertEqual([(c.name, c.arguments) for c in state["calls"]], [("create_object", {"a": 1})])
        self.assertIsNone(state["done"])


class RetryStatusMatrixTest(FaultCase):
    def test_retriable_statuses_are_retried_once_then_succeed(self):
        for status in sorted(provider_transport.RETRY_STATUS):
            with self.subTest(status=status):
                self.sleeps.clear()
                script = Script(http_status(status, "try later"), ok())
                state = self.run_chat(self.openai(), script)
                self.assertEqual(state["text"], "Hello")
                self.assertIsNone(state["error"])
                self.assertEqual(len(script.calls), 2)
                self.assertEqual(len(self.sleeps), 1)

    def test_the_retriable_set_is_the_expected_set(self):
        self.assertEqual(provider_transport.RETRY_STATUS,
                         {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529})

    def test_permanent_statuses_are_never_retried(self):
        for status in (400, 401, 402, 403, 404, 405, 413, 422):
            with self.subTest(status=status):
                script = Script(http_status(status, "nope"), ok())
                state = self.run_chat(self.openai(), script)
                self.assertEqual(len(script.calls), 1)
                self.assertIsNone(state["done"])
                self.assertTrue(str(state["error"]).startswith(f"HTTP {status} "), state["error"])

    def test_quota_and_billing_bodies_are_never_retried(self):
        for body in ('{"error": {"code": "insufficient_quota"}}', "Your credit balance is too low", "billing hard limit reached"):
            with self.subTest(body=body):
                script = Script(http_status(429, body), ok())
                state = self.run_chat(self.openai(), script)
                self.assertEqual(len(script.calls), 1)
                self.assertIn("quota or billing problem", str(state["error"]))
                self.assertFalse(state["error"].retriable)

    def test_five_attempts_then_the_http_error_with_the_first_200_chars_of_the_body(self):
        body = "overloaded " * 50
        script = Script(*[http_status(503, body) for _ in range(6)])
        state = self.run_chat(self.openai(), script)
        self.assertEqual(len(script.calls), 5)
        self.assertEqual(len(self.sleeps), 4)             # no sleep after the final attempt
        message = str(state["error"])
        self.assertTrue(message.startswith("HTTP 503 service unavailable: overloaded"), message)
        self.assertLessEqual(len(message), len("HTTP 503 service unavailable: ") + 200)
        self.assertEqual(state["kinds"].count("error"), 1)

    def test_retry_attempts_are_logged_and_numbered(self):
        self.run_chat(self.openai(), Script(http_status(500), http_status(502), ok()))
        retries = [kw for name, kw in self.logged if name == "provider_retry"]
        self.assertEqual([r["attempt"] for r in retries], [1, 2])
        self.assertEqual({r["status"] for r in retries}, {500, 502})

    def test_the_sleep_announcement_carries_the_next_attempt_and_the_max(self):
        self.run_chat(self.openai(), Script(http_status(500), http_status(500), ok()))
        self.assertEqual([r for _d, r in self.sleeps], [(2, 5), (3, 5)])


class BackoffAndRetryAfterTest(FaultCase):
    def delays(self, *responses):
        self.sleeps.clear()
        self.run_chat(self.openai(), Script(*responses))
        return [d for d, _r in self.sleeps]

    def test_exponential_backoff_with_jitter(self):
        delays = self.delays(*[http_status(503) for _ in range(5)])
        self.assertEqual(len(delays), 4)
        for attempt, delay in enumerate(delays):
            self.assertGreaterEqual(delay, 2 ** attempt)
            self.assertLess(delay, 2 ** attempt + 1.0 + 1e-9)

    def test_backoff_formula_and_cap(self):
        self.assertEqual(provider_transport.backoff_delay(0, jitter=lambda a, b: 0.0), 1.0)
        self.assertEqual(provider_transport.backoff_delay(3, jitter=lambda a, b: 0.5), 8.5)
        self.assertEqual(provider_transport.backoff_delay(10, jitter=lambda a, b: 1.0), 20.0)
        self.assertEqual(provider_transport.backoff_delay(-3, jitter=lambda a, b: 0.0), 1.0)

    def test_retry_after_seconds_is_honoured(self):
        self.assertEqual(self.delays(http_status(429, headers={"Retry-After": "7"}), ok()), [7.0])

    def test_retry_after_is_capped_at_sixty_seconds(self):
        self.assertEqual(self.delays(http_status(429, headers={"Retry-After": "500"}), ok()), [60.0])

    def test_retry_after_zero_means_retry_now(self):
        self.assertEqual(self.delays(http_status(503, headers={"Retry-After": "0"}), ok()), [0.0])

    def test_retry_after_header_name_is_case_insensitive(self):
        self.assertEqual(self.delays(http_status(429, headers={"retry-after": "3"}), ok()), [3.0])

    def test_retry_after_http_date(self):
        when = email.utils.format_datetime(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=20), usegmt=True)
        (delay,) = self.delays(http_status(503, headers={"Retry-After": when}), ok())
        self.assertTrue(17 <= delay <= 21, delay)

    def test_a_past_date_or_garbage_falls_back_to_backoff(self):
        past = email.utils.format_datetime(datetime.datetime(2001, 1, 1, tzinfo=datetime.timezone.utc), usegmt=True)
        (delay,) = self.delays(http_status(503, headers={"Retry-After": past}), ok())
        self.assertEqual(delay, 0.0)                                   # a past date means "now", not "unknown"
        (delay,) = self.delays(http_status(503, headers={"Retry-After": "soon-ish"}), ok())
        self.assertTrue(1.0 <= delay < 2.0, delay)

    def test_retry_after_parsing_unit(self):
        r = provider_transport.retry_after_seconds
        self.assertEqual(r({"Retry-After": "12.5"}), 12.5)
        self.assertEqual(r({"Retry-After": "-4"}), 0.0)
        self.assertIsNone(r({}))
        self.assertIsNone(r(None))
        self.assertIsNone(r({"Retry-After": ""}))
        self.assertIsNone(r({"Retry-After": "Mon, 99 Foo 2026 25:61:61 GMT"}))
        self.assertEqual(provider_transport.retry_delay(0, {"Retry-After": "9999"}), 60.0)

    def test_total_sleep_budget_is_ninety_seconds(self):
        script = Script(*[http_status(429, headers={"Retry-After": "60"}) for _ in range(5)])
        state = self.run_chat(self.openai(), script)
        self.assertEqual([d for d, _r in self.sleeps], [60.0, 30.0])    # 60 + 30 = the whole budget
        self.assertEqual(len(script.calls), 3)                          # then: no budget left, no fourth request
        self.assertTrue(str(state["error"]).startswith("HTTP 429"), state["error"])


class TransportErrorTest(FaultCase):
    def test_connection_level_errors_are_retried(self):
        for exc in (requests.exceptions.ConnectionError("reset"), requests.exceptions.ReadTimeout("slow"),
                    requests.exceptions.ConnectTimeout("no route"), requests.exceptions.ChunkedEncodingError("cut"),
                    OSError("network is unreachable"), http.client.RemoteDisconnected("closed"),
                    requests.exceptions.SSLError("eof")):
            with self.subTest(exc=type(exc).__name__):
                script = Script(exc, ok())
                state = self.run_chat(self.openai(), script)
                self.assertEqual(state["text"], "Hello")
                self.assertEqual(len(script.calls), 2)

    def test_exhausted_transport_retries_name_the_attempt_count(self):
        script = Script(*[requests.exceptions.ConnectionError("refused") for _ in range(5)])
        state = self.run_chat(self.openai(), script)
        self.assertEqual(len(script.calls), 5)
        self.assertEqual(str(state["error"]), "request failed after 5 attempts: refused")

    def test_a_malformed_url_is_not_retried(self):
        for exc in (requests.exceptions.InvalidURL("bad"), requests.exceptions.MissingSchema("no scheme"),
                    requests.exceptions.InvalidSchema("ftp"), requests.exceptions.InvalidHeader("bad header")):
            with self.subTest(exc=type(exc).__name__):
                script = Script(exc, ok())
                state = self.run_chat(self.openai(), script)
                self.assertEqual(len(script.calls), 1)
                self.assertIn(type(exc).__name__, str(state["error"]))

    def test_a_bug_in_the_provider_is_not_retried_and_is_named(self):
        script = Script(KeyError("oops"), ok())
        state = self.run_chat(self.openai(), script)
        self.assertEqual(len(script.calls), 1)
        self.assertEqual(str(state["error"]), "KeyError: 'oops'")

    def test_an_error_raised_in_the_reader_is_delivered_not_swallowed(self):
        state = self.run_chat(self.openai(), Script(lambda: (_ for _ in ()).throw(RuntimeError("inside"))))
        self.assertIn("inside", str(state["error"]))


class EarlyEofAndMidStreamTest(FaultCase):
    def test_a_stream_without_a_terminal_marker_is_retried_and_ends_in_the_contract_error(self):
        cut = openai_reply(text="partial", terminal=None)
        script = Script(*[ok(cut) for _ in range(5)])
        state = self.run_chat(self.openai(), script)
        self.assertEqual(len(script.calls), 5)
        self.assertEqual(str(state["error"]), "stream ended early (no terminal marker)")
        self.assertNotIn("done", state["kinds"])

    def test_reset_precedes_a_retry_after_partial_output(self):
        cut = openai_reply(text="partial", terminal=None)
        state = self.run_chat(self.openai(), Script(ok(cut), ok(openai_reply(text="whole"))))
        self.assertEqual(state["resets"], ["stream_dropped"])
        self.assertEqual(state["text"], "whole")                # the consumer discarded "partial"
        self.assertEqual(state["kinds"].index("reset"), state["kinds"].index("text") + 1)

    def test_no_reset_when_the_failed_attempt_produced_nothing(self):
        state = self.run_chat(self.openai(), Script(http_status(503), ok()))
        self.assertEqual(state["resets"], [])
        self.assertNotIn("reset", state["kinds"])

    def test_a_connection_dropped_mid_stream_resets_with_retry(self):
        broken = Resp(200, openai_reply(text="par", terminal=None), fail=requests.exceptions.ChunkedEncodingError("cut"))
        state = self.run_chat(self.openai(), Script(broken, ok(openai_reply(text="whole"))))
        self.assertEqual((state["resets"], state["text"]), (["retry"], "whole"))

    def test_a_long_generation_that_drops_is_replayed_at_most_once(self):
        long_cut = openai_reply(text="x" * 2500, terminal=None)
        script = Script(*[ok(long_cut) for _ in range(5)])
        state = self.run_chat(self.openai(), script)
        self.assertEqual(len(script.calls), 2)                  # one billed retry, not four
        self.assertEqual(str(state["error"]), "stream ended early (no terminal marker)")

    def test_a_short_drop_does_not_count_as_billed(self):
        short_cut = openai_reply(text="x" * 2000, terminal=None)            # exactly at the threshold: not "more than"
        script = Script(*[ok(short_cut) for _ in range(5)])
        self.run_chat(self.openai(), script)
        self.assertEqual(len(script.calls), 5)

    def test_a_transport_error_after_a_long_generation_is_also_billed_once(self):
        def long_then_cut():
            return Resp(200, openai_reply(text="x" * 2500, terminal=None), fail=requests.exceptions.ConnectionError("cut"))

        script = Script(long_then_cut, long_then_cut, long_then_cut)
        state = self.run_chat(self.openai(), script)
        self.assertEqual(len(script.calls), 2)
        self.assertIn("cut", str(state["error"]))

    def test_reasoning_and_tool_argument_chars_count_toward_the_billed_threshold(self):
        reasoning = ["data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "r" * 2500}}]})]
        script = Script(ok(reasoning), ok(reasoning), ok(reasoning))
        self.run_chat(self.openai(), script)
        self.assertEqual(len(script.calls), 2)

    def test_in_stream_overload_is_retried_with_a_reset(self):
        overloaded = sse([{"type": "message_start", "message": {}},
                          {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                          {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
                          {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}])
        script = Script(ok(overloaded), ok(OK_ANTHROPIC))
        state = self.run_chat(self.anthropic(), script)
        self.assertEqual((state["resets"], state["text"]), (["retry"], "Hello"))
        self.assertEqual(len(script.calls), 2)

    def test_in_stream_non_retriable_error_ends_the_call_once(self):
        bad = sse([{"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}])
        script = Script(ok(bad), ok(OK_ANTHROPIC))
        state = self.run_chat(self.anthropic(), script)
        self.assertEqual(len(script.calls), 1)
        self.assertEqual(state["kinds"].count("error"), 1)
        self.assertIn("invalid_request_error", str(state["error"]))

    def test_in_stream_quota_error_is_not_retried(self):
        quota = sse([{"error": {"message": "You exceeded your current quota", "type": "insufficient_quota"}}])
        script = Script(ok(quota), ok())
        self.run_chat(self.openai(), script)
        self.assertEqual(len(script.calls), 1)


class NotAnEventStreamTest(FaultCase):
    def test_an_html_answer_with_status_200_is_reported_not_retried(self):
        page = ["<!doctype html>", "<html><body>Please sign in to continue</body></html>"]
        script = Script(ok(page), ok())
        state = self.run_chat(self.openai(), script)
        self.assertEqual(len(script.calls), 1)
        self.assertEqual(state["kinds"].count("error"), 1)
        self.assertTrue(str(state["error"]).startswith("the server answered with something other than an event stream: <!doctype html>"),
                        state["error"])
        self.assertIn("Please sign in", str(state["error"]))

    def test_a_json_error_answered_with_200_names_the_error(self):
        body = ['{"error": {"message": "stream is not supported for this model",', ' "type": "invalid_request_error"}}']
        state = self.run_chat(self.anthropic(), Script(ok(body)))
        self.assertIn("stream is not supported for this model", str(state["error"]))

    def test_the_quoted_text_is_clipped(self):
        state = self.run_chat(self.openai(), Script(ok(["x" * 5000])))
        self.assertLessEqual(len(str(state["error"])), len("the server answered with something other than an event stream: ") + 200)

    def test_an_empty_body_is_still_an_early_end_and_is_retried(self):
        script = Script(*[ok([]) for _ in range(5)])
        state = self.run_chat(self.openai(), script)
        self.assertEqual(len(script.calls), 5)
        self.assertEqual(str(state["error"]), "stream ended early (no terminal marker)")

    def test_sse_fields_and_comments_alone_are_not_stray_text(self):
        script = Script(*[ok([": keepalive", "event: ping", "id: 1", "retry: 100"]) for _ in range(5)])
        state = self.run_chat(self.openai(), script)
        self.assertEqual(str(state["error"]), "stream ended early (no terminal marker)")

    def test_text_after_data_lines_is_ignored(self):
        lines = ["data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}), "garbage",
                 "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})]
        state = self.run_chat(self.openai(), Script(ok(lines)))
        self.assertEqual((state["text"], state["done"]), ("ok", "stop"))


class AdaptationTest(FaultCase):
    def test_a_400_adaptation_retries_without_counting_as_an_attempt(self):
        script = Script(
            http_status(400, "Unsupported parameter: 'temperature' is not supported with this model."),
            http_status(503), http_status(503), http_status(503), http_status(503), ok())
        state = self.run_chat(self.openai(), script)
        self.assertEqual(state["text"], "Hello")                         # 1 adaptation + 5 real attempts
        self.assertEqual(len(script.calls), 6)
        self.assertIn("temperature", script.calls[0]["json"])
        self.assertNotIn("temperature", script.calls[1]["json"])
        self.assertEqual(self.sleeps and len(self.sleeps), 4)            # the adaptation itself did not sleep

    def test_the_adaptation_is_learned_for_the_next_worker(self):
        provider = self.openai()
        self.run_chat(provider, Script(http_status(400, "temperature is not supported"), ok()))
        script = Script(ok())
        self.run_chat(providers.OpenAICompatProvider(provider.profile), script)
        self.assertNotIn("temperature", script.calls[0]["json"])

    def test_the_same_complaint_twice_ends_the_call(self):
        script = Script(*[http_status(400, "temperature is not supported") for _ in range(4)])
        state = self.run_chat(self.openai(), script)
        self.assertEqual(len(script.calls), 2)
        self.assertTrue(str(state["error"]).startswith("HTTP 400 bad request"), state["error"])

    def test_adaptations_are_bounded(self):
        provider = self.openai()
        script = Script(*[http_status(400, "whatever") for _ in range(20)])
        with mock.patch.object(provider, "_adapt_to_rejection", return_value=True):
            state = self.run_chat(provider, script)
        self.assertEqual(len(script.calls), provider_chat._MAX_ADAPTATIONS + 1)
        self.assertTrue(str(state["error"]).startswith("HTTP 400"), state["error"])

    def test_an_unnamed_400_is_final(self):
        script = Script(http_status(400, "messages: roles must alternate"), ok())
        state = self.run_chat(self.openai(), script)
        self.assertEqual(len(script.calls), 1)
        self.assertIn("roles must alternate", str(state["error"]))

    def test_optional_fields_are_dropped_one_by_one_across_retries(self):
        provider = self.openai("gpt-5", "https://api.openai.com/v1")
        script = Script(http_status(400, "Unrecognized request argument supplied: stream_options"),
                        http_status(400, "Unsupported parameter: reasoning_effort"), ok())
        state = self.run_chat(provider, script, reasoning="high")
        self.assertEqual(state["text"], "Hello")
        first, second, third = (c["json"] for c in script.calls)
        self.assertIn("stream_options", first)
        self.assertNotIn("stream_options", second)
        self.assertIn("reasoning_effort", second)
        self.assertNotIn("reasoning_effort", third)
        self.assertNotIn("stream_options", third)

    def test_a_tool_choice_rejection_retries_without_it(self):
        script = Script(http_status(400, "tool_choice is not supported by this model"), ok())
        state = self.run_chat(self.openai(), script, tools=[TOOL], tool_choice="required")
        self.assertEqual(state["text"], "Hello")
        self.assertIn("tool_choice", script.calls[0]["json"])
        self.assertNotIn("tool_choice", script.calls[1]["json"])
        self.assertIn("tools", script.calls[1]["json"])



class StopTest(FaultCase):
    def test_stop_before_the_first_request_sends_nothing(self):
        script = Script(ok())
        with mock.patch.object(providers.requests, "post", script):
            with self.assertRaises(providers.StopRequested):
                list(self.openai().chat("s", USER, [], lambda: True))
        self.assertEqual(script.calls, [])

    def test_stop_in_the_middle_of_a_stream_raises_and_closes_the_response(self):
        resp = Resp(200, openai_reply(text="a"), block=threading.Event())
        resp._lines = ["data: " + json.dumps({"choices": [{"delta": {"content": "a"}}]})]    # never terminates
        stop = threading.Event()
        provider = self.openai()
        events = []
        with mock.patch.object(providers.requests, "post", Script(resp)):
            gen = provider.chat("s", USER, [], stop.is_set)
            with self.assertRaises(providers.StopRequested):
                for event in gen:
                    events.append(event)
                    if event[0] == "text":
                        stop.set()
        self.assertEqual(events[0], ("text", "a"))
        self.assertTrue(resp.closed.wait(2), "the in-flight response must be closed")

    def test_closing_the_generator_cancels_the_request(self):
        resp = Resp(200, ["data: " + json.dumps({"choices": [{"delta": {"content": "a"}}]})], block=threading.Event())
        provider = self.openai()
        baseline = provider_transport.AttemptRunner.live_reader_count()
        with mock.patch.object(providers.requests, "post", Script(resp)):
            gen = provider.chat("s", USER, [])
            self.assertEqual(next(gen), ("text", "a"))
            started = time.monotonic()
            gen.close()
            self.assertLess(time.monotonic() - started, 1.5)
        self.assertTrue(resp.closed.wait(2))
        self._wait_readers(baseline)

    def _wait_readers(self, baseline, timeout=3.0):
        deadline = time.monotonic() + timeout
        while provider_transport.AttemptRunner.live_reader_count() > baseline and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertLessEqual(provider_transport.AttemptRunner.live_reader_count(), baseline)

    def test_stop_while_connecting_is_prompt_and_the_reader_ends_when_the_server_answers(self):
        release = threading.Event()
        late = Resp(200, OK_OPENAI)

        def blocked_post():
            release.wait(10)
            return late

        provider = self.openai()
        baseline = provider_transport.AttemptRunner.live_reader_count()
        stop_at = time.monotonic() + 0.3
        started = time.monotonic()
        with mock.patch.object(providers.requests, "post", Script(blocked_post)):
            with self.assertRaises(providers.StopRequested):
                list(provider.chat("s", USER, [], lambda: time.monotonic() >= stop_at))
        self.assertLess(time.monotonic() - started, 2.0)      # 0.3 s + at most the 1 s join of the cancel
        release.set()                                         # the "server" finally answers: the orphan reader must end
        self._wait_readers(baseline)
        self.assertTrue(late.closed.wait(2))

    def test_stop_during_a_backoff_sleep(self):
        # the real sleeper (FaultCase replaced it): Retry-After 30 must be interruptible within 200 ms polls
        provider = self.openai()
        stop_at = time.monotonic() + 0.5
        started = time.monotonic()
        with mock.patch.object(provider_chat.ChatLoop, "_sleep", _REAL_SLEEP):
            with mock.patch.object(providers.requests, "post", Script(http_status(429, headers={"Retry-After": "30"}), ok())):
                with self.assertRaises(providers.StopRequested):
                    list(provider.chat("s", USER, [], lambda: time.monotonic() >= stop_at))
        self.assertLess(time.monotonic() - started, 1.5)

    def test_stop_between_attempts(self):
        stop = threading.Event()
        calls = []

        def post_then_stop():
            calls.append(1)
            stop.set()
            return http_status(503)

        with mock.patch.object(providers.requests, "post", Script(post_then_stop, ok())):
            with self.assertRaises(providers.StopRequested):
                list(self.openai().chat("s", USER, [], stop.is_set))
        self.assertEqual(len(calls), 1)


class TickTest(FaultCase):
    def setUp(self):
        super().setUp()
        for name, value in (("_TICK_S", 0.05), ("_POLL_S", 0.02)):
            patcher = mock.patch.object(provider_chat, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_tick_payload_has_exactly_the_contract_keys(self):
        gate = threading.Event()

        def slow_stream():
            gate.wait(0.3)
            return ok()

        state = self.run_chat(self.openai(), Script(slow_stream))
        self.assertTrue(state["ticks"], "a silent connect must still produce ticks")
        keys = {"elapsed", "reasoning_chars", "text_chars", "tool_chars", "out_tokens", "phase", "tool", "retry", "idle_s"}
        for tick in state["ticks"]:
            self.assertEqual(set(tick), keys)
        self.assertEqual(state["ticks"][0]["phase"], "connecting")
        self.assertIsNone(state["ticks"][0]["out_tokens"])
        elapsed = [t["elapsed"] for t in state["ticks"]]
        self.assertEqual(elapsed, sorted(elapsed))
        idle = [t["idle_s"] for t in state["ticks"]]
        self.assertGreater(idle[-1], idle[0])

    def test_phases_and_cumulative_counters_follow_the_stream(self):
        steps = [
            ("data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "think" * 4}}]}), 0.3),
            ("data: " + json.dumps({"choices": [{"delta": {"content": "hello"}}]}), 0.3),
            ("data: " + json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c", "function": {"name": "create_object", "arguments": '{"a":'}}]}}]}), 0.3),
            ("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}), 0.0),
            ("data: [DONE]", 0.0),
        ]

        class Slow(Resp):
            def iter_lines(self, decode_unicode=True):
                for line, pause in steps:
                    yield line
                    time.sleep(pause)

        state = self.run_chat(self.openai(), Script(Slow()), tools=[TOOL])
        phases = [t["phase"] for t in state["ticks"]]
        self.assertIn("thinking", phases)
        self.assertIn("writing", phases)
        self.assertIn("tool_args", phases)
        self.assertLess(phases.index("thinking"), phases.index("writing"))
        self.assertLess(phases.index("writing"), phases.index("tool_args"))
        last = [t for t in state["ticks"] if t["phase"] == "tool_args"][-1]
        self.assertEqual((last["reasoning_chars"], last["text_chars"], last["tool"]), (20, 5, "create_object"))
        self.assertGreaterEqual(last["tool_chars"], 1)
        for t in state["ticks"]:
            if t["phase"] != "tool_args":
                self.assertIsNone(t["tool"])

    def test_sse_comment_lines_count_as_liveness(self):
        class Quiet(Resp):
            def iter_lines(self, decode_unicode=True):
                for _ in range(8):
                    yield ": keepalive"
                    time.sleep(0.1)
                yield from OK_OPENAI

        state = self.run_chat(self.openai(), Script(Quiet()))
        mid = [t for t in state["ticks"] if t["elapsed"] > 0.4]
        self.assertTrue(mid)
        self.assertTrue(all(t["idle_s"] < 0.35 for t in mid), [t["idle_s"] for t in mid])

    def test_counters_restart_with_each_attempt(self):
        def attempt1():
            return Resp(200, openai_reply(text="a" * 50, terminal=None))

        class Slow(Resp):
            def iter_lines(self, decode_unicode=True):
                time.sleep(0.2)
                yield from OK_OPENAI

        state = self.run_chat(self.openai(), Script(attempt1, Slow()))
        self.assertEqual(state["resets"], ["stream_dropped"])
        after = [t for t in state["ticks"] if t["elapsed"] > 0]
        self.assertTrue(all(t["text_chars"] <= 5 or t["text_chars"] == 50 for t in after))

    def test_ticks_continue_during_a_backoff_sleep(self):
        provider = self.openai()
        with mock.patch.object(provider_chat.ChatLoop, "_sleep", _REAL_SLEEP):
            state = self.run_chat(provider, Script(http_status(503, headers={"Retry-After": "0.4"}), ok()))
        retrying = [t for t in state["ticks"] if t["phase"] == "retrying"]
        self.assertGreaterEqual(len(retrying), 3)
        self.assertEqual(retrying[0]["retry"], [2, 5])
        self.assertEqual(state["text"], "Hello")

    def test_a_slow_consumer_never_blocks_the_reader(self):
        lines = [f"data: " + json.dumps({"choices": [{"delta": {"content": "x"}}]}) for _ in range(3000)]
        lines += openai_reply(text="")[-2:]
        provider = self.openai()
        baseline = provider_transport.AttemptRunner.live_reader_count()
        with mock.patch.object(providers.requests, "post", Script(ok(lines))):
            gen = provider.chat("s", USER, [])
            next(gen)
            time.sleep(0.5)                       # the consumer stalls; the reader must still run to completion
            self.assertLessEqual(provider_transport.AttemptRunner.live_reader_count(), baseline)
            text = "".join(p for k, p in gen if k == "text")
        self.assertEqual(len(text), 2999)


class AttemptRunnerTest(unittest.TestCase):
    def test_events_exceptions_and_the_final_end_marker(self):
        def work(run):
            run.put("a", 1)
            raise ValueError("boom")

        run = provider_transport.AttemptRunner(work)
        run.start()
        run._thread.join(2)
        items = []
        while not run.events.empty():
            items.append(run.events.get())
        self.assertEqual([k for k, _ in items], ["a", "_exc", "_end"])
        self.assertIsInstance(items[1][1], ValueError)

    def test_put_never_blocks(self):
        run = provider_transport.AttemptRunner(lambda r: [r.put("x", i) for i in range(100_000)])
        run.start()
        run._thread.join(5)
        self.assertFalse(run.alive)
        self.assertEqual(run.events.qsize(), 100_001)

    def test_cancel_after_normal_end_is_a_noop_that_reports_success(self):
        run = provider_transport.AttemptRunner(lambda r: None)
        run.start()
        run._thread.join(2)
        self.assertTrue(run.cancel(0.1))
        self.assertTrue(run.abort.is_set())

    def test_cancel_closes_the_response_and_joins_a_blocked_reader(self):
        resp = Resp(200, [], block=threading.Event())

        def work(run):
            run.response = resp
            list(resp.iter_lines())

        run = provider_transport.AttemptRunner(work)
        run.start()
        time.sleep(0.05)
        started = time.monotonic()
        self.assertTrue(run.cancel(1.0))
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(resp.closed.is_set())

    def test_cancel_reports_a_reader_that_will_not_end(self):
        release = threading.Event()
        run = provider_transport.AttemptRunner(lambda r: release.wait(5))
        run.start()
        try:
            self.assertFalse(run.cancel(0.1))
            self.assertGreaterEqual(provider_transport.AttemptRunner.live_reader_count(), 1)
        finally:
            release.set()
            run._thread.join(2)

    def test_the_live_reader_count_returns_to_zero(self):
        base = provider_transport.AttemptRunner.live_reader_count()
        run = provider_transport.AttemptRunner(lambda r: None)
        run.start()
        run._thread.join(2)
        self.assertEqual(provider_transport.AttemptRunner.live_reader_count(), base)

    def test_shutdown_socket_tries_both_attribute_chains_and_survives_fakes(self):
        self.assertFalse(provider_transport.shutdown_socket(object()))
        self.assertFalse(provider_transport.shutdown_socket(None))
        a, b = socket.socketpair()
        try:
            chain1 = mock.Mock()
            chain1.raw = mock.Mock(_fp=mock.Mock(fp=mock.Mock(raw=mock.Mock(_sock=a))), _connection=None)
            self.assertTrue(provider_transport.shutdown_socket(chain1))
            self.assertEqual(b.recv(1), b"")                 # the peer sees the shutdown at once
        finally:
            a.close()
            b.close()
        c, d = socket.socketpair()
        try:
            chain2 = mock.Mock()
            chain2.raw = mock.Mock(_fp=None, _connection=mock.Mock(sock=c))
            self.assertTrue(provider_transport.shutdown_socket(chain2))
        finally:
            c.close()
            d.close()
        closed = socket.socket()
        closed.close()
        broken = mock.Mock()
        broken.raw = mock.Mock(_fp=None, _connection=mock.Mock(sock=closed))
        self.assertFalse(provider_transport.shutdown_socket(broken))      # an already-closed socket is not an error

    def test_close_quietly(self):
        provider_transport.close_quietly(object())
        provider_transport.close_quietly(None)
        provider_transport.close_quietly(mock.Mock(close=mock.Mock(side_effect=OSError("x"))))


class HttpErrorTextTest(unittest.TestCase):
    def test_human_sentences(self):
        e = provider_transport.http_error(429, "slow down")
        self.assertEqual((str(e), e.status, e.retriable, e.kind), ("HTTP 429 rate limited: slow down", 429, True, "http"))
        self.assertEqual(str(provider_transport.http_error(401, "")), "HTTP 401 unauthorized (check the API key)")
        self.assertIn("check the base URL", str(provider_transport.http_error(404, "")))
        self.assertEqual(str(provider_transport.http_error(599, "x")), "HTTP 599 server error: x")
        self.assertEqual(str(provider_transport.http_error(418, "tea")), "HTTP 418 error: tea")

    def test_whitespace_is_collapsed_and_the_body_clipped(self):
        message = str(provider_transport.http_error(500, "a\n\n  b\t" + "c" * 500))
        self.assertTrue(message.startswith("HTTP 500 server error: a b c"))
        self.assertEqual(len(message.split(": ", 1)[1]), 200)

    def test_quota_wins_over_the_status_label(self):
        e = provider_transport.http_error(429, '{"code": "insufficient_quota"}')
        self.assertFalse(e.retriable)
        self.assertTrue(str(e).startswith("HTTP 429 quota or billing problem"))
        self.assertTrue(provider_transport.is_quota_error("Your CREDIT balance is too low"))
        self.assertFalse(provider_transport.is_quota_error("rate limited"))

    def test_in_stream_message_classification(self):
        r = provider_transport.retriable_message
        self.assertTrue(r("overloaded_error Overloaded"))
        self.assertTrue(r("Rate limit reached for requests"))
        self.assertTrue(r("Too many requests"))
        self.assertTrue(r("The server is at capacity"))
        self.assertFalse(r("insufficient_quota: rate limit"))      # quota beats rate limit
        self.assertFalse(r("invalid_request_error"))

    def test_transport_error_classification(self):
        t = provider_transport.is_transport_error
        self.assertTrue(t(requests.exceptions.ConnectionError()))
        self.assertTrue(t(OSError()))
        self.assertTrue(t(http.client.IncompleteRead(b"x")))
        self.assertFalse(t(requests.exceptions.InvalidURL()))
        self.assertFalse(t(ValueError()))
        self.assertFalse(t(KeyError("x")))




# --- a real HTTP server ---------------------------------------------------------------------------------------


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, script):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.script = list(script)
        self.requests = []
        self.closed_at = None
        self.lock = threading.Lock()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


def _chunk(h, text):
    """One HTTP/1.1 chunk (real LLM servers stream chunked bodies; ``requests`` reads those as they arrive)."""
    data = text.encode("utf-8")
    h.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
    h.wfile.flush()


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        server = self.server
        with server.lock:
            server.requests.append({"path": self.path, "headers": dict(self.headers.items()), "body": json.loads(body or b"{}")})
            behaviour = server.script.pop(0) if server.script else None
        self.close_connection = True
        if behaviour is not None:
            behaviour(self)

    # behaviours -------------------------------------------------------
    @staticmethod
    def _sse_headers(h, charset=None):
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream" + (f"; charset={charset}" if charset else ""))
        h.send_header("Transfer-Encoding", "chunked")
        h.send_header("Connection", "close")
        h.end_headers()

    @staticmethod
    def stream(lines, charset=None, delay_headers=0.0):
        def run(h):
            if delay_headers:
                time.sleep(delay_headers)
            _Handler._sse_headers(h, charset)
            for line in lines:
                _chunk(h, line + "\n\n")
            h.wfile.write(b"0\r\n\r\n")
            h.wfile.flush()
        return run

    @staticmethod
    def status(code, text="", headers=None):
        def run(h):
            payload = text.encode("utf-8")
            h.send_response(code)
            for k, v in (headers or {}).items():
                h.send_header(k, v)
            h.send_header("Content-Length", str(len(payload)))
            h.send_header("Connection", "close")
            h.end_headers()
            h.wfile.write(payload)
        return run

    @staticmethod
    def stall_after(lines):
        """Send ``lines``, then say nothing and watch the connection: records when the client hangs up."""
        def run(h):
            _Handler._sse_headers(h)
            for line in lines:
                _chunk(h, line + "\n\n")
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                readable, _, _ = select.select([h.connection], [], [], 0.05)
                if readable:
                    try:
                        data = h.connection.recv(1, socket.MSG_PEEK)
                    except OSError:
                        data = b""
                    if not data:
                        h.server.closed_at = time.monotonic()
                        return
        return run


class RealServerTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
        env.start()
        self.addCleanup(env.stop)
        self._profiles = []
        self.addCleanup(lambda: forget_learned(*self._profiles))
        self.servers = []
        self.addCleanup(self._stop_servers)

    def _stop_servers(self):
        for server in self.servers:
            server.shutdown()
            server.server_close()

    def server(self, *script):
        server = _Server(script)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.servers.append(server)
        return server

    def provider(self, cls, server, model, suffix="", **extra):
        provider_name = {providers.OpenAICompatProvider: "openai_compat", providers.AnthropicProvider: "anthropic"}[cls]
        extra.setdefault("max_tokens", 512)
        p = params.Profile(name="srv", provider=provider_name, base_url=server.base + suffix, model=model,
                           api_key="key", **extra)
        self._profiles.append(p)
        forget_learned(p)
        return cls(p)

    def test_stalled_server_the_cancel_wakes_the_blocked_read_within_a_second(self):
        # socket.shutdown on the reader's socket is what wakes a recv that Response.close() alone cannot
        first = "data: " + json.dumps({"choices": [{"delta": {"content": "tick"}}]})
        server = self.server(_Handler.stall_after([first]))
        provider = self.provider(providers.OpenAICompatProvider, server, "m", "/v1")
        baseline = provider_transport.AttemptRunner.live_reader_count()
        gen = provider.chat("s", USER, [])
        for kind, payload in gen:
            if kind == "text":
                break
        started = time.monotonic()
        gen.close()
        returned = time.monotonic() - started
        self.assertLess(returned, 1.0, f"close() took {returned:.2f}s")
        deadline = time.monotonic() + 2.0
        while server.closed_at is None and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertIsNotNone(server.closed_at, "the server never saw the connection closed")
        self.assertLess(server.closed_at - started, 2.0)
        self.assertLessEqual(provider_transport.AttemptRunner.live_reader_count(), baseline)

    def test_stalled_server_stop_raises_promptly(self):
        server = self.server(_Handler.stall_after(["data: " + json.dumps({"choices": [{"delta": {"content": "a"}}]})]))
        provider = self.provider(providers.OpenAICompatProvider, server, "m", "/v1")
        stop = threading.Event()
        started = None
        with self.assertRaises(providers.StopRequested):
            for kind, _payload in provider.chat("s", USER, [], stop.is_set):
                if kind == "text":
                    stop.set()
                    started = time.monotonic()
        self.assertLess(time.monotonic() - started, 1.5)

    def test_stop_while_the_server_is_still_loading_the_model(self):
        server = self.server(_Handler.stream(OK_OPENAI, delay_headers=1.3))
        provider = self.provider(providers.OpenAICompatProvider, server, "m", "/v1")
        baseline = provider_transport.AttemptRunner.live_reader_count()
        started = time.monotonic()
        stop_at = started + 0.3
        with self.assertRaises(providers.StopRequested):
            list(provider.chat("s", USER, [], lambda: time.monotonic() >= stop_at))
        self.assertLess(time.monotonic() - started, 2.0)
        deadline = time.monotonic() + 4.0           # the orphaned reader ends once the server finally answers
        while provider_transport.AttemptRunner.live_reader_count() > baseline and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertLessEqual(provider_transport.AttemptRunner.live_reader_count(), baseline)

    def test_utf8_survives_a_stream_without_a_charset(self):
        chunk = "data: " + json.dumps({"choices": [{"delta": {"content": "Gehäuse Ø 5 mm ✓ — 外壳"}}]}, ensure_ascii=False)
        stop = "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        server = self.server(_Handler.stream([chunk, stop, "data: [DONE]"]))
        provider = self.provider(providers.OpenAICompatProvider, server, "m", "/v1")
        state = collect(provider.chat("s", USER, []))
        self.assertEqual(state["text"], "Gehäuse Ø 5 mm ✓ — 外壳")

    def test_a_real_503_with_retry_after_is_retried_with_the_header_honoured(self):
        server = self.server(_Handler.status(503, "busy", {"Retry-After": "7"}), _Handler.stream(OK_OPENAI))
        provider = self.provider(providers.OpenAICompatProvider, server, "m", "/v1")
        sleeps = []

        def fake_sleep(loop, delay, call, should_stop, retry):
            sleeps.append(delay)
            return iter(())

        with mock.patch.object(provider_chat.ChatLoop, "_sleep", fake_sleep):
            state = collect(provider.chat("s", USER, []))
        self.assertEqual((state["text"], sleeps), ("Hello", [7.0]))
        self.assertEqual(len(server.requests), 2)

    def test_a_real_429_quota_is_final(self):
        server = self.server(_Handler.status(429, '{"error": {"code": "insufficient_quota"}}'), _Handler.stream(OK_OPENAI))
        state = collect(self.provider(providers.OpenAICompatProvider, server, "m", "/v1").chat("s", USER, []))
        self.assertEqual(len(server.requests), 1)
        self.assertIn("quota or billing problem", str(state["error"]))

    def test_connection_refused_is_retried_then_reported(self):
        server = self.server()
        base = server.base
        server.shutdown()
        server.server_close()
        self.servers.remove(server)
        p = params.Profile(name="srv", provider="openai_compat", base_url=base + "/v1", model="m", api_key="k")
        provider = providers.OpenAICompatProvider(p)
        self._profiles.append(p)
        with mock.patch.object(provider_chat.ChatLoop, "_sleep", lambda *a, **k: iter(())):
            state = collect(provider.chat("s", USER, []))
        self.assertTrue(str(state["error"]).startswith("request failed after 5 attempts"), state["error"])

    def test_anthropic_round_trip_sends_the_cached_request_and_parses_the_reply(self):
        reply = anthropic_reply(blocks=[("thinking", "hm", "sig"), ("text", "Hi "), ("tool", "toolu_1", "create_object", '{"a": 1}')],
                                stop="tool_use", start_usage={"input_tokens": 5, "cache_read_input_tokens": 40},
                                delta_usage={"output_tokens": 9})
        server = self.server(_Handler.stream(reply))
        provider = self.provider(providers.AnthropicProvider, server, "claude-sonnet-4-5", max_tokens=8192)
        state = collect(provider.chat("SYSTEM", USER, [TOOL], tool_choice="required", reasoning="off"))
        (request,) = server.requests
        self.assertEqual(request["path"], "/v1/messages")
        self.assertEqual(request["headers"]["x-api-key"], "key")
        body = request["body"]
        self.assertEqual(body["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(body["tools"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(body["tool_choice"], {"type": "any"})
        self.assertTrue(body["stream"])
        self.assertEqual(state["text"], "Hi ")
        self.assertEqual([(c.id, c.arguments) for c in state["calls"]], [("toolu_1", {"a": 1})])

    def test_openai_round_trip_end_to_end(self):
        lines = openai_reply(calls=[("c1", "create_object", '{"a": 1}')], finish="tool_calls",
                             usage={"prompt_tokens": 7, "completion_tokens": 2})
        server = self.server(_Handler.stream(lines))
        provider = self.provider(providers.OpenAICompatProvider, server, "m", "/v1")
        events = list(provider.chat("S", USER, [TOOL]))
        info = next(p for k, p in events if k == "stop_info")
        self.assertEqual((info["reason"], info["usage"]), ("tool_use", {"prompt_tokens": 7, "completion_tokens": 2}))
        self.assertEqual(server.requests[0]["path"], "/v1/chat/completions")



# -- review additions: line separators, null-valued fields, per-call output caps -------------------------------------

from oscar import provider_stream  # noqa: E402
from test_providers import FakeResponse  # noqa: E402

_ODD_SEPARATORS = "a b c\u0085d"


class RealServerSeparatorTest(RealServerTest):
    """``requests`` splits decoded lines with ``str.splitlines()``, which also breaks at U+2028 / U+2029 / U+0085.
    JSON serializers leave those unescaped, so a chunk holding one was cut in half and silently dropped."""

    def test_text_with_unicode_line_separators_survives_a_real_stream(self):
        lines = ["data: " + json.dumps({"choices": [{"delta": {"content": c}}]}, ensure_ascii=False)
                 for c in ("one ", _ODD_SEPARATORS, " two")]
        lines.append("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}))
        lines.append("data: [DONE]")
        server = self.server(_Handler.stream(lines))
        provider = self.provider(providers.OpenAICompatProvider, server, "m", "/v1")
        state = collect(provider.chat("s", USER, []))
        self.assertEqual(state["text"], "one " + _ODD_SEPARATORS + " two")
        self.assertIsNone(state["error"])

    def test_tool_arguments_with_separators_are_not_silently_shortened(self):
        code = "x = 1  # " + _ODD_SEPARATORS + "\nprint(x)"
        args = json.dumps({"code": code}, ensure_ascii=False)
        half = len(args) // 2
        pieces = [args[:half], args[half:]]
        lines = ["data: " + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "create_object", "arguments": piece}}]}}]}, ensure_ascii=False)
            for piece in pieces]
        lines.append("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}))
        lines.append("data: [DONE]")
        server = self.server(_Handler.stream(lines))
        provider = self.provider(providers.OpenAICompatProvider, server, "m", "/v1")
        state = collect(provider.chat("s", USER, [TOOL]))
        (call,) = state["calls"]
        self.assertEqual(call.args_error, "")
        self.assertEqual(call.arguments, {"code": code})

    def test_anthropic_text_with_separators_survives(self):
        lines = anthropic_reply(blocks=[("text", _ODD_SEPARATORS)])
        lines = [line.replace("\\u2028", " ").replace("\\u2029", " ").replace("\\u0085", "\u0085")
                 for line in lines]
        self.assertIn(" ", "".join(lines))
        server = self.server(_Handler.stream(lines))
        provider = self.provider(providers.AnthropicProvider, server, "claude-sonnet-4-5")
        self.assertEqual(collect(provider.chat("s", USER, []))["text"], _ODD_SEPARATORS)

    def test_crlf_terminated_streams_still_work(self):
        lines = [line + "\r" for line in OK_OPENAI]
        server = self.server(_Handler.stream(lines))
        provider = self.provider(providers.OpenAICompatProvider, server, "m", "/v1")
        state = collect(provider.chat("s", USER, []))
        self.assertEqual(state["text"], "Hello")
        self.assertIsNone(state["error"])


class SseLineIteratorTest(unittest.TestCase):
    def test_a_response_without_the_delimiter_argument_is_iterated_the_old_way(self):
        lines = ["data: {\"a\": 1}", "", "data: [DONE]"]
        self.assertEqual(list(provider_stream.sse_lines(FakeResponse(lines))), ['{"a": 1}', "[DONE]"])

    def test_the_delimiter_is_a_line_feed_and_stray_carriage_returns_split_lines(self):
        seen = {}

        class Resp2:
            def iter_lines(self, decode_unicode=True, delimiter=None):
                seen["delimiter"] = delimiter
                return iter(["data: one\r", "", "data: two\rdata: three"])

        out = list(provider_stream.sse_lines(Resp2()))
        self.assertEqual(seen["delimiter"], "\n")
        self.assertEqual(out, ["one", "two", "three"])

    def test_a_type_error_raised_while_iterating_is_not_mistaken_for_a_missing_argument(self):
        class Resp3:
            def iter_lines(self, decode_unicode=True, delimiter=None):
                def gen():
                    yield "data: ok"
                    raise TypeError("boom")
                return gen()

        out = []
        with self.assertRaises(TypeError):
            for payload in provider_stream.sse_lines(Resp3()):
                out.append(payload)
        self.assertEqual(out, ["ok"])


def _parse(provider, events):
    lines = events if isinstance(events, list) else [events]
    return list(provider._parse_stream(FakeResponse(["data: " + json.dumps(e) for e in lines]), lambda: False))


class NullFieldParserTest(unittest.TestCase):
    """Gateways send ``null`` (or a scalar) where an object belongs: the parsers skip it, they never crash the reader."""

    def setUp(self):
        self._profiles = []
        self.addCleanup(lambda: forget_learned(*self._profiles))

    def make(self, cls, provider, model, base="https://llm.example.com/v1"):
        p = params.Profile(name="p", provider=provider, base_url=base, model=model, api_key="k", max_tokens=512)
        self._profiles.append(p)
        forget_learned(p)
        return cls(p)

    def kinds(self, events):
        return [k for k, _ in events]

    def test_anthropic_null_objects_and_chunks(self):
        provider = self.make(providers.AnthropicProvider, "anthropic", "claude-sonnet-4-5", "https://api.anthropic.com")
        events = _parse(provider, [
            {"type": "message_start", "message": None},
            {"type": "content_block_start", "index": 0, "content_block": None},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": None},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": None}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "hi"}},
            {"type": "content_block_delta", "index": [1], "delta": {"type": "text_delta", "text": "!"}},
            {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "t1", "name": "go"}},
            {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": None}},
            {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"a\": 1}"}},
            {"type": "message_delta", "delta": None, "usage": 5},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        ])
        calls = next(p for k, p in events if k == "tool_calls")
        self.assertEqual([(c.name, c.arguments, c.args_error) for c in calls], [("go", {"a": 1}, "")])
        self.assertEqual("".join(p for k, p in events if k == "text"), "hi!")
        self.assertEqual(self.kinds(events)[-1], "stream_end")

    def test_openai_malformed_choices_and_tool_fragments(self):
        provider = self.make(providers.OpenAICompatProvider, "openai_compat", "m")
        events = _parse(provider, [
            {"choices": {"0": {}}},
            {"choices": [None]},
            {"choices": ["x"]},
            {"choices": [{"delta": None}]},
            {"choices": [{"delta": {"tool_calls": [None, "x", {"index": True, "function": None}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": 7, "function": {"name": 5, "arguments": ["a"]}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "go", "arguments": "{\"b\":"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "2}"}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ])
        # the list-valued ``arguments`` of the broken fragment is carried as JSON text: the call is then malformed, not a crash
        calls = next(p for k, p in events if k == "tool_calls")
        self.assertEqual([c.name for c in calls], ["go"])
        self.assertTrue(calls[0].args_error)

    def test_openai_content_given_as_typed_parts(self):
        provider = self.make(providers.OpenAICompatProvider, "openai_compat", "m")
        events = _parse(provider, [
            {"choices": [{"delta": {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "hmm "}]},
                                                 {"type": "text", "text": "Hello"}, None, {"type": "text", "text": None}]}}]},
            {"choices": [{"delta": {"content": [{"type": "text", "text": " world"}, "!"]}}]},
            {"choices": [{"delta": {"content": 5}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])
        self.assertEqual("".join(p for k, p in events if k == "text"), "Hello world!")
        self.assertEqual("".join(p for k, p in events if k == "reasoning"), "hmm ")



class PerCallCapAdaptationTest(unittest.TestCase):
    def test_a_cap_error_is_compared_with_the_value_the_request_carried(self):
        p = params.Profile(name="cap", provider="openai_compat", base_url="https://llm.example.com/v1", model="capm",
                           api_key="k", max_tokens=1000)
        forget_learned(p)
        self.addCleanup(lambda: forget_learned(p))
        provider = providers.OpenAICompatProvider(p)
        body = provider._body("s", [Message(role="user", content="hi")], [], max_tokens=3000)
        self.assertEqual(body["max_tokens"], 3000)       # the per-call value (the profile's own cap is 1000)
        self.assertTrue(provider._adapt_to_rejection("max_tokens is too large: 3000. This model supports at most 2048 "
                                                     "completion tokens", body))
        again = provider._body("s", [Message(role="user", content="hi")], [], max_tokens=3000)
        self.assertEqual(again["max_tokens"], 2048)
        self.assertFalse(provider._adapt_to_rejection("supports at most 4096 completion tokens", again))   # not lower: no loop



class NonFiniteArgumentsTest(unittest.TestCase):
    """``NaN`` / ``Infinity`` are not JSON. Kept in the history they make every later request fail to serialize
    (``requests`` refuses out-of-range floats), so a call that carries one is reported as malformed instead."""

    def test_nan_infinity_and_overflowing_floats_are_malformed(self):
        for raw in ('{"a": NaN}', '{"a": -Infinity}', '{"a": Infinity, "b": 1}', '{"a": 1e999}', '{"a": [1, NaN]}',
                    '"{\\"a\\": NaN}"'):
            arguments, error = providers.decode_tool_arguments(raw)
            self.assertIn("malformed JSON", error, raw)
            self.assertEqual(arguments, {"_raw": raw})

    def test_ordinary_numbers_and_strings_that_mention_nan_are_untouched(self):
        raw = '{"a": 1.5e3, "b": -0.0, "c": [0.1, 2], "code": "x = float(\'nan\')  # Infinity NaN"}'
        arguments, error = providers.decode_tool_arguments(raw)
        self.assertEqual(error, "")
        self.assertEqual(arguments["a"], 1500.0)
        self.assertEqual(arguments["code"], "x = float('nan')  # Infinity NaN")

    def test_a_replayed_call_never_puts_a_non_finite_number_on_the_wire(self):
        p = params.Profile(name="nf", provider="anthropic", base_url="https://api.anthropic.com", model="claude-sonnet-4-5",
                           api_key="k", max_tokens=512)
        forget_learned(p)
        provider = providers.AnthropicProvider(p)
        events = list(provider._parse_stream(FakeResponse(anthropic_reply(
            blocks=[("tool", "t1", "create_object", '{"Length": NaN}')], stop="tool_use")), lambda: False))
        (call,) = next(payload for kind, payload in events if kind == "tool_calls")
        self.assertTrue(call.args_error)
        history = [Message(role="user", content="go"),
                   Message(role="assistant", tool_calls=[call]),
                   Message(role="user", tool_results=[providers.ToolResult("t1", "create_object", "bad args", True)])]
        body = provider._body("sys", history, [])
        json.dumps(body, allow_nan=False)       # what requests does: must not raise
        self.assertEqual(body["messages"][1]["content"][0]["input"], {})





# --------------------------------------------------------------------------------------------------------------
# A learn hook on every rejection path, text mode under faults, token persistence off the GUI thread


class LearnHooksTest(FaultCase):
    """Every 400 the provider adapts to is also REMEMBERED (``models.learn``), so the next worker starts right."""

    def learned(self, provider):
        return models.learned_fields(provider._caps_subject())

    def test_tools_unsupported_flips_to_text_mode_retries_once_and_is_remembered(self):
        provider = self.openai("tinyllama", "https://llm.example.com/v1")
        script = Script(http_status(400, '{"error": "tinyllama does not support tools"}'),
                        ok(openai_reply(text='Go.\n<tool_call>{"name": "create_object", "arguments": {"a": 1}}</tool_call>')))
        state = self.run_chat(provider, script, tools=[TOOL])
        self.assertEqual([(c.name, c.arguments) for c in state["calls"]], [("create_object", {"a": 1})])
        self.assertEqual(state["text"].strip(), "Go.")
        self.assertIn("tools", script.calls[0]["json"])
        self.assertNotIn("tools", script.calls[1]["json"])
        self.assertEqual(len(script.calls), 2)
        self.assertEqual(self.learned(provider)["tools"], "text")
        later = Script(ok())
        self.run_chat(providers.OpenAICompatProvider(provider.profile), later, tools=[TOOL])
        self.assertNotIn("tools", later.calls[0]["json"])                  # the next worker starts in text mode

    def test_a_tools_complaint_that_persists_in_text_mode_is_final(self):
        provider = self.openai("tinyllama", "https://llm.example.com/v1")
        script = Script(*[http_status(400, "tinyllama does not support tools") for _ in range(3)])
        state = self.run_chat(provider, script, tools=[TOOL])
        self.assertEqual(len(script.calls), 2)
        self.assertTrue(str(state["error"]).startswith("HTTP 400"), state["error"])

    def test_an_ambiguous_image_error_retries_text_only_without_persisting_vision(self):
        provider = self.openai("vlm-1", "https://llm.example.com/v1")
        history = [Message(role="user", content="look", images=["QUJD"])]
        script = Script(http_status(400, "Invalid base64 image data"), ok())
        with mock.patch.object(providers.requests, "post", script):
            state = collect(provider.chat("sys", history, []))
        self.assertEqual(state["text"], "Hello")
        self.assertIn("image_url", json.dumps(script.calls[0]["json"]))
        self.assertNotIn("image_url", json.dumps(script.calls[1]["json"]))
        self.assertNotIn("vision", self.learned(provider))
        self.assertTrue(providers.OpenAICompatProvider(provider.profile).vision_enabled)

    def test_an_unambiguous_image_error_is_persisted(self):
        provider = self.openai("text-1", "https://llm.example.com/v1")
        history = [Message(role="user", content="look", images=["QUJD"])]
        script = Script(http_status(400, '{"error": "This model does not support image input"}'), ok())
        with mock.patch.object(providers.requests, "post", script):
            collect(provider.chat("sys", history, []))
        self.assertIs(self.learned(provider)["vision"], False)
        self.assertFalse(providers.OpenAICompatProvider(provider.profile).vision_enabled)

    def test_tool_choice_reasoning_and_optional_field_rejections_are_learned(self):
        provider = self.openai("gpt-5", "https://api.openai.com/v1")
        script = Script(http_status(400, "Unsupported parameter: tool_choice"),
                        http_status(400, "Unsupported parameter: reasoning_effort"),
                        http_status(400, "Unrecognized request argument supplied: stream_options"), ok())
        self.run_chat(provider, script, tools=[TOOL], tool_choice="required", reasoning="high")
        learned = self.learned(provider)
        self.assertIs(learned["tool_choice_ok"], False)
        self.assertEqual(learned["reasoning_control"], "none")
        self.assertIs(learned["stream_options_ok"], False)
        self.assertEqual(provider._caps().reasoning_control, "none")

    def test_a_context_length_error_and_an_output_cap_error_are_learned(self):
        provider = self.openai("m", "https://llm.example.com/v1")
        state = self.run_chat(provider, Script(http_status(400, "This model's maximum context length is 8192 tokens, you sent 20000 tokens")))
        self.assertIn("HTTP 400", str(state["error"]))
        self.assertEqual(self.learned(provider)["context_window"], 8192)
        self.run_chat(provider, Script(http_status(400, "max_tokens: 1024 > 512, which is the maximum allowed"), ok()))
        self.assertEqual(self.learned(provider)["max_output"], 512)

    def test_the_buffering_of_tool_arguments_is_learned_over_a_whole_call(self):
        provider = self.openai("m", "https://llm.example.com/v1")
        long_args = json.dumps({"code": "z" * 700})
        state = self.run_chat(provider, Script(ok(openai_reply(calls=[("c1", "create_object", long_args)], finish="tool_calls"))), tools=[TOOL])
        self.assertEqual(len(state["calls"]), 1)
        self.assertIs(self.learned(provider)["tool_args_stream"], False)


class TextModeFaultTest(FaultCase):
    def text_provider(self):
        return self.openai("tinyllama", "https://llm.example.com/v1", caps_override={"tools": "text"})

    CALL = '<tool_call>{"name": "create_object", "arguments": {"a": 1}}</tool_call>'

    def test_a_stream_that_drops_inside_a_block_never_leaks_markup_and_the_retry_succeeds(self):
        cut = openai_reply(text="Go. " + self.CALL[:30], terminal=None)
        whole = openai_reply(text="Go. " + self.CALL)
        script = Script(ok(cut), ok(whole))
        state = self.run_chat(self.text_provider(), script, tools=[TOOL])
        self.assertEqual(state["resets"], ["stream_dropped"] if state["resets"] else [])
        self.assertEqual([c.arguments for c in state["calls"]], [{"a": 1}])
        self.assertNotIn("tool_call", state["text"])
        self.assertEqual(state["text"].strip(), "Go.")

    def test_a_429_in_the_middle_of_a_loop_keeps_the_text_history_format(self):
        history = [Message(role="user", content="go"),
                   Message(role="assistant", content="ok", tool_calls=[providers.ToolCall(id="c1", name="create_object", arguments={"a": 1})]),
                   Message(role="user", tool_results=[providers.ToolResult(tool_call_id="c1", name="create_object", content="done")])]
        script = Script(http_status(429, "slow down", {"Retry-After": "1"}), ok())
        with mock.patch.object(providers.requests, "post", script):
            collect(self.text_provider().chat("sys", history, [TOOL]))
        for call in script.calls:
            blob = json.dumps(call["json"])
            self.assertNotIn('"tool_calls"', blob)
            self.assertNotIn('"role": "tool"', blob)
            self.assertIn("<tool_result", blob)
        self.assertEqual(script.calls[0]["json"], script.calls[1]["json"])                # a retry sends the same request

    def test_stop_while_a_block_is_withheld_raises_and_leaks_nothing(self):
        stops = {"n": 0}

        def should_stop():
            stops["n"] += 1
            return stops["n"] > 3

        script = Script(ok(openai_reply(text="Start " + self.CALL, terminal=None)))
        events = []
        with mock.patch.object(providers.requests, "post", script):
            with self.assertRaises(providers.StopRequested):
                for event in self.text_provider().chat("sys", USER, [TOOL], should_stop):
                    events.append(event)
        self.assertNotIn("tool_call", "".join(p for k, p in events if k == "text"))




if __name__ == "__main__":
    unittest.main()
