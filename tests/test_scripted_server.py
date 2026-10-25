"""support.server.ScriptedServer, mock_llm_server and the scripted-provider extensions.

The server is validated against the REAL providers (``OpenAICompatProvider`` and ``AnthropicProvider`` over a real
socket), because its only job is to be indistinguishable from a model endpoint: text, reasoning, tool calls whose
arguments stream in pieces, usage, truncated arguments, HTTP faults, mid-stream drops and stalls, plus the reactive
mode the GUI smoke test relies on.
"""

import json
import os
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

from oscar import params, provider_chat, providers  # noqa: E402
from support.scripted_llm import (Round, ScriptedProvider, advance, bad_json, call, calls, empty, malformed_json,  # noqa: E402
                                  nameless, provider_error, say, think, tick)
from support.server import (HttpFault, ScriptedServer, drop_after, http_error, last_event, raw, render_anthropic,  # noqa: E402
                            render_openai, slow, stall_after)

Message = providers.Message
USER = [Message(role="user", content="hi")]


def collect(events):
    """Fold a provider event stream into one dict."""
    out = {"text": "", "reasoning": "", "kinds": [], "calls": None, "done": None, "error": None, "stop": None,
           "usage": None, "resets": [], "stop_info": None}
    for kind, payload in events:
        out["kinds"].append(kind)
        if kind == "text":
            out["text"] += payload
        elif kind == "reasoning":
            out["reasoning"] += payload
        elif kind == "tool_calls":
            out["calls"] = payload
        elif kind == "done":
            out["done"] = payload
        elif kind == "error":
            out["error"] = payload
        elif kind == "stop_reason":
            out["stop"] = payload
        elif kind == "usage":
            out["usage"] = payload
        elif kind == "reset":              # consumers discard the partial output of the dropped attempt
            out["resets"].append(payload)
            out["text"], out["reasoning"] = "", ""
        elif kind == "stop_info":
            out["stop_info"] = payload
    return out


class ServerCase(unittest.TestCase):
    """Real providers against a ScriptedServer; retry sleeps are recorded, not slept."""

    def setUp(self):
        self.sleeps = []

        def fake_sleep(loop, delay, call_state, should_stop, retry):
            self.sleeps.append((round(delay, 3), tuple(retry)))
            return iter(())

        for patcher in (mock.patch.object(provider_chat.ChatLoop, "_sleep", fake_sleep),
                        mock.patch.object(provider_chat, "log_event", lambda *a, **k: None)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def serve(self, script, **kw):
        server = ScriptedServer(script, **kw)
        self.addCleanup(server.shutdown)
        return server

    def provider(self, server, **extra):
        kind = "anthropic" if server.dialect == "anthropic" else "openai_compat"
        prof = params.Profile(name=f"p-{uuid.uuid4().hex[:6]}", provider=kind, base_url=server.base_url,
                              model=f"m-{uuid.uuid4().hex[:8]}", api_key="key", max_tokens=1024, **extra)
        cls = providers.AnthropicProvider if kind == "anthropic" else providers.OpenAICompatProvider
        return cls(prof)

    def chat(self, server, messages=None, **kw):
        return collect(self.provider(server).chat("system", messages or USER, [], **kw))


DIALECTS = ("openai", "anthropic")


class DialectTest(ServerCase):
    """Each Round builder comes out of the REAL parser as the same events the script described."""

    def each(self, script_factory, check):
        for dialect in DIALECTS:
            with self.subTest(dialect=dialect):
                server = self.serve(script_factory(), dialect=dialect)
                check(self.chat(server))

    def test_text(self):
        def check(r):
            self.assertEqual((r["text"], r["error"]), ("Hello there", None))
            self.assertEqual(r["stop_info"]["reason"], "end_turn")
        self.each(lambda: [say("Hello there")], check)

    def test_text_in_many_chunks_is_concatenated_in_order(self):
        script = lambda: [Round([("text", "a"), ("text", "b"), ("text", "c"), ("stop_reason", "stop"), ("done", "stop")])]  # noqa: E731
        self.each(script, lambda r: self.assertEqual(r["text"], "abc"))

    def test_long_text_is_split_into_several_chunks_and_survives(self):
        text = "word " * 1500
        self.each(lambda: [say(text)], lambda r: self.assertEqual(r["text"], text))

    def test_reasoning_then_text(self):
        def check(r):
            self.assertEqual((r["reasoning"], r["text"]), ("thinking it over", "the answer"))
            self.assertLess(r["kinds"].index("reasoning"), r["kinds"].index("text"))
        self.each(lambda: [say("the answer", reasoning="thinking it over")], check)

    def test_tool_call_arguments_stream_in_pieces_and_decode(self):
        args = {"doc_name": "D", "obj_type": "Part::Box", "obj_properties": {"Length": 10, "Width": 20, "Height": 30},
                "note": "x" * 200}

        def check(r):
            (tc,) = r["calls"]
            self.assertEqual((tc.id, tc.name, tc.arguments, tc.args_error), ("call_x", "create_object", args, ""))
            self.assertEqual(r["stop_info"]["reason"], "tool_use")
        self.each(lambda: [calls(call("create_object", "call_x", **args))], check)

    def test_several_calls_keep_their_order_and_ids(self):
        def check(r):
            self.assertEqual([(c.id, c.name, c.arguments) for c in r["calls"]],
                             [("c1", "get_objects", {"doc_name": "A"}), ("c2", "get_view", {"view_name": "Top"})])
        self.each(lambda: [calls(call("get_objects", "c1", doc_name="A"), call("get_view", "c2", view_name="Top"))], check)

    def test_a_tool_argument_named_name_is_possible(self):
        def check(r):
            self.assertEqual(r["calls"][0].arguments, {"name": "SmokeDoc"})
        self.each(lambda: [calls(call("create_document", "c1", name="SmokeDoc"))], check)

    def test_text_before_a_tool_call(self):
        def check(r):
            self.assertEqual(r["text"], "On it.")
            self.assertEqual(r["calls"][0].name, "get_objects")
        self.each(lambda: [calls(call("get_objects", "c1"), text="On it.")], check)

    def test_usage_is_reported(self):
        usage = {"prompt_tokens": 11, "completion_tokens": 7}
        for dialect, usage in (("openai", {"prompt_tokens": 11, "completion_tokens": 7}),
                               ("anthropic", {"input_tokens": 11, "output_tokens": 7})):
            with self.subTest(dialect=dialect):
                server = self.serve([say("ok", usage=usage)], dialect=dialect)
                r = self.chat(server)
                self.assertEqual(r["stop_info"]["usage"], {"prompt_tokens": 11, "completion_tokens": 7})

    def test_cut_off_reply_is_reported_as_max_tokens(self):
        def check(r):
            self.assertEqual(r["stop_info"]["reason"], "max_tokens")
            self.assertTrue(r["stop_info"]["truncated"])
        self.each(lambda: [think(300)], check)

    def test_truncated_arguments_set_args_error(self):
        def check(r):
            (tc,) = r["calls"]
            self.assertIn("truncated", tc.args_error)
            self.assertIn("_raw", tc.arguments)
        self.each(lambda: [bad_json("execute_code", raw='{"code": "import Part\\nprint(1')], check)

    def test_malformed_arguments_are_not_called_truncated(self):
        def check(r):
            (tc,) = r["calls"]
            self.assertIn("malformed", tc.args_error)
            self.assertNotIn("truncated", tc.args_error)
        self.each(lambda: [malformed_json("create_object", raw='{"doc_name": "D", "obj_type": Part::Box}')], check)

    def test_empty_reply(self):
        def check(r):
            self.assertEqual((r["text"], r["calls"], r["error"]), ("", None, None))
            self.assertIsNotNone(r["done"])
        self.each(lambda: [empty()], check)

    def test_in_stream_error_event(self):
        def check(r):
            self.assertIsInstance(r["error"], Exception)
            self.assertIn("boom", str(r["error"]))
        self.each(lambda: [provider_error("boom") for _ in range(6)], check)

    def test_non_ascii_text_survives_a_charset_less_stream(self):
        text = "Gehäuse für Ω-Platte ✓ \U0001f527"
        self.each(lambda: [say(text)], lambda r: self.assertEqual(r["text"], text))

    def test_unknown_stop_reason_is_passed_through(self):
        server = self.serve([say("x", stop="weird_reason")])
        self.assertEqual(self.chat(server)["stop"], "weird_reason")

    def test_ticks_and_advance_pseudo_events_have_no_wire_form(self):
        script = [Round([advance(50), tick(10), ("text", "ok"), ("stop_reason", "stop"), ("done", "stop")])]
        self.assertEqual(self.chat(self.serve(script))["text"], "ok")

    def test_nameless_call_round_is_an_empty_done(self):
        r = self.chat(self.serve([nameless()]))
        self.assertEqual((r["calls"], r["error"]), (None, None))


class RenderTest(unittest.TestCase):
    def test_openai_body_and_terminal_are_separate(self):
        body, terminal = render_openai(say("hi", usage={"prompt_tokens": 1, "completion_tokens": 2}))
        self.assertTrue(all(b.startswith(b"data: ") and b.endswith(b"\n\n") for b in body + terminal))
        self.assertEqual(terminal[-1], b"data: [DONE]\n\n")
        self.assertIn(b'"finish_reason": "stop"', terminal[0])
        self.assertIn(b'"usage"', terminal[1])
        self.assertNotIn(b"finish_reason", b"".join(body))

    def test_openai_finish_reason_follows_the_calls_when_no_stop_given(self):
        _, terminal = render_openai(Round([("tool_calls", [call("get_objects", "c1")])]))
        self.assertIn(b'"finish_reason": "tool_calls"', terminal[0])

    def test_openai_maps_anthropic_style_stop_names(self):
        _, terminal = render_openai(say("x", stop="max_tokens"))
        self.assertIn(b'"finish_reason": "length"', terminal[0])

    def test_anthropic_event_names_and_block_order(self):
        body, terminal = render_anthropic(Round([("reasoning", "think"), ("text", "say"),
                                                 ("tool_calls", [call("get_objects", "c1", doc_name="A")]),
                                                 ("stop_reason", "tool_use")]))
        text = b"".join(body + terminal).decode()
        kinds = [line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("event: ")]
        self.assertEqual(kinds[0], "message_start")
        self.assertEqual(kinds[-2:], ["message_delta", "message_stop"])
        self.assertEqual(kinds.count("content_block_start"), 3)
        self.assertEqual(kinds.count("content_block_stop"), 3)
        self.assertLess(text.index("thinking_delta"), text.index("text_delta"))
        self.assertLess(text.index("text_delta"), text.index("input_json_delta"))
        self.assertIn("signature_delta", text)

    def test_anthropic_error_has_no_terminal(self):
        body, terminal = render_anthropic(provider_error("Overloaded"))
        self.assertEqual(terminal, [])
        self.assertIn(b"overloaded_error", body[-1])

    def test_openai_error_has_no_terminal(self):
        body, terminal = render_openai(provider_error("rate limit exceeded"))
        self.assertEqual(terminal, [])
        self.assertIn(b"rate_limit_error", body[-1])

    def test_events_without_a_wire_form_are_ignored(self):
        body, _ = render_openai(Round([("tick", {}), ("advance", 5), ("reset", "retry"), ("text", "x")]))
        self.assertEqual(len(body), 1)


class LastEventTest(unittest.TestCase):
    def test_openai_user_tool_and_image_only_messages(self):
        self.assertEqual(last_event({"messages": [{"role": "user", "content": "build a box"}]}), ("user", "", "build a box"))
        convo = {"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "created"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]}
        self.assertEqual(last_event(convo), ("tool", "c1", "created"))

    def test_the_screenshot_caption_the_agent_adds_after_a_view_is_not_a_user_turn(self):
        picture = [{"type": "text", "text": "(screenshot after get_view)"},
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]
        convo = {"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c2b"}]},
            {"role": "tool", "tool_call_id": "c2b", "content": "Screenshot captured."},
            {"role": "user", "content": picture}]}
        self.assertEqual(last_event(convo), ("tool", "c2b", "Screenshot captured."))
        # a real user turn that happens to carry an image (an attachment) still is the last event
        attached = [{"type": "text", "text": "what is this part?"}, picture[1]]
        self.assertEqual(last_event({"messages": [{"role": "user", "content": attached}]}), ("user", "", "what is this part?"))
        # the caption without a picture is plain user text
        text_only = [{"type": "text", "text": "(screenshot of my bracket) why is it thin?"}]
        self.assertEqual(last_event({"messages": [{"role": "user", "content": text_only}]})[0], "user")

    def test_openai_user_text_parts(self):
        convo = {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]}
        self.assertEqual(last_event(convo), ("user", "", "a\nb"))

    def test_anthropic_tool_result_blocks(self):
        convo = {"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                                         {"type": "image", "source": {}}]}]}
        self.assertEqual(last_event(convo), ("tool", "t1", "ok"))

    def test_anthropic_tool_result_with_text_block_content(self):
        convo = {"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t9", "content": [{"type": "text", "text": "line"}]}]}]}
        self.assertEqual(last_event(convo), ("tool", "t9", "line"))

    def test_degenerate_inputs(self):
        for bad in (None, {}, {"messages": []}, {"messages": ["x", None]}, {"messages": [{"role": "assistant", "content": "a"}]}):
            self.assertEqual(last_event(bad), ("none", "", ""))
        self.assertEqual(last_event({"messages": [{"role": "user", "content": "   "}]}), ("none", "", ""))


class ScriptingTest(ServerCase):
    def test_list_script_is_consumed_in_order_and_recorded(self):
        server = self.serve([say("one"), say("two")])
        self.assertEqual(self.chat(server)["text"], "one")
        self.assertEqual(self.chat(server)["text"], "two")
        self.assertEqual(server.request_count, 2)
        self.assertEqual(server.remaining, 0)
        server.assert_exhausted()

    def test_requests_are_recorded_with_body_headers_and_tools(self):
        server = self.serve([say("x")])
        spec = providers.ToolSpec("get_objects", "List objects.", {"type": "object", "properties": {}})
        list(self.provider(server).chat("SYS", USER, [spec]))
        (rec,) = server.requests
        self.assertEqual((rec.method, rec.n, rec.kind, rec.status), ("POST", 1, "round", 200))
        self.assertTrue(rec.path.endswith("/chat/completions"))
        self.assertEqual(rec.headers["authorization"], "Bearer key")
        self.assertEqual(rec.tool_names, ["get_objects"])
        self.assertEqual(rec.last, ("user", "", "hi"))
        self.assertEqual(rec.messages[0]["role"], "system")
        self.assertTrue(rec.wait_finished(2.0))

    def test_anthropic_request_is_recorded_in_its_dialect(self):
        server = self.serve([say("x")], dialect="anthropic")
        self.chat(server)
        (rec,) = server.requests
        self.assertTrue(rec.path.endswith("/v1/messages"))
        self.assertEqual(rec.headers["x-api-key"], "key")
        self.assertEqual(rec.last, ("user", "", "hi"))

    def test_strict_exhaustion_is_a_loud_500(self):
        server = self.serve([say("only")])
        self.chat(server)
        r = self.chat(server)
        self.assertIsNotNone(r["error"])
        self.assertGreaterEqual(len(server.unexpected), 1)      # the provider retries a 500: every try is recorded
        self.assertEqual(server.requests[-1].kind, "exhausted")
        with self.assertRaisesRegex(AssertionError, "unexpected extra provider request #2"):
            server.assert_exhausted()

    def test_unserved_items_fail_assert_exhausted(self):
        server = self.serve([say("a"), say("b")])
        self.chat(server)
        with self.assertRaisesRegex(AssertionError, "1 scripted item"):
            server.assert_exhausted()

    def test_repeat_last_answers_every_further_request(self):
        server = self.serve([say("first"), say("again")], repeat_last=True)
        self.assertEqual([self.chat(server)["text"] for _ in range(4)], ["first", "again", "again", "again"])
        self.assertEqual(server.unexpected, [])

    def test_a_plain_string_item_is_a_text_reply(self):
        self.assertEqual(self.chat(self.serve(["plain"]))["text"], "plain")

    def test_add_appends_to_a_list_script(self):
        server = self.serve([say("a")])
        server.add(say("b"))
        self.assertEqual([self.chat(server)["text"], self.chat(server)["text"]], ["a", "b"])

    def test_reset_rewinds_the_script_and_the_records(self):
        server = self.serve([say("a")])
        self.chat(server)
        server.reset()
        self.assertEqual(server.request_count, 0)
        self.assertEqual(server.remaining, 1)

    def test_wait_for_requests(self):
        server = self.serve([say("a")])
        self.assertFalse(server.wait_for_requests(1, timeout=0.1))
        self.chat(server)
        self.assertTrue(server.wait_for_requests(1, timeout=0.1))

    def test_raise_exc_round_becomes_http_500(self):
        server = self.serve([Round(raise_exc=RuntimeError("model crashed"))] * 6)
        r = self.chat(server)
        self.assertIn("500", str(r["error"]))
        self.assertEqual(server.requests[0].kind, "http_fault")

    def test_raw_html_with_status_200_is_not_an_event_stream(self):
        server = self.serve([raw("<html>Please log in</html>", content_type="text/html")])
        r = self.chat(server)
        self.assertIn("event stream", str(r["error"]))

    def test_unsupported_item_is_a_500_not_a_hang(self):
        server = self.serve([object()] * 6)
        self.assertIsNotNone(self.chat(server)["error"])
        self.assertEqual(server.requests[0].kind, "serve_error")

    def test_models_endpoint(self):
        server = self.serve([], model="my-model")
        data = requests.get(server.url + "/v1/models", timeout=5).json()
        self.assertEqual(data["data"][0]["id"], "my-model")
        self.assertEqual(requests.get(server.url + "/nope", timeout=5).status_code, 404)

    def test_invalid_dialect_is_rejected(self):
        with self.assertRaises(ValueError):
            ScriptedServer([], dialect="cohere")

    def test_instances_do_not_share_state(self):
        a = self.serve([say("A1"), say("A2")])
        b = self.serve([say("B1")])
        self.assertEqual(self.chat(a)["text"], "A1")
        self.assertEqual(self.chat(b)["text"], "B1")
        self.assertEqual(self.chat(a)["text"], "A2")
        self.assertEqual((a.request_count, b.request_count), (2, 1))
        self.assertNotEqual(a.port, b.port)

    def test_shutdown_is_idempotent_and_frees_the_port(self):
        server = ScriptedServer([say("a")])
        port = server.port
        server.shutdown()
        server.shutdown()
        with self.assertRaises(requests.exceptions.ConnectionError):
            requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={}, timeout=2)

    def test_context_manager(self):
        with ScriptedServer([say("a")]) as server:
            self.assertGreater(server.port, 0)
            self.assertTrue(server.base_url.endswith("/v1"))
        self.assertFalse(ScriptedServer([], dialect="anthropic", start=False).base_url.endswith("/v1"))


class ReactiveTest(ServerCase):
    def test_reply_is_decided_from_the_request_not_the_ordinal(self):
        def script(req):
            kind, call_id, text = last_event(req)
            if kind == "user":
                return calls(call("get_objects", "c1", doc_name="D"))
            if kind == "tool" and call_id == "c1":
                return say("saw: " + text)
            return say("?")

        server = self.serve(script)
        prov = self.provider(server)
        first = collect(prov.chat("s", USER, []))
        self.assertEqual(first["calls"][0].name, "get_objects")
        history = USER + [Message(role="assistant", tool_calls=first["calls"]),
                          Message(role="user", tool_results=[providers.ToolResult("c1", "get_objects", "3 objects")])]
        second = collect(prov.chat("s", history, []))
        self.assertEqual(second["text"], "saw: 3 objects")
        # an extra identical request (a retry, a nudge round) gets the SAME answer: nothing shifts
        again = collect(prov.chat("s", history, []))
        self.assertEqual(again["text"], "saw: 3 objects")
        self.assertEqual(server.request_count, 3)

    def test_a_crashing_script_is_reported_on_the_wire(self):
        def script(req):
            raise KeyError("oops")
        server = self.serve(script)
        r = self.chat(server)
        self.assertIsNotNone(r["error"])
        self.assertEqual(server.requests[0].kind, "script_error")

    def test_none_from_a_reactive_script_counts_as_unexpected(self):
        server = self.serve(lambda req: None)
        self.chat(server)
        self.assertGreaterEqual(len(server.unexpected), 1)


class FaultTest(ServerCase):
    def test_429_with_retry_after_is_retried_and_the_delay_is_honoured(self):
        for dialect in DIALECTS:
            with self.subTest(dialect=dialect):
                self.sleeps.clear()
                server = self.serve([http_error(429, retry_after=2), say("recovered")], dialect=dialect)
                r = self.chat(server)
                self.assertEqual((r["text"], r["error"]), ("recovered", None))
                self.assertEqual(server.request_count, 2)
                self.assertEqual(self.sleeps[0][0], 2.0)
                self.assertEqual([x.status for x in server.requests], [429, 200])

    def test_5xx_is_retried(self):
        server = self.serve([http_error(503), http_error(500, "upstream down"), say("fine")])
        self.assertEqual(self.chat(server)["text"], "fine")
        self.assertEqual(server.request_count, 3)

    def test_400_is_not_retried(self):
        server = self.serve([http_error(400, '{"error": {"message": "bad request body"}}'), say("never")])
        r = self.chat(server)
        self.assertIn("400", str(r["error"]))
        self.assertEqual(server.request_count, 1)

    def test_persistent_429_gives_up_with_a_readable_error(self):
        server = self.serve([http_error(429)] * 6)
        r = self.chat(server)
        self.assertIn("429", str(r["error"]))
        self.assertEqual(server.request_count, 5)

    def test_http_error_default_body_is_a_provider_shaped_json(self):
        fault = http_error(429, retry_after="3")
        self.assertIsInstance(fault, HttpFault)
        self.assertEqual(json.loads(fault.body)["error"]["type"], "rate_limit_error")
        self.assertEqual(fault.headers["Retry-After"], "3")

    def test_mid_stream_drop_retries_without_duplicated_text(self):
        for dialect in DIALECTS:
            with self.subTest(dialect=dialect):
                partial = Round([("text", "The housing "), ("text", "has two "), ("stop_reason", "stop"), ("done", "stop")])
                # openai: 2 chunks = both text deltas; anthropic: message_start + block_start + both text deltas
                chunks = 4 if dialect == "anthropic" else 2
                server = self.serve([drop_after(partial, chunks=chunks), say("The housing has two parts.")], dialect=dialect)
                r = self.chat(server)
                self.assertEqual(r["error"], None)
                self.assertIn("stream_dropped", r["resets"])
                self.assertEqual(r["text"], "The housing has two parts.")   # collect() discards the dropped attempt on reset
                self.assertEqual(server.request_count, 2)

    def test_drop_before_the_first_chunk_is_an_early_eof_retry(self):
        server = self.serve([drop_after(say("lost"), chunks=0), say("ok")])
        r = self.chat(server)
        self.assertEqual((r["text"], r["error"]), ("ok", None))

    def test_connection_reset_mid_stream_is_retried(self):
        partial = Round([("text", "abc"), ("text", "def"), ("stop_reason", "stop"), ("done", "stop")])
        server = self.serve([drop_after(partial, chunks=1, style="reset"), say("whole")])
        r = self.chat(server)
        self.assertEqual((r["error"], r["text"]), (None, "whole"))   # the RST may discard the unread chunk: no reset event needed
        self.assertEqual(server.request_count, 2)

    def test_body_cut_without_the_chunk_terminator_is_retried(self):
        partial = Round([("text", "abc"), ("text", "def"), ("stop_reason", "stop"), ("done", "stop")])
        server = self.serve([drop_after(partial, chunks=1, style="cut"), say("whole")])
        r = self.chat(server)
        self.assertEqual((r["error"], r["text"]), (None, "whole"))
        self.assertEqual(server.request_count, 2)

    def test_every_attempt_dropping_ends_in_an_error(self):
        partial = Round([("text", "abc"), ("stop_reason", "stop"), ("done", "stop")])
        server = self.serve([drop_after(partial, chunks=1)] * 6)
        r = self.chat(server)
        self.assertIsNotNone(r["error"])
        self.assertNotIn("done", r["kinds"])

    def test_slow_stream_still_delivers_everything(self):
        server = self.serve([slow(Round([("text", "a"), ("text", "b"), ("text", "c"), ("stop_reason", "stop"),
                                         ("done", "stop")]), gap=0.05)])
        t0 = time.monotonic()
        r = self.chat(server)
        self.assertEqual(r["text"], "abc")
        self.assertGreaterEqual(time.monotonic() - t0, 0.1)

    def test_round_delay_postpones_the_reply(self):
        server = self.serve([Round([("text", "late"), ("stop_reason", "stop"), ("done", "stop")], delay=0.2)])
        t0 = time.monotonic()
        self.assertEqual(self.chat(server)["text"], "late")
        self.assertGreaterEqual(time.monotonic() - t0, 0.2)

    def test_chunk_delay_applies_to_every_gap(self):
        server = self.serve([Round([("text", "a"), ("text", "b"), ("stop_reason", "stop"), ("done", "stop")])],
                            chunk_delay=0.1)
        t0 = time.monotonic()
        self.chat(server)
        self.assertGreaterEqual(time.monotonic() - t0, 0.2)


class StallTest(ServerCase):
    """cancelling a stalled stream returns quickly and the SERVER sees the connection close."""

    def _start_stalled(self):
        partial = Round([("text", "partial answer "), ("text", "never finished"), ("stop_reason", "stop"), ("done", "stop")])
        server = self.serve([stall_after(partial, chunks=1, seconds=30.0)])
        gen = self.provider(server).chat("s", USER, [])
        for kind, payload in gen:
            if kind == "text":
                break
        return server, gen

    def test_closing_a_stalled_stream_is_fast_and_the_server_sees_the_close(self):
        server, gen = self._start_stalled()
        t0 = time.monotonic()
        gen.close()
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertTrue(server.requests[0].wait_closed(3.0), "the server never saw the client disconnect")
        self.assertFalse(server.requests[0].finished)

    def test_stop_while_stalled_ends_the_call(self):
        partial = Round([("text", "hi"), ("stop_reason", "stop"), ("done", "stop")])
        server = self.serve([stall_after(partial, chunks=1, seconds=30.0)])
        stop = threading.Event()
        threading.Timer(0.4, stop.set).start()
        t0 = time.monotonic()
        with self.assertRaises(providers.StopRequested):
            for _ in self.provider(server).chat("s", USER, [], should_stop=stop.is_set):
                pass
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertTrue(server.requests[0].wait_closed(3.0))

    def test_shutdown_releases_a_stalled_handler(self):
        partial = Round([("text", "hi"), ("stop_reason", "stop"), ("done", "stop")])
        server = ScriptedServer([stall_after(partial, chunks=1, seconds=60.0)])
        result = {}

        def run():
            result["r"] = collect(self.provider(server).chat("s", USER, []))

        t = threading.Thread(target=run, daemon=True)
        t.start()
        self.assertTrue(server.wait_for_requests(1, 3.0))
        time.sleep(0.2)
        t0 = time.monotonic()
        server.shutdown()
        self.assertLess(time.monotonic() - t0, 3.0)
        t.join(5.0)


class ScriptedProviderExtensionTest(unittest.TestCase):
    """The additive extensions: positional-only ``call``, ``advance`` pseudo-events, ``malformed_json``, ``tick``."""

    def test_call_accepts_a_name_argument(self):
        c = call("create_document", name="X")
        self.assertEqual((c.name, c.arguments), ("create_document", {"name": "X"}))
        self.assertEqual(call("get_objects", "cid", doc_name="D").id, "cid")
        self.assertEqual(call("get_objects", call_id="cid2").id, "cid2")

    def test_advance_calls_the_injected_clock_and_is_not_yielded(self):
        now = [0.0]
        prov = ScriptedProvider([Round([advance(200), ("text", "x"), ("stop_reason", "stop"), ("done", "stop")])],
                                advance=lambda s: now.__setitem__(0, now[0] + s))
        kinds = [k for k, _ in prov.chat("s", USER, [])]
        self.assertEqual((now[0], kinds), (200.0, ["text", "stop_reason", "done"]))

    def test_advance_without_a_clock_is_ignored(self):
        prov = ScriptedProvider([Round([advance(5), ("text", "x"), ("done", "stop")])])
        self.assertEqual([k for k, _ in prov.chat("s", USER, [])], ["text", "done"])

    def test_tick_payload_has_the_contract_keys(self):
        kind, payload = tick(12.4, reasoning_chars=5120)
        self.assertEqual(kind, "tick")
        self.assertEqual(set(payload), {"elapsed", "reasoning_chars", "text_chars", "tool_chars", "out_tokens", "phase",
                                        "tool", "retry", "idle_s"})
        self.assertEqual((payload["idle_s"], payload["phase"]), (12.4, "thinking"))

    def test_malformed_json_builds_a_decodable_error_round(self):
        (kind, calls_), = [e for e in malformed_json("x").events if e[0] == "tool_calls"]
        self.assertIn("malformed", calls_[0].args_error)
        self.assertTrue(calls_[0].raw_args)


class MockLlmServerTest(unittest.TestCase):
    """mock_llm_server keeps its public surface (gui_smoke imports it) but has no class-level globals."""

    def setUp(self):
        import mock_llm_server
        self.mod = mock_llm_server
        self.servers = []
        patcher = mock.patch.object(provider_chat, "log_event", lambda *a, **k: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        for s in self.servers:
            s.shutdown()

    def start(self, *args, **kw):
        kw.setdefault("port", 0)
        s = self.mod.make_server(*args, **kw)
        self.servers.append(s)
        return s

    def prov(self, server):
        prof = params.Profile(name="mock", provider="openai_compat", base_url=server.base_url,
                              model=f"mock-{uuid.uuid4().hex[:6]}", api_key="", max_tokens=512)
        return providers.OpenAICompatProvider(prof)

    def test_public_names_survive(self):
        for name in ("make_server", "standard_rounds", "attachment_rounds"):
            self.assertTrue(callable(getattr(self.mod, name)), name)
        self.assertEqual(len(self.mod.standard_rounds()), 4)
        self.assertEqual(len(self.mod.attachment_rounds("/x.stl")), 7)
        self.assertTrue(all(isinstance(b, bytes) for b in self.mod.attachment_rounds("/x.stl")))

    def test_ordinal_mode_walks_the_standard_rounds_and_repeats_the_last(self):
        server = self.start(self.mod.standard_rounds())
        prov = self.prov(server)
        got = [collect(prov.chat("s", USER, [])) for _ in range(5)]
        self.assertEqual([c.name for r in got[:3] for c in r["calls"]], ["create_document", "create_object", "get_view"])
        self.assertEqual(got[0]["text"], "Setting up a document.")
        self.assertEqual(got[3]["text"], "I created a 10x10x10 mm box in SmokeDoc.")
        self.assertEqual(got[4]["text"], "I created a 10x10x10 mm box in SmokeDoc.")   # the last item repeats
        self.assertEqual(server.request_count, 5)

    def test_state_is_per_server(self):
        a = self.start(self.mod.standard_rounds())
        b = self.start(self.mod.standard_rounds())
        collect(self.prov(a).chat("s", USER, []))
        collect(self.prov(a).chat("s", USER, []))
        first_of_b = collect(self.prov(b).chat("s", USER, []))
        self.assertEqual(first_of_b["calls"][0].name, "create_document")   # b did not inherit a's counter

    def test_server_object_has_the_surface_gui_smoke_uses(self):
        server = self.start(self.mod.standard_rounds())
        self.assertGreater(server.port, 0)
        self.assertTrue(callable(server.shutdown))

    def test_reactive_smoke_script_drives_both_conversations_by_content(self):
        server = self.start(reactive=self.mod.smoke_reactive("/tmp/smoke_part.stl"))
        prov = self.prov(server)

        def turn(history):
            return collect(prov.chat("s", history, []))

        def step(history, reply):
            history.append(Message(role="assistant", content=reply["text"], tool_calls=reply["calls"] or []))
            if reply["calls"]:
                history.append(Message(role="user", tool_results=[
                    providers.ToolResult(c.id, c.name, "ok") for c in reply["calls"]]))

        history = [Message(role="user", content="Build the smoke test box.")]
        names = []
        for _ in range(4):
            reply = turn(history)
            names += [c.name for c in reply["calls"] or []]
            step(history, reply)
            if not reply["calls"]:
                self.assertEqual(reply["text"], "I created a 10x10x10 mm box in SmokeDoc.")
                break
        self.assertEqual(names, ["create_document", "create_object", "get_view"])
        self.assertEqual(server.request_count, 4)
        # turn 2 on the same conversation: decided by the new user text, not by how many requests came before
        history.append(Message(role="user", content="Import the attached part and add a 5 mm cube. [/tmp/smoke_part.stl]"))
        reply = turn(history)
        self.assertEqual(reply["calls"][0].name, "import_file")
        self.assertEqual(reply["calls"][0].arguments["path"], "/tmp/smoke_part.stl")
        step(history, reply)
        reply = turn(history)
        self.assertEqual(reply["calls"][0].name, "create_object")
        self.assertEqual(reply["calls"][0].arguments["obj_name"], "Spacer")
        step(history, reply)
        reply = turn(history)
        self.assertEqual((reply["calls"], reply["text"], reply["reasoning"] != ""),
                         (None, "I imported the attached part and added a 5 mm cube.", True))
        self.assertEqual(server.request_count, 7)

    def test_reactive_extra_request_ends_the_turn_instead_of_desynchronising(self):
        server = self.start(reactive=self.mod.smoke_reactive())
        history = [Message(role="user", content="Build the smoke test box."),
                   Message(role="assistant", content="", tool_calls=[providers.ToolCall("zzz", "get_objects", {})]),
                   Message(role="user", tool_results=[providers.ToolResult("zzz", "get_objects", "x")])]
        self.assertEqual(collect(self.prov(server).chat("s", history, []))["text"], "Nothing more to do.")


if __name__ == "__main__":
    unittest.main()
