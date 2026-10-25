"""The native Ollama provider: ``/api/chat`` NDJSON, ``options.num_ctx`` (never trust ``/api/show`` as
the active window), tools, ``think``, text tool mode, the published window and the rejection / fault paths.

Field names of the native API are from documentation memory: the stream and request shapes
asserted here are what the provider is written against, not a statement about the installed Ollama.
"""

import json
import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import requests  # noqa: E402

from oscar import models, params, provider_ollama, provider_stream, providers  # noqa: E402
from support.server import ScriptedServer, raw  # noqa: E402
from test_provider_faults import FaultCase, Resp, Script, collect  # noqa: E402
from test_providers import FakeResponse, forget_learned, kinds, payload  # noqa: E402

Message, ToolCall, ToolResult, ToolSpec = providers.Message, providers.ToolCall, providers.ToolResult, providers.ToolSpec

URL = "http://localhost:11434/v1"
TOOL = ToolSpec("create_object", "Create an object. More.", {"type": "object", "properties": {"doc_name": {"type": "string"}}})
TOOL2 = ToolSpec("get_view", "Get a view.", {"type": "object", "properties": {}})
HIST = [Message(role="user", content="make a plate")]


def nd(*chunks):
    """NDJSON lines as ``iter_lines`` yields them."""
    return [json.dumps(c) for c in chunks]


def msg(content="", **extra):
    return {"model": "m", "created_at": "2026-01-01T00:00:00Z", "message": {"role": "assistant", "content": content, **extra},
            "done": False}


def done(reason="stop", **extra):
    return {"model": "m", "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": reason,
            "prompt_eval_count": 40, "eval_count": 7, **extra}


def texts(events):
    return "".join(p for k, p in events if k == "text")


class _Case(unittest.TestCase):
    def setUp(self):
        self._profiles = []
        self.addCleanup(lambda: forget_learned(*self._profiles))

    def prof(self, model="qwen2.5:7b", base_url=URL, **extra):
        extra.setdefault("max_tokens", 2048)
        extra.setdefault("api_key", "")
        p = params.Profile(name="ol", provider="openai_compat", base_url=base_url, model=model, **extra)
        self._profiles.append(p)
        forget_learned(p)
        return p

    def native(self, model="qwen2.5:7b", base_url=URL, **extra):
        return provider_ollama.OllamaNativeProvider(self.prof(model, base_url, **extra))

    def parse(self, provider, chunks, tools=(TOOL,), **opts):
        provider._body("SYS", HIST, list(tools), **opts)
        return list(provider._parse_stream(FakeResponse(nd(*chunks)), lambda: False))


# --------------------------------------------------------------------------------------------------------------


class RoutingTest(_Case):
    def test_an_ollama_host_gets_the_native_provider(self):
        for base in (URL, "http://localhost:11434", "http://ollama.lan:8080/v1", "localhost:11434/v1"):
            with self.subTest(base=base):
                self.assertIsInstance(providers.make_provider(self.prof(base_url=base)), provider_ollama.OllamaNativeProvider)

    def test_the_user_can_insist_on_the_openai_compatible_endpoint(self):
        p = providers.make_provider(self.prof(caps_override={"ollama_native": False}))
        self.assertIs(type(p), providers.OpenAICompatProvider)

    def test_other_hosts_and_providers_are_untouched(self):
        self.assertIs(type(providers.make_provider(self.prof(base_url="https://api.openai.com/v1"))), providers.OpenAICompatProvider)
        self.assertIs(type(providers.make_provider(self.prof(base_url="http://localhost:1234/v1"))), providers.OpenAICompatProvider)
        anth = params.Profile(name="a", provider="anthropic", base_url="http://localhost:11434", model="claude-sonnet-4-5")
        self.assertIs(type(providers.make_provider(anth)), providers.AnthropicProvider)
        # the account never becomes an Ollama (or OpenAI-compatible) provider, whatever its base URL says
        from oscar.provider_originshape import OriginShapeProvider

        for base in ("", URL):
            with self.subTest(account_base=base):
                account = params.Profile(name="o", provider="originshape", base_url=base, model="oscar-default")
                self.assertIs(type(providers.make_provider(account)), OriginShapeProvider)

    def test_make_provider_never_touches_the_network(self):
        with mock.patch.object(requests, "post", side_effect=AssertionError("network")), \
                mock.patch.object(requests, "get", side_effect=AssertionError("network")):
            providers.make_provider(self.prof())

    def test_the_provider_serves_the_openai_compat_profile_kind(self):
        self.assertEqual((provider_ollama.OllamaNativeProvider.name, provider_ollama.OllamaNativeProvider.kind),
                         ("ollama", "openai_compat"))

    def test_switching_back_to_v1_drops_the_published_window(self):
        profile = self.prof()
        providers.make_provider(profile)
        self.assertEqual(models.resolve_caps(profile).context_window, 16384)
        profile.caps_override = {"ollama_native": False}
        providers.make_provider(profile)
        self.assertEqual(models.resolve_caps(profile).context_window, 4096)       # the /v1 window of the table


class WindowTest(_Case):
    def test_the_default_window_is_16384_and_is_published_before_the_first_request(self):
        profile = self.prof()
        self.assertEqual(models.resolve_caps(profile).context_window, 4096)       # the /v1 seed, before any provider exists
        provider_ollama.OllamaNativeProvider(profile)
        caps = models.resolve_caps(profile)
        self.assertEqual((caps.context_window, models.ollama_num_ctx(profile)), (16384, 16384))

    def test_the_user_override_wins_and_is_what_is_sent(self):
        profile = self.prof(caps_override={"context_window": 8192})
        p = provider_ollama.OllamaNativeProvider(profile)
        self.assertEqual(models.resolve_caps(profile).context_window, 8192)
        self.assertEqual(p._body("S", HIST, [])["options"]["num_ctx"], 8192)
        profile.caps_override = {"context_window": 65536}
        self.assertEqual(provider_ollama.OllamaNativeProvider(profile)._body("S", HIST, [])["options"]["num_ctx"], 65536)

    def test_the_trained_maximum_caps_the_default_but_never_becomes_the_window(self):
        profile = self.prof()
        key = models.caps_key(profile)
        params.set_model_caps(key, {"api": {"context_window": 4096, "context_max": 8192},
                                    "api_ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.assertEqual(models.ollama_num_ctx(profile), 8192)
        profile.caps_override = {"context_window": 12000}
        self.assertEqual(models.ollama_num_ctx(profile), 12000)                    # an explicit choice is respected
        big = self.prof("other:model")
        params.set_model_caps(models.caps_key(big), {"api": {"context_max": 131072}, "api_ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.assertEqual(models.ollama_num_ctx(big), 16384)                        # NOT 131072

    def test_the_runtime_layer_is_in_process_only(self):
        profile = self.prof()
        provider_ollama.OllamaNativeProvider(profile)
        self.assertEqual(models.learned_fields(profile), {})
        self.assertEqual(params.get_model_caps(models.caps_key(profile)), {})

    def test_runtime_layer_helpers(self):
        profile = self.prof()
        models.set_runtime_caps(profile, context_window=9999, bogus=1, tools="nonsense")
        caps = models.resolve_caps(profile)
        self.assertEqual((caps.context_window, caps.tools, caps.source), (9999, "native", "learned"))   # the invalid tools value was dropped
        profile.caps_override = {"context_window": 5000}
        self.assertEqual(models.resolve_caps(profile).context_window, 5000)
        models.clear_runtime_caps(profile)
        profile.caps_override = {}
        self.assertEqual(models.resolve_caps(profile).context_window, 4096)

    def test_native_wanted(self):
        self.assertTrue(models.ollama_native_wanted(self.prof()))
        self.assertFalse(models.ollama_native_wanted(self.prof(caps_override={"ollama_native": False})))
        self.assertTrue(models.ollama_native_wanted(self.prof(caps_override={"ollama_native": True})))
        self.assertFalse(models.ollama_native_wanted(self.prof(base_url="https://api.openai.com/v1")))
        self.assertFalse(models.ollama_native_wanted(params.Profile(provider="anthropic", base_url="http://localhost:11434")))


class EndpointTest(_Case):
    def test_endpoint_is_the_server_root_plus_api_chat(self):
        for base, want in ((URL, "http://localhost:11434/api/chat"), (URL + "/", "http://localhost:11434/api/chat"),
                           ("http://localhost:11434", "http://localhost:11434/api/chat"),
                           ("http://h:8080/v1/chat/completions", "http://h:8080/api/chat"),
                           ("http://h:8080/api", "http://h:8080/api/chat"), ("", "http://localhost:11434/api/chat")):
            with self.subTest(base=base):
                self.assertEqual(self.native(base_url=base)._endpoint(), want)

    def test_headers_carry_a_key_only_when_there_is_one(self):
        self.assertEqual(self.native()._headers(), {"content-type": "application/json"})
        keyed = provider_ollama.OllamaNativeProvider(self.prof(model="keyed:1b", api_key="K"))
        self.assertEqual(keyed._headers()["Authorization"], "Bearer K")


class BodyTest(_Case):
    def test_the_request_shape(self):
        body = self.native(temperature=0.3)._body("SYS", HIST, [TOOL, TOOL2])
        self.assertEqual((body["model"], body["stream"]), ("qwen2.5:7b", True))
        self.assertEqual(body["options"], {"num_ctx": 16384, "num_predict": 2048, "temperature": 0.3})
        self.assertEqual(body["messages"][0], {"role": "system", "content": "SYS"})
        self.assertEqual(body["messages"][1], {"role": "user", "content": "make a plate"})
        self.assertEqual(body["tools"][0], {"type": "function", "function": {
            "name": "create_object", "description": "Create an object. More.", "parameters": TOOL.parameters}})
        self.assertNotIn("think", body)
        for absent in ("max_tokens", "tool_choice", "stream_options", "parallel_tool_calls"):
            self.assertNotIn(absent, body)

    def test_no_tools_key_without_tools_and_per_call_max_tokens(self):
        body = self.native()._body("S", HIST, [], max_tokens=300)
        self.assertNotIn("tools", body)
        self.assertEqual(body["options"]["num_predict"], 300)

    def test_temperature_follows_the_sampling_rule(self):
        self.assertIn("temperature", self.native()._body("S", HIST, [])["options"])
        p2 = self.native("other:7b")       # a learned "sampling" flag removes temperature
        p2._learned["sampling"] = False
        self.assertNotIn("temperature", p2._body("S", HIST, [])["options"])
        self.assertNotIn("temperature", self.native("third:7b", caps_override={"sampling_ok": False})._body("S", HIST, [])["options"])

    def test_think_is_sent_only_with_a_reasoning_control_and_an_explicit_effort(self):
        plain = self.native()
        self.assertNotIn("think", plain._body("S", HIST, [], reasoning="high"))            # no control: nothing is sent
        thinking = self.native("qwen3:8b", caps_override={"reasoning_control": "ollama_effort"})
        self.assertNotIn("think", thinking._body("S", HIST, []))                           # auto without a per-call value
        self.assertIs(thinking._body("S", HIST, [], reasoning="off")["think"], False)
        for level in ("low", "medium", "high"):
            self.assertIs(thinking._body("S", HIST, [], reasoning=level)["think"], True)
        gpt = self.native("gpt-oss:20b", caps_override={"reasoning_control": "ollama_effort"})
        self.assertEqual(gpt._body("S", HIST, [], reasoning="high")["think"], "high")
        self.assertIs(gpt._body("S", HIST, [], reasoning="off")["think"], False)
        explicit = self.native("qwen3:8b", reasoning="low", caps_override={"reasoning_control": "ollama_effort"})
        self.assertIs(explicit._body("S", HIST, [])["think"], True)

    def test_tool_choice_none_drops_the_tools_and_forced_choices_are_not_expressible(self):
        p = self.native()
        self.assertNotIn("tools", p._body("S", HIST, [TOOL], tool_choice="none"))
        self.assertIn("tools", p._body("S", HIST, [TOOL], tool_choice="required"))
        self.assertNotIn("tool_choice", p._body("S", HIST, [TOOL], tool_choice="required"))

    def test_text_tool_mode_sends_no_tools_and_appends_the_protocol(self):
        p = self.native("tinyllama")                           # an unknown Ollama model is text mode by the table
        body = p._body("SYS", HIST, [TOOL])
        self.assertNotIn("tools", body)
        self.assertTrue(body["messages"][0]["content"].startswith("SYS\n\nTOOL CALLING PROTOCOL."))
        self.assertIn("- create_object(doc_name: string): Create an object.", body["messages"][0]["content"])

    def test_a_known_family_stays_native(self):
        self.assertIn("tools", self.native("llama3.1:8b")._body("S", HIST, [TOOL]))


class WireTest(_Case):
    def wire(self, provider, *messages):
        out = []
        provider._body("S", [], [TOOL])
        for m in messages:
            out.extend(provider._to_wire(m))
        return out

    def test_assistant_calls_use_object_arguments_and_results_use_tool_name(self):
        call = ToolCall(id="call_1", name="create_object", arguments={"doc_name": "D"})
        out = self.wire(self.native(),
                        Message(role="assistant", content="ok", tool_calls=[call]),
                        Message(role="user", tool_results=[ToolResult(tool_call_id="call_1", name="create_object", content="created")]))
        self.assertEqual(out[0], {"role": "assistant", "content": "ok",
                                  "tool_calls": [{"function": {"name": "create_object", "arguments": {"doc_name": "D"}}}]})
        self.assertEqual(out[1], {"role": "tool", "tool_name": "create_object", "content": "created"})
        self.assertEqual(len(out), 2)

    def test_undecodable_arguments_are_replayed_as_an_empty_object(self):
        bad = ToolCall(id="c", name="create_object", arguments={"_raw": "{"}, args_error="truncated", raw_args="{")
        out = self.wire(self.native(), Message(role="assistant", tool_calls=[bad]))
        self.assertEqual(out[0]["tool_calls"][0]["function"]["arguments"], {})

    def test_images_travel_in_the_images_list_and_screenshots_follow_a_result(self):
        out = self.wire(self.native(),
                        Message(role="user", content="look", images=["AAA", "BBB"]),
                        Message(role="user", tool_results=[ToolResult(tool_call_id="c", name="get_view", content="ok", image_b64="CCC")]))
        self.assertEqual(out[0], {"role": "user", "content": "look", "images": ["AAA", "BBB"]})
        self.assertEqual(out[1]["role"], "tool")
        self.assertEqual(out[2], {"role": "user", "content": "(screenshot after get_view)", "images": ["CCC"]})
        self.assertNotIn("data:image", json.dumps(out))                       # raw base64, no data-URL prefix

    def test_vision_off_degrades_images_to_notes(self):
        p = self.native()
        p._vision_ok = False
        out = self.wire(p, Message(role="user", content="look", images=["AAA"]),
                        Message(role="user", tool_results=[ToolResult(tool_call_id="c", name="get_view", content="ok", image_b64="CCC")]))
        self.assertNotIn('"images"', json.dumps(out))
        self.assertIn("omitted", out[0]["content"])
        self.assertIn("screenshot omitted", out[1]["content"])

    def test_only_the_newest_images_are_kept(self):
        messages = [Message(role="user", content=f"m{i}", images=[f"IMG{i}"]) for i in range(providers.MAX_WIRE_IMAGES + 3)]
        body = self.native()._body("S", messages, [])
        kept = [m["images"][0] for m in body["messages"] if "images" in m]
        self.assertEqual(kept, [f"IMG{i}" for i in range(3, providers.MAX_WIRE_IMAGES + 3)])
        self.assertIn("earlier screenshot omitted", body["messages"][1]["content"])

    def test_text_mode_history_uses_blocks_and_one_user_message(self):
        p = self.native("tinyllama")
        calls = [ToolCall(id="a", name="create_object", arguments={"doc_name": "D"}), ToolCall(id="b", name="get_view", arguments={})]
        results = [ToolResult(tool_call_id="a", name="create_object", content="made"),
                   ToolResult(tool_call_id="b", name="get_view", content="seen", image_b64="IMG")]
        out = self.wire(p, Message(role="assistant", content="go", tool_calls=calls), Message(role="user", tool_results=results))
        self.assertEqual([m["role"] for m in out], ["assistant", "user"])
        self.assertNotIn("tool_calls", out[0])
        self.assertEqual(out[0]["content"].count("<tool_call>"), 2)
        self.assertEqual(out[1]["content"].count("<tool_result"), 2)
        self.assertEqual(out[1]["images"], ["IMG"])

    def test_thinking_is_replayed_only_when_the_caps_say_so(self):
        message = Message(role="assistant", content="x", provider_blocks=[{"type": "reasoning_content", "text": "deep"}])
        self.assertNotIn("thinking", self.wire(self.native(), message)[0])
        replay = self.native(caps_override={"reasoning_replay": True})
        self.assertEqual(self.wire(replay, message)[0]["thinking"], "deep")


# --------------------------------------------------------------------------------------------------------------


class StreamTest(_Case):
    def test_text_stream_with_usage_and_stop_info(self):
        events = self.parse(self.native(), [msg("Hel"), msg("lo"), done()])
        self.assertEqual(texts(events), "Hello")
        self.assertEqual(payload(events, "usage"), {"prompt_eval_count": 40, "eval_count": 7})
        self.assertEqual(payload(events, "stop_reason"), "stop")
        info = payload(events, "stop_info")
        self.assertEqual((info["reason"], info["usage"], info["complete"]), ("end_turn", {"prompt_tokens": 40, "completion_tokens": 7}, True))
        self.assertEqual(payload(events, "done"), "stop")
        self.assertEqual(kinds(events)[-2:], ["done", "stream_end"])
        self.assertEqual(events[-1], ("stream_end", {"complete": True}))

    def test_thinking_becomes_reasoning_events(self):
        events = self.parse(self.native(), [msg("", thinking="plan "), msg("", thinking="it"), msg("Answer."), done()])
        self.assertEqual("".join(p for k, p in events if k == "reasoning"), "plan it")
        self.assertEqual(texts(events), "Answer.")

    def test_inline_think_in_content_is_split_too(self):
        events = self.parse(self.native(), [msg("<thi"), msg("nk>cube</think>"), msg("1000 mm3"), done()])
        self.assertEqual(("".join(p for k, p in events if k == "reasoning"), texts(events)), ("cube", "1000 mm3"))

    def test_whole_tool_calls_with_object_arguments_and_minted_ids(self):
        calls = [{"function": {"name": "create_object", "arguments": {"doc_name": "D"}}},
                 {"function": {"name": "get_view", "arguments": {}}}]
        events = self.parse(self.native(), [msg("Doing it.", tool_calls=calls), done()], tools=[TOOL, TOOL2])
        got = payload(events, "tool_calls")
        self.assertEqual([(c.name, c.arguments) for c in got], [("create_object", {"doc_name": "D"}), ("get_view", {})])
        self.assertTrue(all(c.id.startswith("call_") and len(c.id) == 17 for c in got))
        self.assertEqual(len({c.id for c in got}), 2)
        self.assertEqual(payload(events, "stop_info")["reason"], "tool_use")
        self.assertNotIn("done", kinds(events))
        progress = [p for k, p in events if k == "tool_progress"]
        self.assertEqual([p["name"] for p in progress], ["create_object", "get_view"])

    def test_a_server_supplied_id_is_kept_and_string_arguments_are_decoded(self):
        calls = [{"id": "call_xyz", "function": {"name": "create_object", "arguments": '{"doc_name": "S"}'}}]
        got = payload(self.parse(self.native(), [msg("", tool_calls=calls), done()]), "tool_calls")
        self.assertEqual((got[0].id, got[0].arguments), ("call_xyz", {"doc_name": "S"}))
        bad = [{"function": {"name": "create_object", "arguments": '{"doc_name": "S'}}]
        got = payload(self.parse(self.native(), [msg("", tool_calls=bad), done()]), "tool_calls")
        self.assertTrue(got[0].args_error)

    def test_nameless_tool_calls_are_dropped(self):
        events = self.parse(self.native(), [msg("x", tool_calls=[{"function": {"arguments": {}}}, "junk"]), done()])
        self.assertNotIn("tool_calls", kinds(events))

    def test_length_is_a_token_limit_stop(self):
        events = self.parse(self.native(), [msg("cut off"), done("length")])
        self.assertEqual((payload(events, "stop_reason"), payload(events, "stop_info")["reason"]), ("length", "max_tokens"))
        self.assertTrue(payload(events, "stop_info")["truncated"])

    def test_a_stream_without_done_is_incomplete(self):
        events = self.parse(self.native(), [msg("partial")])
        self.assertEqual(events[-1], ("stream_end", {"complete": False}))
        self.assertNotIn("done", kinds(events))
        self.assertNotIn("tool_calls", kinds(events))

    def test_an_error_chunk_is_a_terminal_error(self):
        events = self.parse(self.native(), [msg("x"), {"error": "model runner has unexpectedly stopped"}])
        self.assertEqual(kinds(events)[-1], "error")
        self.assertIn("qwen2.5:7b: model runner has unexpectedly stopped", str(payload(events, "error")))

    def test_null_fields_do_not_break_the_parser(self):
        events = self.parse(self.native(), [{"message": None, "done": False}, {"message": {"content": None, "thinking": None, "tool_calls": None}},
                                            msg("ok"), {"done": True, "done_reason": None}])
        self.assertEqual(texts(events), "ok")
        self.assertEqual(payload(events, "stop_reason"), "stop")

    def test_stop_is_polled_between_chunks(self):
        p = self.native()
        p._body("S", HIST, [])
        with self.assertRaises(providers.StopRequested):
            list(p._parse_stream(FakeResponse(nd(msg("a"), msg("b"), done())), lambda: True))

    def test_text_mode_calls_are_extracted_and_the_markup_is_withheld(self):
        p = self.native("tinyllama")
        reply = 'On it.\n<tool_call>\n{"name": "create_object", "arguments": {"doc_name": "T"}}\n</tool_call>'
        pieces = [reply[:12], reply[12:30], reply[30:60], reply[60:]]
        events = self.parse(p, [msg(x) for x in pieces] + [done()])
        self.assertEqual(texts(events).strip(), "On it.")
        self.assertEqual([c.arguments for c in payload(events, "tool_calls")], [{"doc_name": "T"}])

    def test_native_mode_block_only_reply_is_a_call(self):
        reply = '<tool_call>{"name": "get_view", "arguments": {}}</tool_call>'
        events = self.parse(self.native("llama3.1:8b"), [msg(reply), done()], tools=[TOOL, TOOL2])
        self.assertEqual([c.name for c in payload(events, "tool_calls")], ["get_view"])
        self.assertEqual(texts(events), "")

    def test_reasoning_is_offered_for_replay_only_when_enabled(self):
        chunks = [msg("", thinking="deep thought"), msg("a"), done()]
        self.assertNotIn("assistant_blocks", kinds(self.parse(self.native(), chunks)))
        replay = self.parse(self.native(caps_override={"reasoning_replay": True}), chunks)
        self.assertEqual(payload(replay, "assistant_blocks"), [{"type": "reasoning_content", "text": "deep thought"}])
        self.assertLess(kinds(replay).index("assistant_blocks"), kinds(replay).index("stop_reason"))

    def test_a_non_ndjson_answer_is_a_clear_non_retriable_error(self):
        p = self.native()
        p._body("S", HIST, [])
        response = FakeResponse(["<html><body>Bad gateway page</body></html>"])
        with self.assertRaises(provider_stream.ProviderError) as cm:
            list(p._parse_stream(response, lambda: False))
        self.assertIn("something other than an NDJSON stream", str(cm.exception))
        self.assertFalse(cm.exception.retriable)

    def test_a_full_window_is_logged(self):
        p = self.native(caps_override={"context_window": 1000})
        logged = []
        with mock.patch.object(provider_ollama, "log_event", lambda name, **kw: logged.append((name, kw))):
            self.parse(p, [msg("x"), done(prompt_eval_count=990)])
            self.parse(p, [msg("x"), done(prompt_eval_count=200)])
        self.assertEqual([n for n, _ in logged], ["ollama_context_full"])

    def test_ndjson_lines_helper(self):
        got = list(provider_stream.ndjson_lines(FakeResponse(["", '{"a": 1}', "  ", "garbage", '[1]', '{"b": 2}'])))
        self.assertEqual(got, [{"a": 1}, {"b": 2}])


# --------------------------------------------------------------------------------------------------------------


class ChatLoopTest(FaultCase):
    """The whole ``chat()`` generator over a patched ``requests.post``: the call shape, retries, adaptation."""

    def native_provider(self, model="qwen2.5:7b", **extra):
        extra.setdefault("max_tokens", 1024)
        p = params.Profile(name="ol", provider="openai_compat", base_url=URL, model=model, api_key="", **extra)
        self._profiles.append(p)
        forget_learned(p)
        return provider_ollama.OllamaNativeProvider(p)

    GOOD = nd(msg("Hello"), done())

    def test_the_post_goes_to_api_chat_with_the_pinned_keyword_shape(self):
        script = Script(Resp(200, self.GOOD))
        state = self.run_chat(self.native_provider(), script)
        self.assertEqual(state["text"], "Hello")
        call = script.calls[0]
        self.assertEqual((call["url"], call["stream"], call["timeout"]), ("http://localhost:11434/api/chat", True, (15, 600)))
        self.assertEqual(call["json"]["options"]["num_ctx"], 16384)
        self.assertEqual(state["kinds"][-2:], ["stop_info", "done"])

    def test_a_404_names_the_model_problem(self):
        script = Script(Resp(404, text='{"error":"model \\"nope\\" not found, try pulling it first"}'))
        state = self.run_chat(self.native_provider("nope"), script)
        self.assertIn("HTTP 404 not found (check the base URL and the model name)", str(state["error"]))
        self.assertIn("try pulling it first", str(state["error"]))
        self.assertEqual(len(script.calls), 1)

    def test_a_server_that_is_not_running_says_so_after_the_retries(self):
        script = Script(*[requests.exceptions.ConnectionError("refused")] * 5)
        provider = self.native_provider()
        provider.backoff_scale = 0
        state = self.run_chat(provider, script)
        self.assertIn("request failed after 5 attempts", str(state["error"]))
        self.assertIn("ollama serve", str(state["error"]))

    def test_does_not_support_tools_flips_to_text_mode_and_retries_once(self):
        script = Script(Resp(400, text='{"error":"registry.ollama.ai/library/qwen2.5:7b does not support tools"}'),
                        Resp(200, nd(msg('<tool_call>{"name": "create_object", "arguments": {"doc_name": "Z"}}</tool_call>'), done())))
        provider = self.native_provider()
        state = self.run_chat(provider, script, tools=[TOOL])
        self.assertIsNone(state["error"])
        self.assertEqual([c.arguments for c in state["calls"]], [{"doc_name": "Z"}])
        self.assertIn("tools", script.calls[0]["json"])
        self.assertNotIn("tools", script.calls[1]["json"])
        self.assertIn("TOOL CALLING PROTOCOL", script.calls[1]["json"]["messages"][0]["content"])
        self.assertEqual(models.learned_fields(provider.profile)["tools"], "text")          # remembered
        self.assertEqual(state["text"].strip(), "")

    def test_does_not_support_thinking_drops_think_and_stops_asking(self):
        script = Script(Resp(400, text='{"error":"qwen3:8b does not support thinking"}'), Resp(200, self.GOOD))
        provider = self.native_provider("qwen3:8b", caps_override={"reasoning_control": "ollama_effort"})
        state = self.run_chat(provider, script, reasoning="high")
        self.assertEqual(state["text"], "Hello")
        self.assertIs(script.calls[0]["json"]["think"], True)
        self.assertNotIn("think", script.calls[1]["json"])
        self.assertIs(models.learned_fields(provider.profile)["think_ok"], False)

    def test_an_image_the_model_cannot_take_degrades_to_text_and_is_remembered_only_when_unambiguous(self):
        provider = self.native_provider()
        history = [Message(role="user", content="look", images=["AAA"])]
        script = Script(Resp(400, text='{"error":"this model is missing data required for image input"}'), Resp(200, self.GOOD))
        with mock.patch.object(providers.requests, "post", script):
            state = collect(provider.chat("S", history, []))
        self.assertEqual(state["text"], "Hello")
        self.assertIn("images", script.calls[0]["json"]["messages"][1])
        self.assertNotIn('"images"', json.dumps(script.calls[1]["json"]["messages"]))
        self.assertIs(models.learned_fields(provider.profile).get("vision"), False)

        other = self.native_provider("llava:7b")
        script = Script(Resp(400, text='{"error":"invalid image: payload too large"}'), Resp(200, self.GOOD))
        with mock.patch.object(providers.requests, "post", script):
            collect(other.chat("S", history, []))
        self.assertNotIn("vision", models.learned_fields(other.profile))                      # a size problem is not "no vision"
        self.assertFalse(other.vision_enabled)                                                # ... but THIS session degraded

    def test_a_stream_that_drops_is_retried_with_a_reset(self):
        script = Script(Resp(200, nd(msg("par"), msg("tial"))), Resp(200, self.GOOD))
        state = self.run_chat(self.native_provider(), script)
        self.assertEqual((state["text"], state["resets"]), ("Hello", ["stream_dropped"]))


# --------------------------------------------------------------------------------------------------------------


class RealServerTest(_Case):
    """A real HTTP round trip (``requests`` + the reader thread) against a local server speaking NDJSON."""

    def test_events_over_a_real_socket(self):
        body = ("\n".join(nd(msg("Real "), msg("socket", thinking=""), done())) + "\n").encode()
        with ScriptedServer([raw(body, content_type="application/x-ndjson")]) as server:
            provider = provider_ollama.OllamaNativeProvider(self.prof(base_url=server.url + "/v1"))
            events = list(provider.chat("SYS", HIST, [TOOL]))
            request = server.requests[0]
        self.assertEqual(request.path, "/api/chat")
        self.assertEqual(request.json["options"]["num_ctx"], 16384)
        self.assertEqual(request.json["tools"][0]["function"]["name"], "create_object")
        self.assertEqual(texts(events), "Real socket")
        self.assertEqual(kinds(events)[-2:], ["stop_info", "done"])

    def test_a_chunk_split_inside_a_json_line_is_reassembled(self):
        lines = nd(msg("A"), msg("B"), done())
        body = ("\n".join(lines) + "\n").encode()
        with ScriptedServer([raw(body, content_type="application/x-ndjson")]) as server:
            provider = provider_ollama.OllamaNativeProvider(self.prof(base_url=server.url))
            events = list(provider.chat("S", HIST, []))
        self.assertEqual(texts(events), "AB")


if __name__ == "__main__":
    unittest.main()
