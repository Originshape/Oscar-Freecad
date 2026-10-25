"""Streaming parser tests for both provider wire formats.

Feeds recorded SSE bodies through the parsers with a fake response object —
no network involved.
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import models, params, provider_probe, providers  # noqa: E402


def forget_learned(*profiles):
    """Drop everything the providers learned about ``profiles`` (in-process mirror, settings file, adaptation flags).

    Capability learning is persisted per (provider, base url, model); tests that make a provider learn something
    must not leak it into the next test that uses the same fake profile.
    """
    providers._LEARNED.clear()
    for profile in profiles:
        for kind in ("anthropic", "openai_compat", ""):
            subject = type("P", (), {"provider": kind, "base_url": profile.base_url, "model": profile.model})()
            key = models.caps_key(subject)
            models._LEARNED.pop(key, None)
            models._RUNTIME.pop(key, None)         # the window a native Ollama provider published
            models._PROBE_FAILED.pop(key, None)
            params.set_model_caps(key, {})


class FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


class Profile:
    def __init__(self):
        self.base_url = "https://example.test"
        self.model = "test-model"
        self.api_key = "k"
        self.temperature = 0.0
        self.max_tokens = 64


def run_stream(provider, events, should_stop=lambda: False):
    collected = {"text": "", "reasoning": "", "tool_calls": [], "done": None, "error": None,
                 "usage": None}
    for kind, payload in provider._parse_stream(FakeResponse(events), should_stop):
        if kind == "text":
            collected["text"] += payload
        elif kind == "reasoning":
            collected["reasoning"] += payload
        elif kind == "usage":
            collected["usage"] = payload
        elif kind == "tool_calls":
            collected["tool_calls"] = payload
        elif kind == "done":
            collected["done"] = payload
        elif kind == "error":
            collected["error"] = payload
    return collected


class AnthropicStreamTest(unittest.TestCase):
    def setUp(self):
        self.provider = providers.AnthropicProvider(Profile())

    def sse(self, events):
        return ["data: " + json.dumps(e) for e in events] + ["data: [DONE]"]

    def test_text_stream(self):
        events = [
            {"type": "message_start", "message": {}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello "}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "world"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]
        out = run_stream(self.provider, self.sse(events))
        self.assertEqual(out["text"], "Hello world")
        self.assertEqual(out["done"], "end_turn")
        self.assertEqual(out["tool_calls"], [])

    def test_tool_use_after_text_uses_block_index(self):
        # Text block at index 0, tool_use at index 1: input_json_delta events
        # address the tool by stream index 1, not by tool id.
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Building it."}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_01", "name": "create_object"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"doc_name": "M'}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": 'yDoc", "obj_type": "Part::Box"}'}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        ]
        out = run_stream(self.provider, self.sse(events))
        self.assertEqual(out["text"], "Building it.")
        self.assertEqual(len(out["tool_calls"]), 1)
        call = out["tool_calls"][0]
        self.assertEqual(call.id, "toolu_01")
        self.assertEqual(call.name, "create_object")
        self.assertEqual(call.arguments["obj_type"], "Part::Box")
        self.assertEqual(out["done"], None)

    def test_two_tools_in_one_turn(self):
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "list_documents"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t2", "name": "get_view"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"view_name": "Top"}'}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_stop"},
        ]
        out = run_stream(self.provider, self.sse(events))
        self.assertEqual([c.name for c in out["tool_calls"]], ["list_documents", "get_view"])
        self.assertEqual(out["tool_calls"][1].arguments, {"view_name": "Top"})

    def test_usage_is_merged_from_start_and_delta(self):
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 12}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
            {"type": "message_stop"},
        ]
        out = run_stream(self.provider, self.sse(events))
        self.assertEqual(out["text"], "ok")
        self.assertEqual(out["usage"], {"input_tokens": 12, "output_tokens": 3})

    def test_error_event(self):
        events = [{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}]
        out = run_stream(self.provider, self.sse(events))
        self.assertIn("overloaded_error", str(out["error"]))


class OpenAIStreamTest(unittest.TestCase):
    def setUp(self):
        self.provider = providers.OpenAICompatProvider(Profile())

    def test_text_stream(self):
        lines = [
            'data: {"choices":[{"delta":{"role":"assistant","content":"Hi"}}]}',
            'data: {"choices":[{"delta":{"content":" there"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        out = run_stream(self.provider, lines)
        self.assertEqual(out["text"], "Hi there")
        self.assertEqual(out["done"], "stop")

    def test_reasoning_only_length_cutoff_is_reported(self):
        # Recorded from z.ai glm-5.3: the thinking used every completion
        # token, then a final chunk with finish_reason "length", empty
        # content and the usage — no answer text, no tool call. The parser
        # must report that stop reason (the agent turns it into a nudge or
        # a visible error) rather than a plain empty "stop".
        lines = [
            'data: {"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"The user wants"}}]}',
            'data: {"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":" a housing…"}}]}',
            'data: {"choices":[{"index":0,"finish_reason":"length","delta":{"role":"assistant","content":""}}],'
            '"usage":{"prompt_tokens":7925,"completion_tokens":8192,"total_tokens":16117,'
            '"completion_tokens_details":{"reasoning_tokens":8189}}}',
            "data: [DONE]",
        ]
        out = run_stream(self.provider, lines)
        self.assertEqual(out["text"], "")
        self.assertEqual(out["reasoning"], "The user wants a housing…")
        self.assertEqual(out["tool_calls"], [])
        self.assertEqual(out["done"], "length")
        self.assertEqual(out["usage"]["completion_tokens_details"]["reasoning_tokens"], 8189)

    def test_in_stream_error_payload_is_an_error(self):
        # After the 200 is sent, gateways deliver failures inside the SSE
        # stream; that must not end the turn as a clean empty answer.
        lines = [
            'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}',
            'data: {"error":{"message":"upstream timeout","code":"1305"}}',
            "data: [DONE]",
        ]
        out = run_stream(self.provider, lines)
        self.assertIsNotNone(out["error"])
        self.assertIn("upstream timeout", str(out["error"]))
        self.assertEqual(out["text"], "")
        self.assertIsNone(out["done"])

    def test_nameless_tool_call_yields_no_call_but_keeps_the_stop_reason(self):
        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_x","function":{"arguments":"{}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
        out = run_stream(self.provider, lines)
        self.assertEqual(out["tool_calls"], [])
        self.assertEqual(out["done"], "tool_calls")  # the agent reports it

    def test_usage_only_final_chunk_is_captured(self):
        # OpenAI (stream_options include_usage): a trailing chunk with no
        # choices carries the usage.
        lines = [
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":1}}',
            "data: [DONE]",
        ]
        out = run_stream(self.provider, lines)
        self.assertEqual(out["text"], "Hi")
        self.assertEqual(out["done"], "stop")
        self.assertEqual(out["usage"], {"prompt_tokens": 10, "completion_tokens": 1})

    def test_parallel_tool_calls_assemble_across_chunks(self):
        # OpenAI streams tool call arguments as fragments keyed by index;
        # ids and names can also arrive split across chunks.
        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"create_","arguments":""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"object","arguments":"{\\"doc_name\\":"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"D\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"get_view","arguments":"{\\"view_name\\":\\"Top\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
        out = run_stream(self.provider, lines)
        self.assertEqual(out["done"], None)
        self.assertEqual(len(out["tool_calls"]), 2)
        first, second = out["tool_calls"]
        self.assertEqual((first.id, first.name), ("call_a", "create_object"))
        self.assertEqual(first.arguments, {"doc_name": "D"})
        self.assertEqual((second.id, second.name), ("call_b", "get_view"))
        self.assertEqual(second.arguments, {"view_name": "Top"})


class WireFormatTest(unittest.TestCase):
    """Message -> wire conversion for both providers."""

    def setUp(self):
        providers._LEARNED.clear()  # no vision downgrade learned by another test

    def test_anthropic_wire(self):
        provider = providers.AnthropicProvider(Profile())
        messages = [
            providers.Message(role="user", content="build a box"),
            providers.Message(
                role="assistant",
                content="On it",
                tool_calls=[providers.ToolCall(id="t1", name="create_object", arguments={"obj_type": "Part::Box"})],
            ),
            providers.Message(
                role="user",
                tool_results=[
                    providers.ToolResult(
                        tool_call_id="t1", name="create_object",
                        content="Object 'Box' created.", image_b64="QUJD",
                    )
                ],
            ),
        ]
        body = provider._body("system prompt", messages, [])
        wire = body["messages"]
        self.assertEqual(body["system"], "system prompt")
        self.assertEqual(wire[0], {"role": "user", "content": [{"type": "text", "text": "build a box"}]})
        assistant_blocks = wire[1]["content"]
        self.assertEqual(assistant_blocks[0], {"type": "text", "text": "On it"})
        self.assertEqual(assistant_blocks[1]["type"], "tool_use")
        result_blocks = wire[2]["content"]
        self.assertEqual(result_blocks[0]["type"], "tool_result")
        # the screenshot rides inside the tool_result as a vision block
        self.assertEqual(result_blocks[0]["content"][1]["source"]["data"], "QUJD")

    def test_openai_wire(self):
        provider = providers.OpenAICompatProvider(Profile())
        messages = [
            providers.Message(role="user", content="build a box"),
            providers.Message(
                role="assistant",
                tool_calls=[providers.ToolCall(id="t1", name="create_object", arguments={"obj_type": "Part::Box"})],
            ),
            providers.Message(
                role="user",
                tool_results=[
                    providers.ToolResult(
                        tool_call_id="t1", name="create_object",
                        content="Object 'Box' created.", image_b64="QUJD",
                    )
                ],
            ),
        ]
        body = provider._body("system prompt", messages, [])
        wire = body["messages"]
        self.assertEqual(wire[0]["role"], "system")
        self.assertEqual(wire[2]["tool_calls"][0]["function"]["name"], "create_object")
        tool_msg = wire[3]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], "t1")
        # tool messages cannot carry images; a follow-up user turn delivers it
        image_turn = wire[4]
        self.assertEqual(image_turn["role"], "user")
        self.assertIn("data:image/png;base64,QUJD", image_turn["content"][1]["image_url"]["url"])


class VisionFallbackTest(unittest.TestCase):
    """Text-only models reject image content blocks (HTTP 400) — the
    provider must detect that, retry text-only, and annotate what was
    omitted instead of failing the conversation."""

    def setUp(self):
        forget_learned(Profile())

    def tearDown(self):
        forget_learned(Profile())

    def test_detection_patterns(self):
        self.assertTrue(providers._looks_like_image_rejection(
            '{"error":{"code":"1210","message":"messages.content.type is invalid, '
            "allowed values: ['text']}}"))
        self.assertTrue(providers._looks_like_image_rejection(
            "image content is not supported by this model"))
        self.assertFalse(providers._looks_like_image_rejection("invalid model id"))

    def test_openai_wire_drops_images_when_disabled(self):
        provider = providers.OpenAICompatProvider(Profile())
        provider._vision_ok = False
        wire = provider._to_wire(providers.Message(role="user", content="look", images=["QUJD"]))
        self.assertEqual(wire[0]["role"], "user")
        self.assertIn("omitted", wire[0]["content"])
        self.assertNotIn("image_url", json.dumps(wire))
        result_wire = provider._to_wire(providers.Message(role="user", tool_results=[
            providers.ToolResult(tool_call_id="t1", name="x", content="done", image_b64="QUJD")]))
        self.assertEqual(len(result_wire), 1)  # no follow-up image turn
        self.assertIn("screenshot omitted", result_wire[0]["content"])

    def test_anthropic_wire_drops_images_when_disabled(self):
        provider = providers.AnthropicProvider(Profile())
        provider._vision_ok = False
        wire = provider._to_wire(providers.Message(role="user", content="look", images=["QUJD"]))
        blocks = wire["content"]
        self.assertTrue(all(b["type"] == "text" for b in blocks))
        self.assertTrue(any("omitted" in b["text"] for b in blocks))
        rwire = provider._to_wire(providers.Message(role="user", tool_results=[
            providers.ToolResult(tool_call_id="t1", name="x", content="done", image_b64="QUJD")]))
        rblocks = rwire["content"][0]["content"]
        self.assertTrue(all(b["type"] == "text" for b in rblocks))
        self.assertIn("screenshot omitted", rblocks[0]["text"])

    def test_chat_retries_without_images_on_400(self):
        from unittest import mock

        provider = providers.OpenAICompatProvider(Profile())
        messages = [providers.Message(role="user", content="hi", images=["QUJD"])]

        class R400:
            status_code = 400
            text = ('{"error":{"code":"1210","message":"messages.content.type is '
                    "invalid, allowed values: ['text']}}\"")

        ok = FakeResponse([
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ])
        ok.status_code = 200

        bodies = []

        def fake_post(url, headers=None, json=None, stream=True, timeout=None):
            bodies.append(json)
            return R400() if len(bodies) == 1 else ok

        with mock.patch.object(providers.requests, "post", side_effect=fake_post):
            events = list(provider.chat("sys", messages, []))

        self.assertFalse(provider._vision_ok)
        self.assertEqual(len(bodies), 2)  # rejected with images, then clean retry
        self.assertIn("image_url", json.dumps(bodies[0]))
        self.assertNotIn("image_url", json.dumps(bodies[1]))
        self.assertEqual([p for kind, p in events if kind == "text"], ["Hi"])
        self.assertEqual([p for kind, p in events if kind == "error"], [])


class ReasoningStreamTest(unittest.TestCase):
    """Reasoning models stream thinking separately from the answer."""

    def test_openai_reasoning_content(self):
        provider = providers.OpenAICompatProvider(Profile())
        lines = [
            'data: {"choices":[{"delta":{"reasoning_content":"think A"}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"…think B"}}]}',
            'data: {"choices":[{"delta":{"content":"Answer"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        out = run_stream(provider, lines)
        self.assertEqual(out["text"], "Answer")
        self.assertEqual(out["reasoning"], "think A…think B")

    def test_anthropic_thinking_delta(self):
        provider = providers.AnthropicProvider(Profile())
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "pondering "}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "deeply"}},
            {"type": "content_block_stop"},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hi"}},
            {"type": "content_block_stop"},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]
        lines = ["data: " + json.dumps(e) for e in events] + ["data: [DONE]"]
        out = run_stream(provider, lines)
        self.assertEqual(out["reasoning"], "pondering deeply")
        self.assertEqual(out["text"], "Hi")


# ---------------------------------------------------------------------------------------------------------------
# Decoder, parsers, stop_info, mapper, request bodies.


def sse(events, done=False):
    """``data:`` lines for a list of JSON events (and an optional ``[DONE]``)."""
    lines = ["data: " + json.dumps(e) for e in events]
    return lines + ["data: [DONE]"] if done else lines


def parse(provider, lines):
    """Every event the provider's parser yields for ``lines`` (including the internal ``stream_end``)."""
    return list(provider._parse_stream(FakeResponse(lines), lambda: False))


def kinds(events):
    return [k for k, _ in events]


def payload(events, kind):
    return next(p for k, p in events if k == kind)


def anthropic_reply(*, blocks=(), stop="end_turn", start_usage=None, delta_usage=None, terminal=True):
    """A complete Anthropic Messages stream: ``blocks`` are ("text", str) / ("tool", id, name, json) / ("thinking", text, sig)."""
    events = [{"type": "message_start", "message": {"usage": start_usage or {}}}]
    for index, block in enumerate(blocks):
        if block[0] == "text":
            events += [{"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
                       {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": block[1]}}]
        elif block[0] == "thinking":
            events += [{"type": "content_block_start", "index": index, "content_block": {"type": "thinking", "thinking": ""}},
                       {"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": block[1]}},
                       {"type": "content_block_delta", "index": index, "delta": {"type": "signature_delta", "signature": block[2]}}]
        else:
            events += [{"type": "content_block_start", "index": index,
                        "content_block": {"type": "tool_use", "id": block[1], "name": block[2]}},
                       {"type": "content_block_delta", "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": block[3]}}]
        events.append({"type": "content_block_stop", "index": index})
    events.append({"type": "message_delta", "delta": {"stop_reason": stop}, "usage": delta_usage or {}})
    if terminal:
        events.append({"type": "message_stop"})
    return sse(events)


def openai_reply(*, text="", calls=(), finish="stop", usage=None, terminal="done"):
    """A complete OpenAI chat stream; ``calls`` are (id, name, arguments-text). ``terminal``: "done" | "finish" | None."""
    lines = []
    if text:
        lines.append({"choices": [{"delta": {"content": text}}]})
    for index, (call_id, name, args) in enumerate(calls):
        lines.append({"choices": [{"delta": {"tool_calls": [
            {"index": index, "id": call_id, "function": {"name": name, "arguments": args}}]}}]})
    if terminal in ("done", "finish"):
        lines.append({"choices": [{"delta": {}, "finish_reason": finish}]})
    if usage:
        lines.append({"choices": [], "usage": usage})
    out = sse(lines)
    return out + ["data: [DONE]"] if terminal == "done" else out


class _ProviderCase(unittest.TestCase):
    """Base: forgets everything the providers learn about the profiles a test builds."""

    def setUp(self):
        self._profiles = []
        self.addCleanup(lambda: forget_learned(*self._profiles))

    def profile(self, provider, base_url, model, **extra):
        p = params.Profile(name="t", provider=provider, base_url=base_url, model=model, api_key="k", **extra)
        self._profiles.append(p)
        forget_learned(p)
        return p

    def anthropic(self, model="claude-sonnet-4-5", **extra):
        extra.setdefault("max_tokens", 8192)
        return providers.AnthropicProvider(self.profile("anthropic", "https://api.anthropic.com", model, **extra))

    def openai(self, model="gpt-5", base_url="https://api.openai.com/v1", **extra):
        extra.setdefault("max_tokens", 4096)
        return providers.OpenAICompatProvider(self.profile("openai_compat", base_url, model, **extra))


TOOL = providers.ToolSpec("create_object", "Create an object. More text.", {"type": "object", "properties": {}})
TOOL2 = providers.ToolSpec("get_view", "Get a view.", {"type": "object", "properties": {}})


class DecodeToolArgumentsTest(unittest.TestCase):
    def decode(self, raw, stop=""):
        return providers.decode_tool_arguments(raw, stop_reason=stop)

    def test_valid_empty_and_whitespace(self):
        self.assertEqual(self.decode('{"a": 1}'), ({"a": 1}, ""))
        self.assertEqual(self.decode(""), ({}, ""))
        self.assertEqual(self.decode("  \n\t "), ({}, ""))
        self.assertEqual(self.decode(None), ({}, ""))
        self.assertEqual(self.decode({"already": "a dict"}), ({"already": "a dict"}, ""))

    def test_byte_order_mark_is_stripped(self):
        self.assertEqual(self.decode('﻿{"a": 1}'), ({"a": 1}, ""))

    def test_raw_newlines_and_tabs_inside_strings_are_accepted(self):
        value, error = self.decode('{"code": "line1\nline2\tx"}')
        self.assertEqual((value, error), ({"code": "line1\nline2\tx"}, ""))

    def test_trailing_commas_are_dropped(self):
        self.assertEqual(self.decode('{"a": [1, 2,], "b": 2,}'), ({"a": [1, 2], "b": 2}, ""))
        self.assertEqual(self.decode('{"a": 1 ,\n}'), ({"a": 1}, ""))

    def test_commas_inside_strings_are_never_touched(self):
        # the script that runs must be byte-identical to the one the model sent
        code = 'x = [1,]\nprint("a,}")'
        raw = json.dumps({"code": code})[:-1] + ",}"        # one stray comma, outside the string
        value, error = self.decode(raw)
        self.assertEqual(error, "")
        self.assertEqual(value["code"], code)

    def test_markdown_fences_are_stripped(self):
        self.assertEqual(self.decode('```json\n{"a": 1}\n```'), ({"a": 1}, ""))
        self.assertEqual(self.decode('```\n{"a": 1}```'), ({"a": 1}, ""))

    def test_double_encoded_object_is_decoded_once_more(self):
        self.assertEqual(self.decode(json.dumps(json.dumps({"a": 1}))), ({"a": 1}, ""))

    def test_a_python_dict_literal_is_accepted(self):
        self.assertEqual(self.decode("{'a': 1, 'b': [True, None]}"), ({"a": 1, "b": [True, None]}, ""))

    def test_the_result_must_be_an_object(self):
        for raw in ("[1, 2]", "42", "true", "null"):
            with self.subTest(raw=raw):
                value, error = self.decode(raw)
                self.assertEqual(value, {"_raw": raw})
                self.assertIn("expected a JSON object", error)
        value, error = self.decode('"just text"')       # a string that is not itself an encoded object
        self.assertEqual(value, {"_raw": '"just text"'})
        self.assertTrue(error.startswith("malformed JSON"), error)

    def test_truncation_is_reported_as_such(self):
        raw = '{"code": "import Part\\nbox = Part.makeBox(1,1'
        value, error = self.decode(raw, stop="max_tokens")
        self.assertEqual(value, {"_raw": raw})
        self.assertTrue(error.startswith("truncated (output cut off at the token limit) at char "), error)
        self.assertIn(f"of {len(raw)}; tail ", error)
        # unbalanced braces mean truncation even when the stop reason is a plain end
        _, error = self.decode(raw, stop="end_turn")
        self.assertTrue(error.startswith("truncated"), error)
        # and the raw provider spelling "length" normalizes to max_tokens
        _, error = self.decode('{"a": [1, 2', stop="length")
        self.assertTrue(error.startswith("truncated"), error)

    def test_balanced_but_invalid_json_is_malformed(self):
        value, error = self.decode('{"a": 1 "b": 2}')
        self.assertEqual(value, {"_raw": '{"a": 1 "b": 2}'})
        self.assertTrue(error.startswith("malformed JSON ("), error)
        self.assertIn("at char", error)
        self.assertIn("tail '", error)

    def test_valid_json_with_a_token_limit_stop_still_decodes(self):
        self.assertEqual(self.decode('{"a": 1}', stop="max_tokens"), ({"a": 1}, ""))

    def test_tail_is_the_last_sixty_characters(self):
        raw = '{"v": "' + "x" * 200 + "END"
        _, error = self.decode(raw, stop="max_tokens")
        tail = error.split("; tail ", 1)[1]
        self.assertEqual(tail, repr(raw[-60:]))
        self.assertTrue(tail.endswith("END'"))

    def test_nothing_in_here_raises_on_garbage(self):
        for raw in ("{", "}", "{'", "\x00\x01", "{" * 5000, "[" * 5000, "'" * 3, "{'a': ", "```", '"'):
            with self.subTest(raw=raw[:10]):
                value, error = self.decode(raw)
                self.assertIsInstance(value, dict)
                self.assertTrue(error or value == {})

    def test_a_huge_python_literal_is_not_evaluated(self):
        raw = "{'k': '" + "a" * 1_100_000 + "'}"
        value, error = self.decode(raw)
        self.assertEqual(value, {"_raw": raw})
        self.assertTrue(error)


class ParsersCarryArgsErrorTest(_ProviderCase):
    """All three stream parsers decode through ``decode_tool_arguments`` and report ``args_error`` / ``raw_args``."""

    def test_anthropic_truncated_call(self):
        raw = '{"code": "import Part\\nprint(1'
        lines = anthropic_reply(blocks=[("tool", "t1", "execute_code", raw)], stop="max_tokens")
        events = parse(self.anthropic(), lines)
        call = payload(events, "tool_calls")[0]
        self.assertEqual(call.arguments, {"_raw": raw})
        self.assertTrue(call.args_error.startswith("truncated"), call.args_error)
        self.assertEqual(call.raw_args, raw)
        self.assertEqual(payload(events, "stop_info")["reason"], "max_tokens")

    def test_anthropic_repaired_call_has_no_error(self):
        lines = anthropic_reply(blocks=[("tool", "t1", "create_object", '{"a": 1,}')], stop="tool_use")
        call = payload(parse(self.anthropic(), lines), "tool_calls")[0]
        self.assertEqual((call.arguments, call.args_error, call.raw_args), ({"a": 1}, "", ""))

    def test_anthropic_empty_arguments_are_an_empty_dict(self):
        lines = anthropic_reply(blocks=[("tool", "t1", "list_documents", "")], stop="tool_use")
        call = payload(parse(self.anthropic(), lines), "tool_calls")[0]
        self.assertEqual((call.arguments, call.args_error), ({}, ""))

    def test_openai_malformed_and_length(self):
        lines = openai_reply(calls=[("c1", "create_object", '{"a": 1 "b"')], finish="length")
        call = payload(parse(self.openai(), lines), "tool_calls")[0]
        self.assertTrue(call.args_error.startswith("truncated"), call.args_error)
        lines = openai_reply(calls=[("c1", "create_object", '{"a": 1 "b": 2}')], finish="tool_calls")
        call = payload(parse(self.openai(), lines), "tool_calls")[0]
        self.assertTrue(call.args_error.startswith("malformed JSON"), call.args_error)
        self.assertEqual(call.arguments, {"_raw": '{"a": 1 "b": 2}'})

    def test_openai_dict_valued_arguments_are_accepted(self):
        lines = sse([{"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c", "function": {"name": "x", "arguments": {"a": 1}}}]}}]},
                     {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}], done=True)
        self.assertEqual(payload(parse(self.openai(), lines), "tool_calls")[0].arguments, {"a": 1})


    def test_raw_args_is_clipped(self):
        raw = '{"v": "' + "y" * 50_000
        lines = anthropic_reply(blocks=[("tool", "t1", "x", raw)], stop="max_tokens")
        call = payload(parse(self.anthropic(), lines), "tool_calls")[0]
        self.assertEqual(len(call.raw_args), 20_000)
        self.assertEqual(call.arguments, {"_raw": raw})   # the legacy marker keeps the full text

    def test_failed_calls_replay_as_empty_arguments(self):
        bad = providers.ToolCall(id="b1", name="x", arguments={"_raw": "{"}, args_error="truncated", raw_args="{")
        msg = providers.Message(role="assistant", tool_calls=[bad])
        self.assertEqual(self.anthropic()._to_wire(msg)["content"][0]["input"], {})
        self.assertEqual(json.loads(self.openai()._to_wire(msg)[0]["tool_calls"][0]["function"]["arguments"]), {})
        legacy = providers.ToolCall(id="b2", name="x", arguments={"_raw": "{"})   # an old consumer's marker alone
        msg = providers.Message(role="assistant", tool_calls=[legacy])
        self.assertEqual(self.anthropic()._to_wire(msg)["content"][0]["input"], {})


class StopInfoTest(_ProviderCase):
    def test_anthropic_usage_is_raw_and_stop_info_is_normalized(self):
        start = {"input_tokens": 12, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000}
        lines = anthropic_reply(blocks=[("text", "ok")], start_usage=start, delta_usage={"output_tokens": 3})
        events = parse(self.anthropic(), lines)
        self.assertEqual(payload(events, "usage"), {**start, "output_tokens": 3})   # raw, never augmented
        info = payload(events, "stop_info")
        self.assertEqual(info, {"reason": "end_turn", "raw": "end_turn", "truncated": False, "complete": True,
                                "usage": {"prompt_tokens": 1112, "completion_tokens": 3}})
        self.assertEqual(kinds(events)[-5:], ["usage", "stop_reason", "stop_info", "done", "stream_end"])

    def test_anthropic_reasons(self):
        cases = {"end_turn": "end_turn", "stop_sequence": "end_turn", "max_tokens": "max_tokens",
                 "refusal": "refusal", "pause_turn": "end_turn"}
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                info = payload(parse(self.anthropic(), anthropic_reply(blocks=[("text", "x")], stop=raw)), "stop_info")
                self.assertEqual((info["reason"], info["raw"], info["truncated"]), (expected, raw, expected == "max_tokens"))

    def test_anthropic_tool_use(self):
        lines = anthropic_reply(blocks=[("tool", "t1", "x", "{}")], stop="tool_use")
        events = parse(self.anthropic(), lines)
        self.assertEqual(payload(events, "stop_info")["reason"], "tool_use")
        self.assertEqual(payload(events, "stop_reason"), "tool_use")           # raw
        self.assertEqual(kinds(events)[-4:], ["stop_reason", "stop_info", "tool_calls", "stream_end"])

    def test_no_usage_gives_empty_normalized_usage_and_no_usage_event(self):
        events = parse(self.anthropic(), anthropic_reply(blocks=[("text", "x")]))
        self.assertNotIn("usage", kinds(events))
        self.assertEqual(payload(events, "stop_info")["usage"], {})

    def test_openai_stop_info(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        events = parse(self.openai(), openai_reply(text="hi", usage=usage))
        self.assertEqual(payload(events, "usage"), usage)
        self.assertEqual(payload(events, "stop_info"), {
            "reason": "end_turn", "raw": "stop", "truncated": False, "complete": True,
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}})
        self.assertEqual(kinds(events)[-4:], ["stop_reason", "stop_info", "done", "stream_end"])

    def test_openai_length_is_truncated_and_tool_calls_with_a_plain_stop_is_tool_use(self):
        info = payload(parse(self.openai(), openai_reply(text="x", finish="length")), "stop_info")
        self.assertEqual((info["reason"], info["truncated"]), ("max_tokens", True))
        info = payload(parse(self.openai(), openai_reply(calls=[("c", "x", "{}")], finish="stop")), "stop_info")
        self.assertEqual(info["reason"], "tool_use")     # some servers say "stop" even with calls
        info = payload(parse(self.openai(), openai_reply(text="x", finish="content_filter")), "stop_info")
        self.assertEqual(info["reason"], "content_filter")

    def test_ollama_style_usage_names_are_normalized(self):
        usage = {"prompt_eval_count": 33, "eval_count": 7}
        info = payload(parse(self.openai(), openai_reply(text="x", usage=usage)), "stop_info")
        self.assertEqual(info["usage"], {"prompt_tokens": 33, "completion_tokens": 7})




class TerminalMarkerTest(_ProviderCase):
    """Without the protocol's terminal marker the parser must not claim a finished reply."""

    def assertIncomplete(self, events):
        self.assertEqual(events[-1], ("stream_end", {"complete": False}))
        for kind in ("done", "tool_calls", "stop_info", "stop_reason", "usage"):
            self.assertNotIn(kind, kinds(events))

    def test_anthropic_without_message_stop(self):
        events = parse(self.anthropic(), anthropic_reply(blocks=[("tool", "t", "x", "{}")], terminal=False))
        self.assertIncomplete(events)
        events = parse(self.anthropic(), [])
        self.assertIncomplete(events)

    def test_openai_without_done_or_finish_reason(self):
        events = parse(self.openai(), openai_reply(text="hi", terminal=None))
        self.assertIncomplete(events)
        self.assertIn("text", kinds(events))        # the partial text still streamed

    def test_openai_finish_reason_alone_is_complete(self):
        events = parse(self.openai(), openai_reply(text="hi", terminal="finish"))
        self.assertEqual(events[-1], ("stream_end", {"complete": True}))
        self.assertEqual(payload(events, "done"), "stop")

    def test_openai_done_without_finish_reason_is_complete(self):
        events = parse(self.openai(), ["data: " + json.dumps({"choices": [{"delta": {"content": "x"}}]}), "data: [DONE]"])
        self.assertEqual(events[-1], ("stream_end", {"complete": True}))



    def test_error_events_are_exceptions_never_dicts(self):
        for provider, lines in (
            (self.anthropic(), sse([{"type": "error", "error": {"type": "api_error", "message": "x"}}])),
            (self.openai(), sse([{"error": {"message": "x", "code": "1"}}])),
        ):
            with self.subTest(provider=provider.name):
                events = parse(provider, lines)
                self.assertEqual(kinds(events), ["error"])
                self.assertIsInstance(events[0][1], Exception)

    def test_in_stream_overload_is_retriable_and_quota_is_not(self):
        err = payload(parse(self.anthropic(), sse([{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}])), "error")
        self.assertTrue(err.retriable)
        err = payload(parse(self.openai(), sse([{"error": {"message": "You exceeded your quota", "type": "insufficient_quota"}}])), "error")
        self.assertFalse(err.retriable)
        err = payload(parse(self.openai(), sse([{"error": {"message": "slow down", "code": "rate_limit_exceeded"}}])), "error")
        self.assertTrue(err.retriable)
        err = payload(parse(self.anthropic(), sse([{"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}])), "error")
        self.assertFalse(err.retriable)


class ToolProgressTest(_ProviderCase):
    def test_anthropic_progress_starts_at_zero_and_ends_with_the_final_size(self):
        args = '{"code": "' + "x" * 50 + '"}'
        lines = anthropic_reply(blocks=[("tool", "t1", "execute_code", args)], stop="tool_use")
        events = [p for k, p in parse(self.anthropic(), lines) if k == "tool_progress"]
        self.assertEqual(events[0], {"name": "execute_code", "chars": 0, "index": 0})
        self.assertEqual(events[-1], {"name": "execute_code", "chars": len(args), "index": 0})
        self.assertEqual(len([e for e in events if e["chars"] == len(args)]), 1)   # one final event, not one per sweep

    def test_progress_is_throttled_per_index(self):
        from oscar.provider_stream import ToolProgress

        tp = ToolProgress(interval=0.5)
        with mock.patch("oscar.provider_stream.time.monotonic", side_effect=[0.0, 0.1, 0.3, 0.6, 0.7]):
            self.assertIsNotNone(tp.update(0, "t", 1))      # t=0.0
            self.assertIsNone(tp.update(0, "t", 2))         # 0.1
            self.assertIsNotNone(tp.update(1, "u", 1))      # 0.3: another index is not throttled by index 0
            self.assertIsNotNone(tp.update(0, "t", 3))      # 0.6 >= 0.0 + 0.5
            self.assertIsNone(tp.update(1, "u", 2))         # 0.7 < 0.3 + 0.5
        self.assertEqual(tp.final(0, "t", 9), ("tool_progress", {"name": "t", "chars": 9, "index": 0}))
        self.assertIsNone(tp.final(0, "t", 9))              # once per index

    def test_openai_emits_progress(self):
        lines = openai_reply(calls=[("c1", "create_object", '{"a": 1}')], finish="tool_calls")
        events = [p for k, p in parse(self.openai(), lines) if k == "tool_progress"]
        self.assertTrue(events and events[-1]["chars"] == len('{"a": 1}'))


class SseLinesTest(unittest.TestCase):
    class Resp:
        def __init__(self, lines):
            self._lines = lines
            self.encoding = None

        def iter_lines(self, decode_unicode=True):
            return iter(self._lines)

    def test_encoding_is_forced_to_utf8(self):
        r = self.Resp(["data: x"])
        list(providers._sse_lines(r))
        self.assertEqual(r.encoding, "utf-8")

    def test_bytes_are_decoded_as_utf8(self):
        r = self.Resp(["data: Größe 3 mm — ✓".encode("utf-8")])
        self.assertEqual(list(providers._sse_lines(r)), ["Größe 3 mm — ✓"])

    def test_invalid_bytes_do_not_raise(self):
        r = self.Resp([b"data: ok\xff"])
        self.assertEqual(list(providers._sse_lines(r)), ["ok�"])

    def test_non_data_lines_and_blank_lines_are_skipped(self):
        r = self.Resp(["", "event: ping", "id: 4", "retry: 10", "data:{\"a\":1}", "data: [DONE]"])
        self.assertEqual(list(providers._sse_lines(r)), ['{"a":1}', "[DONE]"])

    def test_comment_lines_are_liveness_not_data(self):
        from oscar.provider_stream import LIVE

        beats = []
        LIVE.callback = lambda: beats.append(1)
        try:
            r = self.Resp([": keepalive", "data: a", ": ping"])
            self.assertEqual(list(providers._sse_lines(r)), ["a"])
        finally:
            LIVE.callback = None
        self.assertEqual(len(beats), 2)

    def test_a_response_that_forbids_setting_encoding_is_fine(self):
        class Slots:
            __slots__ = ("lines",)

            def iter_lines(self, decode_unicode=True):
                return iter(["data: x"])

        self.assertEqual(list(providers._sse_lines(Slots())), ["x"])

    def test_mojibake_regression_non_ascii_text_survives_a_real_stream(self):
        # a text/event-stream without a charset is decoded as ISO-8859-1 by requests unless the parser forces UTF-8
        provider = providers.OpenAICompatProvider(Profile())
        r = self.Resp([("data: " + json.dumps({"choices": [{"delta": {"content": "Ø 5 mm ✓"}}]}, ensure_ascii=False)).encode("utf-8"),
                       b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}', b"data: [DONE]"])
        out = run_stream(provider, r._lines)
        self.assertEqual(out["text"], "Ø 5 mm ✓")


class ReasoningMapperTest(unittest.TestCase):
    """``Profile.reasoning`` / per-call effort -> request fragment, table by table."""

    def frag(self, control, effort, **kw):
        return providers_reasoning.reasoning_fragment(control, effort, **kw)

    def test_normalize_effort(self):
        n = providers_reasoning.normalize_effort
        self.assertEqual([n(v) for v in ("off", "LOW", " medium ", "high")], ["off", "low", "medium", "high"])
        for value in ("auto", None, "", "extreme", 3, [], "Auto"):
            self.assertIsNone(n(value))

    def test_nothing_without_an_effort_or_a_control(self):
        self.assertEqual(self.frag("openai_effort", None), {})
        self.assertEqual(self.frag("openai_effort", "auto"), {})
        self.assertEqual(self.frag("none", "high"), {})
        self.assertEqual(self.frag("", "high"), {})
        self.assertEqual(self.frag("made_up", "high"), {})

    def test_anthropic_budget(self):
        self.assertEqual(self.frag("anthropic_budget", "off", max_tokens=8192), {})
        for effort, budget in (("low", 1024), ("medium", 4096)):
            self.assertEqual(self.frag("anthropic_budget", effort, max_tokens=8192),
                             {"thinking": {"type": "enabled", "budget_tokens": budget}})
        self.assertEqual(self.frag("anthropic_budget", "high", max_tokens=64000)["thinking"]["budget_tokens"], 16000)

    def test_anthropic_budget_stays_below_max_tokens(self):
        # budget = min(table, max_tokens - 1024), never below 1024, nothing at all when max_tokens <= 2048
        self.assertEqual(self.frag("anthropic_budget", "high", max_tokens=8192)["thinking"]["budget_tokens"], 7168)
        self.assertEqual(self.frag("anthropic_budget", "medium", max_tokens=3000)["thinking"]["budget_tokens"], 1976)
        self.assertEqual(self.frag("anthropic_budget", "low", max_tokens=2049)["thinking"]["budget_tokens"], 1024)
        self.assertEqual(self.frag("anthropic_budget", "low", max_tokens=2048), {})
        self.assertEqual(self.frag("anthropic_budget", "high", max_tokens=0), {})
        self.assertEqual(providers_reasoning.anthropic_budget("high", 4096), 3072)

    def test_openai_effort_and_the_off_spelling(self):
        self.assertEqual(self.frag("openai_effort", "high"), {"reasoning_effort": "high"})
        self.assertEqual(self.frag("openai_effort", "low", model="o3"), {"reasoning_effort": "low"})
        self.assertEqual(self.frag("openai_effort", "off", model="gpt-5"), {"reasoning_effort": "minimal"})
        self.assertEqual(self.frag("openai_effort", "off", model="openai/gpt-5-mini"), {"reasoning_effort": "minimal"})
        self.assertEqual(self.frag("openai_effort", "off", model="o4-mini"), {"reasoning_effort": "low"})


    def test_zai_thinking(self):
        self.assertEqual(self.frag("zai_thinking", "off"), {"thinking": {"type": "disabled"}})
        self.assertEqual(self.frag("zai_thinking", "low"), {"thinking": {"type": "disabled"}})
        self.assertEqual(self.frag("zai_thinking", "medium"), {"thinking": {"type": "enabled"}})
        self.assertEqual(self.frag("zai_thinking", "high"), {"thinking": {"type": "enabled"}})

    def test_ollama_and_gemini_efforts(self):
        for control in ("ollama_effort", "gemini_effort"):
            self.assertEqual(self.frag(control, "off"), {"reasoning_effort": "none"})
            self.assertEqual(self.frag(control, "low"), {"reasoning_effort": "low"})
            self.assertEqual(self.frag(control, "high"), {"reasoning_effort": "high"})

    def test_qwen_field_placement(self):
        self.assertEqual(self.frag("qwen_enable_thinking", "off", base_url="https://dashscope-intl.aliyuncs.com/v1"),
                         {"enable_thinking": False})
        self.assertEqual(self.frag("qwen_enable_thinking", "high", base_url="https://dashscope-intl.aliyuncs.com/v1"),
                         {"enable_thinking": True})
        self.assertEqual(self.frag("qwen_enable_thinking", "low", base_url="http://localhost:8000/v1"),
                         {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(self.frag("qwen_enable_thinking", "medium", base_url="http://localhost:8000/v1"),
                         {"chat_template_kwargs": {"enable_thinking": True}})

    def test_openrouter_and_xai(self):
        self.assertEqual(self.frag("openrouter_reasoning", "off"), {"reasoning": {"enabled": False}})
        self.assertEqual(self.frag("openrouter_reasoning", "low"), {"reasoning": {"effort": "low"}})
        self.assertEqual(self.frag("openrouter_reasoning", "high"), {"reasoning": {"effort": "high"}})
        self.assertEqual([self.frag("xai_effort", e) for e in ("off", "low", "medium", "high")],
                         [{"reasoning_effort": v} for v in ("low", "low", "high", "high")])

    def test_mentions_field_is_a_whole_identifier_match(self):
        m = providers_reasoning.mentions_field
        self.assertTrue(m("unsupported parameter: 'reasoning'", "reasoning"))
        self.assertTrue(m("reasoning.effort is not allowed", "reasoning"))
        self.assertFalse(m("unknown field reasoning_effort", "reasoning"))
        self.assertTrue(m("unknown field reasoning_effort", "reasoning_effort"))
        self.assertFalse(m("cannot restore the session", "store"))
        self.assertFalse(m("stored items", "store"))
        self.assertTrue(m('"store": false is not supported', "store"))
        self.assertFalse(m("", "store"))


def _providers_reasoning_module():
    from oscar import provider_reasoning

    return provider_reasoning


providers_reasoning = _providers_reasoning_module()


class AnthropicBodyTest(_ProviderCase):
    HIST = [providers.Message(role="user", content="hi")]

    def test_no_reasoning_parameter_means_todays_wire_shape(self):
        body = self.anthropic()._body("sys", self.HIST, [TOOL])
        self.assertNotIn("thinking", body)
        self.assertEqual(body["temperature"], 0.2)
        self.assertNotIn("tool_choice", body)
        self.assertNotIn("disable_parallel_tool_use", json.dumps(body))

    def test_auto_per_call_or_profile_sends_nothing(self):
        self.assertNotIn("thinking", self.anthropic()._body("s", self.HIST, [], reasoning="auto"))
        self.assertNotIn("thinking", self.anthropic(reasoning="auto")._body("s", self.HIST, []))

    def test_per_call_effort_enables_thinking_and_drops_temperature(self):
        body = self.anthropic()._body("s", self.HIST, [], reasoning="medium")
        self.assertEqual(body["thinking"], {"type": "enabled", "budget_tokens": 4096})
        self.assertNotIn("temperature", body)
        self.assertLess(body["thinking"]["budget_tokens"], body["max_tokens"])

    def test_explicit_profile_effort_applies_and_per_call_overrides_it(self):
        p = self.anthropic(reasoning="high")
        self.assertEqual(p._body("s", self.HIST, [])["thinking"]["budget_tokens"], 7168)
        self.assertEqual(p._body("s", self.HIST, [], reasoning="low")["thinking"]["budget_tokens"], 1024)
        off = p._body("s", self.HIST, [], reasoning="off")
        self.assertNotIn("thinking", off)
        self.assertIn("temperature", off)

    def test_tiny_output_cap_omits_thinking(self):
        body = self.anthropic(max_tokens=2048)._body("s", self.HIST, [], reasoning="high")
        self.assertNotIn("thinking", body)
        self.assertIn("temperature", body)

    def test_per_call_max_tokens_is_bounded_by_the_learned_cap(self):
        p = self.anthropic()
        self.assertEqual(p._body("s", self.HIST, [], max_tokens=1500)["max_tokens"], 1500)
        p._learned["max_tokens_cap"] = 1000
        self.assertEqual(p._body("s", self.HIST, [], max_tokens=1500)["max_tokens"], 1000)
        self.assertEqual(p._body("s", self.HIST, [])["max_tokens"], 1000)

    def test_off_in_an_open_loop_that_replays_thinking_is_sent_as_low(self):
        p = self.anthropic()
        thinking_turn = providers.Message(role="assistant", provider_blocks=[
            {"type": "thinking", "thinking": "t", "signature": "s"}, {"type": "tool_use", "id": "t1", "name": "x", "input": {}}],
            tool_calls=[providers.ToolCall("t1", "x")])
        loop = [providers.Message(role="user", content="go"), thinking_turn,
                providers.Message(role="user", tool_results=[providers.ToolResult("t1", "x", "ok")])]
        self.assertEqual(p._body("s", loop, [], reasoning="off")["thinking"]["budget_tokens"], 1024)
        # a thinking block of an EARLIER user turn does not pin the loop
        earlier = loop + [providers.Message(role="user", content="new turn")]
        self.assertNotIn("thinking", p._body("s", earlier, [], reasoning="off"))
        # no thinking in the loop: off really is off
        self.assertNotIn("thinking", p._body("s", self.HIST, [], reasoning="off"))

    def test_models_without_a_thinking_control_never_get_thinking(self):
        p = self.anthropic("claude-3-5-sonnet-20241022")
        self.assertNotIn("thinking", p._body("s", self.HIST, [], reasoning="high"))

    def test_a_learned_rejection_disables_thinking_for_the_model(self):
        p = self.anthropic()
        body = p._body("s", self.HIST, [], reasoning="medium")
        self.assertTrue(p._adapt_to_rejection("thinking.budget_tokens: Input should be valid", body))
        self.assertNotIn("thinking", p._body("s", self.HIST, [], reasoning="medium"))
        again = providers.AnthropicProvider(p.profile)     # a new worker: learned for good
        self.assertNotIn("thinking", again._body("s", self.HIST, [], reasoning="medium"))
        self.assertEqual(again._caps().reasoning_control, "none")

    def test_sampling_is_never_sent_to_models_that_reject_it(self):
        self.assertNotIn("temperature", self.anthropic("claude-opus-4-8")._body("s", self.HIST, []))
        self.assertIn("temperature", self.anthropic("claude-haiku-4-5")._body("s", self.HIST, []))


class AnthropicToolChoiceTest(_ProviderCase):
    HIST = [providers.Message(role="user", content="hi")]

    def body(self, p=None, **opts):
        return (p or self.anthropic())._body("s", self.HIST, [TOOL, TOOL2], **opts)

    def test_none_and_auto_omit_tool_choice(self):
        self.assertNotIn("tool_choice", self.body())
        self.assertNotIn("tool_choice", self.body(tool_choice="auto"))

    def test_required_and_named(self):
        self.assertEqual(self.body(tool_choice="required")["tool_choice"], {"type": "any"})
        self.assertEqual(self.body(tool_choice="get_view")["tool_choice"], {"type": "tool", "name": "get_view"})

    def test_none_maps_to_the_none_type(self):
        self.assertEqual(self.body(tool_choice="none")["tool_choice"], {"type": "none"})

    def test_an_unknown_tool_name_is_ignored(self):
        self.assertNotIn("tool_choice", self.body(tool_choice="no_such_tool"))

    def test_forced_choices_are_downgraded_while_thinking_is_on(self):
        p = self.anthropic()
        for choice in ("required", "get_view"):
            body = self.body(p, tool_choice=choice, reasoning="medium")
            self.assertIn("thinking", body)
            self.assertNotIn("tool_choice", body)
        # but "none" is allowed with thinking, and forced choices work with thinking off
        self.assertEqual(self.body(p, tool_choice="none", reasoning="medium")["tool_choice"], {"type": "none"})
        self.assertEqual(self.body(p, tool_choice="required", reasoning="off")["tool_choice"], {"type": "any"})

    def test_no_tools_means_no_tool_choice(self):
        body = self.anthropic()._body("s", self.HIST, [], tool_choice="required")
        self.assertNotIn("tool_choice", body)
        self.assertNotIn("tools", body)

    def test_tools_none_caps_drop_the_array(self):
        p = self.anthropic(caps_override={"tools": "none"})
        self.assertNotIn("tools", self.body(p, tool_choice="required"))

    def test_parallel_tool_calls_false_is_disable_parallel_tool_use(self):
        body = self.body(parallel_tool_calls=False)
        self.assertEqual(body["tool_choice"], {"type": "auto", "disable_parallel_tool_use": True})
        body = self.body(parallel_tool_calls=False, tool_choice="required")
        self.assertEqual(body["tool_choice"], {"type": "any", "disable_parallel_tool_use": True})
        self.assertNotIn("tool_choice", self.body(parallel_tool_calls=True))
        self.assertNotIn("tool_choice", self.body(p=self.anthropic(caps_override={"parallel_calls": False}),
                                                  parallel_tool_calls=False))

    def test_a_tool_choice_rejection_is_learned(self):
        p = self.anthropic()
        body = self.body(p, tool_choice="required")
        self.assertTrue(p._adapt_to_rejection("tool_choice: Input tag 'any' found using 'type' does not match", body))
        self.assertNotIn("tool_choice", self.body(p, tool_choice="required"))
        self.assertFalse(p._caps().tool_choice_ok)


class AnthropicCacheControlTest(_ProviderCase):
    HIST = [providers.Message(role="user", content="hi")]

    def test_system_block_and_last_tool_carry_cache_control_when_the_model_caches(self):
        body = self.anthropic()._body("SYS", self.HIST, [TOOL, TOOL2])
        self.assertEqual(body["system"], [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}])
        self.assertNotIn("cache_control", body["tools"][0])
        self.assertEqual(body["tools"][-1]["cache_control"], {"type": "ephemeral"})

    def test_models_that_do_not_cache_keep_a_plain_string_system(self):
        body = self.anthropic("mystery-model")._body("SYS", self.HIST, [TOOL])
        self.assertEqual(body["system"], "SYS")
        self.assertNotIn("cache_control", json.dumps(body))

    def test_an_empty_system_prompt_never_becomes_an_empty_text_block(self):
        body = self.anthropic()._body("", self.HIST, [TOOL])
        self.assertEqual(body["system"], "")

    def test_request_without_tools_still_caches_the_system_block(self):
        body = self.anthropic()._body("SYS", self.HIST, [])
        self.assertEqual(body["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("tools", body)

    def test_a_400_naming_cache_control_turns_it_off_for_good(self):
        p = self.anthropic()
        body = p._body("SYS", self.HIST, [TOOL])
        self.assertTrue(p._adapt_to_rejection("system.0.cache_control: Extra inputs are not permitted", body))
        plain = p._body("SYS", self.HIST, [TOOL])
        self.assertEqual(plain["system"], "SYS")
        self.assertNotIn("cache_control", json.dumps(plain))
        again = providers.AnthropicProvider(p.profile)
        self.assertNotIn("cache_control", json.dumps(again._body("SYS", self.HIST, [TOOL])))

    def test_cache_control_in_an_error_is_ignored_when_the_request_had_none(self):
        p = self.anthropic("mystery-model")
        body = p._body("SYS", self.HIST, [TOOL])
        self.assertFalse(p._adapt_to_rejection("cache_control is not supported", body))


class OpenAIBodyTest(_ProviderCase):
    HIST = [providers.Message(role="user", content="hi")]

    def test_no_reasoning_parameter_without_an_explicit_effort(self):
        body = self.openai()._body("s", self.HIST, [])
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(body["max_completion_tokens"], 4096)       # reasoning model naming
        self.assertNotIn("temperature", body)
        self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_effort_for_openai_reasoning_models(self):
        p = self.openai("o3-mini")
        self.assertEqual(p._body("s", self.HIST, [], reasoning="high")["reasoning_effort"], "high")
        self.assertEqual(p._body("s", self.HIST, [], reasoning="off")["reasoning_effort"], "low")
        self.assertEqual(self.openai("gpt-5")._body("s", self.HIST, [], reasoning="off")["reasoning_effort"], "minimal")
        self.assertNotIn("reasoning_effort", p._body("s", self.HIST, [], reasoning="auto"))
        self.assertEqual(self.openai("gpt-5", reasoning="low")._body("s", self.HIST, [])["reasoning_effort"], "low")

    def test_host_specific_knob_styles(self):
        custom = "https://llm.example.com/v1"
        body = self.openai("m", custom, caps_override={"reasoning_control": "zai_thinking"})._body(
            "s", self.HIST, [], reasoning="medium")
        self.assertEqual(body["thinking"], {"type": "enabled"})
        body = self.openai("qwen3-max", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")._body(
            "s", self.HIST, [], reasoning="off")
        self.assertIs(body["enable_thinking"], False)
        body = self.openai("qwen3-8b", custom, caps_override={"reasoning_control": "qwen_enable_thinking"})._body(
            "s", self.HIST, [], reasoning="high")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
        body = self.openai("m", custom, caps_override={"reasoning_control": "openrouter_reasoning"})._body(
            "s", self.HIST, [], reasoning="low")
        self.assertEqual(body["reasoning"], {"effort": "low"})

    def test_other_wire_formats_knobs_never_leak_into_chat_completions(self):
        body = self.openai("m", "https://llm.example.com/v1", caps_override={"reasoning_control": "anthropic_budget"})._body(
            "s", self.HIST, [], reasoning="high")
        self.assertNotIn("thinking", body)
        self.assertNotIn("reasoning", body)

    def test_models_without_a_control_get_nothing_even_when_asked(self):
        body = self.openai("gpt-4o")._body("s", self.HIST, [], reasoning="high")
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(body["temperature"], 0.2)

    def test_stream_options_only_for_hosts_that_accept_them_or_after_a_proof(self):
        self.assertIn("stream_options", self.openai()._body("s", self.HIST, []))
        self.assertIn("stream_options", self.openai("m", "https://openrouter.ai/api/v1")._body("s", self.HIST, []))
        p = self.openai("m", "http://localhost:11434/v1")
        self.assertNotIn("stream_options", p._body("s", self.HIST, []))
        p._learned["stream_options_ok"] = True
        self.assertIn("stream_options", p._body("s", self.HIST, []))
        custom = self.openai("m2", "https://llm.example.com/v1")
        self.assertNotIn("stream_options", custom._body("s", self.HIST, []))

    def test_per_call_max_tokens(self):
        p = self.openai("gpt-4o")
        self.assertEqual(p._body("s", self.HIST, [], max_tokens=777)["max_tokens"], 777)
        self.assertEqual(p._body("s", self.HIST, [])["max_tokens"], 4096)

    def test_tool_choice_mapping(self):
        p = self.openai("gpt-4o")
        body = lambda **o: p._body("s", self.HIST, [TOOL, TOOL2], **o)   # noqa: E731
        self.assertNotIn("tool_choice", body())
        self.assertNotIn("tool_choice", body(tool_choice="auto"))
        self.assertEqual(body(tool_choice="required")["tool_choice"], "required")
        self.assertEqual(body(tool_choice="none")["tool_choice"], "none")
        self.assertEqual(body(tool_choice="get_view")["tool_choice"], {"type": "function", "function": {"name": "get_view"}})
        self.assertNotIn("tool_choice", body(tool_choice="no_such_tool"))

    def test_hosts_without_tool_choice_drop_the_tools_for_none_and_ignore_the_rest(self):
        p = self.openai("gpt-4o", caps_override={"tool_choice_ok": False})
        self.assertNotIn("tools", p._body("s", self.HIST, [TOOL], tool_choice="none"))
        body = p._body("s", self.HIST, [TOOL], tool_choice="required")
        self.assertIn("tools", body)
        self.assertNotIn("tool_choice", body)

    def test_tools_none_and_no_tools(self):
        self.assertNotIn("tools", self.openai(caps_override={"tools": "none"})._body("s", self.HIST, [TOOL]))
        body = self.openai()._body("s", self.HIST, [], tool_choice="required")
        self.assertNotIn("tool_choice", body)

    def test_parallel_tool_calls_false(self):
        body = lambda p, **o: p._body("s", self.HIST, [TOOL], **o)   # noqa: E731
        self.assertIs(body(self.openai("gpt-4o"), parallel_tool_calls=False)["parallel_tool_calls"], False)
        self.assertNotIn("parallel_tool_calls", body(self.openai("gpt-4o"), parallel_tool_calls=True))
        self.assertNotIn("parallel_tool_calls", body(self.openai("gpt-4o"), ))
        custom = self.openai("m", "https://llm.example.com/v1")
        self.assertNotIn("parallel_tool_calls", body(custom, parallel_tool_calls=False))   # unproven on a custom host
        custom._learned["parallel_tool_calls_ok"] = True
        self.assertIn("parallel_tool_calls", body(custom, parallel_tool_calls=False))
        serial = self.openai("gpt-4o", caps_override={"parallel_calls": False})
        self.assertNotIn("parallel_tool_calls", body(serial, parallel_tool_calls=False))


class GenericUnknownFieldRuleTest(_ProviderCase):
    """a 400 naming an optional field drops exactly that field, learns it per (host, model), retries."""

    HIST = [providers.Message(role="user", content="hi")]
    MESSAGE = "400 Unrecognized request argument supplied: {field}"

    def check(self, make, field, **opts):
        p = make()
        body = p._body("s", self.HIST, [TOOL, TOOL2], **opts)
        self.assertIn(field, body, f"{field} must be part of the request for this test to mean anything")
        before = {k: v for k, v in body.items() if k != field}
        self.assertTrue(p._adapt_to_rejection(self.MESSAGE.format(field=field), body))
        after = p._body("s", self.HIST, [TOOL, TOOL2], **opts)
        self.assertNotIn(field, after)
        self.assertEqual({k: after[k] for k in before if k in after and k != "messages"},
                         {k: before[k] for k in before if k in after and k != "messages"}, "only that field changes")
        # a second rejection of the same field is not an adaptation any more (no endless retry loop)
        self.assertFalse(p._adapt_to_rejection(self.MESSAGE.format(field=field), after))
        # learned for the next worker of the same model
        again = type(p)(p.profile)
        self.assertNotIn(field, again._body("s", self.HIST, [TOOL, TOOL2], **opts))

    def test_stream_options(self):
        self.check(lambda: self.openai("gpt-4o"), "stream_options")

    def test_parallel_tool_calls(self):
        self.check(lambda: self.openai("gpt-4o"), "parallel_tool_calls", parallel_tool_calls=False)

    def test_tool_choice(self):
        self.check(lambda: self.openai("gpt-4o"), "tool_choice", tool_choice="required")

    def test_reasoning_effort(self):
        self.check(lambda: self.openai("gpt-5"), "reasoning_effort", reasoning="high")

    def test_enable_thinking(self):
        self.check(lambda: self.openai("qwen3-max", "https://dashscope-intl.aliyuncs.com/v1"), "enable_thinking", reasoning="high")

    def test_chat_template_kwargs(self):
        self.check(lambda: self.openai("qwen3-8b", "https://llm.example.com/v1",
                                       caps_override={"reasoning_control": "qwen_enable_thinking"}),
                   "chat_template_kwargs", reasoning="high")

    def test_reasoning_object_on_chat_completions(self):
        self.check(lambda: self.openai("m", "https://llm.example.com/v1",
                                       caps_override={"reasoning_control": "openrouter_reasoning"}),
                   "reasoning", reasoning="medium")

    def test_thinking_on_chat_completions(self):
        self.check(lambda: self.openai("glm-4.6", "https://api.z.ai/api/paas/v4"), "thinking", reasoning="medium")

    def test_thinking_on_anthropic(self):
        self.check(lambda: self.anthropic(), "thinking", reasoning="medium")


    def test_a_field_name_inside_a_longer_identifier_is_not_a_match(self):
        p = self.openai("gpt-5")
        body = p._body("s", self.HIST, [], reasoning="high")
        self.assertFalse(p._adapt_to_rejection("unknown parameter: reasoning_effort_level", body))
        self.assertIn("reasoning_effort", p._body("s", self.HIST, [], reasoning="high"))

    def test_an_unrelated_400_adapts_nothing(self):
        p = self.openai("gpt-4o")
        body = p._body("s", self.HIST, [TOOL])
        self.assertFalse(p._adapt_to_rejection("messages: roles must alternate", body))
        self.assertFalse(p._adapt_to_rejection("", body))
        self.assertFalse(p._adapt_to_rejection("anything", None))

    def test_context_length_errors_are_learned_but_not_retried_as_is(self):
        p = self.openai("gpt-4o", "https://llm.example.com/v1")
        body = p._body("s", self.HIST, [])
        msg = "This model's maximum context length is 8192 tokens. However, your messages resulted in 20000 tokens."
        self.assertFalse(p._adapt_to_rejection(msg, body))
        self.assertEqual(p._caps().context_window, 8192)

    def test_output_cap_errors_set_the_learned_cap_and_max_output(self):
        p = self.anthropic()
        body = p._body("s", self.HIST, [])
        self.assertTrue(p._adapt_to_rejection("max_tokens: 8192 > 4096, which is the maximum allowed number of output tokens", body))
        self.assertEqual(p._body("s", self.HIST, [])["max_tokens"], 4096)
        self.assertEqual(p._caps().max_output, 4096)


# --------------------------------------------------------------------------------------------------------------
# Reasoning replay, tool_args_stream learning, vision learning, list_models


class ReasoningReplayTest(_ProviderCase):
    """``reasoning_content`` is kept in ``provider_blocks`` and replayed only per ``caps.reasoning_replay``."""

    HIST = [providers.Message(role="user", content="hi")]
    LINES = sse([{"choices": [{"delta": {"reasoning_content": "ab"}}]}, {"choices": [{"delta": {"reasoning_content": "cd"}}]},
                 {"choices": [{"delta": {"content": "Answer"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                 {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}], done=True)

    def parsed(self, **extra):
        p = self.openai("m", "https://llm.example.com/v1", **extra)
        p._body("s", self.HIST, [])
        return p, parse(p, self.LINES)

    def test_offered_for_replay_when_the_caps_say_so(self):
        _, events = self.parsed(caps_override={"reasoning_replay": True})
        self.assertEqual(payload(events, "assistant_blocks"), [{"type": "reasoning_content", "text": "abcd"}])
        order = kinds(events)
        self.assertLess(order.index("assistant_blocks"), order.index("usage"))
        self.assertEqual("".join(p for k, p in events if k == "reasoning"), "abcd")

    def test_not_offered_by_default(self):
        _, events = self.parsed()
        self.assertNotIn("assistant_blocks", kinds(events))

    def test_a_reply_without_reasoning_offers_nothing(self):
        p = self.openai("m", "https://llm.example.com/v1", caps_override={"reasoning_replay": True})
        p._body("s", self.HIST, [])
        self.assertNotIn("assistant_blocks", kinds(parse(p, openai_reply(text="plain"))))

    def test_replayed_on_the_assistant_message_only_when_enabled(self):
        message = providers.Message(role="assistant", content="x", provider_blocks=[{"type": "reasoning_content", "text": "abcd"}])
        on = self.openai("m", "https://llm.example.com/v1", caps_override={"reasoning_replay": True})
        on._body("s", [message], [])
        self.assertEqual(on._to_wire(message)[0]["reasoning_content"], "abcd")
        off = self.openai("m", "https://llm.example.com/v1")
        off._body("s", [message], [])
        self.assertNotIn("reasoning_content", off._to_wire(message)[0])


    def test_blocks_of_other_providers_are_ignored(self):
        message = providers.Message(role="assistant", content="x", provider_blocks=[
            {"type": "thinking", "thinking": "t", "signature": "s"}, {"type": "redacted_thinking", "data": "e"}])
        on = self.openai("m", "https://llm.example.com/v1", caps_override={"reasoning_replay": True})
        on._body("s", [message], [])
        self.assertNotIn("reasoning_content", on._to_wire(message)[0])


class ToolArgsStreamLearningTest(_ProviderCase):
    """``tool_args_stream`` is learned from how long arguments arrive: one fragment = a buffering server."""

    HIST = [providers.Message(role="user", content="hi")]
    LONG = '{"code": "' + "x" * 600 + '"}'

    def stream(self, fragments, base="https://llm.example.com/v1", model="m"):
        events = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "create_object", "arguments": fragments[0]}}]}}]}]
        events += [{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": f}}]}}]} for f in fragments[1:]]
        events.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        p = self.openai(model, base)
        p._body("s", self.HIST, [TOOL])
        out = parse(p, sse(events, done=True))
        return p, out

    def test_a_long_call_in_one_fragment_marks_the_server_as_buffering(self):
        p, events = self.stream([self.LONG])
        self.assertIs(p._caps().tool_args_stream, False)
        self.assertIs(models.learned_fields(p._caps_subject())["tool_args_stream"], False)
        self.assertEqual(len(payload(events, "tool_calls")), 1)

    def test_a_long_call_in_many_fragments_marks_it_as_streaming_even_on_a_suspected_host(self):
        fragments = [self.LONG[i:i + 60] for i in range(0, len(self.LONG), 60)]
        p, _ = self.stream(fragments, "http://localhost:11434/v1", "tinyllama")
        self.assertIs(models.resolve_caps(p.profile).tool_args_stream, True)       # the Ollama seed said False

    def test_short_calls_say_nothing(self):
        p, _ = self.stream(['{"a": 1}'])
        self.assertNotIn("tool_args_stream", models.learned_fields(p._caps_subject()))
        self.assertIs(p._caps().tool_args_stream, True)

    def test_an_unchanged_value_is_not_written_again(self):
        p, _ = self.stream([self.LONG])
        with mock.patch.object(models, "learn") as learn:
            p._learn_args_streaming({0: {"id": "c", "name": "n", "args": self.LONG, "n": 1}})
        learn.assert_not_called()

    def test_text_mode_replies_never_touch_it(self):
        p = self.openai("m", "https://llm.example.com/v1", caps_override={"tools": "text"})
        p._body("s", self.HIST, [TOOL])
        parse(p, openai_reply(text='<tool_call>{"name": "create_object", "arguments": {"a": "' + "y" * 600 + '"}}</tool_call>'))
        self.assertNotIn("tool_args_stream", models.learned_fields(p._caps_subject()))


class VisionLearningTest(_ProviderCase):
    def test_an_ambiguous_rejection_degrades_the_session_but_is_not_persisted(self):
        p = self.openai("m", "https://llm.example.com/v1")
        p._set_vision(False, "Invalid base64 image data")
        self.assertFalse(p.vision_enabled)
        self.assertNotIn("vision", models.learned_fields(p._caps_subject()))
        self.assertTrue(providers.OpenAICompatProvider(p.profile).vision_enabled)        # the next worker tries again

    def test_an_unambiguous_one_is_remembered(self):
        p = self.openai("m", "https://llm.example.com/v1")
        p._set_vision(False, "this model does not support image input")
        self.assertIs(models.learned_fields(p._caps_subject())["vision"], False)
        self.assertFalse(providers.OpenAICompatProvider(p.profile).vision_enabled)        # the next worker starts text-only

    def test_the_broad_detector_still_matches_what_it_always_matched(self):
        for text in ("Invalid base64 image data", "this model does not support image input",
                     "this model is missing data required for image input", "image content is not supported"):
            self.assertTrue(providers._looks_like_image_rejection(text), text)


class ListModelsTest(unittest.TestCase):
    class Resp:
        def __init__(self, status=200, payload=None, boom=None, text=""):
            self.status_code, self._payload, self._boom, self.text = status, payload, boom, text

        def json(self):
            if self._boom:
                raise self._boom
            return self._payload

    def profile(self, provider="openai_compat", base="http://localhost:11434/v1", key=""):
        return params.Profile(name="t", provider=provider, base_url=base, model="m", api_key=key)

    def test_lives_with_the_probes_and_stays_importable_from_providers(self):
        self.assertIs(providers.list_models, provider_probe.list_models)

    def test_openai_shape_sorted_and_deduplicated_with_the_key_header(self):
        import requests

        with mock.patch.object(requests, "get", return_value=self.Resp(payload={"data": [{"id": "b"}, {"id": "a"}, {"name": "c"}, {"id": "a"}, "junk"]})) as get:
            self.assertEqual(providers.list_models(self.profile(key="K", base="https://x/v1/")), (["a", "b", "c"], ""))
        self.assertEqual(get.call_args.args[0], "https://x/v1/models")
        self.assertEqual(get.call_args.kwargs["headers"], {"Authorization": "Bearer K"})

    def test_ollama_tags_shape_and_a_bare_list(self):
        import requests

        with mock.patch.object(requests, "get", return_value=self.Resp(payload={"models": [{"name": "qwen2.5:7b"}, {"name": "llama3:8b"}]})):
            self.assertEqual(providers.list_models(self.profile()), (["llama3:8b", "qwen2.5:7b"], ""))
        with mock.patch.object(requests, "get", return_value=self.Resp(payload=[{"id": "z"}])):
            self.assertEqual(providers.list_models(self.profile()), (["z"], ""))

    def test_anthropic_uses_its_headers(self):
        import requests

        with mock.patch.object(requests, "get", return_value=self.Resp(payload={"data": [{"id": "claude-x"}]})) as get:
            self.assertEqual(providers.list_models(self.profile("anthropic", "", "sk")), (["claude-x"], ""))
        self.assertEqual(get.call_args.args[0], "https://api.anthropic.com/v1/models")
        self.assertEqual(get.call_args.kwargs["headers"], {"x-api-key": "sk", "anthropic-version": "2023-06-01"})

    def test_failures_never_raise(self):
        import requests

        cases = ((self.Resp(401, text="bad key"), "HTTP 401: bad key"), (self.Resp(payload={"data": []}), "endpoint returned no models"),
                 (self.Resp(boom=ValueError("x")), "endpoint did not return JSON"),
                 (self.Resp(payload="string"), "endpoint returned no models"), (self.Resp(payload={"data": 5}), "endpoint returned an unexpected answer"))
        for resp, error in cases:
            with mock.patch.object(requests, "get", return_value=resp):
                self.assertEqual(providers.list_models(self.profile()), ([], error))
        with mock.patch.object(requests, "get", side_effect=requests.ConnectionError("refused")):
            names, error = providers.list_models(self.profile())
        self.assertEqual(names, [])
        self.assertTrue(error.startswith("connection failed"))


class ComingSoonProviderTest(unittest.TestCase):
    """An announced provider is listed in the settings dialog but never built. Nothing is announced since the OriginShape
    account became usable; the mechanism is tested with a dummy kind."""

    SOON = {"label": "Example Cloud", "provider": "example_cloud", "note": "Sign in with Example Cloud. Coming soon."}

    def announce(self):
        for patcher in (mock.patch.object(params, "COMING_SOON_PROVIDERS", [dict(self.SOON)]),
                        mock.patch.object(providers, "COMING_SOON_KINDS", {"example_cloud": "Example Cloud"})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_nothing_is_announced_and_no_preset_uses_an_announced_kind(self):
        self.assertEqual(providers.COMING_SOON_KINDS, {})
        self.assertEqual({p["provider"]: p["label"] for p in params.COMING_SOON_PROVIDERS}, providers.COMING_SOON_KINDS)
        self.announce()
        self.assertEqual({p["provider"]: p["label"] for p in params.COMING_SOON_PROVIDERS}, providers.COMING_SOON_KINDS)
        self.assertFalse({p["provider"] for p in params.PROVIDER_PRESETS} & set(providers.COMING_SOON_KINDS))

    def test_a_profile_of_an_announced_provider_is_refused_with_a_clear_message(self):
        self.announce()
        profile = params.Profile(name="Hand-edited", provider="example_cloud", base_url="https://example.invalid/v1", model="m")
        with self.assertRaises(providers.ProviderError) as caught:
            providers.make_provider(profile)
        self.assertEqual(str(caught.exception), "Example Cloud is coming soon and cannot be used yet. "
                                                "Choose another provider in Oscar's settings (⚙).")

    def test_the_originshape_account_is_built_not_refused(self):
        from oscar.provider_originshape import OriginShapeProvider

        profile = params.account_profile()
        provider = providers.make_provider(profile)
        self.assertIsInstance(provider, OriginShapeProvider)
        self.assertEqual(provider.profile.model, "origin1")


if __name__ == "__main__":
    unittest.main()
