"""Regression tests for Oscar's agent loop, providers and tools.

Each test pins a bug that made Oscar take the wrong next step: failures
reported as successes, forgotten answers, broken histories after Stop,
rejected request parameters, half-built objects left behind, screenshots of
the wrong document, … Runs in the console interpreter (no GUI); providers are
exercised with recorded SSE bodies and a fake transport.
"""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault(
    "OSCAR_AI_SETTINGS",
    os.path.join(tempfile.mkdtemp(prefix="oscar_regress_"), "settings.json"),
)
if "FreeCADGui" not in sys.modules:
    try:
        import FreeCADGui  # noqa: F401
    except Exception:
        sys.modules["FreeCADGui"] = types.ModuleType("FreeCADGui")

import FreeCAD as App  # noqa: E402

from oscar import agent, loopcore, params, providers, tools  # noqa: E402
from oscar.providers import Message, StopRequested, ToolCall, ToolResult  # noqa: E402


def _profile(provider="anthropic", model="claude-sonnet-4-5", base_url="https://example.test"):
    return params.Profile(name="t", provider=provider, base_url=base_url,
                          model=model, api_key="k", temperature=0.2, max_tokens=4096)


class FakeResponse:
    def __init__(self, lines, status=200, text=""):
        self._lines = lines
        self.status_code = status
        self.text = text

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


def _sse(events):
    return ["data: " + json.dumps(e) for e in events]


# ---------------------------------------------------------------------------
# tools


class ToolResultTest(unittest.TestCase):
    def setUp(self):
        self.doc = App.newDocument("RegressDoc")

    def tearDown(self):
        try:
            App.closeDocument(self.doc.Name)
        except Exception:
            pass

    def run_tool(self, name, **args):
        return tools.run_tool(name, args)

    def test_failures_are_flagged_as_errors(self):
        text, _img, is_error = self.run_tool(
            "create_object", doc_name="NoSuchDocAtAll", obj_type="Part::Box",
            obj_name="B", include_screenshot=False)
        self.assertTrue(is_error, text)
        self.assertIn("not open", text)

    def test_missing_document_is_not_an_empty_document(self):
        text, _img, is_error = self.run_tool("get_objects", doc_name="GhostDocument")
        self.assertTrue(is_error)
        self.assertIn("not open", text)
        self.assertIn(self.doc.Name, text)  # lists what IS open

    def test_multifuse_takes_object_names(self):
        for i, x in enumerate((20, -20)):
            self.run_tool("create_object", doc_name=self.doc.Name, obj_type="Part::Cylinder",
                          obj_name=f"Hole{i}", include_screenshot=False,
                          obj_properties={"Radius": 3, "Height": 10,
                                          "Placement": {"Base": {"x": x, "y": 0, "z": -1}}})
        text, _img, is_error = self.run_tool(
            "create_object", doc_name=self.doc.Name, obj_type="Part::MultiFuse",
            obj_name="Holes", obj_properties={"Shapes": ["Hole0", "Hole1"]},
            include_screenshot=False)
        self.assertFalse(is_error, text)
        fused = self.doc.getObject("Holes")
        self.assertEqual([o.Name for o in fused.Shapes], ["Hole0", "Hole1"])

    def test_unknown_link_name_is_a_clear_error(self):
        text, _img, is_error = self.run_tool(
            "create_object", doc_name=self.doc.Name, obj_type="Part::Cut",
            obj_name="Cut", obj_properties={"Base": "Nope", "Tool": "Nada"},
            include_screenshot=False)
        self.assertTrue(is_error)
        self.assertIn("'Nope'", text)

    def test_failed_create_leaves_nothing_behind(self):
        text, _img, is_error = self.run_tool(
            "create_object", doc_name=self.doc.Name, obj_type="Part::Cylinder",
            obj_name="Disc", obj_properties={"Radius": 40, "Hieght": 8},
            include_screenshot=False)
        self.assertTrue(is_error)
        self.assertIn("Height", text)  # suggests the real property
        self.assertEqual([o.Name for o in self.doc.Objects], [])
        # a corrected retry gets the requested name, not 'Disc001'
        text, _img, is_error = self.run_tool(
            "create_object", doc_name=self.doc.Name, obj_type="Part::Cylinder",
            obj_name="Disc", obj_properties={"Radius": 40, "Height": 8},
            include_screenshot=False)
        self.assertFalse(is_error, text)
        self.assertIsNotNone(self.doc.getObject("Disc"))

    def test_properties_as_json_string_are_accepted(self):
        text, _img, is_error = self.run_tool(
            "create_object", doc_name=self.doc.Name, obj_type="Part::Box",
            obj_name="StrBox", obj_properties='{"Length": 5}', include_screenshot="false")
        self.assertFalse(is_error, text)
        self.assertAlmostEqual(self.doc.getObject("StrBox").Length.Value, 5.0)

    def test_nested_view_object_does_not_fail(self):
        text, _img, is_error = self.run_tool(
            "create_object", doc_name=self.doc.Name, obj_type="Part::Box",
            obj_name="Colored", include_screenshot=False,
            obj_properties={"Length": 5, "ViewObject": {"ShapeColor": [1, 0, 0]}})
        self.assertFalse(is_error, text)

    def test_unknown_argument_names_the_accepted_ones(self):
        text, _img, is_error = self.run_tool(
            "create_object", doc_name=self.doc.Name, obj_type="Part::Box",
            obj_name="X", properties={"Length": 5})
        self.assertTrue(is_error)
        self.assertIn("obj_properties", text)
        self.assertIsNone(self.doc.getObject("X"))

    def test_execute_code_error_names_the_line_and_keeps_output(self):
        text, _img, is_error = self.run_tool(
            "execute_code", include_screenshot=False,
            code="print('step 1 ok')\nx = 1\ny = undefined_name + 1\n")
        self.assertTrue(is_error)
        self.assertIn("line 3", text)
        self.assertIn("undefined_name + 1", text)
        self.assertIn("step 1 ok", text)

    def test_execute_code_recomputes_forgotten_documents(self):
        text, _img, is_error = self.run_tool(
            "execute_code", include_screenshot=False,
            code=(f"d = App.getDocument('{self.doc.Name}')\n"
                  "b = d.addObject('Part::Box', 'Scripted')\nb.Length = 7\n"))
        self.assertFalse(is_error, text)
        box = self.doc.getObject("Scripted")
        self.assertAlmostEqual(box.Shape.BoundBox.XLength, 7.0, places=5)

    def test_deleting_a_boolean_shows_its_inputs_again(self):
        for name, radius in (("Plate", 20), ("Bore", 5)):
            self.run_tool("create_object", doc_name=self.doc.Name, obj_type="Part::Cylinder",
                          obj_name=name, obj_properties={"Radius": radius, "Height": 5},
                          include_screenshot=False)
        self.run_tool("create_object", doc_name=self.doc.Name, obj_type="Part::Cut",
                      obj_name="Cut", obj_properties={"Base": "Plate", "Tool": "Bore"},
                      include_screenshot=False)
        for name in ("Plate", "Bore"):  # console mode: hide like the GUI would
            self.doc.getObject(name).Visibility = False
        text, _img, is_error = self.run_tool("delete_object", doc_name=self.doc.Name,
                                             obj_name="Cut", include_screenshot=False)
        self.assertFalse(is_error, text)
        self.assertTrue(self.doc.getObject("Plate").Visibility)
        self.assertTrue(self.doc.getObject("Bore").Visibility)

    def test_descriptions_have_no_template_leftovers(self):
        for tool in tools.get_tools():
            self.assertNotIn("{_", tool.description, tool.name)
            self.assertNotIn("{{", tool.description, tool.name)
            self.assertNotIn(".FCSrd", tool.description, tool.name)


# ---------------------------------------------------------------------------
# providers


class AnthropicRequestTest(unittest.TestCase):
    def test_temperature_only_where_accepted(self):
        for model, expected in (
            ("claude-sonnet-4-5", True), ("claude-opus-4-6", True),
            ("claude-3-5-haiku-20241022", True), ("claude-opus-4-20250514", True),
            ("claude-opus-4-7", False), ("claude-opus-5", False),
            ("claude-sonnet-5", False), ("claude-fable-5-1", False),
        ):
            body = providers.AnthropicProvider(_profile(model=model))._body(
                "s", [Message(role="user", content="hi")], [])
            self.assertEqual("temperature" in body, expected, model)

    def test_rejected_parameter_is_dropped_and_retried(self):
        provider = providers.AnthropicProvider(_profile(model="claude-custom-proxy-a"))
        ok = FakeResponse(_sse([
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
            {"type": "message_stop"},
        ]))
        bodies = []

        def fake_post(url, headers=None, json=None, stream=True, timeout=None):
            bodies.append(json)
            if len(bodies) == 1:
                return FakeResponse([], 400, '{"error":{"message":"`temperature` is deprecated for this model."}}')
            return ok

        with mock.patch.object(providers.requests, "post", side_effect=fake_post):
            events = list(provider.chat("s", [Message(role="user", content="hi")], []))
        self.assertIn("temperature", bodies[0])
        self.assertNotIn("temperature", bodies[1])
        self.assertEqual([p for k, p in events if k == "text"], ["ok"])

    def test_output_cap_is_learned_from_the_error(self):
        provider = providers.AnthropicProvider(_profile(model="claude-custom-proxy-b"))
        detail = ("max_tokens: 4096 > 1024, which is the maximum allowed number of "
                  "output tokens for claude-custom-proxy-b")
        self.assertTrue(provider._adapt_to_rejection(detail))
        body = provider._body("s", [Message(role="user", content="hi")], [])
        self.assertEqual(body["max_tokens"], 1024)

    def test_thinking_blocks_are_captured_and_replayed(self):
        provider = providers.AnthropicProvider(_profile(model="claude-opus-5"))
        events = [
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "plan"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "SIG"}},
            {"type": "content_block_start", "index": 1,
             "content_block": {"type": "tool_use", "id": "toolu_1", "name": "list_documents"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        ]
        out = list(provider._parse_stream(FakeResponse(_sse(events)), lambda: False))
        blocks = next(p for k, p in out if k == "assistant_blocks")
        self.assertEqual(blocks[0], {"type": "thinking", "thinking": "plan", "signature": "SIG"})
        calls = next(p for k, p in out if k == "tool_calls")
        message = Message(role="assistant", tool_calls=calls, provider_blocks=blocks)
        wire = provider._to_wire(message)
        self.assertEqual(wire["content"][0]["type"], "thinking")
        self.assertEqual(wire["content"][0]["signature"], "SIG")
        self.assertEqual(wire["content"][1]["type"], "tool_use")

    def test_truncated_tool_input_is_flagged(self):
        provider = providers.AnthropicProvider(_profile())
        events = [
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "tool_use", "id": "toolu_1", "name": "execute_code"}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "input_json_delta", "partial_json": '{"code": "print(1'}},
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}},
            {"type": "message_stop"},
        ]
        out = list(provider._parse_stream(FakeResponse(_sse(events)), lambda: False))
        call = next(p for k, p in out if k == "tool_calls")[0]
        self.assertIn("_raw", call.arguments)
        self.assertIn(("stop_reason", "max_tokens"), out)

    def test_no_empty_text_blocks_and_foreign_ids_normalized(self):
        provider = providers.AnthropicProvider(_profile())
        wire = provider._body("s", [
            Message(role="user", content="go"),
            Message(role="assistant", content="",
                    tool_calls=[ToolCall(id="call.abc:1", name="list_documents")]),
            Message(role="user", tool_results=[
                ToolResult(tool_call_id="call.abc:1", name="list_documents", content="")]),
        ], [])["messages"]
        self.assertEqual(wire[1]["content"][0]["id"], "call_abc_1")
        result = wire[2]["content"][0]
        self.assertEqual(result["tool_use_id"], "call_abc_1")
        self.assertTrue(result["content"][0]["text"])
        text_blocks = [b for m in wire for b in m["content"] if b.get("type") == "text"]
        self.assertTrue(all(b["text"] for b in text_blocks))


class OpenAIRequestTest(unittest.TestCase):
    def test_reasoning_models_get_the_right_parameters(self):
        provider = providers.OpenAICompatProvider(
            _profile("openai_compat", "gpt-5", "https://api.openai.com/v1"))
        body = provider._body("s", [Message(role="user", content="hi")], [])
        self.assertIn("max_completion_tokens", body)
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("temperature", body)
        regular = providers.OpenAICompatProvider(
            _profile("openai_compat", "qwen2.5:7b", "http://localhost:11434/v1"))
        body = regular._body("s", [Message(role="user", content="hi")], [])
        self.assertIn("max_tokens", body)
        self.assertIn("temperature", body)

    def test_token_parameter_rename_is_learned(self):
        provider = providers.OpenAICompatProvider(
            _profile("openai_compat", "vendor-model-x", "https://proxy.example/v1"))
        self.assertTrue(provider._adapt_to_rejection(
            "Unsupported parameter: 'max_tokens' is not supported with this model. "
            "Use 'max_completion_tokens' instead."))
        body = provider._body("s", [Message(role="user", content="hi")], [])
        self.assertIn("max_completion_tokens", body)

    def test_tool_calls_without_index_stay_separate(self):
        provider = providers.OpenAICompatProvider(_profile("openai_compat", "m-noindex"))
        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"id":"a","function":{"name":"get_view","arguments":"{}"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"id":"b","function":{"name":"list_documents","arguments":"{}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
        out = list(provider._parse_stream(FakeResponse(lines), lambda: False))
        calls = next(p for k, p in out if k == "tool_calls")
        self.assertEqual([c.name for c in calls], ["get_view", "list_documents"])
        self.assertEqual([c.arguments for c in calls], [{}, {}])

    def test_old_screenshots_are_dropped_from_the_request(self):
        provider = providers.OpenAICompatProvider(_profile("openai_compat", "m-images"))
        history = [Message(role="user", content="build")]
        for i in range(providers.MAX_WIRE_IMAGES + 3):
            history.append(Message(role="assistant", tool_calls=[ToolCall(id=f"c{i}", name="get_view")]))
            history.append(Message(role="user", tool_results=[
                ToolResult(tool_call_id=f"c{i}", name="get_view", content="ok", image_b64="QUJD")]))
        body = provider._body("s", history, [])
        images = sum(
            1 for m in body["messages"] if isinstance(m.get("content"), list)
            for part in m["content"] if part.get("type") == "image_url"
        )
        self.assertEqual(images, providers.MAX_WIRE_IMAGES)


# ---------------------------------------------------------------------------
# agent loop


class ScriptedProvider:
    vision_enabled = True

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.systems = []
        self.requests = []

    def chat(self, system, messages, specs, should_stop=None):
        self.systems.append(system)
        self.requests.append(list(messages))
        kind, payload = self.rounds.pop(0)
        if kind == "text":
            yield ("text", payload)
            yield ("stop_reason", "end_turn")
            yield ("done", "end_turn")
        elif kind == "cutoff":
            # a reasoning model that spent max_tokens on thinking alone
            yield ("reasoning", payload)
            yield ("stop_reason", "length")
            yield ("done", "length")
        elif kind == "nameless":
            # finish_reason tool_calls, but the provider dropped the call
            yield ("stop_reason", "tool_calls")
            yield ("done", "tool_calls")
        else:
            yield ("stop_reason", "tool_use")
            yield ("tool_calls", payload)


class AgentLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        params.save_profiles([_profile("openai_compat", "scripted", "http://127.0.0.1:9/v1")])
        params.set_active_profile("t")

    def run_turn(self, history, rounds, run_tool=None):
        worker = agent.AgentWorker(history)
        worker.provider = ScriptedProvider(rounds)
        finished = []
        worker.turn_finished.connect(lambda ok, text: finished.append((ok, text)))
        patcher = mock.patch.object(agent._tools, "run_tool", run_tool) if run_tool else None
        if patcher:
            patcher.start()
        try:
            worker._run_turn()
        except StopRequested:
            finished.append((False, "Stopped."))
        finally:
            if patcher:
                patcher.stop()
        return worker, finished

    def test_final_answer_is_remembered(self):
        history = [Message(role="user", content="list my documents")]
        _w, finished = self.run_turn(history, [
            ("tools", [ToolCall(id="c1", name="list_documents")]),
            ("text", "You have one document."),
        ])
        self.assertEqual(finished, [(True, "You have one document.")])
        self.assertEqual(history[-1].role, "assistant")
        self.assertEqual(history[-1].content, "You have one document.")

    def test_thinking_cutoff_is_nudged_then_the_model_acts(self):
        # Recorded from z.ai glm-5.3 with max_tokens 8192: 8189 reasoning
        # tokens deliberating a STEP download URL, then finish_reason
        # "length" with no content and no tool call. The turn ended as a
        # successful empty answer — the panel showed nothing at all.
        # Migrated: "build a Pi 4 housing" is now an enclosure work turn with a template plan; "make a plate"
        # is a generic_part turn whose first geometry call satisfies the gate.
        history = [Message(role="user", content="make a plate")]
        notices = []
        worker = agent.AgentWorker(history)
        worker.notice.connect(notices.append)
        worker.provider = ScriptedProvider([
            ("cutoff", "Do I KNOW a trustworthy URL? " * 40),
            ("tools", [ToolCall(id="c1", name="create_object", arguments={
                "doc_name": "Plate", "obj_type": "Part::Box", "obj_name": "Plate"})]),
            ("text", "Modelled from nominal dimensions."),
        ])
        finished = []
        worker.turn_finished.connect(lambda ok, text: finished.append((ok, text)))
        with mock.patch.object(agent._tools, "run_tool", lambda n, a: ("created", None, False)):
            worker._run_turn()
        self.assertEqual(finished, [(True, "Modelled from nominal dimensions.")])
        # the model was told what happened, with roles still alternating
        self.assertEqual([m.role for m in history[:4]],
                         ["user", "assistant", "user", "assistant"])
        self.assertIn("cut off", history[1].content)
        self.assertIn("output-token limit", history[2].content)
        self.assertEqual(history[3].tool_calls[0].name, "create_object")
        self.assertIn("output-token limit", worker.provider.requests[1][-1].content)
        self.assertTrue(any("output-token limit" in n for n in notices), notices)

    def test_repeated_cutoffs_end_with_a_visible_error(self):
        history = [Message(role="user", content="build a Pi 4 housing")]
        rounds = [("cutoff", "hmm…")] * (agent.MAX_CUTOFF_RETRIES + 1)
        worker, finished = self.run_turn(history, rounds)
        self.assertEqual(len(finished), 1)
        ok, text = finished[0]
        self.assertFalse(ok)
        self.assertIn("output budget", text)
        self.assertIn(f"{agent.MAX_CUTOFF_RETRIES + 1} times in a row", text)
        self.assertEqual(len(worker.provider.requests), agent.MAX_CUTOFF_RETRIES + 1)
        # nothing bogus is recorded as an answer
        self.assertEqual(history[-1].role, "user")

    def test_incomplete_tool_call_is_reported(self):
        # Migrated: "add a box" is a work request, so an empty reply is nudged twice (per-code gate budget 2)
        # before the turn fails: three requests in all, one failure, and no assistant answer in the history.
        history = [Message(role="user", content="add a box")]
        worker, finished = self.run_turn(history, [("nameless", None)] * 3)
        self.assertEqual(len(finished), 1)
        self.assertFalse(finished[0][0])
        self.assertIn("incomplete", finished[0][1])
        self.assertEqual(len(worker.provider.requests), 3)
        self.assertEqual(history[-1].role, "user")
        self.assertFalse([m for m in history if m.role == "assistant" and m.content != loopcore.NO_TEXT_PLACEHOLDER])

    def test_empty_reply_is_not_a_silent_success(self):
        history = [Message(role="user", content="hello?")]
        _w, finished = self.run_turn(history, [("text", "")])
        self.assertEqual(len(finished), 1)
        self.assertFalse(finished[0][0])
        self.assertIn("empty reply", finished[0][1])
        self.assertEqual(len(history), 1)

    def test_stop_mid_round_answers_every_call(self):
        history = [Message(role="user", content="two things")]
        worker_ref = {}

        def stop_after_first(name, args):
            worker_ref["w"].stop()
            return "done", None, False

        worker = agent.AgentWorker(history)
        worker_ref["w"] = worker
        worker.provider = ScriptedProvider([
            ("tools", [ToolCall(id="a", name="list_documents"),
                       ToolCall(id="b", name="list_documents")]),
        ])
        with mock.patch.object(agent._tools, "run_tool", stop_after_first):
            with self.assertRaises(StopRequested):
                worker._run_turn()
        calls = {c.id for m in history for c in m.tool_calls}
        results = {r.tool_call_id for m in history for r in m.tool_results}
        self.assertEqual(calls, results)

    def test_truncated_arguments_are_not_executed(self):
        history = [Message(role="user", content="script it")]
        ran = []
        _w, _f = self.run_turn(history, [
            ("tools", [ToolCall(id="c1", name="execute_code", arguments={"_raw": '{"code": "pri'})]),
            ("text", "retrying later"),
        ], run_tool=lambda name, args: ran.append(name) or ("x", None, False))
        self.assertEqual(ran, [])
        result = history[2].tool_results[0]
        self.assertTrue(result.is_error)
        self.assertIn("not valid JSON", result.content)

    def test_one_screenshot_per_round(self):
        history = [Message(role="user", content="show me three views")]   # migrated: intent other
        _w, _f = self.run_turn(history, [
            ("tools", [ToolCall(id=f"c{i}", name="get_view") for i in range(3)]),
            ("text", "done"),
        ], run_tool=lambda name, args: ("shot", "QUJD", False))
        images = [r.image_b64 for r in history[2].tool_results]
        self.assertEqual(images, ["", "", "QUJD"])

    def test_system_prompt_is_static_and_state_rides_in_the_message(self):
        doc = App.newDocument("StateDoc")
        try:
            history = [Message(role="user", content="add a box")]
            worker, _f = self.run_turn(history, [
                ("tools", [ToolCall(id="c1", name="create_object", arguments={
                    "doc_name": "StateDoc", "obj_type": "Part::Box", "obj_name": "B",
                    "include_screenshot": False})]),
                ("text", "done"),
            ])
            self.assertEqual(len(set(worker.provider.systems)), 1)
            self.assertNotIn("StateDoc", worker.provider.systems[0])
            self.assertIn("<workspace_state>", history[0].content)
            self.assertIn("StateDoc", history[0].content)
        finally:
            App.closeDocument(doc.Name)

    def test_round_cap_ends_with_a_summary(self):
        history = [Message(role="user", content="loop forever")]
        rounds = [("tools", [ToolCall(id=f"c{i}", name="list_documents")])
                  for i in range(agent.MAX_TOOL_ROUNDS)]
        rounds.append(("text", "Summary: stopped early."))
        worker, finished = self.run_turn(history, rounds,
                                         run_tool=lambda name, args: ("[]", None, False))
        self.assertEqual(finished, [(True, "Summary: stopped early.")])
        self.assertIn("Tool budget", history[-2].content)

    def test_compaction_starts_at_a_real_user_turn(self):
        history = []
        for i in range(40):
            history.append(Message(role="user", content=f"q{i}"))
            history.append(Message(role="assistant", tool_calls=[ToolCall(id=f"x{i}", name="t")],
                                   provider_blocks=[{"type": "thinking", "thinking": "", "signature": "S"},
                                                    {"type": "tool_use", "id": f"x{i}", "name": "t", "input": {}}]))
            history.append(Message(role="user", tool_results=[
                ToolResult(tool_call_id=f"x{i}", name="t", content="ok", image_b64="QUJD")]))
        history.append(Message(role="user", content="now"))
        self.assertTrue(agent.compact_history(history))
        self.assertLessEqual(len(history), agent.KEEP_HISTORY_MESSAGES)
        self.assertEqual(history[0].role, "user")
        self.assertFalse(history[0].tool_results)
        self.assertFalse(any(b.get("type") == "thinking"
                             for m in history for b in m.provider_blocks))
        shots = sum(1 for m in history for r in m.tool_results if r.image_b64)
        self.assertLessEqual(shots, agent.KEEP_SCREENSHOTS)


if __name__ == "__main__":
    unittest.main()
