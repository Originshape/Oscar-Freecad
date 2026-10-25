"""oscar.agent over a real socket: the wiring against the REAL providers (``support.server.ScriptedServer``).

The scripted-provider tests prove what the loop hands to ``provider.chat``; these prove what actually goes on the wire
once the real request builder has seen it: the tools array per round, the static system message, the derived view
(no stale governor lines), the reasoning / tool-choice options of the caps, and the truncated-argument path through the
real stream parser. Tools are the ``FakeTools`` of ``test_agent_loop`` (no documents are touched).
"""

import json
import os
import sys
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import agent, params, prompt, providers, tools  # noqa: E402
from oscar.flow import AgentConfig  # noqa: E402
from oscar.models import ModelCaps  # noqa: E402
from oscar.providers import Message  # noqa: E402
from support import personas  # noqa: E402
from support.scripted_llm import bad_json, call, calls, say  # noqa: E402
from support.server import ScriptedServer  # noqa: E402
from support.signal_recorder import SignalRecorder  # noqa: E402
from test_agent_loop import FakeTools, PI_PROMPT, pi_script, pi_tools  # noqa: E402


def system_of(request) -> str:
    body = request.json
    if body.get("system"):
        return body["system"] if isinstance(body["system"], str) else json.dumps(body["system"])
    first = (body.get("messages") or [{}])[0]
    return first.get("content", "") if first.get("role") == "system" else ""


def user_texts(request) -> list[str]:
    out = []
    for message in request.json.get("messages") or []:
        if message.get("role") == "user":
            content = message.get("content")
            out.append(content if isinstance(content, str) else json.dumps(content))
    return out


class WireCase(unittest.TestCase):
    def serve(self, script, **kw):
        server = ScriptedServer(script, **kw)
        self.addCleanup(server.shutdown)
        return server

    def run_agent(self, server, prompt_text, *, caps=None, fake=None, override=None, config=None, **kw):
        prof = params.Profile(name=f"wire-{uuid.uuid4().hex[:6]}", provider="anthropic" if server.dialect == "anthropic" else "openai_compat",
                              base_url=server.base_url, model=f"wire-{uuid.uuid4().hex[:8]}", api_key="key", max_tokens=1024,
                              caps_override=dict(override or {}))
        history = [Message(role="user", content=prompt_text)]
        provider = providers.make_provider(prof)
        worker = agent.AgentWorker(history, provider=provider, profile=prof, caps=caps, config=config or AgentConfig(), **kw)
        self.rec = SignalRecorder(worker)
        self.fake = fake or FakeTools()
        with mock.patch.object(agent._tools, "run_tool", lambda n, a: self.fake(n, a)):
            result = worker.run_turn()
        self.history = history
        self.worker = worker
        return result


class ToolsAndSystemOnTheWireTest(WireCase):
    def test_the_tools_array_grows_and_the_system_message_never_changes(self):
        server = self.serve(pi_script())
        result = self.run_agent(server, PI_PROMPT, fake=pi_tools(), validator=False)
        self.assertEqual(result.outcome, "answered")
        server.assert_no_unexpected()
        names = [set(r.tool_names) for r in server.requests]
        self.assertGreaterEqual(len(names), 5)
        for earlier, later in zip(names, names[1:]):
            self.assertLessEqual(earlier, later)
        self.assertLess(names[0], names[-1])
        self.assertEqual(len({system_of(r) for r in server.requests}), 1)
        self.assertIn("EXECUTION CONTRACT", system_of(server.requests[0]))
        self.assertNotIn("download_file", system_of(server.requests[0]))             # hidden for the enclosure intent

    def test_the_lite_prompt_and_tools_reach_a_small_model(self):
        server = self.serve([calls(call("create_object")), say("Done.")])
        caps = ModelCaps(context_window=8192, max_output=1024)
        result = self.run_agent(server, "make a plate", caps=caps)
        self.assertEqual(result.outcome, "answered")
        lite = {s.name for s in tools.specs(lite=True)}
        for request in server.requests:
            self.assertTrue(set(request.tool_names) <= lite)
        self.assertLess(len(system_of(server.requests[0])), len(prompt.CORE))

    def test_only_the_newest_governor_line_travels(self):
        server = self.serve([calls(call("create_object")), calls(call("edit_object")), calls(call("edit_object", n=2)), say("Done.")])
        self.run_agent(server, "make a plate", validator=False)
        last = server.requests[-1]
        carriers = [t for t in user_texts(last) if "[governor]" in t]
        self.assertEqual(len(carriers), 1)
        self.assertIn("round 3/30", carriers[0])

    def test_the_seeded_plan_is_in_the_first_user_message_on_the_wire(self):
        server = self.serve([calls(call("create_object")), say("Done.")])
        self.run_agent(server, "make a plate", validator=False)
        first = user_texts(server.requests[0])[0]
        self.assertIn("<workspace_state>", first)
        self.assertIn("PLAN (intent generic_part", first)
        self.assertIn("make a plate", first)
        self.assertEqual(user_texts(server.requests[1])[0], first)                   # byte-identical prefix next round


class OptionsOnTheWireTest(WireCase):
    KNOB = {"reasoning": "hidden", "reasoning_control": "openai_effort"}

    def test_reasoning_is_medium_then_low_and_the_execution_rounds_are_capped(self):
        server = self.serve([calls(call("create_object")), say("Done.")])
        self.run_agent(server, "make a plate", override=self.KNOB, validator=False)
        first, second = (r.json for r in server.requests[:2])
        self.assertEqual(first.get("reasoning_effort"), "medium")
        self.assertEqual(second.get("reasoning_effort"), "low")

    def test_a_model_without_a_knob_gets_no_reasoning_parameter(self):
        server = self.serve([calls(call("create_object")), say("Done.")])
        self.run_agent(server, "make a plate", validator=False)
        for request in server.requests:
            self.assertNotIn("reasoning_effort", request.json)

    def test_the_forced_tool_of_the_second_nudge_reaches_the_wire(self):
        server = self.serve([say("plan"), say("plan"), calls(call("create_object")), say("Done.")])
        self.run_agent(server, "make a plate", validator=False)
        third = server.requests[2].json
        self.assertEqual((third.get("tool_choice") or {}).get("function", {}).get("name") or third.get("tool_choice"), "create_object")
        self.assertNotIn("tool_choice", server.requests[0].json)


class AnthropicThinkingOnTheWireTest(WireCase):
    """Thinking blocks of the open tool loop must survive the derived view (Anthropic answers HTTP 400 otherwise)."""

    def test_the_thinking_of_the_open_loop_is_replayed_even_when_older_rounds_were_dropped(self):
        script = [calls(call("create_object", obj_name=f"B{i}"), reasoning=f"considering box {i}") for i in range(7)] + [
            say("Done.", reasoning="all good")]
        server = self.serve(script, dialect="anthropic")
        caps = ModelCaps(context_window=20000, max_output=2048, reasoning="stream", reasoning_control="anthropic_budget",
                         reasoning_replay=True, tool_choice_required_with_reasoning=False)
        self.run_agent(server, "make a plate", caps=caps, fake=FakeTools(texts={"create_object": "z" * 30000}), validator=False)
        server.assert_no_unexpected()
        last = server.requests[-1].json
        self.assertEqual(last["messages"][0]["role"], "user")
        assistants = [m for m in last["messages"] if m["role"] == "assistant" and isinstance(m["content"], list)]
        newest_with_tool = [m for m in assistants if any(b.get("type") == "tool_use" for b in m["content"])][-1]
        self.assertEqual(newest_with_tool["content"][0]["type"], "thinking")                # the open loop keeps its thinking
        self.assertLess(len(last["messages"]), len(self.history))                            # and older rounds really were dropped
        self.assertTrue(any("<task_memory>" in t for t in user_texts(server.requests[-1])))


class WeakModelOnTheWireTest(WireCase):
    def test_a_truncated_script_is_parsed_as_truncated_and_never_run(self):
        server = self.serve(personas.build("giant_script_truncated"))
        result = self.run_agent(server, personas.prompt_for("giant_script_truncated"), validator=False)
        self.assertEqual((result.ok, result.outcome), (True, "answered"))
        self.assertEqual(self.fake.names().count("execute_code"), 1)
        first = [m for m in self.history if m.tool_results][0].tool_results[0]
        self.assertIn("not valid JSON", first.content)
        self.assertIn("under 40 lines", first.content)
        server.assert_no_unexpected()

    def test_plan_only_over_the_wire_fails_after_three_requests(self):
        server = self.serve(personas.build("plan_only"))
        result = self.run_agent(server, personas.prompt_for("plan_only"))
        self.assertEqual((result.ok, result.outcome), (False, "failed"))
        self.assertEqual(server.request_count, 3)
        server.assert_no_unexpected()

    def test_the_view_that_shrank_still_alternates_roles_on_the_wire(self):
        script = [calls(call("create_object", obj_name=f"B{i}")) for i in range(7)] + [say("Done.")]
        server = self.serve(script)
        self.run_agent(server, "make a plate", fake=FakeTools(texts={"create_object": "z" * 9000}),
                       caps=ModelCaps(context_window=20000, max_output=2048), validator=False)
        last = server.requests[-1].json["messages"]
        roles = [m["role"] for m in last if m["role"] != "system"]
        self.assertEqual(roles[0], "user")
        self.assertTrue(any("<task_memory>" in t for t in user_texts(server.requests[-1])))
        self.assertLess(len(last), len(self.history))


if __name__ == "__main__":
    unittest.main()
