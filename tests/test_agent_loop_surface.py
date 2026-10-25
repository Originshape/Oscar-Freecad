"""oscar.agent compatibility surface and the plan tools (split out of test_agent_loop to keep each file under 1500 lines).

``LegacyCompatTest`` pins the legacy module-level names of ``oscar.agent``; ``PlanToolsTest`` drives the
two plan tools through the real ``tools.run_tool`` and cross-checks the pure ``flow`` tool-category table against the live
registry (that check needs FreeCAD, ``tools`` imports ``oscar.cad``).
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

from oscar import agent, flow, loopcore, params, tools  # noqa: E402
from oscar.flow import Ledger  # noqa: E402
from oscar.providers import Message  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402
from support.scripted_llm import ScriptedProvider, say  # noqa: E402


def profile():
    return params.Profile(name="t", provider="openai_compat", model="scripted", base_url="http://127.0.0.1:9/v1")


class LegacyCompatTest(unittest.TestCase):
    def test_legacy_names_still_import_from_agent(self):
        for name in ("AgentWorker", "MAX_TOOL_ROUNDS", "MAX_CUTOFF_RETRIES", "MAX_HISTORY_MESSAGES", "KEEP_HISTORY_MESSAGES",
                     "KEEP_SCREENSHOTS", "CONTEXT_CHAR_BUDGET", "TOOL_RESULT_KEEP_CHARS", "PERSONA", "STRATEGY",
                     "build_system_prompt", "workspace_snapshot", "compact_history", "stub_oversized_history",
                     "_message_size", "needs_execution_nudge", "_EXECUTION_NOTE", "_CUTOFF_NOTE", "_BUDGET_NOTE",
                     "_BAD_ARGUMENTS", "_TRIMMED_NOTE", "_tools", "_summarize_arguments", "_keep_last_screenshot", "_cut_off"):
            self.assertTrue(hasattr(agent, name), name)
        self.assertIs(agent._tools, tools)
        self.assertEqual(agent.MAX_TOOL_ROUNDS, 30)
        self.assertEqual(agent.MAX_CUTOFF_RETRIES, 2)
        self.assertIn("Tool budget", agent._BUDGET_NOTE)
        self.assertEqual(agent._CUTOFF_NOTE, loopcore.CUTOFF_NOTE)
        self.assertTrue(agent._cut_off("length") and agent._cut_off("max_tokens") and not agent._cut_off("stop"))

    def test_worker_surface(self):
        history = [Message(role="user", content="hi")]
        worker = agent.AgentWorker(history, provider=ScriptedProvider([]), profile=profile())
        for name in ("history", "provider", "profile", "_stop", "_cutoffs", "_vision_notice_sent", "stop", "pause", "resume",
                     "_wait_if_paused", "_should_stop", "_run_turn", "_turn_request", "_finish", "_run_tools", "run_turn",
                     "steer", "drain_unsent", "act_now", "ledger"):
            self.assertTrue(hasattr(worker, name), name)
        for sig in ("text_delta", "reasoning_delta", "tool_started", "tool_finished", "turn_finished", "notice", "flow_paused",
                    "step_started", "step_finished", "plan_changed", "progress", "round_reset", "artifacts_changed",
                    "turn_outcome"):
            self.assertTrue(hasattr(worker, sig), sig)
        self.assertEqual(worker.ledger.steps, [])

    def test_constructor_needs_no_profile_when_a_provider_is_injected(self):
        with mock.patch.object(params, "active_profile", lambda: None):
            worker = agent.AgentWorker([Message(role="user", content="hi")], provider=ScriptedProvider([]))
        self.assertEqual(worker.profile.provider, "openai_compat")
        with mock.patch.object(params, "active_profile", lambda: None):
            with self.assertRaises(RuntimeError):
                agent.AgentWorker([Message(role="user", content="hi")])

    def test_provider_and_profile_stay_assignable(self):
        worker = agent.AgentWorker([Message(role="user", content="hello?")], provider=ScriptedProvider([]), profile=profile())
        worker.provider = ScriptedProvider([say("swapped")])
        self.assertEqual(worker.run_turn().text, "swapped")

    def test_compaction_never_cuts_at_a_synthetic_note(self):
        history = []
        for i in range(30):
            history.append(Message(role="user", content=f"q{i}"))
            history.append(Message(role="assistant", content="a"))
            history.append(Message(role="user", content="[finish-gate] x", synthetic=True, tag="gate"))
            history.append(Message(role="assistant", content="b"))
        self.assertTrue(agent.compact_history(history, limit=20, keep=10))
        self.assertFalse(history[0].synthetic)
        self.assertTrue(history[0].content.startswith(agent._TRIMMED_NOTE) or "q" in history[0].content)

    def test_summarize_arguments_ignores_the_raw_marker(self):
        self.assertEqual(agent._summarize_arguments("execute_code", {"_raw": "{"}), "")
        self.assertEqual(agent._summarize_arguments("create_object", {"doc_name": "D", "obj_type": "Part::Box"}),
                         "doc_name=D, obj_type=Part::Box")


class PlanToolsTest(unittest.TestCase):
    def run_plan(self, ledger, name, **args):
        with use_context(ToolContext(ledger=ledger)):
            return tools.run_tool(name, args)

    def test_registry_flags_and_schemas(self):
        for name in ("plan_set", "plan_update"):
            entry = tools.tool_meta(name)
            self.assertIsNotNone(entry, name)
            self.assertEqual((entry.category, entry.mutating, entry.geometry), ("plan", False, False))
            self.assertEqual(set(entry.phases), set(tools.ALL_PHASES))
            self.assertIn(name, tools.CORE_TOOLS)
            self.assertGreater(len(entry.description), 40)
            self.assertEqual(entry.parameters["type"], "object")
            json.dumps(entry.parameters)
        schema = tools.tool_meta("plan_update").parameters
        self.assertEqual(schema["required"], ["id", "status"])
        self.assertEqual(schema["properties"]["status"]["enum"], ["pending", "running", "done", "blocked", "skipped", "changed"])
        steps = tools.tool_meta("plan_set").parameters["properties"]["steps"]["items"]["properties"]
        self.assertEqual(steps["phase"]["enum"], list(flow.PHASES))

    def test_the_category_table_of_flow_matches_the_registry(self):
        """flow is pure and carries its own copy of the tool -> category facts it needs; it must never drift from the matrix."""
        drift = [(t.name, t.category, flow.tool_category(t.name)) for t in tools.get_tools()
                 if flow.tool_category(t.name) != t.category]
        self.assertEqual(drift, [])

    def test_geometry_tools_get_a_category_the_ledger_can_place(self):
        """A geometry tool whose category the ledger does not know would attach evidence nowhere (5.5 rule 2)."""
        placeable = {"build", "source", "params", "fem", "session", "plan"}
        for t in tools.get_tools():
            if t.geometry:
                self.assertIn(t.category, placeable, t.name)

    def test_plan_tools_are_offered_in_every_phase(self):
        for phase in tools.ALL_PHASES:
            names = {s.name for s in tools.specs(phases=[phase])}
            self.assertTrue({"plan_set", "plan_update"} <= names, phase)

    def test_plan_update_changes_the_ledger_and_returns_the_plan(self):
        ledger = Ledger.from_request("make a plate")
        text, image, is_error = self.run_plan(ledger, "plan_update", id="S3", status="running", note="starting")
        self.assertFalse(is_error)
        self.assertIsNone(image)
        self.assertTrue(text.startswith('S3 "Model the part" -> running'))
        self.assertIn("[~] S3 Model the part", text)
        self.assertEqual(ledger.step("S3").status, "running")

    def test_plan_update_errors_are_the_contract_messages(self):
        ledger = Ledger.from_request("make a plate")
        text, _image, is_error = self.run_plan(ledger, "plan_update", id="S99", status="done")
        self.assertTrue(is_error)
        self.assertEqual(text, "plan_update: 'S99' is not in the plan. Plan ids: S1..S5, R1..R2.")
        text, _i, is_error = self.run_plan(ledger, "plan_update", id="S3", status="blocked")
        self.assertTrue(is_error)
        self.assertIn("'note' (the reason) is required", text)

    def test_plan_update_on_a_chat_turn(self):
        text, _i, is_error = self.run_plan(Ledger.from_request("hello?"), "plan_update", id="S1", status="done")
        self.assertTrue(is_error)
        self.assertIn("no plan", text)

    def test_without_a_ledger(self):
        text, _i, is_error = tools.run_tool("plan_set", {})
        self.assertTrue(is_error)
        self.assertIn("no plan is bound to this turn", text)

    def test_plan_set_edits_the_plan(self):
        ledger = Ledger.from_request("make a plate")
        text, _i, is_error = self.run_plan(ledger, "plan_set", steps=[{"title": "Add fillets", "phase": "BUILD"}],
                                           requirements=[{"text": "Edges are rounded"}])
        self.assertFalse(is_error)
        self.assertIn("S6 Add fillets", text)
        self.assertEqual(ledger.requirements[-1].text, "Edges are rounded")

    def test_plan_set_rejects_a_bad_phase_with_the_contract_message(self):
        ledger = Ledger.from_request("make a plate")
        text, _i, is_error = self.run_plan(ledger, "plan_set", steps=[{"title": "x", "phase": "Building"}])
        self.assertTrue(is_error)
        self.assertEqual(text, "plan_set: step 'S6' has phase 'Building'; use one of INTAKE|SOURCE|PLAN|PARAMS|BUILD|VERIFY|DELIVER.")

    def test_plan_set_accepts_json_strings_like_other_tools(self):
        ledger = Ledger.from_request("make a plate")
        text, _i, is_error = self.run_plan(ledger, "plan_set", steps=json.dumps([{"title": "Chamfer", "phase": "BUILD"}]))
        self.assertFalse(is_error, text)
        self.assertTrue(any(s.title == "Chamfer" for s in ledger.steps))

    def test_plan_tools_do_not_count_as_geometry_or_mutation(self):
        ledger = Ledger.from_request("make a plate")
        self.run_plan(ledger, "plan_update", id="S3", status="running")
        self.assertEqual(ledger.last_geometry_step, 0)


if __name__ == "__main__":
    unittest.main()
