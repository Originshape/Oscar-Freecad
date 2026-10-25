"""oscar.agent against the scripted persona harness: the loop must give a defined, honest outcome for
every weak-model behaviour, with the full wiring (request view, per-round tools, memory) in the path.

The personas of ``tests/support/personas.py`` are the model. The tools are the ``FakeTools`` of ``test_agent_loop`` (they
fill ``ctx.outcome`` and tick the ledger like the real pipeline); a few tests use a REAL ``Project`` so the dossier is
checked too. Every test names the failure it prevents: a weak model must never be able to end a turn "green" without
work, loop forever, or leave a call without its result.
"""

import json
import os
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import agent, context  # noqa: E402
from oscar.flow import AgentConfig, WatchdogConfig  # noqa: E402
from oscar.flow_run import NO_CHANGES_TEXT  # noqa: E402
from oscar.models import ModelCaps  # noqa: E402
from support import personas  # noqa: E402
from support.fakes import fake_validation, temp_project  # noqa: E402
from support.scripted_llm import Round, ScriptedProvider, call, calls, say  # noqa: E402
from test_agent_loop import FakeTools, LoopCase, profile, types_clock  # noqa: E402

KNOB = ModelCaps(reasoning="stream", reasoning_control="openai_effort")


def persona_tools(**kw):
    return FakeTools(**kw)


class PlanOnlyTest(LoopCase):
    """A model that describes the plan and never calls a tool (the audit's most common weak-model failure)."""

    def test_plan_only_twice_is_a_failed_turn_never_a_green_one(self):
        result = self.run_turn(personas.prompt_for("plan_only"), personas.build("plan_only"))
        self.assertEqual((result.ok, result.outcome), (False, "failed"))
        self.assertTrue(result.text.startswith(NO_CHANGES_TEXT))
        self.assertEqual(result.geometry_ok, 0)
        self.assertEqual(len(self.provider.requests), 3)                    # the plan, nudge 1, nudge 2 (forced tool)
        self.assertEqual(self.rec.of("turn_outcome")[0][0], "failed")
        self.assertEqual(self.rec.of("turn_finished")[0][0], False)

    def test_the_second_nudge_forces_a_tool_the_model_was_offered(self):
        self.run_turn(personas.prompt_for("plan_only"), personas.build("plan_only"))
        kinds = [r.kwargs.get("tool_choice") for r in self.provider.requests]
        self.assertEqual(kinds[:2], [None, None])
        self.assertIn(kinds[2], set(self.provider.requests[2].tools))

    def test_the_plan_text_is_kept_in_the_history_and_every_nudge_is_marked_synthetic(self):
        self.run_turn(personas.prompt_for("plan_only"), personas.build("plan_only"))
        notes = self.notes("gate")
        self.assertEqual(len(notes), 2)
        for note in notes:
            index = self.history.index(note)
            self.assertEqual(self.history[index - 1].role, "assistant")
            self.assertIn("Here is my plan", self.history[index - 1].content)
        self.assertEqual(context.alternation_problems(self.provider.requests[-1].messages), [])

    def test_a_plan_that_arrives_after_a_real_call_is_still_the_models_answer(self):
        script = [calls(call("create_object")), say(personas.PLAN_TEXT)]
        result = self.run_turn("make a plate", script)
        self.assertEqual((result.ok, result.outcome), (True, "answered"))
        self.assertEqual(result.geometry_ok, 1)


class StallTest(LoopCase):
    """A reasoning model that never stops thinking."""

    def cfg(self):
        return AgentConfig(watchdog=WatchdogConfig(reasoning_chars_plan=3000, reasoning_chars_exec=3000))

    def flood(self, chars=20000, chunk=1000):
        return Round([("reasoning", "x" * chunk) for _ in range(chars // chunk)] + [("stop_reason", "length"), ("done", "length")])

    def test_a_stall_is_cut_off_and_the_retry_acts(self):
        script = personas.build("stalls_then_acts")
        result = self.run_turn(personas.prompt_for("stalls_then_acts"), script, caps=KNOB, config=self.cfg())
        self.assertEqual((result.ok, result.outcome), (True, "answered"))
        self.assertEqual([r.kwargs.get("reasoning") for r in self.provider.requests][:2], ["medium", "off"])
        self.assertEqual(self.worker._watchdog.trips, 1)
        self.assertEqual(self.rec.of("round_reset").count(("watchdog",)), 1)
        self.assertEqual(len(self.notes("watchdog")), 1)

    def test_a_model_that_never_stops_thinking_is_aborted_after_three_trips(self):
        result = self.run_turn("make a plate", [self.flood()] * 3, caps=KNOB, config=self.cfg())
        self.assertEqual((result.ok, result.outcome), (False, "needs_help"))
        self.assertIn("kept thinking instead of acting", result.text)
        self.assertEqual(len(self.provider.requests), 3)
        self.assertEqual(self.rec.of("turn_outcome")[0][0], "needs_help")

    def test_without_a_reasoning_knob_the_flood_ends_through_the_cutoff_path(self):
        script = personas.build("stalls_then_acts")
        result = self.run_turn(personas.prompt_for("stalls_then_acts"), script, caps=ModelCaps())
        self.assertEqual(result.outcome, "answered")
        self.assertEqual(self.worker._watchdog.trips, 0)
        self.assertEqual(len(self.notes("cutoff")), 1)

    def test_cutoff_then_plan_is_nudged_twice_then_acts(self):
        result = self.run_turn(personas.prompt_for("cutoff_then_plan"), personas.build("cutoff_then_plan"))
        self.assertEqual((result.ok, result.outcome), (True, "answered"))
        self.assertEqual(len(self.notes("cutoff")), 1)
        self.assertEqual(len(self.notes("gate")), 1)
        self.assertEqual(result.geometry_ok, 1)

    def test_a_silent_model_is_cut_after_the_silent_budget_of_its_host(self):
        clock = types_clock()
        provider = ScriptedProvider(personas.build("hidden_reasoning", silent_s=150.0), advance=clock.advance)
        history = [agent.Message(role="user", content=personas.prompt_for("hidden_reasoning"))]
        cfg = AgentConfig(watchdog=WatchdogConfig())
        prof = profile()
        prof.base_url = "https://api.openai.com/v1"        # a hosted host: the standard 90 s silence budget applies
        worker = agent.AgentWorker(history, provider=provider, profile=prof, caps=KNOB, config=cfg, clock=clock)
        with mock.patch.object(agent._tools, "run_tool", FakeTools()):
            worker.run_turn()
        self.assertEqual(worker._watchdog.trips, 1)
        self.assertEqual(worker._watchdog.last_reason, "silent")

    def test_a_slow_local_host_is_not_cut(self):
        clock = types_clock()
        provider = ScriptedProvider(personas.build("slow_local", seconds=200.0), advance=clock.advance)
        history = [agent.Message(role="user", content=personas.prompt_for("slow_local"))]
        prof = profile()
        prof.base_url = "http://localhost:11434/v1"
        worker = agent.AgentWorker(history, provider=provider, profile=prof, caps=ModelCaps(reasoning_control="ollama_effort"),
                                   config=AgentConfig(), clock=clock)
        with mock.patch.object(agent._tools, "run_tool", FakeTools()):
            result = worker.run_turn()
        self.assertEqual(worker._watchdog.trips, 0)
        self.assertEqual(result.outcome, "answered")


class WeakArgumentsTest(LoopCase):
    """Malformed and truncated tool-call arguments."""

    def test_a_giant_truncated_script_is_not_run_and_the_model_is_told_to_split(self):
        events = []
        with mock.patch.object(agent, "log_event", lambda kind, **f: events.append((kind, f))):
            result = self.run_turn(personas.prompt_for("giant_script_truncated"), personas.build("giant_script_truncated"))
        self.assertEqual((result.ok, result.outcome), (True, "answered"))
        self.assertEqual(self.fake.names().count("execute_code"), 1)           # only the small script ever ran
        first = self.results_messages()[0].tool_results[0]
        self.assertTrue(first.is_error)
        self.assertIn("not valid JSON", first.content)
        self.assertIn("under 40 lines", first.content)
        self.assertIn("breaker_trip", [k for k, _f in events])
        self.assertEqual([f["rule"] for k, f in events if k == "breaker_trip"], ["truncation"])
        self.assertTrue(any("shorter" in n.content or "smaller" in n.content for n in self.notes_in_results()))

    def notes_in_results(self):
        return [m for m in self.history if m.tool_results and m.content]

    def test_every_call_of_a_bad_round_gets_exactly_one_result(self):
        self.run_turn(personas.prompt_for("giant_script_truncated"), personas.build("giant_script_truncated"))
        for message in self.history:
            if message.role == "assistant" and message.tool_calls:
                following = self.history[self.history.index(message) + 1]
                self.assertEqual([r.tool_call_id for r in following.tool_results], [c.id for c in message.tool_calls])

    def test_malformed_json_is_reported_as_malformed_not_as_truncated(self):
        result = self.run_turn(personas.prompt_for("bad_json_args"), personas.build("bad_json_args"))
        self.assertEqual(result.outcome, "answered")
        first = self.results_messages()[0].tool_results[0]
        self.assertTrue(first.is_error)
        self.assertIn("Fix the JSON", first.content)
        self.assertNotIn("output-token limit", first.content)
        self.assertEqual(self.fake.names()[:1], ["create_document"])             # the bad call never reached the tools

    def test_text_tool_calls_that_the_provider_did_not_parse_are_just_an_answer(self):
        # a weak model that writes its call as plain text: the loop sees a text-only reply and nudges for real work
        text = '<tool_call>{"name": "create_object", "arguments": {}}</tool_call>'
        result = self.run_turn("make a plate", [say(text), say(text), calls(call("create_object")), say("Done.")])
        self.assertEqual(result.outcome, "answered")
        self.assertEqual(result.geometry_ok, 1)
        self.assertEqual(len(self.notes("gate")), 2)


class HonestOutcomeTest(LoopCase):
    """Claims, questions and loops must map to their prescribed outcomes."""

    def test_claiming_steps_done_without_work_is_unverified(self):
        result = self.run_turn(personas.prompt_for("claims_done_without_work"), personas.build("claims_done_without_work"),
                               tools_=FakeTools(), validator=lambda d, c: fake_validation(True))
        self.assertEqual((result.ok, result.outcome), (True, "unverified"))
        self.assertIn("claimed without recorded work", result.text)
        self.assertTrue(result.text.startswith("NOT VERIFIED"))

    def test_asking_the_user_for_a_file_is_needs_input_not_a_failure(self):
        result = self.run_turn(personas.prompt_for("asks_the_user"), personas.build("asks_the_user"), tools_=FakeTools())
        self.assertEqual((result.ok, result.outcome), (True, "needs_input"))
        self.assertIn("attach the STEP file", result.text)
        self.assertEqual(len(self.provider.requests), 2)

    def test_a_loop_of_reads_is_stopped_with_one_note_then_needs_help(self):
        result = self.run_turn(personas.prompt_for("loops_on_reads"), personas.build("loops_on_reads"))
        self.assertEqual((result.ok, result.outcome), (False, "needs_help"))
        self.assertLessEqual(len(self.provider.requests), 14)
        self.assertTrue(any("stall" in (m.content or "").lower() or "no change" in (m.content or "").lower()
                            for m in self.history if m.tool_results))
        self.assertEqual(self.fake.names()[0], "create_document")
        self.assertTrue(set(self.fake.names()[1:]) <= {"get_objects", "get_view"})            # it never changed anything again

    def test_the_same_failing_call_over_and_over_ends_needs_help(self):
        tools_ = FakeTools(fail={"execute_code": True})
        result = self.run_turn(personas.prompt_for("tool_loop_repeat"), personas.build("tool_loop_repeat"), tools_=tools_)
        self.assertEqual((result.ok, result.outcome), (False, "needs_help"))
        self.assertLessEqual(tools_.names().count("execute_code"), 8)

    def test_a_model_that_only_ever_returns_nothing_fails_visibly(self):
        result = self.run_turn("hello", personas.build("empty_reply", limit=6))
        self.assertEqual((result.ok, result.outcome), (False, "failed"))
        self.assertIn("empty reply", result.text)

    def test_a_work_turn_of_empty_replies_is_a_failure_after_the_nudges(self):
        result = self.run_turn("make a plate", personas.build("empty_reply", limit=8))
        self.assertEqual((result.ok, result.outcome), (False, "failed"))

    def test_a_tool_then_stop_persona_is_nudged_with_the_pending_step_ids(self):
        result = self.run_turn(personas.prompt_for("tool_then_stop"), personas.build("tool_then_stop"), tools_=FakeTools())
        self.assertEqual(result.outcome, "unverified")
        gate = self.notes("gate")
        self.assertTrue(gate)
        self.assertIn("STEPS_PENDING", gate[0].content)
        self.assertRegex(gate[0].content, r"S\d")

    def test_a_non_english_request_acts_in_round_one_and_escalates(self):
        result = self.run_turn(personas.prompt_for("plan_in_other_language"), personas.build("plan_in_other_language"))
        self.assertEqual((result.ok, result.outcome), (True, "answered"))
        self.assertEqual(self.worker.ledger.intent, "generic_part")
        self.assertIn("create_object", set(self.provider.requests[0].tools))
        self.assertIn("execute_code", set(self.provider.requests[0].tools))
        self.assertEqual(result.geometry_ok, 1)


class RealProjectPersonaTest(LoopCase):
    """The same weak models with a REAL project folder: the dossier stays consistent whatever the model does."""

    def test_a_failed_plan_only_turn_leaves_a_ledger_and_no_summary_claim(self):
        project = temp_project(chat_id="20261002-111111-p1a1a1")
        result = self.run_turn(personas.prompt_for("plan_only"), personas.build("plan_only"), project=project)
        self.assertEqual(result.outcome, "failed")
        self.assertEqual(project.read_ledger()["intent"], "generic_part")
        self.assertNotIn("## Build Summary", result.text)
        self.assertEqual([s.tool for s in project.read_steps()], [])

    def test_a_needs_input_turn_does_not_finalize_the_project(self):
        project = temp_project(chat_id="20261002-111111-p2a2a2")
        result = self.run_turn(personas.prompt_for("asks_the_user"), personas.build("asks_the_user"), project=project,
                               tools_=FakeTools())
        self.assertEqual(result.outcome, "needs_input")
        self.assertIsNone(project.manifest.get("finalized"))
        self.assertEqual(project.read_ledger()["steps"][1]["status"], "blocked")

    def test_an_unverified_turn_still_gets_the_real_build_summary(self):
        project = temp_project(chat_id="20261002-111111-p3a3a3")
        result = self.run_turn(personas.prompt_for("claims_done_without_work"), personas.build("claims_done_without_work"),
                               project=project, tools_=FakeTools(), validator=lambda d, c: fake_validation(True))
        self.assertEqual(result.outcome, "unverified")
        self.assertIn("## Build Summary", result.text)
        self.assertEqual(project.manifest["finalized"]["outcome"], "unverified")


class StopAndSteerTest(LoopCase):
    def test_stop_while_paused_runs_no_tool_and_the_project_records_a_stopped_turn(self):
        project = temp_project(chat_id="20261002-111111-p4a4a4")
        worker_ref = {}

        def script(record):
            worker_ref["w"].pause()
            threading.Timer(0.3, worker_ref["w"].stop).start()
            return calls(call("create_object"), call("edit_object"))

        worker = self.build("make a plate", script, project=project)
        worker_ref["w"] = worker
        with self.assertRaises(agent.StopRequested):
            worker.run_turn()
        self.assertEqual(self.fake.names(), [])                                    # zero tools executed
        results = self.results_messages()[0].tool_results
        self.assertEqual([r.tool_call_id for r in results], [c.id for c in self.history[1].tool_calls])
        self.assertTrue(all(r.is_error for r in results))
        with open(project.path("steps.jsonl"), encoding="utf-8") as f:
            ends = [json.loads(line) for line in f if '"turn_end"' in line]
        self.assertEqual([e["outcome"] for e in ends], ["stopped"])
        self.assertEqual(project.read_steps(), [])
        self.assertTrue(os.path.isfile(project.path("ledger.json")))

    def test_a_steer_message_reaches_the_first_request_of_a_persona_run(self):
        script = [calls(call("create_object")), calls(call("edit_object")), say("Done.")]
        worker = self.build("make a plate", script)
        worker.steer("make it 5 mm thick", note="[2nd message]")
        worker.run_turn()
        first_user = self.provider.requests[0].messages[0].content
        self.assertIn("[USER, sent while you were working]: [2nd message] make it 5 mm thick", first_user)
        self.assertEqual(worker.drain_unsent(), [])

    def test_a_steer_during_a_long_build_survives_the_request_view(self):
        script = [calls(call("create_object", obj_name=f"B{i}")) for i in range(6)] + [say("Done.")]
        worker = self.build("make a plate", script, tools_=FakeTools(texts={"create_object": "y" * 9000}),
                            caps=ModelCaps(context_window=20000, max_output=2048))
        calls_seen = []
        original = worker._provider_round

        def wrapped():
            calls_seen.append(1)
            if len(calls_seen) == 4:
                worker.steer("also add a 3 mm chamfer", note="")
            return original()

        worker._provider_round = wrapped
        worker.run_turn()
        last = " ".join(m.content or "" for m in self.provider.requests[-1].messages)
        self.assertIn("also add a 3 mm chamfer", last)


class OutcomeTableTest(LoopCase):
    """every outcome is reachable, with the ``ok`` flag of the table and ``turn_outcome`` before ``turn_finished``."""

    def check(self, outcome, ok):
        self.assertEqual(self.rec.of("turn_outcome")[0][0], outcome)
        self.assertEqual(self.rec.of("turn_finished")[0][0], ok)
        self.rec.assert_ordered("turn_outcome", "turn_finished")

    def test_answered(self):
        result = self.run_turn("what is a fillet?", [say("A rounded edge.")])
        self.assertEqual((result.ok, result.outcome, result.text), (True, "answered", "A rounded edge."))
        self.check("answered", True)

    def test_validated(self):
        result = self.run_turn("make a plate", [calls(call("create_object")), say("Done.")],
                               validator=lambda d, c: fake_validation(True))
        self.assertEqual((result.ok, result.outcome), (True, "validated"))
        self.check("validated", True)

    def test_needs_input(self):
        result = self.run_turn(personas.prompt_for("asks_the_user"), personas.build("asks_the_user"))
        self.assertEqual((result.ok, result.outcome), (True, "needs_input"))
        self.check("needs_input", True)

    def test_unverified(self):
        result = self.run_turn(personas.prompt_for("claims_done_without_work"), personas.build("claims_done_without_work"),
                               validator=lambda d, c: fake_validation(True))
        self.assertEqual((result.ok, result.outcome), (True, "unverified"))
        self.check("unverified", True)

    def test_capped(self):
        script = [calls(call("list_documents")), calls(call("list_documents")), calls(call("list_documents")), say("Stopped early.")]
        result = self.run_turn("make a plate", script, config=AgentConfig(max_tool_rounds=3))
        self.assertEqual((result.ok, result.outcome), (True, "capped"))
        self.check("capped", True)

    def test_failed(self):
        result = self.run_turn(personas.prompt_for("plan_only"), personas.build("plan_only"))
        self.assertEqual((result.ok, result.outcome), (False, "failed"))
        self.check("failed", False)

    def test_needs_help(self):
        result = self.run_turn(personas.prompt_for("loops_on_reads"), personas.build("loops_on_reads"))
        self.assertEqual((result.ok, result.outcome), (False, "needs_help"))
        self.check("needs_help", False)

    def test_stopped(self):
        worker = self.build("make a plate", [calls(call("create_object")), say("Done.")], tools_=FakeTools(
            on_call=lambda name, args: worker.stop()))
        worker.run()
        self.assertEqual(self.rec.of("turn_outcome"), [("stopped", "")])
        self.assertEqual(self.rec.of("turn_finished"), [(False, "Stopped.")])
        self.rec.assert_ordered("turn_outcome", "turn_finished")

    def test_a_provider_error_is_failed_with_the_message(self):
        from support.scripted_llm import provider_error

        result = self.run_turn("hello", [provider_error("HTTP 500: upstream down")])
        self.assertEqual((result.ok, result.outcome), (False, "failed"))
        self.assertIn("HTTP 500: upstream down", result.text)
        self.check("failed", False)


class ModuleSanityTest(unittest.TestCase):
    def test_every_persona_used_here_exists(self):
        for name in ("plan_only", "stalls_then_acts", "giant_script_truncated", "bad_json_args", "claims_done_without_work",
                     "asks_the_user", "loops_on_reads", "tool_loop_repeat", "empty_reply", "tool_then_stop",
                     "plan_in_other_language", "hidden_reasoning", "slow_local", "cutoff_then_plan"):
            self.assertIn(name, personas.PERSONAS)


if __name__ == "__main__":
    unittest.main()
