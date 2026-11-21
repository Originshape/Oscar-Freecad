"""Trace replay: recorded failure shapes of real models, played against the NEW build flow.

``tests/fixtures/traces/*.json`` hold what the MODEL did in the failures of the audit (synthesized faithfully, no user data):
a five-minute reasoning round followed by one giant script, the spreadsheet int-vs-Quantity crash and its retry, partial
objects left behind by a failed script, and a plan-only reply. ``support/traces.py`` turns a trace into a script for
``ScriptedProvider``; the flow under test is the real one (real tools, real Project, real documents, an injected clock).
Each test names what the old flow did wrong and asserts what the new one does instead.
"""

import copy
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402

from unittest import mock  # noqa: E402

from oscar import agent, context  # noqa: E402
from oscar.models import ModelCaps  # noqa: E402
from support import traces  # noqa: E402
from support.realflow import RealFlowCase, scripted_profile  # noqa: E402
from support.scripted_llm import Round  # noqa: E402

WALL_BUDGET_S = 20.0       # the injected clock must make a "five minute" trace cost far less than this


class TraceFormatTest(unittest.TestCase):
    """The harness itself: loading, validation, substitution and the shape of the rounds it builds."""

    def test_the_fixture_traces_all_load_and_are_the_four_audit_shapes(self):
        names = traces.list_traces()
        for expected in ("reasoning_marathon_then_giant_script", "spreadsheet_int_quantity_crash",
                         "partial_objects_after_failed_script", "plan_only_reply"):
            self.assertIn(expected, names)
        for name in names:
            with self.subTest(trace=name):
                trace = traces.load_trace(name)
                self.assertEqual(trace["name"], name)
                self.assertTrue(trace["source"].startswith("audit"))
                self.assertTrue(all(isinstance(r, Round) for r in traces.script_from_trace(trace, "D")))

    def test_doc_placeholders_are_replaced_everywhere_and_nothing_else_is(self):
        trace = traces.load_trace("partial_objects_after_failed_script")
        script = traces.script_from_trace(trace, "MyDoc")
        blob = json.dumps([[repr(e) for e in r.events] for r in script])
        self.assertNotIn("{doc}", blob)
        self.assertIn("MyDoc", blob)
        self.assertEqual(traces.substitute({"a": ["x{doc}", 3, None]}, "Q"), {"a": ["xQ", 3, None]})

    def test_the_reasoning_marathon_trace_models_five_minutes_of_injected_time(self):
        trace = traces.load_trace("reasoning_marathon_then_giant_script")
        self.assertEqual(traces.reasoning_seconds(trace), 300.0)
        first = traces.script_from_trace(trace, "D")[0]
        self.assertEqual([e[0] for e in first.events].count("advance"), 20)
        self.assertEqual(sum(len(e[1]) for e in first.events if e[0] == "reasoning"), 6000)

    def test_a_bad_tool_call_round_is_a_truncated_call_with_a_giant_body(self):
        trace = traces.load_trace("reasoning_marathon_then_giant_script")
        rnd = traces.script_from_trace(trace, "D")[1]
        tool_calls = next(p for k, p in rnd.events if k == "tool_calls")
        self.assertEqual(tool_calls[0].name, "execute_code")
        self.assertIn("truncated", tool_calls[0].args_error)
        self.assertGreater(len(tool_calls[0].raw_args), 20000)
        self.assertEqual(("stop_reason", "max_tokens"), next(e for e in rnd.events if e[0] == "stop_reason"))

    def test_tool_names_lists_every_call_in_order(self):
        self.assertEqual(traces.tool_names(traces.load_trace("plan_only_reply")), ["create_document", "create_object"])

    def test_malformed_traces_are_rejected_with_a_message_naming_the_problem(self):
        base = traces.load_trace("plan_only_reply")
        cases = {
            "format": lambda t: t.update(format=2),
            "missing 'prompt'": lambda t: t.pop("prompt"),
            "has no events": lambda t: t["rounds"].__setitem__(0, {"events": []}),
            "unknown type": lambda t: t["rounds"][0]["events"].__setitem__(0, {"type": "shout"}),
            "needs a 'name'": lambda t: t["rounds"][2]["events"][0]["calls"].__setitem__(0, {"args": {}}),
            "needs 'raw'": lambda t: t["rounds"].append({"events": [{"type": "bad_tool_call", "name": "x"}]}),
        }
        for needle, mutate in cases.items():
            with self.subTest(problem=needle):
                bad = copy.deepcopy(base)
                mutate(bad)
                with self.assertRaises(traces.TraceError) as caught:
                    traces.validate(bad, "bad")
                self.assertIn(needle, str(caught.exception))

    def test_a_file_that_is_not_json_is_a_trace_error(self):
        directory = tempfile.mkdtemp(prefix="traces_")
        self.addCleanup(shutil.rmtree, directory, True)
        with open(os.path.join(directory, "broken.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        with self.assertRaises(traces.TraceError):
            traces.load_trace("broken", directory)
        self.assertEqual(traces.list_traces(directory), ["broken"])


class ReplayCase(RealFlowCase):
    """Replays a trace through the real flow and keeps the evidence on ``self``."""

    def replay(self, name: str, **kw):
        self.trace = traces.load_trace(name)
        self.doc = self.doc_name("Tr")
        self.project = self.new_project(self.trace["name"], chat_id="20261002-101010-e1e1e1")
        caps = ModelCaps.from_dict(self.trace.get("caps") or {})
        profile = scripted_profile(self.trace["profile"]["base_url"])
        script = traces.script_from_trace(self.trace, self.doc)
        started = time.monotonic()
        self.result = self.turn(self.trace["prompt"], script, project=self.project, caps=caps, profile=profile, **kw)
        self.wall = time.monotonic() - started
        self.provider.assert_exhausted()
        return self.result

    def results(self):
        """Every tool result the model was shown, in order."""
        return [r for m in self.history for r in m.tool_results]

    def objects(self):
        return [o.Name for o in App.getDocument(self.doc).Objects]


class ReasoningMarathonTest(ReplayCase):
    """Old flow: five minutes of reasoning, then one giant script cut at the token limit, nothing recoverable."""

    def setUp(self):
        super().setUp()
        self.events = []
        with mock.patch.object(agent, "log_event", lambda kind, **fields: self.events.append((kind, fields))):
            self.replay("reasoning_marathon_then_giant_script")

    def test_the_watchdog_cuts_the_five_minute_round_without_waiting_five_minutes(self):
        self.assertEqual(self.worker._watchdog.last_reason, "round_seconds")
        self.assertEqual(self.worker._watchdog.trips, 1)
        self.assertLess(self.wall, WALL_BUDGET_S)
        self.assertLess(self.clock.now - 1000.0, 400)         # the injected clock saw the cut, not the whole round

    def test_the_cut_round_is_retried_with_reasoning_off_and_a_synthetic_watchdog_note(self):
        efforts = [r.kwargs.get("reasoning") for r in self.provider.requests[:2]]
        self.assertEqual(efforts, ["medium", "off"])
        notes = self.notes("watchdog")
        self.assertEqual(len(notes), 1)
        self.assertTrue(notes[0].synthetic)
        self.assertEqual(self.rec.of("round_reset").count(("watchdog",)), 1)

    def test_the_truncated_giant_script_never_runs_and_the_model_is_told_to_split(self):
        first = next(r for r in self.results() if r.name == "execute_code")
        self.assertTrue(first.is_error)
        self.assertIn("under 40 lines", first.content)
        tools_run = [s.tool for s in self.project.read_steps()]
        self.assertEqual(tools_run.count("execute_code"), 1)               # only the small second script was recorded
        giant = [s for s in self.project.read_steps() if len(json.dumps(s.args)) > 5000]
        self.assertEqual(giant, [])

    def test_every_call_of_the_bad_round_still_gets_exactly_one_result(self):
        for message in self.history:
            if message.role == "assistant" and message.tool_calls:
                following = self.history[self.history.index(message) + 1]
                self.assertEqual([r.tool_call_id for r in following.tool_results], [c.id for c in message.tool_calls])
        self.assertEqual(context.alternation_problems(self.provider.requests[-1].messages), [])

    def test_the_split_build_finishes_and_validates(self):
        self.assertEqual((self.result.ok, self.result.outcome), (True, "validated"))
        self.assertEqual(self.objects(), ["Base", "Clamp"])
        self.assertIn("## Build Summary", self.result.text)

    def test_the_telemetry_tells_the_story_with_the_documented_event_names(self):
        kinds = [k for k, _f in self.events]
        for expected in ("turn_start", "round", "watchdog_trip", "bad_arguments", "tool_start", "turn_end"):
            self.assertIn(expected, kinds)
        self.assertLess(kinds.index("watchdog_trip"), kinds.index("bad_arguments"))
        start = next(f for k, f in self.events if k == "turn_start")
        self.assertEqual(start["intent"], "generic_part")
        end = next(f for k, f in self.events if k == "turn_end")
        self.assertEqual(end["outcome"], "validated")
        logged = json.dumps([f for _k, f in self.events], default=str)
        self.assertNotIn("import FreeCAD", logged)                            # never a code body in the log
        # (the ``bad_arguments`` reason quotes the last 60 characters of a truncated call: a short tail, never the body)
        self.assertLess(len(logged), 6000)
        giant = next(f for k, f in self.events if k == "tool_start" and f["arg_names"] == ["_raw"])
        self.assertGreater(giant["arg_chars"], 20000)                          # sizes are logged, not contents


class SpreadsheetCrashTest(ReplayCase):
    """Old flow: ``'int' object has no attribute 'Value'`` with no hint, and the crashed script's sheet left behind."""

    def setUp(self):
        super().setUp()
        self.replay("spreadsheet_int_quantity_crash")

    def test_the_crash_is_reported_with_the_cause_and_the_fix(self):
        crash = next(r for r in self.results() if r.name == "execute_code" and r.is_error)
        self.assertIn("'int' object has no attribute 'Value'", crash.content)
        self.assertIn("at script line", crash.content)
        self.assertIn("hint:", crash.content)
        self.assertIn("qv", crash.content)                                # the helper the hint recommends

    def test_the_failed_script_is_rolled_back_so_the_retry_reuses_the_name(self):
        crash = next(r for r in self.results() if r.name == "execute_code" and r.is_error)
        self.assertIn("ROLLED BACK", crash.content)
        names = self.objects()
        self.assertIn("Params", names)
        self.assertNotIn("Params001", names)
        self.assertEqual(sorted(names), ["Params", "Plate"])

    def test_the_recorded_steps_show_the_crash_then_the_success(self):
        steps = self.project.read_steps()
        self.assertEqual([(s.tool, s.ok) for s in steps], [("create_document", True), ("execute_code", False), ("execute_code", True)])
        failed = steps[1]
        self.assertEqual((failed.snapshot, failed.checkpoint, failed.delta["added"]), ("", "", []))
        self.assertIn("'int' object has no attribute 'Value'", failed.result_head)
        self.assertEqual(steps[2].delta["added"], ["Params", "Plate"])

    def test_the_retry_really_computed_the_length_from_the_unitless_cell(self):
        plate = App.getDocument(self.doc).getObject("Plate")
        self.assertAlmostEqual(plate.Length.Value, 40.0)

    def test_one_crash_does_not_trip_the_failure_breaker_and_the_turn_validates(self):
        self.assertEqual(self.result.outcome, "validated")
        self.assertEqual(self.rec.of("round_reset"), [])
        self.assertEqual([m for m in self.history if getattr(m, "synthetic", False)], [])
        self.assertLess(self.wall, WALL_BUDGET_S)


class PartialObjectsTest(ReplayCase):
    """Old flow: a script died after creating Shell and Lid; both stayed, the retry made Shell001 / Lid001."""

    def setUp(self):
        super().setUp()
        self.replay("partial_objects_after_failed_script")

    def test_the_document_has_exactly_the_objects_of_the_good_script(self):
        names = self.objects()
        self.assertEqual(names, ["Shell", "Lid"])
        self.assertFalse([n for n in names if re.search(r"\d{3}$", n)], names)

    def test_the_model_is_told_what_was_undone(self):
        crash = next(r for r in self.results() if r.name == "execute_code" and r.is_error)
        self.assertIn("ROLLED BACK", crash.content)
        self.assertIn("removed 2 created object(s): Shell, Lid", crash.content)
        self.assertIn("lip does not fit the shell", crash.content)

    def test_the_failed_step_records_no_delta_and_no_checkpoint(self):
        failed = self.project.read_steps()[1]
        self.assertFalse(failed.ok)
        self.assertEqual(failed.delta, {"added": [], "removed": [], "changed": []})
        self.assertEqual((failed.checkpoint, failed.snapshot_note), ("", "step failed"))

    def test_health_after_the_retry_describes_two_clean_solids(self):
        step = self.project.read_steps()[2]
        self.assertTrue(step.ok)
        self.assertIn("2 objects, 2 solids", step.health["line"])
        self.assertEqual(step.health["errors"], 0)
        self.assertEqual((self.result.ok, self.result.outcome), (True, "validated"))

    def test_the_rolled_back_script_is_still_saved_for_the_dossier(self):
        failed = self.project.read_steps()[1]
        if failed.script:
            self.assertTrue(os.path.isfile(self.project.path(failed.script)))
        validated = self.project.read_steps()[2]
        self.assertTrue(os.path.isfile(self.project.path(validated.script)))


class PlanOnlyTest(ReplayCase):
    """Old flow: the plan text was accepted as the answer; nothing was built and the turn looked green."""

    def setUp(self):
        super().setUp()
        self.replay("plan_only_reply")

    def test_the_model_is_nudged_twice_and_the_second_nudge_forces_a_tool(self):
        gate = self.notes("gate")
        self.assertEqual(len(gate), 2)
        kinds = [r.kwargs.get("tool_choice") for r in self.provider.requests]
        self.assertEqual(kinds[:2], [None, None])
        self.assertIn(kinds[2], set(self.provider.requests[2].tools))

    def test_every_nudge_follows_the_plan_that_provoked_it_and_is_marked_synthetic(self):
        for note in self.notes("gate"):
            before = self.history[self.history.index(note) - 1]
            self.assertEqual(before.role, "assistant")
            self.assertIn("Here is my plan", before.content)
        self.assertEqual(context.alternation_problems(self.provider.requests[-1].messages), [])

    def test_the_nudge_names_the_missing_work(self):
        self.assertRegex(self.notes("gate")[0].content, r"NO_WORK|STEPS_PENDING")

    def test_the_forced_call_builds_and_the_turn_ends_with_a_real_result(self):
        self.assertEqual(self.objects(), ["Plate"])
        self.assertEqual((self.result.ok, self.result.outcome), (True, "validated"))
        self.assertEqual(self.result.geometry_ok, 1)
        self.assertEqual([s.tool for s in self.project.read_steps()], ["create_document", "create_object"])

    def test_the_plan_text_is_never_the_final_answer(self):
        self.assertTrue(self.result.text.startswith("Made the plate."))
        self.assertNotIn("Here is my plan", self.result.text)


class ReplayWithoutTheFlowTest(ReplayCase):
    """The kill switch (``flow=False``) really switches the ledger and the finish gate off: no finish-gate note, no banner."""

    def test_the_plan_only_trace_gets_no_gate_note_when_the_flow_is_off(self):
        self.trace = traces.load_trace("plan_only_reply")
        script = traces.script_from_trace(self.trace, self.doc_name("TrOff"))[:2] * 2
        from support.scripted_llm import ScriptedProvider

        provider = ScriptedProvider(script, strict=False, advance=self.clock.advance)
        result = self.turn(self.trace["prompt"], provider=provider, flow=False, validator=False)
        # only the old English keyword nudge can still fire: never the finish gate's ledger-based note
        self.assertEqual([n for n in self.notes("gate") if n.content.startswith("[finish-gate]")], [])
        self.assertNotIn("NOT VERIFIED", result.text)
        self.assertIsNone(result.ledger)
        self.assertEqual(result.geometry_ok, 0)


if __name__ == "__main__":
    unittest.main()
