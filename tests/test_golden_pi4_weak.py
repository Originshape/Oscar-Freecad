"""The golden Pi 4 request against WEAK models.

The flow must give a good result with a weak model or end honestly; it must never end green on work that was not done.
Real tools, real Project, real FreeCAD documents; only the model is scripted:

* a model that only writes plans (``plan_only``): failed, never green;
* a model that dumps one giant script that builds two boxes and then claims "everything is done": ``unverified``,
  the Build Summary does not list a single check the app did not run, and the pending ledger steps are named;
* a model whose tool arguments are broken (truncated JSON, malformed JSON) over and over: ``needs_help``;
* a model that wastes its first rounds (a reasoning flood that trips the watchdog, a plan-only reply, truncated
  arguments) and then follows the golden script: the same ``validated`` result as the strong model, with the
  watchdog, nudge and bad-argument notes visible in the history (``synthetic`` messages).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402

from oscar.flow import AgentConfig, WatchdogConfig  # noqa: E402
from oscar.models import ModelCaps  # noqa: E402
from support import personas  # noqa: E402
from support.golden import GoldenRun  # noqa: E402
from support.realflow import SUMMARY_HEADINGS, RealFlowCase  # noqa: E402
from support.scripted_llm import Round, bad_json, calls, call, malformed_json, say  # noqa: E402

PROMPT = personas.prompt_for("golden_pi4")
CLAIM = "Done! The Pi 4 housing is finished: ports, vents, lid and snap fits are all built and every check passed."
GIANT = """
import FreeCAD as App
import Part
d = App.newDocument('PiCase')
shell = d.addObject('Part::Box', 'Shell')
shell.Length, shell.Width, shell.Height = 94, 62, 24
lid = d.addObject('Part::Box', 'Lid')
lid.Length, lid.Width, lid.Height = 94, 62, 3
lid.Placement.Base = App.Vector(0, 0, 24)
d.recompute()
"""
KNOB = ModelCaps(reasoning="stream", reasoning_control="openai_effort")
WATCHDOG = AgentConfig(watchdog=WatchdogConfig(reasoning_chars_plan=3000, reasoning_chars_exec=3000))


def flood(chars: int = 20000, chunk: int = 1000) -> Round:
    """A reasoning stream that never ends, cut off at the token limit (the watchdog trips on it)."""
    return Round([("reasoning", "x" * chunk) for _ in range(chars // chunk)] + [("stop_reason", "length"), ("done", "length")])


def section(text: str, heading: str) -> str:
    return RealFlowCase.section(text, heading)


class WeakModelsTest(RealFlowCase):
    """The honest endings."""

    def setUp(self):
        super().setUp()
        self.project = self.new_project("Weak Pi 4", chat_id="20261002-101010-90e0a1")

    def run_weak(self, script, **kw):
        self.docs.append("PiCase")
        for name in list(App.listDocuments()):
            App.closeDocument(name)
        return self.turn(PROMPT, script, project=self.project, **kw)

    def assert_never_green(self, result):
        self.assertNotEqual(result.outcome, "validated")
        self.assertNotEqual(self.project.reference_status, "step-measured")
        ledger = self.project.read_ledger() or {}
        statuses = {s["status"] for s in ledger.get("steps", [])}
        self.assertTrue(statuses & {"pending", "doing", "claimed"} or result.outcome in ("failed", "needs_help"), statuses)

    # -- plan only ---------------------------------------------------------------------------------------------------

    def test_a_model_that_only_writes_plans_ends_failed_and_nothing_is_built(self):
        result = self.run_weak(personas.build("plan_only"))
        self.assertEqual((result.outcome, result.ok), ("failed", False))
        self.assertEqual(result.geometry_ok, 0)
        self.assertEqual(self.project.read_steps(), [])
        self.assertEqual(len(self.notes("gate")), 2)                    # it was asked twice to act
        self.assertEqual(self.rec.of("turn_outcome")[0][0], "failed")
        self.assertNotIn("validated", result.text.lower().split("not validated")[0])

    # -- one giant script ---------------------------------------------------------------------------------------------

    def giant(self):
        return [calls(call("execute_code", code=GIANT, call_id="call_w1")), *[say(CLAIM) for _ in range(8)]]

    def test_one_giant_script_and_a_claim_of_success_is_unverified_not_green(self):
        result = self.run_weak(self.giant())
        self.assertEqual(result.outcome, "unverified", result.text[-800:])
        self.assertTrue(result.text.startswith("NOT VERIFIED:"), result.text[:200])
        self.assertTrue(any(f"S{n}" in result.text for n in range(2, 12)))      # the unresolved ids are named
        self.assertTrue(self.project.read_ledger()["steps"])
        self.assertEqual(len(self.notes("gate")), 2)                            # nudged to the per-reason budget, then told the truth

    def test_the_summary_of_a_giant_script_lists_only_what_the_app_recorded(self):
        result = self.run_weak(self.giant())
        text = result.text
        for heading in SUMMARY_HEADINGS:
            self.assertIn(heading, text)
        validated, not_validated = section(text, "Validated"), section(text, "NOT validated")
        for claim in ("Port openings", "Snap-fit retention", "STEP-based collision check", "rebuild.py reproduces",
                      "Parameter editability", "Printability rules"):
            self.assertNotIn(claim, validated)
        self.assertIn("STEP-based collision check", not_validated)
        self.assertIn("reference status: none", text)
        self.assertNotIn("Checks passed", validated)
        # the script itself was recorded and is in the project like any other step
        (step,) = self.project.read_steps()
        self.assertEqual((step.tool, step.ok), ("execute_code", True))
        self.assertTrue(os.path.isfile(os.path.join(self.project.root, step.script)))

    # -- broken arguments -------------------------------------------------------------------------------------------

    def test_truncated_arguments_over_and_over_end_needs_help(self):
        script = [bad_json("execute_code", raw=GIANT[:80], call_id=f"call_t{i}") for i in range(6)]
        result = self.run_weak(script)
        self.assertEqual((result.outcome, result.ok), ("needs_help", False))
        self.assertEqual(result.geometry_ok, 0)
        self.assertEqual(self.project.read_steps(), [])                          # a call with broken arguments is never executed
        self.assertEqual(self.rec.of("turn_outcome")[0][0], "needs_help")
        self.assertIn("arguments", result.text.lower())                           # the stop reason names the truncated arguments
        self.assertEqual(len(self.notes("breaker")), 0)                          # the loop ended, it did not keep nudging

    def test_malformed_json_that_never_gets_fixed_is_never_green(self):
        script = [malformed_json("build_enclosure", raw='{"doc_name": PiCase, "stage": "next"}', call_id=f"call_m{i}")
                  for i in range(6)]
        result = self.run_weak(script)
        self.assertIn(result.outcome, ("needs_help", "failed"))
        self.assertFalse(result.ok)
        self.assertEqual(self.project.read_steps(), [])
        first = [r for m in self.history for r in m.tool_results][0]
        self.assertTrue(first.is_error)
        self.assertIn("not valid JSON", first.content)

    # -- the model asks the user instead of working -------------------------------------------------------------------

    def test_a_model_that_asks_for_the_step_file_and_builds_nothing_is_not_a_failure(self):
        result = self.run_weak([say("Please attach the Pi 4 STEP file so that I can start."),
                                say("Please attach the Pi 4 STEP file so that I can start."),
                                say("Please attach the Pi 4 STEP file so that I can start.")])
        self.assertNotEqual(result.outcome, "validated")
        self.assertEqual(result.geometry_ok, 0)


class RecoveringWeakModelTest(unittest.TestCase):
    """A weak model that wastes its first rounds and then follows the golden script ends exactly like the strong one."""

    @classmethod
    def setUpClass(cls):
        prefix = [flood(),                                               # trips the watchdog, retried at reasoning "off"
                  say(personas.PLAN_TEXT),                               # a plan and no tool: the gate nudges
                  bad_json("find_reference_part", raw='{"query": "raspberry pi 4 mo', call_id="call_wb1")]   # truncated arguments
        cls.golden = GoldenRun(prefix=prefix, caps=KNOB, config=WATCHDOG, chat_id="20261002-101010-90d0a3").start()

    @classmethod
    def tearDownClass(cls):
        cls.golden.stop()

    def test_it_still_ends_validated_with_the_same_summary(self):
        result = self.golden.result
        self.assertEqual((result.outcome, result.ok), ("validated", True), result.text[-1200:])
        self.assertIn("reference status: step-measured", result.text)
        self.assertIn("STEP-based collision check", section(result.text, "Validated"))
        self.assertEqual(self.golden.worker.ledger.pending_required(), [])

    def test_the_wasted_rounds_are_visible_and_marked_synthetic(self):
        case = self.golden.case
        self.assertEqual(len(case.notes("watchdog")), 1)
        self.assertEqual(len(case.notes("gate")), 1)
        self.assertEqual(case.rec.of("round_reset").count(("watchdog",)), 1)

    def test_the_bad_arguments_were_never_executed_and_the_steps_are_the_golden_ones(self):
        steps = self.golden.steps()
        self.assertEqual(len(steps), 19)
        self.assertEqual(steps[0].tool, "source_reference_part")
        self.assertEqual(sum(1 for s in steps if not s.ok), 1)                  # only the deliberate failure of the script
        first = [r for m in self.golden.case.history for r in m.tool_results][0]
        self.assertTrue(first.is_error)
        self.assertIn("not valid JSON", first.content)
        self.assertIn("Not executed", first.content)

    def test_no_round_stayed_silent_past_the_watchdog_caps(self):
        rec = self.golden.case.rec
        self.assertGreaterEqual(rec.count("progress"), len(self.golden.script))
        self.assertLess(self.golden.case.clock() - 1000.0, 240.0)                  # the injected clock never passed the round cap
        self.assertEqual(rec.of("turn_outcome")[0][0], "validated")


if __name__ == "__main__":
    unittest.main()
