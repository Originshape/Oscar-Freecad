"""Flow integrity closeout: view requests, claimed-only requirements, a go-ahead after a finished plan, ``Ledger.on_revert``."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.flow import Ledger  # noqa: E402
from oscar.toolctx import ToolOutcome  # noqa: E402
from support.scripted_llm import call, calls, say  # noqa: E402
from test_agent_loop import LoopCase  # noqa: E402

REQUEST = "make a plate 10x10x2 mm"


def geo(index, name="create_object"):
    return ToolOutcome(name=name, title=name, ok=True, handler_ok=True, mutating=True, geometry=True, step_index=index,
                       delta={"added": [f"Obj{index}"], "removed": [], "changed": []})


def finished(request=REQUEST):
    ledger = Ledger.from_request(request)
    for step in ledger.steps:
        step.status = "done"
    ledger._bump()
    return ledger


class ClaimedRequirementTest(unittest.TestCase):
    def test_a_requirement_covered_only_by_a_claimed_step_is_not_done(self):
        ledger = Ledger.from_request(REQUEST)
        step = ledger.step("S3")
        step.status, step.claimed = "done", True
        ledger._bump()
        covered = [r for r in ledger.requirements if "S3" in r.covered_by]
        self.assertTrue(covered)
        self.assertTrue(all(r.status == "pending" for r in covered), [r.status for r in covered])

    def test_recorded_work_confirms_the_claim_and_the_requirement(self):
        ledger = Ledger.from_request(REQUEST)
        step = ledger.step("S3")
        step.status, step.claimed = "done", True
        ledger.on_tool_result(geo(1))
        ledger.confirm_evidenced_steps()
        covered = [r for r in ledger.requirements if "S3" in r.covered_by]
        self.assertTrue(covered)
        self.assertTrue(all(r.status == "done" for r in covered))


class OnRevertTest(unittest.TestCase):
    def built(self):
        ledger = Ledger.from_request(REQUEST)
        ledger.on_tool_result(geo(1))
        ledger.on_tool_result(geo(2, "edit_object"))
        return ledger

    def test_steps_with_evidence_after_the_restored_point_go_back_to_pending(self):
        ledger = self.built()
        ledger.step("S3").status = "done"
        ledger._bump()
        revision = ledger.revision
        ledger.on_revert(0)
        step = ledger.step("S3")
        self.assertEqual((step.status, step.note, step.evidence), ("pending", "reverted", []))
        self.assertEqual(ledger.last_geometry_step, 0)
        self.assertGreater(ledger.revision, revision)
        self.assertTrue(all(r.status == "pending" for r in ledger.requirements if "S3" in r.covered_by))

    def test_evidence_at_or_before_the_restored_point_survives(self):
        ledger = self.built()
        target = next(s for s in ledger.steps if s.evidence)
        ledger.on_revert(1)
        kept = [e["step_index"] for e in target.evidence]
        self.assertEqual(kept, [1] if kept else [])
        self.assertEqual(ledger.last_geometry_step, 1)

    def test_none_reopens_all_recorded_work_and_claims(self):
        ledger = self.built()
        claimed = ledger.step("S2")
        claimed.status, claimed.claimed = "done", True
        ledger.on_revert(None)
        self.assertEqual((claimed.status, claimed.claimed, claimed.note), ("pending", False, "reverted"))
        self.assertEqual(ledger.validated_after_step, 0)

    def test_it_is_idempotent_and_json_round_trips(self):
        ledger = self.built()
        ledger.on_revert(0)
        again = Ledger.from_dict(json.loads(json.dumps(ledger.to_dict())))
        self.assertEqual([(s.id, s.status, s.note) for s in again.steps], [(s.id, s.status, s.note) for s in ledger.steps])
        ledger.on_revert(0)
        self.assertEqual([(s.id, s.status, s.note) for s in again.steps], [(s.id, s.status, s.note) for s in ledger.steps])

    def test_the_model_is_told_about_the_reverted_steps_and_redoing_one_clears_the_note(self):
        ledger = self.built()
        ledger.on_revert(0)
        self.assertIn("NOTE: the model was reverted", ledger.render_compact())
        ledger.on_tool_result(geo(3))
        redone = next(s for s in ledger.steps if s.evidence)
        self.assertNotEqual(redone.note, "reverted")

    def test_validation_after_the_restored_point_is_forgotten(self):
        ledger = self.built()
        ledger.apply_validation({"ok": True, "gate_errors": [], "fail_rows": [], "after_step": 2})
        ledger.on_revert(1)
        self.assertEqual(ledger.validated_after_step, 0)


class GoAheadAfterFinishedPlanTest(LoopCase):
    def test_a_go_ahead_carries_the_original_request(self):
        carried = finished()
        worker = self.build("continue", [say("ok")], ledger=carried)
        worker._begin_turn()
        self.addCleanup(worker._teardown, "answered", None)
        self.assertEqual(worker._ledger.request, REQUEST)
        self.assertFalse(worker._live)

    def test_a_building_go_ahead_plans_the_original_request_never_the_word_continue(self):
        result = self.run_turn("continue", [calls(call("create_object", n=1)), say("Built it.")], ledger=finished())
        self.assertEqual(result.ledger["request"], REQUEST)
        self.assertNotEqual(result.ledger["request"], "continue")

    def test_a_genuinely_new_request_still_gets_its_own_plan(self):
        worker = self.build("make a gear with 20 teeth", [say("ok")], ledger=finished())
        worker._begin_turn()
        self.addCleanup(worker._teardown, "answered", None)
        self.assertEqual(worker._ledger.request, "make a gear with 20 teeth")
        self.assertTrue(worker._live)


class ViewRequestTurnTest(LoopCase):
    def test_a_pure_view_request_is_an_inspect_turn_that_ends_answered(self):
        result = self.run_turn("show me the bracket", [say("Here is the bracket.")])
        self.assertEqual((result.ok, result.outcome), (True, "answered"))
        self.assertEqual(self.worker.ledger.intent, "inspect")
        self.assertEqual(len(self.provider.requests), 1)


if __name__ == "__main__":
    unittest.main()
