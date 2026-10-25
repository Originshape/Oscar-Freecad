"""Tests of the panel's revert and rewind (oscar/panel_revert.py, panel_rewind.py, panel_follow.py,
panel_render.py rewind link): the plan follows a revert (``Ledger.on_revert``), the per-step ``rewind:`` anchor with its
guards and confirmation, the revert fallback to the Project's turn checkpoint when the undo stack cannot hold the
turn, and the follow-the-build toggle (camera fit only while the user has not moved it, flash of created objects,
persisted preference, header menu).

Same harness as test_stage2_review_panelui.py: a real ``OscarPanel`` on an offscreen QApplication, a fake engine and
fake Project; the last class drives a REAL Project and REAL documents (checkpoints, restore, journal).
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

_IMPORT_ERROR = ""
try:
    import FreeCAD as App
    from oscar import params as _params
    from oscar.flow import Ledger
    from oscar.panel import Block
    from oscar.panel_render import StepView, step_row_html
    from oscar.project_util import RestoreResult
    from test_panel_flow import j
    from test_panel_logic import GOOD_SNAPSHOT
    from test_panel_render import finished, ledger, started
    from test_stage2_review_panelui import _RevertHarness
except Exception as exc:   # pragma: no cover - plain interpreter without FreeCAD / Qt
    _IMPORT_ERROR = str(exc)

    class _RevertHarness(unittest.TestCase):
        """Skips every test of a module whose Qt / FreeCAD imports failed."""

        @classmethod
        def setUpClass(cls):
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")


class _Harness(_RevertHarness, unittest.TestCase):
    """_RevertHarness plus turns whose steps carry a checkpoint and a stored plan."""

    def setUp(self):
        super().setUp()
        self.calls = []

        def on_revert(ledger_self, restored_step=None):
            self.calls.append(restored_step)
            ledger_self.revision = 4242

        patcher = mock.patch.object(Ledger, "on_revert", on_revert, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.p._current_session()["ledger"] = ledger()

    def ckpt_turn(self, *indexes, checkpoint=True, **over):
        p = self.p
        p._turn_snapshot = dict(GOOD_SNAPSHOT)
        self.begin_turn()
        for n, index in enumerate(indexes):
            call = f"call_{n}"
            p._on_tool_started("execute_code")
            p._on_step_started(j(started(call, ordinal=n + 1)))
            fields = dict(step_index=index, checkpoint=f"{index:03d}" if checkpoint else "", title=f"Step {index}")
            fields.update(over)
            p._on_step_finished(j(finished(call, ordinal=n + 1, **fields)))
            p._on_tool_finished("execute_code(code=...)", True, "ok")
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn(True, "ok")
        return self.card()

    def rewind_info(self, step, **over):
        info = {"ok": True, "error": "", "step": step, "checkpoint": f"{step:03d}", "doc": "Doc",
                "objects": ["Base"], "lost_steps": [{"index": 3, "title": "Step 3", "tool": "execute_code"}]}
        info.update(over)
        return info


class RevertLedgerTest(_Harness):
    """Defect: after a verified revert the plan strip still showed steps done for geometry that was gone."""

    def test_a_verified_revert_calls_on_revert_with_the_step_before_the_turn(self):
        card = self.ckpt_turn(7, 8)
        self.p._revert_turn(card)
        self.assertTrue(card.reverted)
        self.assertEqual(self.calls, [6])
        self.assertEqual(self.p._current_session()["ledger"]["revision"], 4242)   # the changed ledger is carried on

    def test_a_turn_without_recorded_steps_passes_none(self):
        card = self.ckpt_turn(-1)
        self.p._revert_turn(card)
        self.assertEqual(self.calls, [None])

    def test_a_failed_revert_leaves_the_plan_alone(self):
        self.engine.revert = {"success": False, "error": "no undo"}
        self.p._revert_turn(self.ckpt_turn(7))
        self.assertEqual(self.calls, [])

    def test_without_ledger_support_the_revert_still_works(self):
        with mock.patch.object(Ledger, "on_revert", None, create=True):
            card = self.ckpt_turn(7)
            self.p._revert_turn(card)
        self.assertTrue(card.reverted)

    def test_the_reverted_rows_are_struck_through_and_offer_no_rewind(self):
        card = self.ckpt_turn(7)
        self.assertIn("rewind:", card.steps[0].html(card.bid, 0))
        self.p._revert_turn(card)
        html = card.steps[0].html(card.bid, 0)
        self.assertIn("<s>", html)
        self.assertNotIn("rewind:", html)

    def test_the_plan_strip_is_redrawn(self):
        card = self.ckpt_turn(7)
        with mock.patch.object(self.p, "_refresh_plan_strip") as refresh:
            self.p._revert_turn(card)
        refresh.assert_called()


class RewindRowTest(_Harness):
    def test_a_step_with_a_checkpoint_offers_rewind_with_the_anchor_form(self):
        card = self.ckpt_turn(2, 3)
        self.assertIn(f"href='rewind:{card.bid}:1'", card.steps[1].html(card.bid, 1))

    def test_a_step_without_checkpoint_or_index_offers_none(self):
        card = self.ckpt_turn(2, checkpoint=False)
        self.assertNotIn("rewind:", card.steps[0].html(card.bid, 0))
        view = StepView(name="execute_code", ok=True, step_index=-1, checkpoint="005")
        self.assertNotIn("rewind:", step_row_html(view, 1, 0))
        self.assertIn("rewind:1:0", step_row_html(StepView(name="x", ok=True, step_index=5, checkpoint="005"), 1, 0))
        self.assertNotIn("rewind:", step_row_html(StepView(name="x", ok=False, step_index=5, checkpoint="005"), 1, 0))


class RewindAnchorTest(_Harness):
    def setUp(self):
        super().setUp()
        self.card = self.ckpt_turn(1, 2, 3)
        self.confirm = mock.patch.object(self.p, "_confirm_rewind", return_value=True)
        self.ask = self.confirm.start()
        self.addCleanup(self.confirm.stop)
        self.project.rewind_info.side_effect = lambda n, doc_name=None: self.rewind_info(n)
        self.project.rewind_to.side_effect = lambda n, doc_name=None: RestoreResult(
            True, "", "Doc", f"{n:03d}", ["Base", "Lid"], ["Base"], True)

    def click(self, row):
        self.p._on_anchor_target(f"rewind:{self.card.bid}:{row}")

    def notices(self):
        return self.p.log.toPlainText()

    def test_a_confirmed_rewind_restores_and_updates_everything(self):
        self.click(1)
        self.project.rewind_to.assert_called_once_with(2)
        views = [s.view for s in self.card.steps]
        self.assertEqual([v.superseded for v in views], [False, False, True])    # only the later step is discarded
        self.assertEqual(self.calls, [2])                                         # the plan forgets the later work
        note = "\n".join(self.p._pending_notes)
        self.assertIn("rewound the model to step 2", note)
        self.assertIn("Steps 3..3", note)
        self.assertIn("Do not assume anything built after that step exists", note)
        self.assertFalse(self.card.revertible)                                    # its undo baseline is stale now
        self.assertIn("Rewound to step 2", self.notices())

    def test_the_confirmation_names_objects_and_lost_steps(self):
        with mock.patch.object(self.p, "_live_object_names", return_value=["Base", "Lid", "Vent"]):
            self.click(1)
        text = self.ask.call_args[0][0]
        self.assertIn("Objects that disappear: Lid, Vent", text)
        self.assertIn("Later steps that are lost (1): 3 Step 3", text)
        self.assertIn("checkpoint 002", text)

    def test_declining_changes_nothing(self):
        self.ask.return_value = False
        self.click(1)
        self.project.rewind_to.assert_not_called()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.p._pending_notes, [])
        self.assertFalse(self.card.steps[2].view.superseded)

    def test_refused_while_a_turn_runs(self):
        self.begin_turn()
        self.click(1)
        self.ask.assert_not_called()
        self.project.rewind_to.assert_not_called()
        self.assertIn("A turn is running", self.notices())

    def test_refused_when_the_step_has_no_checkpoint(self):
        self.card.steps[1].view.checkpoint = ""
        self.click(1)
        self.project.rewind_to.assert_not_called()
        self.assertIn("No checkpoint exists for step 2", self.notices())

    def test_refused_when_the_checkpoint_was_pruned(self):
        self.project.rewind_info.side_effect = lambda n, doc_name=None: self.rewind_info(n, checkpoint="001")
        self.click(1)
        self.project.rewind_to.assert_not_called()
        self.assertIn("no longer exists", self.notices())

    def test_refused_when_the_project_reports_no_checkpoint(self):
        self.project.rewind_info.side_effect = lambda n, doc_name=None: {"ok": False, "error": "no checkpoint at or before step 2"}
        self.click(1)
        self.project.rewind_to.assert_not_called()
        self.assertIn("no checkpoint at or before step 2", self.notices())

    def test_a_failed_restore_changes_nothing_in_the_panel(self):
        self.project.rewind_to.side_effect = lambda n, doc_name=None: RestoreResult(
            False, "object names differ from the checkpoint", "Doc", "002")
        self.click(1)
        self.assertIn("left as it was", self.notices())
        self.assertIn("object names differ", self.notices())
        self.assertEqual(self.calls, [])
        self.assertEqual(self.p._pending_notes, [])
        self.assertFalse(any(s.view.superseded for s in self.card.steps))

    def test_a_superseded_or_unrecorded_step_is_refused(self):
        self.card.steps[0].view.superseded = True
        self.click(0)
        self.assertIn("already discarded", self.notices())
        self.card.steps[1].view.step_index = -1
        self.click(1)
        self.assertIn("not recorded in the project", self.notices())
        self.project.rewind_to.assert_not_called()

    def test_a_bad_anchor_is_ignored(self):
        self.p._on_anchor_target("rewind:999:0")
        self.p._on_anchor_target("rewind:x")
        self.project.rewind_to.assert_not_called()


class RevertFallbackTest(_Harness):
    """The undo stack cannot hold the turn (``incomplete``): the Project's turn checkpoint is restored and the result
    is verified by object-name-set equality before the card is marked reverted."""

    def setUp(self):
        super().setUp()
        self.engine.revert = {"success": False, "incomplete": True, "incomplete_docs": ["Doc"],
                              "error": "the undo stack cannot hold this turn (Doc)"}
        self.project.turn_checkpoints.return_value = [{"id": "T9f3a", "doc": "Doc", "objects": ["Base", "Lid"]}]
        self.project.restore.side_effect = lambda doc, ckpt: RestoreResult(True, "", doc, ckpt, ["Base", "X"],
                                                                           ["Lid", "Base"], True)

    def test_the_turn_checkpoint_is_restored_and_verified(self):
        card = self.ckpt_turn(5, 6)
        self.p._revert_turn(card)
        self.project.turn_checkpoints.assert_called_once_with(5)
        self.project.restore.assert_called_once_with("Doc", "T9f3a")
        self.assertTrue(card.reverted)
        self.project.note_reverted.assert_called_once_with(5, via="checkpoint", checkpoint="T9f3a")
        self.assertEqual(self.calls, [4])
        self.assertIn("project checkpoint", self.p.log.toPlainText())
        self.assertTrue(any("restoring the project checkpoint" in n for n in self.p._pending_notes))

    def test_a_name_set_mismatch_is_not_reported_as_reverted(self):
        self.project.restore.side_effect = lambda doc, ckpt: RestoreResult(True, "", doc, ckpt, [], ["Base", "Extra"], True)
        card = self.ckpt_turn(5)
        self.p._revert_turn(card)
        self.assertFalse(card.reverted)
        self.assertFalse(card.revertible)
        self.assertIn("object names differ", self.p.log.toPlainText())
        self.project.note_reverted.assert_not_called()
        self.assertEqual(self.calls, [])

    def test_a_failed_restore_refuses(self):
        self.project.restore.side_effect = lambda doc, ckpt: RestoreResult(False, "rescued", doc, ckpt)
        card = self.ckpt_turn(5)
        self.p._revert_turn(card)
        self.assertFalse(card.reverted)
        self.assertIn("Revert refused", self.p.log.toPlainText())

    def test_without_a_turn_checkpoint_it_refuses(self):
        self.project.turn_checkpoints.return_value = []
        card = self.ckpt_turn(5)
        self.p._revert_turn(card)
        self.assertFalse(card.reverted)
        self.assertIn("no turn-start checkpoint", self.p.log.toPlainText())
        self.project.restore.assert_not_called()

    def test_documents_created_by_the_turn_are_closed_only_by_the_engine(self):
        # P0 #3: the fallback used to close every document that was not open at turn start (the user's included)
        self.engine.close_turn_documents = mock.Mock(return_value={
            "success": False, "closed_docs": ["Fresh"], "conflicts": [], "warnings": ["'Mine' was left open"],
            "kept_docs": [{"doc": "Mine", "reason": "it was saved to disk"}]})
        card = self.ckpt_turn(5)
        with mock.patch("FreeCAD.listDocuments", return_value={"Doc": object(), "Fresh": object(), "Mine": object()}), \
                mock.patch("FreeCAD.closeDocument") as close:
            self.p._revert_turn(card)
        close.assert_not_called()
        self.engine.close_turn_documents.assert_called_once_with(card.snapshot)
        self.assertTrue(card.reverted)
        log = self.p.log.toPlainText()
        self.assertIn("closed Fresh", log)
        self.assertIn("left open: Mine (it was saved to disk)", log)
        self.assertTrue(any("left open: Mine" in n for n in self.p._pending_notes))

    def test_an_engine_without_ownership_rules_closes_nothing(self):
        card = self.ckpt_turn(5)
        with mock.patch("FreeCAD.listDocuments", return_value={"Doc": object(), "Fresh": object()}), \
                mock.patch("FreeCAD.closeDocument") as close:
            self.p._revert_turn(card)
        close.assert_not_called()
        self.assertTrue(card.reverted)

    def test_a_touched_document_without_a_checkpoint_refuses(self):
        self.engine.revert = dict(self.engine.revert, touched_docs=["Doc", "Other"])
        card = self.ckpt_turn(5)
        self.p._revert_turn(card)
        self.project.restore.assert_not_called()
        self.assertFalse(card.reverted)
        self.assertFalse(card.revertible)
        self.assertIn("no turn checkpoint for Other; nothing was restored", self.p.log.toPlainText())

    def test_changes_made_after_the_turn_are_only_thrown_away_after_asking(self):
        changed = {"Doc": "your own step 'Move' came after this turn"}
        self.engine.changed_since_turn = mock.Mock(return_value=changed)
        card = self.ckpt_turn(5)
        with mock.patch.object(self.p, "_confirm_checkpoint_restore", return_value=False) as ask:
            self.p._revert_turn(card)
        ask.assert_called_once_with(changed)
        self.engine.changed_since_turn.assert_called_once_with(card.snapshot, ["Doc"])
        self.project.restore.assert_not_called()
        self.assertFalse(card.reverted)
        self.assertTrue(card.revertible, "the user may still change their mind")
        self.assertIn("Revert cancelled", self.p.log.toPlainText())
        with mock.patch.object(self.p, "_confirm_checkpoint_restore", return_value=True):
            self.p._revert_turn(card)
        self.project.restore.assert_called_once_with("Doc", "T9f3a")
        self.assertTrue(card.reverted)

    def test_nothing_changed_after_the_turn_means_no_question(self):
        self.engine.changed_since_turn = mock.Mock(return_value={})
        card = self.ckpt_turn(5)
        with mock.patch.object(self.p, "_confirm_checkpoint_restore", side_effect=AssertionError("asked")):
            self.p._revert_turn(card)
        self.assertTrue(card.reverted)

    def test_a_check_that_fails_asks_instead_of_assuming(self):
        self.engine.changed_since_turn = mock.Mock(side_effect=RuntimeError("GUI busy"))
        card = self.ckpt_turn(5)
        with mock.patch.object(self.p, "_confirm_checkpoint_restore", return_value=False) as ask:
            self.p._revert_turn(card)
        self.assertIn("could not be checked", ask.call_args[0][0]["Doc"])
        self.project.restore.assert_not_called()

    def test_an_undo_unsafe_turn_is_restored_and_the_note_says_why(self):
        # P0 #4: a FeaturePlan build is never popped with Document.undo() (FreeCAD can crash there)
        self.engine.revert = {"success": False, "incomplete": True, "undo_unsafe": True, "touched_docs": ["Doc"],
                              "incomplete_docs": ["Doc"], "unsafe_docs": ["Doc"], "undone": {}, "closed_docs": [],
                              "error": "FreeCAD's undo can crash on the PartDesign features this turn built (Doc); "
                                       "restore the turn checkpoint instead"}
        card = self.ckpt_turn(5)
        self.p._revert_turn(card)
        self.project.restore.assert_called_once_with("Doc", "T9f3a")
        self.assertTrue(card.reverted)
        self.assertIn("the undo is unsafe for PartDesign builds, so the project checkpoint was used",
                      self.p.log.toPlainText())

    def test_an_undo_unsafe_turn_without_a_checkpoint_is_refused_not_undone(self):
        self.engine.revert = {"success": False, "incomplete": True, "undo_unsafe": True, "touched_docs": ["Doc"],
                              "error": "FreeCAD's undo can crash on the PartDesign features this turn built (Doc); "
                                       "restore the turn checkpoint instead"}
        self.project.turn_checkpoints.return_value = []
        card = self.ckpt_turn(5)
        self.p._revert_turn(card)
        self.assertFalse(card.reverted)
        log = self.p.log.toPlainText()
        self.assertIn("Revert refused", log)
        self.assertIn("can crash", log)
        self.assertEqual(len(self.engine.reverts), 1)    # the engine was asked once and refused; nothing was undone

    def test_rerun_does_not_build_when_the_fallback_fails(self):
        self.project.turn_checkpoints.return_value = []
        sent = []
        self.ckpt_turn(5)
        with mock.patch.object(self.p, "_send", lambda: sent.append(1)):
            self.p._revert_and_rerun_parametric()
        self.assertEqual(sent, [])
        self.assertIn("parametric build was not started", self.p.log.toPlainText())


class FollowBuildTest(_Harness):
    def setUp(self):
        super().setUp()
        self.addCleanup(_params.set_bool, "FollowBuild", True)
        _params.set_bool("FollowBuild", True)
        self.cam = ["A"]
        self.cam_patch = mock.patch.object(self.p, "_camera_signature", side_effect=lambda: self.cam[0])
        self.gui_patch = mock.patch("oscar.panel_follow.Gui")
        self.fit_patch = mock.patch("oscar.cad.views.fit_view")
        self.cam_patch.start()
        self.gui = self.gui_patch.start()
        self.fit = self.fit_patch.start()
        for patcher in (self.cam_patch, self.gui_patch, self.fit_patch):
            self.addCleanup(patcher.stop)

    def test_the_header_menu_has_the_checkable_toggle_wired_to_the_preference(self):
        action = self.p.follow_action
        self.assertEqual(action.objectName(), "oscarFollowAction")
        self.assertTrue(action.isCheckable() and action.isChecked())
        action.setChecked(False)
        self.assertFalse(self.p._follow_build)
        self.assertFalse(_params.get_bool("FollowBuild", True))     # persisted
        self.p._init_follow()
        self.assertFalse(self.p._follow_build)                      # a new panel reads it back
        action.setChecked(True)
        self.assertTrue(_params.get_bool("FollowBuild", False))

    def test_the_camera_is_fitted_until_the_user_moves_it(self):
        self.assertTrue(self.p._fit_camera())
        self.assertTrue(self.p._fit_camera())                       # nothing moved since the last fit
        self.assertEqual(self.fit.call_count, 2)
        self.cam[0] = "B"                                           # the user orbited
        self.assertFalse(self.p._fit_camera())
        self.p._fit_view_after_turn()
        self.p._refresh_view_live("create_object")
        self.assertEqual(self.fit.call_count, 2)
        self.assertTrue(self.p._fit_camera(force=True))             # an explicit action still frames the model
        self.assertEqual(self.fit.call_count, 3)
        self.assertTrue(self.p._fit_camera())                       # ... and the new position is the baseline
        self.assertEqual(self.fit.call_count, 4)

    def test_with_follow_off_the_camera_is_never_touched(self):
        self.p._set_follow_build(False)
        self.assertFalse(self.p._fit_camera())
        self.p._fit_view_after_turn()
        self.assertEqual(self.fit.call_count, 0)
        self.assertTrue(self.p._fit_camera(force=True))

    def test_a_finished_step_selects_what_it_created_and_fits(self):
        with mock.patch.object(self.p, "_select_objects", return_value=["Shell", "Lid"]) as select:
            self.ckpt_turn(1)
        select.assert_called_with(["Shell", "Lid"])
        self.assertGreaterEqual(self.fit.call_count, 1)

    def test_a_failed_step_or_follow_off_selects_nothing(self):
        self.p._set_follow_build(False)
        with mock.patch.object(self.p, "_select_objects") as select:
            self.ckpt_turn(1)
        select.assert_not_called()
        self.p._set_follow_build(True)
        view = StepView(name="x", ok=False, delta={"added": ["A"]})
        with mock.patch.object(self.p, "_select_objects") as select:
            self.assertFalse(self.p._follow_step(view))
        select.assert_not_called()

    def test_the_flash_clears_only_its_own_selection(self):
        sel = [mock.Mock(Name="Shell"), mock.Mock(Name="Lid")]
        self.gui.Selection.getSelection.return_value = sel
        self.p._flash_names = ["Lid", "Shell"]
        self.p._end_flash()
        self.gui.Selection.clearSelection.assert_called_once()
        self.gui.Selection.clearSelection.reset_mock()
        self.p._flash_names = ["Shell"]                            # the user selected something else meanwhile
        self.p._end_flash()
        self.gui.Selection.clearSelection.assert_not_called()
