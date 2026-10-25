"""Tests of the Project side of revert / rewind against REAL FreeCAD documents (run under freecadcmd):
``rewind_info`` / ``turn_checkpoints`` (what the panel shows in its confirmation and which checkpoint a revert
falls back to), ``note_reverted`` re-syncing the params.json mirror, and the panel's per-step rewind end to end
(real Project, real checkpoints, real restore, journal)."""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

import FreeCAD as App  # noqa: E402

import project_support as S  # noqa: E402
from test_project_gui import RealDocTestCase  # noqa: E402


class _Base(RealDocTestCase):
    def setUp(self):
        super().setUp()
        self.doc = self.new_model()
        self.name = self.doc.Name
        S.record(self.proj, doc=self.name, title="Base")                                   # step 1
        self.doc.addObject("Part::Sphere", "Ball")
        self.doc.recompute()
        S.record(self.proj, doc=self.name, title="Add ball", delta={"added": ["Ball"]})    # step 2
        self.doc.removeObject("Cyl")
        self.doc.recompute()
        S.record(self.proj, doc=self.name, title="Drop cyl", delta={"removed": ["Cyl"]})   # step 3


class RewindInfoTest(_Base):
    def test_it_names_the_checkpoint_the_objects_and_the_lost_steps(self):
        info = self.proj.rewind_info(1)
        self.assertTrue(info["ok"], info)
        self.assertEqual((info["checkpoint"], info["doc"], info["step"]), ("001", self.name, 1))
        self.assertIn("Cyl", info["objects"])
        self.assertNotIn("Ball", info["objects"])
        self.assertEqual([(s["index"], s["title"]) for s in info["lost_steps"]], [(2, "Add ball"), (3, "Drop cyl")])

    def test_superseded_steps_are_not_lost_again(self):
        self.assertTrue(self.proj.rewind_to(2).ok)
        self.assertEqual([s["index"] for s in self.proj.rewind_info(1)["lost_steps"]], [2])   # 3 is already discarded
        self.assertEqual([s["index"] for s in self.proj.rewind_info(2)["lost_steps"]], [])

    def test_unknown_steps_and_pruned_checkpoints_are_reported(self):
        self.assertFalse(self.proj.rewind_info(99)["ok"])
        self.assertFalse(self.proj.rewind_info("x")["ok"])
        for entry in self.proj._checkpoint_entries():
            os.remove(entry["path"])
        info = self.proj.rewind_info(2)
        self.assertFalse(info["ok"])
        self.assertIn("no checkpoint at or before step 2", info["error"])

    def test_it_changes_nothing(self):
        before = self.proj.read_steps(include_superseded=True)
        self.proj.rewind_info(1)
        self.assertEqual(len(self.proj.read_steps(include_superseded=True)), len(before))
        self.assertNotIn("rewind", [json.loads(l)["type"] for l in S.read_lines(self.proj)])


class TurnCheckpointsTest(RealDocTestCase):
    def test_the_turn_checkpoint_of_a_turn_is_found_by_its_steps(self):
        doc = self.new_model()
        S.record(self.proj, doc=doc.Name, title="Older turn", turn_id="T20261001-000001-aaaa")      # step 1
        self.proj.begin_turn("T20261001-000002-0a1b", "next", work=True)
        self.assertEqual(self.proj.ensure_turn_checkpoint(doc.Name), "T0a1b")                      # after step 1
        S.record(self.proj, doc=doc.Name, title="First", turn_id="T20261001-000002-0a1b")          # step 2
        S.record(self.proj, doc=doc.Name, title="Second", turn_id="T20261001-000002-0a1b")         # step 3
        found = self.proj.turn_checkpoints(2)
        self.assertEqual([(f["id"], f["doc"]) for f in found], [("T0a1b", doc.Name)])
        self.assertIn("Cyl", found[0]["objects"])
        self.assertEqual(self.proj.turn_checkpoints(1), [])      # the older turn took none
        self.assertEqual(self.proj.turn_checkpoints("x"), [])
        self.assertEqual(self.proj.turn_checkpoints(99), [])

    def test_a_later_turns_checkpoint_is_not_taken_for_an_earlier_turn_without_one(self):
        doc = self.new_model()
        S.record(self.proj, doc=doc.Name, title="No checkpoint turn", turn_id="T20261001-000001-aaaa")   # step 1
        self.proj.begin_turn("T20261001-000002-0b2c", "next", work=True)
        self.proj.ensure_turn_checkpoint(doc.Name)                                                      # after step 1
        S.record(self.proj, doc=doc.Name, title="Later", turn_id="T20261001-000002-0b2c")               # step 2
        self.assertEqual(self.proj.turn_checkpoints(1), [])


class NoteRevertedParamsTest(RealDocTestCase):
    """Defect: a revert left params.json describing the reverted sheet."""

    def setUp(self):
        super().setUp()
        self.doc = self.new_model()
        S.record(self.proj, doc=self.doc.Name, title="Base")        # step 1
        S.record(self.proj, doc=self.doc.Name, title="Thicker")     # step 2

    def mirror(self):
        return {p["name"]: p["value"] for p in json.load(open(self.proj.path("params.json")))["params"]}

    def test_the_mirror_follows_the_sheet_after_the_revert(self):
        self.doc.getObject("Params").set("B2", "=4 mm")
        self.doc.recompute()
        self.proj.sync_params(self.doc.Name)
        self.assertEqual(self.mirror()["wall_t"], 4.0)
        self.doc.getObject("Params").set("B2", "=2 mm")             # the undo stack popped the edit
        self.doc.recompute()
        self.assertEqual(self.proj.note_reverted(2), 1)
        self.assertEqual(self.mirror()["wall_t"], 2.0)
        self.assertEqual(json.load(open(self.proj.path("params.json")))["step"], 1)

    def test_a_sheet_that_the_reverted_turn_created_leaves_no_stale_mirror(self):
        self.proj.sync_params(self.doc.Name)
        self.assertTrue(os.path.isfile(self.proj.path("params.json")))
        self.doc.getObject("Box").setExpression("Length", None)
        self.doc.removeObject("Params")
        self.doc.recompute()
        self.proj.note_reverted(2)
        self.assertFalse(os.path.exists(self.proj.path("params.json")))

    def test_the_checkpoint_variant_is_journaled(self):
        self.assertEqual(self.proj.note_reverted(2, via="checkpoint", checkpoint="T0a1b"), 1)
        last = json.loads(S.read_lines(self.proj)[-1])
        self.assertEqual((last["to_step"], last["via"], last["restored_checkpoint"]), (1, "checkpoint", "T0a1b"))


_PANEL_ERROR = ""
try:
    from test_panel_rewind import _Harness
    from oscar.panel import Block  # noqa: F401
    from test_panel_flow import j  # noqa: F401
except Exception as exc:   # pragma: no cover
    _PANEL_ERROR = str(exc)

    class _Harness:   # placeholder: the class below is skipped
        pass


@unittest.skipIf(_PANEL_ERROR, f"panel harness unavailable: {_PANEL_ERROR}")
class PanelRewindRealTest(_Harness, unittest.TestCase):
    """The panel's ``rewind:`` anchor with a REAL Project and REAL documents."""

    def setUp(self):
        super().setUp()
        self._before = set(App.listDocuments())
        self.addCleanup(lambda: S.close_docs([n for n in App.listDocuments() if n not in self._before]))
        from test_project_gui import build_model, unique

        self.proj = S.make_project(chat_id="20261001-000000-" + unique("")[:6])
        self.p._current_session()["_project"] = self.proj
        self.doc = build_model(unique("PRW"))
        self.name = self.doc.Name
        S.record(self.proj, doc=self.name, title="Base")
        self.doc.addObject("Part::Sphere", "Ball")
        self.doc.recompute()
        S.record(self.proj, doc=self.name, title="Add ball", delta={"added": ["Ball"]})
        self.doc.removeObject("Cyl")
        self.doc.recompute()
        S.record(self.proj, doc=self.name, title="Drop cyl", delta={"removed": ["Cyl"]})
        self.card = self.ckpt_turn(1, 2, 3)

    def test_a_confirmed_rewind_restores_the_document_and_journals_it(self):
        with mock.patch.object(self.p, "_confirm_rewind", return_value=True) as ask:
            self.p._on_anchor_target(f"rewind:{self.card.bid}:0")
        text = ask.call_args[0][0]
        self.assertIn("Ball", text)                                  # disappears
        self.assertIn("Later steps that are lost (2)", text)
        live = App.getDocument(self.name)
        self.assertEqual(sorted(o.Name for o in live.Objects), sorted(["Params", "Box", "Cyl"]))
        self.assertEqual(App.ActiveDocument.Name, self.name)         # re-activated
        self.assertEqual([s.index for s in self.proj.read_steps()], [1])
        self.assertEqual(self.calls, [1])
        self.assertTrue(all(s.view.superseded for s in self.card.steps[1:]))
        self.assertFalse(self.card.steps[0].view.superseded)
        self.assertTrue(any("rewound the model to step 1" in n for n in self.p._pending_notes))
        notes = open(self.proj.path("NOTES.md"), encoding="utf-8").read()
        self.assertIn("~~Add ball~~", notes)
        self.assertNotIn("Add ball", [s["title"] for s in self.proj.rewind_info(1)["lost_steps"]])

    def test_a_pruned_checkpoint_refuses_and_leaves_the_document(self):
        os.remove(self.proj.path("checkpoints", "001.FCStd"))
        with mock.patch.object(self.p, "_confirm_rewind", return_value=True) as ask:
            self.p._on_anchor_target(f"rewind:{self.card.bid}:0")
        ask.assert_not_called()
        self.assertIsNotNone(App.getDocument(self.name).getObject("Ball"))
        self.assertEqual([s.index for s in self.proj.read_steps()], [1, 2, 3])


@unittest.skipIf(_PANEL_ERROR, f"panel harness unavailable: {_PANEL_ERROR}")
class PanelCheckpointFallbackRealTest(_Harness, unittest.TestCase):
    """P0 #3 end to end: the panel's checkpoint fallback with a REAL Project, REAL documents and the real engine's
    ownership rules. The turn's own document is closed; a document the user opened and worked in during the turn
    survives (the old fallback closed every document that was not open at the turn's start)."""

    TURN = "T20261001-000002-0c3d"

    def setUp(self):
        super().setUp()
        from oscar.cad import engine as engine_module
        from test_project_gui import build_model, unique

        self._before = set(App.listDocuments())
        self.addCleanup(lambda: S.close_docs([n for n in App.listDocuments() if n not in self._before]))
        real = engine_module.get_engine()          # the harness patched oscar.cad.get_engine with a fake
        self.engine.close_turn_documents = real.close_turn_documents
        self.engine.changed_since_turn = real.changed_since_turn
        self.proj = S.make_project(chat_id="20261001-000000-" + unique("")[:6])
        self.p._current_session()["_project"] = self.proj
        self.doc = build_model(unique("PCF"))
        self.name = self.doc.Name
        snapshot = real.begin_turn_snapshot()     # the turn starts with only the work document open (of ours)
        self.proj.begin_turn(self.TURN, "add a ball", work=True)
        self.assertEqual(self.proj.ensure_turn_checkpoint(self.name), "T0c3d")
        self.doc.addObject("Part::Sphere", "Ball")
        self.doc.recompute()
        S.record(self.proj, doc=self.name, title="Add ball", delta={"added": ["Ball"]}, turn_id=self.TURN)   # step 1
        self.agent_doc = real.create_document(unique("AgentDoc"))["document_name"]
        self.user_doc = App.newDocument(unique("UserDoc"))     # meanwhile the user starts a document of their own
        self.user_doc.UndoMode = 1
        self.user_doc.openTransaction("Pad")
        self.user_doc.addObject("Part::Box", "Mine")
        self.user_doc.commitTransaction()
        self.card = self.ckpt_turn(1)
        self.card.snapshot = snapshot
        self.engine.revert = {"success": False, "incomplete": True, "touched_docs": [self.name],
                              "incomplete_docs": [self.name], "error": "the undo stack cannot hold this turn"}

    def test_the_users_document_survives_and_the_turns_own_is_closed(self):
        self.p._revert_turn(self.card)
        log = self.p.log.toPlainText()
        self.assertTrue(self.card.reverted, log)
        self.assertEqual(sorted(o.Name for o in App.getDocument(self.name).Objects), ["Box", "Cyl", "Params"])
        self.assertIn(self.user_doc.Name, App.listDocuments())
        self.assertEqual([o.Name for o in self.user_doc.Objects], ["Mine"])
        self.assertNotIn(self.agent_doc, App.listDocuments())
        self.assertIn(f"closed {self.agent_doc}", log)
        self.assertIn(f"left open: {self.user_doc.Name} (it holds changes made outside Oscar)", log)


class PlanBuildTurnRevertTest(RealDocTestCase):
    """P0 #4 end to end, without a single ``Document.undo()``: the engine reports a turn that rebuilt a FeaturePlan as
    ``undo_unsafe`` and pops nothing; the turn's checkpoint brings the first build back."""

    TURN = "T20261001-000003-0d4e"

    @staticmethod
    def volume(doc) -> float:
        return round(sum(o.Shape.Volume for o in doc.Objects if o.TypeId == "PartDesign::Body"), 3)

    def test_a_plan_rebuild_is_reverted_from_the_turn_checkpoint(self):
        import copy

        from oscar.cad import engine as engine_module
        from oscar.plan import builder
        from test_project_gui import unique

        engine = engine_module.get_engine()
        fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "plans", "plate_4holes.json")
        with open(fixture, encoding="utf-8") as f:
            plan = json.load(f)
        doc = App.newDocument(unique("PlanRevert"))
        doc.UndoMode = 1
        self.assertTrue(builder.apply_plan(doc, plan)["success"])
        first_names, first_volume = sorted(o.Name for o in doc.Objects), self.volume(doc)
        snapshot = engine.begin_turn_snapshot()
        self.proj.begin_turn(self.TURN, "a thicker plate", work=True)
        self.assertEqual(self.proj.ensure_turn_checkpoint(doc.Name), "T0d4e")
        thicker = copy.deepcopy(plan)
        thicker["parameters"]["thickness"]["value"] = 12.0
        self.assertTrue(builder.apply_plan(doc, thicker)["success"])
        step = S.record(self.proj, doc=doc.Name, title="Thicker plate", turn_id=self.TURN)
        self.assertGreater(self.volume(doc), first_volume + 1.0)
        stack = (doc.UndoCount, list(doc.UndoNames))
        self.assertTrue(stack[1][0].startswith("Oscar: plan "), stack)
        report = engine.revert_turn(snapshot)
        self.assertTrue(report["undo_unsafe"], report)
        self.assertEqual((report["success"], report["undone"], report["unsafe_docs"]), (False, {}, [doc.Name]))
        self.assertEqual((doc.UndoCount, list(doc.UndoNames)), stack)          # nothing was popped
        (found,) = self.proj.turn_checkpoints(step.index)
        restored = self.proj.restore(found["doc"], found["id"])
        self.assertTrue(restored.ok, restored.error)
        live = App.getDocument(doc.Name)
        self.assertEqual(sorted(o.Name for o in live.Objects), first_names)
        self.assertAlmostEqual(self.volume(live), first_volume, delta=1.0)
        self.assertEqual([o.Name for o in live.Objects if "Invalid" in (o.State or [])], [])


if __name__ == "__main__":
    unittest.main()
