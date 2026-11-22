"""Project against REAL FreeCAD documents (run under freecadcmd): autosave / checkpoint / turn checkpoint,
restore and rewind (names equal, Params expressions back, doc.Name unchanged, re-activation, one undo
transaction, rescue copy), the Params mirror, the blank-frame test, model export, and replay equality of a small
model through ``run_tool`` + ``replay_steps`` + the generated rebuild.py."""

import contextlib
import io
import json
import os
import sys
import unittest
import uuid
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402

import project_support as S  # noqa: E402
from oscar import project as P  # noqa: E402
from oscar import project_export as X  # noqa: E402
from oscar import project_gui as gui  # noqa: E402
from oscar import project_replay as R  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402


def unique(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:6]}"


def build_model(name: str, with_params: bool = True):
    """A small parametric model: Params sheet (wall_t 2 mm, vent_count 6, wall_x2 = wall_t * 2), Box with
    ``Length = Params.wall_t * 5`` (10 x 10 x 5) and a Cylinder."""
    doc = App.newDocument(name)
    if with_params:
        sheet = doc.addObject("Spreadsheet::Sheet", "Params")
        for cell, value in (("A1", "name"), ("B1", "value"), ("C1", "unit"), ("D1", "source"), ("E1", "note"),
                            ("A2", "wall_t"), ("B2", "=2 mm"), ("C2", "mm"), ("D2", "assumption"), ("E2", "typical FDM wall"),
                            ("A3", "vent_count"), ("B3", "=6"), ("D3", "derived"),
                            ("A4", "wall_x2"), ("C4", "mm"), ("D4", "derived")):
            sheet.set(cell, value)
        for cell, alias in (("B2", "wall_t"), ("B3", "vent_count")):
            sheet.setAlias(cell, alias)
        sheet.set("B4", "=wall_t * 2")   # a formula can only name an alias that already exists
        sheet.setAlias("B4", "wall_x2")
    box = doc.addObject("Part::Box", "Box")
    box.Width, box.Height = 10, 5
    if with_params:
        box.setExpression("Length", "Params.wall_t * 5")
    else:
        box.Length = 10
    cyl = doc.addObject("Part::Cylinder", "Cyl")
    cyl.Radius, cyl.Height = 3, 8
    doc.recompute()
    return doc


class RealDocTestCase(unittest.TestCase):
    """Closes every document a test created."""

    def setUp(self):
        self._before = set(App.listDocuments())
        self.proj = S.make_project(chat_id=f"20261001-000000-{uuid.uuid4().hex[:6]}")

    def tearDown(self):
        S.close_docs([n for n in App.listDocuments() if n not in self._before])

    def new_model(self, **kw):
        return build_model(unique("T3M"), **kw)


class AutosaveCheckpointTest(RealDocTestCase):
    def test_autosave_writes_a_real_fcstd_and_registers_the_document(self):
        doc = self.new_model()
        self.assertEqual(self.proj.autosave(doc.Name), "model.FCStd")
        path = self.proj.path("model.FCStd")
        self.assertTrue(zipfile.is_zipfile(path))
        self.assertEqual(self.proj.manifest["active_doc"], doc.Name)
        self.assertEqual(self.proj.manifest["docs"], [doc.Name])
        self.assertEqual([n for n in os.listdir(self.proj.root) if ".tmp-" in n or n.endswith(".FCBak")], [])
        reopened = App.openDocument(path, True)
        try:
            self.assertEqual(sorted(o.Name for o in reopened.Objects), sorted(o.Name for o in doc.Objects))
        finally:
            App.closeDocument(reopened.Name)

    def test_saving_twice_leaves_no_backup_files(self):
        doc = self.new_model()
        self.proj.autosave(doc.Name)
        self.proj.autosave(doc.Name)
        self.assertEqual([n for n in os.listdir(self.proj.root) if "FCBak" in n or ".tmp-" in n], [])

    def test_a_second_document_is_saved_under_docs(self):
        first, second = self.new_model(), self.new_model()
        self.assertEqual(self.proj.autosave(first.Name), "model.FCStd")
        self.assertEqual(self.proj.autosave(second.Name), f"docs/{second.Name}.FCStd")
        self.assertEqual(self.proj.manifest["docs"], [first.Name, second.Name])
        self.assertEqual(self.proj.manifest["active_doc"], first.Name)

    def test_unknown_and_empty_document_names(self):
        self.assertEqual(self.proj.autosave(""), "")
        self.assertEqual(self.proj.autosave("NoSuchDocument"), "")
        self.assertEqual(self.proj.save_now("NoSuchDocument"), ("", "no document"))

    def test_the_model_is_found_through_near_miss_names(self):
        doc = self.new_model()
        self.assertEqual(self.proj.autosave(doc.Name.lower()), "model.FCStd")

    def test_checkpoint_default_id_and_sidecar(self):
        doc = self.new_model()
        S.record(self.proj, doc=doc.Name)   # step 1, takes the checkpoint "001"
        self.assertEqual(self.proj.checkpoint(doc.Name), "001")
        side = json.load(open(self.proj.path("checkpoints", "001.json")))
        self.assertEqual(side["doc"], doc.Name)
        self.assertEqual(side["step"], 1)
        self.assertIn("Box", side["objects"])
        self.assertIn("Cyl", side["objects"])
        self.assertIn("Params", side["all_objects"])
        self.assertNotIn("Params", side["objects"])   # a sheet is neither geometry nor a container
        self.assertEqual(side["invalid"], [])
        self.assertTrue(zipfile.is_zipfile(self.proj.path("checkpoints", "001.FCStd")))

    def test_checkpoint_label_becomes_the_id(self):
        doc = self.new_model()
        self.assertEqual(self.proj.checkpoint(doc.Name, "before lid!"), "before-lid")
        self.assertTrue(os.path.isfile(self.proj.path("checkpoints", "before-lid.FCStd")))
        self.assertEqual(self.proj.checkpoint(""), "")

    def test_the_hook_takes_checkpoint_snapshot_and_params_for_a_geometry_step(self):
        doc = self.new_model()
        rec = S.record(self.proj, doc=doc.Name, title="Build the box")
        self.assertEqual((rec.checkpoint, rec.snapshot), ("001", "snapshots/001_build-the-box.png"))
        self.assertEqual(rec.doc, doc.Name)
        self.assertEqual(self.proj.read_params()["doc"], doc.Name)   # the Params sheet was mirrored
        self.assertTrue(os.path.isfile(self.proj.path("snapshots", "001_build-the-box.png")))

    def test_a_document_with_only_a_sheet_has_no_snapshot(self):
        doc = App.newDocument(unique("T3S"))
        sheet = doc.addObject("Spreadsheet::Sheet", "Params")
        sheet.set("A2", "wall_t")
        sheet.set("B2", "=2 mm")
        sheet.setAlias("B2", "wall_t")
        doc.recompute()
        rec = S.record(self.proj, doc=doc.Name, tool="build_enclosure", title="Params stage")
        self.assertEqual((rec.snapshot, rec.snapshot_note), ("", "empty document"))
        self.assertEqual(rec.checkpoint, "001")   # a spreadsheet-only step still gets its checkpoint
        self.assertEqual([r["name"] for r in self.proj.read_params()["params"]], ["wall_t"])

    def test_turn_checkpoint_is_created_once_per_turn_and_document(self):
        doc = self.new_model()
        self.proj.begin_turn("T20261001-000001-0a1b", "build", work=True)
        first = self.proj.ensure_turn_checkpoint(doc.Name)
        self.assertEqual(first, "T0a1b")
        path = self.proj.path("checkpoints", "T0a1b.FCStd")
        self.assertTrue(os.path.isfile(path))
        stamp = os.stat(path).st_mtime_ns
        doc.removeObject("Cyl")   # a later call in the same turn must NOT overwrite the BEFORE state
        doc.recompute()
        self.assertEqual(self.proj.ensure_turn_checkpoint(doc.Name), "T0a1b")
        self.assertEqual(os.stat(path).st_mtime_ns, stamp)
        side = json.load(open(self.proj.path("checkpoints", "T0a1b.json")))
        self.assertEqual((side["turn"], side["doc"]), ("T20261001-000001-0a1b", doc.Name))
        self.assertIn("Cyl", side["objects"])
        self.assertEqual(self.proj.manifest["docs"], [doc.Name])
        self.proj.begin_turn("T20261001-000002-0c2d", "again", work=True)
        self.assertEqual(self.proj.ensure_turn_checkpoint(doc.Name), "T0c2d")
        self.assertNotIn("Cyl", json.load(open(self.proj.path("checkpoints", "T0c2d.json")))["objects"])

    def test_turn_checkpoint_of_a_second_document_gets_its_own_file_and_unrelated_documents_are_untouched(self):
        first, second, unrelated = self.new_model(), self.new_model(), self.new_model()
        self.proj.begin_turn("T20261001-000001-0e3f", "build", work=True)
        self.assertEqual(self.proj.ensure_turn_checkpoint(first.Name), "T0e3f")
        self.assertEqual(self.proj.ensure_turn_checkpoint(second.Name), "T0e3f")
        files = sorted(n for n in os.listdir(self.proj.path("checkpoints")) if n.endswith(".FCStd"))
        self.assertEqual(files, [f"T0e3f.FCStd", f"T0e3f.{second.Name}.FCStd"])
        self.assertEqual(self.proj.manifest["docs"], [first.Name, second.Name])   # the unrelated one is never registered
        self.assertNotIn(unrelated.Name, self.proj.manifest["docs"])
        self.assertEqual(self.proj.ensure_turn_checkpoint(first.Name), "T0e3f")    # idempotent per (turn, document)

    def test_turn_checkpoint_needs_a_turn_and_a_document(self):
        doc = self.new_model()
        self.assertEqual(self.proj.ensure_turn_checkpoint(doc.Name), "")   # no turn yet
        self.proj.begin_turn("T20261001-000001-0a1b", "x", work=True)
        self.assertEqual(self.proj.ensure_turn_checkpoint("NoSuchDocument"), "")

    def test_turn_checkpoint_uses_the_tool_context_turn_when_no_turn_was_begun(self):
        doc = self.new_model()
        with use_context(ToolContext(project=self.proj, turn_id="T20261001-000001-7777")):
            self.assertEqual(self.proj.ensure_turn_checkpoint(doc.Name), "T7777")

    def test_turn_checkpoint_skips_an_oversized_document(self):
        doc = self.new_model()
        self.proj.CHECKPOINT_MAX_BYTES = 10
        self.proj.begin_turn("T20261001-000001-0a1b", "x", work=True)
        self.assertEqual(self.proj.ensure_turn_checkpoint(doc.Name), "")
        self.assertFalse(os.path.exists(self.proj.path("checkpoints", "T0a1b.FCStd")))


class RestoreTest(RealDocTestCase):
    def setUp(self):
        super().setUp()
        self.doc = self.new_model()
        self.name = self.doc.Name
        S.record(self.proj, doc=self.name, title="Build the base")   # step 1: checkpoint 001
        self.names_1 = sorted(o.Name for o in self.doc.Objects)
        self.box_volume = self.doc.getObject("Box").Shape.Volume

    def mutate(self):
        """What a later step did: wall_t 2 -> 3, the Cylinder deleted, a Sphere added."""
        self.doc.getObject("Params").set("B2", "=3 mm")
        self.doc.removeObject("Cyl")
        self.doc.addObject("Part::Sphere", "Ball")
        self.doc.recompute()

    def test_restore_brings_back_names_values_and_expressions(self):
        self.mutate()
        S.record(self.proj, doc=self.name, title="Change it")   # step 2
        self.assertEqual(self.doc.getObject("Box").Length.Value, 15.0)
        self.assertIsNone(self.doc.getObject("Cyl"))
        result = self.proj.restore(self.name, "001")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.checkpoint, "001")
        self.assertEqual(result.doc, self.name)
        self.assertIn("Ball", result.objects_before)
        self.assertEqual(sorted(result.objects_after), self.names_1)
        live = App.getDocument(self.name)
        self.assertEqual(sorted(o.Name for o in live.Objects), self.names_1)
        self.assertIsNone(live.getObject("Ball"))
        self.assertIsNotNone(live.getObject("Cyl"))
        sheet = live.getObject("Params")
        self.assertEqual(sheet.get("wall_t").Value, 2.0)                                   # the value is back
        self.assertEqual(sheet.getContents("B4"), "=wall_t * 2")                           # the formulas too
        box = live.getObject("Box")
        self.assertEqual(list(box.ExpressionEngine), [("Length", "Params.wall_t * 5")])    # the expression is back
        self.assertAlmostEqual(box.Shape.Volume, self.box_volume, places=6)
        self.assertEqual(box.Length.Value, 10.0)
        self.assertFalse([o.Name for o in live.Objects if "Invalid" in o.State])

    def test_the_live_document_keeps_its_name_and_is_active_again(self):
        other = self.new_model()
        App.setActiveDocument(other.Name)
        self.mutate()
        result = self.proj.restore(self.name, "001")
        self.assertTrue(result.ok, result.error)
        self.assertTrue(result.reactivated)
        self.assertEqual(App.ActiveDocument.Name, self.name)   # closing the temp document leaves NO active document otherwise
        self.assertIn(self.name, App.listDocuments())
        self.assertEqual(App.getDocument(self.name).Name, self.name)

    def test_no_temp_document_stays_open_and_unrelated_documents_are_untouched(self):
        unrelated = self.new_model()
        unrelated_names = sorted(o.Name for o in unrelated.Objects)
        unrelated.removeObject("Cyl")
        unrelated.recompute()
        before = set(App.listDocuments())
        self.mutate()
        result = self.proj.restore(self.name, "001")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(set(App.listDocuments()), before)
        self.assertIsNone(App.getDocument(unrelated.Name).getObject("Cyl"))   # its own edit was NOT restored
        self.assertNotEqual(sorted(o.Name for o in App.getDocument(unrelated.Name).Objects), unrelated_names)

    def test_restore_is_one_undo_transaction(self):
        self.doc.UndoMode = 1
        self.mutate()
        before_count = self.doc.UndoCount
        result = self.proj.restore(self.name, "001")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(self.doc.UndoMode, 1)   # the previous mode is put back
        self.assertEqual(self.doc.UndoCount, before_count + 1)
        self.assertEqual(self.doc.UndoNames[-1], "Oscar restore 001")
        self.doc.undo()
        self.assertIsNotNone(self.doc.getObject("Ball"))   # one Ctrl+Z reverts the whole rewind
        self.assertIsNone(self.doc.getObject("Cyl"))
        self.assertEqual(self.doc.getObject("Params").get("wall_t").Value, 3.0)

    def test_undo_mode_zero_is_restored_too(self):
        self.doc.UndoMode = 0
        self.mutate()
        self.assertTrue(self.proj.restore(self.name, "001").ok)
        self.assertEqual(self.doc.UndoMode, 0)

    def test_a_numeric_id_and_the_turn_checkpoint_id(self):
        self.proj.begin_turn("T20261001-000001-0a1b", "x", work=True)
        self.assertEqual(self.proj.ensure_turn_checkpoint(self.name), "T0a1b")
        self.mutate()
        self.assertTrue(self.proj.restore(self.name, "1").ok)          # "1" is "001"
        self.mutate()
        result = self.proj.restore(self.name, "T0a1b")                 # the turn's BEFORE state
        self.assertTrue(result.ok, result.error)
        self.assertEqual(sorted(o.Name for o in App.getDocument(self.name).Objects), self.names_1)

    def test_a_missing_checkpoint_lists_the_available_ones(self):
        result = self.proj.restore(self.name, "042")
        self.assertFalse(result.ok)
        self.assertIn("'042' does not exist", result.error)
        self.assertIn("001", result.error)

    def test_a_checkpoint_that_is_not_an_fcstd_is_refused_before_anything_changes(self):
        with open(self.proj.path("checkpoints", "009.FCStd"), "wb") as f:
            f.write(b"not a zip")
        self.mutate()
        names = sorted(o.Name for o in self.doc.Objects)
        result = self.proj.restore(self.name, "009")
        self.assertFalse(result.ok)
        self.assertIn("not a valid FCStd", result.error)
        self.assertEqual(sorted(o.Name for o in App.getDocument(self.name).Objects), names)

    def test_a_zip_that_is_no_document_aborts_the_transaction(self):
        with zipfile.ZipFile(self.proj.path("checkpoints", "008.FCStd"), "w") as z:
            z.writestr("nothing.txt", "x")
        self.mutate()
        names = sorted(o.Name for o in self.doc.Objects)
        result = self.proj.restore(self.name, "008")
        self.assertFalse(result.ok)
        self.assertTrue(result.error)
        self.assertEqual(sorted(o.Name for o in App.getDocument(self.name).Objects), names)   # unchanged
        self.assertEqual(App.ActiveDocument.Name, self.name)   # still active

    def test_the_rescue_copy_puts_the_previous_state_back_when_the_result_does_not_validate(self):
        self.mutate()
        names_modified = sorted(o.Name for o in self.doc.Objects)
        side_path = self.proj.path("checkpoints", "001.json")
        side = json.load(open(side_path))
        side["all_objects"] = side["all_objects"] + ["GhostObject"]   # the checkpoint "should" contain an object it does not
        json.dump(side, open(side_path, "w"))
        result = self.proj.restore(self.name, "001")
        self.assertFalse(result.ok)
        self.assertIn("object names differ from the checkpoint", result.error)
        self.assertIn("restored from the rescue copy", result.error)
        self.assertEqual(sorted(o.Name for o in App.getDocument(self.name).Objects), names_modified)   # BACK to the modified state
        self.assertIsNotNone(App.getDocument(self.name).getObject("Ball"))
        self.assertTrue(os.path.isfile(self.proj.path("checkpoints", "_rescue.FCStd")))
        self.assertEqual(App.ActiveDocument.Name, self.name)

    def test_a_checkpoint_of_another_document_is_refused(self):
        other = self.new_model()
        result = self.proj.restore(other.Name, "001")
        self.assertFalse(result.ok)
        self.assertIn("belongs to document", result.error)

    def test_an_unknown_document_is_refused(self):
        result = self.proj.restore("NoSuchDocument", "001")
        self.assertFalse(result.ok)
        self.assertIn("is not open", result.error)

    def test_a_restore_never_makes_the_rescue_copy_a_checkpoint(self):
        self.mutate()
        self.proj.restore(self.name, "001")
        ids = [e["id"] for e in self.proj._checkpoint_entries()]
        self.assertNotIn("_rescue", ids)


class RewindTest(RealDocTestCase):
    def setUp(self):
        super().setUp()
        self.doc = self.new_model()
        self.name = self.doc.Name
        S.record(self.proj, doc=self.name, title="Base")                                  # step 1
        self.doc.addObject("Part::Sphere", "Ball")
        self.doc.recompute()
        S.record(self.proj, doc=self.name, title="Add ball", delta={"added": ["Ball"]})   # step 2
        self.doc.removeObject("Cyl")
        self.doc.recompute()
        S.record(self.proj, doc=self.name, title="Drop cyl", delta={"removed": ["Cyl"]})  # step 3

    def test_rewind_restores_the_newest_checkpoint_at_or_before_the_step(self):
        result = self.proj.rewind_to(2)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.checkpoint, "002")
        live = App.getDocument(self.name)
        self.assertIsNotNone(live.getObject("Ball"))
        self.assertIsNotNone(live.getObject("Cyl"))
        result = self.proj.rewind_to(1)
        self.assertTrue(result.ok, result.error)
        self.assertIsNone(App.getDocument(self.name).getObject("Ball"))

    def test_rewind_supersedes_later_steps_without_deleting_them(self):
        self.assertTrue(self.proj.rewind_to(1).ok)
        self.assertEqual([s.index for s in self.proj.read_steps()], [1])
        self.assertEqual([(s.index, s.superseded) for s in self.proj.read_steps(include_superseded=True)],
                         [(1, False), (2, True), (3, True)])
        last = json.loads(S.read_lines(self.proj)[-1])
        self.assertEqual((last["type"], last["to_step"], last["restored_checkpoint"]), ("rewind", 1, "001"))
        self.assertEqual(self.proj.manifest["last_geometry_step"], 1)
        self.assertEqual(json.load(open(self.proj.path("project.json")))["last_geometry_step"], 1)
        self.assertEqual(self.proj.last_step_index(), 3)   # indexes are never reused
        notes = open(self.proj.path("NOTES.md"), encoding="utf-8").read()
        self.assertIn("~~Add ball~~", notes)
        steps = ast_steps(self.proj.rebuild_script_text())
        self.assertEqual([s["index"] for s in steps], [1])   # rebuild.py only replays the live steps
        self.assertEqual(S.record(self.proj, doc=self.name, title="After rewind").index, 4)

    def test_rewind_with_a_document_filter_and_bad_arguments(self):
        self.assertFalse(self.proj.rewind_to(99).ok)
        self.assertIn("there is no step 99", self.proj.rewind_to(99).error)
        self.assertFalse(self.proj.rewind_to("x").ok)
        self.assertFalse(self.proj.rewind_to(-1).ok)
        result = self.proj.rewind_to(3, doc_name="NoSuchDocument")
        self.assertFalse(result.ok)
        self.assertIn("no checkpoint at or before step 3 for document 'NoSuchDocument'", result.error)
        self.assertEqual(self.proj.read_steps(include_superseded=True)[-1].index, 3)
        self.assertNotIn("rewind", [json.loads(l)["type"] for l in S.read_lines(self.proj)])   # nothing was journaled

    def test_a_failed_restore_journals_no_rewind(self):
        os.remove(self.proj.path("checkpoints", "001.FCStd"))
        result = self.proj.rewind_to(1)
        self.assertFalse(result.ok)
        self.assertNotIn("rewind", [json.loads(l)["type"] for l in S.read_lines(self.proj)])
        self.assertEqual([s.index for s in self.proj.read_steps()], [1, 2, 3])

    def test_rewind_ignores_a_step_without_a_checkpoint(self):
        """Step 4 changed nothing (no checkpoint): rewinding to it uses the checkpoint of step 3."""
        S.record(self.proj, doc=self.name, title="Nothing", delta={"added": [], "removed": [], "changed": []})
        result = self.proj.rewind_to(4)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.checkpoint, "003")


def ast_steps(text: str) -> list:
    import ast

    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "STEPS":
            return ast.literal_eval(node.value)
    return []


class ParamsMirrorTest(RealDocTestCase):
    def test_sync_params_writes_the_mirror_with_bound_counts(self):
        doc = self.new_model()
        S.record(self.proj, doc=doc.Name)
        rows = self.proj.sync_params(doc.Name)
        by_name = {r["name"]: r for r in rows}
        self.assertEqual(sorted(by_name), ["vent_count", "wall_t", "wall_x2"])
        wall = by_name["wall_t"]
        self.assertEqual((wall["value"], wall["unit"], wall["source"], wall["note"], wall["cell"]),
                         (2.0, "mm", "assumption", "typical FDM wall", "B2"))
        self.assertEqual(wall["expr"], "")           # "=2 mm" is a literal, not a formula
        self.assertEqual(wall["bound"], 1)           # Box.Length is bound to Params.wall_t
        self.assertEqual(by_name["wall_x2"]["expr"], "=wall_t * 2")
        self.assertEqual(by_name["wall_x2"]["value"], 4.0)
        self.assertEqual(by_name["vent_count"]["value"], 6)
        self.assertNotIn("Value", str(by_name["vent_count"]["value"]))   # a plain number: no Quantity leaks into the file
        mirror = json.load(open(self.proj.path("params.json")))
        self.assertEqual((mirror["schema"], mirror["doc"], mirror["sheet"], mirror["step"]), (1, doc.Name, "Params", 1))
        self.assertEqual([r["name"] for r in mirror["params"]], ["wall_t", "vent_count", "wall_x2"])

    def test_a_count_parameter_that_is_not_bound_needs_a_rebuild(self):
        doc = self.new_model()
        rows = {r["name"]: r for r in self.proj.sync_params(doc.Name)}
        self.assertTrue(rows["vent_count"]["needs_rebuild"])
        self.assertFalse(rows["wall_t"]["needs_rebuild"])

    def test_the_mirror_rows_are_exactly_the_rows_of_cad_health_read_param_table(self):
        """The sheet reader exists once (cad.health): the Project mirrors its rows verbatim."""
        from oscar.cad import health

        doc = self.new_model()
        canonical = health.read_param_table(doc)
        table = gui.param_table(doc)
        self.assertEqual(table, {"sheet": canonical["sheet"], "rows": canonical["rows"]})
        self.assertEqual(table["sheet"], "Params")
        by_name = {r["name"]: r for r in table["rows"]}
        self.assertEqual(sorted(by_name), ["vent_count", "wall_t", "wall_x2"])
        for row in table["rows"]:
            self.assertEqual(sorted(row), ["bound", "cell", "expr", "name", "needs_rebuild", "note", "source", "unit", "value"])
        self.assertEqual((by_name["wall_t"]["bound"], by_name["wall_t"]["value"]), (1, 2.0))
        self.assertEqual(by_name["wall_x2"]["expr"], "=wall_t * 2")
        self.assertTrue(by_name["vent_count"]["needs_rebuild"])
        self.assertEqual(self.proj.sync_params(doc.Name), canonical["rows"])
        self.assertFalse(hasattr(gui, "fallback_param_table"), "the private scanner must be gone: one reader")

    def test_a_document_without_a_sheet_writes_nothing(self):
        doc = self.new_model(with_params=False)
        self.assertEqual(self.proj.sync_params(doc.Name), [])
        self.assertFalse(os.path.exists(self.proj.path("params.json")))
        self.assertEqual(gui.param_table(doc), {"sheet": "", "rows": []})

    def test_the_parameters_sheet_name_is_accepted(self):
        doc = App.newDocument(unique("T3P"))
        sheet = doc.addObject("Spreadsheet::Sheet", "Parameters")
        sheet.set("A2", "wall_t")
        sheet.set("B2", "=2 mm")
        sheet.setAlias("B2", "wall_t")
        doc.recompute()
        self.assertEqual([r["name"] for r in self.proj.sync_params(doc.Name)], ["wall_t"])
        self.assertEqual(json.load(open(self.proj.path("params.json")))["sheet"], "Parameters")

    def test_a_non_geometry_parameter_tool_also_syncs(self):
        doc = self.new_model()
        rec = S.record(self.proj, doc=doc.Name, tool="measure_reference_part", geometry=False, args={"doc_name": doc.Name})
        self.assertGreater(rec.index, 0)
        self.assertEqual(self.proj.read_params()["doc"], doc.Name)

    def test_a_formula_error_cell_keeps_its_row(self):
        doc = self.new_model()
        doc.getObject("Params").set("B3", "=1 mm/0")
        doc.recompute()
        rows = {r["name"]: r for r in self.proj.sync_params(doc.Name)}
        self.assertIn("vent_count", rows)
        json.dumps(rows["vent_count"])


class BlankFrameTest(unittest.TestCase):
    """The blank-frame rule on a 64 x 48 sample grid."""

    def grid(self, colour=(40, 40, 40), changed=0, delta=0):
        """A uniform grid with ``changed`` samples shifted by ``delta``, spread over the rows (one column per pass) so
        that no row's median colour flips."""
        rows = [[colour for _ in range(64)] for _ in range(48)]
        for n in range(changed):
            y, x = n % 48, 8 + n // 48
            rows[y][x] = tuple(c + delta for c in colour)
        return rows

    def test_a_uniform_frame_is_blank(self):
        self.assertTrue(gui.is_blank_grid(self.grid()))

    def test_an_empty_grid_is_blank(self):
        self.assertTrue(gui.is_blank_grid([]))
        self.assertTrue(gui.is_blank_grid([[], []]))

    def test_a_vertical_gradient_is_still_blank(self):
        rows = [[(20 + y * 3, 20 + y * 3, 40 + y * 2)] * 64 for y in range(48)]
        self.assertTrue(gui.is_blank_grid(rows))

    def test_a_small_object_makes_it_not_blank(self):
        self.assertFalse(gui.is_blank_grid(self.grid(changed=200, delta=90)))

    def test_the_threshold_is_more_than_99_5_percent(self):
        total = 64 * 48
        limit = int(total * 0.005)   # 15 samples
        self.assertTrue(gui.is_blank_grid(self.grid(changed=limit, delta=90)))        # 99.51% background
        self.assertFalse(gui.is_blank_grid(self.grid(changed=limit + 1, delta=90)))   # 99.48% background

    def test_anti_aliasing_within_six_per_channel_is_background(self):
        self.assertTrue(gui.is_blank_grid(self.grid(changed=500, delta=6)))
        self.assertFalse(gui.is_blank_grid(self.grid(changed=500, delta=7)))

    def test_a_headless_session_has_no_snapshot(self):
        if getattr(App, "GuiUp", False):
            self.skipTest("a GUI session")
        path = os.path.join(S.sandbox_dir(), "x.png")
        self.assertEqual(gui.capture_snapshot("Doc", path, 480, 360), (False, "no 3D view (headless session)"))
        self.assertFalse(os.path.exists(path))


class CaptureSnapshotTest(unittest.TestCase):
    """``capture_snapshot`` in a simulated GUI session (a fake FreeCADGui and a fake ``oscar.cad.views``)."""

    def setUp(self):
        import types
        from unittest import mock

        self.mock = mock
        self.calls = []
        self.fit_calls = []
        self.fake_gui = types.ModuleType("FreeCADGui")
        self.fake_gui.updateGui = lambda: self.calls.append("updateGui")
        self.views = types.ModuleType("oscar.cad.views")
        self.views.fit_view = lambda: self.fit_calls.append(1)
        import oscar.cad as cad_pkg

        patches = [mock.patch.object(App, "GuiUp", True, create=True),
                   mock.patch.dict(sys.modules, {"FreeCADGui": self.fake_gui, "oscar.cad.views": self.views}),
                   mock.patch.object(cad_pkg, "views", self.views, create=True)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.path = os.path.join(S.sandbox_dir(), "snap.png")

    def test_a_good_capture_passes_through(self):
        def checked(path, orientation, width, height, focus, doc_name):
            self.calls.append((orientation, width, height, focus, doc_name))
            open(path, "wb").write(b"png")
            return True, ""

        self.views.capture_checked = checked
        self.assertEqual(gui.capture_snapshot("PiCase", self.path, 480, 360), (True, ""))
        self.assertEqual(self.calls, [("Isometric", 480, 360, None, "PiCase")])

    def test_the_reason_of_a_failed_capture_is_passed_on(self):
        self.views.capture_checked = lambda *a, **k: (False, "blank frame")
        self.assertEqual(gui.capture_snapshot("PiCase", self.path, 480, 360), (False, "blank frame"))
        self.views.capture_checked = lambda *a, **k: (False, "")
        self.assertEqual(gui.capture_snapshot("PiCase", self.path, 480, 360), (False, "no 3D view available"))

    def test_an_exception_in_the_capture_is_a_note_not_a_crash(self):
        def boom(*a, **k):
            raise RuntimeError("view gone")

        self.views.capture_checked = boom
        ok, note = gui.capture_snapshot("PiCase", self.path, 480, 360)
        self.assertFalse(ok)
        self.assertIn("view gone", note)

    def test_without_capture_checked_a_blank_frame_gets_one_retry(self):
        captures = []
        self.views.capture = lambda path, *a: captures.append(path) or True
        blank = [[(10, 10, 10)] * 64 for _ in range(48)]
        busy = [[(10, 10, 10)] * 32 + [(200, 90, 40)] * 16 + [(10, 10, 10)] * 16 for _ in range(48)]
        grids = iter([blank, busy])
        with self.mock.patch.object(gui, "_sample_image", lambda path: next(grids)):
            self.assertEqual(gui.capture_snapshot("PiCase", self.path, 480, 360), (True, ""))
        self.assertEqual(len(captures), 2)
        self.assertEqual(self.fit_calls, [1])
        self.assertIn("updateGui", self.calls)

    def test_without_capture_checked_a_frame_that_stays_blank_is_reported(self):
        self.views.capture = lambda path, *a: True
        blank = [[(10, 10, 10)] * 64 for _ in range(48)]
        with self.mock.patch.object(gui, "_sample_image", lambda path: blank):
            self.assertEqual(gui.capture_snapshot("PiCase", self.path, 480, 360), (False, "blank frame"))
        self.assertEqual(self.fit_calls, [1])

    def test_without_capture_checked_a_missing_view_is_reported(self):
        self.views.capture = lambda path, *a: False
        self.assertEqual(gui.capture_snapshot("PiCase", self.path, 480, 360), (False, "no 3D view available"))


class ExportTest(RealDocTestCase):
    def solid_doc(self):
        doc = App.newDocument(unique("T3E"))
        box = doc.addObject("Part::Box", "Box")
        box.Length, box.Width, box.Height = 20, 10, 5
        cyl = doc.addObject("Part::Cylinder", "Cyl")
        cyl.Radius, cyl.Height = 2, 10
        cyl.Placement.Base = App.Vector(5, 5, -1)
        cut = doc.addObject("Part::Cut", "Cut")
        cut.Base, cut.Tool = box, cyl
        doc.recompute()
        return doc

    def read_volume(self, path):
        import Part

        shape = Part.read(path)
        return shape.Volume

    def test_the_default_exports_step_and_stl_of_the_visible_tips_only(self):
        doc = self.solid_doc()
        result = X.export_tips(self.proj, doc.Name)
        self.assertEqual(result.errors, [])
        self.assertEqual(sorted(f.format for f in result.files), ["step", "stl"])
        self.assertEqual({f.label for f in result.files}, {"Cut"})   # the boolean's inputs are not tips
        for item in result.files:
            self.assertTrue(os.path.isfile(item.path))
            self.assertTrue(item.rel.startswith("exports/"), item.rel)
            self.assertEqual(item.bytes, os.path.getsize(item.path))
            self.assertEqual(len(item.sha256), 64)
            self.assertEqual(item.solids, 1)
        step = next(f for f in result.files if f.format == "step")
        self.assertAlmostEqual(self.read_volume(step.path), doc.getObject("Cut").Shape.Volume, delta=1e-3)

    def test_all_four_formats(self):
        import hashlib

        doc = self.solid_doc()
        result = X.export_tips(self.proj, doc.Name, formats=["step", "stl", "3mf", "brep"])
        self.assertEqual(result.errors, [])
        self.assertEqual(sorted(os.path.splitext(f.path)[1] for f in result.files), [".3mf", ".brep", ".step", ".stl"])
        for item in result.files:
            self.assertGreater(item.bytes, 0)
            self.assertEqual(hashlib.sha256(open(item.path, "rb").read()).hexdigest(), item.sha256)
        self.assertAlmostEqual(self.read_volume(next(f.path for f in result.files if f.format == "brep")),
                               doc.getObject("Cut").Shape.Volume, delta=1e-3)

    def test_names_are_unique_across_exports_and_no_temp_files_remain(self):
        doc = self.solid_doc()
        first = X.export_tips(self.proj, doc.Name, formats=["step"])
        second = X.export_tips(self.proj, doc.Name, formats=["step"])
        self.assertNotEqual(first.files[0].path, second.files[0].path)
        self.assertTrue(second.files[0].path.endswith("-2.step"))
        leftovers = [n for n in os.listdir(self.proj.path("exports")) if n.startswith(".") or ".tmp" in n]
        self.assertEqual(leftovers, [])

    def test_selected_objects_and_the_combined_file(self):
        doc = self.solid_doc()
        result = X.export_tips(self.proj, doc.Name, objects=["Box", "Cyl"], formats=["step"])
        self.assertEqual(sorted(f.label for f in result.files), ["Box", "Cyl"])
        combined = X.export_tips(self.proj, doc.Name, objects=["Box", "Cyl"], formats=["step"], per_part=False)
        self.assertEqual(len(combined.files), 1)
        self.assertEqual((combined.files[0].label, combined.files[0].solids), ("model", 2))

    def test_errors_name_the_problem_and_what_is_accepted(self):
        doc = self.solid_doc()
        for kwargs, needle in (({"objects": ["Nope"]}, "no such object(s) 'Nope'"),
                               ({"formats": ["dwg"]}, "unknown format"),
                               ({"stl_tolerance": 99}, "outside 0.001..5"),
                               ({"stl_tolerance": "x"}, "must be a number"),
                               ({"directory": "relative/dir"}, "must be an absolute path")):
            with self.assertRaises(P.ProjectError) as caught:
                X.export_tips(self.proj, doc.Name, **kwargs)
            self.assertIn(needle, str(caught.exception))
        with self.assertRaises(P.ProjectError) as caught:
            X.export_tips(self.proj, "NoSuchDocument")
        self.assertIn("is not open", str(caught.exception))
        empty = App.newDocument(unique("T3Z"))
        with self.assertRaises(P.ProjectError) as caught:
            X.export_tips(self.proj, empty.Name)
        self.assertIn("no visible solid part", str(caught.exception))
        with self.assertRaises(P.ProjectError) as caught:
            X.export_tips(None, doc.Name)
        self.assertIn("pass 'directory'", str(caught.exception))
        self.assertEqual(os.listdir(self.proj.path("exports")), [])   # failed exports leave no file and no temp folder

    def test_a_directory_override_works_without_a_project(self):
        doc = self.solid_doc()
        target = os.path.join(S.sandbox_dir(), "out")
        result = X.export_tips(None, doc.Name, formats=["stl"], directory=target)
        self.assertEqual(len(result.files), 1)
        self.assertEqual(os.path.dirname(result.files[0].path), target)
        self.assertEqual(result.files[0].rel, result.files[0].path)   # absolute: outside any project
        self.assertEqual([n for n in os.listdir(target) if n.startswith(".")], [])

    def test_parts_inside_an_app_part_are_exported_in_the_global_frame(self):
        import Part  # noqa: F401

        doc = App.newDocument(unique("T3G"))
        container = doc.addObject("App::Part", "Assembly")
        box = doc.addObject("Part::Box", "Box")
        container.addObject(box)
        container.Placement.Base = App.Vector(0, 0, 50)
        doc.recompute()
        result = X.export_tips(self.proj, doc.Name, formats=["step"])
        self.assertEqual([f.label for f in result.files], ["Box"])
        shape = __import__("Part").read(result.files[0].path)
        self.assertAlmostEqual(shape.BoundBox.ZMin, 50.0, places=4)   # the container's placement is applied

    def test_print_orientation_follows_the_spec_and_can_be_switched_off(self):
        import Part

        doc = App.newDocument(unique("T3L"))
        lid = doc.addObject("Part::Feature", "Lid")
        plate = Part.makeBox(40, 30, 2)
        skirt = Part.makeBox(20, 20, 6, App.Vector(10, 5, 2))
        lid.Shape = plate.fuse(skirt)
        doc.recompute()
        before = lid.Shape.Solids[0].CenterOfMass.z   # 3.0: the plate is at the bottom as modelled
        self.proj.save_spec({"name": "case", "print_orientation": {"shell": "floor_down", "lid": "top_down"}})
        flipped = X.export_tips(self.proj, doc.Name, formats=["step"])
        shape = Part.read(flipped.files[0].path)
        self.assertAlmostEqual(shape.BoundBox.ZMin, 0.0, places=4)
        self.assertAlmostEqual(shape.Solids[0].CenterOfMass.z, 8.0 - before, places=3)   # plate on top: flipped about X, dropped to z=0
        as_modelled = X.export_tips(self.proj, doc.Name, formats=["step"], print_orientation=False)
        self.assertAlmostEqual(Part.read(as_modelled.files[0].path).Solids[0].CenterOfMass.z, before, places=3)
        self.proj.save_spec({"name": "case", "print_orientation": {"lid": "as_modeled"}})
        modelled = X.export_tips(self.proj, doc.Name, formats=["step"])
        self.assertAlmostEqual(Part.read(modelled.files[0].path).Solids[0].CenterOfMass.z, before, places=3)

    def test_file_names_use_the_spec_name_and_the_part_label(self):
        doc = self.solid_doc()
        self.proj.save_spec({"name": "Pi4-Case"})
        result = X.export_tips(self.proj, doc.Name, formats=["step"])
        self.assertEqual(os.path.basename(result.files[0].path), "pi4-case-cut.step")

    def test_the_hidden_reference_is_never_exported(self):
        import Part

        doc = self.solid_doc()
        ref = doc.addObject("Part::Feature", "Ref_RPI4B")
        ref.Shape = Part.makeBox(5, 5, 5)
        doc.recompute()
        result = X.export_tips(self.proj, doc.Name, formats=["step"])
        self.assertEqual({f.label for f in result.files}, {"Cut"})


class ReplayEqualityTest(RealDocTestCase):
    """Build a small model through run_tool with a bound Project, then replay it into a fresh document."""

    SCRIPT = ("# Plate with a hole\nimport Part\n"
              "doc = App.getDocument('{doc}')\n"
              "plate = doc.addObject('Part::Box', 'Plate')\nplate.Length = 30\nplate.Width = 20\nplate.Height = 4\n"
              "hole = doc.addObject('Part::Cylinder', 'Hole')\nhole.Radius = 3\nhole.Height = 10\n"
              "hole.Placement.Base = App.Vector(15, 10, -1)\n"
              "cut = doc.addObject('Part::Cut', 'PlateCut')\ncut.Base = plate\ncut.Tool = hole\n"
              "doc.recompute()\nprint('made', cut.Shape.Volume)\n")

    def build(self):
        from oscar import tools

        name = unique("T3R")
        ctx = ToolContext(project=self.proj, turn_id="T20261001-000001-0a1b")
        self.proj.begin_turn(ctx.turn_id, "make a plate", work=True)
        results = []

        def call(tool, **args):
            ctx.call_id = f"c{len(results) + 1}"
            with use_context(ctx):
                results.append(tools.run_tool(tool, args))
            self.assertFalse(results[-1][2], results[-1][0])
            return results[-1][0]

        call("create_document", name=name)
        call("create_object", doc_name=name, obj_type="Part::Box", obj_name="Block",
             obj_properties={"Length": 12, "Width": 8, "Height": 6}, include_screenshot=False)
        call("execute_code", code=self.SCRIPT.format(doc=name))
        call("edit_object", doc_name=name, obj_name="Block", obj_properties={"Height": 9}, include_screenshot=False)
        return name

    def live(self, name):
        return gui.live_volumes(App.getDocument(name))

    def test_replay_steps_reproduces_the_object_set_and_the_volumes(self):
        name = self.build()
        live = self.live(name)
        self.assertEqual(sorted(live), ["Block", "Hole", "Plate", "PlateCut"])
        steps = self.proj.read_steps()
        self.assertEqual([s.tool for s in steps], ["create_document", "create_object", "execute_code", "edit_object"])
        S.close_docs([name])   # a clean session, as rebuild.py runs: the recorded script addresses the document by name
        result = P.replay_steps(steps, project_root=self.proj.root, doc_name=name)
        self.assertTrue(result.ok, result.results)
        self.assertEqual([r["index"] for r in result.results], [1, 2, 3, 4])
        self.assertEqual(result.name_map, {name: name})
        rebuilt = self.live(result.name_map[name])
        self.assertEqual(sorted(rebuilt), sorted(live))
        for obj, volume in live.items():
            self.assertAlmostEqual(rebuilt[obj], volume, delta=abs(volume) * 1e-6 + 1e-9, msg=obj)
        self.assertEqual(P.replay_summary(result), {"docs": {result.name_map[name]: rebuilt}})

    def _override_run(self, fail_override=False):
        calls = []
        made = []

        def fake_run_tool(tool, args):
            calls.append((tool, dict(args)))
            if tool == "create_document":
                made.append(App.newDocument(args["name"]).Name)
                return f"Document '{made[-1]}' created and active.", None, False
            if tool == "set_parameter" and fail_override:
                return "set_parameter: 'wall_t' is not a parameter in sheet 'Params'.", None, True
            return "ok", None, False

        return calls, fake_run_tool

    def test_overrides_are_applied_through_set_parameter_after_set_parameters(self):
        nm = unique("T3O")
        calls, fake = self._override_run()
        steps = P.steps_from_literal([
            {"index": 1, "tool": "create_document", "kind": "tool", "args": {"name": nm}, "doc": nm},
            {"index": 2, "tool": "set_parameters", "kind": "tool", "args": {"doc_name": nm, "params": []}, "doc": nm}])
        result = P.replay_steps(steps, project_root=self.proj.root, doc_name=nm, overrides={"wall_t": "2.4"}, run_tool=fake)
        self.assertTrue(result.ok, result.results)
        self.assertEqual([c[0] for c in calls], ["create_document", "set_parameters", "set_parameter"])
        self.assertEqual(calls[2][1], {"doc_name": nm, "name": "wall_t", "value": "2.4"})
        self.assertTrue(result.results[-1]["override"])

    def test_an_override_replaces_the_value_of_a_recorded_set_parameter(self):
        nm = unique("T3O")
        calls, fake = self._override_run()
        steps = P.steps_from_literal([
            {"index": 1, "tool": "create_document", "kind": "tool", "args": {"name": nm}, "doc": nm},
            {"index": 2, "tool": "set_parameter", "kind": "tool", "args": {"doc_name": nm, "name": "wall_t", "value": "2"}}])
        result = P.replay_steps(steps, project_root=self.proj.root, doc_name=nm, overrides={"wall_t": "2.4"}, run_tool=fake)
        self.assertTrue(result.ok, result.results)
        self.assertEqual([c[0] for c in calls], ["create_document", "set_parameter"])
        self.assertEqual(calls[1][1]["value"], "2.4")

    def test_an_override_that_cannot_be_applied_fails_the_replay_visibly(self):
        nm = unique("T3O")
        calls, fake = self._override_run(fail_override=True)
        steps = P.steps_from_literal([
            {"index": 1, "tool": "create_document", "kind": "tool", "args": {"name": nm}, "doc": nm},
            {"index": 2, "tool": "set_parameters", "kind": "tool", "args": {"doc_name": nm, "params": []}, "doc": nm}])
        result = P.replay_steps(steps, project_root=self.proj.root, doc_name=nm, overrides={"wall_t": "2.4"}, run_tool=fake)
        self.assertFalse(result.ok)
        self.assertIn("override 'wall_t=2.4' was not applied", result.results[-1]["text"])

    def test_overrides_without_any_document_fail_the_replay(self):
        result = P.replay_steps([], project_root=self.proj.root, doc_name="X", overrides={"wall_t": "2.4"},
                                run_tool=lambda t, a: ("ok", None, False))
        self.assertFalse(result.ok)   # asking for an override that has no document to land in is a failure, not a no-op
        self.assertEqual(result.results[-1]["text"], "overrides were not applied: the replay created no document")

    def test_a_tool_only_replay_next_to_the_open_original_uses_a_sibling_document(self):
        name = unique("T3K")
        from oscar import tools

        ctx = ToolContext(project=self.proj, turn_id="T20261001-000001-0a1b")
        with use_context(ctx):
            tools.run_tool("create_document", {"name": name})
            tools.run_tool("create_object", {"doc_name": name, "obj_type": "Part::Box", "obj_name": "Block",
                                             "obj_properties": {"Length": 3, "Width": 4, "Height": 5}, "include_screenshot": False})
        live = self.live(name)
        result = P.replay_steps(self.proj.read_steps(), project_root=self.proj.root, doc_name=name)
        self.assertTrue(result.ok, result.results)
        self.assertEqual(result.name_map, {name: f"{name}_rebuild"})
        self.assertEqual(self.live(f"{name}_rebuild"), live)
        self.assertEqual(self.live(name), live)   # the original was not touched

    def test_a_script_replay_is_refused_while_the_original_document_is_open(self):
        name = self.build()
        before = self.live(name)
        result = P.replay_steps(self.proj.read_steps(), project_root=self.proj.root, doc_name=name)
        self.assertFalse(result.ok)
        self.assertIn(f"document '{name}' is open in this session", result.error)
        self.assertEqual(self.live(name), before)   # nothing ran: the original model is intact
        self.assertEqual(len(result.results), 1)

    def test_replay_stops_at_the_first_failure_unless_told_otherwise(self):
        name = self.build()
        steps = self.proj.read_steps()
        steps[1].replay["args"]["obj_type"] = "Part::NoSuchType"
        S.close_docs([name])
        stopped = P.replay_steps(steps, project_root=self.proj.root, doc_name=name)
        self.assertFalse(stopped.ok)
        self.assertEqual([r["ok"] for r in stopped.results], [True, False])
        S.close_docs(list(stopped.name_map.values()))
        again = P.replay_steps(steps, project_root=self.proj.root, doc_name=name, continue_on_error=True)
        self.assertFalse(again.ok)
        self.assertGreater(len(again.results), 2)

    def test_replay_skips_superseded_failed_and_unreplayable_steps(self):
        name = self.build()
        S.record(self.proj, tool="create_object", ok=False, args={"doc_name": name, "obj_type": "Bad"}, doc=name,
                 text="boom")
        S.record(self.proj, tool="execute_code_async", args={"code": "x"}, doc=name)
        steps = self.proj.read_steps()
        S.close_docs([name])
        result = P.replay_steps(steps, project_root=self.proj.root, doc_name=name)
        self.assertTrue(result.ok, result.results)
        self.assertEqual([r["index"] for r in result.results], [1, 2, 3, 4])

    def test_a_gui_only_script_is_reported_not_run(self):
        name = unique("T3Q")
        S.record(self.proj, tool="create_document", geometry=False, args={"name": name}, text=f"Document '{name}' created and active.")
        S.record(self.proj, tool="execute_code", doc=name, args={"code": "o = App.ActiveDocument\no.ViewObject.x = 1"})
        steps = self.proj.read_steps()
        result = P.replay_steps(steps, project_root=self.proj.root, doc_name=name)
        self.assertFalse(result.ok)
        self.assertEqual(result.results[-1]["text"], "not headless-replayable (uses ViewObject/Gui)")

    def test_a_missing_script_file_is_a_failed_row_not_an_exception(self):
        name = self.build()
        os.remove(self.proj.path(self.proj.read_steps()[2].script))
        S.close_docs([name])
        result = P.replay_steps(self.proj.read_steps(), project_root=self.proj.root, doc_name=name)
        self.assertFalse(result.ok)
        self.assertIn("cannot be read", result.results[-1]["text"])

    def test_the_generated_rebuild_py_reproduces_the_model(self):
        """Run the generated text itself (in-process; the real headless run is export_rebuild_script(run_check))."""
        name = self.build()
        live = self.live(name)
        self.proj.write_rebuild_py()
        text = open(self.proj.path("rebuild.py"), encoding="utf-8").read()
        S.close_docs([name])   # rebuild.py runs in a clean session
        out = io.StringIO()
        code = None
        with contextlib.redirect_stdout(out):
            try:
                exec(compile(text, "rebuild.py", "exec"), {"__name__": "__main__"})
            except SystemExit as exit_:
                code = exit_.code
        output = out.getvalue()
        self.assertEqual(code, 0, output)
        marker, summary = R.parse_rebuild_output(output)
        self.assertTrue(marker, output)
        self.assertIn("PASS step 003 execute_code", output)
        S.close_docs(list(summary["docs"]))
        rebuilt = next(iter(summary["docs"].values()))
        self.assertEqual(R.compare_rebuild({"x": live}, {"x": rebuilt}), [])

    def test_a_broken_replay_is_reported_by_the_comparison(self):
        name = self.build()
        live = self.live(name)
        steps = self.proj.read_steps()
        steps[3].replay["args"]["obj_properties"]["Height"] = 99   # a deliberately corrupted copy of one step
        S.close_docs([name])
        result = P.replay_steps(steps, project_root=self.proj.root, doc_name=name)
        self.assertTrue(result.ok)
        rebuilt = P.replay_summary(result)["docs"][result.name_map[name]]
        problems = R.compare_rebuild({"x": live}, {"x": rebuilt})
        self.assertTrue(any("Block" in p for p in problems), problems)

    def test_run_tool_takes_the_turn_checkpoint_just_in_time_before_the_first_geometry_step(self):
        name = self.build()
        side = json.load(open(self.proj.path("checkpoints", "T0a1b.json")))
        self.assertEqual((side["doc"], side["turn"], side["objects"]), (name, "T20261001-000001-0a1b", []))   # the BEFORE state: empty
        self.assertEqual(sorted(os.listdir(self.proj.path("checkpoints"))).count("T0a1b.FCStd"), 1)
        result = self.proj.restore(name, "T0a1b")   # what the panel's turn revert falls back to
        self.assertTrue(result.ok, result.error)
        self.assertEqual([o.Name for o in App.getDocument(name).Objects], [])

    def test_the_hook_recorded_artifacts_of_the_built_model(self):
        name = self.build()
        steps = self.proj.read_steps()
        self.assertEqual([s.ok for s in steps], [True] * 4)
        self.assertEqual([bool(s.checkpoint) for s in steps], [False, True, True, True])   # create_document is not geometry
        self.assertTrue(all(s.health and s.health["valid"] for s in steps if s.geometry), [s.health for s in steps])
        self.assertTrue(os.path.isfile(self.proj.path("model.FCStd")))
        self.assertEqual(self.proj.manifest["active_doc"], name)
        self.proj.save_ledger({"intent": "other"})
        self.assertEqual(self.proj.verify_artifacts(), [])


if __name__ == "__main__":
    unittest.main()
