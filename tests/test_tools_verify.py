"""The verification tools (``oscar.tools_verify``): ``validate_model``, ``check_fit``,
``run_validation`` and the ``feature_tree_status`` delegation, on REAL FreeCAD documents under ``freecadcmd`` (the
dispatcher runs inline there).

Covered: the exact text of both tools, a failing validation, the Origin-feature and ``App::Part`` traps, the gate rules
(baseline / touched / Ref_* findings never fail the gate of a user's pre-existing problem), Project and Ledger folding,
Stop, argument errors, ``deep`` editability, and the registry facts of the two tools."""

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

import FreeCAD as App  # noqa: E402
import Part  # noqa: E402
import verify_support as S  # noqa: E402

import oscar  # noqa: E402
from oscar import tools, tools_verify as TV  # noqa: E402
from oscar.cad import dispatch, health, health_eval  # noqa: E402
from oscar.cad.health_types import Finding, HealthReport  # noqa: E402
from oscar.flow import Ledger  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402


class VerifyCase(unittest.TestCase):
    """A fresh FreeCAD document per call of ``self.new_doc``; everything is closed afterwards."""

    def setUp(self):
        health_eval.clear_cache()
        self._made = []
        self.addCleanup(self._close)

    def _close(self):
        for name in self._made:
            if name in App.listDocuments():
                App.closeDocument(name)
        health_eval.clear_cache()

    def new_doc(self, name="VT"):
        doc = App.newDocument(name)
        self._made.append(doc.Name)
        return doc

    def enclosure(self, **kw):
        doc = self.new_doc("Enc")
        self.objs = S.build_enclosure(doc, **kw)
        return doc

    def run_tool(self, name, ctx=None, **args):
        with use_context(ctx or ToolContext()):
            return tools.run_tool(name, args)

    def broken_cut(self, doc, name="UserCut"):
        """A Cut with a Base and no Tool: an error in STRICT mode (BOOLEAN_INPUT_MISSING + RECOMPUTE_ERROR)."""
        cut = doc.addObject("Part::Cut", name)
        cut.Base = doc.addObject("Part::Box", name + "Base")
        doc.recompute()
        return cut


# ---------------------------------------------------------------------------------------------- validate_model


class ValidateModelTest(VerifyCase):
    def test_a_sound_model_is_valid_and_prints_the_documented_blocks(self):
        doc = self.enclosure()
        text, image, is_error = self.run_tool("validate_model", doc_name=doc.Name)
        self.assertFalse(is_error, text)
        self.assertIsNone(image)
        lines = text.split("\n")
        self.assertEqual(lines[0], f"VALIDATE doc '{doc.Name}' (deep=False)")
        self.assertTrue(lines[1].startswith(f"[health] doc '{doc.Name}': "), lines[1])
        self.assertIn("VALID", lines[1])
        self.assertIn("PARAMETERS (sheet 'Params', 6 rows, 0 bound to objects)", text)
        self.assertIn("wall_t", text)
        self.assertEqual(lines[-1], "RESULT: VALID")
        self.assertNotIn("EDITABILITY", text)

    def test_the_parameter_table_counts_bound_rows(self):
        doc = self.new_doc("Par")
        S.build_parametric(doc)
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name)
        self.assertFalse(is_error, text)
        self.assertIn("PARAMETERS (sheet 'Params', 4 rows, 4 bound to objects)", text)
        header = next(line for line in text.split("\n") if line.startswith("name"))
        self.assertEqual(header.split(), ["name", "value", "unit", "source", "bound"])
        self.assertRegex(text, r"slot\s+8\s+mm\s+assumption\s+1")

    def test_include_params_false_leaves_the_table_out(self):
        doc = self.enclosure()
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name, include_params=False)
        self.assertFalse(is_error)
        self.assertNotIn("PARAMETERS", text)
        self.assertTrue(text.endswith("RESULT: VALID"))

    def test_a_document_without_a_parameter_sheet_says_so(self):
        doc = self.new_doc()
        doc.addObject("Part::Box", "Box")
        doc.recompute()
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name)
        self.assertFalse(is_error)
        self.assertIn("PARAMETERS: no 'Params' or 'Parameters' spreadsheet in this document", text)

    def test_a_long_parameter_table_is_capped_at_40_rows(self):
        doc = self.new_doc()
        S.params(doc, [(f"param_{i:02d}", f"={i + 1} mm", "mm", "assumption") for i in range(45)])
        text, _, _ = self.run_tool("validate_model", doc_name=doc.Name)
        self.assertIn("PARAMETERS (sheet 'Params', 45 rows, 0 bound to objects)", text)
        self.assertIn("param_39", text)
        self.assertNotIn("param_40 ", text)
        self.assertIn("(+5 more)", text)

    def test_an_unknown_document_is_a_model_readable_error(self):
        self.new_doc("Known")
        text, _, is_error = self.run_tool("validate_model", doc_name="Nope")
        self.assertTrue(is_error)
        self.assertIn("Document 'Nope' is not open", text)
        self.assertIn("Known", text)

    def test_a_failing_model_is_invalid_with_the_findings_and_is_an_error(self):
        doc = self.enclosure()
        # a shell that is two floating pieces, and a boolean with no tool
        doc.removeObject("Shell")
        S.feature(doc, "Shell", Part.makeCompound([S.box(0, 0, 0, 5, 5, 5), S.box(20, 0, 0, 5, 5, 5)]),
                  role="shell", stage="shell", part="Shell")
        self.broken_cut(doc)
        doc.recompute()
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name)
        self.assertTrue(is_error)
        self.assertRegex(text, r"RESULT: INVALID \(\d+ errors\)")
        self.assertIn("ERROR [FRAGMENTS] Shell", text)
        self.assertIn("UserCut", text)
        self.assertTrue(text.startswith(f"VALIDATE doc '{doc.Name}' (deep=False)"))

    def test_a_single_error_says_error_not_errors(self):
        doc = self.new_doc()
        S.feature(doc, "Shell", Part.makeCompound([S.box(0, 0, 0, 5, 5, 5), S.box(20, 0, 0, 5, 5, 5)]), role="shell")
        doc.recompute()
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name)
        self.assertTrue(is_error)
        self.assertIn("RESULT: INVALID (1 error)", text)

    def test_origin_features_never_show_up_as_geometry(self):
        """The Origin of a Body / App::Part reports a 2e+100 bounding box and a garbage volume."""
        doc = self.new_doc()
        body = doc.addObject("PartDesign::Body", "Body")
        body.newObject("PartDesign::AdditiveBox", "AddBox")
        part = doc.addObject("App::Part", "Part")
        box = doc.addObject("Part::Box", "Cube")
        part.addObject(box)
        doc.recompute()
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name)
        self.assertNotIn("e+", text)
        self.assertNotIn("X_Axis", text)
        self.assertNotIn("XY_Plane", text)
        self.assertNotIn("Origin", text)
        self.assertIn("Cube", text)
        self.assertEqual(text.split("\n")[1].split(": ")[1].split(",")[1].strip(), "2 solids")
        self.assertFalse(is_error, text)

    def test_a_shape_inside_a_placed_part_is_recorded_in_the_global_frame(self):
        doc = self.new_doc()
        part = doc.addObject("App::Part", "Holder")
        cube = doc.addObject("Part::Box", "Cube")
        part.addObject(cube)
        part.Placement = App.Placement(App.Vector(0, 0, 50), App.Rotation())
        doc.recompute()
        project = S.FakeProject()
        self.run_tool("validate_model", ToolContext(project=project), doc_name=doc.Name)
        rows = {r["name"]: r for r in project.recorded[-1]["health"]["rows"]}
        self.assertEqual((rows["Cube"]["bbox"][2], rows["Cube"]["bbox"][5]), (50.0, 60.0))
        self.assertEqual(cube.Shape.BoundBox.ZMin, 0.0)       # the container frame (the trap)

    def test_the_stop_button_abandons_the_check(self):
        doc = self.enclosure()
        text, _, is_error = self.run_tool("validate_model", ToolContext(should_stop=lambda: True), doc_name=doc.Name)
        self.assertTrue(is_error)
        self.assertIn("could not be checked", text)
        self.assertIn("Stop", text)

    def test_it_works_without_any_bound_context(self):
        doc = self.enclosure()
        text, _, is_error = tools.run_tool("validate_model", {"doc_name": doc.Name})
        self.assertFalse(is_error, text)

    def test_arguments_are_coerced_from_strings_like_every_tool(self):
        doc = self.enclosure()
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name, deep="false", include_params="false")
        self.assertFalse(is_error, text)
        self.assertIn("(deep=False)", text)
        self.assertNotIn("PARAMETERS", text)


class ObjectTypesTest(VerifyCase):
    """validate_model reads every object type: sketches (DoF, conflicts), PartDesign, meshes, expressions, spreadsheets."""

    def sketch(self, doc, name="Sketch"):
        import Sketcher

        sk = doc.addObject("Sketcher::SketchObject", name)
        sk.addGeometry(Part.LineSegment(App.Vector(0, 0, 0), App.Vector(10, 0, 0)))
        sk.addGeometry(Part.LineSegment(App.Vector(10, 0, 0), App.Vector(10, 5, 0)))
        doc.recompute()
        return sk, Sketcher

    def test_an_under_constrained_sketch_shows_its_dof_as_info_and_is_still_valid(self):
        doc = self.new_doc()
        self.sketch(doc)
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name)
        self.assertFalse(is_error, text)
        self.assertRegex(text, r"INFO \[SKETCH_UNDERCONSTRAINED\] Sketch: .*DoF")
        self.assertTrue(text.endswith("RESULT: VALID"))

    def test_a_fully_constrained_sketch_has_no_dof_line(self):
        doc = self.new_doc()
        sk, Sketcher = self.sketch(doc)
        for c in (Sketcher.Constraint("Coincident", 0, 1, 1, 1), Sketcher.Constraint("Coincident", 0, 1, -1, 1),
                  Sketcher.Constraint("Horizontal", 0), Sketcher.Constraint("Vertical", 1),
                  Sketcher.Constraint("DistanceX", 0, 1, 0, 2, 10.0), Sketcher.Constraint("DistanceY", 1, 1, 1, 2, 5.0)):
            sk.addConstraint(c)
        doc.recompute()
        self.assertEqual(sk.solve(), 0)
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name)
        self.assertFalse(is_error, text)
        self.assertNotIn("SKETCH_UNDERCONSTRAINED", text)

    def test_a_conflicting_sketch_is_an_error(self):
        doc = self.new_doc()
        sk, Sketcher = self.sketch(doc)
        sk.addConstraint(Sketcher.Constraint("Horizontal", 0))
        sk.addConstraint(Sketcher.Constraint("Vertical", 0))
        doc.recompute()
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name)
        self.assertTrue(is_error)
        self.assertIn("ERROR [SKETCH_CONFLICT] Sketch", text)

    def test_a_partdesign_body_and_a_mesh_are_rows_not_noise(self):
        import Mesh

        doc = self.new_doc()
        body = doc.addObject("PartDesign::Body", "Body")
        body.newObject("PartDesign::AdditiveBox", "AddBox")
        mesh = doc.addObject("Mesh::Feature", "Imported")
        mesh.Mesh = Mesh.createBox(4, 4, 4)
        doc.recompute()
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name)
        self.assertFalse(is_error, text)
        self.assertRegex(text, r"Body\s+PartDesign::Body")
        self.assertRegex(text, r"Imported\s+Mesh::Feature")

    def test_an_expression_naming_a_missing_parameter_is_an_error(self):
        doc = self.new_doc()
        S.params(doc, [("wall_t", "=2 mm", "mm", "assumption")])
        box = doc.addObject("Part::Box", "Cube")
        box.setExpression("Length", "Params.wall_t")
        doc.recompute()
        box.setExpression("Width", "Params.no_such_row")
        doc.recompute()
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name)
        self.assertTrue(is_error)
        self.assertIn("ERROR [EXPRESSION_ERROR] Cube", text)
        self.assertRegex(text, r"PARAMETERS \(sheet 'Params', 1 rows, 1 bound to objects\)")

    def test_a_sheet_formula_error_that_leaves_the_dependents_valid_is_still_caught(self):
        """``=1 mm/0`` leaves the sheet and every object bound to it 'Valid' (verified): only the value shows it."""
        doc = self.new_doc()
        sheet = S.params(doc, [("ratio", "=1 mm/0", "mm", "assumption")])
        box = doc.addObject("Part::Box", "Cube")
        box.setExpression("Length", "Params.ratio")
        doc.recompute()
        self.assertNotIn("Invalid", box.State)
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name)
        self.assertTrue(is_error)
        self.assertIn("EXPRESSION_ERROR", text)
        self.assertIsNotNone(sheet)

    def test_a_modified_unsaved_document_with_a_touched_object_is_recomputed_once_and_judged(self):
        doc = self.new_doc()
        box = doc.addObject("Part::Box", "Cube")
        doc.recompute()
        box.Length = 25                                   # touched, not recomputed
        self.assertIn("Touched", box.State)
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name)
        self.assertFalse(is_error, text)
        self.assertNotIn("Touched", box.State)
        self.assertEqual(box.Shape.BoundBox.XLength, 25.0)


class GateRulesTest(VerifyCase):
    """Pre-existing problems of the user never fail a validation of THIS task's work."""

    def setUp(self):
        super().setUp()
        self.doc = self.enclosure()
        self.broken_cut(self.doc)
        self.baseline = {self.doc.Name: [list(k) for k in health.snapshot_via_dispatch(self.doc.Name).finding_keys]}

    def ctx(self, **state):
        return ToolContext(state={"baseline": self.baseline, **state})

    def test_without_a_baseline_every_error_counts(self):
        text, _, is_error = self.run_tool("validate_model", doc_name=self.doc.Name)
        self.assertTrue(is_error)
        self.assertIn("RESULT: INVALID", text)

    def test_a_pre_existing_error_on_an_untouched_object_is_shown_but_not_counted(self):
        text, _, is_error = self.run_tool("validate_model", self.ctx(), doc_name=self.doc.Name)
        self.assertFalse(is_error, text)
        self.assertRegex(text, r"RESULT: VALID \(\d+ pre-existing errors? not counted\)")
        self.assertIn("pre-existing (not caused by this task, not counted)", text)
        self.assertNotIn("ERROR [", text)           # shown as warnings: nothing invites the model to fix the user's object

    def test_the_same_error_counts_once_the_turn_touched_the_object(self):
        ctx = self.ctx(touched={self.doc.Name: {"UserCut"}})
        text, _, is_error = self.run_tool("validate_model", ctx, doc_name=self.doc.Name)
        self.assertTrue(is_error)
        self.assertIn("RESULT: INVALID", text)

    def test_a_new_error_counts_even_with_a_baseline(self):
        S.feature(self.doc, "Shell2", Part.makeCompound([S.box(0, 0, 0, 5, 5, 5), S.box(20, 0, 0, 5, 5, 5)]), role="shell")
        self.doc.recompute()
        text, _, is_error = self.run_tool("validate_model", self.ctx(), doc_name=self.doc.Name)
        self.assertTrue(is_error)
        self.assertIn("ERROR [FRAGMENTS] Shell2", text)
        self.assertIn("pre-existing", text)          # the user's cut is still listed, as a warning

    def test_a_document_missing_from_the_baseline_was_created_this_turn_so_everything_counts(self):
        ctx = ToolContext(state={"baseline": {"SomeOtherDoc": []}})
        _, _, is_error = self.run_tool("validate_model", ctx, doc_name=self.doc.Name)
        self.assertTrue(is_error)

    def test_split_findings_is_pure_and_keys_survive_a_json_round_trip(self):
        err = lambda code, obj: Finding("error", code, obj, "m")           # noqa: E731
        report = HealthReport(doc="D", ts=0.0, findings=[err("A", "x"), err("B", "y"), err("C", ""),
                                                          Finding("warning", "W", "x", "m")])
        baseline = {("A", "x"), ("C", "")}
        counted, pre = TV.split_findings(report, baseline, set())
        self.assertEqual([f.code for f in counted], ["B"])
        self.assertEqual([f.code for f in pre], ["A", "C"])
        counted, _ = TV.split_findings(report, baseline, {"x"})
        self.assertEqual([f.code for f in counted], ["A", "B"])
        counted, pre = TV.split_findings(report, None, set())
        self.assertEqual(([f.code for f in counted], pre), (["A", "B", "C"], []))
        stored = json.loads(json.dumps({"D": [list(k) for k in baseline]}))
        self.assertEqual(TV._baseline_keys(ToolContext(state={"baseline": stored}), "D"), baseline)
        self.assertIsNone(TV._baseline_keys(ToolContext(state={"baseline": stored}), "Other"))
        self.assertIsNone(TV._baseline_keys(ToolContext(), "D"))


class DeepEditabilityTest(VerifyCase):
    def test_deep_rebuilds_with_each_driver_parameter_and_restores_the_sheet(self):
        doc = self.new_doc("Deep")
        objs = S.build_parametric(doc, slot=8.0)
        before = {c: objs["sheet"].getContents(c) for c in ("B2", "B3", "B4", "B5")}
        volume = objs["body"].Shape.Volume
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name, deep=True)
        self.assertFalse(is_error, text)
        self.assertIn("(deep=True)", text)
        self.assertRegex(text, r"EDITABILITY \(4 parameters, \+-10%, [\d.]+ s\)")
        self.assertIn("slot +10% ok, -10% ok", text)
        self.assertIn("len +10% ok, -10% ok", text)
        self.assertTrue(text.endswith("RESULT: VALID"))
        self.assertEqual({c: objs["sheet"].getContents(c) for c in before}, before)
        self.assertAlmostEqual(objs["body"].Shape.Volume, volume, places=6)
        self.assertNotIn("Touched", objs["body"].State)

    def test_a_parameter_that_breaks_the_model_makes_it_invalid(self):
        doc = self.new_doc("Fragile")
        objs = S.build_parametric(doc, slot=9.5)
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name, deep=True)
        self.assertTrue(is_error)
        self.assertRegex(text, r"slot \+10% FAIL\(FRAGMENTS Body\)")
        self.assertIn("RESULT: INVALID", text)
        self.assertEqual(objs["sheet"].getContents("B5"), "=9.5 mm")       # restored even after the break
        self.assertEqual(len(objs["body"].Shape.Solids), 1)

    def test_run_validation_turns_an_editability_break_into_a_gate_error(self):
        doc = self.new_doc("Fragile2")
        S.build_parametric(doc, slot=9.5)
        result = TV.run_validation(doc.Name, deep=True)
        self.assertFalse(result.ok)
        codes = [e["code"] for e in result.gate_errors]
        self.assertIn("EDITABILITY", codes)
        edit = next(e for e in result.gate_errors if e["code"] == "EDITABILITY")
        self.assertEqual(edit["object"], "slot")
        self.assertIn("by +10% breaks the model", edit["message"])
        self.assertIn("editability", result.ran)
        self.assertTrue(result.editability["broken"])

    def test_deep_without_driver_parameters_says_why_and_stays_valid(self):
        doc = self.enclosure()           # the fixture's Params rows are bound to nothing
        text, _, is_error = self.run_tool("validate_model", doc_name=doc.Name, deep=True)
        self.assertFalse(is_error, text)
        self.assertIn("EDITABILITY not run: no bound numeric parameter to edit", text)


# ---------------------------------------------------------------------------------------------- check_fit


class CheckFitTest(VerifyCase):
    def test_a_correct_enclosure_prints_one_line_per_row_and_the_count_line(self):
        doc = self.enclosure()
        text, image, is_error = self.run_tool("check_fit", doc_name=doc.Name)
        self.assertFalse(is_error, text)
        self.assertIsNone(image)
        lines = text.split("\n")
        self.assertEqual(lines[0], f"CHECK_FIT doc '{doc.Name}' (lift_mm=0.5)")
        self.assertEqual(lines[1], "PASS interference.shell_lid  Shell vs Lid: common_volume_mm3 = 0.0 (<= 0.05)")
        self.assertEqual(lines[4], "PASS clearance.pcb_to_wall_min  Shell vs PCB: min_wall_distance_mm = 2.0 (>= 1.95)")
        self.assertTrue(lines[-2].startswith("WARN reference.status  reference is nominal board data"))
        self.assertEqual(lines[-1], "FIT: 6 PASS, 1 WARN, 0 FAIL, 0 SKIP")

    def test_a_fail_row_makes_the_result_an_error_with_the_fix_it_hint(self):
        doc = self.enclosure(overlap=True)
        text, _, is_error = self.run_tool("check_fit", doc_name=doc.Name)
        self.assertTrue(is_error)
        self.assertIn("FAIL interference.shell_lid  Shell vs Lid: common_volume_mm3 = 40.0 (<= 0.05) - ", text)
        self.assertIn("FIT: 5 PASS, 1 WARN, 1 FAIL, 0 SKIP", text)
        self.assertIn("FAIL rows (interference.shell_lid) are defects", text)

    def test_every_documented_failure_mode_is_reported_as_its_own_row(self):
        for kw, rid in ((dict(overlap=True), "interference.shell_lid"), (dict(wide_pcb=True), "clearance.pcb_to_wall_min"),
                        (dict(ref_poke=True), "interference.reference_shell")):
            with self.subTest(rid=rid):
                doc = self.enclosure(**kw)
                text, _, is_error = self.run_tool("check_fit", doc_name=doc.Name)
                self.assertTrue(is_error)
                self.assertRegex(text, rf"FAIL {rid.replace('.', r'[.]')}\b")
                App.closeDocument(doc.Name)

    def test_checks_selects_rows_by_id_or_family(self):
        doc = self.enclosure()
        text, _, _ = self.run_tool("check_fit", doc_name=doc.Name, checks=["interference.shell_lid"])
        self.assertEqual(text.split("\n")[1:], ["PASS interference.shell_lid  Shell vs Lid: common_volume_mm3 = 0.0 "
                                                 "(<= 0.05)", "FIT: 1 PASS, 0 WARN, 0 FAIL, 0 SKIP"])
        text, _, _ = self.run_tool("check_fit", doc_name=doc.Name, checks=["clearance"])
        self.assertIn("clearance.pcb_to_wall_min", text)
        self.assertIn("clearance.lid_to_tallest", text)
        self.assertNotIn("interference", text)

    def test_unknown_check_ids_are_all_listed_with_the_valid_ones(self):
        doc = self.enclosure()
        text, _, is_error = self.run_tool("check_fit", doc_name=doc.Name, checks=["nope", "interference.shell_lid", "x"])
        self.assertTrue(is_error)
        self.assertIn("'nope' is not a check id", text)
        self.assertIn("'x' is not a check id", text)
        self.assertIn("use one of: interference.shell_lid", text)

    def test_argument_problems_are_reported_together_before_any_document_work(self):
        text, _, is_error = self.run_tool("check_fit", doc_name="NoSuchDocument", parts=["only_one"],
                                          checks=["nope"])
        self.assertTrue(is_error)
        self.assertNotIn("is not open", text)         # the arguments are judged first
        self.assertIn("needs at least two object names", text)
        self.assertIn("'nope' is not a check id", text)
        text, _, is_error = self.run_tool("check_fit", doc_name="NoSuchDocument", lift_mm=9, spec=[1])
        self.assertTrue(is_error)                     # the registry already judges bounds and shapes
        self.assertIn("lift_mm", text)
        self.assertIn("spec", text)
        self.assertNotIn("is not open", text)

    def test_too_many_parts_are_refused(self):
        doc = self.enclosure()
        text, _, is_error = self.run_tool("check_fit", doc_name=doc.Name, parts=[f"P{i}" for i in range(11)])
        self.assertTrue(is_error)
        self.assertIn("at most 10", text)

    def test_unresolvable_parts_are_all_named(self):
        doc = self.enclosure()
        text, _, is_error = self.run_tool("check_fit", doc_name=doc.Name, parts=["Ghost", "Params", "Shell"])
        self.assertTrue(is_error)
        self.assertIn("'Ghost' is not an object of document", text)
        self.assertIn("'Params' is not solid geometry", text)

    def test_explicit_parts_and_reference_and_lift(self):
        doc = self.enclosure()
        text, _, is_error = self.run_tool("check_fit", doc_name=doc.Name, parts=["Shell", "Lid"], reference="Ref_TEST",
                                          lift_mm=1.0)
        self.assertFalse(is_error, text)
        self.assertIn("(lift_mm=1)", text)
        self.assertRegex(text, r"PASS retention.lid_lift  Shell vs Lid: lifted_common_volume_mm3 = [\d.]+ \(> 0\)")

    def test_an_unknown_document_lists_the_open_ones(self):
        self.new_doc("Open1")
        text, _, is_error = self.run_tool("check_fit", doc_name="Gone")
        self.assertTrue(is_error)
        self.assertIn("Document 'Gone' is not open", text)

    def test_the_stop_button_abandons_the_checks(self):
        doc = self.enclosure()
        text, _, is_error = self.run_tool("check_fit", ToolContext(should_stop=lambda: True), doc_name=doc.Name)
        self.assertTrue(is_error)
        self.assertIn("did not run", text)

    def test_the_assembled_pose_trap_with_a_lifted_lid(self):
        doc = self.enclosure(lid_lift=15.0, overlap=True)
        text, _, is_error = self.run_tool("check_fit", doc_name=doc.Name)
        self.assertTrue(is_error)
        self.assertIn("FAIL interference.shell_lid", text)
        doc2 = self.enclosure(lid_lift=15.0)
        text, _, is_error = self.run_tool("check_fit", doc_name=doc2.Name)
        self.assertFalse(is_error, text)

    def test_the_reference_status_row_reads_the_ledger_then_the_project(self):
        doc = self.enclosure()
        ledger = S.FakeLedger(reference_status="step-imported")
        text, _, _ = self.run_tool("check_fit", ToolContext(ledger=ledger), doc_name=doc.Name)
        self.assertIn("PASS reference.status  reference_status=step-imported", text)
        project = S.FakeProject(reference_status="catalog-nominal")
        text, _, _ = self.run_tool("check_fit", ToolContext(project=project), doc_name=doc.Name)
        self.assertIn("WARN reference.status", text)

    def test_the_project_spec_adds_the_kit_rows_and_decides_retention(self):
        doc = self.enclosure(hooks=False)
        project = S.FakeProject(spec={"board": "rpi4b", "lid": {"style": "snap_fit"}})
        fake = [{"id": "containment.ports", "kind": "containment", "a": "Shell", "b": "usb_c", "metric": "margin_mm",
                 "value": 0.4, "limit": ">= 0", "status": "PASS", "detail": ""}]
        from oscar.kit import checks as kit_checks

        with mock.patch.object(kit_checks, "fit_rows", return_value=fake):
            text, _, is_error = self.run_tool("check_fit", ToolContext(project=project), doc_name=doc.Name)
        self.assertTrue(is_error)                                         # the barb-less lid does not retain
        self.assertIn("PASS containment.ports  Shell vs usb_c: margin_mm = 0.4 (>= 0)", text)
        self.assertIn("FAIL retention.lid_lift", text)

    def test_a_spec_argument_overrides_the_project_spec(self):
        doc = self.enclosure(hooks=False)
        project = S.FakeProject(spec={"board": "rpi4b", "lid": {"style": "snap_fit"}})
        text, _, is_error = self.run_tool("check_fit", ToolContext(project=project), doc_name=doc.Name,
                                          spec={"board": "rpi4b", "lid": {"style": "screw"}})
        self.assertFalse(is_error, text)
        self.assertIn("SKIP retention.lid_lift  lid style 'screw' has no snap hooks", text)

    def test_origin_features_and_part_containers_do_not_confuse_the_checks(self):
        doc = self.enclosure()
        doc.addObject("PartDesign::Body", "Body")
        doc.recompute()
        text, _, is_error = self.run_tool("check_fit", doc_name=doc.Name)
        self.assertFalse(is_error, text)
        self.assertNotIn("e+", text)
        text, _, is_error = self.run_tool("check_fit", doc_name=doc.Name, parts=["Origin", "Shell"])
        self.assertTrue(is_error)
        self.assertIn("is not solid geometry", text)


# ---------------------------------------------------------------------------------------------- run_validation


class RunValidationTest(VerifyCase):
    def test_the_result_has_every_documented_field(self):
        from support.fakes import FakeValidation
        import dataclasses

        fake_fields = {f.name for f in dataclasses.fields(FakeValidation)}
        mine = {f.name for f in dataclasses.fields(TV.ValidationResult)}
        self.assertLessEqual(fake_fields, mine)
        self.assertEqual(mine - fake_fields, {"editability"})

    def test_a_sound_document_validates(self):
        doc = self.new_doc()
        S.build_parametric(doc)
        result = TV.run_validation(doc.Name)
        self.assertIsInstance(result, TV.ValidationResult)
        self.assertTrue(result.ok)
        self.assertEqual(result.doc, doc.Name)
        self.assertEqual((result.gate_errors, result.fail_rows, result.fit_rows, result.fit_ran), ([], [], [], False))
        self.assertEqual(result.ran, ["health"])
        self.assertEqual(result.not_validated, ["fit.interference", "fit.clearance", "fit.ports", "fit.retention",
                                                "dfm", "editability", "rebuild", "step_collision", "physical"])
        self.assertTrue(result.health["valid"])
        self.assertEqual(result.health["doc"], doc.Name)
        self.assertTrue(result.text.startswith("[health] doc "))
        json.dumps(result.to_dict(), default=str)

    def test_none_resolves_to_the_document_of_the_last_geometry_step_not_the_active_one(self):
        good = self.new_doc("Good")
        S.build_parametric(good)
        bad = self.new_doc("Bad")                       # the newest document is the ACTIVE one
        self.broken_cut(bad)
        self.assertEqual(App.ActiveDocument.Name, bad.Name)
        result = TV.run_validation(None, ctx=ToolContext(state={"last_doc": good.Name}))
        self.assertEqual(result.doc, good.Name)
        self.assertTrue(result.ok)
        by_default = TV.run_validation(None)
        self.assertEqual(by_default.doc, bad.Name)
        self.assertFalse(by_default.ok)

    def test_a_last_doc_that_was_closed_falls_back_to_the_active_document(self):
        doc = self.new_doc("Active")
        S.build_parametric(doc)
        result = TV.run_validation(None, ctx=ToolContext(state={"last_doc": "ClosedLongAgo"}))
        self.assertEqual(result.doc, doc.Name)

    def test_nothing_to_judge_is_none_never_a_clean_result(self):
        self.assertIsNone(TV.run_validation("NoSuchDocument"))
        doc = self.new_doc()
        S.build_parametric(doc)
        self.assertIsNone(TV.run_validation(doc.Name, ctx=ToolContext(should_stop=lambda: True)))

    def test_new_and_touched_errors_fail_the_gate_pre_existing_ones_do_not(self):
        doc = self.enclosure()
        self.broken_cut(doc)
        baseline = {doc.Name: [list(k) for k in health.snapshot_via_dispatch(doc.Name).finding_keys]}
        ok = TV.run_validation(doc.Name, ctx=ToolContext(state={"baseline": baseline}), fit=False)
        self.assertTrue(ok.ok)
        self.assertEqual(ok.gate_errors, [])
        self.assertTrue(any(f["object"] == "UserCut" and f["severity"] == "error" for f in ok.health["findings"]))
        touched = TV.run_validation(doc.Name, ctx=ToolContext(state={"baseline": baseline,
                                                                    "touched": {doc.Name: {"UserCut"}}}), fit=False)
        self.assertFalse(touched.ok)
        self.assertTrue(all(e["object"] == "UserCut" for e in touched.gate_errors))
        self.assertIn("not counted", ok.text)

    def test_an_invalid_vendor_solid_in_the_reference_part_never_fails_the_gate(self):
        doc = self.enclosure()
        bowtie = Part.Face(Part.makePolygon([App.Vector(0, 0, 0), App.Vector(4, 4, 0), App.Vector(4, 0, 0),
                                              App.Vector(0, 4, 0), App.Vector(0, 0, 0)]))
        bad = S.feature(doc, "VendorBad", bowtie.extrude(App.Vector(0, 0, 3)))
        self.objs["ref"].addObject(bad)
        doc.recompute()
        self.assertFalse(bad.Shape.isValid())
        result = TV.run_validation(doc.Name, fit=False)
        self.assertTrue(result.ok, result.gate_errors)
        warned = [f for f in result.health["findings"] if f["object"] == "VendorBad"]
        self.assertTrue(warned and all(f["severity"] == "warning" for f in warned))

    def test_fit_defaults_to_the_intent_or_an_existing_spec(self):
        doc = self.enclosure()
        none = TV.run_validation(doc.Name, ctx=ToolContext(intent="generic_part"))
        self.assertFalse(none.fit_ran)
        self.assertEqual(none.fit_rows, [])
        enc = TV.run_validation(doc.Name, ctx=ToolContext(intent="enclosure"))
        self.assertTrue(enc.fit_ran)
        self.assertEqual(len(enc.fit_rows), 7)
        self.assertEqual(enc.fail_rows, [])
        self.assertTrue(enc.ok)
        self.assertEqual(enc.ran, ["health", "fit.interference", "fit.clearance", "fit.retention"])
        self.assertIn("FIT: ", enc.text)
        spec = TV.run_validation(doc.Name, ctx=ToolContext(project=S.FakeProject(spec={"board": "rpi4b"})))
        self.assertTrue(spec.fit_ran)
        forced_off = TV.run_validation(doc.Name, ctx=ToolContext(intent="enclosure"), fit=False)
        self.assertFalse(forced_off.fit_ran)
        forced_on = TV.run_validation(doc.Name, fit=True)
        self.assertTrue(forced_on.fit_ran)

    def test_a_failing_fit_row_fails_the_result(self):
        doc = self.enclosure(overlap=True)
        result = TV.run_validation(doc.Name, ctx=ToolContext(intent="enclosure"))
        self.assertFalse(result.ok)
        self.assertEqual(result.gate_errors, [])
        self.assertEqual([r["id"] for r in result.fail_rows], ["interference.shell_lid"])
        self.assertIn("FAIL interference.shell_lid", result.text)

    def test_the_spec_adds_dfm_rows_and_the_kit_seam_rows(self):
        from oscar.kit import spec as kit_spec

        doc = self.enclosure()
        project = S.FakeProject(spec=kit_spec.default_spec().to_dict())
        result = TV.run_validation(doc.Name, ctx=ToolContext(project=project, intent="enclosure"))
        self.assertTrue(result.dfm)
        self.assertEqual(result.dfm[0]["id"], "dfm.min_wall")
        self.assertIn("dfm", result.ran)
        self.assertIn("DFM: ", result.text)
        self.assertNotIn("dfm", result.not_validated)

    def test_the_result_is_persisted_through_the_project(self):
        doc = self.enclosure()
        project = S.FakeProject(last_geometry_step=7)
        result = TV.run_validation(doc.Name, ctx=ToolContext(project=project, intent="enclosure"))
        record = project.recorded[-1]
        self.assertEqual((record["doc"], record["after_step"]), (doc.Name, 7))
        self.assertEqual(result.after_step, 7)
        self.assertEqual(record["ran"], result.ran)
        self.assertEqual(len(record["fit_rows"]), 7)
        self.assertNotIn("text", record)

    def test_a_project_that_raises_never_breaks_the_validation(self):
        doc = self.enclosure()
        project = S.FakeProject()
        project.record_validation = mock.Mock(side_effect=OSError("disk full"))
        result = TV.run_validation(doc.Name, ctx=ToolContext(project=project))
        self.assertTrue(result.ok)

    def test_it_runs_from_any_thread(self):
        doc = self.enclosure()
        out = {}

        def work():
            out["result"] = TV.run_validation(doc.Name, ctx=ToolContext(intent="enclosure"))

        # a QApplication left behind by another test would queue the call for a GUI thread that never drains: the
        # console behaviour (inline dispatch, no GUI thread) is what this test is about
        with mock.patch.object(dispatch, "_run_inline", return_value=True):
            t = threading.Thread(target=work)
            t.start()
            t.join(60)
        self.assertIn("result", out, "run_validation did not finish on the helper thread")
        self.assertTrue(out["result"].ok)

    def test_the_bound_context_is_restored_after_the_call(self):
        doc = self.enclosure()
        outer = ToolContext(turn_id="outer")
        with use_context(outer):
            TV.run_validation(doc.Name, ctx=ToolContext(turn_id="inner"))
            self.assertIs(tools.current_context(), outer)


# ---------------------------------------------------------------------------------------------- persistence and ledger


class PartialRecordTest(VerifyCase):
    """``_persist`` hands the Project only what ran: the Project merges partial records (``project_validation``)."""

    def test_the_project_receives_a_partial_record_without_the_redundant_fields(self):
        doc = self.enclosure()
        project = S.FakeProject(last_geometry_step=5)
        self.run_tool("validate_model", ToolContext(project=project), doc_name=doc.Name)
        record = project.recorded[-1]
        self.assertEqual(sorted(record), ["after_step", "doc", "editability", "gate_errors", "health", "ran"])
        self.assertEqual((record["doc"], record["after_step"], record["ran"]), (doc.Name, 5, ["health"]))
        self.run_tool("check_fit", ToolContext(project=project), doc_name=doc.Name)
        fit = project.recorded[-1]
        self.assertEqual(sorted(fit), ["after_step", "doc", "fit_rows", "ran"])
        self.assertEqual(fit["ran"], ["fit.interference", "fit.clearance", "fit.retention"])

    def test_a_project_without_the_method_or_that_raises_is_tolerated(self):
        doc = self.enclosure()
        for project in (object(), mock.Mock(record_validation=mock.Mock(side_effect=ValueError("x")))):
            text, _, is_error = self.run_tool("validate_model", ToolContext(project=project), doc_name=doc.Name)
            self.assertFalse(is_error, text)

    def test_fit_entry_follows_the_build_summary_split(self):
        cases = {("interference", "interference.shell_lid"): "fit.interference",
                 ("clearance", "clearance.lid_to_tallest"): "fit.clearance",
                 ("retention", "retention.lid_lift"): "fit.retention",
                 ("alignment", "hooks.engagement"): "fit.retention",
                 ("alignment", "alignment.standoffs"): "fit.ports",
                 ("containment", "containment.ports"): "fit.ports",
                 ("coverage", "reference.status"): ""}
        from oscar import project_validation

        for (kind, rid), entry in cases.items():
            row = {"kind": kind, "id": rid}
            self.assertEqual(TV.fit_entry(row), entry, rid)
            self.assertEqual(project_validation.group_of(row) or "", entry, rid)

    def test_the_catalog_is_the_projects(self):
        from oscar import project_validation

        self.assertEqual(TV._catalog(), project_validation.CATALOG)


class ProjectAndLedgerTest(VerifyCase):
    def test_validate_model_then_check_fit_leave_one_merged_validation_record(self):
        from support.fakes import temp_project

        doc = self.enclosure()
        project = temp_project()
        ctx = ToolContext(project=project)
        self.run_tool("validate_model", ctx, doc_name=doc.Name)
        record = project.read_validation()
        self.assertEqual((record["ran"], record["ok"], record["doc"]), (["health"], True, doc.Name))
        self.assertEqual(record["fit_rows"], [])
        self.run_tool("check_fit", ctx, doc_name=doc.Name)
        record = project.read_validation()
        self.assertEqual(record["ran"], ["health", "fit.interference", "fit.clearance", "fit.retention"])
        self.assertEqual(len(record["fit_rows"]), 7)
        self.assertTrue(record["ok"])
        self.assertTrue(record["health"]["valid"])
        self.assertEqual({r["status"] for r in record["fit_rows"]}, {"PASS", "WARN"})

    def test_a_later_failing_check_fit_flips_the_merged_record(self):
        from support.fakes import temp_project

        doc = self.enclosure(overlap=True)
        project = temp_project()
        ctx = ToolContext(project=project)
        self.run_tool("validate_model", ctx, doc_name=doc.Name)
        self.run_tool("check_fit", ctx, doc_name=doc.Name)
        record = project.read_validation()
        self.assertFalse(record["ok"])
        self.assertEqual([r["id"] for r in record["fit_rows"] if r["status"] == "FAIL"], ["interference.shell_lid"])

    def test_a_failing_validate_model_reopens_the_ledger_with_a_fix_step(self):
        doc = self.enclosure()
        self.broken_cut(doc)
        ledger = Ledger.from_request("Design a snap-fit enclosure for a Raspberry Pi 4", intent="enclosure")
        before = [s.id for s in ledger.steps]
        text, _, is_error = self.run_tool("validate_model", ToolContext(ledger=ledger, intent="enclosure"),
                                          doc_name=doc.Name)
        self.assertTrue(is_error)
        self.assertFalse(ledger.health_ok)
        self.assertTrue(ledger.open_problems)
        self.assertEqual([s.id for s in ledger.steps if s.id.startswith("F")], ["F1"])
        self.assertEqual(len(ledger.steps), len(before) + 1)

    def test_a_passing_validate_model_does_not_stand_in_for_the_fit_checks(self):
        doc = self.enclosure()
        fake = S.FakeLedger()
        self.run_tool("validate_model", ToolContext(ledger=fake, intent="enclosure"), doc_name=doc.Name)
        self.assertEqual(fake.applied, [])                  # the intent owes fit checks: the finish validation decides
        self.run_tool("validate_model", ToolContext(ledger=fake, intent="generic_part"), doc_name=doc.Name)
        self.assertEqual(len(fake.applied), 1)              # a COMPLETE validation for this turn: folded in
        self.assertTrue(fake.applied[0].ok)
        self.assertEqual(fake.applied[0].after_step, 3)

    def test_a_project_spec_also_means_fit_checks_are_owed(self):
        doc = self.enclosure()
        fake = S.FakeLedger()
        ctx = ToolContext(ledger=fake, project=S.FakeProject(spec={"board": "rpi4b"}), intent="generic_part")
        self.run_tool("validate_model", ctx, doc_name=doc.Name)
        self.assertEqual(fake.applied, [])

    def test_a_ledger_that_raises_never_breaks_the_tool(self):
        doc = self.enclosure()
        fake = S.FakeLedger()
        fake.apply_validation = mock.Mock(side_effect=RuntimeError("boom"))
        text, _, is_error = self.run_tool("validate_model", ToolContext(ledger=fake, intent="generic_part"),
                                          doc_name=doc.Name)
        self.assertFalse(is_error, text)

    def test_the_ledger_after_a_failure_then_a_fix_closes_the_fix_step(self):
        doc = self.enclosure()
        cut = self.broken_cut(doc)
        ledger = Ledger.from_request("Design a snap-fit enclosure for a Raspberry Pi 4", intent="enclosure")
        ctx = ToolContext(ledger=ledger, intent="generic_part")
        self.run_tool("validate_model", ctx, doc_name=doc.Name)
        self.assertTrue(any(s.id == "F1" and s.status != "done" for s in ledger.steps))
        doc.removeObject(cut.Name)
        doc.removeObject("UserCutBase")
        doc.recompute()
        _, _, is_error = self.run_tool("validate_model", ctx, doc_name=doc.Name)
        self.assertFalse(is_error)
        self.assertTrue(ledger.health_ok)
        self.assertTrue(any(s.id == "F1" and s.status == "done" for s in ledger.steps))


# ---------------------------------------------------------------------------------------------- delegation and registry


class FeatureTreeStatusTest(VerifyCase):
    def test_it_delegates_to_validate_model(self):
        doc = self.enclosure()
        text, _, is_error = self.run_tool("feature_tree_status", doc_name=doc.Name)
        self.assertFalse(is_error, text)
        self.assertTrue(text.startswith(f"VALIDATE doc '{doc.Name}' (deep=False)"))
        self.assertIn("PARAMETERS (sheet 'Params'", text)
        self.assertTrue(text.endswith("RESULT: VALID"))

    def test_it_reports_a_broken_feature_tree_as_an_error(self):
        doc = self.new_doc()
        self.broken_cut(doc)
        text, _, is_error = self.run_tool("feature_tree_status", doc_name=doc.Name)
        self.assertTrue(is_error)
        self.assertIn("RESULT: INVALID", text)

    def test_the_tool_name_and_schema_are_unchanged(self):
        entry = tools.tool_meta("feature_tree_status")
        self.assertEqual(entry.parameters["required"], ["doc_name"])
        self.assertEqual(list(entry.parameters["properties"]), ["doc_name"])
        self.assertEqual((entry.category, entry.mutating, entry.geometry), ("verify", False, False))
        self.assertIn("validate_model", entry.description)


class RegistryFactsTest(unittest.TestCase):
    def test_validate_model_follows_the_contract_schema(self):
        entry = tools.tool_meta("validate_model")
        self.assertEqual((entry.category, entry.mutating, entry.geometry), ("verify", False, False))
        self.assertEqual(entry.phases, ("BUILD", "VERIFY", "DELIVER"))
        props = entry.parameters["properties"]
        self.assertEqual(list(props), ["doc_name", "deep", "include_params"])
        self.assertEqual(entry.parameters["required"], ["doc_name"])
        self.assertIs(props["deep"]["default"], False)
        self.assertIs(props["include_params"]["default"], True)
        self.assertIn("VALID or INVALID", entry.description)
        self.assertIn("+-10%", entry.description)

    def test_check_fit_follows_the_contract_schema(self):
        entry = tools.tool_meta("check_fit")
        self.assertEqual((entry.category, entry.mutating, entry.geometry), ("verify", False, False))
        self.assertEqual(list(entry.parameters["properties"]),
                         ["doc_name", "parts", "reference", "checks", "spec", "lift_mm"])
        self.assertEqual(entry.parameters["required"], ["doc_name"])
        lift = entry.parameters["properties"]["lift_mm"]
        self.assertEqual((lift["default"], lift["minimum"], lift["maximum"]), (0.5, 0.05, 5))
        self.assertEqual(entry.parameters["properties"]["parts"]["items"], {"type": "string"})
        self.assertIn("A FAIL is a real defect to fix", entry.description)

    def test_the_schemas_use_no_oneof_anyof_or_ref(self):
        for name in ("validate_model", "check_fit"):
            text = json.dumps(tools.tool_meta(name).parameters)
            for banned in ("oneOf", "anyOf", "$ref", "allOf"):
                self.assertNotIn(banned, text, name)

    def test_the_titles_come_from_the_arguments(self):
        self.assertEqual(tools.derive_title("validate_model", {"deep": True}), "Validate model (deep)")
        self.assertEqual(tools.derive_title("validate_model", {}), "Validate model")
        self.assertEqual(tools.derive_title("check_fit", {}), "Check fit")

    def test_both_tools_are_offered_in_the_verify_phase_and_to_lite_models(self):
        verify = {s.name for s in tools.specs(phases=["VERIFY"])}
        self.assertLessEqual({"validate_model", "check_fit", "feature_tree_status"}, verify)
        self.assertNotIn("validate_model", {s.name for s in tools.specs(phases=["PLAN"])})
        self.assertLessEqual({"validate_model", "check_fit"}, {s.name for s in tools.specs(lite=True)})

    def test_a_non_mutating_verification_writes_no_step_record(self):
        project = S.FakeProject()
        project.on_tool_result = mock.Mock()
        doc = App.newDocument("NoStep")
        try:
            S.build_enclosure(doc)
            with use_context(ToolContext(project=project)):
                tools.run_tool("validate_model", {"doc_name": doc.Name})
        finally:
            App.closeDocument(doc.Name)
        project.on_tool_result.assert_not_called()


if __name__ == "__main__":
    unittest.main()
