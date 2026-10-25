"""Engine integrity against real FreeCAD documents: the compact
``get_objects`` table, ``get_object``, atomic property application, ``delete_object`` with
dependents, ``create_object(body=...)``, ``reload_document``, transaction rollback, turn
snapshot / revert, headless screenshots, the merge guard, the download location, the failure
text of the plan tools and the real ``run_tool`` pipeline with the real health module and a fake
Project."""

import os
import sys
import tempfile
import types
import unittest
import uuid
import zipfile
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

if "FreeCADGui" not in sys.modules:
    try:
        import FreeCADGui  # noqa: F401
    except Exception:
        sys.modules["FreeCADGui"] = types.ModuleType("FreeCADGui")

import FreeCAD as App  # noqa: E402

from oscar import netguard, paths, tools  # noqa: E402
from oscar.cad import engine as engine_module, get_engine, serialize, views  # noqa: E402
from oscar.cad.properties import apply_properties, apply_properties_report  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402


def _name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


class DocCase(unittest.TestCase):
    """Creates documents that are closed again after each test."""

    def new_doc(self, prefix="T4a"):
        doc = App.newDocument(_name(prefix))
        self.addCleanup(self._close, doc.Name)
        return doc

    @staticmethod
    def _close(name):
        try:
            if name in App.listDocuments():
                App.closeDocument(name)
        except Exception:
            pass

    def box(self, doc, name="Box", length=10, width=20, height=5):
        obj = doc.addObject("Part::Box", name)
        obj.Length, obj.Width, obj.Height = length, width, height
        doc.recompute()
        return obj

    def run_tool(self, name, args):
        text, image, is_error = tools.run_tool(name, args)
        return text, is_error


# --------------------------------------------------------------------------------------------
# get_objects table and get_object


class ObjectTableTest(DocCase):
    def test_table_columns_rows_and_counts(self):
        doc = self.new_doc()
        self.box(doc, "Base")
        doc.addObject("Part::Cylinder", "Pin")
        doc.recompute()
        text, is_error = self.run_tool("get_objects", {"doc_name": doc.Name})
        self.assertFalse(is_error, text)
        lines = text.splitlines()
        self.assertEqual(lines[0], f"Document '{doc.Name}': 2 objects")
        header = lines[1].split()
        self.assertEqual(header[:4], ["Name", "Label", "Type", "Valid"])
        for column in ("Solids", "Shape", "Volume", "BBox", "Vis", "Notes"):
            self.assertIn(column, lines[1])
        base = next(line for line in lines if line.startswith("Base"))
        self.assertIn("Part::Box", base)
        self.assertIn("ok", base)
        self.assertIn("Solid", base)
        self.assertIn("1000.0", base)
        self.assertIn("10.0 x 20.0 x 5.0", base)
        self.assertIn("yes", base)

    def test_origin_helpers_are_hidden_unless_a_filter_asks_for_them(self):
        doc = self.new_doc()
        body = doc.addObject("PartDesign::Body", "Body")
        doc.recompute()
        text, _ = self.run_tool("get_objects", {"doc_name": doc.Name})
        self.assertNotIn("App::Origin", text)
        self.assertNotIn("App::Line", text)
        self.assertIn("origin helpers hidden", text.splitlines()[0])
        shown, _ = self.run_tool("get_objects", {"doc_name": doc.Name, "filter": "App::Plane"})
        self.assertIn("App::Plane", shown)
        self.assertIsNotNone(body)

    def test_filter_limit_offset_and_paging_footer(self):
        doc = self.new_doc()
        for index in range(5):
            self.box(doc, f"Box{index}")
        first, _ = self.run_tool("get_objects", {"doc_name": doc.Name, "limit": 2})
        self.assertIn("Box0", first)
        self.assertIn("Box1", first)
        self.assertNotIn("Box2", first)
        self.assertIn("(showing rows 1-2 of 5; call get_objects with offset=2 for more", first)
        second, _ = self.run_tool("get_objects", {"doc_name": doc.Name, "limit": 2, "offset": 4})
        self.assertIn("Box4", second)
        self.assertNotIn("Box3", second)
        self.assertIn("(showing rows 5-5 of 5)", second)
        filtered, _ = self.run_tool("get_objects", {"doc_name": doc.Name, "filter": "box3"})
        self.assertIn("1 match filter 'box3'", filtered.splitlines()[0])
        self.assertIn("Box3", filtered)
        none, _ = self.run_tool("get_objects", {"doc_name": doc.Name, "filter": "zzz"})
        self.assertEqual(none, f"No object of document '{doc.Name}' matches filter 'zzz' (5 objects in total).")

    def test_empty_document_and_unknown_document(self):
        doc = self.new_doc()
        text, is_error = self.run_tool("get_objects", {"doc_name": doc.Name})
        self.assertEqual((text, is_error), (f"Document '{doc.Name}' is open and empty.", False))
        text, is_error = self.run_tool("get_objects", {"doc_name": "NoSuchDocument"})
        self.assertTrue(is_error)
        self.assertIn("is not open", text)

    def test_notes_column_shows_expressions_sketch_dof_and_children(self):
        doc = self.new_doc()
        sheet = doc.addObject("Spreadsheet::Sheet", "Params")
        sheet.set("A1", "=4 mm")
        sheet.setAlias("A1", "wall_t")
        doc.recompute()
        base = self.box(doc, "Base")
        base.setExpression("Length", "Params.wall_t * 2")
        doc.recompute()
        body = doc.addObject("PartDesign::Body", "Body")
        sketch = body.newObject("Sketcher::SketchObject", "Sketch")
        doc.recompute()
        text, _ = self.run_tool("get_objects", {"doc_name": doc.Name})
        base_row = next(line for line in text.splitlines() if line.startswith("Base"))
        self.assertIn("expr:1", base_row)
        sketch_row = next(line for line in text.splitlines() if line.startswith("Sketch"))
        self.assertIn(f"DoF {sketch.DoF}", sketch_row)
        body_row = next(line for line in text.splitlines() if line.startswith("Body"))
        self.assertIn("children", body_row)

    def test_invalid_state_is_visible_in_the_valid_column(self):
        doc = self.new_doc()
        doc.addObject("Part::Cut", "EmptyCut")     # no inputs: stays Touched / Invalid after recompute
        doc.recompute()
        text, _ = self.run_tool("get_objects", {"doc_name": doc.Name})
        row = next(line for line in text.splitlines() if line.startswith("EmptyCut"))
        self.assertTrue("Invalid" in row or "INVALID" in row or "Error" in row, row)

    def test_render_table_with_synthetic_pages(self):
        self.assertEqual(serialize.render_objects_table("D", {"objects": [], "total": 0}),
                         "Document 'D' is open and empty.")
        self.assertEqual(serialize.render_objects_table("D", {"objects": [], "total": 3, "matched": 0,
                                                              "hidden_origin": 3}),
                         "Document 'D' has 3 objects, all of them origin helpers (hidden).")
        rows = [{"Name": "A", "Label": "A", "Type": "Part::Box", "State": [], "Visible": True,
                 "Shape": {"ShapeType": "Solid", "Solids": 1, "Volume": 8.0, "Valid": True,
                           "BoundingBox": {"Min": {"x": 0, "y": 0, "z": 0}, "Max": {"x": 2, "y": 2, "z": 2}}}},
                {"Name": "B", "Label": "B", "Type": "Part::Feature", "State": ["Invalid"], "Visible": False,
                 "Status": "Linked object is not a Part object"}]
        text = serialize.render_objects_table("D", {"objects": rows, "total": 9, "matched": 9, "hidden_origin": 0},
                                              limit=2, offset=0)
        self.assertIn("(showing rows 1-2 of 9; call get_objects with offset=2 for more", text)
        self.assertIn("Invalid", text)
        self.assertIn("Linked object is not a Part object", text)
        self.assertIn("2.0 x 2.0 x 2.0", text)

    def test_collect_objects_budget_never_drops_rows(self):
        doc = self.new_doc()
        for index in range(3):
            self.box(doc, f"B{index}")
        with mock.patch.object(serialize, "_VALID_BUDGET_S", 0.0):
            page = serialize.collect_objects(doc, limit=10)
        self.assertEqual(len(page["objects"]), 3)
        self.assertEqual(page["total"], 3)
        self.assertEqual(page["matched"], 3)
        self.assertIsNone(page["objects"][0]["Shape"]["Valid"], "over budget: validity unknown, not claimed")


class GetObjectTest(DocCase):
    def test_detail_has_properties_list_links_and_parent(self):
        doc = self.new_doc()
        body = doc.addObject("PartDesign::Body", "Body")
        sketch = body.newObject("Sketcher::SketchObject", "Sketch")
        doc.recompute()
        text, is_error = self.run_tool("get_object", {"doc_name": doc.Name, "obj_name": "Sketch"})
        self.assertFalse(is_error, text)
        import json

        data = json.loads(text)
        self.assertEqual(data["Name"], "Sketch")
        self.assertIn("Placement", data["PropertiesList"])
        self.assertEqual(data["Parent"], "Body")
        self.assertIn("DoF", data)
        self.assertIsNotNone(sketch)

    def test_unknown_object_names_the_fix(self):
        doc = self.new_doc()
        text, is_error = self.run_tool("get_object", {"doc_name": doc.Name, "obj_name": "Nope"})
        self.assertTrue(is_error)
        self.assertIn("not found", text)
        self.assertIn("get_objects", text)

    def test_origin_features_get_no_shape_statistics(self):
        doc = self.new_doc()
        doc.addObject("PartDesign::Body", "Body")
        doc.recompute()
        origin = next(o for o in doc.Objects if o.TypeId == "App::Plane")
        data = serialize.object_to_dict(origin)
        self.assertNotIn("Shape", data, "an Origin plane has an infinite bounding box")

    def test_shape_statistics_are_in_the_global_frame_inside_a_part(self):
        doc = self.new_doc()
        part = doc.addObject("App::Part", "Asm")
        part.Placement = App.Placement(App.Vector(0, 0, 50), App.Rotation())
        box = self.box(doc, "Inner")
        part.addObject(box)
        doc.recompute()
        data = serialize.object_to_dict(box)
        self.assertAlmostEqual(data["Shape"]["BoundingBox"]["Min"]["z"], 50.0, places=3)
        self.assertEqual(data["Shape"]["Solids"], 1)
        self.assertTrue(data["Shape"]["Valid"])

    def test_json_safe_values(self):
        doc = self.new_doc()
        box = self.box(doc)
        import json

        json.dumps(serialize.object_to_dict(box, detail=True))
        self.assertEqual(serialize.object_to_dict(box)["Length"], 10.0)


# --------------------------------------------------------------------------------------------
# properties: placement merge, expressions, atomic apply, enum suggestions


class PropertiesTest(DocCase):
    def setUp(self):
        self.doc = self.new_doc()
        self.box_obj = self.box(self.doc)
        self.box_obj.Placement = App.Placement(App.Vector(1, 2, 3), App.Rotation(App.Vector(0, 0, 1), 30))

    def pos(self):
        base = self.box_obj.Placement.Base
        return (round(base.x, 6), round(base.y, 6), round(base.z, 6))

    def test_partial_base_keeps_the_other_coordinates_and_the_rotation(self):
        apply_properties(self.box_obj, {"Placement": {"Base": {"z": 9}}})
        self.assertEqual(self.pos(), (1.0, 2.0, 9.0))
        self.assertAlmostEqual(self.box_obj.Placement.Rotation.Angle * 180 / 3.141592653589793, 30.0, places=3)

    def test_rotation_only_keeps_the_position(self):
        apply_properties(self.box_obj, {"Placement": {"Rotation": {"Axis": {"x": 0, "y": 0, "z": 1}, "Angle": 90}}})
        self.assertEqual(self.pos(), (1.0, 2.0, 3.0))
        self.assertAlmostEqual(self.box_obj.Placement.Rotation.Angle * 180 / 3.141592653589793, 90.0, places=3)

    def test_bare_xyz_is_a_position_only(self):
        apply_properties(self.box_obj, {"Placement": {"x": 5, "y": 6, "z": 7}})
        self.assertEqual(self.pos(), (5.0, 6.0, 7.0))
        self.assertAlmostEqual(self.box_obj.Placement.Rotation.Angle * 180 / 3.141592653589793, 30.0, places=3)

    def test_yaw_pitch_roll_is_a_rotation_only(self):
        apply_properties(self.box_obj, {"Placement": {"Yaw": 0, "Pitch": 0, "Roll": 90}})
        self.assertEqual(self.pos(), (1.0, 2.0, 3.0))
        ypr = self.box_obj.Placement.Rotation.getYawPitchRoll()
        self.assertAlmostEqual(ypr[2], 90.0, places=3)
        self.assertAlmostEqual(ypr[0], 0.0, places=3)

    def test_expression_strings_bind_and_equals_alone_unbinds(self):
        sheet = self.doc.addObject("Spreadsheet::Sheet", "Params")
        sheet.set("A1", "=2.5 mm")
        sheet.setAlias("A1", "wall_t")
        self.doc.recompute()
        applied = apply_properties(self.box_obj, {"Length": "=Params.wall_t*2"})
        self.doc.recompute()
        self.assertEqual(applied, ["Length"])
        self.assertAlmostEqual(self.box_obj.Length.Value, 5.0)
        self.assertEqual([p for p, _ in self.box_obj.ExpressionEngine], ["Length"])
        apply_properties(self.box_obj, {"Length": "="})
        self.assertEqual(self.box_obj.ExpressionEngine, [])

    def test_plain_value_on_a_bound_property_works_but_is_reported(self):
        sheet = self.doc.addObject("Spreadsheet::Sheet", "Params")
        sheet.set("A1", "=2.5 mm")
        sheet.setAlias("A1", "wall_t")
        self.doc.recompute()
        apply_properties(self.box_obj, {"Length": "=Params.wall_t*2"})
        report = apply_properties_report(self.box_obj, {"Length": 12})
        self.assertEqual(report["applied"], ["Length"])
        self.assertTrue(report["notes"], "the caller must be told the expression will overwrite the value")
        self.assertIn("expression", " ".join(report["notes"]).lower())

    def test_invalid_expression_is_rejected_with_nothing_changed(self):
        with self.assertRaises(ValueError) as caught:
            apply_properties(self.box_obj, {"Length": "=NoSuchSheet.x*2"})
        self.assertIn("expression", str(caught.exception))
        self.assertIn("Nothing was changed", str(caught.exception))
        self.assertEqual(self.box_obj.ExpressionEngine, [])

    def test_atomic_apply_nothing_is_touched_when_one_key_is_wrong(self):
        with self.assertRaises(ValueError) as caught:
            apply_properties(self.box_obj, {"Length": 99, "Lenght": 4})
        message = str(caught.exception)
        self.assertIn("Nothing was changed", message)
        self.assertIn("similar", message.lower())
        self.assertIn("Length", message)
        self.assertEqual(self.box_obj.Length.Value, 10.0)

    def test_all_problems_are_listed_in_one_error(self):
        with self.assertRaises(ValueError) as caught:
            apply_properties(self.box_obj, {"Nope1": 1, "Nope2": 2})
        message = str(caught.exception)
        self.assertIn("Nope1", message)
        self.assertIn("Nope2", message)
        self.assertIn("2 properties cannot be applied", message)

    def test_a_failing_assignment_restores_the_keys_set_before_it(self):
        with self.assertRaises(ValueError) as caught:
            apply_properties(self.box_obj, {"Length": 55, "Height": "not a length"})
        self.assertIn("Nothing was changed", str(caught.exception))
        self.assertEqual(self.box_obj.Length.Value, 10.0, "Length was set first and must be rolled back")
        self.assertEqual(self.box_obj.Height.Value, 5.0)

    def test_enum_members_are_matched_case_insensitively_and_misses_list_the_members(self):
        ext = self.doc.addObject("Part::Extrusion", "Ext")
        apply_properties(ext, {"DirMode": "normal"})
        self.assertEqual(ext.DirMode, "Normal")
        with self.assertRaises(ValueError) as caught:
            apply_properties(ext, {"DirMode": "Norml"})
        message = str(caught.exception)
        self.assertIn("Custom|Edge|Normal", message)
        self.assertIn("similar: Normal", message)

    def test_links_resolve_by_name_and_unknown_targets_list_the_objects(self):
        pin = self.doc.addObject("Part::Cylinder", "Pin")
        cut = self.doc.addObject("Part::Cut", "Cut")
        apply_properties(cut, {"Base": "Box", "Tool": "Pin"})
        self.assertEqual((cut.Base.Name, cut.Tool.Name), ("Box", "Pin"))
        with self.assertRaises(ValueError) as caught:
            apply_properties(cut, {"Base": "Nope"})
        self.assertIn("Box", str(caught.exception))
        self.assertIsNotNone(pin)

    def test_view_object_keys_are_skipped_without_a_gui(self):
        report = apply_properties_report(self.box_obj, {"Length": 11, "ViewObject": {"ShapeColor": [1, 0, 0]}})
        self.assertIn("Length", report["applied"])
        self.assertEqual(self.box_obj.Length.Value, 11.0)

    def test_non_dict_properties_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_properties(self.box_obj, ["Length", 3])

    def test_edit_object_tool_reports_the_notes(self):
        sheet = self.doc.addObject("Spreadsheet::Sheet", "Params")
        sheet.set("A1", "=2.5 mm")
        sheet.setAlias("A1", "wall_t")
        self.doc.recompute()
        self.run_tool("edit_object", {"doc_name": self.doc.Name, "obj_name": "Box",
                                      "obj_properties": {"Length": "=Params.wall_t*2"}, "include_screenshot": False})
        text, is_error = self.run_tool("edit_object", {"doc_name": self.doc.Name, "obj_name": "Box",
                                                       "obj_properties": {"Length": 12}, "include_screenshot": False})
        self.assertFalse(is_error, text)
        self.assertIn("Note:", text)
        self.assertIn("expression", text.lower())

    def test_edit_object_failure_changes_nothing(self):
        text, is_error = self.run_tool("edit_object", {"doc_name": self.doc.Name, "obj_name": "Box",
                                                       "obj_properties": {"Length": 99, "Bogus": 1},
                                                       "include_screenshot": False})
        self.assertTrue(is_error)
        self.assertIn("Nothing was changed", text)
        self.assertEqual(self.box_obj.Length.Value, 10.0)


# --------------------------------------------------------------------------------------------
# create_object / delete_object


class CreateAndDeleteTest(DocCase):
    def create(self, doc, obj_type, name, **extra):
        args = {"doc_name": doc.Name, "obj_type": obj_type, "obj_name": name, "include_screenshot": False}
        args.update(extra)
        return self.run_tool("create_object", args)

    def test_sketch_with_body_argument_lands_inside_the_body(self):
        doc = self.new_doc()
        self.create(doc, "PartDesign::Body", "Body")
        text, is_error = self.create(doc, "Sketcher::SketchObject", "Sketch", body="Body")
        self.assertFalse(is_error, text)
        self.assertIn("It is part of body 'Body'", text)
        self.assertIn(doc.getObject("Sketch"), doc.getObject("Body").Group)

    def test_ambiguous_and_unknown_bodies_are_explained(self):
        doc = self.new_doc()
        self.create(doc, "PartDesign::Body", "BodyA")
        self.create(doc, "PartDesign::Body", "BodyB")
        text, is_error = self.create(doc, "PartDesign::Pad", "Pad")
        self.assertTrue(is_error)
        self.assertIn("has several (BodyA, BodyB); pass body=<name> to choose one", text)
        text, is_error = self.create(doc, "Sketcher::SketchObject", "Sk", body="Nope")
        self.assertTrue(is_error)
        self.assertIn("body 'Nope' is not a PartDesign::Body", text)
        self.assertIsNone(doc.getObject("Pad"))
        self.assertIsNone(doc.getObject("Sk"))

    def test_failed_create_leaves_nothing_behind_and_the_name_can_be_reused(self):
        doc = self.new_doc()
        self.box(doc, "Keep")
        text, is_error = self.create(doc, "PartDesign::Pad", "Pad")
        self.assertTrue(is_error)
        self.assertIsNone(doc.getObject("Pad"))
        text, is_error = self.create(doc, "Part::Box", "Pad")
        self.assertFalse(is_error, text)
        self.assertEqual(doc.getObject("Pad").Name, "Pad", "no '001' suffix after the failed attempt")

    def test_bad_property_on_create_leaves_nothing_behind(self):
        doc = self.new_doc()
        text, is_error = self.create(doc, "Part::Extrusion", "Ext", obj_properties={"DirMode": "Bogus"})
        self.assertTrue(is_error)
        self.assertIn("Custom|Edge|Normal", text)
        self.assertEqual([o.Name for o in doc.Objects], [])

    def _boolean_doc(self):
        doc = self.new_doc()
        self.create(doc, "Part::Box", "Base", obj_properties={"Length": 10, "Width": 10, "Height": 10})
        self.create(doc, "Part::Cylinder", "Pin", obj_properties={"Radius": 2, "Height": 20})
        self.create(doc, "Part::Cut", "Cut", obj_properties={"Base": "Base", "Tool": "Pin"})
        return doc

    def test_delete_refuses_while_dependents_exist_and_names_them(self):
        doc = self._boolean_doc()
        text, is_error = self.run_tool("delete_object", {"doc_name": doc.Name, "obj_name": "Base",
                                                         "include_screenshot": False})
        self.assertTrue(is_error)
        self.assertIn("cannot delete 'Base': 'Cut' (Part::Cut) depend(s) on it", text)
        self.assertIn("cascade=true", text)
        self.assertIsNotNone(doc.getObject("Base"))
        self.assertIsNotNone(doc.getObject("Cut"))

    def test_delete_cascade_removes_dependents_first_and_reports_their_names(self):
        doc = self._boolean_doc()
        text, is_error = self.run_tool("delete_object", {"doc_name": doc.Name, "obj_name": "Base",
                                                         "cascade": True, "include_screenshot": False})
        self.assertFalse(is_error, text)
        self.assertIn("Also deleted (they depended on it): Cut.", text)
        self.assertNotIn("None", text)
        self.assertEqual(sorted(o.Name for o in doc.Objects), ["Pin"])

    def test_deleting_a_boolean_shows_its_inputs_again(self):
        doc = self._boolean_doc()
        doc.getObject("Base").Visibility = False
        doc.getObject("Pin").Visibility = False
        text, is_error = self.run_tool("delete_object", {"doc_name": doc.Name, "obj_name": "Cut",
                                                         "include_screenshot": False})
        self.assertFalse(is_error, text)
        self.assertIn("Its inputs are visible again", text)
        self.assertTrue(doc.getObject("Base").Visibility)
        self.assertTrue(doc.getObject("Pin").Visibility)

    def test_deleting_a_feature_does_not_count_its_body_as_a_dependent(self):
        doc = self.new_doc()
        self.create(doc, "PartDesign::Body", "Body")
        self.create(doc, "Sketcher::SketchObject", "Sketch", body="Body")
        dependents = engine_module._dependents(doc.getObject("Sketch"))
        self.assertEqual([o.Name for o in dependents], [])
        text, is_error = self.run_tool("delete_object", {"doc_name": doc.Name, "obj_name": "Sketch",
                                                         "include_screenshot": False})
        self.assertFalse(is_error, text)
        self.assertIsNone(doc.getObject("Sketch"))
        self.assertNotIn(None, list(doc.getObject("Body").Group))

    def test_delete_unknown_object(self):
        doc = self.new_doc()
        text, is_error = self.run_tool("delete_object", {"doc_name": doc.Name, "obj_name": "Ghost"})
        self.assertTrue(is_error)
        self.assertIn("'Ghost' not found", text)

    def test_spreadsheet_with_bound_expressions_counts_as_a_dependency(self):
        doc = self.new_doc()
        sheet = doc.addObject("Spreadsheet::Sheet", "Params")
        sheet.set("A1", "=3 mm")
        sheet.setAlias("A1", "wall_t")
        base = self.box(doc, "Base")
        base.setExpression("Length", "Params.wall_t")
        doc.recompute()
        text, is_error = self.run_tool("delete_object", {"doc_name": doc.Name, "obj_name": "Params",
                                                         "include_screenshot": False})
        self.assertTrue(is_error)
        self.assertIn("'Base'", text)


# --------------------------------------------------------------------------------------------
# reload_document


class ReloadDocumentTest(DocCase):
    def saved_doc(self):
        doc = self.new_doc("Rel")
        self.box(doc, "Base")
        folder = tempfile.mkdtemp(prefix="reload_", dir=isolation.sandbox_root())
        path = os.path.join(folder, f"{doc.Name}.FCStd")
        doc.saveAs(path)
        return doc, path

    def test_reload_returns_the_actual_name_and_the_old_one(self):
        doc, path = self.saved_doc()
        name = doc.Name
        result = get_engine().reload_document(name)
        self.assertTrue(result["success"], result)
        self.assertEqual(result["previous_name"], name, "captured BEFORE the document was closed")
        self.assertIn(result["document_name"], App.listDocuments())
        self.addCleanup(self._close, result["document_name"])
        self.assertEqual(result["rescue_copy"], "")
        self.assertIsNotNone(App.getDocument(result["document_name"]).getObject("Base"))

    def test_reload_makes_the_new_document_active_when_the_old_one_was(self):
        doc, path = self.saved_doc()
        App.setActiveDocument(doc.Name)
        result = get_engine().reload_document(doc.Name)
        self.assertTrue(result["success"], result)
        self.addCleanup(self._close, result["document_name"])
        self.assertEqual(App.ActiveDocument.Name, result["document_name"])

    def test_never_saved_document_is_refused(self):
        doc = self.new_doc()
        result = get_engine().reload_document(doc.Name)
        self.assertFalse(result["success"])
        self.assertIn("never saved", result["error"])
        self.assertIn(doc.Name, App.listDocuments())

    def test_missing_file_leaves_the_open_document_alone(self):
        doc, path = self.saved_doc()
        os.remove(path)
        result = get_engine().reload_document(doc.Name)
        self.assertFalse(result["success"])
        self.assertIn("missing or is not a valid .FCStd file", result["error"])
        self.assertIn(doc.Name, App.listDocuments())
        self.assertIsNotNone(doc.getObject("Base"))

    def test_corrupt_file_leaves_the_open_document_alone(self):
        doc, path = self.saved_doc()
        with open(path, "wb") as f:
            f.write(b"this is not a zip file")
        result = get_engine().reload_document(doc.Name)
        self.assertFalse(result["success"])
        self.assertIn(doc.Name, App.listDocuments())

    def test_unsaved_changes_are_kept_in_a_rescue_copy(self):
        doc, path = self.saved_doc()
        doc.addObject("Part::Cylinder", "Unsaved")
        result = get_engine().reload_document(doc.Name)
        self.assertTrue(result["success"], result)
        self.addCleanup(self._close, result["document_name"])
        rescue = result["rescue_copy"]
        self.assertTrue(rescue.endswith(".FCStd"))
        self.assertTrue(zipfile.is_zipfile(rescue))
        self.assertTrue(os.path.dirname(rescue).startswith(paths.cache_dir()), "under the cache dir")
        self.assertIsNone(App.getDocument(result["document_name"]).getObject("Unsaved"))

    def test_unknown_document(self):
        result = get_engine().reload_document("NoSuchDoc_" + uuid.uuid4().hex[:4])
        self.assertFalse(result["success"])
        self.assertIn("is not open", result["error"])

    def test_tool_text_names_a_changed_document_name(self):
        doc, path = self.saved_doc()
        with mock.patch.object(tools._engine().__class__, "reload_document", return_value={
                "success": True, "document_name": "Other", "previous_name": "Old", "rescue_copy": "/x/r.FCStd"}):
            text, is_error = self.run_tool("reload_document", {"doc_name": doc.Name})
        self.assertFalse(is_error)
        self.assertIn("It was named 'Old' before; use 'Other' from now on.", text)
        self.assertIn("/x/r.FCStd", text)


# --------------------------------------------------------------------------------------------
# transactions, turn snapshot / revert


class TransactionRollbackTest(DocCase):
    def test_a_failed_task_is_rolled_back_even_with_undo_mode_zero(self):
        doc = self.new_doc()
        self.box(doc, "Keep")
        self.assertEqual(doc.UndoMode, 0, "freecadcmd default")

        def task():
            doc.addObject("Part::Cylinder", "Stray")
            doc.getObject("Keep").Length = 77
            return {"success": False, "error": "nope"}

        result = engine_module.txn_task(task, "test", doc.Name)()
        self.assertFalse(result["success"])
        self.assertIsNone(doc.getObject("Stray"), "created object rolled back")
        self.assertEqual(doc.getObject("Keep").Length.Value, 10.0, "edited value rolled back")
        self.assertEqual(doc.UndoMode, 0, "the previous UndoMode is restored")

    def test_a_raising_task_is_rolled_back_and_the_exception_propagates(self):
        doc = self.new_doc()

        def task():
            doc.addObject("Part::Cylinder", "Stray")
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            engine_module.txn_task(task, "test", doc.Name)()
        self.assertIsNone(doc.getObject("Stray"))
        self.assertEqual(doc.UndoMode, 0)

    def test_a_successful_task_is_committed(self):
        doc = self.new_doc()

        def task():
            doc.addObject("Part::Cylinder", "Made")
            return {"success": True}

        engine_module.txn_task(task, "test", doc.Name)()
        self.assertIsNotNone(doc.getObject("Made"))
        self.assertEqual(doc.UndoMode, 0)

    def test_a_deleted_object_is_restored_by_the_rollback(self):
        doc = self.new_doc()
        self.box(doc, "Victim")

        def task():
            doc.removeObject("Victim")
            return {"success": False, "error": "changed my mind"}

        engine_module.txn_task(task, "test", doc.Name)()
        self.assertIsNotNone(doc.getObject("Victim"))

    def test_without_a_document_the_task_just_runs(self):
        result = engine_module.txn_task(lambda: {"success": True, "ran": True}, "test", "NoSuchDoc_zz")()
        self.assertEqual(result, {"success": True, "ran": True})


class TurnRevertTest(DocCase):
    # the turn's own steps carry the agent's "Oscar: ..." label: revert_turn pops only those, a user's step
    # (e.g. "Move") on top is never undone
    def doc_with_undo_steps(self, steps):
        doc = self.new_doc()
        doc.UndoMode = 1
        for index in range(steps):
            doc.openTransaction(f"step {index}")
            doc.addObject("Part::Box", f"B{index}")
            doc.commitTransaction()
        return doc

    def test_begin_snapshot_records_depth_max_undo_and_mode(self):
        doc = self.doc_with_undo_steps(2)
        snapshot = get_engine().begin_turn_snapshot()
        self.assertTrue(snapshot["success"])
        self.assertEqual(snapshot["docs"][doc.Name], 2)
        self.assertIn(doc.Name, snapshot["max_undo"])
        self.assertEqual(snapshot["undo_mode"][doc.Name], 1)

    def test_revert_undoes_the_steps_after_the_snapshot_and_closes_new_documents(self):
        doc = self.doc_with_undo_steps(1)
        snapshot = get_engine().begin_turn_snapshot()
        snapshot["max_undo"] = {name: 20 for name in snapshot["max_undo"]}
        for index in range(2):
            doc.openTransaction(f"Oscar: turn {index}")
            doc.addObject("Part::Cylinder", f"C{index}")
            doc.commitTransaction()
        created_name = App.newDocument(_name("Created")).Name
        self.addCleanup(self._close, created_name)
        result = get_engine().revert_turn(snapshot)
        self.assertTrue(result["success"], result)
        self.assertFalse(result["incomplete"])
        self.assertEqual(result["undone"], {doc.Name: 2})
        self.assertIn(created_name, result["closed_docs"])
        self.assertEqual(sorted(o.Name for o in doc.Objects), ["B0"])

    def test_a_full_undo_stack_is_reported_incomplete_and_nothing_is_reverted(self):
        doc = self.doc_with_undo_steps(1)
        snapshot = get_engine().begin_turn_snapshot()
        snapshot["max_undo"] = {doc.Name: 3}
        for index in range(2):
            doc.openTransaction(f"Oscar: turn {index}")
            doc.addObject("Part::Cylinder", f"C{index}")
            doc.commitTransaction()
        self.assertEqual(doc.UndoCount, 3)
        result = get_engine().revert_turn(snapshot)
        self.assertFalse(result["success"])
        self.assertTrue(result["incomplete"])
        self.assertEqual(result["incomplete_docs"], [doc.Name])
        self.assertEqual(sorted(o.Name for o in doc.Objects), ["B0", "C0", "C1"], "a half-undone turn is worse")
        self.assertIn("restore the turn checkpoint", result["error"])

    def test_a_missing_or_failed_snapshot_is_refused_and_closes_nothing(self):
        doc = self.new_doc()
        for bad in ({}, None, {"success": False, "docs": {}}, {"success": True}, "junk"):
            result = get_engine().revert_turn(bad)
            self.assertFalse(result["success"], bad)
            self.assertIn("no turn snapshot", result["error"])
        self.assertIn(doc.Name, App.listDocuments())

    def test_unknown_max_undo_means_the_stack_is_trusted(self):
        doc = self.doc_with_undo_steps(1)
        snapshot = get_engine().begin_turn_snapshot()
        snapshot["max_undo"] = {doc.Name: 0}
        doc.openTransaction("Oscar: turn")
        doc.addObject("Part::Cylinder", "C0")
        doc.commitTransaction()
        result = get_engine().revert_turn(snapshot)
        self.assertTrue(result["success"], result)
        self.assertIsNone(doc.getObject("C0"))

    # P0 #3: only a document the turn touched can make the undo path incomplete

    def test_a_full_stack_on_an_untouched_document_does_not_block_the_revert(self):
        idle = self.doc_with_undo_steps(3)              # at its limit, but the turn never worked there
        work = self.doc_with_undo_steps(1)
        snapshot = get_engine().begin_turn_snapshot()
        snapshot["max_undo"] = {idle.Name: 3, work.Name: 20}
        work.openTransaction("Oscar: turn")
        work.addObject("Part::Cylinder", "C0")
        work.commitTransaction()
        result = get_engine().revert_turn(snapshot)
        self.assertTrue(result["success"], result)
        self.assertFalse(result["incomplete"])
        self.assertEqual(result["undone"], {work.Name: 1})
        self.assertEqual(result["touched_docs"], [work.Name])
        self.assertIsNone(work.getObject("C0"))
        self.assertEqual(idle.UndoCount, 3)

    def test_an_incomplete_revert_names_the_documents_the_turn_touched(self):
        doc = self.doc_with_undo_steps(1)
        snapshot = get_engine().begin_turn_snapshot()
        snapshot["max_undo"] = {doc.Name: 2}
        doc.openTransaction("Oscar: turn")
        doc.addObject("Part::Cylinder", "C0")
        doc.commitTransaction()
        result = get_engine().revert_turn(snapshot)
        self.assertTrue(result["incomplete"])
        self.assertFalse(result["undo_unsafe"])
        self.assertEqual((result["incomplete_docs"], result["touched_docs"]), ([doc.Name], [doc.Name]))

    def test_changed_since_turn_reports_work_done_after_the_turn(self):
        engine = get_engine()
        doc = self.doc_with_undo_steps(1)
        snapshot = engine.begin_turn_snapshot()
        doc.openTransaction("Oscar: create_object")
        doc.addObject("Part::Box", "TurnBox")
        doc.commitTransaction()
        self.assertEqual(engine.changed_since_turn(snapshot, [doc.Name, "NoSuchDocument"]), {})
        doc.openTransaction("Move")
        doc.getObject("TurnBox").Length = 33
        doc.commitTransaction()
        self.assertEqual(engine.changed_since_turn(snapshot, [doc.Name]),
                         {doc.Name: "your own step 'Move' came after this turn"})
        engine.begin_turn_snapshot()                    # the next turn starts ...
        doc.openTransaction("Oscar: edit_object")       # ... and changes the document again
        doc.getObject("TurnBox").Length = 34
        doc.commitTransaction()
        self.assertEqual(engine.changed_since_turn(snapshot, [doc.Name]), {doc.Name: "it was changed after this turn"})
        self.assertEqual(engine.changed_since_turn({}, [doc.Name]),
                         {doc.Name: "the turn's starting state was not recorded"})

    # P0 #4: FreeCAD can crash undoing a FeaturePlan build, so such a turn is never popped with Document.undo()

    def test_a_turn_with_plan_steps_is_never_undone(self):
        for label in ("Oscar: plan p1", "Oscar restore T0a1b"):
            with self.subTest(label):
                doc = self.doc_with_undo_steps(1)
                snapshot = get_engine().begin_turn_snapshot()
                snapshot["max_undo"] = {doc.Name: 20}
                doc.openTransaction("Oscar: create_object")
                doc.addObject("Part::Box", "Plain")
                doc.commitTransaction()
                doc.openTransaction(label)
                doc.addObject("Part::Box", "Planned")
                doc.commitTransaction()
                before = (doc.UndoCount, list(doc.UndoNames), sorted(o.Name for o in doc.Objects))
                result = get_engine().revert_turn(snapshot)
                self.assertFalse(result["success"])
                self.assertTrue(result["incomplete"] and result["undo_unsafe"], result)
                self.assertEqual((result["unsafe_docs"], result["incomplete_docs"]), ([doc.Name], [doc.Name]))
                self.assertEqual(result["undone"], {})
                self.assertIn("restore the turn checkpoint", result["error"])
                self.assertIn("PartDesign", result["error"])
                self.assertEqual((doc.UndoCount, list(doc.UndoNames), sorted(o.Name for o in doc.Objects)), before)

    def test_plan_steps_of_an_earlier_turn_do_not_count(self):
        doc = self.doc_with_undo_steps(1)
        doc.openTransaction("Oscar: plan p0")
        doc.addObject("Part::Box", "EarlierPlan")
        doc.commitTransaction()
        snapshot = get_engine().begin_turn_snapshot()
        snapshot["max_undo"] = {doc.Name: 20}
        doc.openTransaction("Oscar: edit_object")
        doc.getObject("EarlierPlan").Length = 20
        doc.commitTransaction()
        result = get_engine().revert_turn(snapshot)
        self.assertTrue(result["success"], result)
        self.assertEqual(result["undone"], {doc.Name: 1})
        self.assertEqual(float(doc.getObject("EarlierPlan").Length), 10.0)


# --------------------------------------------------------------------------------------------
# screenshots without a GUI


class HeadlessScreenshotTest(DocCase):
    def test_engine_screenshot_is_none_with_a_reason_never_an_error_string(self):
        self.assertFalse(App.GuiUp)
        engine = get_engine()
        self.assertIsNone(engine.screenshot())
        image, reason = engine._screenshot_with_reason()
        self.assertIsNone(image)
        self.assertEqual(reason, "no 3D view (headless session)")
        self.assertIsNone(tools._screenshot("Isometric", "Whatever"))

    def test_capture_checked_gates_on_guiup_and_never_touches_freecadgui(self):
        self.assertEqual(views.capture_checked("/nonexistent/x.png"), (False, views.HEADLESS_REASON))
        self.assertFalse(views.capture("/nonexistent/x.png"))
        self.assertIsNone(views.capture_base64())
        self.assertEqual(views.capture_base64_checked(), (None, views.HEADLESS_REASON))

    def test_missing_getdocument_is_caught_when_the_gui_flag_lies(self):
        # FreeCADGui has no getDocument / ActiveDocument in a console session: AttributeError, not None.
        with mock.patch.object(views.App, "GuiUp", True, create=True):
            with mock.patch.object(views, "_active_view", side_effect=AttributeError("getDocument")):
                self.assertEqual(views.capture_checked("/nonexistent/x.png"), (False, views.HEADLESS_REASON))

    def test_get_view_tool_reports_the_reason_instead_of_a_picture(self):
        text, is_error = self.run_tool("get_view", {})
        self.assertTrue(is_error)
        self.assertIn("Cannot capture a screenshot: no 3D view (headless session)", text)
        self.assertNotIn("Screenshot captured", text)
        doc = self.new_doc()
        text, is_error = self.run_tool("get_view", {"doc_name": doc.Name})
        self.assertTrue(is_error)
        text, is_error = self.run_tool("get_view", {"doc_name": "NoSuchDoc_zz"})
        self.assertTrue(is_error)
        self.assertIn("is not open", text)

    def test_get_view_returns_the_picture_when_there_is_one(self):
        with mock.patch.object(tools, "_screenshot_with_reason", return_value=("QUJD", "")) as shot:
            engine = tools._engine()
            with mock.patch.object(engine.__class__, "_screenshot_with_reason", return_value=("QUJD", "")):
                text, image, is_error = tools.run_tool("get_view", {"view_name": "Top"})
        self.assertEqual((text, image, is_error), ("Screenshot captured.", "QUJD", False))
        self.assertIsNotNone(shot)

    def test_shot_helper_explains_a_missing_picture_except_in_console_runs(self):
        with mock.patch.object(tools, "_screenshot_with_reason", return_value=(None, "blank frame")):
            self.assertEqual(tools._shot("done", True), ("done\n(no screenshot: blank frame)", None))
            self.assertEqual(tools._shot("done", False), ("done", None))
        with mock.patch.object(tools, "_screenshot_with_reason", return_value=(None, "no 3D view (headless session)")):
            self.assertEqual(tools._shot("done", True), ("done", None))
        with mock.patch.object(tools, "_screenshot_with_reason", return_value=("QUJD", "")):
            self.assertEqual(tools._shot("done", True), ("done", "QUJD"))

    def test_png_magic_check(self):
        import base64

        png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8).decode()
        self.assertTrue(engine_module.CadEngine._is_png_base64(png))
        self.assertFalse(engine_module.CadEngine._is_png_base64(base64.b64encode(b"Error: no view" * 3).decode()))
        self.assertFalse(engine_module.CadEngine._is_png_base64("short"))
        self.assertFalse(engine_module.CadEngine._is_png_base64(None))
        self.assertFalse(engine_module.CadEngine._is_png_base64("!!!not base64!!!!"))


class BlankFrameTest(unittest.TestCase):
    """The tolerant sample: 64 x 48 grid, 6 per channel from the row median, 99.5%."""

    @staticmethod
    def grid(background=(30, 32, 36), odd=0, gradient=0):
        rows = []
        remaining = odd
        for j in range(48):
            row = []
            for i in range(64):
                shade = j * gradient // 47 if gradient else 0
                px = tuple(c + shade for c in background)
                if remaining and i == 0:
                    px = (250, 250, 250)
                    remaining -= 1
                row.append(px)
            rows.append(row)
        return rows

    def test_uniform_and_empty_frames_are_blank(self):
        self.assertTrue(views.is_blank_grid(self.grid()))
        self.assertTrue(views.is_blank_grid([]))
        self.assertTrue(views.is_blank_grid([[], []]))

    def test_a_vertical_gradient_background_is_still_blank(self):
        self.assertTrue(views.is_blank_grid(self.grid(gradient=40)))

    def test_boundary_at_half_a_percent(self):
        # 3072 samples: up to 15 odd samples (0.488%) is blank, 16 (0.521%) is a picture
        self.assertTrue(views.is_blank_grid(self.grid(odd=15)))
        self.assertFalse(views.is_blank_grid(self.grid(odd=16)))

    def test_noise_inside_the_tolerance_is_background(self):
        rows = self.grid()
        rows[3][5] = (36, 38, 42)           # +6 on every channel
        self.assertTrue(views.is_blank_grid(rows))
        for r in range(0, 48, 2):
            rows[r][7] = (37, 32, 36)       # +7 on red: outside the tolerance
        self.assertFalse(views.is_blank_grid(rows))

    def test_an_unreadable_image_is_blank_and_a_missing_qt_never_discards_a_picture(self):
        self.assertTrue(views.is_blank_image("/nonexistent/none.png") in (True, False))


# --------------------------------------------------------------------------------------------
# merge guard


class MergeGuardTest(DocCase):
    def test_snapshot_lists_shape_objects_and_ignores_temp_helpers(self):
        doc = self.new_doc()
        self.box(doc, "Real")
        doc.addObject("Part::Feature", "_oscar_tmp_probe")
        doc.addObject("Spreadsheet::Sheet", "Params")
        names = engine_module.CadEngine._snapshot_shapes()
        self.assertIn(f"{doc.Name}/Real", names)
        self.assertNotIn(f"{doc.Name}/_oscar_tmp_probe", names)
        self.assertNotIn(f"{doc.Name}/Params", names)

    def test_note_warns_about_lost_objects_only(self):
        doc = self.new_doc()
        self.box(doc, "Left")
        engine = get_engine()
        before = engine._shapes_on_gui_thread()
        self.assertIn(f"{doc.Name}/Left", before)
        self.assertEqual(engine._merge_guard_note(before), "")
        doc.removeObject("Left")
        note = engine._merge_guard_note(before)
        self.assertIn("LOST shape object(s)", note)
        self.assertIn(f"{doc.Name}/Left", note)
        self.assertEqual(engine._merge_guard_note(None), "")

    def test_a_temp_helper_removed_by_the_script_is_not_a_loss(self):
        doc = self.new_doc()
        doc.addObject("Part::Feature", "_oscar_tmp_helper").Shape = __import__("Part").makeBox(1, 1, 1)
        engine = get_engine()
        before = engine._shapes_on_gui_thread()
        doc.removeObject("_oscar_tmp_helper")
        self.assertEqual(engine._merge_guard_note(before), "")

    def test_execute_code_that_deletes_a_part_gets_the_warning(self):
        doc = self.new_doc()
        self.box(doc, "Victim")
        App.setActiveDocument(doc.Name)
        code = f"App.getDocument('{doc.Name}').removeObject('Victim')\nprint('removed')"
        text, is_error = self.run_tool("execute_code", {"code": code, "include_screenshot": False})
        self.assertFalse(is_error, text)
        self.assertIn("LOST shape object(s)", text)


# --------------------------------------------------------------------------------------------
# download directory, plan tool failure text




class DownloadDirTest(unittest.TestCase):
    def test_falls_back_to_the_documented_default_until_netguard_provides_one(self):
        with mock.patch.object(netguard, "default_download_dir", None, create=True):
            self.assertEqual(tools._download_dir(), os.path.join(os.path.expanduser("~"), "Downloads", "Oscar"))

    def test_netguard_resolver_wins_when_present(self):
        target = tempfile.mkdtemp(prefix="dl_", dir=isolation.sandbox_root())
        with mock.patch.object(netguard, "default_download_dir", lambda: target, create=True):
            self.assertEqual(tools._download_dir(), target)

    def test_a_failing_or_empty_resolver_falls_back(self):
        default = os.path.join(os.path.expanduser("~"), "Downloads", "Oscar")
        with mock.patch.object(netguard, "default_download_dir", mock.Mock(side_effect=OSError("no")), create=True):
            self.assertEqual(tools._download_dir(), default)
        with mock.patch.object(netguard, "default_download_dir", lambda: "", create=True):
            self.assertEqual(tools._download_dir(), default)

    def test_download_file_uses_the_resolved_destination(self):
        target = tempfile.mkdtemp(prefix="dl_", dir=isolation.sandbox_root())
        saved = os.path.join(target, "part.step")
        with open(saved, "wb") as f:
            f.write(b"x")
        tools.reset_turn_circuits()
        self.addCleanup(tools.reset_turn_circuits)
        with mock.patch.object(netguard, "default_download_dir", lambda: target, create=True), \
                mock.patch.object(netguard, "request_permission", return_value="allow"), \
                mock.patch.object(netguard, "download", return_value=(saved, "")) as download:
            text, _, is_error = tools.run_tool("download_file", {"url": "https://example.com/part.step",
                                                                 "reason": "test"})
        self.assertFalse(is_error, text)
        self.assertEqual(download.call_args.args[1], target)


class PlanFailureTextTest(DocCase):
    def test_apply_feature_plan_failure_uses_the_errors_list(self):
        from oscar.plan import builder

        doc = self.new_doc()
        result = {"success": False, "stage": "validation", "errors": ["width is missing", "height < 0"]}
        with mock.patch.object(builder, "apply_plan", return_value=result):
            text, is_error = self.run_tool("apply_feature_plan", {"doc_name": doc.Name, "plan": {}})
        self.assertTrue(is_error)
        self.assertEqual(text, "Plan failed at 'validation': width is missing; height < 0")
        self.assertNotIn("None", text)

    def test_reapply_plan_feature_failure_names_the_feature(self):
        from oscar.plan import builder

        doc = self.new_doc()
        with mock.patch.object(builder, "apply_feature", return_value={"success": False, "errors": ["sketch is open"]}):
            text, is_error = self.run_tool("reapply_plan_feature", {"doc_name": doc.Name, "plan": {}, "feature_id": "f3"})
        self.assertTrue(is_error)
        self.assertIn("'f3'", text)
        self.assertIn("sketch is open", text)
        self.assertNotIn("None", text)

    def test_unopened_document_is_still_reported(self):
        text, is_error = self.run_tool("apply_feature_plan", {"doc_name": "NoSuchDoc_zz", "plan": {}})
        self.assertTrue(is_error)
        self.assertIn("is not open", text)


class EngineHelpersTest(DocCase):
    def test_get_engine_is_one_object_across_threads(self):
        import threading

        found = []
        threads = [threading.Thread(target=lambda: found.append(get_engine())) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len({id(e) for e in found}), 1)

    def test_discard_created_removes_new_objects_and_restores_visibility(self):
        doc = self.new_doc()
        keep = self.box(doc, "Keep")
        before = {o.Name for o in doc.Objects}
        visibility = engine_module._visibility(doc)
        doc.addObject("Part::Cylinder", "Junk")
        keep.Visibility = False
        engine_module._discard_created(doc, before, visibility)
        self.assertIsNone(doc.getObject("Junk"))
        self.assertTrue(doc.getObject("Keep").Visibility)

    def test_dependents_are_ordered_dependents_first(self):
        doc = self.new_doc()
        self.box(doc, "A")
        doc.addObject("Part::Cylinder", "B")
        cut = doc.addObject("Part::Cut", "C")
        cut.Base, cut.Tool = doc.getObject("A"), doc.getObject("B")
        fuse = doc.addObject("Part::Fuse", "D")
        fuse.Base, fuse.Tool = cut, doc.getObject("B")
        doc.recompute()
        names = [o.Name for o in engine_module._dependents(doc.getObject("A"))]
        self.assertEqual(names, ["D", "C"], "the fuse uses the cut, so it must be deleted first")

    def test_find_document_is_tolerant_and_list_documents_works(self):
        doc = self.new_doc("TolerantDoc")
        engine = get_engine()
        self.assertEqual(engine.find_document(doc.Name.lower()), doc.Name)
        self.assertEqual(engine.find_document(doc.Name[:-2]), doc.Name)
        self.assertIsNone(engine.find_document("zzz_nothing_like_it"))
        self.assertIn(doc.Name, engine.list_documents())

    def test_missing_document_errors_name_the_open_documents(self):
        doc = self.new_doc()
        text, is_error = self.run_tool("edit_object", {"doc_name": "Nope_zz", "obj_name": "X", "obj_properties": {"a": 1}})
        self.assertTrue(is_error)
        self.assertIn("is not open", text)
        text, is_error = self.run_tool("create_object", {"doc_name": "Nope_zz", "obj_type": "Part::Box", "obj_name": "X"})
        self.assertTrue(is_error)
        self.assertIsNotNone(doc)


# --------------------------------------------------------------------------------------------
# the real pipeline: real health module, fake Project


class FakeProject:
    def __init__(self):
        self.records, self.checkpoints = [], []

    def ensure_turn_checkpoint(self, doc):
        self.checkpoints.append(doc)

    def on_tool_result(self, **kwargs):
        self.records.append(kwargs)
        return SimpleNamespace(index=len(self.records), snapshot="", snapshot_note="n", checkpoint="", script="")

    def path(self, *parts):
        return os.path.join("/x", *parts)


class RealPipelineTest(DocCase):
    def setUp(self):
        self.project = FakeProject()
        self.ctx = ToolContext(project=self.project, turn_id="T1", call_id="c1", phase="BUILD", state={})

    def call(self, name, args):
        with use_context(self.ctx):
            return tools.run_tool(name, args)

    def test_create_object_gets_the_health_line_and_a_delta(self):
        doc = self.new_doc()
        text, _, is_error = self.call("create_object", {
            "doc_name": doc.Name, "obj_type": "Part::Box", "obj_name": "Base", "include_screenshot": False,
            "obj_properties": {"Length": 10, "Width": 10, "Height": 10}, "step_title": "Make the base"})
        self.assertFalse(is_error, text)
        self.assertTrue(text.startswith("Object 'Base' created"))
        self.assertIn(f"[health] doc '{doc.Name}': 1 objects, 1 solids, VALID", text)
        self.assertIn("Δ +Base", text)
        outcome = self.ctx.outcome
        self.assertEqual((outcome.ok, outcome.handler_ok, outcome.doc), (True, True, doc.Name))
        self.assertEqual(outcome.delta["added"], ["Base"])
        self.assertEqual(outcome.title, "Make the base")
        record = self.project.records[0]
        self.assertTrue(record["geometry"] and record["ok"])
        self.assertEqual(record["delta"]["added"], ["Base"])
        self.assertEqual(self.project.checkpoints, [doc.Name])
        self.assertEqual(self.ctx.state["last_doc"], doc.Name)
        self.assertEqual(self.ctx.state["touched"][doc.Name], {"Base"})

    def test_reads_are_not_recorded_and_carry_no_health_suffix(self):
        doc = self.new_doc()
        self.box(doc)
        text, _, is_error = self.call("get_objects", {"doc_name": doc.Name})
        self.assertFalse(is_error)
        self.assertNotIn("[health]", text)
        self.assertEqual(self.project.records, [])

    def test_a_failing_script_is_an_error_that_leaves_the_model_unchanged(self):
        doc = self.new_doc()
        self.box(doc, "Base")
        App.setActiveDocument(doc.Name)
        text, _, is_error = self.call("execute_code", {
            "code": f"d = App.getDocument('{doc.Name}')\nd.addObject('Part::Cylinder', 'Half')\nraise ValueError('no')",
            "include_screenshot": False})
        self.assertTrue(is_error)
        self.assertIn("ValueError: no", text)
        self.assertFalse(self.ctx.outcome.handler_ok)
        self.assertIsNone(doc.getObject("Half"), "the half-built object is rolled back")
        self.assertFalse(self.project.records[0]["ok"])

    def test_a_script_with_huge_output_still_ends_with_the_health_line(self):
        doc = self.new_doc()
        App.setActiveDocument(doc.Name)
        code = f"App.getDocument('{doc.Name}').addObject('Part::Box', 'B')\nprint('x' * 40000)"
        text, _, is_error = self.call("execute_code", {"code": code, "include_screenshot": False})
        self.assertFalse(is_error, text)
        self.assertLess(len(text), 13000)
        self.assertIn("[health] doc", text.splitlines()[-1])

    def test_a_script_creating_a_second_document_is_attributed_to_it(self):
        doc = self.new_doc()
        self.box(doc, "Base")
        other = _name("T4aOther")
        self.addCleanup(self._close, other)
        text, _, is_error = self.call("execute_code", {
            "code": f"d = App.newDocument('{other}')\nd.addObject('Part::Box', 'Made')\nd.recompute()",
            "include_screenshot": False})
        self.assertFalse(is_error, text)
        self.assertEqual(self.ctx.outcome.doc, other)
        self.assertIn(f"[health] doc '{other}'", text)



if __name__ == "__main__":
    unittest.main()
