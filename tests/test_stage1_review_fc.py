"""Review tests that need a real FreeCAD (run under freecadcmd by tests/run.py): the plan builder's
re-apply must never delete work it did not build, and the real-document behaviour of the other reviewed modules."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402

import project_support as S  # noqa: E402
from oscar.plan import builder  # noqa: E402

_MOD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLES = os.path.join(_MOD, "tests", "fixtures", "plans")


def _plan(name: str = "plate_4holes") -> dict:
    with open(os.path.join(_EXAMPLES, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


class ReapplyKeepsForeignWorkTest(unittest.TestCase):
    """T6a: applying a plan again replaces ITS OWN previous build; features somebody added to its Body by hand are
    never deleted silently (the builder refuses and says what blocks it)."""

    def setUp(self):
        self.doc = App.newDocument("ReapplyReview")
        self.addCleanup(lambda: App.closeDocument(self.doc.Name) if self.doc.Name in App.listDocuments() else None)

    def test_a_hand_added_feature_in_the_body_blocks_the_reapply_and_survives(self):
        plan = _plan()
        first = builder.apply_plan(self.doc, plan)
        self.assertTrue(first["success"], first.get("error"))
        body = self.doc.getObject(first["body"])
        user = body.newObject("PartDesign::AdditiveBox", "UserBox")
        user.Length = user.Width = user.Height = 5
        self.doc.recompute()
        volume = body.Shape.Volume
        names_before = sorted(o.Name for o in self.doc.Objects)

        second = builder.apply_plan(self.doc, plan)

        self.assertFalse(second["success"])
        self.assertEqual(second["stage"], "document")
        self.assertIn("UserBox", second["error"])
        self.assertIn("did not build", second["error"])
        self.assertIn("body_name", second["error"])
        self.assertEqual(sorted(o.Name for o in self.doc.Objects), names_before, "nothing may be removed")
        self.assertAlmostEqual(self.doc.getObject(first["body"]).Shape.Volume, volume, places=3)

    def test_a_fresh_body_name_builds_next_to_the_hand_edited_one(self):
        plan = _plan()
        first = builder.apply_plan(self.doc, plan)
        body = self.doc.getObject(first["body"])
        body.newObject("PartDesign::AdditiveBox", "UserBox")
        self.doc.recompute()
        other = builder.apply_plan(self.doc, plan, body_name="BodyTwo", sheet_name="ParametersTwo")
        self.assertTrue(other["success"], other.get("error"))
        self.assertIsNotNone(self.doc.getObject("UserBox"))
        self.assertIsNotNone(self.doc.getObject("Body"))
        self.assertIsNotNone(self.doc.getObject("BodyTwo"))

    def test_an_untouched_previous_build_is_still_replaced(self):
        plan = _plan()
        first = builder.apply_plan(self.doc, plan)
        again = builder.apply_plan(self.doc, plan)
        self.assertTrue(again["success"], again.get("error"))
        self.assertTrue(again["replaced"])
        self.assertEqual([o.Name for o in self.doc.Objects if o.TypeId == "PartDesign::Body"], [first["body"]])

    def test_an_object_outside_the_body_is_never_touched(self):
        plan = _plan()
        builder.apply_plan(self.doc, plan)
        self.doc.addObject("Part::Box", "MyOwnBox")
        again = builder.apply_plan(self.doc, plan)
        self.assertTrue(again["success"], again.get("error"))
        self.assertIsNotNone(self.doc.getObject("MyOwnBox"))


class RestoreContainersTest(unittest.TestCase):
    """Project.restore on the document shapes the builders really produce (the original tests use loose Part
    primitives only): a PartDesign Body with sketches, and an App::Part container with a placement and kit tags."""

    def setUp(self):
        self.proj = S.make_project(create=True)
        self.created = []
        self.addCleanup(self._close)

    def _close(self):
        S.close_docs(self.created)

    def _doc(self, name):
        doc = App.newDocument(name)
        self.created.append(doc.Name)
        return doc

    def test_a_partdesign_body_comes_back_valid_bound_and_editable(self):
        doc = self._doc("RestoreBody")
        built = builder.apply_plan(doc, _plan())
        self.assertTrue(built["success"], built.get("error"))
        volume = doc.getObject(built["body"]).Shape.Volume
        names = sorted(o.Name for o in doc.Objects)
        S.record(self.proj, doc=doc.Name, title="Build the plate")                      # checkpoint 001
        doc.addObject("Part::Box", "Junk")
        doc.recompute()

        result = self.proj.restore(doc.Name, "001")

        self.assertTrue(result.ok, result.error)
        self.assertTrue(result.reactivated)
        self.assertEqual(sorted(o.Name for o in doc.Objects), names)
        body = doc.getObject(built["body"])
        self.assertTrue(body.isValid())
        self.assertAlmostEqual(body.Shape.Volume, volume, places=4)
        self.assertEqual(len([o for o in doc.Objects if o.TypeId == "Sketcher::SketchObject"]),
                         len([o for o in body.Group if o.TypeId == "Sketcher::SketchObject"]),
                         "every sketch is a member of the restored Body")
        sheet = doc.getObject(built["spreadsheet"])
        alias = "thickness"
        value = sheet.get(alias)
        sheet.set(sheet.getCellFromAlias(alias), "=" + repr(float(getattr(value, "Value", value)) * 1.5))
        doc.recompute()
        self.assertAlmostEqual(doc.getObject(built["body"]).Shape.Volume / volume, 1.5, places=2,
                               msg="the expressions are still bound, so the restored model is still parametric")

    def test_an_app_part_container_keeps_its_group_placement_tags_and_expressions(self):
        doc = self._doc("RestorePart")
        part = doc.addObject("App::Part", "Part_Shell")
        box = doc.addObject("Part::Box", "A")
        cyl = doc.addObject("Part::Cylinder", "B")
        cut = doc.addObject("Part::Cut", "Shell")
        cut.Base, cut.Tool = box, cyl
        for obj in (box, cyl, cut):
            part.addObject(obj)
        part.Placement = App.Placement(App.Vector(0, 0, 50), App.Rotation())
        cut.addProperty("App::PropertyString", "OscarKit", "Oscar", "tag")
        cut.OscarKit = "enc1"
        sheet = doc.addObject("Spreadsheet::Sheet", "Params")
        sheet.set("A1", "=3 mm")
        sheet.setAlias("A1", "wall_t")
        box.setExpression("Width", "Params.wall_t*4")
        doc.recompute()
        names, volume, group = sorted(o.Name for o in doc.Objects), cut.Shape.Volume, [o.Name for o in part.Group]
        S.record(self.proj, doc=doc.Name, title="Build the shell")
        doc.removeObject("B")
        doc.addObject("Part::Sphere", "Ball")
        part.Placement = App.Placement()
        sheet.set("A1", "=9 mm")
        doc.recompute()

        result = self.proj.restore(doc.Name, "001")

        self.assertTrue(result.ok, result.error)
        self.assertEqual(sorted(o.Name for o in doc.Objects), names)
        part, cut, box = doc.getObject("Part_Shell"), doc.getObject("Shell"), doc.getObject("A")
        self.assertEqual([o.Name for o in part.Group], group)
        self.assertEqual(part.Placement.Base.z, 50.0)
        self.assertEqual(cut.getGlobalPlacement().Base.z, 50.0)
        self.assertAlmostEqual(cut.Shape.Volume, volume, places=3)
        self.assertEqual(cut.OscarKit, "enc1")
        self.assertEqual(list(box.ExpressionEngine), [("Width", "Params.wall_t * 4")])
        self.assertEqual(box.getParentGeoFeatureGroup().Name, "Part_Shell")
        self.assertFalse([o.Name for o in doc.Objects if "Invalid" in o.State])


class ExportModelArgumentsTest(unittest.TestCase):
    """A model-supplied argument of the wrong type is a model-correctable tool error, never a crash."""

    def call(self, **args):
        from oscar import tools

        return tools.run_tool("export_model", {"doc_name": "NoSuchDoc", "directory": os.path.join(os.sep, "tmp"), **args})

    def test_a_non_numeric_tolerance_is_reported_with_the_accepted_range(self):
        for bad in ("fast", [1], "nan"):
            text, _image, is_error = self.call(stl_tolerance=bad)
            self.assertTrue(is_error, repr(bad))
            self.assertIn("stl_tolerance", text)
            self.assertNotIn("Traceback", text)

    def test_a_null_tolerance_means_not_sent(self):
        # reconciled with the merged "optional argument sent as null falls back to its default" fix: the default
        # tolerance applies, so the only problem left is the unknown document
        text, _image, is_error = self.call(stl_tolerance=None)
        self.assertTrue(is_error)
        self.assertNotIn("stl_tolerance", text)
        self.assertIn("NoSuchDoc", text)

    def test_an_out_of_range_tolerance_keeps_its_message(self):
        text, _image, is_error = self.call(stl_tolerance=9)
        self.assertTrue(is_error)
        self.assertIn("outside 0.001..5 mm", text)


if __name__ == "__main__":
    unittest.main()
