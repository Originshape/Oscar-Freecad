"""Native boss synthesis, end to end on the reference-part side: recognition
(``measure.find_bosses``: convex cylinders standing off a recognized PCB face),
the measurement surface (``Measurement.bosses``, ``render_text``) and the plan
emitter (``synthesize.boss_plan_features``) whose fragments validate as
schema-v2 features and build through the EXISTING plan builder as native,
spreadsheet-editable boss features.

Real FreeCAD documents under ``freecadcmd``; synthetic shapes for every
geometric edge case (rounded corners, separate standoffs, through pins,
out-of-range diameters, footprints crossing the board edge, stepped bosses).
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

import FreeCAD as App  # noqa: E402
import Part  # noqa: E402

from oscar.refparts import measure, synthesize  # noqa: E402
from oscar.plan import builder, schema  # noqa: E402

PLATE = 85.0 * 56.0 * 1.4


def plate(length=85.0, width=56.0, thickness=1.4):
    return Part.makeBox(length, width, thickness)


def bossed(x, y, d, h, on="top", length=85.0, width=56.0, thickness=1.4):
    """A plate with ONE cylindrical boss fused onto its ``top`` or ``bottom`` face."""
    base = plate(length, width, thickness)
    r = d / 2.0
    if on == "top":
        cyl = Part.makeCylinder(r, h, App.Vector(x, y, thickness))
    else:
        cyl = Part.makeCylinder(r, h, App.Vector(x, y, 0.0), App.Vector(0, 0, -1.0))
    return base.fuse(cyl)


def measured(shape, name="Ref_X"):
    return measure.analyze([measure.SolidRecord("PCB", shape)], object_name=name)


class FindBossesTest(unittest.TestCase):
    def bosses_of(self, shape, thickness=1.4, length=85.0, width=56.0):
        return measure.find_bosses(measure.SolidRecord("PCB", shape), thickness, length, width)

    def test_a_cylinder_fused_on_the_top_face_is_a_boss(self):
        self.assertEqual(self.bosses_of(bossed(20, 12, 6.0, 5.0)),
                         [{"x": 20.0, "y": 12.0, "d": 6.0, "h": 5.0, "face": "top"}])

    def test_a_cylinder_hanging_below_the_bottom_face(self):
        self.assertEqual(self.bosses_of(bossed(20, 12, 6.0, 2.5, on="bottom")),
                         [{"x": 20.0, "y": 12.0, "d": 6.0, "h": 2.5, "face": "bottom"}])

    def test_rounded_corners_are_not_bosses(self):
        base = plate()
        vertical = [e for e in base.Edges if abs(e.Vertexes[0].Z - e.Vertexes[1].Z) > 0.7]
        self.assertEqual(self.bosses_of(base.makeFillet(3.0, vertical)), [])

    def test_a_cylinder_through_the_board_reads_as_its_top_boss(self):
        # the union of plate and through-cylinder EQUALS plate + top boss: the reading is exact
        shape = plate().fuse(Part.makeCylinder(3.0, 1.4 + 5.0, App.Vector(20, 12, 0.0)))
        self.assertEqual(self.bosses_of(shape),
                         [{"x": 20.0, "y": 12.0, "d": 6.0, "h": 5.0, "face": "top"}])

    def test_separate_standoff_solids_are_not_pcb_bosses(self):
        m = measure.analyze([measure.SolidRecord("PCB", plate()),
                             measure.SolidRecord("STANDOFF", Part.makeCylinder(3.0, 5.0, App.Vector(20, 12, 1.4)))],
                            object_name="Ref_X")
        self.assertEqual(m.bosses, [])

    def test_diameter_out_of_range_and_edge_footprints_are_refused(self):
        self.assertEqual(self.bosses_of(bossed(20, 12, 1.5, 5.0)), [])       # below the boss diameter range
        self.assertEqual(self.bosses_of(bossed(83.0, 28, 6.0, 5.0)), [])     # footprint crosses the board edge
        # the boss itself grows the fused solid's box: only the BOARD outline refuses it
        self.assertEqual(len(self.bosses_of(bossed(83.0, 28, 6.0, 5.0), length=90.0)), 1)

    def test_a_stepped_boss_reports_only_the_step_seated_on_the_face(self):
        shape = plate().fuse(Part.makeCylinder(3.0, 2.0, App.Vector(20, 12, 1.4))) \
                      .fuse(Part.makeCylinder(2.0, 3.0, App.Vector(20, 12, 3.4)))
        self.assertEqual(self.bosses_of(shape),
                         [{"x": 20.0, "y": 12.0, "d": 6.0, "h": 2.0, "face": "top"}])

    def test_bosses_sort_top_first_then_y_x(self):
        shape = bossed(10, 40, 5.0, 4.0)
        shape = shape.fuse(Part.makeCylinder(2.5, 3.0, App.Vector(10, 10, 1.4)))
        shape = shape.fuse(Part.makeCylinder(2.5, 3.0, App.Vector(10, 10, 0.0), App.Vector(0, 0, -1.0)))
        self.assertEqual([(b["face"], b["y"]) for b in self.bosses_of(shape)],
                         [("top", 10.0), ("top", 40.0), ("bottom", 10.0)])


class MeasurementSurfaceTest(unittest.TestCase):
    def test_analyze_reports_bosses_and_render_text_lists_them(self):
        m = measured(bossed(20, 12, 6.0, 5.0))
        self.assertEqual(m.bosses, [{"x": 20.0, "y": 12.0, "d": 6.0, "h": 5.0, "face": "top"}])
        self.assertIn("bosses", m.to_dict())
        text = measure.render_text(m)
        self.assertIn("Bosses: 1 raised cylinder", text)
        self.assertIn("d 6 h 5 on the top face at (20, 12)", text)

    def test_no_boss_line_without_bosses(self):
        m = measured(plate())
        self.assertEqual(m.bosses, [])
        self.assertNotIn("Bosses:", measure.render_text(m))

    def test_a_boss_is_still_not_a_hole(self):
        m = measured(bossed(20, 12, 6.0, 5.0))
        self.assertEqual(m.holes, [])


class EmitterTest(unittest.TestCase):
    def test_fragment_parameters_carry_step_measurement_sources(self):
        params, feats = synthesize.boss_plan_features(measured(bossed(20, 12, 6.0, 5.0)))
        self.assertEqual(sorted(params), ["boss_d", "boss_h"])
        self.assertEqual(params["boss_d"]["value"], 6.0)
        self.assertEqual(params["boss_d"]["source"]["kind"], "step_measurement")
        self.assertIn("Ref_X", params["boss_d"]["source"]["ref"])
        self.assertEqual(feats, [{"id": "boss1", "op": "boss", "face": "top", "length": "boss_h",
                                  "note": "measured boss on the top face at (20, 12)",
                                  "profiles": [{"shape": "circle", "diameter": "boss_d", "center": [20.0, 12.0]}]}])

    def test_a_dict_measurement_works_too(self):
        params, feats = synthesize.boss_plan_features(measured(bossed(20, 12, 6.0, 5.0)).to_dict())
        self.assertEqual([f["id"] for f in feats], ["boss1"])
        self.assertEqual(params["boss_h"]["value"], 5.0)

    def test_equal_bosses_share_one_feature_and_unequal_heights_do_not(self):
        unequal = bossed(20, 12, 6.0, 5.0).fuse(Part.makeCylinder(3.25, 4.0, App.Vector(60, 40, 1.4)))
        params, feats = synthesize.boss_plan_features(measured(unequal))
        self.assertEqual([f["length"] for f in feats], ["boss1_h", "boss2_h"])
        self.assertEqual([p["diameter"] for f in feats for p in f["profiles"]], ["boss1_d", "boss2_d"])
        self.assertEqual(sorted(params), ["boss1_d", "boss1_h", "boss2_d", "boss2_h"])

        equal = bossed(20, 12, 6.0, 5.0).fuse(Part.makeCylinder(3.0, 5.0, App.Vector(60, 40, 1.4)))
        params, feats = synthesize.boss_plan_features(measured(equal))
        self.assertEqual(sorted(params), ["boss_d", "boss_h"])               # one feature, two profiles
        self.assertEqual([f["id"] for f in feats], ["boss1"])
        self.assertEqual([p["diameter"] for p in feats[0]["profiles"]], ["boss_d", "boss_d"])
        self.assertEqual([p["center"] for p in feats[0]["profiles"]], [[20.0, 12.0], [60.0, 40.0]])

    def test_faces_and_heights_split_features(self):
        shape = bossed(20, 12, 6.0, 5.0).fuse(Part.makeCylinder(3.0, 5.0, App.Vector(60, 40, 0.0), App.Vector(0, 0, -1.0)))
        _params, feats = synthesize.boss_plan_features(measured(shape))
        self.assertEqual([(f["face"], f["length"]) for f in feats], [("top", "boss1_h"), ("bottom", "boss2_h")])

    def test_no_bosses_emit_nothing(self):
        self.assertEqual(synthesize.boss_plan_features(measured(plate())), ({}, []))
        self.assertIsNone(synthesize.measured_plan(measured(plate())))

    def test_measured_plan_is_complete_and_in_the_board_frame(self):
        plan = synthesize.measured_plan(measured(bossed(20, 12, 6.0, 5.0)))
        self.assertEqual((plan["id"], plan["part_class"], plan["schema_version"]),
                         ("measured_board", "plate", 2))
        self.assertEqual(plan["parameters"]["plate_l"]["source"]["kind"], "step_measurement")
        self.assertEqual(plan["parameters"]["plate_cx"], {"expr": "plate_l / 2"})
        base = plan["features"][:2]
        self.assertEqual([f["id"] for f in base], ["base_sk", "base_pad"])
        self.assertEqual(base[0]["profile"]["center"], ["plate_cx", "plate_cy"])   # corner-anchored
        _params, bosses = synthesize.boss_plan_features(measured(bossed(20, 12, 6.0, 5.0)))
        self.assertEqual(plan["features"][2:], bosses)

    def test_emission_line_names_the_fragment_and_caps_it(self):
        line = synthesize.emission_line(measured(bossed(20, 12, 6.0, 5.0)))
        self.assertIn("Bosses measured (1)", line)
        self.assertIn('"op":"boss"', line)
        self.assertEqual(synthesize.emission_line(measured(plate())), "")
        many = plate()                                        # four heights -> four features, the line caps at three
        for x, h in ((10, 5.0), (30, 4.0), (50, 3.0), (70, 2.0)):
            many = many.fuse(Part.makeCylinder(3.0, h, App.Vector(x, 28, 1.4)))
        capped = synthesize.emission_line(measured(many))
        self.assertIn("Bosses measured (4)", capped)
        self.assertIn("+1 more", capped)


class NativeBuildTest(unittest.TestCase):
    def new_doc(self, name):
        doc = App.newDocument(name)
        self.addCleanup(lambda n=doc.Name: App.closeDocument(n) if n in App.listDocuments() else None)
        return doc

    def build(self, plan, name="BossEmit"):
        doc = self.new_doc(name)
        report = builder.apply_plan(doc, plan)
        self.assertTrue(report.get("success"),
                        f"{report.get('stage')}/{report.get('feature_id')}: {report.get('error')}")
        return doc, report

    def volume(self, doc):
        return float(doc.getObject("Body").Shape.Volume)

    def set_param(self, doc, alias, value):
        sheet = doc.getObject("Parameters")
        sheet.set(sheet.getCellFromAlias(alias), "=" + repr(float(value)))
        doc.recompute()

    def test_measured_plan_validates_and_builds_a_native_editable_boss(self):
        plan = synthesize.measured_plan(measured(bossed(20, 12, 6.0, 5.0)))
        ok, errors, _warnings = schema.validate(plan)
        self.assertEqual((ok, errors), (True, []))
        doc, report = self.build(plan)
        self.assertAlmostEqual(self.volume(doc), PLATE + math.pi * 9 * 5, delta=0.5)
        self.assertEqual(next(f for f in report["features"] if f["id"] == "boss1")["op"], "boss")
        self.set_param(doc, "boss_h", 8)                          # the driving parameter reshapes it
        self.assertAlmostEqual(self.volume(doc), PLATE + math.pi * 9 * 8, delta=0.5)
        shape = doc.getObject("Body").Shape
        self.assertTrue(shape.isValid())
        self.assertEqual(len(shape.Solids), 1)

    def test_editing_the_plate_grows_it_away_from_the_corner(self):
        plan = synthesize.measured_plan(measured(bossed(20, 12, 6.0, 5.0)))
        doc, _report = self.build(plan, "BossGrow")
        bb = doc.getObject("Body").Shape.BoundBox
        self.assertAlmostEqual(bb.XMax, 85.0, places=3)           # the board frame anchors the corner
        self.set_param(doc, "plate_l", 95)
        bb = doc.getObject("Body").Shape.BoundBox
        self.assertAlmostEqual(bb.XMax, 95.0, places=3)           # grows away from the corner ...
        self.assertAlmostEqual(bb.XMin, 0.0, places=3)           # ... the boss at x=20 stays put
        self.assertAlmostEqual(self.volume(doc), 95 * 56 * 1.4 + math.pi * 9 * 5, delta=0.5)

    def test_two_emitted_bosses_build_and_follow_the_shared_parameter(self):
        shape = bossed(20, 12, 6.0, 5.0).fuse(Part.makeCylinder(3.0, 5.0, App.Vector(60, 40, 1.4)))
        plan = synthesize.measured_plan(measured(shape))
        doc, _report = self.build(plan, "BossTwo")
        self.assertAlmostEqual(self.volume(doc), PLATE + 2 * math.pi * 9 * 5, delta=0.6)
        self.set_param(doc, "boss_d", 8)                          # both circles follow the one cell
        self.assertAlmostEqual(self.volume(doc), PLATE + 2 * math.pi * 16 * 5, delta=1.0)

    def test_a_boss_off_the_face_fails_gracefully_and_keeps_the_last_good_feature(self):
        plan = synthesize.measured_plan(measured(bossed(20, 12, 6.0, 5.0)))
        plan["features"][2]["profiles"][0]["center"] = [200.0, 12.0]   # tampered: misses the face
        ok, _errors, _warnings = schema.validate(plan)            # numbers are legal centres...
        self.assertTrue(ok)
        doc = self.new_doc("BossFail")
        report = builder.apply_plan(doc, plan)
        self.assertFalse(report["success"])
        self.assertEqual(report["feature_id"], "boss1")
        self.assertIn("separate solids", report["error"])
        self.assertEqual(report["completed"], ["base_sk", "base_pad"])
        self.assertAlmostEqual(self.volume(doc), PLATE, delta=0.5)   # the checkpoint kept the plate


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
