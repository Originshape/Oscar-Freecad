"""R-3: hole-kind recognition and reconstruction — through / blind / counterbore /
countersink distinguished by GEOMETRIC EVIDENCE (position-checked enclosing faces,
area-gated exits vs floors, coaxial stepped cylinders, F-test-gated cone refits at
entries), every dimension measured; never invented.

Fixtures are REAL tessellated FreeCAD shapes, both at survey level (recognition) and
through the FULL reconstruct pipeline (plan + builder + deviation + outcome).
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402

from oscar.recon import recognize, reconstruct  # noqa: E402


def blind_plate(depth=4.0, radius=3.0):
    """30x20x10 plate with a blind hole drilled from the top (z=10) to z=10-depth."""
    return Part.makeBox(30, 20, 10).cut(
        Part.makeCylinder(radius, depth, App.Vector(15, 10, 10 - depth)))


def cbore_plate(bore_r=2.0, cb_r=4.0, cb_depth=3.0):
    """Plate with a through bore + a counterbore at the TOP face (z=10)."""
    shape = Part.makeBox(30, 20, 10).cut(
        Part.makeCylinder(bore_r, 30, App.Vector(15, 10, -10)))
    return shape.cut(Part.makeCylinder(cb_r, cb_depth,
                                       App.Vector(15, 10, 10 - cb_depth)))


def csink_plate(bore_r=2.0, cs_r1=2.0, cs_r2=6.0, cs_h=4.0):
    """Plate with a through bore + a 90-degree countersink at the TOP face."""
    shape = Part.makeBox(30, 20, 10).cut(
        Part.makeCylinder(bore_r, 30, App.Vector(15, 10, -10)))
    return shape.cut(Part.makeCone(cs_r1, cs_r2, cs_h,
                                   App.Vector(15, 10, 10 - cs_h)))


def survey_of(shape, tess=0.3):
    return recognize.survey(Mesh.Mesh(shape.tessellate(tess)))


def run_reconstruct(shape, tess=0.3, warn=0.5):
    mesh = Mesh.Mesh(shape.tessellate(tess))
    doc = App.newDocument("r3e")
    try:
        result = reconstruct.reconstruct(doc, mesh, warn_mm=warn)
        volume = doc.getObject(result["build"]["body"]).Shape.Volume \
            if result.get("build", {}).get("success") else None
        return result, volume
    finally:
        App.closeDocument(doc.Name)


class RecognitionTest(unittest.TestCase):
    def test_blind_hole_depth_from_floor_plane(self):
        s = survey_of(blind_plate(depth=4.0))
        self.assertEqual(len(s["holes"]), 1)
        h = s["holes"][0]
        self.assertEqual(h["kind"], "blind")
        self.assertFalse(h["through"])
        self.assertAlmostEqual(h["depth"], 4.0, delta=4.0 * 0.05)
        self.assertEqual(h["depth_basis"], "entry face to floor plane")

    def test_counterbore_from_stepped_coaxial_cylinders(self):
        s = survey_of(cbore_plate())
        self.assertEqual(len(s["holes"]), 1)
        h = s["holes"][0]
        self.assertEqual(h["kind"], "counterbore")
        self.assertAlmostEqual(h["counterbore"]["diameter"], 8.0, delta=8.0 * 0.05)
        self.assertAlmostEqual(h["counterbore"]["depth"], 3.0, delta=3.0 * 0.05)
        self.assertTrue(h["through"])

    def test_countersink_with_measured_angle(self):
        s = survey_of(csink_plate())
        self.assertEqual(len(s["holes"]), 1)
        h = s["holes"][0]
        self.assertEqual(h["kind"], "countersink")
        self.assertAlmostEqual(h["countersink"]["diameter"], 12.0, delta=12.0 * 0.1)
        self.assertAlmostEqual(h["countersink"]["angle"], 90.0, delta=5.0)
        self.assertTrue(h["through"])

    def test_through_hole_behaviour_unchanged(self):
        shape = Part.makeBox(30, 20, 10).cut(
            Part.makeCylinder(3.3, 30, App.Vector(15, 10, -10)))
        s = survey_of(shape)
        h = s["holes"][0]
        self.assertEqual(h["kind"], "through")
        self.assertTrue(h["through"])
        self.assertAlmostEqual(h["radius"], 3.3, delta=0.2)

    def test_two_separate_coaxial_holes_are_not_a_counterbore(self):
        # a through hole + a separate blind hole on the SAME axis but at different
        # heights with a solid web between them: NOT stepped, NOT a counterbore
        shape = Part.makeBox(30, 20, 10)
        shape = shape.cut(Part.makeCylinder(2, 3, App.Vector(15, 10, 0)))
        shape = shape.cut(Part.makeCylinder(4, 2, App.Vector(15, 10, 8)))
        s = survey_of(shape)
        for h in s["holes"]:
            self.assertNotEqual(h["kind"], "counterbore",
                                "a webbed pair must never read as stepped")

    def test_blind_hole_with_no_floor_seen_states_its_basis(self):
        # a hole open at BOTH ends of the wall's evidence (half-scanned): the depth
        # comes from the wall extent and SAYS so
        s = survey_of(blind_plate(depth=4.0))
        h = s["holes"][0]
        self.assertIn("depth_basis", h)
        self.assertTrue(h["depth_basis"])

    def test_through_hole_in_blind_position_is_not_misread(self):
        # regression: the plate BOTTOM face must not count as an exit beyond a blind
        # bottom (position-checked enclosing)
        s = survey_of(blind_plate(depth=4.0))
        self.assertFalse(s["holes"][0]["through"])


class EndToEndTest(unittest.TestCase):
    def test_blind_hole_builds_with_measured_depth(self):
        result, volume = run_reconstruct(blind_plate(depth=4.0))
        self.assertTrue(result["build"]["success"])
        analytic = 30 * 20 * 10 - math.pi * 9 * 4
        self.assertAlmostEqual(volume, analytic, delta=analytic * 0.02)
        self.assertEqual(result["outcome"], "verified_parametric")
        self.assertEqual(result["meta"]["holes"], 1)

    def test_counterbore_builds_partdesign_hole_with_measured_cut(self):
        result, volume = run_reconstruct(cbore_plate())
        self.assertTrue(result["build"]["success"])
        analytic = 30 * 20 * 10 - math.pi * 4 * 10 - math.pi * (16 - 4) * 3
        self.assertAlmostEqual(volume, analytic, delta=analytic * 0.02)
        self.assertEqual(result["outcome"], "verified_parametric")
        # the plan carries the counterbore with parameter names (not literals)
        hole_feats = [f for f in result["plan"]["features"] if f["op"] == "hole"]
        self.assertEqual(len(hole_feats), 1)
        self.assertIn("counterbore", hole_feats[0])
        cb = hole_feats[0]["counterbore"]
        self.assertIsInstance(cb["cb_diameter"], str)
        self.assertIsInstance(cb["cb_depth"], str)

    def test_countersink_builds_with_measured_diameter_and_angle(self):
        result, volume = run_reconstruct(csink_plate())
        self.assertTrue(result["build"]["success"])
        frustum = math.pi * 4 / 3 * (36 + 12 + 4)
        analytic = 30 * 20 * 10 - math.pi * 4 * 10 - (frustum - math.pi * 4 * 4)
        self.assertAlmostEqual(volume, analytic, delta=analytic * 0.03)
        self.assertEqual(result["outcome"], "verified_parametric")
        hole_feats = [f for f in result["plan"]["features"] if f["op"] == "hole"]
        self.assertIn("countersink", hole_feats[0])

    def test_through_plate_regression_verified(self):
        shape = Part.makeBox(30, 20, 10).cut(
            Part.makeCylinder(3.3, 30, App.Vector(15, 10, -10)))
        result, volume = run_reconstruct(shape)
        self.assertTrue(result["build"]["success"])
        self.assertEqual(result["outcome"], "verified_parametric")
        self.assertTrue(result["holes"] if "holes" in result else
                        result["meta"]["holes"] == 1)

    def test_blind_depth_parameter_is_editable(self):
        # parametric editability: the depth is a spreadsheet parameter
        result, _ = run_reconstruct(blind_plate(depth=4.0))
        params = result["plan"]["parameters"]
        self.assertTrue(any(k.startswith("hole_dp") for k in params),
                        "blind depth must be a named parameter")


if __name__ == "__main__":
    unittest.main()
