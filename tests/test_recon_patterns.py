"""P1.3 repeated-hole patterns: a collinear equal-spacing row (>= 3) becomes
ONE Hole plus a native LinearPattern; holes at equal angular steps on a circle
(>= 4) become ONE Hole plus a native PolarPattern — both with editable
count/pitch parameters. Irregular arrangements and 2x2 grids keep individual
Hole features (a subset pattern would round unmeasured corners too).
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402

from oscar.recon import reconstruct  # noqa: E402


def plate_with_holes(holes, w=80, d=40, h=8, r=3.0):
    shape = Part.makeBox(w, d, h)
    for x, y in holes:
        shape = shape.cut(Part.makeCylinder(r, h * 3, App.Vector(x, y, -h)))
    return shape


class HolePatternTest(unittest.TestCase):
    def test_a_row_of_four_builds_one_hole_and_a_linear_pattern(self):
        shape = plate_with_holes(((10, 20), (30, 20), (50, 20), (70, 20)))
        mesh = Mesh.Mesh(shape.tessellate(0.3))
        doc = App.newDocument("P13Lin")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            self.assertEqual(len([o for o in doc.Objects
                                  if o.TypeId == "PartDesign::Hole"]), 1,
                             "the row is one original hole")
            pats = [o for o in doc.Objects if o.TypeId == "PartDesign::LinearPattern"]
            self.assertEqual(len(pats), 1)
            self.assertEqual(pats[0].Occurrences, 4)
            params = result["plan"]["parameters"]
            self.assertEqual(params["pat1_n"]["value"], 4)
            self.assertAlmostEqual(params["pat1_pitch"]["value"], 20.0,
                                   delta=0.15)
            body = doc.getObject(result["build"]["body"])
            self.assertLessEqual(abs(body.Shape.Volume - shape.Volume) / shape.Volume,
                                 0.02)
            self.assertEqual(result["meta"]["patterns"][0]["kind"], "linear")
            self.assertEqual(result["meta"]["patterns"][0]["count"], 4)
            self.assertAlmostEqual(result["meta"]["patterns"][0]["spacing_mm"],
                                   20.0, delta=0.15)
        finally:
            App.closeDocument(doc.Name)

    def test_a_bolt_circle_builds_one_hole_and_a_polar_pattern(self):
        holes = tuple((40 + 15 * math.cos(math.radians(a)),
                       20 + 15 * math.sin(math.radians(a)))
                      for a in range(0, 360, 72))
        shape = plate_with_holes(holes, w=80, d=40, r=2.5)
        mesh = Mesh.Mesh(shape.tessellate(0.3))
        doc = App.newDocument("P13Pol")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            self.assertEqual(len([o for o in doc.Objects
                                  if o.TypeId == "PartDesign::Hole"]), 1)
            pats = [o for o in doc.Objects if o.TypeId == "PartDesign::PolarPattern"]
            self.assertEqual(len(pats), 1)
            self.assertEqual(pats[0].Occurrences, 5)
            body = doc.getObject(result["build"]["body"])
            self.assertLessEqual(abs(body.Shape.Volume - shape.Volume) / shape.Volume,
                                 0.02)
            self.assertEqual(result["meta"]["patterns"][0]["kind"], "polar")
            self.assertAlmostEqual(result["meta"]["patterns"][0]["sweep_deg"],
                                   360.0, delta=2.0)
        finally:
            App.closeDocument(doc.Name)

    def test_irregular_spacing_keeps_individual_holes(self):
        shape = plate_with_holes(((10, 20), (25, 20), (58, 20), (70, 20)))
        mesh = Mesh.Mesh(shape.tessellate(0.3))
        doc = App.newDocument("P13Irr")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["build"]["success"], result["build"])
            self.assertEqual(len([o for o in doc.Objects
                                  if o.TypeId == "PartDesign::Hole"]), 4,
                             "unequal spacing is NOT a pattern")
            self.assertEqual([o for o in doc.Objects
                              if o.TypeId in ("PartDesign::LinearPattern",
                                              "PartDesign::PolarPattern")], [])
            self.assertNotIn("patterns", result["meta"])
        finally:
            App.closeDocument(doc.Name)

    def test_a_two_by_two_grid_keeps_individual_holes(self):
        # rows of two never reach the >= 3 evidence bar; a grid pattern op is
        # not expressible without patterning a pattern (documented v1 limit)
        shape = plate_with_holes(((20, 10), (60, 10), (20, 30), (60, 30)))
        mesh = Mesh.Mesh(shape.tessellate(0.3))
        doc = App.newDocument("P13Grid")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["build"]["success"], result["build"])
            self.assertEqual(len([o for o in doc.Objects
                                  if o.TypeId == "PartDesign::Hole"]), 4)
            self.assertEqual([o for o in doc.Objects
                              if o.TypeId in ("PartDesign::LinearPattern",
                                              "PartDesign::PolarPattern")], [])
        finally:
            App.closeDocument(doc.Name)


if __name__ == "__main__":
    unittest.main()


class ThinTrayTest(unittest.TestCase):
    """P1.5: a thin-walled FLAT tray builds through the prismatic level route
    (stacked pads + a measured pocket for the cavity) — thin-panel geometry
    with plane evidence no longer needs a shell workaround."""

    def test_a_thin_tray_builds_as_pads_plus_pocket(self):
        shape = Part.makeBox(60, 40, 20).cut(Part.makeBox(50, 30, 15, App.Vector(5, 5, 5)))
        mesh = Mesh.Mesh(shape.tessellate(0.3))
        doc = App.newDocument("P15Tray")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            meta = result["meta"]
            self.assertEqual(meta["pads"], 2)
            self.assertEqual(meta["pockets"], 1)
            body = doc.getObject(result["build"]["body"])
            self.assertLessEqual(abs(body.Shape.Volume - shape.Volume) / shape.Volume,
                                 0.02)
            # P1.6: the datum / symmetry evidence rides along for the agent
            self.assertIn("datums", meta, meta.keys())
        finally:
            App.closeDocument(doc.Name)
