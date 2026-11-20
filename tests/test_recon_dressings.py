"""P1.1 multi-dressing tests: multiple fillet / chamfer groups (the far and the
base face each get their own bound op and parameter), a fillet + chamfer
combination, the deliberate no-double-rounding of vertical corners (a full-height
corner round is already expressed by the base sketch's outline arcs), and the
dressing-failure recovery ladder (a failed dress op never discards the successful
base build; what was dropped is recorded in the meta).
"""

import math
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402

from oscar.recon import reconstruct  # noqa: E402


def _edges_at(shape, z, tol=1e-6):
    return [e for e in shape.Edges
            if abs(e.CenterOfMass.z - z) < tol and e.Length > 1e-6]


def _vertical_edges(shape):
    out = []
    for e in shape.Edges:
        d = e.Vertexes[-1].Point - e.Vertexes[0].Point
        if d.Length > 1e-6 and abs(d.normalize().dot(App.Vector(0, 0, 1))) > 1 - 1e-9:
            out.append(e)
    return out


class MultiDressingTest(unittest.TestCase):
    """Far/base perimeter dressings become one bound op per face family."""

    @staticmethod
    def both_faces_filleted(top_r=2.0, bot_r=1.0):
        shape = Part.makeBox(60, 40, 8)
        shape = shape.makeFillet(top_r, _edges_at(shape, 8))
        return shape.makeFillet(bot_r, _edges_at(shape, 0))

    def test_two_fillet_groups_build_one_op_per_face(self):
        shape = self.both_faces_filleted()
        mesh = Mesh.Mesh(shape.tessellate(0.2))
        doc = App.newDocument("P11TwoF")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            params = result["plan"]["parameters"]
            self.assertAlmostEqual(params["fillet_r"]["value"], 2.0,
                                   delta=0.10 * 2.0)
            self.assertAlmostEqual(params["fillet_rb"]["value"], 1.0,
                                   delta=0.20 * 1.0)
            fillets = [o for o in doc.Objects if o.TypeId == "PartDesign::Fillet"]
            self.assertEqual(len(fillets), 2, "one fillet op per face family")
            body = doc.getObject(result["build"]["body"])
            self.assertLessEqual(abs(body.Shape.Volume - shape.Volume) / shape.Volume,
                                 0.02)
            self.assertEqual(sorted(result["meta"]["dressings"]),
                             ["fillet_bot", "fillet_top"],
                             "one fillet op per face family (order follows the "
                             "anchored frame, which a symmetric part may flip)")
        finally:
            App.closeDocument(doc.Name)

    def test_fillet_and_chamfer_combine_on_opposite_faces(self):
        shape = Part.makeBox(50, 35, 8)
        shape = shape.makeFillet(2.0, _edges_at(shape, 8))
        shape = shape.makeChamfer(1.5, _edges_at(shape, 0))
        mesh = Mesh.Mesh(shape.tessellate(0.2))
        doc = App.newDocument("P11FC")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            params = result["plan"]["parameters"]
            self.assertAlmostEqual(params["fillet_r"]["value"], 2.0,
                                   delta=0.10 * 2.0)
            # the measured strip width of a d x 45 deg chamfer is d * sqrt(2)
            self.assertAlmostEqual(params["chamfer_wb"]["value"], 1.5 * math.sqrt(2.0),
                                   delta=0.25 * 1.5 * math.sqrt(2.0))
            # P1.1b: the FEATURE binds the cut distance (1.5), not the strip
            # width (1.5*sqrt(2)) — the old binding overcut every chamfer
            cham = next(o for o in doc.Objects if o.TypeId == "PartDesign::Chamfer")
            self.assertAlmostEqual(float(cham.Size), 1.5, delta=0.15 * 1.5)
            self.assertAlmostEqual(params["chamfer_db"]["value"], 1.5,
                                   delta=0.15 * 1.5)
            self.assertEqual(len([o for o in doc.Objects
                                  if o.TypeId == "PartDesign::Fillet"]), 1)
            self.assertEqual(len([o for o in doc.Objects
                                  if o.TypeId == "PartDesign::Chamfer"]), 1)
            body = doc.getObject(result["build"]["body"])
            self.assertLessEqual(abs(body.Shape.Volume - shape.Volume) / shape.Volume,
                                 0.03)
        finally:
            App.closeDocument(doc.Name)

    def test_vertical_corner_rounds_are_not_doubled(self):
        # a full-height corner round is expressed by the base sketch's outline
        # arcs — a rule-based fillet op on top of that would round them twice
        box = Part.makeBox(60, 40, 10)
        shape = box.makeFillet(3.0, _vertical_edges(box))
        mesh = Mesh.Mesh(shape.tessellate(0.25))
        doc = App.newDocument("P11Vert")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            self.assertEqual([o for o in doc.Objects
                              if o.TypeId in ("PartDesign::Fillet",
                                              "PartDesign::Chamfer")], [],
                             "corner rounds belong to the sketch, not a dress op")
            self.assertNotIn("fillet_r", result["plan"]["parameters"])
            body = doc.getObject(result["build"]["body"])
            self.assertLessEqual(abs(body.Shape.Volume - shape.Volume) / shape.Volume,
                                 0.03)
        finally:
            App.closeDocument(doc.Name)


class DressingRecoveryTest(unittest.TestCase):
    """A failed dressing op never discards the successful base build."""

    def test_failed_fillet_retries_without_dressings(self):
        shape = Part.makeBox(60, 40, 10)
        for x, y in ((15, 10), (45, 10), (15, 30), (45, 30)):
            shape = shape.cut(Part.makeCylinder(3.3, 30, App.Vector(x, y, -10)))
        shape = shape.makeFillet(2.5, _edges_at(shape, 10))
        mesh = Mesh.Mesh(shape.tessellate(0.3))
        doc = App.newDocument("P11Rec")
        try:
            real = reconstruct.builder.apply_plan

            def failing(doc_, plan):
                if any(f.get("op") in ("fillet", "chamfer")
                       for f in plan["features"]):
                    return {"success": False, "stage": "build",
                            "feature_id": "fillet_top",
                            "error": "mocked fillet failure"}
                return real(doc_, plan)

            with mock.patch.object(reconstruct.builder, "apply_plan",
                                   side_effect=failing):
                result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            self.assertIn("dressings_dropped", result["meta"])
            self.assertEqual(result["meta"]["dressings_dropped"]["failed"],
                             "fillet_top")
            self.assertEqual([o for o in doc.Objects
                              if o.TypeId == "PartDesign::Fillet"], [])
            self.assertEqual(len([o for o in doc.Objects
                                  if o.TypeId == "PartDesign::Hole"]), 4,
                             "the holes survive the dropped dressing")
            self.assertNotIn("fillet_r", result["plan"]["parameters"],
                             "the dropped dressing's parameter is pruned")
        finally:
            App.closeDocument(doc.Name)


if __name__ == "__main__":
    unittest.main()
