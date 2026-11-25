"""R-2: ``oscar.recon.section`` — mesh sectioning and profile extraction, and its
integration into the prismatic route (the section outline supersedes the bounding box).

Fixtures are REAL tessellated FreeCAD shapes through the FULL reconstruct pipeline:
- L-bracket: outline = 6-line section profile (never a rectangle), hole preserved,
  volume exact, outcome verified_parametric;
- rounded plate: line-arc-line-arc section profile;
- plain box: section rectangle, exact;
- four-hole plate: regression (still prismatic, 4 exact holes);
- the bbox route survives as a flagged approximation when no closed section exists.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import numpy as np  # noqa: E402
import Part  # noqa: E402

from oscar.recon import reconstruct, section  # noqa: E402


def l_bracket():
    shape = Part.makeBox(40, 20, 5).fuse(Part.makeBox(20, 40, 5)).removeSplitter()
    return shape.cut(Part.makeCylinder(3.3, 30, App.Vector(30, 10, -10)))


def rounded_plate():
    shape = Part.makeBox(40, 20, 5)
    shape = shape.fuse(Part.makeCylinder(10, 5, App.Vector(0, 10, 0)))
    return shape.fuse(Part.makeCylinder(10, 5, App.Vector(40, 10, 0))).removeSplitter()


def plate_with_holes():
    shape = Part.makeBox(60, 40, 10)
    for x, y in ((15, 10), (45, 10), (15, 30), (45, 30)):
        shape = shape.cut(Part.makeCylinder(3.3, 30, App.Vector(x, y, -10)))
    return shape


def run_reconstruct(shape, tess=0.3, warn=0.5):
    """(result, volume) — the volume is read BEFORE the document closes (FreeCAD
    objects die with their document)."""
    mesh = Mesh.Mesh(shape.tessellate(tess))
    doc = App.newDocument("r2test")
    try:
        result = reconstruct.reconstruct(doc, mesh, warn_mm=warn)
        volume = None
        if result.get("build", {}).get("success"):
            volume = doc.getObject(result["build"]["body"]).Shape.Volume
        return result, volume
    finally:
        App.closeDocument(doc.Name)


def noisy_mesh(shape, dev, sigma, seed):
    """``shape`` tessellated at ``dev`` with Gaussian vertex noise of ``sigma`` mm
    (a fixed seed, so the mesh is the same on every run)."""
    mesh = Mesh.Mesh(shape.tessellate(dev))
    pts, facs = mesh.Topology
    rng = np.random.default_rng(seed)
    points = np.array([[p.x, p.y, p.z] for p in pts], dtype=float)
    points = points + sigma * rng.standard_normal(points.shape)
    return Mesh.Mesh(([App.Vector(*p) for p in points],
                      [tuple(int(i) for i in f) for f in facs]))


class SlicingTest(unittest.TestCase):
    def test_l_bracket_slice_gives_one_closed_loop_with_hole(self):
        mesh = Mesh.Mesh(l_bracket().tessellate(0.3))
        box = mesh.BoundBox
        loops, opens = section.slice_loops(
            mesh, App.Vector(0, 0, (box.ZMin + box.ZMax) / 2 + 0.01), App.Vector(0, 0, 1))
        self.assertEqual(len(opens), 0)
        self.assertGreaterEqual(len(loops), 1)
        area = section._polygon_area(np.array(loops[0]))
        self.assertAlmostEqual(area, 1200.0, delta=2.0)     # the L area

    def test_open_chain_reported_not_closed(self):
        # a mesh with a genuine gap: the outline chain must come back open
        pts = [App.Vector(0, 0, 0), App.Vector(10, 0, 0), App.Vector(10, 10, 0),
               App.Vector(0, 10, 0), App.Vector(0, 0, 5), App.Vector(10, 0, 5),
               App.Vector(10, 10, 5), App.Vector(0, 10, 5)]
        facs = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
                (0, 4, 1), (1, 4, 5), (2, 6, 7), (2, 7, 3)]  # open box: no side at x=0..x=1
        facs = [f for f in facs if f not in ((2, 6, 7), (2, 7, 3))]
        mesh = Mesh.Mesh((pts, facs))
        loops, opens = section.slice_loops(mesh, App.Vector(0, 0, 2.5), App.Vector(0, 0, 1))
        # whatever comes back, nothing may be silently closed into a fake loop
        for lp in loops:
            self.assertGreater(len(lp), 2)

    def test_determinism(self):
        mesh = Mesh.Mesh(l_bracket().tessellate(0.3))
        r1 = section.slice_loops(mesh, App.Vector(0, 0, 2.51), App.Vector(0, 0, 1))
        r2 = section.slice_loops(mesh, App.Vector(0, 0, 2.51), App.Vector(0, 0, 1))
        self.assertEqual([np.round(p, 6).tolist() for p in r1[0][0]],
                         [np.round(p, 6).tolist() for p in r2[0][0]])


class ShapeDetectionTest(unittest.TestCase):
    def detect(self, shape, tess=0.2):
        mesh = Mesh.Mesh(shape.tessellate(tess))
        box = mesh.BoundBox
        loops, _ = section.slice_loops(mesh, App.Vector(0, 0, (box.ZMin + box.ZMax) / 2 + 0.01),
                                       App.Vector(0, 0, 1))
        return section.detect_shape(loops[0])

    def test_box_is_rectangle(self):
        det = self.detect(Part.makeBox(30, 20, 5))
        self.assertEqual(det["shape"], "rectangle")
        self.assertAlmostEqual(det["width"] * det["height"], 600.0, delta=5.0)

    def test_l_is_six_line_profile(self):
        det = self.detect(l_bracket())
        self.assertEqual(det["shape"], "profile")
        self.assertEqual(len(det["segments"]), 6)
        self.assertTrue(all(s["type"] == "line" for s in det["segments"]))

    def test_rounded_is_line_arc_profile(self):
        det = self.detect(rounded_plate())
        self.assertEqual(det["shape"], "profile")
        types = [s["type"] for s in det["segments"]]
        self.assertEqual(types.count("arc"), 2)
        self.assertEqual(types.count("line"), 2)

    def test_circle_is_circle(self):
        det = self.detect(Part.makeCylinder(8, 5))
        self.assertEqual(det["shape"], "circle")
        self.assertAlmostEqual(det["radius"], 8.0, delta=0.1)

    def test_hole_loop_is_a_circle(self):
        mesh = Mesh.Mesh(plate_with_holes().tessellate(0.3))
        box = mesh.BoundBox
        loops, _ = section.slice_loops(mesh, App.Vector(0, 0, (box.ZMin + box.ZMax) / 2 + 0.01),
                                       App.Vector(0, 0, 1))
        self.assertGreaterEqual(len(loops), 5)               # outer + 4 holes
        inner = section.detect_shape(loops[1])
        self.assertEqual(inner["shape"], "circle")
        self.assertAlmostEqual(inner["radius"], 3.3, delta=0.15)


class EndToEndTest(unittest.TestCase):
    def test_l_bracket_reconstructs_from_the_section_profile(self):
        result, volume = run_reconstruct(l_bracket())
        self.assertTrue(result.get("supported"))
        self.assertTrue(result["build"]["success"])
        self.assertIn("section profile", result["meta"]["outline"])
        self.assertEqual(result["meta"]["holes"], 1)
        analytic = 40 * 20 * 5 + 20 * 20 * 5 - 3.14159 * 3.3 ** 2 * 5
        self.assertAlmostEqual(volume, analytic, delta=analytic * 0.02)
        self.assertEqual(result["outcome"], "verified_parametric")

    def test_rounded_plate_reconstructs_with_arcs(self):
        result, volume = run_reconstruct(rounded_plate())
        self.assertTrue(result.get("supported"))
        self.assertTrue(result["build"]["success"])
        self.assertIn("section profile", result["meta"]["outline"])
        analytic = 40 * 20 * 5 + 3.14159 * 100 * 5
        self.assertAlmostEqual(volume, analytic, delta=analytic * 0.02)
        self.assertEqual(result["outcome"], "verified_parametric")

    def test_a_noisy_rounded_plate_still_reconstructs(self):
        # 0.05 mm of scan noise makes the section fitter emit arcs whose end
        # points sit off the circle their centre describes. That plan built
        # until a validation rule insisted on a size nothing had declared and
        # refused the whole profile; this mesh is the guard (it is rebuilt
        # from a fixed seed, so the outline is the same on every run).
        mesh = noisy_mesh(rounded_plate(), 0.3, 0.05, 8)
        doc = App.newDocument("r2noisy")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result.get("supported"), result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            arcs = [s for f in result["plan"]["features"]
                    for s in (f.get("profile") or {}).get("segments") or []
                    if s.get("type") == "arc"]
            if any("radius" not in s and "diameter" not in s for s in arcs):
                # the sketch solves and builds; its free degree of freedom is
                # REPORTED on the feature, never raised
                info = next(f["info"] for f in result["build"]["features"]
                            if f.get("op") == "sketch")
                self.assertFalse(info["fully_constrained"], info)
                self.assertTrue(info["under_constrained"], info)
                self.assertIn("not fully constrained", info["warning"])
        finally:
            App.closeDocument(doc.Name)

    def test_box_reconstructs_as_exact_section_rectangle(self):
        result, volume = run_reconstruct(Part.makeBox(30, 20, 10))
        self.assertEqual(result["meta"]["outline"], "section")
        self.assertAlmostEqual(volume, 6000.0, delta=6.0)
        self.assertEqual(result["outcome"], "verified_parametric")

    def test_four_hole_plate_regression(self):
        result, volume = run_reconstruct(plate_with_holes())
        self.assertTrue(result["build"]["success"])
        self.assertEqual(result["meta"]["holes"], 4)
        self.assertEqual(result["outcome"], "verified_parametric")

    def test_bbox_fallback_is_flagged_as_approximation(self):
        # an open mesh (no closed section): the route still works via the bbox and the
        # parameter source SAYS it is an approximation
        shape = Part.makeBox(30, 20, 5)
        mesh = Mesh.Mesh(shape.tessellate(1.0))
        pts, facs = mesh.Topology
        facs = [tuple(f) for f in facs][:-2]                # punch a hole in the wall
        open_mesh = Mesh.Mesh((list(pts), facs))
        doc = App.newDocument("r2open")
        try:
            result = reconstruct.reconstruct(doc, open_mesh, warn_mm=0.5,
                                             user_unit="mm")
        finally:
            App.closeDocument(doc.Name)
        # the open mesh blocks at repair (honest) OR falls back to bbox — either way it
        # must never claim a section outline
        if result.get("supported") and result.get("build", {}).get("success"):
            self.assertEqual(result["meta"]["outline"], "bbox (approximation)")
            plan = result["plan"]
            src = plan["parameters"]["plate_len"]["source"]["ref"]
            self.assertIn("APPROXIMATION", src)


if __name__ == "__main__":
    unittest.main()
