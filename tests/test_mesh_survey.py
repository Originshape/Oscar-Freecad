"""Tests of ``oscar.refparts.mesh_survey``: the bounded face-surface survey of a sewn mesh solid.

Real FreeCAD documents under ``freecadcmd``: synthetic parts tessellated at a coarse, scan-like deviation,
so every recovered dimension carries the chord error of the mesh, and the assertions use honest tolerances.
The open-mesh cases prove the survey reports nothing rather than inventing it; the noise cases fix the
mesh by seed and pin what the survey reads on a scan — and that it says when it could not.
"""

import json
import os
import sys
import unittest
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402
import numpy as np  # noqa: E402

from oscar.plan import builder  # noqa: E402
from oscar.refparts import mesh_survey  # noqa: E402

V = App.Vector
TESS = 0.3          # mm: the coarse source-mesh deviation every tolerance below accounts for
CYL_TOL = 0.65      # mm: chord error of a cylinder at TESS plus the sew tolerance


def mesh_of(shape, deviation=TESS):
    verts, tris = shape.tessellate(deviation)
    return Mesh.Mesh([list(verts[i]) + list(verts[j]) + list(verts[k]) for i, j, k in tris])


def noisy_mesh(shape, sigma, seed, deviation=TESS):
    """``shape`` as ``mesh_of`` makes it, with Gaussian vertex noise of ``sigma`` mm drawn from
    a fixed seed: the same mesh on every run (and the same on FreeCAD 1.1.3 and 1.1.4)."""
    points, facets = mesh_of(shape, deviation).Topology
    P = np.array([[p.x, p.y, p.z] for p in points], dtype=float)
    P = P + sigma * np.random.default_rng(seed).standard_normal(P.shape)
    return Mesh.Mesh(([App.Vector(*p) for p in P], [tuple(int(i) for i in f) for f in facets]))


def plate_with_holes(length=100.0, width=60.0, thickness=8.0, hole_d=10.0,
                     centers=((30, 15), (-30, 15), (30, -15), (-30, -15))):
    base = Part.makeBox(length, width, thickness, V(-length / 2, -width / 2, 0))
    for x, y in centers:
        base = base.cut(Part.makeCylinder(hole_d / 2, thickness + 2, V(x, y, -1)))
    return base


def boss_board():
    """A thin board with four bosses standing on it and four small mounting holes: each
    hole's wall is well under 0.5 % of the surface area."""
    board = Part.makeBox(85, 56, 1.4)
    for x, y in ((20, 12), (65, 12), (20, 44), (65, 44)):
        board = board.fuse(Part.makeCylinder(4.0, 12.0, V(x, y, 1.4)))
    for i in range(4):
        board = board.cut(Part.makeCylinder(1.6, 10, V(15 + i * 18, 28, -1)))
    return board.removeSplitter()


class SurveyCase(unittest.TestCase):
    def setUp(self):
        self.doc = App.newDocument(f"MS{uuid.uuid4().hex[:6]}")
        self.addCleanup(App.closeDocument, self.doc.Name)

    def survey(self, shape, deviation=TESS):
        return mesh_survey.survey_mesh(mesh_of(shape, deviation))

    def cylinders(self, survey, *, concave=None):
        out = survey["cylinders"] if "cylinders" in survey else survey["surface"]["cylinders"]
        if concave is not None:
            out = [c for c in out if c["concave"] == concave]
        return out


class BoxSurveyTest(SurveyCase):
    def test_a_closed_box_mesh_surveys_into_three_plane_pairs_and_no_cylinders(self):
        s = self.survey(Part.makeBox(30, 20, 10))
        self.assertTrue(s["closed"])
        self.assertEqual(s["surface"]["face_count"], 6)
        planes = s["surface"]["planes"]
        normals = {tuple(round(abs(c), 2) for c in p["normal"]) for p in planes}
        self.assertEqual(normals, {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)})
        self.assertEqual(s["surface"]["cylinders"], [])
        by_normal = {tuple(round(abs(c), 2) for c in p["normal"]): p for p in planes}
        self.assertAlmostEqual(by_normal[(1.0, 0.0, 0.0)]["area"], 20 * 10, delta=1.0)
        self.assertAlmostEqual(by_normal[(0.0, 0.0, 1.0)]["area"], 30 * 20, delta=1.0)

    def test_the_bbox_volume_and_facet_facts_come_along(self):
        s = self.survey(Part.makeBox(30, 20, 10))
        self.assertEqual([round(v, 2) for v in s["bbox"]["size"]], [30.0, 20.0, 10.0])
        self.assertAlmostEqual(s["volume"], 6000.0, delta=15.0)
        self.assertGreater(s["facets"], 10)


class HoleSurveyTest(SurveyCase):
    def test_four_through_holes_are_found_concave_with_diameter_position_and_depth(self):
        s = self.survey(plate_with_holes())
        holes = self.cylinders(s, concave=True)
        self.assertEqual(len(holes), 4)
        for h in holes:
            self.assertAlmostEqual(h["diameter"], 10.0, delta=CYL_TOL)
            axis = [round(abs(c), 3) for c in h["axis"]]
            self.assertEqual(axis, [0.0, 0.0, 1.0])
            self.assertAlmostEqual(h["depth"], 8.0, delta=0.35)
        centers = sorted(tuple(round(c, 1) for c in h["center"]) for h in holes)
        self.assertEqual(centers, [(-30.0, -15.0, 4.0), (-30.0, 15.0, 4.0),
                                   (30.0, -15.0, 4.0), (30.0, 15.0, 4.0)])

    def test_a_smaller_hole_diameter_is_told_apart_from_a_larger_one(self):
        s = self.survey(plate_with_holes(hole_d=6.0, centers=((20, 10),)))
        holes = self.cylinders(s, concave=True)
        self.assertEqual(len(holes), 1)
        self.assertAlmostEqual(holes[0]["diameter"], 6.0, delta=CYL_TOL)

    def test_coarser_tessellation_stays_within_its_declared_chord_error(self):
        s = self.survey(plate_with_holes(), deviation=0.8)
        holes = self.cylinders(s, concave=True)
        self.assertEqual(len(holes), 4)
        for h in holes:
            self.assertAlmostEqual(h["diameter"], 10.0, delta=1.6)


class ConvexSurveyTest(SurveyCase):
    def test_a_solid_cylinder_is_convex_with_diameter_and_depth(self):
        s = self.survey(Part.makeCylinder(6.0, 20.0))
        pins = self.cylinders(s, concave=False)
        self.assertEqual(len(pins), 1)
        self.assertAlmostEqual(pins[0]["diameter"], 12.0, delta=CYL_TOL)
        self.assertAlmostEqual(pins[0]["depth"], 20.0, delta=0.35)
        self.assertFalse(pins[0]["concave"])

    def test_coaxial_steps_of_different_radius_are_separate_cylinders(self):
        shape = Part.makeCylinder(10.0, 40.0).fuse(Part.makeCylinder(6.0, 30.0, V(0, 0, 40)))
        shape = shape.removeSplitter()
        s = self.survey(shape)
        pins = sorted(self.cylinders(s, concave=False), key=lambda c: -c["diameter"])
        self.assertEqual([round(p["diameter"], 1) for p in pins], [20.0, 12.0])
        self.assertAlmostEqual(pins[0]["depth"], 40.0, delta=0.35)
        self.assertAlmostEqual(pins[1]["depth"], 30.0, delta=0.35)


class HonestyTest(SurveyCase):
    def test_an_open_mesh_reports_itself_and_withholds_what_it_cannot_know(self):
        mesh = mesh_of(Part.makeBox(30, 20, 10))
        facets = [f.Points for f in mesh.Facets]
        open_mesh = Mesh.Mesh([list(p[0]) + list(p[1]) + list(p[2]) for p in facets[:-2]])
        s = mesh_survey.survey_mesh(open_mesh)
        self.assertFalse(s["closed"])
        self.assertIsNone(s["volume"])
        self.assertTrue(any("not closed" in w for w in s["warnings"]))
        # the facets are still there: the box's planes are recognized, only the volume is withheld
        self.assertGreaterEqual(len(s["surface"]["planes"]), 3)
        self.assertTrue(all(c["concave"] is None for c in s["surface"]["cylinders"]))
        # the bbox facts are still exact: the mesh is measurable even when it is not solid
        self.assertEqual([round(v, 2) for v in s["bbox"]["size"]], [30.0, 20.0, 10.0])

    def test_a_mesh_with_no_triangles_is_rejected_without_raising(self):
        s = mesh_survey.survey_mesh(Mesh.Mesh())
        self.assertFalse(s["closed"])
        self.assertEqual(s["surface"]["cylinders"], [])

    def test_the_survey_of_a_shape_with_many_faces_stays_bounded(self):
        sphere = Part.makeSphere(12.0)
        s = self.survey(sphere, deviation=1.2)
        self.assertTrue(s["closed"])
        kinds = s["surface"]["surface_kinds"]
        self.assertGreater(kinds.get("plane", 0) + kinds.get("sphere", 0), 0)
        self.assertLessEqual(len(s["surface"]["planes"]), 16)


class SolidSurveyTest(SurveyCase):
    def test_survey_solid_works_on_a_b_rep_shape_directly(self):
        s = mesh_survey.survey_solid(plate_with_holes())
        holes = [c for c in s["cylinders"] if c["concave"]]
        self.assertEqual(len(holes), 4)
        for h in holes:
            self.assertAlmostEqual(h["diameter"], 10.0, delta=1e-6)

    def test_an_invalid_shape_reports_itself_without_raising(self):
        s = mesh_survey.survey_solid(Part.Shape())
        self.assertFalse(s["valid"])
        self.assertEqual(s["cylinders"], [])

    def test_outer_cylinders_away_from_the_origin_are_convex(self):
        # the concavity probe is taken at the face's mid-height measured from the cylinder's
        # own axis point. Added to it as an absolute height, it landed outside a stepped
        # shaft and its two upper steps were reported as holes
        shaft = (Part.makeCylinder(10.0, 40.0)
                 .fuse(Part.makeCylinder(6.0, 30.0, V(0, 0, 40)))
                 .fuse(Part.makeCylinder(4.0, 20.0, V(0, 0, 70)))).removeSplitter()
        s = mesh_survey.survey_solid(shaft)
        self.assertEqual(sorted((round(c["diameter"], 1), c["concave"]) for c in s["cylinders"]),
                         [(8.0, False), (12.0, False), (20.0, False)])
        # and a hole far from the origin is still a hole
        plate = Part.makeBox(40, 30, 5, V(100, 100, 50)).cut(
            Part.makeCylinder(3.0, 30, V(120, 115, 40)))
        holes = [c for c in mesh_survey.survey_solid(plate)["cylinders"] if c["concave"]]
        self.assertEqual([round(h["diameter"], 1) for h in holes], [6.0])


class NoiseAndSmallFeatureTest(SurveyCase):
    """What the survey measures on a scan (fixed seeds) and what it says when it cannot."""

    def test_small_holes_on_a_large_plate_are_reported(self):
        # each mounting hole's wall is a sliver of the area: dropped as noise before it was
        # classified, a clean board reported none of its four holes
        s = self.survey(boss_board())
        holes = self.cylinders(s, concave=True)
        pins = self.cylinders(s, concave=False)
        self.assertEqual(len(holes), 4)
        self.assertEqual(len(pins), 4)
        for h in holes:
            self.assertAlmostEqual(h["diameter"], 3.2, delta=CYL_TOL)
        for p in pins:
            self.assertAlmostEqual(p["diameter"], 8.0, delta=CYL_TOL)
        self.assertEqual(s["surface"]["unsurveyed_area_pct"], 0.0)
        self.assertNotIn("not surveyed", mesh_survey.render_text(s))

    def test_holes_survive_a_little_noise(self):
        # 0.01 mm of vertex noise: all four holes are still read, and nothing is invented
        s = mesh_survey.survey_mesh(noisy_mesh(plate_with_holes(), 0.01, 22))
        holes = self.cylinders(s, concave=True)
        self.assertEqual(len(holes), 4)
        for h in holes:
            self.assertAlmostEqual(h["diameter"], 10.0, delta=CYL_TOL)
        self.assertEqual(len(self.cylinders(s)), 4)

    def test_a_noisy_face_is_never_reported_as_a_giant_cylinder(self):
        # the committed plate plan, tessellated like a scan and given 0.02 mm of noise: a
        # face that stopped being flat "fitted" a circle 32 427 mm across with a tiny
        # residual, and it was listed as a cylinder of that size on a 100 mm plate
        doc = App.newDocument("MSfixture")
        self.addCleanup(App.closeDocument, doc.Name)
        with open(os.path.join(os.path.dirname(__file__), "fixtures", "plans", "plate_4holes.json"),
                  encoding="utf-8") as fh:
            report = builder.apply_plan(doc, json.load(fh))
        self.assertTrue(report["success"], report["error"])
        s = mesh_survey.survey_mesh(noisy_mesh(doc.getObject(report["body"]).Shape, 0.02, 22))
        diagonal = sum(v * v for v in s["bbox"]["size"]) ** 0.5
        for c in self.cylinders(s):
            self.assertLessEqual(c["diameter"], diagonal, c)

    def test_a_degraded_survey_says_how_much_it_did_not_read(self):
        # 0.02 mm: the face merges with three of its four hole walls. The survey must not
        # let "one hole listed" read as "the part has one hole"
        s = mesh_survey.survey_mesh(noisy_mesh(plate_with_holes(), 0.02, 11))
        self.assertLess(len(self.cylinders(s, concave=True)), 4)
        self.assertGreater(s["surface"]["unsurveyed_area_pct"], 20.0)
        text = mesh_survey.render_text(s)
        self.assertIn("not surveyed", text)
        self.assertIn("not evidence that the part has none", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
