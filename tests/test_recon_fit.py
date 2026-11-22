"""G2: ``oscar.recon.fit`` — seeded geometric fitting for sphere / cone / torus plus the
LM cylinder polish, and their gated integration into ``recognize.survey``.

Fixtures are REAL tessellated FreeCAD shapes (exact recovery on clean tessellations,
parameter recovery under noise, the classic Kasa-vs-geometric rim-bias A/B, false-cone
rejection on cylinders, and survey-level recognition of cone/sphere/torus regions). The
Kasa-bias experiment asserts the textbook result: the algebraic fit systematically
under-estimates radius on noisy partial arcs while the geometric refinement does not.
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

from oscar.recon import fit, recognize  # noqa: E402


def region_points(shape, tess, *, noise=0.0, seed=7, drop_vertical=True,
                  min_radius=2.0, centroid_samples=False):
    """Facet-corner (or centroid) points + normals of the CURVED walls of a shape
    (planar caps and sector sides dropped — regions in production are single-surface)."""
    p, f = shape.tessellate(tess)
    mesh = Mesh.Mesh((list(p), [tuple(t) for t in f]))
    P, N = [], []
    for t, fc in zip(f, mesh.Facets):
        n = fc.Normal
        if drop_vertical and abs(n.z) > 0.9:
            continue
        if centroid_samples:
            c = np.array([[p[i].x, p[i].y, p[i].z] for i in t]).mean(axis=0)
            if np.hypot(c[0], c[1]) < min_radius:
                continue
            P.append(c)
            N.append([n.x, n.y, n.z])
        else:
            corners = np.array([[p[i].x, p[i].y, p[i].z] for i in t])
            if np.hypot(corners[:, 0], corners[:, 1]).min() < min_radius:
                continue
            for q in corners:
                P.append([q[0], q[1], q[2]])
            for _ in t:
                N.append([n.x, n.y, n.z])
    P = np.asarray(P)
    if noise:
        P = P + noise * np.random.default_rng(seed).standard_normal(P.shape)
    return P, np.asarray(N)


class SphereFitTest(unittest.TestCase):
    def test_clean_cap_recovers_radius_exactly(self):
        sph = Part.makeSphere(8, App.Vector(0, 0, 0), App.Vector(0, 0, 1), 75)
        P, N = region_points(sph, 0.4, drop_vertical=False, min_radius=0.0)
        r = fit.fit_sphere(P)
        self.assertAlmostEqual(r["radius"], 8.0, places=3)
        self.assertAlmostEqual(r["rms"], 0.0, places=5)

    def test_noisy_cap_recovers_radius_within_band(self):
        sph = Part.makeSphere(8, App.Vector(0, 0, 0), App.Vector(0, 0, 1), 75)
        P, N = region_points(sph, 0.4, noise=0.02, drop_vertical=False, min_radius=0.0)
        r = fit.fit_sphere(P)
        # a 75° CAP sees only ~40% of the sphere: the recovered radius carries sampling
        # uncertainty well beyond the noise level — assert the honest band (G2 report
        # notes caps as a limitation; full spheres recover exactly)
        self.assertAlmostEqual(r["radius"], 8.0, delta=0.15)
        self.assertLess(r["rms"], 0.03)

    def test_degenerate_inputs_return_none(self):
        line = np.column_stack([np.linspace(0, 5, 10), np.zeros(10), np.zeros(10)])
        self.assertIsNone(fit.fit_sphere(line))       # collinear: no sphere
        tiny = np.ones((8, 3)) * 0.5                  # coincident points
        self.assertIsNone(fit.fit_sphere(tiny))


class ConeFitTest(unittest.TestCase):
    def test_clean_frustum_recovers_half_angle(self):
        cone = Part.makeCone(4, 9, 12)                    # tan(beta) = 5/12
        P, N = region_points(cone, 0.4)
        c = fit.fit_cone(P, N)
        self.assertAlmostEqual(np.degrees(c["half_angle"]),
                               np.degrees(np.arctan(5.0 / 12.0)), places=2)
        self.assertAlmostEqual(c["radius_min"], 4.0, places=2)
        self.assertAlmostEqual(c["radius_max"], 9.0, places=2)

    def test_noisy_frustum_stays_within_band(self):
        cone = Part.makeCone(4, 9, 12)
        P, N = region_points(cone, 0.4, noise=0.01)
        c = fit.fit_cone(P, N)
        self.assertAlmostEqual(np.degrees(c["half_angle"]),
                               np.degrees(np.arctan(5.0 / 12.0)), delta=0.3)
        self.assertLess(c["rms"], 0.02)


class TorusFitTest(unittest.TestCase):
    def test_clean_torus_recovers_both_radii(self):
        tor = Part.makeTorus(10, 2)
        P, N = region_points(tor, 0.4, min_radius=0.0)
        t = fit.fit_torus(P, N)
        self.assertAlmostEqual(t["major_radius"], 10.0, places=2)
        self.assertAlmostEqual(t["minor_radius"], 2.0, places=2)

    def test_degenerate_tube_rejected(self):
        cyl = Part.makeCylinder(5, 20)                    # a tube of zero major radius
        P, N = region_points(cyl, 0.4)
        self.assertIsNone(fit.fit_torus(P, N))


class CylinderPolishTest(unittest.TestCase):
    def test_full_wall_exact(self):
        cyl = Part.makeCylinder(5, 20)
        P, N = region_points(cyl, 0.4)
        c = fit.fit_cylinder(P, N)
        self.assertAlmostEqual(c["radius"], 5.0, places=3)
        self.assertAlmostEqual(c["rms"], 0.0, places=5)

    def test_kasa_rim_bias_is_removed_by_the_geometric_refinement(self):
        # the textbook experiment: noisy partial arcs — the ALGEBRAIC (Kasa) fit
        # systematically under-estimates the radius; the geometric LM fit does not
        rng = np.random.default_rng(42)
        r_true, noise = 6.0, 0.10
        errs_k, errs_l = [], []
        for trial in range(20):
            th = np.radians(np.linspace(-60, 60, 200)) + rng.uniform(-1, 1, 200) * np.radians(0.2)
            rr = r_true + noise * rng.standard_normal(200)
            P = np.column_stack([rr * np.cos(th), rr * np.sin(th), 10 * rng.random(200)])
            N = np.column_stack([np.cos(th), np.sin(th), np.zeros(200)])
            errs_k.append(fit.fit_cylinder(P, N, refine=False)["radius"] - r_true)
            errs_l.append(fit.fit_cylinder(P, N, refine=True)["radius"] - r_true)
        self.assertGreater(np.mean(errs_l), np.mean(errs_k),
                           "the geometric fit must sit above Kasa's systematic "
                           "underestimate")
        self.assertLess(abs(np.mean(errs_l)), abs(np.mean(errs_k)),
                        "the geometric fit must be measurably closer to truth")

    def test_cap_normals_are_refused(self):
        box = Part.makeBox(10, 10, 10)
        P, N = region_points(box, 2.0, drop_vertical=False, min_radius=0.0)
        self.assertIn(None, (fit.fit_cylinder(P, N),))   # planar normals: not a cylinder


class ClassifyGateTest(unittest.TestCase):
    def test_false_cone_rejected_on_cylinder_walls(self):
        # a cone FITTED to cylinder points must be rejected by the significance gate
        # (classify itself short-circuits at the cylinder — the gate guards the case
        # where a cylinder fit is poor and a cone is a live candidate)
        cyl = Part.makeCylinder(5, 20)
        P, N = region_points(cyl, 0.4)
        got = fit.classify(P, N, tol=0.35, sigma=0.01)
        self.assertEqual(got["kind"], "cylinder")
        cone = fit.fit_cone(P, N)
        cylinder = fit.fit_cylinder(P, N)
        plane = fit.fit_plane(P)
        ok, why = fit.cone_significance(P, cone,
                                        {"rms": cylinder["rms"]},
                                        {"rms": plane[2]}, 0.01)
        self.assertFalse(ok)
        self.assertTrue(why, "the gate must state why the cone was rejected")

    def test_cone_accepted_when_genuinely_conical(self):
        cone = Part.makeCone(4, 9, 12)
        P, N = region_points(cone, 0.4)
        got = fit.classify(P, N, tol=0.35, sigma=0.01)
        self.assertEqual(got["kind"], "cone")
        self.assertEqual(got["report"].get("cone_gate"), "accepted")

    def test_torus_accepted_when_genuinely_toroidal(self):
        tor = Part.makeTorus(10, 2)
        P, N = region_points(tor, 0.4, min_radius=0.0)
        got = fit.classify(P, N, tol=0.35, sigma=0.01)
        self.assertEqual(got["kind"], "torus")

    def test_sphere_accepted_when_genuinely_spherical(self):
        sph = Part.makeSphere(8)
        P, N = region_points(sph, 0.4, drop_vertical=False, min_radius=0.0)
        got = fit.classify(P, N, tol=0.35, sigma=0.01)
        self.assertEqual(got["kind"], "sphere")


class SurveyIntegrationTest(unittest.TestCase):
    def test_cone_solid_surveys_as_cone_regions(self):
        mesh = Mesh.Mesh(Part.makeCone(4, 9, 12).tessellate(0.4))
        s = recognize.survey(mesh)
        cones = [r for r in s["regions"] if r["type"] == "cone"]
        self.assertTrue(cones, "the conical wall must be recognized")
        self.assertAlmostEqual(cones[0]["half_angle_deg"],
                               np.degrees(np.arctan(5.0 / 12.0)), delta=0.3)

    def test_sphere_surveys_as_sphere_regions(self):
        mesh = Mesh.Mesh(Part.makeSphere(8).tessellate(0.4))
        s = recognize.survey(mesh)
        share = sum(r["area_mm2"] for r in s["regions"] if r["type"] == "sphere") \
            / float(mesh.Area)
        self.assertGreaterEqual(share, 0.5,
                                "a sphere is now mostly RECOGNIZED (G2 spec evolution)")
        self.assertLessEqual(s["coverage"]["unrecognized"], 0.1)

    def test_torus_wall_surveys_as_torus_region(self):
        mesh = Mesh.Mesh(Part.makeTorus(10, 2).tessellate(0.4))
        s = recognize.survey(mesh)
        tori = [r for r in s["regions"] if r["type"] == "torus"]
        self.assertTrue(tori)
        self.assertAlmostEqual(tori[0]["major_radius"], 10.0, delta=0.2)

    def test_plate_with_holes_still_surveys_as_prismatic(self):
        shape = Part.makeBox(60, 40, 10)
        for x, y in ((15, 10), (45, 10), (15, 30), (45, 30)):
            shape = shape.cut(Part.makeCylinder(3.3, 30, App.Vector(x, y, -10)))
        mesh = Mesh.Mesh(shape.tessellate(0.3))
        s = recognize.survey(mesh)
        self.assertEqual(s["classification"]["class"], "prismatic")
        self.assertEqual(len(s["holes"]), 4)


if __name__ == "__main__":
    unittest.main()
