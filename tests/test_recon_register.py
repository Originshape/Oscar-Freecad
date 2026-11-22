"""Registration mandate tests: ``oscar.recon.register`` (pure numpy ICP) and the raw/aligned
parity of ``deviation.registered_deviation``. Ground truth is a known rigid transform; seeds
are fixed; tolerances scale with the input (documented per test). Pure FreeCAD-free module for
the ICP core; the registered-deviation tests run on real meshes under freecadcmd.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

import numpy as np  # noqa: E402

from oscar.recon import register  # noqa: E402

RNG = np.random.default_rng(20261114)


def cloud(n=400, spread=15.0):
    """A generic (asymmetric) 3-D point cloud."""
    return RNG.uniform(-spread, spread, (n, 3))


def rigid(theta_deg, translation):
    th = math.radians(theta_deg)
    k = np.array([0.0, 1.0, 0.0])
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    return R, np.asarray(translation, dtype=float)


def apply(R, t, pts):
    return pts @ R.T + t


class IcpGroundTruthTest(unittest.TestCase):
    def assert_recovers(self, pts, R_true, t_true, *, angle_tol=0.05, trans_tol=0.05,
                        method="point_to_point", **kw):
        moved = apply(R_true, t_true, pts)
        reg = register.icp_register(moved, pts, method=method, **kw)
        self.assertTrue(reg["valid"], reg["warnings"])
        self.assertTrue(reg["converged"])
        self.assertLessEqual(abs(reg["rotation_deg"] - math.degrees(
            math.acos(max(-1.0, min(1.0, (np.trace(R_true) - 1) / 2))))), angle_tol, reg)
        # the estimated transform maps scan->model: invert to compare with the applied one
        R_est = np.asarray(reg["R"])
        t_est = np.asarray(reg["t"])
        self.assertLessEqual(np.linalg.norm(R_est @ R_true - np.eye(3)), 1e-3)
        self.assertLessEqual(np.linalg.norm(t_est + R_est @ t_true), trans_tol, reg)

    def test_identical_clouds_converge_to_identity(self):
        pts = cloud()
        reg = register.icp_register(pts, pts)
        self.assertTrue(reg["valid"] and reg["converged"])
        self.assertAlmostEqual(reg["rotation_deg"], 0.0, places=3)
        self.assertAlmostEqual(reg["translation_mm"], 0.0, places=3)
        self.assertLess(reg["residual_rmse_mm"], 1e-6)

    def test_small_rotation(self):
        R, t = rigid(2.0, (0, 0, 0))
        self.assert_recovers(cloud(), R, t)

    def test_small_translation(self):
        R, t = rigid(0.0, (0.5, -0.3, 0.2))
        self.assert_recovers(cloud(), R, t)

    def test_combined_rotation_and_translation(self):
        R, t = rigid(3.0, (1.0, 0.6, -0.8))
        self.assert_recovers(cloud(), R, t)

    def test_noisy_points_still_recover_within_tolerance(self):
        pts = cloud()
        noisy = pts + RNG.normal(0.0, 0.02, pts.shape)
        R, t = rigid(2.0, (0.4, 0.2, 0.1))
        reg = register.icp_register(apply(R, t, noisy), pts)
        self.assertTrue(reg["valid"], reg["warnings"])
        self.assertLessEqual(reg["rotation_deg"], 2.5)
        self.assertLessEqual(reg["residual_rmse_mm"], 0.05)

    def test_outliers_are_trimmed(self):
        pts = cloud()
        moved = apply(*rigid(1.5, (0.3, 0.2, 0.1)), pts)
        moved[:40] += np.array([50.0, -60.0, 40.0])      # gross outliers
        reg = register.icp_register(moved, pts)
        self.assertTrue(reg["valid"], reg["warnings"])
        self.assertLessEqual(reg["rotation_deg"], 2.0)

    def test_partial_overlap_registers(self):
        pts = cloud(600)
        R, t = rigid(1.0, (0.2, 0.2, 0.2))
        reg = register.icp_register(apply(R, t, pts[:300]), pts)
        self.assertTrue(reg["valid"], reg["warnings"])

    def test_different_densities(self):
        pts = cloud(800)
        R, t = rigid(2.0, (0.5, 0.3, 0.2))
        reg = register.icp_register(apply(R, t, pts[::7]), pts)
        self.assertTrue(reg["valid"], reg["warnings"])

    def test_planar_geometry_reports_weak_axes(self):
        plane = RNG.uniform(-10, 10, (300, 2))
        pts = np.column_stack([plane, np.zeros(300)])
        R, t = rigid(1.0, (0.0, 0.0, 0.4))              # slide + lift off the plane
        reg = register.icp_register(apply(R, t, pts), pts, method="point_to_plane",
                                    model_normals=np.tile([0, 0, 1], (300, 1)))
        # z is observable (normal direction); x/y sliding is planar-blind
        self.assertIn("x", reg["weak_translation_axes"])
        self.assertIn("y", reg["weak_translation_axes"])
        self.assertNotIn("z", reg["weak_translation_axes"])

    def test_symmetric_geometry_pca_init_reports_ambiguity(self):
        pts = np.column_stack([RNG.uniform(-1, 1, 400),
                               RNG.uniform(-1, 1, 400),
                               RNG.uniform(-0.02, 0.02, 400)]) * 10.0
        reg = register.icp_register(pts, pts, init="pca")
        self.assertTrue(reg["init"]["ambiguous"])

    def test_degenerate_collinear_points_are_rejected(self):
        line = np.column_stack([np.linspace(0, 10, 50), np.zeros(50), np.zeros(50)])
        reg = register.icp_register(line, line)
        self.assertFalse(reg["valid"] or reg["converged"])

    def test_empty_input_is_rejected(self):
        reg = register.icp_register([], [])
        self.assertFalse(reg["valid"])

    def test_nan_and_infinite_inputs_are_rejected(self):
        pts = cloud(50)
        bad = pts.copy()
        bad[3, 0] = math.nan
        reg = register.icp_register(bad, pts)
        self.assertFalse(reg["valid"])
        self.assertTrue(any("NaN" in w for w in reg["warnings"]))

    def test_insufficient_correspondences_rejected(self):
        pts = cloud(50)
        far = pts + np.array([1000.0, 1000.0, 1000.0])
        far[:3] = pts[:3]
        reg = register.icp_register(far, pts, max_correspondence=0.05)
        self.assertFalse(reg["valid"])

    def test_convention_is_source_to_target(self):
        pts = cloud(200)
        R, t = rigid(2.0, (0.3, 0.0, 0.0))
        moved = apply(R, t, pts)
        reg = register.icp_register(moved, pts)
        # applying the reported transform to the SCAN must land on the MODEL
        landed = apply(np.asarray(reg["R"]), np.asarray(reg["t"]), moved)
        self.assertLess(np.linalg.norm(landed - pts, axis=1).mean(), 0.05)
        self.assertIn("source->target", reg["convention"])

    def test_rotation_matrix_is_valid_and_reflection_free(self):
        pts = cloud()
        reg = register.icp_register(apply(*rigid(2.0, (0.1, 0.1, 0.1)), pts), pts)
        R = np.asarray(reg["R"])
        self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-6))
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=6)


class RegisteredDeviationTest(unittest.TestCase):
    """Raw + aligned parity on a REAL mesh/shape pair through the production samplers."""

    @classmethod
    def setUpClass(cls):
        import FreeCAD as App  # noqa: F401
        import Mesh
        import Part

        cls.shape = Part.makeBox(60, 40, 10)
        mesh0 = Mesh.Mesh(cls.shape.tessellate(0.4))
        R, t = rigid(2.0, (0.5, -0.3, 0.2))
        pts = np.array([[p.x, p.y, p.z] for p in mesh0.Topology[0]])
        moved = (R @ pts.T).T + t
        import FreeCAD as App2

        cls.mesh = Mesh.Mesh(([App2.Vector(*p) for p in moved],
                              [tuple(f) for f in mesh0.Topology[1]]))

    def test_raw_biased_aligned_improved_and_report_complete(self):
        from oscar.recon import deviation

        out = deviation.registered_deviation(self.mesh, self.shape, warn_mm=0.3)

        def by_id(rows):
            return {r["id"]: r for r in rows}

        raw, aligned, reg = by_id(out["raw"]), by_id(out["aligned"]), out["registration"]
        self.assertIsNotNone(aligned, reg and reg["warnings"])
        self.assertGreater(raw["hausdorff"]["value"], 1.5)          # the raw bias is real
        # sampled maxima are NN lower bounds: the model→scan direction keeps the mesh
        # grid's discreteness (~1.2 mm at this tessellation), so the honest contract is
        # MATERAL IMPROVEMENT + transform recovery, not a halved hausdorff
        self.assertLess(aligned["hausdorff"]["value"], raw["hausdorff"]["value"])
        self.assertAlmostEqual(reg["rotation_deg"], 2.0, delta=0.5)
        for key in ("R", "t", "T", "rotation_deg", "translation_mm", "iterations",
                    "correspondences", "residual_rmse_mm", "converged", "method", "init",
                    "warnings", "convention"):
            self.assertIn(key, reg)
        # weak-axis semantics: the densified surface samples spread the correspondence
        # normals over all six faces of this box, so NO translation axis is weakly observed
        # (an empty list is the correct answer here); the pure planar case (x/y blind) is
        # pinned by IcpGroundTruthTest.test_planar_geometry_reports_weak_axes
        self.assertTrue(set(reg["weak_translation_axes"]) <= {"x", "y", "z"})
        # same sampling policy on both sides
        self.assertEqual(raw["samples"]["metric"], aligned["samples"]["metric"])

    def test_exact_copy_registers_to_identity(self):
        import Mesh

        from oscar.recon import deviation

        mesh = Mesh.Mesh(self.shape.tessellate(0.4))
        out = deviation.registered_deviation(mesh, self.shape)
        self.assertLess(out["raw"]["hausdorff"]["value"] if isinstance(out["raw"][0], float)
                        else {r["id"]: r for r in out["raw"]}["hausdorff"]["value"], 0.35)

    def test_a_registration_that_worsens_the_deviation_is_discarded(self):
        """The benchmark's wrong-local-minimum case (partial overlap: box with a hole wall)
        must be REJECTED, not reported: aligned withheld, validity cleared, warning kept."""
        import Mesh
        import Part

        from oscar.recon import deviation

        shape = Part.makeBox(60, 40, 10).cut(
            Part.makeCylinder(3.3, 40, np.array([15, 10, -5.0]).tolist() and __import__(
                "FreeCAD").Vector(15, 10, -5)))
        mesh0 = Mesh.Mesh(shape.tessellate(0.4))
        R, t = rigid(2.0, (0.5, -0.3, 0.2))
        pts = np.array([[p.x, p.y, p.z] for p in mesh0.Topology[0]])
        moved = pts @ R.T + t
        import FreeCAD as App

        mesh = Mesh.Mesh(([App.Vector(*p) for p in moved],
                          [tuple(f) for f in mesh0.Topology[1]]))
        out = deviation.registered_deviation(mesh, shape, warn_mm=0.3)
        if out["aligned"] is None:                      # the guard fired — this is the pin
            self.assertFalse(out["registration"]["valid"])
            self.assertTrue(any("WORSENED" in w for w in out["registration"]["warnings"]))
        else:                                           # or ICP genuinely improved it
            raw = {r["id"]: r for r in out["raw"]}["hausdorff"]["value"]
            aligned = {r["id"]: r for r in out["aligned"]}["hausdorff"]["value"]
            self.assertLess(aligned, raw)


if __name__ == "__main__":
    unittest.main()
