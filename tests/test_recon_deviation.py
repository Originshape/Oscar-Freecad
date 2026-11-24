"""M6: ``oscar.recon.deviation`` — bidirectional sampled deviation rows between the source
mesh and the reconstructed shape. Checked against analytic truth: identical geometry gives
zeros, a known offset gives exactly that offset, and the numpy fallback agrees with cKDTree.
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

from oscar.recon import deviation  # noqa: E402


def rows_by_id(rows):
    return {r["id"]: r for r in rows}


class DeviationTest(unittest.TestCase):
    def test_identical_geometry_is_all_pass_zeros(self):
        shape = Part.makeBox(30, 20, 8)
        mesh = Mesh.Mesh(shape.tessellate(0.3))
        rows = rows_by_id(deviation.deviation_rows(mesh, shape))
        self.assertEqual(rows["samples"]["status"], "PASS")
        for rid in ("scan_to_model.max", "model_to_scan.max", "hausdorff",
                    "scan_to_model.rms", "model_to_scan.p95"):
            self.assertEqual(rows[rid]["status"], "PASS", rows[rid])
            self.assertAlmostEqual(rows[rid]["value"], 0.0, delta=0.05, msg=rid)

    def test_a_known_offset_measures_exactly(self):
        shape = Part.makeBox(30, 20, 8)
        mesh = Mesh.Mesh(Part.makeBox(30, 20, 8, App.Vector(0, 0, 0.5)).tessellate(0.3))
        rows = rows_by_id(deviation.deviation_rows(mesh, shape, warn_mm=0.2))
        self.assertAlmostEqual(rows["hausdorff"]["value"], 0.5, delta=0.05)
        self.assertEqual(rows["hausdorff"]["status"], "WARN")
        self.assertAlmostEqual(rows["model_to_scan.max"]["value"], 0.5, delta=0.05)
        self.assertLessEqual(rows["scan_to_model.rms"]["value"], 0.5)

    def test_the_numpy_fallback_agrees_with_kdtree(self):
        shape = Part.makeCylinder(12, 30)
        mesh = Mesh.Mesh(shape.tessellate(0.3))
        original = deviation._distances
        orig_exact = deviation._exact_raw_distances
        # the exact primary engines must step aside: this test exercises the
        # SAMPLED path and its numpy fallback on BOTH sides of the comparison
        deviation._exact_raw_distances = (
            lambda *a, **k: (None, "forced-sampled", None, "forced-sampled"))

        def forced_numpy(points, targets):
            import numpy as np

            src = np.asarray(points, dtype=float)
            dst = np.asarray(targets, dtype=float)
            out = []
            for start in range(0, len(src), 256):
                block = src[start:start + 256]
                out.extend(float(v) for v in deviation.numpy_linalg_norm(block, dst))
            return out, "numpy-forced"

        try:
            kd = rows_by_id(deviation.deviation_rows(mesh, shape))["hausdorff"]["value"]
            deviation._distances = forced_numpy
            np_rows = rows_by_id(deviation.deviation_rows(mesh, shape))
        finally:
            deviation._distances = original
            deviation._exact_raw_distances = orig_exact
        self.assertLessEqual(abs(np_rows["hausdorff"]["value"] - kd), 0.05)
        self.assertIn("numpy-forced", np_rows["samples"]["detail"])


class MeasuredMaskTest(unittest.TestCase):
    """G6: the measured-only mask — deviation covers MEASURED regions only. Scan vertices
    on synthetic (repair-created) facets are excluded, model samples inside recorded
    missing regions are excluded, the exclusions are REPORTED (a ``measured_only`` row,
    never silent), and ``mask=None`` keeps the historic behaviour exactly."""

    def test_scan_side_exclusion_removes_the_outlier(self):
        import numpy as np

        shape = Part.makeBox(10, 10, 0.01)
        pts = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [5, 5, 30.0]],
                       dtype=float)
        mesh = Mesh.Mesh(([App.Vector(*p) for p in pts],
                          [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]))
        raw = rows_by_id(deviation.deviation_rows(mesh, shape, warn_mm=0.3))
        masked = rows_by_id(deviation.deviation_rows(
            mesh, shape, warn_mm=0.3, mask={"scan_exclude_indices": [4]}))
        self.assertGreater(raw["scan_to_model.max"]["value"], 5.0)
        self.assertAlmostEqual(masked["scan_to_model.max"]["value"], 0.0, delta=0.05)
        self.assertIn("measured_only", masked)
        self.assertEqual(masked["measured_only"]["status"], "PASS")
        self.assertIn("synthetic fill geometry is NOT measurement",
                      masked["measured_only"]["detail"])

    def test_model_region_exclusion_drops_the_unmeasured_pad(self):
        import numpy as np

        # a dense grid-box scan (all faces measured at a 2 mm sampling); the model adds a
        # pad the scan never saw over part of the top face — the ring masks that footprint
        n = 11
        xs = np.linspace(0, 20, n)

        def sheet(z):
            return np.array([[x, y, z] for y in xs for x in xs])

        pts = np.vstack([sheet(0.0), sheet(8.0)])
        N = n * n
        facs = []
        for i in range(n - 1):
            for j in range(n - 1):
                a, b, c, e = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
                facs += [(a, c, b), (b, c, e)]                        # bottom, -z
                facs += [(N + a, N + b, N + c), (N + b, N + e, N + c)]  # top, +z
        for i in range(n - 1):
            for lo_a, hi_a, lo_b, hi_b in ((i * n, N + i * n, (i + 1) * n, N + (i + 1) * n),
                                           (i * n + n - 1, N + i * n + n - 1,
                                            (i + 1) * n + n - 1, N + (i + 1) * n + n - 1)):
                facs += [(lo_a, hi_a, hi_b), (lo_a, hi_b, lo_b)]
        for j in range(n - 1):
            for lo_a, hi_a, lo_b, hi_b in ((j, N + j, j + 1, N + j + 1),
                                           ((n - 1) * n + j, N + (n - 1) * n + j,
                                            (n - 1) * n + j + 1, N + (n - 1) * n + j + 1)):
                facs += [(lo_a, hi_a, hi_b), (lo_a, hi_b, lo_b)]
        scan_mesh = Mesh.Mesh(([App.Vector(*p) for p in pts], facs))
        model = Part.makeBox(20, 20, 8).fuse(Part.makeBox(6, 6, 2, App.Vector(7, 7, 8)))
        # the ring sits slightly OUTSIDE the pad: the model's planar pad-top face samples
        # are exactly its four corner vertices, and boundary-exact points are ambiguous
        # under ray casting — strictly-inside is the honest fixture
        ring = np.array([[6.8, 6.8, 10.0], [6.8, 13.2, 10.0], [13.2, 13.2, 10.0],
                         [13.2, 6.8, 10.0]])
        # tol covers the pad's full height: the model-side exact pass samples
        # the pad WALLS too (edge discretization), not only its top vertices
        mask = {"scan_exclude_indices": [], "model_regions": [
            {"points": ring, "normal": [0, 0, 1.0], "tol_mm": 2.3}],
            "note": "pad footprint unmeasured"}
        raw = rows_by_id(deviation.deviation_rows(scan_mesh, model, warn_mm=0.3))
        masked = rows_by_id(deviation.deviation_rows(scan_mesh, model, warn_mm=0.3,
                                                     mask=mask))
        self.assertGreater(raw["model_to_scan.max"]["value"], 1.8,
                       "the 2 mm pad must dominate the unmasked model_to_scan max")
        self.assertLess(masked["model_to_scan.max"]["value"],
                        raw["model_to_scan.max"]["value"] - 0.4,
                        "the masked rows must not count the unmeasured pad")
        self.assertGreater(masked["measured_only"]["value"], 0)
        self.assertEqual(masked["measured_only"]["status"], "PASS")

    def test_registered_deviation_returns_the_mask_summary(self):
        mesh = Mesh.Mesh(Part.makeBox(30, 20, 8).tessellate(0.3))
        shape = Part.makeBox(30, 20, 8)
        got = deviation.registered_deviation(
            mesh, shape, warn_mm=0.3, mask={"scan_exclude_indices": [0, 1]})
        self.assertEqual(got["mask"]["scan_excluded"], 2)
        self.assertIn("measured_only", rows_by_id(got["raw"]))

    def test_masking_everything_reports_unavailable(self):
        mesh = Mesh.Mesh(Part.makeBox(10, 10, 4).tessellate(1.0))
        shape = Part.makeBox(10, 10, 4)
        rows = rows_by_id(deviation.deviation_rows(
            mesh, shape, mask={"scan_exclude_indices": range(int(mesh.CountPoints))}))
        self.assertEqual(rows["deviation"]["status"], "SKIP",
                         "an empty measured set must not pretend to measure anything")

    def test_no_mask_keeps_the_historic_rows_exactly(self):
        mesh = Mesh.Mesh(Part.makeBox(30, 20, 8).tessellate(0.3))
        shape = Part.makeBox(30, 20, 8)
        rows = deviation.deviation_rows(mesh, shape)
        self.assertNotIn("measured_only", rows_by_id(rows))
        self.assertIn("hausdorff", rows_by_id(rows))


class ExactResidualsTest(unittest.TestCase):
    """V-1: exact worst-point refinement (distToShape) + signed missing/excess clusters.

    Exactness is pinned against analytic truth (a 0.5 mm gap measures 0.500); signs are
    pinned with uniform cylinder fixtures (the model lacks the scan's top → missing; the
    model is taller than the scan → excess); identical geometry yields no clusters.
    Cluster AREAS are per-point-share ESTIMATES — FreeCAD tessellations concentrate
    samples in caps/fans, so the test pins order-of-magnitude bounds, not ±20% (that
    OriginShape acceptance presumes near-uniform scan sampling; the caveat travels with
    every model-side record)."""

    @staticmethod
    def rows_of(got):
        return {r["id"]: r for r in got["rows"]}

    def test_a_half_millimetre_gap_measures_exactly(self):
        pts = [[0, 0, 0], [30, 0, 0], [30, 20, 0], [0, 20, 0], [15, 10, 8.5]]
        mesh = Mesh.Mesh(([App.Vector(*p) for p in pts],
                          [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]))
        box = Part.makeBox(30, 20, 8)
        got = deviation.exact_residuals(mesh, box, warn_mm=0.3)
        row = self.rows_of(got)["exact_scan_to_model.max"]
        self.assertEqual(row["status"], "WARN")
        self.assertAlmostEqual(row["value"], 0.5, places=6,
                               msg="distToShape must measure the 0.5 mm gap exactly")
        # the open pyramid is not a closed solid: the model side skips honestly
        self.assertEqual(self.rows_of(got)["exact_model_to_scan.max"]["status"], "SKIP")

    def test_a_shorter_model_is_a_missing_cluster(self):
        scan = Mesh.Mesh(Part.makeCylinder(10, 20).tessellate(0.3))
        model = Part.makeCylinder(10, 16)
        got = deviation.exact_residuals(scan, model, warn_mm=0.3)
        rows = self.rows_of(got)
        self.assertAlmostEqual(rows["exact_scan_to_model.max"]["value"], 4.0, places=3)
        self.assertEqual(rows["residual_clusters"]["value"], 1)
        cluster = got["clusters"][0]
        self.assertEqual(cluster["kind"], "missing")
        expected = 2 * math.pi * 10 * 4 + math.pi * 10 ** 2      # band + cap ≈ 565 mm²
        self.assertAlmostEqual(cluster["centroid_mm"][2], 18.7, delta=1.5)
        # share-estimate bounds (cap/fan density skew) — order-correct, never "exact"
        self.assertGreater(cluster["area_mm2"], expected * 0.5)
        self.assertLess(cluster["area_mm2"], expected * 2.2)

    def test_a_taller_model_is_an_excess_cluster(self):
        scan = Mesh.Mesh(Part.makeCylinder(10, 16).tessellate(0.3))
        model = Part.makeCylinder(10, 20)
        got = deviation.exact_residuals(scan, model, warn_mm=0.3)
        rows = self.rows_of(got)
        self.assertAlmostEqual(rows["exact_model_to_scan.max"]["value"], 4.0, places=3)
        self.assertGreaterEqual(rows["residual_clusters"]["value"], 1)
        self.assertTrue(all(c["kind"] == "excess" for c in got["clusters"]),
                        got["clusters"])
        self.assertTrue(all("crude" in c.get("note", "") or not c.get("note")
                            for c in got["clusters"]))

    def test_identical_geometry_has_no_offenders(self):
        scan = Mesh.Mesh(Part.makeCylinder(10, 20).tessellate(0.3))
        got = deviation.exact_residuals(scan, Part.makeCylinder(10, 20), warn_mm=0.3)
        rows = self.rows_of(got)
        self.assertAlmostEqual(rows["exact_scan_to_model.max"]["value"], 0.0, places=6)
        self.assertEqual(rows["residual_clusters"]["value"], 0)
        self.assertEqual(got["clusters"], [])

    def test_masked_scan_vertices_never_take_part(self):
        import numpy as np

        shape = Part.makeBox(10, 10, 0.01)
        pts = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [5, 5, 5.0]],
                       dtype=float)
        mesh = Mesh.Mesh(([App.Vector(*p) for p in pts],
                          [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]))
        got = deviation.exact_residuals(mesh, shape, warn_mm=0.3,
                                        mask={"scan_exclude_indices": [4]})
        row = self.rows_of(got)["exact_scan_to_model.max"]
        self.assertLessEqual(row["value"], 0.05,
                             "the excluded spike must not drive the exact rows")

    def test_an_empty_side_skips_honestly(self):
        got = deviation.exact_residuals(Mesh.Mesh(), Part.makeBox(10, 10, 4), warn_mm=0.3)
        self.assertEqual(self.rows_of(got)["exact_scan_to_model.max"]["status"], "SKIP")

    def test_a_tiny_budget_still_reports_rows(self):
        scan = Mesh.Mesh(Part.makeCylinder(10, 20).tessellate(0.5))

        class Clock:
            t = 100.0

            def __call__(self):
                self.t += 10.0            # every check is past the deadline
                return self.t

        got = deviation.exact_residuals(scan, Part.makeCylinder(10, 16), warn_mm=0.3,
                                        budget_s=1.0, clock=Clock())
        self.assertIn("exact_scan_to_model.max", self.rows_of(got))
        self.assertLessEqual(got["refined"]["scan"], 8,
                             "the budget must bind within one refinement batch")

    def test_laptop_limit_skips_expensive_mesh_to_solid_refinement(self):
        scan = Mesh.Mesh(Part.makeCylinder(10, 20).tessellate(0.5))
        model = Part.makeCylinder(10, 16)
        with mock.patch.object(deviation, "MAX_EXACT_MODEL_SCAN_FACETS", 0), \
             mock.patch("oscar.cad.imported.solid_from_mesh") as make_solid:
            got = deviation.exact_residuals(scan, model, warn_mm=0.3)
        make_solid.assert_not_called()
        rows = self.rows_of(got)
        self.assertEqual(rows["exact_model_to_scan.max"]["status"], "SKIP")
        self.assertIn("laptop resource limit", rows["exact_model_to_scan.max"]["detail"])
        self.assertGreater(got["refined"]["scan"], 0,
                           "scan-side exact residuals remain available")


if __name__ == "__main__":
    unittest.main()


class ExactIouTest(unittest.TestCase):
    """D-1: the exact boolean IoU row — analytic values on boxes, honest skips otherwise."""

    def setUp(self):
        import FreeCAD as App

        self.doc = App.newDocument("IouT")
        self.addCleanup(App.closeDocument, self.doc.Name)

    def _mesh_and_shape(self, x, y, z):
        box = Part.makeBox(x, y, z)
        return Mesh.Mesh(box.tessellate(0.3)), box

    def test_identical_boxes_score_one(self):
        from oscar.recon import deviation

        mesh, shape = self._mesh_and_shape(20, 10, 5)
        row = deviation.exact_iou(mesh, shape)
        self.assertEqual(row["iou"], 1.0, row)
        self.assertEqual(row["note"], "")

    def test_a_quarter_volume_scores_the_analytic_value(self):
        from oscar.recon import deviation

        mesh, _shape = self._mesh_and_shape(20, 10, 5)      # scan = the bigger box
        _, small = self._mesh_and_shape(10, 10, 5)          # model inside it: half the volume
        row = deviation.exact_iou(mesh, small)
        self.assertEqual(row["iou"], 0.5, row)              # 500 / (1000 + 1000 - 500)

    def test_the_facet_cap_is_an_honest_skip(self):
        from oscar.recon import deviation

        mesh, shape = self._mesh_and_shape(20, 10, 5)
        row = deviation.exact_iou(mesh, shape, facet_cap=2)   # any mesh exceeds 2 facets
        self.assertIsNone(row["iou"])
        self.assertIn("cap", row["note"])

    def test_an_empty_intersection_of_overlapping_solids_is_a_reported_anomaly(self):
        # the noisy-bracket anomaly: OCC's common() silently returns empty against a
        # degenerate sewn solid — reporting iou 0.0 would claim a measured zero overlap
        # that does not exist. OCC itself computes valid solids correctly (a mirrored
        # box still yields its true intersection), so the guard is exercised through a
        # lying proxy: valid shape, positive volume, overlapping bbox, empty common().
        from oscar.recon import deviation

        class _LyingShape:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def common(self, other):
                return Part.Compound([])

        mesh, shape = self._mesh_and_shape(20, 10, 5)
        row = deviation.exact_iou(mesh, _LyingShape(shape))
        self.assertIsNone(row["iou"], row)
        self.assertIn("despite overlapping", row["note"])

    def test_an_invalid_shape_is_reported_not_substituted(self):
        from oscar.recon import deviation

        mesh, _shape = self._mesh_and_shape(20, 10, 5)
        row = deviation.exact_iou(mesh, None)
        self.assertIsNone(row["iou"])
        self.assertIn("shape", row["note"])
