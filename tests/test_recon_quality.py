"""Stage 1 / K1 + G5: ``oscar.recon.quality`` — noise sigma estimation, the derived,
noise-adaptive recognition gates (plus their wiring into ``recognize.survey``), and the
§2 scan-quality assessment (completeness / integrity / plausibility → readiness).

Sigma oracles are synthetic noisy-plane grids with a fixed seed (the Stage-1 probe ladder:
raw MAD reads ~0.85x the true sigma at k=16, so the module calibrates by 1.18); clean
tessellations must floor at 2% of the mean edge; the survey gates must widen with measured
noise while a clean plate keeps its previous behaviour. Everything runs under freecadcmd on
real meshes; the estimate must never touch the mesh.

G5 fixtures manufacture every defect class synthetically: a box missing its whole top face
(unscanned region), a wall-only tube (two bore ends, through-paired), an annulus+disc over
one bore (hole openings with NO scanned wall, through-paired — the hole exists but the
scanner never saw its wall), a deleted mid-grid quad (pinhole), one displaced vertex
(isolated spike) vs a displaced 3×3 patch (NOT spikes — isolation), and two same-facing
parallel sheets (the unaligned-scan-pass doubled surface; a thin closed box must NOT flag:
its far side faces the other way).
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
import numpy as np  # noqa: E402

from oscar.recon import meshops  # noqa: E402
from oscar.recon import quality  # noqa: E402
from oscar.recon import recognize  # noqa: E402


def noisy_plane_mesh(sigma, size=40.0, n=41, seed=20261114):
    """A triangulated z-noisy plane grid (true noise = sigma mm)."""
    rng = np.random.default_rng(seed)
    g = np.linspace(0.0, size, n)
    xs, ys = np.meshgrid(g, g)
    z = sigma * rng.standard_normal(xs.size)
    pts = np.column_stack([xs.ravel(), ys.ravel(), z])
    facs = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            facs += [(a, b, d), (a, d, c)]
    return Mesh.Mesh(([App.Vector(*p) for p in pts], facs))


def mesh_from(pts, facs):
    return Mesh.Mesh(([App.Vector(*p) for p in pts], [tuple(f) for f in facs]))


def tess_mesh(shape, dev):
    v, f = shape.tessellate(dev)
    pts = np.array([[p.x, p.y, p.z] for p in v])
    return mesh_from(pts, [tuple(int(i) for i in t) for t in f])


def box_missing_top(a=30.0):
    v, f = Part.makeBox(a, a, a).tessellate(0.5)
    pts = np.array([[p.x, p.y, p.z] for p in v])
    facs = [tuple(int(i) for i in t) for t in f]
    keep = [t for t in facs if not all(pts[i][2] > a - 1e-6 for i in t)]
    return mesh_from(pts, keep)


def wall_only_tube(r=6.0, h=20.0, dev=0.5):
    """A scanned cylinder wall with both caps removed: two round rims whose surrounding
    faces ARE the wall (bore ends), coaxial and through-pairable."""
    v, f = Part.makeCylinder(r, h).tessellate(dev)
    pts = np.array([[p.x, p.y, p.z] for p in v])
    facs = [tuple(int(i) for i in t) for t in f]
    keep = []
    for t in facs:
        p = pts[list(t)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if abs(n[2] / max(float(np.linalg.norm(n)), 1e-12)) < 0.9:
            keep.append(t)
    return mesh_from(pts, keep)


def annulus_with_unscanned_hole(r_in=6.0, r_out=20.0, n=16, t=6.0):
    """A top annulus and a bottom disc over the same bore: the hole's wall was never
    scanned — two round openings sitting in flat faces (rim normals aligned with the loop
    normal), plus the disc's round outer rim (the one-sided scan's outer boundary)."""
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    inner = np.column_stack([r_in * np.cos(ang), r_in * np.sin(ang), np.full(n, t)])
    outer = np.column_stack([r_out * np.cos(ang), r_out * np.sin(ang), np.full(n, t)])
    ring2 = np.column_stack([r_in * np.cos(ang), r_in * np.sin(ang), np.zeros(n)])
    pts = np.vstack([inner, outer, [[0.0, 0.0, 0.0]], ring2])
    facs = []
    for i in range(n):
        j = (i + 1) % n
        facs += [(i, n + i, n + j), (i, n + j, j)]            # the annulus, facing +z
        facs.append((2 * n, 2 * n + 1 + j, 2 * n + 1 + i))    # the bottom disc, facing +z
    return mesh_from(pts, facs)


def two_parallel_sheets(size=40.0, delta=0.8, g=9):
    """Two full sheets over the same footprint facing the SAME way ``delta`` apart — the
    unaligned-scan-passes artifact. Open on purpose: a CLOSED orientable shell can never
    hold two same-facing overlapping layers (the void between them forces opposite
    normals), which is why the reconstruction gate is wired with a stubbed report."""
    xs = np.linspace(0.0, size, g)
    X, Y = np.meshgrid(xs, xs)
    lo = np.column_stack([X.ravel(), Y.ravel(), np.zeros(g * g)])
    hi = lo.copy()
    hi[:, 2] = delta
    pts = np.vstack([lo, hi])
    facs = []
    for i in range(g - 1):
        for j in range(g - 1):
            a, b, c, e = i * g + j, i * g + j + 1, (i + 1) * g + j, (i + 1) * g + j + 1
            for off in (0, g * g):
                facs += [(off + a, off + b, off + c), (off + b, off + e, off + c)]
    return pts, np.array(facs)


def grid_sheet(size=20.0, n=21):
    """A flat open grid (n x n vertices) — the spike/pinhole substrate."""
    xs = np.linspace(0.0, size, n)
    X, Y = np.meshgrid(xs, xs)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n * n)])
    facs = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, e = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            facs += [(a, b, e), (a, e, c)]
    return pts, facs


class SigmaLadderTest(unittest.TestCase):
    def test_the_ladder_is_linear_within_20_percent(self):
        for true_sigma in (0.01, 0.02, 0.05):
            est = quality.estimate_sigma(noisy_plane_mesh(true_sigma))
            self.assertAlmostEqual(est["sigma_raw_mm"], true_sigma, delta=0.25 * true_sigma,
                                   msg=f"raw sigma for true {true_sigma}")
            self.assertAlmostEqual(est["sigma_mm"], true_sigma, delta=0.20 * true_sigma,
                                   msg=f"calibrated sigma for true {true_sigma}")

    def test_a_noiseless_grid_floors_at_half_a_percent_of_the_edge(self):
        est = quality.estimate_sigma(noisy_plane_mesh(0.0))
        self.assertAlmostEqual(est["sigma_mm"], est["floor_mm"], places=4)
        self.assertAlmostEqual(est["sigma_mm"], quality.SIGMA_EDGE_FLOOR * est["mean_edge_mm"],
                               delta=1e-3)

    def test_the_estimate_never_touches_the_mesh(self):
        m = noisy_plane_mesh(0.05)
        before = (int(m.CountFacets), int(m.CountPoints), round(float(m.Area), 3))
        quality.estimate_sigma(m)
        self.assertEqual((int(m.CountFacets), int(m.CountPoints), round(float(m.Area), 3)),
                         before)


class CleanMeshTest(unittest.TestCase):
    def test_a_clean_tessellated_cylinder_reads_as_noise_free(self):
        cyl = Part.makeCylinder(6, 20).tessellate(0.4)   # fine: chord sag ~0.003 mm
        mesh = Mesh.Mesh((list(cyl[0]), [tuple(f) for f in cyl[1]]))
        est = quality.estimate_sigma(mesh)
        self.assertLessEqual(est["sigma_mm"], max(est["floor_mm"] * 2.0, 0.01),
                             f"clean cylinder must read as (near) noise-free, got {est}")


class TolerancesTest(unittest.TestCase):
    def test_gates_combine_sigma_and_edge_floor(self):
        t = quality.tolerances(0.1, 2.0)
        self.assertAlmostEqual(t["plane_rms_mm"], 0.25, places=3)      # 2.5*sigma wins
        self.assertAlmostEqual(t["cylinder_rms_mm"], 0.30, places=3)   # 3*sigma wins
        t2 = quality.tolerances(0.0, 5.0)
        self.assertAlmostEqual(t2["plane_rms_mm"], 0.10, places=3)     # 0.02*edge
        self.assertAlmostEqual(t2["cylinder_rms_mm"], 0.50, places=3)  # 0.1*edge

    def test_estimate_tolerances_adds_gates_to_the_sigma(self):
        est = quality.estimate_tolerances(noisy_plane_mesh(0.02))
        self.assertIn("plane_rms_mm", est)
        self.assertGreater(est["plane_rms_mm"], 2.0 * est["sigma_mm"])


class AdaptiveSurveyTest(unittest.TestCase):
    def test_noisy_planes_pass_their_own_widened_gate(self):
        s = recognize.survey(noisy_plane_mesh(0.06))
        q = s["quality"]
        self.assertGreater(q["plane_rms_mm"], recognize.PLANE_RMS_MM,
                           "a 0.06 mm-noisy mesh must widen the fixed 0.05 gate")
        self.assertGreaterEqual(s["coverage"]["planes"], 0.8,
                                f"planes must still be recognized: {s['coverage']}")
        self.assertGreater(q["sigma_mm"], 0.045)

    def test_a_clean_box_keeps_the_prismatic_behaviour(self):
        pts, facs = Part.makeBox(30, 30, 30).tessellate(4.0)
        mesh = Mesh.Mesh((list(pts), [tuple(f) for f in facs]))
        s = recognize.survey(mesh)
        self.assertEqual(s["classification"]["class"], "prismatic")
        self.assertIsNotNone(s["quality"]["sigma_mm"])
        self.assertGreaterEqual(s["coverage"]["planes"], 0.95)

    def test_a_precomputed_quality_dict_is_honoured(self):
        pts, facs = Part.makeBox(30, 30, 30).tessellate(4.0)
        mesh = Mesh.Mesh((list(pts), [tuple(f) for f in facs]))
        s = recognize.survey(mesh, quality={"sigma_mm": 0.0, "mean_edge_mm": 5.0,
                                            "plane_rms_mm": 0.1, "cylinder_rms_mm": 0.5})
        self.assertEqual(s["quality"]["plane_rms_mm"], 0.1)


class NeighbourKernelTest(unittest.TestCase):
    def test_knn_matches_a_brute_force_sort(self):
        rng = np.random.default_rng(7)
        pts = rng.uniform(0.0, 10.0, (120, 3))
        d, i, engine = quality._knn(pts, pts, 16)
        self.assertIn(engine, ("scipy.cKDTree", "numpy-chunked"))
        for row in (0, 17, 60, 119):
            dd = np.sqrt(((pts - pts[row]) ** 2).sum(1))
            order = np.argsort(dd)[1:17]                      # self excluded
            self.assertTrue(np.allclose(np.sort(d[row]), np.sort(dd[order]), atol=1e-9))


# --------------------------------------------------------------------------- G5 (§2)
class ScanQualityReportTest(unittest.TestCase):
    def test_a_clean_closed_box_is_ready(self):
        sq = quality.scan_quality(tess_mesh(Part.makeBox(30, 30, 30), 0.5))
        self.assertEqual(sq["readiness"]["verdict"], "ready", sq["readiness"])
        self.assertEqual(sq["completeness"]["boundary_loops"], 0)
        self.assertEqual(sq["integrity"]["spikes"]["count"], 0)
        self.assertEqual(sq["integrity"]["doubled_surfaces"], [])

    def test_a_coarse_bracket_scan_is_ready_its_corners_are_not_spikes(self):
        shape = Part.makeBox(60, 40, 10)
        for x, y in ((15, 10), (45, 10), (15, 30), (45, 30)):
            shape = shape.cut(Part.makeCylinder(3.3, 30, App.Vector(x, y, -10)))
        sq = quality.scan_quality(Mesh.Mesh(shape.tessellate(0.3)))
        self.assertEqual(sq["integrity"]["spikes"]["count"], 0, sq["integrity"]["spikes"])
        self.assertEqual(sq["readiness"]["verdict"], "ready", sq["readiness"])
        check = sq["integrity"]["doubled_surface_check"]
        self.assertEqual(check["sampled_facets"], quality.LAPTOP_DOUBLE_SURFACE_SAMPLES)
        self.assertGreater(check["total_facets"], check["sampled_facets"])
        self.assertFalse(check["complete"])

    def test_plausibility_reports_both_scales_and_the_detectable_hole_radius(self):
        sq = quality.scan_quality(tess_mesh(Part.makeBox(30, 20, 10), 0.5))
        p = sq["plausibility"]
        self.assertAlmostEqual(p["bbox_diagonal_mm"], math.sqrt(1400.0), places=2)
        self.assertAlmostEqual(p["bbox_diagonal_inch"], math.sqrt(1400.0) / 25.4, places=3)
        expected_r = quality.MIN_HOLE_RIM_POINTS * sq["mean_edge_mm"] / (2.0 * math.pi)
        self.assertAlmostEqual(p["min_detectable_hole_radius_mm"], round(expected_r, 3),
                               places=2)
        self.assertIn("16 rim points", p["min_detectable_basis"])

    def test_the_units_status_is_embedded_in_plausibility(self):
        m = tess_mesh(Part.makeBox(30, 30, 30), 0.5)
        p = quality.scan_quality(m, units_status="POSSIBLE_MM_INCH_MISMATCH")["plausibility"]
        self.assertEqual(p["scale_status"], "POSSIBLE_MM_INCH_MISMATCH")

    def test_a_precomputed_est_is_reused_not_reestimated(self):
        m = tess_mesh(Part.makeBox(30, 30, 30), 0.5)
        est = quality.estimate_tolerances(m)
        self.assertEqual(quality.scan_quality(m, est=est)["sigma_mm"], est["sigma_mm"])

    def test_the_report_never_touches_the_mesh(self):
        m = box_missing_top()
        before = (int(m.CountFacets), int(m.CountPoints), round(float(m.Area), 3))
        quality.scan_quality(m)
        self.assertEqual((int(m.CountFacets), int(m.CountPoints), round(float(m.Area), 3)),
                         before)


class LoopClassificationTest(unittest.TestCase):
    def test_a_box_missing_its_whole_top_face_is_a_missing_region(self):
        sq = quality.scan_quality(box_missing_top())
        loops = sq["completeness"]["loops"]
        self.assertEqual(len(loops), 1, loops)
        self.assertEqual(loops[0]["kind"], "dropout")
        self.assertEqual(loops[0]["detail"], "unscanned_face")
        self.assertEqual(sq["readiness"]["verdict"], "degraded")
        self.assertTrue(any("missing region" in r for r in sq["readiness"]["reasons"]))

    def test_a_wall_only_tube_reports_two_through_paired_bore_ends(self):
        sq = quality.scan_quality(wall_only_tube())
        loops = sq["completeness"]["loops"]
        self.assertEqual(sorted(l["kind"] for l in loops), ["bore_end", "bore_end"], loops)
        paired = [l for l in loops if l.get("through")]
        self.assertEqual(len(paired), 2)
        self.assertAlmostEqual(paired[0]["through"]["depth_mm"], 20.0, places=1)
        self.assertEqual(sq["readiness"]["verdict"], "caution")

    def test_an_unscanned_hole_wall_reports_paired_hole_openings(self):
        sq = quality.scan_quality(annulus_with_unscanned_hole())
        kinds = sorted(l["kind"] for l in sq["completeness"]["loops"])
        self.assertEqual(kinds, ["hole_opening", "hole_opening", "outer_rim"],
                         sq["completeness"]["loops"])
        openings = [l for l in sq["completeness"]["loops"] if l["kind"] == "hole_opening"]
        self.assertTrue(all(l.get("through") for l in openings), openings)
        self.assertAlmostEqual(openings[0]["through"]["depth_mm"], 6.0, places=2)
        self.assertEqual(sq["readiness"]["verdict"], "degraded")
        self.assertTrue(any("hole opening" in r for r in sq["readiness"]["reasons"]))

    def test_a_deleted_mid_grid_quad_is_a_pinhole_not_an_unscanned_face(self):
        pts, facs = grid_sheet()
        mid = 2 * (10 * 20 + 10)                 # the quad at grid cell (10, 10)
        facs = [f for k, f in enumerate(facs) if k not in (mid, mid + 1)]
        sq = quality.scan_quality(mesh_from(pts, facs))
        loops = sq["completeness"]["loops"]
        self.assertEqual(len(loops), 2, loops)
        self.assertEqual(sorted(l["kind"] for l in loops), ["dropout", "outer_rim"])
        hole = next(l for l in loops if l["kind"] == "dropout")
        self.assertEqual(hole["detail"], "pinhole")
        self.assertEqual(sq["readiness"]["verdict"], "caution", sq["readiness"])


class BoundaryRingTest(unittest.TestCase):
    def test_boundary_emits_the_ordered_ring_of_each_loop(self):
        b = meshops.boundary(box_missing_top())
        self.assertEqual(len(b["loops"]), 1)
        ring = b["loops"][0]["ring"]
        self.assertEqual(len(ring), 4)
        self.assertTrue(all(isinstance(i, int) for i in ring))

    def test_quality_loop_entries_carry_the_ring_for_g6(self):
        sq = quality.scan_quality(box_missing_top())
        loop = sq["completeness"]["loops"][0]
        self.assertEqual(len(loop["ring"]), 4)
        pts, _ = quality._pts_fac(box_missing_top())
        ring_pts = pts[np.asarray(loop["ring"], dtype=int)]
        self.assertAlmostEqual(float(ring_pts[:, 2].max()), 30.0, places=3,
                               msg="the missing-top rim ring sits on the open face")


class SpikeTest(unittest.TestCase):
    def test_one_displaced_vertex_is_an_isolated_spike(self):
        pts, _ = grid_sheet()
        spiked = pts.copy()
        spiked[10 * 21 + 10, 2] = 5.0
        got = quality._spikes(spiked, 0.0, 1.0)
        self.assertGreaterEqual(got["count"], 1, got)
        self.assertIn([10.0, 10.0, 5.0], got["examples"])
        self.assertFalse(got["estimated"])

    def test_a_displaced_patch_is_not_spikes(self):
        pts, _ = grid_sheet()
        patch = pts.copy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                patch[(10 + di) * 21 + (10 + dj), 2] = 5.0
        self.assertEqual(quality._spikes(patch, 0.0, 1.0)["count"], 0)


class DoubledSurfaceTest(unittest.TestCase):
    def test_two_same_facing_layers_are_detected_with_their_offset(self):
        pts, facs = two_parallel_sheets()
        sq = quality.scan_quality(mesh_from(pts, facs))
        ds = sq["integrity"]["doubled_surfaces"]
        self.assertEqual(len(ds), 1, ds)
        self.assertAlmostEqual(ds[0]["offset_mm"], 0.8, delta=0.1)
        self.assertGreaterEqual(ds[0]["n_samples"], 5)
        self.assertEqual(sq["readiness"]["verdict"], "degraded")
        self.assertTrue(any("doubled" in r for r in sq["readiness"]["reasons"]))

    def test_a_thin_closed_box_has_no_doubled_surfaces(self):
        sq = quality.scan_quality(tess_mesh(Part.makeBox(30, 30, 3), 0.5))
        self.assertEqual(sq["integrity"]["doubled_surfaces"], [])
        self.assertEqual(sq["readiness"]["verdict"], "ready")

    def test_the_adjacency_sigma_stays_small_where_the_knn_sigma_swallows_layers(self):
        pts, facs = two_parallel_sheets()
        est = quality.estimate_sigma(mesh_from(pts, facs))
        self.assertGreater(est["sigma_mm"], 0.4, "k-NN crosses the layers — it must show")
        self.assertLess(quality._adjacency_sigma(pts, facs), 0.05,
                        "the 1-ring stays in one layer — it must NOT show")


if __name__ == "__main__":
    unittest.main()


class HundredKFacetTierTest(unittest.TestCase):
    """S8: the >=100k-facet performance tier — diagnose + noise estimate + the
    scan-quality report must complete on a very fine mesh inside a generous budget
    (the release tier check; timing depends on the machine, the bound is loose)."""

    def test_the_diagnose_quality_path_handles_a_100k_facet_mesh(self):
        import time

        mesh = tess_mesh(Part.makeSphere(20.0), 0.004)
        self.assertGreater(int(mesh.CountFacets), 100_000,
                           f"fixture too small: {mesh.CountFacets}")
        t0 = time.monotonic()
        from oscar.recon import meshops

        diag = meshops.diagnose(mesh)
        sq = quality.scan_quality(mesh)
        elapsed = time.monotonic() - t0
        self.assertTrue(diag["closed"])
        self.assertIn(sq["readiness"]["verdict"], ("ready", "caution", "degraded"))
        self.assertLess(elapsed, 300.0, f"the 100k tier took {elapsed:.0f}s")
