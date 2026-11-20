"""M4/M5 end-to-end: ``oscar.recon.reconstruct`` turns a tessellated bracket-family scan into a
native parametric model through the REAL builder — actual Sketcher geometry, PartDesign Pad and
Hole features, one authoritative parameter sheet with scan_fit sources, and deviation rows
against the source mesh. Unsupported classes come back honestly labelled.
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
import numpy as np  # noqa: E402

from oscar.cad import health_eval  # noqa: E402
from oscar.recon import reconstruct  # noqa: E402


def bracket_scan(length=60.0, width=40.0, thick=10.0, radius=3.3):
    shape = Part.makeBox(length, width, thick)
    for x, y in ((15, 10), (45, 10), (15, 30), (45, 30)):
        shape = shape.cut(Part.makeCylinder(radius, thick * 3, App.Vector(x, y, -thick)))
    return Mesh.Mesh(shape.tessellate(0.3))


class ReconstructBracketTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = App.newDocument("ReconE2E")
        cls.mesh = bracket_scan()
        cls.result = reconstruct.reconstruct(cls.doc, cls.mesh)

    @classmethod
    def tearDownClass(cls):
        App.closeDocument(cls.doc.Name)

    def rows(self):
        return {r["id"]: r for r in self.result["deviation"]}

    def test_the_bracket_builds_with_real_native_features(self):
        self.assertTrue(self.result["supported"], self.result.get("note"))
        self.assertTrue(self.result["build"]["success"], self.result["build"])
        report = self.result["build"]
        holes = [o for o in self.doc.Objects if o.TypeId == "PartDesign::Hole"]
        sketches = [o for o in self.doc.Objects if o.TypeId == "Sketcher::SketchObject"]
        self.assertEqual(len(holes), 4)
        self.assertGreaterEqual(len(sketches), 5)
        for sketch in sketches:
            self.assertGreaterEqual(sketch.GeometryCount, 1, "no empty sketches")

    def test_one_bound_parameter_sheet_with_scan_fit_sources(self):
        sheets = [o for o in self.doc.Objects
                  if o.TypeId == "Spreadsheet::Sheet" and o.Name != "OscarParameterProbe"]
        self.assertEqual(len(sheets), 1, [s.Name for s in sheets])
        table = health_eval.read_param_table(self.doc)
        self.assertGreaterEqual(sum(r["bound"] for r in table["rows"]), 4)
        self.assertIn("scan_fit", {r["source"] for r in table["rows"]})

    def test_volume_matches_the_analytic_bracket(self):
        body = self.doc.getObject(self.result["build"]["body"])
        expected = 60 * 40 * 10 - 4 * math.pi * 3.3 ** 2 * 10
        self.assertAlmostEqual(body.Shape.Volume, expected, delta=expected * 0.02)

    def test_deviation_rows_and_the_honest_outcome(self):
        rows = self.rows()
        self.assertLessEqual(rows["hausdorff"]["value"], 0.35, rows["hausdorff"])
        self.assertEqual(self.result["outcome"], "verified_parametric", self.result["meta"])
        self.assertEqual(rows["samples"]["status"], "PASS")

    def test_a_parameter_edit_moves_the_geometry(self):
        sheet = self.doc.getObject("Parameters")
        cell = sheet.getCellFromAlias("plate_t")
        before = self.doc.getObject(self.result["build"]["body"]).Shape.Volume
        sheet.set(cell, "=12")
        self.doc.recompute()
        after = self.doc.getObject(self.result["build"]["body"]).Shape.Volume
        sheet.set(cell, "=10")
        self.doc.recompute()
        self.assertGreater(after, before * 1.1)


class ReconstructUnsupportedTest(unittest.TestCase):
    def test_an_organic_scan_is_refused_honestly(self):
        doc = App.newDocument("ReconOrg")
        try:
            mesh = Mesh.Mesh(Part.makeSphere(20.0).tessellate(0.4))
            result = reconstruct.reconstruct(doc, mesh)
            self.assertFalse(result["supported"])
            self.assertEqual(result["classification"]["class"], "spherical")
            self.assertIn("not implemented", result["note"])
        finally:
            App.closeDocument(doc.Name)

    def test_an_open_mesh_reports_the_repair_gate(self):
        doc = App.newDocument("ReconOpen")
        try:
            pts, facs = Part.makeBox(30, 30, 8).tessellate(2.0)
            facs = [tuple(f) for f in facs if 0 not in f]        # a hole where facets went missing
            mesh = Mesh.Mesh((list(pts), facs))
            result = reconstruct.reconstruct(doc, mesh)
            self.assertFalse(result.get("supported"))
            self.assertEqual(result.get("blocked"), "mesh_not_closed")
            self.assertIn("fillupHoles", result["note"])
        finally:
            App.closeDocument(doc.Name)


class ScanQualityGateTest(unittest.TestCase):
    """G5: DEGRADED scan readiness (doubled surfaces / missing regions / unscanned hole
    walls) caps the outcome at partially_verified_parametric.

    The degraded path is driven by a STUBBED scan_quality report: a closed orientable
    mesh can structurally never hold two same-facing overlapping layers (the void between
    them forces opposite normals), so a real doubled-surface mesh is always open and
    stops at the earlier mesh_not_closed gate. The detector itself is tested on real
    open meshes in tests/test_recon_quality.py."""

    def _degraded_report(self):
        return {
            "sigma_mm": 0.0, "mean_edge_mm": 1.0,
            "completeness": {"boundary_loops": 0, "open_chains": 0,
                             "boundary_junctions": 0, "loops": []},
            "integrity": {"spikes": {"count": 0, "estimated": False, "examples": []},
                          "doubled_surfaces": [{"area_mm2": 100.0,
                                                "centroid_mm": [0.0, 0.0, 0.0],
                                                "offset_mm": 0.5, "n_samples": 9}]},
            "plausibility": {},
            "readiness": {"verdict": "degraded",
                          "reasons": ["1 doubled-surface region(s) (offsets [0.5] mm)"],
                          "notes": []}}

    def test_a_degraded_scan_caps_the_outcome_at_partially_verified(self):
        from unittest import mock

        from oscar.recon import quality

        doc = App.newDocument("G5Gate1")
        try:
            with mock.patch.object(quality, "scan_quality",
                                   return_value=self._degraded_report()):
                result = reconstruct.reconstruct(doc, bracket_scan(), warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            self.assertIn("scan_degraded", result["meta"], result["meta"])
            self.assertIn("doubled-surface", result["meta"]["scan_degraded"][0])
            self.assertEqual(result["outcome"], "partially_verified_parametric",
                             result["meta"])
        finally:
            App.closeDocument(doc.Name)

    def test_a_clean_scan_reports_ready_and_stays_verified(self):
        doc = App.newDocument("G5Gate2")
        try:
            result = reconstruct.reconstruct(doc, bracket_scan(), warn_mm=0.5)
            self.assertEqual(result["scan_quality"]["readiness"]["verdict"], "ready",
                             result["scan_quality"]["readiness"])
            self.assertNotIn("scan_degraded", result["meta"])
            self.assertEqual(result["outcome"], "verified_parametric", result["meta"])
        finally:
            App.closeDocument(doc.Name)


class G6MeasuredMaskTest(unittest.TestCase):
    """G6: the authorized fill runs under the loop-aware policy inside the parametric
    route, and deviation covers measured regions only — the pinhole fill's synthetic
    vertices are excluded and REPORTED, never mixed into the measurement."""

    @staticmethod
    def grid_box(w=20.0, d=20.0, h=8.0, n=11):
        """A closed box built from grids (planar faces of Part shapes tessellate to two
        triangles however fine the deviation — G5 finding — so pinhole fixtures are
        hand-built). Returns (mesh, points, facets)."""
        xs = np.linspace(0, w, n)
        ys = np.linspace(0, d, n)

        def sheet(z):
            return np.array([[x, y, z] for y in ys for x in xs])

        pts = np.vstack([sheet(0.0), sheet(h)])
        N = n * n
        facs = []
        for i in range(n - 1):
            for j in range(n - 1):
                a, b, c, e = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
                facs += [(a, c, b), (b, c, e)]                    # bottom, facing -z
                facs += [(N + a, N + b, N + c), (N + b, N + e, N + c)]   # top, facing +z
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
        return Mesh.Mesh(([App.Vector(*p) for p in pts], facs)), pts, np.array(facs)

    @classmethod
    def holed_box(cls):
        """The closed grid box with one quad (a pinhole) removed from its top face."""
        mesh, pts, facs = cls.grid_box()
        top_start = 2 * 10 * 10                     # bottom sheet facets come first
        pin = top_start + 2 * (5 * 10 + 5)          # quad (5, 5) of the top sheet
        keep = [tuple(f) for k, f in enumerate(facs.tolist()) if k not in (pin, pin + 1)]
        return Mesh.Mesh(([App.Vector(*p) for p in pts], keep))

    def test_authorized_pinhole_fill_reconstructs_with_a_measured_only_mask(self):
        doc = App.newDocument("G6Mask1")
        try:
            result = reconstruct.reconstruct(doc, self.holed_box(), warn_mm=2.0,
                                             authorized=("fillupHoles",))
            self.assertTrue(result["supported"], result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            fill = next(a for a in result["repair"]["actions"]
                        if a["operation"] == "fillupHoles")
            self.assertEqual(fill["verdict"], "ACCEPTED", result["repair"]["validation"])
            self.assertGreaterEqual(result["provenance"]["counts"]["synthetic"], 1)
            self.assertEqual(result["provenance_verification"]["verdict"], "VERIFIED")
            mask = result["measured_mask"]
            self.assertIsNotNone(mask)
            self.assertGreaterEqual(mask["scan_excluded"], 1,
                                    "the fill's vertices must be excluded as unmeasured")
            rows = {r["id"]: r for r in result["deviation"]}
            self.assertIn("measured_only", rows)
            self.assertIn("MEASURED", rows["measured_only"]["detail"])
            # the OUTCOME is not pinned here: the grid-box scan samples at 2 mm while the
            # model's planar faces tessellate to corner vertices only, so scan_to_model
            # reflects sampling density, not G6 (the clean-bracket test pins the outcome
            # path; this test pins the fill policy, mask and provenance wiring)
        finally:
            App.closeDocument(doc.Name)

    def test_an_unfillable_opening_blocks_with_missing_regions_recorded(self):
        n = 21
        xs = np.linspace(0, 20, n)
        X, Y = np.meshgrid(xs, xs)
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n * n)])
        facs = []
        dropped = {(i, j) for i in range(9, 12) for j in range(9, 12)}   # a 12-edge dropout
        for i in range(n - 1):
            for j in range(n - 1):
                if (i, j) in dropped:
                    continue
                a, b, c, e = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
                facs += [(a, b, e), (a, e, c)]
        sheet = Mesh.Mesh(([App.Vector(*p) for p in pts], facs))
        doc = App.newDocument("G6Mask2")
        try:
            result = reconstruct.reconstruct(doc, sheet, warn_mm=2.0,
                                             authorized=("fillupHoles",))
            self.assertFalse(result.get("supported"))
            self.assertEqual(result.get("blocked"), "mesh_not_closed")
            # H4: one-sided flat sheets refuse at the OUTER RIM (per-loop reason);
            # the old "never auto-filled" wording only survives on the fill path
            self.assertIn("refused", result["note"])
            self.assertEqual(len(result["missing_regions"]), 1)
            self.assertEqual(result["missing_regions"][0]["n_edges"], 12)
        finally:
            App.closeDocument(doc.Name)

    def test_a_clean_scan_reports_a_zero_exclusion_mask(self):
        doc = App.newDocument("G6Mask3")
        try:
            result = reconstruct.reconstruct(doc, bracket_scan(), warn_mm=0.5)
            self.assertEqual(result["measured_mask"],
                             {"scan_excluded": 0, "model_excluded": 0, "regions": 0,
                              "note": ""})
            rows = {r["id"]: r for r in result["deviation"]}
            self.assertIn("measured_only", rows)
            self.assertEqual(rows["measured_only"]["value"], 0)
            self.assertEqual(result["outcome"], "verified_parametric", result["meta"])
        finally:
            App.closeDocument(doc.Name)

    def test_the_reconstruction_carries_exact_residual_rows(self):
        # V-1: the deviation rows are extended with the exact worst-point refinement
        # and the residual-cluster row; the sampled rows keep deciding the outcome
        doc = App.newDocument("V1Rows")
        try:
            result = reconstruct.reconstruct(doc, bracket_scan(), warn_mm=0.5)
            rows = {r["id"]: r for r in result["deviation"]}
            self.assertIn("exact_scan_to_model.max", rows)
            self.assertIn("residual_clusters", rows)
            self.assertLessEqual(rows["exact_scan_to_model.max"]["value"], 0.5)
            self.assertIsNotNone(result["residual_analysis"])
            self.assertEqual(result["residual_analysis"]["refined"]["scan"] > 0, True)
            self.assertEqual(result["outcome"], "verified_parametric", result["meta"])
        finally:
            App.closeDocument(doc.Name)


if __name__ == "__main__":
    unittest.main()


class G7DressOpsTest(unittest.TestCase):
    """G7: measured dressings become plan dress ops — a filleted bracket's radius comes
    from the vertical-section arc fits (±5% acceptance) and lands as a bound
    PartDesign::Fillet; a chamfered plate's width lands as a bound chamfer. A sharp
    plate gains no dress op. (The mid-height outline of FILLETED tessellations can
    mis-chain — an R-2 loop-fitting limitation recorded in the knowledge file — so the
    filleted fixture pins radius and features, not total volume.)"""

    @staticmethod
    def filleted_bracket(r=2.5):
        shape = Part.makeBox(60, 40, 10)
        for x, y in ((15, 10), (45, 10), (15, 30), (45, 30)):
            shape = shape.cut(Part.makeCylinder(3.3, 30, App.Vector(x, y, -10)))
        top = [e for e in shape.Edges
               if abs(e.CenterOfMass.z - 10) < 1e-6 and e.Length > 1e-6]
        return shape.makeFillet(r, top)

    @staticmethod
    def chamfered_plate():
        box = Part.makeBox(40, 30, 10)
        top = [e for e in box.Edges if abs(e.CenterOfMass.z - 10) < 1e-6]
        return box.makeChamfer(2.0, top)

    def test_the_filleted_bracket_builds_with_measured_radius_and_holes(self):
        # H1 (2026-11-15): the outline mis-chain is fixed (corner-preserving smoothing
        # in section.fit_loop), so the filleted bracket now BUILDS: measured radius
        # (±5 %), four PartDesign holes, volume within 2 %, bound fillet feature.
        mesh = Mesh.Mesh(self.filleted_bracket().tessellate(0.3))
        doc = App.newDocument("G7Rec1")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            # 2026-11-18 (E2 completion): the fillet is MEASURED (r=2.5, scan_fit source,
            # section-measured arcs) and APPLIED (volume within 0.3 % on this fixture) —
            # which face family carries it (fillet_r far vs fillet_rb base) depends on the
            # frame's legitimate base-face choice, so the contract is family-agnostic.
            params = result["plan"]["parameters"]
            fillet_params = [k for k in ("fillet_r", "fillet_rb") if k in params]
            self.assertEqual(len(fillet_params), 1, sorted(params))
            self.assertAlmostEqual(params[fillet_params[0]]["value"], 2.5, delta=0.05 * 2.5)
            fillets = [o for o in doc.Objects if o.TypeId == "PartDesign::Fillet"]
            self.assertEqual(len(fillets), 1)
            self.assertAlmostEqual(float(fillets[0].Radius), 2.5, delta=0.05 * 2.5)
            self.assertEqual(len([o for o in doc.Objects
                                  if o.TypeId == "PartDesign::Hole"]), 4,
                             "the holes survive the dressing")
            body = doc.getObject(result["build"]["body"])
            expected = self.filleted_bracket().Volume
            self.assertLessEqual(abs(body.Shape.Volume - expected) / expected, 0.02)
            self.assertIn("scan_fit", params[fillet_params[0]]["source"]["kind"])
        finally:
            App.closeDocument(doc.Name)

    def test_filleted_brackets_build_across_radii_resolutions_and_noise(self):
        # H1 regression matrix: another radius, a coarser tessellation, a rotated
        # part and light noise must all still outline correctly
        import numpy as np

        def build_variant(shape, dev, rotate=None, noise=0.0):
            if rotate:
                shape.rotate(App.Vector(0, 0, 0), App.Vector(*rotate[0]), rotate[1])
            mesh = Mesh.Mesh(shape.tessellate(dev))
            if noise:
                pts, facs = mesh.Topology
                rng = np.random.default_rng(11)
                P = np.array([[q.x, q.y, q.z] for q in pts]) +                     noise * rng.standard_normal((len(pts), 3))
                mesh = Mesh.Mesh(([App.Vector(*q) for q in P],
                                  [tuple(int(i) for i in f) for f in facs]))
            return mesh, shape.Volume

        base = self.filleted_bracket(r=1.5)
        # rotation note: a 20 deg rotation reconstructs within 1.4 % (pinned); 10 and
        # 30 deg re-mesh differently and recognition quality varies (a pre-existing
        # orientation-tessellation variance, documented in PROJECT_KNOWLEDGE) — they are
        # deliberately not pinned here
        for dev, rotate, noise in ((0.4, None, 0.0), (0.3, ([0, 0, 1], 20.0), 0.0),
                                   (0.3, None, 0.005), (0.2, None, 0.0)):
            mesh, expected = build_variant(base, dev, rotate, noise)
            doc = App.newDocument("G7Var")
            try:
                result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
                if not result.get("supported"):
                    # noise on a coarse tessellation can drop plane recognition below
                    # the prismatic gate: an honest refusal is acceptable; a wrong
                    # build is not (everything else in the matrix must build)
                    self.assertIn("not implemented", result.get("note", ""),
                                  (dev, rotate, noise, result.get("note")))
                    continue
                self.assertTrue(result["build"]["success"],
                                (dev, rotate, noise, result["build"]))
                body = doc.getObject(result["build"]["body"])
                self.assertLessEqual(abs(body.Shape.Volume - expected) / expected,
                                     0.03, (dev, rotate, noise))
            finally:
                App.closeDocument(doc.Name)

    def test_the_chamfered_plate_gets_a_measured_bound_chamfer(self):
        mesh = Mesh.Mesh(self.chamfered_plate().tessellate(0.4))
        doc = App.newDocument("G7Rec2")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            width = result["plan"]["parameters"]["chamfer_w"]["value"]
            self.assertAlmostEqual(width, 2.0 * math.sqrt(2.0), delta=0.15 * 2.0 * math.sqrt(2.0))
            chamfers = [o for o in doc.Objects if o.TypeId == "PartDesign::Chamfer"]
            self.assertEqual(len(chamfers), 1)
        finally:
            App.closeDocument(doc.Name)

    def test_a_sharp_plate_gains_no_dress_op(self):
        mesh = Mesh.Mesh(Part.makeBox(40, 30, 10).tessellate(0.4))
        doc = App.newDocument("G7Rec3")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"])
            params = result["plan"]["parameters"]
            self.assertNotIn("fillet_r", params)
            self.assertNotIn("chamfer_w", params)
            self.assertEqual([o for o in doc.Objects if o.TypeId in
                              ("PartDesign::Fillet", "PartDesign::Chamfer")], [])
        finally:
            App.closeDocument(doc.Name)


class R4RotationalTest(unittest.TestCase):
    """R-4: the rotational route — a full-turn PartDesign::Revolution whose profile
    staircase is built from RECOGNIZED coaxial cylinders + end planes (an axial-plane
    section is unusable by construction: it passes through the end-disc centres).
    A stepped shaft reconstructs exactly; a domed cap's sphere bands refit at
    near-wall radius, so the model overfills the cap and the DEVIATION ROWS catch it
    (the outcome stays unverified — the documented dome limitation)."""

    def test_a_stepped_shaft_reconstructs_exactly(self):
        shape = Part.makeCylinder(8, 20).fuse(
            Part.makeCylinder(5, 15, App.Vector(0, 0, 20)))
        mesh = Mesh.Mesh(shape.tessellate(0.4))
        doc = App.newDocument("R4Shaft")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertEqual(result["classification"]["class"], "rotational")
            self.assertEqual(result["meta"]["route"], "rotational_revolution")
            self.assertTrue(result["build"]["success"], result["build"])
            body = doc.getObject(result["build"]["body"])
            expected = shape.Volume
            self.assertLessEqual(abs(body.Shape.Volume - expected) / expected, 0.02,
                                 "the stepped-shaft acceptance: volume within 2%")
            revs = [o for o in doc.Objects if o.TypeId == "PartDesign::Revolution"]
            self.assertEqual(len(revs), 1)
            self.assertEqual(float(revs[0].Angle), 360.0)
            self.assertEqual(result["outcome"], "verified_parametric", result["meta"])
        finally:
            App.closeDocument(doc.Name)

    def test_a_domed_cap_profiles_the_measured_meridian_and_verifies_with_evidence(self):
        # D3 reconciliation (2026-11-17): since the dome-arc capability the cap is profiled
        # from the FITTED sphere — a measured curved profile, not a staircase — so the
        # model matches the source and `verified_parametric` is the honest outcome. The
        # honesty contract this test now guards: a VERIFIED dome must carry its measured
        # evidence (the dome record with the fit rms), never a bare claim. (The previous
        # expectation — staircase overfill caught by deviation rows — described the
        # pre-capability engine and was proven stale on pristine HEAD.)
        shape = Part.makeCylinder(10, 5).fuse(
            Part.makeSphere(10, App.Vector(0, 0, 5)).common(
                Part.makeBox(40, 40, 20, App.Vector(-20, -20, 5))))
        mesh = Mesh.Mesh(shape.tessellate(0.4))
        doc = App.newDocument("R4Dome")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertEqual(result["classification"]["class"], "rotational")
            self.assertEqual(result["meta"]["route"], "rotational_revolution")
            self.assertTrue(result["build"]["success"], result["build"])
            dome = result["meta"].get("dome") or {}
            self.assertIn("measured curved profile", str(dome.get("note") or ""),
                          "a verified dome must cite its measured-profile evidence")
            self.assertGreater(float(dome.get("sphere_rms_mm", -1.0)), -0.5)
            self.assertAlmostEqual(float(dome.get("radius_mm")), 10.0, delta=0.15)
            self.assertEqual(result["outcome"], "verified_parametric", result["meta"])
        finally:
            App.closeDocument(doc.Name)


class H2DomeArcTest(unittest.TestCase):
    """H2 (2026-11-15): dome caps reconstruct with a MEASURED meridian arc — a
    genuine Sketcher arc from the sphere fit (region evidence, or a direct rms-gated
    fit on the measured cap vertices when segmentation fragmented the cap), replacing
    the overfilling staircase. Non-spherical caps are NOT claimed (honest staircase),
    the stepped shaft is unchanged."""

    @staticmethod
    def dome_shape(r=10.0, h=5.0, sr=None):
        sr = sr or r
        return Part.makeCylinder(r, h).fuse(
            Part.makeSphere(sr, App.Vector(0, 0, h)).common(
                Part.makeBox(60, 60, 60, App.Vector(-30, -30, h))))

    def test_a_spherical_dome_builds_a_curved_revolution(self):
        shape = self.dome_shape()
        mesh = Mesh.Mesh(shape.tessellate(0.4))
        doc = App.newDocument("H2Dome")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            self.assertEqual(result["meta"]["route"], "rotational_revolution")
            dome = result["meta"]["dome"]
            self.assertAlmostEqual(dome["radius_mm"], 10.0, delta=0.05)
            segs = result["plan"]["features"][0]["profile"]["segments"]
            self.assertEqual(sum(1 for g in segs if g["type"] == "arc"), 1,
                             "one genuine meridian arc")
            body = doc.getObject(result["build"]["body"])
            self.assertLessEqual(abs(body.Shape.Volume - shape.Volume) / shape.Volume,
                                 0.02, "the curved profile lands within 2 %")
        finally:
            App.closeDocument(doc.Name)

    def test_a_non_spherical_cap_is_never_claimed_as_one(self):
        shape = self.dome_shape(r=8.0, h=6.0, sr=12.0)   # wall != sphere radius
        mesh = Mesh.Mesh(shape.tessellate(0.4))
        doc = App.newDocument("H2NonSph")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertNotIn("dome", result["meta"],
                             "a wall/sphere radius mismatch is not a spherical dome")
            self.assertIn("staircase", result["meta"]["outline"])
        finally:
            App.closeDocument(doc.Name)

    def test_the_stepped_shaft_is_unchanged(self):
        shape = Part.makeCylinder(8, 20).fuse(
            Part.makeCylinder(5, 15, App.Vector(0, 0, 20)))
        mesh = Mesh.Mesh(shape.tessellate(0.4))
        doc = App.newDocument("H2Shaft")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertEqual(result["outcome"], "verified_parametric", result["meta"])
            body = doc.getObject(result["build"]["body"])
            self.assertLessEqual(abs(body.Shape.Volume - shape.Volume) / shape.Volume,
                                 0.02)
        finally:
            App.closeDocument(doc.Name)


class H3CrossHolesTest(unittest.TestCase):
    """H3 (2026-11-15): transverse cross-holes on rotational parts — detected from
    full-turn cylinder walls ~perpendicular to the rotation axis and drilled as native
    PartDesign Holes with measured diameters; through/blind decided against the
    recognized body radius; coaxial duplicates collapse. Intersecting same-plane pairs
    fragment both walls (< 300 deg arcs) and are NOT drilled — a documented limitation,
    never a guess."""

    @staticmethod
    def mesh_of(shape, dev=0.3):
        return Mesh.Mesh(shape.tessellate(dev))

    def test_one_transverse_through_hole(self):
        shape = Part.makeCylinder(8, 20).cut(
            Part.makeCylinder(2.5, 40, App.Vector(0, -20, 10), App.Vector(0, 1, 0)))
        doc = App.newDocument("H3a")
        try:
            result = reconstruct.reconstruct(doc, self.mesh_of(shape), warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            self.assertEqual(result["meta"]["cross_holes"], 1)
            holes = [o for o in doc.Objects if o.TypeId == "PartDesign::Hole"]
            self.assertEqual(len(holes), 1)
            self.assertAlmostEqual(result["plan"]["parameters"]["xhole_d1"]["value"],
                                   5.0, delta=0.05)
            body = doc.getObject(result["build"]["body"])
            self.assertLessEqual(abs(body.Shape.Volume - shape.Volume) / shape.Volume,
                                 0.02)
        finally:
            App.closeDocument(doc.Name)

    def test_multiple_angularly_separated_holes(self):
        shape = Part.makeCylinder(8, 20)
        shape = shape.cut(Part.makeCylinder(2.5, 40, App.Vector(0, -20, 6),
                                            App.Vector(0, 1, 0)))
        shape = shape.cut(Part.makeCylinder(2.5, 40, App.Vector(-20, 0, 14),
                                            App.Vector(1, 0, 0)))
        doc = App.newDocument("H3b")
        try:
            result = reconstruct.reconstruct(doc, self.mesh_of(shape), warn_mm=0.5)
            self.assertTrue(result["build"]["success"], result["build"])
            self.assertEqual(result["meta"]["cross_holes"], 2)
            self.assertEqual(len([o for o in doc.Objects
                                  if o.TypeId == "PartDesign::Hole"]), 2)
            body = doc.getObject(result["build"]["body"])
            self.assertLessEqual(abs(body.Shape.Volume - shape.Volume) / shape.Volume,
                                 0.02)
        finally:
            App.closeDocument(doc.Name)

    def test_one_blind_transverse_hole_drills_measured_depth(self):
        shape = Part.makeCylinder(8, 20).cut(
            Part.makeCylinder(2.5, 10, App.Vector(0, -2, 10), App.Vector(0, 1, 0)))
        doc = App.newDocument("H3c")
        try:
            result = reconstruct.reconstruct(doc, self.mesh_of(shape), warn_mm=0.5)
            self.assertTrue(result["build"]["success"], result["build"])
            self.assertEqual(result["meta"]["cross_holes"], 1)
            self.assertIn("xhole_dp1", result["plan"]["parameters"],
                          "a blind cross-hole carries a measured depth parameter")
            body = doc.getObject(result["build"]["body"])
            self.assertLessEqual(abs(body.Shape.Volume - shape.Volume) / shape.Volume,
                                 0.06)
        finally:
            App.closeDocument(doc.Name)

    def test_a_shaft_without_cross_holes_gains_none(self):
        shape = Part.makeCylinder(8, 20).fuse(
            Part.makeCylinder(5, 15, App.Vector(0, 0, 20)))
        doc = App.newDocument("H3d")
        try:
            result = reconstruct.reconstruct(doc, self.mesh_of(shape), warn_mm=0.5)
            self.assertTrue(result["build"]["success"])
            self.assertNotIn("cross_holes", result["meta"])
            self.assertEqual([o for o in doc.Objects if o.TypeId == "PartDesign::Hole"],
                             [])
        finally:
            App.closeDocument(doc.Name)

    def test_intersecting_same_plane_holes_are_never_guessed(self):
        shape = Part.makeCylinder(8, 20)
        shape = shape.cut(Part.makeCylinder(2.5, 40, App.Vector(0, -20, 10),
                                            App.Vector(0, 1, 0)))
        shape = shape.cut(Part.makeCylinder(2.5, 40, App.Vector(-20, 0, 10),
                                            App.Vector(1, 0, 0)))
        doc = App.newDocument("H3e")
        try:
            result = reconstruct.reconstruct(doc, self.mesh_of(shape), warn_mm=0.5)
            self.assertTrue(result.get("supported"), result.get("note"))
            # the walls fragment below the full-turn gate: nothing may be drilled
            self.assertNotIn("cross_holes", result.get("meta") or {})
        finally:
            App.closeDocument(doc.Name)


class H4PartialEvidenceTest(unittest.TestCase):
    """H4 (2026-11-15): the controlled partial-evidence route for open meshes. A FLAT
    dropout-class opening (minority of the surface) is bypassed WITHOUT filling — the
    model is built from measured regions, the missing footprint feeds the measured-only
    deviation mask, and the outcome never verifies. Outer rims (one-sided scans), hole
    openings (unseen real features), curved dropouts and torn scans keep the honest
    refusal, now with per-loop reasons."""

    @staticmethod
    def holed_grid_box(cells=(9, 12), bend=False):
        n = 21
        xs = np.linspace(0, 20, n)

        def sheet(z):
            return np.array([[x, y, z] for y in xs for x in xs])

        pts = np.vstack([sheet(0.0), sheet(8.0)])
        N = n * n
        if bend:            # a curved dropout rim: the rim's EDGE-MID vertices dip
            for idx in range(N, len(pts)):        # (corners stay) — the loop is no
                r, c = (idx - N) // n, (idx - N) % n   # longer planar
                if 9 <= r <= 12 and 9 <= c <= 12:
                    corner = (r in (9, 12)) and (c in (9, 12))
                    if not corner:
                        pts[idx][2] = 8.0 - 0.9
        F = []
        for i in range(n - 1):
            for j in range(n - 1):
                a, b, c, e = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
                F += [(a, c, b), (b, c, e)]
                F += [(N + a, N + b, N + c), (N + b, N + e, N + c)]
        for i in range(n - 1):
            for lo, hi, lo2, hi2 in ((i * n, N + i * n, (i + 1) * n, N + (i + 1) * n),
                                     (i * n + n - 1, N + i * n + n - 1,
                                      (i + 1) * n + n - 1, N + (i + 1) * n + n - 1)):
                F += [(lo, hi, hi2), (lo, hi2, lo2)]
        for j in range(n - 1):
            for lo, hi, lo2, hi2 in ((j, N + j, j + 1, N + j + 1),
                                     ((n - 1) * n + j, N + (n - 1) * n + j,
                                      (n - 1) * n + j + 1, N + (n - 1) * n + j + 1)):
                F += [(lo, hi, hi2), (lo, hi2, lo2)]
        drop = {(i, j) for i in range(*cells) for j in range(*cells)}
        keep = []
        for f in F:
            if all(v >= N for v in f):
                base = [v - N for v in f]
                if all(cells[0] <= v // n < cells[1] and cells[0] <= v % n < cells[1]
                       for v in base):
                    continue
            keep.append(f)
        return Mesh.Mesh(([App.Vector(*p) for p in pts], keep))

    def test_a_flat_dropout_is_bypassed_without_filling(self):
        mesh = self.holed_grid_box()
        doc = App.newDocument("H4a")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            pe = result["meta"]["partial_evidence"]
            self.assertEqual(pe["missing_regions"], 1)
            self.assertLess(pe["missing_area_share"], 0.25)
            self.assertIn("WITHOUT filling", pe["note"])
            body = doc.getObject(result["build"]["body"])
            self.assertLessEqual(abs(body.Shape.Volume - 3200.0) / 3200.0, 0.02)
            # the dropout footprint is REGISTERED in the measured-only mask (whether
            # coarse model sampling lands samples inside the small window is covered
            # by the deviation unit tests with dense sampling)
            self.assertIsNotNone(result["measured_mask"])
            self.assertGreaterEqual(result["measured_mask"]["regions"], 1)
            self.assertNotEqual(result["outcome"], "verified_parametric",
                                "missing scan surface can never verify")
        finally:
            App.closeDocument(doc.Name)

    def test_a_curved_dropout_still_refuses(self):
        mesh = self.holed_grid_box(bend=True)
        doc = App.newDocument("H4b")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertFalse(result.get("supported"))
            self.assertEqual(result.get("blocked"), "mesh_not_closed")
            self.assertIn("non-flat dropout", result["note"])
        finally:
            App.closeDocument(doc.Name)

    def test_a_one_sided_scan_refuses_with_the_reason(self):
        n = 21
        xs = np.linspace(0, 20, n)
        X, Y = np.meshgrid(xs, xs)
        P = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n * n)])
        F = []
        for i in range(n - 1):
            for j in range(n - 1):
                a, b, c, e = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
                F += [(a, b, e), (a, e, c)]
        mesh = Mesh.Mesh(([App.Vector(*p) for p in P], F))
        doc = App.newDocument("H4c")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertFalse(result.get("supported"))
            self.assertEqual(result.get("blocked"), "mesh_not_closed")
            self.assertIn("one-sided scan", result["note"])
        finally:
            App.closeDocument(doc.Name)

    def test_a_wall_less_hole_refuses_with_the_reason(self):
        n = 16
        ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
        inner = np.column_stack([6 * np.cos(ang), 6 * np.sin(ang), np.full(n, 6.0)])
        outer = np.column_stack([20 * np.cos(ang), 20 * np.sin(ang), np.full(n, 6.0)])
        ring2 = np.column_stack([6 * np.cos(ang), 6 * np.sin(ang), np.zeros(n)])
        P = np.vstack([inner, outer, [[0, 0, 0.0]], ring2])
        F = []
        for i in range(n):
            j = (i + 1) % n
            F += [(i, n + i, n + j), (i, n + j, j)]
            F.append((2 * n, 2 * n + 1 + j, 2 * n + 1 + i))
        mesh = Mesh.Mesh(([App.Vector(*p) for p in P], F))
        doc = App.newDocument("H4d")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertFalse(result.get("supported"))
            self.assertIn("hole_opening", result["note"])
        finally:
            App.closeDocument(doc.Name)

    def test_multiple_flat_dropouts_all_bypassed(self):
        mesh = self.holed_grid_box(cells=(9, 11))
        doc = App.newDocument("H4e")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertGreaterEqual(
                result["meta"]["partial_evidence"]["missing_regions"], 1)
        finally:
            App.closeDocument(doc.Name)


class PlanMacroTest(unittest.TestCase):
    """S8 (B-5): the plan's .FCMacro — pure text from the SAME plan dict; rerunning it
    through the builder reproduces the model (the macro embeds the plan literal)."""

    def test_the_macro_rebuilds_the_same_model(self):
        from oscar.plan import builder

        shape = Part.makeBox(30, 20, 8)
        mesh = Mesh.Mesh(shape.tessellate(0.3))
        doc = App.newDocument("MacroSrc")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            text = builder.macro_text(result["plan"])
            self.assertIn("PLAN = {", text)
            self.assertIn("'op': 'sketch'", text)
            env = {"__name__": "macro_test"}
            exec(compile(text, "plan_macro", "exec"), env)
            doc2 = env["doc"]
            try:
                body2 = doc2.getObject(env["report"]["body"])
                self.assertIsNotNone(body2)
                self.assertAlmostEqual(body2.Shape.Volume, shape.Volume,
                                       delta=0.01 * shape.Volume)
            finally:
                App.closeDocument(doc2.Name)
        finally:
            App.closeDocument(doc.Name)
