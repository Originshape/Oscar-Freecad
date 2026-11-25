# SPDX-License-Identifier: Apache-2.0
"""General accuracy gates: a VALID recompute is not proof of reconstruction.
(A) the exact residual maxima gate the verified outcome whenever they were COMPUTED
    (a sampled lower-bound Hausdorff alone can miss a small region);
(B) recognized-but-unrealized features (reconciliation `unexplained`) cap the
    outcome at partially verified — the tree does not represent the whole source;
(C) registration is a diagnostic: ALIGNED rows can pass the very gate the RAW rows
    fail (proven at the boundary), and a real reconstruct() feeds the gate exactly
    the RAW rows — never an aligned one (issue #13's cannot-falsey-upgrade rule)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

from oscar.recon.reconstruct import _geometric_ok  # noqa: E402


class GeometricOkTest(unittest.TestCase):
    def test_sampled_hausdorff_within_warn_passes(self):
        rows = [{"id": "hausdorff", "value": 0.2}]
        self.assertTrue(_geometric_ok(0.2, rows, warn_mm=0.3))

    def test_sampled_hausdorff_above_warn_fails(self):
        self.assertFalse(_geometric_ok(0.4, [], warn_mm=0.3))

    def test_a_computed_exact_max_above_warn_blocks_verification(self):
        rows = [{"id": "hausdorff", "value": 0.1},
                {"id": "exact_scan_to_model.max", "value": 0.44, "status": "WARN"},
                {"id": "exact_model_to_scan.max", "value": 0.05, "status": "PASS"}]
        self.assertFalse(_geometric_ok(0.1, rows, warn_mm=0.3))

    def test_both_directions_are_required_when_computed(self):
        rows = [{"id": "hausdorff", "value": 0.1},
                {"id": "exact_scan_to_model.max", "value": 0.05, "status": "PASS"},
                {"id": "exact_model_to_scan.max", "value": 0.9, "status": "WARN"}]
        self.assertFalse(_geometric_ok(0.1, rows, warn_mm=0.3))

    def test_a_skipped_or_missing_exact_row_never_gates(self):
        rows = [{"id": "hausdorff", "value": 0.1},
                {"id": "exact_scan_to_model.max", "value": None, "status": "SKIP"}]
        self.assertTrue(_geometric_ok(0.1, rows, warn_mm=0.3))
        self.assertTrue(_geometric_ok(0.1, [], warn_mm=0.3))

    def test_exact_within_warn_still_passes(self):
        rows = [{"id": "hausdorff", "value": 0.2},
                {"id": "exact_scan_to_model.max", "value": 0.29, "status": "WARN"},
                {"id": "exact_model_to_scan.max", "value": 0.21, "status": "PASS"}]
        self.assertTrue(_geometric_ok(0.2, rows, warn_mm=0.3))


class AlignedNeverUpgradesTest(unittest.TestCase):
    """Issue #13: registration must never be able to upgrade the reconstruction status.
    (A) at the decision boundary: a rigidly misaligned scan/shape pair whose ALIGNED rows
        PASS `_geometric_ok` while its RAW rows FAIL it — the hazard is real;
    (B) through a REAL `reconstruct()`: the outcome gate (spied) consumes exactly the
        stored RAW rows, and no row it sees is an aligned row."""

    @staticmethod
    def _misaligned_mesh(shape, deg, shift):
        import math

        import FreeCAD as App
        import Mesh
        import numpy as np

        th = math.radians(deg)
        k = np.array([0.0, 1.0, 0.0])
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
        base = Mesh.Mesh(shape.tessellate(0.4))
        pts = np.array([[p.x, p.y, p.z] for p in base.Topology[0]])
        moved = pts @ R.T + np.asarray(shift, dtype=float)
        return Mesh.Mesh(([App.Vector(*p) for p in moved],
                          [tuple(f) for f in base.Topology[1]]))

    def test_aligned_rows_pass_the_gate_the_raw_rows_fail(self):
        import FreeCAD as App  # noqa: F401 - Part/Mesh resolve after the runtime is up
        import Part

        from oscar.recon import deviation

        shape = Part.makeBox(60, 40, 10)
        mesh = self._misaligned_mesh(shape, 2.0, (0.5, -0.3, 0.2))
        # warn 1.5 sits BETWEEN the raw (~1.97 mm) and aligned (~1.2 mm, mesh-grid
        # discreteness floor) hausdorff — any warn that separates them proves the hazard
        out = deviation.registered_deviation(mesh, shape, warn_mm=1.5)
        by_id = lambda rows: {r["id"]: r for r in rows}  # noqa: E731
        raw, aligned = by_id(out["raw"]), by_id(out["aligned"])
        self.assertTrue(out["registration"]["valid"], out["registration"]["warnings"])
        self.assertGreater(raw["hausdorff"]["value"], 1.5)       # the raw bias is real
        self.assertLess(aligned["hausdorff"]["value"], 1.5)
        self.assertFalse(_geometric_ok(raw["hausdorff"]["value"], out["raw"], warn_mm=1.5))
        self.assertTrue(_geometric_ok(aligned["hausdorff"]["value"], out["aligned"],
                                      warn_mm=1.5))               # aligned WOULD verify —
        # ...which is exactly why the outcome must never be fed these rows

    def test_the_outcome_gate_consumes_exactly_the_raw_rows(self):
        from unittest import mock

        import FreeCAD as App
        import Mesh
        import Part

        from oscar.recon import reconstruct as rc

        box = Part.makeBox(60, 40, 10).cut(
            Part.makeCylinder(3.3, 30, App.Vector(15, 10, -10)))
        mesh = Mesh.Mesh(box.tessellate(0.4))
        seen = []
        original = rc._geometric_ok

        def spy(*args, **kwargs):
            seen.append((args, kwargs))
            return original(*args, **kwargs)

        doc = App.newDocument("AlignedNeverUpgrades")
        try:
            with mock.patch.object(rc, "_geometric_ok", spy):
                result = rc.reconstruct(doc, mesh, warn_mm=0.3)
        finally:
            App.closeDocument(doc.Name)
        self.assertTrue(seen, "the outcome gate never ran")
        for args, _kwargs in seen:                               # primary gate AND any M1 re-gate
            self.assertEqual(args[1], result["deviation"])        # exactly the stored rows
            for row in args[1]:
                self.assertNotIn("AFTER ICP", row.get("detail", ""))   # none is an aligned row


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
