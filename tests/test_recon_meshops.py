"""M2: ``oscar.recon.meshops`` — diagnostics and copy-based repair.

Defects are manufactured the way real exports produce them (duplicated vertices, flipped
windings, a hole where facets went missing, a zero-area triangle); the repair contract is the
master spec's: the original is never touched, safe operations always run, gated operations only
on authorization, and everything is recorded. Runs under freecadcmd on real meshes.
"""

import os
import sys
import unittest
import unittest.mock as mock  # noqa: F401  (K2 failure-injection test)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import MeshPart  # noqa: E402
import Part  # noqa: E402

from oscar.recon import meshops  # noqa: E402


def dirty_mesh():
    """A 30 mm cube tessellation with 6 duplicated vertices, flipped windings on 7 facets, a
    hole where every facet at vertex 0 went missing, and one zero-area triangle."""
    pts, facs = Part.makeBox(30, 30, 30).tessellate(4.0)
    pts = list(pts)
    facs = [tuple(f) for f in facs]
    base = len(pts)
    pts += [App.Vector(p) for p in pts[:6]]
    for i, f in enumerate(facs):
        if i < 12:
            f = list(f)
            for j in range(3):
                if f[j] < 6:
                    f[j] = base + f[j]
            facs[i] = tuple(f)
    for i in range(5, min(12, len(facs))):
        a, b, c = facs[i]
        facs[i] = (a, c, b)
    facs = [f for f in facs if 0 not in f]
    facs.append((1, 1, 2))
    return Mesh.Mesh((pts, facs))


class DiagnoseTest(unittest.TestCase):
    def test_defects_are_reported_and_the_mesh_is_untouched(self):
        mesh = dirty_mesh()
        before = (int(mesh.CountFacets), int(mesh.CountPoints))
        report = meshops.diagnose(mesh)
        self.assertFalse(report["closed"])
        self.assertTrue(report["inconsistent_normals"])
        self.assertGreaterEqual(report["duplicate_points"], 6)
        self.assertGreaterEqual(report["degenerate_facets"], 1)
        self.assertEqual((int(mesh.CountFacets), int(mesh.CountPoints)), before,
                         "diagnose must never change the mesh")

    def test_a_clean_mesh_reports_clean(self):
        mesh = Mesh.Mesh(Part.makeCylinder(10, 20).tessellate(0.5))
        report = meshops.diagnose(mesh)
        self.assertTrue(report["closed"])
        self.assertFalse(report["non_manifold"] or report["self_intersections"]
                         or report["inconsistent_normals"])
        self.assertEqual((report["duplicate_points"], report["duplicate_facets"],
                          report["degenerate_facets"]), (0, 0, 0))
        self.assertIn("volume_mm3", report)
        self.assertEqual(report["components"], 1)


class RepairTest(unittest.TestCase):
    def test_safe_operations_restore_the_copy_and_record_every_step(self):
        mesh = dirty_mesh()
        original = (int(mesh.CountFacets), int(mesh.CountPoints))
        result = meshops.repair(mesh)
        self.assertTrue(result["resolved"]["opened_to_closed"], result["unresolved"])
        self.assertEqual([a["operation"] for a in result["actions"]],
                         list(meshops.SAFE_OPERATIONS))
        self.assertTrue(all("facets_after" in a for a in result["actions"]))
        self.assertEqual(sorted(result["refused"]), sorted(meshops.GATED_OPERATIONS))
        self.assertEqual((int(mesh.CountFacets), int(mesh.CountPoints)), original,
                         "repair works on a copy; the original must stay untouched")
        self.assertTrue(result["after"]["closed"])
        self.assertFalse(result["after"]["inconsistent_normals"])

    def test_gated_operations_run_only_when_authorized(self):
        mesh = dirty_mesh()
        result = meshops.repair(mesh, authorized=["removeNonManifolds"])
        ops = [a["operation"] for a in result["actions"]]
        self.assertIn("removeNonManifolds", ops)
        self.assertNotIn("removeNonManifolds", result["refused"])
        self.assertIn("fillupHoles", result["refused"])

    def test_safe_operations_are_validated_and_accepted_on_this_fixture(self):
        result = meshops.repair(dirty_mesh())
        self.assertTrue(result["validation"])
        self.assertEqual({v["verdict"] for v in result["validation"]}, {"ACCEPTED"},
                         [v for v in result["validation"]])
        self.assertTrue(all("verdict" in a for a in result["actions"]))


class K2BoundaryTest(unittest.TestCase):
    def cube(self, tess=2.0):
        pts, facs = Part.makeBox(20, 20, 20).tessellate(tess)
        return Mesh.Mesh((list(pts), [tuple(f) for f in facs]))

    def test_closed_mesh_has_no_boundary(self):
        b = meshops.boundary(self.cube())
        self.assertEqual((b["edges"], b["loops"], b["open_chains"]), (0, [], []))

    def test_one_hole_is_one_closed_loop_with_size_and_position(self):
        pts, facs = Part.makeBox(20, 20, 5).tessellate(2.0)
        facs = [tuple(f) for f in facs][:-2]                      # punch a small hole
        m = Mesh.Mesh((list(pts), facs))
        b = meshops.boundary(m)
        self.assertEqual(len(b["loops"]), 1)
        loop = b["loops"][0]
        self.assertGreater(loop["perimeter_mm"], 0.0)
        self.assertTrue(all(0.0 <= c <= 20.0 for c in loop["centroid_mm"]),
                        f"loop centroid outside the part: {loop}")

    def test_two_holes_are_two_loops(self):
        pts, facs = Part.makeBox(20, 20, 5).tessellate(1.0)
        facs = [tuple(f) for f in facs]
        gone = set(range(0, 3)) | set(range(len(facs) - 3, len(facs)))
        facs = [f for i, f in enumerate(facs) if i not in gone]
        m = Mesh.Mesh((list(pts), facs))
        self.assertEqual(len(meshops.boundary(m)["loops"]), 2)

    def test_junction_boundaries_report_open_chains(self):
        pts = [App.Vector(0, 0, 0), App.Vector(10, 0, 0), App.Vector(0, 10, 0),
               App.Vector(10, -10, 0), App.Vector(20, -10, 0), App.Vector(20, -20, 0),
               App.Vector(10, -20, 0), App.Vector(0, -20, 0)]
        facs = [(0, 1, 2), (0, 3, 4), (0, 5, 6), (0, 7, 3)]
        b = meshops.boundary(Mesh.Mesh((pts, facs)))
        self.assertGreaterEqual(b["junctions"], 1)
        self.assertGreaterEqual(len(b["open_chains"]) + len(b["loops"]), 1)


class K2DiagnoseTest(unittest.TestCase):
    def test_components_are_counted_and_detailed_without_deleting_anything(self):
        pts, facs = Part.makeBox(20, 20, 20).tessellate(2.0)
        far = Part.makeBox(5, 5, 5).tessellate(2.0)
        base = len(pts)
        pts = list(pts) + [App.Vector(p.x + 50, p.y, p.z) for p in far[0]]
        facs = [tuple(f) for f in facs] + [tuple(base + i for i in f) for f in far[1]]
        m = Mesh.Mesh((pts, facs))
        before = int(m.CountFacets)
        report = meshops.diagnose(m)
        self.assertEqual(report["components"], 2)
        self.assertEqual(len(report["components_detail"]), 2)
        self.assertEqual(int(m.CountFacets), before, "diagnose deletes nothing")
        self.assertTrue(all(c["size_mm"] for c in report["components_detail"]))

    def test_a_tiny_isolated_component_is_a_candidate_not_a_removal(self):
        pts = [App.Vector(0, 0, 0), App.Vector(10, 0, 0), App.Vector(0, 10, 0),
               App.Vector(50, 0, 0), App.Vector(51, 0, 0), App.Vector(50, 1, 0)]
        m = Mesh.Mesh((pts, [(0, 1, 2), (3, 4, 5)]))
        report = meshops.diagnose(m)
        self.assertIn("tiny_components", report)
        self.assertTrue(any("candidate" in w.lower() for w in report["warnings"]))
        self.assertEqual(int(m.CountFacets), 2, "nothing was deleted")

    def test_adjacent_triangles_are_not_self_intersections(self):
        m = Mesh.Mesh([(App.Vector(0, 0, 0), App.Vector(10, 0, 0), App.Vector(5, 5, 0)),
                       (App.Vector(10, 0, 0), App.Vector(0, 0, 0), App.Vector(5, -5, 0))])
        report = meshops.diagnose(m)
        self.assertEqual(report["self_intersection_pairs_count"], 0)
        self.assertFalse(report["self_intersections"])

    def test_large_mesh_skips_expensive_intersection_query_explicitly(self):
        m = Mesh.Mesh(Part.makeCylinder(10, 20).tessellate(0.5))
        with mock.patch.object(meshops, "MAX_SELF_INTERSECTION_FACETS", 0):
            report = meshops.diagnose(m)
        self.assertIsNone(report["self_intersections"])
        self.assertIsNone(report["self_intersection_pairs_count"])
        self.assertTrue(any("self-intersection check skipped" in note
                            for note in report["limitations"]))

    def test_genuinely_interpenetrating_boxes_report_localized_pairs(self):
        b = Part.makeBox(10, 10, 10)
        m1 = MeshPart.meshFromShape(Shape=b, LinearDeflection=1.0)
        m2 = MeshPart.meshFromShape(Shape=b, LinearDeflection=1.0)
        m2.translate(5, 5, 5)
        both = Mesh.Mesh()
        both.addMesh(m1)
        both.addMesh(m2)
        report = meshops.diagnose(both)
        self.assertGreater(report["self_intersection_pairs_count"], 0)
        self.assertTrue(report["self_intersection_pairs"], "localized pairs must be listed")
        i, j = report["self_intersection_pairs"][0]
        self.assertNotEqual(i, j)

    def test_sliver_small_and_repeated_index_facets_are_counted(self):
        # array-level math (the kernel's constructor silently drops hand-built degenerate
        # facets — verified 2026-11-14 — so the math is proven on raw arrays)
        import numpy as np

        pts = np.array([[0, 0, 0], [10, 0, 0], [5, 0.001, 0],   # sliver apex
                        [0, 0, 5], [1, 0, 0]], dtype=float)
        fac = np.array([[0, 1, 2],                               # sliver (min angle ~0°)
                        [0, 1, 3],                               # healthy
                        [1, 1, 2],                               # repeated index → zero area
                        [0, 3, 4]], dtype=int)                   # small (vs median area)
        q = meshops._facet_quality_arrays(pts, fac)
        self.assertGreaterEqual(q["sliver_facets"], 1)
        self.assertGreaterEqual(q["repeated_index_facets"], 1)
        self.assertGreaterEqual(q["zero_area_facets"], 1)
        self.assertEqual(q["small_facets"], 1,
                         "only the sliver is below 1% of the median facet area")

    def test_mesh_level_counts_on_the_dirty_fixture(self):
        q = meshops.diagnose(dirty_mesh())
        self.assertGreaterEqual(q["repeated_index_facets"], 1)
        self.assertGreaterEqual(q["zero_area_facets"], 1)

    def test_diagnose_is_deterministic_and_read_only_on_a_real_tessellation(self):
        sph = Part.makeSphere(10).tessellate(0.5)
        m = Mesh.Mesh((list(sph[0]), [tuple(f) for f in sph[1]]))
        a = meshops.diagnose(m)
        b = meshops.diagnose(m)
        self.assertEqual(a, b)
        self.assertEqual((int(m.CountFacets), int(m.CountPoints)),
                         (len(sph[1]), len(sph[0])))


class K2ValidationTest(unittest.TestCase):
    def _cube_snap(self, tess=1.0):
        pts, facs = Part.makeBox(20, 20, 20).tessellate(tess)
        m = Mesh.Mesh((list(pts), [tuple(f) for f in facs]))
        return m, meshops._snapshot(m)

    def test_a_geometry_destroying_change_is_rejected(self):
        mesh, before = self._cube_snap(1.0)
        tp, tf = mesh.Topology
        damaged = Mesh.Mesh((list(tp), [tuple(f) for i, f in enumerate(tf)
                                        if i not in (0, 1)]))
        verdict, reasons = meshops._validate_transition("fixDegenerations", before,
                                                        meshops._snapshot(damaged),
                                                        0.01, 1.5)
        self.assertEqual(verdict, "REJECTED")
        self.assertTrue(any("area" in r for r in reasons), reasons)

    def test_vertex_movement_is_rejected_by_the_repair_loop(self):
        mesh = dirty_mesh()
        real = meshops._apply_op

        def secretly_smooths(m, op, mean_edge=None):
            out = real(m, op, mean_edge)
            if op == "fixIndices":
                m.smooth()                # a "cleanup" that actually moves vertices
            return out

        with mock.patch.object(meshops, "_apply_op", secretly_smooths):
            result = meshops.repair(mesh)
        rec = next(v for v in result["validation"] if v["operation"] == "fixIndices")
        self.assertEqual(rec["verdict"], "REJECTED")
        self.assertTrue(any("moved" in r for r in rec["reasons"]), rec["reasons"])

    def test_a_clean_noop_is_accepted(self):
        mesh, before = self._cube_snap()
        verdict, _ = meshops._validate_transition("harmonizeNormals", before,
                                                  meshops._snapshot(mesh.copy()),
                                                  0.01, 1.5)
        self.assertEqual(verdict, "ACCEPTED")

    def test_disappearing_components_need_review(self):
        _, before = self._cube_snap()
        after = dict(before, components=before["components"] - 1)
        verdict, reasons = meshops._validate_transition("removeNonManifolds", before, after,
                                                        0.01, 1.5)
        self.assertEqual(verdict, "NEEDS_REVIEW")
        self.assertTrue(any("disappeared" in r for r in reasons))

    def test_open_mesh_volume_is_not_used_as_a_damage_metric(self):
        pts, facs = Part.makeBox(20, 20, 5).tessellate(2.0)
        open_mesh = Mesh.Mesh((list(pts), [tuple(f) for f in facs][:-2]))
        before = meshops._snapshot(open_mesh)
        self.assertNotIn("volume_mm3", before)
        verdict, reasons = meshops._validate_transition("fixDegenerations", before, before,
                                                        0.01, 1.5)
        self.assertTrue(any("volume not compared" in r for r in reasons), reasons)

    def test_a_failing_operation_is_recorded_failed_and_the_mesh_keeps_its_state(self):
        mesh = dirty_mesh()
        original = meshops._snapshot(mesh)
        real = meshops._apply_op

        def boom(m, op, mean_edge=None):
            if op == "fixIndices":
                raise RuntimeError("kernel exploded")
            return real(m, op, mean_edge)

        with mock.patch.object(meshops, "_apply_op", boom):
            result = meshops.repair(mesh)
        failed = [v for v in result["validation"] if v["verdict"] == "FAILED"]
        self.assertEqual([v["operation"] for v in failed], ["fixIndices"])
        self.assertTrue(any("error" in a for a in result["actions"]))
        # every other operation still ran and validated on the live working copy
        self.assertIn("fixDegenerations", [a["operation"] for a in result["actions"]])
        self.assertEqual(meshops._snapshot(mesh)["facets"], original["facets"],
                         "the ORIGINAL mesh never changes")


class K2SyntheticTest(unittest.TestCase):
    def holed_box(self):
        # G6: the fill policy fills only tiny flat pinholes — the old coarse-box fixture's
        # hole (>= 2% of the area, unscanned-face class) is now SKIPPED by design, so the
        # synthetic-recording fixture is a fine grid sheet with one quad missing
        import numpy as np

        n = 21
        xs = np.linspace(0.0, 20.0, n)
        X, Y = np.meshgrid(xs, xs)
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n * n)])
        facs = []
        for i in range(n - 1):
            for j in range(n - 1):
                if (i, j) == (10, 10):
                    continue
                a, b, c, e = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
                facs += [(a, b, e), (a, e, c)]
        return Mesh.Mesh(([App.Vector(*p) for p in pts], facs))

    def test_a_filled_hole_is_marked_synthetic_and_the_original_is_untouched(self):
        mesh = self.holed_box()
        before = (int(mesh.CountFacets), int(mesh.CountPoints))
        result = meshops.repair(mesh, authorized=["fillupHoles"])
        self.assertEqual((int(mesh.CountFacets), int(mesh.CountPoints)), before)
        self.assertTrue(result["synthetic"], "the fill must be recorded as synthetic")
        rec = result["synthetic"][0]
        self.assertEqual(rec["operation"], "fillupHoles")
        self.assertGreaterEqual(rec["count"], 1)
        self.assertIn("NOT measured scan evidence", rec["note"])
        verdict = next(v for v in result["validation"] if v["operation"] == "fillupHoles")
        self.assertIn(verdict["verdict"], ("ACCEPTED", "NEEDS_REVIEW"))
        self.assertGreaterEqual(result["mesh"].countSegments(), 1,
                                "the fill is registered as a session segment")

    def test_synthetic_provenance_persists_as_a_property_but_segments_do_not(self):
        # verified limitation (2026-11-14): FreeCAD mesh segments do NOT survive save/reopen;
        # the durable record is the object property the tool stamps. Executable spec:
        import json
        import os
        import tempfile

        doc = App.newDocument("k2syn")
        obj = doc.addObject("Mesh::Feature", "Repaired")
        kernel_mesh = self.holed_box()
        kernel_mesh.addSegment([0, 1])     # segments live on the KERNEL mesh …
        obj.Mesh = kernel_mesh             # … the object's .Mesh property is an immutable view
        obj.addProperty("App::PropertyString", "OscarSyntheticRegions", "Oscar",
                        "repair-created regions")
        obj.OscarSyntheticRegions = json.dumps(
            [{"operation": "fillupHoles", "facet_count": 2}])
        path = os.path.join(tempfile.mkdtemp(prefix="k2syn_"), "syn.FCStd")
        doc.saveAs(path)
        App.closeDocument(doc.Name)
        d2 = App.openDocument(path)
        back = d2.getObject("Repaired")
        self.assertEqual(json.loads(back.OscarSyntheticRegions),
                         [{"operation": "fillupHoles", "facet_count": 2}])
        self.assertEqual(back.Mesh.countSegments(), 0,
                         "documented limitation: segments are session-only")
        App.closeDocument(d2.Name)


class G6FillPolicyTest(unittest.TestCase):
    """G6: the loop-aware hole-fill authorization matrix. FILLABLE loops are tiny
    (<= 8 edges) pinhole-class dropouts in a flat neighbourhood; an authorized
    ``fillupHoles`` runs only when one exists, closes only what the policy allows
    (a closure of a forbidden loop rejects the whole trial), and larger dropouts come
    back as ``missing_regions`` — never auto-filled."""

    @staticmethod
    def sheet(size=20.0, n=21, drop_cells=()):
        """An open grid sheet with the given quad cells removed (each cell = 2 facets)."""
        import numpy as np

        xs = np.linspace(0.0, size, n)
        X, Y = np.meshgrid(xs, xs)
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n * n)])
        facs = []
        for i in range(n - 1):
            for j in range(n - 1):
                if (i, j) in drop_cells:
                    continue
                a, b, c, e = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
                facs += [(a, b, e), (a, e, c)]
        return Mesh.Mesh(([App.Vector(*p) for p in pts], facs))

    @staticmethod
    def block(i0, j0, k=1):
        return {(i0 + di, j0 + dj) for di in range(k) for dj in range(k)}

    def _fill(self, result):
        return next(a for a in result["actions"] if a["operation"] == "fillupHoles")

    def test_authorized_fill_closes_only_the_pinhole(self):
        m = self.sheet(drop_cells={(10, 10)})            # one 1-quad pinhole + the rim
        r = meshops.repair(m, authorized=("fillupHoles",))
        action = self._fill(r)
        self.assertEqual(action["verdict"], "ACCEPTED", r["validation"])
        self.assertEqual(action["policy"], {"fillable": 1, "non_fillable": 1})
        self.assertGreaterEqual(sum(s["count"] for s in r["synthetic"]), 1)
        self.assertEqual(r["missing_regions"], [], r["missing_regions"])
        self.assertEqual(int(m.CountFacets), 798, "the original keeps its facets")
        self.assertEqual(int(r["mesh"].CountFacets), 800,
                         "the repaired COPY carries the fill's two facets")

    def test_unauthorized_fill_is_refused(self):
        r = meshops.repair(self.sheet(drop_cells={(10, 10)}))
        self.assertIn("fillupHoles", r["refused"])
        self.assertFalse(r["synthetic"])

    def test_no_fillable_loop_skips_the_authorized_fill(self):
        m = self.sheet(drop_cells=self.block(9, 9, 3))   # a 12-edge dropout: not fillable
        r = meshops.repair(m, authorized=("fillupHoles",))
        action = self._fill(r)
        self.assertEqual(action["verdict"], "SKIPPED")
        self.assertIn("G6 fill policy", action["skipped"])
        self.assertEqual(r["after"]["facets"], r["before"]["facets"],
                         "nothing may change when the policy skips the operation")
        self.assertEqual(len(r["missing_regions"]), 1)
        region = r["missing_regions"][0]
        self.assertEqual((region["kind"], region["detail"]), ("dropout", "unscanned_face"))
        self.assertEqual(region["n_edges"], 12)
        self.assertTrue(all(0 <= i < int(r["mesh"].CountPoints) for i in region["ring"]))

    def test_a_forbidden_closure_rejects_the_trial_wholesale(self):
        # The stock kernel cannot bridge the loops this policy forbids (measured), so the
        # REJECTED safety net is driven through a seam: a fill that closes everything —
        # exactly what a future/other kernel arg might do without the policy check.
        m = self.sheet(size=10.0, n=11,
                       drop_cells={(7, 7)} | self.block(2, 2, 2))
        real_apply = meshops._apply_op

        def close_everything(mesh, op, mean_edge=None):
            if op != "fillupHoles":
                return real_apply(mesh, op, mean_edge)
            mesh.fillupHoles(200)
            return {"max_edge": 200}

        with mock.patch.object(meshops, "_apply_op", side_effect=close_everything):
            r = meshops.repair(m, authorized=("fillupHoles",))
        verdict = next(v for v in r["validation"] if v["operation"] == "fillupHoles")
        self.assertEqual(verdict["verdict"], "REJECTED", verdict)
        self.assertTrue(any("policy forbids" in s for s in verdict["reasons"]),
                        verdict["reasons"])
        self.assertFalse(r["synthetic"], "a rejected trial must leave no synthetic record")
        self.assertEqual(r["after"]["facets"], r["before"]["facets"],
                         "the trial is discarded — the mesh keeps the last good state")

    def test_junction_boundaries_are_handled_conservatively(self):
        # two holes touching at one corner vertex: a boundary junction — the walk is
        # conservative there and the policy must neither crash nor invent a fill
        m = self.sheet(drop_cells={(10, 10), (11, 11)})
        r = meshops.repair(m, authorized=("fillupHoles",))
        fill = self._fill(r)
        self.assertIn(fill["verdict"], ("ACCEPTED", "SKIPPED", "NEEDS_REVIEW"))
        self.assertIsInstance(r["missing_regions"], list)

    def test_an_empty_mesh_repairs_without_crashing(self):
        r = meshops.repair(Mesh.Mesh(), authorized=("fillupHoles",))
        self.assertEqual(r["after"]["facets"], 0)
        self.assertEqual(r["missing_regions"], [])


if __name__ == "__main__":
    unittest.main()
