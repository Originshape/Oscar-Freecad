"""G12: ``oscar.recon.segmerge`` and the survey merge passes — absorb, union-fit merge
and kind upgrade. Fixtures are REAL tessellated FreeCAD shapes:

- a cylinder split by an artificial noisy band must MERGE back into one cylinder region
  with the true radius;
- two coplanar plates side by side (sharing an edge) must merge into ONE plane region;
- two parallel planes separated by a scratch/gap must STAY separate;
- the four-hole plate must keep its exact recognition (regression);
- a cylinder pair forming a frustum must upgrade to a cone through the G2 gate;
- provenance evidence keys must ride the merge untouched.
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

from oscar.recon import fit, recognize, segmerge  # noqa: E402


def mesh_of(shape, tess=0.4):
    p, f = shape.tessellate(tess)
    return Mesh.Mesh((list(p), [tuple(t) for t in f]))


def split_cylinder_mesh(radius=6.0, height=14.0, band=1.0, jitter=0.35):
    """A cylinder whose mid-band vertices are pushed radially outward so the dihedral
    threshold SPLITS the wall into upper/lower regions (the classic over-segmentation
    the G12 merge must repair)."""
    p, f = Part.makeCylinder(radius, height).tessellate(0.4)
    pts = [[q.x, q.y, q.z] for q in p]
    pts = np.array(pts, dtype=float)
    mid = (pts[:, 2] > height / 2 - band / 2) & (pts[:, 2] < height / 2 + band / 2)
    r = np.hypot(pts[:, 0], pts[:, 1])
    scale = np.where(mid, 1.0 + jitter, 1.0)
    pts[:, 0] *= scale
    pts[:, 1] *= scale
    return Mesh.Mesh(([App.Vector(*q) for q in pts], [tuple(t) for t in f]))


class MergePassTest(unittest.TestCase):
    def test_split_cylinder_merges_back_into_one_true_radius_region(self):
        mesh = split_cylinder_mesh()
        s = recognize.survey(mesh)
        walls = [r for r in s["regions"] if r["type"] == "cylinder"
                 and r["axial_span"] > 4.0]
        self.assertTrue(walls, "the split wall must be recognized as cylinder(s)")
        self.assertTrue(any(w["axial_span"] >= 10.0 for w in walls),
                        "a merged wall spans nearly the full height: "
                        + str([(w["axial_span"], w["radius"]) for w in walls]))
        merged = max(walls, key=lambda w: w["axial_span"])
        self.assertAlmostEqual(merged["radius"], 6.0, delta=0.35)

    def test_coplanar_adjacent_plates_merge_into_one_plane(self):
        # two coplanar slabs sharing a full edge — one planar top after merging
        shape = Part.makeBox(20, 10, 3).fuse(Part.makeBox(20, 10, 3,
                                                          App.Vector(0, 10, 0)))
        shape = shape.removeSplitter()
        mesh = mesh_of(shape, 0.8)
        s = recognize.survey(mesh)
        tops = [r for r in s["regions"] if r["type"] == "plane"
                and abs(r["normal"][2]) > 0.99 and r["area_mm2"] > 150]
        self.assertTrue(tops, "the coplanar top must exist as a plane region")
        self.assertGreaterEqual(max(t["area_mm2"] for t in tops), 380.0,
                                "the merged top covers both plates (2 x 200 mm2)")

    def test_scratch_separated_planes_stay_separate(self):
        # two coplanar slabs with a GAP: never merged (not adjacent)
        shape = Part.makeBox(20, 8, 3)
        shape = shape.fuse(Part.makeBox(20, 8, 3, App.Vector(0, 12, 0)))
        shape = shape.removeSplitter()
        mesh = mesh_of(shape, 0.8)
        s = recognize.survey(mesh)
        tops = [r for r in s["regions"] if r["type"] == "plane"
                and abs(r["normal"][2]) > 0.99 and r["area_mm2"] > 100]
        self.assertGreaterEqual(len(tops), 2,
                                "the gapped tops stay separate regions")
        for t in tops:
            self.assertLess(t["area_mm2"], 200.0, "no region spans the gap")


class KindUpgradeTest(unittest.TestCase):
    def test_two_cylinder_strips_of_a_frustum_upgrade_to_cone(self):
        # a cone frustum wall split into upper/lower strips: each strip alone fits a
        # cylinder; the union must upgrade to a cone through the G2 gate
        mesh = mesh_of(Part.makeCone(4, 9, 12), 0.5)
        s = recognize.survey(mesh)
        cones = [r for r in s["regions"] if r["type"] == "cone"]
        self.assertTrue(cones, "the frustum wall must be ONE cone region (upgraded)")
        self.assertAlmostEqual(cones[0]["half_angle_deg"],
                               np.degrees(np.arctan(5.0 / 12.0)), delta=0.5)


class AbsorbTest(unittest.TestCase):
    PTS = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0], [20, 10, 0],
                    [20, 0, 0], [5, 5, 0], [0, 0, 5], [10, 0, 5], [0, 10, 5],
                    [15, 10, 0], [15, 7, 0]], dtype=float)
    FAC = np.array([(0, 1, 2), (1, 3, 2), (1, 5, 3), (5, 4, 3),
                    (2, 3, 10), (3, 4, 10),
                    (0, 2, 7), (2, 3, 7), (3, 1, 7),
                    (3, 10, 11)], dtype=int)
    NORMALS = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
                        [0, 0, 1], [0, 0, 1],
                        [1, 0, 0], [1, 0, 0], [1, 0, 0],
                        [0, 0, 1]], dtype=float)

    def test_small_group_joins_the_compatible_adjacent_region(self):
        groups = [np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7, 8])]
        small = [np.array([9])]        # shares edge (3,10) with region A, +Z like A
        merged, remaining = segmerge.absorb_small(groups, small, self.NORMALS,
                                                  self.FAC, self.PTS, 25.0)
        self.assertEqual(len(remaining), 0)
        self.assertEqual([len(g) for g in merged], [7, 3],
                         "the small facet joined region A (compatible), B untouched")

    def test_incompatible_small_group_is_not_absorbed(self):
        normals = self.NORMALS.copy()
        normals[9] = [0, 0, -1]        # opposite face: incompatible with both regions
        groups = [np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7, 8])]
        small = [np.array([9])]
        merged, remaining = segmerge.absorb_small(groups, small, normals,
                                                  self.FAC, self.PTS, 25.0)
        self.assertEqual([len(g) for g in merged], [6, 3])
        self.assertEqual([len(g) for g in remaining], [1],
                         "an incompatible facet stays unabsorbed (honesty over merging)")

    def test_no_faces_lost_or_duplicated(self):
        groups = [np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7, 8])]
        small = [np.array([9])]
        merged, remaining = segmerge.absorb_small(groups, small, self.NORMALS,
                                                  self.FAC, self.PTS, 25.0)
        before = sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        after = sorted(int(x) for g in merged for x in g) +             sorted(int(x) for g in remaining for x in g)
        self.assertEqual(before, after)
        self.assertEqual(len(after), len(set(after)), "no duplicate membership")


class MergeBehaviourTest(unittest.TestCase):
    def test_incompatible_adjacent_kinds_never_merge(self):
        # plane top + cylinder wall of the plate: adjacency exists, kinds differ and
        # neither is a cylinder pair — the merge pass must leave both alone
        mesh = mesh_of(Part.makeCylinder(6, 10), 0.5)
        s = recognize.survey(mesh)
        kinds = [r["type"] for r in s["regions"]]
        self.assertIn("cylinder", kinds)
        # no region may claim an area larger than the whole mesh (a bogus merge)
        for r in s["regions"]:
            self.assertLessEqual(r["area_mm2"], float(mesh.Area) + 1e-6)

    def test_greedy_rms_ordering_merges_the_best_pair_first(self):
        # unit level: three coplanar strips; the flattest union (lowest rms) wins first,
        # and the pass chains to a single plane region
        pts = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0],
                        [0, 10, 0], [10, 10, 0], [20, 10, 0], [30, 10, 0]],
                       dtype=float)
        fac = np.array([(0, 1, 5), (0, 5, 4),
                        (1, 2, 6), (1, 6, 5),
                        (2, 3, 7), (2, 7, 6)], dtype=int)
        normals = np.zeros((6, 3), dtype=float)
        normals[:, 2] = 1.0
        groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
        hints = ["plane", "plane", "plane"]
        pairs = [(0, 1), (1, 2)]
        merged, kinds, log = segmerge.merge_adjacent(groups, hints, fac, pts, normals,
                                                     0.05, 0.35, 0.01, pairs)
        self.assertEqual([len(g) for g in merged], [6],
                         "three coplanar strips chain-merge into one plane")
        self.assertEqual(kinds, ["plane"])
        self.assertEqual(len(log), 2, "two merges were journaled")
        rms_seq = [m["rms"] for m in log]
        self.assertEqual(rms_seq, sorted(rms_seq),
                         "merges are applied in improving-rms order (greedy)")

    def test_insignificant_cone_upgrade_is_rejected(self):
        # a true cylinder wall split in two: the union fits a CYLINDER perfectly, so the
        # cone candidate (however tried) must lose the G2 gate and the merge stays a
        # cylinder — the frustum upgrade test above covers the accepted side
        from oscar.recon import fit as _fitm

        p, f = Part.makeCylinder(5, 20).tessellate(0.4)
        mesh = Mesh.Mesh((list(p), [tuple(t) for t in f]))
        pts = np.array([[q.x, q.y, q.z] for q in p], dtype=float)
        fac = np.array([tuple(t) for t in f], dtype=int)
        normals = np.array([[fc.Normal.x, fc.Normal.y, fc.Normal.z]
                            for fc in mesh.Facets])
        # wall facets only (drop caps)
        keep = np.abs(normals[:, 2]) < 0.2
        idx = np.nonzero(keep)[0]
        P = pts[fac[idx]].reshape(-1, 3)
        N = np.repeat(normals[idx], 3, axis=0)[:len(P)]
        cone = _fitm.fit_cone(P, N)
        cyl = _fitm.fit_cylinder(P, N)
        plane = _fitm.fit_plane(P)
        ok, why = _fitm.cone_significance(P, cone, {"rms": cyl["rms"]},
                                          {"rms": plane[2]}, 0.01)
        self.assertFalse(ok)
        self.assertTrue(why, "the gate states why the cone was rejected")


class PlateRegressionTest(unittest.TestCase):
    def test_four_hole_plate_recognition_is_unchanged(self):
        shape = Part.makeBox(60, 40, 10)
        for x, y in ((15, 10), (45, 10), (15, 30), (45, 30)):
            shape = shape.cut(Part.makeCylinder(3.3, 30, App.Vector(x, y, -10)))
        mesh = mesh_of(shape, 0.3)
        s = recognize.survey(mesh)
        self.assertEqual(s["classification"]["class"], "prismatic")
        holes = s["holes"]
        self.assertEqual(len(holes), 4, holes)
        for h in holes:
            self.assertAlmostEqual(h["radius"], 3.3, delta=0.2)
            self.assertTrue(h["through"])


class ProvenanceMergeTest(unittest.TestCase):
    def test_evidence_keys_ride_the_merge(self):
        from oscar.recon import provenance

        mesh = split_cylinder_mesh()
        keys = provenance.facet_keys(mesh)
        ledger = provenance.new_ledger()
        # mark the ENTIRE wall synthetic: whichever regions survive the merge, their
        # evidence must read synthetic (the merge must not drop provenance)
        ledger["synthetic"] = list(keys)
        s = recognize.survey(mesh, provenance=ledger)
        tagged = [r for r in s["regions"] if r.get("evidence")]
        walls = [r for r in tagged if r["type"] in ("cylinder", "cone")]
        self.assertTrue(walls, "merged wall regions carry their synthetic evidence")


if __name__ == "__main__":
    unittest.main()
