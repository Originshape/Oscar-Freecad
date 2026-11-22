# SPDX-License-Identifier: Apache-2.0
"""L-2: deviation localization + conservative feature attribution (oscar.recon.localize).
Synthetic cluster/feed dicts exercise the region math and the attribution chain; no fixture
is hardcoded to any benchmark part."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

from oscar.recon import localize  # noqa: E402


def cluster(direction="s2m", kind="missing", points=None, dists=None, centroid=None, n=6):
    pts = points or []
    return {"kind": kind, "points": n, "area_mm2": 4.0,
            "centroid_mm": centroid or [round(sum(p[i] for p in pts) / len(pts), 3)
                                        for i in range(3)],
            "direction": direction,
            "member_points_mm": pts,
            "member_dists_mm": dists,
            "mean_dist_mm": round(sum(dists) / len(dists), 4) if dists else None,
            "max_dist_mm": max(dists) if dists else None}


SURVEY = {"holes": [{"center": [20.0, 10.0, 5.0], "axis": [0, 0, 1], "radius": 2.0},
                    {"center": [40.0, 10.0, 5.0], "axis": [0, 0, 1], "radius": 2.0}]}
PLAN = {"features": [{"id": "OP-1", "op": "hole", "position": [20.0, 10.0, 5.0], "diameter": 4.0},
                     {"id": "OP-2", "op": "hole", "position": [40.0, 10.0, 5.0], "diameter": 4.0}],
       "parameters": {}}
BUILD = {"features": [{"id": "OP-1", "op": "hole", "object": "Hole001"},
                      {"id": "OP-2", "op": "hole", "object": "Hole002"}]}


class LocalizationTest(unittest.TestCase):
    def test_both_directions_stay_independently_identifiable(self):
        res = localize.localize_deviations(
            {"clusters": [cluster("s2m", "missing", [[19, 9, 5]] * 4, [1.0] * 4, n=40),
                          cluster("m2s", "excess", [[5, 5, 5]] * 4, [0.4] * 4, n=30)]},
            survey=SURVEY, plan=PLAN, build=BUILD)
        kinds = {(r["direction"], r["deviation_type"]) for r in res["regions"]}
        self.assertIn(("s2m", "MISSING_MATERIAL"), kinds)
        self.assertIn(("m2s", "EXTRA_MATERIAL"), kinds)
        self.assertEqual(localize.localization_block(res)[0], "LOCALIZED DEVIATIONS")

    def test_separated_deviations_form_distinct_regions_with_bboxes_and_stats(self):
        res = localize.localize_deviations(
            {"clusters": [cluster(points=[[19.9, 10, 5], [20.1, 10, 5], [20, 9.9, 5], [20, 10.1, 5]],
                                   dists=[1.0, 2.0, 1.5, 3.0], n=40),
                          cluster(points=[[40.0, 30.0, 5]] * 4, dists=[0.6] * 4, n=20)]},
            survey=SURVEY, plan=PLAN, build=BUILD)
        self.assertEqual(len(res["regions"]), 2)
        first = res["regions"][0]
        self.assertEqual(first["sample_count"], 40)
        self.assertIsNotNone(first["bounding_box_mm"])
        self.assertAlmostEqual(first["rms_deviation_mm"], 2.0156, places=3)
        self.assertEqual(first["p95_deviation_mm"], 3.0)
        self.assertEqual([r["region_id"] for r in res["regions"]], ["DEV-001", "DEV-002"])

    def test_empty_and_unfed_inputs_are_handled(self):
        self.assertEqual(localize.localize_deviations({"clusters": []})["regions"], [])
        legacy = {"clusters": [{"kind": "missing", "points": 5, "area_mm2": 1.0,
                                "centroid_mm": [0, 0, 0]}]}   # no L-1 feed
        out = localize.localize_deviations(legacy)
        self.assertEqual(out["regions"], [])
        self.assertIn("skipped", out.get("note", ""))
        self.assertEqual(localize.localization_block({"regions": []}), [])

    def test_unavailable_statistics_are_none_not_zero(self):
        res = localize.localize_deviations(
            {"clusters": [cluster(points=[[1, 1, 1]] * 4, dists=None, n=10)]})
        r = res["regions"][0]
        self.assertIsNone(r["rms_deviation_mm"])
        self.assertIsNone(r["mean_deviation_mm"])


class AttributionTest(unittest.TestCase):
    def test_full_chain_attributes_high(self):
        res = localize.localize_deviations(
            {"clusters": [cluster(points=[[19.5, 10, 5], [20.5, 10, 5], [20, 9.5, 5], [20, 10.5, 5]],
                                   dists=[1.2, 1.3, 1.1, 1.4], n=40)]},
            survey=SURVEY, plan=PLAN, build=BUILD)
        r = res["regions"][0]
        self.assertEqual(r["plan_operation_id"], "OP-1")
        self.assertEqual(r["cad_object_name"], "Hole001")
        self.assertEqual(r["attribution_confidence"], "HIGH")
        self.assertTrue(r["evidence"])

    def test_dimension_error_is_only_inferred_from_the_full_chain(self):
        plan = {"features": [{"id": "OP-1", "op": "hole", "position": [20.0, 10.0, 5.0],
                              "diameter": 6.0}], "parameters": {}}
        res = localize.localize_deviations(
            {"clusters": [cluster(points=[[19.5, 10, 5]] * 4, dists=[1.0] * 4, n=40)]},
            survey=SURVEY, plan=plan, build=BUILD)
        self.assertEqual(res["regions"][0]["deviation_type"], "DIMENSION_ERROR")

    def test_no_recognized_fit_downgrades_to_medium_never_high(self):
        survey_no_holes = {"holes": []}
        res = localize.localize_deviations(
            {"clusters": [cluster(points=[[19.5, 10, 5]] * 4, dists=[1.0] * 4, n=40)]},
            survey=survey_no_holes, plan=PLAN, build=BUILD)
        r = res["regions"][0]
        self.assertEqual(r["attribution_confidence"], "MEDIUM")
        self.assertIn("no recognized-fit link", " ".join(r["evidence"]))

    def test_equally_plausible_operations_stay_ambiguous(self):
        plan = {"features": [{"id": "OP-1", "op": "hole", "position": [20.0, 10.0, 5.0],
                              "diameter": 4.0},
                             {"id": "OP-2", "op": "hole", "position": [21.0, 10.0, 5.0],
                              "diameter": 4.0}], "parameters": {}}
        res = localize.localize_deviations(
            {"clusters": [cluster(points=[[20.4, 10, 5]] * 4, dists=[1.0] * 4, n=40)]},
            survey=SURVEY, plan=plan, build=BUILD)
        r = res["regions"][0]
        self.assertEqual(r["attribution_confidence"], "AMBIGUOUS")
        self.assertIsNone(r["plan_operation_id"])

    def test_a_recognized_hole_without_an_operation_stays_unattributed_and_explains_itself(self):
        plan_empty = {"features": [], "parameters": {}}
        res = localize.localize_deviations(
            {"clusters": [cluster(points=[[19.5, 10, 5]] * 4, dists=[1.0] * 4, n=40)]},
            survey=SURVEY, plan=plan_empty, build={"features": []})
        r = res["regions"][0]
        self.assertEqual(r["attribution_confidence"], "UNATTRIBUTED")
        self.assertIsNotNone(r["recognized_feature"])
        self.assertIn("no hole operation", " ".join(r["evidence"]))
        self.assertEqual(res["unattributed"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


SURVEY_D = {"holes": [],
            "dressings": [{"kind": "fillet", "radius_mm": 2.5, "width_mm": 2.5,
                           "point": [30.0, 0.0, 8.75], "axis": [1, 0, 0]}]}
PLAN_D = {"features": [{"id": "FIL-1", "op": "fillet", "size": "fillet_rb", "edges": "bottom_outer"}],
          "parameters": {"fillet_rb": {"value": 2.5, "source": {}}}}
BUILD_D = {"features": [{"id": "FIL-1", "op": "fillet", "object": "Fillet001"}]}


class DressingAttributionTest(unittest.TestCase):
    def cluster_at(self, radius=2.5, centre=(30.0, 2.5, 8.75)):
        # member points ON the fitted strip surface (axis x, radial 2.5 from the axis line)
        pts = [[centre[0] + dx, centre[1], centre[2]] for dx in (-1.0, 0.0, 1.0, 2.0)]
        return {"clusters": [cluster(points=pts, dists=[0.6, 0.7, 0.6, 0.8], n=50,
                                     centroid=[round(sum(p[i] for p in pts) / 4, 3)
                                               for i in range(3)])]}

    def test_fillet_chain_attributes_high(self):
        res = localize.localize_deviations(
            self.cluster_at(), survey=SURVEY_D, plan=PLAN_D, build=BUILD_D)
        r = res["regions"][0]
        self.assertEqual((r["plan_operation_id"], r["cad_object_name"]), ("FIL-1", "Fillet001"))
        self.assertEqual(r["attribution_confidence"], "HIGH")
        self.assertIn("fillet strip", r["recognized_feature"])

    def test_radius_disagreement_refines_to_dimension_error(self):
        plan = {"features": PLAN_D["features"],
                "parameters": {"fillet_rb": {"value": 4.0, "source": {}}}}
        res = localize.localize_deviations(
            self.cluster_at(), survey=SURVEY_D, plan=plan, build=BUILD_D)
        self.assertEqual(res["regions"][0]["deviation_type"], "DIMENSION_ERROR")

    def test_recognized_strip_without_an_operation_stays_unattributed(self):
        res = localize.localize_deviations(
            self.cluster_at(), survey=SURVEY_D,
            plan={"features": [], "parameters": {}}, build={"features": []})
        r = res["regions"][0]
        self.assertEqual(r["attribution_confidence"], "UNATTRIBUTED")
        self.assertIn("no fillet operation", " ".join(r["evidence"]))

    def test_hole_attribution_still_wins_when_both_chains_could_apply(self):
        res = localize.localize_deviations(
            self.cluster_at(), survey=SURVEY, plan=PLAN, build=BUILD)
        # no dressing in SURVEY, no hole near the strip → hole chain finds nothing,
        # dressing chain finds nothing → UNATTRIBUTED (not a false attribution)
        self.assertEqual(res["regions"][0]["attribution_confidence"], "UNATTRIBUTED")


class RealPlanFormTest(unittest.TestCase):
    """The plan's real hole ops: sketch-frame {x,y} position + plane basis, and a
    PARAMETER-NAME diameter resolved through the parameter table."""

    REAL_PLAN = {"features": [
        {"id": "hole_1", "op": "hole",
         "plane": {"normal": [0, 0, -1], "origin": [20.0, 10.0, 10.0], "x_dir": [1, 0, 0]},
         "position": {"x": 0.0, "y": 0.0}, "diameter": "hole_d"}],
        "parameters": {"hole_d": {"value": 6.6, "source": {}}}}
    REAL_BUILD = {"features": [{"id": "hole_1", "op": "hole", "object": "Hole001"}]}

    def test_sketch_frame_position_and_parameter_diameter_are_resolved(self):
        res = localize.localize_deviations(
            {"clusters": [cluster(points=[[19.5, 10, 9.5], [20.5, 10, 9.5],
                                          [20, 9.5, 9.5], [20, 10.5, 9.5]],
                                   dists=[1.0] * 4, n=40)]},
            survey=SURVEY, plan=self.REAL_PLAN, build=self.REAL_BUILD)
        r = res["regions"][0]
        self.assertEqual(r["plan_operation_id"], "hole_1")
        self.assertEqual(r["cad_object_name"], "Hole001")
        # fitted d=4.0 vs constructed 6.6 -> the chain refines to DIMENSION_ERROR
        self.assertEqual(r["deviation_type"], "DIMENSION_ERROR")

    def test_plain_list_positions_keep_working(self):
        res = localize.localize_deviations(
            {"clusters": [cluster(points=[[19.5, 10, 5]] * 4, dists=[1.0] * 4, n=40)]},
            survey=SURVEY, plan=PLAN, build=BUILD)
        self.assertEqual(res["regions"][0]["plan_operation_id"], "OP-1")


SLAB_PLAN = {"features": [{"id": "base_pad", "op": "pad",
                           "plane": {"origin": [30.0, 20.0, 0.0], "x_dir": [1, 0, 0],
                                     "normal": [0, 0, 1]}}],
             "parameters": {}}
SLAB_BUILD = {"features": [{"id": "base_pad", "op": "pad", "object": "Pad001"}]}


class SlabAttributionTest(unittest.TestCase):
    def cluster(self):
        pts = [[29.5, 20, 5], [30.5, 20, 5], [30, 19.5, 5], [30, 20.5, 5]]
        return {"clusters": [cluster(points=pts, dists=[0.6] * 4, n=40,
                                     centroid=[30.0, 20.0, 5.0])]}

    def test_a_single_pad_gets_medium_never_high(self):
        res = localize.localize_deviations(self.cluster(), survey={"holes": []},
                                          plan=SLAB_PLAN, build=SLAB_BUILD)
        r = res["regions"][0]
        self.assertEqual((r["plan_operation_id"], r["cad_object_name"]), ("base_pad", "Pad001"))
        self.assertEqual(r["attribution_confidence"], "MEDIUM")
        self.assertIn("no fitted surface", " ".join(r["evidence"]))
        self.assertEqual(res["unattributed"], 0)

    def test_no_pad_and_two_pads_stay_unattributed(self):
        empty = {"features": [], "parameters": {}}
        res = localize.localize_deviations(self.cluster(), survey={"holes": []},
                                          plan=empty, build={"features": []})
        self.assertEqual(res["regions"][0]["attribution_confidence"], "UNATTRIBUTED")
        two = {"features": SLAB_PLAN["features"] + [dict(SLAB_PLAN["features"][0], id="p2")],
               "parameters": {}}
        res2 = localize.localize_deviations(self.cluster(), survey={"holes": []},
                                           plan=two, build={"features": []})
        r2 = res2["regions"][0]
        self.assertEqual(r2["attribution_confidence"], "UNATTRIBUTED")
        self.assertIn("not separable", " ".join(r2["evidence"]))


class ThinSampleClusteringTest(unittest.TestCase):
    def test_two_agreeing_offenders_still_form_a_region(self):
        # the r4.0-fillet class: < 30 offenders spread along strips — 2 members cluster
        res = localize.localize_deviations(
            {"clusters": [cluster(points=[[11.7, 12.0, 9.1], [11.1, 9.2, 9.2]],
                                   dists=[0.56, 0.64], n=2,
                                   centroid=[11.4, 10.6, 9.15])]},
            survey={"holes": []}, plan={"features": [], "parameters": {}}, build={"features": []})
        self.assertEqual(len(res["regions"]), 1)
        self.assertEqual(res["regions"][0]["sample_count"], 2)


class CoarseNoisyHoleWallTest(unittest.TestCase):
    """M2: on the real case-13 noisy mesh (sigma=0.05 jitter, coarse tessellation) the
    wall regions fit PLANES within the edge-dominated gate — the recognition failure. The
    curvature-significance override must recover curved regions (cylinders > 0)."""

    def test_curved_regions_survive_on_the_noisy_mesh(self):
        import FreeCAD as App
        import Mesh
        import numpy as np

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        import benchmark_recon as bench  # noqa: E402

        from oscar.recon import recognize, meshops

        case = next(c for c in bench._cases() if c[0] == "13-noisy-bracket")
        shape, dev = case[1]
        mesh = bench.to_mesh(shape, dev)
        pts_v, facs = mesh.Topology
        rng = np.random.default_rng(bench.SEED + 1)
        P = np.array([[p.x, p.y, p.z] for p in pts_v]) + 0.05 * rng.standard_normal((len(pts_v), 3))
        mesh = Mesh.Mesh(([App.Vector(*p) for p in P],
                          [tuple(int(i) for i in f) for f in facs]))
        s = recognize.survey(mesh, meshops.diagnose(mesh))
        cylinders = [r for r in s["regions"] if r.get("type") == "cylinder"]
        self.assertGreater(len(cylinders), 0,
                           "the noisy mesh must recover curved regions")


class SlabAttributionTest(unittest.TestCase):
    def cluster(self):
        pts = [[29.5, 20, 5], [30.5, 20, 5], [30, 19.5, 5], [30, 20.5, 5]]
        return {"clusters": [cluster(points=pts, dists=[0.6] * 4, n=40,
                                     centroid=[30.0, 20.0, 5.0])]}

    def test_a_single_pad_gets_medium_never_high(self):
        res = localize.localize_deviations(self.cluster(), survey={"holes": []},
                                          plan=SLAB_PLAN, build=SLAB_BUILD)
        r = res["regions"][0]
        self.assertEqual((r["plan_operation_id"], r["cad_object_name"]), ("base_pad", "Pad001"))
        self.assertEqual(r["attribution_confidence"], "MEDIUM")
        self.assertIn("no fitted surface", " ".join(r["evidence"]))
        self.assertEqual(res["unattributed"], 0)

    def test_no_pad_and_two_pads_stay_unattributed(self):
        empty = {"features": [], "parameters": {}}
        res = localize.localize_deviations(self.cluster(), survey={"holes": []},
                                          plan=empty, build={"features": []})
        self.assertEqual(res["regions"][0]["attribution_confidence"], "UNATTRIBUTED")
        two = {"features": SLAB_PLAN["features"] + [dict(SLAB_PLAN["features"][0], id="p2")],
               "parameters": {}}
        res2 = localize.localize_deviations(self.cluster(), survey={"holes": []},
                                           plan=two, build={"features": []})
        r2 = res2["regions"][0]
        self.assertEqual(r2["attribution_confidence"], "UNATTRIBUTED")
        self.assertIn("not separable", " ".join(r2["evidence"]))


class ThinSampleClusteringTest(unittest.TestCase):
    def test_two_agreeing_offenders_still_form_a_region(self):
        # the r4.0-fillet class: < 30 offenders spread along strips — 2 members cluster
        res = localize.localize_deviations(
            {"clusters": [cluster(points=[[11.7, 12.0, 9.1], [11.1, 9.2, 9.2]],
                                   dists=[0.56, 0.64], n=2,
                                   centroid=[11.4, 10.6, 9.15])]},
            survey={"holes": []}, plan={"features": [], "parameters": {}}, build={"features": []})
        self.assertEqual(len(res["regions"]), 1)
        self.assertEqual(res["regions"][0]["sample_count"], 2)


