# SPDX-License-Identifier: Apache-2.0
"""M1: the bounded deviation-driven correction pass. A HIGH-confidence DIMENSION_ERROR
attribution carries the fitted value — the correction updates the plan parameter to the
FIT (never an invented number), rebuilds, revalidates, and keeps the better deviation.
Bounded: one round, engine-synthesized plans only (an explicit plan_override is the
agent's decision and is never silently edited), revert-on-worse, everything recorded."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

from oscar.recon.reconstruct import corrections_from_regions  # noqa: E402


def region(op, fitted, constructed, conf="HIGH", dtype="DIMENSION_ERROR"):
    return {"region_id": "DEV-001", "direction": "s2m", "deviation_type": dtype,
            "plan_operation_id": op, "cad_object_name": "Hole001",
            "attribution_confidence": conf, "max_deviation_mm": 0.8,
            "recognized_feature": f"recognized hole r={fitted / 2} at [0, 0, 0]",
            "evidence": [f"fitted diameter {fitted:.2f} mm vs constructed {constructed:.2f} mm"]}


PLAN = {"features": [{"id": "hole_1", "op": "hole", "position": {"x": 0.0, "y": 0.0},
                     "diameter": "hole_d", "plane": {"origin": [0, 0, 0], "x_dir": [1, 0, 0], "normal": [0, 0, 1]}},
                    {"id": "fillet_bot", "op": "fillet", "size": "fillet_rb", "edges": "bottom_outer"}],
        "parameters": {"hole_d": {"value": 7.63, "source": {}},
                       "fillet_rb": {"value": 4.0, "source": {}}}}
SURVEY = {"holes": [{"center": [0, 0, 0], "axis": [0, 0, 1], "radius": 3.3145}],
          "dressings": [{"kind": "fillet", "radius_mm": 2.5, "point": [30.0, 0.0, 8.75], "axis": [1, 0, 0]}]}


class CorrectionsFromRegionsTest(unittest.TestCase):
    def test_a_high_dimension_error_yields_the_fitted_value(self):
        got = corrections_from_regions([region("hole_1", 6.629, 7.63)], SURVEY, PLAN)
        self.assertEqual(got, [{"parameter": "hole_d", "from": 7.63, "to": 6.629,
                                "op": "hole_1", "evidence": got[0]["evidence"][:1]
                                if got else []}])
        self.assertEqual(got[0]["from"], 7.63)
        self.assertAlmostEqual(got[0]["to"], 6.629, places=3)

    def test_only_high_confidence_dimension_errors_correct(self):
        for conf in ("MEDIUM", "AMBIGUOUS", "UNATTRIBUTED"):
            self.assertEqual(corrections_from_regions(
                [region("hole_1", 6.629, 7.63, conf=conf)], SURVEY, PLAN), [], conf)
        self.assertEqual(corrections_from_regions(
            [region("hole_1", 6.629, 7.63, dtype="MISSING_MATERIAL")], SURVEY, PLAN), [])

    def test_a_fillet_region_corrects_through_the_dressing_fit(self):
        r = {"region_id": "DEV-002", "direction": "s2m", "deviation_type": "DIMENSION_ERROR",
             "plan_operation_id": "fillet_bot", "attribution_confidence": "HIGH",
             "max_deviation_mm": 0.7, "recognized_feature": "recognized fillet strip r=2.5 at [30.0, 0.0, 8.75]",
             "evidence": ["fitted radius 2.50 mm vs constructed size 4.00 mm"]}
        got = corrections_from_regions([r], SURVEY, PLAN)
        self.assertEqual([(c["parameter"], c["from"], c["to"]) for c in got],
                         [("fillet_rb", 4.0, 2.5)])

    def test_the_fit_must_exist_and_disagree_beyond_tolerance(self):
        no_holes = {"holes": [], "dressings": []}
        self.assertEqual(corrections_from_regions([region("hole_1", 7.63, 7.63)], SURVEY, PLAN), [])
        self.assertEqual(corrections_from_regions([region("hole_1", 6.629, 7.63)], no_holes, PLAN), [])
        # within 10% + warn margin: a snap-level difference is not a correction
        self.assertEqual(corrections_from_regions([region("hole_1", 7.3, 7.63)], SURVEY, PLAN), [])

    def test_bounded_one_correction_per_parameter(self):
        got = corrections_from_regions(
            [region("hole_1", 6.629, 7.63), region("hole_1", 6.7, 7.63)], SURVEY, PLAN)
        self.assertEqual(len(got), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
