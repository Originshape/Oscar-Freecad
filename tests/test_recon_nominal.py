"""G8: ``oscar.recon.intent`` — nominal dimension snapping with evidence-band protection.

The ladder oracles pin the two-stage rule: a candidate qualifies only within 2σ and the
COARSEST qualifying grid wins (10.02 → 10.0 at σ0.05, never 10.1); out-of-band values
keep their measurement; exact values are no-ops; hole diameters get their nearest common
drill size as a SUGGESTION only; positions are untouched. The reconstruct path applies
this to the plan's measured parameters, records every decision in provenance, and the
deviation rows revalidate the snapped model against the scan.
"""

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

from oscar.recon import intent  # noqa: E402
from oscar.recon import reconstruct  # noqa: E402


class LadderTest(unittest.TestCase):
    def test_the_coarsest_in_band_nominal_wins(self):
        got = intent.length_candidate(10.02, 0.05)
        self.assertIsNotNone(got)
        nominal, delta, ratio = got
        self.assertEqual(nominal, 10.0)
        self.assertAlmostEqual(delta, 0.02, places=6)
        self.assertAlmostEqual(ratio, 0.4, places=3)

    def test_out_of_band_keeps_the_measurement(self):
        # a fine σ narrows the band below the tenth grid's half-step: nothing qualifies
        self.assertIsNone(intent.length_candidate(10.65, 0.02))   # Δ0.05 > 2σ=0.04
        nominal, _, _ = intent.length_candidate(10.44, 0.05)      # 10.5 at 1.2σ wins
        self.assertEqual(nominal, 10.5)

    def test_exact_values_are_no_ops(self):
        self.assertIsNone(intent.length_candidate(10.0, 0.05))
        self.assertIsNone(intent.angle_candidate(90.0, 0.1))

    def test_half_grid_inside_the_band_snaps_to_the_half(self):
        # 10.6 at σ0.05: integer 11 is 8σ out; 10.5 is exactly 2σ — in band (≤)
        nominal, delta, _ = intent.length_candidate(10.6, 0.05)
        self.assertEqual(nominal, 10.5)

    def test_tenth_grid_catches_what_coarser_grids_miss(self):
        nominal, _, ratio = intent.length_candidate(10.13, 0.05)   # 10.5/10 far, 10.1 near
        self.assertEqual(nominal, 10.1)
        self.assertLessEqual(ratio, intent.SIGMA_BAND)


class AngleLadderTest(unittest.TestCase):
    def test_a_countersink_angle_finds_the_whole_degree(self):
        nominal, delta, ratio = intent.angle_candidate(89.7, 0.2)
        self.assertEqual(nominal, 90.0)                        # the 15° grid holds 90
        self.assertAlmostEqual(delta, 0.3, places=6)
        self.assertLessEqual(ratio, intent.SIGMA_BAND)

    def test_a_non_quarter_angle_snaps_to_the_degree_not_15s(self):
        nominal, _, _ = intent.angle_candidate(47.3, 0.2)      # 45 is 11σ out
        self.assertEqual(nominal, 47.0)


class DrillTest(unittest.TestCase):
    def test_the_nearest_common_drill_is_a_suggestion_never_forced(self):
        nominal, delta = intent.drill_candidate(6.62)
        self.assertEqual(nominal, 6.6)
        self.assertAlmostEqual(delta, 0.02, places=6)
        params, log = intent.apply_nominals(
            {"hole_d": {"value": 6.62, "source": {"kind": "scan_fit", "ref": "wall fit"}}},
            0.05)
        self.assertEqual(params["hole_d"]["value"], 6.62, "diameters are never snapped")
        self.assertEqual(len(log), 1)
        self.assertFalse(log[0]["applied"])
        self.assertEqual(log[0]["nominal"], 6.6)
        self.assertIn("suggestion only", log[0]["note"])


class ApplyNominalsTest(unittest.TestCase):
    def test_lengths_snap_diameters_suggest_positions_untouched(self):
        params = {
            "plate_len": {"value": 60.04, "source": {"kind": "scan_fit", "ref": "section"}},
            "plate_wid": {"value": 40.06, "source": {"kind": "scan_fit", "ref": "section"}},
            "plate_t": {"value": 10.02, "source": {"kind": "scan_fit", "ref": "planes"}},
            "hole_d": {"value": 6.62, "source": {"kind": "scan_fit", "ref": "wall fit"}},
            "hole_x1": {"value": 15.03, "source": {"kind": "scan_fit", "ref": "axis"}},
            "hole_dp1": {"value": 8.03, "source": {"kind": "scan_fit", "ref": "depth"}},
        }
        out, log = intent.apply_nominals(params, 0.05)
        self.assertEqual(out["plate_len"]["value"], 60.0)
        self.assertEqual(out["plate_wid"]["value"], 40.0)      # 0.06 = 1.2σ → 40
        self.assertEqual(out["plate_t"]["value"], 10.0)
        self.assertEqual(out["hole_d"]["value"], 6.62)         # never forced
        self.assertEqual(out["hole_x1"]["value"], 15.03)       # positions untouched
        self.assertEqual(out["hole_dp1"]["value"], 8.0)        # depths snap (0.03=0.6σ)
        self.assertIn("nominal snap 60.04→60", out["plate_len"]["source"]["ref"])
        applied = {s["param"] for s in log if s["applied"]}
        self.assertEqual(applied, {"plate_len", "plate_wid", "plate_t", "hole_dp1"})
        self.assertIn("hole_d", {s["param"] for s in log if not s["applied"]})

    def test_the_input_dict_is_not_mutated(self):
        params = {"plate_t": {"value": 8.02, "source": {"kind": "scan_fit", "ref": "x"}}}
        intent.apply_nominals(params, 0.05)
        self.assertEqual(params["plate_t"]["value"], 8.02)

    def test_non_numeric_and_unknown_parameters_pass_through(self):
        params = {"plate_t": {"value": "expr"}, "other": {"value": 3.33}}
        out, log = intent.apply_nominals(params, 0.05)
        self.assertEqual(out, params)
        self.assertEqual(log, [])


class ReconstructNominalsTest(unittest.TestCase):
    def test_an_off_nominal_bracket_snaps_and_stays_verified(self):
        shape = Part.makeBox(60.04, 40.06, 10.02)
        for x, y in ((15, 10), (45, 10), (15, 30), (45, 30)):
            shape = shape.cut(Part.makeCylinder(3.3, 30, App.Vector(x, y, -10)))
        mesh = Mesh.Mesh(shape.tessellate(0.3))
        doc = App.newDocument("G8Snap")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"], result.get("note"))
            self.assertTrue(result["build"]["success"], result["build"])
            snaps = result["nominal_snaps"]
            applied = {s["param"]: s for s in snaps if s["applied"]}
            self.assertIn("plate_len", applied)
            # the section's u/v basis maps either side to plate_len — both are nominal
            self.assertIn(applied["plate_len"]["nominal"], (60.0, 40.0))
            self.assertIn("plate_len", result["plan"]["parameters"])
            self.assertEqual(result["plan"]["parameters"]["plate_len"]["value"],
                             applied["plate_len"]["nominal"])
            # provenance carries every measured→nominal decision
            self.assertIn("plate_len", {s["param"] for s in result["provenance"]
                                        .get("nominal_snaps", [])})
            self.assertEqual(result["outcome"], "verified_parametric", result["meta"])
        finally:
            App.closeDocument(doc.Name)

    def test_an_exact_bracket_records_no_meaningful_snaps(self):
        shape = Part.makeBox(60, 40, 10)
        for x, y in ((15, 10), (45, 10), (15, 30), (45, 30)):
            shape = shape.cut(Part.makeCylinder(3.3, 30, App.Vector(x, y, -10)))
        mesh = Mesh.Mesh(shape.tessellate(0.3))
        doc = App.newDocument("G8Exact")
        try:
            result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
            self.assertTrue(result["supported"])
            applied = [s for s in result["nominal_snaps"] if s["applied"]]
            # the plate is exactly nominal: at most float-epsilon snaps may appear, and
            # the drills (Ø6.6 exactly) stay suggestions — never applied
            self.assertTrue(all(s["delta"] < 0.01 for s in applied), applied)
            self.assertNotIn("hole_d", {s["param"] for s in applied})
            self.assertEqual(result["outcome"], "verified_parametric", result["meta"])
        finally:
            App.closeDocument(doc.Name)


if __name__ == "__main__":
    unittest.main()
