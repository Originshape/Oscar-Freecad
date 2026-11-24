"""B-1: the reconstruction benchmark runner — the smoke subset must run end to end
from ONE command, deterministically, and emit the Markdown + JSON evidence table
(the full 15-case run is the owner's manual command; CI runs the smoke subset).
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BenchmarkSmokeTest(unittest.TestCase):
    def _load_runner(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "benchmark_recon", os.path.join(ROOT, "scripts", "benchmark_recon.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_full_case_oracles_match_measured_accuracy_contract(self):
        cases = self._load_runner()._cases()
        self.assertEqual(len(cases), 15)
        self.assertEqual(len({case[0] for case in cases}), 15)
        by_name = {name: (warn, outcome, volume_tol)
                   for name, _spec, warn, outcome, volume_tol in cases}
        # The outcome is a minimum target: 07 may be partially verified or better;
        # 12 remains unverified because Hausdorff exceeds warn_mm. Volume gates below
        # independently catch regressions.
        self.assertEqual(by_name["07-filleted-bracket"],
                         (0.5, "partially_verified_parametric", 0.03))
        self.assertEqual(by_name["12-dome-cap"],
                         (0.5, "reconstruction_unverified", 0.01))

    def test_outcome_oracle_accepts_a_higher_verified_class_without_lowering_the_floor(self):
        mod = self._load_runner()
        meets = mod._outcome_meets
        self.assertTrue(meets("partially_verified_parametric",
                              "partially_verified_parametric"))
        self.assertTrue(meets("verified_parametric",
                              "partially_verified_parametric"))
        self.assertFalse(meets("reconstruction_unverified",
                               "partially_verified_parametric"))
        self.assertFalse(meets("partially_verified_parametric",
                               "verified_parametric"))

    def test_the_smoke_subset_runs_and_writes_evidence(self):
        mod = self._load_runner()
        out = tempfile.mkdtemp(prefix="bench_")
        code = mod.main(["--root", ROOT, "--out", out, "--smoke"])
        self.assertEqual(code, 0, "every smoke case must pass its oracle")
        files = os.listdir(out)
        self.assertTrue(any(f.endswith(".md") for f in files), files)
        data = json.load(open(os.path.join(
            out, next(f for f in files if f.endswith(".json"))), encoding="utf-8"))
        self.assertEqual(len(data["rows"]), 3)
        for row in data["rows"]:
            self.assertEqual(row["oracle"], "PASS", row)
            self.assertIsNotNone(row["outcome"])
            self.assertIsNotNone(row["hausdorff_mm"])


if __name__ == "__main__":
    unittest.main()
