# SPDX-License-License-Identifier: Apache-2.0
"""The §16 intent fields on route classification: confidence is the winner's measured
explanatory share; alternatives are routes whose own gate also passed
(oscar.recon.recognize._classify)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

from oscar.recon import recognize  # noqa: E402


def classify(plane, cyl, unrec=0.05, pairs=2, holes=0, sphere=0.0, total=100.0, box=(50, 40, 10)):
    diag = {"size_mm": list(box), "components": 1}
    return recognize._classify(diag, plane * total, cyl * total, unrec * total, total,
                               [(0, 1)] * pairs, [], [{"radius": 2.0}] * holes, sphere_area=sphere * total)


class IntentFieldsTest(unittest.TestCase):
    def test_a_clear_prismatic_winner_carries_its_plane_share_as_confidence(self):
        result = classify(plane=0.85, cyl=0.05)
        self.assertEqual(result["class"], "prismatic")
        self.assertEqual(result["confidence"], 0.85)
        self.assertNotIn("alternatives", result)      # rotational share 0.05: no real alternative

    def test_a_thin_panel_carries_its_plane_share_and_no_false_alternative(self):
        # the rotational gate wins FIRST when its share crosses, so a thin-panel winner
        # genuinely lacks rotational support: no alternative is invented
        result = classify(plane=0.45, cyl=0.3, sphere=0.05, pairs=0, holes=0, box=(100, 100, 3))
        self.assertEqual(result["class"], "thin_panel")
        self.assertEqual(result["confidence"], 0.45)
        self.assertNotIn("alternatives", result)

    def test_holes_suppress_the_rotational_alternative(self):
        # cyl share crosses 0.5, but a real hole blocks the rotational class AND its
        # alternative: the hybrid result stays quiet instead of asking a false question
        with_holes = classify(plane=0.3, cyl=0.55, pairs=0, holes=1, box=(50, 50, 20))
        self.assertEqual(with_holes["class"], "hybrid_mechanical")
        self.assertNotIn("alternatives", with_holes)

    def test_a_rotational_winner_carries_its_share_with_a_prismatic_alternative_when_close(self):
        result = classify(plane=0.55, cyl=0.38, sphere=0.12, pairs=2, holes=0)
        self.assertEqual(result["class"], "rotational")
        self.assertEqual(result["confidence"], 0.5)          # cyl + sphere share
        self.assertEqual(result["alternatives"], ["prismatic_extrusion"])

    def test_approve_sees_exactly_what_it_reads(self):
        from oscar.recon.approve import CLASSIFICATION_CONFIDENCE_MIN, approval_triggers

        # genuinely ambiguous: mostly unrecognized BUT with strong parallel-plane evidence
        ambiguous = classify(plane=0.51, cyl=0.03, unrec=0.45, pairs=1, holes=0)
        self.assertEqual(ambiguous["class"], "organic_or_hybrid")
        self.assertEqual(ambiguous["alternatives"], ["prismatic_extrusion"])
        triggers = approval_triggers({"classification": ambiguous})
        self.assertTrue(any(t["rule"] == "ambiguous_interpretation" for t in triggers))
        clear = classify(plane=0.9, cyl=0.05)
        self.assertGreaterEqual(clear["confidence"], CLASSIFICATION_CONFIDENCE_MIN)
        self.assertEqual([t for t in approval_triggers({"classification": clear})
                          if t["rule"] == "ambiguous_interpretation"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
