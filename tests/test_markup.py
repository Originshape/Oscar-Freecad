"""The pen's words for where the user drew (oscar.markup). Pure logic —
no Qt, no FreeCAD; runs under any interpreter."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

from oscar import markup  # noqa: E402
from oscar.markup import MarkHit  # noqa: E402


class DescribeTest(unittest.TestCase):
    def test_no_hit_points_to_the_picture(self):
        self.assertEqual(markup.describe([]), "[Marked on the model: see the attached picture.]")

    def test_one_part_its_side_and_the_region(self):
        hits = [MarkHit("Shell", (12.04, -8.0, 20.0), (0, 0, 1)), MarkHit("Shell", (40.0, 15.0, 20.2), (0.1, 0, 0.99))]
        self.assertEqual(markup.describe(hits),
                         "[Marked on the model (see the attached picture): part “Shell”, its top (+Z), "
                         "around x 12 to 40, y −8 to 15, z 20.1 mm.]")

    def test_the_part_hit_most_comes_first_and_sets_the_side_and_region(self):
        hits = [MarkHit("Lid", (0, 0, 5), (0, -1, 0)), MarkHit("Base", (1, 2, 3), (1, 0, 0)), MarkHit("Base", (1, 4, 3), (1, 0, 0)),
                MarkHit("Post", (9, 9, 9), (0, 0, 1))]
        self.assertEqual(markup.describe(hits),
                         "[Marked on the model (see the attached picture): parts “Base”, “Lid” and “Post”, its right side "
                         "(+X), around x 1, y 2 to 4, z 3 mm.]")

    def test_a_curved_or_unknown_surface_names_no_side(self):
        diagonal = MarkHit("Knob", (0, 0, 0), (0.58, 0.58, 0.58))
        unknown = MarkHit("Knob", (10, 0, 0), None)
        self.assertNotIn("its", markup.describe([diagonal, unknown]))
        self.assertNotIn("its", markup.describe([unknown]))

    def test_millimetres_round_halves_up_with_a_real_minus(self):
        self.assertEqual([markup._mm(v) for v in (0.25, -0.04, -2.36, 1234567.0, 3.0)], ["0.3", "0", "−2.4", "1234567", "3"])

    def test_dense_strokes_are_thinned_to_the_pick_limit(self):
        points = [(i / 1000, 0.5) for i in range(1000)]
        picked = markup.sample(points)
        self.assertLessEqual(len(picked), markup.PICKED_POINTS)
        self.assertEqual(picked[0], points[0])
        self.assertEqual(markup.sample(points[:10]), points[:10])


if __name__ == "__main__":
    unittest.main()
