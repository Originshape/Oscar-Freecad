"""V-2: export re-import equality — an exported STEP/BREP is re-imported and its
volume/bbox compared against the shape that was written, so the sha256 (bytes) is
backed by a geometry verdict. Mesh/drawing formats are skipped by design (tessellations
and projections are not the same geometry), a failed re-import is an honest SKIP, and a
real mismatch is a visible MISMATCH note — never a silent pass.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Part  # noqa: E402

from oscar import project_export as X  # noqa: E402


def _write_step(shape, name):
    path = os.path.join(isolation.fresh_dir("v2exp_"), f"{name}.step")
    shape.exportStep(path)
    return path


class VerifyReimportTest(unittest.TestCase):
    def test_a_box_round_trip_is_equal(self):
        shape = Part.makeBox(30, 20, 10)
        got = X.verify_reimport(_write_step(shape, "box"), shape)
        self.assertEqual(got["verdict"], "EQUAL", got)
        self.assertAlmostEqual(got["volume_delta_pct"], 0.0, places=3)
        self.assertLessEqual(got["bbox_max_delta_mm"], 1e-3)

    def test_a_cylinder_round_trip_is_equal(self):
        shape = Part.makeCylinder(8, 25)
        got = X.verify_reimport(_write_step(shape, "cyl"), shape)
        self.assertEqual(got["verdict"], "EQUAL", got)

    def test_a_different_reference_is_a_mismatch(self):
        exported = Part.makeBox(30, 20, 10)
        wrong = Part.makeBox(30, 20, 12)                 # 20 % more volume
        got = X.verify_reimport(_write_step(exported, "mis"), wrong)
        self.assertEqual(got["verdict"], "MISMATCH", got)
        self.assertGreater(abs(got["volume_delta_pct"]), 5.0)
        self.assertIn("volume", got["note"])

    def test_an_unreadable_file_skips_honestly(self):
        target = os.path.join(isolation.fresh_dir("v2bad_"), "nope.step")
        got = X.verify_reimport(target, Part.makeBox(1, 1, 1))
        self.assertEqual(got["verdict"], "SKIPPED")
        self.assertIn("re-import failed", got["note"])


class ExportVerifyWiringTest(unittest.TestCase):
    def _doc_with_box(self, name):
        doc = App.newDocument(name)
        obj = doc.addObject("Part::Feature", "Body")
        obj.Shape = Part.makeBox(40, 25, 8)
        doc.recompute()
        return doc

    def test_step_files_carry_the_verdict_and_stl_does_not(self):
        doc = self._doc_with_box("V2Step")
        try:
            target = isolation.fresh_dir("v2w_")
            result = X.export_tips(None, doc.Name, formats=("step", "stl"),
                                   directory=target)
            self.assertEqual(len(result.files), 2)
            by_fmt = {f.format: f for f in result.files}
            self.assertEqual(by_fmt["step"].verified["verdict"], "EQUAL",
                             by_fmt["step"].verified)
            self.assertIsNone(by_fmt["stl"].verified,
                              "mesh formats are tessellations — never re-import compared")
            self.assertEqual(result.errors, [])
        finally:
            App.closeDocument(doc.Name)

    def test_verify_false_leaves_the_verdict_off(self):
        doc = self._doc_with_box("V2Off")
        try:
            target = isolation.fresh_dir("v2off_")
            result = X.export_tips(None, doc.Name, formats=("step",),
                                   directory=target, verify=False)
            self.assertIsNone(result.files[0].verified)
        finally:
            App.closeDocument(doc.Name)


if __name__ == "__main__":
    unittest.main()
