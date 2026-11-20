"""P5.5: the CADBench adapter runs end to end on synthetic STLs (the official
dataset is not bundled; real-dataset validation is BLOCKED and documented in
the adapter's docstring). Also exercises the P5.4 parametric grader."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402

from oscar.recon import reconstruct  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CadbenchAdapterTest(unittest.TestCase):
    def test_adapter_grades_synthetic_cases_end_to_end(self):
        mod = _load("cadbench_adapter",
                    os.path.join(ROOT, "scripts", "cadbench_adapter.py"))
        tmp = tempfile.mkdtemp(prefix="cb_adapter_")
        cases = os.path.join(tmp, "cases")
        os.makedirs(cases)
        shape = Part.makeBox(40, 30, 8)
        for x, y in ((10, 8), (30, 8), (10, 22), (30, 22)):
            shape = shape.cut(Part.makeCylinder(3.0, 30, App.Vector(x, y, -8)))
        shape.tessellate(0.3)
        stl = os.path.join(cases, "bracket.stl")
        shape.exportStl(stl)
        out = os.path.join(tmp, "out")
        rows = mod.run(cases, out)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["case"], "bracket.stl")
        self.assertEqual(row["outcome"], "verified_parametric", row)
        self.assertIsNotNone(row["iou"])
        self.assertGreaterEqual(row["iou"], 0.99,
                                "an exact reconstruction scores ~1.0 IoU")
        files = os.listdir(out)
        self.assertTrue(any(f.endswith(".json") for f in files), files)
        data = json.load(open(os.path.join(
            out, next(f for f in files if f.endswith(".json"))), encoding="utf-8"))
        self.assertEqual(data["cases"][0]["case"], "bracket.stl")


class GraderTest(unittest.TestCase):
    def test_grader_scores_a_reconstruction_body(self):
        shape = Part.makeBox(60, 40, 10)
        for x, y in ((15, 10), (45, 10), (15, 30), (45, 30)):
            shape = shape.cut(Part.makeCylinder(3.3, 30, App.Vector(x, y, -10)))
        mesh = Mesh.Mesh(shape.tessellate(0.3))
        doc = App.newDocument("GradeA")
        self.addCleanup(App.closeDocument, "GradeA")
        result = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
        self.assertTrue(result["build"]["success"], result["build"])
        from oscar.recon import grader

        rows = grader.grade(doc, result["build"]["body"], mesh=mesh)
        by_id = {r["id"]: r for r in rows}
        self.assertEqual(by_id["grader.parametric.native_features"]["status"], "PASS",
                         by_id["grader.parametric.native_features"])
        self.assertEqual(by_id["grader.parametric.sketches_fully_constrained"]["status"],
                         "PASS")
        self.assertEqual(by_id["grader.parametric.recompute_stable"]["status"], "PASS")
        self.assertEqual(by_id["grader.parametric.edit_probe"]["value"], 2,
                         by_id["grader.parametric.edit_probe"])
        self.assertGreaterEqual(by_id["grader.geometry.iou"]["value"], 0.99,
                                by_id["grader.geometry.iou"])
        self.assertEqual(by_id["grader.geometry.bbox_dz"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
