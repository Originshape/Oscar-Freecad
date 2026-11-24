"""The mesh-to-parametric reconstruction benchmark (issue: reconstruction accuracy).

For each committed plan fixture: build the plan natively, tessellate the Body at a coarse, scan-like
deviation (the "source mesh" — its ground truth is the plan's parameters), then measure what the
survey recovers and what the deviation rows say:

* dimension estimation error per recovered dimension (hole / bore / step diameters, positions, extents);
* deviation rows of the built body against the source mesh (near-zero everywhere);
* parametric editability through the existing +-10% rebuild;
* wall-clock times of build, survey and comparison.

The tolerances are the benchmark's acceptance bars (chord error of the 0.3 mm tessellation included);
they are honest numbers measured on this suite, not aspirations. ``test_*`` methods that only RECORD
timings print them (visible in the module log) and assert generous upper bounds.
"""

import json
import os
import sys
import time
import unittest
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402

from oscar.cad import health_eval  # noqa: E402
from oscar.plan import builder  # noqa: E402
from oscar.refparts import compare, mesh_survey  # noqa: E402
from oscar.tools_verify_edit import editability_test  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "plans")
TESS = 0.3          # mm: the scan-like source-mesh deviation
CYL_TOL = 0.65      # mm: chord error of a cylinder at TESS (+ sew tolerance)
TIMES: dict[str, dict] = {}


def load_plan(name: str) -> dict:
    with open(os.path.join(FIXTURES, f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def source_mesh(body, deviation=TESS) -> "Mesh.Mesh":
    verts, tris = body.Shape.tessellate(deviation)
    return Mesh.Mesh([list(verts[i]) + list(verts[j]) + list(verts[k]) for i, j, k in tris])


class BenchmarkCase(unittest.TestCase):
    def setUp(self):
        self.doc = App.newDocument(f"BM{uuid.uuid4().hex[:6]}")
        self.addCleanup(App.closeDocument, self.doc.Name)

    def build(self, name: str):
        plan = load_plan(name)
        t0 = time.perf_counter()
        result = builder.apply_plan(self.doc, plan)
        build_s = time.perf_counter() - t0
        self.assertTrue(result["success"], result["error"])
        body = self.doc.getObject(result["body"])
        return plan, body, build_s

    def survey_of(self, body):
        mesh = source_mesh(body)
        t0 = time.perf_counter()
        survey = mesh_survey.survey_mesh(mesh)
        survey_s = time.perf_counter() - t0
        return mesh, survey, survey_s

    def compare_rows(self, body, mesh):
        t0 = time.perf_counter()
        rows = compare.deviation_rows(body.Shape, mesh)
        return rows, time.perf_counter() - t0

    def record(self, name: str, **numbers):
        TIMES.setdefault(name, {}).update(numbers)

    @classmethod
    def tearDownClass(cls):
        if TIMES:
            print("\nRECONSTRUCTION BENCHMARK (times in seconds)")
            for name, t in sorted(TIMES.items()):
                print(f"  {name}: " + ", ".join(f"{k}={v}" if isinstance(v, str) else f"{k}={v:.2f}"
                                                for k, v in sorted(t.items())))


class PlateWithFourHolesTest(BenchmarkCase):
    def test_dimensions_are_recovered_within_tolerance_and_deviation_rows_pass(self):
        plan, body, build_s = self.build("plate_4holes")
        mesh, survey, survey_s = self.survey_of(body)
        rows, compare_s = self.compare_rows(body, mesh)
        self.record("plate_4holes", build_s=build_s, survey_s=survey_s, compare_s=compare_s,
                    facets=str(survey["facets"]))
        self.assertTrue(survey["closed"])

        holes = [c for c in survey["surface"]["cylinders"] if c["concave"]]
        self.assertEqual(len(holes), 4)
        for h in holes:
            self.assertAlmostEqual(h["diameter"], plan["parameters"]["hole_dia"]["value"], delta=CYL_TOL)
        # the fixture's own truth: holes inset 20 / 15 mm from the 0-corner of the plate
        centers = sorted(tuple(round(c, 1) for c in h["center"]) for h in holes)
        self.assertEqual(centers, [(20.0, 15.0, 4.0), (20.0, 45.0, 4.0),
                                   (80.0, 15.0, 4.0), (80.0, 45.0, 4.0)])

        size = survey["bbox"]["size"]
        self.assertAlmostEqual(size[0], 100.0, delta=0.05)
        self.assertAlmostEqual(size[1], 60.0, delta=0.05)
        self.assertAlmostEqual(size[2], 8.0, delta=0.05)

        by_id = {r["id"]: r for r in rows}
        self.assertLessEqual(by_id["deviation.bbox_axis"]["value"], 0.05)
        self.assertLessEqual(by_id["deviation.surface_sampled"]["value"], TESS + 0.05)
        self.assertGreaterEqual(by_id["deviation.iou_exact"]["value"], 0.995)
        self.record("plate_4holes", hole_dia_err_mm=round(max(abs(h["diameter"] - 10.0) for h in holes), 3),
                    iou=round(by_id["deviation.iou_exact"]["value"], 5))

    def test_the_model_stays_editable_after_reconstruction(self):
        _plan, body, _t = self.build("plate_4holes")
        table = health_eval.read_param_table(self.doc)
        edit = editability_test(self.doc, table, budget_s=60.0)
        self.record("plate_4holes", editability_s=edit["elapsed_s"])
        self.assertEqual(edit["broken"], [])
        self.assertTrue(edit["restored"])


class LBracketTest(BenchmarkCase):
    def test_the_bore_diameter_and_leg_sizes_are_recovered(self):
        plan, body, build_s = self.build("l_bracket")
        mesh, survey, survey_s = self.survey_of(body)
        rows, compare_s = self.compare_rows(body, mesh)
        self.record("l_bracket", build_s=build_s, survey_s=survey_s, compare_s=compare_s,
                    facets=str(survey["facets"]))
        self.assertTrue(survey["closed"])

        bores = [c for c in survey["surface"]["cylinders"] if c["concave"]]
        self.assertEqual(len(bores), 1)
        self.assertAlmostEqual(bores[0]["diameter"], plan["parameters"]["bore_dia"]["value"], delta=CYL_TOL)

        size = survey["bbox"]["size"]
        self.assertAlmostEqual(size[0], 70.0, delta=0.05)
        self.assertAlmostEqual(size[1], 50.0, delta=0.05)
        self.assertAlmostEqual(size[2], 50.0, delta=0.05)

        by_id = {r["id"]: r for r in rows}
        self.assertLessEqual(by_id["deviation.surface_sampled"]["value"], TESS + 0.05)
        self.assertGreaterEqual(by_id["deviation.iou_exact"]["value"], 0.995)
        self.record("l_bracket", bore_dia_err_mm=round(abs(bores[0]["diameter"] - 12.0), 3),
                    iou=round(by_id["deviation.iou_exact"]["value"], 5))

    def test_the_model_stays_editable(self):
        _plan, _body, _t = self.build("l_bracket")
        table = health_eval.read_param_table(self.doc)
        edit = editability_test(self.doc, table, budget_s=60.0)
        self.record("l_bracket", editability_s=edit["elapsed_s"])
        self.assertEqual(edit["broken"], [])
        self.assertTrue(edit["restored"])


class SteppedShaftTest(BenchmarkCase):
    def test_every_step_diameter_and_length_is_recovered(self):
        plan, body, build_s = self.build("stepped_shaft")
        mesh, survey, survey_s = self.survey_of(body)
        rows, compare_s = self.compare_rows(body, mesh)
        self.record("stepped_shaft", build_s=build_s, survey_s=survey_s, compare_s=compare_s,
                    facets=str(survey["facets"]))
        self.assertTrue(survey["closed"])

        steps = sorted((c for c in survey["surface"]["cylinders"] if not c["concave"]),
                       key=lambda c: -c["diameter"])
        self.assertEqual([round(s["diameter"], 1) for s in steps], [20.0, 12.0, 8.0])
        for step in steps:
            self.assertAlmostEqual(max(abs(c) for c in step["axis"]), 1.0, delta=1e-3)
        self.assertAlmostEqual(steps[0]["depth"], 40.0, delta=0.35)
        self.assertAlmostEqual(steps[1]["depth"], 30.0, delta=0.35)
        self.assertAlmostEqual(steps[2]["depth"], 20.0, delta=0.35)

        by_id = {r["id"]: r for r in rows}
        self.assertGreaterEqual(by_id["deviation.iou_exact"]["value"], 0.995)
        self.record("stepped_shaft", step_dia_err_mm=round(max(
            abs(s["diameter"] - d) for s, d in zip(steps, (20.0, 12.0, 8.0))), 3),
            iou=round(by_id["deviation.iou_exact"]["value"], 5))

    def test_editability_baseline_is_recorded(self):
        # pre-existing on this base: one stepped_shaft parameter breaks at +-10% (recorded, not fixed here)
        _plan, _body, _t = self.build("stepped_shaft")
        table = health_eval.read_param_table(self.doc)
        edit = editability_test(self.doc, table, budget_s=60.0)
        self.record("stepped_shaft", editability_s=edit["elapsed_s"],
                    editability_broken=str(len(edit["broken"])), editability_inert=str(len(edit["inert"])))
        self.assertTrue(edit["restored"])


class WrongDimensionCorrectionTest(BenchmarkCase):
    def test_a_wrongly_estimated_dimension_is_visible_and_fixable_in_the_sheet(self):
        """The correction loop the reconstruction playbook prescribes, proven end to end:
        build with a wrong hole diameter, see the deficit in the deviation rows, fix the
        spreadsheet cell, recompute, see the rows recover."""
        plan, body, _t = self.build("plate_4holes")
        mesh = source_mesh(body)
        sheet = self.doc.getObject("Parameters")
        sheet.set(sheet.getCellFromAlias("hole_dia"), "=12.0")
        self.doc.recompute()
        before = {r["id"]: r for r in compare.deviation_rows(body.Shape, mesh)}
        self.assertLess(before["deviation.volume_share"]["value"], -0.5)   # missing material
        self.assertEqual(before["deviation.volume_share"]["status"], "WARN")

        sheet.set(sheet.getCellFromAlias("hole_dia"), "=10.0")
        self.doc.recompute()
        after = {r["id"]: r for r in compare.deviation_rows(body.Shape, mesh)}
        self.assertLessEqual(abs(after["deviation.volume_share"]["value"]), 0.1)
        self.assertEqual(after["deviation.volume_share"]["status"], "PASS")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
