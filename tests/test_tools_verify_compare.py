"""Issue 003: ``compare_to_reference`` — the reconstruction-vs-reference deviation check.

``oscar.refparts.compare`` (the row math), the verify tool in ``oscar.tools_verify`` (resolution, text, persistence
through the ``check_fit`` pattern) and the catalog wiring (``project_validation`` / ``project_docs``) that files the
rows under ``reference.deviation``, on REAL FreeCAD documents under ``freecadcmd`` (the dispatcher runs inline there).
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402
import Part  # noqa: E402
import verify_support as S  # noqa: E402

from oscar import prompt, project_docs, project_validation, tools  # noqa: E402
from oscar.flow import tool_category  # noqa: E402
from oscar.refparts import compare  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402
from oscar.toolrun import LITE_TOOLS, tool_meta  # noqa: E402


def solid(doc, name, shape, visible=True):
    obj = doc.addObject("Part::Feature", name)
    obj.Shape = shape
    obj.Visibility = visible
    doc.recompute()
    return obj


def box(dx, dy, dz, x=0.0, y=0.0, z=0.0):
    return Part.makeBox(dx, dy, dz, App.Vector(x, y, z))


class CompareCase(unittest.TestCase):
    def setUp(self):
        self._made = []
        self.addCleanup(self._close)

    def _close(self):
        from oscar.cad.health_eval import clear_cache

        for name in self._made:
            if name in App.listDocuments():
                App.closeDocument(name)
        clear_cache()

    def new_doc(self, name="Cmp"):
        doc = App.newDocument(name)
        self._made.append(doc.Name)
        return doc

    def run_tool(self, name, ctx=None, **args):
        with use_context(ctx or ToolContext()):
            return tools.run_tool(name, args)


# ---------------------------------------------------------------------------------------------- registration


class RegistrationTest(unittest.TestCase):
    def test_the_tool_is_registered_and_pinned_everywhere(self):
        meta = tool_meta("compare_to_reference")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.category, "verify")
        self.assertTrue(set(meta.phases) & {"VERIFY", "BUILD", "DELIVER"})
        self.assertIn("compare_to_reference", LITE_TOOLS)
        self.assertEqual(tool_category("compare_to_reference"), "verify")
        self.assertIn("compare_to_reference", prompt.KNOWN_TOOLS)

    def test_the_catalog_and_labels_know_the_entry(self):
        self.assertIn("reference.deviation", project_validation.CATALOG)
        self.assertIn("reference.deviation", project_docs.VALIDATION_LABELS)
        self.assertEqual(project_validation.group_of({"kind": "reference_deviation"}), "reference.deviation")


# ---------------------------------------------------------------------------------------------- row math


class CompareShapesTest(unittest.TestCase):
    def pair(self):
        return box(31.0, 20.0, 9.0, 2.0, 0.0, 0.0), box(30.0, 20.0, 8.0)

    def rows(self, **kw):
        built, ref = self.pair()
        rows = compare.compare_shapes(built, ref, built_name="Built", ref_name="Ref", **kw)
        return {r["id"]: r for r in rows}

    def test_an_exact_copy_passes_every_row(self):
        rows = {r["id"]: r for r in compare.compare_shapes(box(30, 20, 8), box(30, 20, 8), built_name="B", ref_name="R")}
        self.assertTrue(rows)
        for r in rows.values():
            self.assertEqual(r["status"], "PASS", r)
            self.assertEqual(r["kind"], "reference_deviation")
        self.assertAlmostEqual(rows["volume.deviation"]["value"], 0.0, places=3)
        self.assertAlmostEqual(rows["overlap.difference"]["value"], 0.0, places=3)

    def test_a_drifted_copy_warns_with_the_real_numbers(self):
        rows = self.rows()
        self.assertEqual(rows["size.deviation_x"]["status"], "WARN")
        self.assertAlmostEqual(rows["size.deviation_x"]["value"], 1.0, places=3)
        self.assertEqual(rows["size.deviation_y"]["status"], "PASS")
        self.assertEqual(rows["size.deviation_z"]["status"], "WARN")
        self.assertAlmostEqual(rows["offset.deviation"]["value"], 2.55, places=1)
        self.assertEqual(rows["volume.deviation"]["status"], "WARN")
        self.assertAlmostEqual(rows["volume.deviation"]["value"], 16.25, places=1)
        self.assertEqual(rows["overlap.difference"]["status"], "WARN")
        self.assertAlmostEqual(rows["overlap.difference"]["value"], 29.6, places=1)

    def test_tolerances_come_from_the_caller(self):
        rows = self.rows(warn_mm=3.0, warn_pct=20.0)
        self.assertEqual(rows["size.deviation_x"]["status"], "PASS")
        self.assertEqual(rows["offset.deviation"]["status"], "PASS")
        self.assertEqual(rows["volume.deviation"]["status"], "PASS")
        self.assertEqual(rows["overlap.difference"]["status"], "WARN", "the overlap limit is fixed, not an argument")

    def test_a_boolean_failure_degrades_to_skip(self):
        built, ref = self.pair()
        with mock.patch("oscar.refparts.compare._symmetric_difference", side_effect=RuntimeError("boom")):
            rows = {r["id"]: r for r in compare.compare_shapes(built, ref, built_name="B", ref_name="R")}
        self.assertEqual(rows["overlap.difference"]["status"], "SKIP")
        self.assertIn("boom", rows["overlap.difference"]["detail"])
        self.assertEqual(rows["size.deviation_x"]["status"], "WARN", "the other rows still ran")


# ---------------------------------------------------------------------------------------------- the tool


class CompareToolTest(CompareCase):
    def test_an_exact_reconstruction_reports_only_pass(self):
        doc = self.new_doc()
        solid(doc, "Ref_Src", box(30, 20, 8), visible=False)
        solid(doc, "Built", box(30, 20, 8))
        text, image, is_error = self.run_tool("compare_to_reference", doc_name=doc.Name, object="Built",
                                              reference="Ref_Src")
        self.assertFalse(is_error, text)
        self.assertIsNone(image)
        self.assertTrue(text.startswith(f"COMPARE_TO_REFERENCE doc '{doc.Name}' (built 'Built' vs reference 'Ref_Src')"),
                        text)
        self.assertIn("PASS", text)
        self.assertFalse([l for l in text.splitlines() if l.startswith(("WARN", "FAIL", "SKIP"))],
                         "an exact copy must have no warning, failure or skip rows")
        for rid in ("size.deviation_x", "offset.deviation", "volume.deviation", "overlap.difference"):
            self.assertIn(rid, text)

    def test_the_reference_is_hidden_after_a_successful_comparison(self):
        doc = self.new_doc()
        solid(doc, "Ref_Src", box(30, 20, 8), visible=True)      # a plain import leaves it in view
        solid(doc, "Built", box(30, 20, 8), visible=False)
        text, _img, is_error = self.run_tool("compare_to_reference", doc_name=doc.Name,
                                             object="Built", reference="Ref_Src")
        self.assertFalse(is_error, text)
        self.assertIsNotNone(doc.getObject("Ref_Src"))           # preserved, only hidden
        self.assertFalse(doc.getObject("Ref_Src").Visibility)
        self.assertTrue(doc.getObject("Built").Visibility)       # the reconstruction is what shows
        self.assertIn("hidden", text.lower())

    def test_a_drifted_reconstruction_warns_but_never_fails(self):
        doc = self.new_doc()
        solid(doc, "Ref_Src", box(30, 20, 8), visible=False)
        solid(doc, "Built", box(31, 20, 9, 2, 0, 0))
        text, _img, is_error = self.run_tool("compare_to_reference", doc_name=doc.Name, object="Built",
                                             reference="Ref_Src")
        self.assertFalse(is_error, text)
        self.assertIn("WARN size.deviation_x", text)
        self.assertFalse([l for l in text.splitlines() if l.startswith("FAIL")],
                         "a deviation is a warning, never a failure")

    def test_defaults_pick_the_largest_visible_built_object_and_the_reference(self):
        doc = self.new_doc()
        solid(doc, "Ref_Src", box(30, 20, 8), visible=False)
        solid(doc, "Small", box(5, 5, 5))
        solid(doc, "Big", box(30, 20, 8))
        text, _img, is_error = self.run_tool("compare_to_reference", doc_name=doc.Name)
        self.assertFalse(is_error, text)
        self.assertIn("built 'Big'", text)
        self.assertIn("reference 'Ref_Src'", text)

    def test_no_reference_is_a_model_correctable_error(self):
        doc = self.new_doc()
        solid(doc, "Built", box(30, 20, 8))
        text, _img, is_error = self.run_tool("compare_to_reference", doc_name=doc.Name)
        self.assertTrue(is_error)
        self.assertIn("reference", text)

    def test_an_unknown_reference_name_names_the_problem(self):
        doc = self.new_doc()
        solid(doc, "Ref_Src", box(30, 20, 8), visible=False)
        solid(doc, "Built", box(30, 20, 8))
        text, _img, is_error = self.run_tool("compare_to_reference", doc_name=doc.Name, reference="Nope")
        self.assertTrue(is_error)
        self.assertIn("'Nope'", text)

    def test_the_rows_are_persisted_as_a_validation_section(self):
        doc = self.new_doc()
        solid(doc, "Ref_Src", box(30, 20, 8), visible=False)
        solid(doc, "Built", box(30, 20, 8))
        project = S.FakeProject()
        self.run_tool("compare_to_reference", ctx=ToolContext(project=project), doc_name=doc.Name,
                      object="Built", reference="Ref_Src")
        self.assertEqual(len(project.recorded), 1)
        record = project.recorded[0]
        self.assertIn("reference.deviation", record["ran"])
        self.assertTrue(record["fit_rows"])
        self.assertTrue(all(r["kind"] == "reference_deviation" for r in record["fit_rows"]))


# ---------------------------------------------------------------------------------------------- catalog rendering


class CatalogRenderingTest(unittest.TestCase):
    def row(self, status="WARN"):
        return {"id": "overlap.difference", "kind": "reference_deviation", "a": "Built", "b": "Ref_Src",
                "metric": "symmetric_difference_pct", "value": 29.6, "limit": 10.0, "status": status,
                "detail": "built 31 x 20 x 9 vs reference 30 x 20 x 8 mm"}

    def test_a_compared_model_lists_the_entry_as_validated(self):
        validation = {"ran": ["health", "reference.deviation"], "gate_errors": [], "fit_rows": [self.row()],
                      "health": {"valid": True, "counts": {"objects": 2, "solids": 2}, "budget_exceeded": False}}
        validated, not_validated = project_docs.classify_validation(validation)
        entry = next(v for v in validated if v["id"] == "reference.deviation")
        self.assertEqual(entry["label"], project_docs.VALIDATION_LABELS["reference.deviation"])

    def test_without_a_comparison_the_entry_says_not_run(self):
        validation = {"ran": ["health"], "gate_errors": [], "fit_rows": [],
                      "health": {"valid": True, "counts": {"objects": 2, "solids": 2}, "budget_exceeded": False}}
        _validated, not_validated = project_docs.classify_validation(validation)
        entry = next(v for v in not_validated if v["id"] == "reference.deviation")
        self.assertEqual(entry["reason"], "not run")


if __name__ == "__main__":
    unittest.main()
