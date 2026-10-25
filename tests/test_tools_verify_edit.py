"""The editability rebuild of ``validate_model(deep=True)`` (``oscar.tools_verify_edit``) and the pure text
renderings of the verification tools (``oscar.tools_verify_text``).

The rebuild runs on REAL documents under ``freecadcmd``: the driver parameters (the up to 8 with the most bound
properties) are pushed +-10%, the document is recomputed and judged by the STRICT health check, and every cell must be
back verbatim afterwards, also after a crash, a Stop and the time cap."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402
import verify_support as S  # noqa: E402

from oscar import tools_verify_edit as E  # noqa: E402
from oscar import tools_verify_text as T  # noqa: E402
from oscar.cad import health_eval  # noqa: E402
from oscar.cad.health_types import StateSnapshot  # noqa: E402


# ---------------------------------------------------------------------------------------------- text (pure)


class NumberFormatTest(unittest.TestCase):
    def test_fmt_num_is_compact_and_total(self):
        self.assertEqual(T.fmt_num(2), "2")
        self.assertEqual(T.fmt_num(2.0), "2")
        self.assertEqual(T.fmt_num(0.0125), "0.0125")
        self.assertEqual(T.fmt_num(1234567.0), "1.23457e+06")
        self.assertEqual(T.fmt_num(None), "?")
        self.assertEqual(T.fmt_num(float("nan")), "?")
        self.assertEqual(T.fmt_num(True), "True")
        self.assertEqual(T.fmt_num("2 mm"), "2 mm")

    def test_fmt_val_keeps_a_decimal_point_on_floats(self):
        self.assertEqual(T.fmt_val(0.0), "0.0")
        self.assertEqual(T.fmt_val(2.0), "2.0")
        self.assertEqual(T.fmt_val(0.64), "0.64")
        self.assertEqual(T.fmt_val(18432.1), "18432.1")
        self.assertEqual(T.fmt_val(3), "3")
        self.assertEqual(T.fmt_val(None), "?")


class ParamsBlockTest(unittest.TestCase):
    ROWS = [{"name": "wall_t", "value": 2.0, "unit": "mm", "source": "assumption", "bound": 9, "needs_rebuild": False},
            {"name": "hook_count", "value": 4, "unit": "", "source": "user_requirement", "bound": 0,
             "needs_rebuild": True},
            {"name": "broken", "value": None, "unit": "mm", "source": "", "bound": 0}]

    def test_no_sheet_says_so_in_one_line(self):
        for table in (None, {}, {"sheet": "", "rows": []}):
            self.assertEqual(T.params_block(table),
                             "PARAMETERS: no 'Params' or 'Parameters' spreadsheet in this document (no driving "
                             "parameters to check)")

    def test_header_counts_rows_and_bound_rows(self):
        text = T.params_block({"sheet": "Params", "rows": self.ROWS})
        self.assertEqual(text.split("\n")[0], "PARAMETERS (sheet 'Params', 3 rows, 1 bound to objects)")

    def test_columns_are_aligned_and_flags_are_shown(self):
        lines = T.params_block({"sheet": "Params", "rows": self.ROWS}).split("\n")
        self.assertEqual(lines[1].split(), ["name", "value", "unit", "source", "bound"])
        self.assertRegex(lines[2], r"^wall_t\s+2\s+mm\s+assumption\s+9$")
        self.assertIn("0 (rebuild: set_parameter)", lines[3])
        self.assertRegex(lines[4], r"^broken\s+\?\s+mm")
        starts = {line.index("mm") for line in (lines[2], lines[4])}
        self.assertEqual(len(starts), 1)                               # the unit column lines up

    def test_an_empty_sheet_is_just_the_header(self):
        self.assertEqual(T.params_block({"sheet": "Parameters", "rows": []}),
                         "PARAMETERS (sheet 'Parameters', 0 rows, 0 bound to objects)")

    def test_rows_are_capped_with_a_more_line(self):
        rows = [{"name": f"p_{i:03d}", "value": i, "unit": "mm", "source": "assumption", "bound": 1} for i in range(100)]
        lines = T.params_block({"sheet": "Params", "rows": rows}).split("\n")
        self.assertEqual(len(lines), 1 + 1 + T.MAX_PARAM_ROWS + 1)
        self.assertEqual(lines[-1], "(+60 more)")
        small = T.params_block({"sheet": "Params", "rows": rows}, max_rows=3).split("\n")
        self.assertEqual(small[-1], "(+97 more)")
        self.assertEqual(len(small), 1 + 1 + 3 + 1)

    def test_long_names_are_clipped_not_wrapped(self):
        rows = [{"name": "n" * 60, "value": 1, "unit": "mm", "source": "s" * 40, "bound": 1}]
        self.assertEqual(len(T.params_block({"sheet": "Params", "rows": [dict(rows[0], name="n" * 32)]})
                             .split("\n")[2].split()[0]), 32)          # a legal 32-character alias is never cut
        line = T.params_block({"sheet": "Params", "rows": rows}).split("\n")[2]
        self.assertLessEqual(len(line), 40 + 2 + 5 + 2 + 8 + 2 + 18 + 2 + 3 + 4)
        self.assertIn("~", line)


class FitTextTest(unittest.TestCase):
    PASS = {"id": "interference.shell_lid", "kind": "interference", "a": "Shell", "b": "Lid",
            "metric": "common_volume_mm3", "value": 0.0, "limit": "<= 0.05", "status": "PASS", "detail": ""}

    def test_the_documented_pass_line(self):
        self.assertEqual(T.fit_line(self.PASS),
                         "PASS interference.shell_lid  Shell vs Lid: common_volume_mm3 = 0.0 (<= 0.05)")

    def test_a_fail_line_carries_its_detail_a_pass_line_does_not(self):
        fail = dict(self.PASS, status="FAIL", value=4.2, detail="move the lid")
        self.assertEqual(T.fit_line(fail), "FAIL interference.shell_lid  Shell vs Lid: common_volume_mm3 = 4.2 "
                                           "(<= 0.05) - move the lid")
        self.assertNotIn("move", T.fit_line(dict(self.PASS, detail="move the lid")))

    def test_a_skip_line_is_the_reason(self):
        row = {"id": "interference.reference_shell", "status": "SKIP", "detail": "no Ref_* object in the document"}
        self.assertEqual(T.fit_line(row), "SKIP interference.reference_shell  no Ref_* object in the document")
        self.assertEqual(T.fit_line({"id": "x", "status": "skip"}), "SKIP x  not evaluated")

    def test_one_sided_and_coverage_rows(self):
        one = dict(self.PASS, b="", status="WARN", detail="tight")
        self.assertEqual(T.fit_line(one), "WARN interference.shell_lid  Shell: common_volume_mm3 = 0.0 (<= 0.05) - tight")
        cover = {"id": "reference.status", "kind": "coverage", "a": "", "b": "", "metric": "reference_status",
                 "status": "WARN", "detail": "reference is nominal board data"}
        self.assertEqual(T.fit_line(cover), "WARN reference.status  reference is nominal board data")

    def test_a_row_without_a_limit_has_no_empty_parentheses(self):
        row = dict(self.PASS, limit="")
        self.assertEqual(T.fit_line(row), "PASS interference.shell_lid  Shell vs Lid: common_volume_mm3 = 0.0")

    def test_summary_counts_and_unknown_statuses_count_as_skip(self):
        rows = [{"status": "PASS"}] * 9 + [{"status": "WARN"}, {"status": "SKIP"}, {"status": "weird"}, {}]
        self.assertEqual(T.fit_counts(rows), {"PASS": 9, "WARN": 1, "FAIL": 0, "SKIP": 3})
        self.assertEqual(T.fit_summary(rows), "FIT: 9 PASS, 1 WARN, 0 FAIL, 3 SKIP")
        self.assertEqual(T.fit_summary([]), "FIT: 0 PASS, 0 WARN, 0 FAIL, 0 SKIP")

    def test_block_is_one_line_per_row_then_the_summary(self):
        lines = T.fit_block([self.PASS, dict(self.PASS, id="a.b")]).split("\n")
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[-1], "FIT: 2 PASS, 0 WARN, 0 FAIL, 0 SKIP")

    def test_dfm_block_lists_only_warn_and_fail_rows(self):
        rows = [{"id": "dfm.min_wall", "status": "PASS", "detail": "ok"},
                {"id": "dfm.min_web", "status": "WARN", "detail": "web 1.0 < 1.2"},
                {"id": "dfm.hook_strain", "status": "FAIL", "detail": "strain 120%"},
                {"id": "dfm.shrink", "status": "INFO", "detail": "note"}]
        lines = T.dfm_block(rows).split("\n")
        self.assertEqual(lines[0], "DFM: 1 PASS, 1 WARN, 1 FAIL, 1 INFO")
        self.assertEqual(lines[1:], ["WARN dfm.min_web  web 1.0 < 1.2", "FAIL dfm.hook_strain  strain 120%"])
        self.assertEqual(T.dfm_block([]), "")


class EditTextTest(unittest.TestCase):
    RESULT = {"parameters": [
        {"parameter": "wall_t", "plus": {"ok": True, "problems": []}, "minus": {"ok": True, "problems": []}},
        {"parameter": "clearance", "plus": {"ok": True, "problems": []},
         "minus": {"ok": False, "problems": ["FRAGMENTS Shell", "NOT_CLOSED Shell", "third"]}},
        {"parameter": "late", "plus": {"ok": True, "problems": []}, "minus": None}],
        "elapsed_s": 6.24, "budget_s": 20.0}

    def test_the_documented_line_shape(self):
        lines = T.edit_block(self.RESULT).split("\n")
        self.assertEqual(lines[0], "EDITABILITY (3 parameters, +-10%, 6.2 s)")
        self.assertEqual(lines[1], "wall_t +10% ok, -10% ok | clearance +10% ok, -10% FAIL(FRAGMENTS Shell; NOT_CLOSED "
                                   "Shell) | late +10% ok, -10% not checked")
        self.assertEqual(len(lines), 2)

    def test_optional_lines_appear_only_when_there_is_something_to_say(self):
        res = dict(self.RESULT, inert=["a", "b"], not_checked=["z"], skipped=[{"parameter": "k", "reason": "formula"}],
                   restored=False)
        text = T.edit_block(res)
        self.assertIn("inert (changing them moved no object): a, b", text)
        self.assertIn("not checked (20 s cap): z", text)
        self.assertIn("skipped: k (formula)", text)
        self.assertIn("WARNING: a parameter cell could not be restored", text)

    def test_error_and_empty_inputs(self):
        self.assertEqual(T.edit_block({"error": "no sheet", "parameters": []}), "EDITABILITY not run: no sheet")
        self.assertEqual(T.edit_block(None), "")
        self.assertEqual(T.edit_block({}), "")
        self.assertEqual(T.edit_block({"parameters": [], "elapsed_s": 0.0}).split("\n")[0],
                         "EDITABILITY (0 parameters, +-10%, 0.0 s)")

    def test_many_skipped_parameters_are_summarised(self):
        res = {"parameters": [], "skipped": [{"parameter": f"p{i}", "reason": "r"} for i in range(12)]}
        self.assertIn("(+4 more)", T.edit_block(res))


# ---------------------------------------------------------------------------------------------- picking and values (pure)


class PickParametersTest(unittest.TestCase):
    def row(self, name, value=2.0, bound=1, expr="", needs_rebuild=False):
        return {"name": name, "value": value, "bound": bound, "expr": expr, "needs_rebuild": needs_rebuild}

    def test_skipped_rows_say_why(self):
        table = {"rows": [self.row("ok_a"), self.row("formula_a", expr="=ok_a*2"), self.row("hook_count", 4),
                          self.row("stage_p", needs_rebuild=True), self.row("unbound", bound=0), self.row("zero", 0.0),
                          self.row("text", "abc"), self.row("flag", True)]}
        chosen, skipped = E.pick_parameters(table)
        self.assertEqual([r["name"] for r in chosen], ["ok_a"])
        reasons = {s["parameter"]: s["reason"] for s in skipped}
        self.assertEqual(reasons["formula_a"], "formula: follows its sources")
        self.assertIn("count / stage parameter", reasons["hook_count"])
        self.assertIn("count / stage parameter", reasons["stage_p"])
        self.assertEqual(reasons["unbound"], "no object property is bound to it")
        self.assertIn("value is 0", reasons["zero"])
        self.assertEqual(reasons["text"], "not a number")
        self.assertEqual(reasons["flag"], "not a number")

    def test_the_highest_bound_win_with_name_ties_and_the_limit_is_reported(self):
        rows = [self.row(f"p_{i}", bound=1 + (i % 3)) for i in range(12)]
        chosen, skipped = E.pick_parameters({"rows": rows}, limit=8)
        self.assertEqual(len(chosen), 8)
        self.assertEqual([r["bound"] for r in chosen], sorted((r["bound"] for r in chosen), reverse=True))
        self.assertEqual([r["name"] for r in chosen if r["bound"] == 3], sorted(r["name"] for r in chosen if r["bound"] == 3))
        beyond = [s for s in skipped if "beyond the 8 highest-bound" in s["reason"]]
        self.assertEqual(len(beyond), 4)
        self.assertTrue(all(r["bound"] == 1 for r in rows if r["name"] in {b["parameter"] for b in beyond}))

    def test_no_table_picks_nothing(self):
        self.assertEqual(E.pick_parameters({}), ([], []))
        self.assertEqual(E.pick_parameters(None), ([], []))


class PerturbationTest(unittest.TestCase):
    def test_plain_values_move_by_the_percentage(self):
        self.assertAlmostEqual(E._perturbed(2.0, +1, 0.10, "mm"), 2.2)
        self.assertAlmostEqual(E._perturbed(2.0, -1, 0.10, "mm"), 1.8)
        self.assertAlmostEqual(E._perturbed(-5.0, +1, 0.10, "mm"), -5.5)
        self.assertAlmostEqual(E._perturbed(20, +1, 0.5, ""), 30.0)

    def test_an_angle_is_never_pushed_past_its_limit(self):
        self.assertAlmostEqual(E._perturbed(90.0, +1, 0.10, "deg"), 99.0)
        self.assertAlmostEqual(E._perturbed(350.0, +1, 0.10, "deg"), 360.0)          # clamped to a full turn
        self.assertIsNone(E._perturbed(360.0, +1, 0.10, "deg"))                       # already on the limit: left out
        self.assertAlmostEqual(E._perturbed(170.0, +1, 0.10, "°"), 179.99)           # a small angle stays under 180
        self.assertAlmostEqual(E._perturbed(170.0, -1, 0.10, "deg"), 153.0)
        self.assertIsNone(E._perturbed(179.99, +1, 0.10, "Deg"))

    def test_cell_content_is_always_a_formula(self):
        self.assertEqual(E._content(2.2, "mm"), "=2.2 mm")
        self.assertEqual(E._content(22.0, ""), "=22")
        self.assertEqual(E._content(-1.5, " mm "), "=-1.5 mm")
        self.assertEqual(E._content(1 / 3, "mm"), "=0.3333333333 mm")


# ---------------------------------------------------------------------------------------------- the rebuild (real documents)


class _DocSpy:
    """A document that delegates everything and can fail a chosen ``recompute``."""

    def __init__(self, doc, fail_on=()):
        self._doc = doc
        self.calls = 0
        self.fail_on = set(fail_on)

    def recompute(self, *args, **kwargs):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("recompute exploded")
        return self._doc.recompute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._doc, name)


class RebuildCase(unittest.TestCase):
    def setUp(self):
        health_eval.clear_cache()
        self._made = []
        self.addCleanup(self._close)

    def _close(self):
        for name in self._made:
            if name in App.listDocuments():
                App.closeDocument(name)
        health_eval.clear_cache()

    def parametric(self, **kw):
        doc = App.newDocument("EditT")
        self._made.append(doc.Name)
        self.objs = S.build_parametric(doc, **kw)
        return doc

    def table(self, doc):
        return health_eval.read_param_table(doc)

    def cells(self, doc):
        sheet = doc.getObject("Params")
        return {addr: sheet.getContents(addr) for addr in sheet.getUsedCells()}


class EditabilityTest(RebuildCase):
    def test_a_robust_model_survives_every_edit_and_the_sheet_is_restored(self):
        doc = self.parametric(slot=8.0)
        before = self.cells(doc)
        volume = self.objs["body"].Shape.Volume
        result = E.editability_test(doc, self.table(doc))
        self.assertTrue(result["ok"], result["broken"])
        self.assertEqual(result["checked"], 8)
        self.assertEqual([p["parameter"] for p in result["parameters"]], ["hei", "len", "slot", "wid"])   # bound 1 each: by name
        for entry in result["parameters"]:
            self.assertTrue(entry["plus"]["ok"] and entry["minus"]["ok"], entry)
            self.assertFalse(entry["inert"], entry["parameter"])
        self.assertEqual((result["broken"], result["inert"], result["not_checked"]), ([], [], []))
        self.assertTrue(result["restored"])
        self.assertEqual(self.cells(doc), before)
        self.assertAlmostEqual(self.objs["body"].Shape.Volume, volume, places=6)
        self.assertEqual(self.objs["body"].State, ["Up-to-date"] if "Up-to-date" in self.objs["body"].State else
                         self.objs["body"].State)
        self.assertNotIn("Touched", self.objs["body"].State)

    def test_the_values_pushed_are_plus_and_minus_ten_percent(self):
        doc = self.parametric(slot=8.0)
        result = E.editability_test(doc, self.table(doc))
        by = {p["parameter"]: p for p in result["parameters"]}
        self.assertEqual((by["len"]["plus"]["value"], by["len"]["minus"]["value"]), (22.0, 18.0))
        self.assertEqual((by["slot"]["plus"]["value"], by["slot"]["minus"]["value"]), (8.8, 7.2))
        self.assertEqual(by["len"]["unit"], "mm")

    def test_an_edit_that_fragments_the_shell_is_reported_with_its_error_code(self):
        doc = self.parametric(slot=9.5)
        result = E.editability_test(doc, self.table(doc))
        self.assertFalse(result["ok"])
        slot = next(p for p in result["parameters"] if p["parameter"] == "slot")
        self.assertTrue(slot["minus"]["ok"])
        self.assertFalse(slot["plus"]["ok"])
        self.assertEqual(slot["plus"]["problems"], ["FRAGMENTS Body"])
        names = {(b["parameter"], b["delta"]) for b in result["broken"]}
        self.assertIn(("slot", 0.1), names)
        self.assertIn(("wid", -0.1), names)                     # a 9 mm block cannot hold a 9.5 mm slot either
        self.assertTrue(result["restored"])
        self.assertEqual(doc.getObject("Params").getContents("B5"), "=9.5 mm")
        self.assertEqual(len(self.objs["body"].Shape.Solids), 1)

    def test_an_error_that_already_existed_is_not_blamed_on_the_edit(self):
        doc = self.parametric(slot=8.0)
        cut = doc.addObject("Part::Cut", "UserCut")
        cut.Base = doc.addObject("Part::Box", "UserBase")
        doc.recompute()
        result = E.editability_test(doc, self.table(doc))
        self.assertTrue(result["ok"], result["broken"])

    def test_a_parameter_that_moves_nothing_is_inert(self):
        doc = self.parametric(slot=8.0)
        frozen = StateSnapshot(doc=doc.Name, objects={"Body": {"volume": 1.0, "bbox": [0, 0, 0, 1, 1, 1],
                                                              "placement": "p"}})
        with mock.patch.object(health_eval, "snapshot_state", return_value=frozen):
            result = E.editability_test(doc, self.table(doc))
        self.assertEqual(sorted(result["inert"]), ["hei", "len", "slot", "wid"])
        self.assertTrue(all(p["inert"] for p in result["parameters"]))
        self.assertTrue(result["ok"])                         # inert is a warning, not a failure

    def test_a_unitless_parameter_is_read_and_written_without_a_unit(self):
        doc = App.newDocument("UnitlessT")
        self._made.append(doc.Name)
        S.params(doc, [("scale", "=20", "", "assumption")])
        box = doc.addObject("Part::Box", "Cube")
        box.setExpression("Length", "Params.scale * 1 mm")
        doc.recompute()
        result = E.editability_test(doc, self.table(doc))
        self.assertTrue(result["ok"], result)
        entry = result["parameters"][0]
        self.assertEqual((entry["parameter"], entry["unit"], entry["plus"]["value"]), ("scale", "", 22.0))
        self.assertEqual(doc.getObject("Params").getContents("B2"), "20")        # FreeCAD normalises "=20"
        self.assertEqual(box.Length.Value, 20.0)

    def test_the_time_cap_stops_before_the_next_edit_and_lists_what_was_not_reached(self):
        doc = self.parametric(slot=8.0)
        ticks = iter([0.0, 1.0, 2.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0])
        result = E.editability_test(doc, self.table(doc), budget_s=20.0, clock=lambda: next(ticks))
        self.assertEqual(result["checked"], 2)
        self.assertEqual([p["parameter"] for p in result["parameters"]], ["hei"])
        self.assertEqual(result["not_checked"], ["len", "slot", "wid"])
        self.assertTrue(result["restored"])
        self.assertEqual(self.cells(doc)["B2"], "=20 mm")

    def test_a_budget_of_zero_checks_nothing_and_changes_nothing(self):
        doc = self.parametric(slot=8.0)
        before = self.cells(doc)
        result = E.editability_test(doc, self.table(doc), budget_s=0)
        self.assertEqual((result["checked"], result["parameters"]), (0, []))
        self.assertEqual(result["not_checked"], ["hei", "len", "slot", "wid"])
        self.assertEqual(self.cells(doc), before)

    def test_stop_is_honoured_before_every_edit(self):
        doc = self.parametric(slot=8.0)
        calls = []

        def stop():
            calls.append(1)
            return len(calls) > 3

        result = E.editability_test(doc, self.table(doc), should_stop=stop)
        self.assertLessEqual(result["checked"], 3)
        self.assertTrue(result["not_checked"])
        self.assertTrue(result["restored"])
        self.assertEqual(self.cells(doc)["B2"], "=20 mm")

    def test_a_stop_probe_that_raises_counts_as_no_stop(self):
        doc = self.parametric(slot=8.0)
        result = E.editability_test(doc, self.table(doc), should_stop=mock.Mock(side_effect=RuntimeError("x")))
        self.assertEqual(result["checked"], 8)

    def test_a_recompute_that_raises_is_a_finding_and_the_sheet_is_still_restored(self):
        doc = self.parametric(slot=8.0)
        before = self.cells(doc)
        spy = _DocSpy(doc, fail_on={1})
        result = E.editability_test(spy, self.table(doc))
        self.assertFalse(result["ok"])
        crashed = next(p for p in result["parameters"] if p["plus"] and p["plus"]["problems"])
        self.assertIn("RuntimeError: recompute exploded", crashed["plus"]["problems"][0])
        self.assertEqual(self.cells(doc), before)
        self.assertTrue(result["restored"])

    def test_the_limit_of_parameters_is_respected(self):
        doc = self.parametric(slot=8.0)
        result = E.editability_test(doc, self.table(doc), max_params=2)
        self.assertEqual(len(result["parameters"]), 2)
        self.assertEqual(result["checked"], 4)
        self.assertEqual(sum("beyond the 2 highest-bound" in s["reason"] for s in result["skipped"]), 2)

    def test_no_sheet_and_nothing_to_edit_give_a_reason_not_an_exception(self):
        doc = App.newDocument("NoSheetT")
        self._made.append(doc.Name)
        doc.addObject("Part::Box", "Cube")
        doc.recompute()
        no_sheet = E.editability_test(doc, self.table(doc))
        self.assertFalse(no_sheet["ok"])
        self.assertIn("no Params / Parameters spreadsheet", no_sheet["error"])
        S.params(doc, [("hook_count", "=4", "", "user_requirement"), ("free_dim", "=5 mm", "mm", "assumption")])
        nothing = E.editability_test(doc, self.table(doc))
        self.assertFalse(nothing["ok"])
        self.assertIn("no bound numeric parameter to edit", nothing["error"])
        self.assertEqual({s["parameter"] for s in nothing["skipped"]}, {"hook_count", "free_dim"})

    def test_a_cell_that_is_text_is_skipped_not_edited(self):
        doc = self.parametric(slot=8.0)
        sheet = doc.getObject("Params")
        sheet.set("B2", "'twenty")
        doc.recompute()
        result = E.editability_test(doc, self.table(doc))
        self.assertNotIn("len", [p["parameter"] for p in result["parameters"]])
        self.assertEqual(sheet.getContents("B2"), "'twenty")

    def test_the_document_has_no_leftover_objects_or_undo_entries(self):
        doc = self.parametric(slot=8.0)
        names = [o.Name for o in doc.Objects]
        undo = doc.UndoCount
        E.editability_test(doc, self.table(doc))
        self.assertEqual([o.Name for o in doc.Objects], names)
        self.assertEqual(doc.UndoCount, undo)


if __name__ == "__main__":
    unittest.main()
