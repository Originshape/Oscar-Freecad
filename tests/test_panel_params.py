"""Tests of the Parameters workspace tab (oscar/panel_params.py).

Pure tests of the decision logic (parsing, diffing, change collection, unit
rules, nudging, flags) run on any interpreter; the widget tests build a real
``ParamsTab`` on an offscreen QApplication - no FreeCAD document is touched:
data is injected in the ``read_param_table`` shape and Apply goes through a
fake ``apply_fn``, which is exactly the seam the panel host wires to
``tools_kit_params.apply_parameter_edits`` (covered against a real document
by test_panel_tabs.ApplyEditsTest).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

_IMPORT_ERROR = ""
try:
    from PySide import QtCore, QtWidgets
    from oscar.panel_params import (
        COLS_BASE, COLS_FULL, FLAG_APPLIED, FLAG_ERROR, FLAG_MODIFIED, ParamsTab,
        collect_changes, diff_rows, flag_state, format_number, has_evidence, item_flag,
        nudge_text, normalize_input, parse_numeric, row_change, row_is_editable,
        row_value_text, same_as_row,
    )
except Exception as exc:   # pragma: no cover - plain interpreter without Qt
    _IMPORT_ERROR = str(exc)


def row(**kw):
    """A read_param_table-shaped row with the kit defaults."""
    base = {"name": "p", "value": 1.0, "unit": "mm", "expr": "", "source": "",
            "note": "", "cell": "B1", "bound": 0, "needs_rebuild": False}
    base.update(kw)
    return base


def sample_rows():
    return [
        row(name="wall_t", value=3.0, unit="mm", source="step_measurement", bound=2),
        row(name="plate_w", value=60.0, unit="mm", source="assumption"),
        row(name="height", value=12.0, unit="mm", expr="=wall_t * 4", bound=1),
        row(name="vent_count", value=6, unit="", needs_rebuild=True),
    ]


class ParamsTabCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERROR:
            raise unittest.SkipTest("Qt unavailable: %s" % _IMPORT_ERROR)
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            cls.app = QtWidgets.QApplication(sys.argv[:1])


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


class ParseNumericTest(ParamsTabCase):
    def test_accepts_numbers_signs_decimals_exponents_and_units(self):
        self.assertEqual(parse_numeric("4"), (4.0, ""))
        self.assertEqual(parse_numeric(" 6 "), (6.0, ""))
        self.assertEqual(parse_numeric("4 mm"), (4.0, "mm"))
        self.assertEqual(parse_numeric("-2.5e1mm"), (-25.0, "mm"))
        self.assertEqual(parse_numeric("+.5 deg"), (0.5, "deg"))
        self.assertEqual(parse_numeric("2.40 mm"), (2.4, "mm"))

    def test_rejects_formulas_text_double_tokens_and_junk(self):
        for bad in ("=wall_t*2", "abc", "", "   ", "4 5", "4 mm x", "mm", "1e999", None):
            self.assertIsNone(parse_numeric(bad), bad)

    def test_format_number_is_compact(self):
        self.assertEqual(format_number(3.0), "3")
        self.assertEqual(format_number(0.1), "0.1")
        self.assertEqual(format_number(60.0), "60")


class RowLogicTest(ParamsTabCase):
    def test_value_text_formula_number_text_and_missing(self):
        self.assertEqual(row_value_text(row(name="h", value=12.0, unit="mm", expr="=wall_t * 4")), "=wall_t * 4")
        self.assertEqual(row_value_text(row(value=3.0, unit="mm")), "3 mm")
        self.assertEqual(row_value_text(row(value=6, unit="")), "6")
        self.assertEqual(row_value_text(row(value=None)), "?")
        self.assertEqual(row_value_text(row(value="wide")), "wide")

    def test_editable_matrix(self):
        self.assertTrue(row_is_editable(row(value=3.0)))
        self.assertFalse(row_is_editable(row(value=3.0, expr="=wall_t*2")), "derived is read-only")
        self.assertFalse(row_is_editable(row(value=6, needs_rebuild=True)), "rebuild-only is read-only")
        self.assertFalse(row_is_editable(row(value="wide")), "text is read-only")
        self.assertFalse(row_is_editable(row(value=None)), "missing value is read-only")

    def test_same_as_row_ignores_formatting_but_not_units_or_values(self):
        r = row(value=3.0, unit="mm")
        self.assertTrue(same_as_row(r, "3 mm"))
        self.assertTrue(same_as_row(r, "3.0 mm"))
        self.assertTrue(same_as_row(r, "3"), "a bare number takes the row's unit")
        self.assertFalse(same_as_row(r, "4"))
        self.assertFalse(same_as_row(r, "3 deg"), "another unit never matches")
        self.assertFalse(same_as_row(r, "abc"))


class RowChangeTest(ParamsTabCase):
    def test_change_carries_the_row_unit_and_an_apply_text(self):
        change = row_change(row(name="wall_t", value=3.0, unit="mm"), "2.4")
        self.assertEqual(change, {"name": "wall_t", "value": 2.4, "unit": "mm", "apply_text": "2.4 mm"})
        self.assertEqual(row_change(row(name="vents", value=6, unit=""), "7")["apply_text"], "7")

    def test_unchanged_value_is_not_a_change(self):
        self.assertIsNone(row_change(row(value=3.0, unit="mm"), "3.0"))
        self.assertIsNone(row_change(row(value=3.0, unit="mm"), "3 mm"))

    def test_unit_rules(self):
        r = row(value=3.0, unit="mm")
        self.assertEqual(row_change(r, "3 deg")["error"], "unit 'deg' does not match this row's 'mm'")
        count = row(value=6, unit="")
        self.assertIn("unitless", row_change(count, "7 mm")["error"])

    def test_refusals(self):
        r = row(name="wall_t", value=3.0)
        self.assertIn("read-only", row_change(row(name="h", value=1.0, expr="=x"), "4")["error"])
        self.assertIn("empty", row_change(r, "")["error"])
        self.assertIn("formulas", row_change(r, "=wall_t*2")["error"])
        self.assertIn("not a number", row_change(r, "wide")["error"])


class DiffRowsTest(ParamsTabCase):
    def test_added_removed_changed_unchanged(self):
        old = sample_rows()
        new = sample_rows()
        new[1]["value"] = 61.0                      # plate_w changed
        new[3] = row(name="lid_gap", value=0.4)     # vent_count replaced
        new.append(row(name="new_p", value=1.0))    # added
        diff = diff_rows(old, new)
        self.assertEqual(diff["added"], ["lid_gap", "new_p"])
        self.assertEqual(diff["removed"], ["vent_count"])
        self.assertEqual(diff["changed"], ["plate_w"])
        self.assertNotIn("wall_t", diff["changed"])

    def test_unit_source_and_formula_changes_count(self):
        base = sample_rows()
        for field, value in (("unit", "cm"), ("source", "user_requirement"), ("expr", "=plate_w")):
            tweak = sample_rows()
            tweak[0][field] = value
            self.assertEqual(diff_rows(base, tweak)["changed"], ["wall_t"], field)

    def test_real_difference_is_change_and_refresh_equal_is_not(self):
        base = sample_rows()
        tweak = sample_rows()
        tweak[0]["value"] = 3.0 + 1e-6
        self.assertEqual(diff_rows(base, tweak)["changed"], ["wall_t"])
        tweak[0]["value"] = 3.0 + 1e-12
        self.assertEqual(diff_rows(base, tweak)["changed"], [])


class NudgeTextTest(ParamsTabCase):
    def test_plus_minus_ten_percent_with_unit_and_counts(self):
        r = row(value=3.0, unit="mm")
        self.assertEqual(nudge_text(r, 10), "3.3 mm")
        self.assertEqual(nudge_text(r, -10), "2.7 mm")
        self.assertEqual(nudge_text(row(value=6, unit=""), 10), "6.6")

    def test_nudge_builds_on_an_unapplied_edit_when_parseable(self):
        self.assertEqual(nudge_text(row(value=3.0, unit="mm"), 10, "10 mm"), "11 mm")
        self.assertEqual(nudge_text(row(value=3.0, unit="mm"), 10, "=x"), "3.3 mm",
                         "unparseable current text falls back to the refresh value")
        self.assertEqual(nudge_text(row(value=3.0, unit="mm"), 10, "4 deg"), "3.3 mm",
                         "a foreign unit falls back to the refresh value")

    def test_no_numeric_base_is_refused(self):
        self.assertEqual(nudge_text(row(value=None), 10), "")


class EvidenceAndMetaTest(ParamsTabCase):
    def test_has_evidence(self):
        self.assertFalse(has_evidence(sample_rows()))
        self.assertFalse(has_evidence([row(measured="", basis=None)]))
        self.assertTrue(has_evidence([row(basis="measured")]))
        self.assertTrue(has_evidence([row(measured=3.1)]))
        self.assertTrue(has_evidence([row(status="proposed")]))

    def test_flag_state_priority(self):
        self.assertEqual(flag_state("a", pending=["a"], errors=(), applied=()), FLAG_MODIFIED)
        self.assertEqual(flag_state("a", pending=["a"], errors=({"name": "a", "error": "x"},), applied=()), FLAG_ERROR)
        self.assertEqual(flag_state("a", pending=(), errors=(), applied=["a"]), FLAG_APPLIED)
        self.assertEqual(flag_state("a"), "")
        self.assertEqual(flag_state("a", errors=({"name": "", "error": "general"},)), "",
                         "a general error flags no row")

    def test_normalize_input_dict_list_and_defensive(self):
        sheet, rows = normalize_input({"sheet": "Params", "rows": sample_rows(),
                                       "sheets": [{"name": "Params", "rows": 4, "bound": 2}]})
        self.assertEqual((sheet, [r["name"] for r in rows]), ("Params", ["wall_t", "plate_w", "height", "vent_count"]))
        sheet, rows = normalize_input(sample_rows())
        self.assertEqual((sheet, len(rows)), ("", 4))
        sheet, rows = normalize_input({"sheet": "Params", "rows": [row(), {"value": 1.0}, "junk"]})
        self.assertEqual(rows, [row()])
        self.assertEqual(normalize_input(None), ("", []))

    def test_collect_changes_takes_only_changed_editable_cells(self):
        rows = sample_rows()
        tab = ParamsTab()
        tab.refresh_from(rows)
        self.assertEqual(tab.table.item(1, 1).text(), "60 mm")
        tab.table.item(0, 1).setText("4 mm")     # a real change
        tab.table.item(1, 1).setText("60.0")     # same value, other formatting
        changes, errors = collect_changes(rows, tab.table)
        self.assertEqual(changes, [{"name": "wall_t", "value": 4.0, "unit": "mm", "apply_text": "4 mm"}])
        self.assertEqual(errors, [])
        tab.table.item(1, 1).setText("wide")     # local refusal of an edit
        changes, errors = collect_changes(rows, tab.table)
        self.assertEqual([e["name"] for e in errors], ["plate_w"])


# ---------------------------------------------------------------------------
# the widget
# ---------------------------------------------------------------------------


class ParamsTabTest(ParamsTabCase):
    def tab(self, rows=None, apply_fn=None) -> ParamsTab:
        tab = ParamsTab(apply_fn=apply_fn)
        tab.refresh_from({"sheet": "Params", "rows": sample_rows() if rows is None else rows})
        return tab

    def row_of(self, tab: ParamsTab, name: str) -> int:
        for r in range(tab.table.rowCount()):
            if tab.table.item(r, 0).text() == name:
                return r
        return -1

    def editable(self, tab: ParamsTab, name: str) -> bool:
        flag = QtCore.Qt.ItemFlag.ItemIsEditable
        return bool(tab.table.item(self.row_of(tab, name), 1).flags() & flag)

    def test_builds_the_base_table_with_headers_values_and_sources(self):
        tab = self.tab()
        self.assertEqual(tab.table.columnCount(), 4)
        self.assertEqual([tab.table.horizontalHeaderItem(c).text() for c in range(4)],
                         list(COLS_BASE))
        self.assertEqual(tab.table.rowCount(), 4)
        r = self.row_of(tab, "wall_t")
        self.assertEqual(tab.table.item(r, 1).text(), "3 mm")
        self.assertEqual(tab.table.item(r, 2).text(), "mm")
        self.assertEqual(tab.table.item(r, 3).text(), "Measured")
        self.assertTrue(tab.hint.isHidden())
        self.assertIn("Sheet 'Params'", tab.sheet_label.text())

    def test_evidence_columns_appear_only_when_present(self):
        rows = sample_rows()
        tab = self.tab(rows)
        self.assertEqual([tab.table.horizontalHeaderItem(c).text() for c in range(4)], list(COLS_BASE))
        rows[0].update({"measured": 3.2, "deviation": 0.2, "tolerance": 0.5,
                        "confidence": "high", "basis": "measured", "status": "proposed"})
        tab = self.tab(rows)
        self.assertEqual(tab.table.columnCount(), 7)
        self.assertEqual([tab.table.horizontalHeaderItem(c).text() for c in range(7)], list(COLS_FULL))
        r = self.row_of(tab, "wall_t")
        self.assertEqual(tab.table.item(r, 4).text(), "3.2")
        self.assertEqual(tab.table.item(r, 5).text(), "measured")
        self.assertEqual(tab.table.item(r, 6).text(), "proposed")
        self.assertEqual(tab.table.item(r, 4).toolTip(), "deviation 0.2 | tolerance 0.5 | confidence high")

    def test_edit_flags_only_plain_numeric_value_cells(self):
        tab = self.tab()
        self.assertTrue(self.editable(tab, "wall_t"))
        self.assertFalse(self.editable(tab, "height"), "derived is read-only")
        self.assertFalse(self.editable(tab, "vent_count"), "needs-rebuild is read-only")
        for col in (0, 2, 3):
            flag = QtCore.Qt.ItemFlag.ItemIsEditable
            self.assertFalse(tab.table.item(0, col).flags() & flag, col)

    def test_derived_row_shows_its_formula(self):
        tab = self.tab()
        value = tab.table.item(self.row_of(tab, "height"), 1)
        self.assertEqual(value.text(), "=wall_t * 4")
        self.assertIn("Derived", value.toolTip())

    def test_edit_flags_the_row_modified_and_counts_it(self):
        tab = self.tab()
        r = self.row_of(tab, "wall_t")
        tab.table.item(r, 1).setText("4 mm")
        self.assertEqual(item_flag(tab.table.item(r, 0)), FLAG_MODIFIED)
        self.assertEqual(item_flag(tab.table.item(r, 1)), FLAG_MODIFIED)
        self.assertTrue(tab.table.item(r, 1).font().bold())
        self.assertEqual(tab.apply_btn.text(), "Apply (1)")
        self.assertTrue(tab.apply_btn.isVisibleTo(tab))
        self.assertEqual(tab.pending(), {"wall_t": "4 mm"})
        tab.table.item(r, 1).setText("3.0 mm")    # back to the same value
        self.assertEqual(item_flag(tab.table.item(r, 0)), "")
        self.assertEqual(tab.pending(), {})
        self.assertFalse(tab.apply_btn.isVisibleTo(tab))

    def test_discard_restores_the_last_refresh(self):
        tab = self.tab()
        tab.table.item(self.row_of(tab, "wall_t"), 1).setText("4 mm")
        tab.table.item(self.row_of(tab, "plate_w"), 1).setText("61 mm")
        self.assertEqual(len(tab.pending()), 2)
        tab.discard()
        self.assertEqual(tab.pending(), {})
        self.assertEqual(tab.table.item(self.row_of(tab, "wall_t"), 1).text(), "3 mm")
        self.assertEqual(tab.table.item(self.row_of(tab, "plate_w"), 1).text(), "60 mm")
        self.assertEqual(item_flag(tab.table.item(self.row_of(tab, "wall_t"), 0)), "")
        self.assertFalse(tab.apply_btn.isVisibleTo(tab))

    def test_refresh_keeps_live_pending_edits_and_drops_dead_ones(self):
        tab = self.tab()
        tab.table.item(self.row_of(tab, "wall_t"), 1).setText("4 mm")
        kept = sample_rows()
        kept[1]["value"] = 61.0
        tab.refresh_from({"sheet": "Params", "rows": kept})
        self.assertEqual(tab.pending(), {"wall_t": "4 mm"}, "a live edit survives a reload")
        self.assertEqual(tab.table.item(self.row_of(tab, "wall_t"), 1).text(), "4 mm")
        caught_up = sample_rows()
        caught_up[0]["value"] = 4.0            # the sheet now holds the typed value
        tab.refresh_from({"sheet": "Params", "rows": caught_up})
        self.assertEqual(tab.pending(), {}, "an edit the refresh caught up with stops being modified")
        self.assertEqual(tab.table.item(self.row_of(tab, "wall_t"), 1).text(), "4 mm")
        tab.table.item(self.row_of(tab, "wall_t"), 1).setText("5 mm")
        without = [r for r in caught_up if r["name"] != "wall_t"]
        tab.refresh_from({"sheet": "Params", "rows": without})
        self.assertEqual(tab.pending(), {}, "the edit of a deleted row is dropped")

    def test_refresh_flags_rows_the_apply_changed(self):
        tab = self.tab()
        fresh = sample_rows()
        fresh[0]["value"] = 4.0
        tab.refresh_from({"sheet": "Params", "rows": fresh})
        r = self.row_of(tab, "wall_t")
        self.assertEqual(item_flag(tab.table.item(r, 0)), FLAG_APPLIED)
        self.assertEqual(item_flag(tab.table.item(self.row_of(tab, "plate_w"), 0)), "")

    def test_apply_success_writes_through_apply_fn_and_resets_the_rows(self):
        seen = []

        def apply_fn(changes):
            seen.append(changes)
            return True, []

        tab = self.tab(apply_fn=apply_fn)
        tab.table.item(self.row_of(tab, "wall_t"), 1).setText("4 mm")
        tab.apply()
        self.assertEqual(seen, [[{"name": "wall_t", "value": 4.0, "unit": "mm", "apply_text": "4 mm"}]])
        self.assertEqual(tab.pending(), {})
        r = self.row_of(tab, "wall_t")
        self.assertEqual(tab.table.item(r, 1).text(), "4 mm")
        self.assertEqual(item_flag(tab.table.item(r, 0)), FLAG_APPLIED)
        self.assertFalse(tab.apply_btn.isVisibleTo(tab))
        tab.apply()                                   # nothing modified anymore
        self.assertEqual(len(seen), 1, "an empty apply never calls the write path")

    def test_apply_failure_flags_rows_and_keeps_the_edits(self):
        tab = self.tab(apply_fn=lambda changes: (False, [{"name": "wall_t", "error": "rolled back"}]))
        tab.table.item(self.row_of(tab, "wall_t"), 1).setText("4 mm")
        tab.apply()
        self.assertEqual(tab.pending(), {"wall_t": "4 mm"}, "the user can fix and retry")
        self.assertTrue(tab.error_label.isVisibleTo(tab))
        self.assertIn("wall_t", tab.error_label.text())
        self.assertIn("rolled back", tab.error_label.text())
        r = self.row_of(tab, "wall_t")
        self.assertEqual(item_flag(tab.table.item(r, 0)), FLAG_ERROR)
        self.assertIn("rolled back", tab.table.item(r, 1).toolTip())
        self.assertTrue(tab.apply_btn.isVisibleTo(tab))

    def test_local_refusal_never_reaches_apply_fn(self):
        seen = []
        tab = self.tab(apply_fn=lambda changes: (seen.append(changes) or (True, [])))
        tab.table.item(self.row_of(tab, "wall_t"), 1).setText("wide")
        tab.apply()
        self.assertEqual(seen, [], "a bad value is caught before the write path")
        self.assertIn("not a number", tab.error_label.text())
        self.assertEqual(item_flag(tab.table.item(self.row_of(tab, "wall_t"), 0)), FLAG_ERROR)

    def test_unwired_apply_fn_is_refused_loudly_not_silently(self):
        tab = self.tab()                        # no apply_fn: the safe default
        tab.table.item(self.row_of(tab, "wall_t"), 1).setText("4 mm")
        tab.apply()
        self.assertTrue(tab.error_label.isVisibleTo(tab))
        self.assertIn("NotImplementedError", tab.error_label.text())
        self.assertIn("apply_parameter_edits", tab.error_label.text())
        self.assertEqual(tab.pending(), {"wall_t": "4 mm"})

    def test_apply_fn_contract_violation_and_exception_are_shown(self):
        tab = self.tab(apply_fn=lambda changes: {"ok": True})   # wrong shape
        tab.table.item(self.row_of(tab, "wall_t"), 1).setText("4 mm")
        tab.apply()
        self.assertIn("(ok, errors)", tab.error_label.text())
        tab2 = self.tab(apply_fn=lambda changes: (_ for _ in ()).throw(RuntimeError("doc closed")))
        tab2.table.item(self.row_of(tab2, "wall_t"), 1).setText("4 mm")
        tab2.apply()
        self.assertIn("RuntimeError", tab2.error_label.text())
        self.assertIn("doc closed", tab2.error_label.text())
        self.assertEqual(tab2.pending(), {"wall_t": "4 mm"})

    def test_clear_errors_dismisses_the_area_and_restores_flags(self):
        tab = self.tab(apply_fn=lambda changes: (False, ["the document refused"]))
        tab.table.item(self.row_of(tab, "wall_t"), 1).setText("4 mm")
        tab.apply()
        self.assertTrue(tab.error_label.isVisibleTo(tab))
        tab.clear_errors_btn.click()
        self.assertFalse(tab.error_label.isVisibleTo(tab))
        self.assertEqual(item_flag(tab.table.item(self.row_of(tab, "wall_t"), 0)), FLAG_MODIFIED,
                         "the edit itself is still pending")
        self.assertEqual(tab.pending(), {"wall_t": "4 mm"})

    def test_nudge_is_just_another_edit_through_the_pipeline(self):
        seen = []
        tab = self.tab(apply_fn=lambda changes: (seen.append(changes) or (True, [])))
        r = self.row_of(tab, "wall_t")
        tab._nudge_row(r, 10)
        self.assertEqual(tab.table.item(r, 1).text(), "3.3 mm")
        self.assertEqual(tab.pending(), {"wall_t": "3.3 mm"})
        self.assertEqual(item_flag(tab.table.item(r, 0)), FLAG_MODIFIED)
        tab._nudge_row(r, 10)                     # builds on the unapplied edit
        self.assertEqual(tab.table.item(r, 1).text(), "3.63 mm")
        tab.apply()
        self.assertEqual(seen[-1], [{"name": "wall_t", "value": 3.63, "unit": "mm", "apply_text": "3.63 mm"}])
        tab._nudge_row(self.row_of(tab, "height"), 10)   # read-only: ignored
        self.assertEqual(tab.table.item(self.row_of(tab, "height"), 1).text(), "=wall_t * 4")

    def test_search_filters_rows(self):
        tab = self.tab()
        tab.search.setText("wall")
        self.assertEqual(tab.table.rowCount(), 1)
        self.assertEqual(tab.table.item(0, 0).text(), "wall_t")
        tab.search.setText("plate")
        self.assertEqual(tab.table.rowCount(), 1)
        self.assertEqual(tab.table.item(0, 0).text(), "plate_w")

    def test_empty_sheet_shows_the_hint(self):
        tab = ParamsTab()
        tab.refresh_from({"sheet": "", "rows": []})
        self.assertTrue(tab.hint.isVisibleTo(tab))
        self.assertIn("No parameter sheet", tab.hint.text())
        self.assertEqual(tab.table.rowCount(), 0)
        tab.refresh_from({"sheet": "Params", "rows": []})
        self.assertIn("no aliased parameter rows", tab.hint.text())


if __name__ == "__main__":
    unittest.main()
