"""Tests for oscar.kit.params: the Params spreadsheet.

The pure helpers (alias rules, unit inference, the DEFAULT_PARAMS table, STAGE_OF_PARAM) run anywhere; the sheet tests need
FreeCAD (freecadcmd, as tests/run.py does) and are skipped otherwise. They pin the verified runtime facts the kit
relies on: Quantity vs int on ``sheet.get``, `=`-prefixed numeric cells, FreeCAD's own rejection of unit-like aliases,
and ``ensure_sheet`` adopting the sheet instead of creating ``Params001``.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.kit import params, spec  # noqa: E402

try:
    import FreeCAD as App  # noqa: E402
except ImportError:
    App = None

def cell(sheet, addr: str) -> str:
    """Cell contents as the user sees them: FreeCAD returns a text cell as ``'text`` (leading apostrophe)."""
    c = sheet.getContents(addr) or ""
    return c[1:] if c.startswith("'") else c


def squash(text: str) -> str:
    """Formula text without blanks: FreeCAD re-prints expressions with spaces around operators (``wall_t * 2``)."""
    return text.replace(" ", "")


_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "enclosure")
with open(os.path.join(_FIX, "rpi4b_board.json"), encoding="utf-8") as _f:
    BOARD = json.load(_f)


# --------------------------------------------------------------------------------------------------------------------
class AliasRulesTest(unittest.TestCase):
    """valid_alias without a sheet: the local rules (pure, any thread)."""

    def test_good_names(self):
        for name in ("wall_t", "floor_t", "clearance_eff", "hook_ramp_deg", "abc", "a_b", "x" + "y" * 31, "board_len", "pr_nx", "lid_lift"):
            self.assertEqual(params.Params.valid_alias(name), (True, ""), name)
            self.assertEqual(params.valid_alias(name), (True, ""), name)

    def test_shape_rules(self):
        for bad in ("ab", "a", "", "Wall_t", "WALL", "1abc", "_abc", "wall t", "wall-t", "wall.t", "x" * 33, "ümlaut", "wall_t "):
            ok, reason = params.valid_alias(bad)
            self.assertFalse(ok, bad)
            self.assertIn("not a valid parameter name", reason)
            self.assertIn("lower_snake_case", reason)
        for bad in (None, 3, ["a"]):
            self.assertFalse(params.valid_alias(bad)[0])                 # type: ignore[arg-type]

    def test_cell_address_like_names_are_refused(self):
        for bad in ("a1", "ab12", "zz999", "b2"):
            if len(bad) < 3:
                continue                                                  # too short anyway
            ok, reason = params.valid_alias(bad)
            self.assertFalse(ok, bad)
        self.assertIn("cell address", params.valid_alias("ab12")[1])
        self.assertTrue(params.valid_alias("abc1")[0])                    # 3 letters then a digit is not an address

    def test_reserved_and_unit_names_are_refused(self):
        for bad in ("min", "max", "abs", "sin", "cos", "deg", "rad", "psi", "bar", "ohm", "day", "sec", "true", "false", "not",
                    "and", "mil", "mol", "mph", "lbf", "kcal", "thou", "ksi"):       # (2-letter names such as 'pi' fail the shape rule first)
            ok, reason = params.valid_alias(bad)
            self.assertFalse(ok, bad)
            self.assertIn("reserved", reason)
            self.assertIn(f"{bad}_val", reason)                           # the suggestion

    def test_suggest_alias(self):
        self.assertEqual(params.suggest_alias("Wall T"), "wall_t")
        self.assertEqual(params.suggest_alias("  Hook  Length (mm) "), "hook_length_mm")
        self.assertEqual(params.suggest_alias("2nd wall"), "p_2nd_wall")
        self.assertEqual(params.suggest_alias("ab"), "")
        self.assertEqual(params.suggest_alias("A" * 40), "a" * 32)
        self.assertEqual(params.suggest_alias("min"), "")                  # no sensible alias is invented for a reserved word
        self.assertIn("try 'wall_t'", params.valid_alias("Wall T")[1])

    def test_constants_match_the_contract(self):
        self.assertEqual(params.SHEET_NAME, "Params")
        self.assertEqual(params.HEADER, ("name", "value", "unit", "source", "note"))
        self.assertEqual(params.SOURCES, ("user_requirement", "datasheet", "step_measurement", "standard", "derived", "assumption"))
        self.assertEqual(params.UNITS, ("mm", "deg", ""))
        self.assertTrue({"min", "max", "abs", "sin", "cos", "tan", "exp", "log", "sqrt", "pi", "deg", "rad", "mod", "pow", "round", "ceil",
                         "floor", "true", "false", "not", "and", "or", "if", "mm", "cm", "psi", "bar", "ohm", "day", "sec"} <= params.RESERVED)


class UnitInferenceTest(unittest.TestCase):
    def test_millimetre_suffixes_and_names(self):
        for name in ("wall_t", "slot_h", "hook_w", "slot_l", "pilot_d", "corner_r", "hook_len", "board_wid", "hole_off", "port_gap",
                     "lid_fit", "hole_dx", "hole_dy", "tallest_z", "clearance", "pitch", "overhang", "lid_lift", "corner_r", "vent_pitch",
                     "hook_overhang", "window_margin", "pcb_thickness"):
            self.assertEqual(params.infer_unit(name), "mm", name)

    def test_degree_names(self):
        for name in ("hook_ramp_deg", "hook_retention_deg", "angle", "vent_angle"):
            self.assertEqual(params.infer_unit(name), "deg", name)

    def test_unitless_names(self):
        for name in ("hook_count", "vent_count", "shrink_pct", "ratio", "foo", "board", "scale_factor", "aspect_ratio", "x_pct"):
            self.assertEqual(params.infer_unit(name), "", name)

    def test_count_and_pct_win_over_mm_suffixes(self):
        self.assertEqual(params.infer_unit("slot_w_count"), "")
        self.assertEqual(params.infer_unit("wall_pct"), "")


class DefaultTableTest(unittest.TestCase):
    """DEFAULT_PARAMS is the table"""

    def row(self, name):
        return next(r for r in params.DEFAULT_PARAMS if r["name"] == name)

    def test_names_in_the_contract_order(self):
        names = [r["name"] for r in params.DEFAULT_PARAMS]
        want_board = ["board_len", "board_w", "pcb_t", "hole_off", "hole_dx", "hole_dy", "hole_d", "tallest_z", "bottom_z", "pr_nx", "pr_px", "pr_ny", "pr_py"]
        want_design = ["wall_t", "floor_t", "lid_t", "clearance", "shrink_pct", "corner_r", "standoff_h", "standoff_d", "standoff_pilot_d", "lid_gap",
                       "lip_h", "lip_t", "lid_fit", "hook_count", "hook_w", "hook_t", "hook_overhang", "hook_len", "hook_barb_h", "hook_ramp_deg",
                       "hook_retention_deg", "window_margin", "vent_count", "vent_slot_w", "vent_slot_l", "vent_pitch", "port_margin", "port_gap", "lid_lift"]
        want_derived = ["clearance_eff", "margin_nx", "margin_px", "margin_ny", "margin_py", "cav_x0", "cav_x1", "cav_y0", "cav_y1", "cav_len", "cav_wid",
                        "shell_len", "shell_wid", "z_floor_top", "z_floor_bot", "z_cavity_top", "shell_h", "lid_len", "lid_wid"]
        self.assertEqual(names, want_board + want_design + want_derived)
        self.assertEqual(len(names), len(set(names)))

    def test_every_name_is_a_valid_alias(self):
        for r in params.DEFAULT_PARAMS:
            self.assertEqual(params.valid_alias(r["name"]), (True, ""), r["name"])

    def test_sources_units_and_formulas(self):
        for r in params.DEFAULT_PARAMS:
            self.assertEqual(set(r), {"name", "value", "unit", "source", "note", "formula"})
            self.assertIn(r["source"], params.SOURCES)
            self.assertIn(r["unit"], params.UNITS)
            self.assertEqual(r["source"] == "derived", r["formula"] is not None, r["name"])
            if r["formula"]:
                self.assertTrue(r["formula"].startswith("="))
        for name in ("board_len", "board_w", "hole_off", "hole_dx", "hole_dy", "hole_d"):
            self.assertEqual(self.row(name)["source"], "datasheet")
        for name in ("pcb_t", "tallest_z", "bottom_z", "pr_nx", "wall_t", "hook_len", "lid_lift"):
            self.assertEqual(self.row(name)["source"], "assumption")
        self.assertEqual(self.row("shrink_pct")["unit"], "")
        self.assertEqual(self.row("hook_count")["unit"], "")
        self.assertEqual(self.row("hook_ramp_deg")["unit"], "deg")
        self.assertEqual(self.row("wall_t")["unit"], "mm")
        self.assertEqual(self.row("lid_lift")["unit"], "mm")

    def test_values_match_the_pi4_abs_defaults(self):
        for name, want in {"board_len": 85.0, "board_w": 56.0, "pcb_t": 1.4, "hole_off": 3.5, "hole_dx": 58.0, "hole_dy": 49.0, "hole_d": 2.75,
                           "tallest_z": 17.0, "bottom_z": -1.5, "pr_nx": 2.5, "pr_px": 2.0, "pr_ny": 1.0, "pr_py": 0.0, "wall_t": 2.0,
                           "clearance": 0.3, "shrink_pct": 0.8, "hook_count": 4, "hook_len": 12.0, "vent_count": 9, "window_margin": 0.3,
                           "clearance_eff": 0.64, "cav_len": 90.1, "cav_wid": 57.94, "shell_len": 94.1, "shell_wid": 61.94, "shell_h": 24.0,
                           "lid_len": 97.0, "lid_wid": 64.84, "z_floor_bot": -6.0, "z_cavity_top": 18.0}.items():
            self.assertAlmostEqual(self.row(name)["value"], want, places=6, msg=name)
        self.assertEqual(self.row("clearance_eff")["formula"], "=clearance + shrink_pct/100*board_len/2")
        self.assertEqual(self.row("z_floor_bot")["formula"], "=-(standoff_h + floor_t)")

    def test_the_table_equals_the_one_built_from_the_spec_and_board(self):
        rows = spec.param_rows(spec.default_spec(), BOARD)
        self.assertEqual([r["name"] for r in rows], [r["name"] for r in params.DEFAULT_PARAMS])
        for a, b in zip(rows, params.DEFAULT_PARAMS):
            self.assertEqual((a["unit"], a["source"], a["formula"]), (b["unit"], b["source"], b["formula"]), a["name"])
            self.assertAlmostEqual(a["value"], b["value"], places=6, msg=a["name"])

    def test_derived_rows_depend_only_on_earlier_rows(self):
        seen = set()
        for r in params.DEFAULT_PARAMS:
            if r["formula"]:
                import re
                for token in re.findall(r"[a-z][a-z0-9_]*", r["formula"]):
                    if token not in ("max", "min"):
                        self.assertIn(token, seen, f"{r['name']} refers to {token} before it is defined")
            seen.add(r["name"])

    def test_stage_of_param(self):
        names = {r["name"] for r in params.DEFAULT_PARAMS}
        for name, stage in params.STAGE_OF_PARAM.items():
            self.assertIn(name, names)
            self.assertIn(stage, spec.STAGES)
        for name in ("vent_count", "vent_slot_w", "vent_pitch", "hook_count", "hook_w", "hook_len"):
            self.assertIn(name, params.STAGE_OF_PARAM)
        self.assertEqual(params.STAGE_OF_PARAM["vent_count"], "vents")
        self.assertEqual(params.STAGE_OF_PARAM["hook_count"], "hooks")
        for name in ("wall_t", "clearance", "corner_r", "lid_lift"):            # continuous dimensions are expression-bound
            self.assertNotIn(name, params.STAGE_OF_PARAM)

    def test_every_design_param_maps_to_a_spec_field(self):
        for name, path in spec.FIELD_OF_PARAM.items():
            self.assertIn(name, [r["name"] for r in params.DEFAULT_PARAMS])


class PureHelpersTest(unittest.TestCase):
    def test_prefix_aliases(self):
        names = {"wall_t", "floor_t", "clearance"}
        f = params._prefix_aliases
        self.assertEqual(f("wall_t*2", names, "Params"), "Params.wall_t*2")
        self.assertEqual(f("wall_t + floor_t", names, "Params"), "Params.wall_t + Params.floor_t")
        self.assertEqual(f("Params.wall_t*2", names, "Params"), "Params.wall_t*2")           # already qualified
        self.assertEqual(f("max(wall_t; floor_t)", names, "Params"), "max(Params.wall_t; Params.floor_t)")
        self.assertEqual(f("wall_t * 2 mm", names, "Params"), "Params.wall_t * 2 mm")        # units stay
        self.assertEqual(f("Box.Length + wall_t", names, "Params"), "Box.Length + Params.wall_t")
        self.assertEqual(f("unknown_name*2", names, "Params"), "unknown_name*2")
        self.assertEqual(f("wall_t*2", names, "Sheet"), "Sheet.wall_t*2")

    def test_cell_value_always_writes_a_formula_for_numbers(self):
        cv = params._cell_value
        self.assertEqual(cv("wall_t", 2.4, None, None), ("=2.4 mm", "mm"))
        self.assertEqual(cv("bottom_z", -1.5, None, None), ("=-1.5 mm", "mm"))        # a leading '-' would be stored as a string without the '='
        self.assertEqual(cv("hook_count", 4, None, None), ("=4", ""))
        self.assertEqual(cv("hook_ramp_deg", 30, None, None), ("=30 deg", "deg"))
        self.assertEqual(cv("wall_t", "2.4 mm", None, None), ("=2.4 mm", "mm"))
        self.assertEqual(cv("hook_ramp_deg", "45 deg", None, None), ("=45 deg", "deg"))
        self.assertEqual(cv("wall_t", "2.4", None, None), ("=2.4 mm", "mm"))
        self.assertEqual(cv("some_value", "7", None, None), ("=7", ""))
        self.assertEqual(cv("some_value", "7", "mm", None), ("=7 mm", "mm"))
        self.assertEqual(cv("wall_t", "+1.5mm", None, None), ("=1.5 mm", "mm"))
        self.assertEqual(cv("wall_t", "=floor_t*2", None, None), ("=floor_t*2", "mm"))      # formulas are kept as given
        self.assertEqual(cv("shell_len", None, None, "cav_len + 2*wall_t"), ("=cav_len + 2*wall_t", "mm"))
        self.assertEqual(cv("wall_t", 2, None, None), ("=2 mm", "mm"))
        self.assertEqual(cv("wall_t", 1e-7, None, None)[0], "=1e-07 mm")

    def test_cell_value_errors_are_model_readable(self):
        cv = params._cell_value
        cases = [(("wall_t", "2,5 mm", None, None), "decimal separator"), (("wall_t", "abc", None, None), "cannot read 'abc' as a number"),
                 (("wall_t", True, None, None), "is not a number"), (("wall_t", None, None, None), "is not a number"),
                 (("wall_t", float("nan"), None, None), "not a finite number"), (("wall_t", float("inf"), None, None), "not a finite number"),
                 (("wall_t", 2, "cm", None), "unit 'cm' is not supported"), (("wall_t", "2 mm", "deg", None), "is in 'mm' but unit='deg'"),
                 (("wall_t", None, None, "  "), "formula is empty"), (("wall_t", None, None, "="), "formula is empty"), (("wall_t", [1], None, None), "is not a number")]
        for args, needle in cases:
            with self.assertRaises(params.ParamsError, msg=str(args)) as cm:
                cv(*args)
            self.assertIn(needle, str(cm.exception))
            self.assertTrue(str(cm.exception).startswith("Params: '"))

    def test_params_error_is_a_value_error(self):
        self.assertTrue(issubclass(params.ParamsError, ValueError))

    def test_text_cells_are_protected_from_formula_parsing(self):
        self.assertEqual(params._safe_text("=oops"), "'=oops")
        self.assertEqual(params._safe_text("'quoted"), "''quoted")
        self.assertEqual(params._safe_text("plain"), "plain")
        self.assertEqual(params._strip_quote("'=oops"), "=oops")
        self.assertEqual(params._strip_quote("plain"), "plain")


# --------------------------------------------------------------------------------------------------------------------
@unittest.skipIf(App is None, "FreeCAD is needed for the sheet tests (run through freecadcmd)")
class SheetTestBase(unittest.TestCase):
    def setUp(self):
        self.doc = App.newDocument("ParamsTest")
        self.addCleanup(App.closeDocument, self.doc.Name)

    def sheet(self, **kw) -> "params.Params":
        return params.ensure_sheet(self.doc, **kw)

    def raw(self, alias):
        return self.doc.getObject("Params").get(alias)


@unittest.skipIf(App is None, "FreeCAD is needed for the sheet tests (run through freecadcmd)")
class EnsureSheetTest(SheetTestBase):
    """ONE producer: get-or-create the sheet whose Name is Params; never a Params001."""

    def test_creates_the_sheet_with_header_and_tags(self):
        p = self.sheet()
        s = self.doc.getObject("Params")
        self.assertIsNotNone(s)
        self.assertEqual((s.Name, s.Label, s.TypeId), ("Params", "Params", "Spreadsheet::Sheet"))
        self.assertIs(p.sheet, s)
        self.assertIs(p.doc, self.doc)
        self.assertEqual([cell(s, f"{c}1") for c in "ABCDE"], ["name", "value", "unit", "source", "note"])
        self.assertEqual((s.OscarKit, s.OscarStage, s.OscarRole, s.OscarPart), ("enc1", "params", "params", ""))
        self.assertEqual(p.names(), [])

    def test_a_second_call_adopts_the_same_object(self):
        p1 = self.sheet()
        p1.set("wall_t", 2.4, source="user_requirement")
        p2 = self.sheet()
        self.assertIs(p1.sheet, p2.sheet)
        self.assertEqual(p2.names(), ["wall_t"])
        self.assertEqual([o.Name for o in self.doc.Objects if o.TypeId == "Spreadsheet::Sheet"], ["Params"])
        self.assertIsNone(self.doc.getObject("Params001"))

    def test_adopts_a_sheet_somebody_else_created(self):
        mine = self.doc.addObject("Spreadsheet::Sheet", "Params")
        mine.set("A1", "name")
        mine.set("B2", "=4 mm")
        mine.setAlias("B2", "wall_t")
        p = self.sheet()
        self.assertIs(p.sheet, mine)
        self.assertEqual(p.get("wall_t"), 4.0)
        self.assertEqual(cell(mine, "A1"), "name")                          # not re-headered over existing content
        self.assertEqual(cell(mine, "B1"), "")
        self.assertIsNone(self.doc.getObject("Params001"))
        self.assertEqual(mine.OscarStage, "params")

    def test_keeps_existing_tags(self):
        s = self.doc.addObject("Spreadsheet::Sheet", "Params")
        s.addProperty("App::PropertyString", "OscarKit", "Oscar", "")
        s.OscarKit = "other_kit"
        self.sheet()
        self.assertEqual(s.OscarKit, "other_kit")

    def test_custom_kit_id(self):
        p = self.sheet(kit_id="enc7")
        self.assertEqual(p.sheet.OscarKit, "enc7")

    def test_refuses_a_non_sheet_called_params(self):
        self.doc.addObject("Part::Box", "Params")
        with self.assertRaises(params.ParamsError) as cm:
            self.sheet()
        self.assertIn("is a Part::Box, not a Spreadsheet::Sheet", str(cm.exception))

    def test_params_without_create_needs_an_existing_sheet(self):
        with self.assertRaises(params.ParamsError) as cm:
            params.Params(self.doc)
        self.assertIn("has no sheet named 'Params'", str(cm.exception))
        self.sheet()
        self.assertEqual(params.Params(self.doc).names(), [])

    def test_document_name_is_accepted(self):
        p = params.Params(self.doc.Name, create=True)
        self.assertIs(p.doc, self.doc)
        with self.assertRaises(params.ParamsError):
            params.Params("no_such_document")
        with self.assertRaises(params.ParamsError):
            params.Params(42)                                               # type: ignore[arg-type]

    def test_freecad_really_renames_a_second_sheet_params001(self):
        """The verified fact the rule exists for (and the reason ensure_sheet never creates a second sheet)."""
        self.doc.addObject("Spreadsheet::Sheet", "Params")
        second = self.doc.addObject("Spreadsheet::Sheet", "Params")
        self.assertEqual(second.Name, "Params001")

    def test_a_name_clash_on_creation_is_refused_and_cleaned_up(self):
        class FakeDoc:
            Name = "Fake"
            removed: list = []

            def getObject(self, name):
                return None

            def addObject(self, type_id, name):
                class Obj:
                    Name = "Params001"
                return Obj()

            def removeObject(self, name):
                self.removed.append(name)

        doc = FakeDoc()
        with self.assertRaises(params.ParamsError) as cm:
            params.ensure_sheet(doc)
        self.assertIn("named the new sheet 'Params001' instead of 'Params'", str(cm.exception))
        self.assertEqual(doc.removed, ["Params001"])

    def test_alternative_sheet_name(self):
        p = params.Params(self.doc, "Parameters", create=True)
        p.set("wall_t", 2.0)
        self.assertEqual(self.doc.getObject("Parameters").get("wall_t").Value, 2.0)
        self.assertIsNone(self.doc.getObject("Params"))


@unittest.skipIf(App is None, "FreeCAD is needed for the sheet tests (run through freecadcmd)")
class SetGetTest(SheetTestBase):
    def test_quantity_for_unit_cells_and_int_for_unitless_cells(self):
        p = self.sheet()
        p.set("wall_t", 2.4, source="user_requirement", note="stated by the user")
        p.set("hook_count", 4)
        raw_len, raw_count = self.raw("wall_t"), self.raw("hook_count")
        self.assertTrue(hasattr(raw_len, "Value"), "a unit cell reads back as a Quantity")
        self.assertFalse(hasattr(raw_count, "Value"), "a unitless cell reads back as a plain number")
        self.assertIsInstance(raw_count, (int, float))
        with self.assertRaises(AttributeError):
            raw_count.Value                                               # noqa: B018 - the bug of the real run
        # Params.get unifies them
        self.assertEqual(p.get("wall_t"), 2.4)
        self.assertIsInstance(p.get("wall_t"), float)
        self.assertEqual(p.get("hook_count"), 4.0)
        self.assertIsInstance(p.get("hook_count"), float)

    def test_layout_of_a_row(self):
        p = self.sheet()
        p.set("wall_t", 2.4, source="user_requirement", note="stated by the user")
        s = p.sheet
        self.assertEqual(cell(s, "A2"), "wall_t")
        self.assertEqual(s.getContents("B2"), "=2.4 mm")                   # numbers are written as formulas
        self.assertEqual(cell(s, "C2"), "mm")
        self.assertEqual(cell(s, "D2"), "user_requirement")
        self.assertEqual(cell(s, "E2"), "stated by the user")
        self.assertEqual(s.getAlias("B2"), "wall_t")
        self.assertEqual(s.getCellFromAlias("wall_t"), "B2")
        self.assertEqual(s.getAlias("A2"), None)                           # the alias lives on column B only

    def test_negative_numbers_are_numbers_not_strings(self):
        p = self.sheet()
        p.set("bottom_z", -1.5, source="datasheet")
        self.assertNotIsInstance(self.raw("bottom_z"), str)
        self.assertEqual(p.get("bottom_z"), -1.5)
        # the trap the '=' prefix avoids: a cell typed as '-1.5 mm' IS stored as text
        self.doc.getObject("Params").set("B9", "-1.5 mm")
        self.doc.getObject("Params").setAlias("B9", "raw_text_cell")
        self.doc.recompute()
        self.assertIsInstance(self.raw("raw_text_cell"), str)
        with self.assertRaises(params.ParamsError) as cm:
            p.get("raw_text_cell")
        self.assertEqual(str(cm.exception), "Params: 'raw_text_cell' was stored as text, not a number; write it as '=<value> <unit>'")

    def test_value_forms(self):
        p = self.sheet()
        p.set("slot_w", "2.4 mm")
        p.set("hook_ramp_deg", "45 deg")
        p.set("some_value", "7")
        p.set("wall_t", "=2.4 mm")                                          # a formula cell, as given
        p.set("floor_t", "=wall_t*2")
        p.set("lid_t", 3)
        p.set("pr_nx", 2.5, unit="mm")
        p.set("hook_count", "6", unit="")
        self.assertEqual(p.get("slot_w"), 2.4)
        self.assertAlmostEqual(self.raw("hook_ramp_deg").getValueAs("deg").Value, 45.0)
        self.assertEqual(p.get("some_value"), 7.0)
        self.assertEqual(p.get("wall_t"), 2.4)
        self.assertAlmostEqual(p.get("floor_t"), 4.8)
        self.assertEqual(p.get("lid_t"), 3.0)
        self.assertEqual(p.get("pr_nx"), 2.5)
        self.assertEqual(p.get("hook_count"), 6.0)

    def test_an_update_keeps_the_unit_the_row_already_has(self):
        """set_parameter('bracket_thk', 3) must not turn '2 mm' into a bare 3 just because the NAME says nothing about units."""
        p = self.sheet()
        p.set("bracket_thk", "2 mm")
        p.set("tilt_value", "30 deg")
        p.set("hole_total", "6")
        p.set("bracket_thk", 3)
        p.set("tilt_value", 45)
        p.set("hole_total", 8)
        self.assertEqual(self.doc.getObject("Params").get("bracket_thk").Unit.Type, "Length")
        self.assertEqual(self.doc.getObject("Params").get("tilt_value").Unit.Type, "Angle")
        self.assertFalse(hasattr(self.raw("hole_total"), "Value"))
        self.assertEqual((p.get("bracket_thk"), p.get("hole_total")), (3.0, 8.0))
        rows = {r["name"]: r for r in p.table()}
        self.assertEqual((rows["bracket_thk"]["unit"], rows["tilt_value"]["unit"], rows["hole_total"]["unit"]), ("mm", "deg", ""))
        p.set("bracket_thk", 4, unit="mm")                                 # an explicit unit always wins
        p.set("wall_t", 2.0)
        p.set("wall_t", "2.5")                                             # known name, same unit again
        self.assertEqual({r["name"]: r for r in p.table()}["wall_t"]["unit"], "mm")

    def test_a_row_written_by_other_code_without_a_unit_column_keeps_its_quantity_type(self):
        s = self.doc.addObject("Spreadsheet::Sheet", "Params")
        s.set("B2", "=40 mm")
        s.setAlias("B2", "plate_size")
        s.set("B3", "=3")
        s.setAlias("B3", "plate_total")
        self.doc.recompute()
        p = self.sheet()
        p.set("plate_size", 50)
        p.set("plate_total", 4)
        self.assertEqual(s.get("plate_size").Unit.Type, "Length")
        self.assertFalse(hasattr(s.get("plate_total"), "Value"))
        self.assertEqual({r["name"]: r for r in p.table()}["plate_size"]["unit"], "mm")

    def test_value_text_unit_beats_the_existing_unit(self):
        p = self.sheet()
        p.set("some_value", "2 mm")
        p.set("some_value", "30 deg")
        self.assertEqual(self.doc.getObject("Params").get("some_value").Unit.Type, "Angle")

    def test_derived_rows_follow_their_inputs(self):
        p = self.sheet()
        p.set("wall_t", 2.0)
        p.set("shell_len", None, formula="=wall_t*2 + 10 mm", source="derived")
        self.assertAlmostEqual(p.get("shell_len"), 14.0)
        p.set("wall_t", 3.0)
        self.assertAlmostEqual(p.get("shell_len"), 16.0)
        row = {r["name"]: r for r in p.table()}["shell_len"]
        self.assertEqual(squash(row["expr"]), "=wall_t*2+10mm")
        self.assertEqual(row["source"], "derived")

    def test_update_keeps_the_row_and_the_alias(self):
        p = self.sheet()
        p.set("wall_t", 2.0, note="first")
        p.set("hook_count", 4)
        p.set("wall_t", 2.4, source="user_requirement", note="second")
        self.assertEqual(p.names(), ["wall_t", "hook_count"])
        self.assertEqual(p.get("wall_t"), 2.4)
        s = p.sheet
        self.assertEqual((cell(s, "D2"), cell(s, "E2")), ("user_requirement", "second"))
        self.assertEqual(s.getAlias("B2"), "wall_t")

    def test_clearing_a_note_and_unit(self):
        p = self.sheet()
        p.set("some_value", 1, note="note")
        p.set("some_value", 2, note="")
        self.assertEqual(cell(p.sheet, "E2"), "")
        self.assertEqual({r["name"]: r for r in p.table()}["some_value"]["note"], "")

    def test_notes_starting_with_equals_survive(self):
        p = self.sheet()
        p.set("wall_t", 2.0, note="=not a formula")
        self.assertEqual({r["name"]: r for r in p.table()}["wall_t"]["note"], "=not a formula")
        p.set("floor_t", 2.0, note="'quoted")
        self.assertEqual({r["name"]: r for r in p.table()}["floor_t"]["note"], "'quoted")

    def test_get_unknown_name_suggests_similar_ones(self):
        p = self.sheet()
        p.set("wall_t", 2.0)
        p.set("floor_t", 2.0)
        with self.assertRaises(params.ParamsError) as cm:
            p.get("wal_t")
        self.assertIn("'wal_t' is not a parameter in sheet 'Params' (similar: wall_t)", str(cm.exception))
        self.assertIn("parameters: floor_t, wall_t", str(cm.exception))

    def test_a_formula_that_cannot_evaluate_is_refused_and_the_row_restored(self):
        p = self.sheet()
        p.set("wall_t", 2.0, note="keep")
        before = p.sheet.getContents("B2")
        with self.assertRaises(params.ParamsError):
            p.set("wall_t", None, formula="=no_such_param*2")
        self.assertEqual(p.sheet.getContents("B2"), before)
        self.assertEqual(p.get("wall_t"), 2.0)
        n = len(p.names())
        with self.assertRaises(params.ParamsError):
            p.set("brand_new", None, formula="=also_missing + 1")
        self.assertEqual(len(p.names()), n, "a failed new row leaves nothing behind")
        self.assertIsNone(p.sheet.getCellFromAlias("brand_new"))

    def test_division_by_zero_is_refused(self):
        p = self.sheet()
        with self.assertRaises(params.ParamsError) as cm:
            p.set("ratio_x", None, formula="=1 mm / 0")
        self.assertIn("evaluates to", str(cm.exception))
        self.assertEqual(p.names(), [])

    def test_a_text_result_is_refused_with_the_documented_message(self):
        p = self.sheet()
        with self.assertRaises(params.ParamsError) as cm:
            p.set("label_x", None, formula='="abc"')
        self.assertEqual(str(cm.exception), "Params: 'label_x' was stored as text, not a number; write it as '=<value> <unit>'")
        self.assertEqual(p.names(), [])

    def test_bad_input_raises_before_anything_is_written(self):
        p = self.sheet()
        p.set("wall_t", 2.0)
        cells = sorted(p.sheet.getNonEmptyCells())
        for kwargs, needle in ((dict(name="Bad Name", value=1), "not a valid parameter name"), (dict(name="min", value=1), "reserved"),
                               (dict(name="ab1", value=1), "looks like"), (dict(name="ok_name", value="2,5 mm"), "decimal separator"),
                               (dict(name="ok_name", value=1, source="guess"), "source 'guess' is not valid"),
                               (dict(name="ok_name", value=1, unit="cm"), "unit 'cm' is not supported"),
                               (dict(name="ok_name", value=True), "is not a number")):
            name, value = kwargs.pop("name"), kwargs.pop("value")
            with self.assertRaises(params.ParamsError, msg=name) as cm:
                p.set(name, value, **kwargs)
            self.assertIn(needle, str(cm.exception))
            self.assertTrue(str(cm.exception).startswith("Params: "))
        self.assertEqual(sorted(p.sheet.getNonEmptyCells()), cells)

    def test_comma_decimal_is_refused_not_misparsed(self):
        p = self.sheet()
        with self.assertRaises(params.ParamsError):
            p.set("wall_t", "2,5 mm")
        self.assertEqual(p.names(), [])

    def test_names_keep_row_order(self):
        p = self.sheet()
        for n in ("zeta_len", "alpha_len", "mid_len"):
            p.set(n, 1.0)
        self.assertEqual(p.names(), ["zeta_len", "alpha_len", "mid_len"])

    def test_full_default_table_matches_the_python_formulas(self):
        """The derived formulas (';' separators, max(), unary minus) evaluate in the real sheet to the numbers spec.derived gives."""
        p = self.sheet()
        for r in params.DEFAULT_PARAMS:
            p.set(r["name"], r["value"], unit=r["unit"], source=r["source"], note=r["note"], formula=r["formula"])
        dv = spec.derived(spec.default_spec(), BOARD)
        for name, row in dv.items():
            self.assertAlmostEqual(p.get(name), row["value"], places=6, msg=name)
        self.assertAlmostEqual(p.get("clearance_eff"), 0.64)
        self.assertAlmostEqual(p.get("cav_len"), 90.1)
        self.assertAlmostEqual(p.get("cav_wid"), 57.94)
        self.assertAlmostEqual(p.get("shell_len"), 94.1)
        self.assertAlmostEqual(p.get("shell_wid"), 61.94)
        self.assertAlmostEqual(p.get("shell_h"), 24.0)
        self.assertAlmostEqual(p.get("lid_len"), 97.0)
        self.assertAlmostEqual(p.get("lid_wid"), 64.84)
        # editing an input moves every dependent
        p.set("wall_t", 2.4, source="user_requirement")
        p.set("clearance", 0.35)
        self.assertAlmostEqual(p.get("shell_len"), 94.9, places=6)
        self.assertAlmostEqual(p.get("shell_wid"), 62.79, places=6)
        # the table reports the sources
        rows = {r["name"]: r for r in p.table()}
        self.assertEqual(rows["wall_t"]["source"], "user_requirement")
        self.assertEqual(rows["board_len"]["source"], "datasheet")
        self.assertEqual(rows["cav_len"]["source"], "derived")
        self.assertEqual(squash(rows["cav_len"]["expr"]), "=cav_x1-cav_x0")
        self.assertEqual(rows["wall_t"]["expr"], "")
        self.assertTrue(rows["hook_count"]["needs_rebuild"])
        self.assertFalse(rows["wall_t"]["needs_rebuild"])

    def test_a_sheet_without_a_header_is_readable(self):
        s = self.doc.addObject("Spreadsheet::Sheet", "Parameters")
        s.set("A1", "length")
        s.set("B1", "=40 mm")
        s.setAlias("B1", "length")
        s.set("A2", "count")
        s.set("B2", "=3")
        s.setAlias("B2", "count_n")
        self.doc.recompute()
        p = params.Params(self.doc, "Parameters")
        self.assertEqual(p.names(), ["length", "count_n"])
        self.assertEqual(p.get("length"), 40.0)
        self.assertEqual(p.get("count_n"), 3.0)
        p.set("extra_len", 5)
        self.assertEqual(p.names(), ["length", "count_n", "extra_len"])


@unittest.skipIf(App is None, "FreeCAD is needed for the sheet tests (run through freecadcmd)")
class AliasProbeTest(SheetTestBase):
    """FreeCAD's own setAlias is the authority; RESERVED is an extra guard."""

    def test_the_sheet_rejects_unit_names_with_invalid_alias(self):
        s = self.sheet().sheet
        for name in ("min", "psi", "rad", "bar", "deg", "true", "mil", "kcal"):
            with self.assertRaises(Exception) as cm:
                s.setAlias("A1", name)
            self.assertIn("Invalid alias", str(cm.exception), name)
        for name in ("abs", "sin", "tan", "exp", "log", "pow", "ohm", "day", "sec", "round", "ceil", "floor", "mod"):
            s.setAlias("A1", name)                                        # FreeCAD accepts these, so RESERVED is wrong in this direction
            s.setAlias("A1", "")
        s.setAlias("A1", "ab1")                                           # and accepts cell-address-like lowercase names
        s.setAlias("A1", "")

    def test_probe_reports_freecads_message_and_leaves_no_trace(self):
        p = self.sheet()
        p.set("wall_t", 2.0)
        s = p.sheet
        before = sorted(s.getNonEmptyCells())
        self.assertIn("Invalid alias", params._probe_alias(s, "min"))
        self.assertIn("Invalid alias", params._probe_alias(s, "psi"))
        self.assertEqual(params._probe_alias(s, "good_name"), "")
        self.assertEqual(params._probe_alias(s, "abs"), "")               # FreeCAD accepts it: the probe says so
        self.assertEqual(sorted(s.getNonEmptyCells()), before)
        for name in ("good_name", "min", "abs"):
            self.assertFalse(s.getCellFromAlias(name), name)

    def test_valid_alias_with_a_sheet_is_authoritative(self):
        p = self.sheet()
        p.set("wall_t", 2.0)
        s = p.sheet
        self.assertEqual(params.valid_alias("floor_t", sheet=s), (True, ""))
        ok, reason = params.valid_alias("min", sheet=s)
        self.assertFalse(ok)
        self.assertIn("reserved", reason)                                 # the local guard answers first
        self.assertEqual(params.Params.valid_alias("wall_t", sheet=s), (True, ""))      # an alias the sheet already uses is its own
        # a name the LOCAL rules let through but the sheet refuses is reported with FreeCAD's message
        original = params.SHEET_REJECTED
        params.SHEET_REJECTED = frozenset(original - {"mol"})
        try:
            self.assertEqual(params.valid_alias("mol"), (True, ""))        # the local rules alone would accept it
            ok, reason = params.valid_alias("mol", sheet=s)
            self.assertFalse(ok)
            self.assertIn("not accepted as an alias by the spreadsheet (", reason)
            self.assertIn("Invalid alias", reason)
        finally:
            params.SHEET_REJECTED = original

    def test_every_three_letter_name_freecad_rejects_is_rejected_locally_too(self):
        """Brute force over the 17 576 three-letter names (measured on FreeCAD 1.1.3): the local pre-filter has no hole."""
        s = self.sheet().sheet
        letters = "abcdefghijklmnopqrstuvwxyz"
        holes = []
        for a in letters:
            for b in letters:
                for c in letters:
                    name = a + b + c
                    try:
                        s.setAlias("A1", name)
                        s.setAlias("A1", "")
                    except Exception:    # noqa: BLE001
                        if params.valid_alias(name)[0]:
                            holes.append(name)
        self.assertEqual(holes, [], "names FreeCAD rejects that params.valid_alias accepts: add them to SHEET_REJECTED")

    def test_every_listed_rejection_is_still_rejected_by_this_freecad(self):
        s = self.sheet().sheet
        stale = []
        for name in sorted(params.SHEET_REJECTED):
            try:
                s.setAlias("A1", name)
                s.setAlias("A1", "")
                stale.append(name)
            except Exception:    # noqa: BLE001, S110
                pass
        self.assertEqual(stale, [], "SHEET_REJECTED lists names this FreeCAD accepts")

    def test_four_letter_unit_names_sample(self):
        s = self.sheet().sheet
        rng = __import__("random").Random(5)
        letters = "abcdefghijklmnopqrstuvwxyz"
        names = ["".join(rng.choice(letters) for _ in range(4)) for _ in range(3000)] + ["mbar", "mmol", "sqft", "thou", "kcal"]
        for name in names:
            try:
                s.setAlias("A1", name)
                s.setAlias("A1", "")
            except Exception:    # noqa: BLE001
                self.assertFalse(params.valid_alias(name)[0], name)


@unittest.skipIf(App is None, "FreeCAD is needed for the sheet tests (run through freecadcmd)")
class BindAndTableTest(SheetTestBase):
    def test_bind_prefixes_the_sheet_name_and_drives_the_property(self):
        p = self.sheet()
        p.set("wall_t", 2.0)
        p.set("lid_t", 3.0)
        box = self.doc.addObject("Part::Box", "Box")
        p.bind(box, "Length", "wall_t*2")
        p.bind(box, "Width", "=lid_t + wall_t")
        p.bind(box, "Height", "Params.wall_t*5")                          # already qualified
        self.doc.recompute()
        self.assertEqual(squash(dict(box.ExpressionEngine)["Length"]), "Params.wall_t*2")      # FreeCAD re-prints the expression
        self.assertEqual(squash(dict(box.ExpressionEngine)["Width"]), "Params.lid_t+Params.wall_t")
        self.assertAlmostEqual(box.Length.Value, 4.0)
        self.assertAlmostEqual(box.Width.Value, 5.0)
        self.assertAlmostEqual(box.Height.Value, 10.0)
        p.set("wall_t", 2.5)
        self.doc.recompute()
        self.assertAlmostEqual(box.Length.Value, 5.0)

    def test_bind_placement_component_and_functions(self):
        p = self.sheet()
        p.set("lid_lift", 0.0)
        p.set("floor_t", 2.0)
        p.set("wall_t", 3.0)
        box = self.doc.addObject("Part::Box", "Lid")
        p.bind(box, "Placement.Base.z", "lid_lift + 1 mm")
        p.bind(box, "Length", "max(floor_t; wall_t)")
        self.doc.recompute()
        self.assertAlmostEqual(box.Placement.Base.z, 1.0)
        self.assertAlmostEqual(box.Length.Value, 3.0)
        p.set("lid_lift", 15.0)
        self.doc.recompute()
        self.assertAlmostEqual(box.Placement.Base.z, 16.0)

    def test_bind_errors(self):
        p = self.sheet()
        p.set("wall_t", 2.0)
        box = self.doc.addObject("Part::Box", "Box")
        for expr in ("", "  ", "="):
            with self.assertRaises(params.ParamsError, msg=repr(expr)) as cm:
                p.bind(box, "Length", expr)
            self.assertIn("empty expression", str(cm.exception))
        with self.assertRaises(params.ParamsError) as cm:
            p.bind(box, "NoSuchProperty", "wall_t")
        self.assertIn("cannot bind Box.NoSuchProperty to 'Params.wall_t'", str(cm.exception))

    def test_table_rows_bound_counts_and_flags(self):
        p = self.sheet()
        p.set("wall_t", 2.0, source="user_requirement", note="n")
        p.set("hook_count", 4)
        p.set("bottom_z", -1.5, source="datasheet")
        a = self.doc.addObject("Part::Box", "A")
        b = self.doc.addObject("Part::Box", "B")
        p.bind(a, "Length", "wall_t*2")
        p.bind(a, "Width", "wall_t*3")
        p.bind(b, "Height", "wall_t")
        self.doc.recompute()
        rows = p.table()
        self.assertEqual([r["name"] for r in rows], ["wall_t", "hook_count", "bottom_z"])
        for r in rows:
            self.assertEqual(set(r), {"name", "value", "unit", "expr", "source", "note", "cell", "bound", "needs_rebuild"})
        wall = rows[0]
        self.assertEqual((wall["value"], wall["unit"], wall["expr"], wall["source"], wall["note"], wall["cell"]), (2.0, "mm", "", "user_requirement", "n", "B2"))
        self.assertEqual(wall["bound"], 3)
        self.assertEqual(rows[1]["bound"], 0)
        self.assertTrue(rows[1]["needs_rebuild"])
        self.assertFalse(wall["needs_rebuild"])
        self.assertEqual(rows[2]["value"], -1.5)

    def test_bound_count_uses_word_boundaries(self):
        p = self.sheet()
        p.set("wall_t", 2.0)
        p.set("wall_t_extra", 3.0)
        a = self.doc.addObject("Part::Box", "A")
        p.bind(a, "Length", "wall_t_extra")
        self.doc.recompute()
        rows = {r["name"]: r for r in p.table()}
        self.assertEqual(rows["wall_t"]["bound"], 0)
        self.assertEqual(rows["wall_t_extra"]["bound"], 1)

    def test_table_value_is_none_for_an_unreadable_cell(self):
        p = self.sheet()
        p.set("wall_t", 2.0)
        s = p.sheet
        s.set("B9", "-1.5 mm")                                            # stored as text
        s.setAlias("B9", "text_cell")
        self.doc.recompute()
        rows = {r["name"]: r for r in p.table()}
        self.assertIsNone(rows["text_cell"]["value"])
        self.assertEqual(rows["wall_t"]["value"], 2.0)

    def test_sheet_survives_save_and_reopen(self):
        import tempfile
        p = self.sheet()
        p.set("wall_t", 2.4, source="user_requirement")
        p.set("hook_count", 4)
        path = os.path.join(isolation.fresh_dir("params_"), "p.FCStd")
        self.doc.saveCopy(path)
        self.assertTrue(os.path.exists(path))
        doc2 = App.openDocument(path)
        self.addCleanup(App.closeDocument, doc2.Name)
        q = params.Params(doc2)
        self.assertEqual(q.names(), ["wall_t", "hook_count"])
        self.assertEqual(q.get("wall_t"), 2.4)
        self.assertEqual(q.sheet.OscarStage, "params")
        self.assertEqual({r["name"]: r for r in q.table()}["wall_t"]["source"], "user_requirement")


if __name__ == "__main__":
    unittest.main()
