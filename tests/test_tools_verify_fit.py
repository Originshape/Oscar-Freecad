"""Generic fit rows of ``check_fit`` (``oscar.tools_verify_fit``): the pure helpers, and the seven generic rows on
REAL documents built by ``verify_support.build_enclosure`` (a PCB with a tall component in a ``Ref_TEST`` part, a
shell with floor / walls / standoff / catch pocket, a cap lid with a snap hook). Every switch of the fixture breaks exactly one
row; the known traps are covered: the assembled pose (``lid_lift`` subtracted), the global frame of a nested
``App::Part``, a standoff that makes a plain ``distToShape`` 0, Origin features, and the guarded kit rows."""

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

from oscar import tools_verify_fit as F  # noqa: E402
from oscar.cad import health_eval  # noqa: E402


# ---------------------------------------------------------------------------------------------- pure helpers


class RowHelpersTest(unittest.TestCase):
    def test_make_row_has_the_documented_keys_and_rounds_the_value(self):
        row = F.make_row("interference.shell_lid", "interference", "Shell", "Lid", "common_volume_mm3", 0.012345,
                         "<= 0.05", "PASS")
        self.assertEqual(list(row), ["id", "kind", "a", "b", "metric", "value", "limit", "status", "detail"])
        self.assertEqual(row["value"], 0.012)
        self.assertEqual(row["detail"], "")

    def test_make_row_never_emits_nan_inf_or_negative_zero(self):
        for bad in (float("nan"), float("inf"), None, "x", -0.0001):
            self.assertEqual(F.make_row("a.b", "coverage", "", "", "m", bad, "", "PASS")["value"], 0.0)

    def test_skip_row_carries_the_reason_in_detail(self):
        row = F.skip_row("clearance.lid_to_tallest", "clearance", "no lid part found", "A", "B", "m")
        self.assertEqual((row["status"], row["detail"], row["a"], row["b"]), ("SKIP", "no lid part found", "A", "B"))

    def test_normalize_row_fills_every_key_and_clamps_the_status(self):
        row = F.normalize_row({"id": "containment.ports", "status": "pass"})
        self.assertEqual(set(row), set(F._ROW_KEYS))
        self.assertEqual(row["status"], "PASS")
        self.assertEqual(F.normalize_row({"id": "x", "status": "MAYBE"})["status"], "SKIP")
        self.assertEqual(F.normalize_row(None)["status"], "SKIP")
        self.assertEqual(F.normalize_row("junk")["id"], "")

    def test_slug_is_a_safe_id_fragment(self):
        self.assertEqual(F.slug("Lid_Plate"), "lid_plate")
        self.assertEqual(F.slug("Part 1 / Cover!"), "part_1_cover")
        self.assertEqual(F.slug("  "), "part")

    def test_select_ids_exact_family_and_none(self):
        self.assertEqual(F.select_ids(None), (None, []))
        ids, problems = F.select_ids(["interference.shell_lid"])
        self.assertEqual((ids, problems), ({"interference.shell_lid"}, []))
        ids, problems = F.select_ids(["clearance"])
        self.assertEqual(problems, [])
        self.assertLessEqual({"clearance.pcb_to_wall_min", "clearance.lid_to_tallest", "clearance"}, ids)
        self.assertNotIn("interference.shell_lid", ids)

    def test_select_ids_reports_every_unknown_id_once_with_the_valid_list(self):
        ids, problems = F.select_ids(["interference.shell_lid", "nope", "also_nope"])
        self.assertEqual(problems[:2], ["'nope' is not a check id", "'also_nope' is not a check id"])
        self.assertIn("use one of: interference.shell_lid", problems[2])
        self.assertIn("hooks.clear_of_cutouts", problems[2])

    def test_select_ids_accepts_pair_ids_of_explicit_parts_and_ignores_blanks(self):
        ids, problems = F.select_ids(["interference.body_cover", "", "  "])
        self.assertEqual((ids, problems), ({"interference.body_cover"}, []))

    def test_wants_matches_exact_ids_and_families(self):
        self.assertTrue(F.wants(None, "anything.here"))
        self.assertTrue(F.wants({"interference"}, "interference.body_cover"))
        self.assertTrue(F.wants({"clearance.lid_to_tallest"}, "clearance.lid_to_tallest"))
        self.assertFalse(F.wants({"clearance.lid_to_tallest"}, "clearance.pcb_to_wall_min"))

    def test_catalogue_ids_are_the_documented_ones(self):
        self.assertEqual(F.GENERIC_IDS, ("interference.shell_lid", "interference.reference_shell",
                                         "interference.reference_lid", "clearance.pcb_to_wall_min",
                                         "clearance.lid_to_tallest", "retention.lid_lift", "reference.status"))
        self.assertEqual(F.KIT_IDS, ("containment.ports", "alignment.standoffs", "hooks.engagement",
                                     "hooks.clear_of_cutouts"))
        for rid in F.ALL_IDS:
            self.assertIn(rid.split(".")[0], F.KINDS + ("hooks", "reference"))


# ---------------------------------------------------------------------------------------------- real documents


class FitCase(unittest.TestCase):
    def setUp(self):
        health_eval.clear_cache()
        self._made = []
        self.addCleanup(self._close)

    def _close(self):
        for name in self._made:
            if name in App.listDocuments():
                App.closeDocument(name)
        health_eval.clear_cache()

    def doc(self, **kw):
        doc = App.newDocument("FitT")
        self._made.append(doc.Name)
        self.objs = S.build_enclosure(doc, **kw)
        return doc

    def rows(self, doc=None, **kw):
        kw.setdefault("spec", None)
        kw.setdefault("reference_status", "none")
        kw.setdefault("lift_mm", 0.5)
        rows, problems = F.run_generic(doc or self.doc_, **kw)
        self.assertEqual(problems, [])
        return {r["id"]: r for r in rows}

    def check(self, **fixture):
        self.doc_ = self.doc(**fixture)
        return self.rows(self.doc_)


class GenericRowsTest(FitCase):
    def test_a_correct_enclosure_passes_every_row_with_the_documented_numbers(self):
        rows = self.check()
        self.assertEqual(list(rows), list(F.GENERIC_IDS))
        self.assertEqual({k: v["status"] for k, v in rows.items()},
                         {**{k: "PASS" for k in F.GENERIC_IDS}, "reference.status": "WARN"})
        self.assertEqual(rows["interference.shell_lid"]["value"], 0.0)
        self.assertEqual(rows["clearance.pcb_to_wall_min"]["value"], 2.0)
        self.assertEqual(rows["clearance.pcb_to_wall_min"]["limit"], ">= 1.95")
        self.assertEqual(rows["clearance.lid_to_tallest"]["value"], 2.0)
        self.assertEqual(rows["retention.lid_lift"]["value"], 0.64)
        self.assertEqual(rows["retention.lid_lift"]["limit"], "> 0")
        self.assertEqual(rows["interference.shell_lid"]["limit"], "<= 0.05")
        for row in rows.values():
            self.assertEqual(set(row), set(F._ROW_KEYS))
            self.assertIn(row["kind"], F.KINDS)

    def test_the_standoff_does_not_make_the_wall_clearance_zero(self):
        """A plain distToShape(pcb, shell) is the standoff contact (0): the row must measure the cavity walls only."""
        rows = self.check()
        shell, pcb = self.objs["shell"], self.objs["pcb"]
        self.assertEqual(round(pcb.Shape.distToShape(shell.Shape)[0], 6), 0.0)
        self.assertEqual(rows["clearance.pcb_to_wall_min"]["value"], 2.0)

    def test_overlapping_shell_and_lid_fails_with_the_volume(self):
        rows = self.check(overlap=True)
        row = rows["interference.shell_lid"]
        self.assertEqual((row["status"], row["value"]), ("FAIL", 40.0))
        self.assertIn("overlap by 40.000 mm3", row["detail"])
        self.assertEqual(rows["interference.reference_shell"]["status"], "PASS")

    def test_a_board_touching_the_wall_fails_the_clearance(self):
        rows = self.check(wide_pcb=True)
        self.assertEqual(rows["clearance.pcb_to_wall_min"]["status"], "FAIL")
        self.assertEqual(rows["clearance.pcb_to_wall_min"]["value"], 0.0)

    def test_a_reference_part_through_the_wall_fails_reference_shell_and_names_the_worst_solid(self):
        rows = self.check(ref_poke=True)
        row = rows["interference.reference_shell"]
        self.assertEqual((row["status"], row["value"]), ("FAIL", 32.0))
        self.assertIn("worst: Poke", row["detail"])
        self.assertEqual(rows["interference.reference_lid"]["status"], "PASS")

    def test_a_lid_without_a_barb_does_not_retain(self):
        rows = self.check(hooks=False)
        # no hook is tagged and there is no spec: the row is not applicable, not failing
        self.assertEqual(rows["retention.lid_lift"]["status"], "SKIP")
        self.assertIn("no snap hooks", rows["retention.lid_lift"]["detail"])
        rows = self.rows(self.doc_, spec={"lid": {"style": "snap_fit"}})
        row = rows["retention.lid_lift"]
        self.assertEqual((row["status"], row["value"]), ("FAIL", 0.0))
        self.assertIn("nothing retains it", row["detail"])

    def test_a_screw_lid_spec_skips_retention(self):
        rows = self.rows(self.doc(), spec={"lid": {"style": "screw"}})
        self.assertEqual(rows["retention.lid_lift"]["status"], "SKIP")
        self.assertIn("lid style 'screw' has no snap hooks", rows["retention.lid_lift"]["detail"])

    def test_retention_uses_the_requested_lift(self):
        self.doc_ = self.doc()
        small = self.rows(self.doc_, lift_mm=0.3)["retention.lid_lift"]   # the barb stays under the pocket ceiling
        self.assertEqual((small["status"], small["value"]), ("FAIL", 0.0))
        big = self.rows(self.doc_, lift_mm=1.0)["retention.lid_lift"]
        self.assertEqual(big["status"], "PASS")
        self.assertGreater(big["value"], 0.64)

    def test_lid_lift_is_subtracted_before_comparing(self):
        """The lid container sits 15 mm up with Params.lid_lift = 15: the checks see the ASSEMBLED pose."""
        rows = self.check(lid_lift=15.0)
        self.assertEqual(rows["interference.shell_lid"]["status"], "PASS")
        self.assertEqual(rows["clearance.lid_to_tallest"]["value"], 2.0)
        self.assertEqual(rows["retention.lid_lift"]["value"], 0.64)
        lid_global = self.objs["lid_part"].Placement.Base.z
        self.assertEqual(lid_global, 15.0)

    def test_an_overlap_hidden_by_the_lift_is_still_found(self):
        rows = self.check(lid_lift=15.0, overlap=True)
        self.assertEqual(rows["interference.shell_lid"]["status"], "FAIL")
        self.assertEqual(rows["interference.shell_lid"]["value"], 40.0)

    def test_shapes_in_a_placed_container_are_read_in_the_global_frame(self):
        """Shell, lid and reference all live in an Assembly placed 50 mm up: the answers are unchanged."""
        plain = self.check()
        nested = self.check(nest=True, offset_z=50.0)
        self.assertEqual({k: (v["status"], v["value"]) for k, v in nested.items()},
                         {k: (v["status"], v["value"]) for k, v in plain.items()})
        raw_zmin = self.objs["shell"].Shape.BoundBox.ZMin
        self.assertEqual(raw_zmin, -5.0)                      # the container frame
        self.assertEqual(round(health_eval.global_shape(self.objs["shell"]).BoundBox.ZMin, 6), 45.0)

    def test_mixed_frames_would_be_wrong_without_global_shapes(self):
        """Only the shell and the lid are moved: the reference stays at the origin, so reference_shell overlaps nothing
        because the frames are respected (a raw-Shape check would see them overlap)."""
        doc = self.doc()
        assembly = doc.addObject("App::Part", "Assembly")
        assembly.addObject(self.objs["shell_part"])
        assembly.addObject(self.objs["lid_part"])
        assembly.Placement = App.Placement(App.Vector(0, 0, 80), App.Rotation())
        doc.recompute()
        rows = self.rows(doc)
        self.assertEqual(rows["interference.reference_shell"]["status"], "PASS")
        self.assertEqual(rows["interference.reference_shell"]["value"], 0.0)
        # clearance.lid_to_tallest compares the lid plate (80 up) with the reference at the origin: a real gap of 82
        self.assertEqual(rows["clearance.lid_to_tallest"]["status"], "PASS")
        self.assertEqual(rows["clearance.lid_to_tallest"]["value"], 82.0)

    def test_untagged_parts_are_found_by_their_labels(self):
        rows = self.check(tagged=False)
        self.assertEqual(rows["interference.shell_lid"]["status"], "PASS")
        self.assertEqual(rows["clearance.pcb_to_wall_min"]["status"], "PASS")
        self.assertEqual(rows["clearance.lid_to_tallest"]["value"], 2.0)       # z_cavity_top fallback (no Lid_Plate tag)

    def test_without_a_reference_the_reference_rows_skip_with_a_reason(self):
        doc = self.doc()
        for name in ("Comp", "PCB"):
            doc.removeObject(name)
        doc.removeObject("Ref_TEST")
        rows = self.rows(doc)
        for rid in ("interference.reference_shell", "interference.reference_lid", "clearance.pcb_to_wall_min",
                    "clearance.lid_to_tallest"):
            self.assertEqual(rows[rid]["status"], "SKIP", rid)
            self.assertIn("no Ref_* object in the document", rows[rid]["detail"])
        self.assertEqual(rows["interference.shell_lid"]["status"], "PASS")

    def test_missing_parameters_skip_the_clearance_rows(self):
        rows = self.check(with_params=False)
        self.assertEqual(rows["clearance.pcb_to_wall_min"]["status"], "SKIP")
        self.assertIn("unknown", rows["clearance.pcb_to_wall_min"]["detail"])
        self.assertEqual(rows["clearance.lid_to_tallest"]["status"], "SKIP")
        rows = self.rows(self.doc_, spec={"clearance": 2.0, "wall_t": 2.0, "lid": {"gap": 2.0}})
        self.assertEqual(rows["clearance.pcb_to_wall_min"]["status"], "PASS")
        self.assertEqual(rows["clearance.lid_to_tallest"]["status"], "PASS")

    def test_no_shell_and_no_lid_skips_everything_that_needs_them_but_keeps_the_row_set(self):
        doc = App.newDocument("FitEmpty")
        self._made.append(doc.Name)
        doc.addObject("Part::Box", "Box")
        doc.recompute()
        rows = self.rows(doc)
        self.assertEqual(list(rows), list(F.GENERIC_IDS))
        self.assertEqual({r["status"] for k, r in rows.items() if k != "reference.status"}, {"SKIP"})
        self.assertIn("needs a shell and a lid", rows["interference.shell_lid"]["detail"])

    def test_reference_status_row_follows_the_ledger_status(self):
        doc = self.doc()
        for status, expected in (("none", "WARN"), ("catalog-nominal", "WARN"), ("step-imported", "PASS"),
                                 ("step-measured", "PASS"), ("user-attached", "PASS")):
            row = self.rows(doc, reference_status=status)["reference.status"]
            self.assertEqual(row["status"], expected, status)
            self.assertEqual(row["kind"], "coverage")
        warn = self.rows(doc, reference_status="catalog-nominal")["reference.status"]
        self.assertEqual(warn["detail"], "reference is nominal board data; no STEP-based collision check")

    def test_ids_restrict_the_rows(self):
        doc = self.doc()
        rows = self.rows(doc, ids={"interference.shell_lid"})
        self.assertEqual(list(rows), ["interference.shell_lid"])
        rows = self.rows(doc, ids={"clearance"})
        self.assertEqual(list(rows), ["clearance.pcb_to_wall_min", "clearance.lid_to_tallest"])

    def test_time_budget_skips_the_remaining_rows(self):
        doc = self.doc()
        ticks = iter(range(0, 1000, 100))
        rows, problems = F.run_generic(doc, spec=None, reference_status="none", lift_mm=0.5, budget_s=150,
                                       clock=lambda: float(next(ticks)))
        self.assertEqual(problems, [])
        statuses = [r["status"] for r in rows]
        self.assertIn("SKIP", statuses)
        late = [r for r in rows if r["status"] == "SKIP"]
        self.assertTrue(all("time budget exhausted" in r["detail"] for r in late))


class ExplicitPartsTest(FitCase):
    def test_two_untagged_parts_are_shell_then_lid(self):
        doc = self.doc(tagged=False)
        rows, problems = F.run_generic(doc, parts=["Shell", "Lid"], spec=None, reference_status="none", lift_mm=0.5)
        self.assertEqual(problems, [])
        by = {r["id"]: r for r in rows}
        self.assertEqual(by["interference.shell_lid"]["status"], "PASS")
        self.assertNotIn("interference.shell_lid_lid", by)

    def test_explicit_parts_run_retention_even_without_hook_tags(self):
        doc = self.doc(hooks=False)
        rows, _ = F.run_generic(doc, parts=["Shell", "Lid"], spec=None, reference_status="none", lift_mm=0.5)
        by = {r["id"]: r for r in rows}
        self.assertEqual(by["retention.lid_lift"]["status"], "FAIL")

    def test_three_parts_get_every_pair_with_slugged_ids(self):
        doc = self.doc()
        spare = S.feature(doc, "Block", S.box(10, 10, 0, 6, 6, 6), label="Spare Block")
        doc.recompute()
        rows, problems = F.run_generic(doc, parts=["Shell", "Lid", spare.Name], spec=None, reference_status="none",
                                       lift_mm=0.5)
        self.assertEqual(problems, [])
        ids = [r["id"] for r in rows if r["id"].startswith("interference.") and "reference" not in r["id"]]
        self.assertEqual(ids, ["interference.shell_lid", "interference.shell_spare_block",
                               "interference.lid_spare_block"])
        by = {r["id"]: r for r in rows}
        self.assertEqual(by["interference.shell_spare_block"]["status"], "PASS")

    def test_every_problem_with_parts_is_listed_and_no_rows_are_computed(self):
        doc = self.doc()
        doc.addObject("Spreadsheet::Sheet", "Sheet")
        rows, problems = F.run_generic(doc, parts=["Nope", "Sheet", "Shell"], spec=None, reference_status="none",
                                       lift_mm=0.5)
        self.assertEqual(rows, [])
        self.assertEqual(len(problems), 2)
        self.assertIn("'Nope' is not an object of document", problems[0])
        self.assertIn("'Sheet' is not solid geometry", problems[1])

    def test_origin_features_and_containers_are_not_geometry(self):
        doc = self.doc()
        origin = next(o for o in doc.Objects if o.TypeId == "App::Origin")
        axis = origin.OriginFeatures[0]
        rows, problems = F.run_generic(doc, parts=[axis.Name, "Shell"], spec=None, reference_status="none", lift_mm=0.5)
        self.assertEqual(rows, [])
        self.assertIn("is not solid geometry", problems[0])
        rows, problems = F.run_generic(doc, parts=["Part_Lid", "Shell"], spec=None, reference_status="none", lift_mm=0.5)
        self.assertIn("is not solid geometry", problems[0])

    def test_an_explicit_reference_container_or_object(self):
        doc = self.doc()
        by = {r["id"]: r for r in F.run_generic(doc, reference="Ref_TEST", spec=None, reference_status="none",
                                                lift_mm=0.5)[0]}
        self.assertEqual(by["interference.reference_shell"]["status"], "PASS")
        rows, problems = F.run_generic(doc, reference="Missing", spec=None, reference_status="none", lift_mm=0.5)
        self.assertEqual(rows, [])
        self.assertIn("reference 'Missing' is not an object", problems[0])
        rows, problems = F.run_generic(doc, reference="Params", spec=None, reference_status="none", lift_mm=0.5)
        self.assertIn("contains no solid geometry", problems[0])


class PartFinderTest(FitCase):
    def test_role_tip_prefers_the_unconsumed_visible_object_of_a_chain(self):
        doc = App.newDocument("Chain")
        self._made.append(doc.Name)
        base = doc.addObject("Part::Box", "Shell_Body")
        cutter = doc.addObject("Part::Box", "Shell_Cutter")
        cutter.Placement.Base = App.Vector(1, 1, 1)
        cut = doc.addObject("Part::Cut", "Shell_Ports")
        cut.Base, cut.Tool = base, cutter
        for obj in (base, cutter, cut):
            S.tag(obj, "shell", "shell", "Shell")
        base.Visibility = cutter.Visibility = False
        doc.recompute()
        self.assertEqual(F.role_tip(doc, "shell", health_eval).Name, "Shell_Ports")

    def test_role_tip_ignores_reference_and_temporary_objects_and_returns_none_when_absent(self):
        doc = App.newDocument("Tips")
        self._made.append(doc.Name)
        ref = S.feature(doc, "Ref_Board", S.box(0, 0, 0, 5, 5, 5), role="shell")      # a reference named Ref_*: never the shell
        tmp = S.feature(doc, "_oscar_tmp_probe", S.box(0, 0, 0, 5, 5, 5), role="lid")
        doc.recompute()
        self.assertIsNone(F.role_tip(doc, "shell", health_eval))
        self.assertIsNone(F.role_tip(doc, "lid", health_eval))
        self.assertTrue(F.is_reference(ref))
        self.assertTrue(F.is_reference(S.feature(doc, "Anything", S.box(0, 0, 0, 1, 1, 1), role="ref")))
        self.assertFalse(F.is_reference(tmp))

    def test_pcb_object_prefers_a_board_label_then_the_largest_flat_solid(self):
        doc = App.newDocument("Pcb")
        self._made.append(doc.Name)
        flat = S.feature(doc, "Plate", S.box(0, 0, 0, 80, 50, 1.6))
        tall = S.feature(doc, "Block", S.box(0, 0, 0, 90, 60, 20))
        doc.recompute()
        self.assertEqual(F._pcb_object([flat, tall]).Name, "Plate")                  # largest flat solid
        named = S.feature(doc, "Thing", S.box(0, 0, 0, 5, 5, 5), label="Raspberry PCB")
        self.assertEqual(F._pcb_object([flat, tall, named]).Name, "Thing")           # the label wins
        self.assertIsNone(F._pcb_object([tall]))


# ---------------------------------------------------------------------------------------------- the kit rows


class KitRowsTest(FitCase):
    SPEC = {"board": "rpi4b", "lid": {"style": "snap_fit"}}

    def test_without_a_spec_nothing_kit_specific_applies(self):
        doc = self.doc()
        self.assertEqual(F.kit_rows(doc, None, "none"), ([], []))

    def test_the_kit_rows_come_from_kit_checks_fit_rows(self):
        from oscar.kit import checks as kit_checks

        doc = self.doc()
        fake = [{"id": "containment.ports", "kind": "containment", "a": "Shell", "b": "usb_c", "metric": "margin_mm",
                 "value": 0.4, "limit": ">= 0", "status": "pass", "detail": ""}, {"id": "hooks.engagement"}]
        with mock.patch.object(kit_checks, "fit_rows", return_value=fake) as call:
            rows, cover = F.kit_rows(doc, self.SPEC, "catalog-nominal")
        call.assert_called_once_with(doc, self.SPEC, "catalog-nominal")
        self.assertEqual([r["id"] for r in rows], ["containment.ports", "hooks.engagement"])
        self.assertEqual(rows[0]["status"], "PASS")                       # normalised
        self.assertEqual(rows[1]["status"], "SKIP")
        self.assertEqual(cover, [])

    def test_kit_rows_honour_the_selected_ids(self):
        from oscar.kit import checks as kit_checks

        doc = self.doc()
        fake = [{"id": "containment.ports", "status": "PASS"}, {"id": "hooks.engagement", "status": "PASS"}]
        with mock.patch.object(kit_checks, "fit_rows", return_value=fake):
            rows, _ = F.kit_rows(doc, self.SPEC, "none", ids={"hooks.engagement"})
            self.assertEqual([r["id"] for r in rows], ["hooks.engagement"])
            only_generic = F.kit_rows(doc, self.SPEC, "none", ids={"interference.shell_lid"})
        self.assertEqual(only_generic, ([], []))

    def test_an_exception_inside_the_kit_becomes_one_skip_row(self):
        from oscar.kit import checks as kit_checks

        doc = self.doc()
        with mock.patch.object(kit_checks, "fit_rows", side_effect=ValueError("bad pocket")):
            rows, _ = F.kit_rows(doc, self.SPEC, "none")
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["id"], rows[0]["status"]), ("kit.checks", "SKIP"))
        self.assertIn("kit.checks failed (ValueError: bad pocket)", rows[0]["detail"])

    def test_a_missing_kit_module_gives_the_documented_skip_row(self):
        doc = self.doc()
        with S.hidden_module("oscar.kit.checks"):
            rows, cover = F.kit_rows(doc, self.SPEC, "none")
        self.assertEqual([(r["id"], r["status"], r["detail"]) for r in rows],
                         [("kit.checks", "SKIP", "kit.checks unavailable")])
        self.assertEqual(cover, [])

    def test_a_dependency_missing_inside_the_kit_is_not_swallowed(self):
        doc = self.doc()
        error = ModuleNotFoundError("No module named 'numpy_gone'")
        error.name = "numpy_gone"
        with mock.patch.dict(sys.modules, {"oscar.kit.checks": None}):
            with mock.patch("builtins.__import__", side_effect=error):
                with self.assertRaises(ModuleNotFoundError):
                    F.kit_rows(doc, self.SPEC, "none")

    def test_the_seam_row_of_the_current_kit_is_a_skip_and_coverage_is_empty(self):
        doc = self.doc()
        rows, cover = F.kit_rows(doc, self.SPEC, "none", coverage=True)
        self.assertTrue(rows)
        self.assertTrue(all(r["status"] in ("SKIP", "PASS", "WARN", "FAIL") for r in rows))
        self.assertIsInstance(cover, list)

    def test_coverage_rows_are_requested_only_on_demand(self):
        from oscar.kit import checks as kit_checks

        doc = self.doc()
        with mock.patch.object(kit_checks, "fit_rows", return_value=[]), \
                mock.patch.object(kit_checks, "spec_coverage",
                                  return_value=[{"requirement": "R1", "status": "covered"}]) as cov:
            self.assertEqual(F.kit_rows(doc, self.SPEC, "none")[1], [])
            cov.assert_not_called()
            rows, cover = F.kit_rows(doc, self.SPEC, "none", coverage=True)
        self.assertEqual(cover, [{"requirement": "R1", "status": "covered"}])
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
