"""Kit fit rows and requirement coverage (oscar/kit/checks.py -> oscar/kit/_fit.py) on REAL built documents.

The four kit rows (``containment.ports``, ``alignment.standoffs``, ``hooks.engagement``, ``hooks.clear_of_cutouts``) must PASS on every
correct model and FAIL, naming the culprit and the fix, when exactly the thing they test is broken; they SKIP with the reason when the
stage they need is not built or does not apply (a screw lid has no hooks); and they never raise. ``spec_coverage`` is the requirement
table of the Build Summary: ``covered`` only when the geometry exists, with the ledger's ``R<n>`` ids when a ledger is bound.
"""

import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.kit import spec  # noqa: E402

try:
    import FreeCAD as App  # noqa: E402
    from oscar.flow import Ledger  # noqa: E402
    from oscar.kit import build, checks, params  # noqa: E402
    from oscar.kit import geometry as G  # noqa: E402
    from oscar.toolctx import ToolContext, use_context  # noqa: E402
except ImportError:
    App = None

_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "enclosure")
with open(os.path.join(_FIX, "rpi4b_board.json"), encoding="utf-8") as _f:
    BOARD = json.load(_f)

needs_fc = unittest.skipIf(App is None, "FreeCAD is needed (run through freecadcmd)")
ROW_KEYS = ["id", "kind", "a", "b", "metric", "value", "limit", "status", "detail"]
IDS = ["containment.ports", "alignment.standoffs", "hooks.engagement", "hooks.clear_of_cutouts"]
REQUEST = "Design a snap-fit enclosure for the Raspberry Pi 4 Model B with port cutouts, vent slots and a snap-fit lid in ABS."


def make_spec(**changes):
    d = spec.default_spec().to_dict()
    for path, value in changes.items():
        cur = d
        keys = path.split(".")
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = value
    return spec.from_dict(d)


def build_until(doc, sp, last=None):
    """Build stages in order up to and including ``last`` (all when None)."""
    for stage in spec.STAGES:
        r = build.build_stage(doc, sp, stage, board=BOARD)
        assert r.ok, r.message
        if stage == last:
            return


def by_id(rows):
    return {r["id"]: r for r in rows}


class Case(unittest.TestCase):
    def new_doc(self, name="FitTest"):
        doc = App.newDocument(name)
        self.addCleanup(App.closeDocument, doc.Name)
        return doc

    def built(self, sp=None, last=None):
        doc = self.new_doc()
        build_until(doc, sp or spec.default_spec(), last)
        return doc


@needs_fc
class DefaultModelTest(Case):
    @classmethod
    def setUpClass(cls):
        cls.doc = App.newDocument("FitShared")
        build_until(cls.doc, spec.default_spec())
        cls.rows = checks.fit_rows(cls.doc, spec.default_spec(), "catalog-nominal")

    @classmethod
    def tearDownClass(cls):
        App.closeDocument(cls.doc.Name)

    def test_four_rows_in_the_documented_order_with_the_documented_shape(self):
        self.assertEqual([r["id"] for r in self.rows], IDS)
        for r in self.rows:
            self.assertEqual(list(r), ROW_KEYS)
            self.assertIn(r["kind"], ("containment", "alignment"))

    def test_every_row_passes_on_the_correct_model(self):
        for r in self.rows:
            self.assertEqual(r["status"], "PASS", r)

    def test_values_and_details_are_informative(self):
        rows = by_id(self.rows)
        self.assertEqual(rows["containment.ports"]["metric"], "blocked_volume_mm3")
        self.assertLessEqual(rows["containment.ports"]["value"], 0.05)
        self.assertIn("connector windows", rows["containment.ports"]["detail"])
        self.assertIn("4 standoffs", rows["alignment.standoffs"]["detail"])
        self.assertLessEqual(rows["alignment.standoffs"]["value"], 0.1)
        self.assertIn("4 barbs", rows["hooks.engagement"]["detail"])
        self.assertGreater(rows["hooks.clear_of_cutouts"]["value"], 0)
        self.assertEqual(rows["hooks.engagement"]["a"], "Lid")
        self.assertEqual(rows["hooks.engagement"]["b"], "Shell")

    def test_rows_are_json_serialisable(self):
        json.dumps(self.rows)

    def test_the_reference_status_argument_is_accepted_in_any_form(self):
        for status in ("none", "catalog-nominal", "step-measured", ""):
            self.assertEqual([r["status"] for r in checks.fit_rows(self.doc, spec.default_spec(), status)], ["PASS"] * 4)
        self.assertEqual(len(checks.fit_rows(self.doc, spec.default_spec().to_dict())), 4)       # a dict spec and the default status

    def test_the_check_does_not_modify_the_document(self):
        before = sorted((o.Name, o.State.__repr__()) for o in self.doc.Objects)
        checks.fit_rows(self.doc, spec.default_spec(), "none")
        self.assertEqual(sorted((o.Name, o.State.__repr__()) for o in self.doc.Objects), before)


@needs_fc
class SkipTest(Case):
    def test_empty_document_skips_every_row_with_a_reason(self):
        rows = checks.fit_rows(self.new_doc(), spec.default_spec(), "none")
        self.assertEqual([r["status"] for r in rows], ["SKIP"] * 4)
        self.assertIn("shell is not built", rows[0]["detail"])

    def test_each_row_waits_for_its_stage(self):
        doc = self.built(last="shell")
        rows = by_id(checks.fit_rows(doc, spec.default_spec(), "none"))
        self.assertIn("ports stage is not built", rows["containment.ports"]["detail"])
        self.assertIn("standoffs stage is not built", rows["alignment.standoffs"]["detail"])
        self.assertEqual(rows["hooks.engagement"]["status"], "SKIP")
        doc = self.built(last="standoffs")
        rows = by_id(checks.fit_rows(doc, spec.default_spec(), "none"))
        self.assertEqual(rows["alignment.standoffs"]["status"], "PASS")
        self.assertEqual(rows["containment.ports"]["status"], "SKIP")
        doc = self.built(last="ports")
        rows = by_id(checks.fit_rows(doc, spec.default_spec(), "none"))
        self.assertEqual(rows["containment.ports"]["status"], "PASS")
        self.assertIn("shell and the lid are needed", rows["hooks.engagement"]["detail"])
        doc = self.built(last="vents")
        rows = by_id(checks.fit_rows(doc, spec.default_spec(), "none"))
        self.assertIn("hooks stage is not built", rows["hooks.engagement"]["detail"])

    def test_a_screw_lid_has_no_hook_rows_to_judge(self):
        sp = make_spec(**{"lid.style": "screw"})
        rows = by_id(checks.fit_rows(self.built(sp), sp, "none"))
        self.assertEqual(rows["hooks.engagement"]["status"], "SKIP")
        self.assertIn("no snap hooks", rows["hooks.clear_of_cutouts"]["detail"])
        self.assertEqual(rows["containment.ports"]["status"], "PASS")

    def test_no_wall_connector_means_no_window_to_measure(self):
        sp = make_spec(**{"ports.auto": False})
        rows = by_id(checks.fit_rows(self.built(sp), sp, "none"))
        self.assertEqual(rows["containment.ports"]["status"], "SKIP")
        self.assertIn("no connector opens through a wall", rows["containment.ports"]["detail"])

    def test_an_unknown_board_never_raises(self):
        rows = checks.fit_rows(self.new_doc(), {"board": "nonexistent_board"}, "none")
        self.assertEqual([r["id"] for r in rows], IDS)
        self.assertEqual({r["status"] for r in rows}, {"SKIP"})

    def test_garbage_spec_never_raises(self):
        for junk in (None, 5, "x", []):
            try:
                rows = checks.fit_rows(self.new_doc(), junk, "none")
            except Exception as exc:    # noqa: BLE001
                self.fail(f"fit_rows raised for {junk!r}: {exc}")
            self.assertTrue(all(r["status"] == "SKIP" for r in rows), junk)


@needs_fc
class FailTest(Case):
    def test_a_missing_window_fails_containment_and_names_the_connector(self):
        sp = make_spec(**{"ports.exclude": ["usb_c_power"]})
        doc = self.built(sp)
        row = by_id(checks.fit_rows(doc, spec.default_spec(), "none"))["containment.ports"]
        self.assertEqual(row["status"], "FAIL", row)
        self.assertIn("usb_c_power", row["detail"])
        self.assertIn("-y wall", row["detail"])
        self.assertGreater(row["value"], 0.05)
        self.assertIn("build_enclosure(stage='ports'", row["detail"])

    def test_a_window_that_is_too_small_fails(self):
        doc = self.built()
        P = params.Params(doc)
        P.set("port_margin", 0.0)                   # the probe margin shrinks with the sheet ...
        rows = by_id(checks.fit_rows(doc, spec.default_spec(), "none"))
        self.assertEqual(rows["containment.ports"]["status"], "PASS")           # ... and a smaller probe still fits the window
        P.set("port_margin", 2.5)                   # a bigger margin than the window was cut with
        rows = by_id(checks.fit_rows(doc, spec.default_spec(), "none"))
        self.assertEqual(rows["containment.ports"]["status"], "FAIL")

    def test_a_displaced_standoff_fails_alignment(self):
        doc = self.built()
        post = doc.getObject("Shell_StandoffPost1")
        pilot = doc.getObject("Shell_StandoffPilot1")
        for obj in (post, pilot):
            for expression in list(obj.ExpressionEngine):
                obj.setExpression(expression[0], None)
            obj.Placement.Base = obj.Placement.Base + App.Vector(1.5, 0, 0)
        doc.recompute()
        row = by_id(checks.fit_rows(doc, spec.default_spec(), "none"))["alignment.standoffs"]
        self.assertEqual(row["status"], "FAIL", row)
        self.assertIn("hole 1", row["detail"])
        self.assertIn("build_enclosure(stage='standoffs'", row["detail"])

    def test_barbs_that_do_not_reach_far_enough_fail_engagement(self):
        doc = self.built()
        params.Params(doc).set("hook_overhang", 1.6)         # the sheet now asks for more engagement than the geometry has
        row = by_id(checks.fit_rows(doc, spec.default_spec(), "none"))["hooks.engagement"]
        self.assertEqual(row["status"], "FAIL", row)
        self.assertIn("needs", row["detail"])
        self.assertIn("_Barb", row["detail"])

    def test_an_opening_beside_a_hook_fails_clearance(self):
        doc = self.built()
        beam = next(o for o in doc.Objects if o.Name.endswith("_Beam") and o.Name.startswith("Lid_Hook"))
        bb = G.global_shape(beam).BoundBox
        wall = "+y" if bb.YMax > 30 else "-y"
        sp = copy.deepcopy(spec.default_spec().to_dict())
        sp["cutouts"] = [{"id": "near", "part": "shell", "face": wall, "u": (bb.XMin + bb.XMax) / 2 + 3, "v": 6.0, "w": 6.0, "h": 4.0,
                          "r": 0.0, "margin": 0.0, "through": True}]
        row = by_id(checks.fit_rows(doc, sp, "none"))["hooks.clear_of_cutouts"]
        self.assertEqual(row["status"], "FAIL", row)
        self.assertIn("near", row["detail"])
        self.assertLess(row["value"], 0)

    def test_a_row_that_crashes_becomes_a_skip_not_an_exception(self):
        from unittest import mock

        from oscar.kit import _fit

        doc = self.built()
        with mock.patch.object(_fit, "_ROWS", (lambda k: 1 / 0,) * 4):
            rows = checks.fit_rows(doc, spec.default_spec(), "none")
        self.assertEqual([r["status"] for r in rows], ["SKIP"] * 4)
        self.assertIn("ZeroDivisionError", rows[0]["detail"])


@needs_fc
class CoverageTest(Case):
    def keyed(self, rows):
        return {r["key"]: r for r in rows}

    def test_no_document_means_no_information(self):
        self.assertEqual(checks.spec_coverage(spec.default_spec(), BOARD), [])
        self.assertEqual(checks.spec_coverage(spec.default_spec(), BOARD, None), [])

    def test_a_full_model_covers_every_requirement(self):
        doc = self.built()
        rows = checks.spec_coverage(spec.default_spec(), BOARD, doc)
        self.assertEqual([r["key"] for r in rows], ["shell", "ports", "lid", "vents", "snap", "material", "source", "parameters"])
        for r in rows:
            self.assertEqual(set(r), {"requirement", "key", "status", "by", "detail"})
            self.assertEqual(r["status"], "covered", r)
            self.assertEqual(r["requirement"], r["key"])               # no ledger: the key stands in
        k = self.keyed(rows)
        self.assertIn("openings cut", k["ports"]["detail"])
        self.assertIn("4/4 hooks", k["snap"]["detail"])
        self.assertIn("nominal board data", k["source"]["detail"])
        self.assertEqual(k["shell"]["by"], "Shell")
        self.assertEqual(k["lid"]["by"], "Lid")

    def test_nothing_is_covered_before_the_stage_exists(self):
        doc = self.built(last="shell")
        k = self.keyed(checks.spec_coverage(spec.default_spec(), BOARD, doc))
        self.assertEqual(k["shell"]["status"], "covered")
        for key in ("ports", "lid", "vents", "snap"):
            self.assertEqual(k[key]["status"], "missing", key)
        self.assertEqual(k["parameters"]["status"], "covered")

    def test_not_applicable_requirements(self):
        sp = make_spec(**{"lid.style": "screw", "vents.enabled": False})
        doc = self.built(sp)
        k = self.keyed(checks.spec_coverage(sp, BOARD, doc))
        self.assertEqual(k["vents"]["status"], "n/a")
        self.assertEqual(k["snap"]["status"], "n/a")
        self.assertEqual(k["lid"]["status"], "covered")

    def test_a_missing_opening_makes_ports_partial(self):
        sp = make_spec(**{"ports.exclude": ["usb_c_power"]})
        doc = self.built(sp)
        k = self.keyed(checks.spec_coverage(spec.default_spec(), BOARD, doc))
        self.assertEqual(k["ports"]["status"], "partial")

    def test_missing_params_sheet(self):
        doc = self.new_doc()
        k = self.keyed(checks.spec_coverage(spec.default_spec(), BOARD, doc))
        self.assertEqual(k["parameters"]["status"], "missing")
        self.assertEqual(k["source"]["status"], "missing")

    def test_the_bound_ledger_supplies_the_requirement_ids(self):
        doc = self.built()
        ledger = Ledger.from_request(REQUEST)
        with use_context(ToolContext(ledger=ledger)):
            rows = checks.spec_coverage(spec.default_spec(), BOARD, doc)
        ids = {r["key"]: r["requirement"] for r in rows}
        wanted = {r.id for r in ledger.requirements}
        self.assertTrue(set(ids.values()) <= wanted | set(ids), ids)
        self.assertEqual(ids["shell"], "R1")
        self.assertEqual(ids["ports"], "R2")
        self.assertEqual(ids["lid"], "R3")
        self.assertEqual(ids["vents"], "R4")
        self.assertEqual(ids["snap"], "R5")
        self.assertEqual(ids["material"], "R6")
        self.assertEqual(ids["parameters"], "R7")
        self.assertEqual(ids["source"], "source")                  # this request seeded no source requirement: the key stands in


if __name__ == "__main__":
    unittest.main()
