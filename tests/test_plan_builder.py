"""Translator tests: golden plans build end-to-end into native FreeCAD
feature trees (Spreadsheet + Sketcher + PartDesign), the spreadsheet
drives geometry, per-feature checkpoints roll back cleanly, results survive
save/reopen — and the audit's builder defects (REC-1..17) stay fixed:
blind pocket, slot, counterbore / countersink, rectangle centre, post-
conditions, the one result contract, parameterised names, idempotent
re-apply, parameter sync and derived parameters.

Runs in console mode via tests/run.py (Sketcher/PartDesign are App-level;
no GUI needed)."""

import copy
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

import FreeCAD as App  # noqa: E402

from oscar.plan import _buildcore, builder, schema  # noqa: E402

_WORKBENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(_WORKBENCH, "tests", "fixtures", "plans")
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fixtures", "plans", "manifest.json")


def _plan(name: str) -> dict:
    with open(os.path.join(EXAMPLES, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def _truth(name: str) -> dict:
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)["fixtures"][name]


def _param(value: float) -> dict:
    return {"value": float(value), "assumption": "unit test fixture value"}


def _plate(plan_id: str = "t6a_plate", **extra_params) -> dict:
    """A 60 x 40 x 10 plate (centred on the origin, z 0..10) with a ``top``
    plane at z = thickness for sketches that cut downwards."""
    params = {"plate_len": _param(60), "plate_wid": _param(40),
              "thickness": _param(10)}
    params.update({k: _param(v) for k, v in extra_params.items()})
    return {
        "schema_version": 1, "id": plan_id, "part_class": "plate",
        "units": "mm", "target": "freecad",
        "frame": {"planes": {
            "bottom": {"origin": [0, 0, 0], "normal": [0, 0, 1], "x_dir": [1, 0, 0]},
            "top": {"origin": [0, 0, "thickness"], "normal": [0, 0, 1],
                    "x_dir": [1, 0, 0]}}},
        "parameters": params,
        "features": [
            {"id": "base_sketch", "op": "sketch", "plane": {"frame_plane": "bottom"},
             "profile": {"shape": "rectangle", "width": "plate_len",
                         "height": "plate_wid", "center": [0, 0]}},
            {"id": "base_pad", "op": "pad", "sketch": "base_sketch",
             "length": "thickness"},
        ],
    }


PLATE_VOLUME = 60 * 40 * 10


class _DocTest(unittest.TestCase):
    """Fresh documents that are always closed again."""

    def new_doc(self, name: str = "T6aDoc"):
        doc = App.newDocument(name)
        self.addCleanup(self._close, doc.Name)
        return doc

    @staticmethod
    def _close(name: str) -> None:
        if name in App.listDocuments():
            App.closeDocument(name)

    def build(self, plan: dict, name: str = "T6aBuild", **kw):
        doc = self.new_doc(name)
        report = builder.apply_plan(doc, plan, **kw)
        self.assertTrue(report.get("success"),
                        f"{report.get('stage')}/{report.get('feature_id')}: {report.get('error')}")
        return doc, report

    @staticmethod
    def set_param(doc, sheet_name: str, alias: str, value: float) -> None:
        sheet = doc.getObject(sheet_name)
        cell = sheet.getCellFromAlias(alias)
        assert cell, f"no alias {alias!r} in {sheet_name}"
        sheet.set(cell, "=" + repr(float(value)))
        doc.recompute()

    @staticmethod
    def volume(doc, body_name: str = "Body") -> float:
        return float(doc.getObject(body_name).Shape.Volume)


class GoldenBuildTest(_DocTest):
    def _build(self, name, doc_name):
        doc = self.new_doc(doc_name)
        report = builder.apply_plan(doc, _plan(name))
        self.assertTrue(report.get("success"),
                        f"{name}: {report.get('feature_id')}: {report.get('error')}")
        return doc, report

    def test_all_golden_plans_build(self):
        for name in ("plate_4holes", "l_bracket", "stepped_shaft"):
            with self.subTest(plan=name):
                doc, report = self._build(name, f"Build_{name}")
                body = doc.getObject(report["body"])
                shape = body.Shape
                self.assertTrue(shape.isValid())
                self.assertEqual(len(shape.Solids), 1)
                expected = _truth(name)["volume_mm3"]
                self.assertAlmostEqual(shape.Volume, expected, delta=2.0)
                self.assertGreaterEqual(report["feature_count"], 2)
                self.assertGreaterEqual(report["parameter_count"], 4)
                for f in report["features"]:
                    if f["op"] == "sketch":
                        self.assertTrue(
                            f["info"].get("fully_constrained"),
                            f"{name}/{f['id']} not fully constrained: {f['info']}")
                        self.assertGreaterEqual(f["info"]["dimensional_expressions"], 2)
                    else:
                        self.assertIn("volume_delta", f["info"]["metrics"])

    @staticmethod
    def _cell(plan, param):
        row = list(plan["parameters"]).index(param) + 1
        return f"B{row}"

    def test_spreadsheet_resizes_plate(self):
        doc, report = self._build("plate_4holes", "ResizePlate")
        sheet = doc.getObject(report["spreadsheet"])
        body = doc.getObject(report["body"])
        sheet.set(self._cell(_plan("plate_4holes"), "thickness"), "12.0")
        doc.recompute()
        # 100*60*12 - 4 * pi * 5^2 * 12
        expected = 100 * 60 * 12 - 4 * math.pi * 25 * 12
        self.assertAlmostEqual(body.Shape.Volume, expected, delta=3.0)

    def test_spreadsheet_resizes_bracket_depth(self):
        doc, report = self._build("l_bracket", "ResizeBracket")
        sheet = doc.getObject(report["spreadsheet"])
        body = doc.getObject(report["body"])
        sheet.set(self._cell(_plan("l_bracket"), "depth"), "45.0")
        doc.recompute()
        expected = 1100 * 45 - math.pi * 36 * 10
        self.assertAlmostEqual(body.Shape.Volume, expected, delta=3.0)

    def test_spreadsheet_resizes_shaft_radius(self):
        doc, report = self._build("stepped_shaft", "ResizeShaft")
        sheet = doc.getObject(report["spreadsheet"])
        body = doc.getObject(report["body"])
        # r1 10 → 11 (sketch expression chain re-drives the profile)
        sheet.set(self._cell(_plan("stepped_shaft"), "r1"), "11.0")
        doc.recompute()
        d1 = math.pi * 121 * 40
        rest = math.pi * 36 * 30 + math.pi * 16 * 20
        self.assertAlmostEqual(body.Shape.Volume, d1 + rest, delta=8.0)

    def test_build_survives_save_reopen(self):
        doc, report = self._build("plate_4holes", "ReopenDoc")
        path = os.path.join(tempfile.mkdtemp(prefix="plan_fcstd_"),
                            "plate.FCStd")
        doc.saveAs(path)
        App.closeDocument(doc.Name)
        doc2 = App.openDocument(path)
        self.addCleanup(App.closeDocument, doc2.Name)
        body = doc2.getObject(report["body"])
        self.assertIsNotNone(body)
        self.assertTrue(body.Shape.isValid())
        self.assertAlmostEqual(body.Shape.Volume, 45486.7, delta=2.0)
        # expressions survive: resize still works after reopen
        sheet = doc2.getObject(report["spreadsheet"])
        sheet.set(self._cell(_plan("plate_4holes"), "thickness"), "12.0")
        doc2.recompute()
        expected = 100 * 60 * 12 - 4 * math.pi * 25 * 12
        self.assertAlmostEqual(body.Shape.Volume, expected, delta=3.0)

    def test_checkpoint_rolls_back_to_last_good_feature(self):
        plan = _plan("plate_4holes")
        # the first 6 features are valid and the plan validates; then a
        # pocket on the bottom face (it would cut downwards, out of the
        # material) must fail at exactly that feature and leave the
        # document at the last good state (ES-033)
        plan["features"].append({
            "id": "bad_pocket", "op": "pocket",
            "sketch": "base_sketch", "depth": "cut_depth",
        })
        plan["parameters"]["cut_depth"] = {
            "value": 5.0, "assumption": "pocket that cuts nothing (test)"}
        doc = self.new_doc("RollbackDoc")
        report = builder.apply_plan(doc, plan)
        self.assertFalse(report.get("success"))
        self.assertEqual(report.get("feature_id"), "bad_pocket")
        self.assertEqual(report.get("stage"), "build")
        self.assertEqual(len(report.get("completed", [])), 6)
        # document still has the 6 good features as a valid solid
        body = doc.getObject("Body")
        self.assertTrue(body.Shape.isValid())
        self.assertAlmostEqual(body.Shape.Volume, 45486.7, delta=2.0)
        self.assertIsNone(doc.getObject("bad_pocket"))
        self.assertEqual(body.Tip.Name, "hole_d")

    def test_negative_dimension_is_rejected_by_validation_not_the_builder(self):
        plan = _plan("plate_4holes")
        plan["features"].append({"id": "bad_pocket", "op": "pocket",
                                 "sketch": "base_sketch", "depth": "neg_depth"})
        plan["parameters"]["neg_depth"] = _param(-5.0)
        doc = self.new_doc("NegDepthDoc")
        report = builder.apply_plan(doc, plan)
        self.assertFalse(report["success"])
        self.assertEqual(report["stage"], "validation")
        self.assertTrue(any("bad_pocket" in e and "positive" in e
                            for e in report["errors"]), report["errors"])
        self.assertEqual([o.Name for o in doc.Objects], [])

    def test_invalid_plan_fails_validation_before_building(self):
        plan = _plan("l_bracket")
        plan["features"][0]["profile"]["segments"][0]["to"] = ["nope", 0]
        doc = self.new_doc("InvalidPlanDoc")
        report = builder.apply_plan(doc, plan)
        self.assertFalse(report["success"])
        self.assertEqual(report["stage"], "validation")
        self.assertTrue(any("nope" in e for e in report["errors"]))
        # nothing was created
        self.assertEqual([o.Name for o in doc.Objects
                          if o.TypeId.startswith("PartDesign")], [])

    def test_garbage_input_never_raises(self):
        doc = self.new_doc("GarbageDoc")
        for garbage in (None, [], {}, {"parameters": {}, "features": []}, "plan"):
            with self.subTest(plan=repr(garbage)[:20]):
                report = builder.apply_plan(doc, garbage)
                self.assertFalse(report["success"])
                self.assertEqual(report["stage"], "validation")
                self.assertTrue(report["errors"])


class ResultContractTest(_DocTest):
    CONTRACT = {"success", "error", "errors", "feature_id", "stage"}

    def test_success_carries_the_contract_fields(self):
        _doc, report = self.build(_plate(), "ContractOk")
        self.assertTrue(self.CONTRACT <= set(report))
        self.assertEqual((report["error"], report["errors"], report["feature_id"],
                          report["stage"]), ("", [], None, "built"))

    def test_every_failure_has_a_text_a_stage_and_never_none(self):
        doc = self.new_doc("ContractFail")
        results = {
            "validation": builder.apply_plan(doc, {"parameters": {}, "features": []}),
            "document": builder.apply_plan(doc, _plate(), body_name="not valid"),
            "no body": builder.apply_feature(doc, _plate(), "base_pad"),
            "no feature": builder.apply_feature(doc, _plate(), "ghost"),
            "no sheet": builder.sync_parameters(doc, _plate()),
        }
        for label, result in results.items():
            with self.subTest(case=label):
                self.assertFalse(result["success"])
                self.assertTrue(self.CONTRACT <= set(result), result)
                self.assertTrue(result["error"], result)
                self.assertTrue(result["errors"], result)
                text = builder.format_build_failure(result)
                self.assertTrue(text.startswith("Plan failed at '"), text)
                self.assertNotIn("None", text)

    def test_build_failure_reports_feature_stage_and_completed(self):
        plan = _plate(cut=5)
        plan["features"].append({"id": "bad_cut", "op": "pocket",
                                 "sketch": "base_sketch", "depth": "cut"})
        doc = self.new_doc("ContractBuildFail")
        result = builder.apply_plan(doc, plan)
        self.assertEqual((result["stage"], result["feature_id"]), ("build", "bad_cut"))
        text = builder.format_build_failure(result)
        self.assertIn("Plan failed at 'bad_cut': ", text)
        self.assertIn("Completed features: ['base_sketch', 'base_pad']", text)
        self.assertNotIn("None", text)


class FormatBuildFailureTest(unittest.TestCase):
    fmt = staticmethod(builder.format_build_failure)

    def test_feature_failure_with_completed(self):
        text = self.fmt({"success": False, "feature_id": "hole_a", "stage": "build",
                         "error": "boom", "errors": ["boom"], "completed": ["s", "p"]})
        self.assertEqual(text, "Plan failed at 'hole_a': boom\nCompleted features: ['s', 'p']")

    def test_stage_is_used_when_there_is_no_feature(self):
        self.assertEqual(self.fmt({"stage": "validation", "error": "bad plan"}),
                         "Plan failed at 'validation': bad plan")

    def test_errors_are_joined_when_there_is_no_error_text(self):
        text = self.fmt({"stage": "validation", "errors": ["a", "b", "  ", None]})
        self.assertEqual(text, "Plan failed at 'validation': a; b")

    def test_missing_fields_never_print_none(self):
        for result in ({}, {"feature_id": None, "stage": None, "error": None,
                                "errors": None},
                       {"errors": []}, {"error": ""}):
            with self.subTest(result=result):
                text = self.fmt(result)
                self.assertNotIn("None", text)
                self.assertEqual(text, "Plan failed at 'plan': unknown error")

    def test_empty_completed_list_is_still_reported(self):
        self.assertTrue(self.fmt({"error": "x", "completed": []})
                        .endswith("Completed features: []"))

    def test_non_dict_input(self):
        self.assertEqual(self.fmt(None), "Plan failed: unknown error")
        self.assertEqual(self.fmt("oops"), "Plan failed: oops")

    def test_old_style_result_still_formats(self):
        text = self.fmt({"success": False, "stage": "validation",
                         "errors": ["parameters must be an object"]})
        self.assertIn("parameters must be an object", text)

    def test_failure_and_success_helpers(self):
        fail = _buildcore.failure("build", None, ["e1", "e2"], feature_id="x")
        self.assertEqual((fail["error"], fail["errors"], fail["feature_id"]),
                         ("e1; e2", ["e1", "e2"], "x"))
        empty = _buildcore.failure("build")
        self.assertEqual(empty["error"], "unknown build error")
        self.assertEqual(empty["errors"], ["unknown build error"])
        ok = _buildcore.success(body="Body")
        self.assertEqual((ok["success"], ok["stage"], ok["errors"], ok["body"]),
                         (True, "built", [], "Body"))


class PostconditionTest(unittest.TestCase):
    """The pure post-condition rules (volume delta sign, one valid solid, bbox)."""

    @staticmethod
    def m(volume, solids=1, valid=True, bbox=(0, 0, 0, 10, 10, 10)):
        return {"volume": float(volume), "solids": solids, "valid": valid,
                "bbox": list(bbox) if bbox else None}

    def check(self, op, before, after):
        return _buildcore.check_postconditions(op, "feat", before, after)

    def assertFails(self, op, before, after, text):
        with self.assertRaises(_buildcore.BuildError) as ctx:
            self.check(op, before, after)
        self.assertEqual(ctx.exception.feature_id, "feat")
        self.assertIn(text, ctx.exception.message)
        self.assertIn("volume_delta", ctx.exception.metrics)
        return ctx.exception

    def test_additive_and_subtractive_happy_paths(self):
        metrics = self.check("pad", self.m(0, 0, bbox=None), self.m(1000))
        self.assertEqual(metrics["volume_delta"], 1000.0)
        self.check("pad", self.m(1000), self.m(1500, bbox=(0, 0, 0, 12, 10, 10)))
        self.check("pocket", self.m(1000), self.m(900))
        self.check("hole", self.m(1000), self.m(990))
        self.check("revolution", self.m(1000), self.m(1200))

    def test_a_tolerance_halo_on_the_bounding_box_is_not_a_grown_box(self):
        """CI regression: a hole in a 100 x 60 x 8 plate grew the box by 0.036 mm on a runner (BRep tolerance); the check
        used ~1e-6 mm of slack and escalated a correct build."""
        plate = (-50.0, -30.0, 0.0, 50.0, 30.0, 8.0)
        halo = (-50.0, -30.0, -0.036, 50.0, 30.0, 8.036)
        self.check("hole", self.m(1000, bbox=plate), self.m(990, bbox=halo))
        self.check("pocket", self.m(1000, bbox=plate), self.m(990, bbox=halo))
        self.check("fillet", self.m(1000, bbox=plate), self.m(995, bbox=halo))
        self.check("pad", self.m(1000, bbox=plate), self.m(1100, bbox=(-50.0, -30.0, 0.036, 50.0, 30.0, 8.0)))

    def test_a_cut_that_really_grows_the_box_still_fails(self):
        plate = (-50.0, -30.0, 0.0, 50.0, 30.0, 8.0)
        for grown in ((-50.0, -30.0, -2.0, 50.0, 30.0, 8.0),             # 2 mm below the plate
                      (-50.0, -30.0, 0.0, 50.0, 30.0, 8.5),              # 0.5 mm above it
                      (-50.0, -30.0, 0.0, 50.2, 30.0, 8.0)):             # 0.2 mm wider: > 0.1 % of the extent
            self.assertFails("hole", self.m(1000, bbox=plate), self.m(990, bbox=grown), "grew the bounding box")

    def test_the_slack_is_proportional_for_big_parts(self):
        big = (0.0, 0.0, 0.0, 2000.0, 1000.0, 50.0)
        self.check("pocket", self.m(1e6, bbox=big), self.m(9e5, bbox=(0.0, 0.0, 0.0, 2001.5, 1000.0, 50.0)))   # 1.5 mm < 0.1 % of 2 m
        self.assertFails("pocket", self.m(1e6, bbox=big), self.m(9e5, bbox=(0.0, 0.0, 0.0, 2005.0, 1000.0, 50.0)),
                         "grew the bounding box")

    def test_pad_that_adds_nothing_fails(self):
        self.assertFails("pad", self.m(1000), self.m(1000), "added no material")
        self.assertFails("pad", self.m(0, 0, bbox=None), self.m(0, 1), "no volume")

    def test_cut_that_removes_nothing_fails(self):
        for op in ("pocket", "hole"):
            self.assertFails(op, self.m(1000), self.m(1000), "cut nothing")

    def test_cut_that_removes_everything_fails(self):
        err = self.assertFails("pocket", self.m(1000), self.m(0, 0, bbox=None),
                               "removed all material")
        self.assertEqual(err.metrics["volume_after"], 0.0)

    def test_two_solids_and_invalid_solid_fail(self):
        self.assertFails("pad", self.m(1000), self.m(1500, solids=2), "2 separate solids")
        self.assertFails("pad", self.m(1000), self.m(1500, valid=False), "invalid solid")

    def test_bounding_box_rules(self):
        self.assertFails("pad", self.m(1000), self.m(1100, bbox=(0, 0, 0, 9, 10, 10)),
                         "shrank the bounding box")
        self.assertFails("pocket", self.m(1000), self.m(900, bbox=(0, 0, 0, 11, 10, 10)),
                         "grew the bounding box")
        self.assertFails("pad", self.m(1000), self.m(1100, bbox=None), "no usable bounding box")
        self.assertFails("pad", self.m(1000), self.m(1100, bbox=(0, 0, 0, math.inf, 1, 1)),
                         "no usable bounding box")

    def test_tolerance_is_relative(self):
        # a 1e-9 relative change is numerical noise, not a feature
        self.assertFails("pocket", self.m(1e6), self.m(1e6 - 1e-4), "cut nothing")
        self.check("pocket", self.m(1e6), self.m(1e6 - 2.0))

    def test_metrics_are_rounded_and_complete(self):
        metrics = self.check("pad", self.m(1000), self.m(1234.56789,
                                                         bbox=(0, 0, 0, 12.34567, 10, 10)))
        self.assertEqual(set(metrics), {"volume_before", "volume_after",
                                        "volume_delta", "solids", "bbox"})
        self.assertEqual(metrics["volume_after"], 1234.5679)


class PostconditionBuildTest(_DocTest):
    def test_pocket_that_cuts_nothing_fails_with_metrics(self):
        plan = _plate(cut=5)
        plan["features"].append({"id": "air_cut", "op": "pocket",
                                 "sketch": "base_sketch", "depth": "cut"})
        doc = self.new_doc("AirCut")
        result = builder.apply_plan(doc, plan)
        self.assertFalse(result["success"])
        self.assertEqual(result["feature_id"], "air_cut")
        self.assertIn("cut nothing", result["error"])
        self.assertEqual(result["metrics"]["volume_delta"], 0.0)
        self.assertEqual(result["completed"], ["base_sketch", "base_pad"])
        self.assertIsNone(doc.getObject("air_cut"))
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME, delta=0.01)
        self.assertEqual(doc.getObject("Body").Tip.Name, "base_pad")

    def test_pad_inside_the_body_adds_nothing(self):
        plan = _plate(inner_len=10, inner_wid=10, inner_h=2)
        plan["features"] += [
            {"id": "inner_sketch", "op": "sketch", "plane": {"frame_plane": "bottom"},
             "profile": {"shape": "rectangle", "width": "inner_len",
                         "height": "inner_wid", "center": [0, 0]}},
            {"id": "inner_pad", "op": "pad", "sketch": "inner_sketch",
             "length": "inner_h"}]
        doc = self.new_doc("InnerPad")
        result = builder.apply_plan(doc, plan)
        self.assertEqual(result["feature_id"], "inner_pad")
        self.assertTrue("added no material" in result["error"]
                        or "recompute failed" in result["error"], result["error"])

    def test_disconnected_pad_is_not_silently_accepted(self):
        plan = _plate(far_len=10, far_wid=10, far_h=5, far_x=200)
        plan["features"] += [
            {"id": "far_sketch", "op": "sketch", "plane": {"frame_plane": "bottom"},
             "profile": {"shape": "rectangle", "width": "far_len",
                         "height": "far_wid", "center": ["far_x", 0]}},
            {"id": "far_pad", "op": "pad", "sketch": "far_sketch", "length": "far_h"}]
        doc = self.new_doc("FarPad")
        result = builder.apply_plan(doc, plan)
        self.assertFalse(result["success"])
        self.assertEqual(result["feature_id"], "far_pad")
        self.assertIsNone(doc.getObject("far_pad"))
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME, delta=0.01)


class BlindPocketTest(_DocTest):
    """REC: ``pocket.Depth`` does not exist; the property is ``Length``."""

    def plan(self):
        plan = _plate(pocket_w=20, pocket_h=10, pocket_d=4)
        plan["features"] += [
            {"id": "pocket_sketch", "op": "sketch", "plane": {"frame_plane": "top"},
             "profile": {"shape": "rectangle", "width": "pocket_w",
                         "height": "pocket_h", "center": [0, 0]}},
            {"id": "pocket", "op": "pocket", "sketch": "pocket_sketch",
             "depth": "pocket_d"}]
        return plan

    def test_blind_pocket_builds_and_removes_the_volume(self):
        doc, report = self.build(self.plan(), "BlindPocket")
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME - 20 * 10 * 4, delta=0.01)
        pocket = doc.getObject("pocket")
        self.assertEqual(pocket.Type, "Length")
        self.assertAlmostEqual(pocket.Length.Value, 4.0)
        self.assertIn("Parameters.pocket_d", [e[1] for e in pocket.ExpressionEngine])
        info = report["features"][-1]["info"]
        self.assertAlmostEqual(info["metrics"]["volume_delta"], -800.0, delta=0.01)

    def test_pocket_depth_is_expression_driven(self):
        doc, report = self.build(self.plan(), "BlindPocketEdit")
        self.set_param(doc, report["spreadsheet"], "pocket_d", 6)
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME - 20 * 10 * 6, delta=0.01)

    def test_through_pocket_and_midplane_flag(self):
        plan = self.plan()
        pocket = plan["features"][-1]
        del pocket["depth"]
        pocket["through"] = True
        doc, _ = self.build(plan, "ThroughPocket")
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME - 20 * 10 * 10, delta=0.01)
        self.assertEqual(doc.getObject("pocket").Type, "ThroughAll")


class SlotTest(_DocTest):
    """REC: the slot constraint pass unpacked ``(type, index)`` tuples that
    the geometry pass never produced (crash), and pinned the wrong centre."""

    @staticmethod
    def slot_area(length, width):
        return (length - width) * width + math.pi * (width / 2) ** 2

    def plan(self, **slot):
        profile = {"shape": "slot", "length": "slot_len", "width": "slot_wid",
                   "center": [5, 3]}
        profile.update(slot)
        plan = _plate(slot_len=20, slot_wid=6, slot_ang=30)
        plan["features"] += [
            {"id": "slot_sketch", "op": "sketch", "plane": {"frame_plane": "top"},
             "profile": profile},
            {"id": "slot_cut", "op": "pocket", "sketch": "slot_sketch", "through": True}]
        return plan

    def test_horizontal_slot_builds_fully_constrained(self):
        doc, report = self.build(self.plan(), "SlotFlat")
        sketch = next(f for f in report["features"] if f["id"] == "slot_sketch")
        self.assertTrue(sketch["info"]["fully_constrained"], sketch["info"])
        self.assertEqual(sketch["info"]["solver"], 0)
        self.assertNotIn("redundant", sketch["info"])
        self.assertAlmostEqual(self.volume(doc),
                               PLATE_VOLUME - self.slot_area(20, 6) * 10, delta=0.5)
        # the slot is centred where the plan says (5, 3), not at its cap centre
        bb = doc.getObject("slot_sketch").Shape.BoundBox
        self.assertAlmostEqual((bb.XMin + bb.XMax) / 2, 5.0, places=4)
        self.assertAlmostEqual((bb.YMin + bb.YMax) / 2, 3.0, places=4)
        self.assertAlmostEqual(bb.XMax - bb.XMin, 20.0, places=4)
        self.assertAlmostEqual(bb.YMax - bb.YMin, 6.0, places=4)

    def test_angled_slot_with_literal_angle(self):
        for key in ("angle_deg", "angle"):
            with self.subTest(key=key):
                doc, report = self.build(self.plan(**{key: 30}), f"SlotAngled_{key}")
                sketch = next(f for f in report["features"] if f["id"] == "slot_sketch")
                self.assertTrue(sketch["info"]["fully_constrained"])
                bb = doc.getObject("slot_sketch").Shape.BoundBox
                flat = 14.0
                self.assertAlmostEqual(bb.XMax - bb.XMin,
                                       flat * math.cos(math.radians(30)) + 6, places=3)
                self.assertAlmostEqual(bb.YMax - bb.YMin,
                                       flat * math.sin(math.radians(30)) + 6, places=3)
                self.assertAlmostEqual(self.volume(doc),
                                       PLATE_VOLUME - self.slot_area(20, 6) * 10, delta=0.5)

    def test_slot_dimensions_and_angle_follow_the_spreadsheet(self):
        doc, report = self.build(self.plan(angle_deg="slot_ang"), "SlotEdit")
        sheet = report["spreadsheet"]
        self.set_param(doc, sheet, "slot_len", 30)
        self.assertAlmostEqual(self.volume(doc),
                               PLATE_VOLUME - self.slot_area(30, 6) * 10, delta=0.5)
        self.set_param(doc, sheet, "slot_ang", 90)
        bb = doc.getObject("slot_sketch").Shape.BoundBox
        self.assertAlmostEqual(bb.XMax - bb.XMin, 6.0, places=3)
        self.assertAlmostEqual(bb.YMax - bb.YMin, 30.0, places=3)

    def test_slot_not_longer_than_wide_is_a_validation_error(self):
        plan = self.plan()
        plan["parameters"]["slot_len"]["value"] = 6.0
        doc = self.new_doc("SlotCircle")
        result = builder.apply_plan(doc, plan)
        self.assertEqual(result["stage"], "validation")
        self.assertIn("must exceed its width", result["error"])


class HoleCutTest(_DocTest):
    """REC: counterbore / countersink wrote properties a Hole does not have."""

    @staticmethod
    def plan(**hole_extra):
        plan = _plate(hole_dia=4, cb_dia=8, cb_dep=3, cs_ang=90)
        hole = {"id": "hole", "op": "hole", "plane": {"frame_plane": "top"},
                "position": {"x": 0, "y": 0}, "diameter": "hole_dia", "through": True}
        hole.update(hole_extra)
        plan["features"].append(hole)
        return plan

    COUNTERBORE = PLATE_VOLUME - (math.pi * 4 ** 2 * 3 + math.pi * 2 ** 2 * 7)
    CSINK = PLATE_VOLUME - (math.pi * 2 ** 2 * 10 +
                            math.pi * 2 / 3 * (16 + 8 + 4) - math.pi * 4 * 2)

    def test_counterbore_flat_and_nested_spellings(self):
        for label, extra in {
            "flat": {"counterbore": True, "cb_diameter": "cb_dia", "cb_depth": "cb_dep"},
            "nested": {"counterbore": {"cb_diameter": "cb_dia", "cb_depth": "cb_dep"}},
            "nested short": {"counterbore": {"diameter": "cb_dia", "depth": "cb_dep"}},
        }.items():
            with self.subTest(spelling=label):
                doc, _ = self.build(self.plan(**extra), "Cbore")
                hole = doc.getObject("hole")
                self.assertEqual(hole.HoleCutType, "Counterbore")
                self.assertAlmostEqual(hole.HoleCutDiameter.Value, 8.0)
                self.assertAlmostEqual(hole.HoleCutDepth.Value, 3.0)
                self.assertAlmostEqual(self.volume(doc), self.COUNTERBORE, delta=0.5)
                self.assertTrue(hole.State == ["Up-to-date"] or "Invalid" not in hole.State)

    def test_counterbore_dimensions_follow_the_spreadsheet(self):
        doc, report = self.build(self.plan(counterbore=True, cb_diameter="cb_dia",
                                           cb_depth="cb_dep"), "CboreEdit")
        self.set_param(doc, report["spreadsheet"], "cb_dep", 5)
        expected = PLATE_VOLUME - (math.pi * 16 * 5 + math.pi * 4 * 5)
        self.assertAlmostEqual(self.volume(doc), expected, delta=0.5)

    def test_countersink_with_parameter_and_default_angle(self):
        for label, extra in {
            "param angle": {"countersink": {"cs_diameter": "cb_dia", "cs_angle": "cs_ang"}},
            "default angle": {"countersink": True, "cs_diameter": "cb_dia"},
        }.items():
            with self.subTest(case=label):
                doc, _ = self.build(self.plan(**extra), "Csink")
                hole = doc.getObject("hole")
                self.assertEqual(hole.HoleCutType, "Countersink")
                self.assertAlmostEqual(hole.HoleCutCountersinkAngle.Value, 90.0)
                self.assertAlmostEqual(self.volume(doc), self.CSINK, delta=0.5)

    def test_blind_hole_has_a_flat_bottom(self):
        plan = self.plan()
        hole = plan["features"][-1]
        del hole["through"]
        hole["depth"] = "cb_dep"
        doc, _ = self.build(plan, "BlindHole")
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME - math.pi * 4 * 3, delta=0.01)

    def test_contradictory_hole_cuts_are_validation_errors(self):
        both = self.plan(counterbore=True, countersink=True, cb_diameter="cb_dia",
                         cb_depth="cb_dep", cs_diameter="cb_dia")
        doc = self.new_doc("CutBoth")
        result = builder.apply_plan(doc, both)
        self.assertEqual(result["stage"], "validation")
        self.assertIn("mutually exclusive", result["error"])
        smaller = self.plan(counterbore=True, cb_diameter="hole_dia", cb_depth="cb_dep")
        result = builder.apply_plan(doc, smaller)
        self.assertIn("must exceed the hole diameter", result["error"])


class RectangleCenterTest(_DocTest):
    """REC: a rectangle ``center`` was ignored (always symmetric about the origin)."""

    def plan(self, center):
        plan = _plate(box_w=20, box_h=10, box_d=5, ctr_x=10, ctr_y=-5)
        plan["features"] = [
            {"id": "box_sketch", "op": "sketch", "plane": {"frame_plane": "bottom"},
             "profile": {"shape": "rectangle", "width": "box_w", "height": "box_h",
                         "center": center}},
            {"id": "box_pad", "op": "pad", "sketch": "box_sketch", "length": "box_d"}]
        return plan

    def bbox(self, doc):
        bb = doc.getObject("Body").Shape.BoundBox
        return bb.XMin, bb.XMax, bb.YMin, bb.YMax

    def test_literal_centre_is_honoured(self):
        doc, report = self.build(self.plan([10, -5]), "RectCenterLiteral")
        self.assertEqual(self.bbox(doc), (0.0, 20.0, -10.0, 0.0))
        self.assertTrue(report["features"][0]["info"]["fully_constrained"])

    def test_centred_at_origin_by_default(self):
        plan = self.plan([0, 0])
        del plan["features"][0]["profile"]["center"]
        doc, _ = self.build(plan, "RectCenterOrigin")
        self.assertEqual(self.bbox(doc), (-10.0, 10.0, -5.0, 5.0))

    def test_parameter_centre_rides_the_spreadsheet(self):
        doc, report = self.build(self.plan(["ctr_x", "ctr_y"]), "RectCenterParam")
        self.assertEqual(self.bbox(doc), (0.0, 20.0, -10.0, 0.0))
        self.assertTrue(report["features"][0]["info"]["fully_constrained"])
        self.set_param(doc, report["spreadsheet"], "ctr_x", 25)
        self.set_param(doc, report["spreadsheet"], "ctr_y", 8)
        self.set_param(doc, report["spreadsheet"], "box_w", 30)
        for got, want in zip(self.bbox(doc), (10.0, 40.0, 3.0, 13.0)):
            self.assertAlmostEqual(got, want, places=6)


class NamingTest(_DocTest):
    def test_prefix_and_explicit_names(self):
        doc, report = self.build(_plan("plate_4holes"), "NamedBuild", body_name="Housing",
                                 sheet_name="Dims", prefix="p_")
        self.assertEqual((report["body"], report["spreadsheet"]), ("Housing", "Dims"))
        self.assertEqual(doc.getObject("Housing").TypeId, "PartDesign::Body")
        self.assertEqual(doc.getObject("Dims").TypeId, "Spreadsheet::Sheet")
        self.assertIsNone(doc.getObject("Body"))
        self.assertIsNone(doc.getObject("Parameters"))
        for name in ("p_base_sketch", "p_base_pad", "p_hole_a", "p_hole_a_sketch"):
            self.assertIsNotNone(doc.getObject(name), name)
        self.assertIsNone(doc.getObject("base_pad"))
        pad = doc.getObject("p_base_pad")
        self.assertIn("Dims.thickness", [e[1] for e in pad.ExpressionEngine])
        self.set_param(doc, "Dims", "thickness", 12)
        self.assertAlmostEqual(self.volume(doc, "Housing"),
                               100 * 60 * 12 - 4 * math.pi * 25 * 12, delta=3.0)

    def test_expressions_use_the_real_sheet_name(self):
        """REC: re-apply into a document that already has a 'Parameters'
        sheet bound the new model to the OLD sheet."""
        doc = self.new_doc("ForeignSheet")
        foreign = doc.addObject("Spreadsheet::Sheet", "Parameters")
        foreign.set("A1", "mine")
        foreign_body = doc.addObject("PartDesign::Body", "Body")
        doc.recompute()
        report = builder.apply_plan(doc, _plan("plate_4holes"))
        self.assertTrue(report["success"], report)
        self.assertEqual((report["spreadsheet"], report["body"]),
                         ("Parameters001", "Body001"))
        pad = doc.getObject("base_pad")
        self.assertIn("Parameters001.thickness", [e[1] for e in pad.ExpressionEngine])
        self.set_param(doc, "Parameters001", "thickness", 12)
        self.assertAlmostEqual(self.volume(doc, "Body001"),
                               100 * 60 * 12 - 4 * math.pi * 25 * 12, delta=3.0)
        # the user's own objects are untouched
        self.assertEqual(foreign.get("A1"), "mine")
        self.assertEqual(foreign_body.Group, [])

    def test_explicit_name_held_by_a_foreign_object_is_refused(self):
        doc = self.new_doc("ForeignExplicit")
        doc.addObject("Spreadsheet::Sheet", "Parameters")
        result = builder.apply_plan(doc, _plate(), sheet_name="Parameters")
        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "document")
        self.assertIn("'Parameters'", result["error"])
        self.assertIsNone(doc.getObject("Body"))
        self.assertEqual([o.Name for o in doc.Objects], ["Parameters"])

    def test_invalid_names_are_refused_before_touching_the_document(self):
        doc = self.new_doc("BadNames")
        for kw in ({"body_name": "my body"}, {"sheet_name": "1sheet"}, {"prefix": "a-b"}):
            with self.subTest(**kw):
                result = builder.apply_plan(doc, _plate(), **kw)
                self.assertEqual(result["stage"], "document")
                self.assertEqual([o.Name for o in doc.Objects], [])

    def test_every_object_is_tagged_with_the_plan_id(self):
        doc, report = self.build(_plan("plate_4holes"), "Tagged")
        self.assertEqual(_buildcore.tag_of(doc.getObject("Body")), "plate_4holes")
        self.assertEqual(_buildcore.tag_of(doc.getObject("Parameters")), "plate_4holes")
        self.assertEqual(_buildcore.feature_of(doc.getObject("hole_a_sketch")), "hole_a_sketch")
        self.assertEqual(_buildcore.feature_of(doc.getObject("hole_a")), "hole_a")
        self.assertIsNone(_buildcore.tag_of(doc.getObject("XY_Plane")))


class IdempotencyTest(_DocTest):
    def test_applying_the_same_plan_twice_replaces_the_first_build(self):
        doc, first = self.build(_plan("plate_4holes"), "Idem")
        volume = self.volume(doc)
        names = sorted(o.Name for o in doc.Objects)
        second = builder.apply_plan(doc, _plan("plate_4holes"))
        self.assertTrue(second["success"], second)
        self.assertEqual((second["body"], second["spreadsheet"]), ("Body", "Parameters"))
        self.assertIn("Body", second["replaced"])
        self.assertIn("Parameters", second["replaced"])
        types = [o.TypeId for o in doc.Objects]
        self.assertEqual(types.count("PartDesign::Body"), 1)
        self.assertEqual(types.count("Spreadsheet::Sheet"), 1)
        self.assertEqual(types.count("PartDesign::Hole"), 4)
        self.assertAlmostEqual(self.volume(doc), volume, delta=0.01)
        self.assertEqual(sorted(o.Name for o in doc.Objects), names)   # nothing leaked, nothing renamed

    def test_a_changed_plan_with_the_same_id_takes_effect(self):
        doc, _ = self.build(_plan("plate_4holes"), "IdemChange")
        plan = _plan("plate_4holes")
        plan["parameters"]["thickness"]["value"] = 12.0
        report = builder.apply_plan(doc, plan)
        self.assertTrue(report["success"], report)
        self.assertAlmostEqual(self.volume(doc), 100 * 60 * 12 - 4 * math.pi * 25 * 12, delta=3.0)

    def test_a_fixed_plan_replaces_a_partial_failed_build(self):
        bad = _plate(cut=5)
        bad["features"].append({"id": "air_cut", "op": "pocket",
                                "sketch": "base_sketch", "depth": "cut"})
        doc = self.new_doc("IdemFailed")
        self.assertFalse(builder.apply_plan(doc, bad)["success"])
        good = _plate(cut=5)
        report = builder.apply_plan(doc, good)
        self.assertTrue(report["success"], report)
        self.assertEqual([o.TypeId for o in doc.Objects].count("PartDesign::Body"), 1)
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME, delta=0.01)

    def test_two_builds_of_one_plan_coexist_under_explicit_names(self):
        doc, _ = self.build(_plate(), "Twins")
        again = builder.apply_plan(doc, _plate(), body_name="Body_two",
                                   sheet_name="Params_two", prefix="two_")
        self.assertTrue(again["success"], again)
        self.assertEqual(again["replaced"], [])
        self.assertIsNotNone(doc.getObject("Body"))
        self.assertIsNotNone(doc.getObject("two_base_pad"))
        # re-applying with the same explicit names replaces only that build
        third = builder.apply_plan(doc, _plate(), body_name="Body_two",
                                   sheet_name="Params_two", prefix="two_")
        self.assertTrue(third["success"], third)
        self.assertIn("Body_two", third["replaced"])
        self.assertIsNotNone(doc.getObject("Body"))


class ParametersTest(_DocTest):
    def test_derived_parameters_are_formulas_in_dependency_order(self):
        plan = _plate(pocket_w=20)
        plan["parameters"]["pocket_h"] = {"expr": "pocket_w / 2"}
        plan["parameters"]["pocket_d"] = {"expr": "pocket_h - 6 + thickness / 5"}
        plan["features"] += [
            {"id": "pocket_sketch", "op": "sketch", "plane": {"frame_plane": "top"},
             "profile": {"shape": "rectangle", "width": "pocket_w",
                         "height": "pocket_h", "center": [0, 0]}},
            {"id": "pocket", "op": "pocket", "sketch": "pocket_sketch",
             "depth": "pocket_d"}]
        # pocket_d is declared AFTER pocket_h but the order in the dict must not matter
        plan["parameters"] = dict(reversed(list(plan["parameters"].items())))
        doc, report = self.build(plan, "Derived")
        sheet = doc.getObject(report["spreadsheet"])
        self.assertTrue(sheet.getContents(sheet.getCellFromAlias("pocket_h")).startswith("="))
        self.assertAlmostEqual(sheet.get("pocket_h"), 10.0)
        self.assertAlmostEqual(sheet.get("pocket_d"), 6.0)   # 10 - 6 + 2
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME - 20 * 10 * 6, delta=0.01)
        # editing the source re-drives both derived values and the geometry
        self.set_param(doc, report["spreadsheet"], "pocket_w", 30)
        self.assertAlmostEqual(sheet.get("pocket_h"), 15.0)
        self.assertAlmostEqual(sheet.get("pocket_d"), 11.0)
        self.assertLess(self.volume(doc), PLATE_VOLUME - 30 * 15 * 9)



    def test_spreadsheet_rejects_an_alias_and_nothing_is_left_behind(self):
        plan = _plate()
        plan["parameters"]["A1"] = _param(1)       # schema.validate would refuse it; the builder is the 2nd line
        doc = self.new_doc("BadAlias")
        with self.assertRaises(_buildcore.BuildError) as ctx:
            builder.build_spreadsheet(doc, plan)
        self.assertEqual(ctx.exception.stage, "parameters")
        self.assertIn("'A1'", ctx.exception.message)
        self.assertIn("alias", ctx.exception.message)
        self.assertEqual([o.Name for o in doc.Objects], [])

    def test_validation_catches_the_same_names_first(self):
        for bad in ("A1", "mm", "_hidden", "T"):
            with self.subTest(name=bad):
                plan = _plate()
                plan["parameters"][bad] = _param(1)
                ok, errors, _w = schema.validate(plan)
                self.assertFalse(ok)
                self.assertTrue(any(repr(bad) in e for e in errors), errors)

    def test_sync_parameters_upserts_by_alias(self):
        doc, report = self.build(_plate(), "Sync")
        sheet_name = report["spreadsheet"]
        plan = _plate(extra_dim=7)
        plan["parameters"]["thickness"]["value"] = 12.0
        out = builder.sync_parameters(doc, plan, sheet_name)
        self.assertTrue(out["success"], out)
        self.assertEqual(out["updated"], ["thickness"])
        self.assertEqual(out["added"], ["extra_dim"])
        self.assertEqual(sorted(out["unchanged"]), ["plate_len", "plate_wid"])
        sheet = doc.getObject(sheet_name)
        self.assertAlmostEqual(sheet.get("thickness"), 12.0)
        self.assertAlmostEqual(sheet.get("extra_dim"), 7.0)
        self.assertEqual(sheet.getCellFromAlias("extra_dim"), "B4")
        self.assertEqual(sheet.get("A4"), "extra_dim")
        self.assertAlmostEqual(self.volume(doc), 60 * 40 * 12, delta=0.01)   # the pad follows
        again = builder.sync_parameters(doc, plan, sheet_name)
        self.assertEqual((again["added"], again["updated"]), ([], []))

    def test_sync_keeps_the_cell_a_user_moved_an_alias_to(self):
        doc, report = self.build(_plate(), "SyncMoved")
        sheet = doc.getObject(report["spreadsheet"])
        sheet.setAlias("B3", "")
        sheet.set("C3", "=10.0")
        sheet.setAlias("C3", "thickness")
        doc.recompute()
        plan = _plate()
        plan["parameters"]["thickness"]["value"] = 14.0
        out = builder.sync_parameters(doc, plan, report["spreadsheet"])
        self.assertEqual(out["updated"], ["thickness"])
        self.assertAlmostEqual(sheet.get("thickness"), 14.0)
        self.assertEqual(sheet.getCellFromAlias("thickness"), "C3")

    def test_sync_reports_a_missing_sheet_and_a_bad_plan(self):
        doc = self.new_doc("SyncMissing")
        self.assertEqual(builder.sync_parameters(doc, _plate())["stage"], "document")
        doc2, report = self.build(_plate(), "SyncBad")
        out = builder.sync_parameters(doc2, {"parameters": {}, "features": []},
                                      report["spreadsheet"])
        self.assertEqual(out["stage"], "validation")


class ReapplyTest(_DocTest):
    def test_apply_feature_syncs_a_patched_parameter_and_replaces_the_feature(self):
        doc, report = self.build(_plan("plate_4holes"), "Reapply")
        before = self.volume(doc)
        plan = _plan("plate_4holes")
        plan["parameters"]["hole_dia_small"] = _param(6)
        plan["features"][-1]["diameter"] = "hole_dia_small"      # hole_d gets a smaller bore
        out = builder.apply_feature(doc, plan, "hole_d")
        self.assertTrue(out["success"], out)
        self.assertEqual((out["feature_id"], out["object"], out["stage"]), ("hole_d", "hole_d", "built"))
        self.assertEqual(out["completed"], ["base_sketch", "base_pad", "hole_a", "hole_b", "hole_c", "hole_d"])
        self.assertIn("Parameters.hole_dia_small",
                      [e[1] for e in doc.getObject("hole_d").ExpressionEngine])
        self.assertGreater(self.volume(doc), before)
        self.assertEqual(doc.getObject("Body").Tip.Name, "hole_d")
        self.assertIsNotNone(doc.getObject("Parameters").getCellFromAlias("hole_dia_small"))
        self.assertEqual(out["parameters"]["added"], ["hole_dia_small"])

    def test_apply_feature_refuses_when_later_features_depend_on_it(self):
        doc, _ = self.build(_plan("plate_4holes"), "ReapplyLater")
        out = builder.apply_feature(doc, _plan("plate_4holes"), "hole_a")
        self.assertFalse(out["success"])
        self.assertEqual(out["feature_id"], "hole_a")
        self.assertIn("later features exist (hole_b, hole_c, hole_d", out["error"])
        self.assertIn("re-apply the whole plan", builder.format_build_failure(out))

    def test_rebuild_from_rebuilds_the_feature_and_everything_after_it(self):
        doc, _ = self.build(_plan("plate_4holes"), "Rebuild")
        volume = self.volume(doc)
        out = builder.rebuild_from(doc, _plan("plate_4holes"), "hole_b")
        self.assertTrue(out["success"], out)
        self.assertEqual([f["id"] for f in out["features"]],
                         ["hole_b", "hole_c", "hole_d"])
        self.assertEqual(out["completed"], ["base_sketch", "base_pad", "hole_a",
                                            "hole_b", "hole_c", "hole_d"])
        self.assertEqual([o.TypeId for o in doc.Objects].count("PartDesign::Hole"), 4)
        self.assertAlmostEqual(self.volume(doc), volume, delta=0.01)
        self.assertEqual(doc.getObject("Body").Tip.Name, "hole_d")

    def test_rebuild_from_a_sketch_with_a_changed_dimension(self):
        doc, _ = self.build(_plate(), "RebuildSketch")
        plan = _plate()
        plan["parameters"]["plate_len"]["value"] = 80.0
        out = builder.rebuild_from(doc, plan, "base_sketch")
        self.assertTrue(out["success"], out)
        self.assertAlmostEqual(self.volume(doc), 80 * 40 * 10, delta=0.01)

    def test_failed_reapply_reports_the_feature_and_keeps_the_previous_state(self):
        doc, _ = self.build(_plate(cut=5), "ReapplyFail")
        plan = _plate(cut=5)
        plan["features"].append({"id": "air_cut", "op": "pocket",
                                 "sketch": "base_sketch", "depth": "cut"})
        out = builder.apply_feature(doc, plan, "air_cut")
        self.assertFalse(out["success"])
        self.assertEqual((out["feature_id"], out["stage"]), ("air_cut", "build"))
        self.assertIn("cut nothing", out["error"])
        self.assertEqual(out["completed"], ["base_sketch", "base_pad"])
        self.assertIsNone(doc.getObject("air_cut"))
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME, delta=0.01)

    def test_reapply_with_a_prefix_finds_features_by_tag(self):
        doc, _ = self.build(_plan("plate_4holes"), "ReapplyPrefix", prefix="q_")
        out = builder.apply_feature(doc, _plan("plate_4holes"), "hole_d")
        self.assertTrue(out["success"], out)
        self.assertEqual(out["object"], "q_hole_d")       # the prefix was read back from the Body
        self.assertEqual([o.TypeId for o in doc.Objects].count("PartDesign::Hole"), 4)

    def test_reapply_entry_points_report_problems_instead_of_raising(self):
        doc = self.new_doc("ReapplyProblems")
        for fn in (builder.apply_feature, builder.rebuild_from):
            with self.subTest(fn=fn.__name__):
                no_body = fn(doc, _plate(), "base_pad")
                self.assertEqual(no_body["stage"], "document")
                self.assertIn("apply the whole plan first", no_body["error"])
                unknown = fn(doc, _plate(), "ghost")
                self.assertEqual(unknown["stage"], "validation")
                self.assertIn("'ghost'", unknown["error"])
                self.assertIn("base_pad", unknown["error"])
        self.assertEqual(builder.apply_feature(doc, {"parameters": {}}, "x")["stage"],
                         "validation")


class SketchProfilesTest(_DocTest):
    def test_circle_profile_and_polyline_arc_chain(self):
        plan = _plate(disc_d=16, disc_h=3, disc_x=12)
        plan["features"] += [
            {"id": "disc_sketch", "op": "sketch", "plane": {"frame_plane": "top"},
             "profile": {"shape": "circle", "diameter": "disc_d", "center": ["disc_x", 0]}},
            {"id": "disc_pad", "op": "pad", "sketch": "disc_sketch", "length": "disc_h"}]
        doc, report = self.build(plan, "Circle")
        info = report["features"][-2]["info"]
        self.assertTrue(info["fully_constrained"], info)
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME + math.pi * 64 * 3, delta=0.5)
        self.set_param(doc, report["spreadsheet"], "disc_d", 20)
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME + math.pi * 100 * 3, delta=0.5)

    def test_polyline_with_an_arc_builds_and_follows_its_parameters(self):
        """No golden plan has an arc: a rectangle with one rounded corner
        (derived parameters carry the arc's end points) is built, solved and
        parameter-driven."""
        plan = _plate(p_len=30, p_hgt=20, p_r=6, p_h=5)
        plan["parameters"]["p_hr"] = {"expr": "p_hgt - p_r"}
        plan["parameters"]["p_lr"] = {"expr": "p_len - p_r"}
        plan["features"] += [
            {"id": "d_sketch", "op": "sketch", "plane": {"frame_plane": "top"},
             "profile": {"shape": "profile", "start": [0, 0], "segments": [
                 {"type": "line", "to": ["p_len", 0]},
                 {"type": "line", "to": ["p_len", "p_hr"]},
                 {"type": "arc", "to": ["p_lr", "p_hgt"], "center": ["p_lr", "p_hr"],
                  "ccw": True, "radius": "p_r"},
                 {"type": "line", "to": [0, "p_hgt"]},
                 {"type": "line", "to": [0, 0]}]}},
            {"id": "d_pocket", "op": "pocket", "sketch": "d_sketch", "depth": "p_h"}]
        doc, report = self.build(plan, "ArcProfile")
        info = next(f for f in report["features"] if f["id"] == "d_sketch")["info"]
        self.assertEqual(info["geometry"], "profile")
        self.assertNotIn("redundant", info)
        self.assertTrue(info["fully_constrained"], info)
        area = 30 * 20 - (6 * 6 - math.pi * 36 / 4)
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME - area * 5, delta=0.5)
        self.set_param(doc, report["spreadsheet"], "p_r", 8)        # derived end points follow
        area = 30 * 20 - (8 * 8 - math.pi * 64 / 4)
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME - area * 5, delta=0.5)

    def test_a_semicircular_arc_fails_with_an_actionable_message(self):
        plan = _plate(d_len=20, d_r=8, d_h=5)
        plan["features"] += [
            {"id": "d_sketch", "op": "sketch", "plane": {"frame_plane": "top"},
             "profile": {"shape": "profile", "start": [0, -8], "segments": [
                 {"type": "line", "to": ["d_len", -8]},
                 {"type": "arc", "to": ["d_len", 8], "center": ["d_len", 0], "ccw": True,
                  "radius": "d_r"},
                 {"type": "line", "to": [0, 8]},
                 {"type": "line", "to": [0, -8]}]}}]
        doc = self.new_doc("SemiArc")
        result = builder.apply_plan(doc, plan)
        self.assertEqual((result["success"], result["feature_id"]), (False, "d_sketch"))
        self.assertIn("redundant constraint", result["error"])
        self.assertIn("slot", result["error"])
        self.assertIsNone(doc.getObject("d_sketch"))
        self.assertEqual(result["completed"], ["base_sketch", "base_pad"])

    def test_revolution_midplane_and_invert_flags_are_honoured(self):
        plan = _plan("stepped_shaft")
        plan["features"][-1]["midplane"] = True
        doc, report = self.build(plan, "RevFlags")
        self.assertTrue(doc.getObject("revolve").Midplane)


class ModuleHygieneTest(_DocTest):
    def test_builder_modules_import_no_freecad_at_module_level(self):
        """The builder package stays importable without FreeCAD (pure at
        import; the translators import FreeCAD inside functions)."""
        import ast

        for name in ("builder", "_buildcore", "_sketchbuild", "_featurebuild", "schema"):
            path = os.path.join(_WORKBENCH, "oscar", "plan", f"{name}.py")
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in tree.body:
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                for imported in names:
                    self.assertNotIn(imported.split(".")[0],
                                     ("FreeCAD", "FreeCADGui", "Part", "Sketcher", "PySide",
                                      "PySide2", "PySide6"), f"{name}.py imports {imported}")

    def test_no_print_in_library_code(self):
        for name in ("builder", "_buildcore", "_sketchbuild", "_featurebuild"):
            with open(os.path.join(_WORKBENCH, "oscar", "plan", f"{name}.py"),
                      encoding="utf-8") as f:
                self.assertNotIn("print(", f.read(), name)

    def test_apply_plan_reports_a_closed_or_missing_document_instead_of_raising(self):
        result = builder.apply_plan(None, _plate())
        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "document")
        self.assertIn("AttributeError", result["error"])
        self.assertTrue(builder.format_build_failure(result).startswith("Plan failed at 'document'"))



if __name__ == "__main__":
    unittest.main()
