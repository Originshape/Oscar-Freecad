"""Validation and recovery workflow of the FeaturePlan builder, for real in
FreeCAD (console mode): failed operations come back with a machine-readable
``cause`` and recovery ``alternatives`` (the model-facing text renders both);
``reapply_plan_feature(rebuild_later=true)`` rebuilds a feature and its
dependents in place instead of re-applying the whole plan; a parameter change
that stales a plan-built fillet edge link is REPAIRED from the stored rule
instead of refused; and the per-feature validation overhead (body metrics +
post-conditions) stays small. Runs in console mode via tests/run.py."""

import copy
import json
import math
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

import FreeCAD as App  # noqa: E402

from oscar.plan import _buildcore, builder, schema  # noqa: E402

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "fixtures", "plans")


def param(value: float) -> dict:
    return {"value": float(value), "source": {"kind": "user_requirement", "ref": "unit test requirement"}}


def plate_plan(plan_id="vr", features=(), **params) -> dict:
    """A 60 x 40 x 10 plate centred on the origin (z 0..10) with a top plane."""
    plan = {
        "schema_version": 2, "id": plan_id, "part_class": "plate", "units": "mm", "target": "freecad",
        "frame": {"planes": {
            "bottom": {"origin": [0, 0, 0], "normal": [0, 0, 1], "x_dir": [1, 0, 0]},
            "top": {"origin": [0, 0, "thickness"], "normal": [0, 0, 1], "x_dir": [1, 0, 0]}}},
        "parameters": {"plate_len": param(60), "plate_wid": param(40), "thickness": param(10)},
        "features": [
            {"id": "base_sk", "op": "sketch", "plane": {"frame_plane": "bottom"},
             "profile": {"shape": "rectangle", "width": "plate_len", "height": "plate_wid"}},
            {"id": "base_pad", "op": "pad", "sketch": "base_sk", "length": "thickness"}],
    }
    plan["parameters"].update({k: param(v) for k, v in params.items()})
    plan["features"].extend(copy.deepcopy(list(features)))
    return plan


def hole(fid="hole1", x="hole_x", y=0, d="hole_d") -> dict:
    return {"id": fid, "op": "hole", "plane": {"frame_plane": "top"}, "position": {"x": x, "y": y},
            "diameter": d, "through": True}


PLATE = 60 * 40 * 10


class _DocTest(unittest.TestCase):
    def new_doc(self, name="VRDoc"):
        doc = App.newDocument(name)
        self.addCleanup(lambda n=doc.Name: App.closeDocument(n) if n in App.listDocuments() else None)
        return doc

    def build(self, plan, name="VRDoc", **kw):
        doc = self.new_doc(name)
        report = builder.apply_plan(doc, plan, **kw)
        self.assertTrue(report.get("success"),
                        f"{report.get('stage')}/{report.get('feature_id')}: {report.get('error')}")
        return doc, report

    @staticmethod
    def set_param(doc, alias, value, sheet="Parameters"):
        sh = doc.getObject(sheet)
        sh.set(sh.getCellFromAlias(alias), "=" + repr(float(value)))
        doc.recompute()


class CauseContractTest(unittest.TestCase):
    """Every post-condition failure carries a stable cause code and
    recovery alternatives; the result contract and the model-facing text
    expose them."""

    @staticmethod
    def m(volume, solids=1, bbox=(0, 0, 0, 10, 10, 10), valid=True):
        return {"volume": float(volume), "solids": solids, "valid": valid, "bbox": list(bbox)}

    def _fail(self, op, before, after, **kw):
        with self.assertRaises(_buildcore.BuildError) as cm:
            _buildcore.check_postconditions(op, "f", before, after, **kw)
        return cm.exception

    def test_cause_codes_of_every_postcondition_failure(self):
        cases = [
            ("hole", self.m(1000), self.m(0, solids=0), {}, "no_solid"),
            ("pad", self.m(1000), self.m(1000, valid=False), {}, "invalid_solid"),
            ("boss", self.m(1000), self.m(900, solids=2), {}, "multiple_solids"),
            ("pad", self.m(1000), self.m(1000), {}, "added_nothing"),
            ("hole", self.m(1000), self.m(1000), {}, "cut_nothing"),
            ("fillet", self.m(1000), self.m(1000), {}, "changed_nothing"),
            ("hole", self.m(1000), self.m(900, bbox=(0, 0, 0, 12, 10, 10)), {}, "box_grew"),
            ("pad", self.m(1000), self.m(1100, bbox=(1, 0, 0, 10, 10, 10)), {}, "box_shrank"),
        ]
        for op, before, after, kw, cause in cases:
            with self.subTest(cause=cause):
                err = self._fail(op, before, after, **kw)
                self.assertEqual(err.cause, cause)
                self.assertTrue(err.alternatives, f"{cause} must offer alternatives")

    def test_every_cause_carries_its_alternatives_without_being_told(self):
        # a cause code alone must be enough: no raiser may forget the guidance
        for cause in _buildcore.RECOVERY_ALTERNATIVES:
            with self.subTest(cause=cause):
                err = _buildcore.BuildError("f", "boom", cause=cause)
                self.assertEqual(list(err.alternatives),
                                 list(_buildcore.RECOVERY_ALTERNATIVES[cause]),
                                 f"{cause} must carry its recovery alternatives")

    def test_the_recompute_failure_path_offers_alternatives(self):
        # regression: this raiser passes only the cause code
        err = _buildcore.BuildError("f", "recompute failed: x",
                                    cause=_buildcore.CAUSE_RECOMPUTE_FAILED)
        self.assertTrue(err.alternatives, "recompute_failed must offer recovery alternatives")

    def test_an_absent_cause_stays_empty(self):
        err = _buildcore.BuildError("f", "plain failure")
        self.assertEqual(err.cause, "")
        self.assertEqual(err.alternatives, ())

    def test_the_failure_result_carries_cause_and_alternatives(self):
        doc = App.newDocument("VRCause")
        self.addCleanup(lambda: App.closeDocument("VRCause") if "VRCause" in App.listDocuments() else None)
        plan = plate_plan("vr_cause", [hole(x=100)], hole_d=4)   # beside the plate: cuts nothing
        result = builder.apply_plan(doc, plan)
        self.assertFalse(result["success"])
        self.assertEqual(result["feature_id"], "hole1")
        self.assertEqual(result["cause"], "cut_nothing")
        self.assertTrue(result["alternatives"])
        self.assertIn("cut nothing", result["error"])     # the message text is unchanged

    def test_format_build_failure_renders_cause_and_alternatives(self):
        result = _buildcore.failure(_buildcore.STAGE_BUILD, "hole1 cut nothing: volume 1000 -> 1000",
                                    feature_id="hole1", cause="cut_nothing",
                                    alternatives=("flip the sketch plane's normal",
                                                  "move the sketch into the material"))
        text = builder.format_build_failure(result)
        self.assertIn("cut nothing", text)
        self.assertIn("Cause: cut_nothing", text)
        self.assertIn("Try:", text)
        self.assertIn("flip the sketch plane's normal", text)

    def test_a_result_without_a_cause_renders_as_before(self):
        result = _buildcore.failure(_buildcore.STAGE_VALIDATION,
                                    errors=["feature x: op 'pad' requires 'length'"])
        text = builder.format_build_failure(result)
        self.assertNotIn("cause:", text)
        self.assertIn("requires 'length'", text)


class SuffixRebuildTest(_DocTest):
    """``reapply_plan_feature(rebuild_later=true)``: the repair path for a
    feature others depend on — the dependents are rebuilt in place, the
    features before it keep their objects and bindings."""

    PLAN = None  # set below

    def test_the_refusal_names_the_suffix_alternative(self):
        doc, _r = self.build(plate_plan("vr_dep", [hole(x="hole_x")], hole_x=0, hole_d=4))
        result = builder.apply_feature(doc, plate_plan("vr_dep", [hole(x="hole_x")], hole_x=0, hole_d=4),
                                       "base_sk")
        self.assertFalse(result["success"])
        self.assertIn("later features exist", result["error"])
        self.assertIn("rebuild_later", result["error"])
        self.assertEqual(result.get("cause"), "dependents_exist")

    def test_rebuild_from_preserves_the_earlier_features_and_their_bindings(self):
        plan = plate_plan("vr_suf", [hole(x="hole_x"), {"id": "row", "op": "linear_pattern",
                                                        "of": "hole1", "count": "n_holes",
                                                        "spacing": "pitch", "direction": "x"}],
                          hole_x=-10, hole_d=4, n_holes=3, pitch=10)
        doc, _r = self.build(plan, "VRSuf")
        base_pad = doc.getObject("base_pad")
        sketch = doc.getObject("base_sk")
        self.assertTrue(base_pad.ExpressionEngine)      # bound to Parameters.thickness

        patched = copy.deepcopy(plan)
        patched["parameters"]["hole_d"] = param(6)      # a bigger hole: hole1 and its pattern rebuild
        result = builder.rebuild_from(doc, patched, "hole1")
        self.assertTrue(result["success"], result.get("error"))
        # the features BEFORE the rebuild point survive untouched and parametric...
        self.assertIs(doc.getObject("base_sk"), sketch)
        self.assertIs(doc.getObject("base_pad"), base_pad)
        self.assertTrue(base_pad.ExpressionEngine)
        # ...the dependents came back as fresh objects of the same plan...
        self.assertTrue(doc.getObject("row"), "the linear pattern must be rebuilt")
        self.assertIn("row", result["completed"])
        # ...and the geometry follows the patched parameter
        doc.recompute()
        self.assertAlmostEqual(doc.getObject("Body").Shape.Volume,
                               PLATE - 3 * math.pi * 9 * 10, delta=1.0)


class StaleDressingRepairTest(_DocTest):
    """A plan-built fillet whose edge link went stale is repaired from the
    rule stored on the object — and a set_parameter change that would stale
    one repairs it instead of refusing."""

    def plate_with_fillet(self, doc_name="VRFil"):
        plan = plate_plan("vr_fil", [{"id": "rounds", "op": "fillet", "size": "fil_r",
                                      "edges": "vertical"}], fil_r=3)
        doc, _r = self.build(plan, doc_name)
        return doc, plan

    def test_refresh_edge_rules_repairs_a_stale_link(self):
        doc, _plan = self.plate_with_fillet()
        fillet = doc.getObject("rounds")
        base = fillet.Base[0]
        fillet.Base = (base, ["Edge999"])              # a link to nothing: stale
        doc.recompute()
        self.assertIn("Invalid", fillet.State)
        repaired = builder.refresh_edge_rules(doc)
        self.assertIn("rounds", repaired)
        self.assertNotIn("Invalid", fillet.State)
        self.assertTrue(doc.getObject("Body").Shape.isValid())

    def test_set_parameter_repairs_a_repairable_fillet_instead_of_refusing(self):
        from oscar import tools_kit_params
        from oscar.toolctx import ToolContext, use_context

        doc, _plan = self.plate_with_fillet("VRFilSet")
        fillet = doc.getObject("rounds")
        fillet.Base = (fillet.Base[0], ["Edge999"])    # stale the link by hand
        doc.recompute()
        self.assertIn("Invalid", fillet.State)
        ctx = ToolContext()
        with use_context(ctx):
            text = tools_kit_params._set_parameter(doc.Name, "plate_len", 50)
        self.assertIn("50", text)
        self.assertNotIn("Invalid", fillet.State, "the stale fillet must have been repaired")
        self.assertAlmostEqual(doc.getObject("Body").Shape.Volume, 50 * 40 * 10 - 4 * (3 * 3 - math.pi * 9 / 4) * 10, delta=15)


class ValidationOverheadTest(_DocTest):
    """The per-feature validation (body metrics + post-conditions) must stay
    a small share of a real build: generous ceilings, real numbers."""

    def test_body_metrics_overhead_on_a_golden_body(self):
        with open(os.path.join(EXAMPLES, "plate_4holes.json"), encoding="utf-8") as f:
            plan = json.load(f)
        doc, _r = self.build(plan, "VRPerf")
        body = doc.getObject("Body")
        _buildcore.body_metrics(body)                  # warm any caches
        t0 = time.monotonic()
        for _ in range(5):
            _buildcore.body_metrics(body)
        per_call = (time.monotonic() - t0) / 5
        self.assertLess(per_call, 0.5, f"body_metrics costs {per_call:.3f}s per call")

    def test_postconditions_cost_nothing(self):
        m0 = {"volume": 1000.0, "solids": 1, "valid": True, "bbox": [0, 0, 0, 10, 10, 10]}
        m1 = {"volume": 1010.0, "solids": 1, "valid": True, "bbox": [0, 0, 0, 10, 10, 10]}
        t0 = time.monotonic()
        for _ in range(2000):
            _buildcore.check_postconditions("pad", "f", m0, m1)
        self.assertLess(time.monotonic() - t0, 0.5, "the pure post-condition math must be negligible")

    def test_a_golden_build_stays_within_the_validation_budget(self):
        with open(os.path.join(EXAMPLES, "plate_4holes.json"), encoding="utf-8") as f:
            plan = json.load(f)
        doc = self.new_doc("VRPerf2")
        t0 = time.monotonic()
        report = builder.apply_plan(doc, plan)
        elapsed = time.monotonic() - t0
        self.assertTrue(report["success"], report.get("error"))
        self.assertLess(elapsed, 30.0, f"the golden build took {elapsed:.1f}s")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
