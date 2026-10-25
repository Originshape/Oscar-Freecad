"""Schema-v2 plans built for real in FreeCAD: flange with a polar hole pattern,
L-bracket with fillets, hollow box by thickness, chamfer / mirror / linear
pattern / boss, multi-profile sketches, a two-part assembly with
parameter-driven placement, editability (change a parameter, the geometry
follows), idempotent re-apply with the data-loss guards, rollback, the
registry-driven post-conditions and the v1 regression.

Runs in console mode via tests/run.py (Sketcher / PartDesign are App-level)."""

import copy
import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

import FreeCAD as App  # noqa: E402

from oscar.plan import _buildcore, _partbuild, builder, schema

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "fixtures", "plans")


def param(value: float) -> dict:
    return {"value": float(value), "source": {"kind": "user_requirement", "ref": "unit test requirement"}}


def plate_plan(plan_id="v2p", features=(), **params) -> dict:
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
    def new_doc(self, name="V2Doc"):
        doc = App.newDocument(name)
        self.addCleanup(lambda n=doc.Name: App.closeDocument(n) if n in App.listDocuments() else None)
        return doc

    def build(self, plan, name="V2Doc", **kw):
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

    @staticmethod
    def shape(doc, body="Body"):
        return doc.getObject(body).Shape

    def volume(self, doc, body="Body"):
        return float(self.shape(doc, body).Volume)

    def assertSolid(self, doc, body="Body"):
        sh = self.shape(doc, body)
        self.assertTrue(sh.isValid())
        self.assertEqual(len(sh.Solids), 1)

    @staticmethod
    def info(report, fid):
        return next(f for f in report["features"] if f["id"] == fid)




class PatternsAndMirrorTest(_DocTest):
    def test_linear_pattern_with_count_and_spacing_parameters(self):
        plan = plate_plan("lin", [hole(x="hole_x"), {"id": "row", "op": "linear_pattern", "of": "hole1",
                                                     "count": "n_holes", "spacing": "pitch", "direction": "x"}],
                          hole_x=-10, hole_d=4, n_holes=3, pitch=10)
        doc, report = self.build(plan, "Lin")
        one = math.pi * 4 * 10
        self.assertAlmostEqual(self.volume(doc), PLATE - 3 * one, delta=0.5)
        self.set_param(doc, "n_holes", 4)
        self.assertAlmostEqual(self.volume(doc), PLATE - 4 * one, delta=0.5)
        self.set_param(doc, "pitch", 20)          # holes at -10, 10, 30 (edge), 50 (outside)
        self.assertAlmostEqual(self.volume(doc), PLATE - 2.5 * one, delta=0.5)   # -10, 10 and half of the one on the edge
        self.assertSolid(doc)

    def test_a_reversed_direction_walks_the_other_way(self):
        plan = plate_plan("linr", [hole(x="hole_x"), {"id": "row", "op": "linear_pattern", "of": "hole1",
                                                      "count": "n_holes", "spacing": "pitch", "direction": "-x"}],
                          hole_x=20, hole_d=4, n_holes=3, pitch=10)
        doc, _r = self.build(plan, "LinR")
        self.assertAlmostEqual(self.volume(doc), PLATE - 3 * math.pi * 4 * 10, delta=0.5)   # 20, 10, 0: all inside
        plan["features"][-1]["direction"] = "x"      # 20, 30 (half), 40 (outside)
        doc2, _r = self.build(plan, "LinR2")
        self.assertGreater(self.volume(doc2), self.volume(doc) + 1.0)

    def test_mirror_doubles_the_hole(self):
        plan = plate_plan("mir", [hole(x="hole_x"), {"id": "twin", "op": "mirror", "of": "hole1", "plane": "yz"}],
                          hole_x=20, hole_d=4)
        doc, _r = self.build(plan, "Mir")
        self.assertAlmostEqual(self.volume(doc), PLATE - 2 * math.pi * 4 * 10, delta=0.5)
        self.set_param(doc, "hole_x", 15)
        self.assertAlmostEqual(self.volume(doc), PLATE - 2 * math.pi * 4 * 10, delta=0.5)
        bb = self.shape(doc).BoundBox
        self.assertAlmostEqual(bb.XLength, 60.0, places=3)

    def test_a_pattern_of_a_pocket(self):
        plan = plate_plan("pk", [
            {"id": "slot_sk", "op": "sketch", "plane": {"frame_plane": "top"},
             "profile": {"shape": "rectangle", "width": "slot_w", "height": "slot_h", "center": ["slot_x", 0]}},
            {"id": "slot", "op": "pocket", "sketch": "slot_sk", "depth": "slot_depth"},
            {"id": "slots", "op": "polar_pattern", "of": "slot", "count": "n_slots", "angle": "sweep", "axis": "z"}],
            slot_w=4, slot_h=6, slot_x=-20, slot_depth=3, n_slots=2, sweep=180)
        doc, _r = self.build(plan, "PkPat")
        self.assertAlmostEqual(self.volume(doc), PLATE - 2 * 4 * 6 * 3, delta=0.5)

    def test_a_pattern_that_cuts_nothing_new_is_reported(self):
        plan = plate_plan("cn", [hole(x="hole_x"), {"id": "row", "op": "linear_pattern", "of": "hole1",
                                                    "count": "n_holes", "spacing": "pitch", "direction": "x"}],
                          hole_x=20, hole_d=4, n_holes=2, pitch=100)         # the copy lands far outside the plate
        doc = self.new_doc("CutNothing")
        result = builder.apply_plan(doc, plan)
        self.assertFalse(result["success"])
        self.assertEqual(result["feature_id"], "row")
        self.assertIn("linear_pattern cut nothing", result["error"])
        self.assertIn("volume", result["metrics"] and result["error"])






class ThicknessTest(_DocTest):
    @staticmethod
    def box(open_faces="top", direction=None) -> dict:
        feat = {"id": "shell", "op": "thickness", "wall": "wall_t", "open_faces": open_faces}
        if direction:
            feat["direction"] = direction
        return plate_plan("box", [feat], wall_t=2, thickness=30)

    def test_hollow_box_open_at_the_top(self):
        doc, report = self.build(self.box(), "Box")
        self.assertAlmostEqual(self.volume(doc), 60 * 40 * 30 - 56 * 36 * 28, delta=1.0)
        self.assertAlmostEqual(self.shape(doc).BoundBox.ZLength, 30.0, places=3)
        self.assertEqual(self.info(report, "shell")["info"]["open_faces"], ["Face6"])
        self.assertSolid(doc)

    def test_open_at_the_bottom_and_the_wall_is_editable(self):
        doc, _r = self.build(self.box("bottom"), "BoxB")
        self.assertAlmostEqual(self.volume(doc), 60 * 40 * 30 - 56 * 36 * 28, delta=1.0)
        self.set_param(doc, "wall_t", 3)
        self.assertAlmostEqual(self.volume(doc), 60 * 40 * 30 - 54 * 34 * 27, delta=1.0)
        self.set_param(doc, "plate_len", 80)
        self.assertAlmostEqual(self.volume(doc), 80 * 40 * 30 - 74 * 34 * 27, delta=1.5)
        self.assertSolid(doc)

    def test_outward_walls_grow_the_part(self):
        doc, _r = self.build(self.box("top", "outward"), "BoxOut")
        bb = self.shape(doc).BoundBox
        self.assertAlmostEqual(bb.XLength, 64.0, places=2)
        self.assertGreater(self.volume(doc), 0)

    def test_a_shell_thicker_than_the_part_is_reported(self):
        plan = self.box()
        plan["parameters"]["wall_t"] = param(25)
        doc = self.new_doc("BoxBad")
        result = builder.apply_plan(doc, plan)
        self.assertFalse(result["success"])
        self.assertEqual(result["feature_id"], "shell")
        self.assertEqual(result["completed"], ["base_sk", "base_pad"])


class BossTest(_DocTest):
    def test_a_boss_on_the_top_face(self):
        plan = plate_plan("bt", [{"id": "knob", "op": "boss", "face": "top", "length": "knob_h",
                                  "profile": {"shape": "circle", "diameter": "knob_d", "center": [5, 0]}}],
                          knob_d=10, knob_h=8)
        doc, report = self.build(plan, "BossTop")
        self.assertAlmostEqual(self.volume(doc), PLATE + math.pi * 25 * 8, delta=0.5)
        self.assertAlmostEqual(self.shape(doc).BoundBox.ZMax, 18.0, places=3)
        self.assertEqual(self.info(report, "knob")["info"]["sketch"], "knob_sketch")
        self.set_param(doc, "knob_h", 12)                       # editable
        self.assertAlmostEqual(self.shape(doc).BoundBox.ZMax, 22.0, places=3)
        self.set_param(doc, "thickness", 14)                    # the boss rides the face it sits on
        self.assertAlmostEqual(self.shape(doc).BoundBox.ZMax, 26.0, places=3)
        self.assertSolid(doc)

    def test_a_boss_on_the_bottom_face_grows_downwards(self):
        plan = plate_plan("bb", [{"id": "foot", "op": "boss", "face": "bottom", "length": "foot_h",
                                  "profile": {"shape": "circle", "diameter": "foot_d"}}], foot_d=10, foot_h=5)
        doc, _r = self.build(plan, "BossBottom")
        self.assertAlmostEqual(self.shape(doc).BoundBox.ZMin, -5.0, places=3)
        self.assertAlmostEqual(self.volume(doc), PLATE + math.pi * 25 * 5, delta=0.5)

    def test_a_boss_on_a_side_face_uses_y_up_coordinates(self):
        plan = plate_plan("bs", [{"id": "lug", "op": "boss", "face": "right", "length": "lug_l",
                                  "profile": {"shape": "rectangle", "width": "lug_w", "height": "lug_h",
                                              "center": [0, 5]}}], lug_w=10, lug_h=6, lug_l=5)
        doc, _r = self.build(plan, "BossRight")
        bb = self.shape(doc).BoundBox
        self.assertAlmostEqual(bb.XMax, 35.0, places=3)
        self.assertAlmostEqual(self.volume(doc), PLATE + 10 * 6 * 5, delta=0.5)

    def test_a_boss_that_misses_the_face_is_reported(self):
        plan = plate_plan("bm", [{"id": "lug", "op": "boss", "face": "top", "length": "lug_l",
                                  "profile": {"shape": "circle", "diameter": "lug_d", "center": [100, 0]}}],
                          lug_d=6, lug_l=5)
        doc = self.new_doc("BossMiss")
        result = builder.apply_plan(doc, plan)
        self.assertFalse(result["success"])
        self.assertEqual(result["feature_id"], "lug")
        self.assertIn("separate solids", result["error"])

    def test_a_face_the_part_does_not_have(self):
        plan = plate_plan("bn", [{"id": "dome", "op": "boss", "face": "front", "length": "lug_l",
                                  "profile": {"shape": "circle", "diameter": "lug_d"}}], lug_d=6, lug_l=5)
        plan["features"][0]["profile"] = {"shape": "circle", "diameter": "plate_len"}      # a disc: no flat face looks to -Y
        doc = self.new_doc("BossNoFace")
        result = builder.apply_plan(doc, plan)
        self.assertFalse(result["success"])
        self.assertEqual(result["feature_id"], "dome")
        self.assertIn("boss.face: no planar face faces 'front'", result["error"])






class PostconditionRegistryTest(unittest.TestCase):
    """``check_postconditions`` reads the kind from the op registry."""

    @staticmethod
    def m(volume, solids=1, bbox=(0, 0, 0, 10, 10, 10), valid=True):
        return {"volume": float(volume), "solids": solids, "valid": valid, "bbox": list(bbox)}

    def test_kinds_of_the_registry(self):
        self.assertEqual(_buildcore.post_kind("pad"), "additive")
        self.assertEqual(_buildcore.post_kind("hole"), "subtractive")
        self.assertEqual(_buildcore.post_kind("fillet"), "dress")
        self.assertEqual(_buildcore.post_kind("thickness"), "any")
        self.assertEqual(_buildcore.post_kind("linear_pattern"), "follow")
        self.assertEqual(_buildcore.post_kind("linear_pattern", "subtractive"), "subtractive")
        self.assertIsNone(_buildcore.post_kind("nonsense"))

    def test_a_fillet_must_change_the_volume_and_not_grow_the_box(self):
        ok = _buildcore.check_postconditions("fillet", "f", self.m(1000), self.m(990))
        self.assertAlmostEqual(ok["volume_delta"], -10.0)
        with self.assertRaises(_buildcore.BuildError) as cm:
            _buildcore.check_postconditions("fillet", "f", self.m(1000), self.m(1000))
        self.assertIn("fillet changed nothing", cm.exception.message)
        with self.assertRaises(_buildcore.BuildError) as cm:
            _buildcore.check_postconditions("chamfer", "f", self.m(1000), self.m(990, bbox=(0, 0, 0, 11, 10, 10)))
        self.assertIn("grew the bounding box", cm.exception.message)

    def test_a_dress_op_may_add_volume_when_it_rounds_a_concave_edge(self):
        _buildcore.check_postconditions("fillet", "f", self.m(1000), self.m(1004))

    def test_thickness_may_grow_the_box(self):
        _buildcore.check_postconditions("thickness", "t", self.m(1000), self.m(1500, bbox=(-2, -2, -2, 12, 12, 12)))
        with self.assertRaises(_buildcore.BuildError):
            _buildcore.check_postconditions("thickness", "t", self.m(1000), self.m(1000))

    def test_a_pattern_is_judged_like_its_original(self):
        _buildcore.check_postconditions("linear_pattern", "p", self.m(1000), self.m(900), kind="subtractive")
        with self.assertRaises(_buildcore.BuildError) as cm:
            _buildcore.check_postconditions("linear_pattern", "p", self.m(1000), self.m(1000), kind="subtractive")
        self.assertIn("linear_pattern cut nothing", cm.exception.message)
        _buildcore.check_postconditions("polar_pattern", "p", self.m(1000), self.m(1100, bbox=(0, 0, 0, 12, 10, 10)),
                                        kind="additive")

    def test_one_solid_and_validity_hold_for_every_kind(self):
        for op in ("fillet", "thickness", "boss", "linear_pattern"):
            with self.assertRaises(_buildcore.BuildError) as cm:
                _buildcore.check_postconditions(op, "x", self.m(1000), self.m(900, solids=2),
                                                kind="subtractive" if op == "linear_pattern" else None)
            self.assertIn("2 separate solids", cm.exception.message, op)
            with self.assertRaises(_buildcore.BuildError):
                _buildcore.check_postconditions(op, "x", self.m(1000), self.m(900, valid=False),
                                                kind="subtractive" if op == "linear_pattern" else None)


class V1RegressionTest(_DocTest):
    def example(self, name):
        with open(os.path.join(EXAMPLES, f"{name}.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_golden_v1_plans_build_identically_as_version_1_and_2(self):
        for name in ("plate_4holes", "l_bracket", "stepped_shaft"):
            with self.subTest(plan=name):
                plan1 = self.example(name)
                doc1, _r = self.build(plan1, f"V1{name[:4]}")
                plan2 = copy.deepcopy(plan1)
                plan2["schema_version"] = 2
                doc2, report2 = self.build(plan2, f"V2{name[:4]}")
                self.assertAlmostEqual(self.volume(doc1), self.volume(doc2), places=6)
                self.assertEqual(report2["body"], "Body")
                self.assertNotIn("parts", report2)
                self.assertEqual([f["op"] for f in report2["features"]], [f["op"] for f in plan1["features"]])

    def test_a_v1_plan_with_a_v2_op_is_refused_before_anything_is_built(self):
        plan = self.example("plate_4holes")
        plan["features"].append({"id": "round", "op": "fillet", "size": "plate_len"})
        doc = self.new_doc("V1Refuse")
        result = builder.apply_plan(doc, plan)
        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "validation")
        self.assertIn("needs schema_version 2", result["error"])
        self.assertEqual(len(doc.Objects), 0)

    def test_the_solid_builder_table_is_the_registry(self):
        from oscar.plan import _featurebuild
        self.assertEqual(set(_featurebuild.SOLID_BUILDERS),
                         {n for n, o in schema.OPS.items() if o.kind == "solid"})
        for name, spec in schema.OPS.items():
            if spec.kind == "solid":
                self.assertIs(spec.builder, _featurebuild.SOLID_BUILDERS[name])


class RegistryDrivenUnknownOpTest(_DocTest):
    def test_an_op_without_a_translator_is_refused_by_the_builder_not_the_validator(self):
        from unittest import mock

        from oscar.plan import _featurebuild
        plan = plate_plan("noop", [{"id": "round", "op": "fillet", "size": "round_r", "edges": "vertical"}],
                          round_r=1)
        doc = self.new_doc("NoTranslator")
        with mock.patch.dict(_featurebuild.SOLID_BUILDERS):
            del _featurebuild.SOLID_BUILDERS["fillet"]
            result = builder.apply_plan(doc, plan)
        self.assertFalse(result["success"])
        self.assertEqual(result["feature_id"], "round")
        self.assertIn("op 'fillet' unknown", result["error"])


class ToolWiringTest(_DocTest):
    """apply_feature_plan / reapply_plan_feature / feature_tree_status as the model sees them."""

    def test_the_descriptions_teach_schema_2(self):
        from oscar import tools
        plan_tool = tools.tool_meta("apply_feature_plan")
        for needle in ("schema_version 1 or 2", "fillet", "chamfer", "thickness", "linear_pattern", "polar_pattern",
                       "mirror", "boss", "profiles", "parts", "assembly", "source{kind", "edges_of_feature:<id>",
                       "never fused", '"schema_version":2'):
            self.assertIn(needle, plan_tool.description, needle)
        self.assertIn("nearest valid spelling", plan_tool.description)
        self.assertLess(len(tools.tool_meta("reapply_plan_feature").description), 600)
        self.assertIn("schema 1 or 2", tools.tool_meta("reapply_plan_feature").description)
        self.assertIn("multi-part", tools.tool_meta("feature_tree_status").description)

    def test_the_compact_example_in_the_hint_is_a_valid_plan(self):
        from oscar import tools
        text = tools._PLAN_SCHEMA_HINT
        example = json.loads(text[text.index('{"schema_version":2'):])
        ok, errors, _w = schema.validate(example)
        self.assertTrue(ok, errors)
        doc = self.new_doc("HintPlan")
        result = builder.apply_plan(doc, example)
        self.assertTrue(result["success"], result.get("error"))
        self.assertEqual([f["op"] for f in result["features"]], ["sketch", "pad", "fillet"])

    def test_a_typo_reaches_the_model_with_the_field_and_the_nearest_spelling(self):
        from oscar import tools
        doc = self.new_doc("WireTypo")
        plan = plate_plan("typo", [{"id": "f3", "op": "fillet", "size": "round_r", "edges": "vertcal"}], round_r=2)
        text, _image, is_error = tools.run_tool("apply_feature_plan", {"doc_name": doc.Name, "plan": plan})
        self.assertTrue(is_error)
        self.assertIn("feature f3: fillet.edges 'vertcal' not in", text)
        self.assertIn("did you mean 'vertical'?", text)
        self.assertNotIn(": None", text)
        self.assertEqual(len(doc.Objects), 0)                    # nothing was built from a rejected plan

    def test_a_build_failure_names_the_feature_and_what_was_completed(self):
        from oscar import tools
        doc = self.new_doc("WireFail")
        plan = plate_plan("big", [{"id": "round", "op": "fillet", "size": "round_r", "edges": "vertical"}], round_r=45)
        text, _image, is_error = tools.run_tool("apply_feature_plan", {"doc_name": doc.Name, "plan": plan})
        self.assertTrue(is_error)
        self.assertIn("Plan failed at 'round'", text)
        self.assertIn("Completed features: ['base_sk', 'base_pad']", text)

    def test_reapply_plan_feature_takes_a_v2_plan(self):
        from oscar import tools
        doc, _r = self.build(plate_plan("wf", [hole(x="hole_x"), {"id": "row", "op": "linear_pattern", "of": "hole1",
                                                                   "count": "n_holes", "spacing": "pitch"}],
                                        hole_x=-10, hole_d=4, n_holes=3, pitch=10), "WireFeature")
        plan = plate_plan("wf", [hole(x="hole_x"), {"id": "row", "op": "linear_pattern", "of": "hole1",
                                                    "count": "n_holes", "spacing": "pitch"}],
                          hole_x=-10, hole_d=4, n_holes=4, pitch=10)
        text, _image, is_error = tools.run_tool("reapply_plan_feature", {"doc_name": doc.Name, "plan": plan,
                                                                        "feature_id": "row"})
        self.assertFalse(is_error, text)
        self.assertEqual(doc.getObject("row").Occurrences, 4)


if __name__ == "__main__":
    unittest.main()
