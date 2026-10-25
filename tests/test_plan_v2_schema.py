"""FeaturePlan schema v2: the op registry, the v2 vocabulary (fillet, chamfer,
thickness, patterns, mirror, boss, multi-profile sketches), parts / assembly /
references, parameter sources, and the actionable error text (the repair loop
feeds these strings back to the model). Pure Python: no FreeCAD needed.

v1 plans must keep validating exactly as before (golden examples included)."""

import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

from oscar.plan import schema  # noqa: E402

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "fixtures", "plans")


def src(ref: str = "user requirement text", kind: str = "user_requirement") -> dict:
    return {"kind": kind, "ref": ref}


def param(value: float, **extra) -> dict:
    return {"value": float(value), "source": src(), **extra}


def plan_v2(features=None, **extra_params) -> dict:
    """A 60 x 40 x 10 plate (z 0..10) with a top plane, ready to be extended."""
    params = {"plate_len": param(60), "plate_wid": param(40), "thickness": param(10),
              "corner_r": param(3), "hole_d": param(5), "hole_x": param(20),
              "hole_count": param(4), "hole_pitch": param(10), "sweep": param(360),
              "wall": param(2)}
    params.update({k: param(v) for k, v in extra_params.items()})
    feats = [
        {"id": "base_sk", "op": "sketch", "plane": {"frame_plane": "bottom"},
         "profile": {"shape": "rectangle", "width": "plate_len", "height": "plate_wid"}},
        {"id": "base_pad", "op": "pad", "sketch": "base_sk", "length": "thickness"},
    ]
    return {
        "schema_version": 2, "id": "v2_plate", "part_class": "plate", "units": "mm",
        "target": "freecad",
        "frame": {"planes": {
            "bottom": {"origin": [0, 0, 0], "normal": [0, 0, 1], "x_dir": [1, 0, 0]},
            "top": {"origin": [0, 0, "thickness"], "normal": [0, 0, 1], "x_dir": [1, 0, 0]}}},
        "parameters": params,
        "features": feats + (features or []),
    }


def two_parts() -> dict:
    """A base and a lid, each its own part, the lid placed by parameters."""
    p = plan_v2()
    p["parameters"].update({"lid_gap": param(2), "lid_z": {"expr": "thickness + lid_gap"}})
    base = p.pop("features")
    p["parts"] = [
        {"id": "base", "features": base},
        {"id": "lid", "placement": {"z": "lid_z"}, "features": [
            {"id": "lid_sk", "op": "sketch", "plane": {"frame_plane": "bottom"},
             "profile": {"shape": "rectangle", "width": "plate_len", "height": "plate_wid"}},
            {"id": "lid_pad", "op": "pad", "sketch": "lid_sk", "length": "wall"}]},
    ]
    return p


def errors_of(plan: dict) -> list[str]:
    ok, errors, _warnings = schema.validate(plan)
    assert not ok, "the plan was expected to be invalid"
    return errors


def warnings_of(plan: dict) -> list[str]:
    ok, errors, warnings = schema.validate(plan)
    assert ok, errors
    return warnings


class RegistryTest(unittest.TestCase):
    def test_the_registry_holds_every_op_with_its_version(self):
        self.assertEqual(tuple(n for n, o in schema.OPS.items() if o.since == 1), schema.FEATURE_OPS)
        self.assertEqual(set(schema.FEATURE_OPS_V2),
                         {"fillet", "chamfer", "thickness", "linear_pattern", "polar_pattern", "mirror", "boss"})
        self.assertEqual(schema.ALL_FEATURE_OPS, schema.FEATURE_OPS + schema.FEATURE_OPS_V2)

    def test_v1_vocabulary_for_the_planner_prompt_is_unchanged(self):
        self.assertEqual(schema.FEATURE_OPS, ("sketch", "pad", "pocket", "revolution", "hole"))
        self.assertEqual(schema.SCHEMA_VERSION, 1)
        self.assertEqual(schema.SUPPORTED_VERSIONS, (1, 2))

    def test_every_spec_is_complete(self):
        for name, spec in schema.OPS.items():
            self.assertEqual(spec.name, name)
            self.assertIn(spec.kind, ("sketch", "solid"))
            self.assertIn("id", spec.keys)
            self.assertIn("op", spec.keys)
            self.assertTrue(set(spec.required) <= set(spec.keys), name)
            self.assertTrue(set(spec.dims) <= set(spec.keys) | {"x", "y"}, name)
            self.assertIn(spec.post, (None, "additive", "subtractive", "dress", "any", "follow"), name)
            self.assertEqual(spec.post is None, spec.kind == "sketch", name)

    def test_ops_for_a_version(self):
        self.assertEqual(schema.ops_for(1), schema.FEATURE_OPS)
        self.assertEqual(schema.ops_for(2), schema.ALL_FEATURE_OPS)
        self.assertEqual(schema.ops_for("garbage"), schema.FEATURE_OPS)

    def test_every_dimension_of_a_v2_op_is_range_checked(self):
        rules = {(n, f): r for n, o in schema.OPS.items() if o.since == 2 for f, r in o.dims.items()}
        self.assertEqual(rules[("polar_pattern", "angle")], "angle360")
        self.assertEqual(rules[("linear_pattern", "count")], "count")
        self.assertEqual(rules[("fillet", "size")], "positive")


class V2PlanValidatesTest(unittest.TestCase):
    def test_a_plain_plate_validates_without_evidence_when_it_has_sources(self):
        ok, errors, warnings = schema.validate(plan_v2())
        self.assertTrue(ok, errors)

    def test_every_new_op_validates(self):
        features = [
            {"id": "hole1", "op": "hole", "plane": {"frame_plane": "top"},
             "position": {"x": "hole_x", "y": 0}, "diameter": "hole_d", "through": True},
            {"id": "row", "op": "linear_pattern", "of": "hole1", "count": "hole_count",
             "spacing": "hole_pitch", "direction": "-y"},
            {"id": "ring", "op": "polar_pattern", "of": "hole1", "count": "hole_count",
             "angle": "sweep", "axis": "z"},
            {"id": "twin", "op": "mirror", "of": "hole1", "plane": "yz"},
            {"id": "knob", "op": "boss", "face": "top", "length": "wall",
             "profile": {"shape": "circle", "diameter": "hole_d"}},
            {"id": "round", "op": "fillet", "size": "corner_r", "edges": "vertical"},
            {"id": "bevel", "op": "chamfer", "size": "corner_r", "size2": "wall",
             "edges": "edges_of_feature:hole1"},
            {"id": "shell", "op": "thickness", "wall": "wall", "open_faces": "top",
             "direction": "outward"},
        ]
        ok, errors, _w = schema.validate(plan_v2(features))
        self.assertTrue(ok, errors)

    def test_multi_profile_sketch_validates_and_lists_every_dimension(self):
        plan = plan_v2()
        plan["features"][0] = {"id": "base_sk", "op": "sketch", "plane": {"frame_plane": "bottom"}, "profiles": [
            {"shape": "rectangle", "width": "plate_len", "height": "plate_wid"},
            {"shape": "circle", "diameter": "hole_d", "center": ["hole_x", 0]}]}
        ok, errors, _w = schema.validate(plan)
        self.assertTrue(ok, errors)
        roles = {(f, r, ref) for f, r, ref, _rule in schema.dimension_refs(plan)}
        self.assertIn(("base_sk", "sketch.circle.diameter", "hole_d"), roles)
        self.assertIn(("base_sk", "sketch.rectangle.width", "plate_len"), roles)

    def test_dimension_refs_of_the_new_ops_carry_their_rule(self):
        features = [
            {"id": "hole1", "op": "hole", "plane": {"frame_plane": "top"},
             "position": {"x": "hole_x", "y": 0}, "diameter": "hole_d", "through": True},
            {"id": "ring", "op": "polar_pattern", "of": "hole1", "count": "hole_count", "angle": "sweep"},
            {"id": "round", "op": "fillet", "size": "corner_r", "edges": "all_outer"}]
        got = {(f, r): rule for f, r, _ref, rule in schema.dimension_refs(plan_v2(features))}
        self.assertEqual(got[("ring", "polar_pattern.count")], "count")
        self.assertEqual(got[("ring", "polar_pattern.angle")], "angle360")
        self.assertEqual(got[("round", "fillet.size")], "positive")

    def test_features_of_a_parts_plan_are_listed_in_build_order(self):
        plan = two_parts()
        self.assertEqual([f["id"] for f in schema.all_features(plan)],
                         ["base_sk", "base_pad", "lid_sk", "lid_pad"])
        units = schema.plan_units(plan)
        self.assertEqual([u["part"] for u in units], ["base", "lid"])
        self.assertEqual(units[1]["placement"], {"z": "lid_z"})
        self.assertEqual([u["part"] for u in schema.plan_units(plan_v2())], [None])

    def test_a_two_part_plan_with_assembly_and_references_validates(self):
        plan = two_parts()
        plan["parts"][1].pop("placement")
        plan["assembly"] = {"placements": [{"name": "lid_closed", "part": "lid", "z": "lid_z"}]}
        plan["references"] = [{"id": "seat", "kind": "mating_faces", "a": {"part": "base", "face": "top"},
                               "b": {"part": "lid", "face": "bottom"}, "gap": "lid_gap"}]
        ok, errors, warnings = schema.validate(plan)
        self.assertTrue(ok, errors)
        unused = " ".join(w for w in warnings if "never referenced" in w)
        for name in ("lid_z", "lid_gap", "plate_len", "wall"):     # placements, references and features all count
            self.assertNotIn(f"'{name}'", unused)

    def test_a_placement_literal_draws_a_warning_but_zero_does_not(self):
        plan = two_parts()
        plan["parts"][1]["placement"] = {"z": "lid_z", "x": 5, "y": 0}
        warns = warnings_of(plan)
        self.assertTrue(any("literal placement.x 5" in w for w in warns), warns)
        self.assertFalse(any("placement.y" in w for w in warns), warns)


class ParameterSourceTest(unittest.TestCase):
    def test_every_source_kind_is_accepted(self):
        for kind in schema.SOURCE_KINDS:
            if kind == "derived":
                continue
            plan = plan_v2()
            plan["parameters"]["plate_len"] = {"value": 60, "source": src("a perfectly good reference", kind)}
            ok, errors, _w = schema.validate(plan)
            self.assertTrue(ok, (kind, errors))

    def test_v1_evidence_and_assumption_still_work_in_a_v2_plan(self):
        plan = plan_v2()
        plan["parameters"]["plate_len"] = {"value": 60, "assumption": "typical plate length for the test"}
        plan["parameters"]["plate_wid"] = {"value": 40, "evidence": {"op": "user_request", "detail": "the user asked for 40 mm"}}
        ok, errors, _w = schema.validate(plan)
        self.assertTrue(ok, errors)

    def test_a_derived_parameter_stays_a_formula_and_needs_no_source(self):
        plan = plan_v2(half={"expr": "plate_len / 2"} and 1)
        plan["parameters"]["half_len"] = {"expr": "plate_len / 2"}
        plan["parameters"]["half_len2"] = {"expr": "plate_len / 2", "source": {"kind": "derived"}}
        ok, errors, _w = schema.validate(plan)
        self.assertTrue(ok, errors)
        self.assertEqual(schema.parameter_values(plan)["half_len"], 30.0)

    def test_the_error_table(self):
        def mutate(f):
            def run(p):
                f(p["parameters"])
            return run
        cases = {
            "unjustified number": (mutate(lambda ps: ps.__setitem__("plate_len", {"value": 60})),
                                   "needs evidence ({op, detail}), an explicit assumption note or a source ({kind, ref})"),
            "kind typo": (mutate(lambda ps: ps["plate_len"].update(source={"kind": "datasheeet", "ref": "page 3 of the sheet"})),
                          "source.kind 'datasheeet' not in"),
            "kind suggestion": (mutate(lambda ps: ps["plate_len"].update(source={"kind": "datasheeet", "ref": "page 3 of the sheet"})),
                                "did you mean 'datasheet'?"),
            "ref missing": (mutate(lambda ps: ps["plate_len"].update(source={"kind": "datasheet"})),
                            "source.ref is required"),
            "ref too thin": (mutate(lambda ps: ps["plate_len"].update(source={"kind": "datasheet", "ref": "tbd"})),
                             "source.ref is too thin"),
            "not an object": (mutate(lambda ps: ps["plate_len"].update(source="datasheet")),
                              "source must be {'kind'"),
            "stray key": (mutate(lambda ps: ps["plate_len"]["source"].update(url="x")),
                          "unknown key 'url'"),
            "derived without expr": (mutate(lambda ps: ps["plate_len"].update(source={"kind": "derived"})),
                                     "source kind 'derived' needs an 'expr' formula"),
        }
        for label, (change, text) in cases.items():
            with self.subTest(case=label):
                plan = plan_v2()
                change(plan)
                self.assertIn(text, " | ".join(errors_of(plan)))


class VersionGateTest(unittest.TestCase):
    def test_a_v1_plan_may_not_use_a_v2_op_and_is_told_why(self):
        plan = plan_v2([{"id": "round", "op": "fillet", "size": "corner_r"}])
        plan["schema_version"] = 1
        text = " | ".join(errors_of(plan))
        self.assertIn("feature round: op 'fillet' needs schema_version 2 (this plan is version 1)", text)
        self.assertIn("('sketch', 'pad', 'pocket', 'revolution', 'hole')", text)

    def test_a_v1_plan_may_not_use_parts_profiles_assembly_or_references(self):
        plan = two_parts()
        plan["schema_version"] = 1
        plan["features"] = []
        text = " | ".join(errors_of(plan))
        self.assertIn("plan: 'parts' needs schema_version 2", text)
        plan = plan_v2()
        plan["schema_version"] = 1
        plan["features"][0] = {"id": "base_sk", "op": "sketch", "plane": {"frame_plane": "bottom"}, "profiles": [
            {"shape": "rectangle", "width": "plate_len", "height": "plate_wid"}]}
        self.assertIn("feature base_sk: 'profiles' needs schema_version 2", " | ".join(errors_of(plan)))

    def test_the_version_vocabulary(self):
        for bad in (0, 3, "2", True, None):
            plan = plan_v2()
            plan["schema_version"] = bad
            self.assertIn("schema_version must be 1 or 2", " | ".join(errors_of(plan)), bad)

    def test_golden_v1_examples_still_validate_as_version_1_and_as_2(self):
        for name in ("plate_4holes", "l_bracket", "stepped_shaft"):
            with open(os.path.join(EXAMPLES, f"{name}.json"), encoding="utf-8") as f:
                plan = json.load(f)
            ok, errors, _w = schema.validate(plan)
            self.assertTrue(ok, (name, errors))
            plan["schema_version"] = 2
            ok, errors, _w = schema.validate(plan)
            self.assertTrue(ok, (name, errors))


class ErrorTextTableTest(unittest.TestCase):
    """The strings the model sees. Each must name the feature and the field,
    list what is valid and suggest the nearest spelling where there is one."""

    HOLE = {"id": "hole1", "op": "hole", "plane": {"frame_plane": "top"},
            "position": {"x": "hole_x", "y": 0}, "diameter": "hole_d", "through": True}

    def case(self, feature, text, *, extra=None):
        plan = plan_v2([self.HOLE, feature] if extra is None else extra + [feature])
        joined = " | ".join(errors_of(plan))
        self.assertIn(text, joined)
        return joined

    def test_fillet_edge_rule_typo_lists_the_rules_and_suggests(self):
        joined = self.case({"id": "f3", "op": "fillet", "size": "corner_r", "edges": "vertcal"},
                           "feature f3: fillet.edges 'vertcal' not in ['all_outer', 'vertical', 'top_outer', "
                           "'bottom_outer', 'edges_of_feature:<id>']; did you mean 'vertical'?")
        self.assertNotIn("Edge", joined)

    def test_chamfer_edges_of_an_unknown_feature(self):
        self.case({"id": "c1", "op": "chamfer", "size": "corner_r", "edges": "edges_of_feature:hole2"},
                  "feature c1: chamfer.edges 'edges_of_feature:hole2' names feature 'hole2', which does not precede it")
        self.case({"id": "c1", "op": "chamfer", "size": "corner_r", "edges": "edges_of_feature:hole2"},
                  "did you mean 'hole1'?")

    def test_edge_rule_must_be_a_string(self):
        self.case({"id": "f1", "op": "fillet", "size": "corner_r", "edges": 7}, "fillet.edges 7 not in")

    def test_dimension_must_be_a_known_parameter(self):
        self.case({"id": "f1", "op": "fillet", "size": "corner_rr"},
                  "feature f1: fillet.size references unknown parameter 'corner_rr'; did you mean 'corner_r'?")
        self.case({"id": "f1", "op": "fillet", "size": 3},
                  "feature f1: fillet.size must be a parameter name (string), got 3")
        self.case({"id": "f1", "op": "fillet"}, "feature f1: fillet requires 'size' (a parameter name)")

    def test_non_positive_dimensions_name_the_role(self):
        plan = plan_v2([{"id": "f1", "op": "fillet", "size": "corner_r"}], corner_r=-1)
        self.assertIn("feature f1: parameter 'corner_r' (-1) is used as fillet.size and must be positive (> 0)",
                      " | ".join(errors_of(plan)))

    def test_unknown_key_on_a_v2_op(self):
        self.case({"id": "f1", "op": "fillet", "size": "corner_r", "radius": "corner_r"},
                  "feature f1 (fillet): unknown key 'radius'")

    def test_thickness_open_faces(self):
        self.case({"id": "t1", "op": "thickness", "wall": "wall", "open_faces": "tpo"},
                  "feature t1: thickness.open_faces 'tpo' not in ['top', 'bottom']; did you mean 'top'?")
        self.case({"id": "t1", "op": "thickness", "wall": "wall", "open_faces": "none"},
                  "thickness.open_faces 'none' (a sealed hollow) is not supported")
        self.case({"id": "t1", "op": "thickness", "wall": "wall", "direction": "in"},
                  "feature t1: thickness.direction 'in' not in ['inward', 'outward']")

    def test_pattern_original_must_be_an_earlier_pattern_able_feature(self):
        self.case({"id": "p1", "op": "linear_pattern", "of": "hole2", "count": "hole_count", "spacing": "hole_pitch"},
                  "feature p1: linear_pattern.of 'hole2' is not an earlier feature of this part; did you mean 'hole1'?")
        self.case({"id": "p1", "op": "linear_pattern", "of": "base_sk", "count": "hole_count", "spacing": "hole_pitch"},
                  "feature p1: linear_pattern.of 'base_sk' is a 'sketch' feature; a linear_pattern repeats one of")
        self.case({"id": "p1", "op": "mirror", "plane": "yz"}, "feature p1: mirror requires 'of'")

    def test_pattern_direction_axis_and_plane_vocabulary(self):
        self.case({"id": "p1", "op": "linear_pattern", "of": "hole1", "count": "hole_count", "spacing": "hole_pitch",
                   "direction": "xx"}, "linear_pattern.direction 'xx' not in ['x', 'y', 'z', '-x', '-y', '-z']")
        self.case({"id": "p1", "op": "polar_pattern", "of": "hole1", "count": "hole_count", "angle": "sweep",
                   "axis": "w"}, "polar_pattern.axis 'w' not in ['x', 'y', 'z']")
        self.case({"id": "p1", "op": "mirror", "of": "hole1", "plane": "yx"},
                  "mirror.plane 'yx' not in ['xy', 'xz', 'yz']")

    def test_pattern_count_must_be_a_whole_number_of_two_or_more(self):
        for value in (1, 2.5, 600):
            with self.subTest(count=value):
                plan = plan_v2([self.HOLE, {"id": "p1", "op": "linear_pattern", "of": "hole1",
                                            "count": "hole_count", "spacing": "hole_pitch"}], hole_count=value)
                self.assertIn("is used as linear_pattern.count and must be a whole number from 2 to 500",
                              " | ".join(errors_of(plan)))

    def test_polar_angle_is_range_checked(self):
        plan = plan_v2([self.HOLE, {"id": "p1", "op": "polar_pattern", "of": "hole1", "count": "hole_count",
                                    "angle": "sweep"}], sweep=400)
        self.assertIn("used as polar_pattern.angle and must be an angle in (0, 360] degrees",
                      " | ".join(errors_of(plan)))

    def test_boss_face_and_profile(self):
        self.case({"id": "b1", "op": "boss", "face": "tpo", "length": "wall",
                   "profile": {"shape": "circle", "diameter": "hole_d"}},
                  "feature b1: boss.face 'tpo' not in ['top', 'bottom', 'front', 'back', 'left', 'right']; "
                  "did you mean 'top'?")
        self.case({"id": "b1", "op": "boss", "face": "top", "length": "wall"}, "feature b1: sketch needs a profile object")
        self.case({"id": "b1", "op": "boss", "face": "top", "length": "wall",
                   "profile": {"shape": "circle", "diameter": "hole_d"}, "profiles": []},
                  "feature b1: gives both 'profile' and 'profiles'")

    def test_boss_sketch_name_collision(self):
        plan = plan_v2([{"id": "b1", "op": "boss", "face": "top", "length": "wall",
                         "profile": {"shape": "circle", "diameter": "hole_d"}},
                        {"id": "b1_sketch", "op": "fillet", "size": "corner_r"}])
        self.assertIn("collides with the sketch the builder creates for the boss 'b1'", " | ".join(errors_of(plan)))

    def test_a_dressing_feature_needs_something_before_it(self):
        plan = plan_v2()
        plan["features"] = [{"id": "f1", "op": "fillet", "size": "corner_r"}]
        self.assertIn("feature f1: fillet before any sketch", " | ".join(errors_of(plan)))

    def test_profiles_list_errors(self):
        for profiles, text in (([], "profiles must be a non-empty list of profile objects"),
                               (["circle"], "profiles[0] must be a profile object"),
                               ([{"shape": "blob"}], "feature base_sk (profiles[0]): profile shape 'blob' not in")):
            plan = plan_v2()
            plan["features"][0] = {"id": "base_sk", "op": "sketch", "plane": {"frame_plane": "bottom"},
                                   "profiles": profiles}
            self.assertIn(text, " | ".join(errors_of(plan)))

    def test_unknown_op_suggests_the_nearest_v2_op(self):
        plan = plan_v2([{"id": "f1", "op": "filet", "size": "corner_r"}])
        text = " | ".join(errors_of(plan))
        self.assertIn("feature f1: op 'filet' not in bounded set", text)
        self.assertIn("did you mean 'fillet'?", text)


class PartsTest(unittest.TestCase):
    def test_parts_and_features_are_exclusive(self):
        plan = two_parts()
        plan["features"] = [{"id": "x", "op": "fillet", "size": "corner_r"}]
        self.assertIn("plan gives both 'features' and 'parts'", " | ".join(errors_of(plan)))

    def test_parts_must_be_a_non_empty_list_of_objects(self):
        plan = two_parts()
        plan["parts"] = []
        self.assertIn("parts must be a non-empty list", " | ".join(errors_of(plan)))
        plan["parts"] = ["base"]
        self.assertIn("part #0: must be an object", " | ".join(errors_of(plan)))

    def test_part_id_rules(self):
        plan = two_parts()
        plan["parts"][1]["id"] = "base"
        self.assertIn("part base: duplicate id", " | ".join(errors_of(plan)))
        plan = two_parts()
        plan["parts"][0]["id"] = "1st"
        self.assertIn("part #0: id must be an identifier", " | ".join(errors_of(plan)))

    def test_feature_ids_are_unique_across_parts(self):
        plan = two_parts()
        plan["parts"][1]["features"][0]["id"] = "base_sk"
        self.assertIn("feature base_sk: duplicate id", " | ".join(errors_of(plan)))

    def test_a_feature_may_not_reference_a_sketch_of_another_part(self):
        plan = two_parts()
        plan["parts"][1]["features"][1]["sketch"] = "base_sk"
        self.assertIn("feature lid_pad: references sketch 'base_sk' which does not precede it",
                      " | ".join(errors_of(plan)))

    def test_every_part_starts_with_a_sketch(self):
        plan = two_parts()
        plan["parts"][1]["features"].insert(0, {"id": "lone", "op": "fillet", "size": "corner_r"})
        self.assertIn("feature lone: fillet before any sketch", " | ".join(errors_of(plan)))

    def test_part_keys_and_body_names(self):
        plan = two_parts()
        plan["parts"][0]["colour"] = "red"
        self.assertIn("part base: unknown key 'colour'", " | ".join(errors_of(plan)))
        plan = two_parts()
        plan["parts"][0]["body"] = plan["parts"][1]["body"] = "Same"
        self.assertIn("body name 'Same' is used by another part", " | ".join(errors_of(plan)))
        plan = two_parts()
        plan["parts"][0]["body"] = "bad name"
        self.assertIn("body must be an object name", " | ".join(errors_of(plan)))

    def test_placement_errors_name_the_part_and_key(self):
        plan = two_parts()
        plan["parts"][1]["placement"] = {"z": "lid_zz"}
        self.assertIn("part lid: placement.z references unknown parameter 'lid_zz'; did you mean 'lid_z'?",
                      " | ".join(errors_of(plan)))
        plan["parts"][1]["placement"] = {"q": "lid_z"}
        self.assertIn("part lid placement: unknown key 'q'", " | ".join(errors_of(plan)))
        plan["parts"][1]["placement"] = {"z": [1]}
        self.assertIn("placement.z must be a parameter name (or a number)", " | ".join(errors_of(plan)))
        plan["parts"][1]["placement"] = "up"
        self.assertIn("placement must be an object", " | ".join(errors_of(plan)))

    def test_an_empty_part_is_rejected(self):
        plan = two_parts()
        plan["parts"][1]["features"] = []
        self.assertIn("part lid: features must be a non-empty ordered list", " | ".join(errors_of(plan)))


class AssemblyReferencesTest(unittest.TestCase):
    def assembled(self) -> dict:
        plan = two_parts()
        plan["parts"][1].pop("placement")
        plan["assembly"] = {"placements": [{"name": "lid_closed", "part": "lid", "z": "lid_z"}]}
        return plan

    def test_assembly_errors(self):
        cases = {
            "unknown part": (lambda a: a["placements"][0].update(part="lidd"),
                             "assembly placement lid_closed: part 'lidd' is not in parts; did you mean 'lid'?"),
            "duplicate": (lambda a: a["placements"].append(dict(a["placements"][0])),
                          "assembly placement lid_closed: duplicate name"),
            "bad name": (lambda a: a["placements"][0].update(name="1st"),
                         "assembly.placements[0]: name must be an identifier"),
            "unknown param": (lambda a: a["placements"][0].update(z="nope"),
                              "placement.z references unknown parameter 'nope'"),
            "empty": (lambda a: a.update(placements=[]), "assembly.placements must be a non-empty list"),
            "stray key": (lambda a: a.update(colour="x"), "assembly: unknown key 'colour'"),
        }
        for label, (change, text) in cases.items():
            with self.subTest(case=label):
                plan = self.assembled()
                change(plan["assembly"])
                self.assertIn(text, " | ".join(errors_of(plan)))

    def test_a_part_is_placed_once(self):
        plan = self.assembled()
        plan["parts"][1]["placement"] = {"z": "lid_z"}
        self.assertIn("part 'lid' has its own 'placement'; keep one of the two", " | ".join(errors_of(plan)))
        plan = self.assembled()
        plan["assembly"]["placements"].append({"name": "again", "part": "lid", "z": "lid_z"})
        self.assertIn("part 'lid' is already placed by another assembly placement", " | ".join(errors_of(plan)))

    def test_assembly_needs_parts(self):
        plan = plan_v2()
        plan["assembly"] = {"placements": [{"name": "a", "part": "base"}]}
        self.assertIn("assembly / references need parts[]", " | ".join(errors_of(plan)))

    def test_reference_errors(self):
        good = {"id": "seat", "kind": "mating_faces", "a": {"part": "base", "face": "top"},
                "b": {"part": "lid", "face": "bottom"}, "gap": "lid_gap"}
        cases = {
            "kind": ({"kind": "glued"}, "reference seat: kind 'glued' not in ['mating_faces']"),
            "face": ({"b": {"part": "lid", "face": "botom"}}, "reference seat: b.face 'botom' not in"),
            "face hint": ({"b": {"part": "lid", "face": "botom"}}, "did you mean 'bottom'?"),
            "part": ({"a": {"part": "bas", "face": "top"}}, "reference seat: a.part 'bas' is not in parts"),
            "same part": ({"b": {"part": "base", "face": "bottom"}}, "a and b are the same part ('base')"),
            "gap": ({"gap": "lid_gp"}, "gap must be a known parameter name, got 'lid_gp'; did you mean 'lid_gap'?"),
            "shape": ({"a": "base.top"}, "a must be {'part': <part id>, 'face'"),
            "key": ({"tolerance": 1}, "reference seat: unknown key 'tolerance'"),
        }
        for label, (change, text) in cases.items():
            with self.subTest(case=label):
                plan = two_parts()
                plan["references"] = [{**good, **change}]
                self.assertIn(text, " | ".join(errors_of(plan)))

    def test_references_must_be_a_list(self):
        plan = two_parts()
        plan["references"] = {"id": "x"}
        self.assertIn("references must be a non-empty list", " | ".join(errors_of(plan)))


class NeverRaisesTest(unittest.TestCase):
    def test_garbage_in_every_new_place_becomes_errors_not_exceptions(self):
        junk = (None, 5, "x", [], {}, [1], {"a": 1})
        for place in ("parts", "assembly", "references"):
            for value in junk:
                plan = two_parts()
                plan[place] = copy.deepcopy(value)
                ok, errors, _w = schema.validate(plan)
                self.assertTrue(isinstance(errors, list), (place, value))
        for feature in ({"id": "f", "op": "fillet", "size": "corner_r", "edges": None},
                        {"id": "f", "op": "polar_pattern", "of": None, "count": [], "angle": {}},
                        {"id": "f", "op": "boss", "face": 3, "length": None, "profile": 4},
                        {"id": "f", "op": "thickness", "wall": None, "open_faces": []}):
            ok, errors, _w = schema.validate(plan_v2([feature]))
            self.assertFalse(ok, feature)
            self.assertTrue(all(isinstance(e, str) for e in errors))

    def test_a_v2_plan_without_features_or_parts_is_told_what_is_missing(self):
        plan = plan_v2()
        del plan["features"]
        ok, errors, _w = schema.validate(plan)
        self.assertFalse(ok)
        self.assertIn("pass the WHOLE plan object", errors[0])
        self.assertIn("parts[]", errors[0])


if __name__ == "__main__":
    unittest.main()
