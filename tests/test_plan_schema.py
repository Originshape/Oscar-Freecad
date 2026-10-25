"""FeaturePlan schema tests: golden examples validate, malformed plans
are rejected with actionable errors, and the audit's validator gaps stay
closed — reference closure, required fields per op, positivity by role,
unknown keys with a nearest-key suggestion, derived parameters, spreadsheet
alias rules, hole counterbore / countersink spellings.

Pure Python: no FreeCAD needed."""

import copy
import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.plan import schema

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "fixtures", "plans")


def _load(name: str) -> dict:
    with open(os.path.join(EXAMPLES, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def _p(value) -> dict:
    return {"value": float(value), "assumption": "unit test fixture value"}


def _plate() -> dict:
    """A small valid plan: base sketch + pad, plus a top plane."""
    return {
        "schema_version": 1, "id": "schema_plate", "part_class": "plate",
        "units": "mm", "target": "freecad",
        "frame": {"planes": {
            "bottom": {"origin": [0, 0, 0], "normal": [0, 0, 1], "x_dir": [1, 0, 0]},
            "top": {"origin": [0, 0, "thickness"], "normal": [0, 0, 1]}},
            "axes": {"x_axis": {"direction": [1, 0, 0], "point": [0, 0, 0]}}},
        "parameters": {"plate_len": _p(60), "plate_wid": _p(40), "thickness": _p(10),
                       "hole_dia": _p(4)},
        "features": [
            {"id": "base_sketch", "op": "sketch", "plane": {"frame_plane": "bottom"},
             "profile": {"shape": "rectangle", "width": "plate_len",
                         "height": "plate_wid", "center": [0, 0]}},
            {"id": "base_pad", "op": "pad", "sketch": "base_sketch", "length": "thickness"},
        ],
    }


def _errors(plan) -> list[str]:
    ok, errors, _warnings = schema.validate(plan)
    assert ok == (not errors)
    return errors


def _joined(plan) -> str:
    return "\n".join(_errors(plan))


class SchemaTest(unittest.TestCase):
    def test_golden_plans_validate(self):
        for name in ("plate_4holes", "l_bracket", "stepped_shaft"):
            with self.subTest(plan=name):
                ok, errors, warnings = schema.validate(_load(name))
                self.assertTrue(ok, f"{name}: {errors}")
                self.assertEqual(warnings, [], f"{name}: {warnings}")

    def test_minimal_plan_validates(self):
        ok, errors, warnings = schema.validate(_plate())
        self.assertTrue(ok, errors)
        self.assertEqual(warnings, ["parameters never referenced by features: ['hole_dia']"])

    def test_malformed_plan_rejected(self):
        plan = _load("plate_4holes")
        # unevidenced parameter
        plan["parameters"]["hole_dia"] = {"value": 11.0}
        # literal dimension instead of parameter binding
        plan["features"][1]["length"] = 8.0
        # unknown op
        plan["features"].append({"id": "weird", "op": "loft"})
        # sketch reference to a later feature
        plan["features"].append({"id": "bad_pad", "op": "pad",
                                 "sketch": "ghost", "length": "thickness"})
        ok, errors, _ = schema.validate(plan)
        self.assertFalse(ok)
        joined = "\n".join(errors)
        self.assertIn("hole_dia", joined)
        self.assertIn("evidence", joined)
        self.assertIn("base_pad", joined)
        self.assertIn("parameter name", joined)
        self.assertIn("loft", joined)
        self.assertIn("ghost", joined)

    def test_malformed_plan_shapes_rejected_cleanly(self):
        # the demo-run failure: a plan fragment (list) crashed validation
        # with AttributeError instead of an actionable error
        ok, errors, _w = schema.validate(["not", "a", "plan"])
        self.assertFalse(ok)
        self.assertIn("JSON object", errors[0])
        fragment = dict(_load("plate_4holes"))
        fragment["parameters"] = list(fragment["parameters"].values())
        ok, errors, _w = schema.validate(fragment)
        self.assertFalse(ok)
        self.assertIn("parameters must be an object", errors[0])
        fragment = dict(_load("plate_4holes"))
        fragment["features"] = {"id": "x"}
        ok, errors, _w = schema.validate(fragment)
        self.assertFalse(ok)
        self.assertIn("features must be an ordered list", errors[0])

    def test_validate_never_raises_on_hostile_structures(self):
        hostile = [
            None, 7, "plan", [], {},
            {"parameters": {}, "features": []},
            {**_plate(), "features": [None, 3, "x", []]},
            {**_plate(), "frame": [1]},
            {**_plate(), "frame": {"planes": [1], "axes": 5}},
            {**_plate(), "parameters": {"a_b_c": None, "x_y_z": {"value": "str"}}},
        ]
        for plan in hostile:
            with self.subTest(plan=repr(plan)[:40]):
                ok, errors, warnings = schema.validate(plan)
                self.assertFalse(ok)
                self.assertTrue(errors and all(isinstance(e, str) for e in errors))
                self.assertIsInstance(warnings, list)

    def test_dimensional_values_must_bind_params(self):
        plan = _load("l_bracket")
        plan["features"][0]["profile"]["segments"][0]["to"] = [70.0, 0]
        ok, errors, _ = schema.validate(plan)
        # literal coords in profiles are allowed (positions), only named
        # dimensional fields are strict — the plan still validates
        self.assertTrue(ok, errors)


    def test_sketch_basis_orthonormal(self):
        plan = _load("plate_4holes")
        basis = schema.sketch_basis(plan, {"frame_plane": "bottom"})
        n, x, y = basis["normal"], basis["x_dir"], basis["y_dir"]
        dot = lambda a, b: sum(i * j for i, j in zip(a, b))
        self.assertAlmostEqual(dot(n, x), 0.0, places=6)
        self.assertAlmostEqual(dot(n, y), 0.0, places=6)
        self.assertAlmostEqual(dot(x, y), 0.0, places=6)
        self.assertAlmostEqual(sum(c * c for c in n), 1.0, places=6)
        self.assertEqual(basis["origin"], [50, 30, 0])

    def test_explicit_plane_without_x_dir_gets_stable_default(self):
        basis = schema.sketch_basis({"frame": {"planes": {}}},
                                    {"origin": [0, 0, 0],
                                     "normal": [0, 1, 0]})
        self.assertAlmostEqual(sum(c * c for c in basis["x_dir"]), 1.0)

    def test_sketch_basis_binds_origin_parameters(self):
        basis = schema.sketch_basis(_plate(), {"frame_plane": "top"})
        self.assertEqual(basis["origin"], [0.0, 0.0, 10.0])
        self.assertEqual(basis["origin_refs"], [None, None, "thickness"])

    def test_plan_level_fields_and_vocabulary(self):
        for field, value, text in (
                ("schema_version", 3, "schema_version must be 1"),   # migration: 2 is the v2 schema now (see PlanV2SchemaTest)
                ("id", "", "id must be a non-empty string"),
                ("units", "inch", "units 'inch' not supported"),
                ("target", "fusion", "target 'fusion' not supported")):
            with self.subTest(field=field):
                plan = _plate()
                plan[field] = value
                self.assertIn(text, _joined(plan))
        plan = _plate()
        plan["bonus"] = 1
        ok, _e, warnings = schema.validate(plan)
        self.assertTrue(ok)
        self.assertTrue(any("unknown key 'bonus'" in w for w in warnings), warnings)


class ReferenceClosureTest(unittest.TestCase):
    def test_every_parameter_reference_must_exist(self):
        cases = {
            "pad length": (lambda p: p["features"][1].update(length="thicknes"),
                           "references unknown parameter 'thicknes'"),
            "profile width": (lambda p: p["features"][0]["profile"].update(width="plat_len"),
                              "profile.width references unknown parameter 'plat_len'"),
            "profile center": (lambda p: p["features"][0]["profile"].update(center=["nope", 0]),
                               "profile.center[0] references unknown parameter 'nope'"),
            "plane origin": (lambda p: p["frame"]["planes"]["top"].update(origin=[0, 0, "thickness2"]),
                             "must be a number or known parameter name"),
        }
        for label, (mutate, text) in cases.items():
            with self.subTest(case=label):
                plan = _plate()
                mutate(plan)
                self.assertIn(text, _joined(plan))

    def test_unknown_reference_suggests_the_nearest_name(self):
        plan = _plate()
        plan["features"][1]["length"] = "thicknes"
        self.assertIn("did you mean 'thickness'?", _joined(plan))

    def test_hole_references_are_closed_too(self):
        plan = _plate()
        plan["features"].append({
            "id": "hole", "op": "hole", "plane": {"frame_plane": "top"},
            "position": {"x": "px", "y": 0}, "diameter": "hole_diaa", "through": True})
        joined = _joined(plan)
        self.assertIn("hole.diameter references unknown parameter 'hole_diaa'", joined)
        self.assertIn("position.x references unknown parameter 'px'", joined)
        self.assertIn("did you mean 'hole_dia'?", joined)

    def test_cut_references_are_closed(self):
        plan = _plate()
        plan["features"].append({
            "id": "hole", "op": "hole", "plane": {"frame_plane": "top"},
            "position": {"x": 0, "y": 0}, "diameter": "hole_dia", "through": True,
            "counterbore": {"cb_diameter": "ghost_d", "cb_depth": "thickness"}})
        self.assertIn("counterbore diameter references unknown parameter 'ghost_d'",
                      _joined(plan))

    def test_plane_and_axis_references(self):
        plan = _plate()
        plan["features"][0]["plane"] = {"frame_plane": "side"}
        self.assertIn("unknown frame plane 'side' (have: ['bottom', 'top'])", _joined(plan))
        plan = _plate()
        plan["features"] += [
            {"id": "prof", "op": "sketch", "plane": {"frame_plane": "top"},
             "profile": {"shape": "circle", "diameter": "hole_dia"}},
            {"id": "rev", "op": "revolution", "sketch": "prof",
             "axis": {"frame_axis": "y_axis"}}]
        self.assertIn("axis.frame_axis 'y_axis' is not in frame.axes (have: ['x_axis'])",
                      _joined(plan))

    def test_sketch_reference_must_precede(self):
        plan = _plate()
        plan["features"].insert(1, {"id": "early_pad", "op": "pad",
                                    "sketch": "base_sketch", "length": "thickness"})
        plan["features"][0], plan["features"][1] = plan["features"][1], plan["features"][0]
        self.assertIn("pad before any sketch", _joined(plan))
        plan = _plate()
        plan["features"][1]["sketch"] = "later_sketch"
        plan["features"].append({"id": "later_sketch", "op": "sketch",
                                 "plane": {"frame_plane": "top"},
                                 "profile": {"shape": "circle", "diameter": "hole_dia"}})
        self.assertIn("references sketch 'later_sketch' which does not precede it",
                      _joined(plan))

    def test_unused_parameters_are_a_warning_not_an_error(self):
        ok, _e, warnings = schema.validate(_plate())
        self.assertTrue(ok)
        self.assertIn("hole_dia", warnings[0])


class RequiredFieldsTest(unittest.TestCase):
    def test_each_op_states_what_it_requires(self):
        base = _plate()
        cases = {
            "pad needs length": ({"id": "p2", "op": "pad", "sketch": "base_sketch"},
                                 "pad requires 'length'"),
            "pocket needs depth or through": ({"id": "k", "op": "pocket", "sketch": "base_sketch"},
                                              "pocket requires 'depth'"),
            "pocket not both": ({"id": "k", "op": "pocket", "sketch": "base_sketch",
                                 "through": True, "depth": "thickness"},
                                "through-pocket must not also set depth"),
            "revolution needs an axis": ({"id": "r", "op": "revolution", "sketch": "base_sketch"},
                                         "revolution needs axis"),
            "hole needs a diameter": ({"id": "h", "op": "hole", "plane": {"frame_plane": "top"},
                                       "position": {"x": 0, "y": 0}, "through": True},
                                      "hole requires 'diameter'"),
            "hole needs a position": ({"id": "h", "op": "hole", "plane": {"frame_plane": "top"},
                                       "diameter": "hole_dia", "through": True},
                                      "hole needs position {x, y}"),
            "hole needs depth or through": ({"id": "h", "op": "hole", "plane": {"frame_plane": "top"},
                                             "position": {"x": 0, "y": 0}, "diameter": "hole_dia"},
                                            "hole requires 'through': true or a 'depth'"),
            "hole not both": ({"id": "h", "op": "hole", "plane": {"frame_plane": "top"},
                               "position": {"x": 0, "y": 0}, "diameter": "hole_dia",
                               "through": True, "depth": "thickness"},
                              "through-hole must not also set depth"),
            "hole needs a plane": ({"id": "h", "op": "hole", "position": {"x": 0, "y": 0},
                                    "diameter": "hole_dia", "through": True},
                                   "plane must be an object"),
        }
        for label, (feat, text) in cases.items():
            with self.subTest(case=label):
                plan = copy.deepcopy(base)
                plan["features"].append(feat)
                self.assertIn(text, _joined(plan))

    def test_sketch_needs_profile_plane_and_known_shape(self):
        plan = _plate()
        plan["features"][0].pop("profile")
        self.assertIn("sketch needs a profile object", _joined(plan))
        plan = _plate()
        plan["features"][0].pop("plane")
        self.assertIn("plane must be an object", _joined(plan))
        plan = _plate()
        plan["features"][0]["profile"] = {"shape": "ellipse"}
        self.assertIn("profile shape 'ellipse' not in", _joined(plan))

    def test_profile_dimensions_are_required_names(self):
        for shape, key in (("rectangle", "height"), ("circle", "diameter"), ("slot", "length")):
            with self.subTest(shape=shape):
                plan = _plate()
                profile = {"rectangle": {"shape": "rectangle", "width": "plate_len"},
                           "circle": {"shape": "circle"},
                           "slot": {"shape": "slot", "width": "hole_dia"}}[shape]
                plan["features"][0]["profile"] = profile
                self.assertIn(f"profile.{key} must be a parameter name", _joined(plan))

    def test_parameters_need_evidence_or_assumption(self):
        plan = _plate()
        plan["parameters"]["thickness"] = {"value": 10.0}
        self.assertIn("parameter 'thickness': needs evidence", _joined(plan))
        plan["parameters"]["thickness"] = {"value": 10.0, "evidence": {"op": "user_request"}}
        self.assertIn("evidence must be", _joined(plan))

    def test_thin_evidence_and_assumptions_are_rejected(self):
        thin = ("tbd", "n/a", "ok", "measured", "12345", "short", "!!!!!!!!!!!")
        for text in thin:
            with self.subTest(detail=text):
                plan = _plate()
                plan["parameters"]["thickness"] = {
                    "value": 10.0, "evidence": {"op": "user_request", "detail": text}}
                self.assertIn("evidence.detail is too thin", _joined(plan))
                plan["parameters"]["thickness"] = {"value": 10.0, "assumption": text}
                self.assertIn("assumption is too thin", _joined(plan))
        plan = _plate()
        plan["parameters"]["thickness"] = {
            "value": 10.0, "evidence": {"op": "user_request", "detail": "user_request"}}
        self.assertIn("is too thin", _joined(plan))          # a detail that only repeats the op
        plan["parameters"]["thickness"] = {
            "value": 10.0, "evidence": {"op": "user_request", "detail": "the user asked for 10 mm"}}
        self.assertEqual(_errors(plan), [])

    def test_feature_ids(self):
        plan = _plate()
        plan["features"][1]["id"] = "base_sketch"
        self.assertIn("duplicate id", _joined(plan))
        plan = _plate()
        plan["features"][1]["id"] = "1pad"
        self.assertIn("id must be an identifier", _joined(plan))
        plan = _plate()
        plan["features"].append({"id": "h_sketch", "op": "sketch", "plane": {"frame_plane": "top"},
                                 "profile": {"shape": "circle", "diameter": "hole_dia"}})
        plan["features"].append({"id": "h", "op": "hole", "plane": {"frame_plane": "top"},
                                 "position": {"x": 0, "y": 0}, "diameter": "hole_dia",
                                 "through": True})
        self.assertIn("collides with the sketch the builder creates for the hole 'h'",
                      _joined(plan))

    def test_flags_must_be_booleans(self):
        plan = _plate()
        plan["features"][1]["midplane"] = "yes"
        self.assertIn("midplane must be true or false", _joined(plan))


class PositivityTest(unittest.TestCase):
    def check(self, mutate, *texts):
        plan = _plate()
        mutate(plan)
        joined = _joined(plan)
        for text in texts:
            self.assertIn(text, joined)

    def test_every_dimension_role_must_be_positive(self):
        cases = {
            "pad length": (lambda p: p["parameters"].update(thickness=_p(0)),
                           "parameter 'thickness' (0) is used as pad.length and must be positive"),
            "rectangle width": (lambda p: p["parameters"].update(plate_len=_p(-3)),
                                "sketch.rectangle.width and must be positive"),
            "circle diameter": (lambda p: p["features"].append(
                {"id": "c", "op": "sketch", "plane": {"frame_plane": "top"},
                 "profile": {"shape": "circle", "diameter": "zero_d"}}) or
                p["parameters"].update(zero_d=_p(0)),
                "sketch.circle.diameter"),
            "pocket depth": (lambda p: p["features"].append(
                {"id": "k", "op": "pocket", "sketch": "base_sketch", "depth": "neg_d"}) or
                p["parameters"].update(neg_d=_p(-1)),
                "feature k: parameter 'neg_d' (-1) is used as pocket.depth"),
            "hole diameter": (lambda p: p["features"].append(
                {"id": "h", "op": "hole", "plane": {"frame_plane": "top"},
                 "position": {"x": 0, "y": 0}, "diameter": "hole_dia", "through": True}) or
                p["parameters"].update(hole_dia=_p(-4)),
                "hole.diameter"),
        }
        for label, (mutate, text) in cases.items():
            with self.subTest(case=label):
                self.check(mutate, text)

    def test_angle_ranges_and_boundaries(self):
        def rev(angle):
            plan = _plate()
            plan["parameters"]["ang"] = _p(angle)
            plan["features"] += [
                {"id": "prof", "op": "sketch", "plane": {"frame_plane": "top"},
                 "profile": {"shape": "circle", "diameter": "hole_dia"}},
                {"id": "rev", "op": "revolution", "sketch": "prof",
                 "axis": {"direction": [1, 0, 0]}, "angle": "ang"}]
            return _errors(plan)
        self.assertEqual(rev(360), [])           # boundary: a full turn is valid
        self.assertEqual(rev(0.5), [])
        self.assertTrue(any("an angle in (0, 360] degrees" in e for e in rev(0)))
        self.assertTrue(any("an angle in (0, 360] degrees" in e for e in rev(361)))
        self.assertTrue(any("an angle in (0, 360] degrees" in e for e in rev(-90)))

    def test_countersink_angle_range(self):
        def csink(angle):
            plan = _plate()
            plan["parameters"].update(cs_dia=_p(8), cs_ang=_p(angle))
            plan["features"].append({
                "id": "h", "op": "hole", "plane": {"frame_plane": "top"},
                "position": {"x": 0, "y": 0}, "diameter": "hole_dia", "through": True,
                "countersink": {"cs_diameter": "cs_dia", "cs_angle": "cs_ang"}})
            return _errors(plan)
        self.assertEqual(csink(90), [])
        self.assertEqual(csink(179.9), [])
        self.assertTrue(any("an angle in (0, 180) degrees" in e for e in csink(180)))
        self.assertTrue(any("an angle in (0, 180) degrees" in e for e in csink(0)))

    def test_literal_dimensions_are_checked_and_positions_may_be_negative(self):
        plan = _plate()
        plan["features"].append({
            "id": "h", "op": "hole", "plane": {"frame_plane": "top"},
            "position": {"x": -30, "y": -15}, "diameter": "hole_dia", "through": True})
        self.assertEqual(_errors(plan), [])

    def test_each_violation_is_reported_once(self):
        plan = _plate()
        plan["parameters"]["thickness"] = _p(-1)
        errors = [e for e in _errors(plan) if "pad.length" in e]
        self.assertEqual(len(errors), 1)

    def test_slot_and_counterbore_relations(self):
        plan = _plate()
        plan["parameters"].update(slot_l=_p(5), slot_w=_p(5))
        plan["features"].append({"id": "s", "op": "sketch", "plane": {"frame_plane": "top"},
                                 "profile": {"shape": "slot", "length": "slot_l", "width": "slot_w"}})
        self.assertIn("slot length (5) must exceed its width (5)", _joined(plan))
        plan = _plate()
        plan["parameters"].update(cb_d=_p(3), cb_h=_p(12))
        plan["features"].append({
            "id": "h", "op": "hole", "plane": {"frame_plane": "top"},
            "position": {"x": 0, "y": 0}, "diameter": "hole_dia", "depth": "thickness",
            "counterbore": {"cb_diameter": "cb_d", "cb_depth": "cb_h"}})
        ok, errors, warnings = schema.validate(plan)
        self.assertIn("counterbore diameter (3) must exceed the hole diameter (4)", "\n".join(errors))
        self.assertTrue(any("counterbore depth (12) is not smaller than the hole depth" in w
                            for w in warnings), warnings)


class UnknownKeyTest(unittest.TestCase):
    def test_typos_get_the_nearest_key(self):
        cases = {
            "feature": (lambda p: p["features"][1].update(lenght="thickness"),
                        "feature base_pad (pad): unknown key 'lenght'; did you mean 'length'?"),
            "feature depth": (lambda p: p["features"].append(
                {"id": "k", "op": "pocket", "sketch": "base_sketch", "depht": "thickness"}),
                "unknown key 'depht'; did you mean 'depth'?"),
            "profile": (lambda p: p["features"][0]["profile"].update(hieght="plate_wid"),
                        "profile: unknown key 'hieght'; did you mean 'height'?"),
            "plane": (lambda p: p["features"][0]["plane"].update(frame_plan="bottom"),
                      "plane: unknown key 'frame_plan'; did you mean 'frame_plane'?"),
            "parameter": (lambda p: p["parameters"]["thickness"].update(evidance="x"),
                          "parameter 'thickness': unknown key 'evidance'; did you mean 'evidence'?"),
            "hole position": (lambda p: p["features"].append(
                {"id": "h", "op": "hole", "plane": {"frame_plane": "top"},
                 "position": {"x": 0, "y": 0, "z": 1}, "diameter": "hole_dia", "through": True}),
                "position: unknown key 'z'"),
        }
        for label, (mutate, text) in cases.items():
            with self.subTest(case=label):
                plan = _plate()
                mutate(plan)
                self.assertIn(text, _joined(plan))

    def test_an_unrelated_key_lists_the_known_ones_without_a_wrong_suggestion(self):
        plan = _plate()
        plan["features"][1]["zzzzzz"] = 1
        joined = _joined(plan)
        self.assertIn("unknown key 'zzzzzz' (known: ", joined)
        self.assertNotIn("did you mean", joined.split("zzzzzz")[1].split("\n")[0])

    def test_segment_keys(self):
        plan = _load("l_bracket")
        plan["features"][0]["profile"]["segments"][0]["tooo"] = [1, 2]
        self.assertIn("segment 0: unknown key 'tooo'; did you mean 'to'?", _joined(plan))


class SlotAndProfileTest(unittest.TestCase):
    def slot_plan(self, **profile):
        plan = _plate()
        plan["parameters"].update(slot_l=_p(20), slot_w=_p(6), slot_a=_p(30))
        plan["features"].append({"id": "s", "op": "sketch", "plane": {"frame_plane": "top"},
                                 "profile": {"shape": "slot", "length": "slot_l",
                                             "width": "slot_w", **profile}})
        return plan

    def test_slot_angle_spellings(self):
        for key in ("angle", "angle_deg"):
            for value in (30, "slot_a"):
                with self.subTest(key=key, value=value):
                    self.assertEqual(_errors(self.slot_plan(**{key: value})), [])
        self.assertIn("gives both 'angle' and 'angle_deg'",
                      _joined(self.slot_plan(angle=1, angle_deg=2)))
        self.assertIn("unknown parameter 'slot_b'",
                      _joined(self.slot_plan(angle_deg="slot_b")))

    def test_open_polyline_is_rejected(self):
        plan = _load("l_bracket")
        plan["features"][0]["profile"]["segments"].pop()
        self.assertIn("profile is not closed", _joined(plan))

    def test_polyline_segments_are_checked(self):
        plan = _load("l_bracket")
        plan["features"][0]["profile"]["segments"][0]["type"] = "spline"
        self.assertIn("segment 0 type must be line/arc", _joined(plan))
        plan = _load("l_bracket")
        plan["features"][0]["profile"]["segments"][1] = {"type": "arc", "to": [1, 1]}
        joined = _joined(plan)
        self.assertIn("arc segment 1 needs center", joined)
        self.assertIn("needs ccw true/false", joined)


class DerivedParameterTest(unittest.TestCase):
    def plan(self, **derived):
        plan = _plate()
        for name, expr in derived.items():
            plan["parameters"][name] = {"expr": expr}
        return plan

    def test_values_follow_the_formulas_in_dependency_order(self):
        plan = self.plan(half_w="plate_wid / 2", outer="half_w + thickness ^ 2 - 1",
                         diag="sqrt(plate_len^2 + plate_wid^2) + abs(-2) + pi")
        values = schema.parameter_values(plan)
        self.assertEqual(values["half_w"], 20.0)
        self.assertEqual(values["outer"], 119.0)
        self.assertAlmostEqual(values["diag"], math.sqrt(60 ** 2 + 40 ** 2) + 2 + math.pi)
        order = schema.derived_order(plan)
        self.assertLess(order.index("half_w"), order.index("outer"))
        self.assertEqual(schema.resolve(plan, "outer"), 119.0)
        self.assertEqual(schema.resolve(plan, "thickness"), 10.0)

    def test_dependency_order_does_not_depend_on_declaration_order(self):
        plan = _plate()
        plan["parameters"]["outer"] = {"expr": "half_w * 2"}
        plan["parameters"]["half_w"] = {"expr": "plate_wid / 2"}
        self.assertEqual(schema.derived_order(plan), ["half_w", "outer"])
        self.assertEqual(schema.parameter_values(plan)["outer"], 40.0)

    def test_derived_parameters_need_no_evidence_and_may_drive_features(self):
        plan = self.plan(half_w="plate_wid / 2")
        plan["features"][0]["profile"]["height"] = "half_w"
        self.assertEqual(_errors(plan), [])

    def test_cycles_are_reported_with_the_path(self):
        plan = self.plan(a_val="b_val + 1", b_val="c_val * 2", c_val="a_val - 1")
        self.assertIn("derived parameters form a cycle: a_val -> b_val -> c_val -> a_val",
                      _joined(plan))
        self.assertNotIn("a_val", schema.parameter_values(plan))   # not evaluable

    def test_self_reference_is_a_cycle(self):
        self.assertIn("form a cycle", _joined(self.plan(loop_val="loop_val + 1")))

    def test_unknown_reference_and_bad_syntax(self):
        self.assertIn("parameter 'ratio': expr references unknown parameter 'ghost'",
                      _joined(self.plan(ratio="ghost * 2")))
        for bad, text in (("", "non-empty"), ("2 +", "incomplete formula"),
                          ("2 3", "two values need an operator"), ("foo(2)", "unknown function 'foo'"),
                          ("-thickness^2", "ambiguous"), ("2 $ 3", "unexpected character"),
                          ("(2 + 3", "expected ')'")):
            with self.subTest(expr=bad):
                self.assertIn(text, _joined(self.plan(ratio=bad)))

    def test_arithmetic_failures_name_the_parameter(self):
        self.assertIn("parameter 'ratio': expr divides by zero",
                      _joined(self.plan(ratio="thickness / (thickness - 10)")))
        self.assertIn("sqrt of a negative number",
                      _joined(self.plan(ratio="sqrt(thickness - 20)")))
        self.assertIn("power 10^1000 is undefined", _joined(self.plan(ratio="10 ^ 1000")))

    def test_a_disagreeing_value_is_a_warning(self):
        plan = self.plan(half_w="plate_wid / 2")
        plan["parameters"]["half_w"]["value"] = 99.0
        ok, _errors_, warnings = schema.validate(plan)
        self.assertTrue(ok)
        self.assertTrue(any("disagrees with its expr (20)" in w for w in warnings), warnings)

    def test_expression_grammar_helpers(self):
        tree, names = schema.parse_expr("a + b * (c - 2) / sqrt(d)")
        self.assertEqual(names, frozenset({"a", "b", "c", "d"}))
        self.assertEqual(schema.parse_expr("pi")[1], frozenset())
        self.assertTrue(schema.is_derived({"expr": "1"}))
        self.assertFalse(schema.is_derived({"value": 1}))
        self.assertFalse(schema.is_derived(None))
        with self.assertRaises(schema.ExprError):
            schema.parse_expr(None)

    def test_resolve_errors(self):
        plan = self.plan(bad_val="thickness / 0")
        with self.assertRaises(KeyError):
            schema.resolve(plan, "nope")
        with self.assertRaises(ValueError):
            schema.resolve(plan, "bad_val")


class ParameterNameTest(unittest.TestCase):
    def test_alias_rules(self):
        good = ("wall_t", "hole_dia", "a1_dim", "ab12", "x_", "Width", "plate_len2")
        bad = ("mm", "min", "t", "x", "pi", "e", "deg", "A1", "AB12", "_hidden",
               "1abc", "with space", "a-b", "größe", "")
        for name in good:
            with self.subTest(good=name):
                plan = _plate()
                plan["parameters"][name] = _p(1)
                self.assertFalse(any(f"parameter {name!r}:" in e for e in _errors(plan)))
        for name in bad:
            with self.subTest(bad=name):
                plan = _plate()
                plan["parameters"][name] = _p(1)
                self.assertTrue(any(f"parameter {name!r}:" in e for e in _errors(plan)))

    def test_hints_say_what_to_do(self):
        plan = _plate()
        plan["parameters"]["mm"] = _p(1)
        self.assertIn("unit symbol or constant FreeCAD rejects", _joined(plan))
        plan = _plate()
        plan["parameters"]["B7"] = _p(1)
        self.assertIn("cell address", _joined(plan))


class HoleCutTest(unittest.TestCase):
    def test_flat_and_nested_spellings_agree(self):
        flat = schema.hole_cut({"counterbore": True, "cb_diameter": "a", "cb_depth": "b"})
        nested = schema.hole_cut({"counterbore": {"cb_diameter": "a", "cb_depth": "b"}})
        short = schema.hole_cut({"counterbore": {"diameter": "a", "depth": "b"}})
        self.assertEqual(flat, nested)
        self.assertEqual(flat, short)
        self.assertEqual(flat, {"type": "counterbore", "diameter": "a", "depth": "b",
                                "angle": None})
        cs_flat = schema.hole_cut({"countersink": True, "cs_diameter": "d", "cs_angle": 82})
        cs_nested = schema.hole_cut({"countersink": {"cs_diameter": "d", "cs_angle": 82}})
        self.assertEqual(cs_flat, cs_nested)
        self.assertEqual(cs_flat["angle"], 82)
        self.assertIsNone(schema.hole_cut({"countersink": {"diameter": "d"}})["angle"])
        self.assertIsNone(schema.hole_cut({"diameter": "x"}))

    def test_contradictions_raise_planner_facing_text(self):
        for feat, text in (
                ({"counterbore": True, "countersink": True}, "mutually exclusive"),
                ({"counterbore": True, "cb_diameter": "a"}, "needs both cb_diameter and cb_depth"),
                ({"countersink": True}, "needs cs_diameter"),
                ({"cb_diameter": "a"}, "without counterbore: true")):
            with self.subTest(feat=feat):
                with self.assertRaises(ValueError) as ctx:
                    schema.hole_cut(feat)
                self.assertIn(text, str(ctx.exception))


class DimensionRefsTest(unittest.TestCase):
    def test_roles_cover_every_dimension_of_the_plan(self):
        refs = schema.dimension_refs(_load("plate_4holes"))
        roles = {(fid, role) for fid, role, _ref, _rule in refs}
        self.assertIn(("base_sketch", "sketch.rectangle.width"), roles)
        self.assertIn(("base_pad", "pad.length"), roles)
        self.assertIn(("hole_a", "hole.diameter"), roles)
        self.assertNotIn(("hole_a", "hole.depth"), roles)         # through hole
        self.assertTrue(all(rule in ("positive", "angle360", "angle180")
                            for _f, _r, _ref, rule in refs))

    def test_malformed_features_are_skipped_not_raised(self):
        plan = {"features": [None, {"id": 3}, {"id": "x", "op": "sketch", "profile": 5},
                             {"id": "y", "op": "pad", "length": 5}]}
        self.assertEqual(schema.dimension_refs(plan), [("y", "pad.length", 5, "positive")])




if __name__ == "__main__":
    unittest.main()
