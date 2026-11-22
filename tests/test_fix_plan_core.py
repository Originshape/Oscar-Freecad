"""Regression tests for the plan-core review pass (builder / schema /
validate / classify): every case here failed before its fix.

Runs in console mode via tests/run.py (Sketcher/PartDesign are App-level;
no GUI needed)."""

import copy
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

import FreeCAD as App  # noqa: E402

from oscar import tools  # noqa: E402
from oscar.plan import _buildcore, _featurebuild, builder, schema

_WORKBENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(_WORKBENCH, "tests", "fixtures", "plans")


def _example(name: str) -> dict:
    with open(os.path.join(EXAMPLES, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def _plan(params: dict, features: list, frame: dict | None = None) -> dict:
    return {
        "schema_version": 1, "id": "t", "part_class": "plate", "units": "mm",
        "target": "freecad",
        "frame": frame or {"planes": {"base": {
            "origin": [0, 0, 0], "normal": [0, 0, 1], "x_dir": [1, 0, 0]}}},
        "parameters": {k: {"value": v, "assumption": "test fixture dimension"}
                       for k, v in params.items()},
        "features": features,
    }


def _slab(params: dict, extra: list | None = None) -> dict:
    """40x30xthk rectangle pad, then ``extra`` features on top of it."""
    base = {"wid": 40, "hgt": 30, "thk": 10}
    base.update(params)
    features = [
        {"id": "s", "op": "sketch", "plane": {"frame_plane": "base"},
         "profile": {"shape": "rectangle", "width": "wid", "height": "hgt"}},
        {"id": "p", "op": "pad", "sketch": "s", "length": "thk"},
    ] + (extra or [])
    return _plan(base, features)


_TOP = {"origin": [0, 0, "thk"], "normal": [0, 0, 1], "x_dir": [1, 0, 0]}


def _hole(extra: dict | None = None) -> dict:
    return {"id": "h", "op": "hole", "plane": dict(_TOP),
            "position": {"x": 0, "y": 0}, "diameter": "dia", "through": True,
            **(extra or {})}


def _bbox(doc, name="Body"):
    bb = doc.getObject(name).Shape.BoundBox
    return (bb.XMin, bb.XMax, bb.YMin, bb.YMax, bb.ZMin, bb.ZMax)


class _DocCase(unittest.TestCase):
    @staticmethod
    def _close(name):
        if name in App.listDocuments():
            App.closeDocument(name)

    def _doc(self, name="FixRC"):
        doc = App.newDocument(name)
        self.addCleanup(self._close, doc.Name)
        return doc

    def _build(self, plan, doc=None, ok=True):
        doc = doc or self._doc()
        report = builder.apply_plan(doc, plan)
        if ok:
            self.assertTrue(report.get("success"),
                            f"{report.get('feature_id')}: {report.get('error')}")
        return doc, report

    def assertBox(self, doc, expected, delta=0.05, name="Body"):
        got = _bbox(doc, name)
        for g, e, label in zip(got, expected,
                               ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")):
            if e is not None:
                self.assertAlmostEqual(g, e, delta=delta, msg=f"{label}: {got}")

    def assertVolume(self, doc, expected, delta=1.0, name="Body"):
        self.assertAlmostEqual(doc.getObject(name).Shape.Volume, expected,
                               delta=delta)


# ---------------------------------------------------------------------------
# plan-core-02: blind pocket sets Length (PartDesign::Pocket has no Depth)


class BlindPocketTest(_DocCase):
    def test_blind_pocket_builds_with_the_requested_depth(self):
        plan = _slab({"dia": 8, "pd": 4}, [
            {"id": "c", "op": "sketch", "plane": dict(_TOP),
             "profile": {"shape": "circle", "diameter": "dia"}},
            {"id": "k", "op": "pocket", "sketch": "c", "depth": "pd"},
        ])
        doc, _report = self._build(plan)
        self.assertVolume(doc, 40 * 30 * 10 - math.pi * 16 * 4)
        # the depth stays bound to the spreadsheet
        sheet = doc.getObject("Parameters")
        sheet.set(sheet.getCellFromAlias("pd"), "6.0")
        doc.recompute()
        self.assertVolume(doc, 40 * 30 * 10 - math.pi * 16 * 6)


# ---------------------------------------------------------------------------
# plan-core-03: counterbore / countersink use the 1.1 HoleCut* properties
# and accept both the flat and the nested (prompt-documented) spelling


class HoleCutTest(_DocCase):
    PARAMS = {"dia": 10, "cbd": 18, "cbh": 4, "csd": 18, "csa": 90}
    PLAIN = 40 * 30 * 10 - math.pi * 25 * 10
    CBORE = PLAIN - math.pi * (81 - 25) * 4

    def _hole_volume(self, extra):
        doc, _ = self._build(_slab(self.PARAMS, [_hole(extra)]))
        return doc

    def test_plain_hole_has_no_cut(self):
        doc = self._hole_volume({})
        self.assertVolume(doc, self.PLAIN)
        self.assertEqual(doc.getObject("h").HoleCutType, "None")

    def test_counterbore_flat(self):
        doc = self._hole_volume({"counterbore": True,
                                 "cb_diameter": "cbd", "cb_depth": "cbh"})
        self.assertVolume(doc, self.CBORE)

    def test_counterbore_nested(self):
        doc = self._hole_volume({"counterbore": {"cb_diameter": "cbd",
                                                 "cb_depth": "cbh"}})
        self.assertVolume(doc, self.CBORE)

    def test_counterbore_rides_the_spreadsheet(self):
        doc = self._hole_volume({"counterbore": {"cb_diameter": "cbd",
                                                 "cb_depth": "cbh"}})
        sheet = doc.getObject("Parameters")
        sheet.set(sheet.getCellFromAlias("cbd"), "22.0")
        doc.recompute()
        self.assertVolume(doc, self.PLAIN - math.pi * (121 - 25) * 4)

    def test_countersink_flat_and_nested(self):
        for extra in ({"countersink": True, "cs_diameter": "csd",
                       "cs_angle": "csa"},
                      {"countersink": {"cs_diameter": "csd",
                                       "cs_angle": "csa"}}):
            with self.subTest(extra=extra):
                doc = self._hole_volume(extra)
                # 90 deg cone from d18 down to the d10 bore: depth 4
                cone = math.pi * 4 / 3 * (81 + 45 + 25) - math.pi * 25 * 4
                self.assertVolume(doc, self.PLAIN - cone)

    def test_missing_cut_dimension_is_a_validation_error(self):
        plan = _slab(self.PARAMS, [_hole({"counterbore": {"cb_depth": "cbh"}})])
        ok, errors, _w = schema.validate(plan)
        self.assertFalse(ok)
        self.assertTrue(any("cb_diameter" in e and "h" in e for e in errors),
                        errors)


# ---------------------------------------------------------------------------
# plan-core-05: arcs in a polyline profile


def _dshape(ccw: bool, shallow: bool = False) -> dict:
    """30x20 block whose right edge is an arc: bulging out (ccw) or notched
    in (cw). Default is a 10 mm-radius semicircle with the radius left
    free (a plan that does not name it leaves one arc DOF open: v1 limit);
    ``shallow`` is a minor arc with a named radius."""
    if shallow:
        arc = {"type": "arc", "to": ["lenx", "heit"],
               "center": ["acx", 10], "ccw": ccw, "radius": "arc_r"}
    else:
        arc = {"type": "arc", "to": ["lenx", "heit"],
               "center": ["lenx", 10], "ccw": ccw}
    segs = [{"type": "line", "to": ["lenx", 0]}, arc,
            {"type": "line", "to": [0, "heit"]},
            {"type": "line", "to": [0, 0]}]
    params = {"lenx": 30, "heit": 20, "thk": 5}
    if shallow:   # centre 5 mm off the chord: r = sqrt(5^2 + 10^2)
        params.update(acx=25 if ccw else 35, arc_r=math.sqrt(125))
    return _plan(params, [
        {"id": "s", "op": "sketch", "plane": {"frame_plane": "base"},
         "profile": {"shape": "profile", "start": [0, 0], "segments": segs}},
        {"id": "p", "op": "pad", "sketch": "s", "length": "thk"}])


class ProfileArcTest(_DocCase):
    HALF_DISC = math.pi * 100 / 2

    def test_bulging_arc_builds(self):
        doc, _ = self._build(_dshape(True))
        self.assertVolume(doc, (600 + self.HALF_DISC) * 5)

    def test_notch_arc_builds(self):
        # ccw=False swaps the arc's start/end angles; every constraint that
        # addresses "start"/"end" of the arc has to follow
        doc, _ = self._build(_dshape(False))
        self.assertVolume(doc, (600 - self.HALF_DISC) * 5)

    def test_arc_profile_rides_the_spreadsheet(self):
        # minor arc segment: 20 mm chord, r = sqrt(125)
        theta = 2 * math.asin(10 / math.sqrt(125))
        segment = 0.5 * 125 * (theta - math.sin(theta))
        for ccw, sign in ((True, 1), (False, -1)):
            with self.subTest(ccw=ccw):
                doc, report = self._build(_dshape(ccw, shallow=True))
                self.assertVolume(doc, (600 + sign * segment) * 5)
                self.assertTrue(
                    report["features"][0]["info"]["fully_constrained"])
                sheet = doc.getObject("Parameters")
                sheet.set(sheet.getCellFromAlias("lenx"), "40.0")
                doc.recompute()
                self.assertVolume(doc, (800 + sign * segment) * 5)


# ---------------------------------------------------------------------------
# plan-core-04: apply_plan is re-runnable on a document that already holds
# a plan build (failed attempt, or a changed plan)


def _types(doc, type_id):
    return [o.Name for o in doc.Objects if o.TypeId == type_id]


# The undo steps one apply of plate_4holes leaves, newest first: one labelled transaction per feature plus the setup
# (which also removes a previous build). The tests read them instead of popping them: FreeCAD 1.1 can crash in
# Transaction::apply when Document.undo() reverts PartDesign plan steps (P0 #4), so the product never does that either
# (a revert restores the turn checkpoint; tests/test_project_rewind.PlanBuildTurnRevertTest).
PLATE_LABELS = ["Oscar: plan hole_d", "Oscar: plan hole_c", "Oscar: plan hole_b", "Oscar: plan hole_a",
                "Oscar: plan base_pad", "Oscar: plan base_sketch", "Oscar: plan setup"]


def _new_labels(doc, depth: int) -> list:
    """The undo labels added since the stack held ``depth`` entries, newest first."""
    return list(doc.UndoNames)[:doc.UndoCount - depth]


class ReapplyTest(_DocCase):
    PLATE_VOLUME = 100 * 60 * 8 - 4 * math.pi * 25 * 8

    def test_corrected_retry_after_a_failed_build(self):
        doc = self._doc()
        bad = _example("plate_4holes")
        # our schema already rejects thickness 0 at validation, so the build
        # failure comes from a hole that misses the plate (cuts nothing)
        bad["features"][-1]["position"]["x"] = 500.0
        failed = builder.apply_plan(doc, bad)
        self.assertFalse(failed["success"])
        self.assertEqual(failed["stage"], "build")
        self.assertEqual(failed["feature_id"], "hole_d")
        # the fixed plan must not bind to the failed attempt's sheet
        _doc, report = self._build(_example("plate_4holes"), doc=doc)
        self.assertEqual(report["spreadsheet"], "Parameters")
        self.assertEqual(report["body"], "Body")
        self.assertVolume(doc, self.PLATE_VOLUME, delta=2.0)
        self.assertEqual(_types(doc, "Spreadsheet::Sheet"), ["Parameters"])
        self.assertEqual(_types(doc, "PartDesign::Body"), ["Body"])

    def test_reapply_with_changed_values_rebuilds_from_the_new_values(self):
        doc, _ = self._build(_example("plate_4holes"))
        changed = _example("plate_4holes")
        changed["parameters"]["thickness"]["value"] = 12.0
        self._build(changed, doc=doc)
        self.assertEqual(_types(doc, "PartDesign::Body"), ["Body"])
        self.assertEqual(_types(doc, "Spreadsheet::Sheet"), ["Parameters"])
        self.assertVolume(doc, 100 * 60 * 12 - 4 * math.pi * 25 * 12, delta=2.0)
        # and the spreadsheet is live: a cell edit still resizes the model
        sheet = doc.getObject("Parameters")
        sheet.set(sheet.getCellFromAlias("thickness"), "8.0")
        doc.recompute()
        self.assertVolume(doc, self.PLATE_VOLUME, delta=2.0)

    def test_foreign_parameters_sheet_is_never_deleted_or_bound(self):
        doc = self._doc()
        mine = doc.addObject("Spreadsheet::Sheet", "Parameters")
        mine.set("A1", "my notes")
        mine.set("B1", "5")
        doc.recompute()
        before = mine.getContents("A1")
        _doc, report = self._build(_example("plate_4holes"), doc=doc)
        self.assertEqual(report["spreadsheet"], "Parameters001")
        self.assertIsNotNone(doc.getObject("Parameters"))
        self.assertEqual(doc.getObject("Parameters").getContents("A1"), before)
        self.assertVolume(doc, self.PLATE_VOLUME, delta=2.0,
                          name=report["body"])
        # the new model is driven by ITS sheet, not the foreign one
        sheet = doc.getObject("Parameters001")
        sheet.set(sheet.getCellFromAlias("thickness"), "12.0")
        doc.recompute()
        self.assertVolume(doc, 100 * 60 * 12 - 4 * math.pi * 25 * 12,
                          delta=2.0, name=report["body"])

    def test_reapply_with_undo_enabled(self):
        # the interactive tool runs with an undo stack; the cleanup is one
        # transaction and must leave a single consistent build behind, with
        # one labelled undo step per feature (never popped here: P0 #4)
        doc = self._doc()
        doc.UndoMode = 1
        self._build(_example("plate_4holes"), doc=doc)
        depth = doc.UndoCount
        changed = _example("plate_4holes")
        changed["parameters"]["thickness"]["value"] = 12.0
        self._build(changed, doc=doc)
        self.assertEqual(_types(doc, "PartDesign::Body"), ["Body"])
        self.assertVolume(doc, 100 * 60 * 12 - 4 * math.pi * 25 * 12, delta=2.0)
        self.assertEqual(_new_labels(doc, depth), PLATE_LABELS)


# ---------------------------------------------------------------------------
# plan-core-01: slot profiles


def _slot_plan(center, angle=None, params=None) -> dict:
    prof = {"shape": "slot", "length": "len", "width": "wd", "center": center}
    if angle is not None:
        prof["angle"] = angle
    return _plan({"len": 30, "wd": 10, "thk": 5, **(params or {})}, [
        {"id": "s", "op": "sketch", "plane": {"frame_plane": "base"},
         "profile": prof},
        {"id": "p", "op": "pad", "sketch": "s", "length": "thk"}])


class SlotTest(_DocCase):
    @staticmethod
    def _volume(length, width=10.0, thk=5.0):
        return ((length - width) * width + math.pi * (width / 2) ** 2) * thk

    def test_slot_at_origin(self):
        doc, report = self._build(_slot_plan([0, 0]))
        self.assertBox(doc, (-15, 15, -5, 5, 0, 5))
        self.assertVolume(doc, self._volume(30))
        self.assertTrue(report["features"][0]["info"]["fully_constrained"])

    def test_slot_off_origin(self):
        doc, report = self._build(_slot_plan([50, 20]))
        self.assertBox(doc, (35, 65, 15, 25, 0, 5))
        self.assertTrue(report["features"][0]["info"]["fully_constrained"])

    def test_slot_rotated(self):
        doc, report = self._build(_slot_plan([0, 0], 30))
        ext = 10 * math.cos(math.radians(30)) + 5
        self.assertBox(doc, (-ext, ext, -10, 10, 0, 5))
        self.assertVolume(doc, self._volume(30))
        self.assertTrue(report["features"][0]["info"]["fully_constrained"])

    def test_slot_length_edit_keeps_it_centred(self):
        for center, angle in (([0, 0], None), ([50, 20], None),
                              ([50, 20], 30)):
            with self.subTest(center=center, angle=angle):
                doc, _ = self._build(_slot_plan(center, angle))
                sheet = doc.getObject("Parameters")
                sheet.set(sheet.getCellFromAlias("len"), "40.0")
                doc.recompute()
                self.assertVolume(doc, self._volume(40))
                bb = doc.getObject("Body").Shape.BoundBox
                self.assertAlmostEqual(bb.Center.x, center[0], delta=0.05)
                self.assertAlmostEqual(bb.Center.y, center[1], delta=0.05)

    def test_slot_angle_parameter_rotates_the_slot(self):
        for start in (30, 0):
            with self.subTest(start=start):
                doc, _ = self._build(_slot_plan([0, 0], "ang",
                                                params={"ang": start}))
                sheet = doc.getObject("Parameters")
                sheet.set(sheet.getCellFromAlias("ang"), "90.0")
                doc.recompute()
                self.assertBox(doc, (-5, 5, -15, 15, 0, 5))

    def test_slot_with_parameter_bound_centre(self):
        doc, _ = self._build(_slot_plan(["cxp", 20], params={"cxp": 50}))
        self.assertBox(doc, (35, 65, 15, 25, 0, 5))
        sheet = doc.getObject("Parameters")
        sheet.set(sheet.getCellFromAlias("cxp"), "70.0")
        doc.recompute()
        self.assertBox(doc, (55, 85, 15, 25, 0, 5))


# ---------------------------------------------------------------------------
# plan-core-07: rectangle centre is honoured


def _rect_plan(center, params=None) -> dict:
    return _plan({"wid": 40, "hgt": 20, "thk": 5, **(params or {})}, [
        {"id": "s", "op": "sketch", "plane": {"frame_plane": "base"},
         "profile": {"shape": "rectangle", "width": "wid", "height": "hgt",
                     "center": center}},
        {"id": "p", "op": "pad", "sketch": "s", "length": "thk"}])


class RectangleCenterTest(_DocCase):
    def test_origin_centre_unchanged(self):
        doc, report = self._build(_rect_plan([0, 0]))
        self.assertBox(doc, (-20, 20, -10, 10, 0, 5))
        self.assertTrue(report["features"][0]["info"]["fully_constrained"])

    def test_numeric_centre(self):
        doc, report = self._build(_rect_plan([30, 10]))
        self.assertBox(doc, (10, 50, 0, 20, 0, 5))
        self.assertTrue(report["features"][0]["info"]["fully_constrained"])

    def test_parameter_bound_centre_follows_edits(self):
        doc, report = self._build(_rect_plan(["cxp", 10], {"cxp": 30}))
        self.assertBox(doc, (10, 50, 0, 20, 0, 5))
        self.assertTrue(report["features"][0]["info"]["fully_constrained"])
        sheet = doc.getObject("Parameters")
        sheet.set(sheet.getCellFromAlias("cxp"), "50.0")
        doc.recompute()
        self.assertBox(doc, (30, 70, 0, 20, 0, 5))
        # a size edit grows the part about its centre, not the origin
        sheet.set(sheet.getCellFromAlias("wid"), "60.0")
        doc.recompute()
        self.assertBox(doc, (20, 80, 0, 20, 0, 5))


# ---------------------------------------------------------------------------
# plan-core-13: a revolution axis keeps its direction and position


def _revolve_plan(axis: dict, angle=180.0, origin=(0, 0, 0)) -> dict:
    """10x20 profile in the XZ plane (sketch x = world X, y = world Z)."""
    segs = [{"type": "line", "to": ["rr", 0]},
            {"type": "line", "to": ["rr", "zz"]},
            {"type": "line", "to": [0, "zz"]},
            {"type": "line", "to": [0, 0]}]
    return _plan({"rr": 10, "zz": 20, "ang": angle}, [
        {"id": "s", "op": "sketch",
         "plane": {"origin": list(origin), "normal": [0, -1, 0],
                   "x_dir": [1, 0, 0]},
         "profile": {"shape": "profile", "start": [0, 0], "segments": segs}},
        {"id": "r", "op": "revolution", "sketch": "s", "axis": axis,
         "angle": "ang"}])


class RevolutionAxisTest(_DocCase):
    def test_positive_axis_sweeps_to_one_side(self):
        doc, _ = self._build(_revolve_plan({"direction": [0, 0, 1],
                                            "point": [0, 0, 0]}))
        self.assertBox(doc, (-10, 10, 0, 10, 0, 20))
        self.assertVolume(doc, math.pi * 100 * 20 / 2)

    def test_reversed_axis_sweeps_to_the_other_side(self):
        doc, _ = self._build(_revolve_plan({"direction": [0, 0, -1],
                                            "point": [0, 0, 0]}))
        self.assertBox(doc, (-10, 10, -10, 0, 0, 20))
        self.assertVolume(doc, math.pi * 100 * 20 / 2)

    def test_a_point_further_along_the_axis_is_still_the_same_axis(self):
        doc, _ = self._build(_revolve_plan({"direction": [0, 0, 1],
                                            "point": [0, 0, 50]}))
        self.assertBox(doc, (-10, 10, 0, 10, 0, 20))

    def test_axis_that_misses_the_sketch_axis_is_rejected(self):
        for point in ([30, 0, 0], [0, 5, 0]):
            with self.subTest(point=point):
                doc, report = self._build(
                    _revolve_plan({"direction": [0, 0, 1], "point": point}),
                    ok=False)
                self.assertFalse(report["success"])
                self.assertEqual(report["feature_id"], "r")
                self.assertIn("axis", report["error"])

    def test_off_origin_sketch_with_its_axis_on_it_builds(self):
        doc, _ = self._build(_revolve_plan(
            {"direction": [0, 0, 1], "point": [30, 0, 0]}, origin=(30, 0, 0)))
        self.assertBox(doc, (20, 40, 0, 10, 0, 20))

    def test_golden_shaft_still_builds_with_a_frame_axis(self):
        doc, report = self._build(_example("stepped_shaft"))
        self.assertTrue(report["success"])
        bb = doc.getObject("Body").Shape.BoundBox
        self.assertAlmostEqual(bb.Center.x, 0.0, delta=0.05)
        self.assertAlmostEqual(bb.Center.y, 0.0, delta=0.05)


# ---------------------------------------------------------------------------
# plan-core-14: apply_feature takes the (possibly patched) plan's values


class ApplyFeatureSyncTest(_DocCase):
    def _patched(self, **values):
        plan = _example("plate_4holes")
        for name, value in values.items():
            plan["parameters"][name] = (
                {"value": value, "assumption": "patched test value"}
                if name not in plan["parameters"]
                else {**plan["parameters"][name], "value": value})
        return plan

    def test_changed_value_reaches_the_sheet_and_the_sketch(self):
        doc, _ = self._build(_example("plate_4holes"))
        patched = self._patched(hole_dx=35.0)
        patched["features"][-1]["position"].update(x="hole_dx", y=0.0)  # y=0: clear of hole_b, which hole_dx also drives
        result = builder.apply_feature(doc, patched, "hole_d")
        self.assertTrue(result["success"], result)
        self.assertIn("hole_dx", result["parameters_changed"])
        sheet = doc.getObject("Parameters")
        self.assertEqual(float(sheet.getContents(
            sheet.getCellFromAlias("hole_dx"))), 35.0)
        circle = doc.getObject("hole_d_sketch").Geometry[0]
        self.assertAlmostEqual(circle.Center.x, 35.0, delta=1e-6)

    def test_new_parameter_is_added_and_the_feature_builds(self):
        doc, _ = self._build(_example("plate_4holes"))
        patched = self._patched(hole_dia2=6.0)
        patched["features"][-1]["diameter"] = "hole_dia2"
        result = builder.apply_feature(doc, patched, "hole_d")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["parameters_changed"], ["hole_dia2"])
        sheet = doc.getObject("Parameters")
        self.assertIsNotNone(sheet.getCellFromAlias("hole_dia2"))
        circle = doc.getObject("hole_d_sketch").Geometry[0]
        self.assertAlmostEqual(circle.Radius, 3.0, delta=1e-6)

    def test_unchanged_plan_changes_nothing(self):
        doc, _ = self._build(_example("plate_4holes"))
        result = builder.apply_feature(doc, _example("plate_4holes"), "hole_d")
        self.assertTrue(result["success"], result)
        self.assertNotIn("parameters_changed", result)
        self.assertAlmostEqual(doc.getObject("Body").Shape.Volume,
                               100 * 60 * 8 - 4 * math.pi * 25 * 8, delta=2.0)


# ---------------------------------------------------------------------------
# plan-core-12: coaxiality is an angular + positional tolerance, not exact
# 3-decimal equality of the axis vector


def _inspect(guess: str) -> dict:
    return {"part_class_guess": {"class": guess, "confidence": 0.5,
                                 "reasons": ["guess"]}}


def _segment(axis1, axis2, point1=(0, 0, 20), point2=(0, 0, 55)) -> dict:
    return {"regions": [
        {"type": "plane", "normal": [0, 0, 1], "area_mm2": 700,
         "centroid": [0, 0, 0]},
        {"type": "cylinder", "axis": axis1, "axis_point": list(point1),
         "radius_mm": 15.0, "area_mm2": 3800, "centroid": list(point1)},
        {"type": "cylinder", "axis": axis2, "axis_point": list(point2),
         "radius_mm": 10.0, "area_mm2": 1900, "centroid": list(point2)},
    ]}


# ---------------------------------------------------------------------------
# plan-core-09 / -10 / -11: validate() rejects what the builder cannot take


def _validate(plan):
    ok, errors, warnings = schema.validate(plan)
    return ok, errors, warnings


def _mutate(base, fn):
    plan = copy.deepcopy(base)
    fn(plan)
    return plan


class SchemaHardeningTest(unittest.TestCase):
    PLATE = _example("plate_4holes")
    SHAFT = _example("stepped_shaft")

    def assertRejected(self, plan, *needles):
        ok, errors, _w = _validate(plan)
        self.assertFalse(ok, "plan validated but must not")
        joined = "\n".join(errors)
        for needle in needles:
            self.assertIn(needle, joined)

    # -- 09 ------------------------------------------------------------------

    def test_non_finite_parameter_values_are_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf"), 10 ** 400):
            with self.subTest(value=bad):
                plan = _mutate(self.PLATE, lambda p: p["parameters"][
                    "thickness"].__setitem__("value", bad))
                self.assertRejected(plan, "thickness", "finite")

    def test_non_finite_coordinates_are_rejected(self):
        plan = _mutate(self.PLATE, lambda p: p["features"][3]["position"]
                       .__setitem__("y", float("inf")))
        self.assertRejected(plan, "hole_b")

    def test_units_must_be_millimetres(self):
        for bad in ("in", "banana", "cm"):
            with self.subTest(units=bad):
                self.assertRejected(_mutate(self.PLATE, lambda p: p
                                            .__setitem__("units", bad)),
                                    "units")
        for good in ("mm", "MM", "millimeters", "Millimetre"):
            with self.subTest(units=good):
                ok, errors, _w = _validate(_mutate(
                    self.PLATE, lambda p: p.__setitem__("units", good)))
                self.assertTrue(ok, errors)

    # -- 10 ------------------------------------------------------------------

    def test_names_freecad_reserves_are_rejected(self):
        for name in ("A1", "R1", "AB12", "ZZ99", "_wd", "__x", "pi", "mm",
                     "cm", "min", "mil", "bar", "kg", "Hz", "mph", "naïve"):
            with self.subTest(name=name):
                self.assertFalse(schema._valid_param_name(name), name)

    def test_planner_style_names_stay_valid(self):
        for name in ("hole_dia", "x1", "r1", "h1", "step_z1", "bore1_dia",
                     "thickness", "rev_360", "profile_len", "z_total",
                     "hole_dx", "leg_t", "arc_r"):
            with self.subTest(name=name):
                self.assertTrue(schema._valid_param_name(name), name)

    def test_non_string_parameter_name_is_an_error_not_a_crash(self):
        plan = _mutate(self.PLATE, lambda p: p["parameters"].__setitem__(
            5, {"value": 1, "assumption": "x"}))
        self.assertRejected(plan, "5")

    # -- 11 ------------------------------------------------------------------

    def test_unknown_parameter_references_are_named_at_validation(self):
        cases = (
            ("pad length", lambda p: p["features"][1].__setitem__(
                "length", "thicknes"), ("base_pad", "thicknes")),
            ("rect width", lambda p: p["features"][0]["profile"].__setitem__(
                "width", "nope"), ("base_sketch", "nope")),
            ("hole diameter", lambda p: p["features"][2].__setitem__(
                "diameter", "nope"), ("hole_a", "nope")),
            ("hole x", lambda p: p["features"][2]["position"].__setitem__(
                "x", "nope"), ("hole_a", "nope")),
            ("profile centre", lambda p: p["features"][0]["profile"]
             .__setitem__("center", ["nope", 0]), ("base_sketch", "nope")),
        )
        for label, fn, needles in cases:
            with self.subTest(case=label):
                self.assertRejected(_mutate(self.PLATE, fn), *needles)

    def test_unknown_parameter_in_a_revolution_and_slot_and_arc(self):
        self.assertRejected(_mutate(self.SHAFT, lambda p: p["features"][1]
                                    .__setitem__("angle", "nope")),
                            "revolve", "nope")
        self.assertRejected(_slot_plan(["nope", 0]), "nope")
        plan = _dshape(True, shallow=True)
        plan["features"][0]["profile"]["segments"][1]["radius"] = "nope"
        self.assertRejected(plan, "nope")

    def test_required_fields_are_enforced(self):
        self.assertRejected(_mutate(self.PLATE, lambda p: p["features"][1]
                                    .pop("length")), "base_pad", "length")
        self.assertRejected(_mutate(self.PLATE, lambda p: p["features"][2]
                                    .pop("diameter")), "hole_a", "diameter")
        self.assertRejected(_mutate(self.PLATE, lambda p: p["features"][2]
                                    .pop("through")), "hole_a", "through")
        plan = _slab({"pd": 3}, [
            {"id": "c", "op": "sketch", "plane": dict(_TOP),
             "profile": {"shape": "circle", "diameter": "wid"}},
            {"id": "k", "op": "pocket", "sketch": "c"}])
        self.assertRejected(plan, "k", "depth")

    def test_revolution_axis_must_exist(self):
        self.assertRejected(_mutate(self.SHAFT, lambda p: p["features"][1][
            "axis"].__setitem__("frame_axis", "nope")), "revolve", "nope")

    def test_degenerate_planes_are_rejected(self):
        self.assertRejected(_mutate(self.PLATE, lambda p: p["features"][0]
                                    .__setitem__("plane", {"normal": [0, 0, 0]})),
                            "base_sketch", "normal")
        self.assertRejected(_mutate(self.PLATE, lambda p: p["features"][0]
                                    .__setitem__("plane", {
                                        "normal": [0, 0, 1],
                                        "x_dir": [0, 0, 2]})),
                            "base_sketch", "x_dir")
        self.assertRejected(_mutate(self.PLATE, lambda p: p["frame"][
            "planes"]["bottom"].__setitem__("x_dir", [0, 0, 0])), "bottom")

    def test_malformed_shapes_are_errors_not_exceptions(self):
        cases = (
            lambda p: p["features"][1].__setitem__("sketch", ["base_sketch"]),
            lambda p: p["features"][0].__setitem__(
                "plane", {"frame_plane": ["bottom"]}),
            lambda p: p["frame"].__setitem__("planes", None),
            lambda p: p["frame"].__setitem__("planes", "x"),
            lambda p: p["frame"].__setitem__("axes", "x"),
            lambda p: p["frame"].__setitem__("planes", ["a"]),
            lambda p: p["features"].append(None),
            lambda p: p["features"].append([1]),
            lambda p: p["frame"]["planes"].__setitem__("zz", "str"),
            lambda p: p["features"][0].__setitem__("plane", "base"),
            lambda p: p["parameters"]["length"].__setitem__("evidence", "s"),
        )
        for i, fn in enumerate(cases):
            with self.subTest(case=i):
                plan = _mutate(self.PLATE, fn)
                try:
                    ok, errors, _w = _validate(plan)
                except Exception as e:  # noqa: BLE001
                    self.fail(f"validate raised {type(e).__name__}: {e}")
                self.assertFalse(ok)

    def test_validate_never_raises_on_mutated_plans(self):
        values = [None, True, 0, -1, 2.5, float("nan"), float("inf"),
                  10 ** 400, "", "x", "length", [], [1], ["a", "b"], {},
                  {"a": 1}]

        def paths(node, trail=()):
            yield trail
            if isinstance(node, dict):
                for k, v in node.items():
                    yield from paths(v, trail + (k,))
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    yield from paths(v, trail + (i,))

        def at(plan, trail):
            for k in trail[:-1]:
                plan = plan[k]
            return plan

        count = 0
        for name in ("plate_4holes", "l_bracket", "stepped_shaft"):
            base = _example(name)
            for trail in paths(base):
                if not trail:
                    continue
                for value in values + ["<delete>"]:
                    plan = copy.deepcopy(base)
                    holder = at(plan, trail)
                    if value == "<delete>":
                        if isinstance(holder, dict):
                            del holder[trail[-1]]
                        else:
                            continue
                    else:
                        holder[trail[-1]] = value
                    count += 1
                    try:
                        schema.validate(plan)
                    except Exception as e:  # noqa: BLE001
                        self.fail(f"validate raised {type(e).__name__}: {e} "
                                  f"for {name} at {trail} = {value!r}")
        self.assertGreater(count, 1000)


class ResolveErrorTest(unittest.TestCase):
    def test_unresolvable_value_names_its_feature(self):
        plan = _example("plate_4holes")
        with self.assertRaises(builder.BuildError) as cm:
            _buildcore.resolve(plan, "nope", "base_pad length", "base_pad")
        self.assertEqual(cm.exception.feature_id, "base_pad")


class SpreadsheetAliasTest(_DocCase):
    def test_a_name_freecad_refuses_is_a_structured_build_failure(self):
        # validate() cannot list every reserved name; when one slips through
        # the failure is reported, not raised, and no half-filled sheet stays
        plan = _example("plate_4holes")
        plan["parameters"]["A1"] = plan["parameters"].pop("hole_dia")
        for feat in plan["features"]:
            if feat.get("diameter") == "hole_dia":
                feat["diameter"] = "A1"
        original = schema._valid_param_name
        schema._valid_param_name = lambda name: True
        self.addCleanup(setattr, schema, "_valid_param_name", original)
        doc = self._doc()
        report = builder.apply_plan(doc, plan)
        self.assertFalse(report["success"])
        # our result contract names this stage "parameters" (a sheet write
        # failed before any feature ran); theirs lumped it under "build"
        self.assertEqual(report["stage"], "parameters")
        self.assertIn("A1", report["error"])
        self.assertEqual([o.Name for o in doc.Objects], [])

    def test_a_refused_new_name_leaves_the_existing_sheet_untouched(self):
        doc, _ = self._build(_example("plate_4holes"))
        sheet = doc.getObject("Parameters")
        cells = sorted(sheet.getUsedCells())
        patched = _example("plate_4holes")
        patched["parameters"]["A1"] = {"value": 6.0, "assumption": "x"}
        patched["features"][-1]["diameter"] = "A1"
        original = schema._valid_param_name
        schema._valid_param_name = lambda name: True
        self.addCleanup(setattr, schema, "_valid_param_name", original)
        result = builder.apply_feature(doc, patched, "hole_d")
        self.assertFalse(result["success"])
        self.assertIn("A1", result["error"])
        self.assertEqual(sorted(sheet.getUsedCells()), cells)


# ---------------------------------------------------------------------------
# second review pass: plan-01 (the cleanup deleted a user's own part),
# plan-02/03 (the setup and the parameter sync escaped the undo stack),
# plan-05 (the coaxial angle tolerance was defeated by the position one)


def _add_users_own_part(doc):
    """What the user (or Oscar's execute_code) builds: the standard FreeCAD
    parameter sheet - label in A, value in B aliased by that label - and a
    Body holding a sketch and a pad."""
    import Part

    sheet = doc.addObject("Spreadsheet::Sheet", "Parameters")
    for row, (label, value) in enumerate((("Length", 50), ("Width", 30)),
                                         start=1):
        sheet.set(f"A{row}", label)
        sheet.set(f"B{row}", str(value))
        sheet.setAlias(f"B{row}", label)
    doc.recompute()
    body = doc.addObject("PartDesign::Body", "Body")
    sketch = body.newObject("Sketcher::SketchObject", "MySketch")
    corners = [(0, 0), (50, 0), (50, 30), (0, 30)]
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        sketch.addGeometry(Part.LineSegment(App.Vector(a[0], a[1], 0),
                                            App.Vector(b[0], b[1], 0)))
    pad = body.newObject("PartDesign::Pad", "MyPad")
    pad.Profile = sketch
    pad.Length = 10
    doc.recompute()
    return body


def _run_tool(name: str, arguments: dict):
    """tools.run_tool without a 3D view to screenshot (console mode) and
    without appending to the on-disk activity log."""
    with mock.patch.object(tools, "_screenshot", return_value=None), \
            mock.patch("oscar.turnlog.log_event"):
        return tools.run_tool(name, arguments)


class UserPartSurvivesApplyTest(_DocCase):
    PLATE_VOLUME = 100 * 60 * 8 - 4 * math.pi * 25 * 8

    def _assert_users_part_intact(self, doc):
        for name in ("Parameters", "Body", "MySketch", "MyPad"):
            self.assertIsNotNone(doc.getObject(name), f"{name} was deleted")
        self.assertEqual(doc.getObject("Parameters").getAlias("B1"), "Length")
        self.assertAlmostEqual(doc.getObject("Body").Shape.Volume, 15000.0,
                               delta=1.0)

    def test_users_parameters_sheet_and_body_survive_apply_plan(self):
        doc = self._doc()
        _add_users_own_part(doc)
        _doc, report = self._build(_example("plate_4holes"), doc=doc)
        # the new build is its own sheet + Body, next to the user's
        self.assertEqual(report["spreadsheet"], "Parameters001")
        self.assertEqual(report["body"], "Body001")
        self._assert_users_part_intact(doc)
        self.assertVolume(doc, self.PLATE_VOLUME, delta=2.0, name="Body001")

    def test_a_genuine_previous_build_is_replaced_next_to_the_users_part(self):
        doc = self._doc()
        _add_users_own_part(doc)
        self._build(_example("plate_4holes"), doc=doc)
        changed = _example("plate_4holes")
        changed["parameters"]["thickness"]["value"] = 12.0
        _doc, report = self._build(changed, doc=doc)
        self.assertEqual(_types(doc, "Spreadsheet::Sheet"),
                         ["Parameters", "Parameters001"])
        self.assertEqual(_types(doc, "PartDesign::Body"), ["Body", "Body001"])
        self.assertVolume(doc, 100 * 60 * 12 - 4 * math.pi * 25 * 12,
                          delta=2.0, name=report["body"])
        self._assert_users_part_intact(doc)

    def test_the_marker_survives_saving_and_reopening(self):
        doc, _ = self._build(_example("plate_4holes"))
        folder = tempfile.mkdtemp(prefix="plan_marker_")
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        path = os.path.join(folder, "built.FCStd")
        doc.saveAs(path)
        App.closeDocument(doc.Name)
        reopened = App.openDocument(path)
        self.addCleanup(self._close, reopened.Name)
        changed = _example("plate_4holes")
        changed["parameters"]["thickness"]["value"] = 12.0
        _doc, report = self._build(changed, doc=reopened)
        self.assertEqual(_types(reopened, "PartDesign::Body"), ["Body"])
        self.assertEqual(_types(reopened, "Spreadsheet::Sheet"), ["Parameters"])
        self.assertVolume(reopened, 100 * 60 * 12 - 4 * math.pi * 25 * 12,
                          delta=2.0, name=report["body"])

    def test_a_build_without_the_plan_marker_is_left_alone(self):
        # a document saved by the builder before it stamped its output: the
        # sheet and Body cannot be proven ours, so they are not guessed at
        doc, _ = self._build(_example("plate_4holes"))
        for name in ("Parameters", "Body"):
            doc.getObject(name).Label2 = ""
        _doc, report = self._build(_example("plate_4holes"), doc=doc)
        self.assertEqual(report["spreadsheet"], "Parameters001")
        self.assertEqual(report["body"], "Body001")
        self.assertIsNotNone(doc.getObject("base_pad"))
        self.assertVolume(doc, self.PLATE_VOLUME, delta=2.0, name="Body")
        self.assertVolume(doc, self.PLATE_VOLUME, delta=2.0, name="Body001")

    def test_apply_feature_never_adopts_the_users_sheet_or_body(self):
        doc = self._doc()
        _add_users_own_part(doc)
        # the user's sheet shares aliases with the plan on purpose
        users = doc.getObject("Parameters")
        users.set("A3", "hole_dx")
        users.set("B3", "99")
        users.setAlias("B3", "hole_dx")
        doc.recompute()
        users_tree = [o.Name for o in doc.getObject("Body").Group]
        _doc, report = self._build(_example("plate_4holes"), doc=doc)
        patched = _example("plate_4holes")
        patched["parameters"]["hole_dx"]["value"] = 35.0
        patched["features"][-1]["position"].update(x="hole_dx", y=0.0)  # y=0: clear of hole_b, which hole_dx also drives
        result = builder.apply_feature(doc, patched, "hole_d")
        self.assertTrue(result["success"], result)
        self.assertEqual(doc.getObject("Parameters").getContents("B3"), "99")
        self.assertEqual([o.Name for o in doc.getObject("Body").Group],
                         users_tree)
        self.assertIn("hole_d", [o.Name for o in
                                 doc.getObject(report["body"]).Group])
        mine = doc.getObject(report["spreadsheet"])
        self.assertEqual(float(mine.getContents(
            mine.getCellFromAlias("hole_dx"))), 35.0)


class UndoCoversTheWholeApplyTest(_DocCase):
    """The interactive tool runs with an undo stack: every step of an apply is an 'Oscar: plan ...' transaction, one
    per feature, so a turn's plan work is recognisable on the stack. Revert-turn never pops these with
    ``Document.undo()`` (FreeCAD 1.1 can crash in Transaction::apply there, P0 #4): it reports ``undo_unsafe``, changes
    nothing, and the panel restores the turn checkpoint instead (test_project_rewind.PlanBuildTurnRevertTest)."""

    def _undo_doc(self):
        doc = self._doc("UndoRC")
        doc.UndoMode = 1
        return doc

    def test_a_reapply_is_one_labelled_step_per_feature(self):
        doc = self._undo_doc()
        self._build(_example("plate_4holes"), doc=doc)
        depth = doc.UndoCount
        changed = _example("plate_4holes")
        changed["parameters"]["thickness"]["value"] = 12.0
        self._build(changed, doc=doc)
        self.assertEqual(_new_labels(doc, depth), PLATE_LABELS)
        self.assertEqual(_types(doc, "PartDesign::Body"), ["Body"])
        self.assertEqual(_types(doc, "Spreadsheet::Sheet"), ["Parameters"])
        self.assertVolume(doc, 100 * 60 * 12 - 4 * math.pi * 25 * 12, delta=2.0)

    def test_revert_turn_after_a_tool_reapply_is_refused_as_undo_unsafe(self):
        from oscar.cad import get_engine

        doc = self._undo_doc()
        self._build(_example("plate_4holes"), doc=doc)
        engine = get_engine()
        snapshot = engine.begin_turn_snapshot()
        changed = _example("plate_4holes")
        changed["parameters"]["thickness"]["value"] = 12.0
        text, _img, is_error = _run_tool(
            "apply_feature_plan", {"doc_name": doc.Name, "plan": changed})
        self.assertFalse(is_error, text)
        before = (doc.UndoCount, list(doc.UndoNames), sorted(o.Name for o in doc.Objects))
        report = engine.revert_turn(snapshot)
        self.assertFalse(report["success"])
        self.assertTrue(report["incomplete"] and report["undo_unsafe"], report)
        self.assertEqual((report["undone"], report["unsafe_docs"]), ({}, [doc.Name]))
        self.assertIn("restore the turn checkpoint", report["error"])
        self.assertEqual((doc.UndoCount, list(doc.UndoNames), sorted(o.Name for o in doc.Objects)), before)
        self.assertEqual(_types(doc, "PartDesign::Body"), ["Body"])
        self.assertVolume(doc, 100 * 60 * 12 - 4 * math.pi * 25 * 12, delta=2.0)

    def test_a_first_apply_is_one_labelled_step_per_feature(self):
        doc = self._undo_doc()
        depth = doc.UndoCount
        self._build(_example("plate_4holes"), doc=doc)
        self.assertEqual(_new_labels(doc, depth), PLATE_LABELS)
        from oscar.cad.engine import is_undo_unsafe

        self.assertTrue(all(is_undo_unsafe(label) for label in PLATE_LABELS))

    def test_a_failed_reapply_setup_leaves_the_previous_build_in_place(self):
        doc = self._undo_doc()
        self._build(_example("plate_4holes"), doc=doc)
        bad = _example("plate_4holes")
        bad["parameters"]["A1"] = bad["parameters"].pop("hole_dia")
        for feat in bad["features"]:
            if feat.get("diameter") == "hole_dia":
                feat["diameter"] = "A1"
        original = schema._valid_param_name
        schema._valid_param_name = lambda name: True
        self.addCleanup(setattr, schema, "_valid_param_name", original)
        report = builder.apply_plan(doc, bad)
        self.assertFalse(report["success"])
        doc.recompute()
        self.assertEqual(_types(doc, "PartDesign::Body"), ["Body"])
        self.assertEqual(_types(doc, "Spreadsheet::Sheet"), ["Parameters"])
        self.assertVolume(doc, 100 * 60 * 8 - 4 * math.pi * 25 * 8, delta=2.0)


class ApplyFeatureAtomicTest(_DocCase):
    PLATE_VOLUME = 100 * 60 * 8 - 4 * math.pi * 25 * 8

    def _cell(self, doc, alias):
        sheet = doc.getObject("Parameters")
        return float(sheet.getContents(sheet.getCellFromAlias(alias)))

    def test_a_rejected_call_leaves_the_sheet_and_model_alone(self):
        doc, _ = self._build(_example("plate_4holes"))
        patched = _example("plate_4holes")
        patched["parameters"]["thickness"]["value"] = 20.0
        result = builder.apply_feature(doc, patched, "base_pad")
        self.assertFalse(result["success"])
        self.assertIn("later features exist", result["error"])
        self.assertEqual(self._cell(doc, "thickness"), 8.0)
        self.assertVolume(doc, self.PLATE_VOLUME, delta=2.0)

    def test_unknown_op_leaves_the_sheet_alone(self):
        doc, _ = self._build(_example("plate_4holes"))
        patched = _example("plate_4holes")
        patched["parameters"]["hole_dx"]["value"] = 35.0
        with mock.patch.dict(_featurebuild.SOLID_BUILDERS, clear=True):
            result = builder.apply_feature(doc, patched, "hole_d")
        self.assertFalse(result["success"])
        self.assertIn("unknown", result["error"])
        self.assertEqual(self._cell(doc, "hole_dx"), 30.0)

    def test_a_feature_that_fails_to_build_puts_the_sheet_back(self):
        doc, _ = self._build(_example("plate_4holes"))
        patched = _example("plate_4holes")
        patched["parameters"]["hole_dx"]["value"] = 35.0
        patched["parameters"]["extra_dia"] = {"value": 6.0,
                                              "assumption": "test fixture dimension"}
        patched["features"][-1]["diameter"] = "extra_dia"
        with mock.patch.dict(
                _featurebuild.SOLID_BUILDERS,
                {"hole": mock.Mock(side_effect=RuntimeError("boom"))}):
            result = builder.apply_feature(doc, patched, "hole_d")
        self.assertFalse(result["success"])
        self.assertIn("boom", result["error"])
        sheet = doc.getObject("Parameters")
        self.assertEqual(self._cell(doc, "hole_dx"), 30.0)
        self.assertIsNone(sheet.getCellFromAlias("extra_dia"))
        self.assertEqual(sorted(sheet.getUsedCells()),
                         sorted(f"{c}{r}" for c in "AB" for r in range(1, 7)))

    def test_an_apply_feature_turn_is_one_labelled_plan_step(self):
        # the sheet change and the rebuilt feature are ONE undo step labelled as plan work, which a revert restores
        # from the turn checkpoint instead of popping it (FreeCAD can crash undoing PartDesign plan steps, P0 #4)
        doc, _ = self._build(_example("plate_4holes"))
        doc.UndoMode = 1
        depth = doc.UndoCount
        patched = _example("plate_4holes")
        patched["parameters"]["hole_dx"]["value"] = 35.0
        patched["features"][-1]["position"].update(x="hole_dx", y=0.0)  # y=0: clear of hole_b, which hole_dx also drives
        result = builder.apply_feature(doc, patched, "hole_d")
        self.assertTrue(result["success"], result)
        self.assertEqual(self._cell(doc, "hole_dx"), 35.0)
        circle = doc.getObject("hole_d_sketch").Geometry[0]
        self.assertAlmostEqual(circle.Center.x, 35.0, delta=1e-6)
        self.assertEqual(_new_labels(doc, depth), ["Oscar: plan hole_d"])

    def test_a_failed_apply_feature_with_undo_on_changes_nothing(self):
        doc, _ = self._build(_example("plate_4holes"))
        doc.UndoMode = 1
        patched = _example("plate_4holes")
        patched["parameters"]["hole_dx"]["value"] = 35.0
        with mock.patch.dict(
                _featurebuild.SOLID_BUILDERS,
                {"hole": mock.Mock(side_effect=RuntimeError("boom"))}):
            result = builder.apply_feature(doc, patched, "hole_d")
        self.assertFalse(result["success"])
        self.assertEqual(self._cell(doc, "hole_dx"), 30.0)
        self.assertIsNotNone(doc.getObject("hole_d"))
        self.assertVolume(doc, self.PLATE_VOLUME, delta=2.0)

    def test_the_tool_tells_the_agent_which_parameters_were_reset(self):
        doc, _ = self._build(_example("plate_4holes"))
        patched = _example("plate_4holes")
        patched["parameters"]["hole_dx"]["value"] = 35.0
        patched["features"][-1]["position"].update(x="hole_dx", y=0.0)  # y=0: clear of hole_b, which hole_dx also drives
        text, _img, is_error = _run_tool("reapply_plan_feature", {
            "doc_name": doc.Name, "plan": patched, "feature_id": "hole_d"})
        self.assertFalse(is_error, text)
        self.assertIn("hole_dx", text)

    def test_the_plan_tool_says_a_previous_build_is_replaced(self):
        spec = next(t for t in tools.get_tools()
                    if t.name == "apply_feature_plan")
        self.assertIn("replaced", spec.description)


# ---------------------------------------------------------------------------
# reconciliation guards of OUR idempotent apply_plan (plan tag + PLAN_MARK):
# what their marker-based replacement protected must hold here too


class ReplaceOnlyWhatWeBuiltTest(_DocCase):
    PLATE_VOLUME = 100 * 60 * 8 - 4 * math.pi * 25 * 8

    def test_a_stamped_sheet_the_user_added_cells_to_is_not_replaced(self):
        doc, _ = self._build(_example("plate_4holes"))
        sheet = doc.getObject("Parameters")
        sheet.set("D1", "my own note")
        doc.recompute()
        changed = _example("plate_4holes")
        changed["parameters"]["thickness"]["value"] = 12.0
        _doc, report = self._build(changed, doc=doc)
        self.assertEqual(report["spreadsheet"], "Parameters001")
        self.assertEqual(doc.getObject("Parameters").getContents("D1").lstrip("'"),
                         "my own note")
        self.assertEqual(_types(doc, "PartDesign::Body"), ["Body"])
        self.assertVolume(doc, 100 * 60 * 12 - 4 * math.pi * 25 * 12,
                          delta=2.0, name=report["body"])

    def test_features_the_user_added_to_our_body_block_the_replacement(self):
        doc, _ = self._build(_example("plate_4holes"))
        body = doc.getObject("Body")
        body.newObject("PartDesign::Fillet", "MyFillet")
        changed = _example("plate_4holes")
        changed["parameters"]["thickness"]["value"] = 12.0
        report = builder.apply_plan(doc, changed)
        self.assertFalse(report["success"])
        self.assertEqual(report["stage"], "document")
        self.assertIn("MyFillet", report["error"])
        self.assertIsNotNone(doc.getObject("MyFillet"))
        self.assertIsNotNone(doc.getObject("base_pad"))

    def test_an_explicit_name_of_a_users_part_is_refused_everywhere(self):
        doc = self._doc()
        _add_users_own_part(doc)
        refused = builder.apply_plan(doc, _example("plate_4holes"),
                                     body_name="Body")
        self.assertFalse(refused["success"])
        self.assertEqual(refused["stage"], "document")
        _doc, report = self._build(_example("plate_4holes"), doc=doc)
        self.assertEqual(report["body"], "Body001")
        for fn in (builder.apply_feature, builder.rebuild_from):
            result = fn(doc, _example("plate_4holes"), "hole_d",
                        body_name="Body")
            self.assertFalse(result["success"], fn.__name__)
            self.assertEqual(result["stage"], "document", fn.__name__)
        self.assertEqual([o.Name for o in doc.getObject("Body").Group
                          if not o.TypeId.startswith("App::")], ["MySketch", "MyPad"])

    def test_a_failed_reapply_without_an_undo_stack_keeps_the_previous_build(self):
        # the parameters are dry-run before the previous build is removed: a
        # console document cannot take the removal back
        doc, _ = self._build(_example("plate_4holes"))
        self.assertEqual(doc.UndoMode, 0)
        bad = _example("plate_4holes")
        bad["parameters"]["A1"] = bad["parameters"].pop("hole_dia")
        for feat in bad["features"]:
            if feat.get("diameter") == "hole_dia":
                feat["diameter"] = "A1"
        original = schema._valid_param_name
        schema._valid_param_name = lambda name: True
        self.addCleanup(setattr, schema, "_valid_param_name", original)
        report = builder.apply_plan(doc, bad)
        self.assertFalse(report["success"])
        self.assertEqual(report["stage"], "parameters")
        self.assertEqual(report["replaced"], [])
        self.assertEqual(_types(doc, "PartDesign::Body"), ["Body"])
        self.assertEqual(_types(doc, "Spreadsheet::Sheet"), ["Parameters"])
        self.assertVolume(doc, self.PLATE_VOLUME, delta=2.0)

    def test_a_users_object_named_like_a_later_feature_is_not_ours(self):
        doc, _ = self._build(_example("plate_4holes"))
        doc.removeObject("hole_d")                       # the Body lost its last hole
        doc.addObject("Part::Feature", "hole_d")         # a user's, outside the Body
        result = builder.apply_feature(doc, _example("plate_4holes"), "hole_c")
        # that object is not a "later feature" of the build and is never deleted
        self.assertNotIn("later features exist", result.get("error", ""), result)
        self.assertEqual(doc.getObject("hole_d").TypeId, "Part::Feature")

    def test_the_parameter_probe_leaves_no_object_behind(self):
        doc, _ = self._build(_example("plate_4holes"))
        self.assertEqual(_types(doc, "Spreadsheet::Sheet"), ["Parameters"])
        names = {o.Name for o in doc.Objects}
        self.assertFalse([n for n in names if "Probe" in n], names)


@unittest.skipUnless(os.environ.get("OSCAR_TEST_PD_UNDO") == "1",
                     "opt-in reproduction of FreeCAD's undo crash on plan builds (set OSCAR_TEST_PD_UNDO=1)")
class PlanUndoCrashReproductionTest(unittest.TestCase):
    """The upstream reproduction of P0 #4, in a helper process so a crash kills only the helper: build plate_4holes
    with UndoMode 1 and undo all its steps with Document.undo(), twelve times. A crash comes back as the helper's
    signal (``crashed``)."""

    def test_undoing_plan_builds_in_a_helper_process(self):
        from oscar.cad import headless

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = (
            "import json, sys\n"
            "sys.dont_write_bytecode = True\n"
            f"sys.path.insert(0, {root!r})\n"
            "import FreeCAD as App\n"
            "from oscar.plan import builder\n"
            f"plan = json.load(open({os.path.join(EXAMPLES, 'plate_4holes.json')!r}, encoding='utf-8'))\n"
            "for round_ in range(12):\n"
            "    doc = App.newDocument('Repro%d' % round_)\n"
            "    doc.UndoMode = 1\n"
            "    assert builder.apply_plan(doc, plan)['success']\n"
            "    for _ in range(doc.UndoCount):\n"
            "        doc.undo()\n"
            "    App.closeDocument(doc.Name)\n"
            "print('survived')\n")
        result = headless.run(code, timeout=900, allow_network=False)
        self.assertFalse(result.get("crashed"), result.get("error"))
        self.assertTrue(result["success"], result.get("error"))
        self.assertIn("survived", result["output"])


if __name__ == "__main__":
    unittest.main()
