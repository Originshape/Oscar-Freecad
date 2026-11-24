"""Sketch / constraint intelligence of the plan builder: arcs carry a size
constraint (radius or the named diameter), a sketch that solves but is not
fully constrained is refused with a repair hint before any dependent feature
consumes it, conflicting constraints are named, redundancy is counted, and a
sketch no feature consumes is flagged at validation.

Every defect case here was reproduced on FreeCAD 1.1.4 before its fix
(probe 2026-11-22): the radius-less arc built silently with
``fully_constrained: false``, a declared segment ``length`` that contradicted
the geometry passed validation, an off-circle arc center failed deep in the
solver ("SetRadius() - radius should be positive") and a dead sketch built
without a word.

Runs in console mode via tests/run.py (Sketcher/PartDesign are App-level;
no GUI needed)."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

import FreeCAD as App  # noqa: E402

from oscar.plan import _buildcore, _sketchbuild, builder  # noqa: E402


def _p(value: float) -> dict:
    return {"value": float(value), "assumption": "unit test fixture value"}


def _plate(plan_id: str = "intel_plate", **extra_params) -> dict:
    """A 60 x 40 x 10 plate (centred on the origin) with a ``top`` plane at
    z = thickness for sketches that cut downwards."""
    params = {"plate_len": _p(60), "plate_wid": _p(40),
              "thickness": _p(10)}
    params.update({k: _p(v) for k, v in extra_params.items()})
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


def _rounded_corner_pocket(plan_id, derive_from="p_r", **params) -> dict:
    """A pocket whose profile is a rectangle with one rounded corner: the
    arc's end points and centre derive from the size parameter (the same
    derivation the planner is taught), so the arc is exactly determined —
    with or without the radius named on the segment."""
    plan = _plate(plan_id, **params)
    plan["parameters"]["p_hr"] = {"expr": f"p_hgt - {derive_from}"}
    plan["parameters"]["p_lr"] = {"expr": f"p_len - {derive_from}"}
    plan["features"] += [
        {"id": "d_sketch", "op": "sketch", "plane": {"frame_plane": "top"},
         "profile": {"shape": "profile", "start": [0, 0], "segments": [
             {"type": "line", "to": ["p_len", 0]},
             {"type": "line", "to": ["p_len", "p_hr"]},
             {"type": "arc", "to": ["p_lr", "p_hgt"], "center": ["p_lr", "p_hr"],
              "ccw": True},
             {"type": "line", "to": [0, "p_hgt"]},
             {"type": "line", "to": [0, 0]}]}},
        {"id": "d_pocket", "op": "pocket", "sketch": "d_sketch", "depth": "p_h"}]
    return plan


class _DocTest(unittest.TestCase):
    """Fresh documents that are always closed again."""

    def new_doc(self, name: str = "IntelDoc"):
        doc = App.newDocument(name)
        self.addCleanup(self._close, doc.Name)
        return doc

    @staticmethod
    def _close(name: str) -> None:
        if name in App.listDocuments():
            App.closeDocument(name)

    def build(self, plan: dict, name: str = "IntelBuild", ok: bool = True):
        doc = self.new_doc(name)
        report = builder.apply_plan(doc, plan)
        if ok:
            self.assertTrue(report.get("success"),
                            f"{report.get('stage')}/{report.get('feature_id')}: "
                            f"{report.get('error')}")
        return doc, report

    @staticmethod
    def sketch_info(report, fid: str) -> dict:
        return next(f["info"] for f in report["features"] if f["id"] == fid)

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


class ArcSizeConstraintTest(_DocTest):
    """An arc's size must be bound by a constraint (radius, or the named
    diameter) — an arc without one leaves the profile under-constrained."""

    def test_an_arc_without_a_size_is_refused_with_a_repair_hint(self):
        plan = _rounded_corner_pocket("intel_no_rad", p_len=30, p_hgt=20,
                                      p_r=6, p_h=5)
        doc, report = self.build(plan, "ArcNoRad", ok=False)
        self.assertFalse(report["success"])
        self.assertEqual(report["feature_id"], "d_sketch")
        self.assertIn("not fully constrained", report["error"])
        self.assertIn("radius", report["error"])
        self.assertIn("two quarter arcs", report["error"])
        self.assertEqual(report["completed"], ["base_sketch", "base_pad"])
        self.assertIsNone(doc.getObject("d_sketch"))

    def test_the_repaired_plan_with_a_named_radius_builds_and_resizes(self):
        plan = _rounded_corner_pocket("intel_rad", p_len=30, p_hgt=20,
                                      p_r=6, p_h=5)
        plan["features"][2]["profile"]["segments"][2]["radius"] = "p_r"
        doc, report = self.build(plan, "ArcRad")
        info = self.sketch_info(report, "d_sketch")
        self.assertTrue(info["fully_constrained"], info)
        self.assertNotIn("redundant", info)
        area = 30 * 20 - (36 - math.pi * 36 / 4)
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME - area * 5, delta=0.5)
        self.set_param(doc, report["spreadsheet"], "p_r", 8)
        area = 30 * 20 - (64 - math.pi * 64 / 4)
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME - area * 5, delta=0.5)

    def test_an_arc_with_a_diameter_parameter_binds_and_resizes(self):
        # the corner geometry derives from the diameter, so a diameter edit
        # keeps the arc tangent (p_dia = 2 x the corner radius throughout)
        plan = _rounded_corner_pocket("intel_diam", derive_from="p_dia / 2",
                                      p_len=30, p_hgt=20, p_h=5, p_dia=12)
        plan["features"][2]["profile"]["segments"][2]["diameter"] = "p_dia"
        doc, report = self.build(plan, "ArcDiam")
        info = self.sketch_info(report, "d_sketch")
        self.assertTrue(info["fully_constrained"], info)
        self.assertGreaterEqual(info["dimensional_expressions"], 3)
        area = 30 * 20 - (36 - math.pi * 36 / 4)
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME - area * 5, delta=0.5)
        self.set_param(doc, report["spreadsheet"], "p_dia", 16)   # r 6 -> 8
        area = 30 * 20 - (64 - math.pi * 64 / 4)
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME - area * 5, delta=0.5)


class SolverVerdictTest(_DocTest):
    """The sketch verdict gate: solved-but-under-constrained is refused;
    conflicting constraints are named; redundancy is counted."""

    def sketch(self, name: str):
        doc = self.new_doc(name)
        sketch = doc.addObject("Sketcher::SketchObject", "S")
        return doc, sketch

    def test_a_solved_but_under_constrained_sketch_is_refused(self):
        import Part
        doc, sketch = self.sketch("UnderCon")
        sketch.addGeometry(
            Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 0, 1), 5.0), False)
        doc.recompute()
        with self.assertRaises(_buildcore.BuildError) as caught:
            _sketchbuild.sketch_report(sketch, "circle", 0, "gate_probe")
        self.assertIn("not fully constrained", str(caught.exception))

    def test_conflicting_constraints_are_named(self):
        import Part
        import Sketcher
        doc, sketch = self.sketch("Conflict")
        i1 = sketch.addGeometry(
            Part.LineSegment(App.Vector(0, 0, 0), App.Vector(10, 0, 0)), False)
        for value in (10.0, 7.0):        # the second contradicts the first
            i = sketch.addConstraint(
                Sketcher.Constraint("DistanceX", i1, 1, i1, 2, value))
            sketch.renameConstraint(i, f"width_{int(value)}")
        doc.recompute()
        with self.assertRaises(_buildcore.BuildError) as caught:
            _sketchbuild.sketch_report(sketch, "line", 0, "conflict_probe")
        message = str(caught.exception)
        self.assertIn("conflict", message)
        self.assertIn("width_10", message)

    def test_redundant_constraints_are_counted_in_the_report(self):
        import Part
        import Sketcher
        doc, sketch = self.sketch("Redundant")
        i1 = sketch.addGeometry(
            Part.LineSegment(App.Vector(0, 0, 0), App.Vector(10, 0, 0)), False)
        sketch.addConstraint(Sketcher.Constraint("DistanceX", i1, 1, i1, 2, 10.0))
        sketch.addConstraint(Sketcher.Constraint("Distance", i1, 1, i1, 2, 10.0))
        sketch.addConstraint(Sketcher.Constraint("Horizontal", i1))
        doc.recompute()
        info = _sketchbuild.sketch_report(sketch, "line", 0, "redundant_probe")
        self.assertGreaterEqual(info.get("redundant", 0), 1, info)


class DeadSketchTest(_DocTest):
    def test_the_build_result_carries_the_unused_sketch_warning(self):
        plan = _plate("intel_dead")
        plan["features"].append({
            "id": "dead_sketch", "op": "sketch", "plane": {"frame_plane": "top"},
            "profile": {"shape": "circle", "diameter": "plate_len",
                        "center": [0, 0]}})
        doc, report = self.build(plan, "DeadSketch")
        self.assertTrue(any("sketch 'dead_sketch' is never used" in w
                            for w in report["warnings"]), report["warnings"])
        self.assertIsNotNone(doc.getObject("dead_sketch"))   # a warning builds it


class SegmentConsistencyBuildTest(_DocTest):
    """The plan-level consistency checks fire at validation, before any
    geometry exists (the builder never sees the plan)."""

    def test_a_contradictory_declared_length_never_reaches_the_builder(self):
        plan = _plate("intel_lies", d_len=20, d_h=5, d_wrong=99)
        plan["features"] += [
            {"id": "d_sketch", "op": "sketch", "plane": {"frame_plane": "top"},
             "profile": {"shape": "profile", "start": [0, 0], "segments": [
                 {"type": "line", "to": ["d_len", 0], "length": "d_wrong"},
                 {"type": "line", "to": ["d_len", 10]},
                 {"type": "line", "to": [0, 10]},
                 {"type": "line", "to": [0, 0]}]}},
            {"id": "d_pocket", "op": "pocket", "sketch": "d_sketch",
             "depth": "d_h"}]
        doc, report = self.build(plan, "LengthLie", ok=False)
        self.assertEqual(report["stage"], "validation")
        joined = "; ".join(report["errors"])
        self.assertIn("declares length 99", joined)
        self.assertIn("20 mm apart", joined)
        self.assertIsNone(doc.getObject("d_sketch"))

    def test_an_off_circle_arc_center_never_reaches_the_solver(self):
        plan = _plate("intel_offc", d_len=20, d_h=5)
        plan["features"] += [
            {"id": "d_sketch", "op": "sketch", "plane": {"frame_plane": "top"},
             "profile": {"shape": "profile", "start": [0, 0], "segments": [
                 {"type": "line", "to": ["d_len", 0]},
                 {"type": "arc", "to": [23, 4], "center": ["d_len", 0],
                  "ccw": True},
                 {"type": "line", "to": [23, 10]},
                 {"type": "line", "to": [0, 10]},
                 {"type": "line", "to": [0, 0]}]}},
            {"id": "d_pocket", "op": "pocket", "sketch": "d_sketch",
             "depth": "d_h"}]
        doc, report = self.build(plan, "OffCircle", ok=False)
        self.assertEqual(report["stage"], "validation")
        joined = "; ".join(report["errors"])
        self.assertIn("arc segment 1", joined)
        self.assertIn("both must sit on the arc", joined)
        self.assertIsNone(doc.getObject("d_sketch"))

    def test_a_plan_with_rounded_coordinates_builds(self):
        # a model writing coordinates to two decimals is not lying about its shape: the
        # arc's end point sits 0.0002 mm off its circle, exactly as people write them
        plan = _plate("intel_rounded", d_r=10, d_h=5)
        plan["features"] += [
            {"id": "d_sketch", "op": "sketch", "plane": {"frame_plane": "top"},
             "profile": {"shape": "profile", "start": [10, 0], "segments": [
                 {"type": "arc", "to": [5, 8.66], "center": [0, 0], "radius": "d_r",
                  "ccw": True},
                 {"type": "line", "to": [0, 8.66]},
                 {"type": "line", "to": [0, 0]},
                 {"type": "line", "to": [10, 0]}]}},
            {"id": "d_pocket", "op": "pocket", "sketch": "d_sketch", "depth": "d_h"}]
        doc, report = self.build(plan, "Rounded")
        info = self.sketch_info(report, "d_sketch")
        self.assertTrue(info["fully_constrained"], info)
        self.assertAlmostEqual(self.volume(doc), PLATE_VOLUME - 740.08 / 10 * 5, delta=0.5)


if __name__ == "__main__":
    unittest.main()
