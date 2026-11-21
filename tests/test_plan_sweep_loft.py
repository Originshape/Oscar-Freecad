"""Phase 5: sweep (PartDesign Pipe) and loft as FeaturePlan ops — schema
vocabulary, builder translators, deterministic postconditions.

Covers: the four ops (sweep / sweep_cut / loft / loft_cut) validate their
references, build analytically exact solids (a straight-spine cylinder, a
quarter torus on an arc spine, two-section box lofts, three-section ruled
frusta, a swept cut and a lofted cavity), refuse what the kernel cannot do
with an actionable BuildError while leaving nothing partial behind, and
survive save → reopen → recompute with ±10 % parameter edits re-driving the
geometry through the spreadsheet bindings.

Runs in console mode via tests/run.py (Sketcher/PartDesign are App-level; no
GUI needed). FreeCAD behaviour asserted here was probed by execution on
1.1.4 (see the probe notes in the phase report)."""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache BEFORE importing oscar

import FreeCAD as App  # noqa: E402

from oscar.plan import _featurebuild, builder, schema  # noqa: E402


def _param(value: float) -> dict:
    return {"value": float(value), "assumption": "unit test fixture value"}


def _derived(expr: str) -> dict:
    return {"value": 0.0, "expr": expr, "source": {"kind": "derived", "ref": None}}


def _base_plan(plan_id: str, params: dict, features: list,
               version: int = 1) -> dict:
    return {"schema_version": version, "id": plan_id, "part_class": "part",
            "units": "mm", "target": "freecad", "frame": {},
            "parameters": params, "features": features}


def _plate_features(channel: bool = False) -> list:
    """A 60 x 40 x 10 plate; optionally the sketches of a swept channel that
    crosses it along X (a full plate + channel plan is assembled per test)."""
    feats = [
        {"id": "plate_sketch", "op": "sketch",
         "plane": {"normal": [0, 0, 1], "origin": [0, 0, 0]},
         "profile": {"shape": "rectangle", "width": "plate_len",
                     "height": "plate_wid", "center": [0, 0]}},
        {"id": "plate_pad", "op": "pad", "sketch": "plate_sketch",
         "length": "thickness"},
    ]
    if channel:
        feats += [
            {"id": "channel_prof", "op": "sketch",
             "plane": {"normal": [1, 0, 0], "origin": [-40, 0, "channel_z"]},
             "profile": {"shape": "circle", "diameter": "tube_dia"}},
            {"id": "channel_spine", "op": "sketch",
             "plane": {"normal": [0, 0, 1], "origin": [-40, 0, "channel_z"]},
             "profile": {"shape": "profile", "start": [0, 0],
                         "segments": [{"type": "line", "to": ["path_len", 0]}]}},
            {"id": "channel", "op": "sweep_cut", "sketch": "channel_prof",
             "spine": "channel_spine"},
        ]
    return feats


def _plate_params(**extra) -> dict:
    params = {"plate_len": _param(60), "plate_wid": _param(40),
              "thickness": _param(10)}
    params.update(extra)
    return params


PLATE_VOLUME = 60 * 40 * 10


class _DocTest(unittest.TestCase):
    """Fresh documents that are always closed again."""

    def new_doc(self, name: str = "P5Doc"):
        doc = App.newDocument(name)
        self.addCleanup(self._close, doc.Name)
        return doc

    @staticmethod
    def _close(name: str) -> None:
        if name in App.listDocuments():
            App.closeDocument(name)

    def build(self, plan: dict, name: str = "P5Build"):
        doc = self.new_doc(name)
        report = builder.apply_plan(doc, plan)
        self.assertTrue(report.get("success"),
                        f"{report.get('stage')}/{report.get('feature_id')}: "
                        f"{report.get('error')}")
        return doc, report

    def refuse(self, plan: dict, name: str = "P5Refuse"):
        doc = self.new_doc(name)
        report = builder.apply_plan(doc, plan)
        self.assertFalse(report.get("success"), report.get("error"))
        return doc, report

    @staticmethod
    def volume(doc) -> float:
        return float(doc.getObject("Body").Shape.Volume)

    @staticmethod
    def set_param(doc, alias: str, value: float) -> None:
        sheet = doc.getObject("Parameters")
        cell = sheet.getCellFromAlias(alias)
        assert cell, f"no alias {alias!r} in Parameters"
        sheet.set(cell, "=" + repr(float(value)))
        doc.recompute()

    def assertVolume(self, doc, expected: float, note: str = "") -> None:
        got = self.volume(doc)
        self.assertAlmostEqual(got, expected, delta=1e-6 * max(1.0, abs(expected)),
                               msg=f"volume {got} != {expected} {note}")


class RegistryTest(unittest.TestCase):
    """The four ops sit in the registry with their post kinds and translators."""

    def test_ops_registered_v1_with_post_kinds_and_builders(self):
        for op, post in (("sweep", "additive"), ("sweep_cut", "subtractive"),
                         ("loft", "additive"), ("loft_cut", "subtractive")):
            spec = schema.OPS[op]
            self.assertEqual(spec.since, 1, op)
            self.assertEqual(spec.post, post, op)
            self.assertEqual(spec.builder, _featurebuild.SOLID_BUILDERS[op], op)
            self.assertIn(op, schema.FEATURE_OPS)
        self.assertIn("sweep", schema.ops_for(1))
        for op in ("sweep", "sweep_cut", "loft", "loft_cut"):
            for key in ("sketch", "spine", "sections", "mode", "ruled"):
                if key in schema.OPS[op].keys:
                    self.assertIn(key, schema.plan_reference())

    def test_vocabulary_lists_the_keys_the_planner_needs(self):
        text = schema.plan_reference()
        self.assertIn("sweep (solid, post additive)", text)
        self.assertIn("loft_cut (solid, post subtractive)", text)
        for line in text.splitlines():
            if line.startswith(("sweep ", "sweep_cut ", "loft ", "loft_cut ")):
                self.assertIn("required", line)


class SweepValidationTest(unittest.TestCase):
    """The plan vocabulary of sweep / sweep_cut."""

    def _plan(self, **feat):
        sweep = {"id": "sw", "op": "sweep", "sketch": "prof", "spine": "guide"}
        sweep.update(feat)
        return _base_plan("p5v", {"tube_dia": _param(5), "path_len": _param(40)},
                          [{"id": "prof", "op": "sketch",
                            "plane": {"normal": [1, 0, 0]},
                            "profile": {"shape": "circle", "diameter": "tube_dia"}},
                           {"id": "guide", "op": "sketch",
                            "plane": {"normal": [0, 0, 1]},
                            "profile": {"shape": "profile", "start": [0, 0],
                                        "segments": [{"type": "line",
                                                      "to": ["path_len", 0]}]}},
                           sweep])

    def errors(self, plan):
        ok, errors, _ = schema.validate(plan)
        self.assertFalse(ok, errors)
        return " | ".join(errors)

    def test_bad_mode_is_an_error_with_the_enum(self):
        text = self.errors(self._plan(mode="freets"))
        self.assertIn("sweep.mode 'freets' not in", text)
        self.assertIn("'standard'", text)

    def test_spine_must_precede(self):
        text = self.errors(self._plan(spine="nope"))
        self.assertIn("references sketch 'nope' which does not precede it", text)

    def test_spine_must_differ_from_the_profile(self):
        text = self.errors(self._plan(spine="prof"))
        self.assertIn("along itself", text)

    def test_path_key_carries_the_use_instead_hint(self):
        text = self.errors(self._plan(path="guide"))
        self.assertIn("unknown key 'path'", text)
        self.assertIn("guide sketch key is 'spine'", text)

    def test_an_open_polyline_is_legal_only_as_a_spine(self):
        ok, errors, warnings = schema.validate(self._plan())
        self.assertEqual(errors, [], errors)
        # the same open guide consumed by a pad is still refused
        plan = self._plan()
        plan["features"].append({"id": "p", "op": "pad", "sketch": "guide",
                                 "length": "path_len"})
        ok, errors, _ = schema.validate(plan)
        self.assertFalse(ok)
        self.assertTrue(any("profile is not closed" in e for e in errors), errors)


class LoftValidationTest(unittest.TestCase):
    """The plan vocabulary of loft / loft_cut."""

    def _plan(self, **feat):
        loft = {"id": "lf", "op": "loft", "sketch": "base",
                "sections": ["mid", "top"]}
        loft.update(feat)
        return _base_plan(
            "p5vl", {"sec_w": _param(20), "sec_d": _param(10)},
            [{"id": "base", "op": "sketch", "plane": {"normal": [0, 0, 1]},
              "profile": {"shape": "rectangle", "width": "box_w", "height": "box_d"}},
             {"id": "mid", "op": "sketch",
              "plane": {"normal": [0, 0, 1], "origin": [0, 0, 10]},
              "profile": {"shape": "rectangle", "width": "box_w", "height": "box_d"}},
             {"id": "top", "op": "sketch",
              "plane": {"normal": [0, 0, 1], "origin": [0, 0, 20]},
              "profile": {"shape": "rectangle", "width": "box_w", "height": "box_d"}},
             loft])

    def errors(self, plan):
        ok, errors, _ = schema.validate(plan)
        self.assertFalse(ok, errors)
        return " | ".join(errors)

    def test_sections_are_required(self):
        text = self.errors(self._plan(sections=None))
        self.assertIn("loft requires 'sections'", text)
        self.assertIn("non-empty list of earlier sketch ids", text)

    def test_sections_must_precede(self):
        text = self.errors(self._plan(sections=["base", "later"]))
        self.assertIn("loft.sections[1] references sketch 'later' which does "
                      "not precede it", text)
        self.assertIn("names its own profile sketch 'base' as a section", text)

    def test_sections_must_not_repeat(self):
        text = self.errors(self._plan(sections=["mid", "mid", "top"]))
        self.assertIn("repeats ['mid']", text)

    def test_ruled_must_be_a_flag(self):
        text = self.errors(self._plan(ruled="yes"))
        self.assertIn("loft.ruled must be true or false", text)


class SweepBuildTest(_DocTest):
    """Analytically exact pipes through the real builder."""

    def test_straight_spine_sweeps_an_exact_cylinder(self):
        plan = _base_plan(
            "p5_tube", {"tube_dia": _param(10), "path_len": _param(40)},
            [{"id": "tube_prof", "op": "sketch", "plane": {"normal": [1, 0, 0]},
              "profile": {"shape": "circle", "diameter": "tube_dia"}},
             {"id": "tube_guide", "op": "sketch", "plane": {"normal": [0, 0, 1]},
              "profile": {"shape": "profile", "start": [0, 0],
                          "segments": [{"type": "line", "to": ["path_len", 0]}]}},
             {"id": "tube", "op": "sweep", "sketch": "tube_prof",
              "spine": "tube_guide"}])
        doc, report = self.build(plan)
        self.assertVolume(doc, math.pi * 25 * 40)
        body = doc.getObject("Body")
        self.assertEqual(len(body.Shape.Solids), 1)
        pipe = doc.getObject("tube")
        self.assertEqual(pipe.TypeId, "PartDesign::AdditivePipe")
        # object links (not edge indices): the two sketches are the consumption
        self.assertIs(pipe.Profile[0], doc.getObject("tube_prof"))
        self.assertIs(pipe.Spine[0], doc.getObject("tube_guide"))
        info = next(f["info"] for f in report["features"] if f["id"] == "tube")
        self.assertEqual(info["spine"], "tube_guide")
        # the dimensions ride the sheet, not literals
        self.assertTrue(doc.getObject("tube_prof").ExpressionEngine)
        self.assertTrue(doc.getObject("tube_guide").ExpressionEngine)

    def test_arc_spine_quarter_torus_frenet(self):
        ring, tube = 30.0, 5.0
        plan = _base_plan(
            "p5_ring", {"ring_rad": _param(ring), "tube_dia": _param(tube)},
            [{"id": "ring_prof", "op": "sketch",
              "plane": {"normal": [0, 0, 1], "origin": ["ring_rad", 0, 0]},
              "profile": {"shape": "circle", "diameter": "tube_dia"}},
             {"id": "ring_guide", "op": "sketch",
              "plane": {"normal": [0, -1, 0], "x_dir": [1, 0, 0]},
              "profile": {"shape": "profile", "start": ["ring_rad", 0],
                          "segments": [{"type": "arc", "to": [0, "ring_rad"],
                                        "center": [0, 0], "ccw": True,
                                        "radius": "ring_rad"}]}},
             {"id": "ring", "op": "sweep", "sketch": "ring_prof",
              "spine": "ring_guide", "mode": "frenet"}])
        doc, _ = self.build(plan)
        self.assertVolume(doc, math.pi ** 2 * ring * (tube / 2) ** 2 / 2)
        self.assertEqual(doc.getObject("ring").Mode, "Frenet")

    def test_in_plane_spine_is_refused_not_silently_empty(self):
        # measured on 1.1.4: the kernel calls this sweep VALID with 0 volume —
        # the additive post-condition must refuse it
        plan = _base_plan(
            "p5_bad", {"tube_dia": _param(10), "path_len": _param(40)},
            [{"id": "prof", "op": "sketch", "plane": {"normal": [0, 0, 1]},
              "profile": {"shape": "circle", "diameter": "tube_dia"}},
             {"id": "guide", "op": "sketch", "plane": {"normal": [0, 0, 1]},
              "profile": {"shape": "profile", "start": [0, -20],
                          "segments": [{"type": "line", "to": ["path_len", -20]}]}},
             {"id": "sw", "op": "sweep", "sketch": "prof", "spine": "guide"}])
        doc, report = self.refuse(plan)
        self.assertEqual(report["feature_id"], "sw")
        self.assertIn("no volume", report["error"])
        self.assertIsNone(doc.getObject("sw"))          # nothing partial left
        for obj in doc.Objects:
            self.assertNotIn("Invalid", obj.State, obj.Name)
            self.assertNotIn("Touched", obj.State, obj.Name)

    def test_sweep_cut_channel_through_the_plate(self):
        params = _plate_params(tube_dia=_param(6), path_len=_param(80),
                               channel_z=_derived("thickness / 2"))
        doc, report = self.build(_base_plan("p5_chan", params,
                                            _plate_features(channel=True)))
        removed = math.pi * 3 ** 2 * 60      # the channel crosses the plate fully
        self.assertVolume(doc, PLATE_VOLUME - removed)
        self.assertEqual(doc.getObject("channel").TypeId,
                         "PartDesign::SubtractivePipe")
        box = doc.getObject("Body").Shape.BoundBox
        self.assertAlmostEqual(box.XLength, 60, delta=1e-3)   # bbox did not grow


class LoftBuildTest(_DocTest):
    """Analytically exact lofts through the real builder."""

    def test_two_sections_make_an_exact_box(self):
        plan = _base_plan(
            "p5_box", {"box_w": _param(20), "box_d": _param(10), "loft_h": _param(15)},
            [{"id": "base", "op": "sketch", "plane": {"normal": [0, 0, 1]},
              "profile": {"shape": "rectangle", "width": "box_w", "height": "box_d"}},
             {"id": "top", "op": "sketch",
              "plane": {"normal": [0, 0, 1], "origin": [0, 0, "loft_h"]},
              "profile": {"shape": "rectangle", "width": "box_w", "height": "box_d"}},
             {"id": "box", "op": "loft", "sketch": "base",
              "sections": ["top"]}])
        doc, report = self.build(plan)
        self.assertVolume(doc, 20 * 10 * 15)
        loft = doc.getObject("box")
        self.assertEqual(loft.TypeId, "PartDesign::AdditiveLoft")
        self.assertFalse(loft.Ruled)
        self.assertEqual(loft.Sections[0][0], doc.getObject("top"))
        info = next(f["info"] for f in report["features"] if f["id"] == "box")
        self.assertEqual(info["sections"], 1)

    def test_three_ruled_sections_stack_two_exact_frusta(self):
        # 20x10 at z=0 -> 14x7 at z=10 -> 20x10 at z=20, ruled
        def rect(fid, z, w, d):
            return {"id": fid, "op": "sketch",
                    "plane": {"normal": [0, 0, 1], "origin": [0, 0, z]},
                    "profile": {"shape": "rectangle", "width": w, "height": d}}

        plan = _base_plan(
            "p5_frusta", {"bot_w": _param(20), "bot_d": _param(10),
                          "mid_w": _param(14), "mid_d": _param(7)},
            [rect("s0", 0, "bot_w", "bot_d"), rect("s1", 10, "mid_w", "mid_d"),
             rect("s2", 20, "bot_w", "bot_d"),
             {"id": "frusta", "op": "loft", "sketch": "s0",
              "sections": ["s1", "s2"], "ruled": True}])
        doc, _ = self.build(plan)
        frustum = 10 / 3 * (200 + 98 + math.sqrt(200 * 98))
        self.assertVolume(doc, 2 * frustum)
        self.assertTrue(doc.getObject("frusta").Ruled)

    def test_loft_cut_removes_an_exact_frustum_cavity(self):
        def rect(fid, z, w, d):
            return {"id": fid, "op": "sketch",
                    "plane": {"normal": [0, 0, 1], "origin": [0, 0, z]},
                    "profile": {"shape": "rectangle", "width": w, "height": d}}

        feats = _plate_features()
        feats += [rect("cav0", 2, "cav_w", "cav_d"),
                  rect("cav1", 8, "cav_w2", "cav_d2"),
                  {"id": "cavity", "op": "loft_cut", "sketch": "cav0",
                   "sections": ["cav1"], "ruled": True}]
        doc, _ = self.build(_base_plan(
            "p5_cav", _plate_params(cav_w=_param(20), cav_d=_param(10),
                                    cav_w2=_param(10), cav_d2=_param(5)), feats))
        frustum = 6 / 3 * (200 + 50 + math.sqrt(200 * 50))
        self.assertVolume(doc, PLATE_VOLUME - frustum)
        self.assertEqual(doc.getObject("cavity").TypeId,
                         "PartDesign::SubtractiveLoft")

    def test_incompatible_sections_fail_actionably_and_leave_nothing(self):
        # one wire in the profile, TWO in a section -> the kernel refuses;
        # verified by execution: its message names the wire-count rule
        plan = _base_plan(
            "p5_bad_loft", {"circ_dia": _param(10)},
            [{"id": "base", "op": "sketch", "plane": {"normal": [0, 0, 1]},
              "profile": {"shape": "circle", "diameter": "circ_dia"}},
             {"id": "twin", "op": "sketch",
              "plane": {"normal": [0, 0, 1], "origin": [0, 0, 10]},
              "profiles": [{"shape": "circle", "diameter": "circ_dia",
                            "center": [0, 0]},
                           {"shape": "circle", "diameter": "circ_dia",
                            "center": [30, 0]}]},
             {"id": "bad", "op": "loft", "sketch": "base",
              "sections": ["twin"]}], version=2)
        self.assertEqual(schema.validate(plan)[0], True)
        doc, report = self.refuse(plan)
        self.assertEqual(report["feature_id"], "bad")
        self.assertIn("recompute failed", report["error"])
        self.assertIn("same amount of wires", report["error"])
        self.assertIsNone(doc.getObject("bad"))
        names = {o.Name for o in doc.Objects}
        self.assertIn("base", names)
        self.assertIn("twin", names)          # the sketches stay (earlier features)
        for obj in doc.Objects:
            self.assertNotIn("Invalid", obj.State, obj.Name)
            self.assertNotIn("Touched", obj.State, obj.Name)
        self.assertEqual(list(report["completed"]), ["base", "twin"])
        body = doc.getObject("Body")
        self.assertIsNone(body.Tip)           # the tip never points at the corpse


class StabilityTest(_DocTest):
    """Save -> reopen -> forced recompute -> ±10 % edits, all analytically."""

    def _plan(self):
        params = _plate_params(
            tube_dia=_param(5), path_len=_param(80),
            channel_z=_derived("thickness / 2"),
            boss_w=_param(20), boss_d=_param(10),
            boss_w2=_param(14), boss_d2=_param(7), boss_h=_param(10),
            boss_top=_derived("thickness + boss_h"))
        feats = _plate_features(channel=True)
        feats += [
            {"id": "boss_base", "op": "sketch",
             "plane": {"normal": [0, 0, 1], "origin": [0, 0, "thickness"]},
             "profile": {"shape": "rectangle", "width": "boss_w",
                         "height": "boss_d", "center": [0, 0]}},
            {"id": "boss_top_sk", "op": "sketch",
             "plane": {"normal": [0, 0, 1], "origin": [0, 0, "boss_top"]},
             "profile": {"shape": "rectangle", "width": "boss_w2",
                         "height": "boss_d2", "center": [0, 0]}},
            {"id": "boss", "op": "loft", "sketch": "boss_base",
             "sections": ["boss_top_sk"], "ruled": True},
        ]
        return _base_plan("p5_stable", params, feats)

    def test_save_reopen_recompute_and_pm10_edits(self):
        frustum = 10 / 3 * (200 + 98 + math.sqrt(200 * 98))
        channel = math.pi * (5 / 2) ** 2 * 60
        expected = PLATE_VOLUME - channel + frustum
        doc, _ = self.build(self._plan())
        self.assertVolume(doc, expected, "(fresh build)")
        fresh = self.volume(doc)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p5_stable.FCStd")
            doc.saveAs(path)
            App.closeDocument(doc.Name)
            doc = App.openDocument(path)
            self.addCleanup(self._close, doc.Name)

            # forced full recompute keeps it valid and identical
            for obj in doc.Objects:
                obj.touch()
            doc.recompute()
            doc.recompute()
            self.assertEqual(len(doc.getObject("Body").Shape.Solids), 1)
            self.assertTrue(doc.getObject("Body").Shape.isValid())
            self.assertVolume(doc, fresh, "(after reopen + forced recompute)")
            # the links and bindings survived the round trip
            self.assertEqual(doc.getObject("channel").Spine[0].Name,
                             "channel_spine")
            self.assertEqual(doc.getObject("boss").Sections[0][0].Name,
                             "boss_top_sk")
            self.assertTrue(doc.getObject("boss_top_sk").ExpressionEngine)

            # +10 % channel diameter: removed grows to pi*(2.75)^2*60 exactly
            self.set_param(doc, "tube_dia", 5.5)
            self.assertVolume(doc, PLATE_VOLUME - math.pi * 2.75 ** 2 * 60
                              + frustum, "(tube_dia +10 %)")
            self.set_param(doc, "tube_dia", 5)

            # -10 % boss height: the frustum shrinks linearly with h
            self.set_param(doc, "boss_h", 9)
            self.assertVolume(doc, PLATE_VOLUME - channel
                              + 9 / 3 * (200 + 98 + math.sqrt(200 * 98)),
                              "(boss_h -10 %)")
            self.set_param(doc, "boss_h", 10)

            # restored verbatim
            self.assertVolume(doc, fresh, "(restored)")


if __name__ == "__main__":
    unittest.main()
