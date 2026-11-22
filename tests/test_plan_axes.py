"""Issue 003 follow-up, Area 1: subtractive features on ALL six axis directions.

Generic fixtures only (no part-specific names or numbers baked into the implementation):
a plate with through holes, a box drilled on three axes (including two INTERSECTING holes),
and a cylinder (built by revolution) with intersecting cross holes. Expected volumes are
analytic where trivial and an independent Part-boolean oracle where cuts intersect.

Also pins the enriched "cut nothing" error: it must carry the computed facts (drill
direction, profile bbox, material bbox) so a model can self-correct in one round.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # BEFORE importing oscar

import FreeCAD as App  # noqa: E402
import Part  # noqa: E402

from oscar.plan import builder  # noqa: E402


def assumption(value, why):
    return {"value": value, "source": {"kind": "assumption", "ref": why}}


def plan(pid, params, features):
    return {"schema_version": 2, "id": pid, "part_class": "part", "units": "mm",
            "frame": {"planes": {"base": {"origin": [0, 0, 0], "normal": [0, 0, 1]}},
                      "axes": {"z": {"direction": [0, 0, 1], "point": [0, 0, 0]}}},
            "parameters": params, "features": features}


def rectangle_base(params_size):
    return [{"id": "base_sk", "op": "sketch", "plane": {"frame_plane": "base"},
             "profile": {"shape": "rectangle", "width": "size", "height": "size"}},
            {"id": "base_pad", "op": "pad", "sketch": "base_sk", "length": "size"}]


class AxisCase(unittest.TestCase):
    def setUp(self):
        self.docs = []
        self.addCleanup(self._close)

    def _close(self):
        for name in self.docs:
            if name in App.listDocuments():
                App.closeDocument(name)

    def apply(self, plan_dict):
        doc = App.newDocument("Ax" + str(len(self.docs)))
        self.docs.append(doc.Name)
        result = builder.apply_plan(doc, plan_dict)
        return doc, result

    def body_volume(self, doc, result):
        self.assertTrue(result.get("success"), result)
        return doc.getObject(result["body"]).Shape.Volume


class PlateHolesTest(AxisCase):
    """Plate 100 x 60 x 8 with four through holes drilled along -Z (top face, outward +Z)."""

    def test_four_through_holes_match_the_analytic_volume(self):
        thickness, dia = 8.0, 10.0
        cut = 4 * math.pi * (dia / 2) ** 2 * thickness
        features = [
            {"id": "base_sk", "op": "sketch", "plane": {"frame_plane": "base"},
             "profile": {"shape": "rectangle", "width": "size", "height": "size2"}},
            {"id": "base_pad", "op": "pad", "sketch": "base_sk", "length": "thk"},
        ]
        for i, (x, y) in enumerate([(-30, -15), (30, -15), (-30, 15), (30, 15)]):
            features.append({"id": f"h{i}", "op": "hole",
                             "plane": {"normal": [0, 0, 1], "origin": [x, y, thickness]},
                             "position": {"x": 0, "y": 0}, "diameter": "dia", "through": True})
        params = {"size": assumption(100.0, "plate width"), "size2": assumption(60.0, "plate height"),
                  "thk": assumption(thickness, "plate thickness"), "dia": assumption(dia, "hole diameter")}
        doc, result = self.apply(plan("plate", params, features))
        self.assertAlmostEqual(self.body_volume(doc, result), 100 * 60 * thickness - cut, places=2)


class BoxThreeAxesTest(AxisCase):
    """A cube drilled through +Z, +X (intersecting the Z hole) and blind into -Y."""

    SIZE, DIA, BLIND = 40.0, 8.0, 10.0

    def cube_plan(self, holes):
        return plan("box3", {"size": assumption(self.SIZE, "cube edge"), "dia": assumption(self.DIA, "hole diameter"),
                             "blind": assumption(self.BLIND, "blind depth")},
                    rectangle_base("size") + holes)

    def hole(self, fid, normal, origin, **kw):
        return {"id": fid, "op": "hole", "plane": {"normal": normal, "origin": origin},
                "position": {"x": 0, "y": 0}, "diameter": "dia", **kw}

    def test_every_axis_direction_cuts_exactly(self):
        h = self.SIZE / 2
        cases = {  # outward normal at the true face centre of the centred cube
            "+Z": ([0, 0, 1], [0, 0, self.SIZE]), "-Z": ([0, 0, -1], [0, 0, 0]),
            "+X": ([1, 0, 0], [h, 0, h]), "-X": ([-1, 0, 0], [-h, 0, h]),
            "+Y": ([0, 1, 0], [0, h, h]), "-Y": ([0, -1, 0], [0, -h, h]),
        }
        through = math.pi * (self.DIA / 2) ** 2 * self.SIZE
        blind_vol = math.pi * (self.DIA / 2) ** 2 * self.BLIND
        for label, (normal, origin) in cases.items():
            with self.subTest(label):
                blind = label in ("+Y", "-Y")
                feats = self.hole("hz" if label in ("+Z", "-Z") else "h1", normal, origin,
                                  **({"depth": "blind"} if blind else {"through": True}))
                doc, result = self.apply(self.cube_plan([feats]))
                expected = blind_vol if blind else through
                self.assertAlmostEqual(self.body_volume(doc, result), self.SIZE ** 3 - expected, places=2)

    def test_intersecting_through_holes_match_an_independent_oracle(self):
        h = self.SIZE / 2
        p = self.cube_plan([
            self.hole("hz", [0, 0, 1], [0, 0, self.SIZE], through=True),
            self.hole("hx", [1, 0, 0], [h, 0, h], through=True),
        ])
        doc, result = self.apply(p)
        oracle = Part.makeBox(self.SIZE, self.SIZE, self.SIZE, App.Vector(-h, -h, 0))
        oracle = oracle.cut(Part.makeCylinder(self.DIA / 2, self.SIZE, App.Vector(0, 0, 0), App.Vector(0, 0, 1)))
        oracle = oracle.cut(Part.makeCylinder(self.DIA / 2, self.SIZE, App.Vector(h, 0, h), App.Vector(-1, 0, 0)))
        self.assertAlmostEqual(self.body_volume(doc, result), oracle.Volume, delta=0.5)


class CylinderCrossHolesTest(AxisCase):
    """A cylinder built by REVOLUTION with two intersecting cross holes (generic round part)."""

    R, HEIGHT, DIA = 20.0, 40.0, 8.0

    def cylinder_plan(self, holes):
        return plan("cyl", {"cyl_r": assumption(self.R, "cylinder radius"),
                            "hgt": assumption(self.HEIGHT, "cylinder height"),
                            "dia": assumption(self.DIA, "cross hole diameter"),
                            "full_ang": assumption(360.0, "full revolution")},
                    [{"id": "rev_sk", "op": "sketch",
                      "plane": {"normal": [0, -1, 0], "origin": [0, 0, 0]},
                      "profile": {"shape": "rectangle", "width": "cyl_r", "height": "hgt",
                                  "center": [self.R / 2, self.HEIGHT / 2]}},
                     {"id": "rev", "op": "revolution", "sketch": "rev_sk", "angle": "full_ang",
                      "axis": {"frame_axis": "z"}},
                     *holes])

    def test_intersecting_cross_holes_match_an_independent_oracle(self):
        z = self.HEIGHT / 2
        p = self.cylinder_plan([
            {"id": "cx", "op": "hole", "plane": {"normal": [1, 0, 0], "origin": [self.R, 0, z]},
             "position": {"x": 0, "y": 0}, "diameter": "dia", "through": True},
            {"id": "cy", "op": "hole", "plane": {"normal": [0, 1, 0], "origin": [0, self.R, z]},
             "position": {"x": 0, "y": 0}, "diameter": "dia", "through": True},
        ])
        doc, result = self.apply(p)
        oracle = Part.makeCylinder(self.R, self.HEIGHT, App.Vector(0, 0, 0), App.Vector(0, 0, 1))
        oracle = oracle.cut(Part.makeCylinder(self.DIA / 2, 2 * self.R, App.Vector(self.R, 0, z), App.Vector(-1, 0, 0)))
        oracle = oracle.cut(Part.makeCylinder(self.DIA / 2, 2 * self.R, App.Vector(0, self.R, z), App.Vector(0, -1, 0)))
        self.assertAlmostEqual(self.body_volume(doc, result), oracle.Volume, delta=0.5)


class CutNothingErrorTest(AxisCase):
    """The failure must carry the computed facts (drill direction, profile and material boxes)."""

    def wrong_hole(self):
        p = plan("wrong", {"size": assumption(40.0, "cube edge"), "dia": assumption(8.0, "hole diameter")},
                 rectangle_base("size") + [
                     {"id": "h1", "op": "hole", "plane": {"normal": [0, -1, 0], "origin": [0, 20, 20]},
                      "position": {"x": 0, "y": 0}, "diameter": "dia", "through": True}])
        return self.apply(p)

    def test_the_error_names_the_drill_direction_and_both_boxes(self):
        doc, result = self.wrong_hole()
        self.assertFalse(result.get("success"))
        text = str(result.get("error") or result)
        self.assertIn("cut nothing", text)
        self.assertIn("drills towards", text)          # the computed direction vector
        self.assertIn("profile", text)                 # the profile's global box
        self.assertIn("material", text)                # the body's box at failure time

    def test_the_error_tells_holes_have_no_invert(self):
        doc, result = self.wrong_hole()
        text = str(result.get("error") or result)
        self.assertIn("flip the plane normal", text)   # the exact fix for a hole


if __name__ == "__main__":
    unittest.main()
