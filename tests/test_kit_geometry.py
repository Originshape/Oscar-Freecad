"""Tests for oscar.kit.geometry and oscar.kit._cutters: tags, scanning, the Builder primitives and the snap-fit numbers.

The pure parts (formatting, ``barb_profile``, ``hook_plans``) need only FreeCAD's Part module for the polygon prism; everything that
touches a document runs under freecadcmd (tests/run.py) and is skipped on a plain interpreter. The board is the byte-identical copy
of the board JSON in tests/fixtures/enclosure so the numbers do not move when the board library does.
"""

import json
import math
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.kit import spec  # noqa: E402

try:
    import FreeCAD as App  # noqa: E402
    import Part  # noqa: E402
    from oscar.kit import _cutters as C  # noqa: E402
    from oscar.kit import build, params  # noqa: E402
    from oscar.kit import geometry as G  # noqa: E402
except ImportError:
    App = None

_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "enclosure")
with open(os.path.join(_FIX, "rpi4b_board.json"), encoding="utf-8") as _f:
    BOARD = json.load(_f)

needs_fc = unittest.skipIf(App is None, "FreeCAD is needed (run through freecadcmd)")


def exprs_of(obj) -> dict:
    """``{property path: expression}`` of ``obj`` (FreeCAD prefixes some sub-property paths with a dot; stripped here)."""
    return {path.lstrip("."): expr for path, expr in obj.ExpressionEngine}


def default_values() -> dict:
    """The Params values of the default Pi 4 / ABS spec."""
    return {r["name"]: r["value"] for r in spec.param_rows(spec.default_spec(), BOARD)}


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class FormatTest(unittest.TestCase):
    def test_num_is_fixed_notation_without_exponent(self):
        self.assertEqual(G.num(2), "2")
        self.assertEqual(G.num(2.5), "2.5")
        self.assertEqual(G.num(-0.0), "0")
        self.assertEqual(G.num(1e-5), "0.00001")
        self.assertEqual(G.num(1e-12), "0")
        self.assertEqual(G.num(123456.789), "123456.789")
        self.assertNotIn("e", G.num(1e-7))

    def test_mm_and_signed(self):
        self.assertEqual(G.mm(1.0), "1 mm")
        self.assertEqual(G.mm(0.25), "0.25 mm")
        self.assertEqual(G.signed(2.5), " + 2.5 mm")
        self.assertEqual(G.signed(-2.5), " - 2.5 mm")
        self.assertEqual(G.signed(0), " + 0 mm")

    def test_literals_are_valid_expressions(self):
        doc = App.newDocument("FmtTest")
        self.addCleanup(App.closeDocument, doc.Name)
        box = doc.addObject("Part::Box", "B")
        for x in (0.00001, 1.25, -3.5, 120.0):
            box.setExpression("Length", G.mm(abs(x)) if x else "1 mm")
            doc.recompute()
            self.assertAlmostEqual(box.Length.Value, abs(x), places=6)


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class BarbProfileTest(unittest.TestCase):
    def footprint(self, prof, fit):
        """``(height, lower z)`` of the barb at the wall plane t = fit, by linear interpolation along its two sloped edges."""
        root_bottom, tip_bottom, tip_top, root_top = (prof["points"][i] for i in (1, 2, 3, 4))
        o = tip_bottom[0]
        lower = root_bottom[1] + (tip_bottom[1] - root_bottom[1]) * fit / o
        upper = root_top[1] + (tip_top[1] - root_top[1]) * fit / o
        return upper - lower, lower

    def test_defaults_flatten_the_slopes_and_keep_the_engagement(self):
        p = G.barb_profile(0.8, 0.25, 1.0, 30, 45)
        self.assertTrue(p["flattened"])
        self.assertAlmostEqual(p["engagement"], 0.55, places=9)
        self.assertAlmostEqual(p["land"], 0.1, places=6)
        self.assertAlmostEqual(p["scale"], 0.9 / (0.55 * (1 / math.tan(math.radians(30)) + 1.0)), places=6)
        self.assertAlmostEqual(p["ramp_eff_deg"], 43.96, delta=0.05)
        self.assertAlmostEqual(p["retention_eff_deg"], 59.09, delta=0.05)
        self.assertLess(p["retention_eff_deg"], 60.0)           # the defaults stay re-openable
        self.assertAlmostEqual(p["z_wall"], 0.2594, delta=1e-3)

    def test_footprint_at_the_wall_plane_is_barb_h(self):
        for args in ((0.8, 0.25, 1.0, 30, 45), (0.8, 0.25, 2.0, 30, 45), (1.2, 0.3, 1.5, 20, 70), (0.5, 0.2, 0.6, 60, 30)):
            p = G.barb_profile(*args)
            height, lower = self.footprint(p, args[1])
            self.assertAlmostEqual(height, args[2], places=6, msg=str(args))
            self.assertAlmostEqual(lower, p["z_wall"], places=6, msg=str(args))

    def test_tip_reaches_the_overhang_and_the_polygon_is_valid(self):
        for args in ((0.8, 0.25, 1.0, 30, 45), (0.4, 0.1, 0.6, 15, 80), (1.5, 0.25, 2.0, 60, 30)):
            p = G.barb_profile(*args)
            pts = p["points"]
            self.assertEqual(len(pts), 6)
            self.assertEqual(max(t for t, _z in pts), args[0])                  # the tip depth is the overhang
            self.assertLess(min(t for t, _z in pts), 0)                          # the root lies inside the beam (it fuses)
            self.assertGreaterEqual(p["land"], args[2] * 0.1 - 1e-9)             # a flat land remains
            zs = [z for _t, z in pts]
            self.assertAlmostEqual(min(zs), 0.0)                                 # root bottom = beam bottom
            self.assertAlmostEqual(max(zs), p["root_h"])
            self.assertGreater(G.polygon_area([App.Vector(0, t, z) for t, z in pts]), 0.1)

    def test_tall_barb_keeps_the_requested_angles(self):
        p = G.barb_profile(0.8, 0.25, 2.0, 30, 45)
        self.assertFalse(p["flattened"])
        self.assertEqual(p["scale"], 1.0)
        self.assertAlmostEqual(p["ramp_eff_deg"], 30.0, places=6)
        self.assertAlmostEqual(p["retention_eff_deg"], 45.0, places=6)
        self.assertGreater(p["land"], 0.4)

    def test_permanent_snap_has_a_flat_ledge(self):
        p = G.barb_profile(0.8, 0.25, 2.0, 30, 90)
        self.assertAlmostEqual(p["retention_eff_deg"], 90.0, delta=0.01)
        self.assertAlmostEqual(p["points"][3][1], p["points"][4][1], delta=1e-3)          # tip top level with the root top: a horizontal ledge

    def test_invalid_inputs_raise(self):
        for bad in ((0.25, 0.25, 1.0, 30, 45), (0.2, 0.25, 1.0, 30, 45), (0.8, 0.25, 0.0, 30, 45), (0.8, 0.25, -1.0, 30, 45),
                    (0.8, 0.25, 1.0, 0, 45), (0.8, 0.25, 1.0, 30, 91), (0.8, 0.25, 1.0, 100, 45)):
            with self.assertRaises(G.KitGeometryError, msg=str(bad)):
                G.barb_profile(*bad)

    def test_pocket_depth_matches_the_spec_rule(self):
        for ov, fit in ((0.8, 0.25), (1.0, 0.3), (0.5, 0.1)):
            s = spec.default_spec()
            s.hooks.overhang, s.lid.fit = ov, fit
            self.assertAlmostEqual(G.pocket_depth(ov, fit), spec.catch_pocket_depth(s), places=9)

    def test_barb_shape_is_a_valid_positive_solid_on_either_wall(self):
        p = G.barb_profile(0.8, 0.25, 1.0, 30, 45)
        for sign in (-1, 1):
            shape = G.barb_shape(p, 6.0, sign)
            self.assertEqual(len(shape.Solids), 1)
            self.assertTrue(shape.isValid())
            self.assertGreater(shape.Volume, 0)
            self.assertAlmostEqual(shape.BoundBox.XLength, 6.0, places=6)
            bb = shape.BoundBox
            if sign < 0:                      # +y wall: inward is -y, the root lies at +y
                self.assertAlmostEqual(bb.YMin, -0.8, places=6)
                self.assertAlmostEqual(bb.YMax, 0.3, places=6)
            else:
                self.assertAlmostEqual(bb.YMax, 0.8, places=6)
                self.assertAlmostEqual(bb.YMin, -0.3, places=6)
        self.assertAlmostEqual(G.barb_shape(p, 6.0, -1).Volume, G.barb_shape(p, 6.0, 1).Volume, places=6)

    def test_degenerate_polygon_is_refused(self):
        flat = {"points": [(0, 0), (1, 0), (2, 0)]}
        with self.assertRaises(G.KitGeometryError):
            G.barb_shape(flat, 6.0, 1)

    def test_polygon_area(self):
        sq = [App.Vector(0, 0, 0), App.Vector(2, 0, 0), App.Vector(2, 3, 0), App.Vector(0, 3, 0)]
        self.assertAlmostEqual(G.polygon_area(sq), 6.0)
        tilted = [App.Vector(0, 0, 0), App.Vector(0, 2, 0), App.Vector(0, 2, 3), App.Vector(0, 0, 3)]     # plane x = 0
        self.assertAlmostEqual(G.polygon_area(tilted), 6.0)
        self.assertEqual(G.polygon_area(sq[:2]), 0.0)


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class HookPlansTest(unittest.TestCase):
    def test_default_plans(self):
        plans = G.hook_plans(spec.default_spec(), BOARD)
        self.assertEqual(len(plans), 4)
        self.assertEqual([p["wall"] for p in plans], ["+y", "+y", "-y", "-y"])
        self.assertEqual([(p["wall"], p["u"]) for p in plans], [(h["wall"], h["u"]) for h in spec.hook_positions(spec.default_spec(), BOARD)])
        self.assertAlmostEqual(plans[0]["u"], 19.725, places=3)
        for p in plans:
            self.assertEqual((p["w"], p["t"], p["length"]), (6.0, 1.2, 12.0))
            self.assertEqual(p["inward"], -1 if p["wall"] == "+y" else 1)
            self.assertAlmostEqual(p["beam"]["x1"] - p["beam"]["x0"], 6.0)
            pk = p["pocket"]
            self.assertAlmostEqual(pk["w"], 6.4)                    # hooks.w + 0.4
            self.assertAlmostEqual(pk["h"], 1.3)                    # barb_h + window_margin
            self.assertAlmostEqual(pk["depth"], 0.65)               # overhang - fit + 0.1
            self.assertAlmostEqual(pk["x1"] - pk["x0"], 6.4)
            self.assertAlmostEqual((pk["x0"] + pk["x1"]) / 2, p["u"])
            self.assertEqual(p["barb"]["engagement"], 0.55)

    def test_other_styles_have_no_hooks(self):
        for style in ("screw", "friction"):
            s = spec.default_spec()
            s.lid.style = style
            self.assertEqual(G.hook_plans(s, BOARD), [])

    def test_count_and_dict_spec(self):
        d = spec.default_spec().to_dict()
        d["hooks"]["count"] = 2
        plans = G.hook_plans(d, BOARD)
        self.assertEqual([p["wall"] for p in plans], ["+y", "-y"])

    def test_a_measured_pcb_thickness_is_accepted(self):
        a = G.hook_plans(spec.default_spec(), BOARD)
        b = G.hook_plans(spec.default_spec(), BOARD, pcb_t=1.6)
        self.assertEqual([p["wall"] for p in a], [p["wall"] for p in b])


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class TagAndScanTest(unittest.TestCase):
    def setUp(self):
        self.doc = App.newDocument("TagTest")
        self.addCleanup(App.closeDocument, self.doc.Name)

    def obj(self, name, kit="enc1", stage="shell", role="shell", part="Shell", h="abc"):
        o = self.doc.addObject("Part::Box", name)
        if kit is not None:
            G.tag(o, kit_id=kit, stage=stage, role=role, part=part, spec_hash=h)
        return o

    def test_tag_sets_all_five_properties_and_persists_values(self):
        o = self.obj("A", h="0123456789ab")
        self.assertEqual(G.tags_of(o), {"OscarKit": "enc1", "OscarStage": "shell", "OscarRole": "shell", "OscarPart": "Shell",
                                        "OscarSpecHash": "0123456789ab"})
        self.assertEqual(sorted(p for p in o.PropertiesList if p.startswith("Oscar")), sorted(G.TAG_PROPS))
        self.assertEqual(o.getGroupOfProperty("OscarKit"), "Oscar")
        G.tag(o, kit_id="enc2", stage="lid", role="lid", part="Lid", spec_hash="x")        # re-tagging overwrites
        self.assertEqual(G.tags_of(o)["OscarStage"], "lid")

    def test_untagged_object_has_empty_tags(self):
        o = self.obj("Plain", kit=None)
        self.assertEqual(G.tags_of(o), dict.fromkeys(G.TAG_PROPS, ""))
        self.assertFalse(G.is_kit_object(o))

    def test_is_kit_object_filters_by_kit(self):
        o = self.obj("A")
        self.assertTrue(G.is_kit_object(o))
        self.assertTrue(G.is_kit_object(o, "enc1"))
        self.assertFalse(G.is_kit_object(o, "enc2"))

    def test_kit_objects_and_stage_objects(self):
        a = self.obj("A", stage="shell")
        b = self.obj("B", stage="ports")
        c = self.obj("C", kit="enc2", stage="ports")
        plain = self.obj("P", kit=None)
        box = self.obj("Cont", stage="shell", role="container")
        self.assertEqual(G.kit_objects(self.doc, "enc1"), [a, b, box])
        self.assertEqual(G.kit_objects(self.doc), [a, b, c, box])
        self.assertEqual(G.stage_objects(self.doc, "enc1", "ports"), [b])
        self.assertEqual(G.stage_objects(self.doc, "enc1", "shell"), [a])          # containers belong to no single stage
        self.assertNotIn(plain, G.kit_objects(self.doc))

    def test_tip_is_the_newest_stage_and_respects_the_kit(self):
        a = self.obj("A", stage="shell")
        b = self.obj("B", stage="standoffs")
        c = self.obj("C", stage="ports")
        d = self.obj("D", kit="enc2", stage="hooks")
        lid = self.obj("L", stage="lid", role="lid", part="Lid")
        self.assertIs(G.tip(self.doc, "shell", "enc1"), c)
        self.assertIs(G.tip(self.doc, "shell", "enc2"), d)
        self.assertIs(G.tip(self.doc, "shell"), d)                                  # no kit filter: the highest stage index wins
        self.assertIs(G.tip(self.doc, "lid", "enc1"), lid)
        self.assertIsNone(G.tip(self.doc, "lid", "enc2"))
        self.assertEqual(G.chain_objects(self.doc, "shell", "enc1"), [a, b, c])

    def test_tip_ties_go_to_the_newest_object(self):
        self.obj("A", stage="vents")
        b = self.obj("B", stage="vents")
        self.assertIs(G.tip(self.doc, "shell", "enc1"), b)

    def test_tip_ref_falls_back_to_an_untagged_imported_container(self):
        self.assertIsNone(G.tip(self.doc, "ref"))
        part = self.doc.addObject("App::Part", "Ref_RPI4B")
        self.assertIs(G.tip(self.doc, "ref"), part)
        other = self.doc.addObject("App::Part", "Misc")
        self.assertIsNot(G.tip(self.doc, "ref"), other)

    def test_ref_name_and_container_lookup(self):
        self.assertEqual(G.ref_name("rpi4b"), "Ref_RPI4B")
        self.assertEqual(G.ref_name("rpi-4 b"), "Ref_RPI_4_B")
        self.assertIsNone(G.ref_container(self.doc, "rpi4b"))
        part = self.doc.addObject("App::Part", "Part1")
        part.Label = "Ref_RPI4B"
        self.assertIs(G.ref_container(self.doc, "rpi4b"), part)                     # found by Label
        named = self.doc.addObject("App::Part", "Ref_RPI4B")
        self.assertIs(G.ref_container(self.doc, "rpi4b"), named)                    # the Name wins

    def test_set_visible_and_color_are_headless_safe(self):
        o = self.obj("A")
        G.set_visible(o, False)
        self.assertFalse(o.Visibility)
        G.set_visible(o, True)
        self.assertTrue(o.Visibility)
        self.assertIsNone(getattr(o, "ViewObject", None))
        self.assertFalse(G.set_color(o, (1, 0, 0)))                                  # no view object: nothing to colour
        stub = types.SimpleNamespace(ViewObject=types.SimpleNamespace(ShapeColor=None, Visibility=True))
        self.assertTrue(G.set_color(stub, [0.1, 0.2, 0.3]))
        self.assertEqual(stub.ViewObject.ShapeColor, (0.1, 0.2, 0.3))
        G.set_visible(stub, False)
        self.assertFalse(stub.ViewObject.Visibility)
        self.assertFalse(G.set_color(stub, None))

    def test_set_color_survives_a_view_object_that_refuses(self):
        class Refusing:
            def __setattr__(self, name, value):
                raise RuntimeError("no")

        self.assertFalse(G.set_color(types.SimpleNamespace(ViewObject=Refusing()), (0, 0, 0)))


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class RefreshTipsTest(unittest.TestCase):
    def setUp(self):
        self.doc = App.newDocument("TipTest")
        self.addCleanup(App.closeDocument, self.doc.Name)

    def chain(self, names, role="shell", stages=None):
        out = []
        for i, n in enumerate(names):
            o = self.doc.addObject("Part::Box", n)
            G.tag(o, kit_id="enc1", stage=(stages or ["shell", "standoffs", "ports", "vents", "hooks"])[i], role=role, part="Shell")
            out.append(o)
        return out

    def test_only_the_tip_is_visible_and_labelled(self):
        a, b, c = self.chain(["Shell_Body", "Shell_Standoffs", "Shell_Ports"])
        a.Label = "Shell"                                                            # a displaced previous tip
        self.assertEqual(G.refresh_tips(self.doc, "enc1"), [])
        self.assertEqual([o.Name for o in self.doc.getObjectsByLabel("Shell")], ["Shell_Ports"])
        self.assertEqual(a.Label, "Shell_Body")
        self.assertEqual(b.Label, "Shell_Standoffs")
        self.assertEqual([o.Visibility for o in (a, b, c)], [False, False, True])

    def test_the_label_is_never_duplicated_to_shell001(self):
        a, b = self.chain(["Shell_Body", "Shell_Standoffs"])
        G.refresh_tips(self.doc, "enc1")
        self.assertEqual(b.Label, "Shell")
        G.refresh_tips(self.doc, "enc1")                                              # idempotent
        self.assertEqual(b.Label, "Shell")
        self.assertEqual(len(self.doc.getObjectsByLabel("Shell")), 1)

    def test_a_foreign_object_holding_the_label_is_reported_not_stolen(self):
        foreign = self.doc.addObject("Part::Box", "Mine")
        foreign.Label = "Shell"
        (a,) = self.chain(["Shell_Body"])
        warnings = G.refresh_tips(self.doc, "enc1")
        self.assertEqual(foreign.Label, "Shell")
        self.assertEqual(a.Label, "Shell001")
        self.assertEqual(len(warnings), 1)
        self.assertIn("Shell001", warnings[0])
        self.assertIn("rename", warnings[0])

    def test_lid_and_shell_chains_are_independent(self):
        (s,) = self.chain(["Shell_Body"])
        (lid,) = self.chain(["Lid_Plate"], role="lid", stages=["lid"])
        G.refresh_tips(self.doc, "enc1")
        self.assertEqual((s.Label, lid.Label), ("Shell", "Lid"))
        self.assertTrue(s.Visibility and lid.Visibility)

    def test_hides_the_reference_container_and_its_children(self):
        ref = self.doc.addObject("App::Part", "Ref_RPI4B")
        G.tag(ref, kit_id="enc1", stage="reference", role="ref", part="Ref")
        child = self.doc.addObject("Part::Box", "Ref_RPI4B_PCB")
        ref.addObject(child)
        G.refresh_tips(self.doc, "enc1")
        self.assertFalse(ref.Visibility)
        self.assertFalse(child.Visibility)

    def test_nothing_to_do_without_chains(self):
        self.assertEqual(G.refresh_tips(self.doc, "enc1"), [])


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class ShapeFactsTest(unittest.TestCase):
    def setUp(self):
        self.doc = App.newDocument("ShapeTest")
        self.addCleanup(App.closeDocument, self.doc.Name)

    def feature(self, shape, name="F"):
        f = self.doc.addObject("Part::Feature", name)
        f.Shape = shape
        return f

    def test_a_box_is_one_valid_closed_solid(self):
        f = self.feature(Part.makeBox(2, 3, 4))
        facts = G.shape_facts(f)
        self.assertEqual((facts["solids"], facts["valid"], facts["closed"]), (1, True, True))
        self.assertAlmostEqual(facts["volume"], 24.0)
        self.assertEqual(facts["bbox"], [0.0, 0.0, 0.0, 2.0, 3.0, 4.0])
        self.assertEqual(G.single_solid(f), (True, ""))

    def test_two_solids_are_refused(self):
        comp = Part.makeCompound([Part.makeBox(1, 1, 1), Part.makeBox(1, 1, 1, App.Vector(5, 0, 0))])
        ok, why = G.single_solid(self.feature(comp, "Two"))
        self.assertFalse(ok)
        self.assertIn("2 solids", why)
        self.assertIn("'Two'", why)

    def test_a_surface_is_refused(self):
        face = Part.makePlane(2, 2)
        ok, why = G.single_solid(self.feature(face, "Surf"))
        self.assertFalse(ok)
        self.assertIn("0 solids", why)

    def test_an_empty_feature_is_refused_and_never_raises(self):
        f = self.doc.addObject("Part::Feature", "Empty")
        facts = G.shape_facts(f)
        self.assertIsNone(facts["bbox"])
        ok, why = G.single_solid(f)
        self.assertFalse(ok)
        self.assertIn("no shape", why)

    def test_facts_use_the_global_frame(self):
        part = self.doc.addObject("App::Part", "Cont")
        f = self.feature(Part.makeBox(1, 1, 1))
        part.addObject(f)
        part.Placement.Base = App.Vector(0, 0, 50)
        self.doc.recompute()
        self.assertEqual(f.Shape.BoundBox.ZMin, 0.0)                                 # the container frame
        self.assertAlmostEqual(G.shape_facts(f)["bbox"][2], 50.0)                    # the global frame

    def test_common_volume(self):
        a = self.feature(Part.makeBox(2, 2, 2), "A")
        b = self.feature(Part.makeBox(2, 2, 2, App.Vector(1, 1, 1)), "B")
        self.assertAlmostEqual(G.common_volume(a, b), 1.0)
        self.assertEqual(G.common_volume(a, self.doc.addObject("Part::Feature", "Empty")), 0.0)


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class BuilderTest(unittest.TestCase):
    def setUp(self):
        self.doc = App.newDocument("BuilderTest")
        self.addCleanup(App.closeDocument, self.doc.Name)
        r = build.build_stage(self.doc, spec.default_spec(), "params", board=BOARD)
        assert r.ok, r.message
        self.b = G.Builder(self.doc, kit_id="enc1", stage="shell", spec_hash="h" * 12)
        self.v = params.Params(self.doc).values()

    def sketch(self, **kw):
        base = dict(width=90.1, height=57.94, radius=1.0, origin=(-2.8, -1.3, -4.0),
                    exprs={"size_x": "Params.cav_len", "size_y": "Params.cav_wid", "rc": "Params.corner_r - Params.wall_t",
                           "Placement.Base.x": "Params.cav_x0", "Placement.Base.y": "Params.cav_y0", "Placement.Base.z": "Params.z_floor_top"})
        base.update(kw)
        return self.b.rrect_sketch("Sk", **base)

    def test_new_tags_hides_records_and_validates_the_role(self):
        o = self.b.new("Part::Box", "X", role="aux", part="Shell")
        self.assertEqual(G.tags_of(o), {"OscarKit": "enc1", "OscarStage": "shell", "OscarRole": "aux", "OscarPart": "Shell", "OscarSpecHash": "h" * 12})
        self.assertFalse(o.Visibility)
        self.assertEqual(self.b.created, ["X"])
        with self.assertRaises(G.KitGeometryError) as cm:
            self.b.new("Part::Box", "Y", role="bogus")
        self.assertIn("one of", str(cm.exception))
        self.assertNotIn("Y", [o.Name for o in self.doc.Objects])

    def test_rounded_rectangle_sketch_is_fully_constrained_and_named(self):
        sk = self.sketch()
        self.assertEqual(sk.solve(), 0)
        self.assertEqual(sk.DoF, 0)
        self.assertEqual(list(sk.ConflictingConstraints), [])
        self.assertEqual(list(sk.RedundantConstraints), [])
        self.assertEqual([c.Name for c in sk.Constraints if c.Name], ["rc", "size_x", "size_y"])
        self.assertEqual(len(sk.Geometry), 8)

    def test_sketch_expressions_drive_the_outline(self):
        sk = self.sketch()
        self.doc.recompute()
        exprs = exprs_of(sk)
        self.assertEqual(exprs["Constraints.rc"].replace(" ", ""), "Params.corner_r-Params.wall_t")
        self.assertEqual(exprs["Placement.Base.x"], "Params.cav_x0")
        ex = self.b.extrude("Ex", sk, length=23.0, length_expr="Params.z_cavity_top - Params.z_floor_top + 1 mm")
        self.doc.recompute()
        bb = ex.Shape.BoundBox
        self.assertEqual([round(x, 4) for x in (bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax)], [-2.8, -1.3, -4.0, 87.3, 56.64, 19.0])
        params.Params(self.doc).set("wall_t", 2.4)                                   # cavity radius follows corner_r - wall_t
        params.Params(self.doc).set("corner_r", 6.0)
        self.doc.recompute()
        self.assertTrue(ex.Shape.isValid())
        self.assertAlmostEqual(dict((c.Name, c.Value) for c in sk.Constraints if c.Name)["rc"], 3.6, places=6)

    def test_sketch_radius_is_clamped(self):
        tiny = self.sketch(radius=0.0, exprs={})
        self.assertEqual(tiny.solve(), 0)
        self.assertAlmostEqual(dict((c.Name, c.Value) for c in tiny.Constraints if c.Name)["rc"], G.MIN_RADIUS)
        huge = self.b.rrect_sketch("Sk2", width=10, height=6, radius=50, origin=(0, 0, 0), exprs={})
        self.assertEqual(huge.solve(), 0)
        self.assertLess(dict((c.Name, c.Value) for c in huge.Constraints if c.Name)["rc"], 3.0)      # at most half the short side

    def test_empty_outline_is_refused_before_anything_is_created(self):
        count = len(self.doc.Objects)
        for w, h in ((0, 5), (5, 0), (-1, 5)):
            with self.assertRaises(G.KitGeometryError) as cm:
                self.b.rrect_sketch("Bad", width=w, height=h, radius=1, origin=(0, 0, 0), exprs={})
            self.assertIn("empty", str(cm.exception))
        self.assertEqual(len(self.doc.Objects), count)

    def test_extrude_reverse_goes_against_the_normal(self):
        sk = self.b.rrect_sketch("Sk3", width=10, height=10, radius=1, origin=(0, 0, 10), exprs={})
        up = self.b.extrude("Up", sk, length=3.0, length_expr=None)
        down = self.b.extrude("Down", sk, length=3.0, length_expr=None, reverse=True)
        self.doc.recompute()
        self.assertEqual((round(up.Shape.BoundBox.ZMin, 6), round(up.Shape.BoundBox.ZMax, 6)), (10.0, 13.0))
        self.assertEqual((round(down.Shape.BoundBox.ZMin, 6), round(down.Shape.BoundBox.ZMax, 6)), (7.0, 10.0))
        self.assertEqual(len(up.Shape.Solids), 1)

    def test_box_cylinder_and_static_bindings(self):
        box = self.b.box("Bx", (2, 3, 4), (1, 1, 1), exprs={"Height": "Params.wall_t + 1 mm", "Placement.Base.z": "Params.z_floor_top"})
        cyl = self.b.cylinder("Cy", 1.5, 2.0, (0, 0, 0), exprs={"Radius": "Params.standoff_d/2", "Placement.Base.z": None})
        st = self.b.static("St", Part.makeBox(1, 1, 1), (5, 5, 5), exprs={"Placement.Base.z": "Params.z_cavity_top"})
        self.doc.recompute()
        self.assertEqual(box.Height.Value, 3.0)
        self.assertEqual(box.Placement.Base.z, -4.0)
        self.assertEqual(cyl.Radius.Value, 3.0)
        self.assertEqual(cyl.Placement.Base.z, 0.0)                                  # a None expression binds nothing
        self.assertEqual(st.Shape.BoundBox.ZMin, 18.0)
        params.Params(self.doc).set("lid_gap", 2.0)
        self.doc.recompute()
        self.assertEqual(st.Shape.BoundBox.ZMin, 19.0)                               # a static shape follows its bound placement

    def test_booleans_union_and_refine(self):
        a = self.b.box("A", (4, 4, 4), (0, 0, 0))
        c = self.b.box("C", (2, 2, 6), (1, 1, -1))
        d = self.b.box("D", (2, 2, 2), (10, 0, 0))
        cut = self.b.cut("Cut", a, c)
        fuse = self.b.fuse("Fuse", a, d)
        self.assertIs(self.b.union("One", [a]), a)
        multi = self.b.union("Multi", [a, c, d])
        self.assertEqual(multi.TypeId, "Part::MultiFuse")
        with self.assertRaises(G.KitGeometryError):
            self.b.union("None", [])
        self.doc.recompute()
        self.assertAlmostEqual(cut.Shape.Volume, 64 - 16)
        self.assertEqual(len(fuse.Shape.Solids), 2)
        self.assertTrue(cut.Refine and fuse.Refine)

    def test_bind_failure_names_the_object_the_property_and_the_expression(self):
        box = self.b.box("Bx", (1, 1, 1), (0, 0, 0))
        with self.assertRaises(G.KitGeometryError) as cm:
            self.b.bind(box, "Length", "Params.no_such_row + (")
        msg = str(cm.exception)
        self.assertIn("Bx.Length", msg)
        self.assertIn("Params.no_such_row", msg)

    def test_container_is_found_by_tags_not_by_name(self):
        c1 = self.b.container("Part_Shell", "Shell")
        c2 = self.b.container("Part_Shell", "Shell")
        self.assertIs(c1, c2)
        self.assertEqual(c1.TypeId, "App::Part")
        self.assertEqual(G.tags_of(c1)["OscarRole"], "container")
        other = G.Builder(self.doc, kit_id="enc2", stage="shell", spec_hash="x").container("Part_Shell", "Shell")
        self.assertIsNot(other, c1)
        self.assertNotEqual(other.Name, c1.Name)                                      # a second kit gets its own container
        lid = self.b.container("Part_Lid", "Lid")
        self.assertIsNot(lid, c1)

    def test_objects_land_in_their_container(self):
        cont = self.b.container("Part_Shell", "Shell")
        box = self.b.box("InBox", (1, 1, 1), (0, 0, 0), container=cont)
        self.assertIn(box, cont.Group)


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class SpanAndCenterTest(unittest.TestCase):
    """through_span / default_center on the Pi 4 defaults (cavity x -2.8..87.3, y -1.3..56.64, z -4..18; shell outer to -6)."""

    def setUp(self):
        self.v = default_values()

    def test_shell_wall_through_spans(self):
        for face, lo in (("-y", -4.3), ("+y", 55.64), ("-x", -5.8), ("+x", 86.3)):
            sp = C.through_span(self.v, "shell", face)
            self.assertAlmostEqual(sp["lo"], lo, places=6, msg=face)
            self.assertAlmostEqual(sp["len"], 4.0, places=6, msg=face)             # wall_t 2 plus 1 mm each side

    def test_shell_wall_blind_span_cuts_the_outer_half(self):
        sp = C.through_span(self.v, "shell", "-y", blind=True)
        self.assertAlmostEqual(sp["lo"], -4.3)
        self.assertAlmostEqual(sp["len"], 2.0)                                       # half the wall plus the overshoot
        sp = C.through_span(self.v, "shell", "+y", blind=True)
        self.assertAlmostEqual(sp["lo"], 57.64)
        self.assertAlmostEqual(sp["lo"] + sp["len"], 59.64)                          # 1 mm beyond the outer face at 58.64

    def test_shell_floor(self):
        sp = C.through_span(self.v, "shell", "bottom")
        self.assertAlmostEqual(sp["lo"], -7.0)
        self.assertAlmostEqual(sp["len"], 4.0)
        sp = C.through_span(self.v, "shell", "bottom", blind=True)
        self.assertAlmostEqual(sp["lo"], -7.0)
        self.assertAlmostEqual(sp["len"], 2.0)

    def test_lid_plate_and_skirt_band(self):
        top = C.through_span(self.v, "lid", "top")
        self.assertAlmostEqual(top["lo"], 17.0)
        self.assertAlmostEqual(top["len"], 4.0)
        blind = C.through_span(self.v, "lid", "top", blind=True)
        self.assertAlmostEqual(blind["lo"], 19.0)
        self.assertAlmostEqual(blind["len"], 2.0)
        px = C.through_span(self.v, "lid", "+x")
        self.assertAlmostEqual(px["lo"], 89.3)                                       # the shell's outer face
        self.assertAlmostEqual(px["len"], 2.45)                                      # lid_fit + lip_t + 1 mm: out past the skirt
        nx = C.through_span(self.v, "lid", "-x")
        self.assertAlmostEqual(nx["lo"], -7.25)
        self.assertAlmostEqual(nx["lo"] + nx["len"], -4.8)

    def test_faces_a_part_does_not_have(self):
        self.assertIsNone(C.through_span(self.v, "shell", "top"))
        self.assertIsNone(C.through_span(self.v, "lid", "bottom"))
        self.assertIsNone(C.through_span(self.v, "bracket", "top"))

    def test_expressions_name_the_params_rows(self):
        sp = C.through_span(self.v, "shell", "-x")
        self.assertEqual(sp["lo_expr"], "Params.cav_x0 - Params.wall_t - 1 mm")
        self.assertEqual(sp["len_expr"], "Params.wall_t + 2 mm")
        self.assertEqual(C.through_span(self.v, "shell", "+y")["lo_expr"], "Params.cav_y1 - 1 mm")

    def test_default_vent_centre(self):
        cu, cv, ue, ve = C.default_center(self.v, "top")
        self.assertAlmostEqual(cu, 42.25)
        self.assertAlmostEqual(cv, 27.67)
        self.assertEqual((ue, ve), ("Params.cav_x0 + Params.cav_len/2", "Params.cav_y0 + Params.cav_wid/2"))
        cu, cv, *_ = C.default_center(self.v, "+y")
        self.assertAlmostEqual(cu, 42.25)                                              # x centre, z mid-height of the cavity
        self.assertAlmostEqual(cv, 7.0)
        cu, cv, *_ = C.default_center(self.v, "-x")
        self.assertAlmostEqual(cu, 27.67)
        self.assertAlmostEqual(cv, 7.0)

    def test_sanitize(self):
        self.assertEqual(C.sanitize("usb_c-power 1"), "usb_c_power_1")
        self.assertEqual(C.sanitize(""), "x")


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class CutterTest(unittest.TestCase):
    def setUp(self):
        self.doc = App.newDocument("CutterTest")
        self.addCleanup(App.closeDocument, self.doc.Name)
        build.build_stage(self.doc, spec.default_spec(), "params", board=BOARD)
        self.v = params.Params(self.doc).values()
        self.ctx = types.SimpleNamespace(b=G.Builder(self.doc, kit_id="enc1", stage="ports", spec_hash="h"), v=self.v)

    def bbox(self, obj):
        self.doc.recompute()
        bb = obj.Shape.BoundBox
        return [round(x, 4) for x in (bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax)]

    def test_wall_opening_box(self):
        span = C.through_span(self.v, "shell", "-y")
        o = C.rect_cutter(self.ctx, "Cut1", part="Shell", container=None, face="-y", span=span, u0=5.2, u1=17.2, v0=-0.2, v1=6.3)
        self.assertEqual(self.bbox(o), [5.2, -4.3, -0.2, 17.2, -0.3, 6.3])         # wall_t 2 plus 1 mm either side: y -4.3 .. -0.3
        exprs = exprs_of(o)
        self.assertIn("Placement.Base.y", exprs)                                      # the wall-normal position is bound
        self.assertIn("Width", exprs)
        self.assertNotIn("Length", exprs)                                             # the in-plane size is a literal
        self.assertNotIn("Height", exprs)

    def test_side_wall_opening_uses_y_as_u(self):
        span = C.through_span(self.v, "shell", "+x")
        o = C.rect_cutter(self.ctx, "Cut2", part="Shell", container=None, face="+x", span=span, u0=1.0, u1=15.0, v0=0.9, v1=17.5)
        self.assertEqual(self.bbox(o), [86.3, 1.0, 0.9, 90.3, 15.0, 17.5])

    def test_floor_opening_uses_x_and_y(self):
        span = C.through_span(self.v, "shell", "bottom")
        o = C.rect_cutter(self.ctx, "Cut3", part="Shell", container=None, face="bottom", span=span, u0=10, u1=20, v0=5, v1=15)
        self.assertEqual(self.bbox(o), [10.0, 5.0, -7.0, 20.0, 15.0, -3.0])

    def test_in_plane_expressions(self):
        span = C.through_span(self.v, "shell", "-y")
        o = C.rect_cutter(self.ctx, "Cut4", part="Shell", container=None, face="-y", span=span, u0=5.0, u1=15.0, v0=1.0, v1=2.0,
                          exprs={"dv": "Params.z_cavity_top + 1 mm - 1 mm"})
        self.assertEqual(self.bbox(o)[5] - self.bbox(o)[2], 18.0)                      # bound height: 18 + 1 - 1
        self.assertIn("Height", exprs_of(o))

    def test_empty_cutter_is_refused(self):
        span = C.through_span(self.v, "shell", "-y")
        with self.assertRaises(G.KitGeometryError) as cm:
            C.rect_cutter(self.ctx, "Cut5", part="Shell", container=None, face="-y", span=span, u0=5, u1=5, v0=0, v1=1)
        self.assertIn("empty", str(cm.exception))

    def test_rounded_cutter_is_static_and_filleted(self):
        span = C.through_span(self.v, "shell", "+y")
        sharp = C.rect_cutter(self.ctx, "Sharp", part="Shell", container=None, face="+y", span=span, u0=10, u1=20, v0=0, v1=10)
        rounded = C.rect_cutter(self.ctx, "Round", part="Shell", container=None, face="+y", span=span, u0=10, u1=20, v0=0, v1=10, rounded=2.0)
        self.doc.recompute()
        self.assertEqual(rounded.TypeId, "Part::Feature")
        self.assertEqual(list(rounded.ExpressionEngine), [])                          # static: nothing bound
        self.assertLess(rounded.Shape.Volume, sharp.Shape.Volume)
        self.assertAlmostEqual(sharp.Shape.Volume - rounded.Shape.Volume, (4 - math.pi) * 4 * span["len"], places=3)   # four r=2 corners

    def test_slot_array_geometry_and_default_centre(self):
        span = C.through_span(self.v, "lid", "top")
        slots = C.slot_cutters(self.ctx, "V", part="Lid", container=None, face="top", span=span, center=None, count=3, slot_w=2.0,
                               slot_l=24.0, pitch=4.0, angle=0.0, pattern="slots")
        self.assertEqual(len(slots), 3)
        boxes = [self.bbox(s) for s in slots]
        self.assertEqual([b[3] - b[0] for b in boxes], [24.0] * 3)                    # slots lie along u (x)
        self.assertEqual([round(b[4] - b[1], 6) for b in boxes], [2.0] * 3)
        centres = [(b[1] + b[4]) / 2 for b in boxes]
        self.assertAlmostEqual(centres[1], 27.67, places=4)                           # the face centre over the cavity
        self.assertAlmostEqual(centres[0] - centres[1], -4.0, places=4)               # pitch across v
        self.assertAlmostEqual(centres[2] - centres[1], 4.0, places=4)
        self.assertTrue(all(b[2] == 17.0 and b[5] == 21.0 for b in boxes))
        self.assertIn("Placement.Base.x", exprs_of(slots[0]))                         # an unrotated default-centre array is bound

    def test_explicit_centre_is_a_literal(self):
        span = C.through_span(self.v, "lid", "top")
        (s,) = C.slot_cutters(self.ctx, "V", part="Lid", container=None, face="top", span=span, center=[30.0, 20.0], count=1, slot_w=2.0,
                              slot_l=10.0, pitch=4.0, angle=0.0, pattern="slots")
        self.assertEqual(self.bbox(s)[:2], [25.0, 19.0])
        exprs = exprs_of(s)
        self.assertNotIn("Placement.Base.x", exprs)
        self.assertNotIn("Placement.Base.y", exprs)
        self.assertIn("Placement.Base.z", exprs)                                      # only the wall-normal axis stays bound

    def test_rotated_array_keeps_its_centre(self):
        span = C.through_span(self.v, "lid", "top")
        slots = C.slot_cutters(self.ctx, "R", part="Lid", container=None, face="top", span=span, center=[40.0, 25.0], count=3, slot_w=2.0,
                               slot_l=20.0, pitch=5.0, angle=90.0, pattern="slots")
        self.doc.recompute()
        cs = [s.Shape.CenterOfMass for s in slots]
        self.assertAlmostEqual(cs[1].x, 40.0, places=4)
        self.assertAlmostEqual(cs[1].y, 25.0, places=4)
        self.assertAlmostEqual(cs[0].x - cs[1].x, 5.0, places=4)                      # rotated 90 degrees: the pitch runs along x now
        self.assertAlmostEqual(cs[0].y, 25.0, places=4)
        bb = slots[1].Shape.BoundBox
        self.assertAlmostEqual(bb.XLength, 2.0, places=4)                             # and the slots along y
        self.assertAlmostEqual(bb.YLength, 20.0, places=4)

    def test_hole_pattern_on_a_wall_points_along_the_normal(self):
        span = C.through_span(self.v, "shell", "+y")
        holes = C.slot_cutters(self.ctx, "H", part="Shell", container=None, face="+y", span=span, center=[30.0, 8.0], count=2, slot_w=3.0,
                               slot_l=99.0, pitch=6.0, angle=0.0, pattern="holes")
        self.doc.recompute()
        self.assertEqual([h.TypeId for h in holes], ["Part::Cylinder"] * 2)
        for h in holes:
            bb = h.Shape.BoundBox
            self.assertAlmostEqual(bb.YLength, 4.0, places=4)                         # the through span along y: wall_t 2 + 2 x 1 mm
            self.assertAlmostEqual(bb.XLength, 3.0, places=4)
            self.assertAlmostEqual(bb.ZLength, 3.0, places=4)
        self.assertAlmostEqual(holes[1].Shape.CenterOfMass.z - holes[0].Shape.CenterOfMass.z, 6.0, places=4)

    def test_hole_pattern_on_the_x_wall_and_the_top(self):
        for face, axis in (("-x", "XLength"), ("top", "ZLength")):
            span = C.through_span(self.v, "shell" if face != "top" else "lid", face)
            (h,) = C.slot_cutters(self.ctx, f"H{face}", part="Shell", container=None, face=face, span=span, center=[10.0, 5.0], count=1,
                                  slot_w=3.0, slot_l=1.0, pitch=4.0, angle=0.0, pattern="holes")
            self.doc.recompute()
            self.assertAlmostEqual(getattr(h.Shape.BoundBox, axis), span["len"], places=4, msg=face)


if __name__ == "__main__":
    unittest.main()
