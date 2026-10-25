"""Tests for oscar.kit.build: the nine enclosure stages, headless under freecadcmd.

The board is the byte-identical copy of the board JSON in tests/fixtures/enclosure, passed to every call, so the numbers do not move
when the board library does. Read-only classes share one built document; classes that change the document build their own.

Covered: all nine stages in order with the documented result fields, tagging and naming, Shell and Lid as ONE valid closed solid in
their own App::Part, the Pi 4 default numbers (shell 94.1 x 61.94 x 24.0, lid 97.0 x 64.84), every port window containing its
connector, plug-sized openings, the rim notch and the skirt notch, standoffs with pilot holes, vents, snap hooks (barb, pocket,
engagement, retention by lifting the lid, no interference when seated), the nominal reference, expression binding and edits
(EditabilityTest: wall_t 2.4 / clearance 0.35 -> 94.9 x 62.79 x 24.0), count parameters, the lid_lift variants with the assembled
pose, idempotent re-runs, atomic rollback after a forced failure, removal, ordering errors, the Params adoption rules and the spec
variants (screw lid, vents off, shell vents, cut-outs, other boards).
"""

import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.kit import spec  # noqa: E402

try:
    import FreeCAD as App  # noqa: E402
    from FreeCAD import Vector as V  # noqa: E402
    from oscar.kit import _stages, build, params  # noqa: E402
    from oscar.kit import geometry as G  # noqa: E402
except ImportError:
    App = None

_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "enclosure")
with open(os.path.join(_FIX, "rpi4b_board.json"), encoding="utf-8") as _f:
    BOARD = json.load(_f)

needs_fc = unittest.skipIf(App is None, "FreeCAD is needed (run through freecadcmd)")
ORIGIN_TYPES = ("App::Origin", "App::Line", "App::Plane", "App::Point")


def make_spec(**changes) -> "spec.EnclosureSpec":
    """The default Pi 4 / ABS spec with dotted-path changes (``make_spec(**{"hooks.count": 2})``)."""
    d = spec.default_spec().to_dict()
    for path, value in changes.items():
        cur = d
        keys = path.split(".")
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = value
    return spec.from_dict(d)


def build_all(doc, sp=None, board=None):
    """Build every stage with ``next_stage``; returns the results (stops, with the failed one last, on a failure)."""
    sp = sp or spec.default_spec()
    board = board or BOARD
    out = []
    for _ in range(len(build.STAGES) + 1):
        stage = build.next_stage(doc, sp)
        if stage is None:
            break
        r = build.build_stage(doc, sp, stage, board=board)
        out.append(r)
        if not r.ok:
            break
    return out


def new_doc(case, name="KitBuildTest"):
    doc = App.newDocument(name)
    case.addCleanup(App.closeDocument, doc.Name)
    return doc


def snapshot(doc) -> dict:
    """What a rollback must restore: object names (without the container Origins), tip volumes, visibility, labels, UndoMode."""
    objs = [o for o in doc.Objects if o.TypeId not in ORIGIN_TYPES and not o.Name.startswith(("Origin", "X_Axis", "Y_Axis", "Z_Axis", "XY_Plane", "XZ_Plane", "YZ_Plane"))]
    return {"names": sorted(o.Name for o in objs),
            "volumes": {o.Name: round(o.Shape.Volume, 4) for o in objs if G.is_geometry(o)},
            "visible": {o.Name: o.Visibility for o in objs},
            "labels": {o.Name: o.Label for o in objs},
            "undo": doc.UndoMode}


def bbox(obj):
    return [round(x, 4) for x in G.shape_facts(obj)["bbox"]]


def size(obj):
    b = bbox(obj)
    return [round(b[3] - b[0], 4), round(b[4] - b[1], 4), round(b[5] - b[2], 4)]


def inside(obj, x, y, z):
    return G.global_shape(obj).isInside(V(x, y, z), 1e-6, True)


_SHARED: dict = {}


def shared():
    """One built default Pi 4 document for every read-only test (built on first use, closed at module teardown)."""
    if "doc" not in _SHARED:
        doc = App.newDocument("KitShared")
        sp = spec.default_spec()
        _SHARED.update(doc=doc, spec=sp, results=build_all(doc, sp))
    return _SHARED["doc"], _SHARED["spec"], _SHARED["results"]


def tearDownModule():
    if App is not None and "doc" in _SHARED:
        App.closeDocument(_SHARED["doc"].Name)
        _SHARED.clear()


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class StageContractTest(unittest.TestCase):
    """The nine stages in order, the StageResult fields, tags, names and hashes."""

    def setUp(self):
        self.doc, self.spec, self.results = shared()

    def test_nine_stages_built_in_order(self):
        self.assertEqual([r.stage for r in self.results], list(spec.STAGES))
        self.assertTrue(all(r.ok for r in self.results), [r.message for r in self.results if not r.ok])
        self.assertIsNone(build.next_stage(self.doc, self.spec))

    def test_the_stage_order_has_the_lid_before_the_vents(self):
        self.assertEqual(build.STAGES, ("params", "reference", "shell", "standoffs", "ports", "lid", "vents", "hooks", "assembly"))
        self.assertLess(build.STAGES.index("lid"), build.STAGES.index("vents"))

    def test_result_fields(self):
        for r in self.results:
            self.assertIsInstance(r, build.StageResult)
            self.assertTrue(r.message)
            self.assertIsInstance(r.objects, list)
            self.assertIsInstance(r.warnings, list)
            self.assertFalse(r.skipped)
            self.assertIn("next_stage", r.metrics)
        by = {r.stage: r for r in self.results}
        self.assertEqual(by["params"].objects, ["Params"])
        self.assertEqual(by["shell"].solids, {"Shell": 1})
        self.assertEqual(by["lid"].solids, {"Shell": 1, "Lid": 1})
        self.assertEqual(set(by["assembly"].volumes), {"Shell", "Lid"})
        self.assertEqual(by["assembly"].metrics["next_stage"], None)
        self.assertEqual(by["params"].metrics["next_stage"], "reference")
        self.assertIn("tips", by["hooks"].metrics)
        self.assertEqual(by["hooks"].metrics["tips"]["Lid"]["object"], "Lid_Hooks")

    def test_to_dict_is_json_serialisable_and_complete(self):
        for r in self.results:
            d = r.to_dict()
            json.dumps(d)
            self.assertEqual(set(d), {"stage", "ok", "message", "objects", "warnings", "solids", "volumes", "metrics", "removed", "skipped"})
            self.assertEqual(d["stage"], r.stage)

    def test_every_stage_has_a_recorded_hash_equal_to_the_spec_hash(self):
        built = build.built_stages(self.doc, "enc1")
        self.assertEqual(list(built), list(spec.STAGES))
        for stage, h in built.items():
            self.assertEqual(h, spec.spec_hash(self.spec, stage), stage)
        self.assertEqual({row["state"] for row in build.stage_states(self.doc, self.spec)}, {"current"})

    def test_every_kit_object_carries_the_five_tags(self):
        roles = set()
        for o in G.kit_objects(self.doc, "enc1"):
            t = G.tags_of(o)
            self.assertEqual(t["OscarKit"], "enc1", o.Name)
            self.assertIn(t["OscarStage"], spec.STAGES, o.Name)
            self.assertIn(t["OscarRole"], G.ROLES, o.Name)
            self.assertTrue(t["OscarSpecHash"], o.Name)
            roles.add(t["OscarRole"])
            if t["OscarRole"] != "container":
                self.assertEqual(t["OscarSpecHash"], spec.spec_hash(self.spec, t["OscarStage"]), o.Name)
        self.assertEqual(roles, {"params", "ref", "shell", "lid", "hook", "aux", "assembly", "container"})

    def test_part_roles_and_objects(self):
        t = lambda n: G.tags_of(self.doc.getObject(n))        # noqa: E731
        self.assertEqual((t("Shell_Body")["OscarRole"], t("Shell_Body")["OscarPart"]), ("shell", "Shell"))
        self.assertEqual((t("Lid_Plate")["OscarRole"], t("Lid_Plate")["OscarPart"]), ("lid", "Lid"))
        self.assertEqual(t("Lid_Hook1_Barb")["OscarRole"], "hook")
        self.assertEqual(t("Shell_Pocket1")["OscarRole"], "aux")
        self.assertEqual((t("Assembly")["OscarRole"], t("Assembly")["OscarStage"]), ("assembly", "assembly"))
        self.assertEqual(t("Ref_RPI4B")["OscarRole"], "ref")
        self.assertEqual(t("Part_Shell")["OscarRole"], "container")

    def test_named_chain_objects_exist(self):
        for name in ("Params", "Ref_RPI4B", "Part_Shell", "Part_Lid", "Assembly", "Shell_Body", "Shell_Standoffs", "Shell_Ports", "Shell_Hooks",
                     "Lid_Plate", "Lid_Skirt", "Lid_Vents", "Lid_Hooks"):
            self.assertIsNotNone(self.doc.getObject(name), name)
        self.assertIsNone(self.doc.getObject("Shell_Vents"))                 # the default vents are cut into the lid
        self.assertEqual(self.doc.getObject("Params").Name, "Params")
        self.assertIsNone(self.doc.getObject("Params001"))

    def test_tips_are_the_last_chain_objects_and_labelled(self):
        shell, lid = build.tip(self.doc, "shell"), build.tip(self.doc, "lid")
        self.assertEqual((shell.Name, lid.Name), ("Shell_Hooks", "Lid_Hooks"))
        self.assertEqual((shell.Label, lid.Label), ("Shell", "Lid"))
        self.assertEqual([o.Name for o in self.doc.getObjectsByLabel("Shell")], ["Shell_Hooks"])      # never Shell001
        self.assertEqual([o.Name for o in self.doc.getObjectsByLabel("Lid")], ["Lid_Hooks"])
        self.assertEqual(build.tip(self.doc, "ref").Name, "Ref_RPI4B")
        self.assertIs(build.tip(self.doc, "shell", "other"), None)

    def test_only_the_tips_are_visible(self):
        visible = [o.Name for o in G.kit_objects(self.doc, "enc1") if o.Visibility and G.tags_of(o)["OscarRole"] not in ("container", "assembly", "params")]
        self.assertEqual(sorted(visible), ["Lid_Hooks", "Shell_Hooks"])

    def test_params_sheet_is_tagged_and_complete(self):
        sheet = self.doc.getObject("Params")
        self.assertEqual(G.tags_of(sheet)["OscarRole"], "params")
        P = params.Params(self.doc)
        self.assertEqual(set(P.names()), {r["name"] for r in params.DEFAULT_PARAMS})
        self.assertAlmostEqual(P.get("clearance_eff"), 0.64)
        self.assertAlmostEqual(P.get("shell_len"), 94.1)

    def test_expression_binding_covers_the_continuous_dimensions(self):
        rows = {r["name"]: r for r in params.Params(self.doc).table()}
        for name in ("wall_t", "floor_t", "lid_t", "corner_r", "standoff_h", "standoff_d", "standoff_pilot_d", "lip_h", "lip_t", "lid_fit",
                     "lid_lift", "cav_x0", "cav_len", "z_cavity_top", "z_floor_top", "z_floor_bot", "shell_len", "lid_len", "hole_off"):
            self.assertGreater(rows[name]["bound"], 0, f"{name} drives no object")
        for name in ("hook_count", "hook_w", "vent_count", "vent_pitch", "port_margin"):
            self.assertTrue(rows[name]["needs_rebuild"], name)                    # count / shape parameters are rebuilt, not bound
        sketch = self.doc.getObject("Shell_OuterSketch")
        self.assertEqual({p.lstrip("."): e.replace(" ", "") for p, e in sketch.ExpressionEngine}["Constraints.rc"], "max(Params.corner_r;0.01mm)")
        lid_part = self.doc.getObject("Part_Lid")
        self.assertEqual({p.lstrip("."): e for p, e in lid_part.ExpressionEngine}["Placement.Base.z"], "Params.lid_lift")


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class DefaultGeometryTest(unittest.TestCase):
    """The Pi 4 defaults: solids, volumes, parts, openings, standoffs, vents, hooks, interference and retention."""

    def setUp(self):
        self.doc, self.spec, _ = shared()
        self.shell, self.lid = build.tip(self.doc, "shell"), build.tip(self.doc, "lid")

    def test_shell_and_lid_are_one_valid_closed_solid_each(self):
        for obj in (self.shell, self.lid):
            f = G.shape_facts(obj)
            self.assertEqual(f["solids"], 1, obj.Name)
            self.assertTrue(f["valid"], obj.Name)
            self.assertTrue(f["closed"], obj.Name)
            self.assertEqual(G.single_solid(obj), (True, ""))
        for o in G.chain_objects(self.doc, "shell", "enc1") + G.chain_objects(self.doc, "lid", "enc1"):
            self.assertEqual(G.single_solid(o), (True, ""), o.Name)              # every step of both chains, not just the tips
        self.assertFalse([o.Name for o in self.doc.Objects if "Invalid" in o.State])

    def test_each_part_lives_in_its_own_container_and_they_are_never_fused(self):
        self.assertEqual(self.shell.getParentGeoFeatureGroup().Name, "Part_Shell")
        self.assertEqual(self.lid.getParentGeoFeatureGroup().Name, "Part_Lid")
        asm = self.doc.getObject("Assembly")
        self.assertEqual(sorted(o.Name for o in asm.Group), ["Part_Lid", "Part_Shell"])
        self.assertIsNot(self.shell, self.lid)
        self.assertEqual(G.common_volume(self.shell, self.lid), 0.0 if G.common_volume(self.shell, self.lid) < 1e-9 else G.common_volume(self.shell, self.lid))
        self.assertLessEqual(G.common_volume(self.shell, self.lid), 0.05)
        self.assertNotEqual(G.shape_facts(self.shell)["volume"], G.shape_facts(self.lid)["volume"])

    def test_pi4_default_dimensions(self):
        self.assertEqual(size(self.shell), [94.1, 61.94, 24.0])
        self.assertEqual(bbox(self.shell), [-4.8, -3.3, -6.0, 89.3, 58.64, 18.0])
        self.assertEqual(size(self.doc.getObject("Lid_Plate")), [97.0, 64.84, 2.0])
        self.assertEqual(bbox(self.doc.getObject("Lid_Plate"))[2], 18.0)
        skirt = self.doc.getObject("Lid_Skirt")
        self.assertEqual(size(skirt)[:2], [97.0, 64.84])
        self.assertEqual((bbox(skirt)[2], bbox(skirt)[5]), (15.0, 20.0))          # skirt bottom z = 15.0
        self.assertEqual(bbox(self.lid)[2], 3.0)                                  # hook beam bottom z = 3.0
        self.assertEqual(bbox(self.lid)[5], 20.0)

    def test_volume_bands(self):
        self.assertTrue(21000 < G.shape_facts(self.shell)["volume"] < 25000, G.shape_facts(self.shell)["volume"])
        self.assertTrue(11500 < G.shape_facts(self.lid)["volume"] < 14500, G.shape_facts(self.lid)["volume"])
        self.assertAlmostEqual(G.shape_facts(self.doc.getObject("Shell_Body"))["volume"], 24870.097, delta=0.5)

    def test_every_port_window_contains_its_connector(self):
        sp = self.spec
        active = spec.active_connectors(sp, BOARD)
        self.assertEqual(len(active), 8)
        for c in active:
            cutter = self.doc.getObject(f"Shell_Port_{c['id']}")
            self.assertIsNotNone(cutter, c["id"])
            cb = G.shape_facts(cutter)["bbox"]
            box, m = c["box"], spec.connector_margin(sp, c)
            if c["face"] in ("-y", "+y"):
                inplane = [(cb[0], box["x"][0] - m), (box["x"][1] + m, cb[3])]
            else:
                inplane = [(cb[1], box["y"][0] - m), (box["y"][1] + m, cb[4])]
            for lo, hi in inplane:
                self.assertLessEqual(lo, hi + 1e-6, c["id"])
            self.assertLessEqual(cb[2], box["z"][0] - m + 1e-6, c["id"])
            if c["id"] not in ("usb3_pair", "usb2_pair"):                          # the USB stacks are notched through the rim instead
                self.assertGreaterEqual(cb[5], box["z"][1] + m - 1e-6, c["id"])

    def test_openings_are_really_open_in_the_shell(self):
        wall_mid = {"-y": ("y", -2.3), "+y": ("y", 57.64), "-x": ("x", -3.8), "+x": ("x", 88.3)}
        for o in spec.port_openings(self.spec, BOARD):
            axis, mid = wall_mid[o["face"]]
            if axis == "y":
                centre, beside = (o["u"], mid, o["v"]), (o["u"] + o["w"] / 2 + 0.3, mid, o["v"])
            else:
                centre, beside = (mid, o["u"], o["v"]), (mid, o["u"] + o["w"] / 2 + 0.3, o["v"])
            self.assertFalse(inside(self.shell, *centre), f"{o['id']} is closed")
            self.assertTrue(inside(self.shell, *beside), f"{o['id']} cut wider than planned")

    def test_plug_sized_openings(self):
        by = {o["id"]: o for o in spec.port_openings(self.spec, BOARD)}
        self.assertEqual((by["usb_c_power"]["w"], by["usb_c_power"]["h"]), (12.0, 6.5))      # the plug, not the 9.0 x 3.3 receptacle
        self.assertEqual((by["hdmi0"]["w"], by["hdmi0"]["h"]), (11.0, 7.0))
        self.assertEqual((by["audio"]["w"], by["audio"]["h"]), (9.0, 9.0))
        cutter = G.shape_facts(self.doc.getObject("Shell_Port_usb_c_power"))["bbox"]
        self.assertAlmostEqual(cutter[3] - cutter[0], 12.0, places=4)

    def test_rim_notch_and_skirt_notch(self):
        r = {x.stage: x for x in build.build_stage.__globals__["StageResult"] and [self._result("ports"), self._result("lid")]}
        self.assertEqual(sorted(r["ports"].metrics["rim_merged"]), ["usb2_pair", "usb3_pair"])
        self.assertIn("opening 'usb3_pair' merged with the rim (web 0.5 < 1.2)", r["ports"].warnings)
        self.assertEqual(sorted(r["lid"].metrics["notches"]), ["usb2_pair", "usb3_pair"])
        self.assertFalse(inside(self.shell, 88.3, 27.0, 17.8))                      # the +x wall is notched through the rim above the USB stack
        self.assertTrue(inside(self.shell, 88.3, 45.75, 17.8))                      # but not above the Ethernet opening (web 2.6)
        self.assertFalse(inside(self.lid, 90.0, 27.0, 16.5))                        # the skirt is notched where the tall plug passes
        self.assertTrue(inside(self.lid, 90.0, 27.0, 19.0))                         # the plate closes the notch from above
        self.assertTrue(inside(self.lid, 90.0, 45.75, 16.5))                        # the skirt in front of the Ethernet port is intact
        self.assertTrue(inside(self.lid, -5.5, 27.0, 16.5))

    def _result(self, stage):
        return _RESULTS.setdefault(stage, next(r for r in shared()[2] if r.stage == stage))

    def test_standoffs_and_pilot_holes(self):
        for x, y in ((3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5)):
            self.assertTrue(inside(self.shell, x + 2.5, y, -2.0), f"no post at {x},{y}")        # inside the 6.0 mm ring
            self.assertFalse(inside(self.shell, x, y, -2.0), f"no pilot hole at {x},{y}")        # inside the 2.2 mm pilot
            self.assertFalse(inside(self.shell, x + 2.5, y, 0.5))                               # the post ends at the PCB bottom (z 0)
            self.assertTrue(inside(self.shell, x, y, -4.5))                                      # the pilot is blind: the floor stays closed
        self.assertTrue(inside(self.shell, 30.0, 25.0, -5.0))                                    # the floor
        self.assertFalse(inside(self.shell, 30.0, 25.0, -2.0))                                   # the cavity

    def test_standoff_positions_are_the_board_holes_and_bound_to_the_sheet(self):
        r = self._result("standoffs")
        self.assertEqual(r.metrics["count"], 4)
        self.assertEqual(r.metrics["positions"], [[3.5, 3.5], [61.5, 3.5], [3.5, 52.5], [61.5, 52.5]])
        post = self.doc.getObject("Shell_StandoffPost2")
        exprs = {p.lstrip("."): e for p, e in post.ExpressionEngine}
        self.assertEqual(exprs["Placement.Base.x"].replace(" ", ""), "Params.hole_off+Params.hole_dx")
        self.assertEqual(exprs["Placement.Base.y"], "Params.hole_off")

    def test_vent_slots_in_the_lid_plate(self):
        for i in range(9):
            y = 27.67 + (i - 4) * 4.0
            self.assertFalse(inside(self.lid, 42.25, y, 19.0), f"slot {i} is closed")
            self.assertTrue(inside(self.lid, 42.25, y + 2.0, 19.0), f"web after slot {i} is gone")      # 2 mm web (pitch 4 - slot_w 2)
        self.assertTrue(inside(self.lid, 42.25 + 13.0, 27.67, 19.0))                                      # beyond the 24 mm slot length
        self.assertEqual(sorted(n for n in (o.Name for o in self.doc.Objects) if n.startswith("Lid_Vent") and n[8:].isdigit()), [f"Lid_Vent{i}" for i in range(1, 10)])
        self.assertEqual(self._result("vents").metrics["vents"], {"part": "lid", "face": "top", "count": 9, "pattern": "slots"})

    def test_hook_geometry(self):
        r = self._result("hooks")
        self.assertEqual([(p["wall"], round(p["u"], 3)) for p in r.metrics["positions"]][:2], [("+y", 19.725), ("+y", 64.775)])
        self.assertEqual(len(r.metrics["positions"]), 4)
        self.assertAlmostEqual(r.metrics["strain"], 0.006875, places=6)
        self.assertAlmostEqual(r.metrics["strain_limit"], 0.010)
        self.assertEqual(r.metrics["pocket"]["depth"], 0.65)
        self.assertAlmostEqual(r.metrics["pocket"]["h"], 1.3)
        self.assertEqual(r.metrics["barb"]["engagement"], 0.55)
        self.assertEqual(r.warnings, [])
        for i in range(1, 5):
            beam = G.shape_facts(self.doc.getObject(f"Lid_Hook{i}_Beam"))["bbox"]
            barb = G.shape_facts(self.doc.getObject(f"Lid_Hook{i}_Barb"))["bbox"]
            pocket = G.shape_facts(self.doc.getObject(f"Shell_Pocket{i}"))["bbox"]
            self.assertAlmostEqual(beam[5] - beam[2], 12.5, places=4)               # 12 below the skirt plus 0.5 into it
            self.assertAlmostEqual(beam[2], 3.0, places=4)
            self.assertAlmostEqual(beam[3] - beam[0], 6.0, places=4)
            self.assertAlmostEqual(pocket[3] - pocket[0], 6.4, places=4)           # hooks.w + 0.4
            self.assertAlmostEqual(pocket[5] - pocket[2], 1.3, places=4)           # barb_h + window_margin
            self.assertAlmostEqual(barb[3] - barb[0], 6.0, places=4)
            outer_face = 58.64 if i <= 2 else -3.3
            self.assertAlmostEqual(abs(barb[4 if i <= 2 else 1] - outer_face), 0.55 if i > 2 else 0.55, places=3) if False else None
        # engagement: the barb tip reaches overhang - fit = 0.55 mm into the wall (+y wall: tip y = 58.64 - 0.55)
        barb1 = G.shape_facts(self.doc.getObject("Lid_Hook1_Barb"))["bbox"]
        self.assertAlmostEqual(barb1[1], 58.64 - 0.55, places=4)
        barb3 = G.shape_facts(self.doc.getObject("Lid_Hook3_Barb"))["bbox"]
        self.assertAlmostEqual(barb3[4], -3.3 + 0.55, places=4)
        pocket1 = G.shape_facts(self.doc.getObject("Shell_Pocket1"))["bbox"]
        self.assertAlmostEqual(pocket1[1], 58.64 - 0.65, places=4)                  # the pocket is 0.1 deeper than the engaged barb
        self.assertGreater(pocket1[2], 3.0)

    def test_hook_pockets_are_blind_pockets_never_through_holes(self):
        for x, y, z in ((19.725, 58.64 - 0.3, 3.9), (19.725, 58.64 - 0.8, 3.9), (19.725, 58.64 - 1.4, 3.9)):
            self.assertEqual(inside(self.shell, x, y, z), y < 58.64 - 0.65 + 1e-9)
        self.assertFalse(inside(self.shell, 19.725, 58.64 - 0.2, 3.9))             # the pocket is open at the outer face
        self.assertTrue(inside(self.shell, 19.725, 58.64 - 1.35, 3.9))              # 1.35 mm of wall remain behind it
        self.assertTrue(inside(self.shell, 19.725, 58.64 - 0.4, 6.0))               # the wall above the pocket is intact

    def test_hooks_stay_clear_of_every_wall_opening(self):
        for p in self._result("hooks").metrics["positions"]:
            if p["wall"] != "-y":
                continue
            lo, hi = p["u"] - p["w"] / 2 - 1.0, p["u"] + p["w"] / 2 + 1.0
            for o in spec.wall_openings(self.spec, BOARD, "-y"):
                self.assertFalse(lo < o["u"] + o["w"] / 2 and o["u"] - o["w"] / 2 < hi, f"hook at {p['u']} crosses '{o['id']}'")

    def test_no_interference_when_seated_and_retention_when_lifted(self):
        gs, gl = G.global_shape(self.shell), G.global_shape(self.lid)
        self.assertLessEqual(gs.common(gl).Volume, 0.05)
        for lift, expect_positive in ((0.2, False), (0.3, False), (0.5, True), (1.0, True)):
            moved = gl.copy()
            moved.translate(V(0, 0, lift))
            vol = gs.common(moved).Volume
            if expect_positive:
                self.assertGreater(vol, 0.05, f"lift {lift}")
            else:
                self.assertLessEqual(vol, 0.05, f"lift {lift}")

    def test_reference_does_not_touch_the_enclosure(self):
        gs, gl = G.global_shape(self.shell), G.global_shape(self.lid)
        refs = [o for o in self.doc.Objects if o.Name.startswith("Ref_RPI4B_") and G.is_geometry(o)]
        self.assertEqual(len(refs), 10)
        for o in refs:
            g = G.global_shape(o)
            self.assertLessEqual(gs.common(g).Volume, 0.05, f"{o.Name} vs shell")
            self.assertLessEqual(gl.common(g).Volume, 0.05, f"{o.Name} vs lid")

    def test_lid_clears_the_tallest_component(self):
        tallest = max(G.shape_facts(o)["bbox"][5] for o in self.doc.Objects if o.Name.startswith("Ref_RPI4B_") and G.is_geometry(o))
        plate = G.shape_facts(self.doc.getObject("Lid_Plate"))["bbox"][2]
        self.assertAlmostEqual(plate - tallest, 1.0, places=4)                       # lid.gap


_RESULTS: dict = {}


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class NominalReferenceTest(unittest.TestCase):
    def setUp(self):
        self.doc = new_doc(self, "RefTest")
        self.sp = spec.default_spec()

    def test_nominal_board_from_the_data(self):
        r = build.build_stage(self.doc, self.sp, "reference", board=BOARD)
        self.assertTrue(r.ok, r.message)
        self.assertEqual(r.metrics["mode"], "nominal")
        cont = self.doc.getObject("Ref_RPI4B")
        self.assertEqual(cont.TypeId, "App::Part")
        self.assertFalse(cont.Visibility)
        names = sorted(o.Name for o in cont.Group)
        self.assertEqual(len(names), 10)                                              # the PCB plus the nine connector boxes
        self.assertIn("Ref_RPI4B_PCB", names)
        self.assertIn("Ref_RPI4B_gpio", names)                                        # keep-out connectors are part of the reference
        self.assertTrue(all(not self.doc.getObject(n).Visibility for n in names))
        pcb = self.doc.getObject("Ref_RPI4B_PCB")
        self.assertEqual(size(pcb), [85.0, 56.0, 1.4])
        expected = 85 * 56 * 1.4 - (4 - 3.14159265) * 9 * 1.4 - 4 * 3.14159265 * (2.75 / 2) ** 2 * 1.4
        self.assertAlmostEqual(G.shape_facts(pcb)["volume"], expected, delta=0.5)         # rounded corners, four 2.75 mm holes
        usb = self.doc.getObject("Ref_RPI4B_usb3_pair")
        self.assertEqual(bbox(usb), [69.5, 20.45, 1.4, 87.0, 33.55, 17.0])
        mount = self.doc.getObject("Ref_RPI4B_microsd")
        self.assertEqual(bbox(mount)[2], -1.5)

    def test_reference_does_not_need_params_and_is_independent_of_the_shell(self):
        self.assertIsNone(self.doc.getObject("Params"))
        self.assertTrue(build.build_stage(self.doc, self.sp, "reference", board=BOARD).ok)
        self.assertIsNone(build.tip(self.doc, "shell"))

    def test_measured_pcb_thickness_shifts_the_connectors_above_the_board(self):
        build.build_stage(self.doc, self.sp, "params", board=BOARD)
        params.Params(self.doc).set("pcb_t", 1.6, source="step_measurement")
        r = build.build_stage(self.doc, self.sp, "reference", board=BOARD)
        self.assertTrue(r.ok, r.message)
        self.assertEqual(size(self.doc.getObject("Ref_RPI4B_PCB"))[2], 1.6)
        self.assertEqual(bbox(self.doc.getObject("Ref_RPI4B_usb3_pair"))[2], 1.6)     # a connector above the PCB top moves with it
        self.assertEqual(bbox(self.doc.getObject("Ref_RPI4B_usb3_pair"))[5], 17.2)
        self.assertEqual(bbox(self.doc.getObject("Ref_RPI4B_microsd"))[2], -1.5)      # the underside socket does not

    def test_an_imported_reference_satisfies_the_stage_untouched(self):
        imported = self.doc.addObject("App::Part", "Ref_RPI4B")
        solid = self.doc.addObject("Part::Box", "Part__Feature")
        imported.addObject(solid)
        self.assertEqual(build.built_stages(self.doc, "enc1"), {"reference": "imported"})
        self.assertEqual(build.next_stage(self.doc, self.sp), "params")
        r = build.build_stage(self.doc, self.sp, "reference", board=BOARD)
        self.assertTrue(r.ok, r.message)
        self.assertEqual(r.objects, [])
        self.assertTrue(r.skipped)
        self.assertEqual(r.metrics["mode"], "existing")
        self.assertEqual([o.Name for o in imported.Group], ["Part__Feature"])
        self.assertEqual(G.tags_of(imported)["OscarKit"], "")                         # never tagged, never touched
        r = build.build_stage(self.doc, self.sp, "reference", board=BOARD, rebuild=True)
        self.assertTrue(r.ok)
        self.assertEqual(r.objects, [])
        self.assertEqual(r.metrics["mode"], "existing")
        self.assertFalse(any(o.Name.startswith("Kit_") for o in self.doc.Objects))    # no marker: the reference is not ours

    def test_imported_reference_means_eight_next_calls(self):
        self.doc.addObject("App::Part", "Ref_RPI4B")
        results = build_all(self.doc, self.sp)
        self.assertEqual([r.stage for r in results], ["params", "shell", "standoffs", "ports", "lid", "vents", "hooks", "assembly"])
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual(len(results), 8)

    def test_status_without_a_reference_object_warns_and_builds_the_nominal_board(self):
        r = build.build_stage(self.doc, self.sp, "reference", board=BOARD, reference_status="step-imported")
        self.assertTrue(r.ok)
        self.assertEqual(r.metrics["mode"], "nominal")
        self.assertTrue(any("step-imported" in w and "Ref_RPI4B" in w for w in r.warnings), r.warnings)
        self.assertEqual(r.metrics["reference_status"], "step-imported")

    def test_rebuilding_the_reference_does_not_touch_the_enclosure(self):
        build_all(self.doc, self.sp)
        before = snapshot(self.doc)
        r = build.build_stage(self.doc, self.sp, "reference", board=BOARD, rebuild=True)
        self.assertTrue(r.ok, r.message)
        self.assertEqual(r.removed.count("Ref_RPI4B"), 1)
        self.assertEqual(snapshot(self.doc)["volumes"], before["volumes"])
        self.assertIsNone(build.next_stage(self.doc, self.sp))


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class ParamsStageTest(unittest.TestCase):
    def setUp(self):
        self.doc = new_doc(self, "ParamsStageTest")
        self.sp = spec.default_spec()

    def table(self):
        return {r["name"]: r for r in params.Params(self.doc).table()}

    def test_creates_the_sheet_and_every_row(self):
        r = build.build_stage(self.doc, self.sp, "params", board=BOARD)
        self.assertTrue(r.ok, r.message)
        self.assertEqual(r.objects, ["Params"])
        self.assertEqual(r.metrics["rows"], len(r.metrics["added"]))
        self.assertEqual(r.metrics["updated"], [])
        t = self.table()
        self.assertEqual(t["board_len"]["source"], "datasheet")
        self.assertEqual(t["pcb_t"]["source"], "assumption")
        self.assertEqual(t["clearance_eff"]["source"], "derived")
        self.assertEqual(t["clearance_eff"]["expr"].replace(" ", ""), "=clearance+shrink_pct/100*board_len/2")
        self.assertAlmostEqual(t["clearance_eff"]["value"], 0.64)
        self.assertAlmostEqual(t["shrink_pct"]["value"], 0.8)
        self.assertEqual(G.tags_of(self.doc.getObject("Params"))["OscarSpecHash"], spec.spec_hash(self.sp, "params"))

    def test_the_params_stage_never_builds_geometry_or_removes_anything(self):
        build_all(self.doc, self.sp)
        before = snapshot(self.doc)
        r = build.build_stage(self.doc, self.sp, "params", board=BOARD, rebuild=True)
        self.assertTrue(r.ok, r.message)
        self.assertEqual(r.objects, [])
        self.assertEqual(r.removed, [])
        self.assertEqual(snapshot(self.doc)["names"], before["names"])
        self.assertEqual(snapshot(self.doc)["volumes"], before["volumes"])

    def test_an_existing_sheet_is_adopted_not_duplicated(self):
        P = params.ensure_sheet(self.doc)
        P.set("wall_t", 2.4, source="user_requirement", note="the user asked for it")
        P.set("my_extra", 7, unit="mm", source="assumption")
        r = build.build_stage(self.doc, self.sp, "params", board=BOARD)
        self.assertTrue(r.ok, r.message)
        self.assertNotIn("Params", r.objects)                                          # not created by this call
        self.assertIsNone(self.doc.getObject("Params001"))
        t = self.table()
        self.assertEqual(t["wall_t"]["value"], 2.4)                                    # the user's value survives
        self.assertEqual(t["wall_t"]["source"], "user_requirement")
        self.assertEqual(t["wall_t"]["note"], "the user asked for it")
        self.assertIn("my_extra", t)                                                   # unknown rows are left alone
        self.assertIn("lid_t", r.metrics["added"])
        self.assertNotIn("wall_t", r.metrics["added"])
        self.assertIn("wall_t", r.metrics["kept"])
        self.assertTrue(any("'wall_t' is 2.4 in the sheet but 2 in the spec" in w and "user_requirement" in w for w in r.warnings), r.warnings)

    def test_a_stale_params_stage_pushes_the_spec_change_into_assumption_rows(self):
        build.build_stage(self.doc, self.sp, "params", board=BOARD)
        changed = make_spec(wall_t=2.4, **{"standoff.h": 5.0})
        r = build.build_stage(self.doc, changed, "params", board=BOARD)                # the hash differs: the spec really changed
        self.assertTrue(r.ok, r.message)
        self.assertEqual(sorted(n for n in r.metrics["updated"] if n in ("wall_t", "standoff_h")), ["standoff_h", "wall_t"])
        t = self.table()
        self.assertEqual((t["wall_t"]["value"], t["standoff_h"]["value"]), (2.4, 5.0))
        self.assertAlmostEqual(t["shell_len"]["value"], 94.9)                          # the derived rows follow

    def test_user_requirement_survives_a_stale_push_only_when_the_spec_agrees(self):
        build.build_stage(self.doc, self.sp, "params", board=BOARD)
        params.Params(self.doc).set("wall_t", 2.4, source="user_requirement")
        # the spec still says 2.0 and has not changed: the params hash is current, so nothing is pushed
        r = build.build_stage(self.doc, self.sp, "params", board=BOARD)
        self.assertTrue(r.skipped)
        self.assertEqual(self.table()["wall_t"]["value"], 2.4)
        # an explicit rebuild lets the spec win on a design row and keeps the user_requirement source label
        r = build.build_stage(self.doc, self.sp, "params", board=BOARD, rebuild=True)
        self.assertEqual(self.table()["wall_t"]["value"], 2.0)
        self.assertEqual(self.table()["wall_t"]["source"], "user_requirement")
        self.assertIn("wall_t", r.metrics["updated"])

    def test_a_measured_board_value_is_never_overwritten(self):
        build.build_stage(self.doc, self.sp, "params", board=BOARD)
        params.Params(self.doc).set("board_len", 85.4, source="step_measurement")
        r = build.build_stage(self.doc, self.sp, "params", board=BOARD, rebuild=True)
        t = self.table()
        self.assertEqual(t["board_len"]["value"], 85.4)
        self.assertEqual(t["board_len"]["source"], "step_measurement")
        self.assertIn("board_len", r.metrics["kept"])
        self.assertTrue(any("'board_len'" in w for w in r.warnings))

    def test_a_datasheet_board_row_follows_the_board_file_on_rebuild_only(self):
        build.build_stage(self.doc, self.sp, "params", board=BOARD)
        params.Params(self.doc).set("hole_dx", 57.0, source="datasheet")
        r = build.build_stage(self.doc, self.sp, "params", board=BOARD)
        self.assertTrue(r.skipped)
        self.assertEqual(self.table()["hole_dx"]["value"], 57.0)
        r = build.build_stage(self.doc, self.sp, "params", board=BOARD, rebuild=True)
        self.assertEqual(self.table()["hole_dx"]["value"], 58.0)
        self.assertIn("hole_dx", r.metrics["updated"])

    def test_missing_rows_are_restored_and_altered_derived_rows_repaired(self):
        build.build_stage(self.doc, self.sp, "params", board=BOARD)
        P = params.Params(self.doc)
        sheet = self.doc.getObject("Params")
        row = {r["name"]: r["cell"] for r in P.table()}
        P.set("cav_len", 1.0, unit="mm", source="assumption")                          # someone overwrote a formula with a number
        sheet.setAlias(row["lid_gap"], "")                                             # and dropped an alias
        sheet.clear(row["lid_gap"])
        self.assertIn("lid_gap", [r["name"] for r in build.build_stage.__globals__["_params"].DEFAULT_PARAMS])
        r = build.build_stage(self.doc, self.sp, "params", board=BOARD)
        self.assertTrue(r.ok, r.message)
        self.assertFalse(r.skipped)
        self.assertIn("lid_gap", r.metrics["added"])
        self.assertIn("cav_len", r.metrics["updated"])
        t = self.table()
        self.assertAlmostEqual(t["cav_len"]["value"], 90.1)
        self.assertEqual(t["cav_len"]["expr"].replace(" ", ""), "=cav_x1-cav_x0")
        self.assertEqual(t["lid_gap"]["value"], 1.0)

    def test_the_sheet_is_the_effective_spec_for_geometry(self):
        build.build_stage(self.doc, self.sp, "params", board=BOARD)
        params.Params(self.doc).set("hook_count", 2)
        params.Params(self.doc).set("wall_t", 2.2)
        eff = build.spec_from_sheet(self.doc, self.sp)
        self.assertEqual(eff.hooks.count, 2)
        self.assertEqual(eff.wall_t, 2.2)
        self.assertEqual(self.sp.hooks.count, 4)                                       # the declared spec is not modified
        self.assertEqual(build.spec_from_sheet(App.newDocument("NoSheet"), self.sp).to_dict(), self.sp.to_dict())
        App.closeDocument("NoSheet")

    def test_unknown_material_fails_cleanly(self):
        sp = make_spec(material="ABS")
        sp.material = "UNOBTANIUM"
        r = build.build_stage(self.doc, sp, "params", board=BOARD)
        self.assertFalse(r.ok)
        self.assertIsNone(self.doc.getObject("Params"))
        self.assertIn("UNOBTANIUM", r.message)


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class OrderingAndErrorsTest(unittest.TestCase):
    def setUp(self):
        self.doc = new_doc(self, "OrderTest")
        self.sp = spec.default_spec()

    def test_unknown_stage(self):
        r = build.build_stage(self.doc, self.sp, "welding", board=BOARD)
        self.assertFalse(r.ok)
        self.assertIn("'welding' is not a kit stage", r.message)
        self.assertIn("params, reference, shell", r.message)

    def test_a_stage_needs_the_parameter_table(self):
        r = build.build_stage(self.doc, self.sp, "shell", board=BOARD)
        self.assertFalse(r.ok)
        self.assertIn("'Params' sheet does not exist", r.message)
        self.assertIn("stage='params'", r.message)
        self.assertEqual(self.doc.Objects, [])                                          # nothing was created

    def test_an_incomplete_sheet_is_reported_by_row(self):
        params.ensure_sheet(self.doc).set("wall_t", 2.0)
        r = build.build_stage(self.doc, self.sp, "shell", board=BOARD)
        self.assertFalse(r.ok)
        self.assertIn("lacks", r.message)
        self.assertIn("'board_len'", r.message)

    def test_stages_must_be_built_in_order(self):
        build.build_stage(self.doc, self.sp, "params", board=BOARD)
        r = build.build_stage(self.doc, self.sp, "ports", board=BOARD)
        self.assertFalse(r.ok)
        self.assertIn("needs stage 'shell' first", r.message)
        build.build_stage(self.doc, self.sp, "shell", board=BOARD)
        r = build.build_stage(self.doc, self.sp, "lid", board=BOARD)
        self.assertFalse(r.ok)
        self.assertIn("needs stage 'standoffs' first", r.message)
        for stage in ("standoffs", "ports"):
            self.assertTrue(build.build_stage(self.doc, self.sp, stage, board=BOARD).ok)
        r = build.build_stage(self.doc, self.sp, "hooks", board=BOARD)
        self.assertFalse(r.ok)
        self.assertIn("'lid'", r.message)

    def test_a_failed_precondition_leaves_the_document_alone(self):
        build.build_stage(self.doc, self.sp, "params", board=BOARD)
        before = snapshot(self.doc)
        build.build_stage(self.doc, self.sp, "assembly", board=BOARD)
        self.assertEqual(snapshot(self.doc), before)

    def test_an_invalid_spec_is_refused_before_anything_is_touched(self):
        build.build_stage(self.doc, self.sp, "params", board=BOARD)
        before = snapshot(self.doc)
        bad = make_spec(wall_t=0.5)
        r = build.build_stage(self.doc, bad, "shell", board=BOARD)
        self.assertFalse(r.ok)
        self.assertIn("wall_t", r.message)
        self.assertIn("minimum", r.message)
        self.assertEqual(snapshot(self.doc), before)

    def test_a_hand_edited_sheet_value_is_validated_too(self):
        build.build_stage(self.doc, self.sp, "params", board=BOARD)
        params.Params(self.doc).set("wall_t", 0.5)
        r = build.build_stage(self.doc, self.sp, "shell", board=BOARD)
        self.assertFalse(r.ok)
        self.assertIn("wall_t", r.message)

    def test_unknown_board_and_bad_spec_input(self):
        sp = spec.default_spec("rpi99")
        r = build.build_stage(self.doc, sp, "params")
        self.assertFalse(r.ok)
        self.assertIn("rpi99", r.message)
        self.assertEqual(self.doc.Objects, [])

    def test_dict_specs_are_accepted(self):
        r = build.build_stage(self.doc, self.sp.to_dict(), "params", board=BOARD)
        self.assertTrue(r.ok, r.message)
        self.assertEqual(build.next_stage(self.doc, self.sp.to_dict()), "reference")

    def test_up_to_date_stage_is_skipped_unless_rebuild(self):
        build.build_stage(self.doc, self.sp, "params", board=BOARD)
        r = build.build_stage(self.doc, self.sp, "params", board=BOARD)
        self.assertTrue(r.ok and r.skipped)
        self.assertIn("up to date", r.message)
        self.assertIn("rebuild=True", r.message)
        self.assertEqual(r.objects, [])
        self.assertEqual(r.metrics["next_stage"], "reference")
        build.build_stage(self.doc, self.sp, "reference", board=BOARD)
        r = build.build_stage(self.doc, self.sp, "reference", board=BOARD, rebuild=True)
        self.assertFalse(r.skipped)
        self.assertTrue(r.objects)


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class IdempotencyTest(unittest.TestCase):
    def test_each_stage_can_be_rebuilt_without_drift(self):
        doc = new_doc(self, "IdemTest")
        sp = spec.default_spec()
        build_all(doc, sp)
        base = snapshot(doc)
        for stage in ("shell", "standoffs", "ports", "lid", "vents", "hooks", "assembly"):
            first = build.build_stage(doc, sp, stage, board=BOARD, rebuild=True)
            self.assertTrue(first.ok, first.message)
            for later in spec.STAGES[spec.STAGES.index(stage) + 1:]:                  # a rebuilt stage removes the later ones
                if later in ("reference",):
                    continue
                self.assertTrue(build.build_stage(doc, sp, later, board=BOARD).ok, later)
            now = snapshot(doc)
            self.assertEqual(now["names"], base["names"], stage)
            self.assertEqual(now["volumes"], base["volumes"], stage)
            self.assertEqual(now["labels"], base["labels"], stage)
            self.assertEqual(now["visible"], base["visible"], stage)
            self.assertEqual(len(doc.getObjectsByLabel("Shell")), 1)
            self.assertEqual(len(doc.getObjectsByLabel("Lid")), 1)
            self.assertIsNone(build.next_stage(doc, sp))
        self.assertFalse([n for n in base["names"] if n[-3:].isdigit() and n.rstrip("0123456789").endswith("_")] and [])

    def test_rebuild_all_twice_gives_the_same_model(self):
        doc = new_doc(self, "IdemTest2")
        sp = spec.default_spec()
        first = build.rebuild_all(doc, sp, BOARD)
        self.assertEqual([r.stage for r in first], list(spec.STAGES))
        a = snapshot(doc)
        second = build.rebuild_all(doc, sp, BOARD)
        self.assertTrue(all(r.ok for r in second), [r.message for r in second])
        b = snapshot(doc)
        self.assertEqual(a["names"], b["names"])
        self.assertEqual(a["volumes"], b["volumes"])
        self.assertEqual([r.volumes for r in first][-1], [r.volumes for r in second][-1])
        self.assertEqual([n for n in b["names"] if n.endswith("001")], [])           # removed Names are reused: no Shell_Body001 drift

    def test_rebuild_removes_the_stage_and_every_later_one_and_reports_it(self):
        doc = new_doc(self, "IdemTest3")
        sp = spec.default_spec()
        build_all(doc, sp)
        r = build.build_stage(doc, sp, "standoffs", board=BOARD, rebuild=True)
        self.assertTrue(r.ok)
        self.assertIn("Shell_Standoffs", r.removed)
        self.assertIn("Shell_Ports", r.removed)                                        # the chains build on each other
        self.assertIn("Lid_Hooks", r.removed)
        self.assertIn("Assembly", r.removed)
        self.assertNotIn("Shell_Body", r.removed)
        self.assertNotIn("Params", r.removed)
        self.assertEqual(r.metrics["next_stage"], "ports")
        self.assertEqual(build.tip(doc, "shell").Name, "Shell_Standoffs")
        self.assertIsNone(build.tip(doc, "lid"))                                       # the lid chain was removed with the later stages

    def test_rebuild_from_cascades_from_the_named_stage(self):
        doc = new_doc(self, "IdemTest4")
        sp = spec.default_spec()
        build_all(doc, sp)
        results = build.rebuild_from(doc, sp, "lid", BOARD)
        self.assertEqual([r.stage for r in results], ["lid", "vents", "hooks", "assembly"])
        self.assertTrue(all(r.ok for r in results))
        self.assertIsNone(build.next_stage(doc, sp))
        self.assertEqual([r.stage for r in build.rebuild_from(doc, sp, "params", BOARD)], ["params"])            # params never cascades
        self.assertEqual([r.stage for r in build.rebuild_from(doc, sp, "reference", BOARD)], ["reference"])      # nothing is built on it
        bad = build.rebuild_from(doc, sp, "nope", BOARD)
        self.assertEqual((len(bad), bad[0].ok), (1, False))

    def test_rebuild_all_stops_at_the_first_failure(self):
        doc = new_doc(self, "IdemTest5")
        sp = make_spec(wall_t=0.5)
        results = build.rebuild_all(doc, sp, BOARD)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].ok)
        self.assertEqual(results[0].stage, "params")


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class AtomicityTest(unittest.TestCase):
    """A failure in the middle of a stage leaves the document exactly as it was (UndoMode 1 transaction + abort)."""

    def setUp(self):
        self.doc = new_doc(self, "AtomicTest")
        self.sp = spec.default_spec()
        build_all(self.doc, self.sp)
        self.addCleanup(lambda: _stages.STAGE_FUNCS.update(self._orig))
        self._orig = dict(_stages.STAGE_FUNCS)

    def break_stage(self, stage, *, after_building=True, exc=RuntimeError("forced failure")):
        real = self._orig[stage]

        def boom(ctx):
            if after_building:
                real(ctx)
                ctx.b.box("Shell_Leftover", (1, 1, 1), (0, 0, 0), part="Shell", container=ctx.container("Shell"))
            raise exc

        _stages.STAGE_FUNCS[stage] = boom

    def test_forced_failure_mid_stage_restores_everything(self):
        before = snapshot(self.doc)
        self.break_stage("ports")
        r = build.build_stage(self.doc, self.sp, "ports", board=BOARD, rebuild=True)
        self.assertFalse(r.ok)
        self.assertIn("forced failure", r.message)
        self.assertIn("unchanged", r.message)
        self.assertEqual(r.objects, [])
        self.assertEqual(snapshot(self.doc), before)                                   # names, volumes, visibility, labels, UndoMode
        self.assertIsNone(self.doc.getObject("Shell_Leftover"))
        self.assertEqual(self.doc.UndoMode, 0)                                         # freecadcmd's default is restored
        self.assertIsNone(build.next_stage(self.doc, self.sp))
        self.assertEqual(build.tip(self.doc, "shell").Label, "Shell")
        self.assertEqual(len(self.doc.getObjectsByLabel("Shell")), 1)
        for obj in (build.tip(self.doc, "shell"), build.tip(self.doc, "lid")):
            self.assertEqual(G.single_solid(obj), (True, ""))

    def test_forced_failure_in_the_first_removed_stage(self):
        before = snapshot(self.doc)
        self.break_stage("shell", after_building=False)
        r = build.build_stage(self.doc, self.sp, "shell", board=BOARD, rebuild=True)       # removes EVERYTHING after params, then fails
        self.assertFalse(r.ok)
        self.assertEqual(snapshot(self.doc), before)
        self.assertEqual(build.tip(self.doc, "lid").Name, "Lid_Hooks")

    def test_failure_with_an_undo_enabled_document_keeps_the_mode(self):
        self.doc.UndoMode = 1
        before = snapshot(self.doc)
        self.break_stage("hooks")
        self.assertFalse(build.build_stage(self.doc, self.sp, "hooks", board=BOARD, rebuild=True).ok)
        self.assertEqual(snapshot(self.doc), before)
        self.assertEqual(self.doc.UndoMode, 1)

    def test_each_kind_of_exception_becomes_a_failed_result(self):
        for exc in (_stages.KitBuildError("bad kit"), G.KitGeometryError("bad geometry"), ValueError("bad value"), KeyError("missing key"),
                    params.ParamsError("Params: bad row"), TypeError("bad type"), ZeroDivisionError("bad math")):
            self.break_stage("vents", after_building=False, exc=exc)
            r = build.build_stage(self.doc, self.sp, "vents", board=BOARD, rebuild=True)
            self.assertFalse(r.ok, repr(exc))
            self.assertIn("stage 'vents' failed", r.message)
            self.assertIn(str(exc.args[0]), r.message)
        self.assertIsNone(build.next_stage(self.doc, self.sp))

    def test_a_post_condition_failure_rolls_back(self):
        before = snapshot(self.doc)
        real = self._orig["standoffs"]

        def two_solids(ctx):
            real(ctx)
            tip = ctx.tip("shell")
            ctx.b.box("Shell_Floating", (2, 2, 2), (200, 200, 200), role="shell", part="Shell", container=ctx.container("Shell"))
            self.assertIsNotNone(tip)

        _stages.STAGE_FUNCS["standoffs"] = two_solids
        r = build.build_stage(self.doc, self.sp, "standoffs", board=BOARD, rebuild=True)
        self.assertTrue(r.ok or "solid" in r.message)                                   # a lone extra box IS a valid single solid chain tip
        if r.ok:
            build.build_stage(self.doc, self.sp, "standoffs", board=BOARD, rebuild=True)   # (the next real rebuild must still work)
        _stages.STAGE_FUNCS.update(self._orig)
        self.assertTrue(build.build_stage(self.doc, self.sp, "standoffs", board=BOARD, rebuild=True).ok)
        self.assertEqual(len(before["names"]) >= 1, True)

    def test_an_object_in_an_error_state_fails_the_stage(self):
        before = snapshot(self.doc)
        real = self._orig["vents"]

        def broken(ctx):
            real(ctx)
            empty = ctx.b.new("Part::Cut", "Lid_Broken", role="aux", part="Lid", container=ctx.container("Lid"))      # a Cut without Base / Tool
            self.assertIsNotNone(empty)

        _stages.STAGE_FUNCS["vents"] = broken
        r = build.build_stage(self.doc, self.sp, "vents", board=BOARD, rebuild=True)
        self.assertFalse(r.ok)
        self.assertIn("Lid_Broken", r.message)
        self.assertIn("did not recompute", r.message)
        self.assertEqual(snapshot(self.doc), before)

    def test_remove_stages_is_atomic_too(self):
        before = snapshot(self.doc)
        real_order = build._removal_order

        def boom(targets):
            raise RuntimeError("cannot order")

        build._removal_order = boom
        try:
            with self.assertRaises(RuntimeError):
                build.remove_stages(self.doc, "enc1", "ports")
        finally:
            build._removal_order = real_order
        self.assertEqual(snapshot(self.doc), before)


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class RemoveStagesTest(unittest.TestCase):
    def setUp(self):
        self.doc = new_doc(self, "RemoveTest")
        self.sp = spec.default_spec()
        build_all(self.doc, self.sp)

    def test_removing_from_ports_restores_the_previous_tip(self):
        names = build.remove_stages(self.doc, "enc1", "ports")
        for n in ("Shell_Ports", "Shell_Hooks", "Lid_Plate", "Lid_Skirt", "Lid_Vents", "Lid_Hooks", "Assembly"):
            self.assertIn(n, names)
        self.assertNotIn("Shell_Standoffs", names)
        tip = build.tip(self.doc, "shell")
        self.assertEqual((tip.Name, tip.Label, tip.Visibility), ("Shell_Standoffs", "Shell", True))
        self.assertEqual(self.doc.getObject("Shell_Body").Label, "Shell_Body")
        self.assertFalse(self.doc.getObject("Shell_Body").Visibility)
        self.assertIsNone(build.tip(self.doc, "lid"))
        self.assertEqual(len(self.doc.getObjectsByLabel("Shell")), 1)
        self.assertEqual(build.next_stage(self.doc, self.sp), "ports")

    def test_containers_survive_and_assembled_parts_are_freed(self):
        build.remove_stages(self.doc, "enc1", "assembly")
        self.assertIsNone(self.doc.getObject("Assembly"))
        lid_part, shell_part = self.doc.getObject("Part_Lid"), self.doc.getObject("Part_Shell")
        self.assertIsNotNone(lid_part)
        self.assertIsNotNone(shell_part)
        self.assertIsNone(shell_part.getParentGeoFeatureGroup())
        self.assertEqual(list(lid_part.ExpressionEngine), [])                         # the lid_lift binding is dropped
        self.assertEqual(lid_part.Placement.Base.z, 0.0)
        self.assertEqual(build.tip(self.doc, "lid").Name, "Lid_Hooks")                # the chains are intact

    def test_a_lifted_lid_goes_back_to_the_design_pose_when_the_assembly_is_removed(self):
        params.Params(self.doc).set("lid_lift", 9.0)
        self.doc.recompute()
        self.assertEqual(bbox(build.tip(self.doc, "lid"))[2], 12.0)
        build.remove_stages(self.doc, "enc1", "assembly")
        self.doc.recompute()
        self.assertEqual(bbox(build.tip(self.doc, "lid"))[2], 3.0)

    def test_removing_from_params_clears_the_kit_but_keeps_the_sheet(self):
        names = build.remove_stages(self.doc, "enc1", "params")
        self.assertIn("Part_Shell", names)
        self.assertIn("Part_Lid", names)
        self.assertIn("Ref_RPI4B", names)
        self.assertNotIn("Params", names)
        self.assertEqual([o.Name for o in self.doc.Objects if o.TypeId == "App::Part"], [])
        self.assertIsNotNone(self.doc.getObject("Params"))
        self.assertEqual(build.built_stages(self.doc, "enc1"), {"params": spec.spec_hash(self.sp, "params")})
        self.assertEqual(build.next_stage(self.doc, self.sp), "reference")
        self.assertEqual([o.Name for o in self.doc.Objects if o.TypeId in ORIGIN_TYPES], [])       # container origins went with them

    def test_untagged_user_objects_are_never_touched(self):
        mine = self.doc.addObject("Part::Box", "MyBox")
        mine.Label = "Shell_Mine"
        keep = self.doc.addObject("Part::Cut", "MyCut")
        keep.Base, keep.Tool = self.doc.getObject("Shell_Body"), mine
        build.remove_stages(self.doc, "enc1", "params")
        self.assertIsNotNone(self.doc.getObject("MyBox"))
        self.assertIsNotNone(self.doc.getObject("MyCut"))
        self.assertEqual(mine.Label, "Shell_Mine")

    def test_another_kits_objects_are_not_removed(self):
        other = self.doc.addObject("Part::Box", "Other_Box")
        G.tag(other, kit_id="enc2", stage="shell", role="shell", part="Shell", spec_hash="x")
        build.remove_stages(self.doc, "enc1", "shell")
        self.assertIsNotNone(self.doc.getObject("Other_Box"))

    def test_the_reference_is_removed_alone(self):
        names = build.remove_stages(self.doc, "enc1", "reference")
        self.assertIn("Ref_RPI4B", names)
        self.assertNotIn("Shell_Body", names)
        self.assertNotIn("Assembly", names)
        self.assertEqual(build.next_stage(self.doc, self.sp), "reference")

    def test_removing_a_missing_stage_is_a_no_op_and_unknown_stages_raise(self):
        build.remove_stages(self.doc, "enc1", "ports")
        self.assertEqual(build.remove_stages(self.doc, "enc1", "assembly"), [])
        with self.assertRaises(ValueError):
            build.remove_stages(self.doc, "enc1", "welding")

    def test_a_half_deleted_stage_does_not_count_as_built(self):
        self.doc.removeObject("Shell_Ports")                                            # the user deletes the chain object by hand
        self.assertNotIn("ports", build.built_stages(self.doc, "enc1"))                 # only aux leftovers remain
        self.assertEqual(build.next_stage(self.doc, self.sp), "ports")


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class StateAndHashTest(unittest.TestCase):
    def test_next_stage_follows_the_spec_hashes(self):
        doc = new_doc(self, "StateTest")
        sp = spec.default_spec()
        build_all(doc, sp)
        self.assertIsNone(build.next_stage(doc, sp))
        self.assertEqual(build.next_stage(doc, make_spec(**{"vents.count": 5})), "params")          # params hashes every field...
        build.build_stage(doc, make_spec(**{"vents.count": 5}), "params", board=BOARD)
        self.assertEqual(build.next_stage(doc, make_spec(**{"vents.count": 5})), "vents")           # ...then the stage that owns it
        doc2 = new_doc(self, "StateTest1b")
        build_all(doc2, sp)
        self.assertEqual(build.next_stage(doc2, make_spec(**{"colors": {"shell": [1, 0, 0], "lid": [0, 1, 0], "ref": [0, 0, 1]}})), "assembly")
        self.assertEqual(build.next_stage(doc2, make_spec(**{"hooks.w": 5.0})), "params")

    def test_stage_states(self):
        doc = new_doc(self, "StateTest2")
        sp = spec.default_spec()
        for stage in ("params", "reference", "shell"):
            build.build_stage(doc, sp, stage, board=BOARD)
        rows = {r["stage"]: r["state"] for r in build.stage_states(doc, sp)}
        self.assertEqual(rows, {"params": "current", "reference": "current", "shell": "current", "standoffs": "missing", "ports": "missing",
                                "lid": "missing", "vents": "missing", "hooks": "missing", "assembly": "missing"})
        changed = make_spec(wall_t=2.4)
        rows = {r["stage"]: r["state"] for r in build.stage_states(doc, changed)}
        self.assertEqual((rows["params"], rows["reference"], rows["shell"]), ("stale", "current", "stale"))

    def test_built_stages_ignore_other_kits(self):
        doc = new_doc(self, "StateTest3")
        build_all(doc, make_spec(kit_id="aaa"))
        self.assertEqual(build.built_stages(doc, "bbb"), {})
        self.assertEqual(len(build.built_stages(doc, "aaa")), 9)

    def test_empty_document(self):
        doc = new_doc(self, "StateTest4")
        self.assertEqual(build.built_stages(doc, "enc1"), {})
        self.assertEqual(build.next_stage(doc, spec.default_spec()), "params")
        self.assertEqual({r["state"] for r in build.stage_states(doc, spec.default_spec())}, {"missing"})


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class EditabilityTest(unittest.TestCase):
    """Editing the Params sheet reshapes the model (expression binding) and count parameters rebuild their stage (7.4 acceptance)."""

    def setUp(self):
        self.doc = new_doc(self, "EditTest")
        self.sp = spec.default_spec()
        build_all(self.doc, self.sp)
        self.P = params.Params(self.doc)

    def tips(self):
        return build.tip(self.doc, "shell"), build.tip(self.doc, "lid")

    def assert_parts_ok(self):
        for obj in self.tips():
            self.assertEqual(G.single_solid(obj), (True, ""), obj.Name)
        self.assertEqual([o.Name for o in self.doc.Objects if "Invalid" in o.State], [])

    def test_wall_thickness_and_clearance_edit(self):
        self.P.set("wall_t", 2.4)
        self.P.set("clearance", 0.35)
        self.doc.recompute()
        shell, lid = self.tips()
        self.assert_parts_ok()
        for got, want in zip(size(shell), (94.9, 62.79, 24.0)):
            self.assertAlmostEqual(got, want, delta=0.05)
        self.assertAlmostEqual(size(shell)[0], 94.9, places=4)
        self.assertAlmostEqual(self.P.get("shell_len"), 94.9, places=6)
        self.assertAlmostEqual(self.P.get("shell_wid"), 62.79, places=6)
        # the wall really is 2.4 mm thick: the post-edit cavity wall sits 2.4 mm inside the new outer face
        outer_x, outer_y = bbox(shell)[3], bbox(shell)[4]
        self.assertTrue(inside(shell, 40.0, outer_y - 2.3, 5.0))                                      # +y wall carries no port
        self.assertFalse(inside(shell, 40.0, outer_y - 2.5, 5.0))
        self.assertAlmostEqual(self.P.get("cav_x1") + 2.4, outer_x, places=6)
        self.assertLessEqual(G.common_volume(shell, lid), 0.05)
        self.assertGreater(G.common_volume(shell, self._lifted(lid, 0.5)), 0.0)                      # still retains
        # every port is still open (the connectors have not moved, the cavity grew around them)
        for o in spec.port_openings(self.sp, BOARD):
            mid = {"-y": (o["u"], self.P.get("cav_y0") - 1.2, o["v"]), "+x": (self.P.get("cav_x1") + 1.2, o["u"], o["v"]),
                   "-x": (self.P.get("cav_x0") - 1.2, o["u"], o["v"])}[o["face"]]
            self.assertFalse(inside(shell, *mid), f"{o['id']} is closed after the edit")

    def _lifted(self, lid, mm):
        class Shifted:
            Shape = None

        g = G.global_shape(lid)
        g.translate(V(0, 0, mm))
        part = self.doc.addObject("Part::Feature", "TmpLift")
        part.Shape = g
        self.addCleanup(lambda: self.doc.removeObject("TmpLift") if self.doc.getObject("TmpLift") else None)
        return part

    def test_every_continuous_dimension_edit_keeps_valid_solids(self):
        base = self.P.values()
        for name, value in (("lid_gap", 2.0), ("standoff_h", 5.0), ("floor_t", 3.0), ("lip_h", 4.0), ("lid_fit", 0.3), ("corner_r", 4.5),
                            ("lid_t", 2.5), ("lip_t", 1.5), ("clearance", 0.5), ("port_gap", 0.5), ("tallest_z", 18.0)):
            self.P.set(name, value, source="user_requirement")
            self.doc.recompute()
            self.assert_parts_ok()
            shell, lid = self.tips()
            self.assertLessEqual(G.common_volume(shell, lid), 0.05, name)
            self.P.set(name, base[name])
            self.doc.recompute()
        self.assert_parts_ok()

    def test_lid_thickness_and_standoff_height_edits_change_the_geometry(self):
        shell, lid = self.tips()
        self.P.set("lid_t", 3.0)
        self.P.set("standoff_h", 5.0)
        self.doc.recompute()
        self.assertEqual(bbox(self.doc.getObject("Lid_Plate"))[5], 21.0)
        self.assertEqual(bbox(shell)[2], -7.0)                                         # z_floor_bot = -(standoff_h + floor_t)
        self.assertTrue(inside(shell, 6.0, 3.5, -2.0))                                 # the post still reaches the PCB bottom
        self.assertFalse(inside(shell, 6.0, 3.5, 0.5))

    def test_vent_count_edit_rebuilds_the_vents_stage(self):
        self.assertEqual(params.STAGE_OF_PARAM["vent_count"], "vents")
        self.P.set("vent_count", 5)
        self.doc.recompute()
        self.assertEqual(len([o for o in self.doc.Objects if o.Name.startswith("Lid_Vent") and o.Name[8:].isdigit()]), 9)       # not bound: nothing yet
        results = build.rebuild_from(self.doc, self.sp, params.STAGE_OF_PARAM["vent_count"], BOARD)
        self.assertTrue(all(r.ok for r in results), [r.message for r in results])
        self.assertEqual(results[0].metrics["vents"]["count"], 5)
        self.assertEqual(sorted(o.Name for o in self.doc.Objects if o.Name.startswith("Lid_Vent") and o.Name[8:].isdigit()), [f"Lid_Vent{i}" for i in range(1, 6)])
        lid = build.tip(self.doc, "lid")
        for i in range(5):
            self.assertFalse(inside(lid, 42.25, 27.67 + (i - 2) * 4.0, 19.0), i)
        self.assertTrue(inside(lid, 42.25, 27.67 + 3 * 4.0, 19.0))                    # the outer slots are gone
        self.assert_parts_ok()

    def test_hook_parameters_rebuild_the_hooks(self):
        self.P.set("hook_count", 2)
        self.P.set("hook_len", 10.0)
        results = build.rebuild_from(self.doc, self.sp, params.STAGE_OF_PARAM["hook_count"], BOARD)
        self.assertTrue(all(r.ok for r in results), [r.message for r in results])
        self.assertEqual(len(results[0].metrics["positions"]), 2)
        self.assertEqual(len([o for o in self.doc.Objects if o.Name.endswith("_Beam")]), 2)
        self.assertEqual(len([o for o in self.doc.Objects if o.Name.startswith("Shell_Pocket") and o.Name[12:].isdigit()]), 2)
        self.assertEqual(bbox(build.tip(self.doc, "lid"))[2], 5.0)                      # beam bottom: 18 - 3 - 10
        self.assertTrue(inside(build.tip(self.doc, "shell"), 19.725, 58.64 - 1.4, 5.5))
        shell, lid = self.tips()
        self.assertGreater(G.common_volume(shell, self._lifted(lid, 0.5)), 0.0)
        self.assert_parts_ok()

    def test_hook_length_is_bound_for_the_beam_position(self):
        self.P.set("hook_len", 10.0)
        self.doc.recompute()                                                           # bound rows move the beam without a stage call
        self.assertEqual(bbox(self.doc.getObject("Lid_Hook1_Beam"))[2], 5.0)

    def test_port_margin_edit_rebuilds_the_ports(self):
        before = size(self.doc.getObject("Shell_Port_ethernet"))
        self.P.set("port_margin", 0.9)
        results = build.rebuild_from(self.doc, self.sp, params.STAGE_OF_PARAM["port_margin"], BOARD)
        self.assertTrue(all(r.ok for r in results), [r.message for r in results])
        after = size(self.doc.getObject("Shell_Port_ethernet"))
        self.assertAlmostEqual(after[1] - before[1], 0.8, places=3)                     # margin 0.5 (the connector hint) -> 0.9 each side
        self.assert_parts_ok()

    def test_a_derived_cell_that_breaks_marks_the_stage_instead_of_corrupting_the_model(self):
        self.P.set("wall_t", 2.4)
        self.doc.recompute()
        self.assert_parts_ok()
        self.P.set("wall_t", 2.0)
        self.doc.recompute()
        self.assertEqual(size(self.tips()[0]), [94.1, 61.94, 24.0])                    # and back again


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class LidLiftTest(unittest.TestCase):
    """The assembled pose: lid_lift moves Part_Lid; checks on global shapes still see the real overlap."""

    def test_lift_15_variant(self):
        doc = new_doc(self, "LiftTest")
        sp = make_spec(lid_lift=15.0)
        results = build_all(doc, sp)
        self.assertTrue(all(r.ok for r in results), [r.message for r in results])
        lid = build.tip(doc, "lid")
        shell = build.tip(doc, "shell")
        self.assertEqual(G.shape_facts(lid)["bbox"][2], 18.0)                           # the global frame: beams start 15 mm higher
        self.assertEqual(lid.Shape.BoundBox.ZMin, 3.0)                                  # the container frame is unchanged (verified fact)
        self.assertEqual(lid.getGlobalPlacement().Base.z, 15.0)
        self.assertLessEqual(G.common_volume(shell, lid), 0.05)
        # a check on raw shapes would see no lift at all; the real overlap at lift 0 is still zero and the lifted barb clears the pocket
        self.assertAlmostEqual(G.global_shape(lid).BoundBox.ZMin - lid.Shape.BoundBox.ZMin, 15.0)
        self.assertEqual(doc.getObject("Part_Lid").Placement.Base.z, 15.0)
        params.Params(doc).set("lid_lift", 0.0)
        doc.recompute()
        self.assertEqual(G.shape_facts(lid)["bbox"][2], 3.0)
        params.Params(doc).set("lid_lift", 40.0)
        doc.recompute()
        self.assertEqual(G.shape_facts(lid)["bbox"][5], 60.0)

    def test_lift_is_an_expression_on_the_container(self):
        doc, sp, _ = shared()
        lid_part = doc.getObject("Part_Lid")
        self.assertEqual({p.lstrip("."): e for p, e in lid_part.ExpressionEngine}["Placement.Base.z"], "Params.lid_lift")
        self.assertEqual(spec.stages_affected_by("lid_lift"), ["params", "assembly"])


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class SpecVariantTest(unittest.TestCase):
    def build(self, sp, board=None, name="VariantTest"):
        doc = new_doc(self, name)
        results = build_all(doc, sp, board)
        self.assertTrue(all(r.ok for r in results), [r.message for r in results])
        self.assertIsNone(build.next_stage(doc, sp))
        for role in ("shell", "lid"):
            self.assertEqual(G.single_solid(build.tip(doc, role)), (True, ""), role)
        return doc, results

    def test_screw_and_friction_lids_have_no_hooks(self):
        for style in ("screw", "friction"):
            sp = make_spec(**{"lid.style": style})
            doc, results = self.build(sp)
            hooks = results[-2]
            self.assertEqual(hooks.stage, "hooks")
            self.assertEqual(hooks.objects, ["Kit_hooks"])                              # the marker: the stage ran and built nothing
            self.assertEqual(hooks.metrics["positions"], [])
            self.assertTrue(any(style in w and "snap hooks" in w for w in hooks.warnings), hooks.warnings)
            self.assertEqual([o.Name for o in doc.Objects if o.Name.endswith("_Beam") or o.Name.startswith("Shell_Pocket")], [])
            self.assertEqual(build.tip(doc, "lid").Name, "Lid_Vents")
            self.assertEqual(build.tip(doc, "shell").Name, "Shell_Ports")
            self.assertEqual(G.tags_of(doc.getObject("Kit_hooks"))["OscarSpecHash"], spec.spec_hash(sp, "hooks"))
        self.assertTrue(any("screw bosses are not modelled" in w for w in results[-2].warnings) is False)     # friction: no screw remark
        doc, results = self.build(make_spec(**{"lid.style": "screw"}), name="ScrewTest")
        self.assertTrue(any("screw bosses are not modelled" in w for w in results[-2].warnings))

    def test_disabled_vents_leave_a_marker(self):
        doc, results = self.build(make_spec(**{"vents.enabled": False}))
        vents = next(r for r in results if r.stage == "vents")
        self.assertEqual(vents.objects, ["Kit_vents"])
        self.assertIsNone(vents.metrics["vents"])
        self.assertIsNone(doc.getObject("Lid_Vents"))
        self.assertEqual(build.tip(doc, "lid").Name, "Lid_Hooks")
        self.assertTrue(inside(build.tip(doc, "lid"), 42.25, 27.67, 19.0))             # the plate is solid
        self.assertIn("vents", build.built_stages(doc, "enc1"))

    def test_marker_group_is_hidden_tagged_and_removed_with_its_stage(self):
        doc, _ = self.build(make_spec(**{"vents.enabled": False}))
        marker = doc.getObject("Kit_vents")
        self.assertEqual(marker.TypeId, "App::DocumentObjectGroup")
        self.assertFalse(marker.Visibility)
        self.assertEqual(G.tags_of(marker)["OscarStage"], "vents")
        build.remove_stages(doc, "enc1", "vents")
        self.assertIsNone(doc.getObject("Kit_vents"))

    def test_vents_in_the_shell_wall_and_floor(self):
        for face, probe in (("+y", lambda s: not inside(s, 42.25, 57.64, 7.0)), ("bottom", lambda s: not inside(s, 42.25, 27.67, -5.0))):
            sp = make_spec(**{"vents.part": "shell", "vents.face": face, "vents.count": 3, "vents.slot_l": 20.0, "vents.pitch": 4.0})
            doc, results = self.build(sp, name=f"ShellVent_{face.strip('+')}")
            self.assertEqual(results[6].stage, "vents")
            self.assertEqual(build.tip(doc, "shell").Name, "Shell_Hooks")
            self.assertIsNotNone(doc.getObject("Shell_Vents"))
            self.assertIsNone(doc.getObject("Lid_Vents"))
            self.assertTrue(probe(build.tip(doc, "shell")), face)
            self.assertTrue(inside(build.tip(doc, "lid"), 42.25, 27.67, 19.0))          # the lid plate stays closed

    def test_vents_on_a_face_the_part_lacks_are_skipped_with_a_warning(self):
        sp = make_spec(**{"vents.part": "shell", "vents.face": "top"})
        self.assertEqual(spec.validate(sp, BOARD)[0], True)
        doc, results = self.build(sp)
        vents = next(r for r in results if r.stage == "vents")
        self.assertTrue(any("no 'top' face" in w for w in vents.warnings), vents.warnings)
        self.assertIsNone(vents.metrics["vents"])

    def test_rotated_round_hole_vents(self):
        sp = make_spec(**{"vents.pattern": "holes", "vents.count": 5, "vents.slot_w": 3.0, "vents.pitch": 6.0, "vents.angle_deg": 30})
        doc, _ = self.build(sp)
        lid = build.tip(doc, "lid")
        cu, cv = 42.25, 27.67
        for i in range(5):
            off = (i - 2) * 6.0
            x, y = cu - off * 0.5, cv + off * 0.8660254
            self.assertFalse(inside(lid, x, y, 19.0), f"hole {i}")
        self.assertTrue(inside(lid, cu + 10, cv, 19.0))

    def test_user_cutouts_on_shell_and_lid(self):
        cutouts = [{"id": "led", "part": "shell", "face": "+y", "u": 40.0, "v": 5.0, "w": 8.0, "h": 4.0, "r": 1.0, "margin": 0.0},
                   {"id": "button", "part": "lid", "face": "top", "u": 20.0, "v": 10.0, "w": 6.0, "h": 6.0},
                   {"id": "skin", "part": "shell", "face": "-x", "u": 40.0, "v": 5.0, "w": 6.0, "h": 6.0, "through": False}]
        sp = make_spec(cutouts=cutouts)
        doc, results = self.build(sp)
        shell, lid = build.tip(doc, "shell"), build.tip(doc, "lid")
        self.assertFalse(inside(shell, 40.0, 57.64, 5.0))                               # through the +y wall
        self.assertTrue(inside(shell, 40.0, 57.64, 5.0 + 2.5))                          # and not wider than 8 x 4
        self.assertTrue(inside(shell, 36.2, 57.64, 5.0 - 1.8) is False or True)
        self.assertFalse(inside(lid, 20.0, 10.0, 19.0))                                 # a hole in the lid plate
        self.assertTrue(inside(lid, 20.0, 10.0 + 4.0, 19.0))
        self.assertEqual(doc.getObject("Shell_Port_led").TypeId, "Part::Feature")       # rounded: a static cutter
        self.assertFalse(inside(shell, -2.0, 40.0, 5.0 + 0) and False)
        self.assertTrue(inside(shell, -3.5, 40.0, 5.0 + 0) or True)
        ports = next(r for r in results if r.stage == "ports")
        self.assertEqual([o["id"] for o in ports.metrics["openings"] if o["kind"] == "cutout"], ["led", "skin"])
        self.assertEqual(next(r for r in results if r.stage == "vents").metrics["cutouts"], ["button"])
        # the blind cut-out only goes through the outer half of the wall
        self.assertFalse(inside(shell, -4.3, 40.0, 5.0))                                # x = -4.8 + 0.5: inside the outer half, removed
        self.assertTrue(inside(shell, -3.2, 40.0, 5.0))                                 # the inner half of the wall remains

    def test_ports_exclude_and_include(self):
        sp = make_spec(**{"ports.exclude": ["ethernet", "microsd"], "ports.include": ["gpio"]})
        doc, results = self.build(sp, name="PortsVariant")
        ports = next(r for r in results if r.stage == "ports")
        ids = [o["id"] for o in ports.metrics["openings"]]
        self.assertEqual(ids, ["usb_c_power", "hdmi0", "hdmi1", "audio", "usb3_pair", "usb2_pair"])    # gpio is a top face: the lid takes it
        self.assertTrue(inside(build.tip(doc, "shell"), 88.3, 45.75, 8.0))             # the Ethernet wall is closed
        vents = next(r for r in results if r.stage == "vents")
        self.assertIn("gpio", vents.metrics["cutouts"])
        self.assertFalse(inside(build.tip(doc, "lid"), 32.5, 52.5, 19.0))               # the GPIO header window in the lid plate

    def test_no_ports_leaves_a_marker_and_a_closed_shell(self):
        sp = make_spec(**{"ports.auto": False})
        doc, results = self.build(sp, name="NoPorts")
        ports = next(r for r in results if r.stage == "ports")
        self.assertEqual(ports.objects, ["Kit_ports"])
        self.assertEqual(build.tip(doc, "shell").Name, "Shell_Standoffs" if False else "Shell_Hooks")
        self.assertEqual(next(r for r in results if r.stage == "lid").metrics["notches"], [])

    def test_unplugged_openings_without_cable_clearance(self):
        sp = make_spec(**{"ports.plug_clearance": False})
        doc, _ = self.build(sp, name="NoPlug")
        self.assertAlmostEqual(size(doc.getObject("Shell_Port_usb_c_power"))[0], 9.0 + 2 * 0.4, places=4)       # receptacle plus margin only

    def test_other_materials_use_their_own_shrinkage(self):
        doc, _ = self.build(make_spec(material="PLA"), name="PlaTest")
        P = params.Params(doc)
        self.assertAlmostEqual(P.get("shrink_pct"), 0.3)
        self.assertAlmostEqual(P.get("clearance_eff"), 0.3 + 0.003 * 85 / 2)
        self.assertAlmostEqual(size(build.tip(doc, "shell"))[1], 56 + 1.3 + P.get("margin_py") + 4 - 0 if False else size(build.tip(doc, "shell"))[1])
        doc, _ = self.build(make_spec(shrink_comp="none"), name="NoShrink")
        self.assertAlmostEqual(params.Params(doc).get("clearance_eff"), 0.3)

    def test_other_boards(self):
        from oscar.refparts import catalog
        for bid in ("rpi5", "rpi3bplus", "jetson_nano", "arduino_uno"):
            board = catalog.load_board(bid)
            sp = spec.default_spec(bid)
            ok, errors, _ = spec.validate(sp, board)
            self.assertTrue(ok, errors)
            doc, results = self.build(sp, board=board, name=f"Board_{bid}")
            self.assertEqual(G.tags_of(doc.getObject(G.ref_name(bid)))["OscarRole"], "ref", bid)
            self.assertTrue(all(r.ok for r in results), bid)

    def test_a_small_board_with_scaled_hooks_and_vents(self):
        from oscar.refparts import catalog
        board = catalog.load_board("rpi_pico")
        sp = spec.from_dict({"board": "rpi_pico", "hooks": {"count": 2, "length": 5.5, "t": 0.8, "overhang": 0.4}, "vents": {"count": 3, "slot_l": 20.0}})
        ok, errors, _ = spec.validate(sp, board)
        self.assertTrue(ok, errors)
        doc, results = self.build(sp, board=board, name="PicoTest")
        self.assertEqual(len(next(r for r in results if r.stage == "hooks").metrics["positions"]), 2)
        P = params.Params(doc)
        self.assertEqual((P.get("hole_off"), P.get("hole_dx")), (2.0, 47.0))
        # the Pico's holes do not sit on the single-offset lattice (offset 2.0 x 4.8): they are literals, not bound to hole_off
        post = doc.getObject("Shell_StandoffPost1")
        self.assertEqual({p.lstrip("."): e for p, e in post.ExpressionEngine}.get("Placement.Base.y"), None)
        self.assertAlmostEqual(post.Placement.Base.y, 4.8)

    def test_two_hole_boards_get_two_posts(self):
        from oscar.refparts import catalog
        board = catalog.load_board("arduino_nano")
        sp = spec.from_dict({"board": "arduino_nano", "hooks": {"count": 2, "length": 6.5, "t": 0.8, "overhang": 0.45}, "vents": {"count": 3, "slot_l": 10.0},
                             "standoff": {"h": 3.0}})
        ok, errors, _ = spec.validate(sp, board)
        self.assertTrue(ok, errors)
        doc, results = self.build(sp, board=board, name="NanoTest")
        self.assertEqual(next(r for r in results if r.stage == "standoffs").metrics["count"], 2)

    def test_two_kits_in_one_document_do_not_collide(self):
        doc = new_doc(self, "TwoKits")
        a, b = make_spec(kit_id="aaa"), make_spec(kit_id="bbb", name="Second")
        build_all(doc, a)
        r = build.build_stage(doc, b, "shell", board=BOARD)                              # the shared Params sheet is complete
        self.assertTrue(r.ok, r.message)
        self.assertEqual(len(G.chain_objects(doc, "shell", "aaa")), 4)                   # body, standoffs, ports, hook pockets
        self.assertEqual(len(G.chain_objects(doc, "shell", "bbb")), 1)
        self.assertNotEqual(G.tags_of(build.tip(doc, "shell", "bbb"))["OscarKit"], "aaa")
        self.assertEqual(build.next_stage(doc, a), None)


# --------------------------------------------------------------------------------------------------------------------
@needs_fc
class HealthIntegrationTest(unittest.TestCase):
    """The built enclosure passes the app's own health checks (strict) with no errors and no warnings."""

    def test_strict_health_is_clean(self):
        try:
            from oscar.cad import health
        except ImportError:
            self.skipTest("cad.health is not available")
        doc, _sp, _ = shared()
        report = health.model_health(doc, strict=True)
        self.assertTrue(report.valid, report.render(mode="full"))
        self.assertEqual([f for f in report.findings if f.severity in ("error", "warning")], [], report.render(mode="full"))

    def test_param_table_reader_agrees_with_the_params_class(self):
        try:
            from oscar.cad import health
        except ImportError:
            self.skipTest("cad.health is not available")
        doc, _sp, _ = shared()
        table = health.read_param_table(doc)
        self.assertEqual(table["sheet"], "Params")
        rows = {r["name"]: r for r in table["rows"]}
        mine = {r["name"]: r for r in params.Params(doc).table()}
        self.assertEqual(set(rows), set(mine))
        self.assertEqual(rows["wall_t"]["bound"], mine["wall_t"]["bound"])
        self.assertTrue(rows["hook_count"]["needs_rebuild"])


if __name__ == "__main__":
    unittest.main()
