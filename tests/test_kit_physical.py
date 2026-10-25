"""Independent physical measurements of the generated enclosure (a review of oscar/kit).

These tests do not use the kit's own numbers: every claim is measured on the built OCC shapes (``common`` volumes, void
solids of wall slabs, bounding boxes), so a bug in how the stages place their cutters shows up even when the stage reports
success. They were written after the review found that the lid skirt notch / lid cut-out of a ``-x`` or ``-y`` face was bound
to an expression with a sign error (``... - wall_t - fit + lip_t + 1 mm`` instead of ``... - (fit + lip_t + 1 mm)``): the
expression overrides the literal placement on recompute, so the cutter sat INSIDE the cavity and the skirt was never opened.

Covered: the nominal Pi 4 (shell and lid single valid closed solids, bbox numbers, every connector passes its window through
the shell AND the lid, nothing touches the PCB or a connector, standoff posts under the board holes with a blind pilot, the
catch pockets line up with the barbs, the lift test and the seated clearance, vent web), every catalogue board built with its
default spec (no tunnel blocked, no interference), lid cut-outs on every wall face (the skirt void sits where the opening is),
and an audit that every expression the builder binds agrees with the literal value it was created with (distinct parameter
values, so a mixed-up ``Params`` name cannot hide behind equal defaults).
"""

import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.kit import spec  # noqa: E402
from oscar.refparts import catalog  # noqa: E402

try:
    import FreeCAD as App  # noqa: E402
    import Part  # noqa: E402
    from FreeCAD import Vector as V  # noqa: E402
    from oscar.kit import build  # noqa: E402
    from oscar.kit import geometry as G  # noqa: E402
except ImportError:
    App = None

needs_fc = unittest.skipIf(App is None, "FreeCAD is needed (run through freecadcmd)")

_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "enclosure")
with open(os.path.join(_FIX, "rpi4b_board.json"), encoding="utf-8") as _f:
    PI4 = json.load(_f)

TOL = 1e-3          # mm3: "no material"


def variant(board_id="rpi4b", material="ABS", **changes):
    """The default spec of ``board_id`` with dotted-path changes (``variant(**{"hooks.count": 2})``)."""
    d = spec.default_spec(board_id, material).to_dict()
    for path, value in changes.items():
        cur = d
        keys = path.split(".")
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = value
    return spec.from_dict(d)


def build_all(doc, sp, board):
    """Every stage in order; fails the calling test on the first refused stage."""
    for _ in range(len(build.STAGES) + 1):
        stage = build.next_stage(doc, sp)
        if stage is None:
            return
        result = build.build_stage(doc, sp, stage, board=board)
        if not result.ok:
            raise AssertionError(f"stage {stage!r} failed: {result.message}")


def box(x0, y0, z0, x1, y1, z1):
    return Part.makeBox(x1 - x0, y1 - y0, z1 - z0, V(x0, y0, z0))


def common(a, b) -> float:
    return float(a.common(b).Volume)


def voids(slab, solid):
    """The connected regions of ``slab`` that ``solid`` does not fill."""
    return list(slab.cut(solid).Solids)


def bounds(shape):
    b = shape.BoundBox
    return [b.XMin, b.YMin, b.ZMin, b.XMax, b.YMax, b.ZMax]


def tunnel(c: dict, b) -> "Part.Shape":
    """The connector body pushed straight out through its wall to well outside the enclosure (what a plug travels through)."""
    f = c["face"]
    if f == "-y":
        return box(b.XMin, -60, b.ZMin, b.XMax, b.YMin, b.ZMax)
    if f == "+y":
        return box(b.XMin, b.YMax, b.ZMin, b.XMax, 140, b.ZMax)
    if f == "-x":
        return box(-60, b.YMin, b.ZMin, b.XMin, b.YMax, b.ZMax)
    return box(b.XMax, b.YMin, b.ZMin, 160, b.YMax, b.ZMax)


_SHARED: dict = {}


def shared():
    """The default Pi 4 / ABS enclosure, built once for every read-only test."""
    if "doc" not in _SHARED:
        doc = App.newDocument("KitPhysical")
        sp = spec.default_spec()
        build_all(doc, sp, PI4)
        shell, lid = G.tip(doc, "shell"), G.tip(doc, "lid")
        _SHARED.update(doc=doc, spec=sp, shell=G.global_shape(shell), lid=G.global_shape(lid),
                       pcb=G.global_shape(doc.getObject("Ref_RPI4B_PCB")),
                       conns={c["id"]: (c, G.global_shape(doc.getObject("Ref_RPI4B_" + c["id"]))) for c in PI4["connectors"]})
    return _SHARED


def tearDownModule():
    if App is not None and "doc" in _SHARED:
        App.closeDocument(_SHARED["doc"].Name)
        _SHARED.clear()


@needs_fc
class NominalPi4Test(unittest.TestCase):
    """The default Pi 4 / ABS enclosure measured from the outside."""

    @classmethod
    def setUpClass(cls):
        s = shared()
        cls.shell, cls.lid, cls.pcb, cls.conns = s["shell"], s["lid"], s["pcb"], s["conns"]

    def test_shell_and_lid_are_separate_closed_valid_single_solids(self):
        for name, shape in (("shell", self.shell), ("lid", self.lid)):
            self.assertEqual((len(shape.Solids), len(shape.Shells)), (1, 1), name)
            self.assertTrue(shape.isValid() and shape.isClosed(), name)
            self.assertGreater(shape.Volume, 1000.0, name)
        self.assertLess(common(self.shell, self.lid), TOL, "the lid rests on the shell, they must not overlap")

    def test_bounding_boxes_match_the_contract_numbers(self):
        sb, lb = self.shell.BoundBox, self.lid.BoundBox
        self.assertAlmostEqual(sb.XLength, 94.1, delta=0.01)
        self.assertAlmostEqual(sb.YLength, 61.94, delta=0.01)
        self.assertAlmostEqual(sb.ZLength, 24.0, delta=0.01)
        self.assertAlmostEqual(lb.XLength, 97.0, delta=0.01)
        self.assertAlmostEqual(lb.YLength, 64.84, delta=0.01)
        # the lid is centred on the shell (the skirt is the same width on every side)
        self.assertAlmostEqual((lb.XMin + lb.XMax) / 2, (sb.XMin + sb.XMax) / 2, delta=0.01)
        self.assertAlmostEqual((lb.YMin + lb.YMax) / 2, (sb.YMin + sb.YMax) / 2, delta=0.01)

    def test_every_connector_passes_its_window_through_shell_and_lid(self):
        for cid, (c, shape) in self.conns.items():
            if not c.get("cutout"):
                continue
            t = tunnel(c, shape.BoundBox)
            self.assertLess(common(self.shell, t), TOL, f"{cid}: the shell wall blocks the connector")
            self.assertLess(common(self.lid, t), TOL, f"{cid}: the lid skirt blocks the connector")

    def test_no_object_is_left_touched_after_the_stages(self):
        """The containers were left 'Touched' (the GUI then shows a pending recompute although the model is up to date)."""
        doc = shared()["doc"]
        self.assertEqual([o.Name for o in doc.Objects if "Touched" in o.State], [])
        self.assertEqual([o.Name for o in doc.Objects if "Invalid" in o.State], [])

    def test_nothing_touches_the_pcb_or_a_connector(self):
        for name, part in (("shell", self.shell), ("lid", self.lid)):
            self.assertLess(common(part, self.pcb), TOL, f"{name} intersects the PCB")
            for cid, (_c, shape) in self.conns.items():
                self.assertLess(common(part, shape), TOL, f"{name} intersects {cid}")

    def test_wall_windows_are_plug_sized_and_centred_on_the_connectors(self):
        slab = box(-1.0, -3.3, -3.9, 86.0, -1.3, 18)                      # the straight part of the -y wall
        found = sorted(((round(b[0], 2), round(b[3], 2), round(b[2], 2), round(b[5], 2)) for b in map(bounds, voids(slab, self.shell))
                        if b[3] - b[0] > 8.0))
        expected = [(5.2, 17.2, -0.2, 6.3),        # USB-C 12.0 x 6.5 at u 11.2, v 3.05
                    (20.5, 31.5, -0.55, 6.45),     # micro-HDMI 11.0 x 7.0 at u 26.0, v 2.95
                    (34.0, 45.0, -0.55, 6.45),     # micro-HDMI at u 39.5
                    (49.0, 58.0, -0.1, 8.9)]       # audio 9.0 x 9.0 at u 53.5, v 4.4
        self.assertEqual(len(found), 4, found)
        for got, want in zip(found, expected):
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, delta=0.02, msg=f"{got} vs {want}")

    def test_the_tall_usb_windows_are_open_through_the_rim_and_notch_the_skirt(self):
        slab = box(87.3, 0.0, -3.9, 89.3, 55.0, 18)                       # the straight part of the +x wall
        found = sorted(round(b[1], 2) for b in map(bounds, voids(slab, self.shell)) if b[4] - b[1] > 8.0)
        self.assertEqual(found, [1.95, 19.95, 37.25])                      # usb2, usb3, ethernet windows (y0)
        usb = box(87.3, 1.95, 15.0, 91.0, 16.05, 18.0)                     # skirt band in front of the USB 2.0 window
        self.assertLess(common(self.lid, usb), TOL, "the skirt covers the USB window")

    def test_standoff_posts_stand_under_the_board_holes_with_a_blind_pilot(self):
        for x, y in PI4["mounting_holes"]["positions"]:
            post = common(self.shell, Part.makeCylinder(3.0, 4.0, V(x, y, -4.0)))
            self.assertAlmostEqual(post, math.pi * (3.0 ** 2 - 1.1 ** 2) * 4.0, delta=0.5, msg=f"post at {x},{y}")
            self.assertLess(common(self.shell, Part.makeCylinder(1.05, 4.0, V(x, y, -4.0))), TOL, "the pilot hole is not empty")
            self.assertGreater(common(self.shell, Part.makeCylinder(1.05, 1.0, V(x, y, -5.0))), 1.0, "the pilot must be blind (floor below it)")
            self.assertLess(common(self.shell, Part.makeCylinder(3.5, 5.0, V(x, y, 1e-4))), TOL, "the post reaches past the PCB bottom")

    def test_catch_pockets_line_up_with_the_barbs(self):
        hook_solids = G.global_shape(shared()["doc"].getObject("Lid_HookSolids"))
        self.assertLess(common(hook_solids, self.shell), TOL, "a hook touches the shell when the lid is seated")
        barbs = [G.global_shape(shared()["doc"].getObject(f"Lid_Hook{i}_Barb")).BoundBox for i in range(1, 5)]
        pockets = []
        for y0, y1 in ((-3.3, -2.0), (57.3, 58.64)):                       # outer 1.3 mm of the -y and +y walls
            pockets += [p.BoundBox for p in voids(box(-1.0, y0, -3.9, 86.0, y1, 18.0), self.shell) if p.BoundBox.XLength < 8.0]
        self.assertEqual(len(pockets), 4)
        for barb in barbs:
            hit = [p for p in pockets if p.XMin < barb.XMin and barb.XMax < p.XMax]
            self.assertEqual(len(hit), 1, "every barb sits inside exactly one pocket in x")
            p = hit[0]
            self.assertAlmostEqual((p.XMin + p.XMax) / 2, (barb.XMin + barb.XMax) / 2, delta=0.01)
            self.assertGreater(p.XMin, barb.XMin - 0.25)                    # w + 0.4: 0.2 mm each side
            self.assertGreater(barb.ZMax, p.ZMin + 1.0, "the barb must be level with the pocket")
            self.assertLess(p.ZMax, barb.ZMax + 0.6)

    def test_lifting_the_lid_catches_on_the_pockets_but_the_seated_lid_does_not_touch(self):
        def lifted(dz):
            moved = self.lid.copy()
            moved.translate(V(0, 0, dz))
            return common(moved, self.shell)

        self.assertLess(lifted(0.0), TOL)
        self.assertLess(lifted(0.2), TOL, "less than window_margin must still pass freely")
        self.assertGreater(lifted(0.5), 0.1, "the hooks must retain the lid")
        moved = self.lid.copy()
        moved.translate(V(0.2, 0, 0))                                       # sliding along the hook walls is free
        self.assertLess(common(moved, self.shell), TOL)

    def test_vent_slots_keep_the_minimum_web(self):
        plate = box(-6.25, -4.75, 18.0, 90.75, 60.09, 20.0)
        slots = sorted((b for b in map(bounds, voids(plate, self.lid)) if b[3] - b[0] > 20.0), key=lambda b: b[1])
        self.assertEqual(len(slots), 9)
        for a, b in zip(slots, slots[1:]):
            self.assertGreaterEqual(b[1] - a[4], 1.2 - 1e-6, "web between two slots below the ABS minimum")
        for s in slots:
            self.assertAlmostEqual(s[4] - s[1], 2.0, delta=1e-3)
            self.assertAlmostEqual(s[3] - s[0], 24.0, delta=1e-3)
            self.assertGreater(s[0] - (-6.25), 1.2)                         # web to the plate edge
            self.assertGreater(90.75 - s[3], 1.2)
            self.assertAlmostEqual(s[2], 18.0, delta=1e-3)                   # cut through the whole plate
            self.assertAlmostEqual(s[5], 20.0, delta=1e-3)


@needs_fc
class AllBoardsPhysicalTest(unittest.TestCase):
    """Every catalogue board with its default spec: the same measurements, nothing board specific."""

    def check_board(self, board_id):
        board = catalog.load_board(board_id)
        sp = spec.default_spec(board_id, "ABS")
        doc = App.newDocument("Kit" + board_id.replace("_", ""))
        self.addCleanup(App.closeDocument, doc.Name)
        build_all(doc, sp, board)
        shell, lid = G.global_shape(G.tip(doc, "shell")), G.global_shape(G.tip(doc, "lid"))
        for name, shape in (("shell", shell), ("lid", lid)):
            self.assertEqual(len(shape.Solids), 1, f"{board_id} {name}")
            self.assertTrue(shape.isValid(), f"{board_id} {name}")
        self.assertLess(common(shell, lid), TOL, f"{board_id}: shell and lid overlap")
        ref = G.ref_name(board_id)
        pcb = G.global_shape(doc.getObject(ref + "_PCB"))
        for name, part in (("shell", shell), ("lid", lid)):
            self.assertLess(common(part, pcb), TOL, f"{board_id}: {name} intersects the PCB")
        for c in board["connectors"]:
            shape = G.global_shape(doc.getObject(f"{ref}_{c['id']}"))
            for name, part in (("shell", shell), ("lid", lid)):
                self.assertLess(common(part, shape), TOL, f"{board_id}: {name} intersects {c['id']}")
            if c.get("cutout") and c["face"] in spec.WALL_FACES:
                t = tunnel(c, shape.BoundBox)
                self.assertLess(common(shell, t), TOL, f"{board_id}: the shell wall blocks {c['id']}")
                self.assertLess(common(lid, t), TOL, f"{board_id}: the lid skirt blocks {c['id']}")

    def test_every_catalogue_board_builds_a_clean_enclosure(self):
        ids = [b["id"] for b in catalog.list_boards()]
        self.assertGreaterEqual(len(ids), 9)
        for board_id in ids:
            with self.subTest(board=board_id):
                self.check_board(board_id)


@needs_fc
class LidCutoutOnEveryFaceTest(unittest.TestCase):
    """A cut-out of the LID on a wall face opens the skirt where it says (the -x / -y faces were mis-bound)."""

    def lid_volume(self, **changes):
        sp = variant(**changes)
        doc = App.newDocument("KitLidCut")
        self.addCleanup(App.closeDocument, doc.Name)
        build_all(doc, sp, PI4)
        return doc, G.global_shape(G.tip(doc, "lid"))

    def cutout(self, face, u, v=16.5, w=3.0, h=3.0):
        return {"cutouts": [{"id": "c1", "part": "lid", "face": face, "u": u, "v": v, "w": w, "h": h}]}

    def test_every_wall_face_removes_exactly_the_skirt_material_in_the_window(self):
        _doc, plain = self.lid_volume()
        # v 16.5 +- 1.5 = z 15..18 = the skirt band: 3 wide x 3 high x lip_t 1.2 of skirt material (u clear of the USB notches on +x)
        for face, u in (("-y", 40.0), ("+y", 40.0), ("-x", 28.0), ("+x", 18.0)):
            with self.subTest(face=face):
                _d, lid = self.lid_volume(**self.cutout(face, u))
                self.assertAlmostEqual(plain.Volume - lid.Volume, 3.0 * 3.0 * 1.2, delta=0.05)
                self.assertEqual(len(lid.Solids), 1)

    def test_the_skirt_void_sits_in_the_window_for_the_negative_faces(self):
        _d, lid = self.lid_volume(**self.cutout("-y", 40.0))
        found = voids(box(30, -4.75, 15, 50, -3.55, 18), lid)               # the -y skirt itself
        self.assertEqual(len(found), 1)
        self.assertEqual([round(b, 2) for b in bounds(found[0])[0::3]], [38.5, 41.5])
        _d, lid = self.lid_volume(**self.cutout("-x", 28.0))
        found = voids(box(-6.25, 20, 15, -5.05, 36, 18), lid)               # the -x skirt itself
        self.assertEqual(len(found), 1)
        self.assertEqual([round(b, 2) for b in bounds(found[0])[1::3]], [26.5, 29.5])

    def test_a_tall_receptacle_on_a_negative_face_notches_the_skirt(self):
        """The Pico's micro-USB sits on -x and is taller than the skirt bottom: its plug must reach it through the skirt."""
        board = catalog.load_board("rpi_pico")
        sp = spec.default_spec("rpi_pico", "ABS")
        doc = App.newDocument("KitPico")
        self.addCleanup(App.closeDocument, doc.Name)
        build_all(doc, sp, board)
        lid = G.global_shape(G.tip(doc, "lid"))
        conn = G.global_shape(doc.getObject("Ref_RPI_PICO_usb_micro"))
        self.assertLess(common(lid, tunnel(board["connectors"][0], conn.BoundBox)), TOL)


@needs_fc
class ExpressionLiteralAuditTest(unittest.TestCase):
    """Every expression the builder binds must evaluate to the literal the object was created with."""

    DISTINCT = {"wall_t": 2.3, "floor_t": 2.7, "lid_t": 2.9, "clearance": 0.37, "corner_r": 3.7, "lid.gap": 1.3, "lid.fit": 0.21,
                "lid.lip_h": 3.4, "lid.lip_t": 1.3, "standoff.h": 4.4, "standoff.d": 6.6, "standoff.pilot_d": 2.3, "hooks.w": 5.5,
                "hooks.t": 1.3, "hooks.overhang": 0.85, "hooks.length": 12.5, "hooks.barb_h": 1.1, "hooks.window_margin": 0.27,
                "vents.slot_w": 2.1, "vents.slot_l": 22.0, "vents.pitch": 4.3}

    @staticmethod
    def read(obj, prop):
        try:
            if prop.startswith("Constraints."):
                return float(obj.getDatum(prop.split(".", 1)[1]).Value)
            cur = obj
            for part in prop.split("."):
                cur = getattr(cur, part)
            return float(getattr(cur, "Value", cur))
        except Exception:    # noqa: BLE001 - a property the probe cannot read as a number is not audited
            return None

    def audit(self, sp):
        recorded = []
        original = G.Builder.bind

        def spy(builder, obj, prop, expr):
            recorded.append((obj, prop, expr, self.read(obj, prop)))
            return original(builder, obj, prop, expr)

        G.Builder.bind = spy
        doc = App.newDocument("KitAudit")
        self.addCleanup(App.closeDocument, doc.Name)
        try:
            build_all(doc, sp, catalog.load_board(sp.board))
        finally:
            G.Builder.bind = original
        self.assertGreater(len(recorded), 80)
        bad = []
        for obj, prop, expr, literal in recorded:
            if doc.getObject(obj.Name) is None or literal is None:
                continue
            now = self.read(obj, prop)
            if now is not None and abs(now - literal) > 1e-4:
                bad.append(f"{obj.Name}.{prop}: literal {literal:.4g} but '{expr}' = {now:.4g}")
        self.assertEqual(bad, [])

    def test_pi4_with_distinct_parameter_values(self):
        self.audit(variant(**self.DISTINCT))

    def test_pi4_lid_and_shell_cutouts_on_every_face(self):
        for part in ("lid", "shell"):
            for face in ("-x", "+x", "-y", "+y", "top", "bottom"):
                with self.subTest(part=part, face=face):
                    wall = face not in ("top", "bottom")
                    cut = {"id": "c1", "part": part, "face": face, "u": 30.0, "v": 16.5 if wall else 20.0, "w": 6.0, "h": 4.0}
                    d = spec.default_spec().to_dict()
                    d["cutouts"] = [cut]
                    self.audit(spec.from_dict(d))

    def test_boards_with_connectors_on_the_negative_wall(self):
        for board_id in ("rpi_pico", "arduino_uno", "arduino_nano", "esp32_devkitc"):
            with self.subTest(board=board_id):
                self.audit(spec.default_spec(board_id, "ABS"))


@needs_fc
class StructuralEditParityTest(unittest.TestCase):
    """``set_parameter`` of a STRUCTURAL value gives the same model as a fresh build with that value.

    ``lip_h`` / ``lid_gap`` / ``pcb_t`` decide which skirt notches and merged openings exist and ``lid_fit`` feeds the barb profile and
    the catch pocket height; they are expression-bound, but a plain recompute left the old structure (an Ethernet jack blocked by a
    taller skirt, a pocket 0.08 mm off its barb), and the spec hash was re-stamped as if the model were current."""

    FEATURES = ("Lid_Notch_", "Shell_Port_", "Shell_Pocket", "Lid_Hook")

    def run_tool(self, tool_name, **args):
        from oscar import tools
        from oscar.toolctx import ToolContext, use_context

        with use_context(ToolContext(state={})):
            text, _img, is_error = tools.run_tool(tool_name, args)
        self.assertFalse(is_error, text)
        return text

    def built_by_tools(self):
        from oscar import tools_kit_base

        tools_kit_base.forget_specs()
        doc = App.newDocument("KitEditParity")
        self.addCleanup(App.closeDocument, doc.Name)
        self.run_tool("enclosure_spec", action="set", spec={"name": "PiCase"}, doc_name=doc.Name)
        for _ in range(len(build.STAGES)):
            if "All 9 stages are built" in self.run_tool("build_enclosure", doc_name=doc.Name, stage="next"):
                return doc
        self.fail("the nine stages did not finish")

    def fresh(self, name, value):
        """A fresh build with ``name`` = ``value``: the design fields through the spec, a board row through the sheet before the stages."""
        sp = spec.default_spec()
        doc = App.newDocument("KitEditFresh")
        self.addCleanup(App.closeDocument, doc.Name)
        if name in spec.DESIGN_PARAMS:
            sp = variant(**{spec.DESIGN_PARAMS[name][0]: value})
        else:
            self.assertTrue(build.build_stage(doc, sp, "params", board=PI4).ok)
            from oscar.kit import params

            params.Params(doc).set(name, value, source="user_requirement")
        build_all(doc, sp, PI4)
        return doc

    @staticmethod
    def picture(doc):
        names = sorted(o.Name for o in doc.Objects if o.Name.startswith(StructuralEditParityTest.FEATURES))
        shapes = {n: G.global_shape(doc.getObject(n)) for n in names if G.is_geometry(doc.getObject(n))}
        tips = [G.global_shape(G.tip(doc, r)) for r in ("shell", "lid")]
        return (names, {n: (round(s.Volume, 2), [round(x, 3) for x in bounds(s)]) for n, s in shapes.items()},
                [(round(t.Volume, 1), [round(x, 3) for x in bounds(t)]) for t in tips])

    def check(self, name, value):
        edited = self.built_by_tools()
        text = self.run_tool("set_parameter", doc_name=edited.Name, name=name, value=str(value))
        self.assertIn("Rebuilt stage", text, "a structural edit must report the rebuild")
        want, got = self.picture(self.fresh(name, value)), self.picture(edited)
        self.assertEqual(got[0], want[0], f"{name}: the set of features differs from a fresh build")
        self.assertEqual(got[2], want[2], f"{name}: the tip shapes differ from a fresh build")
        self.assertEqual(got[1], want[1], f"{name}: a feature differs from a fresh build")

    def test_a_taller_skirt_notches_the_ethernet_opening(self):
        self.check("lip_h", 3.6)

    def test_a_smaller_lid_gap_changes_the_openings_and_notches(self):
        self.check("lid_gap", 0.5)

    def test_the_skirt_fit_moves_the_barb_profile_and_the_pocket_height(self):
        self.check("lid_fit", 0.3)

    def test_a_thicker_pcb_lifts_the_windows_and_the_reference(self):
        self.check("pcb_t", 1.6)


if __name__ == "__main__":
    unittest.main()
