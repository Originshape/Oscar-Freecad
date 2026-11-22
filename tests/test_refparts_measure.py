"""Tests of the FreeCAD side of the reference parts: the fixture generator, ``measure`` (PCB, frame, holes, connectors),
``measure_params`` (the ``Params`` sheet), ``measure_mesh`` and ``importref`` (the hidden ``Ref_<ID>`` ``App::Part``).

Real FreeCAD documents under ``freecadcmd`` and the generated Pi-4-like STEP files; synthetic shapes for every geometric edge case
(flipped, Y-up, quarter-turned, square, fused, tilted, no PCB at all).
"""

import hashlib
import importlib.util
import json
import os
import sys
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402
import Part  # noqa: E402

from oscar.refparts import catalog, importref, measure, measure_mesh, measure_params  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "refparts")
STEP = os.path.join(FIXTURES, "pi4_like.step")
OFFSET = os.path.join(FIXTURES, "pi4_like_offset.step")
BOARD = catalog.load_board("rpi4b")
CANON_BBOX = {"min": [-2.5, -1.0, -1.5], "max": [87.0, 56.0, 17.0], "size": [89.5, 57.0, 18.5]}
HOLES = ((3.5, 3.5, 2.75), (61.5, 3.5, 2.75), (3.5, 52.5, 2.75), (61.5, 52.5, 2.75))


def load_generator():
    spec = importlib.util.spec_from_file_location("build_pi4_step", os.path.join(FIXTURES, "build_pi4_step.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = load_generator()


def new_doc(prefix="Ref"):
    return App.newDocument(f"{prefix}{uuid.uuid4().hex[:6]}")


def plate(length=85.0, width=56.0, thickness=1.4, holes=HOLES, corner=0.0):
    base = Part.makeBox(length, width, thickness)
    if corner:
        vertical = [e for e in base.Edges if abs(e.Vertexes[0].Z - e.Vertexes[1].Z) > thickness / 2]
        base = base.makeFillet(corner, vertical)
    for x, y, d in holes:
        base = base.cut(Part.makeCylinder(d / 2.0, thickness + 2.0, App.Vector(x, y, -1.0)))
    return base


def block(x0, x1, y0, y1, z0, z1):
    return Part.makeBox(x1 - x0, y1 - y0, z1 - z0, App.Vector(x0, y0, z0))


def pi_shapes(pcb=None):
    shapes = [("PCB", pcb if pcb is not None else plate())]
    for c in BOARD["connectors"]:
        b = c["box"]
        shapes.append((c["id"], block(*b["x"], *b["y"], *b["z"])))
    return shapes


def moved(shapes, axis=None, angle=0.0, move=(0, 0, 0)):
    out = []
    for name, shape in shapes:
        s = shape.copy()
        if axis is not None:
            s.rotate(App.Vector(0, 0, 0), App.Vector(*axis), angle)
        s.translate(App.Vector(*move))
        out.append((name, s))
    return out


def records(shapes):
    return [measure.SolidRecord(name, shape) for name, shape in shapes]


def bbox_of_records(recs):
    return measure._bbox_dict([r.box for r in recs])


def near(testcase, a, b, places=3, msg=None):
    testcase.assertEqual(len(a), len(b), msg)
    for x, y in zip(a, b):
        testcase.assertAlmostEqual(x, y, places=places, msg=msg)


class GeneratorTest(unittest.TestCase):
    def build(self):
        folder = isolation.fresh_dir("gen_")
        return folder, GEN.write_fixtures(folder)

    def test_two_runs_give_identical_bytes(self):
        _a, one = self.build()
        _b, two = self.build()
        self.assertEqual(one["hashes"], two["hashes"])
        for key in ("step", "offset"):
            self.assertEqual(open(one[key], "rb").read(), open(two[key], "rb").read())

    def test_hash_file_matches_the_files_in_sha256sum_format(self):
        folder, result = self.build()
        lines = open(result["sha256"], encoding="utf-8").read().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            digest, name = line.split("  ")
            self.assertEqual(hashlib.sha256(open(os.path.join(folder, name), "rb").read()).hexdigest(), digest, name)
        self.assertEqual(sorted(l.split("  ")[1] for l in lines), ["pi4_like.step", "pi4_like_offset.step"])

    def test_the_mirror_folder_holds_the_catalog_named_copy(self):
        folder, result = self.build()
        self.assertEqual(os.path.basename(result["mirror"]), "mirror")
        self.assertEqual(open(os.path.join(result["mirror"], "rpi4b.step"), "rb").read(), open(result["step"], "rb").read())

    def test_the_time_stamp_is_constant(self):
        _folder, result = self.build()
        head = open(result["step"], encoding="utf-8").read(400)
        self.assertIn(GEN.STAMP, head)
        self.assertTrue(head.startswith("ISO-10303-21;"))

    def test_committed_fixtures_match_their_recorded_hash(self):
        recorded = dict(reversed(l.split("  ")) for l in open(os.path.join(FIXTURES, "pi4_like.sha256"), encoding="utf-8").read().splitlines())
        self.assertEqual(sorted(recorded), ["pi4_like.step", "pi4_like_offset.step"])
        for name, digest in recorded.items():
            self.assertEqual(hashlib.sha256(open(os.path.join(FIXTURES, name), "rb").read()).hexdigest(), digest, name)
        self.assertEqual(open(os.path.join(FIXTURES, "mirror", "rpi4b.step"), "rb").read(), open(STEP, "rb").read())

    def test_the_fixture_is_the_pi4_board_file_as_solids(self):
        shapes = GEN.board_shapes()
        self.assertEqual([n for n, _s in shapes], ["PCB"] + [c["id"] for c in BOARD["connectors"]])
        pcb = dict(shapes)["PCB"]
        self.assertAlmostEqual(pcb.BoundBox.XLength, 85.0)
        self.assertAlmostEqual(pcb.BoundBox.YLength, 56.0)
        self.assertAlmostEqual(pcb.BoundBox.ZLength, 1.4)
        self.assertEqual(len([f for f in pcb.Faces if isinstance(f.Surface, Part.Cylinder)]), 4)
        self.assertTrue(all(len(s.Solids) == 1 for _n, s in shapes))

    def test_offset_variant_has_its_origin_at_the_board_centre_and_the_long_edge_along_y(self):
        shapes = dict(GEN.offset_shapes())
        box = shapes["PCB"].BoundBox
        near(self, (box.XMin, box.XMax, box.YMin, box.YMax, box.ZMin, box.ZMax), (-28, 28, -42.5, 42.5, -0.7, 0.7))


class FindPcbTest(unittest.TestCase):
    def test_slab_solid(self):
        pcb = measure.find_pcb(records(pi_shapes()))
        self.assertEqual((pcb.axis, pcb.method, pcb.name, pcb.index), (2, "solid", "PCB", 0))
        near(self, (pcb.thickness, pcb.extent(0), pcb.extent(1)), (1.4, 85.0, 56.0))

    def test_the_largest_thin_solid_wins_over_a_smaller_one(self):
        shapes = pi_shapes() + [("shield", block(10, 20, 10, 20, 1.4, 2.6))]
        self.assertEqual(measure.find_pcb(records(shapes)).name, "PCB")

    def test_pcb_in_any_axis(self):
        for axis, angle in (((1, 0, 0), 90), ((0, 1, 0), 90), ((1, 0, 0), 180)):
            pcb = measure.find_pcb(records(moved(pi_shapes(), axis, angle)))
            self.assertEqual(pcb.name, "PCB", (axis, angle))
            self.assertAlmostEqual(pcb.thickness, 1.4, places=4)

    def test_nothing_that_looks_like_a_board(self):
        self.assertIsNone(measure.find_pcb(records([("cube", block(0, 20, 0, 20, 0, 20))])))
        self.assertIsNone(measure.find_pcb(records([("chip", block(0, 5, 0, 5, 0, 1))])), "below the minimum footprint")
        self.assertIsNone(measure.find_pcb(records([("foil", block(0, 85, 0, 56, 0, 0.2))])), "thinner than any PCB")
        self.assertIsNone(measure.find_pcb(records([("slab", block(0, 85, 0, 56, 0, 6.0))])), "thicker than any PCB")
        self.assertIsNone(measure.find_pcb([]))

    def test_a_tilted_slab_is_not_taken_for_an_axis_aligned_pcb(self):
        tilted = moved([("PCB", plate())], (0, 1, 0), 30)
        self.assertIsNone(measure.find_pcb(records(tilted)))

    def test_fused_model_is_found_as_a_plane_pair(self):
        fused = pi_shapes()[0][1]
        for _n, shape in pi_shapes()[1:]:
            fused = fused.fuse(shape)
        self.assertEqual(len(fused.Solids), 1)
        pcb = measure.find_pcb(records([("all", fused)]))
        self.assertEqual((pcb.method, pcb.axis), ("faces", 2))
        near(self, (pcb.thickness, pcb.extent(0), pcb.extent(1)), (1.4, 85.0, 56.0))


class FitFrameTest(unittest.TestCase):
    def fit(self, shapes, board=BOARD):
        recs = records(shapes)
        frame = measure.fit_frame(recs, board)
        return frame, measure.apply_frame(recs, frame)

    def assertCanonical(self, placed):
        box = bbox_of_records(placed)
        near(self, box["min"], CANON_BBOX["min"], msg=box)
        near(self, box["max"], CANON_BBOX["max"], msg=box)

    def test_canonical_board_is_already_in_the_frame(self):
        frame, placed = self.fit(pi_shapes())
        self.assertTrue(frame.found and frame.is_identity)
        self.assertIn("already in the board frame", frame.describe())
        self.assertEqual(frame.pcb_size[2], 1.4)
        near(self, frame.pcb_size, (85.0, 56.0, 1.4))
        self.assertCanonical(placed)

    def test_every_orientation_comes_back_to_the_canonical_frame_with_a_board_file(self):
        cases = {"flipped 180 about X": ((1, 0, 0), 180, (3, 4, 5)), "Y-up": ((1, 0, 0), 90, (0, 0, 0)), "Y-up other way": ((1, 0, 0), -90, (7, 8, 9)),
                 "X-up": ((0, 1, 0), 90, (1, 2, 3)), "quarter turn": ((0, 0, 1), 90, (-42.5, 28, 0)), "half turn": ((0, 0, 1), 180, (10, 10, 10)),
                 "three quarters": ((0, 0, 1), 270, (0, 0, 0)), "odd": ((1, 1, 0), 180, (5, -5, 2))}
        for label, (axis, angle, move) in cases.items():
            frame, placed = self.fit(moved(pi_shapes(), axis, angle, move))
            self.assertTrue(frame.found, label)
            self.assertCanonical(placed)

    def test_without_a_board_the_long_edge_is_x_and_the_components_stand_up(self):
        for axis, angle in (((1, 0, 0), 180), ((1, 0, 0), 90), ((0, 0, 1), 90)):
            frame, placed = self.fit(moved(pi_shapes(), axis, angle, (4, 5, 6)), board=None)
            box = bbox_of_records(placed)
            near(self, box["size"], CANON_BBOX["size"], msg=(axis, angle))
            pcb = placed[0]
            near(self, (pcb.box[0], pcb.box[1], pcb.box[2]), (0, 0, 0), msg=(axis, angle))
            self.assertAlmostEqual(box["max"][2], 17.0, places=3, msg="the tall stacks are above the PCB")

    def test_the_frame_is_idempotent(self):
        for axis, angle, move in (((0, 0, 1), 90, (-42.5, 28, 0)), ((1, 0, 0), 180, (3, 4, 5)), ((1, 0, 0), 90, (0, 0, 0))):
            frame, placed = self.fit(moved(pi_shapes(), axis, angle, move))
            again = measure.fit_frame(placed, BOARD)
            self.assertTrue(again.is_identity, (axis, angle))
            near(self, again.translation, (0, 0, 0), places=6)

    def test_a_rigid_move_keeps_volumes(self):
        recs = records(moved(pi_shapes(), (1, 0, 0), 90, (3, 4, 5)))
        placed = measure.apply_frame(recs, measure.fit_frame(recs, BOARD))
        for a, b in zip(recs, placed):
            self.assertAlmostEqual(a.volume, b.volume, places=3, msg=a.name)

    def test_apply_frame_does_not_touch_its_input(self):
        recs = records(moved(pi_shapes(), (0, 0, 1), 90, (3, 4, 5)))
        before = [r.shape.BoundBox.XMin for r in recs]
        measure.apply_frame(recs, measure.fit_frame(recs, BOARD))
        self.assertEqual(before, [r.shape.BoundBox.XMin for r in recs])

    def test_a_square_board_is_oriented_by_its_connector(self):
        board = {"connectors": [{"id": "usb", "face": "+x", "box": {"x": [46, 53], "y": [10, 20], "z": [1.4, 8]}}]}
        base = [("PCB", plate(50, 50, 1.4, holes=())), ("usb", block(46, 53, 10, 20, 1.4, 8))]
        for angle in (0, 90, 180, 270):
            frame, placed = self.fit(moved(base, (0, 0, 1), angle, (0, 0, 0)), board=board)
            near(self, placed[1].box, (46, 10, 1.4, 53, 20, 8), msg=angle)
            near(self, placed[0].box[:3], (0, 0, 0), msg=angle)

    def test_a_fused_model_is_normalized_too(self):
        fused = pi_shapes()[0][1]
        for _n, shape in pi_shapes()[1:]:
            fused = fused.fuse(shape)
        for axis, angle in (((0, 0, 1), 0), ((0, 0, 1), 90), ((1, 0, 0), 180)):
            frame, placed = self.fit(moved([("all", fused)], axis, angle, (3, 4, 5)))
            self.assertEqual(frame.method, "faces")
            self.assertIn("one fused solid", frame.note)
            box = bbox_of_records(placed)
            near(self, box["min"], CANON_BBOX["min"], msg=(axis, angle))
            near(self, box["max"], CANON_BBOX["max"], msg=(axis, angle))

    def test_no_pcb_leaves_the_frame_alone(self):
        recs = records([("cube", block(5, 25, 5, 25, 5, 25))])
        frame = measure.fit_frame(recs, BOARD)
        self.assertFalse(frame.found)
        self.assertTrue(frame.is_identity)
        self.assertIn("no PCB found", frame.describe())
        self.assertEqual(measure.apply_frame(recs, frame)[0].box, recs[0].box)

    def test_exotic_shapes_never_raise(self):
        recs = [measure.SolidRecord("pt", Part.Vertex(App.Vector(1, 2, 3)))]
        frame = measure.fit_frame(recs, BOARD)
        self.assertFalse(frame.found)


class HolesTest(unittest.TestCase):
    def holes_of(self, **kwargs):
        rec = measure.SolidRecord("PCB", plate(**kwargs))
        return measure.find_holes(rec, 1.4)

    def test_four_mounting_holes_and_their_pattern(self):
        holes, corner = self.holes_of()
        self.assertEqual(corner, 0.0)
        self.assertEqual(len(holes), 4)
        self.assertEqual({h["d"] for h in holes}, {2.75})
        mount = measure.mounting_pattern(holes)
        self.assertEqual((mount["d"], mount["count"]), (2.75, 4))
        self.assertEqual(mount["pattern"], {"dx": 58.0, "dy": 49.0, "offset": [3.5, 3.5], "symmetric": True})

    def test_asymmetric_offsets_are_reported_as_such(self):
        holes, _c = self.holes_of(holes=((3.5, 5.0, 2.75), (61.5, 5.0, 2.75), (3.5, 54.0, 2.75), (61.5, 54.0, 2.75)))
        self.assertFalse(measure.mounting_pattern(holes)["pattern"]["symmetric"])

    def test_an_irregular_layout_has_holes_but_no_grid(self):
        holes, _c = self.holes_of(length=68.6, width=53.4, holes=((14.0, 2.5, 3.2), (15.3, 50.7, 3.2), (66.1, 7.6, 3.2), (66.1, 35.5, 3.2)))
        mount = measure.mounting_pattern(holes)
        self.assertEqual((mount["count"], mount["pattern"]), (4, None))

    def test_vias_tiny_and_huge_holes_do_not_count(self):
        holes, _c = self.holes_of(holes=HOLES + ((20, 20, 0.3), (25, 20, 1.0), (30, 30, 12.0)))
        self.assertEqual(sorted({h["d"] for h in holes}), [2.75])

    def test_holes_between_range_limits(self):
        holes, _c = self.holes_of(holes=((10, 10, 1.7), (20, 10, 7.0)))
        self.assertEqual(sorted(h["d"] for h in holes), [1.7, 7.0])
        self.assertIsNone(measure.mounting_pattern(holes), "neither is a mounting-size hole")

    def test_the_most_common_diameter_is_the_mounting_size(self):
        holes, _c = self.holes_of(holes=HOLES + ((20, 20, 3.2), (30, 20, 3.2)))
        mount = measure.mounting_pattern(holes)
        self.assertEqual((mount["d"], mount["count"]), (2.75, 4))
        self.assertEqual(len(holes), 6)

    def test_no_holes(self):
        holes, _c = self.holes_of(holes=())
        self.assertEqual((holes, measure.mounting_pattern(holes)), ([], None))

    def test_corner_radius_comes_from_the_convex_arcs(self):
        _holes, corner = self.holes_of(corner=3.0)
        self.assertAlmostEqual(corner, 3.0, places=3)
        holes, _c = self.holes_of(corner=3.0)
        self.assertEqual(len(holes), 4, "the rounded corners are not holes")

    def test_a_boss_is_not_a_hole(self):
        rec = measure.SolidRecord("PCB", plate(holes=()).fuse(Part.makeCylinder(3.0, 5.0, App.Vector(40, 28, 1.4))))
        holes, _c = measure.find_holes(rec, 1.4)
        self.assertEqual(holes, [])

    def test_cylinders_of_other_directions_are_ignored(self):
        sideways = plate(holes=()).cut(Part.makeCylinder(1.4, 20, App.Vector(20, 28, 0.7), App.Vector(1, 0, 0)))
        holes, _c = measure.find_holes(measure.SolidRecord("PCB", sideways), 1.4)
        self.assertEqual(holes, [])

    def test_cluster_helper(self):
        self.assertEqual(measure._cluster([1.0, 1.05, 5.0, 5.02, 9.0], 0.1), [1.025, 5.01, 9.0])
        self.assertEqual(measure._cluster([], 0.1), [])


class AnalyzeFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = measure.analyze(records(pi_shapes()), board=BOARD, top_n=4, object_name="Ref_RPI4B")

    def test_bounding_box_and_heights(self):
        near(self, self.m.bbox["min"], CANON_BBOX["min"])
        near(self, self.m.bbox["max"], CANON_BBOX["max"])
        self.assertEqual((self.m.tallest_z, self.m.bottom_z), (17.0, -1.5))
        self.assertEqual(self.m.solid_count, 10)

    def test_pcb(self):
        self.assertEqual(self.m.pcb, {"solid": "PCB", "method": "solid", "length": 85.0, "width": 56.0, "thickness": 1.4, "z0": 0.0, "z1": 1.4,
                                      "corner_radius": 0.0})

    def test_mounting_holes(self):
        self.assertEqual(self.m.mounting["pattern"]["dx"], 58.0)
        self.assertEqual(len(self.m.holes), 4)

    def test_connectors_match_the_board_file_exactly(self):
        by_id = {c["matches"]: c for c in self.m.connectors}
        self.assertEqual(sorted(by_id), sorted(c["id"] for c in BOARD["connectors"] if c["id"] != "gpio"))
        usb_c = by_id["usb_c_power"]
        self.assertEqual((usb_c["side"], usb_c["offset"], usb_c["width"], usb_c["height"], usb_c["overhang"]), ("-y", 11.2, 9.0, 3.3, 0.5))
        for cid, conn in by_id.items():
            opening = catalog.opening(next(c for c in BOARD["connectors"] if c["id"] == cid))
            self.assertEqual(conn["side"], opening["face"], cid)
            self.assertAlmostEqual(conn["offset"], opening["u"], places=3, msg=cid)
            self.assertAlmostEqual(conn["width"], opening["w"], places=3, msg=cid)
            self.assertAlmostEqual(conn["height"], opening["h"], places=3, msg=cid)
            self.assertAlmostEqual(conn["overhang"], opening["protrusion"], places=3, msg=cid)

    def test_connector_order_is_by_side_then_offset(self):
        keys = [(measure.SIDES.index(c["side"]), c["offset"]) for c in self.m.connectors]
        self.assertEqual(keys, sorted(keys))

    def test_the_gpio_header_is_a_tall_component_not_a_wall_connector(self):
        self.assertEqual([t["name"] for t in self.m.tall], ["gpio"])
        self.assertEqual(self.m.tall[0]["z1"], 9.9)
        self.assertNotIn("gpio", [c["name"] for c in self.m.connectors])

    def test_solid_table_is_bounded_and_sorted_by_volume(self):
        self.assertEqual(len(self.m.solids), 4)
        volumes = [s["volume"] for s in self.m.solids]
        self.assertEqual(volumes, sorted(volumes, reverse=True))
        self.assertEqual(self.m.solids[0]["name"], "PCB")

    def test_identity_against_the_board_outline(self):
        self.assertEqual((self.m.identity["ok"], self.m.identity["status"]), (True, "match"))
        self.assertEqual(self.m.warnings, [])

    def test_to_dict_is_json_safe_and_complete(self):
        data = json.loads(json.dumps(self.m.to_dict()))
        self.assertEqual(sorted(data), ["bbox", "bottom_z", "connectors", "frame", "holes", "identity", "kind", "mesh", "mounting", "object", "pcb",
                                        "solid_count", "solids", "tall", "tallest_z", "units", "warnings"])
        self.assertEqual(data["object"], "Ref_RPI4B")

    def test_without_a_board_connectors_have_no_match(self):
        m = measure.analyze(records(pi_shapes()), board=None)
        self.assertEqual(len(m.connectors), 8)
        self.assertTrue(all("matches" not in c for c in m.connectors))
        self.assertIsNone(m.identity)

    def test_identity_mismatch_is_a_warning_with_a_unit_hint(self):
        pico = catalog.load_board("rpi_pico")
        m = measure.analyze(records(pi_shapes()), board=pico)
        self.assertFalse(m.identity["ok"])
        self.assertIn("probably a different part", m.warnings[0])

    def test_expected_bbox_of_the_catalog_wins_over_the_outline(self):
        good = measure.analyze(records(pi_shapes()), board=BOARD, expected_bbox=[89.5, 57.0, 18.5])
        self.assertEqual(good.identity["status"], "match")
        bad = measure.analyze(records(pi_shapes()), board=BOARD, expected_bbox=[100, 70, 20])
        self.assertFalse(bad.identity["ok"])
        inches = measure.analyze(records(moved(pi_shapes(), move=(0, 0, 0))), board=None, expected_bbox=[89.5 / 25.4, 57 / 25.4, 18.5 / 25.4])
        self.assertIn("millimetres and read as inches", inches.identity["detail"])

    def test_no_solids(self):
        m = measure.analyze([])
        self.assertEqual((m.solid_count, m.pcb), (0, None))
        self.assertIn("no solid", m.warnings[0])

    def test_a_model_without_a_pcb_still_reports_its_size(self):
        m = measure.analyze(records([("cube", block(5, 25, 5, 25, 5, 25))]), board=BOARD)
        self.assertIsNone(m.pcb)
        near(self, m.bbox["size"], (20, 20, 20))
        self.assertEqual((m.connectors, m.holes, m.tallest_z), ([], [], None))
        self.assertIn("no PCB found", m.warnings[0])

    def test_solids_without_volume_or_with_junk_boxes_are_ignored(self):
        junk = measure.SolidRecord("edge", Part.makeLine(App.Vector(0, 0, 0), App.Vector(5, 0, 0)))
        m = measure.analyze(records(pi_shapes()) + [junk])
        self.assertEqual(m.solid_count, 10)

    def test_tiny_parts_do_not_become_connectors_or_raise_the_height(self):
        shapes = pi_shapes() + [("pin", block(-1, -0.9, 10, 10.1, 1.4, 30.0))]        # 0.29 mm3
        m = measure.analyze(records(shapes), board=BOARD)
        self.assertEqual(m.tallest_z, 17.0)
        self.assertNotIn("pin", [c["name"] for c in m.connectors])

    def test_connectors_on_every_side(self):
        shapes = [("PCB", plate(holes=())), ("a", block(10, 20, -2, 5, 1.4, 5)), ("b", block(10, 20, 53, 58, 1.4, 5)),
                  ("c", block(-2, 5, 10, 20, 1.4, 5)), ("d", block(80, 87, 10, 20, 1.4, 5)), ("e", block(30, 40, 20, 30, 1.4, 12))]
        m = measure.analyze(records(shapes))
        sides = {c["name"]: (c["side"], c["overhang"], c["inset"]) for c in m.connectors}
        self.assertEqual(sides, {"a": ("-y", 2.0, 0.0), "b": ("+y", 2.0, 0.0), "c": ("-x", 2.0, 0.0), "d": ("+x", 2.0, 0.0)})
        self.assertEqual([t["name"] for t in m.tall], ["e"])

    def test_a_flush_connector_and_a_slightly_inset_one_count(self):
        shapes = [("PCB", plate(holes=())), ("flush", block(10, 20, 0, 6, 1.4, 5)), ("inset", block(30, 40, 0.4, 6, 1.4, 5)),
                  ("far", block(50, 60, 1.5, 6, 1.4, 5))]
        sides = {c["name"]: c["inset"] for c in measure.analyze(records(shapes)).connectors}
        self.assertEqual(sides, {"flush": 0.0, "inset": 0.4})

    def test_a_part_under_the_board_away_from_the_edges_is_not_listed(self):
        shapes = [("PCB", plate(holes=())), ("top", block(30, 40, 20, 30, 1.4, 10)), ("under", block(50, 60, 20, 30, -3, 0))]
        m = measure.analyze(records(shapes))
        self.assertEqual((m.connectors, [t["name"] for t in m.tall], m.bottom_z), ([], ["top"], -3.0))

    def test_the_side_with_more_parts_is_up(self):
        m = measure.analyze(records([("PCB", plate(holes=())), ("under", block(30, 40, 20, 30, -3, 0))]))
        self.assertEqual((m.bottom_z, m.tallest_z), (0.0, 4.4), "a board with parts only on one side is turned so they stand up")
        self.assertTrue(m.frame["applied"])

    def test_fused_model_reports_pcb_and_holes_but_no_connectors(self):
        fused = pi_shapes()[0][1]
        for _n, shape in pi_shapes()[1:]:
            fused = fused.fuse(shape)
        m = measure.analyze(records([("all", fused)]), board=BOARD)
        self.assertEqual(m.pcb["method"], "faces")
        near(self, (m.pcb["length"], m.pcb["width"], m.pcb["thickness"]), (85, 56, 1.4))
        self.assertEqual(len(m.holes), 4)
        self.assertEqual(m.connectors, [])
        self.assertTrue(any("one fused solid" in w for w in m.warnings))
        self.assertEqual((m.tallest_z, m.bottom_z), (17.0, -1.5))

    def test_a_pcb_with_a_different_thickness(self):
        m = measure.analyze(records([("PCB", plate(thickness=1.6))] + [(n, s) for n, s in pi_shapes()[1:]]), board=BOARD)
        self.assertEqual(m.pcb["thickness"], 1.6)
        self.assertEqual(m.pcb["z1"], 1.6)

    def test_a_motor_like_reference_without_holes_warns(self):
        m = measure.analyze(records([("PCB", plate(holes=()))]))
        self.assertTrue(any("no mounting holes" in w for w in m.warnings))


class RenderTextTest(unittest.TestCase):
    def test_pi4_report(self):
        m = measure.analyze(records(pi_shapes()), board=BOARD, object_name="Ref_RPI4B")
        text = measure.render_text(m)
        self.assertTrue(text.startswith("Measured Ref_RPI4B (exact B-rep, 10 solids, units mm)."))
        for needle in ("PCB: 85 x 56 x 1.4 mm", "Mounting holes: 4 x diameter 2.75", "pattern 58 x 49 mm, offset 3.5, 3.5",
                       "Connector candidates (8", "-y offset 11.2 width 9 height 3.3 z 1.4..4.7 overhang 0.5 (usb_c_power -> usb_c_power)",
                       "tallest_z 17, bottom_z -1.5", "Tall components", "Identity:"):
            self.assertIn(needle, text)
        self.assertLessEqual(len(text.splitlines()), 22)

    def test_top_n_caps_the_connector_rows(self):
        m = measure.analyze(records(pi_shapes()), board=BOARD, object_name="Ref")
        text = measure.render_text(m, top_n=3)
        self.assertEqual(len([l for l in text.splitlines() if l.startswith("  ") and "overhang" in l]), 3)
        self.assertIn("... 5 more", text)

    def test_empty_measurement_renders(self):
        self.assertIn("0 solids", measure.render_text(measure.analyze([], object_name="X")))

    def test_numbers_have_no_trailing_zeros(self):
        self.assertEqual((measure._mm(85.0), measure._mm(2.75), measure._mm(-0.0), measure._mm(0.1 + 0.2)), ("85", "2.75", "0", "0.3"))


class ParamsSyncTest(unittest.TestCase):
    def setUp(self):
        self.doc = new_doc("Pm")
        self.addCleanup(App.closeDocument, self.doc.Name)
        self.m = measure.analyze(records(pi_shapes()), board=BOARD, object_name="Ref_RPI4B")

    def table(self):
        from oscar.kit import params

        return {r["name"]: r for r in params.Params(self.doc).table()}

    def test_planned_rows(self):
        rows = {r["name"]: r["value"] for r in measure_params.planned_rows(self.m)}
        self.assertEqual(rows, {"board_len": 85.0, "board_w": 56.0, "pcb_t": 1.4, "tallest_z": 17.0, "bottom_z": -1.5, "hole_d": 2.75, "hole_dx": 58.0,
                                "hole_dy": 49.0, "hole_off": 3.5})
        self.assertTrue(all(r["unit"] == "mm" for r in measure_params.planned_rows(self.m)))

    def test_nothing_is_planned_without_a_pcb(self):
        self.assertEqual(measure_params.planned_rows(measure.analyze(records([("c", block(0, 9, 0, 9, 0, 9))]))), [])

    def test_asymmetric_holes_do_not_plan_a_single_offset(self):
        m = measure.analyze(records([("PCB", plate(holes=((3.5, 5.0, 2.75), (61.5, 5.0, 2.75), (3.5, 54.0, 2.75), (61.5, 54.0, 2.75))))]))
        names = [r["name"] for r in measure_params.planned_rows(m)]
        self.assertIn("hole_dx", names)
        self.assertNotIn("hole_off", names)

    def test_write_creates_the_sheet_with_step_measurement_rows(self):
        report = measure_params.sync(self.doc, self.m, write=True)
        self.assertEqual(report["written"], ["board_len", "board_w", "pcb_t", "tallest_z", "bottom_z", "hole_d", "hole_dx", "hole_dy", "hole_off"])
        self.assertEqual((report["skipped"], report["diff"], report["sheet"]), ([], [], True))
        rows = self.table()
        self.assertEqual({n: (r["value"], r["unit"], r["source"]) for n, r in rows.items()}["pcb_t"], (1.4, "mm", "step_measurement"))
        self.assertTrue(all(r["source"] == "step_measurement" for r in rows.values()))
        self.assertEqual(report["names"], list(rows))
        self.assertEqual([o.Name for o in self.doc.Objects if o.TypeId == "Spreadsheet::Sheet"], ["Params"])
        self.assertIn("Ref_RPI4B", rows["board_len"]["note"])

    def test_write_adopts_an_existing_sheet_and_never_duplicates_it(self):
        from oscar.kit import params

        sheet = params.ensure_sheet(self.doc)
        sheet.set("wall_t", 2.4, source="user_requirement")
        sheet.set("board_len", 85.0, source="datasheet")
        sheet.set("pcb_t", 1.6, source="assumption")
        report = measure_params.sync(self.doc, self.m, write=True)
        self.assertIn("board_len", report["written"])
        self.assertEqual([o.Name for o in self.doc.Objects if o.TypeId == "Spreadsheet::Sheet"], ["Params"])
        rows = self.table()
        self.assertEqual((rows["wall_t"]["value"], rows["wall_t"]["source"]), (2.4, "user_requirement"), "other rows are untouched")
        self.assertEqual((rows["board_len"]["source"], rows["pcb_t"]["value"]), ("step_measurement", 1.4))
        self.assertEqual([d["name"] for d in report["diff"]], ["pcb_t"])

    def test_a_user_requirement_is_never_overwritten(self):
        from oscar.kit import params

        params.ensure_sheet(self.doc).set("board_len", 90.0, source="user_requirement")
        report = measure_params.sync(self.doc, self.m, write=True)
        self.assertNotIn("board_len", report["written"])
        self.assertEqual(report["skipped"], [{"name": "board_len", "why": "the user set it (source user_requirement)"}])
        self.assertEqual(self.table()["board_len"]["value"], 90.0)
        self.assertEqual(report["diff"][0]["measured"], 85.0)

    def test_pcb_t_tallest_z_and_bottom_z_are_one_group(self):
        from oscar.kit import params

        params.ensure_sheet(self.doc).set("tallest_z", 20.0, source="user_requirement")
        report = measure_params.sync(self.doc, self.m, write=True)
        for name in ("pcb_t", "tallest_z", "bottom_z"):
            self.assertNotIn(name, report["written"])
        self.assertEqual(sorted(s["name"] for s in report["skipped"]), ["bottom_z", "pcb_t", "tallest_z"])
        self.assertIn("written together or not at all", report["skipped"][0]["why"])
        self.assertIn("board_len", report["written"])
        self.assertNotIn("pcb_t", self.table(), "no half-written group")

    def test_formula_rows_are_left_alone(self):
        from oscar.kit import params

        sheet = params.ensure_sheet(self.doc)
        sheet.set("board_w", 56.0, source="datasheet")
        sheet.set("hole_d", 0, formula="=board_w/20", source="derived")
        report = measure_params.sync(self.doc, self.m, write=True)
        self.assertIn({"name": "hole_d", "why": "it is a formula (derived) row"}, report["skipped"])
        self.assertEqual(self.table()["hole_d"]["expr"].replace(" ", ""), "=board_w/20")

    def test_diff_without_write_changes_nothing_and_creates_no_sheet(self):
        report = measure_params.sync(self.doc, self.m, write=False)
        self.assertEqual((report["sheet"], report["written"], report["diff"]), (False, [], []))
        self.assertIsNone(self.doc.getObject("Params"))
        from oscar.kit import params

        sheet = params.ensure_sheet(self.doc)
        sheet.set("board_len", 86.0, source="assumption")
        sheet.set("pcb_t", 1.4, source="assumption")
        before = self.table()
        report = measure_params.sync(self.doc, self.m, write=False)
        self.assertEqual(report["diff"], [{"name": "board_len", "sheet": 86.0, "measured": 85.0, "source": "assumption"}])
        self.assertEqual((report["written"], report["names"]), ([], ["board_len", "pcb_t"]))
        self.assertEqual(self.table(), before)

    def test_differences_below_the_tolerance_are_not_reported(self):
        from oscar.kit import params

        params.ensure_sheet(self.doc).set("board_len", 85.004, source="assumption")
        self.assertEqual(measure_params.sync(self.doc, self.m, write=False)["diff"], [])

    def test_running_twice_is_stable(self):
        measure_params.sync(self.doc, self.m, write=True)
        first = self.table()
        again = measure_params.sync(self.doc, self.m, write=True)
        self.assertEqual(again["diff"], [])
        self.assertEqual(self.table(), first)

    def test_one_failing_row_does_not_stop_the_others(self):
        from oscar.kit import params

        real = params.Params.set

        def flaky(self, name, *a, **k):
            if name == "hole_dx":
                raise params.ParamsError("Params: 'hole_dx' cannot be written")
            return real(self, name, *a, **k)

        with mock.patch.object(params.Params, "set", flaky):
            report = measure_params.sync(self.doc, self.m, write=True)
        self.assertEqual(report["skipped"], [{"name": "hole_dx", "why": "Params: 'hole_dx' cannot be written"}])
        self.assertIn("hole_dy", report["written"])

    def test_a_foreign_object_called_params_is_reported_not_raised(self):
        self.doc.addObject("Part::Feature", "Params")
        report = measure_params.sync(self.doc, self.m, write=True)
        self.assertEqual(report["written"], [])
        self.assertEqual(report["skipped"][0]["name"], "Params")

    def test_the_fallback_writer_produces_the_same_layout(self):
        with mock.patch.object(measure_params, "_kit_params", return_value=None):
            report = measure_params.sync(self.doc, self.m, write=True)
            again = measure_params.sync(self.doc, self.m, write=True)
        self.assertEqual(len(report["written"]), 9)
        self.assertEqual(again["written"], report["written"], "an existing alias is reused, not duplicated")
        rows = self.table()                       # read back through the real kit reader: same layout
        self.assertEqual(len(rows), 9)
        self.assertEqual((rows["board_len"]["value"], rows["board_len"]["source"], rows["pcb_t"]["unit"]), (85.0, "step_measurement", "mm"))
        sheet = self.doc.getObject("Params")
        self.assertEqual([sheet.getContents(f"{c}1").lstrip("'") for c in "ABCDE"], ["name", "value", "unit", "source", "note"])
        self.assertEqual(sheet.getAlias("B2"), "board_len")

    def test_the_fallback_respects_user_rows_too(self):
        with mock.patch.object(measure_params, "_kit_params", return_value=None):
            measure_params.sync(self.doc, self.m, write=True)
            sheet = self.doc.getObject("Params")
            sheet.set("D2", "user_requirement")
            self.assertEqual(sheet.getAlias("B2"), "board_len")
            report = measure_params.sync(self.doc, self.m, write=True)
        self.assertEqual(report["skipped"][0]["name"], "board_len")


class ImportRefTest(unittest.TestCase):
    def setUp(self):
        self.doc = new_doc("Ir")
        self.addCleanup(App.closeDocument, self.doc.Name)

    def names(self):
        return sorted(o.Name for o in self.doc.Objects)

    def do_import(self, path=STEP, **kwargs):
        kwargs.setdefault("ref_id", "rpi4b")
        kwargs.setdefault("board", BOARD)
        return importref.import_reference(self.doc, path, **kwargs)

    def test_structure_of_the_reference_part(self):
        summary = self.do_import(sha256="ab" * 32)
        part = self.doc.getObject("Ref_RPI4B")
        self.assertEqual(part.TypeId, "App::Part")
        self.assertEqual(summary["object"], "Ref_RPI4B")
        self.assertEqual(len(part.Group), 10)
        self.assertEqual(sorted(o.Label for o in part.Group), sorted(["PCB"] + [c["id"] for c in BOARD["connectors"]]))
        parts = [o for o in self.doc.Objects if o.TypeId == "App::Part"]
        self.assertEqual([p.Name for p in parts], ["Ref_RPI4B"], "the importer container is gone")
        self.assertEqual(len([o for o in self.doc.Objects if o.TypeId == "App::Origin"]), 1, "only the reference's own Origin remains")
        self.assertEqual(len([o for o in self.doc.Objects if o.TypeId in ("App::Line", "App::Plane", "App::Point")]), 7)

    def test_everything_is_hidden_and_tagged(self):
        self.do_import(sha256="cd" * 32)
        part = self.doc.getObject("Ref_RPI4B")
        for obj in self.doc.Objects:
            self.assertFalse(obj.Visibility, obj.Name)
        for obj in [part, *part.Group]:
            self.assertEqual((obj.OscarRole, obj.OscarPart, obj.OscarStage, obj.OscarKit), ("ref", "Ref", "reference", ""), obj.Name)
        self.assertEqual((part.OscarRefId, part.OscarRefFile, part.OscarRefSha256), ("rpi4b", "pi4_like.step", "cd" * 32))

    def test_the_importer_names_are_gone_and_leaves_keep_their_geometry(self):
        self.do_import()
        total = sum(o.Shape.Volume for o in self.doc.getObject("Ref_RPI4B").Group)
        original = sum(s.Volume for _n, s in GEN.board_shapes())
        self.assertAlmostEqual(total, original, places=2)

    def test_summary_numbers(self):
        s = self.do_import()
        self.assertEqual((s["kind"], s["solids"], s["facets"]), ("brep", 10, 0))
        near(self, s["bbox"]["size"], CANON_BBOX["size"])
        self.assertEqual(s["largest"][0]["name"], "PCB")
        self.assertEqual(len(s["largest"]), 5)
        self.assertTrue(s["frame"]["found"] and not s["frame"]["applied"])
        self.assertEqual(s["identity"]["status"], "match")
        self.assertEqual(s["warnings"], [])
        json.dumps(s)

    def test_the_offset_file_is_normalized_into_the_board_frame(self):
        s = self.do_import(OFFSET)
        self.assertTrue(s["frame"]["applied"])
        near(self, s["bbox"]["min"], CANON_BBOX["min"])
        near(self, s["bbox"]["max"], CANON_BBOX["max"])
        part = self.doc.getObject("Ref_RPI4B")
        self.assertNotEqual((part.Placement.Base.x, part.Placement.Base.y), (0, 0))
        # every connector sits where the board file says: the usb_c box is on the -y wall at x 6.7..15.7
        usb = next(o for o in part.Group if o.Label == "usb_c_power")
        box = measure.bbox_of(measure.global_shape(usb))
        near(self, box, (6.7, -0.5, 1.4, 15.7, 6.9, 4.7))

    def test_normalize_false_keeps_the_imported_frame(self):
        s = self.do_import(OFFSET, normalize=False)
        part = self.doc.getObject("Ref_RPI4B")
        self.assertEqual(part.Placement.Base.Length, 0.0)
        self.assertFalse(s["frame"]["applied"])

    def test_without_a_board_file_the_frame_is_still_normalized(self):
        s = self.do_import(OFFSET, ref_id="custom", board=None, name="Ref_CUSTOM")
        near(self, s["bbox"]["size"], CANON_BBOX["size"])
        self.assertEqual(s["identity"], None)

    def test_fused_step_is_imported_and_normalized(self):
        fused = pi_shapes()[0][1]
        for _n, shape in pi_shapes()[1:]:
            fused = fused.fuse(shape)
        folder = isolation.fresh_dir("fused_")
        path = os.path.join(folder, "fused.step")
        GEN._export(moved([("All", fused)], (0, 0, 1), 90, (5, 5, 5)), path, "Fused")
        s = self.do_import(path, ref_id="fused", board=BOARD, name="Ref_FUSED")
        self.assertEqual(s["solids"], 1)
        near(self, s["bbox"]["min"], CANON_BBOX["min"])
        self.assertTrue(any("one fused solid" in w for w in s["warnings"]))

    def test_a_second_reference_under_another_name(self):
        self.do_import()
        s = self.do_import(OFFSET, ref_id="other", name="Ref_OTHER")
        self.assertEqual(s["object"], "Ref_OTHER")
        self.assertEqual(len([o for o in self.doc.Objects if o.TypeId == "App::Part"]), 2)
        self.assertEqual(len({o.Label for o in self.doc.Objects}), len(self.doc.Objects), "labels stay unique")

    def test_a_taken_name_is_refused_and_changes_nothing(self):
        self.do_import()
        before = self.names()
        with self.assertRaises(importref.ReferenceImportError) as cm:
            self.do_import()
        self.assertIn("'Ref_RPI4B' already exists", str(cm.exception))
        self.assertIn("name='Ref_<other>'", str(cm.exception))
        self.assertEqual(self.names(), before)

    def test_invalid_names_are_refused(self):
        for bad in ("1bad", "has space", "a-b", "x" * 70):
            with self.assertRaises(importref.ReferenceImportError, msg=bad):
                self.do_import(name=bad)
        self.assertEqual(self.names(), [])

    def test_a_failed_import_leaves_the_document_exactly_as_it_was(self):
        junk = os.path.join(isolation.fresh_dir("junk_"), "junk.step")
        open(junk, "w").write("ISO-10303-21;\nthis is not a model\n")
        self.doc.addObject("Part::Box", "Box")
        self.doc.recompute()
        before, mode = self.names(), self.doc.UndoMode
        with self.assertRaises(importref.ReferenceImportError) as cm:
            self.do_import(junk)
        self.assertIn("import_reference_part:", str(cm.exception))
        self.assertEqual(self.names(), before)
        self.assertEqual(self.doc.UndoMode, mode, "UndoMode is restored")

    def test_surfaces_without_a_solid_are_refused_and_rolled_back(self):
        path = os.path.join(isolation.fresh_dir("face_"), "face.step")
        GEN._export([("Face", Part.makePlane(10, 10))], path, "Face")
        before = self.names()
        with self.assertRaises(importref.ReferenceImportError) as cm:
            self.do_import(path)
        self.assertIn("no solid bodies", str(cm.exception))
        self.assertEqual(self.names(), before)

    def test_undo_mode_is_restored_after_success_too(self):
        for mode in (0, 1):
            self.doc.UndoMode = mode
            name = f"Ref_M{mode}"
            self.do_import(name=name)
            self.assertEqual(self.doc.UndoMode, mode)

    def test_a_successful_import_is_one_undo_step(self):
        self.doc.UndoMode = 1
        self.do_import()
        self.assertEqual(self.doc.UndoCount, 1)
        self.doc.undo()
        self.assertEqual(self.names(), [])

    def test_the_tags_do_not_make_the_kit_own_the_reference(self):
        self.do_import()
        part = self.doc.getObject("Ref_RPI4B")
        self.assertEqual(part.OscarKit, "", "a kit stage rebuild must never delete an imported reference")

    def test_stl_mesh_import(self):
        import Mesh

        mesh = Mesh.Mesh(Part.makeBox(85, 56, 1.4).tessellate(0.1))
        path = os.path.join(isolation.fresh_dir("stl_"), "board.stl")
        mesh.write(path)
        s = self.do_import(path, ref_id="board", board=None, name="Ref_BOARD")
        self.assertEqual((s["kind"], s["facets"]), ("mesh", 12))
        near(self, s["bbox"]["size"], (85, 56, 1.4))
        part = self.doc.getObject("Ref_BOARD")
        self.assertEqual([o.TypeId for o in part.Group], ["Mesh::Feature"])
        self.assertIn("mesh reference", s["frame"]["description"])
        self.assertTrue(any("cannot be measured exactly" in w for w in s["warnings"]))
        self.assertFalse(part.Group[0].Visibility)




if __name__ == "__main__":
    unittest.main()
