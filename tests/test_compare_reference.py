"""Tests of ``oscar.refparts.compare``: the deviation rows of a built model against its source mesh.

Every row is {"id", "kind", "a", "b", "metric", "value", "limit", "status", "detail"} with status
PASS | WARN | SKIP — never FAIL (the rows inform; they do not gate the turn). The oracles are exact:
a body compared with its own tessellation is near-zero everywhere, a scaled / shifted body reports
the exact deficit, and an open reference mesh degrades to honest SKIP rows instead of fake zeros.
"""

import os
import sys
import unittest
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402

from oscar.refparts import compare  # noqa: E402

V = App.Vector
TESS = 0.3
ROW_KEYS = {"id", "kind", "a", "b", "metric", "value", "limit", "status", "detail"}
STATUSES = {"PASS", "WARN", "SKIP"}


def mesh_of(shape, deviation=TESS):
    verts, tris = shape.tessellate(deviation)
    return Mesh.Mesh([list(verts[i]) + list(verts[j]) + list(verts[k]) for i, j, k in tris])


def plate_with_holes(hole_d=10.0):
    base = Part.makeBox(100, 60, 8, V(-50, -30, 0))
    for x, y in ((30, 15), (-30, 15), (30, -15), (-30, -15)):
        base = base.cut(Part.makeCylinder(hole_d / 2, 10, V(x, y, -1)))
    return base


class CompareCase(unittest.TestCase):
    def setUp(self):
        self.doc = App.newDocument(f"CR{uuid.uuid4().hex[:6]}")
        self.addCleanup(App.closeDocument, self.doc.Name)

    def rows(self, built, mesh, **kw):
        rows = compare.deviation_rows(built, mesh, **kw)
        for r in rows:
            self.assertLessEqual(ROW_KEYS, set(r))
            self.assertIn(r["status"], STATUSES)
            self.assertEqual(r["kind"], "reference_deviation")
        return {r["id"]: r for r in rows}

    def solid(self, name, shape):
        obj = self.doc.addObject("Part::Feature", name)
        obj.Shape = shape
        return obj


class OracleTest(CompareCase):
    def test_a_body_compared_with_its_own_mesh_is_close_everywhere(self):
        shape = plate_with_holes()
        rows = self.rows(shape, mesh_of(shape))
        self.assertLessEqual(rows["deviation.bbox_axis"]["value"], 0.05)
        self.assertLessEqual(rows["deviation.center_offset"]["value"], 0.05)
        self.assertLessEqual(abs(rows["deviation.volume_share"]["value"]), 0.1)   # percent
        # vertices sampled from the body's own tessellation lie ON the surface: a real
        # measurement legitimately reads ~0 (a fake zero would be a SKIP with value 0.0)
        self.assertEqual(rows["deviation.surface_sampled"]["status"], "PASS")
        self.assertIsNotNone(rows["deviation.surface_sampled"]["value"])
        self.assertLessEqual(rows["deviation.surface_sampled"]["value"], TESS + 0.05)
        self.assertGreaterEqual(rows["deviation.iou_exact"]["value"], 0.995)
        self.assertEqual(rows["deviation.iou_exact"]["status"], "PASS")

    def test_a_wrong_hole_diameter_is_reported_as_a_volume_deficit(self):
        source = mesh_of(plate_with_holes(hole_d=10.0))
        built = plate_with_holes(hole_d=7.0)      # holes 3 mm too small: material left in the model
        rows = self.rows(built, source)
        share = rows["deviation.volume_share"]["value"]
        self.assertGreater(share, 0.25)           # percent of the mesh volume
        self.assertEqual(rows["deviation.volume_share"]["status"], "WARN")
        # the bbox cannot see holes: it stays quiet, it is not its job
        self.assertEqual(rows["deviation.bbox_axis"]["status"], "PASS")

    def test_a_shifted_body_reports_the_exact_centre_offset(self):
        shape = plate_with_holes()
        source = mesh_of(shape)
        moved = shape.copy()
        moved.translate(V(1.0, 0, 0))
        rows = self.rows(moved, source)
        self.assertAlmostEqual(rows["deviation.center_offset"]["value"], 1.0, delta=0.05)
        self.assertEqual(rows["deviation.center_offset"]["status"], "WARN")

    def test_a_scaled_body_reports_the_exact_axis_deficit(self):
        source = mesh_of(plate_with_holes())
        longer = Part.makeBox(102, 60, 8, V(-51, -30, 0))   # the same plate, 2 mm longer in x
        for x, y in ((30, 15), (-30, 15), (30, -15), (-30, -15)):
            longer = longer.cut(Part.makeCylinder(3.5, 10, V(x, y, -1)))
        rows = self.rows(longer, source)
        self.assertAlmostEqual(rows["deviation.bbox_axis"]["value"], 2.0, delta=0.1)   # 100 -> 102 mm
        self.assertEqual(rows["deviation.bbox_axis"]["status"], "WARN")


class HonestyTest(CompareCase):
    def test_an_open_reference_mesh_degrades_to_honest_skips_never_fake_zeros(self):
        shape = plate_with_holes()
        facets = [f.Points for f in mesh_of(shape).Facets]
        open_mesh = Mesh.Mesh([list(p[0]) + list(p[1]) + list(p[2]) for p in facets[:-4]])
        rows = self.rows(shape, open_mesh)
        self.assertEqual(rows["deviation.iou_exact"]["status"], "SKIP")
        self.assertTrue(rows["deviation.iou_exact"]["detail"])
        # the sampled surface row still measures (the points are there, the volume is not)
        self.assertIn(rows["deviation.surface_sampled"]["status"], ("PASS", "WARN"))
        self.assertIsNotNone(rows["deviation.surface_sampled"]["value"])
        self.assertIn("sampled", rows["deviation.surface_sampled"]["detail"])
        # a volume share needs the mesh volume: an open mesh has none worth trusting
        self.assertIn(rows["deviation.volume_share"]["status"], ("SKIP", "WARN"))

    def test_a_zero_sample_budget_skips_the_surface_row_with_value_none(self):
        shape = plate_with_holes()
        rows = self.rows(shape, mesh_of(shape), budget_s=0.0)
        row = rows["deviation.surface_sampled"]
        self.assertEqual(row["status"], "SKIP")
        self.assertIsNone(row["value"])
        self.assertTrue(row["detail"])

    def test_the_sample_cap_is_reported_as_a_share_of_the_mesh_points(self):
        shape = plate_with_holes()
        mesh = mesh_of(shape)
        rows = self.rows(shape, mesh, sample_cap=80)
        row = rows["deviation.surface_sampled"]
        self.assertEqual(row["status"], "PASS")
        self.assertRegex(row["detail"], r"sampled \d+/\d+ mesh points")
        sampled = int(row["detail"].split("sampled ")[1].split("/")[0])
        self.assertLessEqual(sampled, 80)


class RenderTest(CompareCase):
    def test_the_text_block_names_every_row_and_its_status(self):
        shape = plate_with_holes()
        rows = compare.deviation_rows(shape, mesh_of(shape))
        text = compare.render_rows(rows, built="Body", reference="Ref_STL")
        self.assertIn("DEVIATION", text)
        self.assertIn("Body", text)
        self.assertIn("Ref_STL", text)
        for r in rows:
            self.assertIn(r["id"], text)
            self.assertIn(r["status"], text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
