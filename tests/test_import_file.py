"""Importing an attached file (``CadEngine.import_file`` and the ``import_file`` tool): every new object measured in
the result, and a mesh (STL / OBJ / PLY) made into a solid on ``as_solid`` so the model can cut, fillet and fuse it.

Against the real FreeCAD runtime in console mode: no GUI, no network.
"""

import math
import os
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

if "FreeCADGui" not in sys.modules:
    try:
        import FreeCADGui  # noqa: F401
    except Exception:
        sys.modules["FreeCADGui"] = types.ModuleType("FreeCADGui")

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402

from oscar import tools  # noqa: E402
from oscar.cad import get_engine, imported  # noqa: E402


def _cube(size=20.0, at=(0.0, 0.0, 0.0)):
    """The 12 triangles of a closed cube, wound outward."""
    x, y, z = at
    v = [(x + dx * size, y + dy * size, z + dz * size)
         for dx, dy, dz in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
    quads = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    return [tri for a, b, c, d in quads for tri in ((v[a], v[b], v[c]), (v[a], v[c], v[d]))]


def _folder():
    return tempfile.mkdtemp(prefix="oscar_import_file_")


def _binary_stl(triangles, name="X_tile.stl"):
    """A binary STL whose header starts with "solid" (as many exporters write it: readers that sniff for ASCII
    by that word get it wrong)."""
    path = os.path.join(_folder(), name)
    with open(path, "wb") as fh:
        fh.write(b"solid X Logo Piece".ljust(80, b" "))
        fh.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            fh.write(struct.pack("<3f", 0, 0, 0))
            for point in tri:
                fh.write(struct.pack("<3f", *point))
            fh.write(b"\0\0")
    return path


def _mesh_file(triangles, name):
    """``triangles`` written by FreeCAD's own Mesh writer, in the format ``name``'s extension says."""
    path = os.path.join(_folder(), name)
    Mesh.Mesh([App.Vector(*point) for tri in triangles for point in tri]).write(path)
    return path


def _new_doc(name):
    doc = App.newDocument(name)
    doc.UndoMode = 1
    return doc


class _DocTestCase(unittest.TestCase):
    """Closes every document a test opened."""

    def setUp(self):
        self._before = set(App.listDocuments())
        self.engine = get_engine()

    def tearDown(self):
        for name in list(App.listDocuments()):
            if name not in self._before:
                try:
                    App.closeDocument(name)
                except Exception:
                    pass


class MeshImportTest(_DocTestCase):
    def test_a_binary_stl_lands_as_a_mesh_with_its_size_and_position(self):
        path = _binary_stl(_cube(20, at=(100, 200, 0)))
        _new_doc("MeshDoc")
        res = self.engine.import_file(path, "MeshDoc")
        self.assertTrue(res["success"], res)
        [part] = res["parts"]
        self.assertEqual((part["kind"], part["triangles"], part["closed"]), ("mesh", 12, True))
        self.assertAlmostEqual(part["volume"], 8000.0, places=2)
        text = tools._imported_text(path, res)
        self.assertIn("mesh of 12 triangles, closed; size 20 × 20 × 20 mm (x 100..120, y 200..220, z 0..20); "
                      "volume 8000 mm³", text)
        self.assertIn("import the file again with as_solid=true", text)

    def test_as_solid_puts_a_solid_in_the_mesh_s_place_in_one_undo_step(self):
        path = _binary_stl(_cube(20, at=(100, 200, 0)))
        doc = _new_doc("SolidDoc")
        undo_before = doc.UndoCount
        res = self.engine.import_file(path, "SolidDoc", as_solid=True)
        self.assertTrue(res["success"], res)
        self.assertEqual(res["solid_notes"], [])
        [name] = res["objects"]
        obj = doc.getObject(name)
        self.assertEqual((obj.TypeId, obj.Label), ("Part::Feature", "X_tile"))
        self.assertEqual([o.TypeId for o in doc.Objects], ["Part::Feature"], "the mesh is gone")
        self.assertEqual(obj.Shape.ShapeType, "Solid")
        self.assertTrue(obj.Shape.isValid())
        self.assertAlmostEqual(obj.Shape.Volume, 8000.0, places=2)
        self.assertEqual(len(obj.Shape.Faces), 6, "the two triangles of each side merge into one face")
        [part] = res["parts"]
        self.assertEqual((part["kind"], part["from_triangles"], part["faces"], part["valid"]), ("solid", 12, 6, True))
        text = tools._imported_text(path, res)
        self.assertIn("solid made from a mesh of 12 triangles, 6 faces; valid solid; size 20 × 20 × 20 mm", text)
        self.assertNotIn("as_solid=true", text)

        # a solid the model can cut
        hole = obj.Shape.cut(Part.makeCylinder(3, 40, App.Vector(110, 210, -10)))
        self.assertAlmostEqual(hole.Volume, 8000.0 - math.pi * 9 * 20, places=1)

        # one undo step takes the whole import back (a reverted turn leaves nothing behind)
        self.assertEqual(doc.UndoCount, undo_before + 1)
        doc.undo()
        self.assertEqual(doc.Objects, [])

    def test_triangles_wound_inside_out_still_make_a_positive_solid(self):
        path = _binary_stl([tuple(reversed(tri)) for tri in _cube(10)], "inside_out.stl")
        doc = _new_doc("FlippedDoc")
        res = self.engine.import_file(path, "FlippedDoc", as_solid=True)
        [part] = res["parts"]
        self.assertEqual(part["kind"], "solid", res)
        self.assertAlmostEqual(doc.getObject(res["objects"][0]).Shape.Volume, 1000.0, places=2)

    def test_a_degenerate_triangle_is_repaired_into_a_valid_solid(self):
        # the top fanned around a point ON its front edge: one triangle of zero area, as exported meshes often have
        s, cube = 10.0, _cube(10)
        corners = [(0, 0, s), (s, 0, s), (s, s, s), (0, s, s)]
        on_edge = (5.0, 0.0, s)
        top = [(corners[0], on_edge, corners[3]), (on_edge, corners[1], corners[2]), (on_edge, corners[2], corners[3]),
               (corners[0], corners[1], on_edge)]
        triangles = [tri for tri in cube if not all(point[2] == s for point in tri)] + top
        raw = Part.Shape()
        raw.makeShapeFromMesh(Mesh.Mesh([App.Vector(*p) for tri in triangles for p in tri]).Topology, imported.SEW_TOLERANCE, False)
        self.assertFalse(Part.Solid(Part.Shell(raw.Faces)).isValid(), "the fixture must need the repair")

        doc = _new_doc("SliverDoc")
        res = self.engine.import_file(_binary_stl(triangles, "sliver.stl"), "SliverDoc", as_solid=True)
        [part] = res["parts"]
        self.assertEqual((part["kind"], part["valid"]), ("solid", True), res)
        shape = doc.getObject(res["objects"][0]).Shape
        self.assertEqual(shape.ShapeType, "Solid")
        self.assertAlmostEqual(shape.Volume, 1000.0, places=2)

    def test_overlapping_pieces_fuse_into_one_solid_and_apart_ones_stay_apart(self):
        # an exported logo is often several closed bodies in one file: sewn as one they make an invalid solid
        doc = _new_doc("PiecesDoc")
        res = self.engine.import_file(_binary_stl(_cube(10) + _cube(10, at=(5, 5, 5)), "overlap.stl"), "PiecesDoc",
                                      as_solid=True)
        [part] = res["parts"]
        self.assertEqual((part["kind"], part["solids"], part["valid"]), ("solid", 1, True), res)
        self.assertAlmostEqual(part["volume"], 2000.0 - 125.0, places=2)
        self.assertEqual(doc.getObject(res["objects"][0]).Shape.ShapeType, "Solid")

        res = self.engine.import_file(_binary_stl(_cube(10) + _cube(10, at=(20, 0, 0)), "apart.stl"), "PiecesDoc",
                                      as_solid=True)
        [part] = res["parts"]
        self.assertEqual((part["kind"], part["solids"], part["valid"]), ("solid", 2, True), res)
        self.assertAlmostEqual(part["volume"], 2000.0, places=2)
        self.assertIn(", 12 faces, 2 solids; valid solid; size 30 × 10 × 10 mm", tools._imported_text("apart.stl", res))

    def test_an_open_mesh_or_one_too_big_stays_a_mesh_and_says_why(self):
        doc = _new_doc("OpenDoc")
        open_path = _binary_stl(_cube(10)[:-1], "open.stl")
        res = self.engine.import_file(open_path, "OpenDoc", as_solid=True)
        self.assertTrue(res["success"], res)
        [part] = res["parts"]
        self.assertEqual((part["kind"], part["closed"]), ("mesh", False))
        self.assertNotIn("volume", part)
        [note] = res["solid_notes"]
        self.assertIn("open stays a mesh: the mesh is not closed", note)
        text = tools._imported_text(open_path, res)
        self.assertIn("mesh of 11 triangles, NOT closed (it has holes)", text)
        self.assertIn("Note: open stays a mesh: the mesh is not closed", text)

        with mock.patch.object(imported, "MAX_SOLID_TRIANGLES", 10):
            res = self.engine.import_file(_binary_stl(_cube(10), "big.stl"), "OpenDoc", as_solid=True)
        self.assertEqual([p["kind"] for p in res["parts"]], ["mesh"])
        self.assertIn("it has 12 triangles; Oscar makes solids of meshes up to 10", res["solid_notes"][0])
        self.assertEqual(len([o for o in doc.Objects if o.TypeId == "Mesh::Feature"]), 2)

    def test_obj_and_ply_meshes_become_solids_too(self):
        for ext in ("obj", "ply"):
            with self.subTest(ext=ext):
                doc = _new_doc(f"Doc{ext}")
                res = self.engine.import_file(_mesh_file(_cube(10), f"cube.{ext}"), doc.Name, as_solid=True)
                self.assertTrue(res["success"], res)
                [part] = res["parts"]
                self.assertEqual((part["kind"], part["faces"]), ("solid", 6), res)
                self.assertAlmostEqual(part["volume"], 1000.0, places=2)


class OtherImportTest(_DocTestCase):
    def test_a_step_reports_its_solid_and_as_solid_changes_nothing(self):
        path = os.path.join(_folder(), "block.step")
        Part.makeBox(10, 20, 30, App.Vector(5, 0, 0)).exportStep(path)
        doc = _new_doc("StepDoc")
        res = self.engine.import_file(path, "StepDoc", as_solid=True)
        self.assertTrue(res["success"], res)
        solids = [p for p in res["parts"] if p["kind"] == "solid"]
        self.assertEqual(len(solids), 1, res)
        self.assertAlmostEqual(solids[0]["volume"], 6000.0, places=2)
        self.assertEqual(res["solid_notes"], [])
        text = tools._imported_text(path, res)
        self.assertIn("solid, 6 faces; size 10 × 20 × 30 mm (x 5..15, y 0..20, z 0..30); volume 6000 mm³", text)
        self.assertNotIn("as_solid", text)
        self.assertTrue(all(o.TypeId != "Mesh::Feature" for o in doc.Objects))

    def test_a_flat_drawing_gives_its_two_dimensions(self):
        try:
            import importDXF  # noqa: F401
        except Exception as e:  # noqa: BLE001 - the Draft importers are not in every build
            self.skipTest(f"no DXF importer: {e}")
        path = os.path.join(_folder(), "outline.dxf")
        lines = [((0, 0), (40, 0)), ((40, 0), (40, 30)), ((40, 30), (0, 30)), ((0, 30), (0, 0))]
        body = "".join(f"0\nLINE\n8\n0\n10\n{a[0]}\n20\n{a[1]}\n30\n0\n11\n{b[0]}\n21\n{b[1]}\n31\n0\n" for a, b in lines)
        with open(path, "w", encoding="ascii") as fh:
            fh.write(f"0\nSECTION\n2\nENTITIES\n{body}0\nENDSEC\n0\nEOF\n")
        _new_doc("DxfDoc")
        res = self.engine.import_file(path, "DxfDoc")
        self.assertTrue(res["success"], res)
        self.assertTrue(res["parts"], res)
        self.assertEqual({p["kind"] for p in res["parts"]}, {"drawing"}, res)
        text = tools._imported_text(path, res)
        self.assertIn("2D drawing; size", text)
        self.assertNotIn("z ", text.split("2D drawing", 1)[1].splitlines()[0])


class LinesTest(unittest.TestCase):
    def test_many_objects_are_counted_with_their_overall_extent(self):
        parts = [{"name": f"Line{i}", "kind": "drawing", "min": [i, 0, 0], "max": [i + 1, 2, 0]} for i in range(15)]
        out = imported.lines(parts)
        self.assertEqual(len(out), imported.MAX_LINES + 1)
        self.assertEqual(out[0], "- Line0: 2D drawing; size 1 × 2 mm (x 0..1, y 0..2)")
        self.assertEqual(out[-1], "- … and 3 more; all 15 together: size 15 × 2 × 0 mm (x 0..15, y 0..2, z 0..0)")


if __name__ == "__main__":
    unittest.main()
