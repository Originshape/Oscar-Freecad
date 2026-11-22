# SPDX-License-Identifier: Apache-2.0
"""PLY point-cloud honesty: a vertices-only PLY is detected, reported as what it is, and refused
for mesh reconstruction with a clear reason — never silently treated as an empty surface mesh
(oscar.refparts.identity.ply_facts + CadEngine.import_file)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402

from oscar.refparts import identity  # noqa: E402

ASCII_MESH_PLY = b"""ply
format ascii 1.0
element vertex 8
property float x
property float y
property float z
element face 12
property list uchar int vertex_indices
end_header
"""

ASCII_POINTS_PLY = b"""ply
format ascii 1.0
element vertex 2048
property float x
property float y
property float z
end_header
"""

BINARY_POINTS_PLY = b"ply\nformat binary_little_endian 1.0\nelement vertex 512\nproperty float x\n" \
                    b"property float y\nproperty float z\nend_header\n" + b"\x00" * 64


class PlyFactsTest(unittest.TestCase):
    def test_a_surface_ply_reports_vertex_and_face_counts(self):
        facts = identity.ply_facts(ASCII_MESH_PLY)
        self.assertEqual((facts["vertices"], facts["faces"], facts["point_cloud"]), (8, 12, False))

    def test_a_vertices_only_ply_is_a_point_cloud(self):
        for blob in (ASCII_POINTS_PLY, BINARY_POINTS_PLY):
            facts = identity.ply_facts(blob)
            self.assertEqual((facts["point_cloud"], facts["faces"]), (True, 0), blob[:20])
            self.assertGreater(facts["vertices"], 0)

    def test_non_ply_data_is_none(self):
        self.assertIsNone(identity.ply_facts(b"solid foo\nendsolid foo\n"))
        self.assertIsNone(identity.ply_facts(b""))
        self.assertIsNone(identity.ply_facts(b"ply\nformat ascii 1.0\n"))   # no end_header


class PointCloudImportTest(unittest.TestCase):
    def _write(self, data: bytes) -> str:
        path = os.path.join(isolation.fresh_dir(), "scan.ply")
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    def test_a_point_cloud_ply_is_refused_with_an_honest_reason(self):
        from oscar.cad import engine

        result = engine.CadEngine().import_file(self._write(ASCII_POINTS_PLY))
        self.assertFalse(result["success"])
        self.assertIn("point cloud", result["error"])
        self.assertIn("2048", result["error"])            # the vertex count is in the message
        self.assertNotIn("import failed", result["error"])  # not an opaque importer traceback

    def test_a_surface_ply_still_imports_as_a_mesh(self):
        from oscar.cad import engine

        path = self._write(ASCII_MESH_PLY + b"0 0 0\n0 1 0\n1 0 0\n0 0 1\n1 1 1\n"
                                             b"0 1 1\n1 0 1\n1 1 0\n" + b"3 0 1 2\n" * 12)
        result = engine.CadEngine().import_file(path)
        self.assertTrue(result["success"], result)
        doc = App.getDocument(result["documents"][0]) if result.get("documents") else None
        if doc is not None:
            App.closeDocument(doc.Name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
