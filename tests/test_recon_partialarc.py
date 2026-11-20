# SPDX-License-Identifier: Apache-2.0
"""E2 (G7): partial-arc cylinder extraction from mixed leftover components — two PARALLEL
fillet strips share the plane-normal axis direction, so the whole-component member fit mixes
them and fails; each CONNECTED piece must be fitted on its own (same G2 gates; weak fits stay
unrecognized). Fixtures are built, nothing hardcoded to one radius."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import numpy as np  # noqa: E402
import Part  # noqa: E402

from oscar.recon import recognize  # noqa: E402


def _mesh_of(shape, dev=0.3):
    return Mesh.Mesh(shape.tessellate(dev))


def _component_mesh():
    """A top face carrying TWO PARALLEL fillet strips (r=3) plus one perpendicular strip —
    the geometry that mixes itself into a single axis-consistent member set."""
    shape = Part.makeBox(60, 30, 10)
    top = [e for e in shape.Edges if abs(e.CenterOfMass.z - 10) < 1e-6]
    return _mesh_of(shape.makeFillet(3.0, top))


class PartialArcExtractionTest(unittest.TestCase):
    def setUp(self):
        self.mesh = _component_mesh()
        self.pts = np.array([[p.x, p.y, p.z] for p in self.mesh.Topology[0]])
        self.fac = np.array([tuple(map(int, f)) for f in self.mesh.Topology[1]])
        self.normals = np.array(recognize._facet_normals(self.mesh, len(self.fac)))
        axes = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        # the real caller has already CLAIMED the planes: feed the curved complement
        # (a plane facet's normal is PARALLEL to a plane normal; a strip's sweeps)
        import math as _math

        idx = [f for f in range(len(self.fac))
               if max(abs(float(self.normals[f] @ a)) for a in axes) < _math.cos(_math.radians(12.0))]
        self.out = recognize._extract_cylinders(
            idx, self.pts, self.fac, self.normals, plane_tol=0.05, sigma=0.0,
            _area=lambda chosen: 1.0, plane_normals=axes)

    def _cylinders(self):
        return [o for o in self.out if o.get("type") == "cylinder"]

    def test_parallel_and_perpendicular_strips_are_each_fitted(self):
        cylinders = self._cylinders()
        radii = sorted(round(c["radius"], 1) for c in cylinders)
        self.assertGreaterEqual(len(cylinders), 3, self.out_summary())
        self.assertTrue(all(abs(r - 3.0) <= 0.15 for r in radii), radii)

    def test_the_fits_carry_partial_arc_evidence(self):
        for c in self._cylinders():
            self.assertIn("rms_mm", c)
            self.assertIn("turn_deg", c)
            self.assertLess(c["rms_mm"], 0.1, c["rms_mm"])

    def test_nothing_is_invented_for_the_planes(self):
        # the flat faces are NOT this function's business (the caller claimed them);
        # whatever it cannot fit stays honestly unrecognized, never guessed
        kinds = [o.get("type") for o in self.out]
        self.assertNotIn("plane", kinds)
        for o in self.out:
            self.assertIn(o.get("type"), ("cylinder", "unrecognized"))

    def out_summary(self):
        return [(o.get("type"), o.get("facets"), o.get("radius")) for o in self.out]


class FalsePositiveResistanceTest(unittest.TestCase):
    def test_a_spherical_blob_stays_unrecognized(self):
        mesh = _mesh_of(Part.makeSphere(8.0), 0.3)
        pts = np.array([[p.x, p.y, p.z] for p in mesh.Topology[0]])
        fac = np.array([tuple(map(int, f)) for f in mesh.Topology[1]])
        normals = np.array(recognize._facet_normals(mesh, len(fac)))
        out = recognize._extract_cylinders(
            list(range(len(fac))), pts, fac, normals, plane_tol=0.05, sigma=0.0,
            _area=lambda chosen: 1.0, plane_normals=[])   # the component pass alone
        self.assertEqual([o["type"] for o in out if o["type"] == "cylinder"], [],
                         "a sphere must not be forced into cylinders")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
