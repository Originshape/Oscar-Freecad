"""G10: ``oscar.recon.datums`` — the 3-2-1 datum PROPOSAL and symmetry scores.

A plate reads XY within 0.05 deg (the acceptance residual); a shaft's dominant axis
proposes Z within 0.05 deg; an equiangular cube flags its primary-plane ambiguity;
rotational and mirror symmetries score by re-occupancy/reflection NN RMS. Everything
is a proposal with residuals — the module never creates objects or moves geometry.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402
import numpy as np  # noqa: E402

from oscar.recon import datums, recognize  # noqa: E402


def _angle_to(n, axis):
    n = np.asarray(n, dtype=float)
    n /= np.linalg.norm(n)
    return math.degrees(math.acos(min(1.0, abs(float(n @ np.asarray(axis, dtype=float))))))


class DatumProposalTest(unittest.TestCase):
    def test_a_plate_proposes_xy_within_the_acceptance_residual(self):
        mesh = Mesh.Mesh(Part.makeBox(40, 30, 8).tessellate(0.5))
        got = datums.propose_datums(mesh, recognize.survey(mesh))
        self.assertLessEqual(_angle_to(got["primary_plane"]["normal"], [0, 0, 1]),
                             datums.ANGLE_FLAG_DEG)
        self.assertIn("largest planar region", got["primary_plane"]["basis"])
        self.assertIn("longest in-plane", got["secondary_axis"]["basis"])

    def test_a_small_tilt_measures_exactly(self):
        shape = Part.makeBox(40, 30, 8)
        shape.rotate(App.Vector(0, 0, 0), App.Vector(0, 1, 0), 0.03)
        mesh = Mesh.Mesh(shape.tessellate(0.5))
        got = datums.propose_datums(mesh, recognize.survey(mesh))
        self.assertAlmostEqual(got["alignment"]["angle_deg"], 0.03, places=3)
        self.assertIn("PROPOSAL", got["alignment"]["note"])

    def test_a_shaft_proposes_its_axis_as_z(self):
        mesh = Mesh.Mesh(Part.makeCylinder(8, 30).tessellate(0.4))
        got = datums.propose_datums(mesh, recognize.survey(mesh))
        self.assertEqual(got["secondary_axis"]["basis"], "dominant cylinder axis")
        self.assertLessEqual(_angle_to(got["secondary_axis"]["direction"], [0, 0, 1]),
                             datums.ANGLE_FLAG_DEG)
        self.assertGreaterEqual(got["secondary_axis"]["perpendicularity_deg"], 89.95,
                             "a shaft on its end cap stands perpendicular to it")

    def test_an_equiangular_cube_flags_its_primary_ambiguity(self):
        mesh = Mesh.Mesh(Part.makeBox(20, 20, 20).tessellate(0.5))
        got = datums.propose_datums(mesh, recognize.survey(mesh))
        self.assertTrue(got["ambiguities"], got["ambiguities"])
        self.assertTrue(any("primary plane" in a for a in got["ambiguities"]))

    def test_symmetry_scores_are_evidence(self):
        plate = Mesh.Mesh(Part.makeBox(40, 30, 8).tessellate(0.5))
        got = datums.propose_datums(plate, recognize.survey(plate))
        self.assertIn("mirror", got["symmetry"])
        self.assertEqual(got["symmetry"]["mirror"]["reflection_rms_mm"], 0.0)
        shaft = Mesh.Mesh(Part.makeCylinder(8, 30).tessellate(0.4))
        got2 = datums.propose_datums(shaft, recognize.survey(shaft))
        self.assertIn("rotational", got2["symmetry"])
        self.assertGreaterEqual(got2["symmetry"]["rotational"]["order"], 2)

    def test_the_proposal_never_touches_the_mesh(self):
        mesh = Mesh.Mesh(Part.makeBox(20, 20, 6).tessellate(1.0))
        before = (int(mesh.CountFacets), int(mesh.CountPoints), round(float(mesh.Area), 3))
        datums.propose_datums(mesh, recognize.survey(mesh))
        self.assertEqual((int(mesh.CountFacets), int(mesh.CountPoints),
                          round(float(mesh.Area), 3)), before)

    def test_a_sparse_or_surveyless_call_degrades_honestly(self):
        mesh = Mesh.Mesh(Part.makeBox(20, 20, 6).tessellate(1.0))
        got = datums.propose_datums(mesh)              # no survey: regions empty
        self.assertIn("secondary_axis", got)
        self.assertEqual(got["tertiary_point"]["basis"],
                         "point-cloud centroid (no usable hole axis)")


if __name__ == "__main__":
    unittest.main()
