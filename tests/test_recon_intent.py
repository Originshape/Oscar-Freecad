"""Stage 11: ``flow_intent.mesh_operation`` — which mesh workflow the user asked for, with the
mandated default (an imported mesh with no operation named = parametric reconstruction) and the
priority rule (a later explicit request always wins over the default). Pure module.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

from oscar.flow_intent import mesh_operation  # noqa: E402


class MeshOperationTest(unittest.TestCase):
    CASES = [
        ("Convert STL to solid", "solid"),
        ("Convert this STL into a solid.", "solid"),
        ("Turn the imported mesh into a solid body", "solid"),
        ("Create a solid from this mesh", "solid"),
        ("Just convert it to a solid", "solid"),
        ("Convert mesh to BREP", "brep"),
        ("Create a BREP from this STL.", "brep"),
        ("Convert the imported mesh into a boundary representation", "brep"),
        ("Make this STL editable and parametric", "parametric"),
        ("Reconstruct this mesh as a native FreeCAD model", "parametric"),
        ("I want to change dimensions later", "parametric"),
        ("Import this STL", "parametric"),
        ("Analyse this STL.", "analyse"),
        ("Tell me the dimensions of this mesh", "analyse"),
        ("Check whether this mesh is watertight", "analyse"),
        ("Repair this STL.", "repair"),
        ("Fix the holes in this mesh", "repair"),
        ("Remove duplicate faces", "repair"),
        ("Fix non-manifold geometry on the scan", "repair"),
        ("Increase the hole diameter in this mesh", "modify"),
        ("Make the base of this scan thicker", "modify"),
        ("Convert to solid and export STEP", "solid"),
        ("Reconstruct parametrically and export STEP", "parametric"),
        # W-1 routing contract (demonstrated regression 2026-11-14: an ordinary CAD follow-up
        # "add a fan hole on the side" was hijacked into the mesh workflow): analyse/modify
        # require MESH context — the same phrasing without a mesh word is plain CAD editing.
        ("Tell me its dimensions", None),
        ("Increase the hole diameter", None),
        ("Make the base thicker", None),
        ("Add a fan hole on the side", None),
    ]

    def test_the_documented_table(self):
        for request, expected in self.CASES:
            self.assertEqual(mesh_operation(request), expected, request)

    def test_capitalisation_and_synonyms(self):
        self.assertEqual(mesh_operation("TURN THE IMPORTED MESH INTO A SOLID BODY"), "solid")
        self.assertEqual(mesh_operation("Make me a brep from this stl"), "brep")
        self.assertEqual(mesh_operation("cOnVeRt the StL into a Solid"), "solid")

    def test_non_mesh_requests_are_left_alone(self):
        self.assertIsNone(mesh_operation("Build a housing for a Raspberry Pi 4"))
        self.assertIsNone(mesh_operation("Export as STEP"))          # export flow, not a mesh op
        self.assertIsNone(mesh_operation("What can you do?"))

    def test_the_default_never_overrides_an_explicit_request(self):
        # §5: the default (parametric) must not swallow an explicit later operation
        self.assertEqual(mesh_operation("Convert STL to solid"), "solid")
        self.assertEqual(mesh_operation("Import this STL"), "parametric")


if __name__ == "__main__":
    unittest.main()
