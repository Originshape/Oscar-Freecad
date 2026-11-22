"""P5.3: save → reopen → recompute stability of reconstruction-built models.
A recon FCStd must reopen with intact aliases, expression bindings and a valid
body, survive a forced recompute, and stay editable (a ±10 % parameter change
moves the geometry and keeps the solid valid), with the original value
restored verbatim."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402

from oscar.recon import reconstruct  # noqa: E402


def bracket_mesh():
    shape = Part.makeBox(60, 40, 10)
    for x, y in ((15, 10), (45, 10), (15, 30), (45, 30)):
        shape = shape.cut(Part.makeCylinder(3.3, 30, App.Vector(x, y, -10)))
    return Mesh.Mesh(shape.tessellate(0.3))


class ReopenStabilityTest(unittest.TestCase):
    def test_reconstructed_model_survives_save_reopen_recompute_and_edit(self):
        out = tempfile.mkdtemp(prefix="reconreopen_")
        path = os.path.join(out, "recon.FCStd")
        doc = App.newDocument("ReopenA")
        try:
            result = reconstruct.reconstruct(doc, bracket_mesh(), warn_mm=0.5)
            self.assertTrue(result["build"]["success"], result["build"])
            body = doc.getObject(result["build"]["body"])
            v0 = body.Shape.Volume
            doc.saveAs(path)
        finally:
            App.closeDocument(doc.Name)

        doc = App.openDocument(path)
        try:
            self.addCleanup(App.closeDocument, doc.Name)
            body = next(o for o in doc.Objects if o.TypeId == "PartDesign::Body")
            sheet = next(o for o in doc.Objects
                         if o.isDerivedFrom("Spreadsheet::Sheet"))
            # everything intact after the round trip
            self.assertAlmostEqual(body.Shape.Volume, v0, delta=v0 * 1e-6)
            self.assertTrue(body.Shape.isValid())
            self.assertIsNotNone(sheet.getCellFromAlias("plate_t"),
                                 "aliases survived the round trip")
            # a forced full recompute keeps it valid
            for o in list(doc.Objects):
                o.touch()
            doc.recompute()
            self.assertTrue(body.Shape.isValid())
            self.assertNotIn("Invalid", body.State)
            # and it is still editable: ±10 % on a driving dimension
            cell = sheet.getCellFromAlias("plate_t")
            base = sheet.getContents(cell)
            sheet.set(cell, "=11.0")
            doc.recompute()
            self.assertTrue(body.Shape.isValid() and "Invalid" not in body.State)
            self.assertAlmostEqual(body.Shape.Volume, v0 * 1.10, delta=v0 * 0.02)
            sheet.set(cell, base)
            doc.recompute()
            self.assertAlmostEqual(body.Shape.Volume, v0, delta=v0 * 1e-6)
        finally:
            pass


if __name__ == "__main__":
    unittest.main()
