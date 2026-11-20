"""M9: TechDraw drawing export (svg / pdf). The page machinery builds headless, but the
writer needs the GUI module — a console session must get the HONEST refusal (never a fake
file), and the GUI path is exercised in the manual checklist. FORMATS carries the new types.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Part  # noqa: E402

from oscar import project_export as X  # noqa: E402


class DrawingExportTest(unittest.TestCase):
    def test_formats_carries_the_drawing_types(self):
        self.assertEqual(X.FORMATS, ("step", "stl", "3mf", "brep", "svg", "pdf"))

    def test_headless_sessions_get_the_honest_refusal(self):
        shape = Part.makeBox(30, 20, 10)
        out = os.path.join(isolation.fresh_dir("draw_"), "part.svg")
        try:
            import TechDrawGui  # noqa: F401

            self.skipTest("GUI module available: the refusal path cannot be exercised here")
        except ImportError:
            pass
        with self.assertRaises(RuntimeError) as ctx:
            X._write_drawing(shape, "svg", out)
        self.assertIn("needs the FreeCAD GUI", str(ctx.exception))
        self.assertFalse(os.path.exists(out), "no fake file may appear on a refusal")


if __name__ == "__main__":
    unittest.main()
