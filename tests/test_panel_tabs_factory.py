# SPDX-License-Identifier: Apache-2.0
"""Integration checks for the Parameters tab seam factory."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402  (the header every test that imports oscar must carry)

isolation.isolate_process()   # sandboxed settings/projects/logs BEFORE importing oscar

from PySide import QtWidgets  # noqa: E402
from oscar.panel_params import ParamsTab  # noqa: E402
from oscar.panel_tabs import make_params_tab  # noqa: E402


class ParamsTabFactoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            cls.app = QtWidgets.QApplication(sys.argv[:1])

    def test_factory_wires_panel_apply_callback_to_params_tab(self):
        seen = []

        def on_apply(changes):
            seen.append(changes)
            return {"ok": True, "text": "Applied."}

        tab = make_params_tab(on_apply=on_apply)
        self.addCleanup(tab.deleteLater)
        self.assertIsInstance(tab, ParamsTab)
        changes = [{"name": "wall_t", "value": 4.0, "apply_text": "4 mm"}]
        self.assertEqual(tab._apply_fn(changes), (True, []))
        self.assertEqual(seen, [changes])

    def test_factory_maps_panel_apply_failure_to_params_tab_errors(self):
        tab = make_params_tab(on_apply=lambda _changes: {"ok": False, "text": "no document"})
        self.addCleanup(tab.deleteLater)
        self.assertEqual(tab._apply_fn([]), (False, ["no document"]))


if __name__ == "__main__":
    unittest.main()
