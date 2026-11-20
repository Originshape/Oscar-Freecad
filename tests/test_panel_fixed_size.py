# SPDX-License-Identifier: Apache-2.0
"""Fixed-size dock regression tests: long interactive questions / messages must never grow Oscar.

Every case is a real ``OscarPanel`` (offscreen QApplication, nothing shown to the user) with
DETERMINISTIC geometry assertions — resize + processEvents, no sleeps. Spec scenarios A-L of the
fixed-size issue; E (wizard card) is intentionally absent: this branch has no WizardCard widget
(yet — the transcript is a QTextBrowser and the interactive surface is the capped QuestionsBar,
whose height cap applies to any payload a future wizard sends through it).

Invariants under test:
- the panel's sizeHint/minimumSizeHint are content-independent (bounded by the bar cap);
- the chat transcript scrolls internally and never contributes its document size;
- no horizontal scrolling appears anywhere at 380-450 px widths;
- the composer and artifacts strip stay inside the panel;
- Parameters / Logs pages keep the tab shell bounded;
- a real QMainWindow docks/floats/moves the panel without content-driven resizing;
- ``_apply_default_dock_geometry`` applies the controlled defaults.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

_IMPORT_ERROR = ""
try:
    from PySide import QtCore, QtWidgets
    from oscar import panel as panel_mod
    from oscar.panel import OscarPanel
except Exception as exc:  # pragma: no cover - plain interpreter without FreeCAD / Qt
    _IMPORT_ERROR = str(exc)
    OscarPanel = object

_LONG = ("Should Oscar fillet the bracket's load-bearing edges, and at what radius should the "
         "transition run to keep the part printable without support material? ") * 40
_LONGER = _LONG * 3


class _PanelCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERROR:
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication(sys.argv[:1])

    def setUp(self):
        self.p = OscarPanel()
        self.addCleanup(self.p.deleteLater)
        self.p.resize(420, 800)
        self.p.show()
        self.pump()

    def pump(self, rounds: int = 6):
        # Bounded event-loop windows so Qt's deferred layouts and the panel's ~100ms render
        # throttle actually run (plain instant rounds return before timers are due).
        for _ in range(rounds):
            self.app.processEvents(QtCore.QEventLoop.AllEvents, 50)

    def bar(self) -> QtWidgets.QWidget:
        bar = self.p.findChild(QtWidgets.QWidget, "oscarQuestionsBar")
        self.assertIsNotNone(bar, "the question bar must be mounted in the chat page")
        return bar

    def show_long_question(self, text=_LONG, options=None):
        labels = options if options is not None else [f"Option {i} — a long choice label" for i in range(6)]
        self.p._show_question_bar(text, labels)
        self.pump()


class ChatContentTest(_PanelCase):
    """A, B, C, D, F, J: content scenarios inside the panel."""

    def test_a_short_question_renders_and_keeps_the_panel_bounded(self):
        self.p._show_question_bar("Fillet radius?", ["2 mm", "4 mm"])
        self.pump()
        bar = self.bar()
        self.assertFalse(bar.isHidden())
        self.assertLessEqual(bar.maximumHeight(), 200)
        self.assertLessEqual(self.p.sizeHint().height(), 1100,
                             "even with a question open the panel hint stays sane")

    def test_b_long_question_cannot_grow_the_panel_the_cap_saturates(self):
        self.p._show_question_bar("Fillet radius?", ["2 mm", "4 mm"])
        self.pump()
        short = self.p.sizeHint().height()
        self.show_long_question()
        once = self.p.sizeHint().height()
        self.show_long_question(_LONGER)
        twice = self.p.sizeHint().height()
        self.assertEqual(once, twice, "3x the question text must not change the panel hint")
        self.assertLessEqual(once - short, 200, "growth is bounded by the bar cap")
        bar = self.bar()
        scroll = bar.findChild(QtWidgets.QScrollArea)
        self.assertEqual(scroll.horizontalScrollBarPolicy(), QtCore.Qt.ScrollBarAlwaysOff)
        self.assertGreater(scroll.verticalScrollBar().maximum(), 0,
                           "the long question scrolls INSIDE the capped bar")

    def test_c_long_description_wraps_without_horizontal_scroll(self):
        self.show_long_question("Decision needed:\n" + _LONG, ["Proceed", "Stop"])
        bar = self.bar()
        self.assertTrue(bar.label.wordWrap())
        scroll = bar.findChild(QtWidgets.QScrollArea)
        self.assertFalse(scroll.horizontalScrollBar().isVisibleTo(bar))
        self.assertFalse(self.p.log.horizontalScrollBar().isVisibleTo(self.p))

    def test_d_long_option_labels_stay_reachable_and_bounded(self):
        self.show_long_question("Pick one:", [LONG_OPTION := (_LONG[:90] + f"… (variant {i})") for i in range(6)])
        bar = self.bar()
        self.assertEqual(len(bar._buttons), 6)
        self.assertLessEqual(self.p.sizeHint().height(), 1100)
        scroll = bar.findChild(QtWidgets.QScrollArea)
        self.assertGreaterEqual(scroll.verticalScrollBar().maximum(), 0)
        self.assertFalse(scroll.horizontalScrollBar().isVisibleTo(bar))

    def test_f_many_messages_scroll_internally_and_never_grow_the_panel(self):
        from oscar.panel_blocks import Block

        self.pump()   # settle the panel's initial (throttled) render first
        before = self.p.sizeHint().height()
        chars_before = self.p.log.document().characterCount()
        # 150 messages: enough to overflow the viewport even with the degenerate font
        # metrics of a font-less offscreen Qt (real displays overflow far sooner).
        for i in range(150):
            self.p._append(Block.ASSISTANT, f"agent message {i}: " + "a reasonably long line. " * 8)
        self.p._render()   # the panel's own synchronous render (worker paths mark dirty + flush)
        self.pump(12)
        mid = self.p.sizeHint().height()
        self.assertEqual(mid, before, "transcript content must not reach the panel hint")
        self.assertGreater(self.p.log.document().characterCount(), chars_before + 1000,
                           "the messages are in the transcript document")
        self.assertGreater(self.p.log.verticalScrollBar().maximum(), 0,
                           "the transcript must scroll inside its viewport")
        for i in range(150):
            self.p._append(Block.ASSISTANT, f"more message {i}: " + "twice as much content. " * 8)
        self.p._render()
        self.pump(12)
        self.assertEqual(self.p.sizeHint().height(), mid)
        self.assertGreater(self.p.log.verticalScrollBar().maximum(), 0)

    def test_j_composer_and_artifacts_stay_inside_the_panel(self):
        self.show_long_question()
        self.p.log.setHtml("<p>" + "message " * 200 + "</p>" * 40)
        self.pump()
        self.assertFalse(self.p.input_frame.isHidden())
        self.assertLessEqual(self.p.input_frame.geometry().bottom(), self.p.height(),
                             "the composer must remain inside the visible panel")
        if not self.p.artifacts_strip.isHidden():
            self.assertLess(self.p.artifacts_strip.geometry().bottom(),
                            self.p.input_frame.geometry().top(),
                            "artifacts stay between the chat and the composer")

    def test_g_narrow_widths_stay_usable(self):
        self.show_long_question()
        for width in (380, 420, 450):
            self.p.resize(width, 700)
            self.pump()
            bar = self.bar()
            self.assertLessEqual(bar.height(), 200, f"bar height at {width}px")
            scroll = bar.findChild(QtWidgets.QScrollArea)
            self.assertFalse(scroll.horizontalScrollBar().isVisibleTo(bar), f"bar h-scroll at {width}px")
            self.assertFalse(self.p.log.horizontalScrollBar().isVisibleTo(self.p), f"log h-scroll at {width}px")
            self.assertTrue(bar.label.wordWrap())

    def test_l_a_bar_answer_still_locks_and_records(self):
        self.p._show_question_bar("Fillet radius?", ["2 mm", "4 mm"])
        self.pump()
        bar = self.bar()
        bar._buttons[0].click()
        self.assertTrue(bar.is_locked())
        self.assertEqual(bar.recorded(), "2 mm")


class SidePagesTest(_PanelCase):
    """K: Parameters and Logs keep the tab shell (and so the panel) bounded."""

    def test_exactly_three_pages_and_each_keeps_a_bounded_hint(self):
        self.assertEqual(self.p.tabs.count(), 3)   # Chat / Parameters / Logs
        hints = []
        for index in range(self.p.tabs.count()):
            self.p.tabs.setCurrentIndex(index)
            self.pump()
            hints.append(self.p.tabs.sizeHint().height())
        self.assertLessEqual(max(hints), 1100, f"per-page tab hints: {hints}")
        # filling the transcript must not change any page's hint
        self.p.tabs.setCurrentIndex(0)
        self.p.log.setHtml("<p>" + "content " * 100 + "</p>" * 60)
        self.pump()
        self.assertLessEqual(self.p.tabs.sizeHint().height(), 1100)

    def test_parameters_table_and_logs_view_are_alive_after_long_chat(self):
        self.show_long_question()
        self.p.log.setHtml("<p>" + "content " * 100 + "</p>" * 60)
        self.pump()
        self.p.tabs.setCurrentIndex(1)
        self.pump()
        table = self.p.tabs.currentWidget().findChild(QtWidgets.QTableWidget)
        self.assertIsNotNone(table, "the Parameters table stays mounted")
        self.p.tabs.setCurrentIndex(2)
        self.pump()
        self.assertIsNotNone(self.p.tabs.currentWidget().findChild(QtWidgets.QScrollArea),
                             "the Logs page keeps its scroll area")


class DockDefaultsTest(_PanelCase):
    """The controlled dock geometry helper (used by ``show_panel`` at creation)."""

    def test_default_geometry_constants_apply_to_a_new_dock(self):
        dock = QtWidgets.QDockWidget("Oscar")
        self.addCleanup(dock.deleteLater)
        panel_mod._apply_default_dock_geometry(dock)
        self.assertEqual((dock.minimumWidth(), dock.minimumHeight()),
                         (panel_mod.DOCK_MIN_WIDTH, panel_mod.DOCK_MIN_HEIGHT))
        self.assertEqual((dock.width(), dock.height()),
                         (panel_mod.DOCK_DEFAULT_WIDTH, panel_mod.DOCK_DEFAULT_HEIGHT))


class DockBehaviourTest(_PanelCase):
    """H, I: real docking, floating and window resizing keep the panel content-driven-free."""

    def setUp(self):
        super().setUp()
        self.mw = QtWidgets.QMainWindow()
        self.addCleanup(self.mw.deleteLater)
        self.dock = QtWidgets.QDockWidget("Oscar", self.mw)
        self.dock.setObjectName("oscarFixedSizeTestDock")
        self.dock.setWidget(self.p)
        self.mw.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dock)   # right, as show_panel does
        panel_mod._apply_default_dock_geometry(self.dock)
        self.mw.resize(1400, 900)
        self.mw.show()
        self.pump()

    def test_h_dock_moves_floats_and_redocks_with_a_long_question_loaded(self):
        self.show_long_question()
        height_docked = self.dock.height()
        for area in (4, 2, 1):   # bottom → right → left
            self.mw.addDockWidget(QtCore.Qt.DockWidgetArea(area), self.dock)
            self.pump()
            self.assertTrue(self.dock.isVisible())
            self.assertFalse(self.p.input_frame.isHidden())
        self.dock.setFloating(True)
        self.pump()
        self.assertTrue(self.dock.isVisible())
        floating_height = self.dock.height()   # Qt keeps the area size when undocking
        self.show_long_question(_LONGER)
        self.pump()
        self.assertEqual(self.dock.height(), floating_height,
                         "longer content must not resize the floating dock")
        self.assertGreaterEqual(self.dock.height(), panel_mod.DOCK_MIN_HEIGHT)
        self.assertFalse(self.p.input_frame.isHidden())
        self.dock.setFloating(False)
        self.mw.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dock)
        self.pump()
        self.assertTrue(self.dock.isVisible())
        self.assertGreaterEqual(self.dock.height(), panel_mod.DOCK_MIN_HEIGHT)

    def test_i_main_window_resizing_does_not_let_content_drive_the_dock(self):
        self.show_long_question()
        self.mw.resize(1500, 1000)
        self.pump()
        big = self.dock.height()
        self.assertLessEqual(big, self.mw.height())
        self.mw.resize(1000, 600)
        self.pump()
        small = self.dock.height()
        self.assertLessEqual(small, self.mw.height())
        self.show_long_question(_LONGER)
        self.pump()
        self.assertEqual(self.dock.height(), small, "longer content did not resize the dock")
        self.assertFalse(self.p.input_frame.isHidden())


if __name__ == "__main__":
    unittest.main()
