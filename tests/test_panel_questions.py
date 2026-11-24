# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-chat QuestionsBar widget."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_IMPORT_ERROR = ""
try:
    from PySide import QtCore, QtWidgets
    from oscar.panel_questions import QuestionsBar
except Exception as exc:  # pragma: no cover - plain interpreter without Qt
    _IMPORT_ERROR = str(exc)


class QuestionsBarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERROR:
            raise unittest.SkipTest(f"Qt unavailable: {_IMPORT_ERROR}")
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            cls.app = QtWidgets.QApplication(sys.argv[:1])

    def test_can_construct_without_a_pending_question(self):
        bar = QuestionsBar()
        self.addCleanup(bar.deleteLater)
        self.assertTrue(bar.isHidden())

    def test_payload_options_and_custom_answer_are_preserved(self):
        bar = QuestionsBar({
            "text": "Which unit should I use?",
            "options": ["Millimetres", "Inches"],
            "allow_custom": True,
        })
        self.addCleanup(bar.deleteLater)
        bar.show()
        self.assertTrue(bar.isVisible())
        self.assertEqual(bar.label.text(), "Which unit should I use?")
        self.assertEqual(bar.options(), ["Millimetres", "Inches"])
        self.assertTrue(bar.allows_custom())

        answers = []
        bar.answered.connect(answers.append)
        bar._buttons[1].click()
        self.assertEqual(answers, ["Inches"])
        self.assertTrue(bar.is_locked())
        self.assertEqual(bar.recorded(), "Inches")

    def test_custom_answer_submit_emits_once_and_locks_the_bar(self):
        bar = QuestionsBar({"text": "Which unit should I use?", "allow_custom": True})
        self.addCleanup(bar.deleteLater)
        answers = []
        bar.custom_answered.connect(answers.append)
        bar.custom_edit.setText("  use inches  ")
        bar.submit_btn.click()
        bar.submit_btn.click()
        self.assertEqual(answers, ["use inches"])
        self.assertTrue(bar.is_locked())
        self.assertEqual(bar.recorded(), "use inches")

    def test_a_long_question_cannot_grow_the_bar_past_its_cap(self):
        long_text = ("Should Oscar fillet the bracket's load-bearing edges, and at what radius "
                     "should the transition run to keep the part printable? ") * 40
        bar = QuestionsBar({
            "text": long_text,
            "options": [f"Option {i} — a deliberately long choice label to widen the row" for i in range(6)],
            "allow_custom": True,
        })
        self.addCleanup(bar.deleteLater)
        bar.show()
        self.assertLessEqual(bar.maximumHeight(), 200, "the bar must carry a height cap")
        bar.resize(420, 2000)   # offered a huge height: the cap, not the content, decides
        self.assertLessEqual(bar.height(), 200)

    def test_the_cap_scrolls_vertically_without_horizontal_scrolling(self):
        bar = QuestionsBar({"text": "wrap " * 400, "options": ["A", "B"], "allow_custom": True})
        self.addCleanup(bar.deleteLater)
        scroll = bar.findChild(QtWidgets.QScrollArea)
        self.assertIsNotNone(scroll, "long questions must scroll inside the bar, not grow it")
        self.assertEqual(scroll.horizontalScrollBarPolicy(), QtCore.Qt.ScrollBarAlwaysOff)
        self.assertFalse(scroll.horizontalScrollBar().isVisibleTo(bar))
        self.assertIs(scroll.widget(), bar._body)
        self.assertTrue(bar.label.wordWrap())


if __name__ == "__main__":
    unittest.main()
