# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-chat QuestionsBar widget."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_IMPORT_ERROR = ""
try:
    from PySide import QtWidgets
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


if __name__ == "__main__":
    unittest.main()
