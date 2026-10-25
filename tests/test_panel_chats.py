"""Renaming and searching chats in the panel (oscar.panel_chats, the ⋯ menu), and the pen's button (oscar.panel_markup):
a real ``OscarPanel`` on an offscreen QApplication, as in test_panel_logic (nothing is shown to the user)."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

_IMPORT_ERROR = ""
try:
    from PySide import QtCore, QtWidgets
    from oscar import chats, panel_chats, panel_markup
    from oscar.providers import Message
    from test_panel_logic import FakeWorker, PanelTestCase
except Exception as exc:   # pragma: no cover - plain interpreter without FreeCAD / Qt
    _IMPORT_ERROR = str(exc)
    PanelTestCase = unittest.TestCase


class _Case(PanelTestCase):
    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERROR:
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")
        super().setUpClass()

    def add_chat(self, title, *messages):
        session = chats.new_session(title)
        session["history"] = [Message(role=role, content=text) for role, text in messages]
        session["blocks"] = None
        self.p.sessions.append(session)
        self.p._rebuild_chat_combo()
        return session

    def action(self, name):
        return next(a for a in self.p.header_menu.actions() if a.objectName() == name)


class RenameTest(_Case):
    def test_the_menu_renames_the_current_chat_and_the_name_is_kept(self):
        session = self.p._current_session()
        self.p.history.append(Message(role="user", content="hello"))
        with mock.patch.object(QtWidgets.QInputDialog, "getText", return_value=("  Lid   for the box ", True)):
            self.action("oscarRenameChatAction").trigger()
        self.assertEqual((session["title"], session["named"]), ("Lid for the box", True))
        self.assertEqual(self.p.chat_combo.currentText(), "Lid for the box")
        self.assertEqual(chats.load(session["id"])["title"], "Lid for the box")

    def test_cancel_or_an_empty_name_changes_nothing(self):
        session = self.p._current_session()
        for answer in (("Other", False), ("   ", True)):
            with mock.patch.object(QtWidgets.QInputDialog, "getText", return_value=answer):
                panel_chats.rename(self.p)
            self.assertEqual(session["title"], "New chat")
            self.assertNotIn("named", session)

    def test_a_chat_named_before_its_first_prompt_keeps_its_name(self):
        session = self.p._current_session()
        with mock.patch.object(QtWidgets.QInputDialog, "getText", return_value=("New chat", True)):
            panel_chats.rename(self.p)   # the same name: nothing to do
        session["title"], session["named"] = "New chat", True
        self.p._title_current_from("Build a flange")
        self.assertEqual(session["title"], "New chat")


class SearchTest(_Case):
    def setUp(self):
        super().setUp()
        self.box = self.add_chat("Workspace: Box", ("user", "Build a 20 mm box"), ("assistant", "Built it."))
        self.bracket = self.add_chat(
            "New chat",
            ("user", "An M3 bracket\n\n--- attached files ---\n[attached file: notes.txt] — flange flange flange"),
            ("assistant", "The bracket has two M3 holes."))

    def test_texts_leave_out_inlined_files_and_the_app_s_own_notes(self):
        self.box["history"].append(Message(role="user", content="[resume] continue", synthetic=True))
        texts = dict(panel_chats.chat_texts([self.box, self.bracket]))
        self.assertEqual(texts["New chat"], [("user", "An M3 bracket"), ("assistant", "The bracket has two M3 holes.")])
        self.assertEqual(texts["Workspace: Box"], [("user", "Build a 20 mm box"), ("assistant", "Built it.")])

    def test_the_dialog_lists_matches_and_enter_chooses_the_selected_one(self):
        dialog = panel_chats.ChatSearchDialog(panel_chats.chat_texts(self.p.sessions))
        self.addCleanup(dialog.deleteLater)
        self.assertIn("Type to search", dialog.hint.text())
        dialog.field.setText("flange")
        self.assertEqual(dialog.results.count(), 0, "inlined file text is not searched")
        self.assertIn("No chat matches", dialog.hint.text())
        dialog.field.setText("m3")
        self.assertEqual(dialog.results.count(), 1)
        self.assertIn("<b>M3</b>", dialog.results.itemWidget(dialog.results.item(0)).text())
        dialog.field.setText("b")   # "Box" (title, first) and "bracket"/"Build" (messages)
        self.assertEqual(dialog.results.count(), 2)
        self.assertEqual(dialog.results.currentRow(), 0)
        down = QtGui_key(QtCore.Qt.Key_Down)
        self.assertTrue(dialog.eventFilter(dialog.field, down))
        self.assertEqual(dialog.results.currentRow(), 1)
        dialog.field.returnPressed.emit()
        self.assertIs(self.p.sessions[dialog.chosen], self.bracket)

    def test_choosing_a_result_opens_that_chat_unless_a_turn_runs(self):
        target = self.p.sessions.index(self.bracket)

        class Choosing:
            def __init__(self, data, parent):
                self.chosen = target

            def exec(self):
                return QtWidgets.QDialog.Accepted

        with mock.patch.object(panel_chats, "ChatSearchDialog", Choosing):
            self.p.worker = FakeWorker()
            panel_chats.search(self.p)
            self.assertIsNot(self.p._current_session(), self.bracket)
            self.p.worker = None
            self.action("oscarSearchChatsAction").trigger()
        self.assertIs(self.p._current_session(), self.bracket)


class PenButtonTest(_Case):
    def test_the_pen_opens_the_markup_unless_a_turn_runs(self):
        with mock.patch.object(panel_markup, "start") as start:
            self.p.worker = FakeWorker()
            self.p.markup_btn.click()
            start.assert_not_called()
            self.p.worker = None
            self.p.markup_btn.click()
        start.assert_called_once()
        self.assertIs(start.call_args.args[0], self.p)
        self.assertEqual(os.path.basename(start.call_args.args[1]), "markup")


def QtGui_key(key):
    from PySide import QtGui

    return QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, QtCore.Qt.NoModifier)


if __name__ == "__main__":
    unittest.main()
