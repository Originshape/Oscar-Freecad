"""Review of the panel, project, chats, paths, netguard, attachments and settings: regression tests for the
defects the review found by adversarial probing. Each test names the defect it pins.

Qt tests run a real ``OscarPanel`` / ``SettingsDialog`` on an offscreen QApplication (nothing is shown to the user,
no network, workers are small fakes); the rest is pure python. Skipped when PySide / FreeCAD are unavailable.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

_IMPORT_ERROR = ""
try:
    from PySide import QtCore, QtWidgets
    from oscar import chats, netguard, panel
    from oscar import params as _params
    from oscar.panel import Block
    from oscar.params import Profile
    from test_panel_flow import FlowTestCase, FlowWorker, j
    from test_panel_logic import GOOD_SNAPSHOT, PanelTestCase
except Exception as exc:   # pragma: no cover - plain interpreter without FreeCAD / Qt
    _IMPORT_ERROR = str(exc)

    class _Unavailable(unittest.TestCase):
        """Base class that skips every test of a module whose Qt / FreeCAD imports failed."""

        @classmethod
        def setUpClass(cls):
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")

    FlowWorker = object
    PanelTestCase = FlowTestCase = _Unavailable


def _profiles():
    return [Profile(name="Alpha", provider="openai_compat", base_url="http://alpha.test/v1", model="m-alpha",
                    api_key="k-alpha"),
            Profile(name="Beta", provider="openai_compat", base_url="http://beta.test/v1", model="m-beta",
                    api_key="k-beta"),
            Profile(name="Gamma", provider="openai_compat", base_url="http://gamma.test/v1", model="m-gamma",
                    api_key="k-gamma")]


class _SettingsCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERROR:
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            cls.app = QtWidgets.QApplication(sys.argv[:1])

    def dialog(self, profiles=None, active="Alpha"):
        from oscar.settings_ui import SettingsDialog

        _params.save_profiles(profiles or _profiles())
        _params.set_active_profile(active)
        dialog = SettingsDialog()
        self.addCleanup(dialog.deleteLater)
        return dialog

    def select(self, dialog, row):
        dialog.list.setCurrentRow(row)
        self.assertEqual(dialog._editor_row, row)


class SettingsRemoveTest(_SettingsCase):
    """Defect: ``_remove`` left ``_editor_row`` on the row that no longer exists, so the row change that follows
    the list reload stashed the REMOVED profile's editor (name, URL, model, key) into the profile that took its
    place in the list."""

    def test_removing_a_middle_profile_leaves_the_next_one_untouched(self):
        dialog = self.dialog()
        self.select(dialog, 1)                       # Beta
        dialog._remove()
        self.assertEqual([p.name for p in dialog.profiles], ["Alpha", "Gamma"])
        gamma = dialog.profiles[1]
        self.assertEqual((gamma.model, gamma.api_key, gamma.base_url), ("m-gamma", "k-gamma", "http://gamma.test/v1"))

    def test_removing_the_first_profile_leaves_the_second_untouched(self):
        dialog = self.dialog()
        self.select(dialog, 0)
        dialog._remove()
        beta = dialog.profiles[0]
        self.assertEqual((beta.name, beta.model, beta.api_key), ("Beta", "m-beta", "k-beta"))
        self.assertEqual(dialog.name_edit.text(), "Beta")            # the editor shows the profile that is selected now

    def test_removing_the_last_profile_selects_its_neighbour_with_its_own_data(self):
        dialog = self.dialog()
        self.select(dialog, 2)
        dialog._remove()
        self.assertEqual([p.name for p in dialog.profiles], ["Alpha", "Beta"])
        self.assertEqual(dialog.key_edit.text(), "k-beta")
        self.assertEqual(dialog.profiles[0].api_key, "k-alpha")

    def test_an_unsaved_edit_of_another_profile_survives_a_removal(self):
        dialog = self.dialog()
        self.select(dialog, 2)
        dialog.model_combo.setEditText("m-gamma-edited")             # edit Gamma, then remove Beta
        self.select(dialog, 1)
        dialog._remove()
        self.assertEqual(dialog.profiles[1].model, "m-gamma-edited")

    def test_the_last_remaining_profile_cannot_be_removed(self):
        dialog = self.dialog(profiles=_profiles()[:1])
        with mock.patch.object(QtWidgets.QMessageBox, "information") as info:
            dialog._remove()
        self.assertEqual(len(dialog.profiles), 1)
        info.assert_called_once()


class SettingsActiveProfileTest(_SettingsCase):
    """Defect: renaming the ACTIVE profile in the dialog made ``_save`` switch the active profile to the first one
    in the list (the old name no longer existed), silently changing the model Oscar talks to."""

    def test_renaming_the_active_profile_keeps_it_active(self):
        dialog = self.dialog(active="Beta")
        self.select(dialog, 1)
        dialog.name_edit.setText("Beta renamed")
        dialog._save()
        self.assertEqual(_params.active_profile_name(), "Beta renamed")
        self.assertEqual(_params.active_profile().model, "m-beta")

    def test_renaming_another_profile_keeps_the_active_one(self):
        dialog = self.dialog(active="Gamma")
        self.select(dialog, 0)
        dialog.name_edit.setText("Alpha 2")
        dialog._save()
        self.assertEqual(_params.active_profile_name(), "Gamma")

    def test_removing_the_active_profile_falls_back_to_the_first(self):
        dialog = self.dialog(active="Beta")
        self.select(dialog, 1)
        dialog._remove()
        dialog._save()
        self.assertEqual(_params.active_profile_name(), "Alpha")


class SettingsUniqueNamesTest(_SettingsCase):
    """Defect: ``_add`` named a new profile ``Profile <count+1>`` and ``_duplicate`` ``<name> copy`` without looking
    at the names in use, and a rename could collide too: two profiles with one name make the panel's picker and
    ``active_profile`` ambiguous (the second one can never be selected)."""

    def test_add_never_reuses_a_name(self):
        dialog = self.dialog(profiles=[Profile(name="Default"), Profile(name="Profile 3")])
        dialog._add()
        names = [p.name for p in dialog.profiles]
        self.assertEqual(len(set(names)), 3, names)

    def test_duplicating_twice_gives_two_names(self):
        dialog = self.dialog()
        self.select(dialog, 0)
        dialog._duplicate()
        self.select(dialog, 0)
        dialog._duplicate()
        names = [p.name for p in dialog.profiles]
        self.assertEqual(len(set(names)), len(names), names)
        self.assertEqual(sum(n.startswith("Alpha copy") for n in names), 2)

    def test_the_new_profile_is_the_selected_one(self):
        dialog = self.dialog(profiles=[Profile(name="Default"), Profile(name="Profile 3")])
        dialog._add()
        self.assertEqual(dialog.name_edit.text(), dialog.profiles[-1].name)

    def test_a_rename_onto_an_existing_name_is_made_unique_at_save(self):
        dialog = self.dialog()
        self.select(dialog, 1)
        dialog.name_edit.setText("Alpha")
        dialog._save()
        names = [p.name for p in _params.load_profiles()]
        self.assertEqual(len(set(names)), 3, names)
        self.assertIn("Alpha", names)

    def test_unique_names_keep_the_active_profile_active(self):
        dialog = self.dialog(active="Beta")
        self.select(dialog, 1)
        dialog.name_edit.setText("Alpha")                           # the active profile takes an existing name
        dialog._save()
        self.assertEqual(_params.active_profile().model, "m-beta")


class SettingsTextFormatTest(_SettingsCase):
    """Defect: the status labels of the dialog showed error text from the network as RICH text (``QLabel`` auto
    detects markup), so a provider error body such as ``<img src=...>`` was rendered, not shown."""

    def test_status_labels_are_plain_text(self):
        dialog = self.dialog()
        for label in (dialog.test_label, dialog.adv_effective):
            self.assertEqual(label.textFormat(), QtCore.Qt.PlainText, label.objectName() or label.text())

    def test_markup_in_an_error_is_shown_verbatim(self):
        dialog = self.dialog()
        dialog.test_label.setText("<b>boom</b>")
        self.assertEqual(dialog.test_label.text(), "<b>boom</b>")
        self.assertEqual(dialog.test_label.textFormat(), QtCore.Qt.PlainText)


class SettingsAdvancedProblemsTest(_SettingsCase):
    """Defect: an invalid Advanced value (a max output of 100 tokens) was dropped silently: the dialog showed it in
    the spin box and stored nothing. The 'Effective' line now says it was not applied."""

    def test_an_invalid_max_output_is_reported_not_silently_dropped(self):
        dialog = self.dialog()
        self.select(dialog, 0)
        dialog.adv_toggle.setChecked(True)
        dialog.adv_max_out.setValue(100)
        self.assertIn("not applied", dialog.adv_effective.text().lower())
        self.assertIn("max output", dialog.adv_effective.text().lower())

    def test_a_valid_override_has_no_warning(self):
        dialog = self.dialog()
        self.select(dialog, 0)
        dialog.adv_max_out.setValue(2048)
        self.assertNotIn("not applied", dialog.adv_effective.text().lower())


class StartupRestoreTest(unittest.TestCase):
    """Defect: ``_load_sessions`` activated the newest chat with ``restore=False``, so after a restart the chat the
    user was looking at showed only the welcome banner: no restore card, no turn records and, worst, no
    'interrupted in the middle of a step' notice after a crash."""

    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERROR:
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            cls.app = QtWidgets.QApplication(sys.argv[:1])

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="oscar_restore_test_")
        patcher = mock.patch.object(chats, "chats_dir", lambda: self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._hook = netguard.on_permission_wait
        self.addCleanup(setattr, netguard, "on_permission_wait", self._hook)

    def panel(self):
        p = panel.OscarPanel()
        self.addCleanup(p.deleteLater)
        return p

    def saved_chat(self, history, **extra):
        session = chats.new_session("Pi 4 housing")
        session["history"] = history
        session.update(extra)
        chats.save(session)
        return session

    def test_the_newest_chat_gets_its_restore_card_at_startup(self):
        self.saved_chat([panel.Message(role="user", content="build a pi case"),
                         panel.Message(role="assistant", content="done")])
        p = self.panel()
        card = [b for b in p.blocks if b.kind == Block.RESTORE]
        self.assertEqual(len(card), 1)
        self.assertIn("Restored “Pi 4 housing” — 2 messages", p.log.toPlainText())

    def test_a_crash_in_the_middle_of_a_step_is_announced_at_startup(self):
        from oscar.providers import ToolCall

        self.saved_chat([panel.Message(role="user", content="build"),
                         panel.Message(role="assistant", content="", tool_calls=[
                             ToolCall(id="c1", name="execute_code", arguments={})])])
        p = self.panel()
        self.assertIn("interrupted in the middle of a step", p.log.toPlainText())

    def test_an_empty_chat_gets_no_card(self):
        p = self.panel()                                             # nothing saved: a fresh chat
        self.assertEqual([b for b in p.blocks if b.kind == Block.RESTORE], [])


class HeartbeatMarkupTest(FlowTestCase):
    """Defect: the heartbeat row is a ``QLabel`` in auto-detect mode and its text carries the tool name the MODEL
    chose, so a tool called ``<img src=...>`` was rendered as markup."""

    def test_the_heartbeat_shows_a_markup_tool_name_verbatim(self):
        from test_panel_render import progress

        p = self.p
        self.begin_turn()
        p._on_progress(j(progress(activity="running_tool", tool="<img src='x'>", started_at=1.0)))
        p._update_heartbeat()
        self.assertEqual(p.heartbeat_label.textFormat(), QtCore.Qt.PlainText)
        self.assertIn("<img src='x'>", p.heartbeat_label.text())


class SteerWithImageResendTest(FlowTestCase):
    """Defect: a steered message that carried an image was handed to the worker as ``typed text + attachment note``,
    but the late-message bookkeeping looked for the bubble by its DISPLAY text, found none and re-sent the note
    too: the chat showed the message twice, the second time with the machine-written attachment block."""

    def late_steer(self):
        from test_panel_flow import png

        p = self.p
        worker = self.begin_turn()
        image = png(os.path.join(self.tmp, "sketch.png"), 64, 48)
        p.add_attachments([image])
        p.input.setPlainText("use the other board")
        p._send()
        verb, content, images = worker.calls[0]
        self.assertIn("sketch.png", content)
        worker._unsent = [{"text": content, "note": "", "images": images}]
        return worker, images

    def test_a_late_message_with_an_image_leaves_one_bubble_and_resends_the_typed_text(self):
        p = self.p
        _worker, images = self.late_steer()
        sent = []
        p._send_text = lambda text, **kw: sent.append((text, kw))
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn(True, "ok")
        self.app.processEvents()
        self.assertEqual([b for b in p.blocks if b.kind == Block.USER], [])       # the re-send shows it again
        self.assertEqual(sent, [("use the other board", {"images": images})])    # typed text only; the image rides along

    def test_after_a_stop_the_typed_text_returns_to_the_input_with_its_image(self):
        p = self.p
        _worker, images = self.late_steer()
        self.finish_turn(False, "Stopped.")
        self.assertEqual(p.input.toPlainText(), "use the other board")
        self.assertEqual(p._carry_images, images)

    def test_a_delivered_message_keeps_its_bubble(self):
        p = self.p
        worker, _images = self.late_steer()
        worker._unsent = []
        bubble = p._queued_blocks[0]
        self.finish_turn(True, "ok")
        self.assertIn(bubble, p.blocks)
        self.assertFalse(bubble.queued)

    def test_two_identical_late_messages_are_matched_one_to_one(self):
        p = self.p
        worker = self.begin_turn()
        for _ in range(2):
            p.input.setPlainText("again")
            p._send()
        worker._unsent = [{"text": "again", "images": []}, {"text": "again", "images": []}]
        sent = []
        p._send_text = lambda text, **kw: sent.append(text)
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn(True, "ok")
        self.app.processEvents()
        self.assertEqual([b for b in p.blocks if b.kind == Block.USER], [])
        self.assertEqual(sent, ["again\n\nagain"])


class _QuitWorker(FlowWorker if not _IMPORT_ERROR else object):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.waited = []

    def wait(self, ms=0):
        self.waited.append(ms)
        return True


class QuitHookTest(PanelTestCase):
    """Defect: nothing stopped a running turn when the application quit, so Qt destroyed a running ``QThread``
    ("QThread: Destroyed while thread is still running") and the process aborted."""

    def test_quitting_stops_the_turn_and_waits_for_the_thread(self):
        p = self.p
        worker = p.worker = _QuitWorker()
        p._shutdown()
        self.assertIn(("stop",), worker.calls)
        self.assertEqual(worker.waited, [panel._QUIT_WAIT_MS])

    def test_a_paused_turn_is_resumed_first_so_it_can_see_the_stop(self):
        p = self.p
        worker = p.worker = _QuitWorker()
        p._paused = True
        p._shutdown()
        self.assertEqual(worker.calls[:2], [("resume",), ("stop",)])


    def test_an_idle_panel_and_a_failing_worker_are_harmless(self):
        p = self.p
        p._shutdown()
        broken = p.worker = mock.Mock()
        broken.stop.side_effect = RuntimeError("gone")
        p._shutdown()                                                  # no exception

    def test_the_hook_is_connected_to_about_to_quit(self):
        calls = []

        class App:
            aboutToQuit = mock.Mock(connect=lambda slot: calls.append(slot))

        p = self.p
        with mock.patch.object(QtWidgets.QApplication, "instance", return_value=App):
            p._register_quit_hook()
        self.assertEqual(calls, [p._shutdown])


class PersistFailureTest(PanelTestCase):
    """Defect: ``_persist_current`` swallowed every error without a trace (a full disk lost chats silently)."""

    def test_a_failing_save_is_logged_not_raised(self):
        p = self.p
        p.history.append(panel.Message(role="user", content="hello"))
        with mock.patch.object(chats, "save", side_effect=OSError("disk full")), \
                mock.patch("oscar.turnlog.log_event") as log:
            p._persist_current()
        self.assertTrue(any(c.args[:1] == ("panel_error",) and c.kwargs.get("where") == "persist"
                            and "disk full" in c.kwargs.get("error", "") for c in log.call_args_list), log.call_args_list)


class DamagedChatFileTest(unittest.TestCase):
    """Defect: ``_load_sessions`` runs in the panel constructor, and a chat file with a wrong-typed field
    (``project_dir`` a list, ``title`` a number) raised there: one damaged file kept Oscar from opening at all."""

    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERROR:
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            cls.app = QtWidgets.QApplication(sys.argv[:1])

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="oscar_damaged_test_")
        patcher = mock.patch.object(chats, "chats_dir", lambda: self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._hook = netguard.on_permission_wait
        self.addCleanup(setattr, netguard, "on_permission_wait", self._hook)

    def write(self, name, payload):
        with open(os.path.join(self.tmp, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def test_wrong_typed_fields_load_as_defaults(self):
        self.write("20260101-000000-abcdef", {"id": 5, "title": 7, "project_dir": ["x"], "ledger": "plan",
                                              "blocks_meta": {"a": 1}, "messages": [], "created": 1, "updated": []})
        session = chats.load("20260101-000000-abcdef")
        self.assertEqual(session["id"], "20260101-000000-abcdef")
        self.assertEqual(session["title"], "Chat")
        self.assertIsNone(session["project_dir"])
        self.assertIsNone(session["ledger"])
        self.assertEqual(session["blocks_meta"], [])
        self.assertEqual((session["created"], session["updated"]), ("", ""))

    def test_the_panel_opens_with_a_damaged_chat_next_to_a_good_one(self):
        self.write("20260101-000000-abcdef", {"id": None, "title": 7, "project_dir": ["x"], "ledger": [1],
                                              "blocks_meta": "x", "messages": [{"role": "user", "content": "hi"},
                                                                                {"role": "assistant", "content": "yo"}]})
        p = panel.OscarPanel()
        self.addCleanup(p.deleteLater)
        self.assertEqual(len(p.sessions), 1)
        self.assertIn("Restored", p.log.toPlainText())

    def test_a_chat_that_cannot_be_loaded_is_skipped_and_logged(self):
        self.write("20260101-000000-abcdef", {"title": "x", "messages": [{"role": "user", "content": "hi"}]})
        with mock.patch.object(chats, "load", side_effect=RuntimeError("boom")), \
                mock.patch("oscar.turnlog.log_event") as log:
            p = panel.OscarPanel()
        self.addCleanup(p.deleteLater)
        self.assertEqual(len(p.sessions), 1)                           # a fresh chat instead
        self.assertTrue(any(c.kwargs.get("where") == "load_chat" for c in log.call_args_list))


class _RevertHarness(FlowTestCase):
    """A panel whose engine and Project are fakes, with a helper that runs one finished mutating turn."""

    def setUp(self):
        super().setUp()
        from test_panel_logic import FakeEngine

        self.engine = FakeEngine()
        patcher = mock.patch("oscar.cad.get_engine", return_value=self.engine)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.project = mock.Mock(root="", manifest={})
        self.p._current_session()["_project"] = self.project

    def turn(self, *indexes):
        from test_panel_render import finished, started

        p = self.p
        p._turn_snapshot = dict(GOOD_SNAPSHOT)
        self.begin_turn()
        for n, index in enumerate(indexes):
            call = f"call_{n}"
            p._on_tool_started("execute_code")
            p._on_step_started(j(started(call, ordinal=n + 1)))
            p._on_step_finished(j(finished(call, ordinal=n + 1, step_index=index)))
            p._on_tool_finished("execute_code(code=...)", True, "ok")
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn(True, "ok")
        return self.card()


class PanelRevertJournalTest(_RevertHarness):
    """The panel tells the Project which steps a verified revert removed, and only after the engine confirmed it."""

    def test_a_verified_revert_journals_the_first_step_of_the_turn(self):
        card = self.turn(7, 8, 9)
        self.p._revert_turn(card)
        self.assertTrue(card.reverted)
        self.project.note_reverted.assert_called_once_with(7)

    def test_a_turn_without_recorded_steps_journals_nothing(self):
        card = self.turn(-1, -1)
        self.p._revert_turn(card)
        self.assertTrue(card.reverted)
        self.project.note_reverted.assert_not_called()

    def test_a_failed_revert_journals_nothing(self):
        self.engine.revert = {"success": False, "error": "no undo"}
        card = self.turn(4)
        self.p._revert_turn(card)
        self.assertFalse(card.reverted)
        self.project.note_reverted.assert_not_called()

    def test_an_incomplete_revert_journals_nothing(self):
        self.engine.after = {"success": True, "docs": {"Doc": 5}}   # the document still holds undo steps of the turn
        card = self.turn(4)
        self.p._revert_turn(card)
        self.assertFalse(card.reverted)
        self.project.note_reverted.assert_not_called()

    def test_a_journal_failure_never_breaks_the_revert(self):
        self.project.note_reverted.side_effect = RuntimeError("disk")
        card = self.turn(4)
        with mock.patch("oscar.turnlog.log_event") as log:
            self.p._revert_turn(card)
        self.assertTrue(card.reverted)
        self.assertTrue(any(c.kwargs.get("where") == "revert_journal" for c in log.call_args_list))


class WorkerEndedTest(FlowTestCase):
    """Defect: the panel relied on ``turn_finished`` alone, so a worker thread that ended without emitting it left the
    panel busy for good (Stop button visible, Send turned into Pause, every chat action refused)."""

    def test_a_thread_that_ended_without_turn_finished_closes_the_turn_as_failed(self):
        p = self.p
        worker = self.begin_turn()
        worker.isRunning = lambda: False
        p._worker_ended()
        self.assertIs(p.worker, worker)                               # one event-loop turn of grace for a queued turn_finished
        self.app.processEvents()
        self.assertIsNone(p.worker)
        self.assertIn("stopped unexpectedly", p.log.toPlainText())
        self.assertTrue(p.stop_btn.isHidden())
        self.assertFalse(p._busy())
        self.assertEqual(p._turn_outcome[0], "failed")

    def test_a_worker_that_is_still_running_is_left_alone(self):
        p = self.p
        worker = self.begin_turn()
        worker.isRunning = lambda: True
        p._worker_ended()
        self.app.processEvents()
        self.assertIs(p.worker, worker)

    def test_a_worker_without_isrunning_is_left_alone(self):
        p = self.p
        worker = self.begin_turn()
        p._worker_ended()
        self.app.processEvents()
        self.assertIs(p.worker, worker)

    def test_after_a_normal_finish_the_late_thread_signal_does_nothing(self):
        p = self.p
        self.begin_turn()
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn(True, "Done.")
        before = len(p.blocks)
        p._worker_ended()
        self.app.processEvents()
        self.assertEqual(len(p.blocks), before)

    def test_a_turn_finished_that_arrives_after_the_thread_signal_is_the_only_ending(self):
        p = self.p
        worker = self.begin_turn()
        worker.isRunning = lambda: False
        p._worker_ended()                                              # the thread signal came first ...
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn(True, "Done.")                           # ... but the real ending was already queued
        self.app.processEvents()
        self.assertNotIn("stopped unexpectedly", p.log.toPlainText())
        self.assertEqual(p.log.toPlainText().count("Done."), 1)

    def test_send_connects_the_thread_signal(self):
        from test_panel_session import CapturingWorker

        class Ended(CapturingWorker):
            def __init__(self, history, parent=None, **kw):
                super().__init__(history, parent, **kw)
                self.finished = type(self).SIG = mock.Mock()

        engine = mock.Mock()
        engine.begin_turn_snapshot.return_value = dict(GOOD_SNAPSHOT)
        p = self.p
        with mock.patch.object(panel, "AgentWorker", Ended), \
                mock.patch.object(_params, "active_profile",
                                  return_value=Profile(name="t", provider="openai_compat", model="m")), \
                mock.patch("oscar.cad.get_engine", return_value=engine):
            p.input.setPlainText("hello")
            p._send()
        Ended.SIG.connect.assert_called_once_with(p._worker_ended)




class RealWorkerReviewTest(FlowTestCase):
    """Per-chat Project binding, persistence and chat switching with a REAL ``AgentWorker`` thread (scripted provider,
    fake tools, a real Project in the test sandbox), the same harness as test_panel_e2e."""

    def setUp(self):
        super().setUp()
        import threading
        import time

        from oscar import agent
        from oscar.cad import dispatch
        from oscar.toolctx import ToolOutcome, current_context
        from test_panel_flow import png

        self.release = threading.Event()
        self.in_tool = threading.Event()
        self.block_on = ""
        self.workers = []
        engine = mock.Mock()
        engine.begin_turn_snapshot.return_value = dict(GOOD_SNAPSHOT)

        def fake_run_tool(name, args):
            ctx = current_context()
            if name == self.block_on:
                self.in_tool.set()
                self.release.wait(20)
            geometry = name == "create_object"
            shot = png(os.path.join(self.tmp, f"{name}_{int(time.monotonic() * 1000)}.png")) if geometry else ""
            outcome = ToolOutcome(
                name=name, call_id=ctx.call_id, title=args.get("step_title") or f"Step {name}", ok=True,
                handler_ok=True, text=f"{name} done", mutating=True, geometry=geometry,
                step_index=1 if geometry else -1, snapshot_path=shot, args=dict(args),
                delta={"added": ["Box"], "removed": [], "changed": []} if geometry else {})
            ctx.outcome = outcome
            if ctx.ledger is not None:
                ctx.ledger.on_tool_result(outcome)
            return f"{name} done", None, False

        dispatch.init()
        self.addCleanup(dispatch.shutdown)
        for patcher in (mock.patch.object(_params, "active_profile",
                                          return_value=Profile(name="t", provider="openai_compat", model="m")),
                        mock.patch("oscar.cad.get_engine", return_value=engine),
                        mock.patch.object(agent._tools, "run_tool", fake_run_tool)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.stop_worker)

    def stop_worker(self):
        for panel_ in [self.p] + getattr(self, "extra_panels", []):
            worker = panel_.worker
            if worker is not None:
                self.release.set()
                worker.stop()
                worker.wait(5000)

    def wait_until(self, condition, seconds=30):
        import time

        deadline = time.monotonic() + seconds
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.01)
        self.app.processEvents()
        return condition()

    def start_turn(self, text, script, on=None):
        from oscar import agent
        from support.fakes import fake_validation
        from support.scripted_llm import ScriptedProvider

        target = on or self.p
        provider = ScriptedProvider(script)
        profile = Profile(name="t", provider="openai_compat", model="scripted")

        def factory(history, **kw):
            self.workers.append(kw)
            return agent.AgentWorker(history, provider=provider, profile=profile,
                                     validator=lambda doc, ctx: fake_validation(True), **kw)

        with mock.patch.object(panel, "AgentWorker", factory):
            target.input.setPlainText(text)
            target._send()
        return provider

    def box_turn(self, text="Build the smoke test box.", on=None):
        from support.scripted_llm import call, calls, say

        target = on or self.p
        self.start_turn(text, [calls(call("create_object", doc_name="D", obj_type="Part::Box", step_title="Make the box")),
                               say("Built.")], on=target)
        self.assertTrue(self.wait_until(lambda: target.worker is None), "the turn did not finish")

    def projects_root(self):
        from oscar import paths

        return paths.projects_dir(probe=False)

    def test_a_restarted_panel_restores_the_chat_and_continues_in_the_same_project(self):
        p = self.p
        self.box_turn()
        session = p._current_session()
        root, sid = session["project_dir"], session["id"]
        self.assertTrue(os.path.isfile(os.path.join(root, "project.json")))
        folders = sorted(os.listdir(self.projects_root()))
        # --- the app restarts: a new panel over the same chats folder ---
        p2 = panel.OscarPanel()
        self.extra_panels = [p2]
        self.addCleanup(p2.deleteLater)
        again = p2._current_session()
        self.assertEqual(again["id"], sid)
        self.assertEqual(again["project_dir"], root)
        cards = [b for b in p2.blocks if b.kind == Block.RESTORE]
        self.assertEqual(len(cards), 1)
        self.assertTrue(cards[0].thumbs)                                    # the snapshot of the geometry step
        self.assertFalse(p2.plan_strip.isHidden())
        self.assertFalse(p2.artifacts_strip.isHidden())
        self.assertIn("Make the box", p2.log.toPlainText())
        # --- the next turn binds the SAME project folder, no second one is created ---
        self.workers.clear()
        self.box_turn("Add a second box.", on=p2)
        self.assertEqual(os.path.realpath(self.workers[-1]["project"].root), os.path.realpath(root))
        self.assertEqual(sorted(os.listdir(self.projects_root())), folders)
        from oscar.project import Project

        self.assertEqual(len(Project.open(root)._journal()[1]), 2)           # both turns are in ONE journal
        self.assertEqual(p2._current_session()["project_dir"], root)

    def test_every_chat_has_its_own_project_and_a_second_chat_never_touches_the_first(self):
        from oscar.project import Project

        p = self.p
        first = p._current_session()
        self.box_turn("Build the first box.")
        root_a = first["project_dir"]
        p._new_chat()
        second = p._current_session()
        self.assertIsNot(second, first)
        self.box_turn("Build the second box.")
        root_b = second["project_dir"]
        self.assertNotEqual(os.path.realpath(root_a), os.path.realpath(root_b))
        self.assertEqual(Project.open(root_b).manifest["chat_id"], second["id"])
        self.assertEqual(Project.open(root_a).manifest["chat_id"], first["id"])
        self.assertEqual(len(Project.open(root_a)._journal()[1]), 1)         # one turn each: nothing crossed over
        self.assertEqual(len(Project.open(root_b)._journal()[1]), 1)
        # back to the first chat: its plan and strips come back, the second chat's do not leak in
        p._on_chat_picked(next(i for i, s in enumerate(p.sessions) if s is first))
        self.assertIn(os.path.basename(root_a), p.artifacts_strip.text() + p._artifacts["project_dir"])
        self.assertEqual(p._artifacts["project_dir"], root_a)

    def test_chat_actions_are_refused_while_a_turn_runs(self):
        from support.scripted_llm import call, calls, say

        p = self.p
        first = p._current_session()
        p._new_chat()
        p._on_chat_picked(next(i for i, s in enumerate(p.sessions) if s is first))
        self.assertIs(p._current_session(), first)
        self.block_on = "create_object"
        self.start_turn("Build the smoke test box.", [
            calls(call("create_object", doc_name="D", obj_type="Part::Box")), say("Done.")])
        self.assertTrue(self.wait_until(self.in_tool.is_set))
        count = len(p.sessions)
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes) as ask:
            p._on_chat_picked(0)
            p._new_chat()
            p._clear_chat()
            p._delete_chat()
        ask.assert_not_called()                                              # refused before any dialog
        self.assertIs(p._current_session(), first)
        self.assertEqual(len(p.sessions), count)
        self.assertEqual(p.chat_combo.currentIndex(), next(i for i, s in enumerate(p.sessions) if s is first))
        self.assertTrue(first["history"])
        self.release.set()
        self.assertTrue(self.wait_until(lambda: p.worker is None))

    def test_the_chat_file_is_saved_after_every_round_and_stays_readable(self):
        from support.scripted_llm import call, calls, say

        p = self.p
        self.block_on = "create_document"
        self.start_turn("Build the smoke test box.", [
            calls(call("create_object", doc_name="D", obj_type="Part::Box")),
            calls(call("create_document", name="D")),
            say("Done.")])
        self.assertTrue(self.wait_until(self.in_tool.is_set))                # round 2 is running; round 1 is complete
        sid = p._current_session()["id"]
        path = os.path.join(self._tmp, sid + ".json")
        saved = chats.load(sid)                                              # while the worker thread is still busy
        self.assertIsNotNone(saved)
        roles = [m.role for m in saved["history"]]
        self.assertEqual(roles[:3], ["user", "assistant", "user"])           # the request, round 1's calls, their results
        self.assertTrue(saved["history"][1].tool_calls)
        self.assertTrue(saved["history"][2].tool_results)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual([n for n in os.listdir(self._tmp) if n.endswith(".tmp")], [])   # no half-written file left behind
        self.release.set()
        self.assertTrue(self.wait_until(lambda: p.worker is None))
        final = chats.load(sid)
        self.assertEqual(final["history"][-1].content, "Done.")
        self.assertEqual(final["project_dir"], p._current_session()["project_dir"])

    def test_deleting_a_chat_removes_its_file_but_keeps_the_project_folder(self):
        p = self.p
        first = p._current_session()
        self.box_turn()
        root = first["project_dir"]
        p._new_chat()
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
            p._on_chat_picked(next(i for i, s in enumerate(p.sessions) if s is first))
            p._delete_chat()
        self.assertFalse(os.path.exists(os.path.join(self._tmp, first["id"] + ".json")))
        self.assertFalse(os.path.exists(os.path.join(self._tmp, first["id"] + ".meta.json")))
        self.assertTrue(os.path.isfile(os.path.join(root, "project.json")))   # the user's files stay
        self.assertNotIn(first, p.sessions)
        self.assertNotIn(first["id"], [m["id"] for m in chats.list_sessions()])

    def test_a_message_steered_into_a_running_turn_ends_as_one_plain_bubble(self):
        from support.scripted_llm import call, calls, say

        p = self.p
        self.block_on = "create_object"
        self.start_turn("Build the smoke test box.", [
            calls(call("create_object", doc_name="D", obj_type="Part::Box")), say("Done."), say("Also done.")])
        self.assertTrue(self.wait_until(self.in_tool.is_set))
        p.input.setPlainText("make it 20 mm wide")
        p._send()
        bubble = p._queued_blocks[0]
        self.assertTrue(bubble.queued)
        self.release.set()
        self.assertTrue(self.wait_until(lambda: p.worker is None))
        self.app.processEvents()
        self.wait_until(lambda: p.worker is None and not p._queued_blocks)
        users = [b for b in p.blocks if b.kind == Block.USER and "20 mm wide" in b.text]
        self.assertEqual(len(users), 1, [b.text for b in p.blocks if b.kind == Block.USER])
        self.assertFalse(users[0].queued)


class SettingsCommandRefreshTest(PanelTestCase):
    """Defect: settings saved through the ``Oscar_Settings`` command left an open Oscar panel showing the previous
    profile and model (only the panel's own gear button reloaded its picker)."""

    def setUp(self):
        super().setUp()
        _params.save_profiles(_profiles())
        _params.set_active_profile("Alpha")
        self.p._reload_profiles()
        self.window = QtWidgets.QMainWindow()
        self.addCleanup(self.window.deleteLater)
        self.dock = QtWidgets.QDockWidget("Oscar", self.window)
        self.dock.setObjectName(panel.DOCK_OBJECT_NAME)
        self.dock.setWidget(self.p)

    def test_the_open_panel_follows_the_saved_active_profile(self):
        from oscar import commands

        self.assertEqual(self.p.profile_combo.currentText(), "Alpha")
        _params.set_active_profile("Gamma")                       # what the dialog's Save did
        with mock.patch.object(commands.Gui, "getMainWindow", create=True, return_value=self.window):
            self.assertTrue(commands.refresh_panel_profiles())
        self.assertEqual(self.p.profile_combo.currentText(), "Gamma")
        self.assertEqual(self.p.model_label.text(), "m-gamma")

    def test_without_a_panel_nothing_happens(self):
        from oscar import commands

        bare = QtWidgets.QMainWindow()
        self.addCleanup(bare.deleteLater)
        with mock.patch.object(commands.Gui, "getMainWindow", create=True, return_value=bare):
            self.assertFalse(commands.refresh_panel_profiles())
        with mock.patch.object(commands.Gui, "getMainWindow", create=True, return_value=None):
            self.assertFalse(commands.refresh_panel_profiles())   # no main window: never raises

    def test_only_a_saved_dialog_refreshes_the_panel(self):
        from oscar import commands, settings_ui

        for accepted in (1, 0):
            class Dialog:
                def __init__(self, parent=None):
                    pass

                def exec(self):
                    return accepted

            with mock.patch.object(settings_ui, "SettingsDialog", Dialog), \
                    mock.patch.object(commands.Gui, "getMainWindow", create=True, return_value=self.window), \
                    mock.patch.object(commands, "refresh_panel_profiles") as refresh:
                commands.SettingsCommand().Activated()
            self.assertEqual(refresh.call_count, 1 if accepted else 0)


class _SendHarness(FlowTestCase):
    """A panel whose ``_send`` starts a recording fake worker (no thread, no network, no CAD)."""

    def setUp(self):
        super().setUp()
        from test_panel_session import CapturingWorker

        CapturingWorker.instances = []
        self.Worker = CapturingWorker
        engine = mock.Mock()
        engine.begin_turn_snapshot.return_value = dict(GOOD_SNAPSHOT)
        for patcher in (mock.patch.object(panel, "AgentWorker", CapturingWorker),
                        mock.patch.object(_params, "active_profile",
                                          return_value=Profile(name="t", provider="openai_compat", model="m")),
                        mock.patch("oscar.cad.get_engine", return_value=engine)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def other_chat_and_back(self):
        """Open a second chat, then return to the first one (the first now has to be re-activated)."""
        p = self.p
        first = p._current_session()
        p._new_chat()
        p._on_chat_picked(next(i for i, s in enumerate(p.sessions) if s is first))
        self.assertIs(p._current_session(), first)
        return first


class WelcomeBannerTest(_SendHarness):
    """Defect: after switching chats ``self._welcome`` still pointed at the OTHER chat's welcome banner, so the first
    message of this chat did not remove its own banner: it stayed above the conversation."""

    def test_the_first_message_of_a_reopened_empty_chat_removes_its_own_banner(self):
        p = self.p
        self.other_chat_and_back()
        self.assertEqual(len([b for b in p.blocks if b.text == panel.WELCOME]), 1)
        p.input.setPlainText("hello")
        p._send()
        self.assertEqual([b for b in p.blocks if b.text == panel.WELCOME], [])

    def test_a_restored_chat_has_no_banner_to_remove(self):
        p = self.p
        session = chats.new_session("old")
        session["history"] = [panel.Message(role="user", content="a"), panel.Message(role="assistant", content="b")]
        session["blocks"] = None
        p.sessions.insert(0, session)
        p._activate_session(session)
        p.input.setPlainText("again")
        p._send()
        self.assertEqual([b for b in p.blocks if b.text == panel.WELCOME], [])


class EmptyChatPersistenceTest(_SendHarness):
    """Defect: every click on "new chat" saved the chat being left, even an empty one: five clicks left five empty chat
    files, and after 40 of them the real chats fell out of the picker (it lists the newest 40)."""

    def files(self):
        return sorted(n for n in os.listdir(self._tmp) if n.endswith(".json") and not n.endswith(".meta.json"))

    def test_new_chat_clicks_leave_no_empty_chat_files(self):
        p = self.p
        for _ in range(5):
            p._new_chat()
        self.assertEqual(self.files(), [])
        self.assertEqual(len(p.sessions), 1)   # the panel already sits in an unused chat: no duplicate empty chats

    def test_a_chat_is_saved_once_it_has_a_message(self):
        p = self.p
        p.input.setPlainText("hello")
        p._send()
        p._new_chat()
        self.assertEqual(len(self.files()), 1)

    def test_clearing_a_saved_chat_still_reaches_its_file(self):
        p = self.p
        p.input.setPlainText("hello")
        p._send()
        p.worker = None
        session = p._current_session()
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
            p._clear_chat()
        self.assertEqual(chats.load(session["id"])["history"], [])

    def test_a_chat_loaded_from_disk_stays_writable_when_it_is_emptied(self):
        session = chats.new_session("old")
        session["history"] = [panel.Message(role="user", content="a"), panel.Message(role="assistant", content="b")]
        chats.save(session)
        p = panel.OscarPanel()
        self.addCleanup(p.deleteLater)
        loaded = p._current_session()
        self.assertEqual(loaded["id"], session["id"])
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
            p._clear_chat()
        self.assertEqual(chats.load(session["id"])["history"], [])


class PdfAttachmentThreadTest(_SendHarness):
    """Defect: ``_send`` assembled the user message on the GUI thread, and reading an attached PDF there can freeze the
    window for seconds (poppler up to 20 s; the built-in reader decompressed a 0.1 MB PDF for 7 s). A PDF is now
    read on a helper thread while the event loop runs and the panel is locked."""

    def setUp(self):
        super().setUp()
        self.pdf = os.path.join(self.tmp, "sheet.pdf")
        with open(self.pdf, "wb") as f:
            f.write(b"%PDF-1.4\n%fake\n")

    def test_the_pdf_is_read_off_the_gui_thread_and_the_window_stays_responsive(self):
        import threading
        import time

        p = self.p
        seen = {}

        def slow_build(text, attachments):
            seen["thread"] = threading.current_thread().name
            time.sleep(0.4)
            return "ASSEMBLED " + text

        def during():
            seen["busy"] = p._busy()
            seen["input_enabled"] = p.input.isEnabled()
            p.input.setPlainText("a second message")
            p._send()                                                  # re-entrant send: ignored
            seen["workers"] = len(self.Worker.instances)

        QtCore.QTimer.singleShot(100, during)
        p.add_attachments([self.pdf])
        p.input.setPlainText("use the datasheet")
        with mock.patch("oscar.attachments.build_user_message", slow_build):
            p._send()
        self.assertEqual(seen["thread"], "oscar-attachments")
        self.assertTrue(seen["busy"])                                  # the timer ran: the event loop was alive
        self.assertFalse(seen["input_enabled"])
        self.assertEqual(seen["workers"], 0)                           # no second turn started meanwhile
        self.assertEqual(len(self.Worker.instances), 1)
        self.assertEqual(p.history[-1].content, "ASSEMBLED use the datasheet")
        self.assertFalse(p._building)
        self.assertTrue(p.input.isEnabled())
        self.assertTrue(p.send_btn.isEnabled())

    def test_chat_switching_is_refused_while_the_pdf_is_read(self):
        import time

        p = self.p
        first = p._current_session()
        p._new_chat()
        p._on_chat_picked(next(i for i, s in enumerate(p.sessions) if s is first))
        seen = {}

        def slow_build(text, attachments):
            time.sleep(0.3)
            return text

        def during():
            p._on_chat_picked(0)
            p._new_chat()
            seen["same"] = p._current_session() is first

        QtCore.QTimer.singleShot(100, during)
        p.add_attachments([self.pdf])
        p.input.setPlainText("read it")
        with mock.patch("oscar.attachments.build_user_message", slow_build):
            p._send()
        self.assertTrue(seen["same"])
        self.assertIs(p._current_session(), first)

    def test_other_attachments_are_assembled_inline(self):
        import threading

        p = self.p
        names = []
        notes = os.path.join(self.tmp, "notes.txt")
        with open(notes, "w") as f:
            f.write("hello")

        def build(text, attachments):
            names.append(threading.current_thread().name)
            return text

        p.add_attachments([notes])
        p.input.setPlainText("see notes")
        with mock.patch("oscar.attachments.build_user_message", build):
            p._send()
        self.assertEqual(names, [threading.current_thread().name])

    def test_a_failing_assembly_still_sends_the_typed_text_with_a_note(self):
        p = self.p

        def boom(text, attachments):
            raise RuntimeError("poppler exploded")

        p.add_attachments([self.pdf])
        p.input.setPlainText("use the datasheet")
        with mock.patch("oscar.attachments.build_user_message", boom), mock.patch("oscar.turnlog.log_event") as log:
            p._send()
        content = p.history[-1].content
        self.assertTrue(content.startswith("use the datasheet"))
        self.assertIn("sheet.pdf", content)
        self.assertIn("could not be read here", content)
        self.assertEqual(len(self.Worker.instances), 1)
        self.assertTrue(any(c.kwargs.get("where") == "attachments" for c in log.call_args_list))
        self.assertTrue(p.input.isEnabled())


class RevertAndRerunTest(_RevertHarness):
    """Defect: "Revert last turn & re-run parametric build" went on to send the build prompt even when the revert was
    refused or failed, so the new build was made on top of the old geometry (two shells, two lids)."""

    def setUp(self):
        super().setUp()
        self.sent = []
        patcher = mock.patch.object(self.p, "_send", lambda: self.sent.append(self.p.input.toPlainText()))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_verified_revert_is_followed_by_the_build(self):
        self.turn(3)
        self.p._revert_and_rerun_parametric()
        self.assertEqual(self.sent, [self.p.PARAMPT])

    def test_a_failed_revert_does_not_start_the_build(self):
        self.engine.revert = {"success": False, "error": "no undo"}
        self.turn(3)
        self.p._revert_and_rerun_parametric()
        self.assertEqual(self.sent, [])
        self.assertIn("parametric build was not started", self.p.log.toPlainText())

    def test_without_a_revertible_turn_it_builds_fresh(self):
        self.p._revert_and_rerun_parametric()
        self.assertEqual(self.sent, [self.p.PARAMPT])
        self.assertIn("building fresh", self.p.log.toPlainText())

    def test_a_running_turn_blocks_it(self):
        self.begin_turn()
        self.p._revert_and_rerun_parametric()
        self.assertEqual(self.sent, [])




def _permission_box(name, title):
    """A message box shaped like netguard's permission dialog (host and file name that contain the old product name)."""
    box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, title,
                                "Oscar's code wants to download https://wiki.freecad.org/step.zip as FreeCAD.step")
    if name:
        box.setObjectName(name)
    return box


class PermissionDialogTimeoutTest(_SettingsCase):
    """Defect: the caller of ``request_permission`` denies after ``DENY_TIMEOUT_S`` but the dialog itself had no timeout:
    it stayed on screen, modal, long after nobody was waiting for its answer. It now closes with the deny."""

    def setUp(self):
        netguard.reset_turn()
        self.addCleanup(netguard.reset_turn)

    def test_an_unanswered_dialog_closes_itself_with_a_deny(self):
        import time

        from oscar.cad import dispatch

        waits = []
        window = QtWidgets.QMainWindow()
        self.addCleanup(window.deleteLater)
        with mock.patch.object(netguard, "DENY_TIMEOUT_S", 0.2), \
                mock.patch.object(sys.modules["FreeCADGui"], "getMainWindow", create=True, return_value=window), \
                mock.patch.object(dispatch, "dispatch", side_effect=lambda fn, timeout=0, op="", **kw: (fn(), False)), \
                mock.patch.object(_params, "get_bool", return_value=False), \
                mock.patch.object(_params, "set_bool") as set_bool, \
                mock.patch.object(netguard, "on_permission_wait", waits.append):
            started = time.monotonic()
            verdict = netguard.request_permission("Oscar wants to download a file:\n\u2022 https://example.com/a.step")
        self.assertEqual(verdict, "deny")
        self.assertLess(time.monotonic() - started, 5.0)
        set_bool.assert_not_called()                                    # a timeout never turns on "always allow"
        self.assertEqual(waits[0], True)
        self.assertEqual(waits[-1], False)


class SteerHashingTest(FlowTestCase):
    """Defect: every Send while a turn ran called ``attachments.make`` on ALL pending files, which hashes a CAD file or a
    PDF (up to 256 MB) on the GUI thread each time, for files that were going to stay pending anyway."""

    def test_only_images_are_examined_when_a_message_is_steered(self):
        from oscar import attachments
        from test_panel_flow import png

        p = self.p
        self.begin_turn()
        stl = os.path.join(self.tmp, "big.stl")
        pdf = os.path.join(self.tmp, "sheet.pdf")
        for path in (stl, pdf):
            with open(path, "wb") as f:
                f.write(b"x")
        image = png(os.path.join(self.tmp, "a.png"), 16, 16)
        p.add_attachments([stl, pdf, image])
        made = []
        real = attachments.make

        def spy(path):
            made.append(os.path.basename(path))
            return real(path)

        p.input.setPlainText("look at this")
        with mock.patch("oscar.attachments.make", spy):
            p._send()
        self.assertEqual(made, ["a.png"])
        self.assertEqual(p.pending, [stl, pdf])                       # the others wait for the next full turn


class ThumbnailChurnTest(FlowTestCase):
    """Defect: with more step thumbnails than the cache holds (96) the render pass evicted an image that the next pass
    needed again, so the retry loop (``QTimer.singleShot(0, _retry_render)``) re-rendered the whole log forever:
    100 % of a core after about 100 geometry steps in a chat, even with every card collapsed."""

    def snapshot_steps(self, count, prefix="c"):
        from test_panel_flow import png

        shot = png(os.path.join(self.tmp, "shot.png"), 64, 48)
        for n in range(count):
            self.rich_step(call_id=f"{prefix}{n}", ordinal=n + 1, title=f"Step {n}", snapshot_path=shot)

    def settle(self, rounds=400):
        for _ in range(rounds):
            self.app.processEvents()
            if not self.p._thumb_retry:
                return True
        return False

    def test_more_visible_thumbnails_than_the_cache_limit_still_converge(self):
        p = self.p
        self.begin_turn()
        self.snapshot_steps(130)
        self.assertTrue(self.settle(), "the render retry loop never ended")
        card = self.card()
        self.assertTrue(all(p._thumbs.has(s.call_id) for s in card.steps))     # every visible row has its image
        self.assertTrue(all(p.log.has_resource(s.call_id) for s in card.steps[:5] + card.steps[-5:]))

    def test_collapsed_cards_cause_no_render_loop_and_the_cache_stays_bounded(self):
        p = self.p
        for turn in range(3):
            self.begin_turn()
            self.snapshot_steps(40, prefix=f"t{turn}_")
            with mock.patch.object(p, "_fit_view_after_turn"):
                self.finish_turn(True, "ok")
        self.assertTrue(self.settle())
        cards = [b for b in p.blocks if b.kind == Block.TOOL and b.steps]
        self.assertEqual(len(cards), 3)
        self.assertTrue(all(not c.expanded for c in cards))
        p._render()
        self.assertTrue(self.settle())                                          # 120 snapshots, none on screen: nothing to retry
        self.assertLessEqual(len(p._thumbs.keys()), p._thumbs.limit)

    def test_expanding_a_card_loads_its_thumbnails_and_converges(self):
        p = self.p
        for turn in range(3):
            self.begin_turn()
            self.snapshot_steps(40, prefix=f"t{turn}_")
            with mock.patch.object(p, "_fit_view_after_turn"):
                self.finish_turn(True, "ok")
        cards = [b for b in p.blocks if b.kind == Block.TOOL and b.steps]
        p._on_anchor_target(f"tool:{cards[1].bid}")
        self.assertTrue(self.settle())
        self.assertTrue(all(p._thumbs.has(s.call_id) for s in cards[1].steps))
        self.assertTrue(all(p.log.has_resource(s.call_id) for s in cards[1].steps))

    def test_the_cache_never_evicts_a_protected_image_and_otherwise_stays_bounded(self):
        from oscar.panel_blocks import ThumbCache
        from test_panel_flow import png

        path = png(os.path.join(self.tmp, "t.png"), 40, 30)
        cache = ThumbCache(limit=3)
        for n in range(6):
            cache.load(f"k{n}", path)
        self.assertEqual(len(cache.keys()), 3)                                  # plain LRU: bounded
        self.assertEqual(cache.keys(), ["k3", "k4", "k5"])
        keep = {f"p{n}" for n in range(6)}
        for n in range(6):
            cache.load(f"p{n}", path, keep=keep)
        self.assertTrue(keep <= set(cache.keys()))                              # all six protected images are still there
        self.assertLessEqual(len(cache.keys()), 3 + 6)


class ShowPanelTest(_SettingsCase):
    """``show_panel`` creates the dock once and raises the same one afterwards (the command, the toolbar button and the
    keyboard shortcut all call it)."""

    def test_the_dock_is_created_once_and_reused(self):
        from oscar.cad import dispatch

        self.addCleanup(setattr, netguard, "on_permission_wait", netguard.on_permission_wait)   # the panel installs its hook
        window = QtWidgets.QMainWindow()
        self.addCleanup(window.deleteLater)
        with mock.patch.object(panel.Gui, "getMainWindow", create=True, return_value=window), \
                mock.patch.object(dispatch, "init") as init, \
                mock.patch.object(chats, "chats_dir", lambda: tempfile.mkdtemp(prefix="oscar_dock_test_")):
            panel.show_panel()
            docks = window.findChildren(QtWidgets.QDockWidget, panel.DOCK_OBJECT_NAME)
            self.assertEqual(len(docks), 1)
            first = docks[0].widget()
            self.assertIsInstance(first, panel.OscarPanel)
            panel.show_panel()
            again = window.findChildren(QtWidgets.QDockWidget, panel.DOCK_OBJECT_NAME)
        self.assertEqual(len(again), 1)
        self.assertIs(again[0].widget(), first)
        self.assertEqual(init.call_count, 2)                          # the GUI-thread dispatcher is armed every time
        self.assertGreaterEqual(again[0].minimumWidth(), 280)
        first.deleteLater()


if __name__ == "__main__":
    unittest.main()
