"""Tests of the panel's own logic (oscar/panel.py).

They construct a real ``OscarPanel`` on an offscreen QApplication (no window is shown to the user, no
agent is started, no network: workers are small fakes). Skipped when PySide / FreeCAD are unavailable
(plain app python); the pure renderer is covered by test_panel_render.py.
"""

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
    from PySide import QtCore, QtGui, QtWidgets
    from oscar import chats, netguard, panel
    from oscar.panel import Block, Step
except Exception as exc:   # pragma: no cover - plain interpreter without FreeCAD / Qt
    _IMPORT_ERROR = str(exc)

GOOD_SNAPSHOT = {"success": True, "docs": {"Doc": 2}, "max_undo": {"Doc": 20}, "undo_mode": {"Doc": 1}}


class FakeWorker:
    """The slice of AgentWorker the panel touches."""

    def __init__(self, steer=True, unsent=()):
        self.calls = []
        self._unsent = list(unsent)
        if not steer:
            self.steer = None

    def steer(self, text, note="", images=()):
        self.calls.append(("steer", text))

    def drain_unsent(self):
        out, self._unsent = self._unsent, []
        return out

    def stop(self):
        self.calls.append(("stop",))

    def resume(self):
        self.calls.append(("resume",))

    def pause(self):
        self.calls.append(("pause",))

    def wait(self):
        return True

    def deleteLater(self):
        pass


class FakeEngine:
    """cad.get_engine() stand-in: scripted revert_turn / begin_turn_snapshot answers."""

    def __init__(self, revert=None, after=None):
        self.revert = revert if revert is not None else {"success": True, "undone": {"Doc": 2}, "closed_docs": []}
        self.after = after if after is not None else {"success": True, "docs": {"Doc": 0}}
        self.reverts = []

    def revert_turn(self, snapshot):
        self.reverts.append(snapshot)
        return self.revert

    def begin_turn_snapshot(self):
        return self.after


class PanelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERROR:
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # never open a window on the user's screen
            cls.app = QtWidgets.QApplication(sys.argv[:1])

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="oscar_panel_test_")
        patcher = mock.patch.object(chats, "chats_dir", lambda: self._tmp)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._hook = netguard.on_permission_wait
        self.addCleanup(setattr, netguard, "on_permission_wait", self._hook)
        self.p = panel.OscarPanel()
        self.addCleanup(self._dispose)

    def _dispose(self):
        self.p._status_timer.stop()
        self.p._render_timer.stop()
        self.p.worker = None
        self.p.deleteLater()

    def plain(self) -> str:
        return self.p.log.toPlainText()

    def finish_turn(self, ok=True, text="Done."):
        self.p._on_turn_finished(ok, text)


class StatusLineTest(PanelTestCase):
    def test_status_is_visible_under_a_live_card(self):
        p = self.p
        p.worker = FakeWorker()
        card = p._ensure_card()
        card.steps.append(Step("execute_code", summary="execute_code …"))
        p._on_netguard_wait(True)
        p._render()
        self.assertIn("Waiting for your approval", self.plain())   # the old code swallowed it once a card existed
        self.assertIn("Working…", self.plain())                    # the card header still counts
        p._on_netguard_wait(False)
        self.assertIn("Working…", p._status)

    def test_status_is_sticky_and_cleared_at_turn_end(self):
        p = self.p
        p.worker = FakeWorker()
        p._ensure_card()
        p._set_status("Running execute_code")
        p._render()
        self.assertIn("Running execute_code", self.plain())
        p.worker = None
        self.finish_turn()
        self.assertEqual(p._status, "")
        self.assertNotIn("Running execute_code", self.plain())

    def test_generic_status_yields_to_streamed_text_but_specific_status_stays(self):
        p = self.p
        p.worker = FakeWorker()
        p._set_status("Thinking")
        p._on_text_delta("Hello")
        self.assertEqual(p._status, "")
        p._set_status("Waiting for your approval — see the Oscar permission dialog…")
        p._on_text_delta(" more")
        self.assertIn("Waiting for your approval", p._status)

    def test_header_is_never_done_while_the_turn_runs(self):
        p = self.p
        p.worker = FakeWorker()
        p._on_tool_started("create_object")
        p._on_tool_finished("create_object(Box)", True, "made Box")
        p._render()
        self.assertIn("Working…", self.plain())    # the step finished but the worker is still running
        self.assertNotIn("✻ Done", self.plain())
        p.worker = None
        self.finish_turn()
        self.assertNotIn("Working…", self.plain())
        self.assertIn("✻ Ran 1 step", self.plain())


class ScrollTest(PanelTestCase):
    def fill(self, count=60):
        p = self.p
        p.resize(500, 400)
        p.show()
        self.app.processEvents()
        for i in range(count):
            p._append(Block.ASSISTANT, f"message {i}\n\n" + "line\n" * 3)
        p._render()
        self.app.processEvents()
        return p.log.verticalScrollBar()

    def test_position_is_restored_when_the_user_scrolled_up(self):
        bar = self.fill()
        self.assertGreater(bar.maximum(), 300)
        bar.setValue(300)
        self.p._render()
        self.assertEqual(bar.value(), 300)
        self.p._append(Block.ASSISTANT, "new message")
        self.p._render()
        self.assertEqual(bar.value(), 300)

    def test_a_deep_position_survives_the_lazy_layout_of_a_long_log(self):
        bar = self.fill(150)
        target = bar.maximum() * 2 // 3
        bar.setValue(target)
        self.p._append(Block.ASSISTANT, "tail\n" * 20)
        self.p._render()
        self.app.processEvents()      # Qt finishes laying the document out; the range grows after setHtml
        self.assertEqual(bar.value(), target)

    def test_a_user_scroll_during_the_layout_is_not_overridden(self):
        bar = self.fill()
        bar.setValue(bar.maximum() // 2)
        self.p._render()
        bar.setValue(10)              # the user moves away before the range settles
        self.p._on_range_changed(0, bar.maximum() + 5000)
        self.assertEqual(bar.value(), 10)
        self.assertEqual(self.p._scroll_mode, "")
        bar.setValue(bar.maximum())   # pinned, then the user scrolls up before the range grows
        self.p._append(Block.ASSISTANT, "more")
        self.p._render()
        self.assertEqual(self.p._scroll_mode, "follow")
        bar.setValue(bar.maximum() // 3)
        self.p._on_range_changed(0, bar.maximum() + 5000)
        self.assertEqual(bar.value(), bar.maximum() // 3)
        self.assertEqual(self.p._scroll_mode, "")

    def test_a_pinned_view_follows_the_bottom(self):
        bar = self.fill()
        bar.setValue(bar.maximum())
        self.p._append(Block.ASSISTANT, "x\n" * 10)
        self.p._render()
        self.app.processEvents()
        self.assertGreaterEqual(bar.value(), bar.maximum() - 8)

    def test_settled_blocks_are_cached_and_live_ones_are_not(self):
        p = self.p
        calls = []
        original = Block.html

        def counting(block):
            calls.append(block.bid)
            return original(block)

        old = p._append(Block.ASSISTANT, "an old answer")
        with mock.patch.object(Block, "html", counting):
            p._render()
            first = calls.count(old.bid)
            p._render()
            self.assertEqual(calls.count(old.bid), first)           # unchanged settled block: served from the cache
            old.text = "an edited answer"
            p._render()
            self.assertEqual(calls.count(old.bid), first + 1)       # changed: rebuilt
            card = p._ensure_card()                                  # a live card has a running clock
            before = calls.count(card.bid)
            p._render()
            p._render()
            self.assertEqual(calls.count(card.bid), before + 2)
        self.assertIn("an edited answer", self.plain())

    def test_cache_does_not_leak_removed_blocks(self):
        p = self.p
        block = p._append(Block.ASSISTANT, "temporary")
        p._render()
        self.assertIn(block.bid, p._frag_cache)
        p.blocks.remove(block)
        p._render()
        self.assertNotIn(block.bid, p._frag_cache)


class PendingNotesTest(PanelTestCase):
    def test_notes_are_kept_per_session(self):
        p = self.p
        first = p._current_session()
        first["history"].append(panel.Message(role="user", content="hi"))   # "new chat" is refused in an unused chat
        p._pending_notes.append("note for chat A")
        self.assertEqual(p._pending_notes, ["note for chat A"])
        p._new_chat()
        second = p._current_session()
        self.assertIsNot(first, second)
        self.assertEqual(p._pending_notes, [])        # the revert note of chat A never leaks into chat B
        p._pending_notes.append("note for chat B")
        p._activate_session(first)
        self.assertEqual(p._pending_notes, ["note for chat A"])
        p._activate_session(second)
        self.assertEqual(p._pending_notes, ["note for chat B"])

    def test_setter_replaces_the_list(self):
        p = self.p
        p._pending_notes = ["x", "y"]
        self.assertEqual(p._pending_notes, ["x", "y"])
        p._pending_notes.clear()
        self.assertEqual(p._pending_notes, [])

    def test_notes_are_not_persisted_in_the_chat_file(self):
        p = self.p
        p._pending_notes.append("transient")
        session = p._current_session()
        session["history"].append(panel.Message(role="user", content="hi"))
        p._persist_current()
        saved = chats.load(session["id"])
        self.assertIsNotNone(saved)
        self.assertNotIn("transient", str(saved))


class SteerWhileBusyTest(PanelTestCase):
    def test_send_while_busy_steers_and_tags_the_bubble(self):
        p = self.p
        worker = FakeWorker()
        p.worker = worker
        p.input.setPlainText("also add vent slots")
        p._send()
        self.assertEqual(worker.calls, [("steer", "also add vent slots")])
        self.assertEqual(p.input.toPlainText(), "")
        bubble = p.blocks[-1]
        self.assertEqual((bubble.kind, bubble.text, bubble.queued), (Block.USER, "also add vent slots", True))
        self.assertEqual(p._queued_blocks, [bubble])
        self.assertIn("queued", self.plain())
        self.assertEqual(len(p.history), 0)   # the worker, not the panel, appends steered text to the history

    def test_the_workers_queued_echo_does_not_add_a_second_notice(self):
        p = self.p
        before = len(p.blocks)
        p._on_notice("Message queued; Oscar will see it at the next step")
        self.assertEqual(len(p.blocks), before)
        p._on_notice("Something else happened")
        self.assertEqual(len(p.blocks), before + 1)

    def test_empty_input_does_nothing(self):
        p = self.p
        worker = FakeWorker()
        p.worker = worker
        p.input.setPlainText("   ")
        p._send()
        self.assertEqual(worker.calls, [])
        self.assertEqual(p._queued_blocks, [])

    def test_worker_without_steer_keeps_the_text_and_says_what_to_do(self):
        p = self.p
        p.worker = FakeWorker(steer=False)
        p.input.setPlainText("hello")
        p._send()
        self.assertEqual(p.input.toPlainText(), "hello")
        self.assertIn("press Esc to stop", p._status)
        self.assertEqual([b for b in p.blocks if b.kind == Block.USER], [])

    def test_steer_failure_keeps_the_text(self):
        p = self.p
        worker = FakeWorker()
        worker.steer = mock.Mock(side_effect=RuntimeError("queue closed"))
        p.worker = worker
        p.input.setPlainText("hello")
        p._send()
        self.assertEqual(p.input.toPlainText(), "hello")
        self.assertIn("queue closed", p._status)

    def test_pending_attachments_stay_pending(self):
        p = self.p
        p.worker = FakeWorker()
        p.pending.append("/tmp/a.png")
        p.input.setPlainText("see this")
        p._send()
        self.assertEqual(p.pending, ["/tmp/a.png"])
        self.assertIn("Attachments", p._status)

    def test_delivered_messages_lose_the_queued_tag_at_turn_end(self):
        p = self.p
        p.worker = FakeWorker()
        p.input.setPlainText("more vents")
        p._send()
        bubble = p._queued_blocks[0]
        self.finish_turn()
        self.assertFalse(bubble.queued)
        self.assertIn(bubble, p.blocks)
        self.assertEqual(p._queued_blocks, [])

    def test_undelivered_message_is_resent_after_a_normal_finish(self):
        p = self.p
        p.worker = FakeWorker(unsent=[{"text": "too late", "note": "", "images": []}])
        p.input.setPlainText("too late")
        p._send()
        bubble = p._queued_blocks[0]
        sent = []
        p._send_text = lambda text: sent.append(text)
        self.finish_turn(ok=True)
        self.assertNotIn(bubble, p.blocks)        # the re-send shows it again: no duplicate bubble
        self.app.processEvents()
        self.assertEqual(sent, ["too late"])

    def test_undelivered_message_returns_to_the_input_after_a_stop(self):
        p = self.p
        p.worker = FakeWorker(unsent=[{"text": "too late", "note": "", "images": []}])
        p.input.setPlainText("too late")
        p._send()
        sent = []
        p._send_text = lambda text: sent.append(text)
        self.finish_turn(ok=False, text="Stopped.")
        self.app.processEvents()
        self.assertEqual(sent, [])                      # a stop never starts new work by itself
        self.assertEqual(p.input.toPlainText(), "too late")
        self.assertIn("not delivered", self.plain())

    def test_send_text_keeps_what_the_user_is_typing(self):
        p = self.p
        p.input.setPlainText("half-typed thought")
        seen = []
        with mock.patch.object(p, "_send", lambda: (seen.append(p.input.toPlainText()), p.input.clear())):
            p._send_text("queued text")
        self.assertEqual(seen, ["queued text"])
        self.assertEqual(p.input.toPlainText(), "half-typed thought")
        p.worker = FakeWorker()
        p._send_text("ignored: a turn is running")
        self.assertEqual(seen, ["queued text"])


class StopTest(PanelTestCase):
    def test_widgets_and_object_names(self):
        p = self.p
        self.assertEqual(p.log.objectName(), "oscarLog")
        self.assertEqual(p.stop_btn.objectName(), "oscarStopButton")
        self.assertFalse(p.log.openLinks())
        self.assertFalse(p.log.openExternalLinks())

    def test_stop_button_and_esc_shortcut_follow_the_busy_state(self):
        p = self.p
        p.show()
        self.app.processEvents()
        self.assertTrue(p.stop_btn.isHidden())
        self.assertFalse(p._esc_shortcut.isEnabled())
        p._set_busy(True)
        self.assertFalse(p.stop_btn.isHidden())
        self.assertTrue(p._esc_shortcut.isEnabled())
        self.assertEqual(p._esc_shortcut.context(), QtCore.Qt.WidgetWithChildrenShortcut)
        p._set_busy(False)
        self.assertTrue(p.stop_btn.isHidden())
        self.assertFalse(p._esc_shortcut.isEnabled())

    def test_stop_button_and_esc_stop_the_worker(self):
        p = self.p
        worker = FakeWorker()
        p.worker = worker
        p.stop_btn.click()
        self.assertEqual(worker.calls, [("stop",)])
        p._paused = True
        p._esc_shortcut.activated.emit()
        self.assertEqual(worker.calls, [("stop",), ("resume",), ("stop",)])   # a paused flow is resumed so it can stop



class DeleteChatTest(PanelTestCase):
    def test_deleting_a_chat_asks_first(self):
        p = self.p
        p._current_session()["history"].append(panel.Message(role="user", content="hi"))   # "new chat" is refused in an unused chat
        p._new_chat()
        self.assertEqual(len(p.sessions), 2)
        victim = p._current_session()
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.No) as ask:
            p._delete_chat()
        ask.assert_called_once()
        self.assertEqual(len(p.sessions), 2)
        self.assertIn(victim, p.sessions)
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
            p._delete_chat()
        self.assertEqual(len(p.sessions), 1)
        self.assertNotIn(victim, p.sessions)

    def test_the_last_chat_is_never_deleted_and_nothing_is_asked(self):
        p = self.p
        with mock.patch.object(QtWidgets.QMessageBox, "question") as ask:
            p._delete_chat()
        ask.assert_not_called()
        self.assertEqual(len(p.sessions), 1)

    def test_busy_panel_does_not_delete(self):
        p = self.p
        p._current_session()["history"].append(panel.Message(role="user", content="hi"))   # "new chat" is refused in an unused chat
        p._new_chat()
        p.worker = FakeWorker()
        with mock.patch.object(QtWidgets.QMessageBox, "question") as ask:
            p._delete_chat()
        ask.assert_not_called()
        self.assertEqual(len(p.sessions), 2)


class LinkTest(PanelTestCase):
    def click(self, target):
        with mock.patch.object(QtGui.QDesktopServices, "openUrl", return_value=True) as opener:
            self.p._on_anchor(QtCore.QUrl(target))
        return [call.args[0].toString() for call in opener.call_args_list]

    def test_web_and_mail_links_open_with_the_desktop(self):
        self.assertEqual(self.click("https://example.com/docs?a=1"), ["https://example.com/docs?a=1"])
        self.assertEqual(self.click("http://example.com/"), ["http://example.com/"])
        self.assertEqual(self.click("mailto:me@example.com"), ["mailto:me@example.com"])

    def test_file_links_open_documents_and_reveal_scripts(self):
        folder = tempfile.mkdtemp(prefix="oscar_links_")
        image = os.path.join(folder, "shell.png")
        script = os.path.join(folder, "rebuild.py")
        for path in (image, script):
            open(path, "w").close()
        self.assertEqual(self.click(QtCore.QUrl.fromLocalFile(image).toString()), [QtCore.QUrl.fromLocalFile(image).toString()])
        self.assertEqual(self.click(QtCore.QUrl.fromLocalFile(folder).toString()), [QtCore.QUrl.fromLocalFile(folder).toString()])
        # a script is never launched: its folder is shown instead
        self.assertEqual(self.click(QtCore.QUrl.fromLocalFile(script).toString()), [QtCore.QUrl.fromLocalFile(folder).toString()])

    def test_missing_file_is_reported_not_opened(self):
        missing = QtCore.QUrl.fromLocalFile("/definitely/not/here.png").toString()
        self.assertEqual(self.click(missing), [])
        self.assertIn("File not found", self.plain())

    def test_failed_open_is_reported(self):
        with mock.patch.object(QtGui.QDesktopServices, "openUrl", return_value=False):
            self.p._on_anchor(QtCore.QUrl("https://example.com/x"))
        self.assertIn("Could not open", self.plain())

    def test_unknown_schemes_and_legacy_anchors_are_unchanged(self):
        self.assertEqual(self.click("javascript:alert(1)"), [])
        self.assertEqual(self.click("ftp://example.com/x"), [])
        p = self.p
        card = p._ensure_card()
        card.steps.append(Step("t", summary="t()", ok=True, dur=1.0, detail="d"))
        card.t0 = 0.0
        card.expanded = False
        self.click(f"tool:{card.bid}")
        self.assertTrue(card.expanded)
        self.click(f"step:{card.bid}:0")
        self.assertTrue(card.steps[0].expanded)
        self.click(f"step:{card.bid}:9")      # out of range: ignored
        self.click(f"step:{card.bid}:x")      # not a number: ignored


class RevertGuardTest(PanelTestCase):
    def card(self, snapshot=None, revertible=True):
        p = self.p
        block = p._append(Block.TOOL, "✻ Ran 1 step · 1.0s")
        block.steps = [Step("execute_code", summary="execute_code()", ok=True, dur=1.0)]
        block.snapshot = dict(GOOD_SNAPSHOT) if snapshot is None else snapshot
        block.revertible = revertible
        return block

    def revert(self, block, engine):
        with mock.patch("oscar.cad.get_engine", return_value=engine):
            self.p._revert_turn(block)

    def test_success_marks_reverted_after_verification_and_adds_the_note(self):
        p = self.p
        block = self.card()
        engine = FakeEngine()
        self.revert(block, engine)
        self.assertTrue(block.reverted)
        self.assertEqual(engine.reverts, [GOOD_SNAPSHOT])
        self.assertIn("Turn reverted", p.blocks[-1].text)
        self.assertTrue(any("I reverted every CAD change" in note for note in p._pending_notes))

    def test_a_document_the_engine_kept_open_is_not_revert_residue(self):
        """Defect (reconcile with main's revert fix): revert_turn deliberately leaves a document open when the
        user changed it after the turn (``kept_docs``); the panel used to call that an incomplete revert."""
        block = self.card()
        engine = FakeEngine(
            revert={"success": True, "undone": {"Doc": 2}, "closed_docs": [],
                    "kept_docs": [{"doc": "Mine", "reason": "you edited it after the turn"}]},
            after={"success": True, "docs": {"Doc": 0, "Mine": 3}})
        self.revert(block, engine)
        self.assertTrue(block.reverted)
        self.assertIn("left open because you changed them since: Mine", self.p.blocks[-1].text)
        self.assertTrue(any("documents kept open" in note for note in self.p._pending_notes))

    def test_an_unlisted_extra_document_is_still_revert_residue(self):
        block = self.card()
        engine = FakeEngine(after={"success": True, "docs": {"Doc": 0, "Stray": 1}})
        self.revert(block, engine)
        self.assertFalse(block.reverted)
        self.assertIn("Revert incomplete", self.p.blocks[-1].text)

    def test_missing_or_failed_snapshot_is_refused(self):
        for snapshot in ({}, {"success": False, "docs": {}}, {"docs": None}, {"success": True}):
            p = self.p
            p._pending_notes.clear()
            block = self.card(snapshot=snapshot)
            engine = FakeEngine()
            self.revert(block, engine)
            self.assertEqual(engine.reverts, [], snapshot)
            self.assertFalse(block.reverted, snapshot)
            self.assertFalse(block.revertible, snapshot)    # no link that can only refuse
            self.assertIn("starting state was not recorded", p.blocks[-1].text)
            self.assertEqual(p._pending_notes, [])

    def test_an_empty_but_successful_snapshot_is_valid(self):
        block = self.card(snapshot={"success": True, "docs": {}})
        engine = FakeEngine(revert={"success": True, "undone": {}, "closed_docs": ["Created"]},
                            after={"success": True, "docs": {}})
        self.revert(block, engine)
        self.assertTrue(block.reverted)   # the turn created a document and the revert closed it

    def test_only_the_latest_non_reverted_turn_can_be_reverted(self):
        p = self.p
        older, newer = self.card(), self.card()
        p._update_revert_flags()
        self.assertEqual((older.revert_enabled, newer.revert_enabled), (False, True))
        self.assertNotIn("revert:", older.html())
        self.assertIn(f"revert:{newer.bid}", newer.html())
        engine = FakeEngine()
        self.revert(older, engine)
        self.assertEqual(engine.reverts, [])
        self.assertFalse(older.reverted)
        self.assertIn("Only the latest turn", p.blocks[-1].text)
        # after reverting the newest, the older one becomes the latest
        self.revert(newer, FakeEngine())
        self.assertTrue(newer.reverted)
        self.assertEqual((older.revert_enabled, newer.revert_enabled), (True, False))

    def test_anchor_of_a_disabled_card_does_nothing(self):
        p = self.p
        older, newer = self.card(), self.card()
        p._update_revert_flags()
        engine = FakeEngine()
        with mock.patch("oscar.cad.get_engine", return_value=engine):
            p._on_anchor(QtCore.QUrl(f"revert:{older.bid}"))
            self.assertEqual(engine.reverts, [])
            p._on_anchor(QtCore.QUrl(f"revert:{newer.bid}"))
        self.assertEqual(len(engine.reverts), 1)
        self.assertTrue(newer.reverted)

    def test_failures_never_report_reverted(self):
        p = self.p
        cases = [
            (FakeEngine(revert={"success": False, "error": "document busy"}), "Revert failed: document busy", True),
            (FakeEngine(revert={"success": True, "undone": {}, "closed_docs": []}), "Nothing to undo", False),
            (FakeEngine(revert={"success": False, "incomplete": True, "error": "the undo stack cannot hold this turn"}),
             "Revert refused", False),
            (FakeEngine(after={"success": True, "docs": {"Doc": 3}}), "Revert incomplete: Doc", True),
            (FakeEngine(after={"success": True, "docs": {"Doc": 0, "Leftover": 1}}), "Revert incomplete: Leftover", True),
        ]
        for engine, text, still_offered in cases:
            block = self.card()
            self.revert(block, engine)
            self.assertFalse(block.reverted, text)
            self.assertIn(text, p.blocks[-1].text)
            self.assertEqual(block.revertible, still_offered, text)
            block.revertible = False   # keep the next case's card the latest
        self.assertEqual(p._pending_notes, [])

    def test_engine_exception_is_a_failure_not_a_crash(self):
        p = self.p
        block = self.card()
        engine = FakeEngine()
        engine.revert_turn = mock.Mock(side_effect=RuntimeError("gui gone"))
        self.revert(block, engine)
        self.assertFalse(block.reverted)
        self.assertIn("Revert failed: gui gone", p.blocks[-1].text)

    def test_verification_that_cannot_run_trusts_the_engine(self):
        block = self.card()
        engine = FakeEngine()
        engine.begin_turn_snapshot = mock.Mock(side_effect=RuntimeError("no gui"))
        self.revert(block, engine)
        self.assertTrue(block.reverted)

    def test_never_mid_flight_and_never_twice(self):
        p = self.p
        block = self.card()
        engine = FakeEngine()
        p.worker = FakeWorker()
        self.revert(block, engine)
        self.assertEqual(engine.reverts, [])
        p.worker = None
        self.revert(block, engine)
        self.assertTrue(block.reverted)
        self.revert(block, engine)
        self.assertEqual(len(engine.reverts), 1)
        not_revertible = self.card(revertible=False)
        self.revert(not_revertible, engine)
        self.assertEqual(len(engine.reverts), 1)

    def test_reverts_are_disabled_while_a_turn_runs(self):
        p = self.p
        block = self.card()
        p._update_revert_flags()
        self.assertTrue(block.revert_enabled)
        with mock.patch.object(panel, "AgentWorker", side_effect=RuntimeError("no agent in tests")):
            p._set_busy(True)
            for b in p.blocks:
                b.revert_enabled = False   # what _send does right after _set_busy
        self.assertNotIn("revert:", block.html())


class RevertibleCardTest(PanelTestCase):
    def run_turn(self, tool, ok, snapshot=None):
        p = self.p
        p._turn_snapshot = dict(GOOD_SNAPSHOT) if snapshot is None else snapshot
        p._turn_mutated = False
        p.worker = FakeWorker()
        p._on_tool_started(tool)
        p._on_tool_finished(f"{tool}()", ok, "result")
        p.worker = None
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn()
        return next(b for b in reversed(p.blocks) if b.kind == Block.TOOL and b.steps)

    def test_read_only_turn_offers_no_revert(self):
        card = self.run_turn("get_objects", True)
        self.assertFalse(card.revertible)
        self.assertNotIn("revert:", card.html())

    def test_mutating_success_offers_revert(self):
        card = self.run_turn("execute_code", True)
        self.assertTrue(card.revertible)
        self.assertTrue(card.revert_enabled)
        self.assertIn(f"revert:{card.bid}", card.html())

    def test_failed_mutating_call_offers_no_revert(self):
        self.assertFalse(self.run_turn("execute_code", False).revertible)

    def test_unusable_snapshot_offers_no_revert(self):
        self.assertFalse(self.run_turn("execute_code", True, snapshot={}).revertible)
        p = self.p
        p.blocks.clear()
        self.assertFalse(self.run_turn("execute_code", True, snapshot={"success": False, "docs": {}}).revertible)

    def test_mutating_names_cover_the_historical_list(self):
        for name in ("create_object", "edit_object", "delete_object", "import_file", "insert_part_from_library",
                     "execute_code", "apply_feature_plan", "reapply_plan_feature", "run_fem_analysis"):
            self.assertIn(name, panel.OscarPanel._MUTATING_TOOLS)

    def test_registry_names_are_added_when_the_registry_offers_them(self):
        from oscar import tools

        with mock.patch.object(tools, "mutating_names", lambda: ["build_enclosure"], create=True):
            self.assertIn("build_enclosure", panel._mutating_tool_names())
        with mock.patch.object(tools, "mutating_names", mock.Mock(side_effect=RuntimeError("x")), create=True):
            self.assertIn("execute_code", panel._mutating_tool_names())   # a broken registry never empties the set

    def test_the_next_turn_takes_over_the_revert_link(self):
        first = self.run_turn("execute_code", True)
        second = self.run_turn("create_object", True)
        self.assertFalse(first.revert_enabled)
        self.assertTrue(second.revert_enabled)


class LegacyRenderTest(PanelTestCase):
    def test_answer_text_uses_the_shared_markdown(self):
        p = self.p
        p._append(Block.ASSISTANT, "The box is 2*3*4 mm. See https://example.com/x")
        p._render()
        html = p.log.toHtml()
        self.assertIn("2*3*4 mm", p.log.toPlainText())
        self.assertNotIn("<i>3</i>", html)
        self.assertIn("https://example.com/x", p.log.toPlainText())

    def test_user_bubble_queued_tag_renders(self):
        p = self.p
        bubble = p._append(Block.USER, "more vents")
        bubble.queued = True
        p._render()
        self.assertIn("queued", self.plain())




if __name__ == "__main__":
    unittest.main()
