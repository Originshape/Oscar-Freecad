"""Regression tests for the panel review fixes (panel-a-*, panel-b-*).

The chat panel (oscar/panel.py), the runtime UI renamer (oscar/ui_rebrand.py)
and the application commands (oscar/commands.py). Runs in the console
interpreter: no real window is ever shown (widgets that need layout carry
WA_DontShowOnScreen, and the platform plugin is forced to ``offscreen`` when
this module creates the QApplication), no network.
"""

import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

# Adapted: the suite's isolation header (test_suite_guard NO_ISOLATION) - temp settings / projects / logs / cache /
# downloads BEFORE importing oscar. The original per-module temp settings file is only a fallback now.
isolation.isolate_process()
os.environ.setdefault(
    "OSCAR_AI_SETTINGS",
    os.path.join(tempfile.mkdtemp(prefix="oscar_fix_panel_"), "settings.json"),
)

if "FreeCADGui" not in sys.modules:
    try:
        import FreeCADGui  # noqa: F401
    except Exception:
        sys.modules["FreeCADGui"] = types.ModuleType("FreeCADGui")

try:
    from PySide import QtCore, QtGui, QtWidgets
    from oscar import panel as panel_mod
except Exception:  # pragma: no cover - GUI-less interpreter
    QtCore = QtGui = QtWidgets = panel_mod = None


def _qapp():
    if QtWidgets is None:
        raise unittest.SkipTest("PySide unavailable")
    app = QtWidgets.QApplication.instance()
    if app is None:
        # a stray test must never open a real window on the desktop
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication(getattr(sys, "argv", ["tests"])[:1])
    return app


def _make_panel():
    """A real OscarPanel that sees no persisted chats and never shows."""
    from oscar import chats

    with mock.patch.object(chats, "list_sessions", return_value=[]):
        panel = panel_mod.OscarPanel()
    # the panel installs a hook on this module global; do not leak it
    from oscar import netguard

    netguard.on_permission_wait = None
    # layout needs a visible widget; this one is never put on screen
    panel.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
    panel.resize(420, 520)
    panel.show()
    return panel


class _PanelCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _qapp()


# ---------------------------------------------------------------------------
# panel-b-03: the chat log must keep its scroll position across re-renders


class ScrollTargetHelperTest(_PanelCase):
    def test_at_bottom_follows_the_end(self):
        self.assertIsNone(panel_mod._scroll_target_for(1000, 1000))
        self.assertIsNone(panel_mod._scroll_target_for(995, 1000))   # within tol
        self.assertIsNone(panel_mod._scroll_target_for(0, 0))        # empty log

    def test_scrolled_up_keeps_the_pixel_value(self):
        self.assertEqual(panel_mod._scroll_target_for(400, 1000), 400)
        self.assertEqual(panel_mod._scroll_target_for(0, 1000), 0)


class ScrollKeepTest(_PanelCase):
    def setUp(self):
        self.panel = _make_panel()
        self.bar = self.panel.log.verticalScrollBar()
        # A chat arrives as a new block list (that is what switching or
        # restoring a chat does); appending to the list the panel was
        # constructed with is not a path the app takes.
        self.panel.blocks = [panel_mod.Block(
            panel_mod.Block.ASSISTANT,
            f"**Message {i}**\n\n- bullet one\n- bullet two\n\nSome longer "
            "paragraph text that wraps over a few lines in a narrow dock "
            "panel so that it takes real vertical space.")
            for i in range(40)]
        self.panel._render()
        self._settle()

    def _state(self):
        return (f"value={self.bar.value()} max={self.bar.maximum()} "
                f"target={self.panel._scroll_target!r} "
                f"viewport_h={self.panel.log.viewport().height()} "
                f"doc_h={self.panel.log.document().size().height():.0f}")

    def tearDown(self):
        self.panel.hide()
        self.panel.deleteLater()

    def _settle(self):
        # Qt lays text out lazily, and how long that takes depends on the
        # platform's fonts (slower on Windows): wait until the scroll range
        # has stopped moving instead of pumping a fixed number of times.
        last, stable = None, 0
        for _ in range(200):
            self.app.processEvents()
            now = (self.bar.maximum(), self.bar.value())
            stable = stable + 1 if now == last else 0
            last = now
            if stable >= 3:
                break
            time.sleep(0.005)

    def _at_end(self):
        return self.bar.value() >= self.bar.maximum() - 8

    def test_follow_survives_streaming_renders(self):
        self.assertTrue(self._at_end(),
                        "initial render should sit at the end: " + self._state())
        block = self.panel._append(panel_mod.Block.ASSISTANT, "")
        for n in range(30):
            block.text += f"streamed line {n} of the answer with some words\n\n"
            self.panel._render()
            self._settle()
            self.assertTrue(self._at_end(),
                            f"lost the end at chunk {n}: "
                            f"{self.bar.value()}/{self.bar.maximum()}")

    def test_user_scroll_position_is_kept_across_renders(self):
        kept = self.bar.maximum() // 2
        self.bar.setValue(kept)           # the user scrolls up
        self._settle()
        for _ in range(8):
            self.panel._render()
            self._settle()
            self.assertEqual(self.bar.value(), kept)

    def test_switching_to_another_chat_starts_at_its_end(self):
        self.bar.setValue(self.bar.maximum() // 2)   # mid-scroll in chat A
        self._settle()
        other = [panel_mod.Block(panel_mod.Block.ASSISTANT,
                                 f"chat B message {i}\n\n" + "text " * 80)
                 for i in range(40)]
        self.panel.blocks = other                      # what a chat switch does
        self.panel._render()
        self._settle()
        self.assertTrue(self._at_end(),
                        f"{self.bar.value()}/{self.bar.maximum()}")

    def test_scrolling_back_to_the_bottom_resumes_following(self):
        self.bar.setValue(self.bar.maximum() // 2)
        self._settle()
        self.panel._render()
        self._settle()
        self.bar.setValue(self.bar.maximum())     # the user returns to the end
        self._settle()
        block = self.panel._append(panel_mod.Block.ASSISTANT, "tail\n\n" * 30)
        self.panel._render()
        self._settle()
        self.assertTrue(self._at_end())
        self.assertIs(block, self.panel.blocks[-1])

    def test_offset_is_forgotten_when_the_log_shrinks_below_it(self):
        # A finished turn collapses its process card: the log gets shorter
        # than the pixel offset the user had parked at. The view then sits
        # at the end, and the next growth must follow it, not jump back.
        long_log = list(self.panel.blocks)
        self.bar.setValue(300)
        self._settle()
        self.panel.blocks[:] = long_log[:1]
        self.panel._render()
        self._settle()
        # shorter than the parked offset (whether or not one block fits the
        # viewport depends on the platform's fonts)
        self.assertLess(self.bar.maximum(), 300)
        # (adapted: the merged panel derives its hold/follow state from the live scrollbar at every render and
        # keeps it in _scroll_mode / _scroll_target, so only the behaviour is asserted, not that state)
        self.panel.blocks[:] = long_log
        self.panel._render()
        self._settle()
        self.assertTrue(self._at_end(),
                        f"{self.bar.value()}/{self.bar.maximum()}")

    def test_offset_below_the_end_survives_a_shorter_log(self):
        # shrinking is only a reason to let go when the offset no longer fits
        long_log = list(self.panel.blocks)
        self.bar.setValue(300)
        self._settle()
        self.panel.blocks[:] = long_log[:30]
        self.panel._render()
        self._settle()
        self.assertEqual(self.bar.value(), 300)


# ---------------------------------------------------------------------------
# panel-a-02 / panel-a-07: markdown rendering must not mangle code or stars


class MarkdownFixTest(_PanelCase):
    def _md(self, text):
        return panel_mod.md_to_html(text)

    # -- panel-a-02: fenced code blocks ----------------------------------

    def test_multiline_code_block_is_kept_verbatim(self):
        out = self._md("```python\nimport Part\n# make a box\n\n"
                       "for i in range(3):\n    print(i)\n```")
        self.assertIn("<pre><code>", out)
        # the comment line, the blank line and the indentation all survive
        self.assertIn("import Part\n# make a box\n\nfor i in range(3):\n"
                      "    print(i)</code></pre>", out)
        self.assertNotIn("<h", out)          # '# comment' is not a heading
        self.assertNotIn("<p ", out)         # no per-line paragraphs, no <p><pre>
        self.assertNotIn("<li", out)

    def test_markdown_lookalikes_inside_code_are_not_rendered(self):
        out = self._md("```\ndef f(*args, **kw):\n    - x\n* y\n```")
        self.assertIn("def f(*args, **kw):\n    - x\n* y", out)
        for tag in ("<li", "<i>", "<b>", "<ul"):
            self.assertNotIn(tag, out)

    def test_text_around_a_code_block_still_renders(self):
        out = self._md("Here:\n```\nx = 1\n```\nDone **now**")
        self.assertIn("Here:</p><pre><code>x = 1</code></pre><p", out)
        self.assertIn("Done <b>now</b>", out)
        self.assertNotIn("<p style='margin:2px 0;'><pre>", out)

    def test_code_block_closes_an_open_list(self):
        out = self._md("- a\n```\nx\n```")
        self.assertIn("</ul><pre><code>x</code></pre>", out)

    def test_unterminated_fence_while_streaming_is_code_not_headings(self):
        out = self._md("text\n```python\nimport Part\n# comment")
        self.assertIn("<pre><code>import Part\n# comment</code></pre>", out)
        self.assertNotIn("<h", out)

    def test_code_survives_the_qt_rich_text_round_trip(self):
        out = self._md("```python\nimport Part\n# make a box\n\n"
                       "for i in range(3):\n    print(i)\n```")
        browser = QtWidgets.QTextBrowser()
        browser.setHtml(out)
        plain = browser.toPlainText()
        self.assertIn("# make a box\n\nfor i in range(3):\n    print(i)", plain)

    def test_nul_in_the_input_cannot_forge_a_placeholder(self):
        out = self._md("\x00CODE0\x00 hello\n```\nx\n```")
        self.assertIn("hello", out)
        self.assertIn("<pre><code>x</code></pre>", out)

    # -- panel-a-07: asterisks are not italics ----------------------------

    def test_multiplication_stars_are_kept(self):
        out = self._md("make a box 80*40*20 mm")
        self.assertIn("80*40*20 mm", out)
        self.assertNotIn("<i>", out)
        self.assertNotIn("<i>", self._md("2 * 3 * 4"))

    def test_glob_patterns_are_kept(self):
        out = self._md("Supported formats: *.stl, *.step, *.dxf")
        self.assertIn("*.stl, *.step, *.dxf", out)
        self.assertNotIn("<i>", out)

    def test_inline_code_content_is_not_reformatted(self):
        out = self._md("Import `*.stl` or *.step files")
        self.assertIn("*.stl</code>", out)
        self.assertNotIn("<i>", out)
        self.assertEqual(out.count("<code"), out.count("</code>"))
        out = self._md("`Params.W * Params.H * 2`")
        self.assertIn("Params.W * Params.H * 2</code>", out)
        self.assertNotIn("<i>", out)
        out = self._md("`**not bold**`")
        self.assertNotIn("<b>", out)

    def test_real_emphasis_still_works(self):
        self.assertIn("<i>italic</i>", self._md("an *italic* word"))
        self.assertIn("<i>a</i>", self._md("one *a* char"))
        self.assertIn("<b>bold</b>", self._md("a **bold** word"))
        self.assertIn("<b>bold</b> and <i>it</i>",
                      self._md("**bold** and *it*"))

    def test_user_bubble_keeps_the_dimensions_as_typed(self):
        block = panel_mod.Block(panel_mod.Block.USER, "make a box 80*40*20 mm")
        self.assertIn("80*40*20 mm", block.html())


# ---------------------------------------------------------------------------
# a stand-in for the agent worker


class _FakeAgent:
    """Just enough of AgentWorker for the panel's stop/pause plumbing."""

    def __init__(self):
        self.stopped = False
        self.resumed = False

    def stop(self):
        self.stopped = True

    def resume(self):
        self.resumed = True

    def pause(self):
        pass


# ---------------------------------------------------------------------------
# panel-b-07: quitting mid-turn must join the worker threads


class _FakeThread:
    """A QThread stand-in. ``stops`` False models a worker stuck in a
    network call that outlasts the shutdown deadline."""

    def __init__(self, stops=True):
        self.running = True
        self.stops = stops
        self.calls = []

    def stop(self):
        self.calls.append("stop")
        if self.stops:
            self.running = False

    def resume(self):
        self.calls.append("resume")

    def isRunning(self):
        return self.running

    def wait(self, ms=0):
        time.sleep(min(ms, 5) / 1000.0)
        return not self.running


def _shutdown_stub(worker=None):
    stub = types.SimpleNamespace(worker=worker)
    return stub, panel_mod.OscarPanel._shutdown_workers


class ShutdownWorkersTest(_PanelCase):
    def test_stops_and_joins_the_worker(self):
        agent = _FakeThread()
        stub, shutdown = _shutdown_stub(agent)
        shutdown(stub, 1000)
        self.assertIn("stop", agent.calls)
        self.assertFalse(agent.isRunning())

    def test_a_paused_worker_is_released_so_it_can_see_the_stop(self):
        agent = _FakeThread()
        stub, shutdown = _shutdown_stub(agent)
        shutdown(stub, 1000)
        self.assertIn("resume", agent.calls)

    def test_a_worker_that_never_stops_cannot_hang_the_quit(self):
        stuck = _FakeThread(stops=False)
        stub, shutdown = _shutdown_stub(stuck)
        started = time.monotonic()
        shutdown(stub, 150)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertIn("stop", stuck.calls)

    def test_nothing_running_is_a_no_op(self):
        stub, shutdown = _shutdown_stub()
        shutdown(stub, 1000)

    def test_a_destroyed_worker_object_is_tolerated(self):
        class Gone(_FakeThread):
            def stop(self):
                raise RuntimeError("Internal C++ object already deleted.")

        stub, shutdown = _shutdown_stub(Gone())
        shutdown(stub, 1000)

    def test_the_panel_hooks_the_application_quit(self):
        class FakeSignal:
            def __init__(self):
                self.slots = []

            def connect(self, slot):
                self.slots.append(slot)

        fake_app = types.SimpleNamespace(aboutToQuit=FakeSignal())
        with mock.patch.object(QtWidgets.QApplication, "instance",
                               return_value=fake_app):
            panel = _make_panel()
        try:
            self.assertEqual(len(fake_app.aboutToQuit.slots), 1)
            agent = panel.worker = _FakeThread()
            fake_app.aboutToQuit.slots[0]()
            self.assertIn("stop", agent.calls)
        finally:
            panel.worker = None
            panel.hide()
            panel.deleteLater()


# ---------------------------------------------------------------------------
# panel-b-04 / panel-b-09: the runtime renamer must not corrupt URLs or
# Oscar's own consent dialogs, and its Report View path must actually work


def _offscreen(widget):
    """Lay a widget out without ever putting it on the desktop."""
    widget.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
    widget.show()
    return widget


def _fake_app(modal):
    return types.SimpleNamespace(activeModalWidget=lambda: modal)


URL = "https://github.com/FreeCAD/FreeCAD-library/raw/master/x.FCStd"


# ---------------------------------------------------------------------------
# panel-b-08: a failed revert must not be reported (or acted on) as done


def _revertible_card(panel, revertible=True):
    card = panel_mod.Block(panel_mod.Block.TOOL, "✻ Ran 1 step · 1.0s")
    card.steps = [panel_mod.Step("create_object", summary="create_object(Box)",
                                 ok=True, dur=1.0)]
    card.revertible = revertible
    # Adapted for the merged panel: a revert needs the turn-start snapshot a real turn records (the panel refuses a
    # card without one); the engine fakes below ignore its content.
    card.snapshot = {"success": True, "docs": {"Doc": 1}}
    panel.blocks.append(card)
    return card


class RevertTurnTest(_PanelCase):
    def setUp(self):
        self.panel = _make_panel()

    def tearDown(self):
        self.panel.hide()
        self.panel.deleteLater()

    def _engine(self, result):
        return mock.patch("oscar.cad.get_engine", return_value=types.SimpleNamespace(
            revert_turn=lambda snapshot: result))

    def test_failed_revert_leaves_the_card_revertible(self):
        card = _revertible_card(self.panel)
        with self._engine({"success": False, "error": "boom"}):
            outcome = self.panel._revert_turn(card)
        self.assertFalse(outcome)
        self.assertFalse(card.reverted)
        html = card.html()
        self.assertIn(f"revert:{card.bid}", html)      # the link is still there
        self.assertNotIn("reverted ✓", html)
        self.assertIn("Revert failed: boom", self.panel.blocks[-1].text)
        self.assertEqual(self.panel._pending_notes, [])

    def test_successful_revert_marks_the_card_and_tells_the_model(self):
        card = _revertible_card(self.panel)
        with self._engine({"success": True, "undone": {"Doc": 2},
                           "closed_docs": []}):
            outcome = self.panel._revert_turn(card)
        self.assertTrue(outcome)
        self.assertTrue(card.reverted)
        self.assertIn("reverted ✓", card.html())
        self.assertEqual(len(self.panel._pending_notes), 1)

    def test_partial_revert_is_reported_as_partial_and_the_model_is_told(self):
        # engine.revert_turn: success False + partial, yet steps WERE undone
        card = _revertible_card(self.panel)
        with self._engine({"success": False, "partial": True,
                           "undone": {"Doc": 1}, "closed_docs": [],
                           "error": "the turn was only partly reverted: x"}):
            outcome = self.panel._revert_turn(card)
        self.assertFalse(outcome)
        self.assertFalse(card.reverted)                  # not claimed as done
        self.assertIn("revert:%s" % card.bid, card.html())  # can be tried again
        self.assertIn("partly reverted", self.panel.blocks[-1].text)
        self.assertEqual(len(self.panel._pending_notes), 1)
        self.assertIn("partly", self.panel._pending_notes[0])

    def test_revert_is_refused_mid_turn(self):
        card = _revertible_card(self.panel)
        self.panel.worker = _FakeAgent()
        try:
            with self._engine({"success": True, "undone": {}, "closed_docs": []}):
                self.assertFalse(self.panel._revert_turn(card))
        finally:
            self.panel.worker = None
        self.assertFalse(card.reverted)

    def test_rerun_does_not_build_on_top_of_a_failed_revert(self):
        _revertible_card(self.panel)
        with self._engine({"success": False, "error": "boom"}), \
                mock.patch.object(self.panel, "_send") as send:
            self.panel._revert_and_rerun_parametric()
        send.assert_not_called()

    def test_rerun_after_a_partial_revert_says_why_it_did_not_rebuild(self):
        _revertible_card(self.panel)
        with self._engine({"success": False, "partial": True,
                           "undone": {"Doc": 1}, "closed_docs": [],
                           "error": "only partly reverted"}), \
                mock.patch.object(self.panel, "_send") as send:
            self.panel._revert_and_rerun_parametric()
        send.assert_not_called()
        self.assertIn("Not rebuilding", self.panel._status)

    def test_rerun_sends_the_parametric_prompt_after_a_good_revert(self):
        _revertible_card(self.panel)
        with self._engine({"success": True, "undone": {"Doc": 1},
                           "closed_docs": []}), \
                mock.patch.object(self.panel, "_send") as send:
            self.panel._revert_and_rerun_parametric()
        send.assert_called_once_with()
        self.assertEqual(self.panel.input.toPlainText(), self.panel.PARAMPT)

    def test_rerun_without_anything_to_revert_still_builds_fresh(self):
        with mock.patch.object(self.panel, "_send") as send:
            self.panel._revert_and_rerun_parametric()
        send.assert_called_once_with()

    def test_reverting_an_older_turn_asks_first(self):
        older = _revertible_card(self.panel)
        newer = _revertible_card(self.panel)
        question = mock.patch.object(
            QtWidgets.QMessageBox, "question",
            return_value=QtWidgets.QMessageBox.No)
        with question as ask:
            self.assertFalse(self.panel._confirm_revert(older))
        ask.assert_called_once()
        with mock.patch.object(QtWidgets.QMessageBox, "question",
                               return_value=QtWidgets.QMessageBox.Yes):
            self.assertTrue(self.panel._confirm_revert(older))
        # the newest revertible turn needs no confirmation
        with mock.patch.object(QtWidgets.QMessageBox, "question") as ask:
            self.assertTrue(self.panel._confirm_revert(newer))
        ask.assert_not_called()

    def test_declined_confirmation_keeps_the_older_turn_unreverted(self):
        older = _revertible_card(self.panel)
        _revertible_card(self.panel)
        with self._engine({"success": True, "undone": {"Doc": 1},
                           "closed_docs": []}), \
                mock.patch.object(QtWidgets.QMessageBox, "question",
                                  return_value=QtWidgets.QMessageBox.No):
            self.panel._on_anchor(QtCore.QUrl(f"revert:{older.bid}"))
        self.assertFalse(older.reverted)


# ---------------------------------------------------------------------------
# panel-a-08: only the automatic narration is hidden under a live card


class StatusLineTest(_PanelCase):
    def setUp(self):
        self.panel = _make_panel()
        self.panel.worker = _FakeAgent()
        self.panel._turn_card = _revertible_card(self.panel)

    def tearDown(self):
        self.panel._status_timer.stop()
        self.panel.worker = None
        self.panel._turn_card = None
        self.panel.hide()
        self.panel.deleteLater()

    def test_automatic_narration_is_shown_under_a_live_card(self):
        # Adapted: the merged panel keeps the status line sticky under a live card on purpose (it is the one place
        # that says what runs right now; a generic "Thinking"/"Working…" yields to the first streamed text), so
        # the narration is NOT hidden any more. What panel-a-08 needs (guard messages are never swallowed) holds
        # in test_guard_messages_are_shown_under_a_live_card.
        for text in ("Thinking", "Running create_object", "Working…"):
            with self.subTest(text=text):
                self.panel._set_status(text)
                self.assertEqual(self.panel._status, text)

    def test_guard_messages_are_shown_under_a_live_card(self):
        for text in ("Wait for the current turn to finish before switching "
                     "chats.",
                     "Wait for the current turn to finish first.",
                     "A turn is already running.",
                     "⏸ Paused", "stopping…"):
            with self.subTest(text=text):
                self.panel._set_status(text)
                self.assertEqual(self.panel._status, text)

    def test_the_netguard_wait_hint_is_shown_then_cleared(self):
        self.panel._on_netguard_wait(True)
        self.assertIn("Waiting for your approval", self.panel._status)
        self.panel._on_netguard_wait(False)       # back to the generic "Working…" status (see the test above)
        self.assertNotIn("Waiting for your approval", self.panel._status)
        self.assertEqual(self.panel._status, "Working…")

    def test_without_a_card_everything_is_shown(self):
        self.panel._turn_card = None
        self.panel._set_status("Thinking")
        self.assertEqual(self.panel._status, "Thinking")


# ---------------------------------------------------------------------------
# panel-a-05: web links in model replies must open in the browser


class AnchorOpenTest(_PanelCase):
    def setUp(self):
        self.panel = _make_panel()

    def tearDown(self):
        self.panel.hide()
        self.panel.deleteLater()

    def _click(self, target):
        with mock.patch.object(QtGui.QDesktopServices, "openUrl") as opener:
            self.panel._on_anchor(QtCore.QUrl(target))
        return opener

    def test_http_and_https_links_open_externally(self):
        for target in ("https://example.com/x", "http://example.com/y"):
            with self.subTest(target=target):
                opener = self._click(target)
                opener.assert_called_once()
                self.assertEqual(opener.call_args[0][0].toString(), target)

    def test_other_schemes_are_never_opened(self):
        for target in ("ftp://example.com/x",
                       "javascript:alert(1)", "x-app://thing"):
            with self.subTest(target=target):
                self._click(target).assert_not_called()

    def test_a_file_link_never_opens_the_file_itself(self):
        # Adapted: the merged panel does open file:// links from the chat, but only by a policy (see
        # panel_anchors._open_external): a plain document / image / CAD file or a folder is opened, anything else
        # (here: an arbitrary system file) is only REVEALED in its folder. What panel-a-05 pins - model text
        # cannot make the panel open an arbitrary local file - holds.
        opener = self._click("file:///etc/passwd")
        for call in opener.call_args_list:
            self.assertNotEqual(call[0][0].toString(), "file:///etc/passwd")
            self.assertEqual(call[0][0].toLocalFile(), "/etc")

    def test_internal_anchors_still_work_and_do_not_open_the_browser(self):
        card = _revertible_card(self.panel)
        opener = self._click(f"tool:{card.bid}")
        opener.assert_not_called()
        self.assertTrue(card.expanded)


# ---------------------------------------------------------------------------
# panel-a-09 / panel-a-13: card header and notice rendering


class CardHeaderTest(_PanelCase):
    def _card(self, steps, t0):
        card = panel_mod.Block(panel_mod.Block.TOOL, "✻ Ran 1 step · 1.0s")
        card.steps = steps
        card.t0 = t0
        card.revertible = True
        return card

    def test_a_running_turn_says_working_between_tools(self):
        # the tool just finished (dur > 0) but the turn is still live
        done = panel_mod.Step("create_object", summary="create_object(Box)",
                              ok=True, dur=0.4)
        card = self._card([done], time.monotonic())
        html = card.html()
        self.assertIn("Working…", html)
        self.assertNotIn("revert changes", html)   # not revertible mid-turn

    def test_a_finished_turn_shows_its_summary_and_the_revert_link(self):
        note = panel_mod.Step(summary="Let me set that up")
        note.plain = True                           # a folded remark: dur 0.0 forever
        done = panel_mod.Step("create_object", summary="create_object(Box)",
                              ok=True, dur=0.4)
        card = self._card([note, done], 0.0)
        html = card.html()
        self.assertNotIn("Working…", html)
        self.assertIn("Ran 1 step", html)
        self.assertIn(f"revert:{card.bid}", html)

    def test_a_live_card_with_only_a_folded_remark_is_still_working(self):
        note = panel_mod.Step(summary="Let me set that up")
        note.plain = True
        self.assertIn("Working…", self._card([note], time.monotonic()).html())

    def test_notice_has_a_single_info_icon(self):
        panel = _make_panel()
        try:
            panel._on_notice("Hello notice")
            html = panel.blocks[-1].html()
        finally:
            panel.hide()
            panel.deleteLater()
        self.assertEqual(html.count("ℹ"), 1)
        self.assertIn("Hello notice", html)


# ---------------------------------------------------------------------------
# panel-b-06: dock position persistence (PySide6 enum handling)


class DockPersistenceTest(_PanelCase):
    def setUp(self):
        from oscar import params

        self.params = params
        self._saved = (params.get_int("DockArea", 2),
                       params.get_bool("DockFloating", False))
        self.window = QtWidgets.QMainWindow()
        self.dock = QtWidgets.QDockWidget("x", self.window)
        self.dock.setObjectName("x")

    def tearDown(self):
        self.params.set_int("DockArea", self._saved[0])
        self.params.set_bool("DockFloating", self._saved[1])
        self.window.deleteLater()

    def test_area_enum_converts_to_the_stored_int(self):
        Qt = QtCore.Qt
        self.assertEqual(panel_mod._dock_area_value(Qt.LeftDockWidgetArea), 1)
        self.assertEqual(panel_mod._dock_area_value(Qt.RightDockWidgetArea), 2)
        self.assertEqual(panel_mod._dock_area_value(Qt.TopDockWidgetArea), 4)
        self.assertEqual(panel_mod._dock_area_value(Qt.BottomDockWidgetArea), 8)
        self.assertEqual(panel_mod._dock_area_value(Qt.NoDockWidgetArea), 0)
        self.assertEqual(panel_mod._dock_area_value(8), 8)    # already an int
        self.assertEqual(panel_mod._dock_area_value(None), 0)

    def test_dock_area_lookup_works_on_a_real_main_window(self):
        self.window.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.dock)
        self.assertEqual(panel_mod.dockWidgetAreaOf(self.window, self.dock),
                         QtCore.Qt.BottomDockWidgetArea)

    def test_moving_the_dock_persists_its_area(self):
        self.window.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock)
        panel_mod._save_dock_state(self.window, self.dock,
                                   QtCore.Qt.LeftDockWidgetArea)
        self.assertEqual(self.params.get_int("DockArea", 0), 1)
        self.assertFalse(self.params.get_bool("DockFloating", True))

    def test_area_is_looked_up_when_the_signal_gave_none(self):
        self.window.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.dock)
        panel_mod._save_dock_state(self.window, self.dock)
        self.assertEqual(self.params.get_int("DockArea", 0), 8)

    def test_floating_keeps_the_last_docked_side(self):
        self.window.addDockWidget(QtCore.Qt.TopDockWidgetArea, self.dock)
        panel_mod._save_dock_state(self.window, self.dock,
                                   QtCore.Qt.TopDockWidgetArea)
        # floating reports NoDockWidgetArea — storing 0 would break restore
        with mock.patch.object(self.dock, "isFloating", return_value=True):
            panel_mod._save_dock_state(self.window, self.dock,
                                       QtCore.Qt.NoDockWidgetArea)
        self.assertEqual(self.params.get_int("DockArea", 0), 4)
        self.assertTrue(self.params.get_bool("DockFloating", False))

    def test_a_stored_area_is_validated_on_restore(self):
        for stored, expected in ((1, 1), (2, 2), (4, 4), (8, 8),
                                 (0, 2), (99, 2), (-3, 2)):
            with self.subTest(stored=stored):
                self.params.set_int("DockArea", stored)
                self.assertEqual(panel_mod._restored_dock_area(), expected)


# ---------------------------------------------------------------------------
# panel-b-11: the Settings command must refresh an open panel


class SettingsCommandTest(_PanelCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from oscar import commands

        cls.commands = commands

    def _run(self, accepted, dock_widget="panel"):
        class FakePanel(QtWidgets.QWidget):
            reloads = 0

            def _reload_profiles(self):
                self.reloads += 1

        window = self._window = QtWidgets.QMainWindow()  # keep the dock alive
        widget = None
        if dock_widget is not None:
            dock = QtWidgets.QDockWidget("Oscar", window)
            dock.setObjectName(panel_mod.DOCK_OBJECT_NAME)
            widget = FakePanel() if dock_widget == "panel" else QtWidgets.QWidget()
            dock.setWidget(widget)

        class FakeDialog:
            def __init__(self, parent=None):
                pass

            def exec(self):
                return accepted

        gui = types.SimpleNamespace(getMainWindow=lambda: window)
        with mock.patch.object(self.commands, "Gui", gui), \
                mock.patch("oscar.settings_ui.SettingsDialog", FakeDialog):
            self.commands.SettingsCommand().Activated()
        return widget

    def test_saving_refreshes_the_open_panel(self):
        self.assertEqual(self._run(True).reloads, 1)

    def test_cancelling_leaves_the_panel_alone(self):
        self.assertEqual(self._run(False).reloads, 0)

    def test_saving_without_a_panel_is_fine(self):
        self._run(True, dock_widget=None)

    def test_a_dock_without_the_panel_widget_is_fine(self):
        self._run(True, dock_widget="other")


if __name__ == "__main__":
    unittest.main()
