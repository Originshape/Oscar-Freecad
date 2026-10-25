"""Tests of the panel's per-chat build state and message plumbing (oscar/panel.py, panel_session.py,
panel_turn.py): what ``_send`` hands the worker (Project, typed request text, carried plan, per-round persistence),
the persisted turn records and the restore card, the templates menu, steering with images, and the mutating-tool
set. The signal-level behaviour (plan strip, step rows, heartbeat, artifacts strip, outcome) is in test_panel_flow.py.

Same harness: a real ``OscarPanel`` on an offscreen QApplication, a fake worker, nothing shown to the user.
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

_IMPORT_ERROR = ""
try:
    from PySide import QtWidgets
    from oscar import chats, panel, params
    from oscar.panel import Block
    from test_panel_flow import FlowTestCase, FlowWorker, j, png
    from test_panel_logic import GOOD_SNAPSHOT, FakeWorker, PanelTestCase
except Exception as exc:   # pragma: no cover - plain interpreter without FreeCAD / Qt
    _IMPORT_ERROR = str(exc)

    class _Unavailable(unittest.TestCase):
        """Base class that skips every test of a module whose Qt / FreeCAD imports failed."""

        @classmethod
        def setUpClass(cls):
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")

    FakeWorker = object
    FlowWorker = object
    PanelTestCase = FlowTestCase = _Unavailable

from test_panel_render import BUILD_SUMMARY, finished, ledger, started  # noqa: E402


# --------------------------------------------------------------------------------------------------
# _send wiring: Project, request text, ledger, persistence


class Sig:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


class CapturingWorker(FlowWorker):
    """Stands in for AgentWorker: records the constructor arguments and the connected slots."""

    instances: list = []
    SIGNALS = ("text_delta", "reasoning_delta", "tool_started", "tool_finished", "turn_finished", "notice",
               "flow_paused", "step_started", "step_finished", "plan_changed", "progress", "round_reset",
               "artifacts_changed", "turn_outcome")

    def __init__(self, history, parent=None, **kw):
        super().__init__()
        self.history, self.kw, self.started = history, kw, False
        for name in self.SIGNALS:
            setattr(self, name, Sig())
        type(self).instances.append(self)

    def start(self):
        self.started = True


class SendWiringTest(FlowTestCase):
    def setUp(self):
        super().setUp()
        CapturingWorker.instances = []
        self.engine = mock.Mock()
        self.engine.begin_turn_snapshot.return_value = dict(GOOD_SNAPSHOT)
        for patcher in (mock.patch.object(panel, "AgentWorker", CapturingWorker),
                        mock.patch.object(params, "active_profile",
                                          return_value=params.Profile(name="t", provider="openai_compat", model="m")),
                        mock.patch("oscar.cad.get_engine", return_value=self.engine)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def send(self, text):
        self.p.input.setPlainText(text)
        self.p._send()
        return CapturingWorker.instances[-1]

    def end_turn(self):
        with mock.patch.object(self.p, "_fit_view_after_turn"):
            self.p._on_turn_finished(True, "ok")

    def test_the_worker_gets_project_request_text_ledger_and_on_round(self):
        from oscar.project import Project

        p = self.p
        worker = self.send("Build the box")
        kw = worker.kw
        self.assertIsInstance(kw["project"], Project)
        self.assertEqual(kw["request_text"], "Build the box")
        self.assertIsNone(kw["ledger"])
        self.assertEqual(kw["attachments"], [])
        self.assertTrue(callable(kw["on_round"]))
        self.assertTrue(worker.started)
        self.assertIs(p.worker, worker)

    def test_the_project_is_bound_after_the_title_and_is_lazy(self):
        p = self.p
        worker = self.send("Build the box")
        project = worker.kw["project"]
        session = p._current_session()
        self.assertEqual(session["title"], "Build the box")
        self.assertEqual(project.manifest["chat_id"], session["id"])
        self.assertTrue(os.path.basename(project.root).startswith("build-the-box"))   # the folder name carries the title
        self.assertFalse(os.path.exists(project.root))                                # nothing written on the GUI thread
        self.assertEqual(session["project_dir"], project.root)
        self.assertNotIn("_project", chats.load(session["id"]) or {})

    def test_all_signals_are_connected(self):
        worker = self.send("Build the box")
        for name in CapturingWorker.SIGNALS:
            self.assertEqual(len(getattr(worker, name).slots), 1, name)
        slots = {name: getattr(worker, name).slots[0] for name in CapturingWorker.SIGNALS}
        self.assertEqual(slots["plan_changed"], self.p._on_plan_changed)
        self.assertEqual(slots["turn_outcome"], self.p._on_turn_outcome)
        self.assertTrue(self.p._heartbeat_timer.isActive())

    def test_a_worker_without_the_new_signals_still_runs(self):
        class Legacy(CapturingWorker):
            SIGNALS = ("text_delta", "reasoning_delta", "tool_started", "tool_finished", "turn_finished", "notice",
                       "flow_paused")

        with mock.patch.object(panel, "AgentWorker", Legacy):
            self.p.input.setPlainText("Build the box")
            self.p._send()
        self.assertTrue(Legacy.instances[-1].started)

    def test_the_second_turn_reuses_the_project_and_carries_the_plan(self):
        p = self.p
        first = self.send("Build the box")
        first.plan_changed.slots[0](j(ledger()))
        self.end_turn()
        second = self.send("Now add a lid")
        self.assertIs(second.kw["project"], first.kw["project"])
        self.assertEqual(second.kw["ledger"]["intent"], "enclosure")
        self.assertEqual(second.kw["ledger"]["revision"], 12)
        self.assertEqual(p._current_session()["title"], "Build the box")

    def test_attachments_and_typed_text_reach_the_worker_separately(self):
        text_file = os.path.join(self.tmp, "notes.txt")
        with open(text_file, "w") as f:
            f.write("hello")
        self.p.add_attachments([text_file])
        worker = self.send("Use the attached notes")
        self.assertEqual(worker.kw["attachments"], [text_file])
        self.assertEqual(worker.kw["request_text"], "Use the attached notes")   # the typed text, not the prompt block
        self.assertIn("hello", self.p.history[-1].content)                       # the attachment text is in the message

    def test_attachment_only_send_uses_the_default_request_text(self):
        text_file = os.path.join(self.tmp, "n.txt")
        with open(text_file, "w") as f:
            f.write("x")
        self.p.add_attachments([text_file])
        self.p.input.setPlainText("")
        self.p._send()
        self.assertEqual(CapturingWorker.instances[-1].kw["request_text"], "Please work with the attached file(s).")

    def test_on_round_persists_the_chat_with_its_build_state(self):
        p = self.p
        worker = self.send("Build the box")
        worker.plan_changed.slots[0](j(ledger()))
        worker.kw["on_round"](p.history)                  # the worker thread calls this after every provider round
        saved = chats.load(p._current_session()["id"])
        self.assertEqual(saved["ledger"]["intent"], "enclosure")
        self.assertEqual(saved["project_dir"], worker.kw["project"].root)
        self.assertEqual(saved["history"][-1].content.splitlines()[0], "Build the box")

    def test_a_project_that_cannot_be_created_does_not_stop_the_turn(self):
        p = self.p
        with mock.patch("oscar.project.Project.for_chat", side_effect=OSError("read-only volume")):
            worker = self.send("Build the box")
        self.assertIsNone(worker.kw["project"])
        self.assertTrue(worker.started)
        self.assertIsNone(p._current_session().get("project_dir"))

    def test_the_turn_state_is_reset_for_every_turn(self):
        p = self.p
        self.send("one")
        first = p._turn_view
        seq = p._turn_seq
        p._turn_outcome = ("validated", "x")
        self.end_turn()
        self.send("two")
        self.assertIsNot(p._turn_view, first)
        self.assertEqual(p._turn_seq, seq + 1)
        self.assertEqual(p._turn_outcome, ("", ""))
        self.assertEqual(p._turn_request, "two")

    def test_turn_outcome_and_finish_end_to_end_through_the_connected_slots(self):
        p = self.p
        worker = self.send("Build the box")
        worker.step_started.slots[0](j(started("c1", "create_object", "Box", ordinal=1)))
        worker.step_finished.slots[0](j(finished("c1", "create_object", "Box", ordinal=1, snapshot_path="")))
        worker.turn_outcome.slots[0]("validated", BUILD_SUMMARY)
        with mock.patch.object(p, "_fit_view_after_turn"):
            worker.turn_finished.slots[0](True, "Built it.\n\n" + BUILD_SUMMARY)
        self.assertIsNone(p.worker)
        self.assertEqual(p.log.toPlainText().count("Build Summary"), 1)

    def test_blocks_meta_ledger_and_title_are_persisted_at_turn_end(self):
        p = self.p
        worker = self.send("Build the box")
        worker.plan_changed.slots[0](j(ledger()))
        worker.step_started.slots[0](j(started("c1", "create_object", "Make the box", ordinal=1)))
        worker.step_finished.slots[0](j(finished("c1", "create_object", "Make the box", ordinal=1, snapshot_path="")))
        worker.turn_outcome.slots[0]("validated", "")
        self.end_turn()
        saved = chats.load(p._current_session()["id"])
        self.assertEqual(len(saved["blocks_meta"]), 1)
        meta = saved["blocks_meta"][0]
        self.assertEqual((meta["request"], meta["outcome"]), ("Build the box", "validated"))
        self.assertEqual(meta["steps"][0]["title"], "Make the box")
        self.assertTrue(meta["header"].endswith("✓ validated"))
        self.assertEqual(saved["ledger"]["intent"], "enclosure")

    def test_a_plain_answer_turn_leaves_no_turn_record(self):
        p = self.p
        self.send("what is a fillet?")
        self.end_turn()
        self.assertEqual(p._current_session().get("blocks_meta") or [], [])

    def test_storage_note_of_the_project_is_shown_once(self):
        p = self.p
        worker = self.send("Build the box")
        worker.kw["project"].manifest["storage_note"] = "Projects are stored in /tmp/x (the Documents folder is not writable)."
        self.end_turn()
        self.assertEqual(p.log.toPlainText().count("is not writable"), 1)
        self.send("again")
        self.end_turn()
        self.assertEqual(p.log.toPlainText().count("is not writable"), 1)


# --------------------------------------------------------------------------------------------------
# templates, steer with images, restore card, mutating tools


def fake_menu(chooser):
    """A QMenu whose ``exec`` returns ``chooser(menu)`` instead of opening a popup (a class-level patch of
    ``QMenu.exec`` does not reach PySide instances, so the panel's ``QtWidgets.QMenu`` is swapped for this)."""

    class FakeMenu(QtWidgets.QMenu):
        labels: list = []

        def exec(self, *_args):
            type(self).labels = [a.text() for a in self.actions() if a.text()]
            return chooser(self)

    return FakeMenu


class TemplatesTest(PanelTestCase):
    def test_entries_come_from_the_prompt_module(self):
        from oscar import prompt

        entries = dict(self.p._template_entries())
        self.assertEqual(len(entries), len(prompt.TEMPLATES))
        self.assertIn("Enclosure for a board (insert)", entries)
        self.assertEqual(entries["Enclosure for a board (insert)"], prompt.TEMPLATES["enclosure_intake"])

    def test_menu_keeps_the_historical_entries_and_appends_the_new_ones(self):
        from oscar import prompt

        p = self.p
        menu = fake_menu(lambda m: next(a for a in m.actions() if a.text() == "Mounting bracket (insert)"))
        with mock.patch.object(QtWidgets, "QMenu", menu):
            p._show_templates()
        labels = menu.labels
        self.assertEqual(labels[:2], ["Parametric feature-tree build (insert)",
                                      "Revert last turn & re-run parametric build"])
        self.assertEqual(len(labels), 2 + len(prompt.TEMPLATES))
        self.assertEqual(p.input.toPlainText(), prompt.TEMPLATES["bracket"])

    def test_the_historical_entries_still_work(self):
        p = self.p
        with mock.patch.object(QtWidgets, "QMenu", fake_menu(lambda m: m.actions()[0])):
            p._show_templates()
        self.assertIn("PartDesign Body", p.input.toPlainText())
        self.assertIn("spreadsheet", p.PARAMPT)

    def test_the_first_placeholder_is_selected_for_overwriting(self):
        p = self.p
        p._insert_template("Build a housing for <board or part> in <ABS>.")
        self.assertEqual(p.input.textCursor().selectedText(), "<board or part>")
        p._insert_template("No placeholders here.")
        self.assertEqual(p.input.textCursor().selectedText(), "")

    def test_a_missing_prompt_module_keeps_the_menu_working(self):
        p = self.p
        with mock.patch.dict(sys.modules, {"oscar.prompt": None}):
            self.assertEqual(p._template_entries(), [])
        with mock.patch.object(p, "_template_entries", return_value=[]), \
                mock.patch.object(QtWidgets, "QMenu", fake_menu(lambda m: m.actions()[0])):
            p._show_templates()
        self.assertIn("PartDesign Body", p.input.toPlainText())

    def test_unknown_template_keys_get_a_readable_label(self):
        p = self.p
        with mock.patch("oscar.prompt.TEMPLATES", {"snap_fit_lid": "Do <x>.", "blank": "  ", "bad": 5}):
            self.assertEqual(p._template_entries(), [("Snap fit lid (insert)", "Do <x>.")])


class SteerImagesTest(FlowTestCase):
    def test_a_late_image_attachment_travels_with_the_steered_message(self):
        p = self.p
        worker = self.begin_turn()
        image = png(os.path.join(self.tmp, "sketch.png"), 64, 48)
        p.add_attachments([image])
        p.input.setPlainText("use this sketch for the lid")
        p._send()
        self.assertEqual(len(worker.calls), 1)
        verb, text, images = worker.calls[0]
        self.assertEqual(verb, "steer")
        self.assertIn("use this sketch for the lid", text)
        self.assertIn("sketch.png", text)                  # the attachment block of build_user_message
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0])                         # base64 PNG
        self.assertEqual(p.pending, [])                    # delivered attachments leave the queue
        bubble = p.blocks[-1]
        self.assertTrue(bubble.queued)
        self.assertIn("sketch.png", bubble.text)

    def test_non_image_attachments_stay_pending_while_busy(self):
        p = self.p
        worker = self.begin_turn()
        stl = os.path.join(self.tmp, "part.stl")
        with open(stl, "w") as f:
            f.write("solid x")
        image = png(os.path.join(self.tmp, "a.png"), 32, 32)
        p.add_attachments([stl, image])
        p.input.setPlainText("look")
        p._send()
        self.assertEqual(p.pending, [stl])
        self.assertIn("Attachments", p._status)
        self.assertEqual(len(worker.calls[0]), 3)

    def test_unreadable_images_stay_pending_and_the_text_is_still_steered(self):
        p = self.p
        worker = self.begin_turn()
        broken = os.path.join(self.tmp, "broken.png")
        with open(broken, "wb") as f:
            f.write(b"nope")
        p.add_attachments([broken])
        p.input.setPlainText("hello")
        p._send()
        self.assertEqual(worker.calls, [("steer", "hello")])
        self.assertEqual(p.pending, [broken])

    def test_undelivered_images_ride_with_the_resend(self):
        p = self.p
        worker = FlowWorker(unsent=[{"text": "late words", "note": "", "images": ["B64IMG"]}])
        self.begin_turn(worker)
        p.input.setPlainText("late words")
        p._send()
        sent = []
        p._send_text = lambda text, **kw: sent.append((text, kw))
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn(True, "ok")
        self.app.processEvents()
        self.assertEqual(sent, [("late words", {"images": ["B64IMG"]})])

    def test_images_of_an_undelivered_message_survive_a_stop_for_the_next_send(self):
        p = self.p
        worker = FlowWorker(unsent=[{"text": "late words", "note": "", "images": ["B64IMG"]}])
        self.begin_turn(worker)
        p.input.setPlainText("late words")
        p._send()
        self.finish_turn(False, "Stopped.")
        self.assertEqual(p.input.toPlainText(), "late words")
        self.assertEqual(p._carry_images, ["B64IMG"])
        with mock.patch.object(panel, "AgentWorker", CapturingWorker), \
                mock.patch.object(params, "active_profile",
                                  return_value=params.Profile(name="t", provider="openai_compat", model="m")), \
                mock.patch("oscar.cad.get_engine", return_value=mock.Mock(begin_turn_snapshot=lambda: dict(GOOD_SNAPSHOT))):
            p._send()
        self.assertEqual(p.history[-1].images, ["B64IMG"])
        self.assertEqual(p._carry_images, [])

    def test_send_text_passes_images_to_the_next_turn(self):
        p = self.p
        with mock.patch.object(p, "_send", lambda: None):
            p._send_text("hello", images=["I1"])
        self.assertEqual(p._carry_images, ["I1"])


class RestoreCardTest(FlowTestCase):
    def session(self, **over):
        session = chats.new_session("Pi 4 housing")
        session["history"] = [panel.Message(role="user", content="build a pi case"),
                              panel.Message(role="assistant", content="done")]
        session["blocks"] = None
        session.update(over)
        self.p.sessions.insert(0, session)
        return session

    def meta(self, snapshot="snapshots/001_shell.png"):
        return [{"v": 1, "request": "Build a housing for a Raspberry Pi 4", "outcome": "validated",
                 "header": "✻ Ran 2 steps · 9.0s · ✓ validated",
                 "steps": [{"title": "Make the shell", "ok": True, "skipped": False, "dur": 1.2, "delta": "+Shell",
                            "snapshot": snapshot}, {"title": "Add vents", "ok": False, "skipped": False, "dur": 0.5,
                                                    "delta": "", "snapshot": ""}]}]

    def test_the_card_shows_the_turn_records_and_thumbnails_of_the_last_turn(self):
        p = self.p
        root = os.path.join(self.tmp, "proj")
        os.makedirs(os.path.join(root, "snapshots"))
        png(os.path.join(root, "snapshots", "001_shell.png"))
        session = self.session(blocks_meta=self.meta(), project_dir=root)
        p._activate_session(session)
        card = next(b for b in p.blocks if b.kind == Block.RESTORE)
        plain = p.log.toPlainText()
        self.assertIn("Restored “Pi 4 housing” — 2 messages", plain)
        self.assertIn("Build a housing for a Raspberry Pi 4", plain)
        self.assertIn("✓ Make the shell", plain)
        self.assertIn("✗ Add vents", plain)
        self.assertIn("✓ validated", plain)
        self.assertEqual(len(card.thumbs), 1)
        self.assertIn("es-img://step/restore.", p.log.toHtml())
        self.assertTrue(p.log.has_resource(next(iter(card.thumbs))))

    def test_evicted_thumbnails_leave_the_card_not_a_broken_image(self):
        p = self.p
        root = os.path.join(self.tmp, "proj")
        os.makedirs(os.path.join(root, "snapshots"))
        png(os.path.join(root, "snapshots", "001_shell.png"))
        session = self.session(blocks_meta=self.meta(), project_dir=root)
        p._activate_session(session)
        card = next(b for b in p.blocks if b.kind == Block.RESTORE)
        self.assertEqual(len(card.thumbs), 1)
        p._thumbs.clear()                                   # what an LRU eviction does
        p._render()
        self.assertEqual(card.thumbs, set())
        self.app.processEvents()
        self.assertNotIn("es-img://", p.log.toHtml())

    def test_a_missing_snapshot_just_leaves_the_thumbnail_out(self):
        p = self.p
        session = self.session(blocks_meta=self.meta("snapshots/gone.png"), project_dir=os.path.join(self.tmp, "none"))
        p._activate_session(session)
        card = next(b for b in p.blocks if b.kind == Block.RESTORE)
        self.assertEqual(card.thumbs, set())
        self.assertNotIn("es-img://", p.log.toHtml())

    def test_an_interrupted_chat_says_so(self):
        p = self.p
        calls = [panel.Message(role="assistant", content="", tool_calls=[
            __import__("oscar.providers", fromlist=["ToolCall"]).ToolCall(id="c1", name="execute_code", arguments={})])]
        session = self.session()
        session["history"].extend(calls)
        p._activate_session(session)
        self.assertIn("interrupted in the middle of a step", p.log.toPlainText())

    def test_damaged_turn_records_never_break_the_card(self):
        p = self.p
        for bad in ("garbage", None, [1, 2, "x"], [{"steps": "no"}], [{"steps": [None, 5]}]):
            session = self.session(blocks_meta=bad)
            session["blocks"] = None
            p._activate_session(session)
            self.assertIn("Restored", p.log.toPlainText())

    def test_a_short_or_fresh_chat_gets_no_card(self):
        p = self.p
        session = self.session()
        session["history"] = [panel.Message(role="user", content="hi")]
        p._activate_session(session)
        self.assertEqual([b for b in p.blocks if b.kind == Block.RESTORE], [])
        p._new_chat()
        self.assertEqual([b for b in p.blocks if b.kind == Block.RESTORE], [])

    def test_a_saved_chat_restores_its_plan_and_artifacts_from_disk(self):
        p = self.p
        root = os.path.join(self.tmp, "proj")
        os.makedirs(root)
        with open(os.path.join(root, "project.json"), "w") as f:
            json.dump({"schema": 1, "id": "proj", "chat_id": "x", "title": "t", "counters": {}, "docs": []}, f)
        with open(os.path.join(root, "NOTES.md"), "w") as f:
            f.write("# notes")
        session = self.session(project_dir=root, ledger=ledger())
        chats.save(session)
        loaded = chats.load(session["id"])
        loaded["blocks"] = None
        p.sessions.insert(0, loaded)
        p._activate_session(loaded)
        self.assertFalse(p.plan_strip.isHidden())
        self.assertIn("Bottom shell", p.plan_strip.toPlainText())
        self.assertFalse(p.artifacts_strip.isHidden())
        self.assertIn("NOTES.md", p.artifacts_strip.text())
        self.assertIsNotNone(loaded.get("_project"))


class ChatListBehaviourTest(PanelTestCase):
    """Behaviour of the chat list: no duplicate empty chats, empty leftovers removed at start, recency order, and a
    delete that never falls back to an empty "New chat" while a real chat exists."""

    def real_chat(self, title="real chat", text="keep me"):
        session = chats.new_session(title)
        session["history"] = [panel.Message(role="user", content=text), panel.Message(role="assistant", content="ok")]
        session["blocks"] = None
        chats.save(session)
        return session

    def reopen(self):
        self.p.deleteLater()
        self.p = panel.OscarPanel()
        self.addCleanup(self.p.deleteLater)
        return self.p

    def test_new_chat_in_an_unused_chat_is_refused_and_says_so(self):
        p = self.p
        before = list(p.sessions)
        for _ in range(3):
            p._new_chat()
        self.assertEqual(p.sessions, before)
        self.assertEqual(p._status, "Already in a new chat.")
        self.assertEqual(len(p.sessions), 1)

    def test_new_chat_in_a_chat_with_messages_creates_one_and_then_refuses_a_second(self):
        p = self.p
        p._current_session()["history"].append(panel.Message(role="user", content="hello"))
        p._new_chat()
        self.assertEqual(len(p.sessions), 2)
        empty = p._current_session()
        self.assertEqual((empty["history"], empty["title"]), ([], "New chat"))
        p._new_chat()                                           # the new chat is unused: no duplicate
        self.assertEqual(len(p.sessions), 2)
        self.assertIs(p._current_session(), empty)

    def test_a_cleared_chat_is_an_unused_chat_too(self):
        p = self.p
        p._current_session()["history"].append(panel.Message(role="user", content="hello"))
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
            p._clear_chat()
        p._new_chat()
        self.assertEqual(len(p.sessions), 1)

    def test_empty_leftover_chat_files_are_deleted_at_start_and_real_chats_stay(self):
        real = self.real_chat()
        for _ in range(3):
            chats.save(chats.new_session())                      # what older builds left behind
        self.assertEqual(len(chats.list_sessions()), 4)
        p = self.reopen()
        self.assertEqual([s["id"] for s in chats.list_sessions()], [real["id"]])   # the leftovers are gone from disk
        self.assertEqual([s["id"] for s in p.sessions], [real["id"]])
        self.assertEqual(p._current_session()["id"], real["id"])

    def test_only_empty_leftovers_start_a_single_fresh_chat(self):
        chats.save(chats.new_session())
        chats.save(chats.new_session())
        p = self.reopen()
        self.assertEqual(len(p.sessions), 1)
        self.assertEqual(p.sessions[0]["history"], [])
        self.assertEqual(chats.list_sessions(), [])

    def test_a_stale_sidecar_never_gets_a_chat_with_messages_deleted(self):
        real = self.real_chat()
        sidecar = os.path.join(self._tmp, real["id"] + ".meta.json")
        with open(sidecar, "w", encoding="utf-8") as f:       # a crash between the two writes: the sidecar says 0
            json.dump({"id": real["id"], "title": real["title"], "updated": real["updated"], "messages": 0}, f)
        os.utime(sidecar, None)
        self.assertEqual(chats.list_sessions()[0]["messages"], 0)
        p = self.reopen()
        self.assertTrue(os.path.isfile(os.path.join(self._tmp, real["id"] + ".json")))
        self.assertEqual([s["id"] for s in p.sessions], [real["id"]])

    def test_a_chat_with_a_plan_but_no_messages_is_not_deleted(self):
        session = chats.new_session("planned")
        session["ledger"] = {"v": 1}
        chats.save(session)
        self.assertFalse(chats.is_unused(session["id"]))
        self.reopen()
        self.assertTrue(os.path.isfile(os.path.join(self._tmp, session["id"] + ".json")))

    def test_the_active_chat_is_always_the_first_in_recency_order(self):
        a, b = self.real_chat("chat a", "aaa"), self.real_chat("chat b", "bbb")
        p = self.reopen()
        by_title = {s["title"]: s for s in p.sessions}
        for title in ("chat a", "chat b", "chat a"):
            p._on_chat_picked(next(i for i, s in enumerate(p.sessions) if s is by_title[title]))
            self.assertIs(p.sessions[0], by_title[title])
            self.assertEqual(p.chat_combo.currentIndex(), 0)
            self.assertEqual(p.chat_combo.currentText(), title)
            self.assertIs(p._current_session(), by_title[title])

    def test_delete_falls_back_to_the_last_used_real_chat_never_an_empty_one(self):
        base = self.real_chat("base", "keep me")
        other = self.real_chat("other", "other text")
        p = self.reopen()
        by_title = {s["title"]: s for s in p.sessions}
        p._activate_session(by_title["base"])
        p._activate_session(by_title["other"])
        p._new_chat()                                            # an empty chat on top
        empty = p._current_session()
        p._activate_session(by_title["base"])
        p._activate_session(empty)                               # the empty chat is the most recent one ...
        p._activate_session(by_title["other"])
        p._activate_session(empty)
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
            p._delete_chat()                                     # ... and is deleted: back to the last-used real chat
        self.assertNotIn(empty, p.sessions)
        self.assertEqual(p.chat_combo.currentText(), "other")
        self.assertTrue(any(m.content == "other text" for m in p.history))
        # deleting a real chat while an empty one is the next most recent still lands on a real chat
        p._new_chat()
        empty2 = p._current_session()
        p._activate_session(by_title["other"])
        p._activate_session(empty2)
        p._activate_session(by_title["other"])
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
            p._delete_chat()
        self.assertEqual(p.chat_combo.currentText(), "base")


class MutatingToolsTest(PanelTestCase):
    def test_the_set_is_derived_from_the_registry(self):
        from oscar import tools

        names = set(tools.mutating_names())
        self.assertTrue(names)
        self.assertTrue(names <= panel.OscarPanel._MUTATING_TOOLS)
        for name in ("apply_feature_plan", "execute_code", "create_object", "set_parameters", "build_enclosure",
                     "export_model", "save_document"):
            if name in tools._HANDLERS:
                self.assertIn(name, panel.OscarPanel._MUTATING_TOOLS)

    def test_a_rich_mutating_step_triggers_the_live_refresh_flag(self):
        p = self.p
        p.worker = FlowWorker()
        p._turn_mutated = False
        from oscar import tools

        name = "export_model" if "export_model" in tools.mutating_names() else "execute_code"
        self.assertNotIn(name, panel._STATIC_MUTATING if name == "export_model" else ())   # registry-derived, not static
        p._on_tool_started(name)
        p._on_tool_finished(f"{name}()", True, "ok")
        self.assertTrue(p._turn_mutated)
