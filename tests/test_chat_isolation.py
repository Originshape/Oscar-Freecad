# SPDX-License-Identifier: Apache-2.0
"""Tests of chat independence (issue #2: a new chat can see or change another chat).

Two scenarios on a real ``OscarPanel`` with fake workers:

- the concurrency case: a build runs in chat 1, a new chat is opened while it processes, a message is
  typed and sent there. Everything the running turn produces — streamed text, process card, plan,
  artifacts, project folder, the final answer, the persisted turn record — must land in chat 1 only;
  the new chat's history, plan, project folder and transcript must stay its own, and its message must
  run as its own turn (never steered into chat 1's conversation) once the panel is free.
- the literal report (``test_issue2_literal_reproduction_...``): chat A finishes BEFORE chat B starts.

The isolation boundary: the panel keeps everything per chat — history, transcript, plan, project
folder, attachments, undelivered images, composer draft. What chats SHARE, by design, is the open
FreeCAD documents themselves: the worker lists every open document for the model (``agent.py``), so a
chat asking "what is in this document?" sees objects another chat created. That is the CAD workspace,
not chat state; this module pins the panel side of the boundary.

Same harness as test_panel_session.py: offscreen QApplication, nothing shown to the user.
"""

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
    from test_panel_flow import FlowTestCase, j, png
    from test_panel_logic import GOOD_SNAPSHOT
    from test_panel_render import ledger
    from test_panel_session import Sig
except Exception as exc:   # pragma: no cover - plain interpreter without FreeCAD / Qt
    _IMPORT_ERROR = str(exc)

    class _Unavailable(unittest.TestCase):
        """Base class that skips every test of a module whose Qt / FreeCAD imports failed."""

        @classmethod
        def setUpClass(cls):
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")

    FlowTestCase = _Unavailable
    Sig = object


class IsolationWorker:
    """Stands in for AgentWorker: records constructor arguments, connected slots and steer calls."""

    instances: list = []
    SIGNALS = ("text_delta", "reasoning_delta", "tool_started", "tool_finished", "turn_finished", "notice",
               "flow_paused", "step_started", "step_finished", "plan_changed", "progress", "round_reset",
               "artifacts_changed", "turn_outcome", "account_problem")

    def __init__(self, history, parent=None, **kw):
        self.history, self.kw, self.started = history, kw, False
        self.steer_calls: list = []
        self.calls: list = []                 # pause / resume / stop, from the flow controls
        for name in self.SIGNALS:
            setattr(self, name, Sig())
        type(self).instances.append(self)

    def start(self):
        self.started = True

    def pause(self):
        self.calls.append("pause")

    def resume(self):
        self.calls.append("resume")

    def stop(self):
        self.calls.append("stop")

    def drain_unsent(self):
        out, self._unsent = getattr(self, "_unsent", []), []
        return out

    def steer(self, text, note="", images=()):
        self.steer_calls.append((text, list(images)))

    def wait(self):
        return True

    def deleteLater(self):
        pass


class ChatIsolationTest(FlowTestCase):
    """The issue's scenario: a build in chat 1, a new chat while it runs, a message sent there."""

    def setUp(self):
        super().setUp()
        IsolationWorker.instances = []
        self.engine = mock.Mock()
        self.engine.begin_turn_snapshot.return_value = dict(GOOD_SNAPSHOT)
        for patcher in (mock.patch.object(panel, "AgentWorker", IsolationWorker),
                        mock.patch.object(params, "active_profile",
                                          return_value=params.Profile(name="t", provider="openai_compat", model="m")),
                        mock.patch("oscar.cad.get_engine", return_value=self.engine),
                        mock.patch.object(self.p, "_fit_view_after_turn"),
                        # a chat opened with a waiting message asks before it sends; the tests answer Yes
                        mock.patch.object(QtWidgets.QMessageBox, "question",
                                          return_value=QtWidgets.QMessageBox.Yes)):
            patcher.start()
            self.addCleanup(patcher.stop)
        # The panel resolves each chat's profile through its own profile list before it sends. In an
        # isolated test the seeded "OriginShape" account profile is signed out, which would refuse every
        # send and no worker would exist; a list without it lets the mocked ``active_profile`` answer.
        self.p._profiles = [params.Profile(name="t", provider="openai_compat", model="m")]

    def send(self, text):
        self.p.input.setPlainText(text)
        self.p._send()
        return IsolationWorker.instances[-1]

    def end_turn(self, worker, ok=True, text="Done."):
        worker.turn_outcome.slots[0]("answered" if ok else "failed", "")
        self.p._on_turn_finished(ok, text)

    # -- issue #2's literal report -----------------------------------------------------------------

    def test_issue2_literal_reproduction_each_chat_is_its_own_conversation(self):
        """Issue #2's exact steps, sequentially: chat A finishes BEFORE chat B starts. B's request carries
        only B (its own history, session id and project folder, no plan of A) and the provider context
        header carries B's chat id; going back to A shows A's history intact and A's next turn carrying
        A's own plan. The open FreeCAD documents stay shared by design (see the module docstring)."""
        from oscar.provider_originshape import context_headers

        p = self.p
        worker_a = self.send("Create a blue 50 mm cube named ChatAObject.")
        session_a = p._current_session()
        p._on_plan_changed(j(ledger()))
        self.end_turn(worker_a, text="Created the blue cube ChatAObject.")

        p._new_chat()                                     # step 2: + for chat B, and it is empty
        session_b = p._current_session()
        self.assertIsNot(session_b, session_a)
        self.assertEqual(session_b["history"], [])
        self.assertIsNone(session_b.get("ledger"))
        self.assertIsNone(session_b.get("project_dir"))

        worker_b = self.send("Create a red 20 mm cylinder named ChatBObject.")   # step 3
        self.assertIs(worker_b.history, session_b["history"])     # step 4: B's request is B's own
        self.assertEqual(worker_b.kw["session_id"], session_b["id"])
        self.assertEqual(len(worker_b.history), 1)
        self.assertIn("ChatBObject", worker_b.history[0].content)
        self.assertNotIn("ChatAObject", worker_b.history[0].content)
        self.assertIsNone(worker_b.kw["ledger"])                  # no plan of A rides it
        self.assertNotEqual(worker_b.kw["project"].root, worker_a.kw["project"].root)   # nor A's folder
        headers = context_headers({"session": worker_b.kw["session_id"], "turn": "t1"})
        self.assertEqual(headers["x-oscar-session"], session_b["id"])   # the provider header names B
        self.assertEqual(headers["x-oscar-turn"], "t1")
        self.end_turn(worker_b, text="Created the red cylinder ChatBObject.")

        p._on_chat_picked(p.sessions.index(session_a))     # step 4b: back to A
        self.assertEqual([str(m.content) for m in session_a["history"]],
                         ["Create a blue 50 mm cube named ChatAObject."])
        worker_a2 = self.send("what size is it?")
        self.assertIs(worker_a2.history, session_a["history"])
        self.assertEqual(worker_a2.kw["session_id"], session_a["id"])
        self.assertEqual(worker_a2.kw["ledger"]["intent"], "enclosure")   # A carries A's plan, not B's
        self.end_turn(worker_a2, text="50 mm.")

    # -- regressions of the review of the isolation PR ---------------------------------------------

    def test_a_blocked_delivery_keeps_the_queued_messages_and_the_draft(self):
        """B1: a queued message whose send is blocked (signed out, new Terms, no key…) keeps waiting in
        its chat — with its bubble — and the composer holds what the user was typing, not the queued text."""
        p = self.p
        worker1 = self.send("Build the box")             # chat 1 runs
        p._new_chat()
        session2 = p._current_session()
        p.input.setPlainText("first queued")
        p._send()
        p.input.setPlainText("second queued")
        p._send()
        p.input.setPlainText("my unsent draft")
        with mock.patch.object(p, "_send_blocker", return_value="signed_out"), \
                mock.patch.object(p, "_block_send"):
            self.end_turn(worker1)
            self.app.processEvents()     # the delivery is one event-loop turn behind the finish
        self.assertEqual(p.input.toPlainText(), "my unsent draft")   # the draft survived the blocked send
        self.assertEqual([w.get("text") for w in p._waiting_sends], ["first queued", "second queued"])
        queued = [b for b in session2["blocks"] if b.kind == Block.USER]
        self.assertEqual([b.text for b in queued], ["first queued", "second queued"])   # bubbles kept
        self.assertTrue(all(b.queued for b in queued))
        self.assertEqual(session2["history"], [])                    # nothing was sent into it
        self.assertIsNone(p.worker)                                  # and no turn started

    def test_picking_the_chat_shown_again_keeps_the_composer(self):
        """B2: QComboBox.activated also fires for the row already selected; re-picking the chat shown must
        not replace the message being typed (the workspace / remote-revert paths re-activate the shown
        chat the same way)."""
        p = self.p
        session = p._current_session()
        p.input.setPlainText("half-typed message")
        p._on_chat_picked(p.sessions.index(session))     # the picker's signal for the row already chosen
        self.assertEqual(p.input.toPlainText(), "half-typed message")
        session["_draft"] = "an old draft"
        p._activate_session(session)                     # re-activating the shown chat changes nothing
        self.assertEqual(p.input.toPlainText(), "half-typed message")

    def test_a_queued_message_goes_out_behind_another_chats(self):
        """B3: delivery searches the queue for the chat shown instead of stopping at another chat's entry."""
        p = self.p
        worker_a = self.send("chat A first message")
        self.end_turn(worker_a)
        session_a = p._current_session()
        p._new_chat()
        worker_b = self.send("chat B first message")
        self.end_turn(worker_b)
        session_b = p._current_session()
        p._new_chat()
        worker_c = self.send("Build the box")            # chat C runs
        p._on_chat_picked(p.sessions.index(session_a))
        p.input.setPlainText("for chat A")
        p._send()                                        # queued in A
        p._on_chat_picked(p.sessions.index(session_b))   # B is shown when C's turn ends
        p.input.setPlainText("for chat B")
        p._send()                                        # queued in B, behind A's entry
        before = len(IsolationWorker.instances)
        self.end_turn(worker_c)
        self.app.processEvents()
        started = IsolationWorker.instances[before:]
        self.assertTrue(started, "chat B is shown and idle but its queued message did not go out")
        self.assertEqual(started[-1].kw["session_id"], session_b["id"])
        self.assertEqual([w.get("text") for w in p._waiting_sends], ["for chat A"])   # A's keeps waiting

    def test_pause_stop_and_esc_belong_to_the_running_turns_chat(self):
        """R1: in a chat with nothing running the send button sends (the message queues there), and the
        stop button, Esc and the heartbeat of the other chat's turn do not show in it."""
        p = self.p
        worker1 = self.send("Build the box")
        session1 = p._current_session()
        p._new_chat()
        session2 = p._current_session()
        self.assertTrue(p.stop_btn.isHidden())           # chat 2 is idle: no stop control of chat 1's turn
        self.assertFalse(p._esc_shortcut.isEnabled())    # Esc is the application's again
        self.assertEqual(p.send_btn.text(), "➤")         # the button sends
        self.assertFalse(p.send_btn.isHidden())          # and shows: chat 2's composer is idle (shell redesign)
        self.assertTrue(p.heartbeat_row.isHidden())      # chat 1's heartbeat does not show in chat 2
        p.input.setPlainText("a question in chat two")
        p._on_flow_button()                              # ➤ sends: the message queues in chat 2
        self.assertEqual(worker1.calls, [])              # chat 1's turn was never paused or stopped
        queued = [b for b in session2["blocks"] if b.kind == Block.USER]
        self.assertEqual([b.text for b in queued], ["a question in chat two"])

        p._on_chat_picked(p.sessions.index(session1))    # back in the turn's own chat: its controls show
        self.assertFalse(p.stop_btn.isHidden())
        self.assertTrue(p._esc_shortcut.isEnabled())
        self.assertTrue(p.send_btn.isHidden())           # the Stop square has the trailing slot (shell redesign)
        self.assertFalse(p.heartbeat_row.isHidden())
        p._on_pause_button()                             # the heartbeat's pause control pauses chat 1's turn
        self.assertEqual(worker1.calls, ["pause"])
        p._on_flow_paused(True)
        self.assertEqual(p.pause_btn.text(), "▶")        # and ▶ resumes it

    def test_an_account_failure_on_the_first_request_gives_the_message_back_to_its_chat(self):
        """R2: the turn's first request failed for an account reason while ANOTHER chat is shown: the
        message leaves its chat's history and returns to THAT chat's draft — never to the composer of the
        chat on screen."""
        p = self.p
        worker1 = self.send("Build the paid thing")
        session1 = p._current_session()
        stl = os.path.join(self.tmp, "paid.stl")
        with open(stl, "w") as f:
            f.write("solid x")
        p.add_attachments([stl])                         # attached while the turn runs
        p._new_chat()
        session2 = p._current_session()
        p.input.setPlainText("chat two is typing")
        p._on_account_problem(j({"kind": "auth", "code": "signed_out", "first_request": True}))
        self.end_turn(worker1, ok=False, text="Provider error: HTTP 401")
        self.assertEqual(session1["history"], [])        # the message left chat 1's history
        self.assertEqual(session1["_draft"], "Build the paid thing")   # it waits in ITS chat's composer
        self.assertEqual(session1["pending"], [stl])     # its attachment too
        self.assertFalse(any(b.kind == Block.USER for b in session1["blocks"]))   # its bubble is gone
        self.assertEqual(p.input.toPlainText(), "chat two is typing")  # the shown chat's composer untouched
        self.assertEqual(session2["history"], [])
        p._on_chat_picked(p.sessions.index(session1))    # opening chat 1 shows the returned message
        self.assertEqual(p.input.toPlainText(), "Build the paid thing")

    def test_an_undelivered_steer_waits_in_its_chats_composer(self):
        """R5: a steered message the stopped turn could not deliver returns to the chat that steered it —
        as that chat's draft when another chat is on screen, not only as a notice line."""
        p = self.p
        worker1 = self.send("Build the box")
        session1 = p._current_session()
        worker1._unsent = [{"text": "late words", "note": "", "images": []}]
        p._new_chat()
        self.end_turn(worker1, ok=False, text="Stopped.")   # stops while chat 2 is shown
        self.assertEqual(session1.get("_draft"), "late words")
        self.assertEqual(p.input.toPlainText(), "")          # the shown chat's composer is not hijacked
        self.assertTrue(any("back in its chat's input" in (b.text or "")                     # said so there
                            for b in session1["blocks"]))

    def test_a_waiting_send_asks_when_its_chat_is_opened_later(self):
        """A message that kept waiting (its chat was not shown when the panel became free) asks before it
        is sent; declining keeps it as the composer text of its chat instead."""
        p = self.p
        worker1 = self.send("Build the box")
        session1 = p._current_session()
        p._new_chat()
        session2 = p._current_session()
        p.input.setPlainText("late question")
        p._send()                                        # queues in chat 2
        p._on_chat_picked(p.sessions.index(session1))    # chat 1 stays shown: nothing delivers yet
        self.end_turn(worker1, text="done")
        self.app.processEvents()
        self.assertEqual([w.get("text") for w in p._waiting_sends], ["late question"])
        with mock.patch.object(QtWidgets.QMessageBox, "question",
                               return_value=QtWidgets.QMessageBox.No) as ask:
            p._on_chat_picked(p.sessions.index(session2))
            self.app.processEvents()
        ask.assert_called_once()
        self.assertEqual(p._waiting_sends, [])           # declined: it left the queue
        self.assertEqual(p.input.toPlainText(), "late question")   # it waits in the composer instead
        self.assertEqual(session2["history"], [])
        worker2 = self.send("late question")             # sending it by hand works, as its own turn
        self.assertEqual(worker2.kw["session_id"], session2["id"])

    # -- the reported scenario ---------------------------------------------------------------------

    def test_new_chat_while_busy_and_a_message_there_never_touch_chat_one(self):
        p = self.p
        worker1 = self.send("Build this component parametrically: one Part")
        session1 = p._current_session()
        self.assertTrue(worker1.started)
        chat1_history = session1["history"]

        # chat 1 works: a plan, a tool step, streaming text, artifacts
        p._on_plan_changed(j(ledger()))
        p._on_tool_started("execute_code")
        p._on_step_started(j({"call_id": "call_a1", "name": "execute_code", "title": "Make the body",
                              "args_summary": "code=...", "round": 1, "ordinal": 1, "mutating": True,
                              "geometry": True, "phase": "BUILD", "started_at": 1790000000.5}))
        p._on_text_delta("chat one is building")

        # the issue's step 2: a new chat while chat 1 still processes
        p._new_chat()
        session2 = p._current_session()
        self.assertIsNot(session2, session1)
        self.assertEqual(session2["history"], [])

        # the issue's step 3: a message typed and sent in the new chat
        p.input.setPlainText("unrelated question about the document")
        p._send()
        self.assertEqual(worker1.steer_calls, [])            # never handed to chat 1's conversation
        self.assertEqual(session2["history"], [])            # and not chat 2's yet: it waits for the panel
        queued = [b for b in session2["blocks"] if b.kind == Block.USER]
        self.assertEqual(len(queued), 1)
        self.assertTrue(queued[0].queued)
        self.assertEqual(p.input.toPlainText(), "")           # the text left the input with the queued bubble

        # chat 1 keeps streaming while chat 2 is shown: nothing of it may appear in chat 2
        p._on_text_delta(" — still building")
        self.assertFalse(any("chat one" in (b.text or "") for b in session2["blocks"]))
        card1 = next(b for b in session1["blocks"] if b.kind == Block.TOOL and b.steps)
        self.assertTrue(any("chat one is building" in ((s.summary or "") + (s.detail or ""))
                            for s in card1.steps))     # it streams as the card's live note, in chat 1

        # chat 1's artifacts / project folder land in chat 1, not in chat 2
        project1 = os.path.join(self.tmp, "chat-one-project")
        p._on_artifacts_changed(j({"project_dir": project1, "files": []}))
        self.assertEqual(session1.get("project_dir"), project1)
        self.assertIsNone(session2.get("project_dir"))

        # chat 1's turn ends while chat 2 is shown: its answer, card and plan stay in chat 1
        self.end_turn(worker1, text="chat one answer")
        self.assertTrue(any("chat one answer" in (b.text or "") for b in session1["blocks"]))
        self.assertFalse(any("chat one answer" in (b.text or "") for b in session2["blocks"]))
        self.assertEqual(session1["history"], chat1_history)      # the worker's list, unchanged
        self.assertEqual(len(chat1_history), 1)                   # its user message (the fake worker writes no reply)
        self.assertTrue(chat1_history[0].content.startswith("Build this component"))
        self.assertEqual(session1.get("ledger", {}).get("intent"), "enclosure")
        self.assertIsNone(session2.get("ledger"))
        self.assertIsNone(p.worker)                               # the panel is free again

        # the queued message goes out as chat 2's OWN turn, with chat 2's (empty) history
        self.app.processEvents()      # the delivery is one event-loop turn behind the finish
        worker2 = IsolationWorker.instances[-1]
        self.assertIsNot(worker2, worker1)
        self.assertTrue(worker2.started)
        self.assertIs(worker2.history, session2["history"])
        self.assertEqual(worker2.kw["request_text"], "unrelated question about the document")
        self.assertIs(p._current_session(), session2)
        bubbles = [b for b in session2["blocks"] if b.kind == Block.USER]
        self.assertEqual(len(bubbles), 1)
        self.assertFalse(bubbles[0].queued)                       # the queued bubble became the real one
        self.assertEqual(p.input.toPlainText(), "")                # _send_text restored nothing over it

        # chat 2's answer lands in chat 2, and chat 1's transcript is untouched by it
        before = [b.text for b in session1["blocks"]]
        self.end_turn(worker2, text="chat two answer")
        self.assertTrue(any("chat two answer" in (b.text or "") for b in session2["blocks"]))
        self.assertEqual([b.text for b in session1["blocks"]], before)

    def test_switching_back_to_the_running_chat_mid_turn_keeps_its_live_state(self):
        p = self.p
        worker1 = self.send("Build the box")
        session1 = p._current_session()
        p._on_plan_changed(j(ledger()))
        p._new_chat()
        session2 = p._current_session()
        index1 = p.sessions.index(session1)

        p._on_chat_picked(index1)                    # back to chat 1 while its worker runs
        self.assertIs(p._current_session(), session1)
        p._on_text_delta("chat one streams on")
        p._render()
        self.assertIn("chat one streams on", self.plain())
        p._on_chat_picked(p.sessions.index(session2))
        self.assertNotIn("chat one streams on", self.plain())   # chat 2's screen shows chat 2 only

        self.end_turn(worker1, text="chat one streams on")   # the full text of what streamed
        self.assertTrue(any("chat one streams on" in (b.text or "") for b in session1["blocks"]))

    def test_several_chats_each_request_and_carryover_is_its_own(self):
        """The acceptance list: create/switch/continue several chats, switching DURING a running turn —
        every worker request carries its own chat's history and session id, a finished turn never touches
        another chat, and each chat carries its own plan into its next turn."""
        p = self.p
        worker1 = self.send("Build the box")            # chat 1
        session1 = p._current_session()
        self.assertEqual(worker1.kw["session_id"], session1["id"])
        self.assertIs(worker1.history, session1["history"])
        p._on_plan_changed(j(ledger()))

        # a second chat while chat 1's turn runs; a message queued there
        p._new_chat()
        session2 = p._current_session()
        p.input.setPlainText("second chat task")
        p._send()
        p._on_chat_picked(p.sessions.index(session1))    # back to chat 1 mid-turn

        # a third chat, only looked at: nothing of the others is in it
        p._new_chat()
        session3 = p._current_session()
        self.assertEqual(session3["history"], [])
        self.assertIsNone(session3.get("ledger"))
        self.assertEqual(session3["blocks"], [p._welcome])

        # back to chat 1 again: its card and streamed state are right there
        p._on_chat_picked(p.sessions.index(session1))
        p._on_tool_started("execute_code")
        p._render()
        self.assertTrue(any(b.kind == Block.TOOL and b.steps for b in session1["blocks"]))

        # chat 1's turn ends while chat 1 is shown: chat 2's queued message must NOT fire into it
        self.end_turn(worker1, text="box done")
        self.app.processEvents()
        self.assertIs(p._current_session(), session1)
        self.assertEqual(len(IsolationWorker.instances), 1)
        self.assertEqual(session2["history"], [])

        # opening chat 2 delivers what waited there — as chat 2's OWN turn
        p._on_chat_picked(p.sessions.index(session2))
        self.app.processEvents()
        worker2 = IsolationWorker.instances[-1]
        self.assertIsNot(worker2, worker1)
        self.assertEqual(worker2.kw["session_id"], session2["id"])
        self.assertIs(worker2.history, session2["history"])
        self.assertEqual(worker2.kw["request_text"], "second chat task")
        self.assertIsNone(worker2.kw["ledger"])          # chat 2 carries no plan of chat 1

        # continuing chat 1 afterwards: it carries ITS OWN plan, on its own history
        self.end_turn(worker2, text="second done")
        p._on_chat_picked(p.sessions.index(session1))
        worker3 = self.send("continue the box")
        self.assertEqual(worker3.kw["session_id"], session1["id"])
        self.assertIs(worker3.history, session1["history"])
        self.assertEqual(worker3.kw["ledger"]["intent"], "enclosure")
        self.end_turn(worker3, text="continued")

    def test_attachments_and_draft_are_kept_per_chat(self):
        """An attached file/image belongs to the chat it was attached in, and so does the unfinished
        composer text: switching chats must neither send chat 1's files with chat 2's message nor show
        chat 1's draft in chat 2's composer."""
        p = self.p
        worker1 = self.send("Build the box")
        session1 = p._current_session()
        stl = os.path.join(self.tmp, "part.stl")
        with open(stl, "w") as f:
            f.write("solid x")
        image = png(os.path.join(self.tmp, "a.png"), 32, 32)
        p.add_attachments([stl, image])
        p.input.setPlainText("half-written thought about the box")   # an unsent draft

        p._new_chat()                                               # while chat 1's turn runs
        session2 = p._current_session()
        self.assertEqual(p.pending, [])                             # chat 2 starts with no attachments
        self.assertEqual(p.input.toPlainText(), "")                 # and without chat 1's draft
        self.end_turn(worker1, text="done")

        worker2 = self.send("what is in the document?")             # a real send in chat 2
        self.assertEqual(worker2.kw["attachments"], [])             # chat 1's files did not ride it
        self.assertFalse(session2["history"][-1].images)            # nor its image
        p.input.setPlainText("chat two's unfinished reply")         # chat 2's own draft, mid-turn

        p._on_chat_picked(p.sessions.index(session1))               # back to chat 1
        self.assertEqual(p.pending, [stl, image])                   # its attachments are back
        self.assertEqual(p.input.toPlainText(), "half-written thought about the box")

        p._on_chat_picked(p.sessions.index(session2))               # and chat 2's draft survives too
        self.assertEqual(p.input.toPlainText(), "chat two's unfinished reply")

    def test_late_steer_images_return_to_the_chat_that_sent_them(self):
        """An image of a steer that could not be delivered goes back to the chat that steered it — even
        when the turn ends while a DIFFERENT chat is shown — and rides only that chat's next message."""
        p = self.p
        worker1 = self.send("Build the box")
        session1 = p._current_session()
        worker1._unsent = [{"text": "late words", "note": "", "images": ["B64IMG"]}]
        p._new_chat()
        session2 = p._current_session()
        self.end_turn(worker1, ok=False, text="Stopped.")           # stops while chat 2 is shown

        self.assertEqual(session1.get("carry_images"), ["B64IMG"])  # home it came from...
        self.assertEqual(p._carry_images, [])                       # ...not the chat on screen

        worker2 = self.send("plain question")                       # chat 2's next message is clean
        self.assertFalse(session2["history"][-1].images)
        self.end_turn(worker2, text="answered")

        p._on_chat_picked(p.sessions.index(session1))
        self.send("try again")                                      # chat 1's next message carries it
        self.assertEqual(session1["history"][-1].images, ["B64IMG"])
        self.assertEqual(p._carry_images, [])

    def test_an_stl_attached_near_another_chat_never_crosses_over(self):
        """The report's literal scenario: an STL pending while another chat runs a turn — it never rides the
        other chat's message, and it goes out with ITS chat's next one, into its own project folder."""
        p = self.p
        worker1 = self.send("Build the box")            # chat 1's turn runs
        session1 = p._current_session()
        stl = os.path.join(self.tmp, "bracket.stl")
        with open(stl, "w") as f:
            f.write("solid bracket\nendsolid bracket\n")
        p.add_attachments([stl])                        # the STL waits in chat 1

        p._new_chat()                                   # a new chat while the turn runs
        p.input.setPlainText("import this other part")
        p._send()                                       # queues there (chat 1 still running)
        self.end_turn(worker1, text="done")
        self.app.processEvents()                        # the queued send runs as chat 2's own turn
        worker2 = IsolationWorker.instances[-1]
        self.assertEqual(worker2.kw["attachments"], [])                    # chat 1's STL did not ride it
        self.assertIsNone(worker2.kw["ledger"])                            # nor chat 1's plan
        self.assertNotEqual(worker2.kw["project"].root, worker1.kw["project"].root)   # nor its folder

        p._on_chat_picked(p.sessions.index(session1))   # back in chat 1
        self.end_turn(worker2, text="ok")
        worker3 = self.send("now use the bracket")
        self.assertEqual(worker3.kw["attachments"], [stl])                 # it goes out with its own chat
        self.assertEqual(worker3.kw["session_id"], session1["id"])

    def test_a_send_while_busy_in_the_same_chat_still_steers(self):
        """The same chat's own turn is steered, as before: the bubble is its queued steer, never a waiting
        send of the cross-chat queue."""
        p = self.p
        session1 = p._current_session()
        worker1 = self.send("Build the box")
        p.input.setPlainText("and make it 40 mm")
        p._send()
        self.assertEqual(worker1.steer_calls, [("and make it 40 mm", [])])   # same chat: the steer path, as before
        self.assertEqual(p._waiting_sends, [])                               # the queue is for OTHER chats only
        queued = [b for b in session1["blocks"] if b.kind == Block.USER]
        self.assertEqual(queued[-1].text, "and make it 40 mm")
        self.assertTrue(queued[-1].queued)


    def test_provider_headers_and_persisted_state_discriminate_the_chats(self):
        """Phase 5 coverage: BOTH chats' provider contexts carry their OWN ``x-oscar-session``
        (never each other's), and reloading the persisted chats keeps each chat's history
        separate - the restart half of the isolation boundary."""
        from oscar.provider_originshape import context_headers
        from oscar import chats as _chats

        p = self.p
        worker_a = self.send("Chat A message")
        session_a = p._current_session()
        self.end_turn(worker_a)
        p._new_chat()
        worker_b = self.send("Chat B message")
        session_b = p._current_session()
        self.end_turn(worker_b)
        for session, worker in ((session_a, worker_a), (session_b, worker_b)):
            headers = context_headers({"session": worker.kw["session_id"], "turn": "t"})
            self.assertEqual(headers["x-oscar-session"], session["id"])
        self.assertNotEqual(worker_a.kw["session_id"], worker_b.kw["session_id"])
        loaded_a = _chats.load(session_a["id"])
        loaded_b = _chats.load(session_b["id"])
        self.assertEqual([str(m.content) for m in loaded_a["history"]], ["Chat A message"])
        self.assertEqual([str(m.content) for m in loaded_b["history"]], ["Chat B message"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
