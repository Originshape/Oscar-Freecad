"""Tests of chat independence (Issue: a new chat is influenced by a previous chat that is still processing).

The scenario of the report, on a real ``OscarPanel`` with fake workers: a build runs in chat 1, a new chat
is opened while it processes, a message is typed and sent there. Everything the running turn produces —
streamed text, process card, plan, artifacts, project folder, the final answer, the persisted turn record —
must land in chat 1 only; the new chat's history, plan, project folder and transcript must stay its own,
and its message must run as its own turn (never steered into chat 1's conversation) once the panel is free.

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
    from oscar import chats, panel, params
    from oscar.panel import Block
    from test_panel_flow import FlowTestCase, png
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


def j(payload) -> str:
    import json

    return json.dumps(payload)


class IsolationWorker:
    """Stands in for AgentWorker: records constructor arguments, connected slots and steer calls."""

    instances: list = []
    SIGNALS = ("text_delta", "reasoning_delta", "tool_started", "tool_finished", "turn_finished", "notice",
               "flow_paused", "step_started", "step_finished", "plan_changed", "progress", "round_reset",
               "artifacts_changed", "turn_outcome", "account_problem")

    def __init__(self, history, parent=None, **kw):
        self.history, self.kw, self.started = history, kw, False
        self.steer_calls: list = []
        for name in self.SIGNALS:
            setattr(self, name, Sig())
        type(self).instances.append(self)

    def start(self):
        self.started = True

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
                        mock.patch.object(self.p, "_fit_view_after_turn")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def send(self, text):
        self.p.input.setPlainText(text)
        self.p._send()
        return IsolationWorker.instances[-1]

    def end_turn(self, worker, ok=True, text="Done."):
        worker.turn_outcome.slots[0]("answered" if ok else "failed", "")
        self.p._on_turn_finished(ok, text)

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
        worker3 = self.send("try again")                            # chat 1's next message carries it
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

    def test_a_send_while_busy_in_the_same_chat_still_stears(self):
        p = self.p
        worker1 = self.send("Build the box")
        p.input.setPlainText("and make it 40 mm")
        p._send()
        self.assertEqual(worker1.steer_calls, [("and make it 40 mm", [])])   # same chat: the steer path, as before
        self.assertIs(p._current_session(), p._current_session())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
