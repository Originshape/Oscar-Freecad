# SPDX-License-Identifier: Apache-2.0
"""The ask_user tool handler headless: validation errors, the wait/resume round trip through the
live registry, wizard answers, Stop release and the conversation binding (oscar.tools_ask).

No Qt and no panel surface here — the registry handshake is exercised for real by resolving the
request from a second thread while the handler blocks."""

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

from oscar import askuser, tools_ask  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402
from oscar.tools import ToolError  # noqa: E402


def _resolve_when_open(registry, conversation_id, run_id, answer, timeout_s=5.0):
    """Resolve the first open request of (conversation, run) from another thread, as the panel would."""
    def run():
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pending = registry.pending(conversation_id=conversation_id)
            ours = [p for p in pending if p.run_id == run_id]
            if ours:
                registry.resolve(ours[0].id, answer, conversation_id=conversation_id, run_id=run_id)
                return
            time.sleep(0.01)
    t = threading.Thread(target=run)
    t.start()
    return t


class AskUserToolTest(unittest.TestCase):
    def setUp(self):
        self._saved = askuser.REGISTRY
        askuser.REGISTRY = askuser.RequestRegistry()      # per-test isolation of the live registry
        self._saved_surface = tools_ask.surface
        tools_ask.surface = None

    def tearDown(self):
        askuser.REGISTRY.shutdown()
        askuser.REGISTRY = self._saved
        tools_ask.surface = self._saved_surface

    def _ctx(self, conversation_id="chat-1", run_id="turn-1", stop=None):
        return ToolContext(conversation_id=conversation_id, turn_id=run_id,
                           should_stop=(stop or (lambda: False)))

    def test_a_malformed_question_is_rejected_not_repaired(self):
        with use_context(self._ctx()):
            with self.assertRaises(ToolError) as raised:
                tools_ask.ask_user(title="", options=[{"label": "a"}, {"label": "b"}])
        self.assertIn("title", str(raised.exception))

    def test_a_turn_without_a_conversation_refuses_to_ask(self):
        with use_context(ToolContext(turn_id="turn-1")):
            with self.assertRaises(ToolError) as raised:
                tools_ask.ask_user(title="Units?", options=[{"label": "mm", "likeliest": True},
                                                            {"label": "in"}])
        self.assertIn("conversation", str(raised.exception))

    def test_the_handler_blocks_then_resumes_with_the_answer(self):
        resolver = _resolve_when_open(askuser.REGISTRY, "chat-1", "turn-1",
                                      {"answer": "Millimetres", "index": 0})
        with use_context(self._ctx()):
            text = tools_ask.ask_user(title="Which unit?", subtitle="185 x 12 x 200",
                                      options=[{"label": "Millimetres", "likeliest": True},
                                               {"label": "Inches"}])
        resolver.join(timeout=5)
        self.assertIn("USER ANSWERED:", text)
        self.assertIn("Millimetres", text)
        self.assertIn('"index": 0', text)
        self.assertIn('"request_id"', text)                 # the full protocol reaches the model

    def test_a_free_text_custom_answer_round_trips(self):
        resolver = _resolve_when_open(askuser.REGISTRY, "chat-1", "turn-1",
                                      {"answer": "Metres", "index": -1, "value": "m"})
        with use_context(self._ctx()):
            text = tools_ask.ask_user(title="Which unit?", allow_free_text=True,
                                      options=[{"label": "Millimetres", "likeliest": True},
                                               {"label": "Inches"}])
        resolver.join(timeout=5)
        self.assertIn("Metres", text)
        self.assertIn('"index": -1', text)

    def test_a_wizard_returns_every_answer_in_order(self):
        answer = {"answers": [{"answer": "Millimetres", "index": 0},
                              {"answer": "Approve all values", "index": 0}]}
        resolver = _resolve_when_open(askuser.REGISTRY, "chat-1", "turn-1", answer)
        with use_context(self._ctx()):
            text = tools_ask.ask_user(wizard=[
                {"title": "Which unit?", "options": [{"label": "Millimetres", "likeliest": True},
                                                     {"label": "Inches"}]},
                {"title": "Approve the rounded dimensions?",
                 "options": [{"label": "Approve all values", "likeliest": True},
                             {"label": "Review individually"}]}])
        resolver.join(timeout=5)
        self.assertIn("Q: Which unit?", text)
        self.assertIn("Q: Approve the rounded dimensions?", text)
        self.assertIn("Approve all values", text)
        self.assertIn('"answers"', text)

    def test_stop_releases_the_waiter_with_a_tool_error(self):
        stop = threading.Event()

        def flip():
            time.sleep(0.05)
            stop.set()

        threading.Thread(target=flip).start()
        with use_context(self._ctx(stop=stop.is_set)):
            with self.assertRaises(ToolError) as raised:
                tools_ask.ask_user(title="Which unit?",
                                   options=[{"label": "Millimetres", "likeliest": True},
                                            {"label": "Inches"}])
        self.assertIn("cancelled", str(raised.exception))
        self.assertEqual(askuser.REGISTRY.pending(), [])   # nothing left dangling

    def test_a_wrong_conversation_cannot_answer_and_stop_can(self):
        # the real chat's request is opened by the handler; a stale answer from another chat bounces
        stop = threading.Event()
        ctx = self._ctx(conversation_id="chat-1", run_id="turn-1", stop=stop.is_set)

        def stale_then_stop():
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                pending = askuser.REGISTRY.pending(conversation_id="chat-1")
                if pending:
                    ok, _err = askuser.REGISTRY.resolve(
                        pending[0].id, {"answer": "Inches", "index": 1},
                        conversation_id="OTHER-CHAT", run_id="turn-1")
                    assert not ok                                   # stale: never settles the live request
                    time.sleep(0.05)
                    stop.set()                                     # Stop ends the wait honestly
                    return
                time.sleep(0.01)

        interloper = threading.Thread(target=stale_then_stop)
        interloper.start()
        with use_context(ctx):
            with self.assertRaises(ToolError) as raised:
                tools_ask.ask_user(title="Which unit?",
                                   options=[{"label": "Millimetres", "likeliest": True},
                                            {"label": "Inches"}])
        interloper.join(timeout=5)
        self.assertIn("cancelled", str(raised.exception))


class RunToolPipelineTest(unittest.TestCase):
    """Issue #20's architecture end to end: the handler runs through ``tools.run_tool`` on a real
    worker thread, blocks on the registry, and ONLY the registry submission from the "GUI" side
    wakes it — ``run_tool`` then returns the structured answer to the "agent" (the worker thread
    joins with the answer, proving the worker resumed instead of merely the registry flipping)."""

    def setUp(self):
        self._saved = askuser.REGISTRY
        askuser.REGISTRY = askuser.RequestRegistry()      # per-test isolation of the live registry
        self._saved_surface = tools_ask.surface
        tools_ask.surface = None

    def tearDown(self):
        askuser.REGISTRY.shutdown()
        askuser.REGISTRY = self._saved
        tools_ask.surface = self._saved_surface

    def test_run_tool_blocks_then_returns_the_submitted_answer(self):
        from oscar import tools

        results = []

        def worker():
            ctx = ToolContext(conversation_id="chat-9", turn_id="turn-9",
                              should_stop=lambda: False)
            with use_context(ctx):
                try:
                    results.append(tools.run_tool("ask_user", {
                        "title": "Which unit is Base.STL in?",
                        "options": [{"label": "Millimetres", "likeliest": True},
                                    {"label": "Inches"}]}))
                except Exception as exc:   # noqa: BLE001 - surfaced by the assertions below
                    results.append(("crash", exc))

        agent = threading.Thread(target=worker)
        agent.start()
        deadline = time.monotonic() + 5.0
        request = None
        while time.monotonic() < deadline:
            pending = askuser.REGISTRY.pending(conversation_id="chat-9")
            if pending:
                request = pending[0]
                break
            time.sleep(0.01)
        self.assertIsNotNone(request, "the worker must have registered the pending request")
        ok, err = askuser.REGISTRY.submit_answer(request.id, {"answer": "Inches", "index": 1},
                                                 conversation_id="chat-9", run_id="turn-9")
        self.assertEqual((ok, err), (True, ""))
        agent.join(timeout=5)
        self.assertFalse(agent.is_alive(), "the worker must resume as soon as the answer lands")
        text, image_b64, is_error = results[0]
        self.assertFalse(is_error)
        self.assertIsNone(image_b64)
        self.assertIn("USER ANSWERED:", text)
        self.assertIn("Inches", text)
        self.assertIn('"index": 1', text)
        self.assertIn('"conversation_id": "chat-9"', text)   # the full answer protocol arrived
        self.assertEqual(askuser.REGISTRY.pending(), [])      # nothing left dangling


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
