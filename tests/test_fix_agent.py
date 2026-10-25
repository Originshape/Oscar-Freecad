"""Regression tests for the agent-loop review fixes (agent-loop-02 .. 06).

Each class pins one defect: the context-size guard charging base64 length for
images (03), the execution nudge firing on empty replies (04) and on the
synthetic cut-off note instead of the user's prompt (05), the nudge keyword
match reading attachments / app notes and mid-word hits (06), and provider
construction (which may block on the network) running on the GUI thread
(02). Runs in the console interpreter (no GUI, no network): the provider is a
scripted stand-in.
"""

import os
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

# Importing FreeCADGui (or stubbing it where there is no GUI) is what makes
# the PySide shim importable in the bundled interpreter, as in the other
# modules; freecadcmd already has it, which is why this passed there.
if "FreeCADGui" not in sys.modules:
    try:
        import FreeCADGui  # noqa: F401
    except Exception:
        sys.modules["FreeCADGui"] = types.ModuleType("FreeCADGui")

from PySide import QtCore  # noqa: E402

from oscar import agent, attachments, params  # noqa: E402
from oscar.providers import Message, ToolCall, ToolResult  # noqa: E402


class _Scripted:
    """Provider stand-in: one scripted reply per request. Running past the
    script yields an error event, so a spurious extra request fails the
    assertions instead of raising inside the generator."""

    vision_enabled = True

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.requests = []

    def chat(self, system, messages, specs, should_stop=None):
        self.requests.append(list(messages))
        if not self.rounds:
            yield ("error", RuntimeError("script exhausted"))
            return
        kind, payload = self.rounds.pop(0)
        if kind == "text":
            yield ("text", payload)
            yield ("stop_reason", "end_turn")
        elif kind == "cutoff":
            yield ("reasoning", payload)
            yield ("stop_reason", "length")
        elif kind == "nameless":
            # finish_reason tool_calls, but the provider dropped the call
            yield ("stop_reason", "tool_calls")
        else:
            yield ("stop_reason", "tool_use")
            yield ("tool_calls", payload)
        yield ("done", "")


class _AgentCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        params.save_profiles([params.Profile(
            name="fixagent", provider="openai_compat",
            base_url="http://127.0.0.1:9/v1", model="scripted", api_key="k",
            temperature=0.2, max_tokens=4096)])
        params.set_active_profile("fixagent")

    def run_turn(self, history, rounds, run_tool=None, flow=None):
        worker = agent.AgentWorker(history, flow=flow)
        worker.provider = _Scripted(rounds)
        finished, notices = [], []
        worker.turn_finished.connect(lambda ok, text: finished.append((ok, text)))
        worker.notice.connect(notices.append)
        with mock.patch.object(agent._tools, "run_tool",
                               run_tool or (lambda n, a: ("[]", None, False))):
            worker._run_turn()
        return worker, finished, notices


# ---------------------------------------------------------------------------
# agent-loop-03: an image costs a fixed amount, not len(base64)


def _shot_pair(i, image_chars, text="ok"):
    call = ToolCall(id=f"c{i}", name="get_view", arguments={})
    result = ToolResult(tool_call_id=f"c{i}", name="get_view", content=text,
                        image_b64="i" * image_chars)
    return [Message(role="assistant", content="", tool_calls=[call]),
            Message(role="user", tool_results=[result])]


def _images_left(history):
    return [r for m in history for r in m.tool_results if r.image_b64]


class ImageBudgetTest(unittest.TestCase):
    def test_prompt_photo_does_not_wipe_tool_screenshots(self):
        # a 1280 px photo re-encoded as PNG is ~1.8M base64 characters
        history = [Message(role="user", content="model this part",
                           images=["p" * 1_800_000])]
        for i in range(3):
            history += _shot_pair(i, 100_000)
            self.assertFalse(agent.stub_oversized_history(history))
            self.assertEqual(len(_images_left(history)), i + 1)
        self.assertEqual(len(history[0].images[0]), 1_800_000)

    def test_two_ordinary_screenshots_do_not_trigger_stubbing(self):
        history = [Message(role="user", content="build it")]
        for i in range(2):
            history += _shot_pair(i, 107_000, text="x" * 3000)
        before = [len(r.content) for m in history for r in m.tool_results]
        self.assertFalse(agent.stub_oversized_history(history))
        self.assertEqual(before, [len(r.content) for m in history
                                  for r in m.tool_results])
        self.assertEqual(len(_images_left(history)), 2)

    def test_single_large_screenshot_survives(self):
        history = [Message(role="user", content="look")] + _shot_pair(0, 213_000)
        self.assertFalse(agent.stub_oversized_history(history))
        self.assertEqual(len(_images_left(history)), 1)

    def test_oversized_text_still_stubs_text_and_old_images(self):
        history = [Message(role="user", content="build it")]
        for i in range(12):
            history += _shot_pair(i, 100_000, text="x" * 40_000)
        history.append(Message(role="user", content="continue"))
        self.assertTrue(agent.stub_oversized_history(history))
        sizes = [len(r.content) for m in history for r in m.tool_results]
        elided = [s for s in sizes if s <= agent.TOOL_RESULT_KEEP_CHARS + 60]
        self.assertGreaterEqual(len(elided), len(sizes) - 5)
        shots = _images_left(history)
        self.assertLessEqual(len(shots), agent.KEEP_SCREENSHOTS)
        # the freshest screenshot is the one that stays
        self.assertIs(shots[-1], history[-2].tool_results[0])

    def test_newest_screenshot_is_never_shed_for_budget(self):
        # text the stubber cannot shrink (the user's own message) must not
        # cost the model the screenshot it just asked for
        history = [Message(role="user", content="x" * 300_000)]
        for i in range(3):
            history += _shot_pair(i, 100_000)
        agent.stub_oversized_history(history)
        shots = _images_left(history)
        self.assertEqual(len(shots), 1)
        self.assertIs(shots[0], history[-1].tool_results[0])


# ---------------------------------------------------------------------------
# agent-loop-04: an empty reply is reported, not "nudged"


# Adapted: the execution nudge now lives only on the OSCAR_FLOW=0 (kill-switch) path of the loop;
# with the flow on, the finish gate replaced it (see FlowGateReplacesNudgeTest below for those behaviours). These
# classes therefore run the worker with flow=False, which is the code path that carries the nudge.
class EmptyReplyNotNudgedTest(_AgentCase):
    def _assert_reported_not_nudged(self, rounds, expect_in_error):
        history = [Message(role="user", content="add a box")]
        worker, finished, notices = self.run_turn(history, rounds, flow=False)
        self.assertEqual(len(worker.provider.requests), 1)
        self.assertEqual(notices, [])
        self.assertEqual(len(history), 1)  # no blank assistant turn, no note
        self.assertEqual(len(finished), 1)
        self.assertFalse(finished[0][0])
        self.assertIn(expect_in_error, finished[0][1])

    def test_dropped_tool_call_is_reported(self):
        self._assert_reported_not_nudged([("nameless", None)], "incomplete")

    def test_whitespace_only_reply_is_reported(self):
        self._assert_reported_not_nudged([("text", "  \n ")], "empty reply")

    def test_plan_only_reply_is_still_nudged_exactly_once(self):
        history = [Message(role="user", content="add a box")]
        worker, finished, notices = self.run_turn(history, [
            ("text", "Plan: 1. create a box"),
            ("text", "Plan: 1. create a box (again)"),
        ], flow=False)
        self.assertEqual(len(worker.provider.requests), 2)
        self.assertEqual(len([n for n in notices
                              if "without doing any CAD work" in n]), 1)
        self.assertEqual(history[1].content, "Plan: 1. create a box")
        self.assertEqual(history[2].content, agent._EXECUTION_NOTE)
        self.assertEqual(finished, [(True, "Plan: 1. create a box (again)")])


class FlowGateReplacesNudgeTest(_AgentCase):
    """With the flow on (the default) the finish gate replaced the execution nudge. The behaviours the tests above
    pin for the legacy nudge hold here in the gate's own terms: an empty or dropped reply is never a success and is
    reported with its own message, a plan-only reply never ends a work turn as an answer, and no execution note
    (the legacy nudge text) is ever injected."""

    def _end_state(self, rounds, request="add a box"):
        history = [Message(role="user", content=request)]
        worker, finished, notices = self.run_turn(history, rounds, flow=True)
        self.assertEqual(len(finished), 1)
        self.assertFalse(any(m.content == agent._EXECUTION_NOTE for m in history))
        self.assertFalse(any("without doing any CAD work" in n for n in notices))
        return worker, finished[0], history

    def test_dropped_tool_call_is_reported_not_succeeded(self):
        _w, (ok, text), _h = self._end_state([("nameless", None)] * 6)
        self.assertFalse(ok)
        self.assertIn("incomplete", text)

    def test_whitespace_only_reply_is_reported_not_succeeded(self):
        _w, (ok, text), _h = self._end_state([("text", "  \n ")] * 6)
        self.assertFalse(ok)
        self.assertIn("empty reply", text)

    def test_plan_only_reply_never_ends_a_work_turn_as_an_answer(self):
        _w, (ok, text), _h = self._end_state([("text", "Plan: 1. create a box")] * 6)
        self.assertFalse(ok)
        self.assertIn("no changes", text)

    def test_plan_after_a_cutoff_is_still_pushed_to_execute(self):
        history = [Message(role="user", content="add a box")]
        worker, finished, notices = self.run_turn(history, [
            ("cutoff", "Hmm, which bolt pattern? " * 20),
            ("text", "Plan: 1. create a box"),
            ("tools", [ToolCall(id="c1", name="create_object", arguments={})]),
            ("text", "Built the box."),
        ], flow=True)
        self.assertTrue(any("unfinished work" in n for n in notices), notices)
        self.assertEqual(finished, [(True, "Built the box.")])


# ---------------------------------------------------------------------------
# agent-loop-05: the nudge judges the user's prompt, not a synthetic note


class NudgeAfterCutoffTest(_AgentCase):
    def test_plan_after_a_cutoff_is_still_nudged_to_execute(self):
        history = [Message(role="user", content="Build a bracket with two holes")]
        worker, finished, notices = self.run_turn(history, [
            ("cutoff", "Hmm, which bolt pattern? " * 20),
            ("text", "Plan: 1. plate 2. holes"),
            ("tools", [ToolCall(id="c1", name="create_object", arguments={})]),
            ("text", "Built the bracket."),
        ], flow=False)
        self.assertTrue(any("without doing any CAD work" in n for n in notices),
                        notices)
        self.assertTrue(any(m.role == "user" and m.content == agent._EXECUTION_NOTE
                            for m in history))
        self.assertEqual(finished, [(True, "Built the bracket.")])


# ---------------------------------------------------------------------------
# agent-loop-06: only the user's own words, whole words


_REVERT_NOTE = (
    "[I reverted every CAD change from one of your earlier turns; documents "
    "closed: Bracket. Objects that turn created or edited are back to their "
    "previous state — check the workspace state below.]"
)


# What panel._revert_turn queues after a revert that only partly worked. The
# engine's error text is free-form, so it may hold brackets of its own.
_PARTIAL_REVERT_NOTE = (
    "[I tried to revert one of your earlier turns but it was only partly "
    "undone; documents closed: Bracket (the turn was only partly reverted: "
    "your own step 'Pad [Sketch]' sits on top of this turn's changes in "
    "'Bracket', so 2 of its step(s) were left in place). Part of what that "
    "turn created or edited may still be there — check the workspace state "
    "below.]"
)


class NudgeHeuristicTest(_AgentCase):
    def nudged(self, text):
        return agent.needs_execution_nudge(text, 0)

    def test_attached_file_content_is_not_the_users_request(self):
        msg = ("What does this file say?\n\n--- attached files ---\n"
               "[attached file: notes.txt]\n~~~~\nthe part list and the case "
               "design\n~~~~")
        self.assertFalse(self.nudged(msg))

    def test_revert_note_is_not_the_users_request(self):
        self.assertFalse(self.nudged(_REVERT_NOTE + "\n\nWhy did that fail?"))

    def test_mid_word_hits_are_not_work_words(self):
        for text in ("Tell me about the whole workflow", "inventory list",
                     "an event was lost", "show me the keypad object",
                     "they sat apart"):
            self.assertFalse(self.nudged(text), text)

    def test_real_work_requests_still_nudge(self):
        for text in ("Build a bracket with two holes", "make 3 boxes",
                     "modelling a gear", "Design a mounting plate",
                     "designing a hinge", "3D print this lid",
                     "cut a slot in the plate"):
            self.assertTrue(self.nudged(text), text)

    def test_re_prefixed_and_compound_work_verbs_still_nudge(self):
        # llm-06: plain substring matching caught these; the word-boundary
        # rewrite must keep them
        for text in ("rebuild it with 3 mm walls", "Please redesign the lid",
                     "redesign the lid with a snap fit", "remodel this gear",
                     "re-create the lid", "recreate it", "redraw it", "reshape it",
                     "resketch the profile", "redo it with a 2 mm wall",
                     "cutout for the USB-C port, 9x3.5 mm", "add a cut-out for the switch"):
            self.assertTrue(self.nudged(text), text)

    def test_re_prefix_does_not_bring_back_mid_word_hits(self):
        for text in ("replace the battery", "prevent the event", "reapply the inventory",
                     "they sat apart", "show me the keypad object", "an event was lost",
                     "I whole-heartedly agree", "retire the old revision",
                     "a reply about the department"):
            self.assertFalse(self.nudged(text), text)

    def test_work_request_behind_app_notes_still_nudges(self):
        self.assertTrue(self.nudged(_REVERT_NOTE + "\n\nBuild a plate"))
        self.assertTrue(self.nudged(
            agent._TRIMMED_NOTE + "Create a housing for the Pi"))
        self.assertTrue(self.nudged(
            "Build a plate\n\n--- attached files ---\n[attached file: a.txt]"))

    def test_only_known_app_notes_are_stripped(self):
        # user text that merely starts with a bracket is still the request
        self.assertTrue(self.nudged("[Pi 4] build a housing"))
        # several pending notes are joined by newlines
        self.assertFalse(self.nudged(
            _REVERT_NOTE + "\n" + _REVERT_NOTE + "\n\nWhy did that fail?"))
        self.assertFalse(self.nudged(_REVERT_NOTE))

    def test_partial_revert_note_is_not_the_users_request(self):
        # llm-05: it says "created", is longer than a short question, and used
        # to be counted as the user's own words
        self.assertFalse(self.nudged(_PARTIAL_REVERT_NOTE + "\n\nwhy did that fail?"))
        self.assertFalse(self.nudged(_PARTIAL_REVERT_NOTE + "\n\nThanks"))
        self.assertFalse(self.nudged(_PARTIAL_REVERT_NOTE))
        self.assertFalse(self.nudged(
            _REVERT_NOTE + "\n" + _PARTIAL_REVERT_NOTE + "\n\nWhy did that fail?"))

    def test_partial_revert_note_without_documents_or_brackets(self):
        note = ("[I tried to revert one of your earlier turns but it was only partly "
                "undone (some changes could not be undone). Part of what that turn "
                "created or edited may still be there — check the workspace state "
                "below.]")
        self.assertFalse(self.nudged(note + "\n\nWhat happened to the lid"))
        self.assertTrue(self.nudged(note + "\n\nBuild a plate"))

    def test_work_request_behind_a_partial_revert_note_still_nudges(self):
        self.assertTrue(self.nudged(_PARTIAL_REVERT_NOTE + "\n\nRebuild it with a 3 mm wall"))

    def test_the_panel_still_emits_the_notes_the_agent_strips(self):
        # the two modules share these texts only by value; a reworded panel
        # note would silently go back to being read as the user's request
        # Adapted: the panel is split into panel*.py modules (the revert notes live in panel_revert.py).
        oscar_dir = os.path.dirname(agent.__file__)
        text = ""
        for name in sorted(os.listdir(oscar_dir)):
            if name.startswith("panel") and name.endswith(".py"):
                with open(os.path.join(oscar_dir, name), encoding="utf-8") as fh:
                    text += fh.read()
        for prefix in agent._APP_NOTE_PREFIXES:
            if prefix.startswith("[I "):
                self.assertIn(prefix, text)

    def test_question_about_an_attached_file_is_answered_in_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "notes.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("supplier notes: the part list, the case drawing")
            content = attachments.build_user_message(
                "What does this file say", [attachments.make(path)])
            history = [Message(role="user", content=content)]
            worker, finished, notices = self.run_turn(
                history, [("text", "It lists supplier notes.")])
        self.assertEqual(notices, [])
        self.assertEqual(len(worker.provider.requests), 1)
        self.assertEqual(finished, [(True, "It lists supplier notes.")])


# ---------------------------------------------------------------------------
# agent-loop-02: provider construction must not run on the GUI thread


class ProviderSetupThreadTest(_AgentCase):
    def _run_threaded(self, make_provider):
        """Start the worker for real; DirectConnection delivers the signal on
        the worker thread (no event loop is spinning under freecadcmd)."""
        with mock.patch.object(agent, "make_provider", make_provider):
            worker = agent.AgentWorker([Message(role="user", content="hello")])
            finished = []
            worker.turn_finished.connect(
                lambda ok, text: finished.append((ok, text)),
                QtCore.Qt.DirectConnection)
            worker.start()
            self.assertTrue(worker.wait(10_000))
        return worker, finished

    def setUp(self):
        self.seen = []

    def test_constructor_does_not_build_the_provider(self):
        def make(profile):
            self.seen.append(threading.current_thread())
            return _Scripted([("text", "hi")])

        with mock.patch.object(agent, "make_provider", make):
            worker = agent.AgentWorker([Message(role="user", content="hello")])
        self.assertEqual(self.seen, [])
        self.assertIsNone(worker.provider)

    def test_provider_is_built_once_off_the_calling_thread(self):
        def make(profile):
            self.seen.append(threading.current_thread())
            return _Scripted([("text", "hi")])

        _w, finished = self._run_threaded(make)
        self.assertEqual(len(self.seen), 1)
        self.assertIsNot(self.seen[0], threading.current_thread())
        self.assertEqual(finished, [(True, "hi")])

    def test_provider_setup_failure_ends_the_turn_with_the_message(self):
        def make(profile):
            raise RuntimeError("provider setup failed: boom")

        worker, finished = self._run_threaded(make)
        self.assertEqual(finished, [
            (False, "provider setup failed: boom")])
        self.assertIsNone(worker.provider)
        self.assertEqual(len(worker.history), 1)

    def test_an_injected_provider_is_kept(self):
        # tests (and gui_smoke) assign worker.provider before run()
        injected = _Scripted([("text", "ok")])
        with mock.patch.object(agent, "make_provider") as make:
            worker = agent.AgentWorker([Message(role="user", content="hello")])
            worker.provider = injected
            finished = []
            worker.turn_finished.connect(
                lambda ok, text: finished.append((ok, text)),
                QtCore.Qt.DirectConnection)
            worker.start()
            self.assertTrue(worker.wait(10_000))
        make.assert_not_called()
        self.assertIs(worker.provider, injected)
        self.assertEqual(finished, [(True, "ok")])


if __name__ == "__main__":
    unittest.main()
