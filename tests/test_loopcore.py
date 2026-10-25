"""oscar.loopcore: the shared agent-loop helpers (pure Python, runs without FreeCAD)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import loopcore, stopreason  # noqa: E402
from oscar.providers import Message, ToolCall, ToolResult  # noqa: E402


class StopReasonReexportTest(unittest.TestCase):
    def test_one_mapping_shared_with_the_providers(self):
        self.assertIs(loopcore.normalize_stop_reason, stopreason.normalize_stop_reason)
        self.assertIs(loopcore.is_cut_off, stopreason.is_cut_off)

    def test_cut_off_covers_both_wire_names(self):
        self.assertTrue(loopcore.is_cut_off("length"))
        self.assertTrue(loopcore.is_cut_off("max_tokens"))
        self.assertFalse(loopcore.is_cut_off("end_turn"))
        self.assertFalse(loopcore.is_cut_off(None))
        self.assertFalse(loopcore.is_cut_off(""))

    def test_normalization_table(self):
        for raw, norm in (("stop", "end_turn"), ("tool_calls", "tool_use"), ("tool_use", "tool_use"),
                          ("length", "max_tokens"), ("", "unknown"), (None, "unknown"), ("weird", "unknown"),
                          ("refusal", "refusal"), ("content_filter", "content_filter")):
            self.assertEqual(loopcore.normalize_stop_reason(raw), norm, raw)


class NotesTest(unittest.TestCase):
    def test_fixed_notes_keep_their_pinned_words(self):
        self.assertIn("output-token limit", loopcore.CUTOFF_NOTE)
        self.assertIn("Tool budget", loopcore.BUDGET_NOTE)
        text = loopcore.WATCHDOG_NOTE.format(chars=1234, secs=95, now='S5 "Shell"')
        self.assertIn("1234 characters", text)
        self.assertIn("95 seconds", text)
        self.assertIn('S5 "Shell"', text)
        self.assertIn("under 300 words", text)

    def test_make_note_is_a_synthetic_user_message(self):
        note = loopcore.make_note("hello", "gate")
        self.assertEqual((note.role, note.content, note.tag), ("user", "hello", "gate"))
        self.assertTrue(note.synthetic)
        self.assertFalse(note.tool_results)


class ArgumentProblemTest(unittest.TestCase):
    def test_clean_call_has_no_problem(self):
        self.assertEqual(loopcore.args_problem(ToolCall(id="1", name="x", arguments={"a": 1})), "")

    def test_args_error_wins(self):
        call = ToolCall(id="1", name="x", arguments={"_raw": "{"}, args_error="truncated at char 3")
        self.assertEqual(loopcore.args_problem(call), "truncated at char 3")

    def test_legacy_raw_marker_is_a_problem(self):
        call = ToolCall(id="1", name="x", arguments={"_raw": '{"code": "pri'})
        self.assertTrue(loopcore.args_problem(call))

    def test_object_without_the_fields_is_tolerated(self):
        class Bare:
            name = "x"

        self.assertEqual(loopcore.args_problem(Bare()), "")

    def test_message_says_not_valid_json_with_the_right_guidance(self):
        truncated = ToolCall(id="1", name="execute_code", arguments={"_raw": "{"}, args_error="truncated (cut off)")
        text = loopcore.bad_arguments_message(truncated)
        self.assertIn("not valid JSON", text)
        self.assertIn("under 40 lines", text)
        malformed = ToolCall(id="2", name="x", arguments={"_raw": "{a}"}, args_error="Expecting property name")
        text = loopcore.bad_arguments_message(malformed)
        self.assertIn("not valid JSON", text)
        self.assertIn("double quotes", text)
        # a cut-off stop reason makes any bad call a truncation
        text = loopcore.bad_arguments_message(malformed, "max_tokens")
        self.assertIn("under 40 lines", text)

    def test_message_for_a_legacy_call_without_reason(self):
        text = loopcore.bad_arguments_message(ToolCall(id="1", name="x", arguments={"_raw": "{"}))
        self.assertIn("not valid JSON", text)
        self.assertIn("could not be decoded", text)


class CallChatTest(unittest.TestCase):
    class Legacy:
        def __init__(self):
            self.seen = None

        def chat(self, system, messages, tools, should_stop=None):
            self.seen = (system, list(messages), list(tools), should_stop)
            yield ("done", "end_turn")

    class Modern:
        def __init__(self):
            self.kwargs = None

        def chat(self, system, messages, tools, should_stop=None, *, tool_choice=None, reasoning=None, max_tokens=None):
            self.kwargs = {"tool_choice": tool_choice, "reasoning": reasoning, "max_tokens": max_tokens}
            yield ("done", "end_turn")

    class Wide:
        def __init__(self):
            self.kwargs = None

        def chat(self, system, messages, tools, should_stop=None, **kwargs):
            self.kwargs = kwargs
            yield ("done", "end_turn")

    def test_legacy_provider_gets_only_its_four_parameters(self):
        provider = self.Legacy()
        stop = lambda: False  # noqa: E731
        events = list(loopcore.call_chat(provider, "sys", [1], [2], stop, tool_choice="required", reasoning="off"))
        self.assertEqual(events, [("done", "end_turn")])
        self.assertEqual(provider.seen, ("sys", [1], [2], stop))

    def test_options_the_provider_accepts_are_passed(self):
        provider = self.Modern()
        list(loopcore.call_chat(provider, "s", [], [], None, tool_choice="none", reasoning="low", max_tokens=100,
                                parallel_tool_calls=False))
        self.assertEqual(provider.kwargs, {"tool_choice": "none", "reasoning": "low", "max_tokens": 100})

    def test_none_options_mean_provider_default(self):
        provider = self.Wide()
        list(loopcore.call_chat(provider, "s", [], [], None, tool_choice=None, reasoning="off"))
        self.assertEqual(provider.kwargs, {"reasoning": "off"})

    def test_var_keyword_provider_gets_everything(self):
        provider = self.Wide()
        list(loopcore.call_chat(provider, "s", [], [], None, anything=1))
        self.assertEqual(provider.kwargs, {"anything": 1})

    def test_closing_the_generator_closes_the_providers(self):
        closed = []

        class Slow:
            def chat(self, system, messages, tools, should_stop=None):
                try:
                    yield ("text", "a")
                    yield ("text", "b")
                finally:
                    closed.append(True)

        gen = loopcore.call_chat(Slow(), "s", [], [], None)
        self.assertEqual(next(gen), ("text", "a"))
        gen.close()
        self.assertEqual(closed, [True])


class SyntheticAndRequestTest(unittest.TestCase):
    def test_is_synthetic(self):
        self.assertTrue(loopcore.is_synthetic(Message(role="user", content="x", synthetic=True)))
        self.assertFalse(loopcore.is_synthetic(Message(role="user", content="x")))
        self.assertFalse(loopcore.is_synthetic(object()))

    def test_plain_typed_text(self):
        history = [Message(role="user", content="Build a flange")]
        self.assertEqual(loopcore.real_user_request(history), "Build a flange")

    def test_attachments_and_workspace_state_are_cut(self):
        content = ("what is in this document?\n\n--- attached files ---\nbracket for a motor with 4 holes\n"
                   "<workspace_state>\nOpen documents:\n</workspace_state>")
        self.assertEqual(loopcore.real_user_request([Message(role="user", content=content)]),
                         "what is in this document?")

    def test_workspace_state_without_attachments(self):
        content = "add a box\n\n<workspace_state>\n- Doc: 2 objects\n</workspace_state>"
        self.assertEqual(loopcore.real_user_request([Message(role="user", content=content)]), "add a box")

    def test_leading_revert_note_is_dropped(self):
        content = ("[I reverted every CAD change you made after that point; check the workspace state below.]\n\n"
                   "what is in this document?")
        self.assertEqual(loopcore.real_user_request([Message(role="user", content=content)]),
                         "what is in this document?")

    def test_several_note_paragraphs_are_dropped(self):
        content = "[The user rewound the chat.]\n\n[I reverted every change.]\n\nmake the walls thicker"
        self.assertEqual(loopcore.real_user_request([Message(role="user", content=content)]),
                         "make the walls thicker")

    def test_a_lone_bracketed_text_is_the_users_own(self):
        self.assertEqual(loopcore.real_user_request([Message(role="user", content="[sic]")]), "[sic]")

    def test_governor_and_phase_lines_are_stripped(self):
        content = "continue\n[governor] round 3/30 · BUILD\n[phase] BUILD: build one step per call"
        self.assertEqual(loopcore.real_user_request([Message(role="user", content=content)]), "continue")

    def test_synthetic_and_tool_result_messages_are_skipped(self):
        history = [
            Message(role="user", content="build a bracket"),
            Message(role="assistant", content="ok", tool_calls=[ToolCall(id="1", name="t")]),
            Message(role="user", tool_results=[ToolResult(tool_call_id="1", name="t", content="done")]),
            Message(role="assistant", content="I stopped"),
            Message(role="user", content="[finish-gate] you stopped", synthetic=True, tag="gate"),
        ]
        self.assertEqual(loopcore.real_user_request(history), "build a bracket")

    def test_no_user_message_gives_empty_text(self):
        self.assertEqual(loopcore.real_user_request([]), "")
        self.assertEqual(loopcore.real_user_request(None), "")
        self.assertEqual(loopcore.real_user_request([Message(role="assistant", content="hi")]), "")


class SmallUtilitiesTest(unittest.TestCase):
    def test_args_hash_is_stable_and_discriminating(self):
        a = loopcore.args_hash("create_object", {"x": 1, "y": [1, 2]})
        self.assertEqual(a, loopcore.args_hash("create_object", {"y": [1, 2], "x": 1}))   # key order does not matter
        self.assertNotEqual(a, loopcore.args_hash("create_object", {"x": 2, "y": [1, 2]}))
        self.assertNotEqual(a, loopcore.args_hash("edit_object", {"x": 1, "y": [1, 2]}))
        self.assertEqual(len(a), 12)
        int(a, 16)

    def test_args_hash_accepts_unserializable_values(self):
        self.assertEqual(len(loopcore.args_hash("t", {"o": object()})), 12)
        self.assertEqual(len(loopcore.args_hash("t", None)), 12)

    def test_clip_leaves_short_text_alone(self):
        self.assertEqual(loopcore.clip("short", 100), "short")
        self.assertEqual(loopcore.clip("x" * 100, 100), "x" * 100)

    def test_clip_keeps_head_and_tail_with_a_marker(self):
        text = "".join(str(i % 10) for i in range(5000))
        out = loopcore.clip(text, 1000)
        self.assertTrue(out.startswith(text[:670]))
        self.assertTrue(out.endswith(text[-330:]))
        self.assertIn("chars omitted", out)
        self.assertLess(len(out), 1100)

    def test_clip_non_positive_limit_and_non_strings(self):
        self.assertEqual(loopcore.clip("abc", 0), "abc")
        self.assertEqual(loopcore.clip(12345, 3).count("omitted"), 1)

    def test_first_line(self):
        self.assertEqual(loopcore.first_line("\n  \nhello\nworld"), "hello")
        self.assertEqual(loopcore.first_line(""), "")
        self.assertEqual(loopcore.first_line(None), "")
        long = loopcore.first_line("x" * 500, 20)
        self.assertEqual(len(long), 20)
        self.assertTrue(long.endswith("…"))




if __name__ == "__main__":
    unittest.main()
