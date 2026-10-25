"""Agent execution-contract tests: a work prompt must end in tool use
(the no-tool plan answer is nudged once into execution), questions are
left alone, and oversized histories are stubbed mid-turn without breaking
the tool-call/result pairing."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import agent  # noqa: E402
from oscar.providers import Message, ToolCall, ToolResult  # noqa: E402


class ExecutionNudgeTest(unittest.TestCase):
    def test_work_request_without_tools_is_nudged(self):
        self.assertTrue(agent.needs_execution_nudge(
            "Build a housing for a Raspberry Pi 4 with vent slots", 0))

    def test_question_is_left_alone(self):
        self.assertFalse(agent.needs_execution_nudge(
            "What is the difference between pad and pocket?", 0))

    def test_short_factual_prompt_left_alone(self):
        self.assertFalse(agent.needs_execution_nudge("what units?", 0))

    def test_no_nudge_after_tools_ran(self):
        self.assertFalse(agent.needs_execution_nudge(
            "build a bracket", 3))

    def test_strategy_carries_the_contract(self):
        self.assertIn("EXECUTION CONTRACT", agent.STRATEGY)
        self.assertIn("never end the turn at", agent.STRATEGY)

    def test_nudge_note_demands_step_by_step_execution(self):
        self.assertIn("step by step", agent._EXECUTION_NOTE.lower())
        self.assertIn("tool", agent._EXECUTION_NOTE.lower())


class HistoryStubbingTest(unittest.TestCase):
    def _history(self, rounds=12, result_chars=40_000):
        history = [Message(role="user",
                           content="Build a housing for the Pi 4.")]
        for i in range(rounds):
            calls = [ToolCall(id=f"c{i}", name="create_object",
                              arguments={"doc": "Doc", "obj_type": "Part::Box"})]
            history.append(Message(role="assistant", content=f"step {i}",
                                   tool_calls=calls))
            results = [ToolResult(tool_call_id=f"c{i}", name="create_object",
                                  content="x" * result_chars,
                                  image_b64="i" * 50_000)]
            history.append(Message(role="user", tool_results=results))
        history.append(Message(role="user", content="continue"))
        return history

    def test_small_history_untouched(self):
        history = self._history(rounds=2)
        before = [len(r.content) for m in history for r in m.tool_results]
        self.assertFalse(agent.stub_oversized_history(history))
        after = [len(r.content) for m in history for r in m.tool_results]
        self.assertEqual(before, after)

    def test_oversized_history_stubbed_pairing_intact(self):
        history = self._history()  # ~12 * 90k chars — way over budget
        total_before = sum(agent._message_size(m) for m in history)
        self.assertGreater(total_before, agent.CONTEXT_CHAR_BUDGET)
        self.assertTrue(agent.stub_oversized_history(history,
                                                     budget=100_000))
        # pairing: every tool call id has its matching result
        called = [c.id for m in history for c in m.tool_calls]
        answered = {r.tool_call_id for m in history
                    for r in m.tool_results}
        self.assertEqual(set(called), answered)
        # old results truncated (the newest rounds stay verbatim within
        # the budget — that is the design), old images stripped...
        sizes = [len(r.content) for m in history for r in m.tool_results]
        stubbed = [s for s in sizes
                   if s <= agent.TOOL_RESULT_KEEP_CHARS + 60]
        self.assertGreaterEqual(len(stubbed), len(sizes) - 3)
        total_after = sum(agent._message_size(m) for m in history)
        self.assertLess(total_after, 2 * 100_000)
        images = [len(r.image_b64) for m in history
                  for r in m.tool_results if r.image_b64]
        self.assertLessEqual(len(images), agent.KEEP_SCREENSHOTS)
        # ...newest message untouched
        self.assertEqual(history[-1].content, "continue")

    def test_estimation_counts_images(self):
        # An image costs a fixed amount, not len(base64): the size of the
        # encoding says nothing about the tokens the model spends on it.
        for chars in (100_000, 2_000_000):
            m = Message(role="user", tool_results=[
                ToolResult(tool_call_id="x", name="get_view", content="ok",
                           image_b64="z" * chars)])
            self.assertEqual(agent._message_size(m),
                             len("ok") + agent.IMAGE_COST_CHARS)
            photo = Message(role="user", content="hi", images=["z" * chars])
            self.assertEqual(agent._message_size(photo),
                             len("hi") + agent.IMAGE_COST_CHARS)
        no_image = Message(role="user", tool_results=[
            ToolResult(tool_call_id="x", name="get_view", content="ok")])
        self.assertEqual(agent._message_size(no_image), len("ok"))


class MergeGuardTest(unittest.TestCase):
    """Scripted fusing that removes parts from the tree must be called out
    in the tool result, so the next round restores per-part objects."""

    def _doc_with_parts(self, n=2):
        import FreeCAD as App

        name = f"MergeGuard{n}"
        if name in App.listDocuments():
            App.closeDocument(name)
        doc = App.newDocument(name)
        for i in range(n):
            doc.addObject("Part::Box", f"Part{i}")
        doc.recompute()
        return doc

    def test_losing_parts_is_flagged(self):
        import FreeCAD as App

        from oscar.cad import get_engine

        doc = self._doc_with_parts(2)
        engine = get_engine()
        code = ("import FreeCAD as App\n"
                "doc = App.getDocument('MergeGuard2')\n"
                "fused = doc.Part0.Shape.fuse(doc.Part1.Shape)\n"
                "doc.removeObject('Part1')\n"
                "doc.Part0.Shape = fused\n")
        result = engine.execute_code(code)
        self.assertTrue(result.get("success"), result)
        self.assertIn("LOST shape object", result.get("message", ""))
        self.assertIn("Part1", result.get("message", ""))
        App.closeDocument(doc.Name)

    def test_adding_a_part_is_not_flagged(self):
        import FreeCAD as App

        from oscar.cad import get_engine

        doc = self._doc_with_parts(1)
        engine = get_engine()
        result = engine.execute_code(
            "import FreeCAD as App\n"
            "doc = App.getDocument('MergeGuard1')\n"
            "doc.addObject('Part::Cylinder', 'Added')\n")
        self.assertTrue(result.get("success"))
        self.assertNotIn("LOST shape object", result.get("message", ""))
        App.closeDocument(doc.Name)


if __name__ == "__main__":
    unittest.main()
