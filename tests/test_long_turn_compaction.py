"""Long turns on a small context window: a 30-round build and its continuations keep the task.

The old agent compacted by rewriting the stored history in place, so after two or three "continue" turns the model no
longer saw what it had been asked for. The new loop derives a REQUEST VIEW from the append-only history and prepends a
pinned memory (original request, plan, parameters, decisions, reference) once rounds had to be dropped.

Here a REAL ``AgentWorker`` (real ``run_tool``, real Project, real documents, scripted model, injected clock) builds a
30-round model on a 12k-token window and is then continued three times through the SAME chat history, exactly like the
panel does ("continue" appended to the stored messages, the ledger carried over). After every turn the view of the LAST
provider request must still contain the original request, the plan, the parameters from ``params.json`` and the
decisions from the notes, and must never contain the word "continue" as the task.
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

from oscar import agent_ctx, context  # noqa: E402
from oscar.models import ModelCaps  # noqa: E402
from oscar.providers import Message  # noqa: E402
from support.realflow import RealFlowCase  # noqa: E402
from support.scripted_llm import call, calls, say  # noqa: E402

REQUEST = "Build a housing for a Raspberry Pi 4 with a snap-fit lid in ABS, 3 mm walls and twelve vent slots."
CAPS = ModelCaps(context_window=12000, max_output=1024)
FILLER = "Continuing with the next compartment of the tray; checking the clearance to its neighbour first. " * 14
BIG_CODE = "x = 1  # " + "padding to make the script long " * 60
PARAMS = [{"name": "wall_t", "value": 3.0, "unit": "mm", "source": "user_requirement"},
          {"name": "compartments", "value": 12, "unit": "", "source": "user_requirement"},
          {"name": "lid_clearance", "value": 0.3, "unit": "mm", "source": "assumption"}]


def view_text(messages) -> str:
    return "\n".join(m.content or "" for m in messages)


class LongBuildCase(RealFlowCase):
    """One chat: ``self.project``, ``self.history`` (shared by every turn) and the ledger carried between turns."""

    def setUp(self):
        super().setUp()
        self.doc = self.doc_name("Lt")
        self.project = self.new_project("Pi 4 housing", chat_id="20261002-101010-c0c0c0")
        with open(self.project.path("params.json"), "w", encoding="utf-8") as f:
            json.dump({"schema": 1, "params": PARAMS}, f)
        self.project.note("decision", "wall_t is 3.0 mm because the user asked for 3 mm walls")
        self.history = [Message(role="user", content=REQUEST)]
        self.ledger = None
        self.views: list[dict] = []
        self.counter = 0
        self.results = []

    def build_rounds(self, n: int, *, first: bool = False) -> list:
        """``n`` tool rounds (a plate each, every fifth one by script) with filler text, then the answer."""
        script = []
        if first:
            script.append(calls(call("create_document", name=self.doc), text=FILLER))
            n -= 1
        for _ in range(n):
            self.counter += 1
            if self.counter % 5 == 0:
                code = self.code(self.doc, f"o = d.addObject('Part::Feature', 'S{self.counter}')\n"
                                           f"o.Shape = Part.makeBox(5, 5, 5)\nd.recompute()\n# {BIG_CODE}")
                script.append(calls(call("execute_code", code=code, include_screenshot=False), text=FILLER))
            else:
                script.append(calls(self.box(self.doc, f"C{self.counter}", Length=5, Width=5, Height=5), text=FILLER))
        # the model says "done" while required steps of the plan are still open: the gate nudges it, the turn ends
        # unverified and the NEXT "continue" resumes this very plan (a finished ledger would start a new one)
        script.extend(say("That is all for now.") for _ in range(8))
        return script

    def run_one(self, script, prompt: str):
        """Appends ``prompt`` to the shared history (the panel's send) and runs one turn over it."""
        if prompt != REQUEST:
            self.history.append(Message(role="user", content=prompt))
        events: list = []
        with mock.patch.object(agent_ctx, "log_event", lambda kind, **f: events.append((kind, f))):
            worker = self.make_worker(prompt, script, project=self.project, caps=CAPS, history=self.history,
                                      ledger=self.ledger)
            result = worker.run_turn()
        self.ledger = worker.ledger
        self.views.append({"events": [f for k, f in events if k == "context_view"], "requests": list(self.provider.requests),
                           "result": result})
        self.results.append(result)
        return result

    def last_view(self, turn: int) -> list:
        return self.views[turn]["requests"][-1].messages

    def drive(self, continuations: int = 3, rounds: int = 30, more: int = 12):
        self.run_one(self.build_rounds(rounds, first=True), REQUEST)
        for _ in range(continuations):
            self.run_one(self.build_rounds(more), "continue")


class ThirtyRoundBuildTest(LongBuildCase):
    """The first turn: 30 rounds on a window that cannot hold them (one run, many checks: a build of 30 real rounds is slow)."""

    def objects(self):
        import FreeCAD as App
        return [o.Name for o in App.getDocument(self.doc).Objects]

    def test_the_build_reaches_a_non_failed_outcome_with_all_the_objects(self):
        result = self.run_one(self.build_rounds(30, first=True), REQUEST)
        self.assertEqual(result.outcome, "capped", result.text[:200])        # 30 tool rounds, steps still open
        self.assertGreaterEqual(len(self.project.read_steps()), 28)
        self.assertGreaterEqual(len([o for o in self.objects() if o.startswith(("C", "S"))]), 25)

    def lt_check_rounds_really_were_dropped_so_the_test_exercises_compaction(self):
        dropped = [f["dropped_rounds"] for f in self.views[0]["events"]]
        self.assertGreater(max(dropped), 5, dropped)
        self.assertEqual(len(dropped), len(self.views[0]["requests"]))

    def lt_check_the_view_of_the_last_round_still_carries_the_original_request_plan_and_params(self):
        view = self.last_view(0)
        self.assertEqual(view[0].role, "user")
        first = view[0].content
        self.assertIn("<task_memory>", first)
        self.assertIn(REQUEST, first)
        self.assertIn("PLAN:", first)
        self.assertIn("wall_t=3 mm", first)
        self.assertIn("compartments=12", first)
        self.assertIn("DECISIONS:", first)

    def lt_check_the_stored_history_keeps_every_message_while_the_view_shrinks(self):
        request_view = self.last_view(0)
        self.assertGreater(len(self.history), len(request_view) + 20)
        self.assertEqual(self.history[0].content.count(REQUEST), 1)
        self.assertEqual(self.history[0].role, "user")

    def lt_check_the_view_stays_within_the_window_budget(self):
        budget = int(0.70 * CAPS.context_window) - min(CAPS.max_output, 8192)
        for f in self.views[0]["events"]:
            self.assertLessEqual(f["after"], budget * 1.15, f)

    def lt_check_old_scripts_are_elided_in_the_view_but_kept_verbatim_on_disk(self):
        text = view_text(self.last_view(0))
        self.assertNotIn(BIG_CODE, text)
        self.assertGreater(max(f["elided_scripts"] for f in self.views[0]["events"]), 0)
        scripts = [s for s in self.project.read_steps() if s.script]
        self.assertTrue(scripts)
        for s in scripts:
            with open(self.project.path(s.script), encoding="utf-8") as f:
                self.assertIn(BIG_CODE, f.read())

    def lt_check_every_view_alternates_roles_and_starts_with_the_user(self):
        for request in self.views[0]["requests"]:
            self.assertEqual(request.messages[0].role, "user")
            self.assertEqual(context.alternation_problems(request.messages), [])

    def lt_check_the_system_prompt_is_one_string_for_the_whole_turn(self):
        self.assertEqual(len({r.system for r in self.views[0]["requests"]}), 1)


    def test_the_view_of_a_30_round_turn_keeps_the_task_and_stays_in_budget(self):
        self.run_one(self.build_rounds(30, first=True), REQUEST)
        for name in sorted(n for n in dir(self) if n.startswith("lt_check_")):
            with self.subTest(check=name):
                getattr(self, name)()


class ContinuationsTest(LongBuildCase):
    """Three continuations of the same chat: the task survives every one of them."""

    def lt_check_every_turn_ends_without_failing(self):
        for i, result in enumerate(self.results):
            with self.subTest(turn=i):
                self.assertNotEqual(result.outcome, "failed", result.text[:200])
                self.assertTrue(result.ok)

    def lt_check_the_original_request_is_in_the_view_of_every_turn(self):
        for turn in range(4):
            with self.subTest(turn=turn):
                self.assertIn(REQUEST, view_text(self.last_view(turn)))

    def lt_check_continue_is_never_pinned_as_the_task(self):
        for turn in range(1, 4):
            with self.subTest(turn=turn):
                memory = self.last_view(turn)[0].content
                self.assertIn("<task_memory>", memory)
                head = memory.split("<task_memory>")[1][:400]
                self.assertIn(REQUEST, head)
                self.assertNotIn("REQUEST: continue", memory)

    def lt_check_the_ledger_is_carried_so_the_plan_does_not_restart(self):
        self.assertEqual(self.last_view(0)[0].content.count("PLAN:"), 1)
        self.assertEqual(self.ledger.request, REQUEST)
        self.assertTrue(self.ledger.has_open_required_steps())
        for turn in range(1, 4):
            self.assertIn("PLAN:", self.last_view(turn)[0].content)

    def lt_check_the_parameters_and_decisions_survive_every_continuation(self):
        for turn in range(4):
            with self.subTest(turn=turn):
                memory = self.last_view(turn)[0].content
                for needle in ("wall_t=3 mm", "compartments=12", "lid_clearance=0.3 mm", "wall_t is 3.0 mm"):
                    self.assertIn(needle, memory)

    def lt_check_the_stored_history_is_append_only_with_the_flow_on(self):
        """The new flow never rewrites the stored messages (``compact_history`` is the kill-switch path only); the request
        survives in the first message and in the pinned memory."""
        first = self.history[0].content
        self.assertTrue(first.startswith(REQUEST), first[:120])
        self.assertEqual(self.history[-1].role, "assistant")
        self.assertGreater(len(self.history), 40)
        self.assertIn(REQUEST, self.last_view(3)[0].content)

    def lt_check_the_memory_is_bounded_and_does_not_grow_each_turn(self):
        sizes = [len(self.last_view(t)[0].content) for t in range(4)]
        self.assertLess(max(sizes), 9000, sizes)
        self.assertLess(sizes[3], sizes[1] * 1.6, sizes)

    def lt_check_the_project_recorded_every_step_of_all_four_turns(self):
        steps = self.project.read_steps()
        self.assertGreaterEqual(len(steps), 29 + 3 * 12 - 4)
        self.assertEqual([s.index for s in steps], list(range(1, len(steps) + 1)))
        self.assertTrue(all(s.ok for s in steps))

    def last_lt_check_new_notes_after_the_first_turn_reach_the_next_memory(self):
        self.project.note("decision", "the lid slides on two 1 mm rails")
        self.run_one(self.build_rounds(6), "continue")
        self.assertIn("the lid slides on two 1 mm rails", self.last_view(4)[0].content)

    def lt_check_a_continuation_uses_the_calibrated_estimate_not_an_unbounded_view(self):
        for turn in range(1, 4):
            for f in self.views[turn]["events"]:
                self.assertLess(f["after"], CAPS.context_window, f)

    def test_three_continuations_keep_the_task(self):
        self.drive()
        for name in sorted(n for n in dir(self) if n.startswith("lt_check_")):
            with self.subTest(check=name):
                getattr(self, name)()
        self.last_lt_check_new_notes_after_the_first_turn_reach_the_next_memory()


class NoProjectContinuationTest(RealFlowCase):
    """Without a Project the memory is just the request and the plan: that alone must survive."""

    def test_the_request_survives_a_long_turn_and_a_continuation_without_a_project(self):
        doc = self.doc_name("Lp")
        history = [Message(role="user", content=REQUEST)]
        script = [calls(call("create_document", name=doc), text=FILLER)]
        script += [calls(self.box(doc, f"P{i}", Length=4, Width=4, Height=4), text=FILLER) for i in range(26)]
        script.extend(say("Done.") for _ in range(6))
        worker = self.make_worker(REQUEST, script, caps=CAPS, history=history)
        worker.run_turn()
        self.assertIn(REQUEST, view_text(self.provider.requests[-1].messages))
        history.append(Message(role="user", content="continue"))
        script2 = [calls(self.box(doc, f"Q{i}", Length=4, Width=4, Height=4), text=FILLER) for i in range(8)] + [say("More.")] * 6
        worker2 = self.make_worker("continue", script2, caps=CAPS, history=history, ledger=worker.ledger)
        worker2.run_turn()
        self.assertIn(REQUEST, view_text(self.provider.requests[-1].messages))


if __name__ == "__main__":
    unittest.main()
