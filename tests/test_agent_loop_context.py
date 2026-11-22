"""oscar.agent: the context wiring driven by a scripted provider.

Covers what ``oscar.agent_ctx`` adds to the loop: the static system prompt of a turn (``prompt.build_system_prompt`` with
the turn-start intent's tool names, the lite switch and the playbooks), the per-round tool list (phase / intent /
lite, visibility that only grows), the derived request view (history never mutated, governor lines only in the newest
results message, the pinned memory only when rounds were dropped, calibration), the floor rule, the round-0 plan
line, the model caps (cache-only in the constructor, metadata in ``run()``), the reasoning / forced tool-choice
guard, the health baseline and the default validator. Fakes are the ones of ``test_agent_loop`` (``FakeTools``,
``StubProject``, ``LoopCase``); the projects in ``ProjectWiringTest`` are REAL ``oscar.project.Project`` objects.
"""

import json
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import agent, agent_ctx, context, flow, loopcore, models, prompt, tools  # noqa: E402
from oscar.flow import AgentConfig, Ledger  # noqa: E402
from oscar.models import ModelCaps  # noqa: E402
from oscar.providers import Message, ToolResult  # noqa: E402
from support.fakes import fake_validation, temp_project  # noqa: E402
from support.scripted_llm import Round, call, calls, say  # noqa: E402
from test_agent_loop import (FakeTools, LoopCase, PI_PROMPT, StubProject, pi_script, pi_tools,  # noqa: E402
                             profile)

BIG = "x" * 9000          # a tool result that eats context quickly


def lenient(script):
    """A callable script: the given rounds in order, then a plain ``say`` forever (for tests that only look at the first
    request and must not die on the gate nudges that follow)."""
    rounds = list(script)

    def next_round(record):
        return rounds.pop(0) if rounds else say("(done)")

    return next_round


def names_of(request) -> set[str]:
    return set(request.tools)


def stop_info(prompt_tokens: int, reason: str = "end_turn") -> tuple:
    return ("stop_info", {"reason": reason, "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 5}})


def say_with_usage(text: str, prompt_tokens: int) -> Round:
    return Round([("text", text), stop_info(prompt_tokens), ("stop_reason", "end_turn"), ("done", "end_turn")])


def calls_with_usage(*tool_calls, prompt_tokens: int) -> Round:
    return Round([stop_info(prompt_tokens, "tool_use"), ("stop_reason", "tool_use"), ("tool_calls", list(tool_calls))])


class SystemPromptTest(LoopCase):
    def test_the_agent_names_delegate_to_the_prompt_module(self):
        self.assertEqual(agent.build_system_prompt(), prompt.build_system_prompt())
        self.assertEqual(agent.build_system_prompt(), prompt.CORE)
        self.assertEqual(agent.build_system_prompt(lite=True), prompt.LITE)
        self.assertIs(agent.STRATEGY, prompt.CORE)
        self.assertTrue(prompt.CORE.startswith(agent.PERSONA))
        for literal in ("EXECUTION CONTRACT", "never end the turn at", "HOUSING/ENCLOSURE", "download_file"):
            self.assertIn(literal, agent.STRATEGY)

    def test_workspace_snapshot_is_the_context_function(self):
        with mock.patch.object(agent._context, "workspace_snapshot", lambda max_docs, max_objects: f"{max_docs}/{max_objects}"):
            self.assertEqual(agent.workspace_snapshot(), "8/40")
            self.assertEqual(agent.workspace_snapshot(2, 5), "2/5")
        self.assertEqual(agent.workspace_snapshot(), context.workspace_snapshot())

    def test_the_turn_prompt_names_only_tools_the_intent_offers(self):
        self.run_turn("make a plate", [calls(call("create_object")), say("Done.")])
        generic = self.provider.requests[0].system
        self.assertIn("download_file", generic)                      # generic_part sees the download tool
        self.run_turn(PI_PROMPT, [calls(call("create_object")), say("Done."), say("Done."), say("Done.")], tools_=pi_tools())
        housing = self.provider.requests[0].system
        self.assertNotIn("download_file", housing)                   # hidden for the enclosure intents (INTENT_HIDDEN)
        self.assertIn("EXECUTION CONTRACT", housing)

    def test_the_prompt_is_exactly_what_prompt_builds_for_the_turn_start_names(self):
        self.run_turn("make a plate", [calls(call("create_object")), say("Done.")])
        names = {s.name for s in tools.specs(intent="generic_part")}
        self.assertEqual(self.provider.requests[0].system,
                         # migration: a generic_part turn offered apply_feature_plan now carries the structured_part playbook
                         prompt.build_system_prompt(ModelCaps(), lite=False, playbooks=("structured_part",),
                                                    tool_names=names))

    def test_one_system_string_per_turn_even_when_the_intent_escalates(self):
        script = [calls(call("get_objects")), calls(call("create_object")), say("Done.")]
        self.run_turn("show me three views", script)
        self.assertEqual(self.worker.ledger.intent, "generic_part")          # escalated mid-turn
        self.assertEqual(len({r.system for r in self.provider.requests}), 1)
        names = {s.name for s in tools.specs(intent="other")}                # computed from the turn-START intent
        self.assertEqual(self.provider.requests[0].system, prompt.build_system_prompt(ModelCaps(), lite=False, tool_names=names))

    def test_no_document_name_ever_reaches_the_system_prompt(self):
        self.run_turn("make a plate called SecretPlateDoc", [calls(call("create_object", doc_name="SecretPlateDoc")), say("Done.")])
        self.assertNotIn("SecretPlateDoc", self.provider.requests[0].system)

    def test_small_models_get_the_lite_prompt_and_the_lite_tools(self):
        caps = ModelCaps(context_window=12000, max_output=2048)
        self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], caps=caps)
        first = self.provider.requests[0]
        lite_names = {s.name for s in tools.specs(lite=True)}
        self.assertTrue(names_of(first) <= lite_names, names_of(first) - lite_names)
        self.assertEqual(first.system, prompt.build_system_prompt(caps, lite=True,
                                                                  tool_names={s.name for s in tools.specs(intent="generic_part", lite=True)}))
        self.assertLess(len(first.system), len(prompt.CORE))
        self.assertIs(self.fake.ctxs[0].state["lite"], True)

    def test_tools_that_are_not_lite_are_not_offered_to_a_lite_model(self):
        self.run_turn("make a plate", lenient([say("ok")]), caps=ModelCaps(tools="text"))
        self.assertTrue(names_of(self.provider.requests[0]) <= {s.name for s in tools.specs(lite=True)})

    def test_fem_intent_loads_the_fem_playbook_and_other_intents_none(self):
        self.assertEqual(agent_ctx.playbooks_for("fem", "x", set()), ("fem",))
        self.assertEqual(agent_ctx.playbooks_for("enclosure", "a Pi case", {"insert_part_from_library"}), ())
        self.assertEqual(agent_ctx.playbooks_for("generic_part", "make a plate", {"insert_part_from_library"}), ())

    def test_the_parts_library_playbook_needs_standard_parts_and_the_tool(self):
        self.assertEqual(agent_ctx.playbooks_for("generic_part", "add M3 screws and a bearing", {"insert_part_from_library"}),
                         ("parts_library",))
        self.assertEqual(agent_ctx.playbooks_for("generic_part", "add M3 screws", {"get_objects"}), ())


class ToolListTest(LoopCase):
    def test_every_round_offers_the_specs_of_the_visible_phases(self):
        worker = self.build(PI_PROMPT, pi_script(), tools_=pi_tools())
        seen = []
        original = agent._tools.specs

        def spy(*args, **kwargs):
            out = original(*args, **kwargs)
            if kwargs.get("phases"):
                seen.append((sorted(kwargs["phases"]), kwargs.get("intent"), kwargs.get("lite"), [s.name for s in out]))
            return out

        with mock.patch.object(agent_ctx._tools, "specs", spy):
            worker.run_turn()
        first = self.provider.requests[0]
        self.assertEqual(names_of(first), {s.name for s in original(phases=flow.visible_phases(Ledger.from_request(PI_PROMPT)),
                                                                     intent="enclosure")})
        self.assertTrue(all(intent == "enclosure" and lite is False for _p, intent, lite, _n in seen))
        self.assertGreaterEqual(len(seen), len(self.provider.requests))

    def test_visibility_only_grows_over_a_build(self):
        self.run_turn(PI_PROMPT, pi_script(), tools_=pi_tools())
        sets = [names_of(r) for r in self.provider.requests]
        self.assertGreaterEqual(len(sets), 5)
        for earlier, later in zip(sets, sets[1:]):
            self.assertLessEqual(earlier, later)
        self.assertLess(sets[0], sets[-1])                       # the array did grow over the build

    def test_the_first_round_of_a_work_turn_does_not_offer_every_phase(self):
        self.run_turn(PI_PROMPT, pi_script(), tools_=pi_tools())
        everything = {s.name for s in tools.specs(intent="enclosure")}
        self.assertLess(names_of(self.provider.requests[0]), everything)

    def test_a_turn_without_steps_is_offered_every_phase(self):
        self.run_turn("hello there", [say("Hi.")])
        self.assertEqual(names_of(self.provider.requests[0]), {s.name for s in tools.specs(intent="other")})
        self.assertIn("create_object", names_of(self.provider.requests[0]))
        self.assertIn("execute_code", names_of(self.provider.requests[0]))

    def test_an_escalated_turn_never_loses_a_tool(self):
        self.run_turn("show me three views", [calls(call("get_objects")), calls(call("create_object")), say("Made a box.")])
        sets = [names_of(r) for r in self.provider.requests]
        self.assertEqual(self.worker.ledger.intent, "generic_part")
        for earlier, later in zip(sets, sets[1:]):
            self.assertLessEqual(earlier, later)
        # `other` sees every category; `generic_part` hides fem: the union keeps it for this turn
        fem_tools = {t.name for t in tools.get_tools() if t.category == "fem"}
        if fem_tools:
            self.assertTrue(fem_tools <= sets[-1])

    def test_a_followup_edit_of_a_chat_that_built_something_sees_every_tool(self):
        carried = Ledger.from_request("make a plate")
        self.run_turn("add a fan hole on the side", [calls(call("create_object")), say("Done.")], ledger=carried)
        first = names_of(self.provider.requests[0])
        ledger = Ledger.from_request("add a fan hole on the side")
        self.assertEqual(first, {s.name for s in tools.specs(intent=ledger.intent)})           # every phase, no filter
        self.assertLess({s.name for s in tools.specs(phases=flow.visible_phases(ledger), intent=ledger.intent)}, first)

    def test_earlier_tool_calls_in_the_history_make_the_next_turn_a_build_in_progress(self):
        history = [Message(role="user", content="make a plate"),
                   Message(role="assistant", content="", tool_calls=[call("create_object")]),
                   Message(role="user", tool_results=[ToolResult(tool_call_id="x", name="create_object", content="done")]),
                   Message(role="assistant", content="Made it."),
                   Message(role="user", content="add a fan hole on the side")]
        self.run_turn(history, [calls(call("edit_object")), say("Done.")])
        ledger = Ledger.from_request("add a fan hole on the side")
        self.assertEqual(names_of(self.provider.requests[0]), {s.name for s in tools.specs(intent=ledger.intent)})

    def test_a_fresh_chat_is_not_a_build_in_progress(self):
        worker = self.build("make a plate", [say("x")])
        worker._begin_turn()
        self.assertFalse(worker._all_phases)
        worker._teardown("answered", None)


    def test_the_kill_switch_offers_every_registered_tool(self):
        self.run_turn(PI_PROMPT, [calls(call("create_object")), say("Done.")], flow=False)
        self.assertEqual(names_of(self.provider.requests[0]), {s.name for s in tools.specs()})

    def test_lite_specs_are_compact(self):
        self.run_turn("make a plate", lenient([say("ok")]), caps=ModelCaps(context_window=8192, max_output=1024, tools="text"))
        offered = {s.name: s for s in tools.specs(lite=True)}
        sent = names_of(self.provider.requests[0])
        self.assertTrue(sent and sent <= set(offered))
        for spec in offered.values():
            self.assertLessEqual(len(spec.description), 200)
            self.assertNotIn("step_title", spec.parameters.get("properties", {}))

    def test_the_forced_tool_of_a_gate_nudge_is_one_that_was_offered(self):
        self.run_turn("make a plate", [say("plan"), say("plan"), calls(call("create_object")), say("Done.")])
        forced = self.provider.requests[2].kwargs.get("tool_choice")
        self.assertIn(forced, names_of(self.provider.requests[2]))


class RequestViewTest(LoopCase):
    def long_build(self, rounds: int, **kw):
        script = [calls(call("create_object", obj_name=f"B{i}")) for i in range(rounds)] + [say("Built it.")]
        tools_ = FakeTools(texts={"create_object": BIG})
        return self.run_turn("make a plate", script, tools_=tools_, **kw)

    def test_the_stored_history_is_append_only(self):
        history_log = []

        def script(record):
            history_log.append([(m.role, m.content, len(m.tool_calls), len(m.tool_results)) for m in self.history])
            if len(history_log) > 7:
                return say("Done.")
            return calls(call("create_object", obj_name=f"B{len(history_log)}"))

        self.run_turn("make a plate", script, tools_=FakeTools(texts={"create_object": BIG}),
                      caps=ModelCaps(context_window=20000, max_output=2048))
        for earlier, later in zip(history_log, history_log[1:]):
            self.assertEqual(later[:len(earlier)], earlier)               # nothing was rewritten or removed
        view_sizes = [len(r.messages) for r in self.provider.requests]
        self.assertLess(view_sizes[-2], len(self.history))                # the view shrank, the history did not

    def test_old_governor_lines_are_removed_from_the_view_not_from_the_history(self):
        script = [calls(call("create_object")), calls(call("edit_object")), calls(call("edit_object")), say("Done.")]
        self.run_turn("make a plate", script)
        stored = [m for m in self.history if m.tool_results]
        self.assertTrue(all("[governor]" in m.content for m in stored))
        last = self.provider.requests[-1].messages
        carriers = [m for m in last if "[governor]" in (m.content or "")]
        self.assertEqual(len(carriers), 1)                                # only the newest results message keeps its line
        self.assertIs(carriers[0].tool_results != [], True)

    def test_two_consecutive_views_share_their_prefix_byte_for_byte(self):
        self.run_turn("make a plate", [calls(call("create_object")), calls(call("edit_object")), say("Done.")])
        a, b = self.provider.requests[1].messages, self.provider.requests[2].messages
        self.assertEqual(len(b), len(a) + 2)
        for left, right in zip(a[:-1], b):
            self.assertEqual((left.role, left.content), (right.role, right.content))
        self.assertNotIn("<task_memory>", " ".join(m.content or "" for m in b))

    def test_the_memory_is_not_materialized_while_everything_fits(self):
        self.run_turn(PI_PROMPT, pi_script(), tools_=pi_tools())
        for request in self.provider.requests:
            self.assertNotIn("<task_memory>", " ".join(m.content or "" for m in request.messages))

    def test_rounds_are_dropped_into_the_pinned_memory_when_the_window_is_small(self):
        events = []
        with mock.patch.object(agent_ctx, "log_event", lambda kind, **f: events.append((kind, f))):
            result = self.long_build(7, caps=ModelCaps(context_window=20000, max_output=2048))
        self.assertEqual(result.outcome, "answered")
        views = [f for k, f in events if k == "context_view"]
        self.assertEqual(len(views), len(self.provider.requests))
        self.assertTrue(any(f["dropped_rounds"] > 0 for f in views), views)
        last = self.provider.requests[-1].messages
        self.assertEqual(last[0].role, "user")
        self.assertIn("<task_memory>", last[0].content)
        self.assertIn("make a plate", last[0].content)
        self.assertIn("PLAN:", last[0].content)
        self.assertIn("INTENT: generic_part", last[0].content)
        self.assertGreater(len(self.history), len(last))                  # the stored history kept every message
        self.assertEqual(self.history[0].role, "user")

    def test_the_view_alternates_roles_after_notes_and_cuts(self):
        script = [say("plan"), say("plan"), calls(call("create_object")), say("Done.")]
        self.run_turn("make a plate", script, tools_=FakeTools(texts={"create_object": BIG}))
        for request in self.provider.requests:
            self.assertEqual(context.alternation_problems(request.messages), [])

    def test_the_view_is_what_the_provider_gets_not_the_stored_list(self):
        worker = self.build("make a plate", [calls(call("create_object")), say("Done.")])
        seen = []
        original = agent_ctx._context.build_request_view

        def spy(history, **kw):
            view, stats = original(history, **kw)
            seen.append((history is worker.history, view is not history))
            return view, stats

        with mock.patch.object(agent_ctx._context, "build_request_view", spy):
            worker.run_turn()
        self.assertEqual(seen, [(True, True), (True, True)])

    def test_usage_from_the_provider_calibrates_the_next_view(self):
        script = [calls_with_usage(call("create_object"), prompt_tokens=777), say_with_usage("Done.", 800)]
        worker = self.build("make a plate", script)
        worker.run_turn()
        self.assertEqual(worker._usage_prompt, 800)
        self.assertGreater(worker._sent_estimate, 0)

    def test_a_reply_without_usage_keeps_the_last_calibration_pair(self):
        script = [calls_with_usage(call("create_object"), prompt_tokens=500), say("Done.")]
        worker = self.build("make a plate", script)
        worker.run_turn()
        self.assertEqual(worker._usage_prompt, 500)

    def test_context_view_is_logged_every_round(self):
        events = []
        with mock.patch.object(agent_ctx, "log_event", lambda kind, **f: events.append((kind, f))):
            self.run_turn("make a plate", [calls(call("create_object")), say("Done.")])
        views = [f for k, f in events if k == "context_view"]
        self.assertEqual(len(views), 2)
        for field in ("before", "after", "dropped_rounds", "memory", "too_small", "tools", "messages"):
            self.assertIn(field, views[0])

    def _long_history(self):
        return [Message(role="user", content=f"msg {i}") if i % 2 == 0 else Message(role="assistant", content=f"ans {i}")
                for i in range(agent.MAX_HISTORY_MESSAGES + 10)] + [Message(role="user", content="make a plate")]

    def test_the_legacy_in_place_compaction_still_runs_at_turn_start_with_the_kill_switch(self):
        self.run_turn(self._long_history(), lenient([say("ok")]), flow=False)
        self.assertLess(len(self.history), agent.MAX_HISTORY_MESSAGES)
        self.assertIn("Earlier conversation was trimmed", self.history[0].content)

    def test_the_stored_history_stays_append_only_with_the_flow_on(self):
        history = self._long_history()
        before = [(m.role, m.content) for m in history]
        self.run_turn(history, lenient([say("ok")]), flow=True)
        after = [(m.role, m.content) for m in self.history]
        self.assertEqual(after[:len(before) - 1], before[:-1])          # only the newest message gains its workspace block
        self.assertTrue(after[len(before) - 1][1].startswith("make a plate"))
        self.assertNotIn("Earlier conversation was trimmed", self.history[0].content)


class FloorRuleTest(LoopCase):
    def force(self, too_small_when):
        """Wrap ``build_request_view`` so the floor verdict follows ``too_small_when(tools, memory)``."""
        original = agent_ctx._context.build_request_view
        calls_ = []

        def wrapper(history, **kw):
            view, stats = original(history, **kw)
            stats = dict(stats, too_small=bool(too_small_when(kw["tools"], kw["memory"])), window=2048)
            calls_.append((len(kw["tools"]), len(kw["memory"]), kw["system"]))
            return view, stats

        patcher = mock.patch.object(agent_ctx._context, "build_request_view", wrapper)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls_

    def test_a_window_that_cannot_hold_the_prompt_fails_the_turn_with_a_notice(self):
        result = self.run_turn("make a plate", [say("never asked")], caps=ModelCaps(context_window=1024, max_output=256),
                               validator=False)
        self.assertFalse(result.ok)
        self.assertEqual(result.outcome, "failed")
        self.assertIn("too small", result.text)
        self.assertIn("16384", result.text)
        self.assertEqual(self.provider.requests, [])                      # nothing was sent
        notices = [a[0] for a in self.rec.of("notice")]
        self.assertIn(result.text, notices)
        self.assertEqual(self.rec.of("turn_outcome")[0][0], "failed")
        self.assertEqual(self.rec.of("turn_finished")[0], (False, result.text))

    def test_the_first_step_is_ultra_lite_tools_and_a_short_memory(self):
        calls_ = self.force(lambda specs, memory: len(specs) > len(context.ULTRA_LITE_TOOLS))
        result = self.run_turn("make a plate", [calls(call("create_object")), say("Done.")])
        self.assertEqual(result.outcome, "answered")
        offered = names_of(self.provider.requests[0])
        self.assertTrue(offered <= set(context.ULTRA_LITE_TOOLS), offered)
        self.assertGreaterEqual(len(calls_), 3)                           # the normal try, the ultra try, the next round
        self.assertEqual(calls_[1][0], len(offered))
        self.assertLessEqual(calls_[1][1], context.ULTRA_LITE_MEMORY_CHARS + 200)
        self.assertEqual(self.provider.requests[0].system, self.provider.requests[1].system)   # one switch, then stable
        self.assertIs(self.fake.ctxs[0].state["lite"], True)
        self.assertTrue(any("reduced tool set" in a[0] for a in self.rec.of("notice")))

    def test_ultra_lite_that_still_does_not_fit_refuses(self):
        self.force(lambda specs, memory: True)
        result = self.run_turn("make a plate", [say("never asked")])
        self.assertEqual((result.ok, result.outcome), (False, "failed"))
        self.assertIn("too small", result.text)
        self.assertEqual(self.provider.requests, [])

    def test_ultra_lite_is_sticky_for_the_turn(self):
        self.force(lambda specs, memory: len(specs) > len(context.ULTRA_LITE_TOOLS))
        self.run_turn("make a plate", [calls(call("create_object")), calls(call("edit_object")), say("Done.")])
        for request in self.provider.requests:
            self.assertTrue(names_of(request) <= set(context.ULTRA_LITE_TOOLS))


class PlanLineTest(LoopCase):
    def first_user(self):
        return self.provider.requests[0].messages[0].content

    def test_the_seeded_plan_rides_inside_the_workspace_block_of_the_first_message(self):
        self.run_turn(PI_PROMPT, pi_script(), tools_=pi_tools())
        text = self.first_user()
        self.assertIn("<workspace_state>", text)
        block = text[text.index("<workspace_state>"):text.index("</workspace_state>")]
        self.assertIn("PLAN (intent enclosure", block)
        self.assertIn("S1", block)
        self.assertIn("REQUIREMENTS:", block)

    def test_the_plan_line_is_stored_once_and_never_rewritten(self):
        self.run_turn(PI_PROMPT, pi_script(), tools_=pi_tools())
        stored = self.history[0].content
        self.assertEqual(stored.count("PLAN (intent"), 1)
        for request in self.provider.requests:
            self.assertEqual(request.messages[0].content, stored)         # byte-identical first message every round

    def test_the_plan_line_does_not_leak_into_the_typed_request(self):
        self.run_turn(PI_PROMPT, pi_script(), tools_=pi_tools())
        self.assertEqual(loopcore.real_user_request(self.history), PI_PROMPT)

    def test_a_chat_turn_gets_no_plan_line(self):
        self.run_turn("what is the volume of a 10 mm cube?", [say("1000 mm3.")])
        self.assertNotIn("PLAN (intent", self.first_user())

    def test_the_kill_switch_gets_no_plan_line(self):
        self.run_turn(PI_PROMPT, [calls(call("create_object")), say("Done.")], flow=False)
        self.assertNotIn("PLAN (intent", self.first_user())

    def test_a_resumed_turn_gets_the_carried_plan(self):
        carried = Ledger.from_request("make a plate")
        self.run_turn("continue", [calls(call("create_object")), say("Done.")], ledger=carried)
        self.assertIn("PLAN (intent generic_part", self.first_user())

    def test_a_block_is_created_when_the_snapshot_is_empty(self):
        with mock.patch.object(agent_ctx._context, "workspace_snapshot", lambda *a: ""), \
                mock.patch.object(agent, "workspace_snapshot", lambda *a: ""):
            self.run_turn("make a plate", [calls(call("create_object")), say("Done.")])
        text = self.first_user()
        self.assertIn("<workspace_state>\nPLAN (intent generic_part", text)
        self.assertTrue(text.rstrip().endswith("</workspace_state>"))

    def test_inject_plan_line_is_idempotent_and_keeps_other_blocks(self):
        content = "hello\n\n<workspace_state>\nOpen documents:\n- D: 1 object(s)\n</workspace_state>"
        once = agent_ctx.inject_plan_line(content, "PLAN (intent x, phase INTAKE)\n[ ] S1 a")
        self.assertEqual(agent_ctx.inject_plan_line(once, "PLAN (intent x, phase INTAKE)\n[ ] S1 a"), once)
        self.assertTrue(once.index("Open documents") < once.index("PLAN (intent") < once.index("</workspace_state>"))
        self.assertEqual(agent_ctx.inject_plan_line(content, ""), content)


class CapsWiringTest(LoopCase):
    def test_the_constructor_resolves_caps_without_the_network(self):
        history = [Message(role="user", content="hi")]
        with mock.patch.object(models, "probe_metadata", side_effect=AssertionError("network in the constructor")):
            worker = agent.AgentWorker(history, provider=self.build("x", [say("a")]).provider, profile=profile())
        self.assertIsInstance(worker.caps, ModelCaps)

    def test_run_folds_provider_metadata_into_the_caps(self):
        history = [Message(role="user", content="hi")]
        worker = agent.AgentWorker(history, profile=profile())
        self.assertEqual(worker.caps.context_window, 32768)
        seen = {}
        with mock.patch.object(models, "probe_metadata", lambda p: {"context_window": 12000}), \
                mock.patch.object(agent.AgentWorker, "run_turn", lambda self_: seen.update(window=self_.caps.context_window,
                                                                                         lite=self_._lite)):
            worker.run()
        self.assertEqual(seen, {"window": 12000, "lite": True})

    def test_injected_caps_and_providers_are_never_probed(self):
        with mock.patch.object(models, "probe_metadata", side_effect=AssertionError("probed")):
            worker = self.build("make a plate", lenient([say("ok")]))
            worker.run()
            worker2 = agent.AgentWorker([Message(role="user", content="hi")], provider=self.provider, profile=profile())
            worker2._refresh_caps()

    def test_a_failing_probe_leaves_the_cache_only_caps(self):
        worker = agent.AgentWorker([Message(role="user", content="hi")], profile=profile())
        before = worker.caps
        with mock.patch.object(models, "probe_metadata", side_effect=RuntimeError("down")):
            worker._refresh_caps()
        self.assertIs(worker.caps, before)

    def test_reasoning_auto_is_medium_then_low_for_a_model_with_a_knob(self):
        caps = ModelCaps(reasoning="stream", reasoning_control="anthropic_budget")
        self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], caps=caps)
        self.assertEqual([r.kwargs.get("reasoning") for r in self.provider.requests], ["medium", "low"])
        self.assertEqual(self.provider.requests[0].kwargs.get("max_tokens"), None)
        self.assertEqual(self.provider.requests[1].kwargs.get("max_tokens"), min(self.worker.profile.max_tokens, 12000))

    def test_a_forced_tool_turns_thinking_off_where_the_model_cannot_combine_them(self):
        caps = ModelCaps(reasoning="stream", reasoning_control="anthropic_budget", tool_choice_required_with_reasoning=False)
        self.run_turn("make a plate", [say("plan"), say("plan"), calls(call("create_object")), say("Done.")], caps=caps)
        third = self.provider.requests[2].kwargs
        self.assertEqual(third.get("tool_choice"), "create_object")
        self.assertEqual(third.get("reasoning"), "off")

    def test_a_forced_tool_is_dropped_when_thinking_must_stay_on(self):
        caps = ModelCaps(reasoning="stream", reasoning_control="anthropic_budget", tool_choice_required_with_reasoning=False,
                         reasoning_replay=True)
        events = []
        with mock.patch.object(agent, "log_event", lambda kind, **f: events.append((kind, f))):
            self.run_turn("make a plate", [say("plan"), say("plan"), calls(call("create_object")), say("Done.")], caps=caps)
        third = self.provider.requests[2].kwargs
        self.assertNotIn("tool_choice", third)
        self.assertEqual(third.get("reasoning"), "low")
        self.assertIn("tool_choice_dropped", [k for k, _f in events])

    def test_a_forced_tool_with_reasoning_already_off_is_kept(self):
        caps = ModelCaps(reasoning="stream", reasoning_control="anthropic_budget", tool_choice_required_with_reasoning=False)
        prof = profile()
        prof.reasoning = "off"
        worker = self.build("make a plate", [say("plan"), say("plan"), calls(call("create_object")), say("Done.")], caps=caps)
        worker.profile = prof
        worker.run_turn()
        self.assertEqual(self.provider.requests[2].kwargs.get("tool_choice"), "create_object")

    def test_models_that_reject_tool_choice_never_get_it(self):
        self.run_turn("make a plate", [say("plan"), say("plan"), calls(call("create_object")), say("Done.")],
                      caps=ModelCaps(tool_choice_ok=False))
        self.assertTrue(all("tool_choice" not in r.kwargs for r in self.provider.requests))

    def test_the_turn_start_event_names_the_wiring(self):
        events = []
        with mock.patch.object(agent, "log_event", lambda kind, **f: events.append((kind, f))):
            self.run_turn("run a static analysis on the bracket", lenient([calls(call("create_object")), say("Done.")]))
        start = next(f for k, f in events if k == "turn_start")
        self.assertIs(start["lite"], False)
        self.assertEqual(start["playbooks"], ["fem"])


class BaselineTest(LoopCase):
    def fake_snapshots(self, mapping):
        snaps = {doc: types.SimpleNamespace(finding_keys=list(keys)) for doc, keys in mapping.items()}
        calls_ = []

        def fake(max_docs=5, *, timeout=30.0):
            calls_.append(timeout)
            return snaps

        from oscar.cad import health

        patcher = mock.patch.object(health, "snapshot_all_via_dispatch", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls_

    def test_a_work_turn_takes_the_baseline_once_before_the_first_call(self):
        taken = self.fake_snapshots({"Doc": [("FRAGMENTS", "Shell"), ("INVALID", "Sketch")]})
        seen = []
        validator = lambda doc, ctx: seen.append(dict(ctx.state["baseline"])) or fake_validation(True)  # noqa: E731
        self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], validator=validator)
        self.assertEqual(len(taken), 1)
        self.assertEqual(seen, [{"Doc": [("FRAGMENTS", "Shell"), ("INVALID", "Sketch")]}])
        self.assertEqual(self.fake.ctxs[0].state["baseline"], seen[0])

    def test_the_baseline_snapshot_is_bounded_by_a_timeout(self):
        taken = self.fake_snapshots({})
        self.run_turn("make a plate", [calls(call("create_object")), say("Done.")])
        self.assertEqual(taken, [10.0])

    def test_a_chat_turn_never_touches_the_gui_for_a_baseline(self):
        taken = self.fake_snapshots({"Doc": []})
        self.run_turn("what is a fillet?", [say("A rounded edge.")])
        self.assertEqual(taken, [])

    def test_an_escalated_turn_takes_it_right_before_its_first_geometry_call(self):
        order = []
        taken = self.fake_snapshots({"Doc": [("X", "Y")]})
        tools_ = FakeTools(on_call=lambda name, args: order.append((name, len(taken))))
        self.run_turn("show me three views", [calls(call("get_objects")), calls(call("create_object")), say("Done.")], tools_=tools_)
        self.assertEqual(order, [("get_objects", 0), ("create_object", 1)])
        self.assertEqual(self.fake.ctxs[1].state["baseline"], {"Doc": [("X", "Y")]})

    def test_a_failing_snapshot_leaves_an_empty_baseline_and_the_turn_goes_on(self):
        from oscar.cad import health

        with mock.patch.object(health, "snapshot_all_via_dispatch", side_effect=RuntimeError("gui gone")):
            result = self.run_turn("make a plate", [calls(call("create_object")), say("Done.")])
        self.assertEqual(result.outcome, "answered")
        self.assertEqual(self.fake.ctxs[0].state["baseline"], {})

    def test_the_kill_switch_takes_no_baseline(self):
        taken = self.fake_snapshots({"Doc": []})
        self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], flow=False)
        self.assertEqual(taken, [])


class DefaultValidatorTest(LoopCase):
    def install(self, run):
        module = types.ModuleType("oscar.tools_verify")
        module.run_validation = run
        patcher = mock.patch.dict(sys.modules, {"oscar.tools_verify": module})   # import_module() consults sys.modules first
        patcher.start()
        self.addCleanup(patcher.stop)
        return module

    def test_the_real_validator_runs_for_the_document_the_turn_changed(self):
        seen = []

        def run_validation(doc, *, ctx=None, deep=False, fit=None):
            seen.append((doc, ctx.turn_id, ctx is not None))
            return fake_validation(True)

        self.install(run_validation)

        def stamp(name, args):
            from oscar.toolctx import current_context

            current_context().state["last_doc"] = "PlateDoc"

        tools_ = FakeTools(on_call=stamp)
        result = self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], tools_=tools_)
        self.assertEqual(result.outcome, "validated")
        self.assertEqual(seen[0][0], "PlateDoc")
        self.assertTrue(seen[0][1].startswith("T"))

    def test_without_a_document_the_turn_made_no_validation_claim(self):
        calls_ = []
        self.install(lambda doc, **kw: calls_.append(doc) or fake_validation(True))
        result = self.run_turn("make a plate", [calls(call("create_object")), say("Done.")])
        self.assertEqual(calls_, [])                        # never "the active document"
        self.assertEqual(result.outcome, "answered")

    def test_an_injected_validator_wins_over_the_real_one(self):
        self.install(lambda doc, **kw: self.fail("the real validator must not run"))
        result = self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], validator=lambda d, c: fake_validation(True))
        self.assertEqual(result.outcome, "validated")

    def test_false_means_never_validate(self):
        self.install(lambda doc, **kw: self.fail("the real validator must not run"))

        def stamp(name, args):
            from oscar.toolctx import current_context

            current_context().state["last_doc"] = "PlateDoc"

        result = self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], tools_=FakeTools(on_call=stamp),
                               validator=False)
        self.assertEqual(result.outcome, "answered")
        self.assertIsNone(result.validation)

    def test_a_missing_module_means_no_claim(self):
        def missing(name, *args, **kwargs):
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)

        with mock.patch.object(agent_ctx.importlib, "import_module", missing):
            self.assertIsNone(agent_ctx.default_validator())
            result = self.run_turn("make a plate", [calls(call("create_object")), say("Done.")])
        self.assertEqual(result.outcome, "answered")

    def test_a_module_without_the_function_means_no_claim(self):
        module = types.ModuleType("oscar.tools_verify")
        with mock.patch.object(agent_ctx.importlib, "import_module", lambda name: module):
            self.assertIsNone(agent_ctx.default_validator())

    def test_an_import_error_inside_the_module_is_not_swallowed(self):
        def broken(name, *args, **kwargs):
            raise ModuleNotFoundError("No module named 'numpy_missing'", name="numpy_missing")

        with mock.patch.object(agent_ctx.importlib, "import_module", broken):
            with self.assertRaises(ModuleNotFoundError):
                agent_ctx.default_validator()


class ProjectWiringTest(LoopCase):
    """A REAL ``Project`` under the sandbox: the loop reads it for the memory and finalizes it."""

    def project(self, **kw):
        return temp_project(create=False, **kw)

    def test_the_memory_inputs_come_from_the_projects_on_disk_records(self):
        project = self.project()
        project.ensure_dir()
        with open(project.path("params.json"), "w", encoding="utf-8") as f:
            json.dump({"schema": 1, "params": [{"name": "wall_t", "value": 2.0, "unit": "mm", "source": "assumption"}]}, f)
        project.provenance_add(kind="reference", claim="Pi 4 outline", source="catalog:rpi4b", status="nominal")
        project.note("decision", "wall_t chosen 2.0 (typical FDM)")
        worker = self.build("make a plate", [say("ok")], project=project)
        worker._begin_turn()
        inputs = worker._memory_inputs()
        self.assertEqual(inputs.request, "make a plate")
        self.assertEqual([p["name"] for p in inputs.params], ["wall_t"])
        self.assertIn("wall_t chosen 2.0 (typical FDM)", inputs.decisions)
        self.assertEqual(inputs.project_dir, project.root)
        self.assertEqual(inputs.ledger["intent"], "generic_part")
        text = context.pinned_memory(inputs)
        self.assertIn("wall_t=2 mm (assumption)", text)
        self.assertIn("DECISIONS: wall_t chosen 2.0", text)
        worker._teardown("answered", None)

    def test_the_memory_without_a_project_is_just_request_and_plan(self):
        worker = self.build("make a plate", [say("ok")])
        worker._begin_turn()
        inputs = worker._memory_inputs()
        self.assertEqual((inputs.params, inputs.reference, inputs.decisions, inputs.project_dir), ([], None, [], ""))
        worker._teardown("answered", None)

    def test_a_chat_turn_memory_has_no_ledger_and_the_turns_own_request(self):
        worker = self.build("what is a fillet?", [say("ok")])
        worker._begin_turn()
        inputs = worker._memory_inputs()
        self.assertIsNone(inputs.ledger)
        self.assertEqual(inputs.request, "what is a fillet?")
        worker._teardown("answered", None)

    def test_a_resumed_turn_pins_the_original_request_not_continue(self):
        carried = Ledger.from_request("make a plate with two holes")
        worker = self.build("continue", [say("ok")], ledger=carried)
        worker._begin_turn()
        self.assertEqual(worker._memory_inputs().request, "make a plate with two holes")
        worker._teardown("answered", None)

    def test_objects_come_from_the_health_reports_the_pipeline_collected(self):
        worker = self.build("make a plate", [say("ok")])
        worker._begin_turn()
        worker._state["last_health"] = {"Doc": {"rows": [
            {"name": "Plate", "type": "Part::Box", "solids": 1, "volume": 1000.0, "visible": True, "valid": True},
            {"name": "Tool", "type": "Part::Cut", "solids": 1, "volume": 5.0, "consumed": True}]}}
        rows = worker._object_rows()
        self.assertEqual([r["name"] for r in rows], ["Plate"])
        self.assertEqual(rows[0]["doc"], "Doc")
        worker._teardown("answered", None)

    def test_attachment_and_download_paths_are_remembered_once(self):
        history = [Message(role="user", content="import it\n\n--- attached files ---\n[attached file: a.step] — CAD file\n  path: /tmp/a.step")]
        worker = self.build(history, [say("ok")], attachments=["/tmp/b.stl"])
        worker._begin_turn()
        worker.history.append(Message(role="user", tool_results=[
            ToolResult(tool_call_id="1", name="download_file", content="saved to /tmp/dl/pi4.step (12 kB)"),
            ToolResult(tool_call_id="2", name="get_objects", content="/tmp/ignored.step"),
            ToolResult(tool_call_id="3", name="download_file", content="failed /tmp/dl/bad.step", is_error=True)]))
        files = worker._memory_inputs().files
        self.assertEqual(files, ["/tmp/b.stl", "/tmp/a.step", "/tmp/dl/pi4.step"])
        self.assertEqual(worker._memory_inputs().files, files)            # idempotent (incremental scan)
        worker._teardown("answered", None)

    def test_the_reference_entry_is_resolved_from_the_provenance_entries(self):
        project = self.project()
        project.ensure_dir()
        entry = project.provenance_add(kind="reference", claim="Pi 4", source="catalog:rpi4b", status="nominal")
        project.set_reference("rpi4b", "Raspberry Pi 4B", "catalog-nominal", entry)
        worker = self.build("make a plate", [say("ok")], project=project)
        worker._begin_turn()
        reference = worker._memory_inputs().reference
        self.assertEqual(reference["status"], "catalog-nominal")
        self.assertEqual(reference["entry"]["source"], "catalog:rpi4b")
        self.assertIn("REFERENCE: rpi4b", context.pinned_memory(worker._memory_inputs()))
        worker._teardown("answered", None)

    def test_only_decision_notes_are_pinned_not_assumptions(self):
        project = self.project()
        project.ensure_dir()
        project.note("decision", "wall_t chosen 2.0")
        project.note("assumption", "PCB is 1.4 mm thick")
        project.note("unresolved", "no STEP file")
        worker = self.build("make a plate", [say("ok")], project=project)
        worker._begin_turn()
        self.assertEqual(worker._memory_inputs().decisions, ["wall_t chosen 2.0"])
        worker._teardown("answered", None)

    def test_a_stub_project_without_readers_does_not_break_the_memory(self):
        worker = self.build("make a plate", [say("ok")], project=StubProject("/tmp/stubproj"))
        worker._begin_turn()
        inputs = worker._memory_inputs()
        self.assertEqual((inputs.params, inputs.reference, inputs.decisions, inputs.project_dir), ([], None, [], "/tmp/stubproj"))
        worker._teardown("answered", None)

    def test_a_dropping_view_of_a_real_project_turn_pins_params_and_finishes_with_the_real_summary(self):
        project = self.project()
        project.ensure_dir()
        with open(project.path("params.json"), "w", encoding="utf-8") as f:
            json.dump({"schema": 1, "params": [{"name": "wall_t", "value": 2.0, "unit": "mm", "source": "assumption"}]}, f)
        script = [calls(call("create_object", obj_name=f"B{i}")) for i in range(7)] + [say("Built the plate.")]
        result = self.run_turn("make a plate", script, tools_=FakeTools(texts={"create_object": BIG}), project=project,
                               caps=ModelCaps(context_window=20000, max_output=2048), validator=lambda d, c: fake_validation(True))
        self.assertEqual(result.outcome, "validated")
        self.assertIn("## Build Summary", result.text)
        memory_views = [r.messages[0].content for r in self.provider.requests if "<task_memory>" in (r.messages[0].content or "")]
        self.assertTrue(memory_views)
        self.assertIn("wall_t=2 mm (assumption)", memory_views[-1])
        self.assertTrue(os.path.isfile(project.path("NOTES.md")))
        self.assertTrue(os.path.isfile(project.path("ledger.json")))


class RobustnessTest(LoopCase):
    def test_a_defect_in_view_building_falls_back_to_the_stored_history(self):
        events = []
        with mock.patch.object(agent_ctx, "log_event", lambda kind, **f: events.append((kind, f))), \
                mock.patch.object(agent_ctx._context, "build_request_view", side_effect=RuntimeError("view bug")):
            result = self.run_turn("make a plate", [calls(call("create_object")), say("Done.")])
        self.assertEqual((result.ok, result.outcome), (True, "answered"))
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual([f["where"] for k, f in events if k == "context_error"], ["view", "view"])
        self.assertEqual(len(self.provider.requests[1].messages), 3)           # the stored history, unchanged

    def test_a_broken_memory_builder_only_shortens_the_memory(self):
        worker = self.build("make a plate", [say("ok")])
        worker._begin_turn()
        with mock.patch.object(agent_ctx._context, "pinned_memory", side_effect=ValueError("memory bug")):
            self.assertEqual(worker._pinned_memory(6000, []), "")
        worker._teardown("answered", None)

    def test_windows_and_posix_paths_in_download_results_are_remembered(self):
        worker = self.build("make a plate", [say("ok")])
        worker._begin_turn()
        worker.history.append(Message(role="user", tool_results=[
            ToolResult(tool_call_id="1", name="download_file", content=r"saved to C:\Users\me\Downloads\pi4.STEP (1 MB)"),
            ToolResult(tool_call_id="2", name="source_reference_part", content="saved /tmp/dl/board.stp and /tmp/dl/readme.txt")]))
        self.assertEqual(worker._scan_files(), [r"C:\Users\me\Downloads\pi4.STEP", "/tmp/dl/board.stp"])
        worker._teardown("answered", None)

    def test_at_most_twelve_files_are_pinned(self):
        worker = self.build("make a plate", [say("ok")])
        worker._begin_turn()
        worker.history.append(Message(role="user", tool_results=[
            ToolResult(tool_call_id=str(i), name="download_file", content=f"saved /tmp/dl/f{i}.step") for i in range(20)]))
        self.assertEqual(len(worker._scan_files()), 12)
        worker._teardown("answered", None)

    def test_a_steered_message_is_pinned_as_a_decision(self):
        worker = self.build("make a plate", [say("ok")])
        worker._begin_turn()
        worker.steer("make it 5 mm thick")
        worker._deliver_steer()
        self.assertIn("user, mid-turn: make it 5 mm thick", worker._memory_inputs().decisions)
        worker._teardown("answered", None)

    def test_an_empty_history_is_a_valid_view(self):
        worker = agent.AgentWorker([], provider=self.build("x", [say("a")]).provider, profile=profile(), caps=ModelCaps())
        worker._begin_turn()
        view, specs = worker._request_for_round()
        self.assertEqual(view, [])
        self.assertTrue(specs)
        worker._teardown("answered", None)


class ModuleSizeTest(unittest.TestCase):
    def test_the_loop_modules_stay_under_1500_lines(self):
        base = os.path.join(os.path.dirname(__file__), "..", "oscar")
        for name in ("agent.py", "agent_ctx.py", "agent_legacy.py", "flow.py", "flow_guard.py", "flow_intent.py",
                     "flow_run.py", "loopcore.py", "tools_flow.py"):
            with open(os.path.join(base, name), encoding="utf-8") as f:
                self.assertLess(sum(1 for _ in f), 1500, name)


class StructuredPartPlaybookSelectionTest(unittest.TestCase):
    """``playbooks_for`` selects the structured_part playbook for prismatic intents when apply_feature_plan is offered."""

    OFFERED = {"apply_feature_plan", "execute_code", "validate_model"}

    def test_selected_for_bracket_fixture_and_generic_part_when_the_tool_is_offered(self):
        for intent in ("bracket", "fixture", "generic_part"):
            self.assertEqual(agent_ctx.playbooks_for(intent, "make a plate", self.OFFERED), ("structured_part",), intent)

    def test_not_selected_without_the_tool(self):
        for intent in ("bracket", "fixture", "generic_part"):
            self.assertEqual(agent_ctx.playbooks_for(intent, "make a plate", {"execute_code"}), (), intent)
            self.assertEqual(agent_ctx.playbooks_for(intent, "make a plate", set()), (), intent)

    def test_not_selected_for_other_intents(self):
        for intent in ("enclosure", "inspect", "question", "other", "modify", "deliver"):
            self.assertNotIn("structured_part", agent_ctx.playbooks_for(intent, "make a plate", self.OFFERED), intent)
        self.assertEqual(agent_ctx.playbooks_for("fem", "x", self.OFFERED), ("fem",))

    def test_it_combines_with_the_parts_library_playbook(self):
        books = agent_ctx.playbooks_for("generic_part", "a bracket with M3 screws",
                                        self.OFFERED | {"insert_part_from_library"})
        self.assertEqual(set(books), {"parts_library", "structured_part"})

    def test_a_generic_part_turn_carries_it_and_a_housing_turn_does_not(self):
        case = unittest.TestCase()
        del case                                    # (the turn-level checks live in SystemPromptTest)
        for intent in ("generic_part", "bracket", "fixture"):
            self.assertIn("apply_feature_plan", {s.name for s in tools.specs(intent=intent)}, intent)
        names = {s.name for s in tools.specs(intent="enclosure")}
        self.assertNotIn("apply_feature_plan", names)  # build_enclosure builds a housing: the playbook follows the tool


if __name__ == "__main__":
    unittest.main()
