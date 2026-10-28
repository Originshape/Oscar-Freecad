"""oscar.agent: the structured build loop, driven by a scripted provider and fakes (no network, no GUI).

Every test builds an ``AgentWorker`` with an injected provider / profile / project / validator and runs the turn
synchronously (``run_turn`` / ``run``). ``tools.run_tool`` is replaced by ``FakeTools``, which behaves like the
real pipeline as far as the loop can tell (it fills ``ctx.outcome`` and ticks the ledger); the two plan tools
run for real. The Project is a recording stub with the interface
"""

import json
import os
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import agent, flow, loopcore, params, tools  # noqa: E402
from oscar.flow import AgentConfig, Ledger, WatchdogConfig  # noqa: E402
from oscar.models import ModelCaps  # noqa: E402
from oscar.providers import Message, StopRequested, ToolCall  # noqa: E402
from oscar.toolctx import ToolContext, ToolOutcome, current_context  # noqa: E402
from support.fakes import fake_validation  # noqa: E402
from support.scripted_llm import (Round, ScriptedProvider, bad_json, call, calls, empty, nameless,  # noqa: E402
                                  provider_error, say, think)
from support.signal_recorder import SignalRecorder  # noqa: E402

REAL_RUN_TOOL = agent._tools.run_tool

PI_PROMPT = "Build a housing for a Raspberry Pi 4 with a bottom shell, a lid and vents. Source the STEP file."
BRACKET_PROMPT = "Design a bracket for a NEMA17 motor with 4 mounting holes"
GEOMETRY = frozenset({"create_object", "edit_object", "delete_object", "execute_code", "build_enclosure", "add_cutout",
                      "add_vents", "import_file"})
MUTATING = GEOMETRY | {"create_document", "set_parameters", "enclosure_spec", "source_reference_part",
                       "import_reference_part", "measure_reference_part", "export_model", "save_document"}
PARAM_NAMES = list(flow.ENCLOSURE_REQUIRED_PARAMS)


class FakeTools:
    """Stands in for ``tools.run_tool``: scripted failures and extras, like the real pipeline fills ``ctx.outcome``."""

    def __init__(self, *, fail=None, health_fail=(), texts=None, extra=None, hooks=True, real=("plan_set", "plan_update"),
                 on_call=None):
        self.calls: list[tuple[str, dict]] = []
        self.ctxs: list[ToolContext] = []
        self.fail = dict(fail or {})              # name -> number of failures (True = always)
        self.health_fail = set(health_fail)
        self.texts = dict(texts or {})
        self.extra = dict(extra or {})
        self.hooks = hooks
        self.real = set(real)
        self.on_call = on_call
        self.step_index = 0

    def names(self) -> list[str]:
        return [name for name, _args in self.calls]

    def _failing(self, name: str) -> bool:
        left = self.fail.get(name)
        if left is True:
            return True
        if left:
            self.fail[name] = left - 1
            return True
        return False

    def __call__(self, name, arguments):
        if name in self.real:
            return REAL_RUN_TOOL(name, arguments)
        ctx = current_context()
        self.calls.append((name, dict(arguments)))
        self.ctxs.append(ctx)
        if self.on_call is not None:
            self.on_call(name, arguments)
        failed = self._failing(name)
        text = self.texts.get(name, f"{name} done" if not failed else f"{name} failed: boom")
        is_error = failed or name in self.health_fail
        if not self.hooks:
            return text, None, is_error
        geometry, mutating = name in GEOMETRY, name in MUTATING
        if mutating and not failed:
            self.step_index += 1
        outcome = ToolOutcome(
            name=name, call_id=ctx.call_id, title=name, ok=not is_error, handler_ok=not failed, text=text,
            mutating=mutating, geometry=geometry, step_index=self.step_index if mutating and not failed else -1,
            args=dict(arguments), extra=dict(self.extra.get(name, {})),
            delta={"added": ["Obj"], "removed": [], "changed": []} if geometry and not failed else {})
        ctx.outcome = outcome
        if ctx.ledger is not None:
            ctx.ledger.on_tool_result(outcome)
        return text, None, is_error


class StubProject:
    """The Project interface the agent uses, recording every call."""

    def __init__(self, root: str = ""):
        self.root = root
        self.chat_id = "chat1"
        self.calls: list[tuple] = []
        self.saved: list[dict] = []
        self.summary = "## Build Summary\n### Model tree\n- Shell\n### Validated\n- health"
        self.fail_on: set[str] = set()

    def _rec(self, name, *args):
        self.calls.append((name, *args))
        if name in self.fail_on:
            raise OSError(f"{name}: disk full")

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def begin_turn(self, turn_id, request, *, doc_names=(), work=False):
        self._rec("begin_turn", turn_id, request, work)

    def end_turn(self, turn_id, outcome, *, rounds=0):
        self._rec("end_turn", turn_id, outcome, rounds)

    def save_ledger(self, ledger):
        self._rec("save_ledger")
        self.saved.append(ledger if isinstance(ledger, dict) else ledger.to_dict())

    def archive_ledger(self, turn_id):
        self._rec("archive_ledger", turn_id)

    def finalize(self, *, ledger, outcome, validation, export=None):
        self._rec("finalize", outcome, ledger, validation)
        return types.SimpleNamespace(markdown=self.summary, data={})

    def write_notes_md(self, ledger=None):
        self._rec("write_notes_md")
        return ""

    def write_rebuild_py(self):
        self._rec("write_rebuild_py")

    def last_geometry_step(self):
        return 0

    def ensure_turn_checkpoint(self, doc):
        return ""

    def path(self, *parts):
        return os.path.join(self.root, *parts)


def profile():
    return params.Profile(name="t", provider="openai_compat", model="scripted", base_url="http://127.0.0.1:9/v1")


class LoopCase(unittest.TestCase):
    """Builds workers around a scripted provider and keeps the bookkeeping out of the tests."""

    def setUp(self):
        self.fake = FakeTools()
        self._patch = None

    def build(self, prompt, script, *, tools_=None, **kw):
        history = [Message(role="user", content=prompt)] if isinstance(prompt, str) else prompt
        self.fake = tools_ or self.fake
        if self._patch is None:
            self._patch = mock.patch.object(agent._tools, "run_tool", lambda n, a: self.fake(n, a))
            self._patch.start()
            self.addCleanup(self._patch.stop)
        kw.setdefault("caps", ModelCaps())
        kw.setdefault("config", AgentConfig())
        self.provider = ScriptedProvider(script)
        self.history = history
        self.worker = agent.AgentWorker(history, provider=self.provider, profile=profile(), **kw)
        self.rec = SignalRecorder(self.worker)
        return self.worker

    def run_turn(self, prompt, script, **kw):
        worker = self.build(prompt, script, **kw)
        return worker.run_turn()

    def results_messages(self):
        return [m for m in self.history if m.tool_results]

    def notes(self, tag):
        return [m for m in self.history if m.synthetic and m.tag == tag]


def pi_script(*, finish="Built the Pi 4 housing."):
    return [
        calls(call("source_reference_part", query="Raspberry Pi 4")),
        calls(call("create_document", title="PiCase"), call("enclosure_spec", board="rpi4")),
        calls(call("build_enclosure", stage="shell"), call("build_enclosure", stage="standoffs")),
        calls(call("build_enclosure", stage="lid")),
        calls(call("build_enclosure", stage="vents"), call("build_enclosure", stage="assembly")),
        say(finish),
    ]


def pi_tools(**kw):
    extra = {"source_reference_part": {"reference_result": {"status": "nominal", "id": "rpi4"}},
             "enclosure_spec": {"param_names": PARAM_NAMES, "spec_ok": True}}
    return FakeTools(extra=extra, **kw)


class ChatTurnTest(LoopCase):
    def test_a_question_is_plain_chat(self):
        project = StubProject()
        result = self.run_turn("What is the difference between pad and pocket?", [say("Pad adds material; pocket removes it.")],
                               project=project)
        self.assertEqual((result.ok, result.outcome, result.text), (True, "answered", "Pad adds material; pocket removes it."))
        self.assertIsNone(result.ledger)
        self.assertEqual(self.rec.of("plan_changed"), [])
        self.assertEqual(self.rec.of("turn_outcome"), [("answered", "")])
        self.assertEqual(self.rec.of("turn_finished"), [(True, "Pad adds material; pocket removes it.")])
        self.assertEqual(self.history[-1].content, "Pad adds material; pocket removes it.")
        self.assertNotIn("finalize", project.names())
        begin = next(c for c in project.calls if c[0] == "begin_turn")
        self.assertEqual((begin[2], begin[3]), ("What is the difference between pad and pocket?", False))
        self.assertEqual(project.calls[-1][:3], ("end_turn", begin[1], "answered"))

    def test_read_only_tools_in_a_chat_turn_get_no_governor_line(self):
        self.run_turn("list my documents", [calls(call("list_documents")), say("One document.")])
        self.assertEqual(self.results_messages()[0].content, "")
        self.assertEqual(self.history[-1].content, "One document.")

    def test_turn_end_signals_come_last_in_order(self):
        self.run_turn("hello?", [say("Hi.")])
        names = self.rec.names()
        self.assertEqual(names[-2:], ["turn_outcome", "turn_finished"])

    def test_empty_ledger_is_not_carried(self):
        carried = Ledger.from_request("make a plate")
        result = self.run_turn("thanks", [say("You're welcome.")], ledger=carried)
        self.assertEqual(result.outcome, "answered")
        self.assertEqual(result.ledger, carried.to_dict())

    def test_refusal_is_a_failure_with_a_hint(self):
        result = self.run_turn("hello?", [Round([("stop_reason", "refusal"), ("done", "refusal")])])
        self.assertFalse(result.ok)
        self.assertIn("safety refusal", result.text)

    def test_cut_off_answer_gets_a_footer_and_no_blocks(self):
        result = self.run_turn("hello?", [say("Partial answer", stop="length")])
        self.assertTrue(result.ok)
        self.assertTrue(result.text.endswith("(The reply was cut off at the output-token limit.)"))

    def test_a_reply_shortened_for_low_credits_says_so(self):
        """OriginShape lowers max_tokens to what the credits pay for and says ``credits`` in x-oscar-normalized; the
        cut-off note names the credits, not the model's output budget (audit 2026-10-28)."""
        worker = self.build("hello?", [say("Partial answer", stop="length")])
        worker.provider.last_normalized = "max_tokens,credits"
        result = worker.run_turn()
        self.assertTrue(result.ok)
        self.assertIn("your OriginShape credits are low", result.text)
        self.assertNotIn("output-token limit", result.text)
        worker = self.build("hello?", [Round([("stop_reason", "max_tokens"), ("done", "max_tokens")])])
        worker.provider.last_normalized = "max_tokens,credits"
        self.assertIn("credits are low", worker._empty_reply_message("max_tokens"))
        self.assertNotIn("output budget", worker._empty_reply_message("max_tokens"))

    def test_provider_error_fails_the_turn(self):
        result = self.run_turn("hello?", [provider_error("HTTP 503")])
        self.assertEqual((result.ok, result.outcome, result.text), (False, "failed", "Provider error: HTTP 503"))
        self.assertEqual(self.rec.of("turn_finished"), [(False, "Provider error: HTTP 503")])


class WorkTurnTest(LoopCase):
    def test_enclosure_turn_is_validated_with_the_build_summary(self):
        project = StubProject()
        worker = self.build(PI_PROMPT, pi_script(), tools_=pi_tools(), project=project,
                            validator=lambda doc, ctx: fake_validation(True))
        result = worker.run_turn()
        self.assertEqual((result.ok, result.outcome), (True, "validated"))
        self.assertTrue(result.text.startswith("Built the Pi 4 housing.\n\n## Build Summary"))
        self.assertEqual(self.history[-1].content, "Built the Pi 4 housing.")        # only the model text enters the history
        self.assertEqual(self.rec.of("turn_outcome")[0][0], "validated")
        self.assertIn("## Build Summary", self.rec.of("turn_outcome")[0][1])
        self.assertEqual(result.rounds, 6)
        self.assertEqual(result.tool_rounds, 5)
        self.assertEqual(result.geometry_ok, 5)
        # project lifecycle
        names = project.names()
        self.assertEqual(names[0], "begin_turn")
        self.assertTrue(project.calls[0][3])                                          # work=True
        self.assertEqual(names.count("finalize"), 1)
        self.assertEqual(project.calls[names.index("finalize")][1], "validated")
        self.assertEqual(names[-1] if names[-1] != "save_ledger" else names[-2], "end_turn")
        self.assertIn("save_ledger", names)
        final = project.saved[-1]
        self.assertEqual(next(s for s in final["steps"] if s["id"] == "S12")["status"], "done")
        self.assertTrue(final["health_ok"])
        # the ledger the panel carries
        self.assertEqual(result.ledger["intent"], "enclosure")
        self.assertEqual(result.ledger["reference_status"], "catalog-nominal")
        self.assertEqual(worker.ledger.step("S3").note, flow.NO_STEP_NOTE)

    def test_plan_is_emitted_at_t0_and_after_every_change(self):
        self.run_turn(PI_PROMPT, pi_script(), tools_=pi_tools(), validator=lambda d, c: fake_validation(True),
                      project=StubProject())
        plans = self.rec.json_of("plan_changed")
        self.assertEqual(self.rec.names()[0], "plan_changed")
        self.assertEqual(plans[0]["intent"], "enclosure")
        self.assertTrue(all(s["status"] != "done" for s in plans[0]["steps"] if s["id"] != "S1"))
        revisions = [p["revision"] for p in plans]
        self.assertEqual(revisions, sorted(revisions))
        self.assertGreater(len(plans), 4)
        self.assertEqual(plans[-1]["steps"][-1]["status"], "done")

    def test_per_call_signal_order(self):
        self.run_turn("make a plate", [calls(call("create_object", obj_type="Part::Box")), say("Done.")],
                      validator=lambda d, c: fake_validation(True))
        names = [n for n in self.rec.names() if n in ("tool_started", "step_started", "step_finished", "tool_finished",
                                                      "plan_changed")]
        first = names.index("tool_started")
        self.assertEqual(names[first:first + 5], ["tool_started", "step_started", "step_finished", "tool_finished", "plan_changed"])

    def test_step_payloads(self):
        self.run_turn("make a plate", [calls(call("create_object", obj_type="Part::Box", step_title="Make the plate"),
                                             call("execute_code", code="print(1)")), say("Done.")])
        started = self.rec.json_of("step_started")
        finished = self.rec.json_of("step_finished")
        self.assertEqual([s["name"] for s in started], ["create_object", "execute_code"])
        s = started[0]
        for key in ("call_id", "name", "title", "args_summary", "round", "ordinal", "mutating", "geometry", "phase", "started_at"):
            self.assertIn(key, s)
        self.assertEqual((s["title"], s["round"], s["ordinal"], s["mutating"], s["geometry"]), ("Make the plate", 1, 1, True, True))
        self.assertEqual(started[1]["ordinal"], 2)
        f = finished[1]
        for key in ("call_id", "name", "title", "ok", "skipped", "text", "dur_s", "round", "ordinal", "code", "args_json",
                    "delta", "health", "step_index", "snapshot_path", "snapshot_note", "checkpoint", "script_path"):
            self.assertIn(key, f)
        self.assertEqual((f["code"], f["args_json"], f["ok"], f["skipped"]), ("print(1)", "", True, False))
        self.assertEqual(finished[0]["code"], "")
        self.assertIn("obj_type", finished[0]["args_json"])
        self.assertEqual(finished[0]["delta"]["added"], ["Obj"])
        tf = self.rec.of("tool_finished")
        self.assertEqual(tf[0][1], True)
        self.assertTrue(tf[0][0].startswith("create_object("))

    def test_governor_line_rides_in_every_results_message(self):
        self.run_turn(PI_PROMPT, pi_script(), tools_=pi_tools(), validator=lambda d, c: fake_validation(True))
        msgs = self.results_messages()
        self.assertEqual(len(msgs), 5)
        for n, m in enumerate(msgs, 1):
            self.assertIn(f"[governor] round {n}/30", m.content)
        self.assertIn("[phase] ", msgs[0].content)
        self.assertIn("health ?", msgs[0].content)
        self.assertEqual(msgs[0].content.count("[governor]"), 1)

    def test_the_system_prompt_is_the_same_every_round(self):
        self.run_turn(PI_PROMPT, pi_script(), tools_=pi_tools())
        self.assertEqual(len({r.system for r in self.provider.requests}), 1)

    def test_work_without_a_validator_is_answered_not_validated(self):
        result = self.run_turn("make a plate", [calls(call("create_object", obj_type="Part::Box")), say("Modelled the plate.")])
        self.assertEqual((result.ok, result.outcome, result.text), (True, "answered", "Modelled the plate."))
        self.assertIsNone(result.validation)
        self.assertEqual(self.worker.ledger.step("S4").note, "automatic validation unavailable")

    def test_a_validator_that_returns_none_makes_no_claim(self):
        result = self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], validator=lambda d, c: None)
        self.assertEqual(result.outcome, "answered")

    def test_a_broken_validator_does_not_kill_the_turn(self):
        def boom(doc, ctx):
            raise RuntimeError("validator exploded")

        result = self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], validator=boom)
        self.assertEqual((result.ok, result.outcome), (True, "answered"))

    def test_validator_receives_the_turn_context(self):
        seen = []

        def validator(doc, ctx):
            seen.append((doc, ctx.turn_id, ctx.ledger, ctx.state, current_context() is ctx))
            return fake_validation(True)

        project = StubProject()
        self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], validator=validator, project=project)
        doc, turn_id, ledger, state, bound = seen[0]
        self.assertTrue(turn_id.startswith("T"))
        self.assertIs(ledger, self.worker.ledger)
        self.assertIs(state, self.fake.ctxs[0].state)
        self.assertTrue(bound)

    def test_validation_runs_once_per_geometry_change(self):
        runs = []
        self.run_turn("make a plate", [calls(call("create_object")), say("Done."), ],
                      validator=lambda d, c: runs.append(1) or fake_validation(True))
        self.assertEqual(len(runs), 1)

    def test_tool_context_carries_the_turn_state(self):
        self.run_turn("make a plate", [calls(call("create_object"), call("execute_code")), calls(call("edit_object")), say("ok")])
        first, second, third = self.fake.ctxs
        self.assertIs(first.state, second.state)
        self.assertIs(second.state, third.state)
        self.assertEqual(first.state["lite"], False)
        self.assertEqual((first.round, third.round), (1, 2))
        self.assertTrue(first.call_id and first.call_id != second.call_id)
        self.assertEqual((first.intent, first.phase), ("generic_part", "BUILD"))
        self.assertTrue(first.turn_id and first.turn_id == third.turn_id)
        self.assertTrue(callable(first.should_stop))
        self.assertIsNotNone(first.ledger)

    def test_the_tool_runs_inside_a_bound_context_and_outside_it_the_context_is_inert(self):
        self.run_turn("make a plate", [calls(call("create_object")), say("ok")])
        self.assertFalse(current_context().active)

    def test_a_patched_run_tool_without_hooks_still_accrues_evidence(self):
        tools_ = FakeTools(hooks=False)
        result = self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], tools_=tools_)
        self.assertEqual(result.geometry_ok, 1)
        self.assertEqual(self.worker.ledger.step("S3").status, "done")
        self.assertEqual(result.outcome, "answered")


class GateTest(LoopCase):
    def test_pending_steps_are_nudged_then_unverified(self):
        script = [calls(call("create_object")), say("Done."), say("Done really."), say("Done!")]
        result = self.run_turn(PI_PROMPT, script, tools_=pi_tools(), validator=lambda d, c: fake_validation(True))
        self.assertEqual((result.ok, result.outcome), (True, "unverified"))
        self.assertTrue(result.text.startswith("NOT VERIFIED: this turn ended with items the app could not close:"))
        self.assertIn("S2", result.text)
        self.assertEqual(len(self.provider.requests), 4)
        gate = self.notes("gate")
        self.assertEqual(len(gate), 2)
        self.assertTrue(gate[0].content.startswith("[finish-gate] You stopped, but the app found unfinished work:"))
        self.assertIn("STEPS_PENDING", gate[0].content)
        # roles alternate around every injected note
        for note in gate:
            i = self.history.index(note)
            self.assertEqual(self.history[i - 1].role, "assistant")
        self.assertEqual(self.history[-1].content, "Done!")
        self.assertTrue(result.reasons)
        self.assertEqual(self.rec.of("turn_outcome")[0][0], "unverified")
        self.assertIn("NOT VERIFIED", self.rec.of("turn_finished")[0][1])

    def test_unverified_text_carries_the_project_summary(self):
        project = StubProject()
        result = self.run_turn(PI_PROMPT, [calls(call("create_object")), say("a"), say("b"), say("c")], tools_=pi_tools(),
                               project=project)
        self.assertTrue(result.text.rstrip().endswith("### Validated\n- health"))
        self.assertEqual(project.calls[[c[0] for c in project.calls].index("finalize")][1], "unverified")
        # the DELIVER step stays open: the app does not claim a delivery it could not make
        self.assertNotEqual(self.worker.ledger.step("S12").status, "done")

    def test_second_no_work_nudge_forces_the_first_offered_hint_tool(self):
        result = self.run_turn("make a plate", [say("I would model a plate like this."), say("Still planning."), say("Plan.")])
        self.assertEqual((result.ok, result.outcome), (False, "failed"))
        self.assertTrue(result.text.startswith("Oscar made no changes in the model for this request"))
        self.assertIn("the model made no change", result.text)
        kwargs = [r.kwargs.get("tool_choice") for r in self.provider.requests]
        self.assertEqual(kwargs, [None, None, "create_object"])
        self.assertEqual(len(self.provider.requests), 3)
        self.assertEqual(self.history[-1].role, "user")
        self.assertEqual(self.notes("gate")[0].tag, "gate")

    def test_the_forced_tool_applies_to_one_call_only(self):
        script = [say("plan"), say("plan"), calls(call("create_object")), say("Done.")]
        result = self.run_turn("make a plate", script)
        self.assertEqual(result.outcome, "answered")
        self.assertEqual([r.kwargs.get("tool_choice") for r in self.provider.requests], [None, None, "create_object", None])

    def test_claims_without_work_are_not_accepted(self):
        script = [calls(call("create_object")),
                  calls(call("plan_update", id="S4", status="done", note="holes are in")),
                  say("All done."), say("All done."), say("All done.")]
        result = self.run_turn(BRACKET_PROMPT, script)
        self.assertEqual(result.outcome, "unverified")
        self.assertIn("S4 claimed without recorded work", result.text)
        self.assertTrue(any("CLAIM_ONLY" in m.content for m in self.notes("gate")))

    def test_blocked_step_and_a_question_to_the_user_is_needs_input(self):
        project = StubProject()
        script = [calls(call("plan_update", id="S2", status="blocked", note="I need the STEP file from you")),
                  say("Please attach the STEP file for the Pi 4 and I will continue.")]
        result = self.run_turn(PI_PROMPT, script, tools_=pi_tools(), project=project, validator=lambda d, c: fake_validation(True))
        self.assertEqual((result.ok, result.outcome), (True, "needs_input"))
        self.assertEqual(result.text, "Please attach the STEP file for the Pi 4 and I will continue.")
        self.assertEqual(self.rec.of("turn_outcome"), [("needs_input", "")])
        self.assertNotIn("finalize", project.names())
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(self.notes("gate"), [])

    def test_health_failure_at_the_end_blocks_validated(self):
        results = [fake_validation(False), fake_validation(False), fake_validation(False)]
        script = [calls(call("create_object")), say("Done."), say("Done."), say("Done.")]
        result = self.run_turn("make a plate", script, validator=lambda d, c: results.pop(0))
        self.assertEqual(result.outcome, "unverified")
        self.assertIn("Shell: 3 solids", result.text)
        self.assertIsNotNone(self.worker.ledger.step("F1"))

    def test_the_model_can_fix_the_findings_and_end_validated(self):
        verdicts = [fake_validation(False), fake_validation(True)]
        runs = []

        def validator(doc, ctx):
            runs.append(1)
            return verdicts.pop(0)

        script = [calls(call("create_object")), say("Done."), calls(call("edit_object")), say("Fixed.")]
        result = self.run_turn("make a plate", script, validator=validator)
        self.assertEqual((result.ok, result.outcome), (True, "validated"))
        self.assertEqual(len(runs), 2)
        self.assertEqual(self.worker.ledger.step("F1").status, "done")
        self.assertTrue(any("HEALTH_FAIL" in m.content for m in self.notes("gate")))

    def test_a_preexisting_problem_that_the_validator_excuses_does_not_block(self):
        validation = fake_validation(True)
        validation.health = dict(validation.health, valid=False)      # the user's own broken sketch: not a gate error
        result = self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], validator=lambda d, c: validation)
        self.assertEqual(result.outcome, "validated")

    def test_fit_failures_block(self):
        validation = fake_validation(True, fail_rows=[{"id": "usb_c", "detail": "0.3 mm interference"}])
        script = [calls(call("create_object")), say("Done."), say("Done."), say("Done.")]
        result = self.run_turn("make a plate", script, validator=lambda d, c: validation)
        self.assertEqual(result.outcome, "unverified")
        self.assertIn("usb_c", result.text)

    def test_open_errors_of_the_last_round_block(self):
        tools_ = FakeTools(fail={"execute_code": True})
        script = [calls(call("create_object")), calls(call("execute_code", code="x")), say("Done."), say("Done."), say("Done.")]
        result = self.run_turn("make a plate", script, tools_=tools_)
        self.assertEqual(result.outcome, "unverified")
        self.assertIn("execute_code", result.text)
        self.assertTrue(any("OPEN_ERRORS" in m.content for m in self.notes("gate")))

    def test_an_error_followed_by_a_success_of_the_same_tool_is_resolved(self):
        tools_ = FakeTools(fail={"get_object": 1})
        script = [calls(call("get_object", n=1), call("get_object", n=2)), say("Done.")]
        result = self.run_turn("list my documents", script, tools_=tools_)
        self.assertEqual(result.outcome, "answered")
        self.assertEqual(self.worker._open_errors, [])
        again = FakeTools(fail={"get_object": 1})
        self.run_turn("list my documents", [calls(call("get_object", n=2), call("get_object", n=1)), say("Done.")], tools_=again)
        self.assertEqual(self.worker._open_errors, [])          # fails first then... (n differs, same tool): resolved
        late = FakeTools(fail={"get_object": True})
        self.run_turn("list my documents", [calls(call("get_object", n=2)), say("Done.")], tools_=late)
        self.assertEqual(len(self.worker._open_errors), 1)
        self.assertTrue(self.worker._open_errors[0].startswith("get_object: "))

    def test_a_failure_line_printed_by_a_script_blocks_until_superseded(self):
        tools_ = FakeTools(texts={"execute_code": "FAIL: the hole misses the wall\n[done]"})
        script = [calls(call("execute_code", code="x")), say("Done."), say("Done."), say("Done.")]
        result = self.run_turn("make a plate", script, tools_=tools_)
        self.assertEqual(result.outcome, "unverified")
        self.assertIn("the hole misses the wall", result.text)
        tools_.texts["execute_code"] = "all checks passed"
        self.run_turn("make a plate", [calls(call("execute_code", code="x")), say("Done.")], tools_=tools_)
        self.assertEqual(self.rec.of("turn_outcome")[0][0], "answered")

    def test_a_deliver_turn_needs_a_deliverable(self):
        script = [say("I can export that."), say("Sure."), say("OK.")]
        result = self.run_turn("export it as STL", script)
        self.assertEqual(result.outcome, "failed")
        ok = self.run_turn("export it as STL", [calls(call("export_model", format="stl")), say("Exported.")])
        self.assertIn(ok.outcome, ("answered", "validated"))
        self.assertTrue(ok.ok)

    def test_an_empty_final_reply_after_the_work_gets_one_nudge(self):
        script = [calls(call("create_object")), empty(), say("Built the plate.")]
        result = self.run_turn("make a plate", script)
        self.assertEqual((result.ok, result.text), (True, "Built the plate."))
        self.assertEqual([m.tag for m in self.history if m.synthetic], ["wrapup"])

    def test_an_empty_final_reply_twice_fails(self):
        result = self.run_turn("make a plate", [calls(call("create_object")), empty(), empty()])
        self.assertFalse(result.ok)
        self.assertIn("empty reply", result.text)

    def test_the_total_note_ceiling_ends_the_turn(self):
        cfg = AgentConfig(recovery_notes_max=1)
        script = [say("plan"), say("plan"), say("plan")]
        result = self.run_turn("make a plate", script, config=cfg)
        self.assertEqual(result.outcome, "failed")
        self.assertEqual(len(self.provider.requests), 2)       # one nudge, then the ceiling


class SteerTest(LoopCase):
    def test_a_message_queued_before_the_turn_rides_in_the_first_request(self):
        worker = self.build("list my documents", [say("ok")])
        worker.steer("also add a lid", note="[note]")
        notices = []
        worker.notice.connect(notices.append)
        worker.run_turn()
        first = self.provider.requests[0].messages[-1]
        self.assertTrue(first.content.startswith("list my documents"))
        self.assertIn("\n[USER, sent while you were working]: [note] also add a lid", first.content)

    def test_steer_emits_a_notice(self):
        worker = self.build("list my documents", [say("ok")])
        notices = []
        worker.notice.connect(notices.append)
        worker.steer("x")
        self.assertEqual(notices, ["Message queued; Oscar will see it at the next step"])

    def test_images_travel_with_the_steer(self):
        worker = self.build("list my documents", [say("ok")])
        worker.steer("look at this", images=["QUJD"])
        worker.run_turn()
        self.assertEqual(self.provider.requests[0].messages[-1].images, ["QUJD"])

    def test_a_message_sent_while_the_final_answer_streams_is_delivered_not_lost(self):
        worker_ref = {}

        def script(record):
            if record.n == 1:
                worker_ref["w"].steer("one more thing: add a fillet")
                return say("Done.")
            return say("Added the fillet too.")

        worker = self.build("list my documents", script)
        worker_ref["w"] = worker
        result = worker.run_turn()
        self.assertEqual(result.text, "Added the fillet too.")
        last = self.provider.requests[1].messages[-1]
        self.assertEqual((last.role, last.synthetic, last.tag), ("user", True, "steer"))
        self.assertIn("one more thing: add a fillet", last.content)
        self.assertEqual(self.provider.requests[1].messages[-2].content, "Done.")     # the answer stays in the history
        self.assertEqual(self.notes("gate"), [])                                       # not a gate round

    def test_steered_text_during_a_tool_round_is_appended_to_the_results_message(self):
        worker_ref = {}

        def on_call(name, args):
            worker_ref["w"].steer("make it 5 mm thick")

        worker = self.build("make a plate", [calls(call("create_object")), say("ok")], tools_=FakeTools(on_call=on_call))
        worker_ref["w"] = worker
        worker.run_turn()
        tail = self.provider.requests[1].messages[-1]
        self.assertTrue(tail.tool_results)
        self.assertIn("[USER, sent while you were working]: make it 5 mm thick", tail.content)

    def test_drain_unsent_returns_what_arrived_too_late(self):
        worker = self.build("hello?", [say("Hi.")])
        worker.run_turn()
        worker.steer("late message", note="n", images=["x"])
        self.assertEqual(worker.drain_unsent(), [{"text": "late message", "note": "n", "images": ["x"]}])
        self.assertEqual(worker.drain_unsent(), [])


class RoundBudgetTest(LoopCase):
    def test_cap_wrap_up_and_capped_outcome(self):
        cfg = AgentConfig(max_tool_rounds=3)
        script = [calls(call("list_documents")), calls(call("list_documents")), calls(call("list_documents")),
                  say("I stopped early: the plate is not modelled yet.")]
        result = self.run_turn("make a plate", script, config=cfg)
        self.assertEqual((result.ok, result.outcome), (True, "capped"))
        self.assertIn("(Stopped after 3 tool rounds with 1 required steps open: S3. Press Continue to keep going.)", result.text)
        self.assertTrue(result.text.startswith("I stopped early"))
        self.assertIn("Tool budget", self.results_messages()[-1].content)
        self.assertEqual(self.provider.requests[-1].kwargs.get("tool_choice"), "none")
        self.assertEqual(self.provider.requests[0].kwargs.get("tool_choice"), None)
        self.assertEqual(result.tool_rounds, 3)

    def test_nudge_and_gate_rounds_do_not_count(self):
        cfg = AgentConfig(max_tool_rounds=3)
        script = [say("plan"), calls(call("create_object")), calls(call("list_documents")), calls(call("list_documents")),
                  say("Done.")]
        result = self.run_turn("make a plate", script, config=cfg)
        self.assertEqual(result.tool_rounds, 3)
        self.assertEqual(result.rounds, 5)

    def test_calls_in_the_wrap_up_reply_are_dropped(self):
        cfg = AgentConfig(max_tool_rounds=1)
        script = [calls(call("list_documents")), calls(call("create_object"), text="one more"), ]
        result = self.run_turn("list my documents", script, config=cfg)
        self.assertEqual(self.fake.names(), ["list_documents"])     # the create_object of the wrap-up reply never ran
        self.assertTrue(result.text.endswith("(Stopped after 1 tool rounds — ask me to continue to keep going.)"))

    def test_the_cap_is_raised_while_a_step_keeps_closing(self):
        cfg = AgentConfig(max_tool_rounds=4)
        script = [calls(call("list_documents")), calls(call("list_documents")),
                  calls(call("plan_update", id="S3", status="skipped", note="the base is not needed")),
                  calls(call("list_documents")), calls(call("create_object")), say("Done.")]
        result = self.run_turn(BRACKET_PROMPT, script, config=cfg)
        governors = [m.content for m in self.results_messages()]
        self.assertIn("round 3/14", governors[2])
        self.assertEqual(self.worker.ledger.round_cap, 14)
        self.assertNotIn("capped", result.outcome)

    def test_the_cap_is_not_raised_without_progress(self):
        cfg = AgentConfig(max_tool_rounds=3)
        script = [calls(call("list_documents"))] * 3 + [say("Out of rounds.")]
        self.run_turn("hello?", script, config=cfg)
        self.assertEqual(len(self.provider.requests), 4)

    def test_eighty_percent_reminder_in_the_governor(self):
        cfg = AgentConfig(max_tool_rounds=5)
        script = [calls(call("create_object")), calls(call("list_documents")), calls(call("list_documents")),
                  calls(call("list_documents")), say("done")]
        self.run_turn("make a plate", script, config=cfg)
        self.assertIn("80% of the round budget used", self.results_messages()[3].content)


class BreakerLoopTest(LoopCase):
    def test_a_repeated_failing_call_is_told_off(self):
        tools_ = FakeTools(fail={"create_object": 2})
        script = [calls(call("create_object", n=1)), calls(call("create_object", n=1)), calls(call("create_object", n=2)),
                  say("Done.")]
        result = self.run_turn("make a plate", script, tools_=tools_)
        second = self.results_messages()[1].content
        self.assertIn("[breaker] You repeated the same failing call.", second)
        self.assertEqual(result.outcome, "answered")

    def test_three_trips_end_needs_help(self):
        tools_ = FakeTools(fail={"create_object": True})
        script = [calls(call("create_object", n=1))] * 6
        result = self.run_turn("make a plate", script, tools_=tools_)
        self.assertEqual((result.ok, result.outcome), (False, "needs_help"))
        self.assertTrue(result.text.startswith("Oscar is stuck: the same call kept failing"))
        self.assertTrue(result.text.endswith('or say "skip it".'))
        self.assertEqual(self.rec.of("turn_outcome"), [("needs_help", "")])
        # every call of the last round still got its result
        self.assertEqual(len(self.results_messages()), 6)

    def test_three_failures_in_a_row_trip_the_streak_rule(self):
        tools_ = FakeTools(fail={"get_object": True, "get_objects": True, "get_view": True})
        script = [calls(call("get_object"), call("get_objects"), call("get_view")), say("Gave up.")]
        self.run_turn("hello?", script, tools_=tools_)
        self.assertIn("[breaker] Three calls in a row failed.", self.results_messages()[0].content)

    def test_two_truncated_replies_in_a_row(self):
        script = [bad_json("execute_code"), bad_json("execute_code"), say("I will send smaller calls.")]
        self.run_turn("hello?", script)
        self.assertIn("not valid JSON", self.results_messages()[0].tool_results[0].content)
        self.assertIn("[breaker] Two replies in a row were cut off.", self.results_messages()[1].content)
        self.assertEqual(self.fake.names(), [])

    def test_health_failures_count_toward_the_streak_only(self):
        tools_ = FakeTools(health_fail={"execute_code"})
        script = [calls(call("execute_code", code="a")), calls(call("execute_code", code="a")),
                  calls(call("execute_code", code="a")), say("Done."), say("Done."), say("Done.")]
        self.run_turn("make a plate", script, tools_=tools_)
        contents = [m.content for m in self.results_messages()]
        self.assertFalse(any("You repeated the same failing call" in c for c in contents))
        self.assertIn("[breaker] Three calls in a row failed.", contents[2])

    def test_a_loop_of_read_only_calls_in_a_work_turn_stalls(self):
        script = [calls(call("get_objects"))] * 12
        result = self.run_turn(BRACKET_PROMPT, script)
        stall = [m.content for m in self.results_messages() if "[stall]" in m.content]
        self.assertEqual(len(stall), 2)
        self.assertIn("Six tool rounds passed without any change to the model or the plan. Next step: S3", stall[0])
        self.assertEqual((result.ok, result.outcome), (False, "needs_help"))
        self.assertIn("Oscar is stuck:", result.text)

    def test_exploring_in_a_chat_turn_never_stalls(self):
        script = [calls(call("get_objects"))] * 10 + [say("That is everything in the document.")]
        result = self.run_turn("list my documents", script)
        self.assertEqual(result.outcome, "answered")

    def test_a_mutating_call_resets_the_stall_counter(self):
        script = [calls(call("get_objects"))] * 5 + [calls(call("create_object"))] + [calls(call("get_objects"))] * 5 + [say("done")] * 4
        result = self.run_turn(BRACKET_PROMPT, script)
        self.assertFalse(any("[stall]" in m.content for m in self.results_messages()))
        self.assertNotEqual(result.outcome, "needs_help")


class DependentsRuleTest(LoopCase):
    def test_later_mutating_calls_of_a_failed_round_are_not_run(self):
        tools_ = FakeTools(fail={"create_object": 1})
        script = [calls(call("create_object", n=1), call("edit_object", n=2), call("get_objects"), call("execute_code")),
                  say("Stopped."), say("Stopped."), say("Stopped.")]
        self.run_turn("make a plate", script, tools_=tools_)
        self.assertEqual(tools_.names(), ["create_object", "get_objects"])            # the read-only call still ran
        results = self.results_messages()[0].tool_results
        self.assertEqual([r.is_error for r in results], [True, True, False, True])
        self.assertTrue(results[1].content.startswith("Not executed: an earlier call in this round failed (create_object: "))
        self.assertTrue(results[1].content.endswith("Fix that first, then re-issue this call."))
        finished = self.rec.json_of("step_finished")
        self.assertEqual([f["skipped"] for f in finished], [False, True, False, True])
        self.assertEqual(len(self.rec.of("tool_started")), 4)
        self.assertEqual(len(self.rec.of("tool_finished")), 4)

    def test_a_health_failure_does_not_trigger_the_rule(self):
        tools_ = FakeTools(health_fail={"create_object"})
        script = [calls(call("create_object"), call("edit_object")), say("ok"), say("ok"), say("ok")]
        self.run_turn("make a plate", script, tools_=tools_)
        self.assertEqual(tools_.names(), ["create_object", "edit_object"])

    def test_skipped_calls_are_not_counted_by_the_breaker(self):
        tools_ = FakeTools(fail={"create_object": 1})
        script = [calls(call("create_object"), call("edit_object"), call("execute_code")), say("ok"), say("ok"), say("ok")]
        self.run_turn("make a plate", script, tools_=tools_)
        self.assertFalse(any("[breaker]" in m.content for m in self.results_messages()))


class WatchdogLoopTest(LoopCase):
    caps = ModelCaps(reasoning="stream", reasoning_control="openai_effort")

    def cfg(self):
        return AgentConfig(watchdog=WatchdogConfig(reasoning_chars_plan=300, reasoning_chars_exec=300))

    def runaway(self):
        return Round([("reasoning", "x" * 400), ("text", "late"), ("stop_reason", "end_turn"), ("done", "end_turn")])

    def test_first_round_gets_medium_then_low(self):
        script = [calls(call("create_object")), say("done")]
        self.run_turn("make a plate", script, caps=self.caps, config=self.cfg())
        self.assertEqual([r.kwargs.get("reasoning") for r in self.provider.requests], ["medium", "low"])
        self.assertLessEqual(self.provider.requests[1].kwargs.get("max_tokens"), 12000)
        self.assertNotIn("max_tokens", self.provider.requests[0].kwargs)

    def test_models_without_a_knob_get_no_reasoning_option(self):
        self.run_turn("hello?", [say("Hi.")], caps=ModelCaps())
        self.assertNotIn("reasoning", self.provider.requests[0].kwargs)

    def test_an_explicit_profile_effort_is_left_to_the_provider(self):
        worker = self.build("hello?", [say("Hi.")], caps=self.caps)
        worker.profile.reasoning = "high"
        worker.run_turn()
        self.assertNotIn("reasoning", self.provider.requests[0].kwargs)

    def test_trips_escalate_and_the_third_ends_the_turn(self):
        result = self.run_turn("make a plate", [self.runaway()] * 3, caps=self.caps, config=self.cfg())
        self.assertEqual((result.ok, result.outcome), (False, "needs_help"))
        self.assertTrue(result.text.startswith("Oscar is stuck: the model kept thinking instead of acting"))
        kwargs = [r.kwargs for r in self.provider.requests]
        self.assertEqual(kwargs[0].get("reasoning"), "medium")
        self.assertEqual((kwargs[1].get("reasoning"), kwargs[1].get("tool_choice")), ("off", None))
        self.assertEqual((kwargs[2].get("reasoning"), kwargs[2].get("tool_choice")), ("off", "required"))
        notes = self.notes("watchdog")
        self.assertEqual(len(notes), 2)
        self.assertIn("The app stopped your previous reply after", notes[0].content)
        self.assertIn('Act now: make the tool call for the next step (S3 "Model the part")', notes[0].content)
        for note in notes:
            self.assertEqual(self.history[self.history.index(note) - 1].content,
                             "(this reply was stopped by the app before it contained a tool call)")
        self.assertEqual(self.rec.of("round_reset").count(("watchdog",)), 3)
        self.assertTrue(any("thinking too long" in n[0] for n in self.rec.of("notice")))
        # the cut-off text never reached the history
        self.assertFalse(any(m.content == "late" for m in self.history))

    def test_a_recovered_round_resets_the_forced_options(self):
        script = [self.runaway(), calls(call("create_object")), say("Done.")]
        result = self.run_turn("make a plate", script, caps=self.caps, config=self.cfg())
        self.assertEqual(result.outcome, "answered")
        self.assertEqual([r.kwargs.get("reasoning") for r in self.provider.requests], ["medium", "off", "low"])

    def test_three_good_rounds_forgive_earlier_trips(self):
        script = [self.runaway(), calls(call("create_object")), calls(call("edit_object")), calls(call("edit_object", n=2)),
                  self.runaway(), self.runaway(), say("Done.")]
        result = self.run_turn("make a plate", script, caps=self.caps, config=self.cfg())
        self.assertNotEqual(result.outcome, "needs_help")       # 1 trip, 3 good rounds, then 2 more trips: still below 3 consecutive

    def test_models_without_a_knob_only_get_notices(self):
        clock = types_clock()

        def script(record):
            clock.advance(130)
            return Round([("reasoning", "x" * 50000), ("stop_reason", "end_turn"), ("done", "end_turn")])

        worker = self.build("hello?", script, caps=ModelCaps(), clock=clock)
        result = worker.run_turn()
        self.assertEqual(worker._watchdog.trips, 0)                       # no trip: the model has no knob to turn
        self.assertEqual(result.outcome, "failed")                        # (the empty reply is reported as before)
        self.assertTrue(any("press Act now" in n[0] for n in self.rec.of("notice")))
        self.assertEqual(self.notes("watchdog"), [])

    def test_silent_round_trips_with_an_injected_clock(self):
        clock = types_clock()

        def script(record):
            if record.n == 1:
                clock.advance(900)       # the 127.0.0.1 profile is a slow ("custom") host: 840 s allowed, not 240
                return Round([("text", "a"), ("tick", {"phase": "thinking"}), ("text", "b"), ("stop_reason", "end_turn"), ("done", "end_turn")])
            return say("ok")

        result = self.run_turn("hello?", script, caps=self.caps, clock=clock)
        self.assertEqual(result.text, "ok")
        self.assertEqual(len(self.notes("watchdog")), 1)
        self.assertEqual(self.worker._watchdog.last_reason, "round_seconds")
        self.assertEqual(self.provider.requests[1].kwargs.get("reasoning"), "off")

    def test_act_now_stops_the_running_call_like_a_trip(self):
        worker_ref = {}
        sent = []

        class Provider(ScriptedProvider):
            def chat(self, system, messages, tools, should_stop=None, **kwargs):
                self.requests.append(types.SimpleNamespace(n=len(self.requests) + 1, system=system, messages=list(messages),
                                                           tools=[t.name for t in tools], kwargs=dict(kwargs)))
                if len(self.requests) == 1:
                    yield ("text", "thinking out loud")
                    worker_ref["w"].act_now()
                    sent.append(1)
                    yield ("text", "more")
                    yield ("stop_reason", "end_turn")
                    yield ("done", "end_turn")
                else:
                    yield ("text", "Acting now.")
                    yield ("stop_reason", "end_turn")
                    yield ("done", "end_turn")

        history = [Message(role="user", content="hello?")]
        worker = agent.AgentWorker(history, provider=Provider([]), profile=profile(), caps=self.caps, config=AgentConfig())
        worker_ref["w"] = worker
        result = worker.run_turn()
        self.assertEqual(result.text, "Acting now.")
        self.assertEqual([m.tag for m in history if m.synthetic], ["watchdog"])
        self.assertEqual(worker.provider.requests[1].kwargs.get("reasoning"), "off")
        self.assertEqual(worker._watchdog.trips, 1)

    def test_watchdog_traits_follow_the_caps_and_the_host(self):
        worker = self.build("hello?", [say("x")], caps=ModelCaps(reasoning_control="openai_effort"))
        self.assertEqual(worker._watchdog_traits()[0], True)
        worker.caps = ModelCaps()
        self.assertEqual(worker._watchdog_traits()[0], False)
        worker.caps = ModelCaps(tool_args_stream=False)
        self.assertTrue(worker._watchdog_traits()[1])
        worker.caps = ModelCaps()
        worker.profile.provider, worker.profile.base_url = "openai_compat", "http://localhost:11434/v1"
        self.assertTrue(worker._watchdog_traits()[1])           # ollama is a slow host

    def test_the_kill_switch_turns_the_watchdog_off(self):
        result = self.run_turn("make a plate", [self.runaway(), calls(call("create_object")), say("Done.")],
                               caps=self.caps, config=self.cfg(), flow=False)
        self.assertEqual(self.notes("watchdog"), [])
        self.assertEqual(len(self.provider.requests), 3)


def types_clock():
    class Clock:
        now = 1000.0

        def __call__(self):
            return self.now

        def advance(self, seconds):
            self.now += seconds

    return Clock()


class CutoffTest(LoopCase):
    def test_cutoff_notes_are_synthetic_and_tagged(self):
        self.run_turn("hello?", [think(100), say("Hi.")])
        notes = self.notes("cutoff")
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].content, loopcore.CUTOFF_NOTE)
        self.assertEqual(self.history[self.history.index(notes[0]) - 1].role, "assistant")

    def test_the_request_is_captured_once_per_turn(self):
        worker = self.build("make a plate", [think(100), calls(call("create_object")), say("ok")])
        worker.run_turn()
        self.assertEqual(worker._turn_request(), "make a plate")      # the cut-off note did not replace it
        self.assertEqual(worker.ledger.request, "make a plate")


class PauseStopTest(LoopCase):
    def test_stop_while_paused_executes_zero_tools(self):
        worker_ref = {}

        def script(record):
            worker_ref["w"].pause()
            threading.Timer(0.3, worker_ref["w"].stop).start()
            return calls(call("create_object"), call("edit_object"))

        worker = self.build("make a plate", script)
        worker_ref["w"] = worker
        with self.assertRaises(StopRequested):
            worker.run_turn()
        self.assertEqual(self.fake.names(), [])
        results = self.results_messages()[0].tool_results
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.is_error and "stopped the turn" in r.content for r in results))
        self.assertTrue(self.rec.of("flow_paused")[0] == (True,))

    def test_stop_right_before_the_tool_run_still_answers_every_call(self):
        worker_ref = {}

        def on_call(name, args):
            worker_ref["w"].stop()

        worker = self.build("make a plate", [calls(call("create_object"), call("edit_object"))], tools_=FakeTools(on_call=on_call))
        worker_ref["w"] = worker
        with self.assertRaises(StopRequested):
            worker.run_turn()
        calls_ = {c.id for m in self.history for c in m.tool_calls}
        results = {r.tool_call_id for m in self.history for r in m.tool_results}
        self.assertEqual(calls_, results)

    def test_resume_releases_the_pause(self):
        worker_ref = {}

        def script(record):
            if record.n == 1:
                worker_ref["w"].pause()
                threading.Timer(0.25, worker_ref["w"].resume).start()
                return calls(call("create_object"))
            return say("ok")

        worker = self.build("make a plate", script)
        worker_ref["w"] = worker
        worker.run_turn()
        self.assertEqual(self.rec.of("flow_paused"), [(True,), (False,)])
        self.assertEqual(self.fake.names(), ["create_object"])

    def test_run_emits_stopped_after_the_teardown(self):
        project = StubProject()
        persisted = []
        worker_ref = {}

        def script(record):
            worker_ref["w"].stop()
            return say("never delivered")

        worker = self.build("make a plate", script, project=project, on_round=lambda h: persisted.append(len(h)))
        worker_ref["w"] = worker
        worker.run()
        names = self.rec.names()
        self.assertEqual(names[-2:], ["turn_outcome", "turn_finished"])
        self.assertEqual(self.rec.of("turn_outcome"), [("stopped", "")])
        self.assertEqual(self.rec.of("turn_finished"), [(False, "Stopped.")])
        end = next(c for c in project.calls if c[0] == "end_turn")
        self.assertEqual(end[2], "stopped")
        self.assertTrue(persisted)
        self.assertIn("save_ledger", project.names())
        self.assertIn("write_notes_md", project.names())         # a work turn leaves its notes even when stopped

    def test_run_emits_failed_for_an_unexpected_exception(self):
        project = StubProject()
        script = [Round(raise_exc=RuntimeError("boom"))]
        worker = self.build("make a plate", script, project=project)
        worker.run()
        self.assertEqual(self.rec.names()[-2:], ["turn_outcome", "turn_finished"])
        self.assertEqual(self.rec.of("turn_outcome"), [("failed", "")])
        self.assertEqual(self.rec.of("turn_finished"), [(False, "RuntimeError: boom")])
        self.assertEqual(next(c for c in project.calls if c[0] == "end_turn")[2], "failed")

    def test_direct_run_turn_raises_the_stop(self):
        worker = self.build("make a plate", [say("x")])
        worker.stop()
        with self.assertRaises(StopRequested):
            worker.run_turn()
        self.assertEqual(self.rec.of("turn_finished"), [])


class IntentCarryTest(LoopCase):
    def test_other_intent_sees_the_build_tools_and_escalates_on_the_first_geometry_call(self):
        project = StubProject()
        script = [calls(call("create_object", obj_type="Part::Box")), say("Made a box.")]
        result = self.run_turn("show me three views", script, project=project)
        names = self.provider.requests[0].tools
        self.assertIn("create_object", names)
        self.assertIn("execute_code", names)
        self.assertEqual(self.worker.ledger.intent, "generic_part")
        self.assertEqual(result.ledger["intent"], "generic_part")
        self.assertEqual(self.rec.names().count("plan_changed") >= 1, True)
        self.assertEqual(self.rec.json_of("plan_changed")[0]["intent"], "generic_part")
        self.assertIn("[governor] round 1/30", self.results_messages()[0].content)
        self.assertEqual(result.geometry_ok, 1)

    def test_escalation_archives_the_carried_ledger(self):
        project = StubProject()
        carried = Ledger.from_request("make a gear")
        self.run_turn("thanks", [calls(call("create_object")), say("Made a box.")], project=project, ledger=carried)
        self.assertIn("archive_ledger", project.names())

    def test_continue_resumes_the_carried_ledger(self):
        carried = Ledger.from_request("make a plate")
        result = self.run_turn("continue", [calls(call("create_object")), say("Done.")], ledger=carried)
        self.assertIs(self.worker.ledger, carried)
        self.assertEqual(result.ledger["request"], "make a plate")
        self.assertEqual(carried.step("S3").status, "done")
        self.assertEqual(self.rec.json_of("plan_changed")[0]["request"], "make a plate")

    def test_the_resume_prompt_continues(self):
        carried = Ledger.from_request("make a plate")
        self.run_turn(carried.resume_prompt(), [calls(call("create_object")), say("Done.")], ledger=carried)
        self.assertIs(self.worker.ledger, carried)

    def test_thanks_does_not_resume(self):
        carried = Ledger.from_request("make a plate")
        before = carried.to_dict()
        result = self.run_turn("thanks", [say("You're welcome.")], ledger=carried)
        self.assertIsNot(self.worker.ledger, carried)
        self.assertEqual(result.ledger, before)
        self.assertEqual(result.outcome, "answered")
        self.assertEqual(self.rec.of("plan_changed"), [])

    def test_a_question_in_the_middle_of_a_build_leaves_the_plan_alone(self):
        carried = Ledger.from_request("make a plate")
        before = carried.to_dict()
        result = self.run_turn("why did you stop?", [calls(call("get_objects")), say("I was waiting for you.")], ledger=carried)
        self.assertEqual(result.ledger, before)
        self.assertEqual(carried.to_dict(), before)

    def test_a_new_work_request_replaces_and_archives_the_old_plan(self):
        project = StubProject()
        carried = Ledger.from_request("make a plate")
        result = self.run_turn("make a gear", [calls(call("create_object")), say("Done.")], ledger=carried, project=project)
        self.assertEqual(result.ledger["request"], "make a gear")
        self.assertIn("archive_ledger", project.names())
        self.assertIsNot(self.worker.ledger, carried)

    def test_the_carried_ledger_may_be_a_dict(self):
        carried = Ledger.from_request("make a plate").to_dict()
        self.run_turn("continue", [calls(call("create_object")), say("Done.")], ledger=carried)
        self.assertEqual(self.worker.ledger.step("S3").status, "done")

    def test_the_typed_text_decides_not_notes_or_attachments(self):
        content = ("[I reverted every CAD change you made after that point; check the workspace state below.]\n\n"
                   "what is in this document?\n\n--- attached files ---\nbracket for a motor: add 4 holes and a lid")
        result = self.run_turn([Message(role="user", content=content)], [say("It holds one box.")])
        self.assertEqual((result.outcome, result.ledger), ("answered", None))
        self.assertEqual(self.rec.of("plan_changed"), [])
        explicit = self.run_turn([Message(role="user", content="[note]\n\nanything at all")], [say("ok")],
                                 request_text="list my documents")
        self.assertEqual(explicit.outcome, "answered")
        self.assertEqual(self.worker.ledger.request, "list my documents")

    def test_attachments_reach_only_the_classifier(self):
        worker = self.build("here you go", [say("I see a STEP file.")] * 3, attachments=["/tmp/part.step"])
        worker.run_turn()
        self.assertEqual(worker.ledger.intent, "generic_part")


class KillSwitchTest(LoopCase):
    def test_flow_off_runs_without_ledger_gate_governor_or_breaker(self):
        project = StubProject()
        tools_ = FakeTools(fail={"create_object": True})
        script = [calls(call("create_object", n=1)), calls(call("create_object", n=1)), calls(call("create_object", n=1)),
                  say("Done?")]
        result = self.run_turn("make a plate", script, tools_=tools_, project=project, flow=False)
        self.assertTrue(result.ok)
        self.assertEqual(result.outcome, "answered")
        self.assertEqual(result.text, "Done?")
        self.assertEqual(self.rec.of("plan_changed"), [])
        for m in self.results_messages():
            self.assertEqual(m.content, "")
        self.assertIsNone(self.fake.ctxs[0].ledger)
        self.assertIs(self.fake.ctxs[0].project, project)           # Project recording stays on
        self.assertEqual(project.calls[0][0], "begin_turn")
        self.assertFalse(project.calls[0][3])
        self.assertIn("end_turn", project.names())
        self.assertNotIn("finalize", project.names())

    def test_the_turn_flag_reaches_run_tool_through_the_context_state(self):
        """run_tool skips its health / ledger hooks from ``ctx.state['flow_enforcement']`` (the pref and the argument, not only the env)."""
        self.run_turn("make a plate", [calls(call("create_object", n=1)), say("Done?")], flow=False)
        self.assertIs(self.fake.ctxs[0].state.get("flow_enforcement"), False)
        self.run_turn("make a plate", [calls(call("create_object", n=1)), say("Built the plate.")], flow=True)
        self.assertIs(self.fake.ctxs[-1].state.get("flow_enforcement"), True)

    def test_the_legacy_keyword_nudge_is_reachable_only_with_the_switch_off(self):
        result = self.run_turn("make a plate", [say("I would do this."), say("Still words.")], flow=False)
        self.assertEqual(result.text, "Still words.")
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(self.notes("gate")[0].content, agent._EXECUTION_NOTE)
        self.assertTrue(any("without doing any CAD work" in n[0] for n in self.rec.of("notice")))

    def test_the_environment_variable_is_honoured(self):
        with mock.patch.dict(os.environ, {"OSCAR_FLOW": "0"}):
            worker = self.build("make a plate", [say("words"), say("more words")])
        self.assertFalse(worker.config.flow_enforcement)
        worker.run_turn()
        self.assertEqual(self.rec.of("plan_changed"), [])

    def test_an_explicit_flow_argument_wins_over_the_environment(self):
        with mock.patch.dict(os.environ, {"OSCAR_FLOW": "0"}):
            worker = self.build("make a plate", [say("words")], flow=True)
        self.assertTrue(worker.config.flow_enforcement)


class PersistenceTest(LoopCase):
    def test_on_round_runs_after_every_round_and_at_the_end(self):
        sizes = []
        self.run_turn("make a plate", [calls(call("create_object")), calls(call("edit_object")), say("Done.")],
                      on_round=lambda h: sizes.append(len(h)))
        self.assertGreaterEqual(len(sizes), 4)
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(sizes[-1], len(self.history))

    def test_a_failing_save_never_kills_the_turn(self):
        project = StubProject()
        project.fail_on = {"save_ledger", "end_turn", "write_notes_md", "begin_turn", "finalize"}

        def explode(history):
            raise OSError("disk full")

        result = self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], project=project, on_round=explode,
                               validator=lambda d, c: fake_validation(True))
        self.assertTrue(result.ok)
        self.assertEqual(result.outcome, "validated")

    def test_the_ledger_is_saved_when_the_plan_changes(self):
        project = StubProject()
        self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], project=project)
        self.assertTrue(project.saved)
        self.assertEqual(project.saved[-1]["intent"], "generic_part")

    def test_artifacts_are_announced_for_a_materialized_project(self):
        root = tempfile.mkdtemp(prefix="proj_", dir=isolation.sandbox_root())
        for name in ("model.FCStd", "NOTES.md"):
            open(os.path.join(root, name), "w").close()
        os.makedirs(os.path.join(root, "snapshots"))
        open(os.path.join(root, "snapshots", "001_x.png"), "w").close()
        project = StubProject(root)
        self.run_turn("make a plate", [calls(call("create_object")), say("Done.")], project=project)
        payloads = self.rec.json_of("artifacts_changed")
        self.assertTrue(payloads)
        last = payloads[-1]
        self.assertEqual(last["project_dir"], root)
        kinds = {f["kind"] for f in last["files"]}
        self.assertTrue({"fcstd", "notes", "snapshot"} <= kinds)
        self.assertEqual(self.rec.names()[-3:], ["artifacts_changed", "turn_outcome", "turn_finished"])


class ProgressTest(LoopCase):
    def test_ticks_become_progress_events(self):
        tick = {"elapsed": 1.2, "reasoning_chars": 512, "text_chars": 0, "tool_chars": 0, "out_tokens": None,
                "phase": "thinking", "tool": None, "retry": None, "idle_s": 0.4}
        script = [Round([("tick", tick), ("reasoning", "hmm"), ("tick", dict(tick, phase="writing", text_chars=10)),
                         ("text", "Hi."), ("stop_reason", "end_turn"), ("done", "end_turn")])]
        self.run_turn("hello?", script)
        events = self.rec.json_of("progress")
        thinking = events[0]
        for key in ("round", "round_cap", "phase", "step_id", "step_title", "steps_done", "steps_total", "elapsed_round",
                    "elapsed_turn", "reasoning_chars", "text_chars", "tool", "tool_chars", "activity", "retry", "idle_s",
                    "started_at"):
            self.assertIn(key, thinking)
        self.assertEqual((thinking["activity"], thinking["reasoning_chars"], thinking["round"], thinking["round_cap"]),
                         ("thinking", 512, 1, 30))
        self.assertEqual(events[1]["activity"], "writing")

    def test_running_a_tool_is_announced(self):
        self.run_turn("make a plate", [calls(call("create_object")), say("ok")])
        events = self.rec.json_of("progress")
        running = [e for e in events if e["activity"] == "running_tool"]
        self.assertEqual(running[0]["tool"], "create_object")
        self.assertEqual(running[0]["phase"], "BUILD")

    def test_the_loop_works_when_ticks_never_arrive(self):
        result = self.run_turn("hello?", [say("Hi.")])
        self.assertEqual(result.text, "Hi.")


class ResetEventTest(LoopCase):
    def test_a_reset_voids_the_partial_text_and_is_forwarded(self):
        script = [Round([("text", "partial"), ("reset", "retry"), ("text", "Whole answer."), ("stop_reason", "end_turn"),
                         ("done", "end_turn")])]
        result = self.run_turn("hello?", script)
        self.assertEqual(result.text, "Whole answer.")
        self.assertEqual(self.rec.of("round_reset"), [("retry",)])


class RealPipelineTest(unittest.TestCase):
    """The loop with the REAL ``tools.run_tool`` (documents are real; the provider is scripted)."""

    def test_a_real_build_ticks_the_ledger_once_and_counts_the_geometry(self):
        import FreeCAD as App

        script = [calls(ToolCall(id="c0", name="create_document", arguments={"name": "LoopRealDoc"}),
                        call("create_object", doc_name="LoopRealDoc", obj_type="Part::Box", obj_name="Plate",
                             include_screenshot=False)),
                  say("Modelled the plate.")]
        history = [Message(role="user", content="make a plate")]
        worker = agent.AgentWorker(history, provider=ScriptedProvider(script), profile=profile(), caps=ModelCaps(),
                                   validator=lambda d, c: fake_validation(True))
        rec = SignalRecorder(worker)
        try:
            result = worker.run_turn()
            self.assertEqual((result.ok, result.outcome), (True, "validated"))
            self.assertEqual(result.geometry_ok, 1)
            step = worker.ledger.step("S3")
            self.assertEqual(len([e for e in step.evidence if e["kind"] == "auto"]), 1)      # ticked by run_tool, not twice
            self.assertEqual(step.status, "done")
            finished = rec.json_of("step_finished")
            self.assertEqual([f["name"] for f in finished], ["create_document", "create_object"])
            self.assertTrue(all(f["ok"] for f in finished), finished)
            self.assertIsNotNone(App.getDocument("LoopRealDoc").getObject("Plate"))
        finally:
            if "LoopRealDoc" in App.listDocuments():
                App.closeDocument("LoopRealDoc")

    def test_a_real_failure_is_reported_and_blocks_the_gate(self):
        script = [calls(call("create_object", doc_name="NoSuchLoopDoc", obj_type="Part::Box", obj_name="B",
                             include_screenshot=False)), say("Done."), say("Done."), say("Done.")]
        history = [Message(role="user", content="make a plate")]
        worker = agent.AgentWorker(history, provider=ScriptedProvider(script), profile=profile(), caps=ModelCaps())
        result = worker.run_turn()
        self.assertEqual((result.ok, result.outcome), (False, "failed"))
        self.assertEqual(result.geometry_ok, 0)
        self.assertTrue(history[2].tool_results[0].is_error)


class TelemetryTest(LoopCase):
    def capture(self):
        events = []
        patcher = mock.patch.object(agent, "log_event", lambda kind, **fields: events.append((kind, fields)))
        patcher.start()
        self.addCleanup(patcher.stop)
        return events

    def kinds(self, events):
        return [k for k, _f in events]

    def test_turn_lifecycle_events(self):
        events = self.capture()
        secret = "SECRET_CODE_BODY = 42"
        script = [say("plan"), calls(call("execute_code", code=secret)), say("Done.")]
        result = self.run_turn("make a plate", script, tools_=FakeTools())
        kinds = self.kinds(events)
        self.assertEqual(kinds[0], "turn_start")
        self.assertEqual(kinds[-1], "turn_end")
        start = dict(events[0][1])
        self.assertEqual((start["intent"], start["flow"], start["round_cap"]), ("generic_part", True, 30))
        self.assertEqual(set(start["profile"]), {"name", "provider", "model", "base_url"})
        self.assertEqual(len(start["prompt_hash"]), 12)
        self.assertEqual(start["turn_id"][0], "T")
        rounds = [f for k, f in events if k == "round"]
        self.assertEqual([r["n"] for r in rounds], [1, 2, 3])
        for key in ("n", "tool_round", "phase", "stop_reason", "text_chars", "reasoning_chars", "tool_calls", "usage", "wall_s"):
            self.assertIn(key, rounds[0])
        end = dict(events[-1][1])
        self.assertEqual((end["outcome"], end["rounds"], end["tool_rounds"], end["geometry_ok"]), (result.outcome, 3, 1, 1))
        self.assertIn("gate_nudge", kinds)
        self.assertEqual(next(f for k, f in events if k == "gate_nudge")["codes"][0], "NO_WORK")
        tool_start = next(f for k, f in events if k == "tool_start")
        self.assertEqual((tool_start["name"], tool_start["arg_names"]), ("execute_code", ["code"]))
        self.assertTrue(tool_start["script_hash"])
        self.assertNotIn(secret, json.dumps(events, default=str))                  # never code bodies
        self.assertNotIn("api_key", json.dumps(events, default=str))

    def test_recovery_events(self):
        events = self.capture()
        caps = ModelCaps(reasoning="stream", reasoning_control="openai_effort")
        cfg = AgentConfig(watchdog=WatchdogConfig(reasoning_chars_plan=100, reasoning_chars_exec=100), max_tool_rounds=2)
        script = [Round([("reasoning", "x" * 200), ("stop_reason", "end_turn"), ("done", "end_turn")]), think(50),
                  calls(call("get_objects")), calls(call("get_objects")), say("Out of rounds.")]
        self.run_turn("hello?", script, caps=caps, config=cfg)
        kinds = self.kinds(events)
        self.assertIn("watchdog_trip", kinds)
        self.assertIn("cutoff_retry", kinds)
        trip = next(f for k, f in events if k == "watchdog_trip")
        self.assertEqual((trip["reason"], trip["round"]), ("reasoning_chars", 1))

    def test_breaker_and_bad_argument_events(self):
        events = self.capture()
        script = [bad_json("execute_code"), bad_json("execute_code"), say("smaller calls from now on")]
        self.run_turn("hello?", script)
        kinds = self.kinds(events)
        self.assertEqual(kinds.count("bad_arguments"), 2)
        self.assertEqual(next(f for k, f in events if k == "breaker_trip")["rule"], "truncation")

    def test_steer_and_stop_events(self):
        events = self.capture()
        worker = self.build("list my documents", [say("ok")])
        worker.steer("hi")
        worker.run_turn()
        self.assertIn("steer_delivered", self.kinds(events))
        worker2 = self.build("list my documents", [say("ok")])
        worker2.stop()
        worker2.run()
        self.assertIn("user_stop", self.kinds(events))
        self.assertEqual(events[-1][0], "turn_end")
        self.assertEqual(events[-1][1]["outcome"], "stopped")

    def test_validation_and_cap_events(self):
        events = self.capture()
        cfg = AgentConfig(max_tool_rounds=4)
        script = [calls(call("list_documents")), calls(call("list_documents")),
                  calls(call("plan_update", id="S3", status="skipped", note="not needed")), calls(call("create_object")), say("Done.")]
        self.run_turn(BRACKET_PROMPT, script, config=cfg, validator=lambda d, c: fake_validation(True))
        kinds = self.kinds(events)
        self.assertIn("round_cap_raised", kinds)
        self.assertEqual(next(f for k, f in events if k == "round_cap_raised")["cap"], 14)


class ReviewRegressionTest(LoopCase):
    """Defects found by an earlier review (each test names the failure it pins)."""

    def capture(self):
        events = []
        patcher = mock.patch.object(agent, "log_event", lambda kind, **fields: events.append((kind, fields)))
        patcher.start()
        self.addCleanup(patcher.stop)
        return events

    def test_a_stopped_turn_logs_exactly_one_turn_end_and_one_user_stop(self):
        events = self.capture()
        worker_ref = {}

        def script(record):
            worker_ref["w"].stop()
            return say("never delivered")

        worker = self.build("make a plate", script)
        worker_ref["w"] = worker
        worker.run()
        kinds = [k for k, _f in events]
        self.assertEqual((kinds.count("turn_end"), kinds.count("user_stop")), (1, 1))
        self.assertLess(kinds.index("user_stop"), kinds.index("turn_end"))
        self.assertEqual(next(f for k, f in events if k == "turn_end")["outcome"], "stopped")

    def test_a_crashed_turn_logs_one_turn_end_that_carries_the_exception(self):
        events = self.capture()
        worker = self.build("make a plate", [Round(raise_exc=RuntimeError("boom"))])
        worker.run()
        ends = [f for k, f in events if k == "turn_end"]
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["outcome"], "failed")
        self.assertEqual(ends[0]["reasons"], ["RuntimeError: boom"])

    def test_a_script_failure_line_counts_even_when_the_health_check_also_flagged_the_call(self):
        tools_ = FakeTools(texts={"execute_code": "FAIL: the lid does not close\n[health] NOT OK"}, health_fail={"execute_code"})
        script = [calls(call("execute_code", code="x")), say("Done."), say("Done."), say("Done.")]
        result = self.run_turn("make a plate", script, tools_=tools_)
        self.assertEqual(result.outcome, "unverified")
        self.assertTrue(any("SCRIPT_FAIL" in m.content for m in self.notes("gate")))
        self.assertIn("the lid does not close", result.text)

    def test_bookkeeping_that_raises_never_leaves_tool_calls_without_results(self):
        events = self.capture()
        with mock.patch.object(agent, "format_governor", side_effect=RuntimeError("governor bug")):
            result = self.run_turn("make a plate", [calls(call("create_object"), call("get_objects")), say("Built the plate.")])
        self.assertTrue(result.ok)
        assistant = next(m for m in self.history if m.tool_calls)
        results = self.results_messages()[0]
        self.assertEqual([r.tool_call_id for r in results.tool_results], [c.id for c in assistant.tool_calls])
        self.assertIn(("loop_error", "round_notes"), [(k, f.get("where")) for k, f in events])
        self.assertEqual(self.provider.requests[-1].messages[-1].tool_results[0].tool_call_id, assistant.tool_calls[0].id)

    def test_the_skipped_calls_of_a_crashed_round_do_not_blame_the_user(self):
        worker = self.build("make a plate", [calls(call("create_object"), call("get_objects"))])
        with mock.patch.object(worker, "_execute", side_effect=RuntimeError("tool crashed hard")):
            with self.assertRaises(RuntimeError):
                worker.run_turn()
        texts = [r.content for r in self.results_messages()[0].tool_results]
        self.assertEqual(len(texts), 2)
        for text in texts:
            self.assertIn("internal error", text)
            self.assertNotIn("user stopped", text)

    def test_dropped_wrap_up_calls_are_not_replayed_as_provider_blocks(self):
        """An Anthropic assistant message that replays a tool_use block without a result is an HTTP 400 on the next request."""
        blocks = [{"type": "thinking", "thinking": "hmm", "signature": "s"}, {"type": "text", "text": "Wrapping up."},
                  {"type": "tool_use", "id": "cX", "name": "create_object", "input": {"obj_type": "Part::Box"}}]
        wrap_up = Round([("text", "Wrapping up."), ("assistant_blocks", blocks), ("stop_reason", "tool_use"),
                         ("tool_calls", [call("create_object", call_id="cX", obj_type="Part::Box")])])
        result = self.run_turn("make a plate", [calls(call("list_documents")), wrap_up], config=AgentConfig(max_tool_rounds=1))
        self.assertEqual(result.outcome, "capped")
        self.assertEqual(self.fake.names(), ["list_documents"])
        stored = self.history[-1]
        self.assertEqual(stored.role, "assistant")
        self.assertEqual([b["type"] for b in stored.provider_blocks], ["thinking", "text"])
        self.assertFalse(any(b.get("type") == "tool_use" for b in stored.provider_blocks))

    def test_the_telemetry_context_carries_the_chat_id_of_a_real_project(self):
        """Project has no ``chat_id`` attribute (the id lives in its manifest); the context must still get it."""
        from oscar.project import Project

        root = tempfile.mkdtemp(prefix="proj_", dir=isolation.sandbox_root())
        project = Project(root, chat_id="20261001-000000-abc123", title="Plate", create=False)
        seen = {}
        original = agent._turnlog.set_context

        def spy(**fields):
            seen.update(fields)
            original(**fields)

        with mock.patch.object(agent._turnlog, "set_context", spy):
            self.run_turn("hello?", [say("Hi.")], project=project)
        self.assertEqual(seen.get("chat_id"), "20261001-000000-abc123")
        self.assertEqual(seen.get("model"), "scripted")
        self.assertTrue(seen.get("turn_id", "").startswith("T"))
        self.assertEqual(agent._turnlog.get_context(), {})            # cleared by the teardown

    def test_every_project_call_of_the_loop_matches_the_real_project_api(self):
        """The loop guards its persistence calls, so an API mismatch would only show as a silent ``persist_error``."""
        from support.fakes import temp_project

        events = self.capture()
        project = temp_project(create=False)
        result = self.run_turn("make a plate", [calls(call("create_object", obj_type="Part::Box")), say("Built the plate.")],
                               project=project, validator=lambda d, c: fake_validation(True))
        self.assertEqual(result.outcome, "validated")
        self.assertEqual([(k, f) for k, f in events if k in ("persist_error", "project_error")], [])
        self.assertTrue(os.path.isfile(os.path.join(project.root, "ledger.json")))
        self.assertTrue(os.path.isfile(os.path.join(project.root, "NOTES.md")))
        saved = json.load(open(os.path.join(project.root, "ledger.json"), encoding="utf-8"))
        self.assertEqual(saved["intent"], "generic_part")
        self.assertIn("## Build Summary", result.text)
        self.assertEqual(result.project_dir, project.root)
        for later in ("save_ledger", "finalize"):
            self.assertNotIn(later, [f.get("where") for k, f in events if k == "persist_error"])

    def test_the_kill_switch_never_changes_a_config_object_the_caller_shares(self):
        shared = AgentConfig()
        worker = self.build("make a plate", [say("x")], config=shared, flow=False)
        self.assertFalse(worker.config.flow_enforcement)
        self.assertTrue(shared.flow_enforcement)
        self.assertIsNot(worker.config, shared)

    def test_artifacts_are_announced_after_the_plan_change_of_the_same_call(self):
        root = tempfile.mkdtemp(prefix="proj_", dir=isolation.sandbox_root())
        open(os.path.join(root, "model.FCStd"), "w").close()
        self.run_turn("make a plate", [calls(call("create_object", obj_type="Part::Box")), say("Done.")],
                      project=StubProject(root))
        names = [n for n in self.rec.names() if n in ("tool_started", "step_started", "step_finished", "tool_finished",
                                                      "plan_changed", "artifacts_changed")]
        first = names.index("tool_started")
        self.assertEqual(names[first:first + 6], ["tool_started", "step_started", "step_finished", "tool_finished",
                                                  "plan_changed", "artifacts_changed"])


if __name__ == "__main__":
    unittest.main()
