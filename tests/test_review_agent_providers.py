"""Skeptic review of the agent loop and the providers: adversarial tests.

Every class here pins a defect that the review found (and fixed) or a hostile situation the loop must survive:

* a text-mode tool call cut off at the output-token limit (a weak model writing a long script into a ``<tool_call>``
  block) used to be released as the reply, raw markup and all, instead of being answered with the "split the script"
  guidance a native call gets;
* a ``</tool_call>`` tag quoted inside a JSON string left tag remnants in the reply;
* two tool calls with one id (a local server that numbers its calls ``0`` in every reply) produced a conversation that
  strict hosts reject;
* "continue" after a FINISHED build became the request of the pinned task memory;
* the connection test demoted a model to text tool mode (remembered for good) after a transient error;
* seeded chaos runs: random weak-model scripts through the REAL loop (and the real text-mode and Anthropic wire) must
  keep every invariant (each call answered, ids unique, alternation, one ``turn_finished``, valid requests);
* a fuzz of the derived request view (compaction): every view is a valid request, whatever the history and window.
"""

import json
import os
import random
import sys
import types
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import agent, context, flow, flow_intent, loopcore, params, provider_check, provider_text, providers, tools  # noqa: E402
from oscar import models as _models  # noqa: E402
from oscar.flow import AgentConfig, Ledger, OUTCOMES  # noqa: E402
from oscar.models import ModelCaps  # noqa: E402
from oscar.provider_transport import ProviderError, StopRequested  # noqa: E402
from oscar.providers import Message, ToolCall, ToolResult  # noqa: E402
from support.realflow import RealFlowCase  # noqa: E402
from support.scripted_llm import (Round, ScriptedProvider, bad_json, call, calls, empty, malformed_json, nameless,  # noqa: E402
                                  provider_error, say, think)
from support.server import ScriptedServer  # noqa: E402
from support.signal_recorder import SignalRecorder  # noqa: E402
from test_agent_loop import FakeTools, LoopCase, PI_PROMPT, StubProject, profile  # noqa: E402
from test_agent_loop_wire import WireCase  # noqa: E402

NAMES = ("execute_code", "get_objects", "create_object")


def make_filter(text_mode=True):
    return provider_text.TextStreamFilter(NAMES, text_mode=text_mode)


def run_filter(chunks, *, stop="stop", text_mode=True):
    """``(shown text, reasoning, calls, events)`` of a stream fed chunk by chunk."""
    flt = make_filter(text_mode)
    events = []
    for chunk in chunks:
        events += flt.feed(chunk)
    tail, found = flt.finish(stop_reason=stop, complete=True, extract=True)
    events += tail
    shown = "".join(p for k, p in events if k == "text")
    reasoning = "".join(p for k, p in events if k == "reasoning")
    return shown, reasoning, found, events


class TruncatedTextCallTest(unittest.TestCase):
    """A text-mode call block cut off by the output-token limit is a truncated CALL, not the reply."""

    CUT = ('I will build it.\n<tool_call>\n{"name": "execute_code", "arguments": {"code": "import FreeCAD\\nx = 1\\n'
           'doc = App.newDocument()\\nfor i in range(10):\\n   pass')

    def test_a_block_cut_at_the_token_limit_becomes_a_truncated_call(self):
        shown, _reasoning, found, events = run_filter([self.CUT[:30], self.CUT[30:90], self.CUT[90:]], stop="length")
        self.assertEqual([c.name for c in found], ["execute_code"])
        self.assertTrue(found[0].args_error.startswith("truncated"), found[0].args_error)
        self.assertIn("_raw", found[0].arguments)
        self.assertTrue(found[0].raw_args)
        self.assertEqual(shown.strip(), "I will build it.")            # the user never sees the raw markup
        self.assertNotIn("tool_call", shown)
        self.assertTrue(any(kind == "tool_progress" for kind, _p in events))

    def test_the_loop_answers_it_with_the_split_the_script_guidance(self):
        _shown, _r, found, _e = run_filter([self.CUT], stop="length")
        self.assertTrue(loopcore.args_problem(found[0]))
        message = loopcore.bad_arguments_message(found[0], "length")
        self.assertIn("not valid JSON", message)
        self.assertIn("smaller execute_code calls", message)

    def test_a_normal_stop_with_an_unfinished_block_is_still_the_reply_text(self):
        shown, _r, found, _e = run_filter([self.CUT], stop="stop")
        self.assertEqual(found, [])
        self.assertIn("<tool_call>", shown)           # nothing says it was a call: it is text

    def test_a_cut_before_the_tool_name_is_known_stays_text(self):
        shown, _r, found, _e = run_filter(['Next.\n<tool_call>\n{"na'], stop="length")
        self.assertEqual(found, [])
        self.assertIn("<tool_call>", shown)

    def test_an_unknown_tool_name_is_never_made_a_call(self):
        cut = '<tool_call>\n{"name": "format_disk", "arguments": {"code": "x'
        _shown, _r, found, _e = run_filter([cut], stop="length")
        self.assertEqual(found, [])

    def test_native_mode_never_turns_markup_into_a_call(self):
        _shown, _r, found, _e = run_filter([self.CUT], stop="length", text_mode=False)
        self.assertEqual(found, [])

    def test_a_fenced_block_cut_off_is_a_truncated_call_too(self):
        cut = 'Here:\n```json\n{"name": "execute_code", "arguments": {"code": "x = 1\\ny = 2\\n'
        shown, _r, found, _e = run_filter([cut[:20], cut[20:]], stop="length")
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].args_error.startswith("truncated"))
        self.assertNotIn("```json", shown)

    def test_a_block_that_completed_before_the_cut_still_runs(self):
        text = ('<tool_call>{"name": "get_objects", "arguments": {}}</tool_call>\n'
                '<tool_call>\n{"name": "execute_code", "arguments": {"code": "x = 1')
        _shown, _r, found, _e = run_filter([text], stop="length")
        self.assertEqual([c.name for c in found], ["get_objects"])                 # the finished call, as before
        self.assertFalse(found[0].args_error)


class QuotedCloseTagTest(unittest.TestCase):
    def test_a_closing_tag_inside_a_json_string_does_not_split_the_call(self):
        text = ('Now.\n<tool_call>\n{"name": "execute_code", "arguments": {"code": "print(\'</tool_call>\')"}}\n'
                '</tool_call>\nDone.')
        found, rest = provider_text.extract_text_tool_calls(text, NAMES)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].arguments["code"], "print('</tool_call>')")
        self.assertNotIn("tool_call", rest)
        self.assertIn("Now.", rest)
        self.assertIn("Done.", rest)

    def test_the_same_through_the_stream_filter_in_any_chunking(self):
        text = '<tool_call>{"name": "get_objects", "arguments": {"doc_name": "a</tool_call>b"}}</tool_call>After.'
        reference = run_filter([text])
        self.assertEqual([c.arguments for c in reference[2]], [{"doc_name": "a</tool_call>b"}])
        self.assertEqual(reference[0].strip(), "After.")
        for size in (1, 2, 3, 7, 11):
            chunks = [text[i:i + size] for i in range(0, len(text), size)]
            shown, _r, found, _e = run_filter(chunks)
            self.assertEqual([c.arguments for c in found], [{"doc_name": "a</tool_call>b"}], size)
            self.assertEqual(shown.strip(), "After.", size)

    def test_a_plain_block_is_unchanged(self):
        text = '<tool_call>{"name": "get_objects", "arguments": {"doc_name": "A"}}</tool_call>'
        found, rest = provider_text.extract_text_tool_calls(text, NAMES)
        self.assertEqual(([c.name for c in found], rest), (["get_objects"], ""))


class TruncatedTextCallRoundTripTest(WireCase):
    """The same defect through the REAL OpenAI-compatible provider in text tool mode, over a socket."""

    def test_a_cut_script_is_not_run_not_shown_and_answered_with_guidance(self):
        cut = ('Building the plate.\n<tool_call>\n{"name": "execute_code", "arguments": {"code": "'
               + "import FreeCAD\\n" * 60)
        server = self.serve([Round([("text", cut), ("stop_reason", "max_tokens"), ("done", "max_tokens")]), say("All done.")])
        fake = FakeTools()
        result = self.run_agent(server, "show me three views", override={"tools": "text"}, fake=fake)
        self.assertEqual((result.ok, result.outcome, result.text), (True, "answered", "All done."))
        self.assertEqual(fake.calls, [])                                    # a truncated call never runs
        self.assertNotIn("tool_call", self.rec.text())                      # the raw markup was never shown
        self.assertEqual(len(server.requests), 2)
        second = server.requests[1].json["messages"]
        self.assertFalse(any("tool_calls" in m or m["role"] == "tool" for m in second))      # text tool mode wire
        last = second[-1]["content"]
        self.assertIn('<tool_result id="', last)
        self.assertIn("not valid JSON", last)
        self.assertIn("smaller execute_code calls", last)
        replayed = next(m["content"] for m in second if m["role"] == "assistant")
        self.assertIn('"arguments": {}', replayed)                          # an undecodable call is replayed empty

    def test_the_breaker_sees_two_cut_replies_in_a_row(self):
        cut = '<tool_call>\n{"name": "execute_code", "arguments": {"code": "' + "x = 1\\n" * 50
        cut_round = Round([("text", cut), ("stop_reason", "max_tokens"), ("done", "max_tokens")])
        server = self.serve([cut_round, cut_round, say("Smaller steps next.")], strict=False)
        result = self.run_agent(server, "show me three views", override={"tools": "text"}, fake=FakeTools())
        self.assertEqual(result.outcome, "answered")
        self.assertTrue(any("[breaker]" in (m.content or "") for m in self.history if m.tool_results))


class HiddenToolInTextModeTest(WireCase):
    """A call to a registered tool that this round does not offer runs in text mode too ("visibility is not permission")."""

    BLOCK = '<tool_call>\n{"name": "build_enclosure", "arguments": {"stage": "shell"}}\n</tool_call>'

    def test_the_call_runs_and_is_not_shown_as_text(self):
        server = self.serve([Round([("text", "Starting the shell.\n" + self.BLOCK), ("stop_reason", "stop"), ("done", "stop")]),
                             say("Built the shell.")], strict=False, repeat_last=True)
        fake = FakeTools()
        self.run_agent(server, PI_PROMPT, override={"tools": "text"}, fake=fake, validator=False)
        first = server.requests[0]
        self.assertNotIn("- build_enclosure(", system_text(first))          # it was NOT offered in this phase ...
        self.assertEqual(fake.names()[:1], ["build_enclosure"])               # ... and the call still ran
        self.assertNotIn("tool_call", self.rec.text())

    def test_a_name_that_is_not_registered_is_still_just_text(self):
        server = self.serve([Round([("text", '<tool_call>\n{"name": "format_disk", "arguments": {}}\n</tool_call>'),
                                    ("stop_reason", "stop"), ("done", "stop")]), say("ok")], strict=False, repeat_last=True)
        fake = FakeTools()
        self.run_agent(server, "show me three views", override={"tools": "text"}, fake=fake)
        self.assertEqual(fake.names(), [])

    def test_the_agent_hands_the_registry_to_its_provider(self):
        worker = agent.AgentWorker([Message(role="user", content="hi")], provider=ScriptedProvider([]), profile=profile(),
                                   caps=ModelCaps())
        worker._begin_turn()
        self.addCleanup(worker._teardown, "answered", None)
        self.assertEqual(worker.provider.known_tool_names, frozenset(t.name for t in tools.get_tools()))


def system_text(request) -> str:
    body = request.json
    first = (body.get("messages") or [{}])[0]
    return first.get("content", "") if first.get("role") == "system" else str(body.get("system") or "")


class UniqueCallIdsTest(unittest.TestCase):
    def test_duplicates_in_one_reply_are_renamed_and_the_first_keeps_its_id(self):
        batch = [ToolCall(id="0", name="a"), ToolCall(id="0", name="b"), ToolCall(id="0", name="c")]
        used: set = set()
        self.assertEqual(loopcore.unique_call_ids(batch, used), 2)
        self.assertEqual([c.id for c in batch], ["0", "0_2", "0_3"])
        self.assertEqual(used, {"0", "0_2", "0_3"})

    def test_an_id_the_conversation_already_holds_is_renamed(self):
        used = {"call_1"}
        batch = [ToolCall(id="call_1", name="a"), ToolCall(id="call_2", name="b")]
        self.assertEqual(loopcore.unique_call_ids(batch, used), 1)
        self.assertEqual([c.id for c in batch], ["call_1_2", "call_2"])

    def test_the_new_id_never_collides_with_an_existing_one(self):
        used = {"x", "x_2"}
        batch = [ToolCall(id="x", name="a")]
        loopcore.unique_call_ids(batch, used)
        self.assertEqual(batch[0].id, "x_3")

    def test_an_empty_id_gets_a_fresh_one(self):
        batch = [ToolCall(id="", name="a"), ToolCall(id="", name="b")]
        self.assertEqual(loopcore.unique_call_ids(batch, set()), 2)
        self.assertEqual(len({c.id for c in batch}), 2)
        self.assertTrue(all(c.id.startswith("call_") for c in batch))

    def test_unique_ids_are_left_alone_and_the_call_is_idempotent(self):
        batch = [ToolCall(id="a", name="a"), ToolCall(id="b", name="b")]
        used: set = set()
        self.assertEqual(loopcore.unique_call_ids(batch, used), 0)
        self.assertEqual(loopcore.unique_call_ids(batch, set()), 0)
        self.assertEqual([c.id for c in batch], ["a", "b"])

    def test_provider_blocks_follow_the_rename_in_order(self):
        batch = [ToolCall(id="t", name="a"), ToolCall(id="t", name="b")]
        blocks = [{"type": "thinking", "thinking": "x", "signature": "s"},
                  {"type": "tool_use", "id": "t", "name": "a", "input": {}},
                  {"type": "tool_use", "id": "t", "name": "b", "input": {}},
                  {"type": "tool_use", "id": "other", "name": "c", "input": {}}]
        loopcore.unique_call_ids(batch, set(), blocks)
        self.assertEqual([b.get("id") for b in blocks if b["type"] == "tool_use"], ["t", "t_2", "other"])
        self.assertEqual(blocks[0]["type"], "thinking")


class DuplicateIdsInTheLoopTest(LoopCase):
    def reply(self, *ids):
        return Round([("stop_reason", "tool_use"), ("tool_calls", [ToolCall(id=i, name="create_object",
                                                                         arguments={"n": k}) for k, i in enumerate(ids)])])

    def test_every_call_of_the_conversation_has_its_own_id_and_its_own_result(self):
        script = [self.reply("0", "0"), self.reply("0", "1", "1"), say("Done.")]
        self.run_turn("make a plate 10x10x2 mm", script)
        ids = [c.id for m in self.history for c in m.tool_calls]
        self.assertEqual(len(ids), 5)
        self.assertEqual(len(set(ids)), 5)
        answered = [r.tool_call_id for m in self.history for r in m.tool_results]
        self.assertEqual(sorted(answered), sorted(ids))
        for index, message in enumerate(self.history):
            if message.tool_calls:
                following = self.history[index + 1]
                self.assertEqual([r.tool_call_id for r in following.tool_results], [c.id for c in message.tool_calls])

    def test_the_provider_sees_the_unique_ids_not_the_duplicates(self):
        self.run_turn("make a plate 10x10x2 mm", [self.reply("7", "7"), say("Done.")])
        sent = [c.id for m in self.provider.requests[-1].messages for c in m.tool_calls]
        self.assertEqual(sent, ["7", "7_2"])

    def test_ids_of_earlier_turns_count_too(self):
        history = [Message(role="user", content="make a plate"),
                   Message(role="assistant", content="", tool_calls=[ToolCall(id="c", name="create_object")]),
                   Message(role="user", content="", tool_results=[ToolResult(tool_call_id="c", name="create_object", content="ok")]),
                   Message(role="assistant", content="Built it."),
                   Message(role="user", content="make another plate 5 mm thick")]
        self.run_turn(history, [self.reply("c"), say("Done.")])
        ids = [c.id for m in self.history for c in m.tool_calls]
        self.assertEqual(ids, ["c", "c_2"])

    def test_a_renamed_call_is_logged(self):
        with mock.patch.object(agent, "log_event") as logged:
            self.run_turn("make a plate 10x10x2 mm", [self.reply("z", "z"), say("Done.")])
        self.assertTrue(any(c.args[0] == "call_ids_renamed" and c.kwargs["count"] == 1 for c in logged.call_args_list))


class GoAheadTest(unittest.TestCase):
    def test_the_go_ahead_table(self):
        for text in ("continue", "Continue", "yes", "ok", "go on", "do it", "keep going", "sigue", "weiter", "继续"):
            self.assertTrue(flow_intent.is_goahead(text), text)
        for text in ("", "   ", "continue?", "yes?", "make it 5 mm", "continue the shell with a thicker wall and a lid",
                     "what is the wall thickness", "thanks"):
            self.assertFalse(flow_intent.is_goahead(text), text)

    def test_is_continuation_is_unchanged_by_the_refactor(self):
        open_plan = Ledger.from_request("make a plate")
        self.assertTrue(flow_intent.is_continuation("continue", open_plan))
        self.assertTrue(flow_intent.is_continuation(open_plan.resume_prompt(), open_plan))
        self.assertFalse(flow_intent.is_continuation("continue?", open_plan))
        self.assertFalse(flow_intent.is_continuation("thanks", open_plan))
        self.assertFalse(flow_intent.is_continuation("continue", None))


class PinnedRequestTest(LoopCase):
    """"continue" after a FINISHED build must not become the request of the pinned task memory."""

    @staticmethod
    def finished(request):
        ledger = Ledger.from_request(request)
        for step in ledger.steps:
            step.status = "done"
        ledger._bump()
        return ledger

    def memory_request(self, typed, carried, **kw):
        worker = self.build(typed, [], ledger=carried, **kw)
        worker._begin_turn()
        self.addCleanup(worker._teardown, "answered", None)
        return worker._memory_inputs().request

    def test_a_bare_go_ahead_keeps_the_task_request(self):
        carried = self.finished("make a plate 10x10x2 mm")
        for word in ("continue", "yes", "go on"):
            self.assertEqual(self.memory_request(word, carried), "make a plate 10x10x2 mm", word)

    def test_a_question_or_a_new_request_is_its_own_request(self):
        carried = self.finished("make a plate 10x10x2 mm")
        self.assertEqual(self.memory_request("what is the wall thickness?", carried), "what is the wall thickness?")
        self.assertEqual(self.memory_request("make a gear with 20 teeth", carried), "make a gear with 20 teeth")

    def test_an_open_plan_resumes_with_its_own_request(self):
        carried = Ledger.from_request("make a plate 10x10x2 mm")
        self.assertEqual(self.memory_request("continue", carried), "make a plate 10x10x2 mm")

    def test_without_a_carried_plan_the_word_stays_as_typed(self):
        self.assertEqual(self.memory_request("continue", None), "continue")


class EarlierTurnsAreNotExcusedTest(LoopCase):
    """The turn-start baseline excuses what the USER left broken, never what the model itself built in an earlier turn."""

    def stub_project(self, steps):
        project = StubProject()
        project.read_steps = lambda include_superseded=False: list(steps)
        return project

    @staticmethod
    def step(doc, added=(), changed=(), superseded=False):
        return types.SimpleNamespace(doc=doc, delta={"added": list(added), "removed": [], "changed": [{"name": n} for n in changed]},
                                     superseded=superseded)

    def begin(self, project, snapshots):
        worker = self.build("make a gear with 20 teeth", [], project=project)
        with mock.patch("oscar.cad.health.snapshot_all_via_dispatch", lambda *a, **k: snapshots):
            worker._begin_turn()
        self.addCleanup(worker._teardown, "answered", None)
        return worker

    def test_objects_the_model_built_earlier_are_seeded_as_touched(self):
        project = self.stub_project([self.step("D", added=["Shell", "Lid"], changed=["Shell"]), self.step("Other", added=["X"])])
        snaps = {"D": types.SimpleNamespace(finding_keys=[("FRAGMENTS", "Lid"), ("INVALID", "UserBox")])}
        worker = self.begin(project, snaps)
        self.assertEqual(worker._state["touched"]["D"], {"Shell", "Lid"})
        self.assertEqual(worker._state["touched"]["Other"], {"X"})
        baseline = {tuple(k) for k in worker._state["baseline"]["D"]}
        self.assertEqual(baseline, {("FRAGMENTS", "Lid"), ("INVALID", "UserBox")})            # the baseline itself is unchanged

    def test_the_gate_counts_the_models_own_earlier_error_and_still_excuses_the_users(self):
        from oscar.cad.health import Finding
        from oscar.tools_verify import split_findings

        project = self.stub_project([self.step("D", added=["Lid"])])
        snaps = {"D": types.SimpleNamespace(finding_keys=[("FRAGMENTS", "Lid"), ("INVALID", "UserBox")])}
        worker = self.begin(project, snaps)
        report = types.SimpleNamespace(errors=[Finding("error", "FRAGMENTS", "Lid", "3 solids"),
                                               Finding("error", "INVALID", "UserBox", "bad shape")])
        baseline = {tuple(k) for k in worker._state["baseline"]["D"]}
        counted, excused = split_findings(report, baseline, worker._state["touched"]["D"])
        self.assertEqual([f.object for f in counted], ["Lid"])
        self.assertEqual([f.object for f in excused], ["UserBox"])

    def test_a_rewound_step_does_not_count(self):
        project = self.stub_project([self.step("D", added=["Gone"], superseded=True)])
        worker = self.begin(project, {})
        self.assertNotIn("D", worker._state.get("touched", {}))

    def test_a_project_without_steps_or_a_broken_reader_changes_nothing(self):
        worker = self.begin(self.stub_project([]), {"D": types.SimpleNamespace(finding_keys=[])})
        self.assertEqual(worker._state.get("touched", {}), {})
        broken = StubProject()
        broken.read_steps = mock.Mock(side_effect=OSError("disk"))
        worker = self.begin(broken, {})
        self.assertEqual(worker._state["baseline"], {})


class EarlierTurnErrorsEndToEndTest(RealFlowCase):
    """Real tools, real Project, real health: the model's own leftover from an earlier turn blocks ``validated``."""

    def two_turns(self, first_turn_object_is_users):
        doc = self.open_doc("ET") if first_turn_object_is_users else self.doc_name("ET")
        project = self.new_project("earlier turn", chat_id=f"20261002-101010-{uuid.uuid4().hex[:6]}")
        history = [Message(role="user", content="make a plate 10x10x2 mm")]
        broken = call("execute_code", include_screenshot=False,
                      code=self.code(doc, "d.addObject('Part::Cut', 'MyCut')\nd.recompute()"))
        if first_turn_object_is_users:
            import FreeCAD as App

            App.getDocument(doc).addObject("Part::Cut", "MyCut")
            App.getDocument(doc).recompute()
            first = [calls(self.box(doc, "Plate")), say("Made the plate.")]
        else:
            first = [calls(call("create_document", name=doc), broken)] + [say("Done.")] * 4
        self.turn("", first, project=project, history=history)
        history.append(Message(role="user", content="make a spacer 5x5x5 mm"))
        second = [calls(self.box(doc, "Spacer")), say("Made the spacer.")] + [say("Made the spacer.")] * 4
        result = self.make_worker("", second, project=project, history=history).run_turn()
        return result

    def test_an_error_the_model_left_in_an_earlier_turn_still_blocks_validated(self):
        result = self.two_turns(first_turn_object_is_users=False)
        self.assertNotEqual(result.outcome, "validated")
        self.assertIn("MyCut", json.dumps(result.validation["gate_errors"]))

    def test_the_same_object_left_by_the_user_is_still_excused(self):
        result = self.two_turns(first_turn_object_is_users=True)
        self.assertEqual(result.outcome, "validated")
        self.assertEqual(result.validation["gate_errors"], [])


class AnswerToAnOpenQuestionTest(LoopCase):
    """The reply to a ``needs_input`` turn ("please attach the STEP file" ...) resumes the plan instead of replacing it."""

    ASKED = PI_PROMPT

    def waiting_ledger(self):
        ledger = Ledger.from_request(self.ASKED)
        ledger.plan_update("S2", "blocked", "please attach the STEP file")
        return ledger

    def test_the_table(self):
        waiting = self.waiting_ledger()
        for reply in ("I attached the STEP file", "ABS", "use ABS and 2 mm walls", "skip it", "Raspberry Pi 4 Model B",
                      "yes use the nominal data", "here you go, the file is attached"):
            self.assertTrue(flow_intent.is_continuation(reply, waiting), reply)
        for reply in ("thanks", "Thanks a lot!", "what is a snap fit?", "make a gear with 20 teeth", "add a fan hole on the side",
                      "export it as STEP", "x " * 60, ""):
            self.assertFalse(flow_intent.is_continuation(reply, waiting), reply)

    def test_an_attachment_alone_answers(self):
        waiting = self.waiting_ledger()
        self.assertFalse(flow_intent.is_continuation("", waiting))
        self.assertTrue(flow_intent.is_continuation("", waiting, attachments=["/tmp/pi4.step"]))
        self.assertTrue(flow_intent.is_continuation("this one", waiting, attachments=["/tmp/pi4.step"]))
        self.assertFalse(flow_intent.is_continuation("make a gear", waiting, attachments=["/tmp/pi4.step"]))

    def test_without_a_question_the_old_rules_hold(self):
        open_plan = Ledger.from_request(self.ASKED)                       # nothing blocked by the model
        for reply in ("I attached the STEP file", "ABS", "skip it", "use ABS and 2 mm walls"):
            self.assertFalse(flow_intent.is_continuation(reply, open_plan), reply)
        finished = self.waiting_ledger()
        for step in finished.steps:
            step.status = "done"
        finished._bump()
        self.assertFalse(flow_intent.is_continuation("I attached the STEP file", finished))
        self.assertFalse(flow_intent.answers_open_question("ABS", None))

    def test_the_loop_resumes_the_enclosure_plan_after_needs_input(self):
        first = self.run_turn(self.ASKED, [calls(call("plan_update", id="S2", status="blocked", note="please attach the STEP file")),
                                           say("Please attach the STEP file of your board.")])
        self.assertEqual((first.ok, first.outcome), (True, "needs_input"))
        self.history.append(Message(role="user", content="I attached the STEP file"))
        worker = self.build(self.history, [calls(call("create_object"))] + [say("Continuing.")] * 6, ledger=first.ledger,
                            attachments=["/tmp/pi4.step"], validator=False)
        worker.run_turn()
        self.assertEqual(worker.ledger.intent, "enclosure")
        self.assertEqual(worker.ledger.request, self.ASKED)                  # the plan was resumed, not replaced
        self.assertEqual(worker.ledger.step("S2").status, "blocked")
        plan = [p for p in self.rec.json_of("plan_changed")][0]
        self.assertEqual(plan["intent"], "enclosure")
        self.assertIn("PLAN (intent enclosure", self.provider.requests[0].messages[-1].content)

    def test_a_new_order_after_needs_input_still_replaces_the_plan(self):
        first = self.run_turn(self.ASKED, [calls(call("plan_update", id="S2", status="blocked", note="please attach the STEP file")),
                                           say("Please attach the STEP file of your board.")])
        self.history.append(Message(role="user", content="forget that, make a gear with 20 teeth"))
        worker = self.build(self.history, [calls(call("create_object")), say("Done.")], ledger=first.ledger, validator=False)
        worker.run_turn()
        self.assertEqual(worker.ledger.intent, "generic_part")


class CompatNamesTest(unittest.TestCase):
    """the names existing tests, the panel and the smoke test import must keep working."""

    AGENT = ("AgentWorker MAX_TOOL_ROUNDS MAX_CUTOFF_RETRIES MAX_HISTORY_MESSAGES KEEP_HISTORY_MESSAGES KEEP_SCREENSHOTS "
             "CONTEXT_CHAR_BUDGET TOOL_RESULT_KEEP_CHARS PERSONA STRATEGY build_system_prompt workspace_snapshot "
             "compact_history stub_oversized_history _message_size needs_execution_nudge _EXECUTION_NOTE _CUTOFF_NOTE "
             "_BUDGET_NOTE _BAD_ARGUMENTS _TRIMMED_NOTE _tools _summarize_arguments _keep_last_screenshot _cut_off").split()
    PROVIDERS = ("Message ToolCall ToolResult ToolSpec StopRequested BaseProvider AnthropicProvider OpenAICompatProvider "
                 "make_provider list_models MAX_WIRE_IMAGES _looks_like_image_rejection "
                 "_anthropic_accepts_sampling _LEARNED").split()
    WORKER = "stop pause resume _wait_if_paused _should_stop _run_turn _turn_request _finish _run_tools".split()
    PROVIDER_METHODS = "_adapt_to_rejection _parse_stream _body _to_wire _headers _endpoint".split()

    def test_agent_names(self):
        self.assertEqual([n for n in self.AGENT if not hasattr(agent, n)], [])
        self.assertEqual([n for n in self.WORKER if not hasattr(agent.AgentWorker, n)], [])
        self.assertEqual((agent.MAX_TOOL_ROUNDS, agent.MAX_CUTOFF_RETRIES), (30, 2))
        self.assertIs(agent._tools, tools)

    def test_pinned_prompt_literals_and_notes(self):
        for literal in ("EXECUTION CONTRACT", "never end the turn at", "HOUSING/ENCLOSURE", "download_file"):
            self.assertIn(literal, agent.STRATEGY)
        self.assertIn("step by step", agent._EXECUTION_NOTE)
        self.assertIn("tool", agent._EXECUTION_NOTE)
        self.assertIn("Tool budget", agent._BUDGET_NOTE)
        self.assertIn("not valid JSON", agent._BAD_ARGUMENTS)

    def test_provider_names(self):
        self.assertEqual([n for n in self.PROVIDERS if not hasattr(providers, n)], [])
        for cls in (providers.AnthropicProvider, providers.OpenAICompatProvider):
            self.assertEqual([n for n in self.PROVIDER_METHODS if not hasattr(cls, n)], [], cls.__name__)

    def test_tools_names(self):
        for name in ("get_tools", "specs", "run_tool", "ToolError", "reset_turn_circuits", "_HANDLERS", "VIEW_NAMES"):
            self.assertTrue(hasattr(tools, name), name)

    def test_the_legacy_helpers_still_behave(self):
        self.assertTrue(agent.needs_execution_nudge("build a housing", 0))
        self.assertFalse(agent.needs_execution_nudge("build a housing", 2))
        self.assertTrue(agent._cut_off("length"))
        self.assertTrue(agent._cut_off("max_tokens"))
        self.assertFalse(agent._cut_off("stop"))
        self.assertEqual(agent._summarize_arguments("get_objects", {"doc_name": "A"}), "doc_name=A")


# ---------------------------------------------------------------------------------------------------------------------
# The connection test


class _Provider:
    """``chat`` replays ``behaviour(provider, tools) -> events``; records the tool mode of every call."""

    name = "fake"

    def __init__(self, prof, behaviour, log):
        self.profile, self.behaviour, self.log = prof, behaviour, log
        self.vision_enabled = True

    def chat(self, system, messages, tools, should_stop=None, **opts):
        self.log.append((self.profile.caps_override or {}).get("tools", "auto") if tools else "plain")
        yield from self.behaviour(self, tools)


def _echo():
    return ("tool_calls", [ToolCall(id="c1", name="echo", arguments={"text": "oscar-ok"})])


class ConnectionTestTransientErrorTest(unittest.TestCase):
    """A transient failure of the native tool test must never demote the model to text tool mode."""

    def setUp(self):
        self.log: list = []
        self.profile = params.Profile(name="chk", provider="openai_compat", base_url="http://127.0.0.1:9/v1",
                                      model=f"chk-{uuid.uuid4().hex[:8]}", api_key="k")
        key = _models.caps_key(self.profile)
        self.addCleanup(lambda: (_models._LEARNED.pop(key, None), params.set_model_caps(key, {})))

    def run_check(self, native_error):
        def behaviour(provider, tools):
            if not tools:
                yield ("text", "ready")
                yield ("done", "stop")
                return
            mode = (provider.profile.caps_override or {}).get("tools")
            if mode == "text":
                yield _echo()                                  # the text protocol would work
            else:
                yield ("error", native_error)

        return provider_check.run_connection_test(self.profile, make=lambda p: _Provider(p, behaviour, self.log), probe=False)

    def test_a_rate_limit_is_not_evidence_against_native_tools(self):
        report = self.run_check(ProviderError("HTTP 429 rate limited: slow down", status=429, retriable=True, kind="http"))
        self.assertFalse(report.ok)
        self.assertEqual(report.tools, "")
        self.assertNotIn("tools", _models.learned_fields(self.profile))
        self.assertNotIn("text", self.log)                                  # the text mode was not even tried
        line = [l for l in report.lines if l.step == "tools"][0]
        self.assertIsNone(line.ok)
        self.assertIn("try again", line.text)

    def test_a_connection_that_died_after_every_retry_is_transient_too(self):
        report = self.run_check(RuntimeError("request failed after 5 attempts: connection reset"))
        self.assertEqual(report.tools, "")
        self.assertNotIn("tools", _models.learned_fields(self.profile))

    def test_a_real_refusal_still_falls_back_to_text_mode(self):
        report = self.run_check(ProviderError("HTTP 400 bad request: tools are not supported", status=400, kind="http"))
        self.assertTrue(report.ok)
        self.assertEqual(report.tools, "text")
        self.assertEqual(_models.learned_fields(self.profile)["tools"], "text")

    def test_the_transient_rule_is_exactly_this(self):
        transient = provider_check._transient
        self.assertTrue(transient(ProviderError("x", retriable=True)))
        self.assertTrue(transient(RuntimeError("request failed after 3 attempts: boom")))
        self.assertTrue(transient(RuntimeError("stream ended early (no terminal marker)")))
        self.assertFalse(transient(ProviderError("HTTP 401 unauthorized", status=401)))
        self.assertFalse(transient(RuntimeError("HTTP 400 bad request")))


# ---------------------------------------------------------------------------------------------------------------------
# The derived request view (compaction) fuzz


def view_problems(view, window):
    """Everything wrong with ``view`` as a provider request, in absolute terms (no baseline)."""
    problems = list(context.alternation_problems(view))
    ids: list[str] = []
    for message in view:
        if message.role == "assistant":
            ids.extend(c.id for c in message.tool_calls)
    if len(ids) != len(set(ids)):
        problems.append("duplicate tool call ids in the view")
    if view and (view[0].role != "user" or view[0].tool_results):
        problems.append("the view does not start with a plain user message")
    if view and view[-1].role != "user":
        problems.append("the view does not end with a user message")
    return problems


def random_history(rng):
    history = []
    serial = 0
    for turn in range(rng.randint(1, 4)):
        history.append(Message(role="user", content=f"request {turn} " + "x" * rng.randint(0, 800)))
        for _ in range(rng.randint(0, 8)):
            batch = []
            for _k in range(rng.randint(1, 3)):
                serial += 1
                code = "line\n" * rng.randint(1, 200)
                batch.append(ToolCall(id=f"c{serial}", name=rng.choice(["execute_code", "get_objects", "create_object"]),
                                      arguments={"code": code} if rng.random() < .6 else {"a": 1}))
            blocks = []
            if rng.random() < .5:
                blocks.append({"type": "thinking", "thinking": "t" * rng.randint(0, 500), "signature": "sig"})
            if rng.random() < .3:
                blocks.append({"type": "text", "text": "hello"})
            blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in batch]
            history.append(Message(role="assistant", content="" if rng.random() < .5 else "text", tool_calls=batch,
                                   provider_blocks=blocks))
            results = [ToolResult(tool_call_id=c.id, name=c.name, content="r" * rng.randint(0, 3000),
                                  image_b64="AAAA" if rng.random() < .2 else "") for c in batch]
            history.append(Message(role="user", content="[governor] round\nline" if rng.random() < .5 else "",
                                   tool_results=results))
            if rng.random() < .15:
                history.append(Message(role="assistant", content="(placeholder)"))
                history.append(Message(role="user", content="[note]", synthetic=True, tag="gate"))
        if rng.random() < .8:
            history.append(Message(role="assistant", content=f"final answer {turn}"))
    if history[-1].role == "assistant":
        history.append(Message(role="user", content="follow up request"))
    return history


class RequestViewFuzzTest(unittest.TestCase):
    MEMORY = "<task_memory>\nREQUEST (verbatim, turn 1):\nrequest 0\n</task_memory>"

    def test_every_view_is_a_valid_request_whatever_the_history_and_the_window(self):
        cut = 0
        for seed in range(500):
            rng = random.Random(seed)
            history = random_history(rng)
            window = rng.choice([4096, 6000, 8192, 16384, 32768])
            caps = ModelCaps(context_window=window, max_output=1024)
            before = [(m.role, m.content, len(m.tool_calls), len(m.tool_results)) for m in history]
            view, stats = context.build_request_view(history, memory=self.MEMORY, caps=caps, system="s" * 500, tools=[])
            self.assertEqual(view_problems(view, window), [], f"seed {seed} window {window} {stats}")
            cut += bool(stats.get("dropped_rounds"))
            # the stored history is never touched
            self.assertEqual(before, [(m.role, m.content, len(m.tool_calls), len(m.tool_results)) for m in history], seed)
        self.assertGreater(cut, 50)      # the fuzz really cuts rounds

    def test_the_open_loop_keeps_its_thinking_blocks(self):
        """Compaction may drop the thinking of earlier turns, never that of the tool loop the model is in."""
        compared = 0
        for seed in range(200):
            rng = random.Random(10_000 + seed)
            history = random_history(rng)
            caps = ModelCaps(context_window=rng.choice([4096, 8192]), max_output=1024)
            view, _stats = context.build_request_view(history, memory=self.MEMORY, caps=caps, system="s" * 500, tools=[])
            last_user = max(i for i, m in enumerate(history) if m.role == "user" and not m.tool_results and not m.synthetic)
            open_loop = {tuple(c.id for c in m.tool_calls): m for m in history[last_user:] if m.role == "assistant" and m.tool_calls}

            def thinks(message):
                return bool(message.provider_blocks and message.provider_blocks[0].get("type") == "thinking")

            for message in view:
                original = open_loop.get(tuple(c.id for c in message.tool_calls)) if message.tool_calls else None
                if original is not None:
                    compared += 1
                    self.assertEqual(thinks(original), thinks(message), f"seed {seed}")
        self.assertGreater(compared, 30)

    def test_the_newest_governor_line_survives_and_older_ones_do_not(self):
        history = [Message(role="user", content="make a plate")]
        for n in range(4):
            call_ = ToolCall(id=f"c{n}", name="get_objects")
            history.append(Message(role="assistant", content="", tool_calls=[call_]))
            history.append(Message(role="user", content=f"[governor] round {n}/30", tool_results=[
                ToolResult(tool_call_id=f"c{n}", name="get_objects", content="ok")]))
        view, _stats = context.build_request_view(history, memory="", caps=ModelCaps(), system="", tools=[])
        carriers = [m.content for m in view if "[governor]" in (m.content or "")]
        self.assertEqual(carriers, ["[governor] round 3/30"])


# ---------------------------------------------------------------------------------------------------------------------
# Chaos: random weak-model behaviour through the real loop

PROMPTS = (PI_PROMPT, "make a plate 10x10x2 mm", "what is a pocket?", "add vents to the lid", "export it as STEP",
           "Design a bracket for a NEMA17 motor with 4 mounting holes", "continue", "fillet the top edges by 2 mm")
FOLLOW_UPS = ("continue", "add a fan hole on the side", "thanks", "make the walls thicker", "what is the wall thickness?")
CHAOS_TOOLS = ("create_document", "create_object", "execute_code", "get_objects", "get_view", "plan_update", "plan_set",
               "enclosure_spec", "build_enclosure", "set_parameters", "validate_model", "export_model", "bogus_tool",
               "source_reference_part")


def chaos_round(rng, nth):
    r = rng.random()
    if r < .10:
        return say("I will plan this: first X then Y." * rng.randint(1, 3))
    if r < .14:
        return empty()
    if r < .20:
        return bad_json("execute_code")
    if r < .24:
        return malformed_json("execute_code")
    if r < .27:
        return nameless()
    if r < .30:
        return think(rng.randint(10, 9000))
    if r < .32:
        return provider_error("HTTP 500")
    batch = []
    for k in range(rng.randint(1, 4)):
        name = rng.choice(CHAOS_TOOLS)
        args = {}
        if name == "plan_update":
            args = {"id": rng.choice(["S1", "S3", "S5", "R1", "S99"]), "status": rng.choice(["done", "skipped", "blocked", "running"]),
                    "note": rng.choice(["", "because"])}
        elif name == "build_enclosure":
            args = {"stage": rng.choice(["shell", "lid", "next"])}
        elif name == "execute_code":
            args = {"code": "x=1\n" * rng.randint(1, 50)}
        cid = rng.choice([f"c{nth}_{k}", "dup", "dup", ""]) if rng.random() < .15 else f"c{nth}_{k}"
        batch.append(ToolCall(id=cid or f"c{nth}_{k}", name=name, arguments=args))
    text = rng.choice(["", "", "ok let me do it"])
    return Round(([("text", text)] if text else []) + [("stop_reason", "tool_use"), ("tool_calls", batch)])


def history_problems(history):
    """What is wrong with the STORED conversation: a call without its result, a repeated id, two assistants in a row."""
    for index, message in enumerate(history):
        if message.role == "assistant" and message.tool_calls:
            nxt = history[index + 1] if index + 1 < len(history) else None
            ids = [c.id for c in message.tool_calls]
            if nxt is None or nxt.role != "user":
                return f"tool calls without a results message at {index}"
            if sorted(r.tool_call_id for r in nxt.tool_results) != sorted(ids):
                return f"results {[r.tool_call_id for r in nxt.tool_results]} do not answer {ids} at {index}"
            if len(set(ids)) != len(ids):
                return f"duplicate ids {ids} at {index}"
        if index and message.role == "assistant" and history[index - 1].role == "assistant":
            return f"two assistant messages in a row at {index}"
    all_ids = [c.id for m in history for c in m.tool_calls]
    if len(all_ids) != len(set(all_ids)):
        return "a tool call id repeats in the conversation"
    return ""


def signal_problems(rec, result):
    names = rec.names()
    if names.count("turn_finished") != 1:
        return f"turn_finished emitted {names.count('turn_finished')} times"
    if names.count("turn_outcome") != 1:
        return f"turn_outcome emitted {names.count('turn_outcome')} times"
    if names.index("turn_outcome") > names.index("turn_finished"):
        return "turn_finished before turn_outcome"
    if result is not None and result.outcome not in OUTCOMES:
        return f"unknown outcome {result.outcome}"
    return ""


class LoopChaosTest(unittest.TestCase):
    """Random scripts of a hostile model through the real loop: the conversation stays valid and the turn always ends."""

    def run_single(self, seed):
        rng = random.Random(seed)
        prompt = rng.choice(PROMPTS)
        script = [chaos_round(rng, i) for i in range(rng.randint(1, 14))]
        fake = FakeTools(extra={"enclosure_spec": {"param_names": list(flow.ENCLOSURE_REQUIRED_PARAMS), "spec_ok": True}},
                         fail={rng.choice(CHAOS_TOOLS): rng.randint(0, 3)})
        history = [Message(role="user", content=prompt)]
        provider = ScriptedProvider(script, strict=False)
        with mock.patch.object(agent._tools, "run_tool", lambda n, a: fake(n, a)):
            worker = agent.AgentWorker(history, provider=provider, profile=profile(), caps=ModelCaps(), config=AgentConfig(),
                                       project=StubProject() if rng.random() < .5 else None,
                                       validator=(lambda d, c: None) if rng.random() < .3 else False)
            rec = SignalRecorder(worker)
            result = worker.run_turn()
        return history, provider, rec, result

    def test_single_turns(self):
        for seed in range(200):
            history, provider, rec, result = self.run_single(seed)
            self.assertEqual(history_problems(history), "", f"seed {seed}")
            self.assertEqual(signal_problems(rec, result), "", f"seed {seed}")
            for request in provider.requests:
                self.assertEqual(context.alternation_problems(request.messages), [], f"seed {seed} request {request.n}")

    def test_multi_turn_chats_with_stop_steer_small_windows_and_the_kill_switch(self):
        for seed in range(120):
            rng = random.Random(10_000 + seed)
            caps = ModelCaps(context_window=rng.choice([4096, 6000, 8192, 32768, 128000]), max_output=1024)
            history: list[Message] = []
            carried = None
            fake = FakeTools(extra={"enclosure_spec": {"param_names": list(flow.ENCLOSURE_REQUIRED_PARAMS), "spec_ok": True}},
                             fail={rng.choice(CHAOS_TOOLS): rng.randint(0, 3)})
            stop_at = rng.randint(1, 12) if rng.random() < .25 else None
            counter = {"n": 0}
            holder: dict = {}

            def hook(name, args, counter=counter, holder=holder, rng=rng, stop_at=stop_at):
                counter["n"] += 1
                worker = holder.get("worker")
                if worker is None:
                    return
                if stop_at and counter["n"] == stop_at:
                    worker.stop()
                elif rng.random() < .1:
                    worker.steer("also make it 5 mm thick")

            fake.on_call = hook
            for turn in range(rng.randint(1, 3)):
                history.append(Message(role="user", content=rng.choice(PROMPTS) if turn == 0 else rng.choice(FOLLOW_UPS)))
                provider = ScriptedProvider([chaos_round(rng, i + 100 * turn) for i in range(rng.randint(1, 14))], strict=False)
                flow_on = rng.random() > .1
                with mock.patch.object(agent._tools, "run_tool", lambda n, a: fake(n, a)):
                    worker = agent.AgentWorker(history, provider=provider, profile=profile(), caps=caps, config=AgentConfig(),
                                               flow=flow_on, ledger=carried, project=StubProject() if rng.random() < .5 else None,
                                               validator=(lambda d, c: None) if rng.random() < .3 else False)
                    holder["worker"] = worker
                    rec = SignalRecorder(worker)
                    try:
                        result = worker.run_turn()
                    except StopRequested:
                        result = None
                if flow_on and worker.ledger.steps:
                    carried = worker.ledger.to_dict()
                    self.assertEqual(Ledger.from_dict(carried).to_dict()["steps"], carried["steps"], f"seed {seed}")
                self.assertEqual(history_problems(history), "", f"seed {seed} turn {turn}")
                if result is not None:
                    self.assertEqual(signal_problems(rec, result), "", f"seed {seed} turn {turn}")
                for request in provider.requests:
                    self.assertEqual(context.alternation_problems(request.messages), [],
                                     f"seed {seed} turn {turn} request {request.n}")
                if result is None:
                    break                   # a stopped turn ends the chat in this scenario


# ---------------------------------------------------------------------------------------------------------------------
# Chaos on the wire


def text_block(name, args, style):
    obj = {"name": name, "arguments": args}
    text = json.dumps(obj)
    if style == 0:
        return "<tool_call>\n" + text + "\n</tool_call>"
    if style == 1:
        return "```json\n" + text + "\n```"
    if style == 2:
        return text
    if style == 3:
        return "<tool_call>\n" + text.replace('"', "'") + "\n</tool_call>"                  # python dict
    if style == 4:
        return "<tool_call>\n" + text[:-1] + ",}\n</tool_call>"                             # trailing comma
    if style == 5:
        return "[TOOL_CALLS]" + json.dumps([obj])
    if style == 6:
        return "<tool_call>\n" + text[:max(10, len(text) // 2)]                             # cut off
    return "<tool_call>" + json.dumps({"name": name, "arguments": json.dumps(args)}) + "</tool_call>"   # double encoded


def text_round(rng):
    r = rng.random()
    if r < .12:
        return Round([("text", "Here is my plan: 1) shell 2) lid 3) vents. Let me know."), ("stop_reason", "stop"), ("done", "stop")])
    if r < .16:
        return Round([("text", ""), ("stop_reason", "stop"), ("done", "stop")])
    parts = []
    if rng.random() < .5:
        parts.append(rng.choice(["Next step.", "<think>hmm let me think</think>Now building.", "I'll do it."]))
    stop = "stop"
    for _ in range(rng.randint(1, 3)):
        name = rng.choice(["create_document", "create_object", "execute_code", "get_objects", "build_enclosure", "plan_update", "bogus"])
        args = {"code": "x=1\n" * rng.randint(1, 30)} if name == "execute_code" else {"stage": "shell"} if name == "build_enclosure" \
            else {"id": "S1", "status": "running"} if name == "plan_update" else {"doc_name": "A"}
        style = rng.choice([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
        parts.append(text_block(name, args, style))
        if style == 6:
            stop = "max_tokens"
            break
    return Round([("text", "\n".join(parts)), ("stop_reason", stop), ("done", stop)])


class TextModeChaosTest(WireCase):
    """Random replies of a model with no native tool calling, through the real provider over a socket."""

    def test_the_text_wire_stays_valid_and_the_turn_ends(self):
        for seed in range(60):
            rng = random.Random(500 + seed)
            script = [text_round(rng) for _ in range(rng.randint(1, 8))] + [say("Done building.")]
            server = ScriptedServer(script, dialect="openai", strict=False, repeat_last=True)
            self.addCleanup(server.shutdown)
            prof = params.Profile(name=f"ct{seed}", provider="openai_compat", base_url=server.base_url,
                                  model=f"ct-{uuid.uuid4().hex[:8]}", api_key="k", max_tokens=1024, caps_override={"tools": "text"})
            history = [Message(role="user", content=rng.choice([PI_PROMPT, "make a plate 10x10x2 mm", "what is a pocket?"]))]
            fake = FakeTools()
            with mock.patch.object(agent._tools, "run_tool", lambda n, a: fake(n, a)):
                worker = agent.AgentWorker(history, provider=providers.make_provider(prof), profile=prof, config=AgentConfig(),
                                           validator=False)
                rec = SignalRecorder(worker)
                result = worker.run_turn()
            self.assertEqual(signal_problems(rec, result), "", f"seed {seed}")
            self.assertEqual(history_problems(history), "", f"seed {seed}")
            for request in server.requests:
                body = request.json
                self.assertNotIn("tools", body, f"seed {seed}")
                self.assertFalse(any("tool_calls" in m or m["role"] == "tool" for m in body["messages"]), f"seed {seed}")
                roles = [m["role"] for m in body["messages"]][1:]
                self.assertTrue(all(a != b for a, b in zip(roles, roles[1:])), f"seed {seed}: roles {roles}")


def anthropic_round(rng, nth):
    think = "t" * rng.choice([20, 200, 7000, 13000])
    r = rng.random()
    if r < .15:
        return say("My plan: shell, lid, vents.", reasoning=think)
    if r < .2:
        return Round([("reasoning", "x" * 900), ("stop_reason", "max_tokens"), ("done", "max_tokens")])
    if r < .3:
        return say("", reasoning=think)
    batch = [call(rng.choice(["create_object", "execute_code", "get_objects", "plan_update", "bogus", "build_enclosure"]),
                  call_id=rng.choice([f"toolu_{nth}_{i}", f"toolu_{nth}_{i}", "toolu_dup"])) for i in range(rng.randint(1, 3))]
    return calls(*batch, text=rng.choice(["", "ok"]), reasoning=think)


def anthropic_problems(body):
    """What the Messages API would reject in ``body`` (the rules that matter for a tool loop with thinking)."""
    messages = body["messages"]
    seen: set = set()
    for index, message in enumerate(messages):
        content = message["content"]
        if message["role"] != "assistant" or isinstance(content, str):
            continue
        uses = [b for b in content if b.get("type") == "tool_use"]
        for use in uses:
            if use["id"] in seen:
                return f"tool_use id {use['id']} repeats"
            seen.add(use["id"])
        if uses:
            nxt = messages[index + 1] if index + 1 < len(messages) else None
            if nxt is None or nxt["role"] != "user" or isinstance(nxt["content"], str):
                return "tool_use without a tool_result message"
            answered = [b.get("tool_use_id") for b in nxt["content"] if b.get("type") == "tool_result"]
            if sorted(answered) != sorted(u["id"] for u in uses):
                return f"tool_results {answered} do not answer {[u['id'] for u in uses]}"
    last = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    if "thinking" in body and last >= 0 and isinstance(messages[last]["content"], list):
        blocks = messages[last]["content"]
        if any(b.get("type") == "tool_use" for b in blocks) and blocks[0].get("type") not in ("thinking", "redacted_thinking"):
            return "the final assistant tool_use turn has no leading thinking block while thinking is enabled"
    if "thinking" in body and (body.get("tool_choice") or {}).get("type") in ("any", "tool"):
        return "a forced tool_choice together with thinking"
    if messages[0]["role"] != "user":
        return "the first message is not a user message"
    return ""


class AnthropicChaosTest(WireCase):
    """Thinking blocks, nudges, cut-off replies and repeated ids through the real Anthropic provider."""

    def test_the_messages_api_rules_hold_for_every_request(self):
        for seed in range(50):
            rng = random.Random(900 + seed)
            script = [anthropic_round(rng, i) for i in range(rng.randint(2, 9))] + [say("Done.", reasoning="ok")]
            server = ScriptedServer(script, dialect="anthropic", strict=False, repeat_last=True)
            self.addCleanup(server.shutdown)
            prof = params.Profile(name=f"ca{seed}", provider="anthropic", base_url=server.base_url, model="claude-sonnet-5-5",
                                  api_key="k", max_tokens=16000)
            history = [Message(role="user", content=rng.choice([PI_PROMPT, "make a plate 10x10x2 mm"]))]
            fake = FakeTools()
            with mock.patch.object(agent._tools, "run_tool", lambda n, a: fake(n, a)):
                worker = agent.AgentWorker(history, provider=providers.make_provider(prof), profile=prof, config=AgentConfig(),
                                           validator=False)
                rec = SignalRecorder(worker)
                result = worker.run_turn()
            self.assertEqual(signal_problems(rec, result), "", f"seed {seed}")
            for request in server.requests:
                self.assertEqual(anthropic_problems(request.json), "", f"seed {seed} request {request.n if hasattr(request, 'n') else ''}")


if __name__ == "__main__":
    unittest.main()
