"""support.personas: every persona exists, has the shape its table row promises, and runs through the
scripted provider (and, for the transport personas, through a real provider over ScriptedServer).

These tests pin the personas themselves; the loop behaviour they provoke is asserted by the loop tests that use them.
"""

import json
import os
import sys
import threading
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import params, provider_chat, providers  # noqa: E402
from support import personas  # noqa: E402
from support.personas import PERSONAS, PROMPTS, Sequence, build, prompt_for  # noqa: E402
from support.scripted_llm import Round, ScriptedProvider  # noqa: E402
from support.server import HttpFault, ScriptedServer, StreamFault  # noqa: E402

Message = providers.Message
USER = [Message(role="user", content="go")]

TABLE_NAMES = {
    "plan_only", "tool_then_stop", "cutoff_then_plan", "stalls_then_acts", "giant_script_truncated", "bad_json_args",
    "text_tool_calls", "think_tags", "flaky_429_then_ok", "drops_mid_stream", "empty_reply", "tool_loop_repeat",
    "hidden_reasoning", "claims_done_without_work", "asks_the_user", "loops_on_reads", "plan_in_other_language",
    "slow_local", "golden_pi4"}


def pull(script, n):
    """The first ``n`` rounds a script serves (a list or a Sequence)."""
    if isinstance(script, list):
        return list(script[:n])
    return [script(None) for _ in range(n)]


def events(rnd, kind):
    return [p for k, p in rnd.events if k == kind]


def tool_calls_of(rnd):
    return [c for payload in events(rnd, "tool_calls") for c in payload]


def run_through_provider(script, requests_wanted):
    prov = ScriptedProvider(script, strict=False)
    out = []
    for _ in range(requests_wanted):
        out.append(list(prov.chat("s", USER, [])))
    return out


class RegistryTest(unittest.TestCase):
    def test_every_persona_of_the_contract_table_is_registered(self):
        self.assertEqual(TABLE_NAMES - set(PERSONAS), set())

    def test_the_five_review_personas_exist(self):
        for name in ("claims_done_without_work", "asks_the_user", "loops_on_reads", "plan_in_other_language", "slow_local"):
            self.assertIn(name, PERSONAS)

    def test_every_persona_builds_with_defaults_and_has_a_prompt(self):
        for name in PERSONAS:
            with self.subTest(persona=name):
                script = build(name)
                self.assertTrue(isinstance(script, (list, Sequence)))
                self.assertTrue(isinstance(prompt_for(name), str) and prompt_for(name).strip())

    def test_every_prompt_belongs_to_a_persona(self):
        self.assertEqual(set(PROMPTS) - set(PERSONAS), set())
        self.assertEqual(set(PERSONAS) - set(PROMPTS), set())

    def test_unknown_persona_names_the_known_ones(self):
        with self.assertRaisesRegex(KeyError, "plan_only"):
            build("no_such_persona")

    def test_the_name_is_the_function_name(self):
        for name, fn in PERSONAS.items():
            self.assertEqual(fn.__name__, name)

    def test_scripts_are_independent_between_calls(self):
        a, b = build("plan_only"), build("plan_only")
        a[0].events.append(("text", "mutated"))
        self.assertNotIn(("text", "mutated"), b[0].events)


class SequenceTest(unittest.TestCase):
    def rnd(self, text):
        return Round([("text", text), ("done", "stop")])

    def test_prefix_then_tail_forever(self):
        seq = Sequence([self.rnd("a"), self.rnd("b")], self.rnd("tail"))
        self.assertEqual([events(seq(None), "text")[0] for _ in range(5)], ["a", "b", "tail", "tail", "tail"])
        self.assertEqual(seq.served, 5)

    def test_tail_rounds_are_copies(self):
        seq = Sequence([], self.rnd("t"))
        first = seq(None)
        first.events.append(("text", "mutated"))
        self.assertEqual(events(seq(None), "text"), ["t"])

    def test_callable_tail_gets_its_own_index(self):
        seq = Sequence([self.rnd("p")], lambda i: self.rnd(f"t{i}"))
        self.assertEqual([events(seq(None), "text")[0] for _ in range(4)], ["p", "t0", "t1", "t2"])

    def test_no_tail_is_strict(self):
        seq = Sequence([self.rnd("only")])
        seq(None)
        with self.assertRaisesRegex(AssertionError, "unexpected extra provider request #2"):
            seq(None)

    def test_limit_stops_a_runaway_loop(self):
        seq = Sequence([], self.rnd("x"), limit=5, name="runaway")
        for _ in range(5):
            seq(None)
        with self.assertRaisesRegex(AssertionError, "runaway.*more than 5 rounds"):
            seq(None)

    def test_it_ignores_whatever_argument_the_caller_passes(self):
        seq = Sequence([self.rnd("a")], self.rnd("b"))
        self.assertEqual(events(seq({"messages": []}), "text"), ["a"])
        self.assertEqual(events(seq(object()), "text"), ["b"])

    def test_thread_safe_counting(self):
        seq = Sequence([], self.rnd("x"), limit=10_000)

        def spin():
            for _ in range(200):
                seq(None)

        threads = [threading.Thread(target=spin) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(seq.served, 1600)

    def test_it_is_a_valid_script_for_provider_and_server(self):
        seq = Sequence([self.rnd("a")], self.rnd("b"))
        prov = ScriptedProvider(seq)
        self.assertEqual([p for k, p in prov.chat("s", USER, []) if k == "text"], ["a"])
        self.assertEqual([p for k, p in prov.chat("s", USER, []) if k == "text"], ["b"])


class ShapeTest(unittest.TestCase):
    """One test per row of the persona table: the behaviour it promises is present in the rounds."""

    def test_plan_only_has_text_and_never_a_call(self):
        rounds = build("plan_only")
        self.assertEqual(len(rounds), 3)
        for r in rounds:
            self.assertEqual(tool_calls_of(r), [])
            self.assertIn("1.", events(r, "text")[0])
        self.assertEqual(len(build("plan_only", times=2)), 2)

    def test_tool_then_stop_acts_once_then_says_done_forever(self):
        first, second, third, fourth = pull(build("tool_then_stop"), 4)
        self.assertEqual([c.name for c in tool_calls_of(first)], ["create_document", "create_object"])
        for r in (second, third, fourth):
            self.assertEqual((tool_calls_of(r), events(r, "text")), ([], [personas.DONE_TEXT]))

    def test_cutoff_then_plan(self):
        first, plan, act, done = build("cutoff_then_plan")
        self.assertEqual((events(first, "stop_reason"), events(first, "tool_calls"), events(first, "text")),
                         (["length"], [], []))
        self.assertTrue(events(first, "reasoning"))
        self.assertEqual((tool_calls_of(plan), events(plan, "text")), ([], [personas.PLAN_TEXT]))
        self.assertTrue(tool_calls_of(act))
        self.assertEqual(tool_calls_of(done), [])

    def test_stalls_then_acts_floods_reasoning_then_acts(self):
        flood, act, done = build("stalls_then_acts", reasoning_chars=5000, chunk=500)
        chunks = events(flood, "reasoning")
        self.assertEqual((len(chunks), sum(map(len, chunks))), (10, 5000))
        self.assertEqual(events(flood, "stop_reason"), ["length"])
        self.assertTrue(tool_calls_of(act))
        self.assertEqual(sum(len(c) for c in events(build("stalls_then_acts")[0], "reasoning")), 20000)

    def test_giant_script_truncated_twice_then_small(self):
        rounds = build("giant_script_truncated")
        for r in rounds[:2]:
            (call,) = tool_calls_of(r)
            self.assertEqual(call.name, "execute_code")
            self.assertIn("truncated", call.args_error)
            self.assertTrue(call.raw_args.startswith('{"code":'))
            self.assertEqual(events(r, "stop_reason"), ["max_tokens"])
        self.assertNotEqual(tool_calls_of(rounds[0])[0].id, tool_calls_of(rounds[1])[0].id)
        (small,) = tool_calls_of(rounds[2])
        self.assertEqual((small.args_error, len(small.arguments["code"]) < 100), ("", True))

    def test_bad_json_args_is_malformed_not_truncated(self):
        (call,) = tool_calls_of(build("bad_json_args")[0])
        self.assertIn("malformed", call.args_error)
        self.assertNotIn("truncated", call.args_error)
        self.assertNotEqual(events(build("bad_json_args")[0], "stop_reason"), ["max_tokens"])

    def test_text_tool_calls_blocks_are_valid_when_joined_and_split_inside_the_tags(self):
        first, answer = build("text_tool_calls")
        pieces = events(first, "text")
        joined = "".join(pieces)
        self.assertGreater(len(pieces), 2)
        self.assertTrue(any("<tool_call>" not in p and "<tool_" in p for p in pieces) or any(
            p.count("<") != p.count(">") for p in pieces), "a tag should be split across chunks")
        blocks = joined.split("<tool_call>")[1:]
        payloads = [json.loads(b.split("</tool_call>")[0]) for b in blocks]
        self.assertEqual([p["name"] for p in payloads], ["create_document", "create_object"])
        self.assertEqual(tool_calls_of(answer), [])
        unsplit = build("text_tool_calls", split=False)[0]
        self.assertEqual(len(events(unsplit, "text")), 1)
        self.assertEqual(events(unsplit, "text")[0], joined)

    def test_think_tags_span_chunks(self):
        (r,) = build("think_tags")
        pieces = events(r, "text")
        self.assertGreaterEqual(len(pieces), 4)
        joined = "".join(pieces)
        self.assertTrue(joined.startswith("<think>") and "</think>" in joined)
        self.assertNotIn("<think>", pieces[0])             # the opening tag itself is split
        self.assertTrue(joined.endswith("The volume is 1000 cubic millimetres."))

    def test_transport_personas_are_server_items(self):
        flaky = build("flaky_429_then_ok")
        self.assertIsInstance(flaky[0], HttpFault)
        self.assertEqual((flaky[0].status, flaky[0].headers["Retry-After"]), (429, "1"))
        self.assertEqual(events(flaky[1], "text"), ["Recovered after the rate limit."])
        dropped = build("drops_mid_stream")
        self.assertIsInstance(dropped[0], StreamFault)
        self.assertEqual((dropped[0].mode, dropped[0].after), ("drop", 2))
        self.assertEqual("".join(events(dropped[1], "text")), "The housing has two parts: a shell and a lid.")
        self.assertTrue("".join(events(dropped[1], "text")).startswith("".join(events(dropped[0].round, "text"))))

    def test_empty_reply_is_empty_for_as_long_as_asked(self):
        seq = build("empty_reply", limit=4)
        for r in pull(seq, 4):
            self.assertEqual((events(r, "text"), tool_calls_of(r)), ([], []))
        with self.assertRaises(AssertionError):
            seq(None)

    def test_tool_loop_repeat_repeats_the_same_call_with_fresh_ids(self):
        rounds = pull(build("tool_loop_repeat"), 6)
        calls = [c for r in rounds[1:] for c in tool_calls_of(r)]
        self.assertEqual(len({(c.name, json.dumps(c.arguments, sort_keys=True)) for c in calls}), 1)
        self.assertEqual(len({c.id for c in calls}), len(calls))

    def test_hidden_reasoning_is_silence_and_ticks_only(self):
        first, second, third = build("hidden_reasoning", silent_s=120, step_s=15)
        self.assertEqual(sum(s for s in events(first, "advance")), 120)
        ticks = events(first, "tick")
        self.assertEqual(len(ticks), 8)
        self.assertEqual([t["idle_s"] for t in ticks], [15.0 * i for i in range(1, 9)])
        self.assertEqual({t["reasoning_chars"] for t in ticks}, {0})
        self.assertEqual(events(first, "reasoning") + events(first, "text"), [])
        self.assertTrue(tool_calls_of(first) and tool_calls_of(second))
        self.assertEqual(tool_calls_of(third), [])

    def test_claims_done_without_work_claims_the_remaining_steps_only(self):
        first, claims, answer, again = pull(build("claims_done_without_work"), 4)
        self.assertEqual([c.name for c in tool_calls_of(first)], ["create_document", "create_object"])
        updates = tool_calls_of(claims)
        self.assertEqual([(c.name, c.arguments["id"], c.arguments["status"]) for c in updates],
                         [("plan_update", f"S{i}", "done") for i in range(6, 11)])
        self.assertEqual((tool_calls_of(answer), tool_calls_of(again)), ([], []))
        custom = build("claims_done_without_work", step_ids=("S3",))
        self.assertEqual([c.arguments["id"] for c in tool_calls_of(pull(custom, 2)[1])], ["S3"])

    def test_asks_the_user_blocks_with_a_note_and_makes_no_geometry(self):
        block, ask = build("asks_the_user")
        (upd,) = tool_calls_of(block)
        self.assertEqual((upd.name, upd.arguments["id"], upd.arguments["status"]), ("plan_update", "S2", "blocked"))
        self.assertIn("attach", upd.arguments["note"])
        self.assertEqual(tool_calls_of(ask), [])
        self.assertTrue(events(ask, "text")[0].rstrip().endswith("?"))

    def test_loops_on_reads_only_reads(self):
        names = {c.name for r in pull(build("loops_on_reads"), 12)[1:] for c in tool_calls_of(r)}
        self.assertEqual(names, {"get_objects", "get_view"})

    def test_plan_in_other_language_acts_in_the_first_reply(self):
        first, final = build("plan_in_other_language")
        self.assertTrue(tool_calls_of(first))
        self.assertIn("Ich", events(first, "text")[0])
        self.assertIn("Fertig", events(final, "text")[0])

    def test_slow_local_is_one_long_silence_then_a_complete_call(self):
        slow_round, final = build("slow_local", seconds=200)
        self.assertEqual(events(slow_round, "advance"), [200.0])
        self.assertEqual([k for k, _ in slow_round.events if k not in ("advance",)], ["stop_reason", "tool_calls"])
        self.assertEqual(tool_calls_of(final), [])


class GoldenPi4Test(unittest.TestCase):
    TABLE = (["find_reference_part", "source_reference_part", "create_document", "import_reference_part",
              "measure_reference_part", "enclosure_spec"] + ["build_enclosure"] * 2 + ["get_view", "execute_code", "get_objects"]
             + ["build_enclosure"] * 6 + ["add_cutout", "set_parameter", "set_parameter", "validate_model", "check_fit",
                                          "export_model", "export_rebuild_script"])

    def names(self, script):
        return [c.name for r in script for c in tool_calls_of(r)]

    def test_the_script_follows_the_table_of_section_10_5(self):
        script = build("golden_pi4")
        self.assertEqual(len(script), 25)
        self.assertEqual(self.names(script), self.TABLE)
        self.assertEqual(tool_calls_of(script[-1]), [])
        self.assertIn("Pi 4", events(script[-1], "text")[0])

    def test_one_call_per_reply_and_unique_ids(self):
        script = build("golden_pi4")
        self.assertTrue(all(len(tool_calls_of(r)) == 1 for r in script[:-1]))
        ids = [c.id for r in script for c in tool_calls_of(r)]
        self.assertEqual(len(ids), len(set(ids)))

    def test_arguments_of_the_named_calls(self):
        by_round = {i: tool_calls_of(r)[0] for i, r in enumerate(build("golden_pi4")[:-1])}
        self.assertEqual(by_round[0].arguments, {"query": "raspberry pi 4 model b"})
        self.assertEqual(by_round[2].arguments, {"name": "PiCase"})
        self.assertEqual(by_round[4].arguments, {"doc_name": "PiCase", "write_params": True})
        self.assertEqual(by_round[5].arguments["spec"], {"name": "PiCase", "material": "ABS"})
        self.assertEqual(by_round[17].arguments, {"doc_name": "PiCase", "part": "lid", "face": "top", "u": 65,
                                                  "v": 45, "w": 6, "h": 6, "id": "status_led"})
        self.assertEqual((by_round[18].arguments, by_round[19].arguments),
                         ({"doc_name": "PiCase", "name": "wall_t", "value": "2.4 mm"},
                          {"doc_name": "PiCase", "name": "clearance", "value": 0.35}))
        self.assertEqual(by_round[22].arguments["formats"], ["step", "stl"])
        self.assertTrue(by_round[23].arguments["run_check"])

    def test_the_injected_failure_is_a_script_that_cannot_run(self):
        (call,) = tool_calls_of(build("golden_pi4")[9])
        self.assertEqual(call.name, "execute_code")
        with self.assertRaises(NameError):
            exec(compile(call.arguments["code"], "<golden>", "exec"), {})

    def test_nominal_variant_drops_the_import_and_measurement_and_builds_the_reference_stage(self):
        names = self.names(build("golden_pi4", nominal=True))
        self.assertEqual(len(build("golden_pi4", nominal=True)), 24)
        self.assertEqual(names.count("build_enclosure"), 9)             # the nine kit stages: nothing imported builds "reference"
        self.assertNotIn("import_reference_part", names)
        self.assertNotIn("measure_reference_part", names)
        self.assertIn("source_reference_part", names)

    def test_lid_lift_variant_sets_the_parameter_before_check_fit(self):
        script = build("golden_pi4", lid_lift=15)
        names = self.names(script)
        self.assertEqual(len(script), 26)
        lift = [(i, c) for i, c in enumerate(c for r in script for c in tool_calls_of(r))
                if c.arguments.get("name") == "lid_lift"]
        self.assertEqual(len(lift), 1)
        index, call = lift[0]
        self.assertEqual((call.name, call.arguments["value"]), ("set_parameter", 15))
        self.assertGreater(index, names.index("set_parameter") + 1)      # after wall_t and clearance
        self.assertLess(index, names.index("check_fit"))

    def test_the_prompt_is_the_one_of_the_contract(self):
        self.assertEqual(prompt_for("golden_pi4"),
                         "Build a housing for a Raspberry Pi 4. Bottom shell with cutouts for every I/O port, air vent slots, "
                         "snap-fit lid, ABS. And source the Pi 4 STEP file yourself.")


class ProviderRunTest(unittest.TestCase):
    """Every persona is servable by the scripted provider without surprises."""

    def test_all_list_personas_run_to_the_end_through_the_scripted_provider(self):
        for name in sorted(PERSONAS):
            script = build(name)
            if not isinstance(script, list) or any(not isinstance(r, Round) for r in script):
                continue                                             # server items and Sequences are covered elsewhere
            with self.subTest(persona=name):
                runs = run_through_provider(script, len(script))
                self.assertEqual(len(runs), len(script))
                for evs in runs:
                    kinds = [k for k, _ in evs]
                    self.assertNotIn("advance", kinds)                  # the pseudo-event is consumed, never yielded
                    self.assertIn(kinds[-1], ("done", "tool_calls"))

    def test_sequence_personas_serve_a_scripted_provider_past_their_prefix(self):
        for name in ("tool_then_stop", "claims_done_without_work", "loops_on_reads", "tool_loop_repeat", "empty_reply"):
            with self.subTest(persona=name):
                runs = run_through_provider(build(name), 8)
                self.assertEqual(len(runs), 8)

    def test_hidden_reasoning_and_slow_local_move_the_injected_clock(self):
        for name, expect in (("hidden_reasoning", 120.0), ("slow_local", 200.0)):
            with self.subTest(persona=name):
                now = [0.0]
                prov = ScriptedProvider(build(name), advance=lambda s: now.__setitem__(0, now[0] + s))
                list(prov.chat("s", USER, []))
                self.assertEqual(now[0], expect)


class PersonasOverTheWireTest(unittest.TestCase):
    """The transport personas against a real provider: they do what their docstring says."""

    def setUp(self):
        self.sleeps = []
        for patcher in (mock.patch.object(provider_chat.ChatLoop, "_sleep",
                                          lambda loop, delay, call, stop, retry: self.sleeps.append(delay) or iter(())),
                        mock.patch.object(provider_chat, "log_event", lambda *a, **k: None)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_persona(self, name):
        server = ScriptedServer(build(name))
        self.addCleanup(server.shutdown)
        prof = params.Profile(name="p", provider="openai_compat", base_url=server.base_url, model=f"m-{uuid.uuid4().hex[:6]}",
                              api_key="k", max_tokens=256)
        text, kinds = "", []
        for kind, payload in providers.OpenAICompatProvider(prof).chat("s", USER, []):
            kinds.append(kind)
            if kind == "text":
                text += payload
            elif kind == "reset":
                text = ""
        return text, kinds, server

    def test_flaky_429_then_ok(self):
        text, kinds, server = self.run_persona("flaky_429_then_ok")
        self.assertEqual(text, "Recovered after the rate limit.")
        self.assertEqual((server.request_count, self.sleeps), (2, [1.0]))

    def test_drops_mid_stream_has_no_duplicated_text_after_the_retry(self):
        text, kinds, server = self.run_persona("drops_mid_stream")
        self.assertEqual(text, "The housing has two parts: a shell and a lid.")
        self.assertIn("reset", kinds)
        self.assertEqual(server.request_count, 2)

    def test_truncated_script_persona_reaches_the_real_parser_as_truncated(self):
        server = ScriptedServer(build("giant_script_truncated"))
        self.addCleanup(server.shutdown)
        prof = params.Profile(name="p", provider="openai_compat", base_url=server.base_url, model=f"m-{uuid.uuid4().hex[:6]}",
                              api_key="k", max_tokens=256)
        calls = [p for k, p in providers.OpenAICompatProvider(prof).chat("s", USER, []) if k == "tool_calls"][0]
        self.assertIn("truncated", calls[0].args_error)


class PromptIntentTest(unittest.TestCase):
    """The prompts of the personas drive the loop's intent classifier the way the persona rows assume."""

    @classmethod
    def setUpClass(cls):
        try:
            from oscar import flow
        except ModuleNotFoundError as e:
            raise unittest.SkipTest(f"oscar.flow is not available: {e.name}") from e
        cls.flow = flow

    def test_enclosure_prompts(self):
        for name in ("tool_then_stop", "claims_done_without_work", "asks_the_user", "golden_pi4"):
            self.assertEqual(self.flow.classify_intent(prompt_for(name)), "enclosure", name)

    def test_generic_part_prompts(self):
        for name in ("plan_only", "cutoff_then_plan", "stalls_then_acts", "tool_loop_repeat", "loops_on_reads"):
            self.assertEqual(self.flow.classify_intent(prompt_for(name)), "generic_part", name)

    def test_the_non_english_prompt_is_other(self):
        self.assertEqual(self.flow.classify_intent(prompt_for("plan_in_other_language")), "other")


if __name__ == "__main__":
    unittest.main()
