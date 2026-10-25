"""oscar.context: token estimates, pinned task memory, the derived request view and the workspace snapshot."""

import dataclasses
import json
import math
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import context, flow, models, prompt  # noqa: E402
from oscar.context import MemoryInputs  # noqa: E402
from oscar.providers import Message, ToolCall, ToolResult, ToolSpec  # noqa: E402

BRIEF = ("Build a housing for a Raspberry Pi 4. Bottom shell with cutouts for every I/O port, air vent slots, "
         "snap-fit lid, ABS. And source the Pi 4 STEP file yourself.")


# -- builders ------------------------------------------------------------------------------------------------------

def user(text: str, **kw) -> Message:
    return Message(role="user", content=text, **kw)


def asst(i: int, text: str = "", *, code: str | None = None, name: str = "execute_code",
         blocks: list | None = None) -> Message:
    args = {"code": code} if code is not None else {"n": i}
    return Message(role="assistant", content=text, tool_calls=[ToolCall(f"c{i}", name, args)],
                   provider_blocks=list(blocks or []))


def results(i: int, text: str = "ok", *, image: str = "", governor: bool = True) -> Message:
    content = f"round {i} done"
    if governor:
        content += f'\n[governor] round {i}/30 · BUILD · steps {i}/12 done · now: S5 "Shell" · health OK'
    return Message(role="user", content=content,
                   tool_results=[ToolResult(f"c{i}", "execute_code", text, image_b64=image)])


def conversation(rounds: int, *, result_chars: int = 800, code: str | None = None, image: str = "") -> list[Message]:
    history = [user(BRIEF)]
    for i in range(1, rounds + 1):
        history.append(asst(i, code=code))
        history.append(results(i, ("r%d " % i) * (result_chars // 4), image=image))
    return history


def frozen(messages) -> str:
    return json.dumps([dataclasses.asdict(m) for m in messages], sort_keys=True, default=str)


def caps(window: int, max_output: int = 1000, **kw) -> models.ModelCaps:
    return models.ModelCaps(context_window=window, max_output=max_output, **kw)


def memory_text(**kw) -> str:
    inp = MemoryInputs(request=kw.pop("request", BRIEF), ledger=kw.pop("ledger", None), params=kw.pop("params", []),
                       objects=kw.pop("objects", []), reference=kw.pop("reference", None), files=kw.pop("files", []),
                       decisions=kw.pop("decisions", []), **kw)
    return context.pinned_memory(inp)


def spec(name: str, size: int = 400) -> ToolSpec:
    return ToolSpec(name, "d" * size, {"type": "object", "properties": {"a": {"type": "string", "description": "x" * size}}})


class ViewCase(unittest.TestCase):
    """Shared structural assertions for a request view."""

    def check_view(self, view):
        context.assert_alternating(view)
        self.assertEqual(view[0].role, "user")
        self.assertFalse(view[0].tool_results)
        for i, m in enumerate(view):
            if m.role == "assistant" and m.tool_calls:
                nxt = view[i + 1]
                self.assertEqual({c.id for c in m.tool_calls}, {r.tool_call_id for r in nxt.tool_results})


# -- estimates -----------------------------------------------------------------------------------------------------

class EstimateTest(unittest.TestCase):
    def test_text_rule_is_ceil_chars_over_3_5(self):
        self.assertEqual(context.estimate_tokens_text(""), 0)
        self.assertEqual(context.estimate_tokens_text(None), 0)
        for n, expected in ((1, 1), (7, 2), (8, 3), (35, 10), (36, 11), (3500, 1000)):
            self.assertEqual(context.estimate_tokens_text("a" * n), expected, n)
            self.assertEqual(context.estimate_tokens_text("a" * n), math.ceil(n / 3.5))

    def test_message_overhead_and_text(self):
        self.assertEqual(context.estimate_tokens([user("a" * 35)]), 8 + 10)
        self.assertEqual(context.estimate_tokens([]), 0)
        self.assertEqual(context.estimate_tokens([user("")]), 8)

    def test_tool_call_arguments_count_a_third_of_their_json(self):
        call = ToolCall("c1", "execute_code", {"code": "x" * 297})
        expected = 8 + len(json.dumps(call.arguments)) / 3.0
        self.assertEqual(context.estimate_tokens([Message(role="assistant", tool_calls=[call])]), math.ceil(expected))

    def test_images_are_flat_costed(self):
        self.assertEqual(context.estimate_tokens([user("", images=["AAAA", "BBBB"])]), 8 + 3000)
        shot = Message(role="user", tool_results=[ToolResult("c1", "get_view", "", image_b64="A" * 100000)])
        self.assertEqual(context.estimate_tokens([shot]), 8 + 1500)   # the size of the base64 string does not matter
        plain = Message(role="user", tool_results=[ToolResult("c1", "get_view", "", image_b64="")])
        self.assertEqual(context.estimate_tokens([shot]) - context.estimate_tokens([plain]), 1500)

    def test_system_and_tool_specs_are_included(self):
        base = context.estimate_tokens([user("hi")])
        with_system = context.estimate_tokens([user("hi")], "s" * 70)
        self.assertEqual(with_system - base, 20)
        tool = ToolSpec("t", "d" * 35, {"type": "object"})
        raw = (len(json.dumps(tool.parameters)) + 35) / 3.5
        self.assertEqual(context.estimate_tokens([], "", [tool]), math.ceil(raw))
        self.assertEqual(context.estimate_tokens([], "", [{"name": "t", "description": "d" * 35, "parameters": {"type": "object"}}]),
                         math.ceil(raw))

    def test_thinking_blocks_cost_their_text(self):
        plain = Message(role="assistant", content="ok")
        thinking = Message(role="assistant", content="ok", provider_blocks=[
            {"type": "thinking", "thinking": "t" * 70, "signature": "s"}, {"type": "text", "text": "ok"}])
        self.assertEqual(context.estimate_tokens([thinking]) - context.estimate_tokens([plain]), 20)

    def test_monotonic_in_messages(self):
        history = conversation(5)
        sizes = [context.estimate_tokens(history[:n]) for n in range(len(history) + 1)]
        self.assertEqual(sizes, sorted(sizes))

    def test_calibration_is_multiplicative(self):
        history = conversation(6)
        raw = context.estimate_tokens(history)
        self.assertEqual(context.estimate_tokens(history, usage_prompt_tokens=int(raw * 1.5)), int(raw * 1.5))
        # the usage figure covers only the first ten messages: its ratio scales the estimate of ALL of them
        prefix_raw = context.estimate_tokens(history[:10])
        scaled = context.estimate_tokens(history, usage_prompt_tokens=2 * prefix_raw, since_index=10)
        self.assertAlmostEqual(scaled, 2 * raw, delta=4)
        self.assertGreater(scaled, raw * 1.9)

    def test_calibration_factor_is_clipped(self):
        history = conversation(4)
        raw = context.estimate_tokens(history)
        self.assertAlmostEqual(context.estimate_tokens(history, usage_prompt_tokens=raw * 100), raw * 3.0, delta=3)
        self.assertAlmostEqual(context.estimate_tokens(history, usage_prompt_tokens=1), raw * 0.5, delta=3)
        self.assertEqual(context.estimate_tokens(history, usage_prompt_tokens=0), raw)       # no usable figure: raw
        self.assertEqual(context.estimate_tokens(history, usage_prompt_tokens=None), raw)


# -- round boundaries ------------------------------------------------------------------------------------------------

class RoundBoundariesTest(unittest.TestCase):
    def test_boundaries_are_assistant_after_user(self):
        self.assertEqual(context.round_boundaries(conversation(3)), [1, 3, 5])
        self.assertEqual(context.round_boundaries([]), [])
        self.assertEqual(context.round_boundaries([user("a")]), [])

    def test_index_zero_and_assistant_after_assistant_are_not_boundaries(self):
        self.assertEqual(context.round_boundaries([asst(1), results(1), asst(2)]), [2])
        self.assertEqual(context.round_boundaries([user("a"), asst(1), asst(2)]), [1])

    def test_consecutive_user_messages_are_fine(self):
        history = [user("a"), user("b"), asst(1), results(1), user("screenshot"), asst(2)]
        self.assertEqual(context.round_boundaries(history), [2, 5])

    def test_synthetic_messages_are_never_cut_points(self):
        history = [user("a"), Message(role="assistant", content="x", synthetic=True, tag="wrapup"), results(1)]
        self.assertEqual(context.round_boundaries(history), [])

    def test_a_cut_never_separates_a_call_from_its_results(self):
        history = conversation(6)
        for i in context.round_boundaries(history):
            kept = history[i:]
            self.assertEqual(kept[0].role, "assistant")
            self.assertTrue({c.id for c in kept[0].tool_calls} <= {r.tool_call_id for r in kept[1].tool_results})


# -- script elision ------------------------------------------------------------------------------------------------

class ElideScriptTest(unittest.TestCase):
    CODE = "\n".join(f"obj{i} = doc.addObject('Part::Box', 'B{i}')" for i in range(20))

    def test_returns_a_copy_and_leaves_the_original_alone(self):
        call = ToolCall("c1", "execute_code", {"code": self.CODE, "timeout": 30})
        out = context.elide_script_args(call, "scripts/007_shell.py")
        self.assertIsNot(out, call)
        self.assertEqual(call.arguments["code"], self.CODE)
        self.assertEqual(out.arguments["code"], "[script elided: 20 lines, saved as scripts/007_shell.py]")
        self.assertEqual(out.arguments["timeout"], 30)
        self.assertEqual((out.id, out.name), ("c1", "execute_code"))

    def test_without_a_saved_path(self):
        out = context.elide_script_args(ToolCall("c1", "execute_code", {"code": self.CODE}), None)
        self.assertEqual(out.arguments["code"], "[script elided: 20 lines]")

    def test_threshold_is_600_characters(self):
        for size, elided in ((600, False), (601, True)):
            call = ToolCall("c1", "execute_code", {"code": "x" * size})
            out = context.elide_script_args(call, None)
            self.assertEqual(out.arguments["code"] != "x" * size, elided, size)
            self.assertIsNot(out, call)

    def test_headless_and_async_scripts_are_elided_other_tools_are_not(self):
        for name, expect in (("execute_code_headless", True), ("execute_code_async", True), ("create_object", False)):
            out = context.elide_script_args(ToolCall("c1", name, {"code": self.CODE}), "scripts/001_x.py")
            self.assertEqual("elided" in out.arguments["code"], expect, name)

    def test_undecodable_arguments_pass_through(self):
        call = ToolCall("c1", "execute_code", {"_raw": "{bad"}, args_error="invalid JSON", raw_args="{bad")
        out = context.elide_script_args(call, "scripts/001_x.py")
        self.assertEqual(out.arguments, {"_raw": "{bad"})
        self.assertEqual(out.args_error, "invalid JSON")
        non_string = ToolCall("c1", "execute_code", {"code": 12345})
        self.assertEqual(context.elide_script_args(non_string, None).arguments, {"code": 12345})

    def test_script_paths_from_steps(self):
        steps = [mock.Mock(call_id="c1", script="scripts/001_a.py", superseded=False),
                 mock.Mock(call_id="c2", script="", superseded=False),
                 mock.Mock(call_id="c3", script="scripts/003_c.py", superseded=True),
                 mock.Mock(call_id="c4", script="scripts/004_d.py", superseded=False)]
        self.assertEqual(context.script_paths_from_steps(steps), {"c1": "scripts/001_a.py", "c4": "scripts/004_d.py"})
        self.assertEqual(context.script_paths_from_steps([]), {})
        self.assertEqual(context.script_paths_from_steps(None), {})


# -- pinned memory -------------------------------------------------------------------------------------------------

def ledger_dict(text: str = BRIEF) -> dict:
    return flow.Ledger.from_request(text).to_dict()


class PinnedMemoryTest(unittest.TestCase):
    def full(self, **kw):
        defaults = dict(
            ledger=ledger_dict(),
            params=[{"name": "wall_t", "value": 2.0, "unit": "mm", "source": "assumption"},
                    {"name": "clearance", "value": 0.3, "unit": "mm", "source": "assumption"},
                    {"name": "holes", "value": 4, "unit": "", "source": "datasheet"}],
            objects=[{"name": "Shell", "type": "Part::Feature", "solids": 1, "volume": 18432.4, "doc": "PiCase"},
                     {"name": "Part_Shell", "type": "App::Part", "doc": "PiCase"},
                     {"name": "Ref_RPI4B", "type": "App::Part", "hidden": True, "doc": "PiCase"},
                     {"name": "Params", "type": "Spreadsheet::Sheet", "doc": "PiCase"}],
            reference={"id": "rpi4b", "status": "catalog-nominal", "source": "catalog:rpi4b"},
            files=["/abs/x.step"], decisions=["wall_t chosen 2.0 (typical FDM)", "user declined the download -> nominal data"],
            project_dir="/abs/projects/pi4-3f9a1c")
        defaults.update(kw)
        return memory_text(**defaults)

    def test_skeleton_sections_in_order(self):
        text = self.full()
        order = ["<task_memory>", "REQUEST (verbatim, turn 1):", BRIEF, "INTENT: enclosure · PHASE:", "PLAN:",
                 "REQUIREMENTS:", "PARAMETERS (sheet 'Params'):", "OBJECTS (doc 'PiCase'):", "REFERENCE:", "FILES:",
                 "DECISIONS:", "</task_memory>"]
        positions = [text.index(h) for h in order]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(text.startswith("<task_memory>\n") and text.endswith("\n</task_memory>"))

    def test_line_formats(self):
        text = self.full()
        self.assertIn("PARAMETERS (sheet 'Params'): wall_t=2 mm (assumption) | clearance=0.3 mm (assumption) | holes=4 (datasheet)", text)
        self.assertIn("OBJECTS (doc 'PiCase'): Params | Shell [solid, 18432 mm3] | Part_Shell [App::Part] | Ref_RPI4B [App::Part, hidden]", text)
        self.assertIn("REFERENCE: rpi4b · status catalog-nominal · source catalog:rpi4b · sha256 n/a", text)
        self.assertIn("FILES: project /abs/projects/pi4-3f9a1c | file /abs/x.step", text)
        self.assertIn("DECISIONS: wall_t chosen 2.0 (typical FDM) | user declined the download -> nominal data", text)
        self.assertIn("[x] S1 Capture requirements and assumptions", text)
        self.assertIn("<- now", text)

    def test_the_memory_has_no_round_counter_or_revision(self):
        a = ledger_dict()
        b = ledger_dict()
        a.update(round=3, revision=4, updated="2026-10-01T10:00:00")
        b.update(round=17, revision=99, updated="2026-10-01T23:59:59")
        text_a, text_b = self.full(ledger=a), self.full(ledger=b)
        self.assertEqual(text_a, text_b)
        self.assertNotRegex(text_a, r"(?i)\brev\b|revision|\bROUND\b|\bround \d")

    def test_empty_sections_are_omitted(self):
        text = memory_text()
        self.assertEqual(text, f"<task_memory>\nREQUEST (verbatim, turn 1):\n{BRIEF}\n</task_memory>")
        for header in ("PLAN", "PARAMETERS", "OBJECTS", "REFERENCE", "FILES", "DECISIONS", "INTENT"):
            self.assertNotIn(header, text)

    def test_invalid_objects_are_flagged_and_visibility_comes_from_the_health_rows(self):
        text = memory_text(objects=[{"name": "Lid", "type": "Part::Feature", "solids": 1, "volume": 4000.0, "valid": False,
                                     "visible": True, "doc": "D"},
                                    {"name": "Ref_X", "type": "App::Part", "visible": False, "doc": "D"}])
        self.assertIn("Lid [solid, 4000 mm3, INVALID]", text)
        self.assertIn("Ref_X [App::Part, hidden]", text)

    def test_status_none_reference_without_id_is_omitted(self):
        self.assertNotIn("REFERENCE", memory_text(reference={"id": "", "name": "", "status": "none", "entry": ""}))
        self.assertIn("REFERENCE: rpi4b", memory_text(reference={"id": "rpi4b", "status": "none"}))

    def test_reference_reads_the_entry_when_it_is_a_dict(self):
        text = memory_text(reference={"id": "rpi4b", "status": "step-imported",
                                      "entry": {"source": "https://example.test/rpi4b.step", "sha256": "ab" * 32}})
        self.assertIn("source https://example.test/rpi4b.step · sha256 " + "ab" * 6, text)
        self.assertNotIn("ab" * 7, text)   # shortened to 12 hex characters

    def test_plan_lines_are_those_of_render_compact(self):
        ledger = flow.Ledger.from_request(BRIEF)
        compact = ledger.render_compact().split("\n")
        text = context.pinned_memory(MemoryInputs(request=BRIEF, ledger=ledger.to_dict(), params=[], objects=[],
                                                  reference=None, files=[], decisions=[]))
        lines = text.split("\n")
        start = lines.index("PLAN:")
        self.assertEqual(lines[start + 1:start + len(compact)], compact[1:])

    def test_plan_fallback_matches_render_compact_when_flow_is_unavailable(self):
        data = ledger_dict()
        data["steps"][1]["status"] = "running"
        data["steps"][2]["status"] = "skipped"
        data["steps"][2]["note"] = "no STEP file: nominal board data used"
        via_flow = context._plan_lines(data)
        with mock.patch.object(flow.Ledger, "from_dict", side_effect=ValueError("broken")):
            fallback = context._plan_lines(data)
        self.assertEqual(fallback, via_flow)

    def test_request_is_verbatim_and_survives_every_clip(self):
        request = "Build a housing for <it>.\nLine two with unicode: Ø12 mm, 90° — and `backticks`."
        many = dict(params=[{"name": f"p_{i:02d}", "value": i, "unit": "mm", "source": "derived"} for i in range(60)],
                    objects=[{"name": f"Obj{i:03d}", "type": "Part::Feature", "solids": 1, "volume": 1000 + i, "doc": "D"}
                             for i in range(120)],
                    decisions=[f"decision number {i} " * 6 for i in range(30)], files=[f"/abs/file{i}.step" for i in range(30)],
                    ledger=ledger_dict(request))
        for limit in (6000, 3000, 1800, 1200, 800, 100):
            text = memory_text(request=request, max_chars=limit, **many) if False else context.pinned_memory(
                MemoryInputs(request=request, reference=None, **many), max_chars=limit)
            self.assertIn(request, text, limit)
            self.assertTrue(text.startswith("<task_memory>") and text.endswith("</task_memory>"), limit)

    def test_clip_order_objects_first_then_parameters(self):
        base = dict(request=BRIEF, ledger=None, reference=None, files=[], decisions=[],
                    params=[{"name": f"p_{i:02d}", "value": i, "unit": "mm", "source": "derived"} for i in range(10)],
                    objects=[{"name": f"Obj{i:03d}", "type": "Part::Feature", "solids": 1, "volume": 1000 + i, "doc": "D"}
                             for i in range(80)])
        full = context.pinned_memory(MemoryInputs(**base), max_chars=10_000)
        self.assertIn("+40 more: get_objects", full)                  # the default view shows the first 40 objects
        mid = context.pinned_memory(MemoryInputs(**base), max_chars=len(full) - 300)
        self.assertLessEqual(len(mid), len(full) - 300)
        self.assertIn("p_09=9 mm (derived)", mid)                      # parameters untouched while objects shrink
        self.assertNotIn("Obj039 [", mid)
        self.assertIn("Obj000 [solid, 1000 mm3]", mid)
        # so tight that even an empty OBJECTS line does not leave room for every parameter
        no_objects = context.pinned_memory(MemoryInputs(**{**base, "objects": []}), max_chars=10_000)
        tight = context.pinned_memory(MemoryInputs(**base), max_chars=len(no_objects) + 20)
        self.assertLessEqual(len(tight), len(no_objects) + 20)
        self.assertIn("OBJECTS (doc 'D'): ... +80 more: get_objects", tight)      # objects shrunk to nothing first
        self.assertIn("PARAMETERS (sheet 'Params'): p_00=0 mm (derived)", tight)
        shorter = context.pinned_memory(MemoryInputs(**base), max_chars=len(no_objects) - 60)
        self.assertLessEqual(len(shorter), len(no_objects) - 60)
        self.assertIn("... +", shorter.split("PARAMETERS")[1].split("\n")[0])        # and then the parameters shrink too
        self.assertIn(BRIEF, shorter)

    def test_caps_at_24_parameters_and_40_objects(self):
        text = memory_text(params=[{"name": f"p_{i:02d}", "value": i} for i in range(30)],
                           objects=[{"name": f"Obj{i:03d}", "doc": "D"} for i in range(45)])
        self.assertIn("p_23=23", text)
        self.assertNotIn("p_24=24", text)
        self.assertIn("... +6 more: get_parameters", text)
        self.assertIn("... +5 more: get_objects", text)

    def test_tool_names_decide_the_parameter_hint(self):
        inp = MemoryInputs(request=BRIEF, ledger=None, params=[{"name": f"p_{i:02d}", "value": i} for i in range(30)],
                           objects=[], reference=None, files=[], decisions=[])
        self.assertIn("more: get_parameters", context.pinned_memory(inp))
        self.assertIn("more: get_parameters", context.pinned_memory(inp, tool_names={"get_parameters", "get_objects"}))
        lite = context.pinned_memory(inp, tool_names={"get_objects"})
        self.assertNotIn("get_parameters", lite)
        self.assertIn("see the Params sheet", lite)

    def test_ultra_lite_budget_keeps_request_and_plan(self):
        many = dict(params=[{"name": f"p_{i:02d}", "value": i, "unit": "mm", "source": "derived"} for i in range(40)],
                    objects=[{"name": f"Obj{i:03d}", "type": "Part::Feature", "solids": 1, "volume": 99, "doc": "D"}
                             for i in range(80)],
                    decisions=[f"decision {i}" for i in range(12)], files=["/abs/a.step"], reference={"id": "rpi4b", "status": "none"})
        text = context.pinned_memory(MemoryInputs(request=BRIEF, ledger=ledger_dict(), **many),
                                     max_chars=context.ULTRA_LITE_MEMORY_CHARS)
        self.assertLessEqual(len(text), context.ULTRA_LITE_MEMORY_CHARS)
        self.assertIn(BRIEF, text)
        self.assertIn("[ ] S2 Source the reference part   <- now", text)

    def test_the_plan_collapses_to_ids_when_nothing_else_can_shrink(self):
        text = context.pinned_memory(MemoryInputs(request=BRIEF, ledger=ledger_dict(), params=[], objects=[], reference=None,
                                                  files=[], decisions=[]), max_chars=500)
        self.assertIn("PLAN: S1 [x] | S2 [ ] <- now | S3 [ ]", text)
        self.assertNotIn("Capture requirements", text)
        self.assertIn(BRIEF, text)

    def test_turn_index_and_unusable_inputs(self):
        text = context.pinned_memory(MemoryInputs(request="  hi  ", ledger={"intent": "question"}, params=["junk", None],
                                                  objects=[None, {"nope": 1}], reference="junk", files=["", None],
                                                  decisions=["", "  "], turn_index=3))
        self.assertIn("REQUEST (verbatim, turn 3):\nhi\nINTENT: question · PHASE: INTAKE", text)
        self.assertNotIn("PARAMETERS", text)
        self.assertNotIn("OBJECTS", text)
        self.assertNotIn("FILES", text)
        self.assertNotIn("DECISIONS", text)

    def test_blocked_steps_show_their_reason(self):
        data = ledger_dict()
        data["steps"][1].update(status="blocked", note="user declined the download")
        self.assertIn("[!] S2 Source the reference part - blocked: user declined the download", self.full(ledger=data))

    def test_deterministic(self):
        self.assertEqual(self.full(), self.full())

    def test_huge_request_is_clipped_at_the_hard_cap_only(self):
        request = "x" * (context.REQUEST_HARD_CAP + 500)
        text = context.pinned_memory(MemoryInputs(request=request, ledger=None, params=[], objects=[], reference=None,
                                                  files=[], decisions=[]), max_chars=1200)
        self.assertIn("x" * context.REQUEST_HARD_CAP, text)
        self.assertIn("[request clipped: 500 more characters]", text)


# -- the request view ----------------------------------------------------------------------------------------------

class NoPressureViewTest(ViewCase):
    def test_a_short_conversation_passes_through_unchanged(self):
        history = conversation(4)
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(200_000))
        self.check_view(view)
        self.assertEqual(len(view), len(history))
        self.assertEqual(view[0].content, history[0].content)
        self.assertEqual({k: stats[k] for k in ("dropped_rounds", "elided_scripts", "clipped_results", "dropped_images")},
                         {"dropped_rounds": 0, "elided_scripts": 0, "clipped_results": 0, "dropped_images": 0})
        self.assertFalse(stats["memory"])
        self.assertFalse(stats["too_small"])
        self.assertEqual(stats["before"], stats["after"])

    def test_the_view_is_a_copy_and_the_history_is_never_mutated(self):
        history = conversation(30, code="x = 1  # " + "y" * 800)
        before = frozen(history)
        view, _ = context.build_request_view(history, memory=memory_text(), caps=caps(4000, 500))
        self.assertEqual(frozen(history), before)
        for m in view:
            self.assertFalse(any(m is h for h in history))
        view2, _ = context.build_request_view(history, memory=memory_text(), caps=caps(200_000))
        self.assertEqual(frozen(history), before)
        self.assertEqual(len(view2), len(history))
        view2[1].content = "changed"
        view2[1].tool_calls.clear()
        self.assertEqual(frozen(history), before)
        # calls and results are copies too: a provider adapting the request in place cannot reach the stored history
        view3, _ = context.build_request_view(history, memory=memory_text(), caps=caps(200_000))
        view3[2].tool_results[0].content = "mutated"
        view3[2].tool_results[0].image_b64 = "X"
        view3[1].tool_calls[0].arguments["n"] = 999
        view3[1].tool_calls[0].name = "other"
        self.assertEqual(frozen(history), before)

    def test_empty_history(self):
        view, stats = context.build_request_view([], memory="x", caps=None)
        self.assertEqual(view, [])
        self.assertEqual(stats["after"], 0)

    def test_unknown_caps_use_the_32768_window(self):
        history = conversation(3)
        _, stats = context.build_request_view(history, memory="", caps=None)
        self.assertEqual(stats["window"], 32768)
        self.assertEqual(stats["budget"], int(0.7 * 32768) - 4096)

    def test_budget_is_70_percent_of_the_window_minus_the_output_reserve(self):
        _, stats = context.build_request_view(conversation(2), memory="", caps=caps(100_000, 64_000))
        self.assertEqual(stats["budget"], int(0.7 * 100_000) - 8192)   # the reserve is capped at 8192
        _, stats = context.build_request_view(conversation(2), memory="", caps=caps(10_000, 2_000))
        self.assertEqual(stats["budget"], 7000 - 2000)


class GovernorLinesTest(ViewCase):
    def test_only_the_newest_results_message_keeps_the_governor_lines(self):
        history = conversation(6)
        history[-1].content += "\n[phase] BUILD: build one named step per call."
        before = frozen(history)
        view, _ = context.build_request_view(history, memory="", caps=caps(200_000))
        for m in view[:-1]:
            self.assertNotIn("[governor]", m.content)
            self.assertNotIn("[phase]", m.content)
        self.assertIn("[governor] round 6/30", view[-1].content)
        self.assertIn("[phase] BUILD:", view[-1].content)
        self.assertTrue(view[2].content.startswith("round 1 done"))
        self.assertEqual(view[2].content, "round 1 done")
        self.assertEqual(frozen(history), before)

    def test_an_older_results_message_keeps_its_real_text(self):
        history = [user("go"), asst(1), results(1), asst(2), results(2)]
        history[2].content = "first line\n[governor] round 1/30 · BUILD\nlast line"
        view, _ = context.build_request_view(history, memory="", caps=caps(200_000))
        self.assertEqual(view[2].content, "first line\nlast line")


class CompactionTest(ViewCase):
    def test_a_125_message_build_keeps_the_brief_and_the_newest_rounds(self):
        history = conversation(62)
        self.assertEqual(len(history), 125)
        before = frozen(history)
        memory = memory_text(ledger=ledger_dict(), params=[{"name": "wall_t", "value": 2, "unit": "mm", "source": "assumption"}])
        view, stats = context.build_request_view(history, memory=memory, caps=caps(6000, 1000))
        self.check_view(view)
        self.assertEqual(frozen(history), before)
        self.assertTrue(stats["memory"])
        self.assertGreater(stats["dropped_rounds"], 0)
        self.assertLess(stats["after"], stats["before"])
        self.assertLessEqual(stats["after"], stats["budget"])
        self.assertTrue(view[0].content.startswith("<task_memory>"))
        self.assertIn(BRIEF, view[0].content)
        self.assertIn("wall_t=2 mm (assumption)", view[0].content)
        # the newest two rounds are untouched (the older of the two lost its governor line)
        for mine, theirs in zip(view[-4:], history[-4:]):
            self.assertEqual(mine.tool_calls, theirs.tool_calls)
            self.assertEqual([r.content for r in mine.tool_results], [r.content for r in theirs.tool_results])
        self.assertEqual(view[-1].content, history[-1].content)
        # whole rounds: what is left is a suffix of the history
        kept = len(view) - 1
        self.assertEqual([m.tool_calls for m in view[1:] if m.role == "assistant"],
                         [m.tool_calls for m in history[-kept:] if m.role == "assistant"])

    def test_compaction_shrinks_in_the_documented_order(self):
        code = "\n".join(f"line_{i} = {i}  # " + "p" * 40 for i in range(30))   # 30 lines, > 600 chars
        history = conversation(20, code=code, result_chars=2000)
        # budget so tight only (a) + (b) fit: dropping rounds must not be needed
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(10_000, 1000))
        self.check_view(view)
        self.assertEqual(stats["elided_scripts"], 18)       # every call older than the newest two rounds
        self.assertGreater(stats["clipped_results"], 0)
        self.assertEqual(stats["dropped_rounds"], 0)
        # a still tighter window needs (d) as well
        view2, stats2 = context.build_request_view(history, memory=memory_text(), caps=caps(3000, 500))
        self.check_view(view2)
        self.assertGreater(stats2["dropped_rounds"], 0)
        self.assertTrue(stats2["memory"])

    def test_old_results_keep_the_first_300_and_last_100_characters(self):
        history = conversation(6, result_chars=2000)
        original = history[2].tool_results[0].content
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(3500, 500))
        self.check_view(view)
        self.assertGreater(stats["clipped_results"], 0)
        old = next(m for m in view if m.tool_results and m.tool_results[0].tool_call_id == "c1")
        text = old.tool_results[0].content
        self.assertEqual(text, original[:300] + " ...(elided)... " + original[-100:])
        self.assertEqual(len(text), 300 + len(" ...(elided)... ") + 100)
        newest = view[-1].tool_results[0].content
        self.assertEqual(newest, history[-1].tool_results[0].content)

    def test_script_elision_marks_the_saved_path_and_rewrites_the_provider_blocks_copy(self):
        code = "\n".join(f"shape_{i} = Part.makeBox(10, 10, {i + 1})  # " + "q" * 30 for i in range(25))
        history = [user(BRIEF)]
        for i in range(1, 13):
            blocks = [{"type": "thinking", "thinking": "t", "signature": f"sig{i}"},
                      {"type": "tool_use", "id": f"c{i}", "name": "execute_code", "input": {"code": code, "timeout": 30}}]
            history.append(asst(i, code=code, blocks=blocks))
            history.append(results(i, "ok"))
        before = frozen(history)
        paths = {f"c{i}": f"scripts/{i:03d}_step.py" for i in range(1, 13)}
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(6000, 1000), script_paths=paths)
        self.check_view(view)
        self.assertEqual(stats["elided_scripts"], 10)
        self.assertEqual(frozen(history), before)
        by_id = {m.tool_calls[0].id: m for m in view if m.role == "assistant" and m.tool_calls}
        marker = "[script elided: 25 lines, saved as scripts/003_step.py]"
        self.assertEqual(by_id["c3"].tool_calls[0].arguments["code"], marker)
        tool_use = next(b for b in by_id["c3"].provider_blocks if b["type"] == "tool_use")
        self.assertEqual(tool_use["input"], {"code": marker, "timeout": 30})
        self.assertEqual(by_id["c3"].provider_blocks[0], history[5].provider_blocks[0])   # the thinking block is untouched
        for i in (11, 12):                                        # the newest two rounds keep their code everywhere
            self.assertEqual(by_id[f"c{i}"].tool_calls[0].arguments["code"], code)
            self.assertEqual(by_id[f"c{i}"].provider_blocks, history[2 * i - 1].provider_blocks)
        self.assertLess(stats["after"], stats["before"])

    def test_script_elision_without_a_project_has_no_path(self):
        code = "x = 1  # " + "z" * 700
        history = conversation(10, code=code)
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(4000, 500))
        self.check_view(view)
        old = next(m for m in view if m.role == "assistant" and m.tool_calls and m.tool_calls[0].id == "c8")
        self.assertEqual(old.tool_calls[0].arguments["code"], "[script elided: 1 lines]")


    def test_images_beyond_the_newest_three_are_dropped_and_flat_costed(self):
        history = conversation(10, result_chars=40, image="A" * 5000)
        _, stats_all = context.build_request_view(history, memory="", caps=caps(200_000))
        self.assertGreaterEqual(stats_all["before"], 10 * 1500)
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(20_000, 1000))
        self.check_view(view)
        self.assertEqual(stats["dropped_images"], 7)
        shots = [bool(m.tool_results and m.tool_results[0].image_b64) for m in view if m.tool_results]
        self.assertEqual(shots, [False] * 7 + [True] * 3)
        self.assertEqual(stats["dropped_rounds"], 0)
        self.assertLess(stats["after"], stats["before"])
        # the stored history still has all ten
        self.assertEqual(sum(1 for m in history if m.tool_results and m.tool_results[0].image_b64), 10)

    def test_user_message_images_are_dropped_and_leave_a_note(self):
        history = [user("look at these", images=["a" * 100, "b" * 100]), asst(1), results(1), asst(2), results(2),
                   asst(3), results(3), asst(4), results(4, image="IMG")]
        history[0].content = ""
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(2500, 500), keep_screenshots=1)
        self.check_view(view)
        self.assertGreaterEqual(stats["dropped_images"], 2)
        for m in view:
            if m.role == "user" and not m.tool_results:
                self.assertTrue(m.content or m.images)

    def test_screenshot_keep_count_never_exceeds_the_provider_cap(self):
        history = conversation(14, result_chars=40, image="A" * 50)
        view, _ = context.build_request_view(history, memory=memory_text(), caps=caps(30_000, 1000), keep_screenshots=50)
        kept = sum(1 for m in view for r in m.tool_results if r.image_b64)
        self.assertLessEqual(kept, 6)


class SyntheticNotesTest(ViewCase):
    def history(self):
        h = [user(BRIEF)]
        for i in range(1, 9):
            h += [asst(i), results(i, "ok " * 80)]
        h += [Message(role="assistant", content="(this reply was cut off at the output-token limit)"),
              user("[cut off note] act now", synthetic=True, tag="cutoff")]
        for i in range(9, 15):
            h += [asst(i), results(i, "ok " * 80)]
        h += [Message(role="assistant", content="All done."),
              user("[gate] required steps are still open", synthetic=True, tag="gate")]
        for i in range(15, 21):
            h += [asst(i), results(i, "ok " * 80)]
        h += [Message(role="assistant", content="paused"), user("also add a lid", synthetic=True, tag="steer")]
        for i in range(21, 25):
            h += [asst(i), results(i, "ok " * 80)]
        return h

    def test_every_window_gives_an_alternating_view_with_paired_notes(self):
        history = self.history()
        before = frozen(history)
        seen_cuts = set()
        for window in range(700, 9000, 150):
            view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(window, 200))
            self.check_view(view)
            seen_cuts.add(stats["dropped_rounds"])
            for i, m in enumerate(view):
                if m.synthetic:
                    self.assertGreater(i, 0, f"window {window}: a synthetic note opens the view")
                    self.assertEqual(view[i - 1].role, "assistant", f"window {window}: a note lost its assistant message")
                    self.assertFalse(view[i - 1].tool_calls)
            self.assertFalse(stats.get("alternation_fallback"), window)
        self.assertEqual(frozen(history), before)
        self.assertGreater(len(seen_cuts), 5)    # the sweep really exercised many different cuts

    def test_a_cut_right_after_a_note_drops_the_note_with_its_assistant_message(self):
        history = self.history()
        note_index = next(i for i, m in enumerate(history) if m.tag == "cutoff")
        pair = (history[note_index - 1], history[note_index])
        for window in range(700, 9000, 100):
            view, _ = context.build_request_view(history, memory=memory_text(), caps=caps(window, 200))
            present = [any(m is p or (m.content == p.content and m.role == p.role) for m in view) for p in pair]
            self.assertEqual(present[0], present[1], f"window {window}: the cutoff pair was split")

    def test_cutoff_gate_and_steer_notes_alternate_without_any_cut(self):
        history = self.history()
        view, stats = context.build_request_view(history, memory="", caps=caps(500_000, 4096))
        self.check_view(view)
        self.assertEqual(len(view), len(history))
        self.assertEqual([m.tag for m in view if m.synthetic], ["cutoff", "gate", "steer"])


class ReasoningBlocksTest(ViewCase):
    @staticmethod
    def thinking_history(rounds: int, *, thinking_chars: int = 400) -> list[Message]:
        history = [user(BRIEF)]
        for i in range(1, rounds + 1):
            blocks = [{"type": "thinking", "thinking": f"t{i} " * (thinking_chars // 4), "signature": f"sig{i}"},
                      {"type": "tool_use", "id": f"c{i}", "name": "execute_code", "input": {"n": i}}]
            history.append(asst(i, blocks=blocks))
            history.append(results(i, "ok " * 60))
        return history

    def test_open_loop_keeps_its_blocks_and_rounds_are_whole(self):
        history = self.thinking_history(40)
        before = frozen(history)
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(2500, 200, reasoning_replay=True))
        self.check_view(view)
        self.assertEqual(frozen(history), before)
        self.assertGreater(stats["dropped_rounds"], 0)
        self.assertEqual(stats["dropped_thinking"], 0)       # one user turn: nothing is "earlier"
        assistants = [m for m in view if m.role == "assistant"]
        self.assertEqual(assistants[-1].provider_blocks, history[-2].provider_blocks)
        for m in assistants:
            self.assertEqual(m.provider_blocks[0]["type"], "thinking")
            self.assertTrue(m.provider_blocks[0]["signature"].startswith("sig"))
            self.assertEqual(m.provider_blocks[1]["id"], m.tool_calls[0].id)       # block and call of one round stay together

    def test_thinking_of_earlier_turns_is_dropped_before_any_round_is(self):
        history = [user(BRIEF)]
        for i in range(1, 11):
            history += [asst(i, blocks=[{"type": "thinking", "thinking": "t" * 3500, "signature": f"s{i}"},
                                        {"type": "tool_use", "id": f"c{i}", "name": "execute_code", "input": {"n": i}}]),
                        results(i, "ok")]
        history += [Message(role="assistant", content="turn one answer"), user("now add a lid")]
        for i in range(11, 14):
            history += [asst(i, blocks=[{"type": "thinking", "thinking": "u" * 3500, "signature": f"s{i}"},
                                        {"type": "tool_use", "id": f"c{i}", "name": "execute_code", "input": {"n": i}}]),
                        results(i, "ok")]
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(20_000, 1000))
        self.check_view(view)
        self.assertEqual(stats["dropped_thinking"], 10)
        self.assertEqual(stats["dropped_rounds"], 0)
        by_id = {m.tool_calls[0].id: m for m in view if m.role == "assistant" and m.tool_calls}
        self.assertEqual([b["type"] for b in by_id["c3"].provider_blocks], ["tool_use"])
        self.assertEqual([b["type"] for b in by_id["c12"].provider_blocks], ["thinking", "tool_use"])


    def test_over_budget_is_accepted_rather_than_dropping_the_open_loop_blocks(self):
        history = self.thinking_history(3, thinking_chars=40_000)
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(2000, 200, reasoning_replay=True))
        self.check_view(view)
        self.assertGreater(stats["after"], stats["budget"])
        self.assertTrue(stats["too_small"])
        for m in view:
            if m.role == "assistant":
                self.assertEqual(m.provider_blocks[0]["type"], "thinking")

    def test_anthropic_body_of_a_40_round_thinking_history_starts_with_its_thinking_block(self):
        try:
            from oscar import providers
            from oscar.params import Profile
        except ImportError:
            self.skipTest("providers need FreeCAD")
        history = self.thinking_history(40)
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(2500, 200, reasoning_replay=True))
        self.assertGreater(stats["dropped_rounds"], 0)
        body = providers.AnthropicProvider(Profile())._body("system", view, [])
        wire = body["messages"]
        self.assertEqual(wire[0]["role"], "user")
        last_assistant = [m for m in wire if m["role"] == "assistant"][-1]
        self.assertEqual(last_assistant["content"][0]["type"], "thinking")
        self.assertEqual(last_assistant["content"][0]["signature"], "sig40")
        for m in wire:
            if m["role"] == "assistant":
                self.assertEqual(m["content"][0]["type"], "thinking")
        roles = [m["role"] for m in wire]
        self.assertTrue(all(a != b or a == "user" for a, b in zip(roles, roles[1:])))


class MemoryMaterializationTest(ViewCase):
    def test_memory_is_added_only_when_rounds_are_dropped_or_the_request_left(self):
        history = conversation(4)
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(200_000))
        self.assertFalse(stats["memory"])
        self.assertNotIn("<task_memory>", "".join(m.content for m in view))

    def test_a_request_that_left_the_window_is_merged_into_the_first_user_message(self):
        trimmed = "[Earlier conversation was trimmed to save context.]\n\nmake the lid thicker"
        history = [user(trimmed), asst(1), results(1), asst(2), results(2)]
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(200_000))
        self.check_view(view)
        self.assertTrue(stats["memory"])
        self.assertEqual(len(view), len(history))                   # merged, not prepended
        self.assertTrue(view[0].content.startswith("<task_memory>"))
        self.assertTrue(view[0].content.endswith(trimmed))
        self.assertEqual(history[0].content, trimmed)

    def test_a_history_that_starts_with_an_assistant_message_gets_the_memory_prepended(self):
        history = [asst(1), results(1), asst(2), results(2)]
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(200_000))
        self.check_view(view)
        self.assertEqual(len(view), len(history) + 1)
        self.assertEqual(view[0].role, "user")
        self.assertTrue(view[0].content.startswith("<task_memory>"))

    def test_without_a_memory_text_a_cut_leaves_a_plain_note(self):
        view, stats = context.build_request_view(conversation(30), memory="", caps=caps(3000, 500))
        self.check_view(view)
        self.assertGreater(stats["dropped_rounds"], 0)
        # nothing else remembers what was asked, so the dropped request rides along with the note
        self.assertTrue(view[0].content.startswith(context._OMITTED_NOTE))
        self.assertIn("CURRENT REQUEST", view[0].content)
        self.assertIn(BRIEF, view[0].content)

    def multi_turn(self):
        history = conversation(6)
        history += [Message(role="assistant", content="the shell is done"),
                    user("make the lid thicker\n\n<workspace_state>\nOpen documents:\n- PiCase\n</workspace_state>")]
        for i in range(7, 25):
            history += [asst(i), results(i, "ok " * 200)]
        return history

    def test_the_latest_user_request_survives_a_cut_when_the_memory_holds_another_one(self):
        history = self.multi_turn()
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(2500, 300))
        self.check_view(view)
        self.assertGreater(stats["dropped_rounds"], 0)
        self.assertNotIn("make the lid thicker", "".join(m.content for m in view[1:] if m.role == "user" and not m.tool_results))
        head = view[0].content
        self.assertIn("CURRENT REQUEST (the user's latest message, verbatim):\nmake the lid thicker", head)
        self.assertNotIn("workspace_state", head)                     # only what the user typed
        self.assertTrue(head.endswith("</task_memory>"))
        self.assertLess(head.index(BRIEF), head.index("CURRENT REQUEST"))

    def test_no_duplicate_when_the_memory_already_has_the_latest_request(self):
        history = self.multi_turn()
        memory = memory_text(request="make the lid thicker", turn_index=2)
        view, stats = context.build_request_view(history, memory=memory, caps=caps(2500, 300))
        self.assertGreater(stats["dropped_rounds"], 0)
        self.assertEqual(view[0].content, memory)

    def test_the_guard_stays_quiet_while_the_latest_request_is_still_in_the_view(self):
        history = self.multi_turn()
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(30_000, 1000))
        self.assertFalse(stats["memory"])
        self.assertIn("make the lid thicker", "".join(m.content for m in view))

    def test_consecutive_views_are_byte_identical_except_for_their_last_messages(self):
        history = conversation(12)
        memory = memory_text(ledger=ledger_dict())
        previous = None
        for n in range(3, len(history) + 1, 2):
            view, stats = context.build_request_view(history[:n], memory=memory, caps=caps(200_000))
            self.assertFalse(stats["memory"])
            if previous is not None:
                self.assertEqual(frozen(previous[:-1]), frozen(view[:len(previous) - 1]), n)
            previous = view

    def test_the_view_of_a_conversation_is_deterministic(self):
        history = conversation(40)
        a, sa = context.build_request_view(history, memory=memory_text(), caps=caps(3000, 500))
        b, sb = context.build_request_view(history, memory=memory_text(), caps=caps(3000, 500))
        self.assertEqual(frozen(a), frozen(b))
        self.assertEqual(sa, sb)


class FloorRuleTest(ViewCase):
    def test_a_4096_window_cannot_hold_the_prompt_and_tools(self):
        system = prompt.CORE
        tools = [spec(f"tool_{i}") for i in range(20)]
        history = conversation(6)
        view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(4096, 4096), system=system, tools=tools)
        self.check_view(view)
        self.assertTrue(stats["too_small"])
        _, ok_stats = context.build_request_view(history, memory=memory_text(), caps=caps(32768, 4096), system=system, tools=tools)
        self.assertFalse(ok_stats["too_small"])

    def test_negative_budget_still_returns_a_valid_view(self):
        view, stats = context.build_request_view(conversation(20), memory=memory_text(), caps=caps(4096, 4096))
        self.check_view(view)
        self.assertEqual(stats["budget"], 0)       # 0.7 * 4096 - 4096 < 0 is floored at 0
        self.assertGreater(stats["dropped_rounds"], 0)

    def test_the_lite_prompt_fits_where_the_core_prompt_does_not(self):
        small = caps(8192, 1024)
        tools = [spec(f"tool_{i}", 150) for i in range(19)]
        history = conversation(3)
        _, core = context.build_request_view(history, memory="", caps=small, system=prompt.CORE, tools=[spec(f"t{i}") for i in range(40)])
        _, lite = context.build_request_view(history, memory="", caps=small, system=prompt.LITE, tools=tools)
        self.assertTrue(core["too_small"])
        self.assertFalse(lite["too_small"])

    def test_notice_text(self):
        text = context.too_small_notice(4096)
        self.assertIn("4096 tokens", text)
        self.assertIn("16384", text)
        self.assertIn("num_ctx", text)
        self.assertIn("build_enclosure", context.ULTRA_LITE_TOOLS)
        self.assertEqual(context.ULTRA_LITE_MEMORY_CHARS, 1200)


class CalibrationTest(ViewCase):
    def setUp(self):
        self.history = conversation(30, result_chars=900)
        self.caps = caps(10_000, 1000)       # budget 6000

    def view(self, **kw):
        return context.build_request_view(self.history, memory=memory_text(), caps=self.caps, **kw)

    def test_the_uncalibrated_estimate_triggers_compaction(self):
        _, stats = self.view()
        self.assertGreater(stats["before"], stats["budget"])

    def test_a_measured_lower_usage_lowers_the_estimate(self):
        _, base = self.view()
        view, stats = self.view(usage_prompt_tokens=int(base["raw_after"] * 0.5), sent_estimate=base["raw_after"])
        self.assertEqual(stats["k"], 0.5)
        self.assertLess(stats["before"], base["before"])
        self.check_view(view)

    def test_a_measured_higher_usage_raises_it_and_compacts_harder(self):
        _, base = self.view()
        _, hard = self.view(usage_prompt_tokens=base["raw_after"] * 2, sent_estimate=base["raw_after"])
        self.assertEqual(hard["k"], 2.0)
        self.assertGreaterEqual(hard["dropped_rounds"], base["dropped_rounds"])
        self.assertGreater(hard["before"], base["before"])

    def test_the_factor_is_clipped_to_the_band(self):
        _, stats = self.view(usage_prompt_tokens=10_000_000, sent_estimate=1000)
        self.assertEqual(stats["k"], 3.0)
        _, stats = self.view(usage_prompt_tokens=1, sent_estimate=1000)
        self.assertEqual(stats["k"], 0.5)

    def test_without_the_previous_estimate_the_factor_can_only_raise(self):
        _, low = self.view(usage_prompt_tokens=100)
        self.assertEqual(low["k"], 1.0)
        _, high = self.view(usage_prompt_tokens=500_000)
        self.assertGreater(high["k"], 1.0)
        self.assertLessEqual(high["k"], 3.0)

    def test_raw_after_is_the_uncalibrated_estimate_of_the_view(self):
        view, stats = self.view(usage_prompt_tokens=500_000, sent_estimate=1000)
        self.assertEqual(stats["raw_after"], context.estimate_tokens(view))


class AlternationTest(unittest.TestCase):
    def test_valid_views(self):
        context.assert_alternating([user("a"), asst(1), results(1)])
        context.assert_alternating([user("a"), asst(1), results(1), user("screenshot"), user("more")])
        context.assert_alternating([user("a"), Message(role="assistant", content="hi"), user("b")])

    def test_the_view_must_start_with_a_plain_user_message(self):
        for view in ([asst(1), results(1)], [results(1), asst(2), results(2)]):
            with self.assertRaises(AssertionError):
                context.assert_alternating(view)

    def test_two_consecutive_assistant_messages_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "both assistant"):
            context.assert_alternating([user("a"), Message(role="assistant", content="x"), asst(1), results(1)])

    def test_unanswered_and_orphaned_tool_calls_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "no results"):
            context.assert_alternating([user("a"), asst(1)])
        with self.assertRaisesRegex(AssertionError, "unanswered"):
            context.assert_alternating([user("a"), asst(1), results(2)])
        with self.assertRaisesRegex(AssertionError, "without their tool calls"):
            context.assert_alternating([user("a"), Message(role="assistant", content="x"), results(1)])

    def test_trailing_assistant_and_empty_views_are_rejected(self):
        with self.assertRaises(AssertionError):
            context.assert_alternating([user("a"), Message(role="assistant", content="x")])
        with self.assertRaises(AssertionError):
            context.assert_alternating([])
        self.assertEqual(context.alternation_problems([user("a")]), [])

    def test_a_cut_that_would_break_alternation_falls_back_to_no_cut(self):
        history = conversation(12, result_chars=1500)
        bad = [2, 4, 6, 8]   # indices of results messages: a cut there would orphan their tool results
        with mock.patch.object(context, "round_boundaries", return_value=bad):
            view, stats = context.build_request_view(history, memory=memory_text(), caps=caps(3000, 500))
        self.assertTrue(stats["alternation_fallback"])
        context.assert_alternating(view)
        self.assertEqual(stats["dropped_rounds"], 0)

    def test_a_malformed_stored_history_is_reported_not_raised(self):
        history = [user("a"), Message(role="assistant", content="x"), Message(role="assistant", content="y"), user("b")]
        view, stats = context.build_request_view(history, memory="", caps=caps(200_000))
        self.assertEqual(len(view), 4)       # the same defect as the stored history is not "introduced" by the view
        self.assertFalse(stats.get("alternation_fallback"))


class FuzzTest(ViewCase):
    """Seeded random histories (rounds, thinking, images, notes, new turns) across random windows: the invariants hold."""

    @staticmethod
    def random_history(rng) -> list[Message]:
        history = [user(BRIEF)]
        n = 0
        for _ in range(rng.randint(1, 30)):
            n += 1
            kind = rng.random()
            code = f"c{n} = {n}  # " + "z" * rng.randint(0, 1200) if rng.random() < 0.4 else None
            blocks = []
            if rng.random() < 0.4:
                blocks = [{"type": "thinking", "thinking": "t" * rng.randint(10, 3000), "signature": f"s{n}"},
                          {"type": "tool_use", "id": f"c{n}", "name": "execute_code",
                           "input": {"code": code} if code else {"n": n}}]
            if kind < 0.70:
                history += [asst(n, code=code, blocks=blocks),
                            results(n, "ok " * rng.randint(1, 600), image="IMG" if rng.random() < 0.3 else "")]
            elif kind < 0.78:
                history += [Message(role="assistant", content="cut off"), user("act now", synthetic=True, tag="cutoff")]
            elif kind < 0.86:
                history += [Message(role="assistant", content="done?"), user("steps open", synthetic=True, tag="gate")]
            elif kind < 0.93:
                history += [Message(role="assistant", content="the answer"), user(f"next request {n}")]
            else:
                history += [Message(role="assistant", content="working"), user("also a lid", synthetic=True, tag="steer")]
        return history

    @staticmethod
    def assistants(messages):
        return [(m.content, tuple(c.id for c in m.tool_calls)) for m in messages if m.role == "assistant"]

    def test_invariants_over_random_histories(self):
        import random
        rng = random.Random(20261002)
        cuts = memories = 0
        for _ in range(400):
            history = self.random_history(rng)
            before = frozen(history)
            window = rng.choice([700, 1500, 3000, 8000, 30000, 128000])
            model_caps = caps(window, rng.choice([200, 1000, 4096]), reasoning_replay=rng.random() < 0.5)
            extra = {}
            if rng.random() < 0.5:
                extra = dict(usage_prompt_tokens=rng.randint(100, 90_000), sent_estimate=rng.randint(100, 90_000))
            view, stats = context.build_request_view(history, memory=memory_text(), caps=model_caps, **extra)
            self.assertEqual(frozen(history), before)
            self.check_view(view)
            self.assertEqual(view[-1].role, "user")
            kept, original = self.assistants(view), self.assistants(history)
            self.assertEqual(kept, original[len(original) - len(kept):])          # whole rounds: a suffix of the history
            self.assertEqual(stats["dropped_rounds"], len(original) - len(kept))
            self.assertEqual(stats["raw_after"], context.estimate_tokens(view))
            self.assertFalse(stats.get("alternation_fallback"))
            if not stats["memory"]:
                self.assertEqual(len(view), len(history))
                self.assertEqual(view[0].content, history[0].content)
            for mine, theirs in zip(view[-3:], history[-3:]):                       # the newest rounds are untouched
                self.assertEqual(mine.tool_calls, theirs.tool_calls)
                self.assertEqual(mine.provider_blocks, theirs.provider_blocks)
            again, again_stats = context.build_request_view(history, memory=memory_text(), caps=model_caps, **extra)
            self.assertEqual(frozen(view), frozen(again))
            self.assertEqual(stats, again_stats)
            cuts += bool(stats["dropped_rounds"])
            memories += bool(stats["memory"])
        self.assertGreater(cuts, 40)       # the random sweep reached the cutting paths
        self.assertGreater(memories, 40)


# -- workspace snapshot ----------------------------------------------------------------------------------------------

try:
    import FreeCAD as _FreeCAD
except ImportError:   # pure-Python run
    _FreeCAD = None


@unittest.skipIf(_FreeCAD is None, "FreeCAD not importable")
class WorkspaceSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.doc = _FreeCAD.newDocument("SnapDoc")
        self.addCleanup(lambda: _FreeCAD.closeDocument(self.doc.Name))

    def lines(self):
        text = context.workspace_snapshot(max_docs=50)
        block = text.split(f"- {self.doc.Name}", 1)[1].split("\n- ", 1)[0] if f"- {self.doc.Name}" in text else ""
        return text, [ln for ln in block.split("\n")[1:]]

    def test_pins_params_and_collapses_parts(self):
        self.doc.addObject("Part::Box", "Loose")
        part = self.doc.addObject("App::Part", "Part_Shell")
        for name in ("Shell", "Floor", "Wall"):
            part.addObject(self.doc.addObject("Part::Box", name))
        self.doc.addObject("Spreadsheet::Sheet", "Params")
        notes = self.doc.addObject("App::FeaturePython", "NOTES")
        self.assertIsNotNone(notes)
        self.doc.recompute()
        text, lines = self.lines()
        self.assertTrue(text.startswith("Open documents:"))
        names = [ln.split()[0] for ln in lines]
        self.assertEqual(names[:2], ["Params", "NOTES"])           # pinned objects come first
        self.assertIn("    Part_Shell [App::Part] (3 children)", lines)
        for hidden_child in ("Shell", "Floor", "Wall"):
            self.assertNotIn(hidden_child, names)                  # collapsed into the part's line
        self.assertIn("Loose", names)

    def test_origin_features_are_not_listed_and_not_counted(self):
        self.doc.addObject("App::Part", "Part_Empty")
        self.doc.recompute()
        text, lines = self.lines()
        self.assertEqual([ln.strip() for ln in lines], ["Part_Empty [App::Part] (0 children)"])
        self.assertIn(f"- {self.doc.Name}", text)
        self.assertIn(": 1 object(s)", text.split(f"- {self.doc.Name}", 1)[1].split("\n", 1)[0])

    def test_one_child_is_singular(self):
        part = self.doc.addObject("App::Part", "Part_One")
        part.addObject(self.doc.addObject("Part::Box", "Only"))
        _, lines = self.lines()
        self.assertIn("    Part_One [App::Part] (1 child)", lines)

    def test_hidden_label_and_type_are_shown(self):
        box = self.doc.addObject("Part::Box", "Cube")
        box.Label = "My cube"
        box.Visibility = False
        _, lines = self.lines()
        self.assertIn("    Cube 'My cube' [Part::Box, hidden]", lines)

    def test_objects_beyond_the_limit_are_summarized(self):
        for i in range(12):
            self.doc.addObject("Part::Box", f"B{i:02d}")
        text = context.workspace_snapshot(max_docs=50, max_objects=5)
        block = text.split(f"- {self.doc.Name}", 1)[1].split("\n- ", 1)[0]
        object_lines = [ln for ln in block.split("\n")[1:] if ln.startswith("    ")]
        self.assertEqual(len(object_lines), 6)
        self.assertEqual(object_lines[-1], "    ... 7 more (use get_objects)")

    def test_active_document_is_marked_and_counts_are_right(self):
        self.doc.addObject("Part::Box", "A")
        self.doc.addObject("Part::Box", "B")
        text = context.workspace_snapshot(max_docs=50)
        self.assertRegex(text, rf"- {self.doc.Name}( \(active\))?: 2 object\(s\)")

    def test_no_documents(self):
        with mock.patch.object(_FreeCAD, "listDocuments", return_value={}):
            self.assertEqual(context.workspace_snapshot(), "No documents are open (create one with create_document).")

    def test_never_raises(self):
        with mock.patch.object(_FreeCAD, "listDocuments", side_effect=RuntimeError("boom")):
            self.assertEqual(context.workspace_snapshot(), "")

    def test_console_runs_have_no_gui_lines(self):
        if getattr(_FreeCAD, "GuiUp", False):
            self.skipTest("GUI is up")
        text = context.workspace_snapshot()
        self.assertNotIn("Selection:", text)
        self.assertNotIn("Active workbench:", text)

    def test_document_limit_adds_a_pointer(self):
        other = _FreeCAD.newDocument("SnapDocTwo")
        self.addCleanup(lambda: _FreeCAD.closeDocument(other.Name))
        text = context.workspace_snapshot(max_docs=1)
        self.assertIn("more documents (use list_documents)", text)

    def test_gui_lines_when_the_gui_is_up(self):
        gui = mock.Mock()
        gui.Selection.getSelectionEx.return_value = [mock.Mock(ObjectName="Shell", SubElementNames=["Face3", "Face5"]),
                                                     mock.Mock(ObjectName="Lid", SubElementNames=[])]
        gui.activeWorkbench.return_value.name.return_value = "PartDesignWorkbench"
        fake_app = mock.Mock(GuiUp=True)
        with mock.patch.dict(sys.modules, {"FreeCADGui": gui}):
            lines = context._gui_lines(fake_app)
        self.assertEqual(lines, ["Selection: Shell.Face3, Shell.Face5, Lid", "Active workbench: PartDesignWorkbench"])

    def test_gui_lines_survive_gui_failures(self):
        gui = mock.Mock()
        gui.Selection.getSelectionEx.side_effect = RuntimeError("no view")
        gui.activeWorkbench.side_effect = RuntimeError("no workbench")
        with mock.patch.dict(sys.modules, {"FreeCADGui": gui}):
            self.assertEqual(context._gui_lines(mock.Mock(GuiUp=True)), [])
        self.assertEqual(context._gui_lines(mock.Mock(GuiUp=False)), [])


if __name__ == "__main__":
    unittest.main()
