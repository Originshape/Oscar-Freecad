"""Tests of the panel's pure presentation layer (oscar/panel_render.py), driven by hand-written
payloads that follow the worker's signal schemas exactly, plus the Qt-side pieces that can be
checked without a window (Block / Step html).

panel_render itself has no Qt and no FreeCAD: those tests also run under the plain app python.
"""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import panel_render as pr  # noqa: E402
from oscar.panel_render import StepView, TurnView  # noqa: E402

# --------------------------------------------------------------------------------------------------
# payload fixtures (ledger, signals)


def ledger(**over) -> dict:
    base = {
        "version": 1, "intent": "enclosure", "phase": "BUILD", "max_phase": "BUILD",
        "request": "Build a housing for a Raspberry Pi 4.", "needs_reference": True,
        "reference_hint": "a Raspberry Pi 4", "reference_status": "catalog-nominal",
        "requirements": [{"id": "R1", "text": "A bottom shell", "status": "pending", "covered_by": ["S5"],
                          "evidence": [], "note": ""}],
        "steps": [
            {"id": "S1", "title": "Source the board", "phase": "SOURCE", "acceptance": "", "status": "done",
             "note": "", "owner": "model", "required": True, "covers": [], "stages": [], "stages_done": [],
             "hint_tools": [], "claimed": False, "waived_reason": "", "template": True, "evidence": []},
            {"id": "S2", "title": "Pick parameters", "phase": "PARAMS", "status": "done", "note": "", "owner": "model",
             "required": True, "claimed": True, "stages": [], "stages_done": []},
            {"id": "S3", "title": "Import the STEP file", "phase": "SOURCE", "status": "skipped",
             "note": "no STEP file: nominal board data used", "owner": "app", "required": True},
            {"id": "S4", "title": "Check <b>clearances</b>", "phase": "BUILD", "status": "blocked",
             "note": "needs a <i>decision</i>", "owner": "model", "required": True},
            {"id": "S5", "title": "Bottom shell: floor, walls, standoffs", "phase": "BUILD", "status": "running",
             "note": "", "owner": "model", "required": True, "stages": ["shell", "standoffs"],
             "stages_done": ["shell"], "hint_tools": ["build_enclosure"]},
            {"id": "S6", "title": "Lid", "phase": "BUILD", "status": "pending", "owner": "model", "required": True},
            {"id": "S7", "title": "Optional vents", "phase": "BUILD", "status": "pending", "owner": "model",
             "required": False},
            {"id": "S8", "title": "Dimensions changed by the user", "phase": "BUILD", "status": "changed",
             "note": "the user moved the port", "owner": "model", "required": True},
        ],
        "waivers": [], "round": 7, "round_cap": 30, "revision": 12, "updated": "2026-10-01T21:40:41",
    }
    base.update(over)
    return base


def started(call_id="call_a1", name="execute_code", title="Make the shell body", **over) -> dict:
    payload = {"call_id": call_id, "name": name, "title": title, "args_summary": "code=...", "round": 6,
               "ordinal": 7, "mutating": True, "geometry": True, "phase": "BUILD", "started_at": 1790000000.5}
    payload.update(over)
    return payload


def finished(call_id="call_a1", name="execute_code", title="Make the shell body", **over) -> dict:
    payload = {
        "call_id": call_id, "name": name, "title": title, "ok": True, "skipped": False,
        "text": "Created Shell (Part::Cut), 1 solid.", "dur_s": 1.84, "round": 6, "ordinal": 7,
        "code": "import Part\nbox = Part.makeBox(1, 2, 3)\n", "args_json": "",
        "delta": {"added": ["Shell"], "removed": [], "changed": ["Lid"]},
        "health": {"valid": True, "errors": 0, "warnings": 0, "line": "[health] doc 'PiCase': valid"},
        "step_index": 7, "snapshot_path": "/abs/path/snapshots/007_shell.png", "snapshot_note": "",
        "checkpoint": "007", "script_path": "/abs/scripts/007_shell.py",
    }
    payload.update(over)
    return payload


def progress(**over) -> dict:
    payload = {"round": 7, "round_cap": 30, "phase": "BUILD", "step_id": "S5",
               "step_title": "Bottom shell: floor, walls, standoffs", "steps_done": 4, "steps_total": 10,
               "elapsed_round": 12.4, "elapsed_turn": 140.2, "reasoning_chars": 1820, "text_chars": 0, "tool": "",
               "tool_chars": 0, "activity": "thinking", "retry": None, "idle_s": 1.2, "started_at": 1790000000.5}
    payload.update(over)
    return payload


BUILD_SUMMARY = """## Build Summary
**Outcome:** validated · reference status: catalog-nominal · project: /Users/me/Documents/Oscar/Projects/pi4-ab12cd
### Model tree
| Name | Type | Solids | Volume mm3 | BBox mm |
| --- | --- | --- | --- | --- |
| Shell | Part::Cut | 1 | 18432.1 | 94.0 x 65.0 x 22.0 |
### Requirements
| id | requirement | status | steps |
| --- | --- | --- | --- |
| R2 | Cutouts for every I/O port | done | S6 (8/8 openings cut) |
### Parameters
| name | value | unit | source |
| --- | --- | --- | --- |
### Provenance
- Reference: Raspberry Pi 4 - status **catalog-nominal** - source: catalog - sha256: n/a (no file)
### Validated
- health: valid (7 objects, 2 solids)
- fit: 9 PASS, 1 WARN, 0 FAIL
### NOT validated
- STEP-based collision check (the reference is nominal board data; no STEP file was imported)
- Physical fit, printing, thermal behaviour and ABS shrinkage on a real printer
### Open items
- S9 blocked: waiting for the user
**Files:** NOTES.md, rebuild.py, model.FCStd, exports/
"""

SUMMARY_HEADINGS = ("Build Summary", "Model tree", "Requirements", "Parameters", "Provenance", "Validated",
                    "NOT validated", "Open items")


def balanced(markup: str, tags=("ul", "ol", "table", "div", "pre", "p", "li")) -> bool:
    """Every opening tag of ``tags`` has its closing tag (``<p`` must not match ``<pre``)."""
    return all(len(re.findall(rf"<{t}[\s>]", markup)) == markup.count(f"</{t}>") for t in tags)


# --------------------------------------------------------------------------------------------------


class MarkdownTest(unittest.TestCase):
    def test_basics_and_escaping(self):
        out = pr.md_to_html("**bold** and *it* and `code` and <script>alert(1)</script>")
        self.assertIn("<b>bold</b>", out)
        self.assertIn("<i>it</i>", out)
        self.assertIn("<code", out)
        self.assertNotIn("<script>", out)
        self.assertIn("&lt;script&gt;", out)

    def test_empty_and_odd_input_never_raises(self):
        self.assertEqual(pr.md_to_html(""), "")
        self.assertEqual(pr.md_to_html(None), "")
        for odd in ("\ue000 0 \ue001", "```", "|", "| a |\n|---|", "1.", "- ", "**", "*", "[x](", "\x00\x01",
                    "```\n```\n```", "#" * 9, "![img](https://x)"):
            self.assertIsInstance(pr.md_to_html(odd), str)

    def test_dimension_math_is_not_italic(self):
        for text in ("2*3*4 mm", "the box is 2*3*4 mm and 5*6 mm", "a * b * c", "x*y*z", "wall*2 + floor*2"):
            self.assertNotIn("<i>", pr.md_to_html(text), text)
        self.assertIn("2*3*4 mm", pr.md_to_html("2*3*4 mm"))
        # python power is not bold either
        self.assertNotIn("<b>", pr.md_to_html("x**2 + y**2 = r**2"))
        # real emphasis still works, also next to punctuation
        self.assertIn("<i>real</i>", pr.md_to_html("this is *real*."))
        self.assertIn("<b>real</b>", pr.md_to_html("(**real**)"))
        self.assertIn("<b>Outcome:</b>", pr.md_to_html("**Outcome:** validated"))

    def test_emphasis_never_reaches_into_code(self):
        out = pr.md_to_html("use `a*b*c` and `**x**` here")
        self.assertIn("a*b*c</code>", out)
        self.assertIn("**x**</code>", out)
        self.assertNotIn("<b>x</b>", out)
        self.assertNotIn("<i>b</i>", out)

    def test_ordered_lists(self):
        out = pr.md_to_html("Plan:\n1. First\n2. Second\n3. Third")
        self.assertEqual(out.count("<ol"), 1)
        self.assertEqual(out.count("<li>"), 3)
        self.assertNotIn("<ul", out)
        self.assertNotIn(" start=", out)
        self.assertTrue(balanced(out))

    def test_ordered_list_keeps_its_numbers_across_an_interruption(self):
        out = pr.md_to_html("1. one\n- detail\n2. two\n3. three")
        self.assertIn("<ol style", out)
        self.assertIn("<ol start='2'", out)
        self.assertTrue(balanced(out))

    def test_ordered_list_paren_and_not_decimals(self):
        self.assertIn("<ol", pr.md_to_html("1) a\n2) b"))
        self.assertNotIn("<ol", pr.md_to_html("2.5 mm wall\n3.2 mm lid"))

    def test_bullets_still_work_and_lists_close(self):
        out = pr.md_to_html("- a\n* b\n\ntext after")
        self.assertEqual(out.count("<li>"), 2)
        self.assertLess(out.index("</ul>"), out.index("text after"))
        self.assertTrue(balanced(out))

    def test_fenced_block_is_protected_from_line_handling(self):
        src = "intro\n```python\n# not a heading\n- not a bullet\n1. not numbered\n| a | b |\n|---|---|\n\nlast **line**\n```\nafter"
        out = pr.md_to_html(src)
        self.assertEqual(out.count("<pre>"), 1)
        pre = out[out.index("<pre>"):out.index("</pre>")]
        for fragment in ("# not a heading", "- not a bullet", "1. not numbered", "| a | b |", "last **line**"):
            self.assertIn(fragment, pre)
        for tag in ("<li>", "<h", "<table", "<b>", "<ol"):
            self.assertNotIn(tag, out[out.index("<pre>"):out.index("</pre>")])
        self.assertIn("after", out[out.index("</pre>"):])
        self.assertIn("intro", out[:out.index("<pre>")])
        # the blank line inside the fence survives
        self.assertIn("\n\nlast", pre)

    def test_fenced_block_escapes_html_and_handles_streaming(self):
        out = pr.md_to_html("```\n<script>x</script>\nPart.makeBox(1,2,3)")   # unterminated: still streaming
        self.assertIn("<pre><code>", out)
        self.assertIn("&lt;script&gt;", out)
        self.assertNotIn("<script>", out)
        self.assertIn("Part.makeBox(1,2,3)</code></pre>", out)
        self.assertTrue(balanced(out))
        # two blocks in one message
        two = pr.md_to_html("```\na\n```\ntext\n```\nb\n```")
        self.assertEqual(two.count("<pre>"), 2)
        self.assertIn("text", two)

    def test_list_closes_before_a_fence(self):
        out = pr.md_to_html("- item\n```\ncode\n```")
        self.assertLess(out.index("</ul>"), out.index("<pre>"))
        self.assertTrue(balanced(out))

    def test_markdown_links_are_anchors(self):
        self.assertIn("<a href='https://example.com/x'>docs</a>", pr.md_to_html("see [docs](https://example.com/x) now"))
        self.assertIn("href='oscar://artifact/open-doc'", pr.md_to_html("[Open](oscar://artifact/open-doc)"))
        self.assertIn("href='file:///tmp/a.png'", pr.md_to_html("[img](file:///tmp/a.png)"))
        # a scripting scheme never becomes an anchor
        for bad in ("[x](javascript://alert(1))", "[x](data://abc)"):
            self.assertNotIn("<a ", pr.md_to_html(bad), bad)

    def test_bare_urls_are_linkified_without_the_sentence_punctuation(self):
        out = pr.md_to_html("Docs at https://example.com/a?b=1&c=2. Also (https://x.org/p), and https://y.io/z!")
        self.assertIn("<a href='https://example.com/a?b=1&amp;c=2'>https://example.com/a?b=1&amp;c=2</a>.", out)
        self.assertIn("<a href='https://x.org/p'>https://x.org/p</a>)", out)
        self.assertIn("<a href='https://y.io/z'>https://y.io/z</a>!", out)
        # a balanced parenthesis is part of the URL
        self.assertIn("href='https://en.wikipedia.org/wiki/A_(b)'", pr.md_to_html("https://en.wikipedia.org/wiki/A_(b)"))

    def test_emphasis_around_a_url_and_one_line_triple_backticks(self):
        out = pr.md_to_html("see **https://example.com/a** now")
        self.assertIn("<b><a href='https://example.com/a'>https://example.com/a</a></b>", out)
        # a line that opens AND closes ``` is inline code: it must not swallow the rest of the message
        out = pr.md_to_html("```quick``` then text\nnext line\n- item")
        self.assertNotIn("<pre>", out)
        self.assertIn("quick</code>", out)
        self.assertIn("next line", out)
        self.assertIn("<li>item</li>", out)

    def test_urls_in_code_and_quotes(self):
        self.assertNotIn("<a ", pr.md_to_html("run `curl https://example.com/x`"))
        self.assertNotIn("<a ", pr.md_to_html("```\nhttps://example.com/x\n```"))
        out = pr.md_to_html('He said "https://example.com/q" loudly')
        self.assertIn("href='https://example.com/q'", out)
        self.assertNotIn("&quot;'", out)

    def test_absolute_paths_are_linkified_and_arithmetic_is_not(self):
        out = pr.md_to_html("Saved to /Users/me/Projects/pi4/model.FCStd.")
        self.assertIn("<a href='file:///Users/me/Projects/pi4/model.FCStd'>/Users/me/Projects/pi4/model.FCStd</a>.", out)
        for plain in ("and/or", "10/12/2026", "a/b/c", "1/2/3", "mm/s/s", "w/o", "/etc"):
            self.assertNotIn("<a ", pr.md_to_html(plain), plain)
        # spaces and odd characters are percent-encoded in the href only
        self.assertIn("href='file:///tmp/a%20b.png'", pr.md_to_html("[x](file:///tmp/a%20b.png)"))
        win = pr.md_to_html(r"open C:\Users\me\box.step now")
        self.assertIn("href='file:///C:/Users/me/box.step'", win)

    def test_tables(self):
        out = pr.md_to_html("| dim | mm |\n|---|---|\n| length | **80** |\n| path | /tmp/a/b.txt |")
        self.assertIn("<table", out)
        self.assertIn("dim</th>", out)
        self.assertIn("<b>80</b></td>", out)
        self.assertIn("href='file:///tmp/a/b.txt'", out)
        self.assertNotIn("---", out)
        self.assertNotIn("<table", pr.md_to_html("a | b\nplain line"))

    def test_windows_line_endings(self):
        out = pr.md_to_html("1. one\r\n2. two\r\n\r\n```\r\n- code\r\n```\r\n| a | b |\r\n|---|---|\r\n| 1 | 2 |")
        self.assertNotIn("\r", out)
        self.assertEqual(out.count("<li>"), 2)
        self.assertIn("<pre><code>- code</code></pre>", out)
        self.assertIn("<table", out)

    def test_private_use_marks_in_the_input_cannot_forge_a_stash_slot(self):
        out = pr.md_to_html("a \ue000 0 \ue001 b `x`")
        self.assertNotIn("\ue000", out)
        self.assertIn("a  0  b", out)

    def test_headings_and_rules(self):
        out = pr.md_to_html("# one\n## two\n### three\n#### four\n---\ntext")
        self.assertIn("<h3", out)
        self.assertIn("<h4", out)
        self.assertIn("<h5", out)
        self.assertIn("<hr>", out)
        self.assertNotIn("<hr>", pr.md_to_html("- a\n- b"))


class ClipTest(unittest.TestCase):
    def test_clip_text(self):
        self.assertEqual(pr.clip_text("abc", 10), "abc")
        self.assertEqual(pr.clip_text("x" * 20, 10), "x" * 10 + "\n…")
        # a producer that already ended with an ellipsis does not get a second one
        self.assertEqual(pr.clip_text("y" * 9 + "…" + "tail", 10), "y" * 9 + "…")
        self.assertEqual(pr.clip_text("short…", 100), "short…")
        self.assertEqual(pr.clip_text(None, 5), "")
        self.assertEqual(pr.DETAIL_LIMIT, 12000)


class HeartbeatTest(unittest.TestCase):
    NOW = 1790000100.0

    def test_contract_example(self):
        text = pr.heartbeat_text(progress(), now=self.NOW)
        self.assertEqual(text, "round 7/30 · BUILD · thinking 12 s · 1.8k chars")

    def test_every_activity(self):
        cases = {
            "writing": (dict(activity="writing", text_chars=420, reasoning_chars=0), "writing 12 s · 420 chars"),
            "tool_args": (dict(activity="tool_args", tool="execute_code", tool_chars=2500, reasoning_chars=0),
                          "writing execute_code arguments 12 s · 2.5k chars"),
            "retrying": (dict(activity="retrying", retry=[2, 5], reasoning_chars=0), "retrying 2/5"),
            "waiting_approval": (dict(activity="waiting_approval", reasoning_chars=0), "waiting for your approval"),
            "paused": (dict(activity="paused", reasoning_chars=0), "paused"),
        }
        for activity, (over, tail) in cases.items():
            text = pr.heartbeat_text(progress(**over), now=self.NOW)
            self.assertTrue(text.startswith("round 7/30 · BUILD · "), text)
            self.assertTrue(text.endswith(tail), f"{activity}: {text}")

    def test_running_tool_and_validating_are_counted_by_the_panel_clock(self):
        text = pr.heartbeat_text(progress(activity="running_tool", tool="execute_code", elapsed_round=0.1,
                                          reasoning_chars=0), now=1790000000.5 + 75)
        self.assertEqual(text, "round 7/30 · BUILD · running execute_code 1 min 15 s")
        text = pr.heartbeat_text(progress(activity="validating", reasoning_chars=0), now=1790000000.5 + 9.7)
        self.assertEqual(text, "round 7/30 · BUILD · validating 9 s")
        # the clock never runs backwards
        self.assertIn("running tool 0 s", pr.heartbeat_text(progress(activity="running_tool", reasoning_chars=0), now=5.0))

    def test_silence_warning_only_for_provider_activities(self):
        self.assertIn("no data for 45 s", pr.heartbeat_text(progress(idle_s=45), now=self.NOW))
        self.assertNotIn("no data", pr.heartbeat_text(progress(idle_s=29), now=self.NOW))
        self.assertNotIn("no data", pr.heartbeat_text(progress(activity="running_tool", idle_s=99), now=self.NOW))

    def test_missing_and_malformed_payloads(self):
        for bad in (None, {}, [], "x", 5):
            self.assertEqual(pr.heartbeat_text(bad, now=self.NOW), "")
        self.assertEqual(pr.heartbeat_text({"activity": "thinking"}, now=self.NOW), "thinking 0 s")
        weird = pr.heartbeat_text({"round": "x", "round_cap": None, "elapsed_round": "nan", "activity": "mystery_mode"},
                                  now=self.NOW)
        self.assertEqual(weird, "mystery mode")
        self.assertEqual(pr.heartbeat_text({"round": 3, "activity": "paused"}, now=self.NOW), "round 3 · paused")


class BadgeTest(unittest.TestCase):
    def test_contract_labels(self):
        expect = {"validated": "✓ validated", "unverified": "⚠ unverified", "capped": "⚠ capped",
                  "needs_input": "? needs input", "failed": "✗ failed", "needs_help": "✗ needs help"}
        for outcome, label in expect.items():
            self.assertEqual(pr.outcome_badge(outcome)[0], label)
        self.assertEqual(pr.outcome_badge("answered"), ("", "ok"))
        self.assertEqual(pr.outcome_badge("validated")[1], "ok")
        self.assertEqual(pr.outcome_badge("unverified")[1], "warn")
        self.assertEqual(pr.outcome_badge("failed")[1], "error")
        self.assertEqual(pr.outcome_badge("needs_input")[1], "neutral")
        self.assertEqual(pr.outcome_badge("stopped")[1], "error")
        self.assertEqual(pr.outcome_badge("nonsense"), ("", "neutral"))
        self.assertEqual(pr.outcome_badge(None), ("", "neutral"))

    def test_every_flow_outcome_has_a_badge_entry(self):
        try:
            from oscar import flow
        except Exception:
            self.skipTest("oscar.flow unavailable")
        for outcome in flow.OUTCOMES:
            self.assertIn(outcome, pr._BADGES, outcome)

    def test_colors(self):
        self.assertNotEqual(pr.outcome_color("ok"), pr.outcome_color("error"))
        self.assertEqual(pr.outcome_color("neutral", "#123456"), "#123456")
        self.assertEqual(pr.outcome_color("zzz", "#123456"), "#123456")


class PlanStripTest(unittest.TestCase):
    def test_hidden_without_steps(self):
        for led in (None, {}, {"steps": []}, {"steps": "x"}, {"steps": [None, 3]}, [], "x"):
            self.assertEqual(pr.plan_strip_html(led), "", repr(led))

    def test_header_and_icons(self):
        out = pr.plan_strip_html(ledger())
        self.assertIn("step 5/8 · round 7/30 · BUILD", out)   # S5 is the running step: 5th of 8
        self.assertIn("href='plan:toggle'", out)
        rows = {m.group(2): m.group(1) for m in re.finditer(r"<span style='color:[^']*;'>(.)</span> <span[^>]*>(S\d)</span>", out)}
        self.assertEqual(rows, {"S1": "✓", "S2": "✓", "S3": "↷", "S4": "✗", "S5": "◌", "S6": "○", "S7": "○", "S8": "↻"})

    def test_every_step_is_listed_with_its_title(self):
        out = pr.plan_strip_html(ledger())
        for title in ("Source the board", "Pick parameters", "Import the STEP file", "Bottom shell: floor, walls, standoffs",
                      "Lid", "Optional vents", "Dimensions changed by the user"):
            self.assertIn(title, out)

    def test_current_step_is_highlighted_with_the_accent(self):
        out = pr.plan_strip_html(ledger(), accent="#ff0000")
        self.assertIn("<b style='color:#ff0000;'>Bottom shell: floor, walls, standoffs</b>", out)
        self.assertEqual(out.count("<b style="), 1)

    def test_first_pending_step_is_current_when_nothing_runs(self):
        led = ledger()
        led["steps"][4]["status"] = "pending"
        out = pr.plan_strip_html(led)
        self.assertIn("step 5/8", out)   # S4 is blocked = terminal, so the first non-terminal required step is S5
        # all terminal: the header shows the total
        for step in led["steps"]:
            step["status"] = "done"
        self.assertIn("step 8/8", pr.plan_strip_html(led))

    def test_optional_steps_do_not_become_current(self):
        led = ledger()
        for step in led["steps"]:
            step["status"] = "done"
        led["steps"][6].update(status="pending", required=False)
        self.assertIn("step 8/8", pr.plan_strip_html(led))
        self.assertIn("(optional)", pr.plan_strip_html(led))

    def test_stage_progress_claimed_notes_and_reference(self):
        out = pr.plan_strip_html(ledger())
        self.assertIn("(1/2)", out)             # S5: one of two stages done
        self.assertIn("(claimed)", out)         # S2: done by the model's word only
        self.assertIn("no STEP file: nominal board data used", out)   # skipped note
        self.assertIn("reference: catalog-nominal", out)
        self.assertNotIn("reference:", pr.plan_strip_html(ledger(reference_status="none")))
        self.assertNotIn("reference:", pr.plan_strip_html(ledger(needs_reference=False)))

    def test_titles_and_notes_are_escaped(self):
        out = pr.plan_strip_html(ledger())
        self.assertNotIn("<b>clearances</b>", out)
        self.assertIn("&lt;b&gt;clearances&lt;/b&gt;", out)
        self.assertIn("&lt;i&gt;decision&lt;/i&gt;", out)

    def test_never_collapses_by_itself_but_can_be_collapsed(self):
        long_plan = ledger(steps=[{"id": f"S{i}", "title": f"Step {i}", "phase": "BUILD",
                                   "status": "running" if i == 20 else "pending", "required": True}
                                  for i in range(1, 41)])
        out = pr.plan_strip_html(long_plan)
        self.assertEqual(out.count("Step "), 40)
        self.assertIn("▾", out)
        collapsed = pr.plan_strip_html(long_plan, collapsed=True)
        self.assertIn("▸", collapsed)
        self.assertIn("Step 20", collapsed)
        self.assertNotIn("Step 3<", collapsed)
        self.assertIn("step 20/40", collapsed)

    def test_round_cap_and_phase_are_optional(self):
        led = ledger(round=0, round_cap=0, phase="")
        out = pr.plan_strip_html(led)
        self.assertIn("Plan · step 5/8<", out)
        self.assertNotIn("round", out.split("</a>")[0])
        self.assertIn("round 4 ·", pr.plan_strip_html(ledger(round=4, round_cap=0)))

    def test_unknown_and_missing_fields(self):
        out = pr.plan_strip_html({"steps": [{"id": "S1"}, {"title": "No id", "status": "weird"}]})
        self.assertIn("S1", out)
        self.assertIn("No id", out)
        self.assertIn("?", out)

    def test_colors_are_sanitised(self):
        out = pr.plan_strip_html(ledger(), accent="red'><script>", muted="#888'x")
        self.assertNotIn("<script>", out)
        self.assertNotIn("'x", out)


class StepRowTest(unittest.TestCase):
    def view(self, **over) -> StepView:
        v = StepView()
        v.apply(started())
        v.apply(finished(**over))
        v.running = False
        return v

    def test_running_row(self):
        v = StepView(running=True)
        v.apply(started())
        out = pr.step_row_html(v, 4, 2, muted="#999999", expanded=True)
        self.assertIn("◌ Make the shell body …", out)
        self.assertIn("href='step:4:2'", out)
        self.assertNotIn("<pre>", out)   # a running row has nothing to expand yet
        self.assertNotIn("copy:", out)

    def test_finished_row_icons_and_duration(self):
        out = pr.step_row_html(self.view(), 1, 0, muted="#999999", expanded=False)
        self.assertIn("✓ Make the shell body · 1.8s", out)
        failed = pr.step_row_html(self.view(ok=False), 1, 0, muted="#999999", expanded=False)
        self.assertIn("✗ Make the shell body", failed)
        skipped = pr.step_row_html(self.view(skipped=True, ok=True), 1, 0, muted="#999999", expanded=False)
        self.assertIn("↷ Make the shell body", skipped)
        self.assertIn("skipped", skipped)
        slow = pr.step_row_html(self.view(dur_s=125.0), 1, 0, muted="#999999", expanded=False)
        self.assertIn("2m05s", slow)

    def test_title_falls_back_to_the_tool_name(self):
        v = self.view(title="")
        v.title = ""
        self.assertIn("✓ execute_code", pr.step_row_html(v, 1, 0, muted="#999999", expanded=False))

    def test_delta_chips_and_health_chip(self):
        out = pr.step_row_html(self.view(), 1, 0, muted="#999999", expanded=False)
        self.assertIn(">+Shell<", out)
        self.assertIn(">~Lid<", out)
        self.assertNotIn("⚠", out)   # healthy: no chip
        bad = pr.step_row_html(self.view(health={"valid": False, "errors": 2, "warnings": 1, "line": "x"}), 1, 0,
                               muted="#999999", expanded=False)
        self.assertIn("⚠ 2 errors", bad)
        warn = pr.step_row_html(self.view(health={"valid": True, "errors": 0, "warnings": 1, "line": "x"}), 1, 0,
                                muted="#999999", expanded=False)
        self.assertIn("⚠ 1 warning<", warn)
        many = pr.step_row_html(self.view(delta={"added": [f"O{i}" for i in range(9)], "removed": ["Old"]}), 1, 0,
                                muted="#999999", expanded=False)
        self.assertIn("+4 more", many)
        self.assertIn(">-Old<", pr.step_row_html(self.view(delta={"removed": ["Old"]}), 1, 0, muted="#999999", expanded=False))

    def test_thumbnail_is_a_clickable_resource_reference(self):
        out = pr.step_row_html(self.view(), 4, 2, muted="#999999", expanded=False)
        self.assertIn("<a href='show:4:2'><img src='es-img://step/call_a1' width='160'></a>", out)
        self.assertEqual(pr.thumb_url("call_a1"), "es-img://step/call_a1")
        self.assertEqual(pr.thumb_url("a b/c"), "es-img://step/a%20b%2Fc")
        out = pr.step_row_html(self.view(call_id="x y"), 1, 0, muted="#999999", expanded=False)
        self.assertIn("src='es-img://step/x%20y'", out)

    def test_snapshot_note_replaces_a_missing_thumbnail(self):
        out = pr.step_row_html(self.view(snapshot_path="", snapshot_note="snapshot skipped: view was blank"), 1, 0,
                               muted="#999999", expanded=False)
        self.assertNotIn("<img", out)
        self.assertIn("snapshot skipped: view was blank", out)
        self.assertNotIn("<img", pr.step_row_html(self.view(snapshot_path=""), 1, 0, muted="#999999", expanded=False))

    def test_collapsed_row_hides_the_detail(self):
        out = pr.step_row_html(self.view(), 1, 0, muted="#999999", expanded=False)
        self.assertNotIn("Created Shell", out)
        self.assertNotIn("<pre>", out)

    def test_expanded_row_shows_code_in_pre_with_a_copy_anchor(self):
        out = pr.step_row_html(self.view(), 4, 2, muted="#999999", expanded=True)
        self.assertIn("<pre>import Part\nbox = Part.makeBox(1, 2, 3)", out)
        self.assertIn("<a href='copy:4:2'>copy</a>", out)
        self.assertIn("Created Shell (Part::Cut), 1 solid.", out)
        self.assertIn("added: Shell", out)
        self.assertIn("changed: Lid", out)
        self.assertIn("[health] doc &#x27;PiCase&#x27;: valid", out)

    def test_expanded_row_without_code_shows_the_arguments(self):
        out = pr.step_row_html(self.view(code="", args_json='{"name": "Box"}'), 1, 0, muted="#999999", expanded=True)
        self.assertNotIn("copy:", out)
        self.assertIn("arguments:", out)
        self.assertIn("{&quot;name&quot;: &quot;Box&quot;}", out)

    def test_full_output_up_to_the_limit_then_one_ellipsis(self):
        long_text = "line\n" * 2500   # 12500 chars
        out = pr.step_row_html(self.view(text=long_text), 1, 0, muted="#999999", expanded=True)
        self.assertEqual(out.count("line\n"), 2400)   # 12000 chars shown, nothing of it lost before the limit
        self.assertEqual(out.count("…"), 1)
        mid = pr.step_row_html(self.view(text="x" * 9000), 1, 0, muted="#999999", expanded=True)
        self.assertIn("x" * 9000, mid)
        self.assertNotIn("…", mid)

    def test_no_output_hint(self):
        out = pr.step_row_html(self.view(text="", code=""), 1, 0, muted="#999999", expanded=True)
        self.assertIn("(no output)", out)

    def test_payload_text_is_escaped_everywhere(self):
        evil = "<script>alert(1)</script>"
        out = pr.step_row_html(self.view(title=evil, text=evil, code=evil, snapshot_path="", snapshot_note=evil,
                                         delta={"added": [evil]}, health={"errors": 1, "line": evil}),
                               1, 0, muted="#999999", expanded=True)
        self.assertNotIn("<script>", out)
        self.assertGreaterEqual(out.count("&lt;script&gt;"), 5)

    def test_ids_are_coerced(self):
        out = pr.step_row_html(self.view(), "7", "3", muted="#999999", expanded=False)
        self.assertIn("step:7:3", out)
        self.assertIn("step:0:0", pr.step_row_html(self.view(), None, "x", muted="#999999", expanded=False))


class StepViewTest(unittest.TestCase):
    def test_apply_coerces_and_ignores_unknown_keys(self):
        v = StepView()
        v.apply({"call_id": 5, "round": "6", "dur_s": "1.5", "ok": 0, "mystery": 1, "delta": {"added": ["A"], "bad": 3},
                 "health": "nope", "step_index": None})
        self.assertEqual((v.call_id, v.round, v.dur_s, v.ok, v.step_index), ("5", 6, 1.5, False, -1))
        self.assertEqual(v.delta, {"added": ["A"]})
        self.assertIsNone(v.health)
        self.assertFalse(hasattr(v, "mystery"))

    def test_properties(self):
        v = StepView(name="t", running=True)
        self.assertEqual((v.icon, v.failed, v.default_expanded), ("◌", False, False))
        v.running, v.ok = False, False
        self.assertEqual((v.icon, v.failed, v.default_expanded), ("✗", True, True))
        v.skipped = True
        self.assertEqual((v.icon, v.failed, v.default_expanded), ("↷", False, False))
        self.assertEqual(StepView(name="t").label, "t")
        self.assertEqual(StepView().label, "tool")
        v.apply({"delta": {"added": ["A"], "changed": ["B"], "removed": ["C"]}})
        self.assertEqual(v.changed_objects(), ["A", "B"])


class TurnViewTest(unittest.TestCase):
    def make(self):
        self.t = [1000.0]
        return TurnView(clock=lambda: self.t[0])

    def test_live_header_is_never_done(self):
        tv = self.make()
        self.assertEqual(tv.header(), "✻ Working… 0s")
        tv.on_step_started(json.dumps(started()))
        self.t[0] = 1012.4
        self.assertEqual(tv.header(), "✻ Working… 12s")
        tv.on_step_finished(json.dumps(finished()))
        # between rounds: nothing running, the worker still works
        self.assertEqual(tv.header(), "✻ Working… 12s")
        self.assertNotIn("Done", tv.header())
        self.assertEqual(tv.header_kind, "live")
        self.assertEqual(tv.header(now=1020.0), "✻ Working… 20s")

    def test_step_lifecycle_from_documented_payloads(self):
        tv = self.make()
        view = tv.on_step_started(json.dumps(started()))
        self.assertTrue(view.running)
        self.assertEqual(view.title, "Make the shell body")
        self.assertEqual(tv.running_steps, [view])
        same = tv.on_step_finished(json.dumps(finished()))
        self.assertIs(same, view)
        self.assertFalse(view.running)
        self.assertEqual((view.ok, view.dur_s, view.step_index), (True, 1.84, 7))
        self.assertEqual(view.delta["added"], ["Shell"])
        self.assertEqual(view.snapshot_path, "/abs/path/snapshots/007_shell.png")
        self.assertEqual(len(tv.steps), 1)
        self.assertEqual(tv.running_steps, [])

    def test_never_two_rows_for_one_call(self):
        tv = self.make()
        tv.on_step_started(started())
        tv.on_step_started(started(title="renamed"))   # a repeated start updates the same row
        self.assertEqual(len(tv.steps), 1)
        self.assertEqual(tv.steps[0].title, "renamed")
        tv.on_step_finished(finished())
        tv.on_step_finished(finished(text="again"))     # a repeated finish too
        tv.on_step_started(started())                   # a late start does not revive a finished row
        self.assertEqual(len(tv.steps), 1)
        self.assertFalse(tv.steps[0].running)
        self.assertEqual(tv.steps[0].text, "again")

    def test_finish_without_start_and_name_fallback(self):
        tv = self.make()
        tv.on_step_finished(finished(call_id="orphan"))
        self.assertEqual(len(tv.steps), 1)
        self.assertFalse(tv.steps[0].running)
        # an empty call_id completes the last running row of the same tool
        tv.on_step_started(started(call_id="c2", name="create_object"))
        tv.on_step_started(started(call_id="c3", name="execute_code"))
        tv.on_step_finished(finished(call_id="", name="execute_code"))
        self.assertTrue(tv.steps[1].running)
        self.assertFalse(tv.steps[2].running)
        # a finish without "ok" counts as ok
        tv.on_step_finished({"call_id": "c2", "name": "create_object", "text": "done"})
        self.assertTrue(tv.steps[1].ok)

    def test_parallel_calls_keep_their_own_rows(self):
        tv = self.make()
        tv.on_step_started(started(call_id="a", title="A"))
        tv.on_step_started(started(call_id="b", title="B"))
        tv.on_step_finished(finished(call_id="b", title="B", ok=False, text="boom"))
        tv.on_step_finished(finished(call_id="a", title="A"))
        self.assertEqual([(s.title, s.ok) for s in tv.steps], [("A", True), ("B", False)])

    def test_malformed_payloads_are_ignored(self):
        tv = self.make()
        for bad in (None, "", "not json", "[1, 2]", "5", b"\xff", 7, [1]):
            self.assertIsNone(tv.on_step_started(bad))
            self.assertIsNone(tv.on_step_finished(bad))
            self.assertFalse(tv.on_plan(bad))
            self.assertFalse(tv.on_progress(bad))
            self.assertFalse(tv.on_artifacts(bad))
        self.assertEqual(tv.steps, [])
        self.assertIsNone(tv.plan)
        # a dict and a bytes payload are accepted
        self.assertTrue(tv.on_plan(ledger()))
        self.assertTrue(tv.on_progress(json.dumps(progress()).encode()))

    def test_failed_step_is_expanded_and_shows_the_full_output(self):
        tv = self.make()
        tv.on_step_started(started())
        long_error = "Traceback (most recent call last):\n" + "  frame\n" * 400 + "ValueError: wall too thin"
        tv.on_step_finished(finished(ok=False, text=long_error))
        view = tv.steps[0]
        self.assertTrue(tv.is_expanded(view))
        rows = tv.rows_html(3)
        self.assertIn("ValueError: wall too thin", rows)
        self.assertEqual(rows.count("  frame\n"), 400)
        self.assertIn("✗ Make the shell body", rows)
        # a succeeded row is collapsed by default
        tv.on_step_started(started(call_id="ok1", title="Fine"))
        tv.on_step_finished(finished(call_id="ok1", title="Fine", text="all good"))
        self.assertNotIn("all good", tv.rows_html(3))

    def test_toggle_flips_the_default(self):
        tv = self.make()
        tv.on_step_started(started(call_id="a"))
        tv.on_step_finished(finished(call_id="a", text="body-a"))
        tv.on_step_started(started(call_id="b"))
        tv.on_step_finished(finished(call_id="b", ok=False, text="body-b"))
        a, b = tv.steps
        self.assertEqual((tv.is_expanded(a), tv.is_expanded(b)), (False, True))
        tv.toggle(tv.row_key(a))
        tv.toggle(tv.row_key(b))
        self.assertEqual((tv.is_expanded(a), tv.is_expanded(b)), (True, False))
        self.assertIn("body-a", tv.rows_html(1))
        self.assertNotIn("body-b", tv.rows_html(1))
        tv.toggle("a")
        self.assertFalse(tv.is_expanded(a))
        # rows without a call_id are keyed by position
        tv2 = self.make()
        tv2.on_step_finished({"name": "x", "ok": True})
        self.assertEqual(tv2.row_key(tv2.steps[0]), "#0")

    def test_legacy_header_after_the_outcome(self):
        tv = self.make()
        tv.on_step_started(started(call_id="a"))
        tv.on_step_finished(finished(call_id="a"))
        tv.on_step_started(started(call_id="b"))
        tv.on_step_finished(finished(call_id="b"))
        self.t[0] = 1014.0
        tv.on_outcome("answered", "")
        self.assertEqual(tv.header(), "✻ Ran 2 steps · 14.0s")
        self.t[0] = 5000.0   # frozen at the end
        self.assertEqual(tv.header(), "✻ Ran 2 steps · 14.0s")
        self.assertEqual(tv.header_kind, "ok")

    def test_header_with_thoughts_and_badges(self):
        tv = self.make()
        tv.on_step_started(started(call_id="a"))
        tv.on_step_finished(finished(call_id="a"))
        tv.on_step_started(started(call_id="b"))
        tv.on_step_finished(finished(call_id="b"))
        tv.add_thought(0.8)
        tv.add_thought(1.4)
        self.t[0] = 1014.0
        tv.on_outcome("validated", BUILD_SUMMARY)
        self.assertEqual(tv.header(), "✻ Thought 2.2s · Ran 2 steps · 14.0s · ✓ validated")
        self.assertEqual(tv.header_kind, "ok")
        self.assertTrue(tv.header().index("Ran 2 steps") < tv.header().index("validated"))   # badge AFTER the legacy text

    def test_header_badge_for_every_outcome(self):
        expect = {"validated": " · ✓ validated", "unverified": " · ⚠ unverified", "capped": " · ⚠ capped",
                  "needs_input": " · ? needs input", "failed": " · ✗ failed", "needs_help": " · ✗ needs help",
                  "answered": ""}
        for outcome, suffix in expect.items():
            tv = self.make()
            tv.on_step_started(started())
            tv.on_step_finished(finished())
            tv.on_outcome(outcome)
            self.assertEqual(tv.header(), "✻ Ran 1 step · 0.0s" + suffix, outcome)

    def test_header_variants(self):
        tv = self.make()
        tv.on_outcome("answered")
        self.assertEqual(tv.header(), "✻ Done")        # nothing ran
        tv = self.make()
        tv.add_thought(2.2)
        tv.on_outcome("answered")
        self.assertEqual(tv.header(), "✻ Thought 2.2s")
        tv = self.make()
        tv.on_step_started(started())
        tv.on_step_finished(finished(ok=False))
        tv.on_outcome("failed")
        self.assertEqual(tv.header(), "✻ Ran 1 step · 0.0s · ⚠ failed steps · ✗ failed")
        self.assertEqual(tv.header_kind, "error")

    def test_stop_closes_running_rows(self):
        tv = self.make()
        tv.on_step_started(started())
        tv.on_outcome("stopped", "")
        self.assertEqual(tv.running_steps, [])
        self.assertFalse(tv.steps[0].ok)
        self.assertIn("Interrupted", tv.steps[0].text)
        self.assertNotIn("Working", tv.header())
        self.assertTrue(tv.header().endswith("■ stopped"))
        tv.on_outcome("stopped", "")   # idempotent
        self.assertEqual(len(tv.steps), 1)

    def test_finish_without_outcome(self):
        tv = self.make()
        tv.on_step_started(started())
        tv.on_step_finished(finished(ok=False))
        tv.finish()
        self.assertTrue(tv.finished)
        self.assertEqual(tv.header_kind, "warn")
        self.assertNotIn("Working", tv.header())

    def test_round_reset_clears_the_live_text(self):
        tv = self.make()
        tv.on_text("partial reply ")
        tv.on_text("that will be void")
        self.assertEqual(tv.live_text, "partial reply that will be void")
        tv.on_round_reset("retry")
        self.assertEqual(tv.live_text, "")
        self.assertEqual(tv.resets, ["retry"])
        for _ in range(60):
            tv.on_round_reset("gate")
        self.assertEqual(len(tv.resets), 50)
        tv.on_text("x" * 300000)
        self.assertEqual(len(tv.live_text), 200000)

    def test_progress_and_heartbeat(self):
        tv = self.make()
        self.assertEqual(tv.heartbeat(), "")
        tv.on_progress(json.dumps(progress()))
        self.assertEqual(tv.heartbeat(now=1790000100.0), "round 7/30 · BUILD · thinking 12 s · 1.8k chars")
        tv.on_progress(progress(activity="running_tool", tool="execute_code", reasoning_chars=0))
        self.assertEqual(tv.heartbeat(now=1790000000.5 + 30), "round 7/30 · BUILD · running execute_code 30 s")
        tv.on_outcome("answered")
        self.assertEqual(tv.heartbeat(), "")   # nothing left to beat once the turn is over

    def test_plan_artifacts_and_summary_renderers(self):
        tv = self.make()
        self.assertEqual(tv.plan_html(), "")
        self.assertEqual(tv.artifacts_html(), "")
        self.assertEqual(tv.summary_html(), "")
        tv.on_plan(json.dumps(ledger()))
        tv.on_artifacts(json.dumps({"project_dir": "/p", "files": [{"kind": "notes", "name": "NOTES.md", "path": "/p/NOTES.md"}]}))
        tv.on_outcome("validated", BUILD_SUMMARY)
        self.assertIn("step 5/8", tv.plan_html())
        self.assertIn("oscar://artifact/open-notes", tv.artifacts_html())
        self.assertIn("Model tree", tv.summary_html())
        self.assertEqual(tv.outcome, "validated")
        self.assertEqual(tv.summary, BUILD_SUMMARY)

    def test_elapsed_uses_the_injected_clock_and_started_at(self):
        tv = TurnView(clock=lambda: 50.0, started_at=20.0)
        self.assertEqual(tv.elapsed(), 30.0)
        self.assertEqual(tv.elapsed(now=15.0), 0.0)   # never negative

    def test_add_thought_ignores_garbage(self):
        tv = self.make()
        tv.add_thought(-3)
        tv.add_thought("x")
        self.assertEqual(tv.thought_s, 0.0)
        self.assertTrue(tv.has_thoughts)


class SummaryCardTest(unittest.TestCase):
    def test_card_contains_every_heading_and_the_tables(self):
        out = pr.summary_card_html(BUILD_SUMMARY)
        for heading in SUMMARY_HEADINGS:
            self.assertIn(heading, out)
        self.assertEqual(out.count("<table"), 3)
        self.assertIn("reference status: catalog-nominal", out)
        self.assertIn("<b>catalog-nominal</b>", out)
        self.assertIn("href='file:///Users/me/Documents/Oscar/Projects/pi4-ab12cd'", out)
        self.assertIn("STEP-based collision check", out)
        self.assertTrue(out.startswith("<div style='border:1px solid"))
        self.assertTrue(balanced(out))

    def test_blank_and_clipped(self):
        for blank in ("", "   \n", None):
            self.assertEqual(pr.summary_card_html(blank), "")
        huge = "## Build Summary\n" + "- row\n" * 10000
        self.assertLess(pr.summary_card_html(huge).count("<li>"), 4000)

    def test_summary_is_escaped(self):
        self.assertNotIn("<script>", pr.summary_card_html("## Build Summary\n- <script>x</script>"))

    def test_split_build_summary(self):
        answer = "I built the case.\n\nIt has four standoffs."
        head, summary = pr.split_build_summary(answer + "\n\n" + BUILD_SUMMARY)
        self.assertEqual(head, answer)
        self.assertTrue(summary.startswith("## Build Summary"))
        self.assertTrue(summary.endswith("exports/"))
        # the summary is rendered once: head + card contain it exactly once
        combined = pr.md_to_html(head) + pr.summary_card_html(summary)
        self.assertEqual(combined.count("Build Summary"), 1)

    def test_split_without_a_summary_or_with_only_a_summary(self):
        self.assertEqual(pr.split_build_summary("just text"), ("just text", ""))
        self.assertEqual(pr.split_build_summary(""), ("", ""))
        self.assertEqual(pr.split_build_summary(None), ("", ""))
        head, summary = pr.split_build_summary(BUILD_SUMMARY)
        self.assertEqual(head, "")
        self.assertTrue(summary.startswith("## Build Summary"))
        # only a whole line equal to the heading cuts
        for inline in ("see ## Build Summary here", "### Build Summary", "## Build Summary of the work"):
            self.assertEqual(pr.split_build_summary(inline), (inline, ""), inline)
        # the NOT VERIFIED banner stays with the answer
        head, _ = pr.split_build_summary("NOT VERIFIED: skipped checks.\nText\n\n## Build Summary\nx")
        self.assertEqual(head, "NOT VERIFIED: skipped checks.\nText")


class ArtifactsStripTest(unittest.TestCase):
    PAYLOAD = {"project_dir": "/abs/project", "files": [
        {"kind": "snapshot", "name": "001_a.png", "path": "/abs/project/snapshots/001_a.png"},
        {"kind": "fcstd", "name": "docs/Lid.FCStd", "path": "/abs/project/docs/Lid.FCStd"},
        {"kind": "fcstd", "name": "model.FCStd", "path": "/abs/project/model.FCStd"},
        {"kind": "fcstd", "name": "model.20261001.FCBak", "path": "/abs/project/model.20261001.FCBak"},
        {"kind": "notes", "name": "NOTES.md", "path": "/abs/project/NOTES.md"},
        {"kind": "rebuild", "name": "rebuild.py", "path": "/abs/project/rebuild.py"},
        {"kind": "export", "name": "exports/shell.step", "path": "/abs/project/exports/shell.step"},
        {"kind": "export", "name": "exports/lid.step", "path": "/abs/project/exports/lid.step"},
        {"kind": "provenance", "name": "provenance.json", "path": "/abs/project/provenance.json"},
    ]}

    def test_chips_and_anchors(self):
        out = pr.artifacts_strip_html(self.PAYLOAD)
        for anchor in ("oscar://artifact/open-doc'", "oscar://artifact/open-doc/docs%2FLid.FCStd'", "oscar://artifact/open-notes'",
                       "oscar://artifact/open-rebuild'", "oscar://artifact/open-exports'",
                       "oscar://artifact/open-provenance'", "oscar://artifact/open-dir'"):
            self.assertIn(anchor, out)
        self.assertIn("exports (2)", out)
        self.assertIn("model.FCStd", out)
        self.assertNotIn("FCBak", out)        # backups are not artifacts
        self.assertNotIn("001_a.png", out)    # snapshots are thumbnails, not chips
        self.assertLess(out.index("model.FCStd"), out.index("Lid.FCStd"))   # the primary document first

    def test_single_export_keeps_its_name(self):
        out = pr.artifacts_strip_html({"files": [{"kind": "export", "name": "exports/a.step"}]})
        self.assertIn("exports/a.step", out)
        self.assertNotIn("open-dir", out)

    def test_empty_and_malformed(self):
        for empty in (None, {}, {"files": []}, {"files": [None, 3]}, "bad", "[]", {"files": "x"}):
            self.assertEqual(pr.artifacts_strip_html(empty), "", repr(empty))
        self.assertIn("open-dir", pr.artifacts_strip_html({"project_dir": "/p"}))
        self.assertIn("open-dir", pr.artifacts_strip_html(json.dumps({"project_dir": "/p"})))

    def test_names_are_escaped(self):
        out = pr.artifacts_strip_html({"files": [{"kind": "notes", "name": "<b>x</b>.md"}]})
        self.assertNotIn("<b>x", out)
        self.assertIn("&lt;b&gt;", out)

    def test_parse_artifact_anchor(self):
        self.assertEqual(pr.parse_artifact_anchor("oscar://artifact/open-doc"), ("open-doc", ""))
        self.assertEqual(pr.parse_artifact_anchor("oscar://artifact/open-doc/docs%2FLid.FCStd"), ("open-doc", "docs/Lid.FCStd"))
        self.assertEqual(pr.parse_artifact_anchor("oscar://artifact/"), ("", ""))
        self.assertEqual(pr.parse_artifact_anchor("oscar://other/open-doc/x"), ("", ""))
        self.assertEqual(pr.parse_artifact_anchor(None), ("", ""))


class AnchorHelpersTest(unittest.TestCase):
    def test_parse_anchor(self):
        self.assertEqual(pr.parse_anchor("show:12:3"), ("show", ["12", "3"]))
        self.assertEqual(pr.parse_anchor("tool:5"), ("tool", ["5"]))
        self.assertEqual(pr.parse_anchor("plan:toggle"), ("plan", ["toggle"]))
        self.assertEqual(pr.parse_anchor("continue:"), ("continue", [""]))
        self.assertEqual(pr.parse_anchor("plain"), ("plain", []))
        self.assertEqual(pr.parse_anchor("oscar://artifact/open-doc"), ("oscar", ["artifact/open-doc"]))
        self.assertEqual(pr.parse_anchor(None), ("", []))


    def test_classify_file_link_never_opens_executables(self):
        for ok in ("/p/a.png", "/p/model.FCStd", "/p/a.STEP", "C:\\p\\box.stl", "/p/notes.md", "/p/x.json"):
            self.assertEqual(pr.classify_file_link(ok), "open", ok)
        for risky in ("/p/rebuild.py", "/p/run.sh", "/p/Tool.app", "/p/x.command", "/p/a.exe", "/p/noext", "/p/.bashrc",
                      "/p/archive.zip", "", None):
            self.assertEqual(pr.classify_file_link(risky), "reveal", repr(risky))


class ModuleContractTest(unittest.TestCase):
    def test_public_names(self):
        for name in ("md_to_html", "plan_strip_html", "step_row_html", "heartbeat_text", "summary_card_html",
                     "artifacts_strip_html", "outcome_badge", "StepView", "TurnView"):
            self.assertTrue(hasattr(pr, name), name)
            self.assertIn(name, pr.__all__)

    def test_signatures_follow_the_contract(self):
        import inspect

        sig = inspect.signature(pr.plan_strip_html)
        self.assertEqual(sig.parameters["accent"].default, "#1e6ee8")
        self.assertEqual(sig.parameters["muted"].default, "#888888")
        self.assertEqual(list(inspect.signature(pr.step_row_html).parameters)[:3], ["view", "bid", "index"])
        self.assertTrue(inspect.signature(pr.step_row_html).parameters["expanded"].kind is inspect.Parameter.KEYWORD_ONLY)
        self.assertTrue(inspect.signature(pr.heartbeat_text).parameters["now"].kind is inspect.Parameter.KEYWORD_ONLY)

    def test_module_is_pure(self):
        src = open(pr.__file__, encoding="utf-8").read()
        for forbidden in ("import FreeCAD", "PySide", "from oscar", "from ."):
            self.assertNotIn(forbidden, src)


class AccountRenderTest(unittest.TestCase):
    """The OriginShape account's badge, menu and cards: plain data in, text and escaped HTML out."""

    @staticmethod
    def state(**snapshot) -> dict:
        snap = {"status": "approved", "email": "ada@example.com", "plan_name": "Trial", "credits_available": 42.5,
                "credits_low": False}
        snap.update(snapshot)
        return {"known": True, "signed_in": True, "kind": "oauth", "email": "ada@example.com", "snapshot": snap,
                "error": "", "error_code": ""}

    def test_credits_are_written_for_people(self):
        for value, text in ((42.5, "42.5 credits"), (1, "1 credit"), (0, "0 credits"), (-2.5, "0 credits"),
                            (1250, "1,250 credits"), (0.01, "0.01 credits"), ("x", "0 credits"),
                            (float("nan"), "0 credits"), (None, "0 credits")):
            self.assertEqual(pr.format_credits(value), text, value)

    def test_the_badge_shows_the_balance_like_the_website(self):
        """Credits a running request holds are not "out of credits" (audit 2026-10-28: FreeCAD said 0 while the
        website said 12)."""
        text, tone, tip = pr.account_badge(self.state(credits_balance=12, credits_available=0))
        self.assertEqual((text, tone), ("12 credits", ""))
        self.assertIn("12 credits held by a running request", tip)
        self.assertEqual(pr.account_badge(self.state(credits_balance=0, credits_available=0))[:2], ("0 credits", "warn"))
        self.assertEqual(pr.account_balance({"credits_balance": 10, "credits_available": 7.5}), (10.0, 2.5))
        self.assertEqual(pr.account_balance({"credits_available": 4}), (4.0, 0.0))

    def test_a_trial_account_is_offered_a_plan_not_credit_packs(self):
        """Credit packs are for subscribers (billing/routes.py), so a trial account never sees "Buy credits"."""
        trial = self.state(plan="trial")
        actions = [action for action, _label in pr.account_menu(trial)]
        self.assertIn("subscribe", actions)
        self.assertNotIn("buy", actions)
        self.assertIn(("subscribe", "Choose a plan…"), pr.account_menu(trial))
        self.assertIn(("buy", "Buy credits…"), pr.account_menu(self.state(plan="pro")))
        text, links = pr.account_card("no_credits", trial=True)
        self.assertIn("Choose a plan", text)
        self.assertEqual([url for url, _label in links], ["oscar://account/subscribe", "oscar://settings"])
        self.assertIn("oscar://account/buy", [url for url, _label in pr.account_card("no_credits")[1]])

    def test_the_badge_follows_the_account(self):
        self.assertEqual(pr.account_badge(None)[:2], ("…", ""))
        self.assertEqual(pr.account_badge({"known": False})[:2], ("…", ""))
        self.assertEqual(pr.account_badge({"known": True, "signed_in": False})[:2], ("Sign in", ""))
        text, tone, tip = pr.account_badge(self.state())
        self.assertEqual((text, tone), ("42.5 credits", ""))
        self.assertIn("ada@example.com", tip)
        self.assertIn("Trial plan", tip)
        self.assertEqual(pr.account_badge(self.state(credits_available=0))[:2], ("0 credits", "warn"))
        self.assertEqual(pr.account_badge(self.state(credits_available=-3))[:2], ("0 credits", "warn"))
        self.assertIn("running low", pr.account_badge(self.state(credits_available=2, credits_low=True))[2])
        self.assertEqual(pr.account_badge(self.state(status="waitlisted"))[:2], ("Waitlist", ""))
        self.assertEqual(pr.account_badge(self.state(status="suspended"))[:2], ("Suspended", "warn"))
        self.assertEqual(pr.account_badge({**self.state(), "snapshot": None})[:2], ("Signed in", ""))
        refused = {"known": True, "signed_in": False, "error": "The OriginShape server URL must use https://.",
                   "error_code": "insecure_url"}
        self.assertEqual(pr.account_badge(refused), ("Sign in", "warn", "The OriginShape server URL must use https://."))
        failed = {**self.state(), "error": "Could not reach the server.", "error_code": "unavailable"}
        self.assertIn("last check failed: Could not reach the server.", pr.account_badge(failed)[2])
        self.assertEqual(pr.account_badge({"known": True, "signed_in": True, "snapshot": "junk"})[0], "Signed in")

    def test_the_menu_follows_the_account(self):
        actions = lambda state: [a for a, _label in pr.account_menu(state) if a not in ("", "-")]   # noqa: E731
        self.assertEqual(actions(None), ["refresh"])
        self.assertEqual(actions({"known": True, "signed_in": False}), ["signin", "device"])
        self.assertEqual(actions(self.state()), ["buy", "manage", "refresh", "signout"])
        rows = pr.account_menu(self.state(status="waitlisted"))
        self.assertEqual(rows[0], ("", "Signed in as ada@example.com"))
        self.assertIn(("", "Your account is on the waitlist"), rows)

    def test_an_oscar_the_server_no_longer_supports_says_update(self):
        state = {**self.state(), "update": "required", "latest_version": "0.4.0", "min_version": "0.3.0",
                 "oscar_version": "0.2.0"}
        text, tone, tooltip = pr.account_badge(state)
        self.assertEqual((text, tone), ("Update Oscar", "warn"))
        self.assertIn("this Oscar (0.2.0) is older than the server supports: update to 0.3.0 or newer", tooltip)
        self.assertIn(("download", "Update Oscar to 0.4.0…"), pr.account_menu(state))
        markdown, actions = pr.account_card("outdated")
        self.assertIn("too old", markdown)
        self.assertEqual(actions[0], ("oscar://account/download", "Get the update"))
        self.assertIn("href='oscar://account/download'", pr.account_card_html(markdown, actions, kind="outdated"))
        self.assertIn("#d9902b", pr.account_card_html(markdown, actions, kind="outdated"))

    def test_a_newer_oscar_is_mentioned_without_getting_in_the_way(self):
        state = {**self.state(), "update": "available", "latest_version": "0.3.0", "min_version": "0.2.0",
                 "oscar_version": "0.2.0"}
        text, tone, tooltip = pr.account_badge(state)
        self.assertEqual((text, tone), ("42.5 credits", ""))
        self.assertIn("Oscar 0.3.0 is available (this is 0.2.0)", tooltip)
        self.assertIn(("download", "Update Oscar to 0.3.0…"), pr.account_menu(state))
        for quiet in (self.state(), {**self.state(), "update": ""}):
            self.assertNotIn("download", [action for action, _label in pr.account_menu(quiet)])
            self.assertNotIn("available", pr.account_badge(quiet)[2])
        # Signed out (the public update check): the menu offers the update next to the sign-ins.
        signed_out = {"known": True, "signed_in": False, "update": "available", "latest_version": "0.3.0"}
        self.assertEqual([action for action, _label in pr.account_menu(signed_out)], ["signin", "device", "download"])

    def test_every_card_has_text_and_only_the_panels_own_links(self):
        for kind in pr.ACCOUNT_CARD_KINDS:
            text, actions = pr.account_card(kind)
            self.assertTrue(text.strip(), kind)
            self.assertTrue(actions, kind)
            self.assertTrue(all(target in pr.ACCOUNT_TARGETS for target, _label in actions), kind)
            self.assertNotIn("back in the input", text)
        for kind in ("signed_out", "waitlisted", "no_credits"):
            self.assertIn("back in the input", pr.account_card(kind, returned=True)[0])
        self.assertIn("no API key", pr.account_card("onboarding", reason="no_key")[0])
        self.assertEqual(pr.account_card("nonsense"), ("", []))
        targets = [t for t, _label in pr.account_card("no_credits")[1]]
        self.assertEqual(targets, ["oscar://account/buy", "oscar://account/subscribe", "oscar://settings"])

    def test_the_card_escapes_text_and_drops_foreign_links(self):
        markup = pr.account_card_html("**Hi** <script>alert(1)</script>", [
            ("oscar://account/buy", "<b>Buy</b>"), ("https://evil.example/x", "Evil"),
            ("oscar://account/buy' onclick='x", "Quote"), ("javascript:alert(1)", "Script"), ("oscar://settings", "S")],
            kind="no_credits")
        self.assertEqual(markup.count("<a "), 2)
        self.assertIn("href='oscar://account/buy'", markup)
        self.assertIn("href='oscar://settings'", markup)
        self.assertIn("&lt;b&gt;Buy&lt;/b&gt;", markup)
        for bad in ("evil.example", "javascript", "onclick", "<script>"):
            self.assertNotIn(bad, markup)
        self.assertIn("#d9902b", markup)                                   # a warning-coloured border
        self.assertIn("OriginShape account", markup)
        self.assertNotIn("#d9902b", pr.account_card_html(*pr.account_card("onboarding"), kind="onboarding"))
        self.assertEqual(pr.account_card_html("", []), "")
        self.assertEqual(pr.account_card_html("  ", [("https://evil.example", "x")]), "")
        self.assertNotIn("keeps going from the plan", pr.account_card_html(*pr.account_card("waitlisted")))

    def test_which_account_problems_have_a_card(self):
        card = pr.account_problem_card
        self.assertEqual(card({"kind": "account", "code": "insufficient_credits"}), "no_credits")
        self.assertEqual(card(json.dumps({"kind": "account", "code": "waitlisted"})), "waitlisted")
        self.assertEqual(card({"kind": "auth", "code": "signed_out"}), "signed_out")
        for other in ({"kind": "account", "code": "rate_limited"}, {"kind": "account", "code": "suspended"},
                      {"kind": "http", "code": "llm_not_configured"}, {"kind": "http", "code": "insufficient_credits"},
                      {"kind": "auth", "code": "insecure_url"}, "not json", None, 42, []):
            self.assertEqual(card(other), "", other)


# --------------------------------------------------------------------------------------------------
# the real producer: replay the signals of a worker turn into a TurnView


class WorkerPayloadReplayTest(unittest.TestCase):
    """TurnView must understand what ``agent.AgentWorker`` really emits (not only the hand-written fixtures)."""

    def replay(self, prompt, script):
        try:
            from unittest import mock

            from oscar import agent, params
            from oscar.providers import Message
            from support.scripted_llm import ScriptedProvider
            from support.signal_recorder import SignalRecorder
        except Exception as exc:   # plain interpreter without FreeCAD
            self.skipTest(f"agent unavailable: {exc}")
        history = [Message(role="user", content=prompt)]
        profile = params.Profile(name="t", provider="openai_compat", model="scripted", base_url="http://127.0.0.1:9/v1")
        worker = agent.AgentWorker(history, provider=ScriptedProvider(script), profile=profile)
        rec = SignalRecorder(worker)
        with mock.patch.object(agent._tools, "run_tool", lambda name, args: (f"{name} ok", None, False)):
            result = worker.run_turn()
        tv = TurnView(clock=lambda: 1790000100.0, started_at=1790000090.0)
        for name, args in rec.events:
            if name == "plan_changed":
                tv.on_plan(args[0])
            elif name == "step_started":
                tv.on_step_started(args[0])
            elif name == "step_finished":
                tv.on_step_finished(args[0])
            elif name == "progress":
                tv.on_progress(args[0])
            elif name == "round_reset":
                tv.on_round_reset(args[0])
            elif name == "artifacts_changed":
                tv.on_artifacts(args[0])
            elif name == "turn_outcome":
                tv.on_outcome(*args)
        return tv, rec, result

    def test_a_scripted_turn_builds_the_expected_view(self):
        from support.scripted_llm import call, calls, say

        tv, rec, result = self.replay(
            "make a plate",
            [calls(call("create_object", obj_type="Part::Box", step_title="Make the plate"),
                   call("execute_code", code="print(1)")), say("Done.")])
        self.assertEqual([v.name for v in tv.steps], ["create_object", "execute_code"])
        first, second = tv.steps
        self.assertEqual(first.title, "Make the plate")
        self.assertTrue(first.ok and not first.running)
        self.assertEqual(second.code, "print(1)")
        self.assertEqual([v.ordinal for v in tv.steps], [1, 2])
        self.assertTrue(tv.finished)
        self.assertEqual(tv.outcome, result.outcome)
        self.assertIsInstance(tv.plan, dict)
        # rows render, the code row carries its copy anchor, nothing is left running
        rows = tv.rows_html(5)
        self.assertIn("Make the plate", rows)
        row = pr.step_row_html(second, 5, 1, muted="#888888", expanded=True)
        self.assertIn("<pre>print(1)</pre>", row)
        self.assertIn("copy:5:1", row)
        self.assertEqual(tv.running_steps, [])
        self.assertTrue(tv.header().startswith("✻ Ran 2 steps"))
        plan_steps = (tv.plan or {}).get("steps") or []
        for step in plan_steps:   # the strip lists every step the worker's ledger carries
            self.assertIn(step["title"].split(":")[0], tv.plan_html())

    def test_progress_payloads_make_a_heartbeat(self):
        from support.scripted_llm import call, calls, say

        tv, rec, _ = self.replay("make a plate", [calls(call("create_object", obj_type="Part::Box")), say("Done.")])
        beats = [pr.heartbeat_text(json.loads(a[0]), now=1790000100.0) for a in rec.of("progress")]
        self.assertTrue(beats, "the worker emitted no progress")
        self.assertTrue(all(b for b in beats), beats)
        self.assertTrue(any(b.startswith("round ") for b in beats), beats)

    def test_turn_finished_text_splits_cleanly(self):
        from support.scripted_llm import say

        _, rec, result = self.replay("hello?", [say("Hi there.")])
        ok, text = rec.of("turn_finished")[-1]
        head, summary = pr.split_build_summary(text)
        self.assertEqual((head, summary), ("Hi there.", ""))


# --------------------------------------------------------------------------------------------------
# Qt-side pieces that need no window: skipped under an interpreter without PySide / FreeCAD


def _qt_modules():
    try:
        from oscar import panel_blocks
        return panel_blocks, None
    except Exception:
        return None


class BlockRenderTest(unittest.TestCase):
    def setUp(self):
        mods = _qt_modules()
        if mods is None:
            self.skipTest("Qt / FreeCAD unavailable in this interpreter")
        self.blocks = mods[0]

    def test_live_header_is_never_done_mid_turn(self):
        Block, Step = self.blocks.Block, self.blocks.Step
        card = Block(Block.TOOL, "")
        card.expanded = True
        card.t0 = 1.0           # the turn is still running
        card.steps = [Step("create_object", summary="create_object(Box)", ok=True, dur=0.3, detail="made")]
        out = card.html()       # every step finished, yet the worker is still between rounds
        self.assertIn("Working…", out)
        self.assertNotIn("Done", out)
        card.t0 = 0.0           # turn over
        self.assertNotIn("Working…", card.html())
        self.assertIn("✻ Done", card.html())
        self.assertTrue(card.settled())

    def test_step_detail_keeps_twelve_thousand_characters(self):
        Block, Step = self.blocks.Block, self.blocks.Step
        self.assertEqual(self.blocks._DETAIL_LIMIT, 12000)
        card = Block(Block.TOOL, "✻ Ran 1 step · 1.0s")
        card.expanded = True
        step = Step("t", summary="t()", ok=True, dur=1.0, detail="a" * 11000)
        step.expanded = True
        card.steps = [step]
        out = card.html()
        self.assertIn("a" * 11000, out)
        self.assertNotIn("…", out)
        step.detail = "b" * 13000
        out = card.html()
        self.assertIn("b" * 12000, out)
        self.assertNotIn("b" * 12001, out)
        self.assertEqual(out.count("…"), 1)
        # a result the agent already clipped with an ellipsis gets no second one
        step.detail = "c" * 400 + "…"
        self.assertEqual(card.html().count("…"), 1)

    def test_revert_link_only_on_the_enabled_card(self):
        Block, Step = self.blocks.Block, self.blocks.Step
        card = Block(Block.TOOL, "✻ Ran 1 step · 1.0s")
        card.steps = [Step("t", summary="t()", ok=True, dur=1.0)]
        card.revertible = True
        self.assertIn(f"href='revert:{card.bid}'", card.html())
        card.revert_enabled = False
        self.assertNotIn("revert:", card.html())
        card.revert_enabled = True
        card.reverted = True
        self.assertIn("reverted ✓", card.html())
        self.assertNotIn("revert:", card.html())

    def test_queued_user_bubble(self):
        Block = self.blocks.Block
        bubble = Block(Block.USER, "also add vents")
        self.assertNotIn("queued", bubble.html())
        bubble.queued = True
        self.assertIn("queued", bubble.html())
        self.assertIn("also add vents", bubble.html())

    def test_signature_tracks_every_visible_change(self):
        Block, Step = self.blocks.Block, self.blocks.Step
        card = Block(Block.TOOL, "x")
        step = Step("t", summary="t()", ok=True, dur=1.0, detail="d")
        card.steps = [step]
        seen = {card.signature()}
        for mutate in (lambda: setattr(card, "expanded", True), lambda: setattr(step, "expanded", True),
                       lambda: setattr(step, "detail", "d2"), lambda: setattr(card, "text", "y"),
                       lambda: setattr(card, "revertible", True), lambda: setattr(card, "reverted", True),
                       lambda: setattr(card, "revert_enabled", False), lambda: card.steps.append(Step("u", ok=True))):
            mutate()
            sig = card.signature()
            self.assertNotIn(sig, seen)
            seen.add(sig)

    def test_settled_excludes_running_clocks(self):
        Block, Step = self.blocks.Block, self.blocks.Step
        card = Block(Block.TOOL, "")
        self.assertTrue(card.settled())
        card.t0 = 1.0
        self.assertFalse(card.settled())
        assistant = Block(Block.ASSISTANT, "text")
        thought = Step(is_thought=True)
        thought.t0 = 5.0
        assistant.steps = [thought]
        self.assertFalse(assistant.settled())

    def test_panel_reexports(self):
        try:
            from oscar import panel
        except Exception:
            self.skipTest("oscar.panel unavailable")
        self.assertIs(panel.md_to_html, pr.md_to_html)
        self.assertIs(panel.Block, self.blocks.Block)
        self.assertIs(panel.Step, self.blocks.Step)
        for name in ("apply_feature_plan", "execute_code"):
            self.assertIn(name, panel.OscarPanel._MUTATING_TOOLS)
        self.assertIn("PartDesign Body", panel.OscarPanel.PARAMPT)
        self.assertEqual(panel._DETAIL_LIMIT, 12000)




# --------------------------------------------------------------------------------------------------
# Header text, summary block, call ids, turn records, restore card, heartbeat clock, plan view


class HeaderTextTest(unittest.TestCase):
    def test_legacy_formats(self):
        self.assertEqual(pr.header_text(thought_s=None, steps=3, total_s=14.0), "✻ Ran 3 steps · 14.0s")
        self.assertEqual(pr.header_text(thought_s=2.2, steps=2, total_s=14.0), "✻ Thought 2.2s · Ran 2 steps · 14.0s")
        self.assertEqual(pr.header_text(thought_s=2.2, steps=0, total_s=3.0), "✻ Thought 2.2s")
        self.assertEqual(pr.header_text(thought_s=None, steps=1, total_s=0.04), "✻ Ran 1 step · 0.0s")
        self.assertEqual(pr.header_text(thought_s=None, steps=0, total_s=0.0), "✻ Done")

    def test_failed_steps_and_the_badge_come_after_the_legacy_text(self):
        text = pr.header_text(thought_s=None, steps=3, total_s=14.0, failed=True, outcome="unverified")
        self.assertEqual(text, "✻ Ran 3 steps · 14.0s · ⚠ failed steps · ⚠ unverified")
        self.assertTrue(text.startswith("✻ Ran 3 steps"))                       # never before

    def test_done_is_only_for_answered_validated_or_unset(self):
        for outcome in ("", "answered", "validated"):
            self.assertTrue(pr.header_text(thought_s=None, steps=0, total_s=0, outcome=outcome).startswith("✻ Done"))
        for outcome in ("unverified", "capped", "needs_input", "failed", "needs_help", "stopped"):
            text = pr.header_text(thought_s=None, steps=0, total_s=0, outcome=outcome)
            self.assertNotIn("Done", text, outcome)
            self.assertTrue(text.startswith("✻ Ended"), outcome)

    def test_negative_durations_are_clamped_and_unknown_outcomes_carry_no_badge(self):
        self.assertEqual(pr.header_text(thought_s=None, steps=2, total_s=-5.0, outcome="mystery"), "✻ Ran 2 steps · 0.0s")

    def test_turn_view_header_uses_the_same_function(self):
        tv = TurnView(clock=lambda: 1000.0)
        tv.on_step_started(started())
        tv.on_step_finished(finished(ok=False))
        tv.add_thought(2.2)
        tv.on_outcome("failed", "")
        self.assertEqual(tv.header(), pr.header_text(thought_s=2.2, steps=1, total_s=0.0, failed=True, outcome="failed"))
        empty = TurnView(clock=lambda: 1000.0)
        empty.on_outcome("unverified", "")
        self.assertEqual(empty.header(), "✻ Ended · ⚠ unverified")


class SummaryBlockTest(unittest.TestCase):
    def test_card_plus_continue_link(self):
        markup = pr.summary_block_html(BUILD_SUMMARY, [("continue:", "Continue")])
        self.assertIn("Build Summary", markup)
        self.assertIn("href='continue:'", markup)
        self.assertIn("▶ Continue", markup)
        self.assertTrue(markup.index("Build Summary") < markup.index("continue:"))
        self.assertTrue(balanced(markup))

    def test_actions_only_and_nothing_at_all(self):
        only = pr.summary_block_html("", [("continue:", "Continue")])
        self.assertNotIn("<h4", only)
        self.assertIn("continue:", only)
        self.assertEqual(pr.summary_block_html("", []), "")
        self.assertEqual(pr.summary_block_html("  \n", None), "")

    def test_hostile_targets_and_labels_cannot_inject_markup(self):
        markup = pr.summary_block_html("", [("x' onclick='evil", "a"), ("ok:", "<script>alert(1)</script>"),
                                            ("a b", "c"), ("<b>:", "d"), ("", "e")])
        self.assertNotIn("onclick", markup)
        self.assertNotIn("<script>", markup)
        self.assertIn("&lt;script&gt;", markup)
        self.assertEqual(markup.count("<a "), 1)


class UniqueCallIdTest(unittest.TestCase):
    def test_turn_ordinal_and_call_id_make_the_key(self):
        self.assertEqual(pr.unique_call_id({"call_id": "call_0", "ordinal": 3}, 2), "t2.3.call_0")
        self.assertNotEqual(pr.unique_call_id({"call_id": "call_0", "ordinal": 1}, 1),
                            pr.unique_call_id({"call_id": "call_0", "ordinal": 1}, 2))        # same id, another turn
        self.assertNotEqual(pr.unique_call_id({"call_id": "call_0", "ordinal": 1}, 1),
                            pr.unique_call_id({"call_id": "call_0", "ordinal": 2}, 1))        # same id, same turn

    def test_json_strings_missing_fields_and_garbage(self):
        self.assertEqual(pr.unique_call_id(json.dumps({"call_id": "c", "ordinal": 1}), 1), "t1.1.c")
        self.assertEqual(pr.unique_call_id({"call_id": "c"}, 4), "t4.0.c")
        self.assertEqual(pr.unique_call_id({"ordinal": 7}, 4), "t4.7.")
        self.assertEqual(pr.unique_call_id({}, 4), "")
        self.assertEqual(pr.unique_call_id("garbage", 4), "")
        self.assertEqual(pr.unique_call_id(None, 4), "")

    def test_the_key_is_a_valid_thumbnail_url_component(self):
        key = pr.unique_call_id({"call_id": "call/with space:odd", "ordinal": 2}, 1)
        url = pr.thumb_url(key)
        self.assertTrue(url.startswith("es-img://step/"))
        self.assertNotIn(" ", url)
        self.assertEqual(url.count("/"), 3)                # es-img: // step / <key>: the key adds no slash


class StepViewDeltaTest(unittest.TestCase):
    def test_changed_entries_may_be_project_style_dicts(self):
        view = StepView()
        view.apply({"delta": {"added": ["A"], "removed": [{"name": "B"}], "changed": [
            {"name": "Lid", "props": ["Height", "Width"], "volume_before": 1.0, "volume_after": 2.0}, "Plain", {"props": []}]}})
        self.assertEqual(view.delta, {"added": ["A"], "removed": ["B"], "changed": ["Lid", "Plain"]})
        self.assertEqual(view.changed_props, {"Lid": ["Height", "Width"]})
        self.assertEqual(view.changed_objects(), ["A", "Lid", "Plain"])

    def test_chips_and_detail_show_names_not_dict_reprs(self):
        view = StepView(running=False)
        view.apply({"title": "t", "text": "x", "delta": {"added": [], "removed": [], "changed": [
            {"name": "Lid", "props": ["Height"]}]}})
        row = pr.step_row_html(view, 1, 0, expanded=True)
        self.assertIn("~Lid", row)
        self.assertNotIn("{", row)
        self.assertIn("Lid (Height)", row)

    def test_rev_moves_with_every_change(self):
        view = StepView()
        before = view.rev
        view.apply({"title": "x"})
        self.assertGreater(view.rev, before)
        mid = view.rev
        view.touch()
        self.assertEqual(view.rev, mid + 1)
        tv = TurnView(clock=lambda: 1.0)
        started_view = tv.on_step_started(started())
        r = started_view.rev
        tv.on_step_finished(finished())
        self.assertGreater(started_view.rev, r)


class HeartbeatClockTest(unittest.TestCase):
    def make(self):
        self.t = [5000.0]
        return TurnView(clock=lambda: self.t[0])

    def test_provider_counters_keep_moving_between_payloads(self):
        tv = self.make()
        tv.on_progress(progress(activity="thinking", elapsed_round=5.0, idle_s=1.0, reasoning_chars=1500))
        self.assertEqual(tv.heartbeat(), "round 7/30 · BUILD · thinking 5 s · 1.5k chars")
        self.t[0] += 3
        self.assertEqual(tv.heartbeat(), "round 7/30 · BUILD · thinking 8 s · 1.5k chars")

    def test_a_silent_stream_shows_no_data_without_a_new_payload(self):
        tv = self.make()
        tv.on_progress(progress(activity="writing", elapsed_round=5.0, idle_s=1.0, text_chars=0))
        self.t[0] += 45
        self.assertIn("no data for 46 s", tv.heartbeat())

    def test_a_new_payload_resets_the_gap(self):
        tv = self.make()
        tv.on_progress(progress(activity="thinking", elapsed_round=5.0, idle_s=1.0))
        self.t[0] += 40
        tv.on_progress(progress(activity="thinking", elapsed_round=45.0, idle_s=0.5))
        self.assertNotIn("no data", tv.heartbeat())
        self.assertIn("thinking 45 s", tv.heartbeat())

    def test_tools_and_waiting_states_are_not_compensated(self):
        tv = self.make()
        tv.on_progress(progress(activity="waiting_approval"))
        self.t[0] += 100
        self.assertEqual(tv.heartbeat(), "round 7/30 · BUILD · waiting for your approval")
        tv.on_progress(progress(activity="running_tool", tool="x", started_at=self.t[0] - 10))
        self.t[0] += 5
        self.assertIn("running x 15 s", tv.heartbeat())             # counted from started_at

    def test_an_explicit_now_means_the_payload_is_current(self):
        tv = self.make()
        tv.on_progress(progress(activity="thinking", elapsed_round=12.0, idle_s=1.0, reasoning_chars=1820))
        self.assertEqual(tv.heartbeat(now=9e9), "round 7/30 · BUILD · thinking 12 s · 1.8k chars")

    def test_finished_turns_and_missing_progress_have_no_heartbeat(self):
        tv = self.make()
        self.assertEqual(tv.heartbeat(), "")
        tv.on_progress(progress())
        tv.finish()
        self.assertEqual(tv.heartbeat(), "")


class PlanViewTest(unittest.TestCase):
    def test_progress_updates_round_cap_and_phase_of_the_plan(self):
        tv = TurnView(clock=lambda: 1.0)
        tv.on_plan(ledger())
        tv.on_progress(progress(round=12, round_cap=40, phase="VERIFY"))
        view = tv.plan_view()
        self.assertEqual((view["round"], view["round_cap"], view["phase"]), (12, 40, "VERIFY"))
        self.assertEqual(tv.plan["round"], 7)                          # the stored ledger is not mutated
        html_text = tv.plan_html()
        self.assertIn("round 12/40", html_text)
        self.assertIn("VERIFY", html_text)

    def test_missing_or_partial_progress_keeps_the_ledger_values(self):
        tv = TurnView(clock=lambda: 1.0)
        self.assertIsNone(tv.plan_view())
        tv.on_plan(ledger())
        self.assertIs(tv.plan_view(), tv.plan)
        tv.on_progress({"activity": "thinking"})
        view = tv.plan_view()
        self.assertEqual((view["round"], view["round_cap"], view["phase"]), (7, 30, "BUILD"))

    def test_the_intent_chip_shows_unless_other_or_collapsed(self):
        self.assertIn("intent: enclosure", pr.plan_strip_html(ledger()))
        self.assertNotIn("intent:", pr.plan_strip_html(ledger(intent="other")))
        self.assertNotIn("intent:", pr.plan_strip_html(ledger(intent="")))
        self.assertNotIn("intent:", pr.plan_strip_html(ledger(), collapsed=True))
        self.assertIn("intent: &lt;b&gt;", pr.plan_strip_html(ledger(intent="<b>x</b>")))


class TurnMetaTest(unittest.TestCase):
    def views(self, count=2, root="/p"):
        out = []
        for i in range(count):
            v = StepView(title=f"Step {i}", ok=i != 1, dur_s=1.26, snapshot_path=f"{root}/snapshots/{i:03d}_x.png" if i != 1 else "")
            v.delta = {"added": [f"O{i}"], "removed": [], "changed": ["Lid"]}
            out.append(v)
        return out

    def test_record_is_compact_json_safe_and_relative(self):
        meta = pr.make_turn_meta("  Build   a\nhousing  ", "validated", "✻ Ran 2 steps · ✓ validated", self.views(), root="/p")
        self.assertEqual(json.loads(json.dumps(meta)), meta)
        self.assertEqual(meta["request"], "Build a housing")
        self.assertEqual(meta["outcome"], "validated")
        self.assertEqual(meta["steps"][0], {"title": "Step 0", "ok": True, "skipped": False, "dur": 1.3,
                                            "delta": "+O0 ~Lid", "snapshot": "snapshots/000_x.png"})
        self.assertEqual(meta["steps"][1]["ok"], False)
        self.assertEqual(meta["steps"][1]["snapshot"], "")

    def test_paths_outside_the_project_stay_absolute_and_long_input_is_cut(self):
        views = self.views(1, root="/elsewhere")
        meta = pr.make_turn_meta("x" * 500, "failed", "h", views, root="/p")
        self.assertEqual(meta["steps"][0]["snapshot"], "/elsewhere/snapshots/000_x.png")
        self.assertLessEqual(len(meta["request"]), 162)
        self.assertEqual(pr.make_turn_meta("r", "o", "h", views, root="")["steps"][0]["snapshot"],
                         "/elsewhere/snapshots/000_x.png")

    def test_step_count_is_bounded(self):
        meta = pr.make_turn_meta("r", "validated", "h", self.views(75), root="/p")
        self.assertEqual(len(meta["steps"]), 60)
        self.assertEqual(meta["more"], 15)

    def test_trim_keeps_well_formed_records_and_the_newest_twenty(self):
        good = [{"steps": [], "request": str(i)} for i in range(25)]
        trimmed = pr.trim_turn_metas(good + ["x", None, {"steps": "no"}, 5])
        self.assertEqual([m["request"] for m in trimmed], [str(i) for i in range(5, 25)])
        for bad in (None, "x", 5, {"steps": []}, [None]):
            self.assertEqual(pr.trim_turn_metas(bad), [])

    def test_restore_thumbs_pick_the_newest_turn_with_snapshots(self):
        older = pr.make_turn_meta("a", "validated", "h", self.views(3), root="/p")
        newer = pr.make_turn_meta("b", "validated", "h", [StepView(title="no snapshot")], root="/p")
        thumbs = pr.restore_thumbs([older, newer], "k7")
        self.assertEqual([t[1] for t in thumbs], ["snapshots/000_x.png", "snapshots/002_x.png"])
        self.assertEqual([t[0] for t in thumbs], ["restore.k7.0", "restore.k7.1"])
        self.assertEqual(pr.restore_thumbs([], "k"), [])
        many = pr.make_turn_meta("c", "validated", "h", [StepView(title=str(i), snapshot_path=f"/p/s/{i}.png") for i in range(9)], root="/p")
        self.assertEqual(len(pr.restore_thumbs([many], "k", limit=4)), 4)
        self.assertEqual(pr.restore_thumbs([many], "k", limit=4)[-1][1], "s/8.png")

    def test_restore_card_html(self):
        meta = pr.make_turn_meta("Build <b>x</b>", "validated", "✻ Ran 2 steps · ✓ validated", self.views(), root="/p")
        markup = pr.restore_card_html("My <chat>", 12, [meta], notice="Interrupted <now>", thumbs={"restore.r.0"})
        self.assertIn("↩ Restored “My &lt;chat&gt;” — 12 messages of task context. Continue where you left off.", markup)
        self.assertIn("⚠ Interrupted &lt;now&gt;", markup)
        self.assertIn("<b>Build &lt;b&gt;x&lt;/b&gt;</b>", markup)
        self.assertIn("✓ Step 0", markup)
        self.assertIn("✗ Step 1", markup)
        self.assertIn("+O0 ~Lid", markup)
        self.assertEqual(markup.count("<img"), 1)
        self.assertIn("es-img://step/restore.r.0", markup)
        self.assertTrue(balanced(markup))

    def test_restore_card_limits_turns_steps_and_thumbnails(self):
        metas = [pr.make_turn_meta(f"turn {i}", "validated", "h", self.views(12), root="/p") for i in range(6)]
        markup = pr.restore_card_html("t", 3, metas, max_turns=2)
        self.assertNotIn("turn 3", markup)
        self.assertIn("turn 4", markup)
        self.assertIn("turn 5", markup)
        self.assertIn("+4 more steps", markup)
        none_loaded = pr.restore_card_html("t", 3, metas, thumbs=set())
        self.assertNotIn("<img", none_loaded)
        self.assertEqual(pr.restore_card_html("t", 1, None), pr.restore_card_html("t", 1, []))
        self.assertEqual(pr.restore_card_html("t", 1, "garbage").count("Restored"), 1)

    def test_restore_card_survives_damaged_records(self):
        metas = [{"steps": [None, 5, {"title": None, "ok": "yes"}, {"skipped": True, "title": "Skip me"}], "request": None}]
        markup = pr.restore_card_html("t", 1, metas)
        self.assertIn("(no request text)", markup)
        self.assertIn("↷ Skip me", markup)


class SnapshotNoteTest(unittest.TestCase):
    def view(self, note):
        v = StepView(title="t", running=False)
        v.apply({"snapshot_note": note})
        return v

    def test_only_informative_notes_are_drawn(self):
        self.assertNotIn("not a geometry tool", pr.step_row_html(self.view("not a geometry tool"), 1, 0))
        for note in ("no 3D view (headless session)", "blank frame", "empty document", "skipped (GUI busy)"):
            self.assertIn(note, pr.step_row_html(self.view(note), 1, 0))
        self.assertNotIn("font-size:small", pr.step_row_html(self.view(""), 1, 0))


if __name__ == "__main__":
    unittest.main()
