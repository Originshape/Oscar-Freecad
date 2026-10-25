"""oscar.flow: intent classification, requirements, template plans, the Ledger, the phase machine, the finish gate
and the governor line (pure Python, runs without FreeCAD). The watchdog and the failure breaker are in
test_flow_guard.py."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import flow, loopcore  # noqa: E402
from oscar.flow import (AgentConfig, GateVerdict, Ledger, LedgerError, TurnState, derive_phase, finish_gate,  # noqa: E402
                        format_governor, phase_exit_met, visible_phases)
from oscar.toolctx import ToolOutcome  # noqa: E402

PI = ("Build a housing for a Raspberry Pi 4. Bottom shell with cutouts for every I/O port, air vent slots, "
      "snap-fit lid, ABS.")
PI_FULL = PI + " Source the STEP file from the vendor."

INTENT_TABLE = [
    (PI, "enclosure"),
    ("Can you design an enclosure for my Arduino Uno?", "enclosure"),
    ("make a case for the esp32 devkit", "enclosure"),
    ("Construye una carcasa para Raspberry Pi 4", "enclosure"),
    ("I need a jig to drill 4 holes in this plate", "fixture"),
    ("Design a bracket for a NEMA17 motor", "bracket"),
    ("Build a flange, 80 mm diameter, 8 mm thick, 6 bolt holes", "generic_part"),
    ("Build the smoke test box.", "generic_part"),
    ("three boxes", "generic_part"),
    ("what's in this document?", "inspect"),
    ("What is the difference between pad and pocket?", "question"),
    ("how do I make a fillet?", "question"),
    ("run a static analysis on the bracket", "fem"),
    ("export it as STL", "deliver"),
    ("make the vents 3 mm wider", "modify"),
    ("continue", "other"),
    ("list my documents", "inspect"),
    ("hello?", "other"),
    ("Build a model of a gear, show me the dimensions", "generic_part"),
    ("Build a box for my keys", "enclosure"),
    ("export the model as STEP", "deliver"),
    ("Build a housing for a Pi 4 and export it as STL", "enclosure"),
    ("show me three views", "other"),
    ("Is it possible to build a housing for a Pi 4 with a snap-fit lid?", "enclosure"),
    ("Do you know how to make a case for an ESP32?", "enclosure"),
    ("Are you able to design a bracket for a NEMA17?", "bracket"),
    ("add a fan hole on the side", "generic_part"),
    ("cut a slot for the ribbon cable", "generic_part"),
    ("drill four 3 mm holes", "generic_part"),
    ("Hola, hazme una pieza cilindrica de 20 mm", "generic_part"),
    ("Erstelle eine Halterung fuer einen Motor", "other"),
    ("thanks", "other"),
]


def oc(name, *, geometry=True, mutating=None, handler_ok=True, ok=None, step_index=-1, args=None, extra=None,
       delta=None, title=""):
    """A ToolOutcome as ``tools.run_tool`` would leave it."""
    return ToolOutcome(name=name, title=title or name, ok=handler_ok if ok is None else ok, handler_ok=handler_ok,
                       mutating=geometry if mutating is None else mutating, geometry=geometry, step_index=step_index,
                       args=args or {}, extra=extra or {}, delta=delta or {"added": ["Obj"], "removed": [], "changed": []})


def good_validation(**extra):
    base = {"ok": True, "gate_errors": [], "fail_rows": [], "after_step": 0}
    base.update(extra)
    return base


class IntentTest(unittest.TestCase):
    def test_contract_table(self):
        for text, intent in INTENT_TABLE:
            with self.subTest(text=text):
                self.assertEqual(flow.classify_intent(text), intent)

    def test_revert_note_and_attachment_text_do_not_change_the_class(self):
        content = ("[I reverted every CAD change you made after that point; check the workspace state below.]\n\n"
                   "what is in this document?\n\n--- attached files ---\nbracket for a motor, 20 mm, add 4 holes\n")
        typed = loopcore.real_user_request([_msg(content)])
        self.assertEqual(flow.classify_intent(typed), "inspect")
        # without the input rule the note alone would have made it a "modify"
        self.assertEqual(flow.classify_intent(content.split("\n\n")[0]), "modify")

    def test_attachment_paths_only_through_the_argument(self):
        self.assertEqual(flow.classify_intent("hello?", attachments=["/tmp/part.step"]), "generic_part")
        self.assertEqual(flow.classify_intent("hello?", attachments=["/tmp/notes.txt"]), "other")

    def test_unknown_and_empty_text(self):
        self.assertEqual(flow.classify_intent(""), "other")
        self.assertEqual(flow.classify_intent("   \n "), "other")

    def test_a_question_frame_is_a_request(self):
        self.assertEqual(flow.classify_intent("Can you build a housing for a Pi 4?"), "enclosure")
        self.assertEqual(flow.classify_intent("Why is the lid loose?"), "question")

    def test_a_huge_paste_is_not_quadratic(self):
        import time

        for text in ("box " * 30000 + "zzz", "case for " * 20000, "Build a housing, " * 5000, "x. " * 30000):
            started = time.monotonic()
            ledger = Ledger.from_request(text)
            self.assertLess(time.monotonic() - started, 2.0, text[:12])
            self.assertLessEqual(len(ledger.request), 4000)
            self.assertLessEqual(len(ledger.requirements), 13)

    def test_non_string_input_is_other(self):
        self.assertEqual(flow.classify_intent(None), "other")
        self.assertEqual(flow.extract_requirements(None, "enclosure")[-1]["covers"], ["S12"])
        self.assertEqual(flow.needs_reference(None, "enclosure"), (False, ""))

    def test_intent_constants(self):
        self.assertEqual(len(flow.INTENTS), 10)
        self.assertEqual(set(flow.WORK), {"enclosure", "fixture", "bracket", "generic_part", "modify", "fem"})
        self.assertNotIn("deliver", flow.WORK)


def _msg(content):
    from oscar.providers import Message

    return Message(role="user", content=content)


class NeedsReferenceTest(unittest.TestCase):
    def test_pi_housing_needs_a_reference(self):
        needs, hint = flow.needs_reference(PI, "enclosure")
        self.assertTrue(needs)
        self.assertIn("Raspberry Pi 4", hint)
        self.assertLessEqual(len(hint), 60)

    def test_box_for_keys_does_not(self):
        self.assertEqual(flow.needs_reference("Build a box for my keys", "enclosure")[0], False)

    def test_only_enclosure_and_fixture(self):
        self.assertFalse(flow.needs_reference("Design a bracket for a NEMA17 motor", "bracket")[0])
        self.assertFalse(flow.needs_reference("make a plate", "generic_part")[0])
        self.assertTrue(flow.needs_reference("a jig for a NEMA17 motor", "fixture")[0])

    def test_attached_cad_file_makes_it_true(self):
        self.assertTrue(flow.needs_reference("make a box for this", "enclosure", attachments=["/tmp/x.step"])[0])

    def test_source_words(self):
        self.assertTrue(flow.needs_reference("an enclosure; download the datasheet first", "enclosure")[0])


def _prev_ledger():
    return Ledger.from_request("make a plate")


class ContinuationTest(unittest.TestCase):
    def test_short_go_aheads_resume(self):
        prev = _prev_ledger()
        for text in ("continue", "Continue", "yes go ahead", "ok", "go on", "keep going", "sigue", "weiter", "继续", "do it"):
            with self.subTest(text=text):
                self.assertTrue(flow.is_continuation(text, prev))

    def test_thanks_and_hello_do_not(self):
        prev = _prev_ledger()
        for text in ("thanks", "hello?", "ok then?", "continue building the lid please now", "why did you stop?", ""):
            with self.subTest(text=text):
                self.assertFalse(flow.is_continuation(text, prev))

    def test_the_resume_prompt_is_a_continuation(self):
        prev = _prev_ledger()
        self.assertTrue(flow.is_continuation(prev.resume_prompt(), prev))

    def test_needs_a_previous_ledger_with_open_steps(self):
        self.assertFalse(flow.is_continuation("continue", None))
        done = _prev_ledger()
        for s in done.steps:
            s.status = "done"
        self.assertFalse(flow.is_continuation("continue", done))
        self.assertFalse(flow.is_continuation("continue", Ledger.from_request("hello?")))


class RequirementsTest(unittest.TestCase):
    def test_pi_prompt_requirements_follow_the_rules_in_order(self):
        reqs = flow.extract_requirements(PI_FULL, "enclosure")
        texts = [r["text"] for r in reqs]
        self.assertEqual([r["id"] for r in reqs], [f"R{i}" for i in range(1, len(reqs) + 1)])
        self.assertEqual(texts[:6], ["A bottom shell that holds the board", "Cutouts for every I/O port", "A lid",
                                     "Air vent slots", "Snap-fit retention of the lid", "Material: ABS"])
        self.assertIn("Source the reference part", texts)
        self.assertEqual(texts[-2:], ["Every driving dimension is a named parameter",
                                      "Delivered as a rebuildable project (notes, rebuild script, exports)"])
        covers = {r["text"]: r["covers"] for r in reqs}
        self.assertEqual(covers["A bottom shell that holds the board"], ["S5"])
        self.assertEqual(covers["Cutouts for every I/O port"], ["S6"])
        self.assertEqual(covers["A lid"], ["S7"])
        self.assertEqual(covers["Air vent slots"], ["S8"])
        self.assertEqual(covers["Snap-fit retention of the lid"], ["S9"])
        self.assertEqual(covers["Material: ABS"], ["S4"])
        self.assertEqual(covers["Source the reference part"], ["S2", "S3"])
        self.assertEqual(len(reqs), 9)

    def test_plural_words_fire_the_rules(self):
        reqs = flow.extract_requirements("Make an enclosure with cutouts for all ports and several vents", "enclosure")
        texts = [r["text"] for r in reqs]
        self.assertIn("Cutouts for every I/O port", texts)
        self.assertIn("Air vent slots", texts)

    def test_no_rule_no_requirement(self):
        texts = [r["text"] for r in flow.extract_requirements("Build a box for my keys", "enclosure")]
        self.assertNotIn("A lid", texts)
        self.assertNotIn("Air vent slots", texts)
        self.assertNotIn("Snap-fit retention of the lid", texts)

    def test_extra_clauses_become_requirements_verified_at_the_end(self):
        reqs = flow.extract_requirements("Build an enclosure, with a company logo engraved on the front, "
                                         "and the wall should look rounded", "enclosure")
        extra = [r for r in reqs if r["covers"] == ["S11"]]
        self.assertTrue(extra)
        self.assertTrue(all(12 <= len(r["text"]) <= 160 for r in extra))

    def test_at_most_twelve(self):
        text = ". ".join(f"Requirement number {i} describes a feature of the enclosure" for i in range(30))
        self.assertLessEqual(len(flow.extract_requirements(text, "enclosure")), 12)

    def test_other_intents_get_one_requirement_per_clause_and_a_deliverable(self):
        reqs = flow.extract_requirements("Build a flange, 80 mm diameter, 8 mm thick, with six bolt holes", "generic_part")
        self.assertGreaterEqual(len(reqs), 2)
        self.assertEqual(reqs[-1]["text"].split(" (")[0], "Delivered as a rebuildable project")
        self.assertEqual(reqs[-1]["covers"], ["S5"])
        self.assertTrue(all(r["covers"] == ["S3"] for r in reqs[:-1]))

    def test_the_first_build_step_per_intent(self):
        for intent, first, delivered in (("fixture", "S5", "S8"), ("bracket", "S3", "S7"), ("modify", "S1", "S3")):
            with self.subTest(intent=intent):
                reqs = flow.extract_requirements("a motor mount that holds the part and a second clause here", intent)
                self.assertEqual(reqs[0]["covers"], [first])
                self.assertEqual(reqs[-1]["covers"], [delivered])

    def test_attachment_text_never_reaches_the_requirements(self):
        content = PI + "\n\n--- attached files ---\nA bracket with motor, sensor and many holes. Add a fan, a display."
        typed = loopcore.real_user_request([_msg(content)])
        self.assertEqual(flow.extract_requirements(typed, "enclosure"), flow.extract_requirements(PI, "enclosure"))


class TemplateTest(unittest.TestCase):
    def ids(self, ledger):
        return [s.id for s in ledger.steps]

    def test_enclosure_template_is_exact(self):
        ledger = Ledger.from_request(PI_FULL)
        rows = [(s.id, s.phase, s.title, s.owner) for s in ledger.steps]
        self.assertEqual(rows, [
            ("S1", "INTAKE", "Capture requirements and assumptions", "app"),
            ("S2", "SOURCE", "Source the reference part", "model"),
            ("S3", "SOURCE", "Import and measure the reference", "model"),
            ("S4", "PARAMS", "Parameters table and spec", "model"),
            ("S5", "BUILD", "Bottom shell: floor, walls, standoffs", "model"),
            ("S6", "BUILD", "Port cutouts for every connector", "model"),
            ("S7", "BUILD", "Lid with lip", "model"),
            ("S8", "BUILD", "Air vent slots", "model"),
            ("S9", "BUILD", "Snap hooks and catch windows", "model"),
            ("S10", "BUILD", "Assembly with lid_lift parameter", "model"),
            ("S11", "VERIFY", "Validate model and check fit", "app"),
            ("S12", "DELIVER", "Save project, export, notes", "app")])
        self.assertEqual(ledger.step("S1").status, "done")
        self.assertTrue(all(ledger.step(i).required for i in ("S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10")))
        self.assertEqual(ledger.step("S5").stages, ["shell", "standoffs"])
        self.assertEqual(ledger.step("S2").hint_tools[:1], ["find_reference_part"])
        self.assertTrue(ledger.needs_reference)

    def test_unrequested_steps_are_kept_greyed(self):
        ledger = Ledger.from_request("Build a box for my keys")
        s6, s8, s9 = ledger.step("S6"), ledger.step("S8"), ledger.step("S9")
        for s in (s6, s8, s9):
            self.assertFalse(s.required)
            self.assertEqual((s.status, s.note), ("skipped", flow.NOT_REQUESTED))
        self.assertFalse(ledger.step("S2").required)       # no reference needed
        self.assertTrue(ledger.step("S5").required)

    def test_the_ledger_covers_every_requirement(self):
        ledger = Ledger.from_request(PI_FULL)
        self.assertTrue(all(r.covered_by for r in ledger.requirements))
        self.assertEqual(ledger.requirement("R1").covered_by, ["S5"])
        self.assertEqual(ledger.requirement("R6").covered_by, ["S4"])
        self.assertEqual(ledger.requirement("R7").covered_by, ["S2", "S3"])

    def test_other_templates(self):
        shapes = {
            "make a plate": ["S1", "S2", "S3", "S4", "S5"],
            "Design a bracket for a NEMA17 motor": ["S1", "S2", "S3", "S4", "S5", "S6", "S7"],
            "make the vents 3 mm wider": ["S1", "S2", "S3"],
            "run a static analysis on the bracket": ["S1", "S2", "S3", "S4"],
            "export it as STL": ["S1"],
            "I need a jig for a NEMA17 motor with clamps": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
        }
        for text, ids in shapes.items():
            with self.subTest(text=text):
                self.assertEqual(self.ids(Ledger.from_request(text)), ids)

    def test_generic_part_required_flags(self):
        ledger = Ledger.from_request("make a plate")
        self.assertEqual([(s.id, s.required, s.owner) for s in ledger.steps],
                         [("S1", True, "app"), ("S2", False, "model"), ("S3", True, "model"),
                          ("S4", True, "app"), ("S5", True, "app")])
        self.assertEqual(ledger.step("S4").phase, "VERIFY")


    def test_deliver_template(self):
        ledger = Ledger.from_request("export it as STL")
        self.assertEqual(ledger.intent, "deliver")
        self.assertEqual((ledger.step("S1").phase, ledger.step("S1").owner), ("DELIVER", "model"))
        self.assertTrue(ledger.is_work_turn())

    def test_plain_chat_has_an_empty_ledger(self):
        for text in ("what's in this document?", "how do I make a fillet?", "hello?"):
            ledger = Ledger.from_request(text)
            self.assertEqual((ledger.steps, ledger.phase), ([], "INTAKE"))
            self.assertFalse(ledger.is_work_turn())
            self.assertFalse(ledger.has_open_required_steps())


class LedgerBasicsTest(unittest.TestCase):
    def test_json_roundtrip(self):
        ledger = Ledger.from_request(PI_FULL)
        ledger.on_tool_result(oc("create_object", step_index=3))
        again = Ledger.from_dict(json.loads(ledger.to_json()))
        self.assertEqual(again.to_dict(), ledger.to_dict())
        self.assertEqual(json.loads(ledger.to_json())["version"], 1)

    def test_from_dict_is_tolerant(self):
        ledger = Ledger.from_dict({"intent": "bracket", "steps": [{"id": "S1", "title": "x", "phase": "BUILD", "extra": 1}],
                                   "requirements": [{"id": "R1", "text": "t"}], "unknown": True})
        self.assertEqual(ledger.intent, "bracket")
        self.assertEqual(ledger.step("S1").status, "pending")
        self.assertEqual(Ledger.from_dict({}).steps, [])
        self.assertEqual(Ledger.from_dict(None).intent, "other")

    def test_current_step_is_the_running_one_else_the_first_open_required(self):
        ledger = Ledger.from_request(PI_FULL)
        self.assertEqual(ledger.current_step().id, "S2")
        ledger.plan_update("S5", "running")
        self.assertEqual(ledger.current_step().id, "S5")

    def test_counts_and_resume_prompt(self):
        ledger = Ledger.from_request("make a plate")
        c = ledger.counts()
        self.assertEqual((c["done"], c["total"]), (1, 4))
        self.assertIn('Next: S3 "Model the part"', ledger.resume_prompt())
        self.assertTrue(ledger.resume_prompt().startswith("Continue the task. Plan status:"))

    def test_render_compact_format(self):
        ledger = Ledger.from_request(PI_FULL)
        ledger.plan_update("S2", "skipped", "no network")
        ledger.plan_update("S5", "running")
        text = ledger.render_compact(40)
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("PLAN (intent enclosure, phase BUILD, "))
        self.assertIn("[x] S1 Capture requirements and assumptions", lines)
        self.assertIn("[-] S2 Source the reference part - no network", lines)
        self.assertTrue(any(l.startswith("[~] S5 ") and l.endswith("<- now") for l in lines))
        self.assertIn("[ ] S6 Port cutouts for every connector", lines)
        self.assertTrue(lines[-1].startswith("REQUIREMENTS: R1 pending (S5)"))

    def test_render_compact_collapses_long_plans(self):
        ledger = Ledger.from_request(PI_FULL)
        text = ledger.render_compact(max_lines=6)
        self.assertIn("... +", text)
        self.assertLessEqual(len(text.splitlines()), 8)

    def test_every_mutation_bumps_the_revision(self):
        ledger = Ledger.from_request("make a plate")
        rev = ledger.revision
        ledger.plan_update("S3", "running")
        self.assertGreater(ledger.revision, rev)
        rev = ledger.revision
        ledger.plan_set(requirements=[{"text": "Add a chamfer on every edge"}])
        self.assertGreater(ledger.revision, rev)

    def test_requirement_status_is_derived(self):
        ledger = Ledger.from_request("make a plate")
        req = ledger.requirements[0]
        self.assertEqual(req.status, "pending")
        ledger.plan_update("S3", "blocked", "needs a file")
        self.assertEqual((ledger.requirement(req.id).status, ledger.requirement(req.id).note), ("blocked", "needs a file"))
        ledger.plan_update("S3", "done", "built")
        # a step the model only CLAIMED done (no recorded work) never makes a requirement done (I2 / I3)
        self.assertEqual(ledger.requirement(req.id).status, "pending")
        ledger.on_tool_result(oc("create_object", step_index=1))
        ledger.confirm_evidenced_steps()
        self.assertEqual(ledger.requirement(req.id).status, "done")


class PlanUpdateTest(unittest.TestCase):
    def ledger(self):
        return Ledger.from_request(PI_FULL)

    def test_unknown_id(self):
        with self.assertRaises(LedgerError) as cm:
            self.ledger().plan_update("S99", "done")
        self.assertEqual(str(cm.exception), "plan_update: 'S99' is not in the plan. Plan ids: S1..S12, R1..R9.")

    def test_bad_status(self):
        with self.assertRaises(LedgerError) as cm:
            self.ledger().plan_update("S5", "finished")
        self.assertEqual(str(cm.exception),
                         "plan_update: status must be one of pending|running|done|blocked|skipped|changed, got 'finished'.")

    def test_reason_required(self):
        for status in ("blocked", "skipped", "changed"):
            with self.assertRaises(LedgerError) as cm:
                self.ledger().plan_update("S5", status)
            self.assertEqual(str(cm.exception),
                             "plan_update: 'note' (the reason) is required when status is 'blocked', 'skipped' or 'changed'.")

    def test_app_owned_steps_cannot_be_set(self):
        with self.assertRaises(LedgerError) as cm:
            self.ledger().plan_update("S11", "done")
        self.assertEqual(str(cm.exception), 'plan_update: S11 "Validate model and check fit" is completed by the app after '
                                            "a passing validation; do not set it. Run validate_model / check_fit instead.")
        with self.assertRaises(LedgerError):
            self.ledger().plan_update("S12", "done")

    def test_requirement_status_follows_its_steps(self):
        with self.assertRaises(LedgerError) as cm:
            self.ledger().plan_update("R3", "done")
        self.assertEqual(str(cm.exception), "plan_update: requirement 'R3' follows its steps (S7); update the step instead. "
                                            "On a requirement only blocked, skipped or changed (with a note) can be set.")

    def test_a_requirement_can_be_skipped_with_a_note(self):
        ledger = self.ledger()
        ledger.plan_update("R4", "skipped", "user dropped the vents")
        self.assertEqual((ledger.requirement("R4").status, ledger.requirement("R4").note), ("skipped", "user dropped the vents"))

    def test_done_without_evidence_is_a_claim(self):
        ledger = self.ledger()
        text = ledger.plan_update("S6", "done", "cutouts exist", evidence="Shell_Body")
        self.assertIn("claimed", text)
        step = ledger.step("S6")
        self.assertTrue(step.claimed)
        self.assertEqual(step.evidence[-1]["kind"], "claim")
        self.assertEqual(ledger.claimed_only_steps(), [step])

    def test_done_with_recorded_work_is_not_a_claim(self):
        ledger = self.ledger()
        ledger.on_tool_result(oc("build_enclosure", args={"stage": "ports"}, step_index=2))
        ledger.step("S6").status = "pending"          # stage tick completed it; reopen to test the claim path
        ledger.on_tool_result(oc("execute_code", step_index=3))
        text = ledger.plan_update("S6", "done", "cut")
        self.assertNotIn("claimed", text)
        self.assertFalse(ledger.step("S6").claimed)

    def test_running_demotes_the_other_running_step(self):
        ledger = self.ledger()
        ledger.plan_update("S5", "running")
        ledger.plan_update("S6", "running")
        self.assertEqual([s.id for s in ledger.steps if s.status == "running"], ["S6"])

    def test_blocked_with_a_note(self):
        ledger = self.ledger()
        ledger.plan_update("S2", "blocked", "user declined the download")
        self.assertEqual(ledger.step("S2").status, "blocked")
        self.assertTrue(ledger.has_blocked_with_note())


class PlanSetTest(unittest.TestCase):
    def test_merge_keeps_template_steps_and_renames_by_id(self):
        ledger = Ledger.from_request("make a plate")
        warnings = ledger.plan_set(steps=[{"id": "S3", "title": "Model the plate with two holes"}])
        self.assertEqual(warnings, [])
        self.assertEqual(ledger.step("S3").title, "Model the plate with two holes")
        self.assertEqual(len(ledger.steps), 5)

    def test_new_steps_get_the_next_id_in_phase_order(self):
        ledger = Ledger.from_request("make a plate")
        ledger.plan_set(steps=[{"title": "Add fillets", "phase": "BUILD", "acceptance": "edges rounded"}])
        self.assertEqual([s.id for s in ledger.steps], ["S1", "S2", "S3", "S6", "S4", "S5"])
        self.assertFalse(ledger.step("S6").template)

    def test_bad_phase_is_rejected_and_changes_nothing(self):
        ledger = Ledger.from_request("make a plate")
        before = ledger.to_json()
        with self.assertRaises(LedgerError) as cm:
            ledger.plan_set(steps=[{"title": "x", "phase": "Building"}])
        self.assertEqual(str(cm.exception), "plan_set: step 'S6' has phase 'Building'; use one of "
                                            "INTAKE|SOURCE|PLAN|PARAMS|BUILD|VERIFY|DELIVER.")
        self.assertEqual(ledger.to_json(), before)

    def test_new_requirement_unclaimed_is_verified_at_verify(self):
        ledger = Ledger.from_request("make a plate")
        ledger.plan_set(requirements=[{"text": "Edges must be chamfered"}])
        new = ledger.requirements[-1]
        self.assertEqual(new.id, "R3")
        self.assertEqual(new.covered_by, ["S4"])

    def test_covers_edits_on_steps_and_requirements(self):
        ledger = Ledger.from_request("make a plate")
        ledger.plan_set(requirements=[{"id": "R1", "text": ledger.requirements[0].text, "covers": ["S2"]}])
        self.assertEqual(ledger.requirement("R1").covered_by, ["S2"])
        ledger.plan_set(steps=[{"id": "S3", "title": "Model the part", "covers": ["R1"]}])
        self.assertEqual(ledger.step("S3").covers, ["R1"])
        warnings = ledger.plan_set(steps=[{"id": "S3", "title": "Model the part", "covers": ["R42"]}])
        self.assertTrue(any("R42" in w for w in warnings))

    def test_replace_retires_open_model_steps(self):
        ledger = Ledger.from_request("make a plate")
        ledger.plan_set(steps=[{"title": "Sketch the outline", "phase": "BUILD"}], replace=True)
        old = ledger.step("S3")
        self.assertEqual((old.status, old.note, old.required), ("changed", "replaced by the model's plan", False))
        self.assertTrue(any(s.title == "Sketch the outline" and s.required for s in ledger.steps))

    def test_app_owned_steps_are_protected(self):
        ledger = Ledger.from_request("make a plate")
        warnings = ledger.plan_set(steps=[{"id": "S4", "title": "Anything goes"}])
        self.assertTrue(any("S4" in w and "app" in w for w in warnings))
        self.assertEqual(ledger.step("S4").title, "Validate model")

    def test_unknown_step_without_a_title(self):
        ledger = Ledger.from_request("make a plate")
        with self.assertRaises(LedgerError):
            ledger.plan_set(steps=[{"id": "S77"}])

    def test_limits(self):
        ledger = Ledger.from_request("make a plate")
        with self.assertRaises(LedgerError):
            ledger.plan_set(steps=[{"title": f"step {i}"} for i in range(41)])

    def test_plan_set_on_an_empty_ledger_creates_a_plan(self):
        ledger = Ledger.from_request("hello?")
        ledger.plan_set(steps=[{"title": "Draw it", "phase": "BUILD"}])
        self.assertEqual([s.id for s in ledger.steps], ["S1"])
        self.assertTrue(ledger.is_work_turn())


class AutoTickTest(unittest.TestCase):
    def enclosure(self, text=PI_FULL):
        return Ledger.from_request(text)

    def test_a_failed_handler_ticks_nothing(self):
        ledger = self.enclosure()
        self.assertFalse(ledger.on_tool_result(oc("create_object", handler_ok=False, ok=False)))
        self.assertEqual(ledger.last_geometry_step, 0)

    def test_geometry_attaches_evidence_to_one_step_and_sets_it_running(self):
        ledger = Ledger.from_request("make a plate")
        self.assertTrue(ledger.on_tool_result(oc("create_object", step_index=4, title="Create Part::Box Plate")))
        step = ledger.step("S3")
        self.assertEqual(step.status, "running")
        self.assertEqual(len(step.evidence), 1)
        entry = step.evidence[0]
        self.assertEqual((entry["step_index"], entry["tool"], entry["kind"], entry["objects"]), (4, "create_object", "auto", ["Obj"]))
        self.assertEqual(ledger.last_geometry_step, 4)

    def test_evidence_does_not_complete_an_unstaged_step_by_itself(self):
        ledger = Ledger.from_request("make a plate")
        ledger.on_tool_result(oc("create_object", step_index=1))
        self.assertNotEqual(ledger.step("S3").status, "done")
        self.assertEqual(ledger.confirm_evidenced_steps(), ["S3"])
        self.assertEqual(ledger.step("S3").status, "done")
        self.assertEqual(ledger.step("S3").evidence[-1]["kind"], "confirmed")

    def test_health_failure_still_counts_as_work_but_cannot_be_confirmed(self):
        ledger = Ledger.from_request("make a plate")
        ledger.on_tool_result(oc("create_object", step_index=1, ok=False))     # handler ok, health found new errors
        self.assertEqual(ledger.last_geometry_step, 1)
        self.assertEqual(ledger.confirm_evidenced_steps(), [])
        ledger.on_tool_result(oc("edit_object", step_index=2))                   # a later clean call fixes it
        self.assertEqual(ledger.confirm_evidenced_steps(), ["S3"])

    def test_out_of_order_work_is_still_counted(self):
        ledger = self.enclosure()
        ledger.on_tool_result(oc("execute_code", step_index=1))                # built before sourcing anything
        self.assertTrue(any(e["kind"] == "auto" for s in ledger.steps for e in s.evidence))

    def test_build_enclosure_stages_complete_staged_steps(self):
        ledger = self.enclosure()
        ledger.on_tool_result(oc("build_enclosure", args={"stage": "shell"}, step_index=1))
        self.assertEqual(ledger.step("S5").stages_done, ["shell"])
        self.assertNotEqual(ledger.step("S5").status, "done")
        ledger.on_tool_result(oc("build_enclosure", args={"stage": "standoffs"}, step_index=2))
        self.assertEqual(ledger.step("S5").status, "done")
        ledger.on_tool_result(oc("build_enclosure", args={"stage": "ports"}, step_index=3))
        self.assertEqual(ledger.step("S6").status, "done")

    def test_next_stage_is_matched_on_the_resolved_name_only(self):
        ledger = self.enclosure()
        ledger.on_tool_result(oc("build_enclosure", args={"stage": "next"}, step_index=1))
        self.assertEqual(ledger.step("S5").stages_done, [])

    def test_nominal_reference_completes_the_source_steps(self):
        ledger = self.enclosure()
        ledger.on_tool_result(oc("source_reference_part", geometry=False, mutating=True,
                                 extra={"reference_result": {"status": "nominal", "id": "rpi4"}}))
        self.assertEqual(ledger.reference_status, "catalog-nominal")
        self.assertEqual(ledger.step("S2").status, "done")
        s3 = ledger.step("S3")
        self.assertEqual((s3.status, s3.note), ("skipped", flow.NO_STEP_NOTE))
        self.assertTrue(any(w["what"] == "S3" and w["by"] == "app" for w in ledger.waivers))

    def test_obtained_reference_keeps_the_status_until_the_import(self):
        ledger = self.enclosure()
        ledger.on_tool_result(oc("source_reference_part", geometry=False, mutating=True,
                                 extra={"reference_result": {"status": "obtained", "sha256": "ab" * 32}}))
        self.assertEqual(ledger.step("S2").status, "done")
        self.assertIn("sha256 abababababab", ledger.step("S2").note)
        self.assertEqual(ledger.reference_status, "none")
        ledger.on_tool_result(oc("import_reference_part", args={"source": "catalog"}, step_index=2,
                                 extra={"reference_result": {"origin": "catalog"}}))
        self.assertEqual(ledger.reference_status, "step-imported")
        ledger.on_tool_result(oc("measure_reference_part", geometry=False, mutating=True, step_index=3))
        self.assertEqual(ledger.reference_status, "step-measured")
        self.assertEqual(ledger.step("S3").status, "done")

    def test_a_user_attached_file_is_recorded_as_such(self):
        ledger = self.enclosure()
        ledger.on_tool_result(oc("import_reference_part", step_index=1, args={"path": "/tmp/pi4.step"}))
        self.assertEqual(ledger.reference_status, "user-attached")
        ledger.on_tool_result(oc("measure_reference_part", geometry=False, mutating=True, step_index=2))
        self.assertEqual(ledger.reference_status, "user-attached")           # a measurement does not upgrade it
        self.assertEqual(ledger.step("S3").status, "done")

    def test_a_declined_download_with_board_data_falls_back_to_nominal(self):
        for result in ({"status": "declined", "has_board": True}, {"status": "failed", "reference_status": "catalog-nominal"}):
            ledger = self.enclosure()
            ledger.on_tool_result(oc("source_reference_part", geometry=False, mutating=True, extra={"reference_result": result}))
            self.assertEqual(ledger.reference_status, "catalog-nominal", result)
            self.assertEqual(ledger.step("S2").status, "done")
            self.assertIn("nominal data used", ledger.step("S3").note)

    def test_declined_download_without_catalog_data_blocks_the_source_step(self):
        ledger = self.enclosure()
        ledger.on_tool_result(oc("source_reference_part", geometry=False, mutating=True,
                                 extra={"reference_result": {"status": "declined", "catalog": False, "reason": "user said no"}}))
        self.assertEqual(ledger.step("S2").status, "blocked")
        self.assertEqual(ledger.reference_status, "none")

    def test_params_step_needs_the_required_names_and_the_spec(self):
        ledger = self.enclosure()
        names = list(flow.ENCLOSURE_REQUIRED_PARAMS)
        ledger.on_tool_result(oc("set_parameters", geometry=False, mutating=True, step_index=1,
                                 extra={"param_names": names[:2]}))
        self.assertNotEqual(ledger.step("S4").status, "done")
        ledger.on_tool_result(oc("enclosure_spec", geometry=False, mutating=True, step_index=2,
                                 extra={"param_names": names, "spec_ok": True}))
        self.assertTrue(ledger.spec_ok)
        self.assertEqual(ledger.step("S4").status, "done")

    def test_deliver_tool_evidence_does_not_complete_the_deliver_step(self):
        ledger = Ledger.from_request("make a plate")
        ledger.on_tool_result(oc("export_model", geometry=False, mutating=True, step_index=5))
        self.assertNotEqual(ledger.step("S5").status, "done")


    def test_first_geometry_call_of_an_other_turn_escalates_first(self):
        ledger = Ledger.from_request("hello?")
        self.assertEqual(ledger.steps, [])
        ledger.on_tool_result(oc("create_object", step_index=1))
        self.assertEqual(ledger.intent, "generic_part")
        self.assertEqual([s.id for s in ledger.steps], ["S1", "S2", "S3", "S4", "S5"])
        self.assertEqual(ledger.step("S3").evidence[0]["step_index"], 1)      # the call counts toward the new template
        self.assertTrue(ledger.is_work_turn())

    def test_read_only_calls_never_escalate(self):
        ledger = Ledger.from_request("what is in this document?")
        ledger.on_tool_result(oc("get_objects", geometry=False, mutating=False))
        self.assertEqual((ledger.intent, ledger.steps), ("inspect", []))

    def test_escalate_is_idempotent_and_only_for_chat_intents(self):
        ledger = Ledger.from_request("how do I make a fillet?")
        ledger.escalate("generic_part")
        rev = ledger.revision
        ledger.escalate("generic_part")
        self.assertEqual(ledger.revision, rev)
        pi = self.enclosure()
        pi.escalate("generic_part")
        self.assertEqual(pi.intent, "enclosure")

    def test_an_open_fix_step_gets_the_next_geometry_call(self):
        ledger = Ledger.from_request("make a plate")
        ledger.on_tool_result(oc("create_object", step_index=1))
        ledger.confirm_evidenced_steps()
        ledger.apply_validation(good_validation(ok=False, gate_errors=[{"message": "Plate: 3 solids"}]))
        fix = ledger.step("F1")
        self.assertIsNotNone(fix)
        ledger.on_tool_result(oc("edit_object", step_index=2))
        self.assertEqual(fix.evidence[-1]["step_index"], 2)


class ConfirmAndValidationTest(unittest.TestCase):
    def built(self):
        ledger = Ledger.from_request("make a plate")
        ledger.on_tool_result(oc("create_object", step_index=1))
        ledger.confirm_evidenced_steps()
        return ledger

    def test_staged_steps_are_confirmed_only_when_every_stage_is_done(self):
        ledger = Ledger.from_request(PI_FULL)
        ledger.on_tool_result(oc("build_enclosure", args={"stage": "shell"}, step_index=1))
        self.assertEqual(ledger.confirm_evidenced_steps(), [])

    def test_a_claim_is_not_confirmed(self):
        ledger = Ledger.from_request("make a plate")
        ledger.plan_update("S3", "done", "I built it")
        self.assertEqual(ledger.confirm_evidenced_steps(), [])
        self.assertEqual(ledger.claimed_only_steps()[0].id, "S3")

    def test_a_claim_is_backed_by_later_recorded_work(self):
        ledger = Ledger.from_request("make a plate")
        ledger.plan_update("S3", "done", "I built it")
        ledger.on_tool_result(oc("create_object", step_index=1))
        self.assertEqual(ledger.claimed_only_steps(), [])

    def test_passing_validation_completes_verify_and_sets_the_facts(self):
        ledger = self.built()
        ledger.apply_validation(good_validation(after_step=1))
        self.assertEqual(ledger.step("S4").status, "done")
        self.assertTrue(ledger.health_ok)
        self.assertEqual(ledger.validated_after_step, 1)
        self.assertEqual(ledger.open_problems, [])

    def test_failing_validation_regresses_to_a_fix_step(self):
        ledger = self.built()
        ledger.apply_validation(good_validation(ok=False, gate_errors=[{"message": "Plate: 3 solids, expected 1"}]))
        self.assertFalse(ledger.health_ok)
        self.assertIn("Plate: 3 solids, expected 1", ledger.open_problems[0])
        fix = ledger.step("F1")
        self.assertEqual((fix.phase, fix.owner, fix.required, fix.template), ("BUILD", "model", True, False))
        self.assertTrue(fix.title.startswith("Fix validation findings: Plate"))
        self.assertEqual(ledger.step("S4").status, "pending")
        self.assertEqual([s.id for s in ledger.steps].index("F1") + 1, [s.id for s in ledger.steps].index("S4"))
        self.assertEqual(ledger.phase, "BUILD")
        self.assertIn(ledger.max_phase, ("BUILD", "VERIFY", "DELIVER"))

    def test_regress_is_idempotent_while_the_fix_step_is_open(self):
        ledger = self.built()
        bad = good_validation(ok=False, fail_rows=[{"id": "usb_c", "detail": "interference 0.3 mm"}])
        ledger.apply_validation(bad)
        ledger.apply_validation(bad)
        self.assertEqual([s.id for s in ledger.steps if s.id.startswith("F")], ["F1"])

    def test_a_later_pass_closes_the_fix_step(self):
        ledger = self.built()
        ledger.apply_validation(good_validation(ok=False, gate_errors=[{"message": "x"}]))
        ledger.apply_validation(good_validation(after_step=2))
        self.assertEqual(ledger.step("F1").status, "done")
        self.assertEqual(ledger.step("S4").status, "done")

    def test_waived_fail_rows_do_not_regress(self):
        ledger = self.built()
        ledger.archive_waiver("usb_c", "tolerance accepted by the user", "user")
        ledger.apply_validation(good_validation(ok=False, fail_rows=[{"id": "usb_c", "detail": "x"}]))
        self.assertIsNone(ledger.step("F1"))
        self.assertEqual(ledger.step("S4").status, "done")

    def test_no_validation_skips_verify_without_a_claim(self):
        ledger = self.built()
        ledger.apply_validation(None)
        s = ledger.step("S4")
        self.assertEqual((s.status, s.note), ("skipped", "automatic validation unavailable"))

    def test_mark_delivered(self):
        ledger = self.built()
        ledger.mark_delivered()
        self.assertEqual(ledger.step("S5").status, "done")
        other = self.built()
        other.mark_delivered(project_bound=False)
        self.assertEqual((other.step("S5").status, other.step("S5").note), ("skipped", "no project bound"))


class PhaseTest(unittest.TestCase):
    def test_empty_ledger_is_intake(self):
        self.assertEqual(derive_phase(Ledger.from_request("hello?")), "INTAKE")

    def test_phase_follows_the_current_step(self):
        ledger = Ledger.from_request(PI_FULL)
        self.assertEqual(derive_phase(ledger), "SOURCE")
        ledger.plan_update("S2", "skipped", "no network")
        ledger.plan_update("S3", "skipped", "no network")
        self.assertEqual(derive_phase(ledger), "PARAMS")
        ledger.plan_update("S5", "running")
        self.assertEqual(derive_phase(ledger), "BUILD")

    def test_all_required_terminal_is_deliver(self):
        ledger = Ledger.from_request("make a plate")
        ledger.on_tool_result(oc("create_object", step_index=1))
        ledger.confirm_evidenced_steps()
        ledger.apply_validation(good_validation())
        ledger.mark_delivered()
        self.assertEqual(derive_phase(ledger), "DELIVER")

    def test_visible_phases_without_steps_is_every_phase(self):
        self.assertEqual(visible_phases(Ledger.from_request("hello?")), frozenset(flow.PHASES))

    def test_visible_phases_grow_monotonically_and_never_shrink(self):
        ledger = Ledger.from_request(PI_FULL)
        seen = []
        seen.append(visible_phases(ledger))
        self.assertEqual(seen[-1], frozenset({"INTAKE", "SOURCE"}))
        ledger.plan_update("S2", "skipped", "x")
        ledger.plan_update("S3", "skipped", "x")
        seen.append(visible_phases(ledger))
        ledger.plan_update("S4", "skipped", "x")
        seen.append(visible_phases(ledger))
        self.assertIn("BUILD", seen[-1])              # BUILD is offered from PLAN on, while the plan is still being refined
        ledger.regress("something broke")
        seen.append(visible_phases(ledger))
        for before, after in zip(seen, seen[1:]):
            self.assertTrue(before <= after, (before, after))
        ledger.plan_update("S5", "running")
        self.assertTrue(seen[-1] <= visible_phases(ledger))

    def test_max_phase_never_decreases(self):
        ledger = Ledger.from_request("make a plate")
        ledger.on_tool_result(oc("create_object", step_index=1))
        ledger.confirm_evidenced_steps()
        ledger.apply_validation(good_validation())
        high = ledger.max_phase
        ledger.apply_validation(good_validation(ok=False, gate_errors=[{"message": "x"}]))
        self.assertEqual(ledger.phase, "BUILD")
        self.assertEqual(ledger.max_phase, high)

    def test_phase_exit_met(self):
        ledger = Ledger.from_request(PI_FULL)
        self.assertEqual(phase_exit_met(ledger, "INTAKE"), (True, ""))
        met, why = phase_exit_met(ledger, "SOURCE")
        self.assertFalse(met)
        self.assertIn("S2", why)
        ledger.plan_update("S2", "skipped", "x")
        ledger.plan_update("S3", "skipped", "x")
        self.assertTrue(phase_exit_met(ledger, "SOURCE")[0])
        self.assertTrue(phase_exit_met(ledger, "PLAN")[0])
        ledger.plan_set(requirements=[{"text": "A requirement no step covers yet", "covers": []}])
        self.assertTrue(phase_exit_met(ledger, "PLAN")[0])      # unclaimed requirements are verified at VERIFY
        self.assertFalse(phase_exit_met(Ledger.from_request("hello?"), "INTAKE")[0])


def state(**kw):
    base = dict(is_work_turn=True, geometry_ok=1, deliver_ok=0, open_errors=[], last_script_fail="", validation=None,
                last_geometry_step=1, gate_nudges={}, nudges_total=0)
    base.update(kw)
    return TurnState(**base)


def built_plate():
    ledger = Ledger.from_request("make a plate")
    ledger.on_tool_result(oc("create_object", step_index=1))
    ledger.confirm_evidenced_steps()
    return ledger


class FinishGateTest(unittest.TestCase):
    cfg = AgentConfig()

    def test_a_non_work_turn_always_passes(self):
        verdict = finish_gate(Ledger.from_request("hello?"), state(is_work_turn=False, geometry_ok=0), self.cfg)
        self.assertTrue(verdict.ok)
        self.assertEqual((verdict.codes, verdict.nudge, verdict.exhausted), ([], "", False))

    def test_finished_work_passes(self):
        ledger = built_plate()
        self.assertTrue(finish_gate(ledger, state(validation=good_validation()), self.cfg).ok)

    def test_no_work(self):
        ledger = Ledger.from_request("make a plate")
        verdict = finish_gate(ledger, state(geometry_ok=0, last_geometry_step=0), self.cfg)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.codes[0], "NO_WORK")
        self.assertIn("you made no change in the model; this request needs geometry", verdict.nudge)
        self.assertTrue(verdict.nudge.startswith("[finish-gate] You stopped, but the app found unfinished work:"))
        self.assertIn('Continue with S3 now by calling the tools.', verdict.nudge)
        self.assertTrue(verdict.nudge.endswith("Do not write a summary yet."))
        self.assertIsNone(verdict.tool_choice)

    def test_the_second_no_work_nudge_forces_a_tool(self):
        ledger = Ledger.from_request("make a plate")
        verdict = finish_gate(ledger, state(geometry_ok=0, last_geometry_step=0, gate_nudges={"NO_WORK": 1}, nudges_total=1),
                              self.cfg, offered={"create_object", "execute_code", "get_objects"})
        self.assertFalse(verdict.exhausted)
        self.assertEqual(verdict.tool_choice, "create_object")
        verdict = finish_gate(ledger, state(geometry_ok=0, last_geometry_step=0, gate_nudges={"NO_WORK": 1}, nudges_total=1),
                              self.cfg, offered={"get_objects"})
        self.assertEqual(verdict.tool_choice, "required")

    def test_no_work_is_exhausted_after_two_nudges_of_the_same_code(self):
        ledger = Ledger.from_request("make a plate")
        verdict = finish_gate(ledger, state(geometry_ok=0, last_geometry_step=0, gate_nudges={"NO_WORK": 2}, nudges_total=2),
                              self.cfg)
        self.assertTrue(verdict.exhausted)
        self.assertEqual(verdict.nudge, "")

    def test_total_budget(self):
        ledger = Ledger.from_request("make a plate")
        verdict = finish_gate(ledger, state(geometry_ok=0, last_geometry_step=0, gate_nudges={"NO_WORK": 1}, nudges_total=3),
                              self.cfg)
        self.assertTrue(verdict.exhausted)

    def test_blocked_step_with_a_note_and_no_work_needs_input(self):
        ledger = Ledger.from_request(PI_FULL)
        ledger.plan_update("S2", "blocked", "I need the STEP file: please attach it")
        verdict = finish_gate(ledger, state(geometry_ok=0, last_geometry_step=0), self.cfg, reply_nonempty=True)
        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.outcome_hint, "needs_input")
        empty = finish_gate(ledger, state(geometry_ok=0, last_geometry_step=0), self.cfg, reply_nonempty=False)
        self.assertFalse(empty.ok)

    def test_deliverable_missing(self):
        ledger = Ledger.from_request("export it as STL")
        verdict = finish_gate(ledger, state(geometry_ok=0, deliver_ok=0, last_geometry_step=0), self.cfg)
        self.assertEqual(verdict.codes[0], "DELIVERABLE_MISSING")
        self.assertIn("the requested file was not produced (no successful export_model / save_document)", verdict.nudge)
        ledger2 = Ledger.from_request("export it as STL")
        ledger2.plan_update("S1", "done", "exported")
        ok = finish_gate(ledger2, state(geometry_ok=0, deliver_ok=1, last_geometry_step=0), self.cfg)
        self.assertNotIn("DELIVERABLE_MISSING", ok.codes)

    def test_open_errors(self):
        verdict = finish_gate(built_plate(), state(open_errors=["execute_code: Boom"]), self.cfg)
        self.assertEqual(verdict.codes, ["OPEN_ERRORS"])
        self.assertIn("unresolved errors from your last tool calls: execute_code: Boom", verdict.nudge)
        self.assertIn("Fix the items above by calling the tools.", verdict.nudge)

    def test_health_fail_uses_the_validation_gate_errors(self):
        validation = good_validation(ok=False, gate_errors=[{"message": "Shell: 3 solids (18000.0, 41.2, 12.9 mm3), expected 1"}])
        verdict = finish_gate(built_plate(), state(validation=validation), self.cfg)
        self.assertEqual(verdict.codes, ["HEALTH_FAIL"])
        self.assertIn("- HEALTH_FAIL: Shell: 3 solids", verdict.nudge)

    def test_script_fail(self):
        verdict = finish_gate(built_plate(), state(last_script_fail="FAIL: hole misses the wall"), self.cfg)
        self.assertEqual(verdict.codes, ["SCRIPT_FAIL"])
        self.assertIn("your last script printed a failure line: 'FAIL: hole misses the wall'. Treat it as a failed check "
                      "and fix the cause", verdict.nudge)

    def test_fit_fail_and_waivers(self):
        validation = good_validation(ok=False, fail_rows=[{"id": "usb_c"}, {"id": "hdmi0"}])
        verdict = finish_gate(built_plate(), state(validation=validation), self.cfg)
        self.assertEqual(verdict.codes, ["FIT_FAIL"])
        self.assertIn("check_fit reports FAIL: usb_c, hdmi0. Fix the geometry or the parameter (set_parameter) and run "
                      "check_fit again", verdict.nudge)
        ledger = built_plate()
        ledger.archive_waiver("usb_c", "accepted", "user")
        ledger.archive_waiver("hdmi0", "accepted", "user")
        self.assertTrue(finish_gate(ledger, state(validation=validation), self.cfg).ok)

    def test_steps_pending_lists_the_steps_with_details(self):
        ledger = Ledger.from_request(PI_FULL)
        ledger.plan_update("S2", "skipped", "x")
        ledger.plan_update("S3", "skipped", "x")
        ledger.on_tool_result(oc("build_enclosure", args={"stage": "shell"}, step_index=1))
        verdict = finish_gate(ledger, state(), self.cfg)
        self.assertIn("STEPS_PENDING", verdict.codes)
        text = verdict.nudge
        self.assertIn('S5 "Bottom shell: floor, walls, standoffs" has stages still to build: standoffs', text)
        self.assertIn('S6 "Port cutouts for every connector" has no recorded work', text)
        self.assertNotIn("S11", text)                    # app-owned steps are not listed
        self.assertIn("If the geometry for a step already exists, call plan_update(id, \"done\", note=\"<where it is>\")", text)

    def test_claim_only(self):
        ledger = built_plate()
        ledger.plan_set(steps=[{"title": "Round the corners", "phase": "BUILD"}])
        ledger.plan_update("S6", "done", "I did it")
        verdict = finish_gate(ledger, state(), self.cfg)
        self.assertEqual(verdict.codes, ["CLAIM_ONLY"])
        self.assertIn("S6 were marked done without any recorded work; do the work with tools, or call plan_update(id, "
                      '"skipped" or "blocked", note="<reason>")', verdict.nudge)
        done = finish_gate(ledger, state(gate_nudges={"CLAIM_ONLY": 2}, nudges_total=2), self.cfg)
        self.assertTrue(done.exhausted)
        self.assertEqual(done.outcome_hint, "claims")
        self.assertIn("S6 claimed without recorded work", done.reasons[0])

    def test_reference_unresolved(self):
        ledger = Ledger.from_request(PI_FULL)
        ledger.on_tool_result(oc("execute_code", step_index=1))              # built without ever sourcing the part
        verdict = finish_gate(ledger, state(), self.cfg)
        self.assertIn("REFERENCE_UNRESOLVED", verdict.codes)
        self.assertIn("the reference part has no recorded status; call source_reference_part, or plan_update S2 with a reason",
                      verdict.nudge)

    def test_reference_blocked_by_the_app_is_unresolved_but_a_model_decision_is_not(self):
        app = Ledger.from_request(PI_FULL)
        app.on_tool_result(oc("source_reference_part", geometry=False, mutating=True,
                              extra={"reference_result": {"status": "failed", "catalog": False, "reason": "offline"}}))
        app.on_tool_result(oc("execute_code", step_index=1))
        self.assertIn("REFERENCE_UNRESOLVED", finish_gate(app, state(), self.cfg).codes)
        model = Ledger.from_request(PI_FULL)
        model.plan_update("S2", "skipped", "user prefers nominal dimensions")
        model.on_tool_result(oc("execute_code", step_index=1))
        self.assertNotIn("REFERENCE_UNRESOLVED", finish_gate(model, state(), self.cfg).codes)

    def test_codes_come_in_rule_order(self):
        ledger = Ledger.from_request("make a plate")
        verdict = finish_gate(ledger, state(geometry_ok=0, last_geometry_step=0, open_errors=["x: y"],
                                            last_script_fail="FAIL: z"), self.cfg)
        self.assertEqual(verdict.codes, ["NO_WORK", "OPEN_ERRORS", "SCRIPT_FAIL", "STEPS_PENDING"])

    def test_a_continuation_with_no_new_geometry_does_not_owe_work(self):
        ledger = built_plate()
        verdict = finish_gate(ledger, state(geometry_ok=0, last_geometry_step=1), self.cfg)
        self.assertNotIn("NO_WORK", verdict.codes)

    def test_verdict_type(self):
        self.assertIsInstance(finish_gate(built_plate(), state(), self.cfg), GateVerdict)


class ComposeTest(unittest.TestCase):
    def test_unverified_text_starts_with_the_banner(self):
        text = flow.compose_unverified(["S6 claimed without recorded work", "Shell: 3 solids"], "All done!", "## Build Summary\n...")
        self.assertTrue(text.startswith("NOT VERIFIED: this turn ended with items the app could not close:\n- S6 claimed"))
        self.assertIn("- Shell: 3 solids\nThe summary below is the model's own account; treat any claim of success in it as "
                      "unverified.", text)
        self.assertIn("\nAll done!\n\n## Build Summary", text)
        self.assertFalse(flow.compose_unverified(["x"], "").endswith("\n"))

    def test_capped_text(self):
        text = flow.compose_capped("Wrapped up.", 30, ["S6", "S7"], "## Build Summary")
        self.assertTrue(text.startswith("Wrapped up.\n\n(Stopped after 30 tool rounds with 2 required steps open: S6, S7. "
                                        "Press Continue to keep going.)"))
        self.assertTrue(text.endswith("## Build Summary"))
        self.assertTrue(flow.compose_capped("", 3, ["S3"]).startswith("(Stopped after 3 tool rounds"))

    def test_needs_help_text(self):
        text = flow.compose_needs_help("the same call kept failing.")
        self.assertEqual(text, 'Oscar is stuck: the same call kept failing. What I need from you: attach the missing file, '
                               'relax a requirement, or say "skip it".')
        self.assertIn("Oscar is stuck: the model kept failing.", flow.compose_needs_help(""))

    def test_failed_text(self):
        text = flow.compose_failed(["the model made no change in the document although the request needs geometry"])
        self.assertTrue(text.startswith("Oscar made no changes in the model for this request"))
        self.assertIn("the model made no change", text.splitlines()[-1])
        self.assertTrue(flow.compose_failed(["x"], "The model tried to call a tool but the call arrived incomplete.")
                        .startswith("The model tried to call a tool"))


class GovernorTest(unittest.TestCase):
    def test_grammar(self):
        ledger = Ledger.from_request(PI_FULL)
        ledger.plan_update("S2", "skipped", "x")
        ledger.plan_update("S3", "skipped", "x")
        ledger.plan_update("S4", "skipped", "x")
        ledger.plan_update("S7", "running")
        line = format_governor(ledger, round_no=18, cap=30, health="OK")
        self.assertTrue(line.startswith("[governor] round 18/30 · BUILD · steps 1/12 done · now: S7 \"Lid with lip\" · pending: "))
        self.assertIn(" · pending: S5, S6, S8, S9, S10", line)
        self.assertTrue(line.endswith("· health OK"))
        self.assertNotIn("\n", line)

    def test_health_labels(self):
        self.assertEqual(flow.health_label(None), "?")
        self.assertEqual(flow.health_label({}), "?")
        self.assertEqual(flow.health_label({"d": {"valid": True, "counts": {"errors": 0}}}), "OK")
        self.assertEqual(flow.health_label({"d": {"valid": False, "counts": {"errors": 2}}}), "2 ERR")
        self.assertEqual(flow.health_label({"d": {"valid": False}}), "1 ERR")

    def test_unknown_health_before_the_first_report(self):
        line = format_governor(Ledger.from_request("make a plate"), round_no=1, cap=30)
        self.assertTrue(line.endswith("health ?"))

    def test_pending_ids_are_capped_with_a_plus_n(self):
        ledger = Ledger.from_request(PI_FULL)
        line = format_governor(ledger, round_no=1, cap=30)
        self.assertRegex(line, r"pending: S3, S4, S5, S6, S7, S8 \+\d+")

    def test_eighty_percent_reminder(self):
        ledger = Ledger.from_request("make a plate")
        self.assertNotIn("80%", format_governor(ledger, round_no=23, cap=30))
        self.assertIn("80% of the round budget used: finish the required steps, then stop and report.",
                      format_governor(ledger, round_no=24, cap=30))

    def test_phase_line_on_a_phase_change(self):
        ledger = Ledger.from_request("make a plate")
        text = format_governor(ledger, round_no=2, cap=30, phase_changed=True)
        first, second = text.split("\n")
        self.assertTrue(first.startswith("[governor]"))
        self.assertTrue(second.startswith(f"[phase] {ledger.phase}: "))
        self.assertLessEqual(len(second.split(": ", 1)[1].split()), 80)

    def test_line_is_at_most_400_chars(self):
        ledger = Ledger.from_request("make a plate")
        ledger.plan_set(steps=[{"title": "x" * 110, "phase": "BUILD"} for _ in range(12)])
        self.assertLessEqual(len(format_governor(ledger, round_no=1, cap=30).split("\n")[0]), 400)

    def test_every_phase_has_guidance(self):
        for phase in flow.PHASES:
            self.assertTrue(flow.phase_guidance(phase))
        self.assertEqual(flow.phase_guidance("NOPE"), "")


class AgentConfigTest(unittest.TestCase):
    def test_defaults_are_the_decision_values(self):
        cfg = AgentConfig()
        self.assertEqual((cfg.max_tool_rounds, cfg.max_rounds_hard, cfg.raise_step, cfg.warn_fraction), (30, 60, 10, 0.8))
        self.assertEqual((cfg.cutoff_retries, cfg.gate_nudges_max, cfg.breaker_trips_max, cfg.stall_rounds), (2, 3, 3, 6))
        self.assertEqual((cfg.recovery_notes_max, cfg.reasoning_plan, cfg.reasoning_exec, cfg.exec_max_tokens),
                         (12, "medium", "low", 12000))
        self.assertTrue(cfg.auto_validate and cfg.steer and cfg.flow_enforcement)

    def test_instances_do_not_share_the_watchdog_config(self):
        a, b = AgentConfig(), AgentConfig()
        a.watchdog.silent_s = 1.0
        self.assertEqual(b.watchdog.silent_s, 90.0)

    def test_outcomes_constant(self):
        self.assertEqual(flow.OUTCOMES, ("answered", "validated", "unverified", "failed", "capped", "stopped",
                                         "needs_help", "needs_input"))


class ReviewRegressionTest(unittest.TestCase):
    """Defects found by an earlier review (each test names the failure it pins)."""

    def pi(self):
        return Ledger.from_request(PI_FULL)

    def test_render_compact_survives_any_line_budget(self):
        ledger = self.pi()
        for max_lines in (-1, 0, 1, 2, 3, 4, 5, 8, 16, 100):
            text = ledger.render_compact(max_lines)
            self.assertTrue(text.startswith("PLAN ("))
            self.assertLessEqual(len(text.splitlines()), max(max_lines, 3))

    def test_render_compact_keeps_the_current_step_when_it_can(self):
        ledger = self.pi()
        text = ledger.render_compact(4)
        self.assertIn("<- now", text)
        self.assertIn("+11 more", text)

    def test_plan_set_applies_covers_to_a_new_step_without_an_id(self):
        ledger = self.pi()
        ledger.plan_set(None, [{"title": "Check the strain relief", "phase": "BUILD", "covers": ["R1", "R2"]}])
        new = ledger.step("S13")
        self.assertEqual((new.title, new.covers), ("Check the strain relief", ["R1", "R2"]))
        self.assertIn("S13", ledger.requirement("R1").covered_by)

    def test_plan_set_applies_covers_to_a_new_requirement_without_an_id(self):
        ledger = self.pi()
        ledger.plan_set([{"text": "The lid opens with one hand", "covers": ["S7"]}])
        req = ledger.requirements[-1]
        self.assertEqual((req.id, req.covered_by), ("R10", ["S7"]))

    def test_plan_set_accepts_covers_as_a_plain_string(self):
        ledger = self.pi()
        warnings = ledger.plan_set(None, [{"title": "Extra", "covers": "R1, R2"}])
        self.assertEqual(ledger.step("S13").covers, ["R1", "R2"])
        self.assertEqual(warnings, [])
        warnings = ledger.plan_set([{"id": "R1", "covers": "S5 S6"}])
        self.assertEqual(sorted(s.id for s in ledger.steps if "R1" in s.covers), ["S5", "S6"])

    def test_from_dict_tolerates_a_garbled_version(self):
        for version in ("x", None, [], {}):
            ledger = Ledger.from_dict({"version": version, "intent": "enclosure"})
            self.assertEqual(ledger.version, Ledger.VERSION)

    def test_a_spec_call_alone_never_stands_in_for_modelling_work(self):
        ledger = Ledger.from_request("Design a bracket for a NEMA17 motor with 4 mounting holes")
        before = [(s.id, len(s.evidence)) for s in ledger.steps]
        ledger.on_tool_result(ToolOutcome(name="enclosure_spec", ok=True, handler_ok=True, mutating=True, geometry=False))
        self.assertEqual([(s.id, len(s.evidence)) for s in ledger.steps], before)
        ledger.on_tool_result(ToolOutcome(name="create_object", ok=True, handler_ok=True, mutating=True, geometry=True,
                                          step_index=1, delta={"added": ["Body"]}))
        self.assertEqual(len(ledger.step("S3").evidence), 1)

    def test_a_spec_call_still_credits_the_step_that_lists_it(self):
        ledger = self.pi()
        ledger.on_tool_result(ToolOutcome(name="enclosure_spec", ok=True, handler_ok=True, mutating=True, geometry=False,
                                          extra={"spec_ok": True}))
        self.assertEqual([e["tool"] for e in ledger.step("S4").evidence], ["enclosure_spec"])

    def test_the_governor_title_is_cut_with_an_ellipsis_not_an_elision_marker(self):
        ledger = Ledger.from_request("make a plate")
        ledger.plan_set(None, [{"id": "S3", "title": "Model " + "the very long plate " * 12}])
        line = format_governor(ledger, round_no=2, cap=30)
        self.assertNotIn("omitted", line)
        self.assertIn("…", line)
        self.assertLessEqual(len(line.splitlines()[0]), 400)

    def test_flow_run_names_are_still_reachable_from_flow(self):
        for name in ("AgentConfig", "format_governor", "health_label", "phase_guidance", "compose_unverified",
                     "compose_capped", "compose_needs_help", "compose_failed", "NOT_VERIFIED_HEADER",
                     "NOT_VERIFIED_FOOTER", "NO_CHANGES_TEXT"):
            self.assertTrue(hasattr(flow, name), name)
        self.assertEqual(sorted(n for n in flow.__all__ if not hasattr(flow, n)), [])


class PhaseGuidanceWiringTest(unittest.TestCase):
    """The guidance of the ``[phase]`` line is ``prompt.PHASE_GUIDANCE``; the built-in text is only a fallback."""

    def test_every_phase_uses_the_prompt_text(self):
        from oscar import prompt

        for phase in flow.PHASES:
            self.assertEqual(flow.phase_guidance(phase), prompt.PHASE_GUIDANCE[phase])
            self.assertTrue(flow.phase_guidance(phase))

    def test_the_governor_phase_line_carries_the_prompt_text(self):
        from oscar import prompt

        ledger = Ledger.from_request("make a plate")
        line = format_governor(ledger, round_no=1, cap=30, phase_changed=True)
        self.assertTrue(line.splitlines()[1].startswith(f"[phase] {ledger.phase}: "))
        self.assertIn(prompt.PHASE_GUIDANCE[ledger.phase], line)

    def test_the_fallback_serves_a_missing_prompt_module_and_an_unknown_phase_is_empty(self):
        import importlib
        from unittest import mock

        from oscar import flow_run

        def missing(name, *args, **kwargs):
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)

        with mock.patch.object(importlib, "import_module", missing):
            self.assertEqual(flow.phase_guidance("BUILD"), flow_run._FALLBACK_GUIDANCE["BUILD"])
            self.assertEqual(flow.phase_guidance("NOPE"), "")


if __name__ == "__main__":
    unittest.main()
