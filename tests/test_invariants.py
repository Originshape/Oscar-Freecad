"""The four invariants as executable assertions.

Every test drives ``AgentWorker.run_turn`` with the REAL ``tools.run_tool`` (so the Project hook, the health hook, the
ledger tick and the finish gate all run for real), a REAL ``Project`` in the test sandbox and REAL FreeCAD documents. The
model is a script: it is the only fake, and the scripts are deliberately uncooperative (no ``step_title``, no notes, no
plan updates) because the invariants must hold WITHOUT model cooperation.

* I1: every mutating step yields a step record, a saved script, a snapshot, an object delta and a health result.
* I2: the final answer is model commentary plus the app-generated Build Summary; validated / NOT validated comes from
  RECORDED checks only.
* I3: no turn ends "success" with unresolved required steps or failing health unless blocked / waived and shown.
* I4: whatever is claimed as sourced has a recorded source, hash and status; overclaiming is impossible while the
  reference is catalog-nominal.
"""

import copy
import json
import os
import re
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402

from oscar import netguard  # noqa: E402
from oscar.cad import health as cad_health  # noqa: E402
from oscar.flow import Ledger  # noqa: E402
from oscar.project import REFERENCE_STATUSES  # noqa: E402
from oscar.refparts import catalog  # noqa: E402
from support import personas  # noqa: E402
from support.fakes import fake_validation  # noqa: E402
from support.realflow import PI_PROMPT, RealFlowCase, sha256_of  # noqa: E402
from support.scripted_llm import call, calls, say  # noqa: E402

CYLINDER = "o = d.addObject('Part::Feature', 'Cyl')\no.Shape = Part.makeCylinder(3, 10)\nd.recompute()"
WEAK_DONE = "All steps are done."


def broken_cut_script(case, doc: str, name: str = "MyCut"):
    """A script that succeeds but leaves an invalid ``Part::Cut`` behind (the health hook flags it)."""
    return call("execute_code", include_screenshot=False,
                code=case.code(doc, f"d.addObject('Part::Cut', '{name}')\nd.recompute()"))


# --------------------------------------------------------------------------------------------------------------
# I1


class I1StepRecordsTest(RealFlowCase):
    """Every mutating step: record + script + snapshot + delta + health, with no model cooperation."""

    def build(self, title_arg=None):
        self.doc = self.doc_name("I1")
        self.project = self.new_project("I1 plate", chat_id="20261002-101010-a1a1a1")
        cyl = call("execute_code", code=self.code(self.doc, CYLINDER), include_screenshot=False)
        box = self.box(self.doc)
        if title_arg is not None:
            box.arguments["step_title"] = title_arg
        script = [calls(call("create_document", name=self.doc)),
                  calls(box, cyl, call("get_objects", doc_name=self.doc)),
                  say("Built a plate and a cylinder.")]
        self.result = self.turn("make a plate and a cylinder", script, project=self.project)
        self.steps = self.project.read_steps()
        return self.steps

    def test_the_contract_example_yields_exactly_the_recorded_steps(self):
        steps = self.build()
        self.assertEqual(self.result.outcome, "validated")
        self.assertEqual([s.tool for s in steps], ["create_document", "create_object", "execute_code"])   # get_objects is read-only
        self.assertEqual([s.index for s in steps], [1, 2, 3])
        self.assertEqual((steps[0].snapshot, steps[0].checkpoint), ("", ""))       # mutating, but not a geometry tool
        self.assertFalse(steps[0].geometry)
        self.assertTrue(all(s.ok for s in steps))

    def test_every_geometry_step_has_title_delta_health_snapshot_and_checkpoint(self):
        steps = self.build()
        for s in steps[1:]:
            with self.subTest(step=s.index, tool=s.tool):
                self.assertTrue(s.geometry)
                self.assertTrue(s.title and len(s.title) <= 80, s.title)
                self.assertIsInstance(s.delta, dict)
                self.assertTrue(s.delta.get("added"))
                self.assertTrue(s.health and "valid" in s.health)
                self.assertTrue(s.health["line"].startswith("[health]"))
                self.assertTrue(s.snapshot or s.snapshot_note)
                if s.snapshot:
                    self.assertGreater(os.path.getsize(self.project.path(s.snapshot)), 0)
                self.assertTrue(s.checkpoint)
                self.assertTrue(os.path.isfile(self.project.path("checkpoints", s.checkpoint + ".FCStd")))

    def test_the_delta_names_exactly_what_the_step_added(self):
        steps = self.build()
        self.assertEqual(steps[1].delta["added"], ["Plate"])
        self.assertEqual(steps[2].delta["added"], ["Cyl"])
        self.assertEqual((steps[1].delta["removed"], steps[2].delta["removed"]), ([], []))

    def test_the_script_is_saved_verbatim_with_a_header(self):
        steps = self.build()
        path = self.project.path(steps[2].script)
        self.assertRegex(steps[2].script, r"^scripts/003_.+\.py$")
        with open(path, encoding="utf-8") as f:
            saved = f.read()
        self.assertIn(CYLINDER, saved)
        self.assertTrue(saved.lstrip().startswith("#"), "the saved script starts with a header comment")
        self.assertEqual(steps[1].script, "")                     # a typed tool call has no script file (its args replay it)

    def test_the_working_model_is_saved_next_to_the_dossier(self):
        self.build()
        self.assertTrue(os.path.isfile(self.project.path("model.FCStd")))
        self.assertTrue(os.path.isfile(self.project.path("steps.jsonl")))

    def test_the_model_title_is_clipped_to_80_characters_and_an_empty_one_is_derived(self):
        steps = self.build(title_arg="T" * 300)
        self.assertLessEqual(len(steps[1].title), 80)
        self.assertTrue(steps[1].title.startswith("T"))
        steps = self.build(title_arg="")
        self.assertIn("Part::Box", steps[1].title)                  # derived from the arguments

    def test_the_health_of_each_step_describes_the_document_after_that_step(self):
        steps = self.build()
        self.assertIn("1 objects, 1 solids", steps[1].health["line"])
        self.assertIn("2 objects, 2 solids", steps[2].health["line"])
        self.assertEqual((steps[1].health["errors"], steps[2].health["errors"]), (0, 0))

    def test_a_failed_script_is_recorded_without_snapshot_or_checkpoint(self):
        doc = self.doc_name("I1f")
        project = self.new_project("I1 failed", chat_id="20261002-101010-a1a1a2")
        bad = call("execute_code", code=self.code(doc, "raise RuntimeError('boom')"), include_screenshot=False)
        script = [calls(call("create_document", name=doc)), calls(bad), calls(self.box(doc)), say("Done.")]
        self.turn("make a plate", script, project=project)
        steps = project.read_steps()
        failed = steps[1]
        self.assertEqual((failed.tool, failed.ok), ("execute_code", False))
        self.assertEqual(failed.snapshot, "")
        self.assertEqual(failed.snapshot_note, "step failed")
        self.assertEqual(failed.checkpoint, "")
        self.assertIn("boom", failed.result_head)
        self.assertEqual(failed.delta["added"], [])
        self.assertTrue(steps[2].ok and steps[2].checkpoint)

    def test_parallel_calls_of_one_round_are_recorded_in_call_order(self):
        doc = self.doc_name("I1p")
        project = self.new_project("I1 parallel", chat_id="20261002-101010-a1a1a3")
        script = [calls(call("create_document", name=doc)),
                  calls(self.box(doc, "A"), self.box(doc, "B"), self.box(doc, "C")), say("Done.")]
        self.turn("make three boxes", script, project=project)
        steps = project.read_steps()
        self.assertEqual([s.delta["added"] for s in steps[1:]], [["A"], ["B"], ["C"]])
        self.assertEqual([s.round for s in steps], [1, 2, 2, 2])

    def test_the_dossier_is_complete_even_if_the_model_never_reads_or_plans(self):
        self.build()
        for name in ("project.json", "steps.jsonl", "ledger.json", "NOTES.md", "rebuild.py", "validation.json"):
            self.assertTrue(os.path.isfile(self.project.path(name)), name)
        for s in self.steps[1:]:
            self.assertTrue(os.path.isfile(self.project.path(s.snapshot)))


# --------------------------------------------------------------------------------------------------------------
# I2


class I2BuildSummaryTest(RealFlowCase):
    """The final answer is the model's commentary plus the app's Build Summary; the claims come from records."""

    def work(self, final="Built a plate.", **kw):
        self.doc = self.doc_name("I2")
        self.project = self.new_project("I2 plate", chat_id="20261002-101010-a2a2a2")
        script = [calls(call("create_document", name=self.doc), self.box(self.doc)), say(final)]
        return self.turn("make a plate", script, project=self.project, **kw)

    def test_the_text_carries_every_heading_of_the_contract(self):
        result = self.work()
        self.assertIn(result.outcome, ("validated", "unverified"))
        self.check_structure(result, self.project)

    def test_the_model_commentary_comes_first(self):
        result = self.work("Built a plate.")
        self.assertTrue(result.text.startswith("Built a plate."))
        self.assertGreater(result.text.index("## Build Summary"), 0)

    def test_a_one_word_answer_still_gets_the_full_summary(self):
        result = self.work("ok")
        self.assertTrue(result.text.startswith("ok"))
        self.check_structure(result, self.project)

    def test_an_empty_answer_is_asked_for_once_and_the_summary_is_still_complete(self):
        self.doc = self.doc_name("I2e")
        self.project = self.new_project("I2 empty", chat_id="20261002-101010-a2a2a5")
        script = [calls(call("create_document", name=self.doc), self.box(self.doc)), say(""), say("Built a plate.")]
        result = self.turn("make a plate", script, project=self.project)
        self.assertEqual(result.outcome, "validated")
        self.assertTrue(result.text.startswith("Built a plate."))
        self.assertEqual(len(self.notes("wrapup")), 1)
        self.check_structure(result, self.project)

    def test_two_empty_answers_are_a_visible_failure_not_a_green_summary(self):
        self.doc = self.doc_name("I2e2")
        self.project = self.new_project("I2 empty2", chat_id="20261002-101010-a2a2a6")
        script = [calls(call("create_document", name=self.doc), self.box(self.doc)), say(""), say("")]
        result = self.turn("make a plate", script, project=self.project)
        self.assertEqual((result.ok, result.outcome), (False, "failed"))
        self.assertIn("empty reply", result.text)

    def test_the_same_text_reaches_the_signals_and_the_panel_card(self):
        result = self.work()
        self.assertEqual(self.rec.of("turn_finished")[0], (True, result.text))
        outcome, summary = self.rec.of("turn_outcome")[0]
        self.assertEqual(outcome, result.outcome)
        self.assertIn("## Build Summary", summary)

    def test_validated_lists_only_what_was_recorded(self):
        result = self.work()
        validated = self.section(result.text, "Validated")
        self.assertIn("Model health", validated)
        recorded = self.project.read_validation()
        self.assertTrue(recorded["ok"])
        self.assertEqual(recorded["health"]["valid"], True)
        for unrecorded in ("Clearances", "Part-to-part interference", "Snap-fit retention", "Port openings", "Printability"):
            self.assertNotIn(unrecorded, validated)

    def test_everything_that_did_not_run_is_listed_as_not_validated(self):
        result = self.work()
        not_validated = self.section(result.text, "NOT validated")
        for unrecorded in ("Clearances", "Part-to-part interference", "Snap-fit retention", "Port openings",
                           "Printability", "Parameter editability", "STEP-based collision check"):
            self.assertIn(unrecorded, not_validated)

    def test_a_model_that_claims_checks_the_app_never_ran_cannot_move_them_into_validated(self):
        claim = ("I verified all clearances, the snap-fit lift test, the port openings and a STEP collision check; "
                 "everything passed.")
        result = self.work(claim)
        self.assertTrue(result.text.startswith(claim))
        validated = self.section(result.text, "Validated")
        for word in ("clearance", "snap", "collision", "interference", "port"):
            self.assertNotIn(word, validated.lower())
        self.assertIn("STEP-based collision check", self.section(result.text, "NOT validated"))

    def _self_validated(self):
        doc = self.doc_name("I2v")
        project = self.new_project("I2 validated", chat_id="20261002-101010-a2a2a3")
        script = [calls(call("create_document", name=doc), self.box(doc)), calls(call("validate_model", doc_name=doc)),
                  say("Checked.")]
        return self.turn("make a plate", script, project=project), project

    def test_a_validation_the_model_really_ran_is_recorded(self):
        result, project = self._self_validated()
        recorded = project.read_validation()
        self.assertTrue(recorded["ok"])
        self.assertEqual(recorded["gate_errors"], [])
        self.assertEqual(result.text.startswith("Checked."), True)

    def test_a_model_that_validated_itself_still_ends_validated_with_the_summary(self):
        result, project = self._self_validated()
        self.assertEqual(result.outcome, "validated")
        self.check_structure(result, project)

    def test_the_model_tree_comes_from_the_document_not_from_the_model(self):
        result = self.work("I built a tower of 12 spheres.")
        tree = self.section(result.text, "Model tree")
        self.assertIn("Plate", tree)
        self.assertNotIn("sphere", tree.lower())

    def test_the_requirements_table_is_derived_from_the_recorded_steps(self):
        result = self.work()
        table = self.section(result.text, "Requirements")
        self.assertIn("| R1 |", table)
        self.assertRegex(table, r"\| R1 \| [^|]+ \| done \|")
        ledger = self.project.read_ledger()
        self.assertEqual([r["status"] for r in ledger["requirements"]][0], "done")

    def test_a_chat_turn_has_no_summary_and_creates_no_project(self):
        project = self.new_project("I2 chat", chat_id="20261002-101010-a2a2a4", create=False)
        result = self.turn("what is a fillet?", [say("A rounded edge.")], project=project)
        self.assertEqual((result.outcome, result.text), ("answered", "A rounded edge."))
        self.assertNotIn("## Build Summary", result.text)
        self.assertFalse(os.path.isdir(project.root))

    def test_the_summary_is_bounded(self):
        result = self.work()
        summary = result.text[result.text.index("## Build Summary"):]
        self.assertLessEqual(len(summary.splitlines()), 72)
        self.assertLessEqual(len(summary), 7200)


# --------------------------------------------------------------------------------------------------------------
# I3


class I3NoFalseSuccessTest(RealFlowCase):
    """The outcome of a turn is decided by the recorded state, never by what the model says."""

    def assert_structural(self, result, project):
        """ok-with-outcome-validated implies zero open required steps and a valid recorded health."""
        if result.outcome == "validated":
            ledger = Ledger.from_dict(project.read_ledger())
            self.assertEqual(ledger.pending_required(), [])
            recorded = project.read_validation()
            self.assertTrue(recorded["ok"])
            self.assertEqual(recorded["gate_errors"], [])          # health.valid may be False for the USER's own broken object (C2)
        if not result.ok:
            self.assertIn(result.outcome, ("failed", "needs_help", "stopped"))

    def enclosure(self, script, **kw):
        self.project = self.new_project("I3 case", chat_id="20261002-101010-a3a3a3")
        result = self.turn(PI_PROMPT, script, project=self.project, **kw)
        self.assert_structural(result, self.project)
        return result

    # scenario A ------------------------------------------------------------------------------------------

    def test_a_pending_required_step_when_the_model_says_done_is_unverified(self):
        doc = self.doc_name("I3a")
        script = [calls(call("create_document", name=doc), self.box(doc)), say("Done."), say("Done."), say("Done."),
                  say("Done."), say("Done.")]
        result = self.enclosure(script)
        self.assertEqual((result.ok, result.outcome), (True, "unverified"))
        self.assertTrue(result.text.startswith("NOT VERIFIED:"))
        pending = [s["id"] for s in self.project.read_ledger()["steps"] if s["required"] and s["status"] in ("pending", "running")]
        self.assertTrue(pending)
        for sid in pending[:4]:                                            # the first lines name the ids (the rest is "+N more")
            self.assertIn(sid, result.text)
        self.assertIn("Open items", result.text)

    def test_the_gate_nudges_the_model_with_the_pending_ids_a_bounded_number_of_times(self):
        doc = self.doc_name("I3an")
        script = [calls(call("create_document", name=doc), self.box(doc))] + [say("Done.")] * 8
        self.enclosure(script)
        gate = self.notes("gate")
        self.assertTrue(1 <= len(gate) <= 3, len(gate))
        self.assertRegex(gate[0].content, r"STEPS_PENDING.*S\d")
        self.assertLessEqual(len(self.provider.requests), 1 + 1 + 3 + 1)

    # scenario B / B2 -------------------------------------------------------------------------------------

    def test_a_blocked_step_with_a_note_is_allowed_and_shown(self):
        doc = self.doc_name("I3b")
        note = "user declined download"
        script = [calls(call("plan_update", id="S2", status="blocked", note=note)),
                  calls(call("create_document", name=doc), self.box(doc)), say("Done."), say("Done."), say("Done.")]
        result = self.enclosure(script)
        self.assertIn(result.outcome, ("validated", "unverified"))
        step = next(s for s in self.project.read_ledger()["steps"] if s["id"] == "S2")
        self.assertEqual(step["status"], "blocked")
        self.assertEqual(step["note"], note)
        self.assertIn(note, result.text)                                   # visible, not hidden
        self.assertNotIn("S2 \"Source the reference part\" has no recorded work", result.text)

    def test_asking_the_user_for_a_file_with_zero_geometry_is_needs_input_not_failure(self):
        result = self.enclosure(personas.build("asks_the_user"))
        self.assertEqual((result.ok, result.outcome), (True, "needs_input"))
        self.assertEqual(result.geometry_ok, 0)
        step = next(s for s in self.project.read_ledger()["steps"] if s["id"] == "S2")
        self.assertEqual((step["status"], step["note"]), ("blocked", "please attach the STEP file"))
        self.assertIsNone(self.project.manifest.get("finalized"))          # the work is not over: nothing is finalized

    def test_a_blocked_step_without_a_note_does_not_excuse_the_turn(self):
        script = [calls(call("plan_update", id="S2", status="blocked", note="")), say("Done."), say("Done."), say("Done."),
                  say("Done.")]
        result = self.enclosure(script)
        self.assertNotEqual(result.outcome, "needs_input")
        self.assertNotEqual(result.outcome, "validated")

    # scenario C / C2 -------------------------------------------------------------------------------------

    def test_a_new_health_error_at_finish_is_never_validated(self):
        doc = self.doc_name("I3c")
        script = [calls(call("create_document", name=doc)), calls(broken_cut_script(self, doc)), say("Done."),
                  say("Done."), say("Done."), say("Done.")]
        project = self.new_project("I3 broken", chat_id="20261002-101010-a3a3a4")
        result = self.turn("make a plate", script, project=project)
        self.assertNotEqual(result.outcome, "validated")
        self.assertEqual(result.validation["gate_errors"][0]["object"], "MyCut")
        self.assertIn("MyCut", json.dumps(project.read_validation()))
        self.assertNotIn("valid (", self.section(result.text, "Validated"))

    def test_a_preexisting_broken_object_does_not_block_the_gate(self):
        broken = self.open_doc("I3c2")
        App.getDocument(broken).addObject("Part::Cut", "UsersBrokenCut")
        App.getDocument(broken).recompute()
        project = self.new_project("I3 broken before", chat_id="20261002-101010-a3a3a5")
        result = self.turn("make a plate", [calls(self.box(broken, "NewBox")), say("Added a box.")], project=project)
        self.assertEqual((result.ok, result.outcome), (True, "validated"))
        self.assertEqual(result.validation["gate_errors"], [])
        self.assertIn("UsersBrokenCut", json.dumps(project.read_validation()["health"]))      # visible, but not a gate error
        self.assert_structural(result, project)

    def test_an_error_on_an_object_the_turn_modified_counts_even_if_it_existed_before(self):
        broken = self.open_doc("I3c3")
        App.getDocument(broken).addObject("Part::Cut", "UsersBrokenCut")
        App.getDocument(broken).recompute()
        edit = call("execute_code", include_screenshot=False,
                    code=self.code(broken, "d.getObject('UsersBrokenCut').Label = 'Renamed'\nd.recompute()"))
        script = [calls(edit), say("Done."), say("Done."), say("Done."), say("Done.")]
        project = self.new_project("I3 touched", chat_id="20261002-101010-a3a3a6")
        result = self.turn("modify the cut", script, project=project)
        self.assertNotEqual(result.outcome, "validated")

    # scenario D ------------------------------------------------------------------------------------------

    def test_a_work_turn_with_zero_geometry_and_no_blocked_step_is_failed(self):
        project = self.new_project("I3 plan only", chat_id="20261002-101010-a3a3a7")
        result = self.turn(personas.prompt_for("plan_only"), personas.build("plan_only"), project=project)
        self.assertEqual((result.ok, result.outcome), (False, "failed"))
        self.assertEqual(result.geometry_ok, 0)
        self.assertNotIn("## Build Summary", result.text)
        self.assert_structural(result, project)

    def test_a_geometry_call_that_failed_is_not_geometry_work(self):
        doc = self.doc_name("I3d")
        bad = call("execute_code", code=self.code(doc, "raise RuntimeError('nope')"), include_screenshot=False)
        script = [calls(call("create_document", name=doc)), calls(bad), say("Done."), say("Done."), say("Done.")]
        project = self.new_project("I3 failed", chat_id="20261002-101010-a3a3a8")
        result = self.turn("make a plate", script, project=project)
        self.assertEqual(result.geometry_ok, 0)
        self.assertNotEqual(result.outcome, "validated")

    # scenario E ------------------------------------------------------------------------------------------

    def test_claims_without_recorded_work_are_unverified_and_named(self):
        result = self.enclosure(personas.build("claims_done_without_work"))
        self.assertEqual((result.ok, result.outcome), (True, "unverified"))
        self.assertIn("claimed without recorded work", result.text)
        self.assertTrue(result.text.startswith("NOT VERIFIED:"))
        ledger = self.project.read_ledger()
        claimed = [s["id"] for s in ledger["steps"] if s.get("claimed") and s["required"]]
        self.assertTrue(claimed)
        for sid in claimed:
            self.assertIn(sid, result.text)

    def test_requirements_covered_only_by_claimed_steps_are_not_done(self):
        result = self.enclosure(personas.build("claims_done_without_work"))
        ledger = self.project.read_ledger()
        steps = {s["id"]: s for s in ledger["steps"]}
        checked = 0
        for req in ledger["requirements"]:
            covering = [steps[sid] for sid in req["covered_by"] if sid in steps]
            if covering and all(s.get("claimed") for s in covering):
                checked += 1
                self.assertNotEqual(req["status"], "done", req)
        self.assertGreater(checked, 0)
        self.assertNotRegex(self.section(result.text, "Requirements"), r"\| Snap-fit retention of the lid \| done \|")

    # the kill switch and a validator that cannot judge -------------------------------------------------------

    def test_without_a_validator_the_best_work_outcome_is_answered_never_validated(self):
        doc = self.doc_name("I3v")
        project = self.new_project("I3 no validator", chat_id="20261002-101010-a3a3a9")
        result = self.turn("make a plate", [calls(call("create_document", name=doc), self.box(doc)), say("Done.")],
                          project=project, validator=False)
        self.assertEqual((result.ok, result.outcome), (True, "answered"))
        self.assertNotIn("validated", result.outcome)

    def test_a_failing_validation_blocks_validated_whatever_the_ledger_says(self):
        doc = self.doc_name("I3vf")
        project = self.new_project("I3 failing validator", chat_id="20261002-101010-a3a3b1")
        script = [calls(call("create_document", name=doc), self.box(doc)), say("Done."), say("Done."), say("Done."),
                  say("Done.")]
        result = self.turn("make a plate", script, project=project, validator=lambda d, c: fake_validation(False))
        self.assertEqual(result.outcome, "unverified")
        self.assert_structural(result, project)

    def test_every_persona_ends_in_an_outcome_consistent_with_the_ok_flag(self):
        cases = (("plan_only", False, "failed"), ("claims_done_without_work", True, "unverified"),
                 ("asks_the_user", True, "needs_input"), ("tool_then_stop", True, "unverified"))
        for name, ok, outcome in cases:
            with self.subTest(persona=name):
                self.close_docs()
                self.setUp()
                self.docs.append(personas.DOC)
                project = self.new_project(f"I3 {name}")
                result = self.turn(personas.prompt_for(name), personas.build(name), project=project)
                self.assertEqual((result.ok, result.outcome), (ok, outcome))
                self.assert_structural(result, project)


# --------------------------------------------------------------------------------------------------------------
# I4


STEP_URL = "https://raw.githubusercontent.com/example/pi/main/pi4.step"


class I4ProvenanceTest(RealFlowCase):
    """What is claimed as sourced has a recorded source, hash and status; the summary never overclaims."""

    def source_then_build(self, *, final="Done.", import_step=False, project=None):
        doc = self.doc_name("I4")
        self.project = project or self.new_project("I4 case", chat_id="20261002-101010-a4a4a4")
        rounds = [calls(call("source_reference_part", id="rpi4b", reason="reference geometry for the housing")),
                  calls(call("create_document", name=doc))]
        if import_step:
            rounds.append(calls(call("import_reference_part", doc_name=doc, id="rpi4b")))
        rounds += [calls(self.box(doc, "Shell")), say(final), say(final), say(final), say(final), say(final)]
        self.result = self.turn(PI_PROMPT, rounds, project=self.project)
        self.prov = self.project.read_provenance()
        return self.result

    def entry(self):
        ref = self.prov["reference"]
        return next(e for e in self.prov["entries"] if e["id"] == ref["entry"])

    def test_no_sourcing_at_all_means_status_none_and_no_entries(self):
        doc = self.doc_name("I4n")
        project = self.new_project("I4 none", chat_id="20261002-101010-a4a4a5")
        result = self.turn("make a plate", [calls(call("create_document", name=doc), self.box(doc)), say("Done.")], project=project)
        self.assertEqual(project.read_provenance()["reference"]["status"], "none")
        self.assertEqual(project.read_provenance()["entries"], [])
        self.assertIn("reference status: none", result.text)
        self.assertIn("STEP-based collision check", self.section(result.text, "NOT validated"))

    def test_catalog_nominal_records_source_and_status_and_has_no_hash(self):
        result = self.source_then_build()
        self.assertEqual(self.prov["reference"]["status"], "catalog-nominal")
        entry = self.entry()
        self.assertTrue(entry["source"].startswith("catalog:rpi4b"))
        self.assertEqual((entry["status"], entry["sha256"], entry["local_path"]), ("nominal", "", ""))
        self.check_structure(result, self.project)

    def test_the_catalog_nominal_summary_cannot_claim_a_step_based_check(self):
        result = self.source_then_build(final="Verified against the real STEP file, collision check passed.")
        self.assertIn("reference status: catalog-nominal", result.text)
        self.assertIn("STEP-based collision check", self.section(result.text, "NOT validated"))
        self.assertNotIn("STEP", self.section(result.text, "Validated"))
        provenance = self.section(result.text, "Provenance")
        self.assertIn("**catalog-nominal**", provenance)
        self.assertIn("sha256: n/a", provenance)

    def test_an_obtained_step_has_a_hash_and_a_file_that_matches_it(self):
        self.use_mirror()
        result = self.source_then_build(import_step=True)
        ref = self.prov["reference"]
        self.assertEqual(ref["status"], "step-imported")
        obtained = next(e for e in self.prov["entries"] if e["status"] == "obtained")
        self.assertEqual(len(obtained["sha256"]), 64)
        path = self.project.path(obtained["local_path"])
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(sha256_of(path), obtained["sha256"])                  # the recorded hash is the real file's hash
        self.assertEqual(obtained["detected_type"], "step")
        self.check_structure(result, self.project)

    def test_a_step_import_changes_the_not_validated_reason_not_the_honesty(self):
        self.use_mirror()
        result = self.source_then_build(import_step=True)
        text = result.text
        self.assertIn("reference status: step-imported", text)
        line = next(x for x in self.section(text, "NOT validated").splitlines() if "STEP-based collision check" in x)
        self.assertNotIn("no reference part was sourced", line)       # a STEP is there: the reason is the check, not the source
        self.assertIn("sha256:", self.section(text, "Provenance"))
        self.assertNotIn("sha256: n/a", self.section(text, "Provenance"))

    def test_an_entry_exists_for_every_reference_status_other_than_none(self):
        self.source_then_build()
        ref = self.prov["reference"]
        self.assertIn(ref["status"], REFERENCE_STATUSES)
        self.assertIn(ref["entry"], [e["id"] for e in self.prov["entries"]])

    def test_a_declined_download_falls_back_to_nominal_and_says_declined(self):
        cat = copy.deepcopy(catalog._catalog())
        next(e for e in cat["entries"] if e["id"] == "rpi4b")["step_urls"] = [
            {"url": STEP_URL, "mirror": False, "expected_sha256": "", "expected_bbox": [], "license": "test licence"}]
        with mock.patch.object(catalog, "_catalog", return_value=cat), \
                mock.patch.object(netguard, "request_permission", return_value="deny"):
            result = self.source_then_build()
        statuses = [e["status"] for e in self.prov["entries"]]
        self.assertIn("declined", statuses)
        self.assertEqual(self.prov["reference"]["status"], "catalog-nominal")
        declined = next(e for e in self.prov["entries"] if e["status"] == "declined")
        self.assertTrue(declined["source"])
        self.assertEqual(declined["sha256"], "")
        self.assertIn("STEP-based collision check", self.section(result.text, "NOT validated"))

    def test_the_summary_reads_the_status_from_provenance_not_from_the_model(self):
        result = self.source_then_build(final="reference status: step-measured (I measured the real STEP).")
        summary = result.text[result.text.index("## Build Summary"):]
        self.assertIn("reference status: catalog-nominal", summary)
        self.assertNotIn("step-measured", summary)

    def test_a_reference_the_model_only_names_in_text_is_not_provenance(self):
        doc = self.doc_name("I4t")
        project = self.new_project("I4 text", chat_id="20261002-101010-a4a4a6")
        claim = "Wrapped the exact STEP model of the Raspberry Pi 4 from the manufacturer."
        script = [calls(call("create_document", name=doc), self.box(doc, "Shell")), say(claim)] + [say(claim)] * 5
        result = self.turn(PI_PROMPT, script, project=project)
        self.assertEqual(project.read_provenance()["reference"]["status"], "none")
        self.assertIn("reference status: none", result.text)
        self.assertNotEqual(result.outcome, "validated")

    def test_the_generic_structure_check_holds_for_the_obtained_flow_too(self):
        self.use_mirror()
        result = self.source_then_build(import_step=True)
        self.check_structure(result, self.project)
        self.assertRegex(result.text, re.compile(r"reference status: step-imported"))


if __name__ == "__main__":
    unittest.main()
