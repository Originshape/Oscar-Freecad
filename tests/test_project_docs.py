"""Generated text of the Project: NOTES.md headings and sections, the Build Summary layout and the validated /
NOT validated rule, rebuild.py text, the literal round trip and the headless-check comparison helpers. Pure
file logic (no FreeCAD document)."""

import ast
import json
import os
import re
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import project_support as S  # noqa: E402
from oscar import project as P  # noqa: E402
from oscar import project_docs as D  # noqa: E402
from oscar import project_replay as R  # noqa: E402

NOTES_HEADINGS = ["## 1. Request", "## 2. Requirements and status", "## 3. Plan", "## 4. Parameters",
                  "## 5. Provenance", "## 6. Assumptions", "## 7. Validation", "## 8. Unresolved",
                  "## 9. Steps log", "## 10. Artifacts"]
SUMMARY_HEADINGS = ["## Build Summary", "### Model tree", "### Requirements", "### Parameters", "### Provenance",
                    "### Validated", "### NOT validated", "### Open items"]

LABELS = {
    "health": "Model health (valid solids, no recompute errors)",
    "fit.interference": "Part-to-part interference",
    "fit.clearance": "Clearances",
    "fit.ports": "Port openings contain every connector",
    "fit.retention": "Snap-fit retention (lid lift test)",
    "dfm": "Printability rules (minimum wall, vent web, hook strain)",
    "editability": "Parameter editability (+-10% rebuild)",
    "rebuild": "rebuild.py reproduces the model headless",
    "step_collision": "STEP-based collision check",
}


def validation(**over):
    """A complete, passing validation record; keyword overrides replace whole fields."""
    data = {
        "after_step": 1, "ok": True, "doc": "PiCase",
        "ran": ["health", "fit.interference", "fit.clearance", "fit.ports", "fit.retention", "dfm"],
        "health": {"valid": True, "counts": {"objects": 7, "solids": 2, "errors": 0, "warnings": 0},
                   "findings": [], "rows": [{"name": "Shell", "type": "Part::Cut", "solids": 1, "volume": 18432.1,
                                             "bbox": [0, 0, 0, 94, 65, 22]}]},
        "gate_errors": [],
        "fit_rows": [{"id": "interference.shell_lid", "kind": "interference", "status": "PASS"},
                     {"id": "clearance.board_shell", "kind": "clearance", "status": "WARN", "detail": "tight"},
                     {"id": "ports.usb_c", "kind": "containment", "status": "PASS"},
                     {"id": "hooks.retention", "kind": "retention", "status": "PASS"}],
        "dfm": [{"id": "dfm.min_wall", "status": "PASS"}],
        "coverage": [], "not_validated": ["step_collision", "physical"],
    }
    data.update(over)
    return data


def ledger(**over):
    data = {
        "intent": "enclosure", "request": "Build a housing for a Raspberry Pi 4", "needs_reference": True,
        "requirements": [
            {"id": "R1", "text": "Fits a Pi 4", "status": "done", "covered_by": ["S1"]},
            {"id": "R2", "text": "Cutouts for every I/O port", "status": "pending", "covered_by": ["S2"]}],
        "steps": [
            {"id": "S1", "title": "Shell", "phase": "BUILD", "status": "done", "note": "", "required": True,
             "covers": ["R1"], "evidence": [{"step_index": 1, "kind": "auto"}]},
            {"id": "S2", "title": "Ports", "phase": "BUILD", "status": "blocked", "note": "no data for DSI",
             "required": True, "covers": ["R2"], "evidence": []}],
    }
    data.update(over)
    return data


def section(text: str, heading: str) -> str:
    """The text of one ``## `` section of NOTES.md (without the heading line)."""
    start = text.index(heading) + len(heading)
    nxt = re.search(r"\n## ", text[start:])
    return text[start:start + nxt.start()] if nxt else text[start:]


class ClassifyValidationTest(unittest.TestCase):
    def ids(self, items):
        return [i["id"] for i in items]

    def test_catalog_order_and_labels(self):
        self.assertEqual(D.VALIDATION_CATALOG, ("health", "fit.interference", "fit.clearance", "fit.ports",
                                                "fit.retention", "dfm", "editability", "rebuild", "step_collision",
                                                "physical"))
        self.assertEqual(P.VALIDATION_CATALOG, D.VALIDATION_CATALOG)
        for key, label in LABELS.items():
            self.assertEqual(D.VALIDATION_LABELS[key], label)
        self.assertEqual(D.VALIDATION_LABELS["physical"],
                         "Physical fit, printing, thermal behaviour and <MATERIAL> shrinkage on a real printer")

    def test_every_entry_is_exactly_once_validated_or_not(self):
        for record in (None, {}, validation(), validation(ran=[]), validation(gate_errors=[{"code": "X"}])):
            good, bad = D.classify_validation(record)
            self.assertEqual(sorted(self.ids(good) + self.ids(bad)), sorted(D.VALIDATION_CATALOG))

    def test_nothing_ran_everything_but_physical_says_not_run(self):
        good, bad = D.classify_validation(None)
        self.assertEqual(good, [])
        reasons = {b["id"]: b["reason"] for b in bad}
        self.assertEqual([k for k, v in reasons.items() if v == "not run"],
                         ["health", "fit.interference", "fit.clearance", "fit.ports", "fit.retention", "dfm",
                          "editability", "rebuild"])
        self.assertIn("no reference part was sourced", reasons["step_collision"])
        self.assertIn("physical", reasons)

    def test_a_passing_record_validates_what_ran_in_catalog_order(self):
        good, bad = D.classify_validation(validation(), reference_status="catalog-nominal")
        self.assertEqual(self.ids(good), ["health", "fit.interference", "fit.clearance", "fit.ports",
                                          "fit.retention", "dfm"])
        self.assertEqual(self.ids(bad), ["editability", "rebuild", "step_collision", "physical"])
        by_id = {g["id"]: g for g in good}
        self.assertEqual(by_id["health"]["result"], "valid (7 objects, 2 solids)")
        self.assertEqual(by_id["fit.interference"]["result"], "1 PASS")
        self.assertEqual(by_id["fit.clearance"]["result"], "1 WARN")
        self.assertEqual(by_id["fit.ports"]["result"], "1 PASS")
        self.assertEqual(by_id["fit.retention"]["result"], "1 PASS")   # the hooks.* rows count as retention
        self.assertEqual(by_id["dfm"]["label"], LABELS["dfm"])

    def test_a_fail_row_is_never_validated(self):
        rows = [{"id": "interference.shell_lid", "kind": "interference", "status": "FAIL", "detail": "12 mm3"},
                {"id": "interference.lid_ref", "kind": "interference", "status": "PASS"}]
        good, bad = D.classify_validation(validation(fit_rows=rows))
        self.assertNotIn("fit.interference", self.ids(good))
        reason = {b["id"]: b["reason"] for b in bad}["fit.interference"]
        self.assertEqual(reason, "FAIL: interference.shell_lid")
        good, bad = D.classify_validation(validation(dfm=[{"id": "dfm.min_wall", "status": "FAIL"}]))
        self.assertNotIn("dfm", self.ids(good))
        self.assertEqual({b["id"]: b["reason"] for b in bad}["dfm"], "FAIL: dfm.min_wall")

    def test_an_entry_that_ran_but_has_only_skipped_rows_is_not_validated(self):
        rows = [{"id": "ports.x", "kind": "containment", "status": "SKIP", "detail": "no spec"}]
        good, bad = D.classify_validation(validation(fit_rows=rows))
        self.assertNotIn("fit.ports", self.ids(good))
        self.assertEqual({b["id"]: b["reason"] for b in bad}["fit.ports"], "skipped: no spec")

    def test_an_entry_that_did_not_run_is_not_validated_even_with_rows(self):
        good, bad = D.classify_validation(validation(ran=["health"]))
        self.assertEqual(self.ids(good), ["health"])
        self.assertEqual({b["id"]: b["reason"] for b in bad}["dfm"], "not run")

    def test_health_with_gate_errors_is_not_validated(self):
        errors = [{"code": "FRAGMENTS", "object": "Shell", "message": "3 solids"},
                  {"code": "INVALID_SHAPE", "object": "Lid", "message": "bad"}]
        good, bad = D.classify_validation(validation(gate_errors=errors))
        self.assertNotIn("health", self.ids(good))
        reason = {b["id"]: b["reason"] for b in bad}["health"]
        self.assertTrue(reason.startswith("2 error(s): FRAGMENTS Shell; INVALID_SHAPE Lid"), reason)

    def test_preexisting_health_errors_are_reported_but_do_not_block(self):
        health = {"valid": False, "counts": {"objects": 4, "solids": 1, "errors": 2, "warnings": 0}, "findings": []}
        good, _ = D.classify_validation(validation(health=health))
        self.assertEqual({g["id"]: g["result"] for g in good}["health"], "no new errors (2 pre-existing)")

    def test_physical_is_always_not_validated_and_names_the_material(self):
        for record in (None, validation(ran=list(D.VALIDATION_CATALOG)), validation(ok=True)):
            good, bad = D.classify_validation(record, material="ABS")
            self.assertNotIn("physical", self.ids(good))
            physical = next(b for b in bad if b["id"] == "physical")
            self.assertEqual(physical["label"], "Physical fit, printing, thermal behaviour and ABS shrinkage on a real printer")
        _, bad = D.classify_validation(None)
        self.assertIn("and material shrinkage", next(b for b in bad if b["id"] == "physical")["label"])

    def test_step_collision_needs_a_step_reference_and_computed_rows(self):
        ref_rows = [{"id": "interference.reference_shell", "kind": "interference", "status": "PASS"}]
        base = validation()["fit_rows"]
        # nominal reference: never validated, with the exact reason
        for status in ("none", "catalog-nominal"):
            good, bad = D.classify_validation(validation(fit_rows=base + ref_rows), reference_status=status)
            self.assertNotIn("step_collision", self.ids(good))
        _, bad = D.classify_validation(validation(), reference_status="catalog-nominal")
        self.assertEqual({b["id"]: b["reason"] for b in bad}["step_collision"],
                         "the reference is nominal board data; no STEP file was imported")
        # a STEP reference but no reference rows computed
        for status in ("step-imported", "step-measured", "user-attached"):
            good, bad = D.classify_validation(validation(), reference_status=status)
            self.assertNotIn("step_collision", self.ids(good), status)
            self.assertEqual({b["id"]: b["reason"] for b in bad}["step_collision"],
                             "interference against the imported reference was not run")
        # a STEP reference with computed rows
        for status in ("step-imported", "step-measured", "user-attached"):
            good, _ = D.classify_validation(validation(fit_rows=base + ref_rows), reference_status=status)
            self.assertIn("step_collision", self.ids(good), status)
        # skipped reference rows do not count
        skipped = [{"id": "interference.reference_shell", "kind": "interference", "status": "SKIP"}]
        good, _ = D.classify_validation(validation(fit_rows=base + skipped), reference_status="step-imported")
        self.assertNotIn("step_collision", self.ids(good))
        # a failing reference row
        failing = [{"id": "interference.reference_shell", "kind": "interference", "status": "FAIL"}]
        good, bad = D.classify_validation(validation(fit_rows=base + failing), reference_status="step-imported")
        self.assertNotIn("step_collision", self.ids(good))
        # interference itself not run
        good, _ = D.classify_validation(validation(fit_rows=base + ref_rows, ran=["health"]), reference_status="step-imported")
        self.assertNotIn("step_collision", self.ids(good))

    def test_rebuild_and_editability(self):
        good, _ = D.classify_validation(validation(ran=["health", "editability"]), rebuild_ok=True)
        self.assertTrue({"editability", "rebuild"} <= set(self.ids(good)))
        good, _ = D.classify_validation(validation(), rebuild_ok=False)
        self.assertNotIn("rebuild", self.ids(good))


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.proj = S.make_project(create=True)

    def build(self, led=None, outcome="validated", val=None):
        return self.proj.build_summary(led, outcome, val)

    def test_headings_and_the_reference_status_line(self):
        self.proj.set_reference("rpi4b", "Raspberry Pi 4 Model B", "catalog-nominal", "E1")
        summary = self.build(ledger(), "validated", validation())
        text = summary.markdown
        positions = [text.index(h) for h in SUMMARY_HEADINGS]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(text.startswith("## Build Summary\n**Outcome:** validated · reference status: catalog-nominal · project: "))
        self.assertIn(f"project: {self.proj.root}", text)
        self.assertIn("reference status: catalog-nominal", text)
        self.assertTrue(text.rstrip().split("\n")[-1].startswith("**Files:**"))

    def test_requirements_is_omitted_without_requirements(self):
        text = self.build(ledger(requirements=[]), "answered", validation()).markdown
        self.assertNotIn("### Requirements", text)
        for heading in SUMMARY_HEADINGS:
            if heading != "### Requirements":
                self.assertIn(heading, text)
        self.assertNotIn("### Requirements", self.build(None, "answered", None).markdown)

    def test_requirement_rows_come_from_the_ledger_and_the_coverage_detail(self):
        val = validation(coverage=[{"requirement": "R2", "status": "missing", "by": ["S2"], "detail": "8/11 openings cut"},
                                   {"requirement": "R1", "status": "covered", "by": ["S1"], "detail": "board fits"}])
        text = self.build(ledger(), "needs_help", val).markdown
        self.assertIn("| R1 | Fits a Pi 4 | done | S1 (board fits) |", text)
        self.assertIn("| R2 | Cutouts for every I/O port | pending - geometry missing | S2 (8/11 openings cut) |", text)

    def test_validated_and_not_validated_bullets(self):
        self.proj.set_reference("rpi4b", "Raspberry Pi 4 Model B", "catalog-nominal", "E1")
        text = self.build(ledger(), "validated", validation()).markdown
        validated = text.split("### Validated\n")[1].split("\n### ")[0]
        not_validated = text.split("### NOT validated\n")[1].split("\n### ")[0]
        self.assertIn("- Model health (valid solids, no recompute errors): valid (7 objects, 2 solids)", validated)
        self.assertIn("- Part-to-part interference: 1 PASS", validated)
        self.assertIn("- STEP-based collision check (the reference is nominal board data; no STEP file was imported)",
                      not_validated)
        self.assertTrue(re.search(r"- Physical fit, printing, thermal behaviour and material shrinkage on a real printer \(", not_validated))
        self.assertNotIn("Physical fit", validated)
        self.assertIn("- rebuild.py reproduces the model headless (not run)", not_validated)

    def test_the_spec_material_names_the_physical_line(self):
        self.proj.save_spec({"name": "x", "material": "PETG"})
        text = self.build(ledger(), "validated", validation()).markdown
        self.assertIn("thermal behaviour and PETG shrinkage on a real printer", text)

    def test_a_validation_older_than_the_last_geometry_step_counts_for_nothing(self):
        S.record(self.proj)
        S.record(self.proj)   # last geometry step = 2
        summary = self.build(ledger(), "validated", validation(after_step=1))
        text = summary.markdown
        validated = text.split("### Validated\n")[1].split("\n### ")[0]
        self.assertEqual(validated.strip(), "- nothing validated")
        self.assertIn("stale: the model changed after the last validation", text)
        self.assertTrue(any("validate again" in item for item in summary.data["open_items"]))
        self.assertEqual(summary.data["validated"], [])
        fresh = self.build(ledger(), "validated", validation(after_step=2)).markdown
        self.assertIn("Model health (valid solids, no recompute errors): valid", fresh)

    def test_open_items_collect_blocked_steps_notes_gate_errors_and_fit_failures(self):
        self.proj.note("unresolved", "port heights approximate")
        val = validation(gate_errors=[{"code": "FRAGMENTS", "object": "Shell", "message": "3 solids"}],
                         fit_rows=[{"id": "interference.shell_lid", "kind": "interference", "status": "FAIL",
                                    "detail": "12 mm3"}])
        items = self.build(ledger(), "needs_help", val).data["open_items"]
        joined = "\n".join(items)
        self.assertIn("S2 blocked: no data for DSI", joined)
        self.assertIn("N1: port heights approximate", joined)
        self.assertIn("health error FRAGMENTS Shell: 3 solids", joined)
        self.assertIn("fit FAIL interference.shell_lid", joined)

    def test_pending_required_steps_are_open_items(self):
        led = ledger(steps=[{"id": "S3", "title": "Lid", "phase": "BUILD", "status": "pending", "required": True,
                             "covers": [], "evidence": []},
                            {"id": "S4", "title": "Optional", "phase": "BUILD", "status": "pending", "required": False,
                             "covers": [], "evidence": []}])
        items = self.build(led, "needs_help", None).data["open_items"]
        self.assertEqual(items, ["S3 pending: Lid"])

    def test_provenance_line_and_sha256(self):
        src = S.new_temp_file(S.sandbox_dir(), "pi4.step", b"STEP DATA")
        import hashlib

        digest = hashlib.sha256(b"STEP DATA").hexdigest()
        entry = self.proj.provenance_add(kind="download", claim="STEP model", source="https://example.org/pi4.step",
                                         status="obtained", confidence="measured", local_path=src, detected_type="step",
                                         reference_status="step-imported")
        self.proj.set_reference("rpi4b", "Raspberry Pi 4 Model B", "step-imported", entry)
        text = self.build(ledger(), "validated", validation()).markdown
        self.assertIn(f"- Reference: Raspberry Pi 4 Model B - status **step-imported** - source: https://example.org/pi4.step - sha256: {digest}", text)
        other = S.make_project(chat_id="20261001-000000-bb0002", create=True)
        other.provenance_add(kind="reference", claim="c", source="catalog:rpi4b", status="nominal",
                             reference_status="catalog-nominal")
        other.set_reference("rpi4b", "Raspberry Pi 4 Model B", "catalog-nominal", "E1")
        self.assertIn("sha256: n/a (no file)", other.build_summary(None, "validated", None).markdown)

    def test_an_unnamed_reference_is_not_shown_as_none(self):
        self.proj.provenance_add(kind="reference", claim="c", source="s", status="nominal", reference_status="catalog-nominal")
        text = self.build(None, "validated", None).markdown
        self.assertIn("- Reference: (unnamed) - status **catalog-nominal**", text)
        self.assertIn("- Reference: none - status **none**", S.make_project(chat_id="20261001-000000-cc0003", create=True)
                      .build_summary(None, "x", None).markdown)

    def test_model_tree_falls_back_to_the_health_rows_of_the_validation(self):
        text = self.build(ledger(), "validated", validation()).markdown
        self.assertIn("| Shell | Part::Cut | 1 | 18432.1 | 94.0 x 65.0 x 22.0 |", text)

    def test_parameters_table(self):
        self.proj.ensure_dir()
        import json as _json

        with open(self.proj.path("params.json"), "w") as f:
            _json.dump({"schema": 1, "params": [{"name": "wall_t", "value": 2.0, "unit": "mm", "source": "assumption"},
                                                {"name": "vent_count", "value": 6, "unit": "", "source": "derived"}]}, f)
        text = self.build(ledger(), "validated", validation()).markdown
        self.assertIn("| wall_t | 2 | mm | assumption |", text)
        self.assertIn("| vent_count | 6 |  | derived |", text)

    def test_the_summary_is_bounded_whatever_the_inputs(self):
        self.proj.ensure_dir()
        params = [{"name": f"param_{i}", "value": float(i), "unit": "mm", "source": "derived", "note": "n" * 300}
                  for i in range(200)]
        with open(self.proj.path("params.json"), "w") as f:
            json.dump({"schema": 1, "params": params}, f)
        reqs = [{"id": f"R{i}", "text": "requirement " * 40, "status": "pending", "covered_by": [f"S{i}"]} for i in range(60)]
        steps = [{"id": f"S{i}", "title": "t" * 200, "phase": "BUILD", "status": "blocked", "note": "why " * 100,
                  "required": True, "covers": [f"R{i}"], "evidence": []} for i in range(60)]
        rows = [{"name": f"Part{i}", "type": "Part::Cut", "solids": 1, "volume": 1.0, "bbox": [0] * 6}
                for i in range(120)]
        val = validation(health={"valid": True, "counts": {"objects": 120, "solids": 120}, "findings": [], "rows": rows},
                         fit_rows=[{"id": f"interference.{i}", "kind": "interference", "status": "PASS"} for i in range(80)])
        summary = self.build(ledger(requirements=reqs, steps=steps), "needs_help", val)
        lines = summary.markdown.split("\n")
        self.assertLessEqual(len(lines), 70)
        self.assertLessEqual(len(summary.markdown), 7000)
        for heading in SUMMARY_HEADINGS:
            self.assertIn(heading, summary.markdown)
        self.assertRegex(summary.markdown, r"\+\d+ more")

    def test_data_mirrors_the_markdown(self):
        summary = self.build(ledger(), "validated", validation())
        for key in ("outcome", "reference_status", "project_dir", "tree", "requirements", "parameters", "provenance",
                    "validated", "not_validated", "open_items", "files"):
            self.assertIn(key, summary.data)
        json.dumps(summary.data)
        self.assertEqual(summary.data["project_dir"], self.proj.root)
        self.assertEqual([v["id"] for v in summary.data["validated"]][:2], ["health", "fit.interference"])

    def test_the_files_line_lists_what_exists(self):
        self.proj.write_notes_md()
        self.proj.write_rebuild_py()
        S.new_temp_file(self.proj.path("exports"), "a.step")
        text = self.build(None, "validated", None).markdown
        self.assertIn("**Files:** NOTES.md, rebuild.py", text)
        self.assertTrue(text.rstrip().endswith("exports/"))

    def test_the_rebuild_check_is_validated_only_until_the_model_changes(self):
        S.record(self.proj)
        self.proj.record_rebuild_check(True, "1 object matched")
        good = self.build(ledger(), "validated", validation(after_step=1)).data["validated"]
        self.assertIn("rebuild", [g["id"] for g in good])
        S.record(self.proj)   # a later geometry step invalidates the check
        good = self.build(ledger(), "validated", validation(after_step=2)).data["validated"]
        self.assertNotIn("rebuild", [g["id"] for g in good])
        self.proj.record_rebuild_check(False, "volumes differ")
        good = self.build(ledger(), "validated", validation(after_step=2)).data["validated"]
        self.assertNotIn("rebuild", [g["id"] for g in good])

    def test_render_summary_is_pure_and_deterministic(self):
        kwargs = dict(outcome="validated", reference_status="none", project_dir="/p", tree=[], requirements=[],
                      steps=[], coverage=None, params=[], provenance={}, validated=[], not_validated=[],
                      open_items=[], files=[])
        self.assertEqual(D.render_summary(**kwargs), D.render_summary(**kwargs))
        text = D.render_summary(**kwargs)
        self.assertIn("- nothing validated", text)
        self.assertIn("**Files:** none", text)


class NotesTest(unittest.TestCase):
    def setUp(self):
        self.proj = S.make_project(create=True)

    def text(self, led=None):
        self.proj.write_notes_md(led)
        return open(self.proj.path("NOTES.md"), encoding="utf-8").read()

    def test_title_and_the_ten_headings_in_order(self):
        text = self.text()
        self.assertTrue(text.startswith("# Pi 4 housing - project notes\n_Generated by Oscar "))
        self.assertIn(". Do not edit: it is rewritten after every step._", text.split("\n")[1])
        positions = [text.index("\n" + h + "\n") for h in NOTES_HEADINGS]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(len(re.findall(r"^## ", text, re.M)), 10)

    def test_request_comes_from_the_ledger_then_the_first_turn(self):
        self.assertIn("(no request recorded)", self.text())
        self.proj.begin_turn(S.TURN, "make me a bracket", work=True)
        self.assertIn("make me a bracket", section(self.text(), "## 1. Request"))
        self.assertIn("Build a housing for a Raspberry Pi 4", section(self.text(ledger()), "## 1. Request"))

    def test_requirements_and_plan_tables(self):
        text = self.text(ledger())
        req = section(text, "## 2. Requirements and status")
        self.assertIn("| id | requirement | status | evidence steps |", req)
        self.assertIn("| R1 | Fits a Pi 4 | done | #1 |", req)
        self.assertIn("| R2 | Cutouts for every I/O port | pending | - |", req)
        plan = section(text, "## 3. Plan")
        self.assertIn("| id | phase | step | status | note |", plan)
        self.assertIn("| S2 | BUILD | Ports | blocked | no data for DSI |", plan)

    def test_empty_sections_say_none(self):
        text = self.text()
        self.assertIn("(none)", section(text, "## 2. Requirements and status"))
        self.assertIn("(none)", section(text, "## 3. Plan"))
        self.assertIn("(none)", section(text, "## 4. Parameters"))
        self.assertIn("(no sources recorded)", section(text, "## 5. Provenance"))
        self.assertIn("(none)", section(text, "## 6. Assumptions"))
        self.assertIn("(none)", section(text, "## 8. Unresolved"))
        self.assertIn("(no steps recorded)", section(text, "## 9. Steps log"))

    def test_parameters_table_and_the_needs_rebuild_marker(self):
        with open(self.proj.path("params.json"), "w") as f:
            json.dump({"schema": 1, "params": [
                {"name": "wall_t", "value": 2.0, "unit": "mm", "expr": "", "source": "assumption", "note": "typical FDM wall", "bound": 9},
                {"name": "clearance_eff", "value": None, "unit": "mm", "expr": "=clearance + 1", "source": "derived", "bound": 6},
                {"name": "vent_count", "value": 6, "unit": "", "source": "derived", "bound": 0, "needs_rebuild": True}]}, f)
        params = section(self.text(), "## 4. Parameters")
        self.assertIn("| name | value | unit | source | note | bound |", params)
        self.assertIn("| wall_t | 2 | mm | assumption | typical FDM wall | 9 |", params)
        self.assertIn("| clearance_eff | =clearance + 1 | mm | derived |  | 6 |", params)
        self.assertIn("| vent_count | 6 |  | derived |  | needs rebuild (use set_parameter) |", params)

    def test_provenance_section(self):
        self.proj.set_reference("rpi4b", "Raspberry Pi 4 Model B", "catalog-nominal", "E1")
        self.proj.provenance_add(kind="reference", claim="Board outline", source="catalog:rpi4b", status="nominal",
                                 confidence="datasheet")
        prov = section(self.text(), "## 5. Provenance")
        self.assertIn("reference status: catalog-nominal (Raspberry Pi 4 Model B)", prov)
        self.assertIn("| E1 | reference | Board outline | catalog:rpi4b | nominal | datasheet | - |", prov)

    def test_assumptions_collect_notes_assumed_parameters_and_the_material_line(self):
        with open(self.proj.path("params.json"), "w") as f:
            json.dump({"schema": 1, "params": [{"name": "wall_t", "value": 2.0, "unit": "mm", "source": "assumption",
                                                "note": "typical FDM wall"},
                                               {"name": "clearance", "value": 0.3, "unit": "mm", "source": "datasheet"}]}, f)
        self.proj.note("assumption", "the lid is a cap lid", refs=["S5"])
        self.proj.note("decision", "not an assumption")
        self.proj.save_spec({"name": "pi4", "material": "ABS", "shrink_comp": "clearance"})
        assumptions = section(self.text(), "## 6. Assumptions")
        self.assertIn("- N1 (S5): the lid is a cap lid", assumptions)
        self.assertIn("- parameter wall_t = 2 mm (assumed): typical FDM wall", assumptions)
        self.assertNotIn("clearance = 0.3", assumptions)
        self.assertNotIn("not an assumption", assumptions)
        self.assertIn("ABS shrinkage 0.8% is compensated by clearance only; not validated on a printer.", assumptions)
        self.proj.save_spec({"name": "pi4", "material": "PLA", "shrink_comp": "none"})
        self.assertIn("PLA shrinkage 0.3% is NOT compensated; not validated on a printer.",
                      section(self.text(), "## 6. Assumptions"))

    def test_validation_section_uses_the_shared_rule(self):
        self.proj.record_validation(validation())
        body = section(self.text(), "## 7. Validation")
        self.assertIn("Validated:\n- Model health (valid solids, no recompute errors): valid (7 objects, 2 solids)", body)
        self.assertIn("\nNOT validated:\n", body)
        self.assertIn("- Parameter editability (+-10% rebuild) (not run)", body)
        self.assertLess(body.index("Validated:"), body.index("NOT validated:"))

    def test_validation_section_without_a_record(self):
        body = section(self.text(), "## 7. Validation")
        self.assertIn("- (nothing validated yet)", body)
        self.assertIn("- Physical fit, printing, thermal behaviour and material shrinkage on a real printer", body)

    def test_unresolved_section(self):
        self.proj.note("unresolved", "port heights approximate")
        val = validation(health={"valid": False, "counts": {}, "rows": [], "findings": [
            {"severity": "error", "code": "FRAGMENTS", "object": "Shell", "message": "3 solids"},
            {"severity": "info", "code": "SKETCH_UNDERCONSTRAINED", "object": "Sk", "message": "dof 2"}]},
            fit_rows=[{"id": "interference.shell_lid", "kind": "interference", "status": "FAIL", "detail": "12 mm3"}])
        self.proj.record_validation(val)
        body = section(self.text(ledger()), "## 8. Unresolved")
        self.assertIn("- S2 blocked: no data for DSI (Ports)", body)
        self.assertIn("health error FRAGMENTS Shell: 3 solids", body)
        self.assertNotIn("SKETCH_UNDERCONSTRAINED", body)   # info findings are not unresolved
        self.assertIn("fit FAIL interference.shell_lid: 12 mm3", body)
        self.assertIn("- N1: port heights approximate", body)

    def test_unresolved_lists_what_rebuild_cannot_replay(self):
        S.record(self.proj, tool="execute_code_async", geometry=True, args={"code": "x"})
        S.record(self.proj, tool="source_reference_part", geometry=False, args={"reason": "r"})
        S.record(self.proj, tool="create_object", ok=False, text="failed: bad type")
        body = section(self.text(), "## 8. Unresolved")
        self.assertIn("rebuild.py does not replay execute_code_async (steps 1)", body)
        self.assertIn("rebuild.py does not replay source_reference_part (steps 2)", body)
        self.assertIn("rebuild.py skips 1 failed call(s): steps 3", body)

    def test_steps_log_with_superseded_steps_struck_through(self):
        S.record(self.proj, title="Shell")
        S.record(self.proj, title="Lid")
        S.record(self.proj, tool="create_object", title="Broken", ok=False, text="Bad arguments: unknown")
        with self.proj._lock:
            self.proj._journal_line({"type": "rewind", "ts": "x", "to_step": 1, "restored_checkpoint": "001"})
        log = section(self.text(), "## 9. Steps log")
        self.assertIn("| # | tool | title | ok | snapshot | note |", log)
        self.assertIn("| 1 | execute_code | Shell | yes | - | no document |", log)
        self.assertIn("| ~~2~~ | ~~execute_code~~ | ~~Lid~~ | ~~yes~~ | - | ~~no document~~ |", log)
        self.assertIn("~~Broken~~", log)
        self.assertIn("Bad arguments: unknown", log)   # a failed step shows the head of its result

    def test_table_cells_are_neutralised(self):
        self.proj.note("assumption", "a | b\nc")
        S.record(self.proj, title="pipe | and\nnewline")
        text = self.text()
        self.assertIn("| pipe \\| and newline |", text)    # escaped exactly once
        self.assertNotIn("pipe | and", text)
        self.assertNotIn("\\\\|", text)                   # never double-escaped
        self.assertIn("- N1: a \\| b c", text)

    def test_artifacts_section_lists_existing_files(self):
        S.new_temp_file(self.proj.path("exports"), "a.step")
        S.new_temp_file(self.proj.path("refs"), "abc_pi.step")
        body = section(self.text(), "## 10. Artifacts")
        self.assertIn("- exports/a.step", body)
        self.assertIn("- refs/abc_pi.step", body)

    def test_the_text_is_deterministic(self):
        S.record(self.proj)
        self.proj.note("plan", "x")
        first = self.text(ledger())
        second = self.text(ledger())
        self.assertEqual(first, second)

    def test_every_step_regenerates_notes_and_rebuild(self):
        S.record(self.proj, title="First thing")
        notes = open(self.proj.path("NOTES.md"), encoding="utf-8").read()
        self.assertIn("First thing", notes)
        self.assertTrue(os.path.isfile(self.proj.path("rebuild.py")))
        S.record(self.proj, title="Second thing")
        self.assertIn("Second thing", open(self.proj.path("NOTES.md"), encoding="utf-8").read())

    def test_regeneration_takes_under_50_ms_for_200_steps(self):
        for i in range(200):
            S.record(self.proj, title=f"Step number {i}", args={"code": f"print({i})"})
        self.proj.save_ledger(ledger())
        best = 1e9
        for _ in range(5):
            started = time.perf_counter()
            self.proj.write_notes_md()
            self.proj.write_rebuild_py()
            best = min(best, time.perf_counter() - started)
        self.assertLess(best, 0.05, f"NOTES.md + rebuild.py regeneration took {best * 1000:.0f} ms for 200 steps")


class RebuildTextTest(unittest.TestCase):
    def setUp(self):
        self.proj = S.make_project(create=True)

    def literal(self, text: str, name: str):
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign) and node.targets[0].id == name:
                return ast.literal_eval(node.value)
        raise AssertionError(f"{name} not found")

    def test_the_script_compiles_and_has_the_documented_parts(self):
        S.record(self.proj, tool="create_document", geometry=False, args={"name": "PiCase"},
                 text="Document 'PiCase' created and active.")
        S.record(self.proj, tool="create_object", args={"doc_name": "PiCase", "obj_type": "Part::Box", "obj_name": "Box"})
        text = self.proj.rebuild_script_text()
        compile(text, "rebuild.py", "exec")
        self.assertTrue(text.startswith("#!/usr/bin/env python\n# Generated by Oscar. Rebuilds \"Pi 4 housing\" by replaying"))
        for needle in ("OSCAR_REBUILD_SET", "import FreeCAD as App", "REBUILD_REFUSED", "project.replay_steps",
                       "project.steps_from_literal(STEPS)", "REBUILD_RESULT ", "REBUILD_OK", "REBUILD_FAILED",
                       "sys.exit(_RC)", "getattr(App, \"GuiUp\", False)"):
            self.assertIn(needle, text, needle)
        self.assertIn("(the app refuses to replay into a document that is open in it)", text)
        self.assertEqual(self.literal(text, "DOC_NAME"), "PiCase")
        self.assertEqual(self.literal(text, "PROJECT_DIR"), self.proj.root)
        self.assertEqual(os.path.basename(self.literal(text, "OSCAR_MOD_DIR")),
                         os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__import__("oscar").__file__)))))

    def test_sys_exit_is_guarded_so_the_gui_console_never_exits(self):
        text = self.proj.rebuild_script_text()
        tail = text.split("_RC = main()")[1]
        self.assertIn('if not getattr(App, "GuiUp", False):', tail)
        self.assertIn("sys.exit(_RC)", tail)
        self.assertGreater(tail.index("sys.exit(_RC)"), tail.index("GuiUp"))

    def test_steps_is_a_python_literal_in_index_order(self):
        S.record(self.proj, tool="create_document", geometry=False, args={"name": "D"}, text="Document 'D' created")
        S.record(self.proj, tool="execute_code", args={"code": "print(1)"}, title="Make a plate")
        S.record(self.proj, tool="edit_object", args={"doc_name": "D", "obj_name": "Plate", "flag": True, "none": None})
        text = self.proj.rebuild_script_text()
        steps = self.literal(text, "STEPS")
        self.assertEqual([s["index"] for s in steps], [1, 2, 3])
        self.assertEqual([s["kind"] for s in steps], ["tool", "script", "tool"])
        self.assertEqual(steps[1]["file"], "scripts/002_make-a-plate.py")
        self.assertEqual(steps[1]["headless"], False)
        self.assertEqual(steps[2]["args"], {"doc_name": "D", "obj_name": "Plate", "flag": True, "none": None})
        literal_text = text.split("STEPS = ")[1].split("\n\ndef main")[0]
        self.assertIn("'flag': True", literal_text)   # a Python literal, never json.dumps output
        self.assertNotIn("true", literal_text)
        self.assertNotIn("null", literal_text)
        self.assertIn("'none': None", literal_text)

    def test_only_successful_replayable_live_steps_are_in_the_script(self):
        S.record(self.proj, tool="create_object", args={"doc_name": "D", "obj_type": "Part::Box"})       # 1 kept
        S.record(self.proj, tool="create_object", args={"doc_name": "D", "obj_type": "Bad"}, ok=False)   # 2 failed
        S.record(self.proj, tool="execute_code_async", args={"code": "x"})                               # 3 not replayable
        S.record(self.proj, tool="source_reference_part", geometry=False, args={"reason": "r"})          # 4 not replayable
        S.record(self.proj, tool="edit_object", args={"doc_name": "D", "obj_name": "Box"})               # 5 superseded
        with self.proj._lock:
            self.proj._journal_line({"type": "rewind", "ts": "x", "to_step": 1, "restored_checkpoint": "001"})
        steps = self.literal(self.proj.rebuild_script_text(), "STEPS")
        self.assertEqual([s["index"] for s in steps], [1])

    def test_paths_with_quotes_backslashes_and_spaces_survive(self):
        root = os.path.join(S.temp_root(), "O'Brien \"quoted\" back\\slash")
        proj = P.Project.for_chat("20261001-000000-dd0004", 'He said "hi"', projects_root=root, create=True)
        S.record(proj, tool="create_object", args={"doc_name": "D", "obj_type": "Part::Box"})
        text = proj.rebuild_script_text()
        compile(text, "rebuild.py", "exec")
        self.assertEqual(self.literal(text, "PROJECT_DIR"), proj.root)
        self.assertIn("by replaying", text.split("\n")[1])
        self.assertNotIn('""', text.split("\n")[1])

    def test_params_are_informational(self):
        with open(self.proj.path("params.json"), "w") as f:
            json.dump({"schema": 1, "params": [{"name": "wall_t", "value": 2.0, "unit": "mm"},
                                               {"name": "vent_count", "value": 6, "unit": ""}]}, f)
        text = self.proj.rebuild_script_text()
        self.assertEqual(self.literal(text, "PARAMS"), {"wall_t": "2 mm", "vent_count": "6"})

    def test_write_rebuild_py_returns_the_relative_path_and_writes_atomically(self):
        self.assertEqual(self.proj.write_rebuild_py(), "rebuild.py")
        self.assertEqual(sorted(n for n in os.listdir(self.proj.root) if "rebuild" in n), ["rebuild.py"])
        compile(open(self.proj.path("rebuild.py"), encoding="utf-8").read(), "rebuild.py", "exec")

    def test_the_literal_round_trips_through_steps_from_literal(self):
        S.record(self.proj, tool="create_document", geometry=False, args={"name": "D"}, text="Document 'D' created")
        S.record(self.proj, tool="execute_code", args={"code": "print(1)"}, title="Make a plate", doc="D")
        S.record(self.proj, tool="execute_code_headless", geometry=False, args={"code": "print(2)"}, title="Headless bit")
        S.record(self.proj, tool="execute_code", args={"code": "o.ViewObject.Visibility = False"}, title="Hide it")
        records = P.steps_from_literal(self.literal(self.proj.rebuild_script_text(), "STEPS"))
        self.assertEqual([r.index for r in records], [1, 2, 3, 4])
        self.assertTrue(all(r.ok and not r.superseded for r in records))
        self.assertEqual(records[0].replay, {"kind": "tool", "name": "create_document", "args": {"name": "D"}})
        self.assertEqual(records[1].replay, {"kind": "script", "file": "scripts/002_make-a-plate.py",
                                             "headless": False, "gui_only": False})
        self.assertEqual(records[2].replay["headless"], True)
        self.assertEqual(records[3].replay["gui_only"], True)
        self.assertEqual(records[1].doc, "D")
        self.assertEqual(records[1].title, "Make a plate")
        self.assertEqual(records[0].tool, "create_document")

    def test_steps_from_literal_defaults(self):
        records = P.steps_from_literal([{"index": 4, "tool": "x"}, {"index": 5, "tool": "y", "kind": "tool"}])
        self.assertEqual(records[0].replay["kind"], "none")
        self.assertEqual(records[1].replay, {"kind": "tool", "name": "y", "args": {}})
        self.assertEqual(P.steps_from_literal([]), [])
        self.assertEqual(P.steps_from_literal(None), [])


class RebuildCheckHelpersTest(unittest.TestCase):
    def test_parse_rebuild_output(self):
        out = ("PASS step 001 create_document\nFAIL step 002 edit_object\n"
               'REBUILD_RESULT {"docs": {"PiCase": {"Shell": 18432.1, "Sk": null}}}\nREBUILD_OK\n')
        marker, summary = R.parse_rebuild_output(out)
        self.assertTrue(marker)
        self.assertEqual(summary, {"docs": {"PiCase": {"Shell": 18432.1, "Sk": None}}})

    def test_parse_rebuild_output_edge_cases(self):
        self.assertEqual(R.parse_rebuild_output(""), (False, None))
        self.assertEqual(R.parse_rebuild_output(None), (False, None))
        self.assertEqual(R.parse_rebuild_output("REBUILD_FAILED\n"), (False, None))
        self.assertEqual(R.parse_rebuild_output("REBUILD_RESULT {broken\nREBUILD_OK"), (True, None))
        self.assertEqual(R.parse_rebuild_output('REBUILD_RESULT [1, 2]\nREBUILD_OK'), (True, None))
        self.assertEqual(R.parse_rebuild_output("  REBUILD_OK  \r\n")[0], True)   # padded / CRLF tolerant
        self.assertEqual(R.parse_rebuild_output("NOT_REBUILD_OK_REALLY")[0], False)

    def test_compare_equal(self):
        live = {"D": {"Shell": 18432.1, "Lid": 5000.0, "Sk": None}}
        self.assertEqual(R.compare_rebuild(live, {"D": {"Shell": 18432.1 * (1 + 1e-9), "Lid": 5000.0, "Sk": None}}), [])
        self.assertEqual(R.compare_rebuild({}, {}), [])
        self.assertEqual(R.compare_rebuild({"D": {}}, {"D": {}}), [])

    def test_compare_reports_each_kind_of_difference(self):
        live = {"D": {"Shell": 100.0, "Lid": 50.0}, "E": {"A": 1.0}}
        rebuilt = {"D": {"Shell": 100.001, "Extra": 3.0}}
        problems = R.compare_rebuild(live, rebuilt)
        joined = "\n".join(problems)
        self.assertIn("document 'E' was not rebuilt", joined)
        self.assertIn("object names differ in 'D'", joined)
        self.assertIn("missing after rebuild: ['Lid']", joined)
        self.assertIn("extra: ['Extra']", joined)
        self.assertIn("volume of 'D.Shell' differs", joined)

    def test_the_tolerance_is_relative_and_inclusive_of_none(self):
        self.assertEqual(R.compare_rebuild({"D": {"A": 1e6}}, {"D": {"A": 1e6 + 0.5}}), [])        # 5e-7 relative
        self.assertEqual(len(R.compare_rebuild({"D": {"A": 1e6}}, {"D": {"A": 1e6 + 5}})), 1)      # 5e-6 relative
        self.assertEqual(len(R.compare_rebuild({"D": {"A": None}}, {"D": {"A": 3.0}})), 1)
        self.assertEqual(len(R.compare_rebuild({"D": {"A": 3.0}}, {"D": {"A": None}})), 1)
        self.assertEqual(len(R.compare_rebuild({"D": {"A": 3.0}}, {"D": {"A": float("nan")}})), 1)
        self.assertEqual(R.compare_rebuild({"D": {"A": 0.0}}, {"D": {"A": 0.0}}), [])


if __name__ == "__main__":
    unittest.main()
