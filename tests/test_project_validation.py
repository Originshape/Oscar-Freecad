"""The Project (PURE: runs on a plain python as well as under freecadcmd): consuming the
``ValidationResult`` (``project_validation.normalize`` / ``merge`` / ``stale_reason`` / ``tree_from_health``,
``Project.record_validation`` / ``finalize`` / ``build_summary``), the provenance claim rule, stale exports,
``artifact_list`` for the panel, and the threading rules that must hold for EVERY Project method that waits on
the GUI thread (no lock across a dispatch; every ``saveCopy`` temp name ends in ``.FCStd``)."""

import copy
import dataclasses
import json
import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import project_support as S  # noqa: E402
from oscar import project as P  # noqa: E402
from oscar import project_export as X  # noqa: E402
from oscar import project_gui as gui  # noqa: E402
from oscar import project_validation as V  # noqa: E402

NOW = "2026-10-01T21:52:10"


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def health(**over):
    data = {"doc": "PiCase", "ts": 1.0, "valid": True, "counts": {"objects": 7, "solids": 2, "errors": 0, "warnings": 0},
            "findings": [], "rows": [], "delta": None, "fragments": [], "budget_exceeded": False}
    data.update(over)
    return data


def full(**over):
    """A complete passing ValidationResult dict (the shape) with one row per fit entry."""
    data = {
        "ok": True, "doc": "PiCase", "health": health(), "gate_errors": [],
        "fit_rows": [
            {"id": "interference.shell_lid", "kind": "interference", "a": "Shell", "b": "Lid",
             "metric": "common_volume_mm3", "value": 0.0, "limit": "<= 0.05", "status": "PASS", "detail": ""},
            {"id": "clearance.pcb_to_wall_min", "kind": "clearance", "status": "PASS"},
            {"id": "containment.ports", "kind": "containment", "status": "PASS"},
            {"id": "hooks.engagement", "kind": "alignment", "status": "PASS"},
            {"id": "retention.lid_lift", "kind": "retention", "status": "PASS"},
            {"id": "reference.status", "kind": "coverage", "status": "WARN", "detail": "nominal board data"}],
        "fit_ran": True, "fail_rows": [], "dfm": [{"id": "dfm.min_wall", "status": "PASS", "detail": "2.0 >= 1.2"}],
        "coverage": [{"requirement": "R2", "status": "covered", "by": ["S6"], "detail": "11/11 openings cut"}],
        "ran": ["health", "fit.interference", "fit.clearance", "fit.ports", "fit.retention", "dfm"],
        "not_validated": ["editability", "rebuild", "step_collision", "physical"], "text": "VALIDATE ...",
        "after_step": 2}
    data.update(over)
    return data


def health_only(**over):
    data = {"ok": True, "doc": "PiCase", "health": health(counts={"objects": 9, "solids": 3, "errors": 0, "warnings": 0}),
            "gate_errors": [], "fit_rows": [], "dfm": [], "coverage": [], "ran": ["health"],
            "not_validated": ["physical"], "after_step": 2}
    data.update(over)
    return data


class NormalizeTest(unittest.TestCase):
    def norm(self, data, default=5):
        return V.normalize(data, now=NOW, default_after_step=default)

    def test_an_empty_dict_gets_every_field_with_defaults(self):
        rec = self.norm({})
        self.assertEqual(sorted(rec), sorted(["schema", "ts", "after_step", "doc", "ok", "health", "gate_errors",
                                              "fit_rows", "dfm", "coverage", "ran", "not_validated"]))
        self.assertEqual((rec["schema"], rec["ts"], rec["after_step"], rec["doc"]), (1, NOW, 5, ""))
        self.assertEqual((rec["health"], rec["gate_errors"], rec["fit_rows"], rec["dfm"], rec["coverage"], rec["ran"],
                          rec["not_validated"]), ({}, [], [], [], [], [], []))
        self.assertTrue(rec["ok"])

    def test_a_validation_result_keeps_the_contract_fields_and_drops_the_renderings(self):
        rec = self.norm(full())
        for gone in ("text", "fit_ran", "fail_rows"):
            self.assertNotIn(gone, rec)
        self.assertEqual((rec["doc"], rec["after_step"], rec["ran"]), ("PiCase", 2, full()["ran"]))
        self.assertEqual(rec["coverage"][0]["requirement"], "R2")
        self.assertEqual(rec["health"]["counts"]["objects"], 7)

    def test_ok_is_computed_when_missing_and_taken_as_given_otherwise(self):
        self.assertFalse(self.norm({"gate_errors": [{"code": "FRAGMENTS", "object": "Shell"}]})["ok"])
        self.assertFalse(self.norm({"fit_rows": [{"id": "x", "status": "fail"}]})["ok"])
        self.assertTrue(self.norm({"fit_rows": [{"id": "x", "status": "WARN"}]})["ok"])
        self.assertFalse(self.norm({"ok": False})["ok"])
        self.assertTrue(self.norm({"ok": True, "gate_errors": [{"code": "X"}]})["ok"])   # the producer decides

    def test_wrongly_typed_fields_become_defaults_instead_of_raising(self):
        rec = self.norm({"health": "nope", "gate_errors": "x", "fit_rows": [1, None, {"id": "a", "status": "PASS"}],
                         "dfm": {"a": 1}, "coverage": 7, "ran": ["health", 3, None], "not_validated": "physical",
                         "after_step": "abc", "doc": None})
        self.assertEqual((rec["health"], rec["gate_errors"], rec["dfm"], rec["coverage"], rec["not_validated"]),
                         ({}, [], [], [], []))
        self.assertEqual(rec["fit_rows"], [{"id": "a", "status": "PASS"}])
        self.assertEqual((rec["ran"], rec["after_step"], rec["doc"]), (["health"], 5, ""))

    def test_after_step_is_coerced_and_zero_is_kept(self):
        self.assertEqual(self.norm({"after_step": "7"})["after_step"], 7)
        self.assertEqual(self.norm({"after_step": 0})["after_step"], 0)
        self.assertEqual(self.norm({"after_step": None})["after_step"], 5)
        self.assertEqual(self.norm({"after_step": float("inf")})["after_step"], 5)

    def test_non_finite_numbers_and_objects_become_json_safe_text(self):
        rec = self.norm({"health": health(ts=float("nan"), rows=[{"name": "A", "volume": float("inf")}])})
        json.dumps(rec, allow_nan=False)

    def test_the_input_is_not_modified(self):
        data = full()
        before = copy.deepcopy(data)
        self.norm(data)
        self.assertEqual(data, before)

    def test_health_rows_and_findings_are_capped(self):
        rows = [{"name": f"O{i}", "solids": 1} for i in range(450)]
        findings = [{"severity": "info", "code": "X", "object": f"O{i}", "message": "m"} for i in range(260)]
        rec = self.norm({"health": health(rows=rows, findings=findings)})
        self.assertEqual(len(rec["health"]["rows"]), V.HEALTH_ROWS_CAP)
        self.assertEqual(rec["health"]["rows_omitted"], 450 - V.HEALTH_ROWS_CAP)
        self.assertEqual(len(rec["health"]["findings"]), V.HEALTH_FINDINGS_CAP)
        small = self.norm({"health": health(rows=rows[:10])})
        self.assertNotIn("rows_omitted", small["health"])


class MergeTest(unittest.TestCase):
    def rec(self, data):
        return V.normalize(data, now=NOW, default_after_step=0)

    def test_another_model_state_is_never_carried_over(self):
        old = self.rec(full(after_step=2))
        for other in (health_only(after_step=3), health_only(after_step=1), health_only(doc="Other")):
            new = self.rec(other)
            self.assertIs(V.merge(old, new), new)
        new = self.rec(health_only())
        self.assertIs(V.merge(None, new), new)
        self.assertIs(V.merge("garbage", new), new)

    def test_health_alone_after_a_full_run_keeps_the_fit_dfm_and_coverage_evidence(self):
        merged = V.merge(self.rec(full()), self.rec(health_only()))
        self.assertEqual(merged["ran"], ["health", "fit.interference", "fit.clearance", "fit.ports", "fit.retention", "dfm"])
        self.assertEqual(len(merged["fit_rows"]), 6)
        self.assertEqual(merged["dfm"][0]["id"], "dfm.min_wall")
        self.assertEqual(merged["coverage"][0]["requirement"], "R2")
        self.assertEqual(merged["health"]["counts"]["objects"], 9, "the fresh health report wins")
        self.assertEqual(merged["not_validated"], ["physical", "editability", "rebuild", "step_collision"])
        self.assertTrue(merged["ok"])
        self.assertEqual(merged["ts"], NOW)

    def test_a_fit_check_alone_replaces_only_its_own_rows_and_keeps_health(self):
        new = self.rec({"doc": "PiCase", "after_step": 2, "ran": ["fit.interference"], "ok": False,
                        "fit_rows": [{"id": "interference.shell_lid", "kind": "interference", "status": "FAIL",
                                      "detail": "12 mm3"}]})
        merged = V.merge(self.rec(full()), new)
        by_id = {r["id"]: r["status"] for r in merged["fit_rows"]}
        self.assertEqual(by_id["interference.shell_lid"], "FAIL")
        self.assertEqual(by_id["clearance.pcb_to_wall_min"], "PASS")
        self.assertEqual(len([r for r in merged["fit_rows"] if r["id"] == "interference.shell_lid"]), 1, "no duplicate row")
        self.assertNotIn("reference.status", by_id, "a coverage row follows the fit run that is fresh")
        self.assertEqual(merged["health"]["counts"]["objects"], 7)
        self.assertEqual(merged["ran"], ["health", "fit.interference", "fit.clearance", "fit.ports", "fit.retention", "dfm"])
        self.assertFalse(merged["ok"])

    def test_a_row_that_belongs_to_no_entry_is_kept_while_no_fit_entry_is_fresh(self):
        new = self.rec(health_only())
        merged = V.merge(self.rec(full()), new)
        self.assertIn("reference.status", {r["id"] for r in merged["fit_rows"]})

    def test_deep_editability_adds_its_id_without_losing_the_rest(self):
        new = self.rec(health_only(ran=["health", "editability"]))
        merged = V.merge(self.rec(full()), new)
        self.assertEqual(merged["ran"], ["health", "fit.interference", "fit.clearance", "fit.ports", "fit.retention",
                                         "dfm", "editability"])
        self.assertNotIn("editability", merged["not_validated"])

    def test_editability_of_the_old_record_is_carried_by_id(self):
        old = self.rec(full(ran=full()["ran"] + ["editability"]))
        self.assertIn("editability", V.merge(old, self.rec(health_only()))["ran"])

    def test_failures_and_gate_errors_of_the_old_record_keep_ok_false(self):
        failing = full(ok=False, fit_rows=[{"id": "interference.shell_lid", "kind": "interference", "status": "FAIL"}])
        merged = V.merge(self.rec(failing), self.rec(health_only(ran=["health", "dfm"], dfm=[])))
        self.assertTrue(any(r["status"] == "FAIL" for r in merged["fit_rows"]))
        self.assertFalse(merged["ok"])
        gated = full(ok=False, gate_errors=[{"code": "FRAGMENTS", "object": "Shell", "message": "3 solids"}])
        new = self.rec({"doc": "PiCase", "after_step": 2, "ran": ["dfm"], "dfm": [], "ok": True})
        merged = V.merge(self.rec(gated), new)
        self.assertEqual(merged["gate_errors"][0]["code"], "FRAGMENTS")
        self.assertFalse(merged["ok"])

    def test_a_fresh_gate_result_replaces_the_old_one(self):
        gated = full(ok=False, gate_errors=[{"code": "FRAGMENTS", "object": "Shell", "message": "3 solids"}])
        merged = V.merge(self.rec(gated), self.rec(health_only()))
        self.assertEqual(merged["gate_errors"], [])
        self.assertTrue(merged["ok"])

    def test_a_full_new_record_replaces_everything(self):
        new = self.rec(full(fit_rows=[{"id": "a", "kind": "clearance", "status": "WARN"}], dfm=[]))
        merged = V.merge(self.rec(full()), new)
        self.assertEqual([r["id"] for r in merged["fit_rows"]], ["a"])
        self.assertEqual(merged["dfm"], [])

    def test_new_coverage_rows_win_and_an_empty_list_does_not_wipe_the_old_ones(self):
        new = self.rec(health_only(coverage=[{"requirement": "R9", "status": "covered"}]))
        self.assertEqual(V.merge(self.rec(full()), new)["coverage"][0]["requirement"], "R9")
        self.assertEqual(V.merge(self.rec(full()), self.rec(health_only()))["coverage"][0]["requirement"], "R2")

    def test_an_old_record_with_damaged_fields_does_not_break_the_merge(self):
        old = {"doc": "PiCase", "after_step": 2, "ran": "health", "fit_rows": "x", "health": 7}
        merged = V.merge(old, self.rec(health_only()))
        self.assertEqual(merged["ran"], ["health"])

    def test_unknown_run_ids_are_kept_after_the_catalog_ones(self):
        merged = V.merge(self.rec(full()), self.rec(health_only(ran=["health", "custom.check"])))
        self.assertEqual(merged["ran"][-1], "custom.check")
        self.assertEqual(merged["ran"][0], "health")


class StaleReasonTest(unittest.TestCase):
    class Step:
        def __init__(self, index, geometry=True, ok=True, superseded=False):
            self.index, self.geometry, self.ok, self.superseded = index, geometry, ok, superseded

    def test_fresh_stale_and_rewound(self):
        steps = [self.Step(1), self.Step(2)]
        self.assertEqual(V.stale_reason({"after_step": 2}, steps), "")
        self.assertIn("changed at step 2", V.stale_reason({"after_step": 1}, steps))
        self.assertIn("validate again", V.stale_reason({"after_step": 1}, steps))
        self.assertEqual(V.stale_reason(None, steps), "")
        self.assertEqual(V.stale_reason({}, []), "")

    def test_a_rewind_below_the_validation_is_stale(self):
        steps = [self.Step(1), self.Step(2, superseded=True), self.Step(3, superseded=True)]
        self.assertIn("rewound to step 1", V.stale_reason({"after_step": 3}, steps))

    def test_an_after_step_above_the_newest_without_a_rewind_is_not_called_stale(self):
        self.assertEqual(V.stale_reason({"after_step": 4}, [self.Step(1)]), "")
        self.assertEqual(V.stale_reason({"after_step": 4}, [self.Step(1), self.Step(2, geometry=False, superseded=True)]), "")

    def test_failed_and_non_geometry_steps_do_not_make_a_validation_stale(self):
        steps = [self.Step(1), self.Step(2, ok=False), self.Step(3, geometry=False)]
        self.assertEqual(V.stale_reason({"after_step": 1}, steps), "")

    def test_a_missing_after_step_counts_as_zero(self):
        self.assertTrue(V.stale_reason({"doc": "PiCase"}, [self.Step(1)]).startswith("the validation ran after step 0"))
        self.assertEqual(V.stale_reason({"doc": "PiCase"}, []), "")
        self.assertEqual(V.stale_reason({"after_step": "x"}, []), "")


class TreeFromHealthTest(unittest.TestCase):
    def test_only_solids_are_listed_tips_first_then_by_volume(self):
        rows = [
            {"name": "Sketch", "type": "Sketcher::SketchObject", "solids": 0, "shape_type": "Wire", "volume": None},
            {"name": "Base", "type": "Part::Box", "solids": 1, "volume": 500.0, "consumed": True, "visible": False},
            {"name": "Shell", "type": "Part::Cut", "solids": 1, "volume": 18432.1, "bbox": [0, 0, 0, 94, 65, 22], "visible": True},
            {"name": "Lid", "type": "Part::Cut", "solids": 1, "volume": 9000.0, "visible": True},
            {"name": "Hidden", "type": "Part::Box", "solids": 1, "volume": 99999.0, "visible": False},
            {"name": "Ref_RPI4B", "type": "App::Part", "solids": 3, "volume": 70000.0, "visible": False, "role": "ref"},
            {"name": "Imported", "type": "Mesh::Feature", "solids": 0, "shape_type": "Mesh", "volume": None}]
        tree = V.tree_from_health({"health": {"rows": rows}})
        self.assertEqual([r["name"] for r in tree], ["Shell", "Lid", "Hidden", "Ref_RPI4B", "Base", "Imported"])
        self.assertEqual([r["name"] for r in tree if r["tip"]], ["Shell", "Lid"])
        self.assertFalse(next(r for r in tree if r["name"] == "Base")["tip"])
        self.assertFalse(next(r for r in tree if r["name"] == "Ref_RPI4B")["tip"])
        self.assertEqual(tree[0]["bbox"], [0, 0, 0, 94, 65, 22])

    def test_missing_or_broken_input_gives_an_empty_tree(self):
        for value in (None, {}, {"health": None}, {"health": {"rows": "x"}}, {"health": {"rows": [1, None]}}):
            self.assertEqual(V.tree_from_health(value), [])


class RecordedValidationTest(unittest.TestCase):
    """``Project.record_validation`` consumes the ValidationResult and merges partial records."""

    def setUp(self):
        self.proj = S.make_project(create=True)
        S.record(self.proj)
        S.record(self.proj, call_id="c2")   # newest geometry step = 2 (no document: no GUI work)

    def stored(self):
        return self.proj.read_validation()

    def test_the_validation_result_dataclass_is_stored_with_the_contract_fields_only(self):
        @dataclasses.dataclass
        class ValidationResult:
            ok: bool
            doc: str
            health: dict
            gate_errors: list
            fit_rows: list
            fit_ran: bool
            fail_rows: list
            dfm: list
            coverage: list
            ran: list
            not_validated: list
            text: str
            after_step: int

        data = full()
        self.proj.record_validation(ValidationResult(**data))
        rec = self.stored()
        self.assertEqual(sorted(rec), sorted(["schema", "ts", "after_step", "doc", "ok", "health", "gate_errors", "fit_rows",
                                              "dfm", "coverage", "ran", "not_validated"]))
        self.assertEqual((rec["after_step"], rec["doc"], rec["ok"]), (2, "PiCase", True))
        self.assertEqual(rec["ran"], data["ran"])

    def test_none_is_refused(self):
        with self.assertRaises(P.ProjectError):
            self.proj.record_validation(None)
        self.assertIsNone(self.stored())

    def test_a_partial_record_after_a_full_one_keeps_the_evidence(self):
        self.proj.record_validation(full())
        self.proj.record_validation(health_only(health=health(counts={"objects": 11, "solids": 3, "errors": 0, "warnings": 0})))
        rec = self.stored()
        self.assertEqual(rec["ran"], full()["ran"])
        self.assertEqual(len(rec["fit_rows"]), 6)
        self.assertEqual(rec["health"]["counts"]["objects"], 11)
        notes = read_text(self.proj.path("NOTES.md"))
        self.assertIn("Part-to-part interference", notes.split("## 7. Validation")[1].split("## 8.")[0])

    def test_a_record_after_more_geometry_replaces_instead_of_carrying_stale_evidence(self):
        self.proj.record_validation(full())
        S.record(self.proj, call_id="c3")   # the model changed: step 3
        self.proj.record_validation(health_only(after_step=None))
        rec = self.stored()
        self.assertEqual((rec["after_step"], rec["ran"], rec["fit_rows"], rec["dfm"]), (3, ["health"], [], []))

    def test_a_partial_record_without_after_step_uses_the_newest_geometry_step(self):
        self.proj.record_validation(full())
        data = health_only()
        data.pop("after_step")
        self.proj.record_validation(data)
        self.assertEqual(self.stored()["ran"], full()["ran"], "same step 2: the full record is kept")

    def test_a_validation_of_another_document_replaces_the_record(self):
        self.proj.record_validation(full())
        self.proj.record_validation(health_only(doc="Other"))
        rec = self.stored()
        self.assertEqual((rec["doc"], rec["ran"], rec["fit_rows"]), ("Other", ["health"], []))

    def test_health_rows_are_capped_on_disk(self):
        rows = [{"name": f"O{i}", "solids": 1, "volume": 1.0} for i in range(500)]
        self.proj.record_validation(full(health=health(rows=rows)))
        rec = self.stored()
        self.assertEqual(len(rec["health"]["rows"]), V.HEALTH_ROWS_CAP)
        self.assertEqual(rec["health"]["rows_omitted"], 100)

    def test_a_disk_failure_is_logged_not_raised(self):
        with mock.patch.object(P, "_atomic_write_json", side_effect=OSError("disk full")):
            self.assertIsNone(self.proj.record_validation(full()))


class SummaryUsesTheRecordTest(unittest.TestCase):
    def setUp(self):
        self.proj = S.make_project(create=True)
        S.record(self.proj)
        S.record(self.proj, call_id="c2")
        self.ledger = {"intent": "enclosure", "request": "case", "requirements": [], "steps": []}

    def validated(self, summary):
        return [v["id"] for v in summary.data["validated"]]

    def test_finalize_stores_the_validation_first_so_notes_and_the_summary_agree(self):
        summary = self.proj.finalize(ledger=self.ledger, outcome="validated", validation=full(), export=False)
        self.assertEqual(self.proj.read_validation()["ran"], full()["ran"])
        self.assertEqual(self.validated(summary), ["health", "fit.interference", "fit.clearance", "fit.ports",
                                                   "fit.retention", "dfm"])
        notes = read_text(self.proj.path("NOTES.md"))
        self.assertIn("Model health (valid solids, no recompute errors): valid (7 objects, 2 solids)", notes)

    def test_finalize_merges_a_partial_validation_with_the_stored_one(self):
        self.proj.record_validation(full())
        summary = self.proj.finalize(ledger=self.ledger, outcome="validated", validation=health_only(), export=False)
        self.assertEqual(self.validated(summary)[:2], ["health", "fit.interference"])
        self.assertIn("valid (9 objects, 3 solids)", summary.markdown)

    def test_without_a_validation_argument_the_stored_record_is_used(self):
        self.proj.record_validation(full())
        summary = self.proj.finalize(ledger=self.ledger, outcome="answered", validation=None, export=False)
        self.assertEqual(self.validated(summary)[0], "health")
        self.assertNotIn("stale", summary.markdown)

    def test_without_any_record_nothing_is_claimed(self):
        summary = self.proj.finalize(ledger=self.ledger, outcome="answered", validation=None, export=False)
        self.assertEqual(summary.data["validated"], [])
        self.assertIn("nothing validated", summary.markdown)

    def test_a_stored_record_older_than_the_model_is_stale(self):
        self.proj.record_validation(full())
        S.record(self.proj, call_id="c3")
        summary = self.proj.build_summary(self.ledger, "answered", None)
        self.assertEqual(summary.data["validated"], [])
        self.assertTrue(any("validate again" in i for i in summary.data["open_items"]))

    def test_a_rewind_below_the_validation_makes_it_stale(self):
        self.proj.record_validation(full())
        with self.proj._lock:
            self.proj._append_raw({"type": "rewind", "ts": NOW, "to_step": 1, "restored_checkpoint": "001"})
            self.proj._jcache = None
        self.assertEqual(self.proj.last_geometry_step(), 1)
        summary = self.proj.build_summary(self.ledger, "answered", None)
        self.assertEqual(summary.data["validated"], [])
        self.assertTrue(any("rewound to step 1" in i for i in summary.data["open_items"]), summary.data["open_items"])
        self.assertIn("stale", summary.markdown)

    def test_a_passed_validation_that_is_older_than_the_model_does_not_count(self):
        summary = self.proj.build_summary(self.ledger, "validated", full(after_step=1))
        self.assertEqual(summary.data["validated"], [])

    def test_failures_of_a_merged_record_show_up_as_open_items_and_not_validated(self):
        failing = full(ok=False, fit_rows=[{"id": "interference.shell_lid", "kind": "interference", "status": "FAIL",
                                            "detail": "12 mm3"}])
        self.proj.record_validation(failing)
        summary = self.proj.finalize(ledger=self.ledger, outcome="needs_help", validation=health_only(), export=False)
        self.assertTrue(any("fit FAIL interference.shell_lid" in i for i in summary.data["open_items"]))
        self.assertNotIn("fit.interference", self.validated(summary))

    def test_a_health_report_that_ran_out_of_time_says_so(self):
        summary = self.proj.build_summary(self.ledger, "validated", full(health=health(budget_exceeded=True)))
        line = next(v["result"] for v in summary.data["validated"] if v["id"] == "health")
        self.assertIn("some objects were not checked (time budget)", line)
        ok = self.proj.build_summary(self.ledger, "validated", full())
        self.assertNotIn("time budget", next(v["result"] for v in ok.data["validated"] if v["id"] == "health"))

    def test_pre_existing_errors_are_open_items_but_not_gate_errors(self):
        pre = {"severity": "error", "code": "SKETCH_CONFLICT", "object": "UserSketch", "message": "conflicting constraints"}
        summary = self.proj.build_summary(self.ledger, "validated", full(health=health(
            valid=False, counts={"objects": 7, "solids": 2, "errors": 1, "warnings": 0}, findings=[pre])))
        self.assertIn("pre-existing: SKETCH_CONFLICT UserSketch: conflicting constraints", summary.data["open_items"])
        line = next(v["result"] for v in summary.data["validated"] if v["id"] == "health")
        self.assertEqual(line, "no new errors (1 pre-existing)")

    def test_the_model_tree_falls_back_to_the_health_rows_when_no_document_is_readable(self):
        rows = [{"name": "Shell", "type": "Part::Cut", "solids": 1, "volume": 18432.1, "bbox": [0, 0, 0, 94, 65, 22],
                 "visible": True},
                {"name": "Sketch", "type": "Sketcher::SketchObject", "solids": 0, "shape_type": "Wire"}]
        summary = self.proj.build_summary(self.ledger, "validated", full(health=health(rows=rows)))
        self.assertEqual([r["name"] for r in summary.data["tree"]], ["Shell"])
        self.assertIn("| Shell | Part::Cut | 1 | 18432.1 |", summary.markdown)

    def test_a_wrongly_typed_validation_still_raises_before_anything_is_written(self):
        before = sorted(os.listdir(self.proj.root))
        with self.assertRaises(P.ProjectError):
            self.proj.finalize(ledger=self.ledger, outcome="answered", validation="not a dict")
        self.assertEqual(sorted(os.listdir(self.proj.root)), before)


class ProvenanceClaimTest(unittest.TestCase):
    def setUp(self):
        self.proj = S.make_project(create=True)
        self.good = dict(kind="reference", claim="Board outline 85 x 56 mm", source="catalog:rpi4b", status="nominal")

    def test_an_empty_or_blank_or_missing_claim_is_refused_and_records_nothing(self):
        for bad in ("", "   ", "\n\t", None, 5, ["x"], {"a": 1}):
            with self.assertRaises(P.ProjectError, msg=repr(bad)) as caught:
                self.proj.provenance_add(**{**self.good, "claim": bad})
            self.assertIn("'claim' must be a non-empty string", str(caught.exception))
        with self.assertRaises(P.ProjectError) as caught:
            self.proj.provenance_add(**{k: v for k, v in self.good.items() if k != "claim"})
        self.assertIn("'claim'", str(caught.exception))
        self.assertEqual(self.proj.read_provenance()["entries"], [])
        self.assertFalse(os.path.exists(self.proj.path("provenance.json")))

    def test_the_claim_is_stripped_and_bounded(self):
        entry = self.proj.provenance_add(**{**self.good, "claim": "  padded  "})
        self.assertEqual(self.proj.read_provenance()["entries"][0]["claim"], "padded")
        self.assertEqual(entry, "E1")
        self.proj.provenance_add(**{**self.good, "claim": "z" * 5000})
        self.assertEqual(len(self.proj.read_provenance()["entries"][1]["claim"]), 2000)

    def test_a_refused_call_does_not_copy_a_file_into_refs(self):
        source = S.new_temp_file(S.sandbox_dir(), "pi4.step", b"ISO-10303-21;")
        with self.assertRaises(P.ProjectError):
            self.proj.provenance_add(**{**self.good, "claim": "", "local_path": source, "kind": "download",
                                        "status": "obtained"})
        self.assertFalse(os.path.isdir(self.proj.path("refs")) and os.listdir(self.proj.path("refs")))

    def test_the_other_rules_still_apply(self):
        with self.assertRaises(P.ProjectError):
            self.proj.provenance_add(**{**self.good, "source": ""})
        with self.assertRaises(P.ProjectError):
            self.proj.provenance_add(**{**self.good, "status": "fine"})


class StaleExportsTest(unittest.TestCase):
    def setUp(self):
        self.proj = S.make_project(create=True)
        S.record(self.proj)
        self.ledger = {"intent": "enclosure", "request": "case", "requirements": [], "steps": []}

    def items(self):
        return self.proj.build_summary(self.ledger, "answered", None).data["open_items"]

    def put_export(self):
        S.new_temp_file(self.proj.path("exports"), "case-shell.step")

    def test_exports_are_fresh_right_after_they_were_written(self):
        self.put_export()
        self.proj.mark_exported()
        self.assertEqual(self.proj.manifest["exported_after_step"], 1)
        self.assertFalse(any("exports/" in i for i in self.items()))

    def test_exports_older_than_the_model_are_an_open_item_and_a_notes_line(self):
        self.put_export()
        self.proj.mark_exported()
        S.record(self.proj, call_id="c2")
        items = [i for i in self.items() if i.startswith("exports/")]
        self.assertEqual(items, ["exports/ was written after step 1 but the model is at step 2; export again "
                                 "(export_model) so the files match the model"])
        self.proj.write_notes_md()
        notes = read_text(self.proj.path("NOTES.md"))
        self.assertIn("exports/ was written after step 1 but the model is at step 2", notes.split("## 8. Unresolved")[1])
        self.proj.mark_exported()
        self.assertFalse(any(i.startswith("exports/") for i in self.items()))

    def test_no_claim_without_a_mark_or_without_files(self):
        S.record(self.proj, call_id="c2")
        self.assertFalse(any(i.startswith("exports/") for i in self.items()))   # no exports/ at all
        self.put_export()
        self.assertFalse(any(i.startswith("exports/") for i in self.items()))   # files, but never marked: no claim
        self.proj.manifest["exported_after_step"] = "garbage"
        self.assertFalse(any(i.startswith("exports/") for i in self.items()))

    def test_a_rewind_below_the_export_makes_it_stale_too(self):
        S.record(self.proj, call_id="c2")
        self.put_export()
        self.proj.mark_exported()   # after step 2
        with self.proj._lock:
            self.proj._append_raw({"type": "rewind", "ts": NOW, "to_step": 1, "restored_checkpoint": "001"})
            self.proj._jcache = None
        self.assertTrue(any(i.startswith("exports/ was written after step 2 but the model is at step 1") for i in self.items()))

    def test_mark_exported_never_raises(self):
        with mock.patch.object(P.Project, "_save_manifest", side_effect=OSError("nope")):
            self.assertIsNone(self.proj.mark_exported())


class ArtifactListTest(unittest.TestCase):
    """The ``artifacts_changed`` file entries the panel strip needs (``panel_render.artifacts_strip_html``)."""

    def setUp(self):
        self.proj = S.make_project(create=True)
        for rel in ("model.FCStd", "docs/Second.FCStd", "NOTES.md", "rebuild.py", "provenance.json",
                    "exports/case-shell.step", "exports/case-lid.stl", "exports/case-lid.stl.tmp-1-ab",
                    "exports/.hidden", "docs/Second.20261001.FCBak", "model.20261001-1.FCBak"):
            S.new_temp_file(os.path.dirname(self.proj.path(*rel.split("/"))), os.path.basename(rel))
        os.makedirs(self.proj.path("exports", "a-folder"))
        for n in range(1, 9):
            S.new_temp_file(self.proj.path("snapshots"), f"{n:03d}_step.png")

    def test_entries_have_exactly_kind_name_path_and_absolute_existing_paths(self):
        entries = self.proj.artifact_list()
        self.assertTrue(entries)
        for entry in entries:
            self.assertEqual(sorted(entry), ["kind", "name", "path"])
            self.assertTrue(os.path.isabs(entry["path"]))
            self.assertTrue(os.path.isfile(entry["path"]), entry)
            self.assertEqual(entry["path"], self.proj.path(*entry["name"].split("/")))
        self.assertTrue({e["kind"] for e in entries} <= {"fcstd", "notes", "rebuild", "provenance", "export", "snapshot"})

    def test_what_is_listed_and_what_is_not(self):
        names = [e["name"] for e in self.proj.artifact_list()]
        self.assertEqual(names[:5], ["model.FCStd", "docs/Second.FCStd", "NOTES.md", "rebuild.py", "provenance.json"])
        self.assertEqual([n for n in names if n.startswith("exports/")], ["exports/case-lid.stl", "exports/case-shell.step"])
        self.assertEqual([n for n in names if n.startswith("snapshots/")], [f"snapshots/{n:03d}_step.png" for n in range(3, 9)])
        for ignored in (".hidden", ".FCBak", ".tmp-", "a-folder"):
            self.assertFalse(any(ignored in n for n in names), ignored)
        self.assertEqual([n for n in names if n.endswith(".FCStd")], ["model.FCStd", "docs/Second.FCStd"])

    def test_kinds_come_in_the_panel_order(self):
        kinds = []
        for entry in self.proj.artifact_list():
            if not kinds or kinds[-1] != entry["kind"]:
                kinds.append(entry["kind"])
        self.assertEqual(kinds, ["fcstd", "notes", "rebuild", "provenance", "export", "snapshot"])

    def test_a_project_that_is_not_materialized_has_no_artifacts(self):
        self.assertEqual(S.make_project().artifact_list(), [])

    def test_the_panel_strip_renders_every_chip_from_the_list(self):
        from oscar import panel_render

        html = panel_render.artifacts_strip_html({"project_dir": self.proj.root, "files": self.proj.artifact_list()})
        for action in ("open-doc", "open-notes", "open-rebuild", "open-exports", "open-provenance", "open-dir"):
            self.assertIn(f"oscar://artifact/{action}", html, action)
        self.assertIn("docs%2FSecond.FCStd", html, "a second document is a chip of its own")
        self.assertIn("exports (2)", html)
        self.assertNotIn("snapshots/", html, "snapshots are thumbnails, not chips")
        self.assertNotIn("FCBak", html)


class PlanOrientationTest(unittest.TestCase):
    """``project_export.plan_orientation``: the kit spec keys ``print_orientation.shell|lid`` decide, nothing else."""

    SPEC = {"print_orientation": {"shell": "floor_down", "lid": "top_down"}}

    def test_the_two_kit_roles_follow_the_spec(self):
        self.assertEqual(X.plan_orientation(self.SPEC, "shell"), ("floor_down", ""))
        self.assertEqual(X.plan_orientation(self.SPEC, "lid"), ("top_down", ""))

    def test_as_modeled_is_a_value_of_its_own(self):
        spec = {"print_orientation": {"shell": "as_modeled", "lid": "as_modeled"}}
        self.assertEqual(X.plan_orientation(spec, "shell"), ("as_modeled", ""))
        self.assertEqual(X.plan_orientation(spec, "lid"), ("as_modeled", ""))

    def test_other_roles_a_disabled_flag_and_a_missing_spec_mean_as_modelled(self):
        for role in ("", "ref", "hook", "params"):
            self.assertEqual(X.plan_orientation(self.SPEC, role), ("as_modeled", ""))
        self.assertEqual(X.plan_orientation(self.SPEC, "lid", enabled=False), ("as_modeled", ""))
        for spec in (None, {}, {"print_orientation": None}, {"print_orientation": "top_down"}, {"print_orientation": []},
                     {"print_orientation": {"lid": ""}}, {"print_orientation": {"lid": None}}, "not a dict", 7):
            self.assertEqual(X.plan_orientation(spec, "lid"), ("as_modeled", ""), repr(spec))

    def test_a_value_the_kit_does_not_know_is_reported_and_exported_as_modelled(self):
        mode, note = X.plan_orientation({"print_orientation": {"lid": "upside_down"}}, "lid")
        self.assertEqual(mode, "as_modeled")
        self.assertEqual(note, "print_orientation.lid 'upside_down' is not one of top_down, as_modeled; "
                               "the lid was exported as modelled")
        # a value valid for the lid is not valid for the shell (and the other way round)
        mode, note = X.plan_orientation({"print_orientation": {"shell": "top_down"}}, "shell")
        self.assertEqual(mode, "as_modeled")
        self.assertIn("print_orientation.shell 'top_down' is not one of floor_down, as_modeled", note)

    def test_the_accepted_values_are_the_kits(self):
        from oscar.kit import spec as kit_spec

        accepted = X.known_orientations()
        self.assertEqual(accepted["shell"], tuple(kit_spec.SHELL_ORIENTATIONS))
        self.assertEqual(accepted["lid"], tuple(kit_spec.LID_ORIENTATIONS))
        self.assertEqual(kit_spec.default_spec().to_dict()["print_orientation"], self.SPEC["print_orientation"])

    def test_without_the_kit_the_built_in_defaults_apply(self):
        import builtins

        real_import = builtins.__import__

        def refuse(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 1 and name == "kit" and globals and globals.get("__name__") == "oscar.project_export":
                raise ModuleNotFoundError("No module named 'oscar.kit'", name="oscar.kit")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch.object(builtins, "__import__", refuse):
            self.assertEqual(X.known_orientations(), {"shell": ("floor_down", "as_modeled"), "lid": ("top_down", "as_modeled")})
        with mock.patch.object(builtins, "__import__", refuse):
            self.assertEqual(X.plan_orientation(self.SPEC, "lid"), ("top_down", ""))

    def test_a_broken_dependency_inside_the_kit_is_not_swallowed(self):
        import builtins

        real_import = builtins.__import__

        def explode(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 1 and name == "kit" and globals and globals.get("__name__") == "oscar.project_export":
                raise ModuleNotFoundError("No module named 'numpy_nope'", name="numpy_nope")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch.object(builtins, "__import__", explode):
            with self.assertRaises(ModuleNotFoundError):
                X.known_orientations()


class SpyDoc(S.FakeDoc):
    """A FakeDoc that remembers the path the CODE asked ``saveCopy`` for (``FakeDoc.saved_to`` has the extension
    FreeCAD would have appended, which hides a wrong temp name)."""

    def __init__(self, name="Fake", **kw):
        super().__init__(name, **kw)
        self.requested: list[str] = []

    def saveCopy(self, path):  # noqa: N802 - FreeCAD API name
        self.requested.append(path)
        return super().saveCopy(path)


class RevisionTwoRulesTest(unittest.TestCase):
    """Threading rules for EVERY method that waits on the GUI thread: no lock is held across a dispatch (a fake GUI
    thread takes the lock and reads while the worker is blocked inside it) and every ``saveCopy`` temp name ends in
    ``.FCStd`` (FreeCAD appends the extension to any other name and writes there)."""

    def setUp(self):
        self.fake = SpyDoc("Doc")
        self.proj = S.make_project(create=True)

        def gui_side():
            got = self.proj._lock.acquire(timeout=1.0)
            if got:
                self.proj._lock.release()
            started = time.monotonic()
            self.proj.read_steps()
            self.proj.artifact_list()
            self.proj.read_params()
            self.proj.read_validation()
            self.proj.last_geometry_step()
            return got and time.monotonic() - started < 0.5

        self.gui = S.FakeGui(before_task=gui_side).install()
        self.addCleanup(self.gui.uninstall)
        rows = [{"name": "wall_t", "value": 2.0, "unit": "mm", "expr": "", "source": "assumption", "note": "",
                 "cell": "B2", "bound": 1, "needs_rebuild": False}]
        patches = {
            "resolve_doc": lambda name: self.fake if name in ("", "Doc") else None,
            "has_geometry": lambda doc: True,
            "object_names": lambda doc: (["Box"], ["Box"]),
            "param_table": lambda doc: {"sheet": "Params", "rows": rows},
            "model_tree": lambda doc: [{"name": "Box", "type": "Part::Box", "solids": 1, "volume": 1.0,
                                        "bbox": [0, 0, 0, 1, 1, 1], "tip": True}],
            "restore_from_file": lambda doc, path, label: {"ok": True, "error": "", "committed": True,
                                                           "reactivated": True, "names_after": ["Box"], "invalid": []},
        }
        for name, fn in patches.items():
            patcher = mock.patch.object(gui, name, fn)
            patcher.start()
            self.addCleanup(patcher.stop)

    def on_worker(self, fn):
        out = {}

        def run():
            try:
                out["value"] = fn()
            except BaseException as exc:  # noqa: BLE001
                out["error"] = exc

        worker = threading.Thread(target=run)
        worker.start()
        worker.join(60)
        self.assertFalse(worker.is_alive(), "the worker is stuck (a lock held across a dispatch?)")
        if "error" in out:
            raise out["error"]
        return out.get("value")

    def test_no_method_holds_the_lock_while_it_waits_for_the_gui_thread(self):
        S.record(self.proj, doc="Doc")           # a recorded geometry step with a checkpoint 001
        S.record(self.proj, doc="Doc", call_id="c2")
        self.proj.begin_turn(S.TURN, "work", work=True)
        seen_before = len(self.gui.observations)
        calls = {
            "hook": lambda: S.record(self.proj, doc="Doc", call_id="c3"),
            "turn_checkpoint": lambda: self.proj.ensure_turn_checkpoint("Doc"),
            "autosave": lambda: self.proj.autosave("Doc"),
            "checkpoint": lambda: self.proj.checkpoint("Doc", "manual"),
            "snapshot": lambda: self.proj.snapshot("Doc", "pic", 9),
            "sync_params": lambda: self.proj.sync_params("Doc"),
            "restore": lambda: self.proj.restore("Doc", "001"),
            "rewind_to": lambda: self.proj.rewind_to(1),
            "verify_artifacts": lambda: self.proj.verify_artifacts(),
            "build_summary": lambda: self.proj.build_summary(None, "answered", None),
            "finalize": lambda: self.proj.finalize(ledger={"intent": "other", "steps": []}, outcome="answered",
                                                   validation=full(after_step=1), export=False),
        }
        for name, fn in calls.items():
            before = len(self.gui.observations)
            self.on_worker(fn)
            ran = self.gui.observations[before:]
            self.assertTrue(ran, f"{name}: no GUI task ran, the test proves nothing")
            self.assertTrue(all(ran), f"{name}: the Project lock was held while the worker waited on the GUI thread")
        self.assertGreater(len(self.gui.observations), seen_before + len(calls))

    def test_the_harness_detects_a_lock_held_across_a_dispatch(self):
        def bad():
            with self.proj._lock:
                return gui.gui_call(lambda: 1, 5.0, "bad")

        self.on_worker(bad)
        self.assertEqual(self.gui.observations, [False])

    def test_every_save_path_uses_an_fcstd_temp_name(self):
        S.record(self.proj, doc="Doc")
        self.proj.begin_turn(S.TURN, "work", work=True)
        self.proj.ensure_turn_checkpoint("Doc")
        self.proj.checkpoint("Doc", "manual")
        self.proj.autosave("Doc")
        self.proj.restore("Doc", "001")            # writes the rescue copy first
        self.proj.save_now("Doc")
        self.assertGreaterEqual(len(self.fake.requested), 6)
        self.assertEqual(self.fake.requested, self.fake.saved_to, "FreeCAD would have had to append .FCStd to a name")
        for path in self.fake.requested:
            self.assertTrue(path.endswith(".FCStd"), path)
            self.assertIn(".tmp-", os.path.basename(path))
            self.assertFalse(path.endswith(".FCStd.FCStd"), path)
        leftovers = []
        for folder, _, files in os.walk(self.proj.root):
            leftovers += [os.path.join(folder, f) for f in files if ".tmp-" in f or f.endswith(".FCBak")]
        self.assertEqual(leftovers, [])
        self.assertTrue(os.path.isfile(self.proj.path("checkpoints", "_rescue.FCStd")))
        names = sorted(os.listdir(self.proj.path("checkpoints")))
        self.assertIn("001.FCStd", names)
        self.assertIn("manual.FCStd", names)
        self.assertEqual([n for n in names if n.endswith(".tmp")], [])


if __name__ == "__main__":
    unittest.main()
