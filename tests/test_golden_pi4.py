"""The golden Pi 4 scenario: a scripted model drives the REAL tools in freecadcmd through the
"Build a housing for a Raspberry Pi 4 ..." request, end to end.

Only the model (``ScriptedProvider``), the 3D-view snapshotter (a 1x1 PNG) and the clock are fake. Everything else is the
product: ``AgentWorker``, the finish gate and ledger, ``tools.run_tool`` with the Project / health hooks, the reference
catalog and the offline STEP mirror (``tests/fixtures/refparts``), the enclosure kit, ``validate_model(deep)`` /
``check_fit``, the exports and the headless rebuild check. ``GoldenPi4Test`` plays the run ONCE (``setUpClass``) and
asserts the program acceptance criteria that need no live LLM in separate ``test_*`` methods; the mutating checks (rewind,
the lid-lift variant) are named ``test_9x`` so they run after every read-only one. ``GoldenPi4NominalTest`` is the same
prompt without the STEP file (``catalog-nominal``): the turn is still validated, and the summary says what it could not
check.
"""

import copy
import dataclasses
import glob
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402

from oscar import chats, project_gui  # noqa: E402
from oscar.cad.serialize import global_shape, is_geometry  # noqa: E402
from oscar.kit import geometry as kit_geometry  # noqa: E402
from oscar.project import REFERENCE_STATUSES, replay_steps, replay_summary  # noqa: E402
from oscar.project_replay import compare_rebuild  # noqa: E402
from oscar.providers import Message  # noqa: E402
from support.golden import DOC, GoldenRun  # noqa: E402
from support.realflow import SUMMARY_HEADINGS, RealFlowCase  # noqa: E402

PORTS = ("usb_c_power", "hdmi0", "hdmi1", "audio", "ethernet", "usb3_pair", "usb2_pair", "microsd")
WALL_BUDGET_S = 120.0       # contract: under 90 s on a developer machine; the margin is for a loaded CI runner
BBOX_BEFORE = (94.1, 61.94, 24.0)
BBOX_AFTER = (94.9, 62.79, 24.0)

# which line of the Build Summary's Validated list each recorded check family produces
VALIDATED_LINES = {"health": "Model health", "fit.interference": "Part-to-part interference", "fit.clearance": "Clearances",
                   "fit.ports": "Port openings contain every connector", "fit.retention": "Snap-fit retention",
                   "dfm": "Printability rules", "editability": "Parameter editability"}


def bbox(shape) -> tuple[float, float, float]:
    b = shape.BoundBox
    return (b.XLength, b.YLength, b.ZLength)


def section(text: str, heading: str) -> str:
    return RealFlowCase.section(text, heading)


def bullet_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("- ")]


class GoldenBase(unittest.TestCase):
    """Shared helpers; subclasses set ``RUN`` in ``setUpClass``."""

    RUN: GoldenRun = None

    @classmethod
    def tearDownClass(cls):
        if cls.RUN is not None:
            cls.RUN.stop()

    # -- accessors ---------------------------------------------------------------------------------------------

    @property
    def project(self):
        return self.RUN.project

    @property
    def result(self):
        return self.RUN.result

    @property
    def doc(self):
        return App.getDocument(DOC)

    def steps(self):
        return self.project.read_steps()

    def tool_texts(self, tool: str) -> list[str]:
        """The texts the model was shown for each call of ``tool`` (from the recorded history)."""
        return [r.content for m in self.RUN.case.history for r in m.tool_results if r.name == tool]

    def validation(self) -> dict:
        return self.project.read_validation() or {}

    def fit_rows(self) -> dict:
        return {r["id"]: r for r in self.validation().get("fit_rows", [])}

    def dfm_rows(self) -> dict:
        return {r["id"]: r for r in self.validation().get("dfm", [])}

    def tip(self, role: str, doc=None):
        return kit_geometry.tip(self.doc if doc is None else doc, role)

    def geometry_steps(self):
        return [s for s in self.steps() if s.geometry and s.ok]


class CommonChecks:
    """Assertions both variants (with and without the STEP file) must satisfy."""

    FIRST_GEOMETRY_WITHIN_ROUNDS = 6      # the scripted persona's first geometry comes in its first N rounds

    def test_first_geometry_comes_within_the_first_rounds(self):
        first = next(s for s in self.steps() if s.geometry and s.ok)
        self.assertLessEqual(first.round, self.FIRST_GEOMETRY_WITHIN_ROUNDS, first.title)

    def test_no_synthetic_note_other_than_the_expected_ones(self):
        history = self.RUN.case.history
        self.assertEqual([m.tag for m in history if getattr(m, "synthetic", False)], [])   # a cooperative model is never nudged
        self.assertEqual(self.RUN.worker._watchdog.trips, 0)


# ======================================================================================================================
# the run with the STEP file (status step-measured)

class GoldenPi4Test(CommonChecks, GoldenBase):
    @classmethod
    def setUpClass(cls):
        cls.RUN = GoldenRun(persist=True, chat_id="20261002-101010-90d0a1").start()

    # -- 1. outcome and Build Summary ----------------------------------------------------------------------------

    def test_01_the_turn_is_validated_with_the_full_build_summary(self):
        text = self.result.text
        self.assertEqual((self.result.outcome, self.result.ok), ("validated", True), text[-1500:])
        self.assertTrue(self.result.text.startswith("Built the Pi 4 housing"))   # the model's commentary first
        self.assertIn("## Build Summary", text)
        for heading in SUMMARY_HEADINGS:
            self.assertIn(heading, text)
        self.assertIn("reference status: step-measured", text)
        self.assertIn("STEP-based collision check", section(text, "Validated"))
        self.assertNotIn("STEP-based collision check", section(text, "NOT validated"))
        self.assertEqual(self.RUN.worker.ledger.pending_required(), [])

    def test_02_validated_and_not_validated_match_the_recorded_checks(self):
        text = self.result.text
        validated, not_validated = section(text, "Validated"), section(text, "NOT validated")
        recorded = self.validation()
        self.assertTrue(recorded.get("ok"), recorded.get("fail_rows"))
        for key in recorded["ran"]:
            self.assertIn(VALIDATED_LINES[key], validated, key)
        # rebuild.py: listed only because the check really ran and passed after the last geometry step
        check = self.project.manifest["rebuild_check"]
        self.assertTrue(check["ok"])
        self.assertGreaterEqual(check["after_step"], self.project.last_geometry_step())
        self.assertIn("rebuild.py reproduces the model headless", validated)
        # and nothing is claimed beyond what ran: every line of Validated maps to a recorded family
        known = list(VALIDATED_LINES.values()) + ["rebuild.py reproduces", "STEP-based collision check"]
        for line in bullet_lines(validated):
            self.assertTrue(any(k in line for k in known), line)
        self.assertIn("Physical fit, printing", not_validated)          # the app cannot check a real print
        self.assertEqual(len(bullet_lines(not_validated)), 1, not_validated)

    def test_03_the_model_text_cannot_widen_the_validated_list(self):
        # the model's closing text names STEP and STL exports only; the summary is generated from records
        self.assertNotIn("Checks passed", section(self.result.text, "Validated"))
        self.assertEqual(self.tool_texts("create_document")[0].split("'")[1], DOC)   # PiCase, not PiCase1

    # -- 2. the project folder -----------------------------------------------------------------------------------

    def test_10_the_project_folder_is_complete(self):
        root = self.project.root
        for name in ("project.json", "ledger.json", "steps.jsonl", "params.json", "provenance.json", "NOTES.md", "rebuild.py",
                     "model.FCStd", "spec.json", "validation.json"):
            self.assertTrue(os.path.isfile(os.path.join(root, name)), name)
        exports = sorted(os.listdir(os.path.join(root, "exports")))
        self.assertEqual(exports, ["picase-lid.step", "picase-lid.stl", "picase-shell.step", "picase-shell.stl"])
        for name in exports:
            self.assertGreater(os.path.getsize(os.path.join(root, "exports", name)), 1000, name)
        self.assertEqual(len(glob.glob(os.path.join(root, "refs", "*rpi4b.step"))), 1)
        self.assertTrue(os.path.isdir(os.path.join(root, "checkpoints")))
        self.assertTrue(os.path.isdir(os.path.join(root, "snapshots")))
        with open(os.path.join(root, "rebuild.py"), encoding="utf-8") as f:
            self.assertIn("REBUILD_OK", f.read())
        with open(os.path.join(root, "NOTES.md"), encoding="utf-8") as f:
            notes = f.read()
        for heading in ("## 1. Request", "## 2. Requirements and status", "## 4. Parameters", "## 5. Provenance",
                        "## 9. Steps log", "## 10. Artifacts"):
            self.assertIn(heading, notes)
        self.assertIn("Raspberry Pi 4", notes)

    def test_11_one_step_record_per_mutating_call(self):
        steps = self.steps()
        self.assertEqual(len(steps), 19)
        self.assertEqual([s.index for s in steps], list(range(1, 20)))
        self.assertEqual(steps[0].tool, "source_reference_part")
        self.assertEqual(steps[-1].tool, "export_rebuild_script")
        self.assertEqual([s.tool for s in steps[5:14]], ["build_enclosure", "build_enclosure", "execute_code"] + ["build_enclosure"] * 6)
        read_only = {"find_reference_part", "get_view", "get_objects", "validate_model", "check_fit"}
        self.assertFalse(read_only & {s.tool for s in steps})

    def test_12_every_successful_geometry_step_has_checkpoint_health_and_snapshot(self):
        geometry = self.geometry_steps()
        self.assertEqual(len(geometry), 12)   # the import, the eight stages, add_cutout and the two set_parameter calls
        for s in geometry:
            self.assertTrue(s.checkpoint, s.title)
            self.assertTrue(os.path.isfile(os.path.join(self.project.root, "checkpoints", s.checkpoint + ".FCStd")), s.title)
            self.assertTrue(s.health and s.health["line"].startswith("[health]"), s.title)
            self.assertTrue(s.delta, s.title)
            if s.snapshot:
                self.assertGreater(os.path.getsize(os.path.join(self.project.root, s.snapshot)), 0)
            else:   # only a document with no geometry yet may go without one
                self.assertEqual(s.snapshot_note, "empty document", s.title)
        # exactly one app snapshot per BUILD step: the PNG name carries the step index
        pngs = sorted(os.listdir(os.path.join(self.project.root, "snapshots")))
        self.assertEqual(len(pngs), len([s for s in geometry if s.snapshot]))
        self.assertEqual(len(set(pngs)), len(pngs))

    def test_13_the_injected_failure_is_recorded_without_snapshot_and_is_followed_by_recovery(self):
        steps = self.steps()
        (failed,) = [s for s in steps if not s.ok]
        self.assertEqual(failed.tool, "execute_code")
        self.assertEqual((failed.snapshot, failed.checkpoint, failed.script), ("", "", ""))
        self.assertTrue(failed.snapshot_note)
        self.assertTrue(any(s.ok and s.index > failed.index and s.geometry for s in steps))
        text = self.tool_texts("execute_code")[0]
        self.assertIn("NameError", text)
        self.assertIn("ROLLED BACK", text)
        # the failed script left no saved script file behind
        scripts = os.path.join(self.project.root, "scripts")
        self.assertEqual(os.listdir(scripts) if os.path.isdir(scripts) else [], [])

    def test_14_the_ledger_is_terminal_and_the_app_confirmed_validation_and_delivery(self):
        ledger = self.project.read_ledger()
        statuses = {s["id"]: s["status"] for s in ledger["steps"]}
        self.assertEqual(len(statuses), 12)
        self.assertTrue(all(v in ("done", "skipped", "blocked") for v in statuses.values()), statuses)
        self.assertEqual((statuses["S11"], statuses["S12"]), ("done", "done"))
        by_id = {s["id"]: s for s in ledger["steps"]}
        for step_id in ("S11", "S12"):
            self.assertEqual({e["kind"] for e in by_id[step_id]["evidence"]}, {"auto"}, step_id)   # the app, not the model
        self.assertEqual(ledger["reference_status"], "step-measured")
        self.assertGreaterEqual(self.RUN.rec.count("plan_changed"), 12)

    # -- 5. the two parts ----------------------------------------------------------------------------------------

    def test_20_shell_and_lid_are_separate_valid_single_solids(self):
        shell, lid = self.tip("shell"), self.tip("lid")
        self.assertIsNotNone(shell)
        self.assertIsNotNone(lid)
        self.assertNotEqual(shell.Name, lid.Name)
        for part in (shell, lid):
            shape = part.Shape
            self.assertEqual(len(shape.Solids), 1, part.Name)
            self.assertTrue(shape.isValid(), part.Name)
            self.assertTrue(shape.isClosed(), part.Name)
        self.assertAlmostEqual(shell.Shape.common(lid.Shape).Volume, 0.0, delta=0.05)   # never fused: two parts that only touch
        for name in ("Part_Shell", "Part_Lid", "Ref_RPI4B"):
            self.assertIsNotNone(self.doc.getObject(name), name)
        self.assertFalse(self.doc.getObject("Ref_RPI4B").Visibility)
        self.assertIsNotNone(self.doc.getObject("Assembly"))

    def test_21_every_connector_has_an_opening(self):
        for port in PORTS:
            self.assertIsNotNone(self.doc.getObject(f"Shell_Port_{port}"), port)

    # -- 6. dimensions ---------------------------------------------------------------------------------------------

    def params(self) -> dict:
        sheet = self.doc.getObject("Params")
        out = {}
        for p in self.project.read_params()["params"]:
            value = sheet.get(p["name"])
            out[p["name"]] = float(getattr(value, "Value", value))
        return out

    def test_30_the_edits_rebuilt_the_shell_to_the_new_dimensions(self):
        after = bbox(global_shape(self.tip("shell")))
        for got, want in zip(after, BBOX_AFTER):
            self.assertAlmostEqual(got, want, delta=0.05)
        params = self.params()
        self.assertAlmostEqual(params["wall_t"], 2.4, delta=1e-6)
        self.assertAlmostEqual(params["clearance"], 0.35, delta=1e-6)
        # the wall read from the geometry: outer minus cavity, per side
        outer, cavity = self.doc.getObject("Shell_Outer").Shape.BoundBox, self.doc.getObject("Shell_Cavity").Shape.BoundBox
        self.assertAlmostEqual((outer.XLength - cavity.XLength) / 2, 2.4, delta=0.05)
        self.assertAlmostEqual((outer.YLength - cavity.YLength) / 2, 2.4, delta=0.05)

    def test_31_before_the_edits_the_shell_was_smaller(self):
        assembly = next(s for s in self.steps() if s.tool == "build_enclosure" and s.title.endswith("assembly"))
        path = os.path.join(self.project.root, "checkpoints", assembly.checkpoint + ".FCStd")
        before = App.openDocument(path)
        try:
            dims = bbox(global_shape(self.tip("shell", before)))
            info = f"checkpoint {assembly.checkpoint} of {self.project.root}"
        finally:
            App.closeDocument(before.Name)
        for got, want in zip(dims, BBOX_BEFORE):
            self.assertAlmostEqual(got, want, delta=0.05, msg=info)

    def test_32_the_reference_and_the_parameters_keep_their_provenance(self):
        sheet = self.doc.getObject("Params")
        self.assertEqual(sheet.Name, "Params")
        self.assertIsNone(self.doc.getObject("Params001"))
        rows = {p["name"]: p for p in self.project.read_params()["params"]}
        self.assertEqual(rows["board_len"]["source"], "step_measurement")
        self.assertEqual(rows["board_w"]["source"], "step_measurement")
        self.assertEqual((rows["pcb_t"]["source"], rows["tallest_z"]["source"]), ("step_measurement",) * 2)
        self.assertEqual(rows["wall_t"]["source"], "user_requirement")

    # -- 7. fit, DFM, editability ----------------------------------------------------------------------------------

    def test_40_check_fit_has_no_failures_and_every_named_row_passes(self):
        rows = self.fit_rows()
        self.assertEqual([r["id"] for r in rows.values() if r["status"] == "FAIL"], [])
        for rid in ("interference.shell_lid", "interference.reference_shell", "interference.reference_lid", "retention.lid_lift",
                    "clearance.pcb_to_wall_min", "hooks.clear_of_cutouts", "hooks.engagement", "containment.ports",
                    "alignment.standoffs", "reference.status"):
            self.assertEqual(rows[rid]["status"], "PASS", rows[rid])
        self.assertIn("8 connector windows", rows["containment.ports"]["detail"])
        self.assertGreater(rows["retention.lid_lift"]["value"], 0)       # a lift of 0.5 mm still interferes
        self.assertIn("step-measured", rows["reference.status"]["detail"])
        text = self.tool_texts("check_fit")[0]
        self.assertIn("0 FAIL", text)
        self.assertIn("lift_mm=0.5", text)

    def test_41_dfm_rows(self):
        rows = self.dfm_rows()
        self.assertEqual(rows["dfm.hook_strain"]["status"], "PASS")
        self.assertIn("0.0069", rows["dfm.hook_strain"]["detail"])
        self.assertEqual(rows["dfm.rim_web"]["status"], "WARN")
        self.assertIn("usb3_pair", rows["dfm.rim_web"]["detail"])
        self.assertEqual([r["id"] for r in rows.values() if r["status"] == "FAIL"], [])

    def test_42_deep_validation_rebuilds_the_driver_parameters_at_plus_minus_ten_percent(self):
        (text,) = self.tool_texts("validate_model")
        self.assertIn("EDITABILITY", text)
        edit = text.split("EDITABILITY", 1)[1].splitlines()[1]
        for name in ("wall_t", "lid_fit", "hook_len", "hole_off", "floor_t"):
            self.assertIn(f"{name} +10% ok, -10% ok", edit)
        self.assertNotRegex(edit, r"(\+|-)10% (fail|invalid|error)")
        self.assertIn("RESULT: VALID", text)
        self.assertEqual(self.validation()["health"]["valid"], True)

    # -- 10. provenance -------------------------------------------------------------------------------------------

    def test_50_provenance_records_the_reference_with_hash_and_local_copy(self):
        prov = self.project.read_provenance()
        ref = prov["reference"]
        self.assertEqual(ref["status"], "step-measured")
        self.assertIn(ref["status"], REFERENCE_STATUSES)
        entries = {e["id"]: e for e in prov["entries"]}
        self.assertEqual([e["kind"] for e in prov["entries"]][:3], ["download", "reference", "measurement"])
        for e in prov["entries"]:
            if e["kind"] in ("download", "reference"):
                self.assertRegex(e["sha256"], r"^[0-9a-f]{64}$")
        download = entries["E1"]
        self.assertEqual(download["status"], "obtained")
        self.assertTrue(os.path.isfile(self.project.path(download["local_path"])))
        self.assertTrue(entries[ref["entry"]]["source"])

    # -- 11. signals -----------------------------------------------------------------------------------------------

    def test_60_signals(self):
        rec = self.RUN.rec
        finished = {p["step_index"]: p for p in rec.json_of("step_finished") if p.get("step_index", -1) >= 0}
        for s in self.geometry_steps():
            payload = finished[s.index]
            if s.snapshot:
                self.assertTrue(payload["snapshot_path"], s.title)
            else:
                self.assertEqual(s.snapshot_note, "empty document")
        self.assertGreaterEqual(rec.count("progress"), len(self.RUN.script))   # at least one per round
        rec.assert_ordered("turn_outcome", "turn_finished")
        self.assertEqual(rec.of("turn_outcome")[0][0], "validated")
        self.assertEqual(rec.count("turn_finished"), 1)

    # -- 12. persistence ---------------------------------------------------------------------------------------------

    def test_61_the_chat_file_holds_the_project_the_ledger_and_the_whole_tool_loop(self):
        session = self.RUN.session
        session["project_dir"] = self.project.root
        session["ledger"] = self.project.read_ledger()
        session["history"] = list(self.RUN.case.history) + [Message(role="user", content="note", synthetic=True, tag="gate")]
        chats.save(session)
        loaded = chats.load(session["id"])
        self.assertEqual(loaded["project_dir"], self.project.root)
        self.assertEqual(len(loaded["ledger"]["steps"]), 12)
        history = loaded["history"] if "history" in loaded else loaded["messages"]
        calls = [c for m in history for c in m.tool_calls]
        self.assertEqual(len(calls), len(self.RUN.script) - 1)             # every call of the tool loop, in order
        self.assertEqual([c.name for c in calls][:3], ["find_reference_part", "source_reference_part", "create_document"])
        self.assertEqual(sum(len(m.tool_results) for m in history), len(calls))
        self.assertEqual((history[-1].synthetic, history[-1].tag), (True, "gate"))   # the new fields round-trip

    # -- 13. speed and watchdog --------------------------------------------------------------------------------------

    def test_62_the_scripted_run_is_fast_and_never_silent(self):
        self.assertLess(self.RUN.wall_s, WALL_BUDGET_S)
        self.assertEqual(self.RUN.worker._watchdog.trips, 0)
        self.assertEqual(self.RUN.case.notes("watchdog"), [])
        # no round went by without a signal, and the injected clock never ran past a watchdog cap
        self.assertLess(self.RUN.case.clock() - 1000.0, 90.0)

    # -- 9. replay equality ------------------------------------------------------------------------------------------

    def replay(self, steps):
        result = replay_steps(steps, project_root=self.project.root)
        self.addCleanup(lambda: [App.closeDocument(n) for n in set(result.name_map.values()) if n in App.listDocuments()])
        return result

    def test_70_replaying_the_recorded_steps_into_a_fresh_document_reproduces_the_model(self):
        live = project_gui.live_volumes(self.doc)
        result = self.replay(self.steps())
        self.assertTrue(result.ok, [r for r in result.results if not r["ok"]])
        self.assertEqual(result.name_map, {DOC: DOC + "_rebuild"})
        rebuilt = replay_summary(result)["docs"][DOC + "_rebuild"]
        self.assertEqual(compare_rebuild({DOC: live}, {DOC: rebuilt}), [])
        solids = {n for n, v in live.items() if v is not None}
        self.assertGreater(len(solids), 60)
        self.assertEqual(solids, {n for n, v in rebuilt.items() if v is not None})
        for name in solids:       # is_geometry objects with solids only: Origin features have garbage volumes
            self.assertLessEqual(abs(live[name] - rebuilt[name]), 1e-6 * max(live[name], 1e-9), name)
        for name in live:
            self.assertTrue(is_geometry(self.doc.getObject(name)), name)

    def test_71_a_corrupted_step_makes_the_replay_fail_and_the_comparison_report_it(self):
        steps = [copy.deepcopy(s) for s in self.steps()]
        victim = next(s for s in steps if s.tool == "build_enclosure" and s.title.endswith("shell"))
        victim.replay = dict(victim.replay, args=dict(victim.replay["args"], stage="no_such_stage"))
        result = self.replay(steps)
        self.assertFalse(result.ok)
        self.assertEqual([r["index"] for r in result.results if not r["ok"]], [victim.index])
        live = project_gui.live_volumes(self.doc)
        broken = dict(live)
        broken[next(iter(broken))] = (broken[next(iter(broken))] or 1.0) * 1.01
        problems = compare_rebuild({DOC: live}, {DOC: broken})
        self.assertEqual(len(problems), 1)
        self.assertTrue(compare_rebuild({DOC: live}, {DOC: {}}))

    def test_72_the_headless_rebuild_check_of_the_tool_passed(self):
        (text,) = self.tool_texts("export_rebuild_script")
        self.assertRegex(text, r"Rebuild check: PASS \(\d+ s\): REBUILD_OK, \d+ object\(s\) in 1 document\(s\) match")
        self.assertIn("1e-06", text)

    # -- 9b. rewind (mutating: runs after every read-only test) ------------------------------------------------------

    def test_90_rewind_to_the_shell_stage_and_replay_back(self):
        steps = self.steps()
        shell = next(s for s in steps if s.tool == "build_enclosure" and s.title.endswith("shell"))
        final_volumes = project_gui.live_volumes(self.doc)
        final_names = set(final_volumes)
        before_params = self.params()
        result = self.project.rewind_to(shell.index, DOC)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.doc, DOC)
        self.assertTrue(result.reactivated)
        self.assertEqual(App.ActiveDocument.Name, DOC)
        doc = self.doc
        self.assertLess(len(doc.Objects), len(final_names))
        self.assertIsNone(doc.getObject("Part_Lid"))                               # the later stages are gone
        self.assertIsNotNone(doc.getObject("Shell_Body"))
        sheet = doc.getObject("Params")
        cell = sheet.getCellFromAlias("wall_t")
        self.assertEqual(self.params()["wall_t"], 2.0, f"{sheet.getContents(cell)} after restoring checkpoint {result.checkpoint}")      # the default, not the edit
        self.assertNotEqual(self.params()["wall_t"], before_params["wall_t"])
        # replay the later steps back into the restored document
        later = [s for s in steps if s.index > shell.index]
        replayed = replay_steps(later, project_root=self.project.root)
        self.assertTrue(replayed.ok, [r for r in replayed.results if not r["ok"]])
        back = project_gui.live_volumes(self.doc)
        self.assertEqual(compare_rebuild({DOC: final_volumes}, {DOC: back}), [])
        self.assertEqual(set(back), final_names)
        self.assertAlmostEqual(self.params()["wall_t"], 2.4, delta=1e-6)

    # -- 7b. the lid-lift variant ------------------------------------------------------------------------------------

    def test_95_a_lid_lift_of_15_mm_keeps_the_assembled_checks_valid(self):
        from oscar import tools

        text, _img, is_error = tools.run_tool("set_parameter", {"doc_name": DOC, "name": "lid_lift", "value": 15})
        self.assertFalse(is_error, text)
        text, _img, is_error = tools.run_tool("check_fit", {"doc_name": DOC})
        self.assertFalse(is_error, text)
        self.assertIn("PASS interference.shell_lid", text)                # the assembled pose is evaluated, not the lifted one
        self.assertIn("0 FAIL", text)
        self.assertRegex(text, r"retention\.lid_lift")


# ======================================================================================================================
# the same prompt without the STEP file

class GoldenPi4NominalTest(CommonChecks, GoldenBase):
    @classmethod
    def setUpClass(cls):
        cls.RUN = GoldenRun(nominal=True, chat_id="20261002-101010-90d0a2").start()

    def test_the_turn_is_still_validated_and_says_what_it_could_not_check(self):
        text = self.result.text
        self.assertEqual((self.result.outcome, self.result.ok), ("validated", True), text[-1500:])
        self.assertIn("reference status: catalog-nominal", text)
        self.assertIn("STEP-based collision check", section(text, "NOT validated"))
        self.assertNotIn("STEP-based collision check", section(text, "Validated"))
        for heading in SUMMARY_HEADINGS:
            self.assertIn(heading, text)

    def test_source_reference_part_falls_back_to_the_catalog_nominal_dimensions(self):
        (text,) = self.tool_texts("source_reference_part")
        self.assertIn("status=nominal", text)
        ledger = self.project.read_ledger()
        statuses = {s["id"]: s["status"] for s in ledger["steps"]}
        self.assertEqual(statuses["S3"], "skipped")           # nothing to import: skipped by the app
        self.assertEqual(ledger["reference_status"], "catalog-nominal")
        prov = self.project.read_provenance()
        self.assertEqual(prov["reference"]["status"], "catalog-nominal")
        entry = next(e for e in prov["entries"] if e["id"] == prov["reference"]["entry"])
        self.assertEqual(entry["status"], "nominal")
        self.assertEqual(entry["sha256"], "")                 # no file, so no hash is claimed

    def test_check_fit_warns_that_the_reference_is_nominal(self):
        rows = self.fit_rows()
        self.assertEqual(rows["reference.status"]["status"], "WARN")
        self.assertIn("nominal board data", rows["reference.status"]["detail"])
        self.assertIn("no STEP-based collision check", rows["reference.status"]["detail"])
        self.assertEqual([r["id"] for r in rows.values() if r["status"] == "FAIL"], [])

    def test_the_model_is_built_from_the_nominal_board_data(self):
        shell, lid = self.tip("shell"), self.tip("lid")
        for part in (shell, lid):
            self.assertEqual(len(part.Shape.Solids), 1)
            self.assertTrue(part.Shape.isValid())
        for got, want in zip(bbox(global_shape(shell)), BBOX_AFTER):
            self.assertAlmostEqual(got, want, delta=0.05)
        rows = {p["name"]: p for p in self.project.read_params()["params"]}
        self.assertNotEqual(rows["board_len"]["source"], "step_measurement")   # never claimed as measured
        self.assertEqual(self.RUN.worker.ledger.pending_required(), [])

    def test_the_step_records_are_the_same_minus_the_import_and_the_measurement(self):
        tools = [s.tool for s in self.steps()]
        self.assertNotIn("import_reference_part", tools)
        self.assertNotIn("measure_reference_part", tools)
        self.assertEqual(len(tools), 18)                      # 19 minus those two, plus the nominal "reference" stage
        self.assertEqual(tools.count("build_enclosure"), 9)
        self.assertEqual(len(self.geometry_steps()), 12)      # the nine stages (the "reference" one builds the board), the cutout, two edits
        self.assertEqual(len([s for s in self.steps() if not s.ok]), 1)
        # with no Ref_ object the first (spreadsheet-only) params stage has nothing to photograph
        params_stage = next(s for s in self.steps() if s.tool == "build_enclosure" and s.title.endswith("params"))
        self.assertEqual(params_stage.snapshot, "")
        self.assertEqual(params_stage.snapshot_note, "empty document")


if __name__ == "__main__":
    unittest.main()
