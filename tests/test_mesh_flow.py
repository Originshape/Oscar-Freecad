"""W-1: the mesh intent router wired into the workflow — ``mesh_operation`` consulted at
``_begin_turn`` time (via ``Ledger.from_request(mesh_intent=...)``), the M0–M14 mesh Ledger
template, evidence-driven step completion (a tool call alone never ticks a step), the
ambiguous-vs-unsupported distinction, controlled retargeting, serialization of
``mesh_intent``, playbook activation and the reconstruction finish gate on stored intent.

Pure python (flow + flow_intent only); the FreeCAD-side mesh behaviour these steps record
is covered by the recon suites (test_recon_*), which must stay green alongside this file.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

from oscar.agent_ctx import playbooks_for  # noqa: E402
from oscar.flow import (Ledger, finish_gate, mesh_ambiguous,  # noqa: E402
                        mesh_operation, seed_template)
from oscar.flow_intent import classify_intent  # noqa: E402


def ledger_for(text, mesh_intent=None):
    if mesh_intent is None:
        mesh_intent = "ambiguous" if mesh_ambiguous(text) else mesh_operation(text)
    return Ledger.from_request(text, mesh_intent=mesh_intent)


class _Outcome:
    """The minimal tool-outcome shape ``Ledger.on_tool_result`` reads."""

    def __init__(self, mesh_flow, name="reconstruct_mesh", geometry=True, step_index=2):
        self.name = name
        self.category = "build"
        self.geometry = geometry
        self.handler_ok = True
        self.step_index = step_index
        self.args = {"mode": (mesh_flow or {}).get("mode", "parametric")}
        self.extra = {"mesh_flow": mesh_flow} if mesh_flow is not None else {}


class RouterTest(unittest.TestCase):
    def test_explicit_intents(self):
        cases = {
            "analyse this mesh for defects": "analyse",
            "please repair the mesh": "repair",
            "convert this mesh to a solid": "solid",
            "make a brep from this stl": "brep",
            "reconstruct this scan as editable cad": "parametric",
            "make the hole in this mesh bigger": "modify",
        }
        for text, want in cases.items():
            self.assertEqual(mesh_operation(text), want, text)

    def test_explicit_instruction_overrides_inference(self):
        # both word-classes present: the router's explicit priority decides (parametric wins)
        self.assertEqual(mesh_operation("convert this mesh to a solid, parametric please"),
                         "parametric")

    def test_mesh_import_without_operation_defaults_to_parametric(self):
        self.assertEqual(mesh_operation("import base.stl"), "parametric")

    def test_non_mesh_requests_are_unaffected(self):
        self.assertIsNone(mesh_operation("build me a bracket with 4 holes"))
        led = ledger_for("build me a bracket with 4 holes")
        self.assertIsNone(led.mesh_intent)
        self.assertNotIn("M1", [s.id for s in led.steps],
                         "non-mesh requests keep their normal template")

    def test_mesh_phrasing_without_reconstruction_trigger_words(self):
        led = ledger_for("turn this stl into native freecad")
        self.assertEqual(led.mesh_intent, "parametric")
        self.assertIn("M11", [s.id for s in led.steps])

    def test_ambiguity(self):
        self.assertTrue(mesh_ambiguous("convert this mesh to a solid or parametric model?"))
        self.assertFalse(mesh_ambiguous("reconstruct this mesh"))       # one operation only
        self.assertFalse(mesh_ambiguous("build me a bracket"))          # not mesh at all
        led = ledger_for("convert this mesh to a solid or parametric model?")
        self.assertEqual(led.mesh_intent, "ambiguous")
        self.assertEqual([s.id for s in led.steps], ["M0"],
                         "no executable build workflow before the user chooses")


class TemplateTest(unittest.TestCase):
    def test_step_ids_are_stable_per_intent(self):
        self.assertEqual([s.id for s in seed_template("generic_part", [], False,
                                                      mesh_intent="analyse")],
                         ["M1", "M2", "M3", "M4"])
        self.assertEqual([s.id for s in seed_template("generic_part", [], False,
                                                      mesh_intent="repair")],
                         ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M14"])
        self.assertEqual([s.id for s in seed_template("generic_part", [], False,
                                                      mesh_intent="solid")],
                         ["M1", "M2", "M3", "M4", "M8", "M12", "M14"])
        self.assertEqual([s.id for s in seed_template("generic_part", [], False,
                                                      mesh_intent="parametric")],
                         ["M1", "M2", "M3", "M4", "M5", "M6", "M7",
                          "M9", "M10", "M11", "M12", "M13", "M14"])

    def test_every_step_declares_phase_owner_criteria_and_dependency(self):
        steps = seed_template("generic_part", [], False, mesh_intent="parametric")
        for s in steps:
            self.assertTrue(s.phase and s.title and s.acceptance, s.id)
        deps = {s.id: s.acceptance for s in steps}
        self.assertIn("(after M4)", deps["M9"])
        self.assertIn("(after M11)", deps["M12"])

    def test_brep_is_labelled_faceted_never_parametric(self):
        steps = {s.id: s for s in seed_template("generic_part", [], False, mesh_intent="brep")}
        self.assertIn("Faceted BREP", steps["M8"].title)
        self.assertIn("non-parametric", steps["M8"].title)
        self.assertNotIn("M11", steps, "a brep request must not seed parametric construction")

    def test_modify_runs_through_the_parametric_route(self):
        steps = [s.id for s in seed_template("generic_part", [], False, mesh_intent="modify")]
        self.assertIn("M11", steps)
        title = next(s.title for s in seed_template("generic_part", [], False,
                                                    mesh_intent="modify") if s.id == "M11")
        self.assertIn("requested change", title)


class EvidenceTest(unittest.TestCase):
    """Completion only from recorded facts; failures block with reasons."""

    def ledger(self):
        return ledger_for("reconstruct this mesh as editable cad")

    def test_successful_parametric_flow_completes_the_steps(self):
        led = self.ledger()
        led.on_tool_result(_Outcome({
            "mode": "parametric", "sha256": "ab" * 32, "facets": 9, "diag": {"closed": True},
            "units": "SCALE_PLAUSIBLE", "units_unconfirmed": False,
            "quality": {"sigma_mm": 0.05}, "survey": {"regions": []}, "plan": {"features": []},
            "build_success": True, "deviation": [{"id": "hausdorff"}],
            "outcome": "verified_parametric"}))
        by = {s.id: s.status for s in led.steps}
        for sid in ("M1", "M2", "M3", "M4", "M9", "M10", "M11", "M12"):
            self.assertEqual(by[sid], "done", sid)
        # no repair evidence on this clean run: the optional repair stages stay skipped
        for sid in ("M5", "M6", "M7"):
            self.assertEqual(by[sid], "skipped", sid)

    def test_a_tool_call_without_facts_completes_nothing(self):
        led = self.ledger()
        led.on_tool_result(_Outcome(None, geometry=True))
        # the generic path may mark a hinted step "running" and optional steps sit
        # "skipped", but NOTHING completes or blocks without recorded facts
        self.assertTrue(all(s.status in ("pending", "running", "skipped") for s in led.steps),
                        [(s.id, s.status) for s in led.steps])

    def test_failed_build_blocks_M11_and_M12(self):
        led = self.ledger()
        led.on_tool_result(_Outcome({
            "mode": "parametric", "sha256": "ab" * 32, "diag": True,
            "build_success": False, "build_error": "plan failed at 'pad'",
            "outcome": "reconstruction_failed"}))
        by = {s.id: s for s in led.steps}
        self.assertEqual(by["M11"].status, "blocked")
        self.assertIn("plan failed", by["M11"].note)
        self.assertEqual(by["M12"].status, "blocked")
        self.assertIn("no deviation evidence", by["M12"].note)

    def test_rejected_repair_blocks_M7_and_unauthorized_gating_blocks_M6(self):
        led = ledger_for("repair this mesh")
        led.on_tool_result(_Outcome({
            "mode": "repair", "sha256": "ab" * 32, "diag": True, "survey": None,
            "repair_verdicts": {"ACCEPTED": 5, "REJECTED": 1},
            "gated_needed": True, "authorized": False}))
        by = {s.id: s for s in led.steps}
        self.assertEqual(by["M5"].status, "done")
        self.assertEqual(by["M6"].status, "blocked")
        self.assertIn("authorization", by["M6"].note)
        self.assertEqual(by["M7"].status, "blocked")
        self.assertIn("REJECTED", by["M7"].note)

    def test_unconfirmed_units_block_M3(self):
        led = self.ledger()
        led.on_tool_result(_Outcome({
            "mode": "parametric", "sha256": "ab" * 32, "diag": True,
            "units": "POSSIBLE_MM_INCH_MISMATCH", "units_unconfirmed": True}))
        m3 = next(s for s in led.steps if s.id == "M3")
        self.assertEqual(m3.status, "blocked")
        self.assertIn("ask the user", m3.note)

    def test_solid_conversion_completes_M8_only_with_a_created_object(self):
        led = ledger_for("convert this mesh to a solid")
        led.on_tool_result(_Outcome({"mode": "solid", "sha256": "ab" * 32, "diag": True,
                                     "solid_created": True}))
        by = {s.id: s for s in led.steps}
        self.assertEqual(by["M8"].status, "done")
        led2 = ledger_for("convert this mesh to a solid")
        led2.on_tool_result(_Outcome({"mode": "solid", "sha256": "ab" * 32, "diag": True,
                                      "solid_created": False}))
        self.assertEqual(next(s for s in led2.steps if s.id == "M8").status, "pending")


class LifecycleTest(unittest.TestCase):
    def test_mesh_intent_survives_serialization(self):
        led = ledger_for("reconstruct this mesh")
        import json

        back = Ledger.from_dict(json.loads(led.to_json()))
        self.assertEqual(back.mesh_intent, "parametric")
        self.assertEqual([s.id for s in back.steps], [s.id for s in led.steps])

    def test_controlled_retarget_keeps_valid_evidence(self):
        led = ledger_for("reconstruct this mesh")
        for sid in ("M1", "M2", "M3", "M4"):
            s = next(x for x in led.steps if x.id == sid)
            s.status, s.claimed = "done", False
        self.assertTrue(led.retarget_mesh_intent("analyse"))
        self.assertEqual(led.mesh_intent, "analyse")
        self.assertEqual([(s.id, s.status) for s in led.steps],
                         [("M1", "done"), ("M2", "done"), ("M3", "done"), ("M4", "done")])
        self.assertTrue(any(w["what"] == "mesh_intent" and w["status"] == "changed"
                            for w in led.waivers))
        self.assertFalse(led.retarget_mesh_intent("analyse"))       # same intent: no-op
        self.assertFalse(led.retarget_mesh_intent("ambiguous"))     # ambiguity is not a target


class GateAndPlaybookTest(unittest.TestCase):
    def test_playbook_activates_from_mesh_intent_without_trigger_words(self):
        names = {"reconstruct_mesh", "measure_reference_part", "compare_to_reference"}
        self.assertIn("reconstruction",
                      playbooks_for("generic_part", "turn this stl into native freecad",
                                    names, mesh_intent="parametric"))
        # wording fallback still works on its own:
        self.assertIn("reconstruction",
                      playbooks_for("generic_part", "please convert the imported step to cad",
                                    names))
        self.assertNotIn("reconstruction",
                          playbooks_for("generic_part", "build a bracket", names))

    @staticmethod
    def _state(**kw):
        from oscar.flow import TurnState

        base = dict(is_work_turn=True, geometry_ok=1, deliver_ok=0, open_errors=[],
                    last_script_fail="", validation=None, last_geometry_step=1,
                    gate_nudges={}, nudges_total=0)
        base.update(kw)
        return TurnState(**base)

    def test_finish_gate_fires_on_mesh_intent_without_reconstruction_wording(self):
        from oscar.flow import AgentConfig

        led = ledger_for("turn this stl into native freecad")     # no trigger words at all
        led.reference_status = "none"
        verdict = finish_gate(led, self._state(), AgentConfig())
        self.assertIn("RECONSTRUCTION_UNVERIFIED", verdict.codes,
                      "a mesh reconstruction must not escape accuracy validation by wording")

    def test_finish_gate_clears_when_deviation_ran(self):
        from oscar.flow import AgentConfig

        led = ledger_for("turn this stl into native freecad")
        led.reference_status = "none"
        validation = {"ran": ["recon.deviation"], "findings": [], "rows": []}
        verdict = finish_gate(led, self._state(validation=validation), AgentConfig())
        self.assertNotIn("RECONSTRUCTION_UNVERIFIED", verdict.codes)


class W1HardeningTest(unittest.TestCase):
    """The two approved W-1 hardening fixes: M13 is never auto-completed by validation,
    and valid repair evidence survives repair->parametric retargeting."""

    @staticmethod
    def _validation(ok=True):
        return {"ran": ["recon.deviation"], "gate_errors": [] if ok else [{"message": "boom"}],
                "fail_rows": [], "findings": [], "after_step": 3}

    def _mesh_ledger_with_validation(self, ok=True):
        led = ledger_for("reconstruct this mesh")
        led.on_tool_result(_Outcome({
            "mode": "parametric", "sha256": "ab" * 32, "diag": True,
            "units": "SCALE_PLAUSIBLE", "quality": {"sigma_mm": 0.05},
            "survey": {"regions": []}, "plan": {"features": []}, "build_success": True,
            "deviation": [{"id": "hausdorff", "value": 0.1}], "outcome": "verified_parametric"}))
        led.apply_validation(self._validation(ok=ok))
        return led

    def test_passing_validation_completes_M12_but_not_M13(self):
        led = self._mesh_ledger_with_validation(ok=True)
        by = {s.id: s.status for s in led.steps}
        self.assertEqual(by["M12"], "done")
        self.assertEqual(by["M13"], "pending",
                         "user review must NEVER complete from automatic validation")

    def test_failed_validation_leaves_M13_pending(self):
        led = self._mesh_ledger_with_validation(ok=False)
        by = {s.id: s.status for s in led.steps}
        self.assertNotEqual(by["M13"], "done")
        self.assertEqual(by["M13"], "pending")

    def test_no_validator_does_not_skip_M13(self):
        led = ledger_for("reconstruct this mesh")
        led.apply_validation(None)          # "automatic validation unavailable" path
        m13 = next(s for s in led.steps if s.id == "M13")
        self.assertEqual(m13.status, "pending", "M13 must not be auto-skipped either")

    def test_non_mesh_verify_steps_still_auto_complete(self):
        led = Ledger.from_request("make a plate")
        led.on_tool_result(_Outcome(None, name="create_object", geometry=True))
        led.confirm_evidenced_steps()
        led.apply_validation(self._validation(ok=True))
        verify = [s for s in led.steps if s.phase == "VERIFY" and s.owner == "app"]
        self.assertTrue(verify and all(s.status == "done" for s in verify),
                        "the enclosure/generic VERIFY step must keep auto-completing")

    @staticmethod
    @staticmethod
    def _repair_outcome(din="d-in", dout="d-out", ops=(("fillupHoles", {"max_edge": 5}),),
                        authorized=True, verdicts=None, step_index=2, source=None):
        specs = [{"op": name, "params": dict(params)} for name, params in ops]
        return _Outcome({
            "mode": "repair", "sha256": "ab" * 32, "diag": True, "survey": None,
            "repair_verdicts": verdicts or {"ACCEPTED": 5},
            "gated_needed": True, "authorized": authorized,
            "authorized_ops": specs,
            "source_digest": source or din,
            "input_digest": din, "input_unit": "unknown", "input_mm": None,
            "converted": False,
            "output_digest": dout, "output_unit": "unknown", "output_mm": None},
            step_index=step_index)

    @staticmethod
    def _repaired(din="d-in", dout="d-out", ops=(("fillupHoles", {"max_edge": 5}),),
                  source=None):
        led = ledger_for("repair this mesh")
        led.on_tool_result(W1HardeningTest._repair_outcome(din=din, dout=dout, ops=ops,
                                                           source=source))
        return led

    def test_valid_M5_M7_evidence_survives_repair_to_parametric(self):
        led = self._repaired()
        by = {s.id: s.status for s in led.steps}
        for sid in ("M5", "M6", "M7"):
            self.assertEqual(by[sid], "done", sid)
        m5_before = next(s for s in led.steps if s.id == "M5").evidence
        m7_before = next(s for s in led.steps if s.id == "M7").evidence
        self.assertTrue(led.retarget_mesh_intent("parametric"))
        by = {s.id: s.status for s in led.steps}
        for sid in ("M5", "M6", "M7"):
            self.assertEqual(by[sid], "done", sid)
        m5_after = next(s for s in led.steps if s.id == "M5")
        m7_after = next(s for s in led.steps if s.id == "M7")
        self.assertEqual(m5_after.evidence, m5_before,
                         "the ORIGINAL evidence objects are preserved, not copied")
        self.assertEqual(m7_after.evidence, m7_before)
        self.assertEqual(m7_after.evidence[-1]["mesh"]["output"], "d-out")
        self.assertEqual(getattr(led, "mesh_digest", None), "d-out")
        self.assertIn("M11", by)
        ids = [s.id for s in led.steps]
        self.assertEqual(len(ids), len(set(ids)))

    def test_changed_live_mesh_digest_drops_repair_evidence(self):
        led = self._repaired()
        led.mesh_live = dict(led.mesh_live, input="d-elsewhere")  # live mesh is NOT d-out
        led.retarget_mesh_intent("parametric")
        by = {s.id: s.status for s in led.steps}
        for sid in ("M5", "M6", "M7"):
            self.assertEqual(by[sid], "pending", sid)
            self.assertIn("historical", next(s for s in led.steps if s.id == sid).note)

    def test_changed_repair_result_blocks_rather_than_preserves(self):
        led = self._repaired()
        led.on_tool_result(self._repair_outcome(verdicts={"ACCEPTED": 3, "REJECTED": 2},
                                                authorized=False, step_index=5))
        by = {s.id: s for s in led.steps}
        self.assertEqual(by["M7"].status, "blocked")
        self.assertEqual(by["M6"].status, "blocked")

    def test_unapproved_gating_is_never_preserved_as_approved(self):
        led = self._repaired()
        m6 = next(s for s in led.steps if s.id == "M6")
        m6.status, m6.note = "blocked", "gated repair needs explicit user authorization"
        led.retarget_mesh_intent("parametric")
        m6_after = next(s for s in led.steps if s.id == "M6")
        self.assertNotEqual(m6_after.status, "done")   # the block was NOT preserved as
        self.assertIn("authorization", m6_after.note)  # approved; the reason survives

    def test_repeated_identical_intent_creates_no_duplicates(self):
        led = self._repaired()
        self.assertFalse(led.retarget_mesh_intent("repair"))
        ids = [s.id for s in led.steps]
        self.assertEqual(len(ids), len(set(ids)))

    def test_other_transitions_still_behave(self):
        led = ledger_for("reconstruct this mesh")
        for sid in ("M1", "M2", "M3", "M4"):
            s = next(x for x in led.steps if x.id == sid)
            s.status, s.claimed = "done", False
        led.retarget_mesh_intent("solid")
        self.assertEqual([s.id for s in led.steps],
                         ["M1", "M2", "M3", "M4", "M8", "M12", "M14"])
        led2 = ledger_for("reconstruct this mesh")
        for sid in ("M1", "M2", "M3", "M4"):
            s = next(x for x in led2.steps if x.id == sid)
            s.status, s.claimed = "done", False
        led2.retarget_mesh_intent("analyse")
        self.assertEqual([s.id for s in led2.steps], ["M1", "M2", "M3", "M4"])


class GeometryBoundEvidenceTest(unittest.TestCase):
    """W-1 final fix: M5-M7 evidence and M6 authorization are bound to working-mesh
    GEOMETRY IDENTITY (K3 mesh digests), not to the source-file sha or step indices."""

    def test_same_source_file_changed_working_mesh_invalidates(self):
        led = W1HardeningTest._repaired(din="d1", dout="d2")
        # a later call on a DIFFERENT working mesh (same source-file sha in the flow):
        led.on_tool_result(_Outcome({"mode": "analyse", "sha256": "ab" * 32,
                                     "diag": True, "input_digest": "d9"}, step_index=2))
        by = {s.id: s.status for s in led.steps}
        for sid in ("M5", "M6", "M7"):
            self.assertEqual(by[sid], "pending", sid)

    def test_same_step_index_different_digest_invalidates(self):
        led = W1HardeningTest._repaired(din="d1", dout="d2")
        led.on_tool_result(_Outcome({          # SAME step_index as the repair record
            "mode": "parametric", "sha256": "ab" * 32, "diag": True,
            "repair_verdicts": {"ACCEPTED": 1}, "gated_needed": False, "authorized": True,
            "authorized_ops": [{"op": "fillupHoles", "params": {"max_edge": 5}}],
            "input_digest": "d9", "output_digest": "d9x"}, step_index=2))
        self.assertIn("digest mismatch",
                      next(s for s in led.steps if s.id == "M7").note)

    def test_valid_unchanged_chain_is_preserved_on_retarget(self):
        led = W1HardeningTest._repaired()
        ev_before = list(next(s for s in led.steps if s.id == "M7").evidence)
        led.retarget_mesh_intent("parametric")
        m7 = next(s for s in led.steps if s.id == "M7")
        self.assertEqual((m7.status, m7.evidence), ("done", ev_before))
        ident = m7.evidence[-1]["mesh"]
        self.assertEqual((ident["input"], ident["output"], ident["ops"]),
                         ("d-in", "d-out",
                          [{"op": "fillupHoles", "params": {"max_edge": 5}}]))

    def test_changed_repair_output_makes_the_chain_stale(self):
        led = W1HardeningTest._repaired(din="d1", dout="d2")
        # the working mesh advanced PAST the recorded chain (a different repair output the
        # chain never recorded) — the live digest no longer matches the chain output:
        led.mesh_live = dict(led.mesh_live, input="d3")
        led.retarget_mesh_intent("parametric")
        by = {s.id: s.status for s in led.steps}
        for sid in ("M5", "M6", "M7"):
            self.assertEqual(by[sid], "pending", sid)
            self.assertTrue(next(s for s in led.steps if s.id == sid).evidence,
                            "the recorded chain history is retained")

    def test_M5_recommendation_becomes_stale(self):
        led = W1HardeningTest._repaired(din="d1", dout="d2")
        led.mesh_live = dict(led.mesh_live, input="d-live-changed")
        led.retarget_mesh_intent("parametric")
        m5 = next(s for s in led.steps if s.id == "M5")
        self.assertEqual(m5.status, "pending")
        self.assertIn("historical", m5.note)
        self.assertTrue(m5.evidence, "the stale recommendation history is retained")

    def test_M6_approval_invalid_after_operation_change_without_authorization(self):
        led = W1HardeningTest._repaired(ops=(("fillupHoles", {"max_edge": 5}),))
        # a DIFFERENT gated operation runs with NO new authorization
        led.on_tool_result(W1HardeningTest._repair_outcome(
            ops=(("removeNonManifolds", {}),), authorized=False, step_index=4))
        m6 = next(s for s in led.steps if s.id == "M6")
        self.assertEqual(m6.status, "blocked")
        self.assertIn("authorization", m6.note)

    def test_M6_approval_superseded_by_new_operation_set(self):
        led = W1HardeningTest._repaired(ops=(("fillupHoles", {"max_edge": 5}),))
        # a DIFFERENT op set runs on a different mesh WITHOUT a matching authorization:
        # the old approval (fillupHoles on d-in) must not carry over as current
        led.on_tool_result(W1HardeningTest._repair_outcome(
            din="d9", dout="d9x", ops=(("fillupHoles", {"max_edge": 5}), ("removeNonManifolds", {})),
            authorized=False, step_index=4))
        led.retarget_mesh_intent("parametric")
        m6 = next(s for s in led.steps if s.id == "M6")
        self.assertIn(m6.status, ("pending", "blocked"))
        self.assertNotEqual(m6.status, "done")   # never preserved as approved for d9

    def test_M6_approval_invalid_after_geometry_change(self):
        led = W1HardeningTest._repaired(din="d1", dout="d2")
        led.on_tool_result(_Outcome({"mode": "analyse", "sha256": "ab" * 32,
                                     "diag": True, "input_digest": "d9"}, step_index=3))
        led.retarget_mesh_intent("parametric")
        self.assertEqual(next(s for s in led.steps if s.id == "M6").status, "pending")

    def test_repair_to_parametric_with_matching_evidence(self):
        led = W1HardeningTest._repaired(din="d1", dout="d2")
        self.assertTrue(led.retarget_mesh_intent("parametric"))
        self.assertTrue(all(next(s for s in led.steps if s.id == sid).status == "done"
                            for sid in ("M5", "M6", "M7")))

    def test_repair_to_parametric_with_mismatched_evidence(self):
        led = W1HardeningTest._repaired(din="d1", dout="d2")
        led.mesh_live = dict(led.mesh_live, input="d-unmatched")  # neither d1 nor d2
        self.assertTrue(led.retarget_mesh_intent("parametric"))
        self.assertTrue(all(next(s for s in led.steps if s.id == sid).status == "pending"
                            for sid in ("M5", "M6", "M7")))

    def test_missing_live_digest_leaves_steps_visibly_unverified(self):
        led = W1HardeningTest._repaired()
        for attr in ("mesh_digest", "mesh_live"):
            if hasattr(led, attr):  # e.g. a ledger reopened without the live identity
                delattr(led, attr)
        led.retarget_mesh_intent("parametric")
        for sid in ("M5", "M6", "M7"):
            st = next(s for s in led.steps if s.id == sid)
            self.assertEqual(st.status, "pending")
            self.assertIn("unavailable", st.note)
        # legacy evidence with NO identity at all is treated the same way:
        led2 = ledger_for("repair this mesh")
        for sid in ("M5", "M6", "M7"):
            s = next(x for x in led2.steps if x.id == sid)
            s.status, s.claimed = "done", False
        led2.retarget_mesh_intent("parametric")
        self.assertTrue(all(next(s for s in led2.steps if s.id == sid).status != "done"
                            for sid in ("M5", "M6", "M7")),
                        "claims with no evidence and no identity are never preserved")

    def test_newer_failed_repair_blocks_completed_evidence(self):
        led = W1HardeningTest._repaired(din="d1", dout="d2")
        led.on_tool_result(W1HardeningTest._repair_outcome(
            verdicts={"REJECTED": 2}, authorized=True, step_index=9))
        m7 = next(s for s in led.steps if s.id == "M7")
        self.assertEqual(m7.status, "blocked")
        self.assertIn("REJECTED", m7.note)

    def test_historical_evidence_remains_without_being_current(self):
        led = W1HardeningTest._repaired(din="d1", dout="d2")
        n_before = len(next(s for s in led.steps if s.id == "M5").evidence)
        led.mesh_live = dict(led.mesh_live, input="d-other")
        led.retarget_mesh_intent("parametric")
        m5 = next(s for s in led.steps if s.id == "M5")
        self.assertEqual(m5.status, "pending")
        self.assertEqual(len(m5.evidence), n_before,
                         "history kept, only the CLAIM to currency is dropped")
        self.assertIn("historical", m5.note)


class AuthorizationSpecTest(unittest.TestCase):
    """W-1 final correction A: M6 approval is bound to the EXACT executed repair
    specification (operation names + effective kernel parameters), not bare names."""

    OPS = (("fillupHoles", {"max_edge": 5}),)

    @staticmethod
    def _ledger_with_approval(spec=None, din="d-in"):
        led = W1HardeningTest._repaired(din=din, dout="d-out",
                                        ops=(("fillupHoles", {"max_edge": 5}),))
        if spec is not None:                      # overwrite M6's recorded approval spec
            m6 = next(s for s in led.steps if s.id == "M6")
            m6.evidence[-1]["mesh"] = dict(m6.evidence[-1]["mesh"], ops=list(spec))
        return led

    def test_same_mesh_same_op_same_params_reusable(self):
        led = self._ledger_with_approval()
        self.assertTrue(led.retarget_mesh_intent("parametric"))
        self.assertTrue(all(next(s for s in led.steps if s.id == sid).status == "done"
                            for sid in ("M5", "M6", "M7")))

    def test_same_mesh_changed_derived_parameter_rejected(self):
        led = self._ledger_with_approval()          # approved max_edge=5
        led.on_tool_result(W1HardeningTest._repair_outcome(
            ops=(("fillupHoles", {"max_edge": 10}),), authorized=False, step_index=4))
        m6 = next(s for s in led.steps if s.id == "M6")
        self.assertEqual(m6.status, "blocked")

    def test_same_mesh_changed_operation_rejected(self):
        led = self._ledger_with_approval()
        led.on_tool_result(W1HardeningTest._repair_outcome(
            ops=(("removeNonManifolds", {}),), authorized=False, step_index=4))
        self.assertEqual(next(s for s in led.steps if s.id == "M6").status, "blocked")

    def test_changed_input_geometry_rejected(self):
        led = self._ledger_with_approval(din="d1")
        led.on_tool_result(W1HardeningTest._repair_outcome(
            din="d2", dout="d2x", authorized=False, step_index=4))
        m6 = next(s for s in led.steps if s.id == "M6")
        self.assertIn(m6.status, ("blocked", "pending"))
        self.assertNotEqual(m6.status, "done")

    def test_changed_derivation_constant_rejects_old_approval(self):
        # the REAL derivation: FILL_ARG_EDGES drives the executed fillupHoles argument,
        # so a changed constant yields a different recorded spec for the same mesh
        import FreeCAD as App
        import Mesh
        import Part
        from oscar.recon import meshops

        pts, facs = Part.makeBox(20, 20, 5).tessellate(2.0)
        mesh = Mesh.Mesh((list(pts), [tuple(f) for f in facs]))
        real = meshops.FILL_ARG_EDGES
        try:
            meshops.FILL_ARG_EDGES = 2.0
            spec_a = next(a["spec"] for a in meshops.repair(mesh, authorized=["fillupHoles"])["actions"]
                          if a["operation"] == "fillupHoles")
            meshops.FILL_ARG_EDGES = 8.0
            spec_b = next(a["spec"] for a in meshops.repair(mesh, authorized=["fillupHoles"])["actions"]
                          if a["operation"] == "fillupHoles")
        finally:
            meshops.FILL_ARG_EDGES = real
        self.assertNotEqual(spec_a, spec_b),
        self.assertLess(spec_a["max_edge"], spec_b["max_edge"])
        # and the ledger-side consequence: an approval recorded for spec_a never covers b
        led = self._ledger_with_approval(spec=[{"op": "fillupHoles", "params": spec_a}])
        led.on_tool_result(W1HardeningTest._repair_outcome(
            ops=(("fillupHoles", spec_b),), authorized=False, step_index=4))
        self.assertEqual(next(s for s in led.steps if s.id == "M6").status, "blocked")

    def test_save_reopen_retains_the_exact_approved_specification(self):
        import json

        led = self._ledger_with_approval()
        m6_spec = _newest_spec(led, "M6")
        back = Ledger.from_dict(json.loads(led.to_json()))
        self.assertEqual(_newest_spec(back, "M6"), m6_spec,
                         "the exact spec dict survives Ledger serialization")

    def test_legacy_name_only_evidence_is_not_parameter_specific(self):
        led = W1HardeningTest._repaired()
        for sid in ("M5", "M6", "M7"):
            m = next(s for s in led.steps if s.id == sid)
            m.evidence[-1]["mesh"] = dict(m.evidence[-1]["mesh"],
                                          ops=["fillupHoles"])   # legacy strings
        led.retarget_mesh_intent("parametric")
        self.assertTrue(all(next(s for s in led.steps if s.id == sid).status == "pending"
                            for sid in ("M5", "M6", "M7")),
                        "name-only approval is never treated as parameter-specific")


def _newest_spec(ledger, sid):
    from oscar.flow_mesh import _newest_identity

    ident = _newest_identity(ledger, sid)
    return list(ident.get("ops") or []) if ident else None


class UnitIdentityTest(unittest.TestCase):
    """W-1 final correction B: identities distinguish source / unit-normalized working
    input / repaired output, and comparisons only run within a verified representation."""

    @staticmethod
    def _parametric_outcome(source, base_mm, out_mm, step_index=2):
        return _Outcome({
            "mode": "parametric", "sha256": "ab" * 32, "diag": True,
            "units": "UNIT_KNOWN", "quality": {"sigma_mm": 0.05},
            "survey": {"regions": []}, "plan": {"features": []}, "build_success": True,
            "deviation": [{"id": "hausdorff"}], "outcome": "verified_parametric",
            "repair_verdicts": {"ACCEPTED": 5}, "gated_needed": True, "authorized": True,
            "authorized_ops": [{"op": "fillupHoles", "params": {"max_edge": 5}}],
            "source_digest": source, "input_digest": base_mm,
            "input_unit": "mm", "input_mm": base_mm, "converted": True,
            "output_digest": out_mm, "output_unit": "mm", "output_mm": out_mm},
            step_index=step_index)

    def test_inch_source_confirmed_inch_workflow(self):
        led = ledger_for("reconstruct this mesh")
        led.on_tool_result(self._parametric_outcome("d-src-inch", "d-mm", "d-mm-rep"))
        by = {s.id: s.status for s in led.steps}
        for sid in ("M5", "M6", "M7"):
            self.assertEqual(by[sid], "done", sid)
        ident = _newest_spec  # noqa: F841  (readability)
        from oscar.flow_mesh import _newest_identity
        m7 = _newest_identity(led, "M7")
        self.assertEqual((m7["source"], m7["input"], m7["input_unit"], m7["converted"]),
                         ("d-src-inch", "d-mm", "mm", True))
        self.assertTrue(led.retarget_mesh_intent("modify"))
        self.assertTrue(all(next(s for s in led.steps if s.id == sid).status == "done"
                            for sid in ("M5", "M6", "M7")))

    def test_inch_source_converted_to_millimetres_records_both_identities(self):
        led = ledger_for("reconstruct this mesh")
        led.on_tool_result(self._parametric_outcome("d-src-inch", "d-mm", "d-mm-rep"))
        from oscar.flow_mesh import _newest_identity

        m5 = _newest_identity(led, "M5")
        self.assertEqual(m5["source"], "d-src-inch")
        self.assertEqual(m5["input"], "d-mm")
        self.assertEqual(m5["input_unit"], "mm")

    def test_equivalent_geometry_in_inch_and_mm_is_not_falsely_invalidated(self):
        # repair MODE on the raw inch source, then a converted parametric call on the
        # SAME source: the mm call vouches for the conversion (converted + same source)
        led = W1HardeningTest._repaired(din="d-src-inch", dout="d-inch-rep",
                                        source="d-src-inch")
        led.on_tool_result(self._parametric_outcome("d-src-inch", "d-mm", "d-mm-rep",
                                                    step_index=4))
        m7 = next(s for s in led.steps if s.id == "M7")
        self.assertNotIn("digest mismatch", m7.note)
        self.assertEqual(m7.status, "done")

    def test_inch_repair_then_mm_parametric_preserves_across_units(self):
        led = W1HardeningTest._repaired(din="d-src-inch", dout="d-inch-rep",
                                        source="d-src-inch")
        led.on_tool_result(self._parametric_outcome("d-src-inch", "d-mm", "d-mm-rep",
                                                    step_index=4))
        self.assertTrue(led.retarget_mesh_intent("parametric"))
        self.assertTrue(all(next(s for s in led.steps if s.id == sid).status == "done"
                            for sid in ("M5", "M6", "M7")),
                        "a valid repair is not invalidated merely by confirmed units")

    def test_mm_repair_followed_by_matching_later_operation(self):
        led = W1HardeningTest._repaired(din="d1", dout="d2")
        led.on_tool_result(_Outcome({"mode": "analyse", "sha256": "ab" * 32,
                                     "diag": True, "source_digest": "d1",
                                     "input_digest": "d1", "input_unit": "unknown",
                                     "input_mm": None, "converted": False},
                                    step_index=3))
        led.retarget_mesh_intent("parametric")
        self.assertTrue(all(next(s for s in led.steps if s.id == sid).status == "done"
                            for sid in ("M5", "M6", "M7")))

    def test_actual_geometry_change_after_conversion_invalidates(self):
        led = ledger_for("reconstruct this mesh")
        led.on_tool_result(self._parametric_outcome("S1", "mm1", "mm1-rep"))
        # a later converted call on a DIFFERENT source vouches only for S1 (no re-repair,
        # so the demotion is not superseded by a new current chain):
        led.on_tool_result(_Outcome({"mode": "analyse", "sha256": "ab" * 32,
                                     "diag": True, "source_digest": "S2",
                                     "input_digest": "mm2", "input_unit": "mm",
                                     "input_mm": "mm2", "converted": True},
                                    step_index=4))
        m7 = next(s for s in led.steps if s.id == "M7")
        self.assertIn("digest mismatch", m7.note)
        self.assertEqual(m7.status, "pending")

    def test_missing_or_ambiguous_unit_metadata_is_conservative(self):
        # two unknown-unit identities with no canonical digests and no conversion
        # voucher: only EXACT input equality preserves — anything else demotes
        led = W1HardeningTest._repaired(din="d1", dout="d2")
        led.on_tool_result(_Outcome({"mode": "analyse", "sha256": "ab" * 32,
                                     "diag": True, "source_digest": "dX",
                                     "input_digest": "dX", "input_unit": "unknown",
                                     "input_mm": None, "converted": False},
                                    step_index=3))
        m7 = next(s for s in led.steps if s.id == "M7")
        self.assertIn("digest mismatch", m7.note)

    def test_history_retained_when_identity_cannot_be_established(self):
        led = W1HardeningTest._repaired(din="d1", dout="d2")
        n = len(next(s for s in led.steps if s.id == "M6").evidence)
        led.on_tool_result(_Outcome({"mode": "analyse", "sha256": "ab" * 32,
                                     "diag": True, "source_digest": "dX",
                                     "input_digest": "dX", "input_unit": "unknown",
                                     "input_mm": None, "converted": False},
                                    step_index=3))
        led.retarget_mesh_intent("parametric")
        m6 = next(s for s in led.steps if s.id == "M6")
        self.assertEqual(m6.status, "pending")
        self.assertGreaterEqual(len(m6.evidence), n)
        self.assertIn("historical", m6.note)


class WorkTurnTest(unittest.TestCase):
    def test_a_mesh_turn_is_always_a_work_turn(self):
        # "analyse this mesh" classifies as a non-work intent ("other") — the mesh intent
        # still makes it a live work turn with the mesh template
        self.assertNotIn(classify_intent("analyse this mesh"),
                         ("enclosure", "fixture", "bracket", "generic_part", "modify", "fem"))
        led = ledger_for("analyse this mesh")
        self.assertTrue(led.is_work_turn())
        self.assertEqual(led.intent, "generic_part")


if __name__ == "__main__":
    unittest.main()
