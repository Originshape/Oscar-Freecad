"""Stage 12: the ``reconstruct_mesh`` tool — every mode through the real registry, on real
FreeCAD documents: analyse (read-only), repair (a COPY, original untouched, gated refusal),
solid (honest NON-PARAMETRIC classification) and parametric (native features + bound sheet +
deviation rows + RESULT CLASS). Registration pins included (tool #45).
"""

import math
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402

from oscar import prompt, tools  # noqa: E402
from oscar.flow import tool_category  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402
from oscar.toolrun import LITE_TOOLS, tool_meta  # noqa: E402


def bracket_mesh():
    shape = Part.makeBox(60, 40, 10)
    for x, y in ((15, 10), (45, 10), (15, 30), (45, 30)):
        shape = shape.cut(Part.makeCylinder(3.3, 30, App.Vector(x, y, -10)))
    return Mesh.Mesh(shape.tessellate(0.3))


class ReconToolCase(unittest.TestCase):
    def setUp(self):
        self._made = []
        self.addCleanup(self._close)

    def _close(self):
        from oscar.cad.health_eval import clear_cache

        for name in self._made:
            if name in App.listDocuments():
                App.closeDocument(name)
        clear_cache()

    def doc_with_mesh(self, name, mesh=None):
        doc = App.newDocument(name)
        self._made.append(doc.Name)
        obj = doc.addObject("Mesh::Feature", "Scan")
        obj.Mesh = mesh if mesh is not None else bracket_mesh()
        doc.recompute()
        return doc

    def run_tool(self, _ctx=None, **args):
        with use_context(_ctx or ToolContext()):
            return tools.run_tool("reconstruct_mesh", args)


class ReconToolModesTest(ReconToolCase):
    def test_registration_pins(self):
        self.assertIn("reconstruct_mesh", LITE_TOOLS)
        self.assertEqual(tool_category("reconstruct_mesh"), "build")
        self.assertIn("reconstruct_mesh", prompt.KNOWN_TOOLS)
        self.assertIsNotNone(tool_meta("reconstruct_mesh"))

    def test_analyse_changes_nothing_and_classifies_the_scan(self):
        doc = self.doc_with_mesh("RAnal")
        before = sorted(o.Name for o in doc.Objects)
        text, _img, is_error = self.run_tool(doc_name=doc.Name, mode="analyse")
        self.assertFalse(is_error, text)
        self.assertIn("closed=True", text)
        self.assertIn("class=prismatic", text)
        self.assertIn("SCAN QUALITY: READY", text)
        self.assertIn("RESULT CLASS: ANALYSIS ONLY", text)
        self.assertEqual(sorted(o.Name for o in doc.Objects), before)

    def test_repair_creates_a_copy_and_names_the_refused_operations(self):
        doc = self.doc_with_mesh("RRep")
        ctx = ToolContext(turn_id="turn-repair", call_id="call-repair",
                          request="Repair the imported scan")
        with mock.patch("oscar.netguard.ask_user", return_value="safe_only"):
            text, _img, is_error = self.run_tool(_ctx=ctx, doc_name=doc.Name, mode="repair")
        self.assertFalse(is_error, text)
        self.assertIsNotNone(doc.getObject("Scan_repaired"))
        self.assertEqual(int(doc.getObject("Scan").Mesh.CountFacets),
                         int(doc.getObject("Scan_repaired").Mesh.CountFacets))
        self.assertIn("gated operations not run without human approval", text)
        self.assertIn("RESULT CLASS: REPAIRED MESH", text)

    def test_llm_cannot_supply_repair_authorization(self):
        doc = self.doc_with_mesh("RNoLLMAuth")
        text, _img, is_error = self.run_tool(doc_name=doc.Name, mode="repair",
                                             authorize_repair=["fillupHoles"])
        self.assertTrue(is_error)
        self.assertIn("unknown parameter", text)
        self.assertIsNone(doc.getObject("Scan_repaired"))

    def test_explicit_human_approval_is_bound_to_this_mesh_request_and_call(self):
        doc = self.doc_with_mesh("RApproved")
        ctx = ToolContext(turn_id="turn-approved", call_id="call-approved",
                          request="Repair only this imported scan")
        with mock.patch("oscar.netguard.ask_user", return_value="approve") as ask:
            text, _img, is_error = self.run_tool(_ctx=ctx, doc_name=doc.Name, mode="repair")
        self.assertFalse(is_error, text)
        self.assertIsNotNone(doc.getObject("Scan_repaired"))
        self.assertIn("RESULT CLASS: REPAIRED MESH", text)
        args, kwargs = ask.call_args
        self.assertIn("Approve mesh repair operations", args[0])
        self.assertIn(ctx.request, kwargs["informative"])
        self.assertIn("Mesh digest:", kwargs["informative"])
        self.assertEqual(kwargs["default"], "cancel")

    def test_rejected_repair_never_executes(self):
        doc = self.doc_with_mesh("RRejected")
        ctx = ToolContext(turn_id="turn-rejected", call_id="call-rejected",
                          request="Repair this scan")
        with mock.patch("oscar.netguard.ask_user", return_value="cancel"):
            text, _img, is_error = self.run_tool(_ctx=ctx, doc_name=doc.Name, mode="repair")
        self.assertTrue(is_error, text)
        self.assertIn("user approval was not granted", text)
        self.assertIsNone(doc.getObject("Scan_repaired"))

    def test_repair_approval_uses_gui_dispatch_for_preflight_and_execution(self):
        from oscar import tools_recon

        doc = self.doc_with_mesh("RDispatch")
        ctx = ToolContext(turn_id="turn-dispatch", call_id="call-dispatch",
                          request="Repair this scan")
        real_gui_call = tools_recon._gui_call
        dispatched = []

        def record_dispatch(fn, op, timeout=tools_recon._TIMEOUT_S):
            dispatched.append(op)
            return real_gui_call(fn, op, timeout)

        with mock.patch.object(tools_recon, "_gui_call", side_effect=record_dispatch), \
                mock.patch("oscar.netguard.ask_user", return_value="approve"):
            text, _img, is_error = self.run_tool(_ctx=ctx, doc_name=doc.Name, mode="repair")
        self.assertFalse(is_error, text)
        self.assertEqual(dispatched, ["reconstruct_mesh repair approval", "reconstruct_mesh"])
        self.assertIsNotNone(doc.getObject("Scan_repaired"))

    def test_stale_and_reused_repair_approvals_are_rejected(self):
        from oscar import tools_recon
        from oscar.recon import provenance

        doc = self.doc_with_mesh("RStaleApproval")
        mesh_obj = doc.getObject("Scan")
        mesh = mesh_obj.Mesh
        request, turn_id, call_id = "repair this scan", "turn", "call"
        specs = tools_recon._repair_proposal(mesh)
        digest = provenance.mesh_digest(mesh)
        packed = tuple((op, tuple(sorted(spec.items()))) for op, spec in specs.items())
        approval = tools_recon._RepairApproval(
            doc_name=doc.Name, mesh_name=mesh_obj.Name, mesh_digest=digest,
            operations=tuple(specs), specs=packed,
            request_digest=tools_recon._request_digest(request), turn_id=turn_id,
            call_id=call_id,
            binding=tools_recon._repair_binding(digest, specs, request, turn_id, call_id))
        self.assertEqual(tools_recon._consume_repair_approval(
            approval, doc.Name, mesh_obj, request, turn_id, call_id), tuple(specs))
        with self.assertRaisesRegex(ValueError, "already used"):
            tools_recon._consume_repair_approval(
                approval, doc.Name, mesh_obj, request, turn_id, call_id)

        request_mismatch = tools_recon._RepairApproval(
            doc_name=doc.Name, mesh_name=mesh_obj.Name, mesh_digest=digest,
            operations=tuple(specs), specs=packed,
            request_digest=tools_recon._request_digest(request), turn_id=turn_id,
            call_id=call_id,
            binding=tools_recon._repair_binding(digest, specs, request, turn_id, call_id))
        with self.assertRaisesRegex(ValueError, "stale"):
            tools_recon._consume_repair_approval(
                request_mismatch, doc.Name, mesh_obj, "different repair request", turn_id, call_id)
        # A rejected mismatched attempt does not consume the still-valid approval.
        self.assertEqual(tools_recon._consume_repair_approval(
            request_mismatch, doc.Name, mesh_obj, request, turn_id, call_id), tuple(specs))

        specs = tools_recon._repair_proposal(mesh)
        digest = provenance.mesh_digest(mesh)
        stale = tools_recon._RepairApproval(
            doc_name=doc.Name, mesh_name=mesh_obj.Name, mesh_digest=digest,
            operations=tuple(specs),
            specs=tuple((op, tuple(sorted(spec.items()))) for op, spec in specs.items()),
            request_digest=tools_recon._request_digest(request), turn_id=turn_id,
            call_id=call_id,
            binding=tools_recon._repair_binding(digest, specs, request, turn_id, call_id))
        mesh_obj.Mesh = Mesh.Mesh(Part.makeBox(70, 40, 10).tessellate(0.3))
        with self.assertRaisesRegex(ValueError, "stale"):
            tools_recon._consume_repair_approval(
                stale, doc.Name, mesh_obj, request, turn_id, call_id)

    def test_solid_is_honestly_labelled_non_parametric(self):
        doc = self.doc_with_mesh("RSol")
        text, _img, is_error = self.run_tool(doc_name=doc.Name, mode="solid")
        self.assertFalse(is_error, text)
        solid = doc.getObject("Scan_solid")
        self.assertIsNotNone(solid)
        self.assertEqual(solid.TypeId, "Part::Feature")
        self.assertTrue(solid.Shape.isValid())
        self.assertIn("RESULT CLASS: NON-PARAMETRIC SOLID", text)

    def test_parametric_builds_native_features_with_deviation_and_result_class(self):
        doc = self.doc_with_mesh("RPar")
        text, _img, is_error = self.run_tool(doc_name=doc.Name)     # mode defaults to parametric
        self.assertFalse(is_error, text)
        self.assertEqual(len([o for o in doc.Objects if o.TypeId == "PartDesign::Hole"]), 4)
        sheets = [o for o in doc.Objects
                  if o.TypeId == "Spreadsheet::Sheet" and o.Name != "OscarParameterProbe"]
        self.assertEqual(len(sheets), 1)
        self.assertIn("scan_to_model.rms", text)
        self.assertIn("RESULT CLASS: PARAMETRIC CAD MODEL", text)
        body = next(o for o in doc.Objects if o.TypeId == "PartDesign::Body")
        expected = 60 * 40 * 10 - 4 * math.pi * 3.3 ** 2 * 10
        self.assertAlmostEqual(body.Shape.Volume, expected, delta=expected * 0.02)

    def test_errors_carry_the_reason(self):
        doc = self.doc_with_mesh("RErr")
        text, _img, is_error = self.run_tool(doc_name=doc.Name, object="NoSuch")
        self.assertTrue(is_error, text)
        self.assertIn("NoSuch", text)


class ScanQualityRenderTest(unittest.TestCase):
    """G5: the SCAN QUALITY block and the degraded-outcome LIMITATION wording (pure
    renderer tests — the detector itself is covered by tests/test_recon_quality.py)."""

    @staticmethod
    def _parametric(scan_quality=None, scan_degraded=None):
        meta = {"route": "prismatic_extrusion", "holes": 0}
        if scan_degraded:
            meta["scan_degraded"] = scan_degraded
        return {"mode": "parametric", "mesh": "Scan", "meta": meta, "supported": True,
                "build": {"success": True, "body": "Body", "spreadsheet": "Parameters",
                          "feature_count": 2, "parameter_count": 3},
                "classification": {"class": "prismatic"},
                "units": None, "deviation": [], "registration": None,
                "scan_quality": scan_quality}

    def test_a_degraded_parametric_result_states_the_cap_and_the_reasons(self):
        from oscar.tools_recon import _render

        text = _render(self._parametric(
            scan_degraded=["1 doubled-surface region(s) (offsets [0.5] mm)"]))
        self.assertIn("LIMITATION: scan quality DEGRADED — the outcome is capped at "
                      "PARTIAL", text)
        self.assertIn("doubled-surface", text)

    def test_a_caution_scan_reports_its_notes_without_capping(self):
        from oscar.tools_recon import _render

        sq = {"readiness": {"verdict": "caution", "reasons": [],
                            "notes": ["outer scan rim present (one-sided/open scan)"]}}
        text = _render(self._parametric(scan_quality=sq))
        self.assertIn("scan note: outer scan rim present", text)
        self.assertNotIn("capped at PARTIAL", text)

    def test_the_analysis_payload_carries_the_scan_quality_digest(self):
        from oscar.tools_recon import _scan_quality_digest

        sq = {"readiness": {"verdict": "degraded", "reasons": ["r"], "notes": []},
              "completeness": {"boundary_loops": 1, "open_chains": 0,
                               "loops": [{"kind": "hole_opening", "detail": "",
                                          "n_edges": 16, "spanned_area_mm2": 113.0,
                                          "area_share": 0.09, "radius_mm": 6.0,
                                          "rim_alignment": 1.0,
                                          "through": {"other": 1, "depth_mm": 6.0}}]},
              "integrity": {"spikes": {"count": 0, "estimated": False, "examples": []},
                            "doubled_surfaces": []},
              "plausibility": {"bbox_diagonal_mm": 43.9}}
        digest = _scan_quality_digest(sq)
        self.assertEqual(digest["readiness"]["verdict"], "degraded")
        self.assertEqual(digest["loops"][0]["kind"], "hole_opening")
        self.assertEqual(digest["loops"][0]["through"]["depth_mm"], 6.0)
        self.assertIsNone(_scan_quality_digest(None))


class G6FillPolicyRenderTest(unittest.TestCase):
    """G6: the repair-mode surface of the loop-aware fill policy — SKIPPED authorization,
    MISSING REGIONS, and the measured-only deviation line."""

    @staticmethod
    def holed_sheet(n=21, drop_i=(9, 12)):
        """An open grid sheet with a 3×3-cell dropout (a 12-edge non-fillable loop)."""
        import numpy as np

        xs = np.linspace(0, 20, n)
        X, Y = np.meshgrid(xs, xs)
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n * n)])
        dropped = {(i, j) for i in range(*drop_i) for j in range(*drop_i)}
        facs = []
        for i in range(n - 1):
            for j in range(n - 1):
                if (i, j) in dropped:
                    continue
                a, b, c, e = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
                facs += [(a, b, e), (a, e, c)]
        return Mesh.Mesh(([App.Vector(*p) for p in pts], facs))

    def test_repair_mode_reports_the_skip_and_the_missing_regions(self):
        case = ReconToolCase()
        case.setUp()
        self.addCleanup(case._close)
        doc = case.doc_with_mesh("G6Rep", mesh=self.holed_sheet())
        ctx = ToolContext(turn_id="turn-g6", call_id="call-g6", request="Repair the mesh")
        with mock.patch("oscar.netguard.ask_user", return_value="approve"):
            text, _img, is_error = case.run_tool(_ctx=ctx, doc_name=doc.Name, mode="repair")
        self.assertFalse(is_error, text)
        self.assertIn("SKIPPED fillupHoles: G6 fill policy", text)
        self.assertIn("MISSING REGIONS: 1 unfilled opening(s)", text)
        self.assertIn("never auto-filled", text)
        self.assertEqual(int(doc.getObject("Scan").Mesh.CountFacets),
                         int(doc.getObject("Scan_repaired").Mesh.CountFacets),
                         "a skipped fill must not change a single facet")

    def test_parametric_render_states_the_measured_only_mask(self):
        from oscar.tools_recon import _render

        text = _render({"mode": "parametric", "mesh": "Scan", "supported": True,
                        "meta": {"route": "prismatic_extrusion", "holes": 0},
                        "build": {"success": True, "body": "Body",
                                  "spreadsheet": "Parameters", "feature_count": 2,
                                  "parameter_count": 3},
                        "classification": {"class": "prismatic"},
                        "units": None, "deviation": [], "registration": None,
                        "measured_mask": {"scan_excluded": 4, "model_excluded": 0,
                                          "regions": 0, "note": ""}})
        self.assertIn("MEASURED-ONLY DEVIATION: 4 synthetic scan vertex/vertices", text)
        self.assertIn("measured regions only", text)


class PlanPreviewTest(ReconToolCase):
    """P2.3/P2.4: mode='plan' previews the FeaturePlan (validated, nothing
    built); plan_override builds the agent's amended plan through the same
    builder, honesty caps and deviation rows."""

    def test_plan_mode_previews_without_building(self):
        doc = self.doc_with_mesh("RPlan")
        before = sorted(o.Name for o in doc.Objects)
        text, _img, is_error = self.run_tool(doc_name=doc.Name, mode="plan")
        self.assertFalse(is_error, text)
        self.assertIn("RESULT CLASS: PLAN PREVIEW", text)
        self.assertIn("feature(s)", text)
        self.assertEqual(sorted(o.Name for o in doc.Objects), before,
                         "a preview creates nothing")
        self.assertNotIn("Body", [o.TypeId for o in doc.Objects])

    def test_plan_override_builds_the_amended_plan(self):
        import json

        doc = self.doc_with_mesh("ROver")
        text, _img, is_error = self.run_tool(doc_name=doc.Name, mode="plan")
        self.assertFalse(is_error, text)
        # amend: thicken the plate by editing the plan the preview returned —
        # fetch it through the engine (the render is text; the plan rides in
        # the analysis payload path, so rebuild it via reconstruct.preview)
        from oscar.recon import reconstruct

        result = reconstruct.reconstruct(doc, doc.getObject("Scan").Mesh,
                                         warn_mm=0.5, preview=True)
        self.assertTrue(result["preview"])
        self.assertTrue(result["validation"]["ok"], result["validation"])
        plan = result["plan"]
        plan["parameters"]["plate_t"]["value"] = 12.0     # the amendment
        text, _img, is_error = self.run_tool(doc_name=doc.Name,
                                             plan_override=plan)
        self.assertFalse(is_error, text)
        self.assertIn("agent_amended_plan", text)
        body = next(o for o in doc.Objects if o.TypeId == "PartDesign::Body")
        bb = body.Shape.BoundBox
        self.assertAlmostEqual(bb.ZLength, 12.0, delta=0.05,
                               msg="the amended thickness drives the build")
        sheet = doc.getObject("Parameters")
        self.assertAlmostEqual(float(sheet.get("plate_t")), 12.0, delta=1e-6)

    def test_the_survey_cache_reuses_the_recognition(self):
        from oscar.recon import reconstruct

        doc = self.doc_with_mesh("RCache")
        mesh = doc.getObject("Scan").Mesh
        r1 = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
        self.assertTrue(r1["build"]["success"], r1["build"])
        key = next(k for k in reconstruct._SURVEY_CACHE
                   if k[1:] and r1["survey"] is reconstruct._SURVEY_CACHE[k]["survey"])
        r2 = reconstruct.reconstruct(doc, mesh, warn_mm=0.5)
        self.assertTrue(r2["build"]["success"], r2["build"])
        self.assertIs(r2["survey"], reconstruct._SURVEY_CACHE[key]["survey"],
                      "the second call reuses the cached survey")


if __name__ == "__main__":
    unittest.main()


class UnitsAskGateTest(ReconToolCase):
    """§16 units ask wired into reconstruct_mesh: a gate-ambiguous mesh with a live card surface
    asks once through the ask_user seam, the answer becomes the durable OscarSourceUnit, and the
    same execution continues (no second run)."""

    def test_a_gate_ambiguous_mesh_asks_through_the_card_and_applies_the_answer(self):
        from test_recon_units import box_mesh

        from oscar import tools_ask

        doc = self.doc_with_mesh("UnitsAskT", box_mesh(50.8, 25.4, 12.7))
        cards = []

        def fake_ask(**card):
            cards.append(card)
            index = next(i for i, o in enumerate(card["options"]) if o["label"].startswith("inch"))
            return {"answer": "inch", "index": index}

        surface_keep = tools_ask.surface
        tools_ask.surface = lambda request: None          # the panel is mounted
        try:
            with mock.patch.object(tools_ask, "ask_user", side_effect=fake_ask):
                text, _img, is_error = self.run_tool(
                    ToolContext(conversation_id="chat-1", turn_id="turn-1"),
                    doc_name=doc.Name, mode="parametric")
        finally:
            tools_ask.surface = surface_keep
        self.assertFalse(is_error, text)
        self.assertEqual(len(cards), 1, "exactly one card, no re-ask")
        self.assertTrue(any(o.get("likeliest") for o in cards[0]["options"]), cards[0])
        self.assertEqual(doc.getObject("Scan").OscarSourceUnit, "inch")

    def test_headless_stays_silent_and_honest(self):
        from test_recon_units import box_mesh

        from oscar import tools_ask

        doc = self.doc_with_mesh("UnitsAskHeadless", box_mesh(50.8, 25.4, 12.7))
        surface_keep = tools_ask.surface
        tools_ask.surface = None                          # no panel: no card may appear
        try:
            with mock.patch.object(tools_ask, "ask_user",
                                   side_effect=AssertionError("no surface, no ask")):
                text, _img, is_error = self.run_tool(ToolContext(), doc_name=doc.Name,
                                                     mode="parametric")
        finally:
            tools_ask.surface = surface_keep
        self.assertFalse(is_error, text)
        self.assertNotIn("OscarSourceUnit", doc.getObject("Scan").PropertiesList)


class InterpretationAskGateTest(ReconToolCase):
    """§16 ambiguous-interpretation ask wired into reconstruct_mesh: the classification probe
    feeds the card, the user's accepted alternative steers THIS run (prefer_class), and headless
    never asks. The probe seam is faked here; its real classification path is covered by
    test_recon_classify."""

    AMBIGUOUS_ROTATIONAL = {"class": "rotational", "confidence": 0.45,
                            "alternatives": ["prismatic_extrusion"]}

    def test_accepting_the_alternative_steer_prefer_class_into_reconstruct(self):
        from test_recon_units import box_mesh

        from oscar import tools_ask, tools_recon
        from oscar.recon import reconstruct as _recon

        doc = self.doc_with_mesh("InterpAskT", box_mesh(50.0, 25.0, 12.0))   # mm grid: no units card
        cards = []

        def fake_ask(**card):
            cards.append(card)
            index = next(i for i, o in enumerate(card["options"]) if o["label"].startswith("Accept"))
            return {"answer": "Accept the alternative", "index": index}

        seen = {}

        def recording_reconstruct(*args, **kwargs):
            seen["prefer_class"] = kwargs.get("prefer_class")
            return _real_reconstruct(*args, **kwargs)

        _real_reconstruct = _recon.reconstruct
        surface_keep = tools_ask.surface
        tools_ask.surface = lambda request: None
        try:
            with mock.patch.object(tools_ask, "ask_user", side_effect=fake_ask), \
                    mock.patch.object(tools_recon, "_classification_probe",
                                      return_value=dict(self.AMBIGUOUS_ROTATIONAL)), \
                    mock.patch.object(_recon, "reconstruct", side_effect=recording_reconstruct):
                text, _img, is_error = self.run_tool(
                    ToolContext(conversation_id="chat-1", turn_id="turn-1"),
                    doc_name=doc.Name, mode="parametric")
        finally:
            tools_ask.surface = surface_keep
        self.assertFalse(is_error, text)
        self.assertEqual(len(cards), 1, "exactly one card")
        self.assertTrue(any(o.get("likeliest") for o in cards[0]["options"]), cards[0])
        self.assertEqual(seen.get("prefer_class"), "prismatic")   # the accepted alternative

    def test_headless_probes_nothing(self):
        from test_recon_units import box_mesh

        from oscar import tools_ask, tools_recon

        doc = self.doc_with_mesh("InterpAskHeadless", box_mesh(50.8, 25.4, 12.7))
        surface_keep = tools_ask.surface
        tools_ask.surface = None
        try:
            with mock.patch.object(tools_recon, "_classification_probe",
                                   side_effect=AssertionError("no surface, no probe")):
                text, _img, is_error = self.run_tool(ToolContext(), doc_name=doc.Name,
                                                     mode="parametric")
        finally:
            tools_ask.surface = surface_keep
        self.assertFalse(is_error, text)
