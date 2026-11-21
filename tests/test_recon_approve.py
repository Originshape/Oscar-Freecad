"""§16 user-approval contract (spec §21.2 U1–U8b, headless subset) + the §15.1 F–K block.

``oscar.recon.approve`` is pure (no FreeCAD) — those tests run anywhere. The F–K sheet
tests open a real Spreadsheet document (the ONLY FreeCAD usage in this module).
netguard.ask_user returns NO_HUMAN in console sessions and callers must refuse, so the
U-tests stub the payload/recorder level exactly as the spec prescribes; real dialogs
stay with the per-milestone GUI checklist.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402

from oscar.recon import approve, param_table  # noqa: E402


def sufficient_state():
    """U1's reliable-dimension state: every D11 rule passes — units known, scan ready,
    holes established, classification confident, deltas inside tolerance."""
    return {
        "units": {"status": "UNIT_KNOWN", "dims_source": [60.0, 40.0, 10.0],
                  "evidence": ["source unit 'mm' from file metadata"]},
        "scan": {"readiness": {"verdict": "ready", "reasons": [], "notes": []}},
        "holes": [{"id": "hole_1", "parameter": "hole_dp1", "through": True,
                   "diameter": 6.6, "confidence": "high"}],
        "classification": {"class": "prismatic", "confidence": 0.92,
                           "alternatives": ["rotational"],
                           "evidence": {"plane_share": 0.78}},
        "datums": {"ambiguities": []},
        "routes": [{"name": "prismatic", "native_parametric": True,
                    "coverage_estimate": 0.95}],
        "deltas": [{"parameter": "plate_len", "measured": 60.02, "proposed": 60.0,
                    "tolerance": 0.2, "confidence": 0.95, "basis": "measured",
                    "critical": True, "evidence": ["bbox u length"]}],
    }


def worn_bore_delta():
    """U4's out-of-round 6.55 mm bore: an inferred standard-size proposal (6.6)."""
    return {"parameter": "hole_d1", "measured": 6.55, "proposed": 6.6,
            "tolerance": 0.2, "confidence": 0.55, "basis": "inferred",
            "wear": {"kind": "out_of_round", "magnitude_mm": 0.12},
            "evidence": ["bore wall fit rms 0.09 mm; profile out-of-round 0.12 mm"]}


def route_trigger():
    """A route question with a faceted fallback beside the parametric route."""
    state = sufficient_state()
    state["routes"] = [{"name": "prismatic", "native_parametric": True,
                        "coverage_estimate": 0.6},
                       {"name": "faceted", "native_parametric": False,
                        "coverage_estimate": 1.0}]
    triggers = approve.approval_triggers(state)
    return triggers[0]


class U1NoQuestionTest(unittest.TestCase):
    def test_reliable_dimensions_proceed_without_any_trigger(self):
        self.assertEqual(approve.approval_triggers(sufficient_state()), [])
        # and a plain measured delta produces a factual row, not a question
        row = approve.row_from_delta({"parameter": "plate_len", "measured": 60.02,
                                      "tolerance": 0.2, "confidence": 0.95})
        self.assertEqual((row["basis"], row["status"]), ("measured", "proposed"))


class U2FocusedQuestionTest(unittest.TestCase):
    def state(self):
        state = sufficient_state()
        state["holes"] = [{"id": "hole_1", "parameter": "hole_dp1", "through": False,
                           "diameter": 6.6, "depth": 8.2, "tolerance": 0.15,
                           "confidence": "medium",
                           "depth_basis": "cylinder wall axial extent (no floor plane seen)"}]
        state["scan"] = {"readiness": {"verdict": "degraded", "reasons": [
            "1 hole opening(s) with NO scanned wall (0 through-paired)"], "notes": []}}
        return state

    def test_near_through_blind_hole_yields_exactly_one_focused_question(self):
        triggers = approve.approval_triggers(self.state())
        self.assertEqual(len(triggers), 1, triggers)
        trigger = triggers[0]
        self.assertEqual(trigger["rule"], "through_blind_uncertain")
        self.assertEqual(trigger["parameter"], "hole_dp1")
        self.assertEqual(trigger["threshold"],
                         "a floor plane closing the bore (blind) or a second exit face (through)")
        self.assertIn("no floor plane seen", trigger["actual"])
        self.assertTrue(any("NO scanned wall" in e for e in trigger["evidence"]),
                        trigger["evidence"])
        payload = approve.build_question(trigger)
        for key in ("measured", "proposal", "tolerance", "confidence", "reason",
                    "evidence", "options", "deviation"):
            self.assertIn(key, payload)
        self.assertEqual(payload["measured"]["depth"], 8.2)
        self.assertEqual(payload["proposal"], {"kind": "blind", "depth": 8.2})
        self.assertEqual(payload["tolerance"], 0.15)
        self.assertEqual(payload["confidence"], 0.6)
        self.assertIn("cannot be determined", payload["reason"])
        keys = [o["key"] for o in payload["options"]]
        self.assertEqual(keys, ["accept", "retain_measured", "other_value"])

    def test_accept_retain_other_each_produce_the_correct_outcome(self):
        row = {"parameter": "hole_dp1", "measured": 8.2, "proposal": 8.2,
               "tolerance": 0.15, "confidence": 0.6}
        accept = approve.answer_outcome(row, {"choice": "accept"})
        self.assertEqual((accept["value"], accept["basis"], accept["status"]),
                         (8.2, "user", "approved"))
        self.assertEqual(accept["proposal_disposition"], "accepted")
        retain = approve.answer_outcome(row, {"choice": "retain_measured"})
        self.assertEqual((retain["value"], retain["basis"], retain["status"]),
                         (8.2, "user", "approved"))
        self.assertEqual(retain["proposal_disposition"], "rejected")
        other = approve.answer_outcome(row, {"choice": "other_value", "value": "through"})
        self.assertEqual((other["value"], other["basis"], other["status"]),
                         ("through", "user", "approved"))
        self.assertIsNone(other["deviation"])
        with self.assertRaises(ValueError):
            approve.answer_outcome(row, {"choice": "other_value"})       # no value given
        with self.assertRaises(ValueError):
            approve.answer_outcome({"parameter": "x"}, {"choice": "accept"})  # nothing to accept
        with self.assertRaises(ValueError):
            approve.answer_outcome(row, {"choice": "maybe"})


class U3UnitQuestionTest(unittest.TestCase):
    def trigger(self):
        state = sufficient_state()
        state["units"] = {"status": "POSSIBLE_MM_INCH_MISMATCH",
                          "dims_source": [190.5, 12.7, 50.8],
                          "source_unit_note": "STL stores no unit metadata",
                          "evidence": ["dimensions sit exactly on the inch fraction grid"]}
        triggers = approve.approval_triggers(state)
        self.assertEqual([t["rule"] for t in triggers], ["units_undeterminable"])
        return triggers[0]

    def test_unit_ambiguity_asks_with_normalized_interpretations_and_recommendation(self):
        payload = approve.build_question(self.trigger())
        self.assertEqual(payload["parameter"], "units")
        readings = {i["unit"]: i["dims"] for i in payload["interpretations"]}
        self.assertEqual(readings["mm"], [190.5, 12.7, 50.8])
        self.assertEqual(readings["cm"], [19.05, 1.27, 5.08])
        self.assertEqual(readings["inch"], [7.5, 0.5, 2.0])
        self.assertIn("inch", payload["recommendation"].lower())
        self.assertEqual([o["key"] for o in payload["options"]],
                         ["mm", "cm", "inch", "other_value"])

    def test_confirmed_unit_persisted_and_applied(self):
        store = approve.AnswerStore(tempfile.mkdtemp(prefix="u3_"), "proj")
        trigger = self.trigger()
        store.record("units", {"choice": "inch", "value": "inch"}, trigger)
        result = approve.decide(trigger, approve.AnswerStore(store.directory, "proj"),
                                human_available=False)
        self.assertEqual(result["decision"], "answered")          # never re-asked
        self.assertEqual(result["answer"]["value"], "inch")
        self.assertEqual(approve.unit_factor("inch"), 25.4)
        self.assertEqual(approve.unit_factor("mm"), 1.0)
        self.assertEqual(approve.unit_factor("furlongs"), None)


class U4WornBoreTest(unittest.TestCase):
    def trigger(self):
        state = sufficient_state()
        state["deltas"] = [worn_bore_delta()]
        triggers = approve.approval_triggers(state)
        self.assertEqual([t["rule"] for t in triggers], ["worn_feature"])  # within tolerance
        return triggers[0]

    def test_worn_bore_proposes_standard_size_with_deviation_never_silently_applied(self):
        trigger = self.trigger()
        self.assertEqual(trigger["parameter"], "hole_d1")
        self.assertEqual(trigger["measured"], 6.55)
        self.assertEqual(trigger["proposal"], 6.6)
        payload = approve.build_question(trigger)
        self.assertEqual(payload["deviation"], 0.05)
        self.assertEqual(payload["proposal_basis"], "inferred")   # never presented as measured
        self.assertIn("out_of_round", " ".join(payload["evidence"]))
        # the pre-answer row: basis inferred, status proposed — NOT approved, NOT applied
        row = approve.row_from_delta(worn_bore_delta())
        self.assertEqual((row["measured"], row["deviation"], row["basis"], row["status"]),
                         (6.55, 0.05, "inferred", "proposed"))

    def test_retain_measured_builds_with_6_55(self):
        row = {"parameter": "hole_d1", "measured": 6.55, "proposal": 6.6,
               "tolerance": 0.2, "confidence": 0.55}
        outcome = approve.answer_outcome(row, {"choice": "retain_measured"})
        self.assertEqual(outcome["value"], 6.55)
        self.assertEqual(outcome["deviation"], 0.0)
        self.assertEqual((outcome["basis"], outcome["status"]), ("user", "approved"))
        self.assertEqual(outcome["proposal_disposition"], "rejected")


class U5PersistenceTest(unittest.TestCase):
    def test_answer_survives_a_new_store_and_is_reused_without_asking(self):
        directory = tempfile.mkdtemp(prefix="u5_")
        state = sufficient_state()
        state["deltas"] = [dict(worn_bore_delta(), tolerance=0.04)]   # also out-of-tolerance
        triggers = approve.approval_triggers(state)
        trigger = next(t for t in triggers if t["parameter"] == "hole_d1")
        store = approve.AnswerStore(directory, "proj-1")
        store.record("hole_d1", {"choice": "accept", "value": 6.6}, trigger)
        # a FRESH instance on the same directory/project: persisted, not memory
        reopened = approve.AnswerStore(directory, "proj-1")
        self.assertTrue(reopened.has_answer("hole_d1"))
        answer = reopened.lookup("hole_d1")
        self.assertEqual((answer["choice"], answer["value"]), ("accept", 6.6))
        # headless decide with the recorded answer: answered, reused, no question
        result = approve.decide(trigger, reopened, human_available=False)
        self.assertEqual(result["decision"], "answered")
        self.assertTrue(result["reused"])
        self.assertNotIn("question", result)
        # the trigger record was logged with the answer (spec §16.3)
        self.assertEqual(reopened.lookup("hole_d1")["rule"], "worn_feature")


class U6IsolationTest(unittest.TestCase):
    def test_no_repeated_questions_for_a_confirmed_parameter(self):
        store = approve.AnswerStore(tempfile.mkdtemp(prefix="u6_"), "proj-1")
        state = sufficient_state()
        state["deltas"] = [dict(worn_bore_delta(), tolerance=0.04)]  # worn AND out-of-tolerance
        triggers = approve.approval_triggers(state)
        self.assertEqual(len(triggers), 2, [t["rule"] for t in triggers])
        # one QUESTION per parameter even when two rules fail it
        self.assertEqual(len(approve.pending_questions(triggers)), 1)
        store.record("hole_d1", {"choice": "retain_measured", "value": 6.55}, triggers[0])
        for trigger in triggers:                      # multi-step flow, same parameter
            result = approve.decide(trigger, store, human_available=True)
            self.assertEqual(result["decision"], "answered")
            self.assertTrue(result["reused"])

    def test_answers_never_leak_across_projects(self):
        directory = tempfile.mkdtemp(prefix="u6x_")
        store_a = approve.AnswerStore(directory, "proj-A")
        store_a.record("hole_d1", {"choice": "accept", "value": 6.6})
        store_b = approve.AnswerStore(directory, "proj-B")
        self.assertFalse(store_b.has_answer("hole_d1"))
        self.assertIsNone(store_b.lookup("hole_d1"))
        state = sufficient_state()
        state["deltas"] = [worn_bore_delta()]
        trigger = approve.approval_triggers(state)[0]
        self.assertEqual(approve.decide(trigger, store_b, human_available=False)
                         ["decision"], "needs_input")     # B must be asked itself
        # and the two projects use separate files
        self.assertNotEqual(store_a.path, store_b.path)


class U7DeferTest(unittest.TestCase):
    def test_unknown_basis_defers_and_never_invents(self):
        state = sufficient_state()
        state["deltas"] = [{"parameter": "hole_dp2", "basis": "none",
                            "tolerance": 0.2, "confidence": None,
                            "evidence": ["hole opening with NO scanned wall"]}]
        self.assertEqual(approve.approval_triggers(state), [])   # nothing to propose → no question
        row = approve.row_from_delta(state["deltas"][0])
        self.assertEqual((row["basis"], row["status"]), ("none", "deferred"))
        self.assertIsNone(row["measured"])
        self.assertIsNone(row["deviation"])
        # a user MAY still supply the value later — then it is basis user, approved
        outcome = approve.answer_outcome({"parameter": "hole_dp2", "measured": None},
                                         {"choice": "other_value", "value": 9.0})
        self.assertEqual((outcome["value"], outcome["basis"], outcome["status"]),
                         (9.0, "user", "approved"))


class U8aNoHumanTest(unittest.TestCase):
    def test_no_human_denies_with_needs_input_and_never_authorizes_fallback(self):
        trigger = route_trigger()
        self.assertEqual(trigger["rule"], "materially_different_routes")
        self.assertEqual(trigger["authorization_scope"], "fallback_route")
        store = approve.AnswerStore(tempfile.mkdtemp(prefix="u8a_"), "proj-1")
        result = approve.decide(trigger, store, human_available=False)
        self.assertEqual(result["decision"], "needs_input")
        self.assertEqual(result["reason"], "no_human")
        self.assertFalse(result["authorizes_fallback"])
        self.assertFalse(result["authorizes_repair"])
        self.assertIn("question", result)                      # identical payload, per §16.2
        self.assertIn("never", result["note"])
        # nothing was recorded by a denial
        self.assertFalse(store.has_answer("route"))


class U8bAuthorizedNotUpgradedTest(unittest.TestCase):
    def test_recorded_approval_authorizes_but_never_upgrades_the_outcome(self):
        store = approve.AnswerStore(tempfile.mkdtemp(prefix="u8b_"), "proj-1")
        trigger = route_trigger()
        store.record("route", {"choice": "accept", "value": "faceted"}, trigger)
        result = approve.decide(trigger, store, human_available=False)
        self.assertEqual(result["decision"], "answered")
        self.assertTrue(result["authorized"])
        self.assertTrue(result["authorization_only"])
        # the ceiling stays the fallback ceiling — approval is authorization only (§18.3)
        self.assertEqual(result["outcome_ceiling"], "BEST AVAILABLE APPROXIMATION")
        self.assertIn("unchanged", result["note"])
        # retaining the parametric route authorizes nothing
        store.record("route", {"choice": "retain_measured"}, trigger)
        result2 = approve.decide(trigger, store)
        self.assertNotIn("authorized", result2)


class TriggerMatrixTest(unittest.TestCase):
    def test_classification_ambiguity_datums_tolerance_and_confidence_rules(self):
        state = sufficient_state()
        state["classification"] = {"class": "prismatic", "confidence": 0.35,
                                   "alternatives": ["rotational"],
                                   "evidence": {"plane_share": 0.52}}
        state["datums"] = {"ambiguities": ["primary: two equal-area planes"]}
        state["deltas"] = [
            {"parameter": "plate_t", "measured": 10.0, "proposed": 10.6,
             "tolerance": 0.2, "confidence": 0.9, "basis": "inferred"},
            {"parameter": "plate_len", "measured": 60.02, "proposed": 60.0,
             "tolerance": 0.2, "confidence": 0.3, "basis": "measured", "critical": True},
        ]
        rules = [t["rule"] for t in approve.approval_triggers(state)]
        self.assertEqual(rules, ["ambiguous_interpretation",
                                 "base_direction_symmetry_ambiguous",
                                 "out_of_tolerance_proposal",
                                 "low_confidence_critical"])
        state["classification"]["confidence"] = 0.9      # confident → that trigger clears
        state["datums"]["ambiguities"] = []
        state["deltas"][0]["proposed"] = 10.1
        state["deltas"][1]["confidence"] = 0.95
        self.assertEqual(approve.approval_triggers(state), [])

    def test_routes_equivalent_do_not_ask(self):
        state = sufficient_state()
        state["routes"] = [{"name": "prismatic", "native_parametric": True,
                            "coverage_estimate": 0.9},
                           {"name": "extruded", "native_parametric": True,
                            "coverage_estimate": 0.88}]
        self.assertEqual(approve.approval_triggers(state), [])


class FKSheetTest(unittest.TestCase):
    """§15.1: the F–K block on a real spreadsheet — A–E untouched, idempotent."""

    def setUp(self):
        self.doc = App.newDocument("ApproveFK")
        self.sheet = self.doc.addObject("Spreadsheet::Sheet", "Parameters")
        rows = (("plate_t", "=10.0", "scan_fit", "opposed-plane thickness sampling"),
                ("hole_d1", "=6.6", "scan_fit", "bore wall cylinder fit"),
                ("hole_dp1", "=8.2", "scan_fit", "cylinder wall axial extent"))
        for i, (name, value, kind, ref) in enumerate(rows, start=1):
            self.sheet.set(f"A{i}", name)
            self.sheet.set(f"B{i}", value)
            self.sheet.setAlias(f"B{i}", name)
            self.sheet.set(f"D{i}", kind)
            self.sheet.set(f"E{i}", ref)
        self.doc.recompute()
        self.addCleanup(App.closeDocument, self.doc.Name)

    def _text(self, cell):
        return str(self.sheet.getContents(cell)).lstrip("'")

    def test_writes_exactly_the_fk_block_and_never_touches_a_to_e(self):
        before = {c: self._text(f"{c}{r}") for r in (1, 2, 3) for c in "ABCDE"}
        rows = [approve.row_from_delta(worn_bore_delta()),                     # J inferred K proposed
                {"parameter": "plate_t", "measured": 10.02, "deviation": 0.02,
                 "tolerance": 0.2, "confidence": 0.93, "basis": "measured",
                 "status": "approved"},
                {"parameter": "hole_dp1", "tolerance": 0.15, "basis": "none",
                 "status": "deferred"}]                                        # basis none → defer
        report = param_table.append_fk_rows(self.sheet, rows)
        self.assertEqual(sorted(report["written"]), ["hole_d1", "hole_dp1", "plate_t"])
        self.assertEqual(report["missing"], [])
        self.assertAlmostEqual(float(self._text("F2")), 6.55, places=6)
        self.assertAlmostEqual(float(self._text("G2")), 0.05, places=6)
        self.assertAlmostEqual(float(self._text("H2")), 0.2, places=6)
        self.assertAlmostEqual(float(self._text("I2")), 0.55, places=6)
        self.assertEqual(self._text("J2"), "inferred")
        self.assertEqual(self._text("K2"), "proposed")
        self.assertEqual(self._text("J1"), "measured")
        self.assertEqual(self._text("K1"), "approved")
        self.assertEqual(self._text("J3"), "none")       # basis none → defer, never invent
        self.assertEqual(self._text("K3"), "deferred")
        after = {c: self._text(f"{c}{r}") for r in (1, 2, 3) for c in "ABCDE"}
        self.assertEqual(before, after)

    def test_idempotent_rerite_changes_nothing(self):
        rows = [approve.row_from_delta(worn_bore_delta())]
        first = param_table.append_fk_rows(self.sheet, rows)
        self.assertEqual(first["written"], {"hole_d1": ["F", "G", "H", "I", "J", "K"]})
        second = param_table.append_fk_rows(self.sheet, rows)
        self.assertEqual(second["written"], {})
        self.assertEqual(second["unchanged"], ["hole_d1"])

    def test_bad_enums_rejected_before_anything_is_written(self):
        with self.assertRaises(ValueError) as caught:
            param_table.append_fk_rows(self.sheet, [
                {"parameter": "plate_t", "basis": "guessed", "status": "approved"}])
        self.assertIn("guessed", str(caught.exception))
        with self.assertRaises(ValueError):
            param_table.append_fk_rows(self.sheet, [
                {"parameter": "plate_t", "basis": "measured", "status": "maybe"}])
        with self.assertRaises(ValueError):
            param_table.append_fk_rows(self.sheet, [
                {"parameter": "plate_t", "basis": "measured", "status": "approved",
                 "measured": "thick"}])
        for column in "FGHIJK":                              # nothing was half-written
            self.assertEqual(self._text(f"{column}1"), "")

    def test_rows_without_alias_are_reported_missing(self):
        report = param_table.append_fk_rows(self.sheet, [
            {"parameter": "nonexistent", "basis": "measured", "status": "proposed",
             "measured": 1.0}])
        self.assertEqual(report["missing"], ["nonexistent"])
        self.assertEqual(report["written"], {})


if __name__ == "__main__":
    unittest.main()
