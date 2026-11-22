# SPDX-License-Identifier: Apache-2.0
"""The §16 ↔ ask_user bridge: question cards, structured answers, and the units decision
round-trip with reuse (oscar.recon.askwire). Pure python — the ``ask`` seam is a lambda."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

from oscar import askuser  # noqa: E402
from oscar.recon import askwire  # noqa: E402
from oscar.recon.approve import AnswerStore  # noqa: E402

MISMATCH = {"status": "POSSIBLE_MM_INCH_MISMATCH", "dims_source": [185.0, 12.0, 200.0],
            "source_unit_note": "STL stores no unit metadata", "evidence": "inch grid exact"}
AMBIGUOUS = {"status": "SCALE_AMBIGUOUS", "dims_source": [10.0, 10.0, 10.0]}


def store():
    return AnswerStore(os.path.join(isolation.fresh_dir(), "a"), "proj-1")


class CardTest(unittest.TestCase):
    def test_a_units_question_becomes_a_valid_card_with_the_recommended_likeliest(self):
        from oscar.recon.approve import approval_triggers, build_question

        trigger = approval_triggers({"units": MISMATCH})[0]
        card = askwire.card_from_question(build_question(trigger))
        normalized, err = askuser.validate_question(card)
        self.assertEqual(err, card and "")
        labels = [o["label"] for o in normalized["options"]]
        self.assertEqual([o["likeliest"] for o in normalized["options"]], [False, False, True],
                         labels)                                          # [mm, cm, INCH]: inch recommended
        self.assertTrue(normalized["allow_free_text"])
        self.assertIn("inch", normalized["subtitle"] + normalized["title"])

    def test_a_worn_feature_question_marks_the_proposal_likeliest(self):
        question = {"title": "Confirm hole_d1", "reason": "worn/out-of-round feature",
                    "measured": 7.92, "proposal": 8.0, "tolerance": 0.1, "confidence": 0.55,
                    "options": [{"key": "accept", "label": "Accept the proposal (8)"},
                                {"key": "retain_measured", "label": "Retain the measured value (7.92)"},
                                {"key": "other_value", "label": "Provide another value"}]}
        card = askwire.card_from_question(question)
        normalized, err = askuser.validate_question(card)
        self.assertEqual(err, "")
        self.assertTrue(normalized["options"][0]["likeliest"])
        self.assertIn("8", normalized["options"][0]["label"])
        self.assertEqual(len(normalized["options"]), 2)                  # other → free text

    def test_structured_answers_map_back_to_choices(self):
        question = {"options": [{"key": "mm", "label": "mm"}, {"key": "inch", "label": "inch"},
                                {"key": "other_value", "label": "other"}]}
        self.assertEqual(askwire.answer_from_structured({"answer": "mm", "index": 0}, question),
                         {"choice": "mm", "value": None})
        self.assertEqual(askwire.answer_from_structured({"answer": "Metres", "index": -1, "value": "m"},
                                                        question),
                         {"choice": "other_value", "value": "m"})


class ResolveUnitsTest(unittest.TestCase):
    def test_a_gate_free_report_never_asks(self):
        unit, outcome = askwire.resolve_units({"status": "UNIT_KNOWN"}, ask=lambda c: (_ for _ in ()).throw(
            AssertionError("must not ask")), store=store())
        self.assertEqual((unit, outcome["decision"]), (None, "not_required"))

    def test_the_card_is_asked_the_answer_applied_and_then_reused_without_asking(self):
        s = store()
        asked = []

        def ask(card):
            asked.append(card)
            index = next(i for i, o in enumerate(card["options"]) if o["label"].startswith("inch"))
            return {"answer": "inch", "index": index}

        unit, outcome = askwire.resolve_units(MISMATCH, ask=ask, store=s)
        self.assertEqual(unit, "inch")
        self.assertEqual(outcome["decision"], "answered")
        self.assertEqual(len(asked), 1)
        recorded = s.lookup("units")
        self.assertEqual(recorded["choice"], "inch")

        unit2, outcome2 = askwire.resolve_units(MISMATCH, ask=lambda c: (_ for _ in ()).throw(
            AssertionError("must reuse, not re-ask")), store=s)
        self.assertEqual((unit2, outcome2["decision"], outcome2["reused"]), ("inch", "answered", True))

    def test_a_typed_answer_becomes_the_unit_value(self):
        unit, outcome = askwire.resolve_units(AMBIGUOUS, store=store(),
                                              ask=lambda card: {"answer": "centimetres", "index": -1,
                                                                "value": "cm"})
        self.assertEqual(unit, "cm")
        self.assertEqual(outcome["decision"], "answered")

    def test_headless_is_the_honest_denial(self):
        unit, outcome = askwire.resolve_units(MISMATCH, ask=None, store=store())
        self.assertIsNone(unit)
        self.assertEqual(outcome["decision"], "needs_input")

    def test_a_failing_card_denies_rather_than_falls_back(self):
        def boom(card):
            raise RuntimeError("no surface")

        unit, outcome = askwire.resolve_units(MISMATCH, ask=boom, store=store())
        self.assertIsNone(unit)
        self.assertEqual(outcome["decision"], "ask_failed")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


AMBIGUOUS_CLASSIFICATION = {"class": "organic_or_hybrid", "confidence": 0.0,
                            "alternatives": ["prismatic_extrusion"],
                            "evidence": {"plane_share": 0.51, "rotational_share": 0.03}}
CLEAR_CLASSIFICATION = {"class": "prismatic", "confidence": 0.85}


class ResolveInterpretationTest(unittest.TestCase):
    def test_a_clear_winner_never_asks(self):
        unit, outcome = askwire.resolve_interpretation(
            CLEAR_CLASSIFICATION, ask=lambda c: (_ for _ in ()).throw(AssertionError("must not ask")),
            store=store())
        self.assertEqual((unit, outcome["decision"]), (None, "not_required"))

    def test_the_measured_winner_is_the_likeliest_card_option(self):
        cards = []

        def ask(card):
            cards.append(card)
            keep = next(i for i, o in enumerate(card["options"])
                        if "Retain" in o["label"] or "organic" in o["label"])
            return {"answer": card["options"][keep]["label"], "index": keep}

        preferred, outcome = askwire.resolve_interpretation(AMBIGUOUS_CLASSIFICATION, ask=ask, store=store())
        self.assertEqual(outcome["decision"], "answered")
        self.assertIsNone(preferred)                       # retain: measured winner stands
        normalized, err = askuser.validate_question(cards[0])
        self.assertEqual(err, "")
        marked = [o["label"] for o in normalized["options"] if o["likeliest"]]
        self.assertEqual(len(marked), 1)
        self.assertIn("organic_or_hybrid", marked[0])      # the evidence winner is the default

    def test_accepting_the_alternative_switches_the_route_for_this_run(self):
        def ask(card):
            i = next(i for i, o in enumerate(card["options"])
                     if o["label"].startswith("Accept"))
            return {"answer": "accept", "index": i}

        preferred, _outcome = askwire.resolve_interpretation(AMBIGUOUS_CLASSIFICATION, ask=ask, store=store())
        self.assertEqual(preferred, "prismatic")           # prismatic_extrusion -> prismatic

    def test_the_answer_is_recorded_and_then_reused_without_asking(self):
        s = store()

        def accept(card):
            i = next(i for i, o in enumerate(card["options"]) if o["label"].startswith("Accept"))
            return {"answer": "Accept", "index": i}

        askwire.resolve_interpretation(AMBIGUOUS_CLASSIFICATION, ask=accept, store=s)
        preferred, outcome = askwire.resolve_interpretation(
            AMBIGUOUS_CLASSIFICATION, ask=lambda c: (_ for _ in ()).throw(AssertionError("reuse, do not ask")),
            store=s)
        self.assertEqual((preferred, outcome["decision"], outcome["reused"]),
                         ("prismatic", "answered", True))

    def test_headless_and_failed_cards_never_switch_the_route(self):
        preferred, outcome = askwire.resolve_interpretation(AMBIGUOUS_CLASSIFICATION, ask=None, store=store())
        self.assertIsNone(preferred)
        self.assertEqual(outcome["decision"], "needs_input")

        def boom(card):
            raise RuntimeError("no surface")

        preferred, outcome = askwire.resolve_interpretation(AMBIGUOUS_CLASSIFICATION, ask=boom, store=store())
        self.assertIsNone(preferred)
        self.assertEqual(outcome["decision"], "ask_failed")
