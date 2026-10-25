"""The intent table: what a typed request turns into, with the phrasings real users type.

``classify_intent`` is deterministic and model-free, so the table is the specification: polite requests, go-aheads,
non-English requests, follow-up edits, questions that merely mention a part, deliver / inspect / view requests. A miss
costs only gate strictness (the turn is then escalated to ``generic_part`` on its first successful geometry call), and
the documented misses are pinned as such. The second half runs the real loop to show what each class of request does.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.flow import INTENTS, WORK, Ledger, classify_intent, extract_requirements, is_continuation, needs_reference  # noqa: E402

# (typed text, intent)
TABLE = [
    # direct and polite build requests
    ("Build a housing for a Raspberry Pi 4 with a snap-fit lid in ABS", "enclosure"),
    ("Build me a Raspberry Pi 4 housing", "enclosure"),
    ("Can you build a snap-fit enclosure for a Raspberry Pi 4?", "enclosure"),
    ("Could you please make me a bracket for a 20mm tube", "bracket"),
    ("Please design a housing for my Arduino Uno", "enclosure"),
    ("I want a case for the ESP32 with a lid", "enclosure"),
    ("I need an L-bracket with two M4 holes", "bracket"),
    ("Would you make the wall 3 mm?", "modify"),
    ("Could you add a vent to the lid?", "generic_part"),
    ("do a snap fit enclosure for ESP32", "enclosure"),
    ("Design a fixture to hold a phone for drilling", "fixture"),
    ("I need a stand for my phone", "bracket"),
    ("make a 40x20x5 mm plate with a 6 mm hole", "generic_part"),
    ("create a cylinder 20 mm", "generic_part"),
    ("a gear with 20 teeth, module 2", "generic_part"),
    # non-English
    ("Construye una carcasa para Raspberry Pi 4", "enclosure"),
    ("caja para Arduino con tapa", "enclosure"),
    ("Erstelle bitte ein Gehäuse für den Raspberry Pi", "enclosure"),
    ("Gehäuse für Raspberry Pi 4 mit Deckel", "enclosure"),
    ("Fais une boîte pour mon Arduino", "enclosure"),
    ("корпус для raspberry pi 4", "enclosure"),
    ("树莓派4的外壳", "enclosure"),
    ("ラズベリーパイ4のケース", "enclosure"),
    ("Konstruiere bitte eine Platte mit zwei Bohrungen.", "other"),          # no keyword rule: escalated on the first geometry call
    # follow-up edits on an existing model
    ("add a fan hole on the side", "generic_part"),
    ("add a fan hole on the side of the housing", "enclosure"),
    ("add a 5mm vent slot to the lid", "generic_part"),
    ("drill a 5 mm hole in the left wall", "generic_part"),
    ("cut a slot in the side", "generic_part"),
    ("make the wall thicker", "modify"),
    ("change the wall thickness to 3 mm", "modify"),
    ("increase the clearance to 0.4", "modify"),
    ("Lid is too tight, increase the clearance", "modify"),
    ("fix the broken fillet", "modify"),
    ("make it taller", "modify"),
    ("rotate it 90 degrees", "modify"),
    ("set wall_t to 2.4 mm", "generic_part"),
    # reconcile with main: the legacy nudge treated 'redesign' / '3D print' as work words; the flow must too
    ("Please redesign the lid", "modify"),
    ("Redesign the case so it fits a Pi 5", "modify"),
    ("rework the vents to be wider", "modify"),
    ("3D print this lid", "deliver"),
    ("print the shell", "deliver"),
    # questions and chat: no work
    ("what is the volume of a 10 mm cube", "inspect"),
    ("what's in the document?", "inspect"),
    ("list all objects", "inspect"),
    ("show me the model", "inspect"),
    ("show me the bracket dimensions", "inspect"),
    ("show me three views", "other"),
    ("how do I make a fillet?", "question"),
    ("is a 1.2 mm wall enough for ABS?", "question"),
    ("Why did the lid fail?", "question"),
    ("what can you do?", "question"),
    ("what is wall_t?", "question"),
    ("hello", "other"),
    ("thanks", "other"),
    ("looks good", "other"),
    ("help", "other"),
    # deliver
    ("export it as STL", "deliver"),
    ("export the model to STEP", "deliver"),
    ("save the project", "deliver"),
    ("send me the file", "deliver"),
    # fem
    ("run a stress analysis on the bracket", "fem"),
    # documented misses (no keyword rule): never WORK, so no gate until the first geometry call escalates the turn
    ("delete the lid", "other"),
    ("thicker walls please", "other"),
    ("undo that", "other"),
    ("can you source the STEP file for the Pi 4?", "other"),
]


class ClassifyTableTest(unittest.TestCase):
    def test_every_row(self):
        for text, expected in TABLE:
            with self.subTest(text=text):
                self.assertEqual(classify_intent(text), expected)

    def test_the_table_is_large_and_covers_every_intent(self):
        self.assertGreaterEqual(len(TABLE), 30)
        self.assertEqual({intent for _t, intent in TABLE}, set(INTENTS))
        self.assertEqual(len({t for t, _i in TABLE}), len(TABLE), "duplicate phrasing in the table")

    def test_every_result_is_a_known_intent(self):
        for text, _intent in TABLE:
            self.assertIn(classify_intent(text), INTENTS)

    def test_case_whitespace_and_trailing_punctuation_do_not_change_the_class(self):
        for text in ("Build a bracket for a 20 mm tube", "make the wall thicker", "export it as STL"):
            base = classify_intent(text)
            for variant in (text.upper(), "  " + text + "   ", text + "!", text.replace(" ", "  "), "\n" + text):
                with self.subTest(variant=variant):
                    self.assertEqual(classify_intent(variant), base)

    def test_polite_frames_do_not_turn_a_build_request_into_a_question(self):
        for frame in ("Can you {}?", "Could you {}?", "Would you {}?", "Please {}", "I want you to {}", "I'd like to {}",
                      "Is it possible to {}?"):
            with self.subTest(frame=frame):
                self.assertEqual(classify_intent(frame.format("build a snap-fit enclosure for a Raspberry Pi 4")), "enclosure")

    def test_a_real_question_that_mentions_a_part_stays_a_question(self):
        for text in ("What is the best wall thickness for an ABS enclosure?", "How thick should the bracket be?",
                     "Which material is better for a snap-fit lid?", "Does a Raspberry Pi 4 need vents?"):
            with self.subTest(text=text):
                self.assertNotIn(classify_intent(text), WORK)

    def test_empty_and_odd_input_never_raises(self):
        for text in ("", "   ", "\n\n", "?", "!!!", "\x00", "a" * 10000, "ü" * 5000, None, 42):
            with self.subTest(text=str(text)[:20]):
                self.assertIn(classify_intent(text), INTENTS)         # type: ignore[arg-type]

    def test_a_huge_paste_is_classified_fast_on_the_first_4000_characters(self):
        import time

        text = "build a bracket " + "lorem ipsum, " * 20000
        started = time.monotonic()
        self.assertEqual(classify_intent(text), "bracket")
        self.assertLess(time.monotonic() - started, 1.0)

    def test_a_cad_attachment_marks_an_unknown_language_request_as_work(self):
        self.assertEqual(classify_intent("Bitte passend dazu", attachments=["/tmp/part.step"]), "generic_part")
        self.assertEqual(classify_intent("Bitte passend dazu"), "other")
        self.assertEqual(classify_intent("Bitte passend dazu", attachments=["/tmp/notes.txt"]), "other")

    def test_the_work_set_is_exactly_the_modelling_intents(self):
        self.assertEqual(WORK, {"enclosure", "fixture", "bracket", "generic_part", "modify", "fem"})
        for text, intent in TABLE:
            work_turn = Ledger.from_request(text).is_work_turn()
            if intent == "deliver":
                self.assertTrue(work_turn, text)         # not a WORK intent, but its plan has open required steps
            else:
                self.assertEqual(intent in WORK, work_turn, text)


class ViewRequestTest(unittest.TestCase):
    """'show me the bracket' asks for a picture, not for a bracket."""

    VIEW_REQUESTS = ("show me the bracket", "let me see the bracket", "can you show me the bracket?", "show me the housing",
                     "view the bracket from the top", "show the bracket in a section view", "zoom in on the bracket")

    def test_a_view_request_is_not_a_work_intent(self):
        for text in self.VIEW_REQUESTS:
            with self.subTest(text=text):
                self.assertNotIn(classify_intent(text), WORK)

    def test_a_view_request_is_inspect(self):
        for text in self.VIEW_REQUESTS + ("please show me the lid", "display the enclosure", "look at the housing"):
            with self.subTest(text=text):
                self.assertEqual(classify_intent(text), "inspect")

    def test_a_show_me_a_part_request_is_still_a_build(self):
        self.assertEqual(classify_intent("show me a bracket design"), "bracket")
        self.assertEqual(classify_intent("show me a housing for a raspberry pi 4"), "enclosure")

    def test_show_me_with_an_inspect_noun_is_inspect_today(self):
        self.assertEqual(classify_intent("show me the model"), "inspect")
        self.assertEqual(classify_intent("show me the objects"), "inspect")

    def test_a_view_phrase_with_a_build_verb_is_still_work(self):
        self.assertIn(classify_intent("show me a bracket for a 20 mm tube, then add two holes"), WORK)
        self.assertIn(classify_intent("build the bracket and show me the result"), WORK)


class ContinuationTableTest(unittest.TestCase):
    def open_ledger(self):
        return Ledger.from_request("Build a housing for a Raspberry Pi 4 with a snap-fit lid in ABS")

    def test_go_aheads_resume_an_open_plan(self):
        ledger = self.open_ledger()
        for text in ("continue", "Continue", "yes go ahead", "yes", "go on please", "go ahead", "proceed", "keep going", "ok",
                     "okay", "do it", "resume", "weiter", "sigue", "继续", "продолжай", "carry on"):
            with self.subTest(text=text):
                self.assertTrue(is_continuation(text, ledger))

    def test_questions_thanks_and_long_messages_are_not_go_aheads(self):
        ledger = self.open_ledger()
        for text in ("ok then?", "continue?", "thanks", "looks good", "hello", "what next?", "sure",
                     "continue with the lid but make it 5 mm taller", "yes please go ahead and finish all of it now",
                     "", "   ", "stop"):
            with self.subTest(text=text):
                self.assertFalse(is_continuation(text, ledger))

    def test_without_an_open_plan_nothing_is_a_continuation(self):
        self.assertFalse(is_continuation("continue", None))
        done = Ledger.from_request("what is a fillet?")                 # no required steps
        self.assertFalse(is_continuation("continue", done))

    def test_the_resume_prompt_of_the_panel_anchor_is_a_continuation(self):
        ledger = self.open_ledger()
        self.assertTrue(is_continuation(ledger.resume_prompt(), ledger))
        self.assertTrue(is_continuation("Continue the task. Plan status: S1 done, S2 pending", ledger))

    def test_a_continuation_phrase_is_never_an_intent_by_itself(self):
        for text in ("continue", "yes go ahead", "ok", "weiter"):
            self.assertEqual(classify_intent(text), "other")


class ReferenceAndRequirementsTest(unittest.TestCase):
    def test_a_named_board_needs_a_reference_and_the_hint_names_it(self):
        needs, hint = needs_reference("Build a housing for a Raspberry Pi 4 with a lid", "enclosure")
        self.assertTrue(needs)
        self.assertIn("Raspberry Pi 4", hint)
        for text in ("Gehäuse für Raspberry Pi 4 mit Deckel", "caja para Arduino con tapa", "do a snap fit enclosure for ESP32"):
            with self.subTest(text=text):
                self.assertTrue(needs_reference(text, "enclosure")[0])

    def test_no_part_named_means_no_reference(self):
        self.assertEqual(needs_reference("Can you build me a snap-fit enclosure?", "enclosure"), (False, ""))
        self.assertEqual(needs_reference("make a plate", "generic_part"), (False, ""))
        self.assertEqual(needs_reference("Build a Raspberry Pi 4 housing", "bracket"), (False, ""))

    def test_step_by_step_is_not_a_request_for_a_step_file(self):
        self.assertFalse(needs_reference("build an enclosure, explain it step-by-step", "enclosure")[0])
        self.assertTrue(needs_reference("build an enclosure and download the STEP file", "enclosure")[0])

    def test_an_enclosure_request_yields_a_requirement_per_asked_feature(self):
        reqs = extract_requirements("Bottom shell with cutouts for every I/O port, air vent slots, snap-fit lid, ABS", "enclosure")
        texts = " | ".join(r["text"] for r in reqs)
        for needle in ("bottom shell", "Cutouts", "lid", "vent", "Snap-fit", "ABS"):
            self.assertIn(needle.lower(), texts.lower())
        self.assertLessEqual(len(reqs), 12)
        self.assertEqual([r["id"] for r in reqs], [f"R{i}" for i in range(1, len(reqs) + 1)])

    def test_a_follow_up_edit_has_its_own_small_plan(self):
        ledger = Ledger.from_request("add a fan hole on the side")
        self.assertEqual(ledger.intent, "generic_part")
        self.assertTrue(ledger.is_work_turn())
        self.assertTrue(ledger.pending_required())
        self.assertEqual(ledger.request, "add a fan hole on the side")
        self.assertTrue(any("fan hole" in r.text for r in ledger.requirements))

    def test_non_work_requests_have_no_requirements_and_no_plan(self):
        for text in ("what is a fillet?", "hello", "list all objects", "thanks"):
            with self.subTest(text=text):
                ledger = Ledger.from_request(text)
                self.assertEqual(extract_requirements(text, ledger.intent), [])
                self.assertFalse(ledger.has_open_required_steps())


# -- the same requests through the REAL loop (needs FreeCAD: skipped on a plain interpreter) -------------------------

try:
    from support.realflow import RealFlowCase
    from support.scripted_llm import call, calls, say
    from oscar.providers import Message
except ImportError:      # a pure interpreter: the table above still runs
    RealFlowCase = None

if RealFlowCase is not None:

    class IntentLoopTest(RealFlowCase):
        """What each class of request does once it reaches ``AgentWorker`` (real tools, real Project)."""

        def test_a_question_is_answered_without_plan_project_or_tools_changes(self):
            project = self.new_project("Q", chat_id="20261002-101010-c1c1c1", create=False)
            result = self.turn("how do I make a fillet?", [say("Select the edge and use Part Fillet.")], project=project)
            self.assertEqual((result.ok, result.outcome), (True, "answered"))
            self.assertIsNone(result.ledger)
            self.assertFalse(os.path.isdir(project.root))

        def test_an_inspect_request_reads_and_answers_without_a_plan(self):
            doc = self.open_doc("IntentRead")
            project = self.new_project("Read", chat_id="20261002-101010-c1c1c2")
            script = [calls(call("get_objects", doc_name=doc)), say("The document is empty.")]
            result = self.turn("list all objects", script, project=project)
            self.assertEqual((result.ok, result.outcome), (True, "answered"))
            self.assertEqual(self.worker.ledger.intent, "inspect")
            self.assertFalse(self.worker.ledger.is_work_turn())
            self.assertEqual(project.read_steps(), [])

        def test_a_polite_build_request_opens_the_enclosure_plan(self):
            self.turn("Can you build a snap-fit enclosure for a Raspberry Pi 4?", [say("x")] * 5)
            ledger = self.worker.ledger
            self.assertEqual(ledger.intent, "enclosure")
            self.assertTrue(ledger.needs_reference)
            self.assertIn("Raspberry Pi 4", ledger.reference_hint)
            self.assertIn("PLAN (intent enclosure", self.history[0].content)
            self.assertIn("source_reference_part", set(self.provider.requests[0].tools))

        def test_a_non_english_request_gets_the_same_plan_as_the_english_one(self):
            self.turn("Erstelle bitte ein Gehäuse für den Raspberry Pi 4 mit Deckel", [say("x")] * 5)
            german = self.worker.ledger
            self.turn("Build a housing for a Raspberry Pi 4 with a lid", [say("x")] * 5)
            english = self.worker.ledger
            self.assertEqual((german.intent, german.needs_reference), (english.intent, english.needs_reference))
            self.assertEqual([s.id for s in german.steps], [s.id for s in english.steps])

        def test_a_follow_up_edit_is_a_new_small_turn_in_the_same_project(self):
            doc = self.doc_name("IntentEdit")
            project = self.new_project("Edit", chat_id="20261002-101010-c1c1c3")
            first = self.turn("make a plate", [calls(call("create_document", name=doc), self.box(doc, "Plate")),
                                               say("Made the plate.")], project=project)
            self.assertEqual(first.outcome, "validated")
            hole = call("execute_code", include_screenshot=False, code=self.code(
                doc, "o = d.addObject('Part::Cylinder', 'FanHole')\no.Radius = 3\no.Height = 10\nd.recompute()"))
            history = self.history + [Message(role="user", content="add a fan hole on the side")]
            second = self.turn("add a fan hole on the side", [calls(hole), say("Added the fan hole.")], project=project,
                               history=history)
            self.assertEqual((second.ok, second.outcome), (True, "validated"))
            self.assertEqual(self.worker.ledger.request, "add a fan hole on the side")
            steps = project.read_steps()
            self.assertEqual([s.index for s in steps], [1, 2, 3])
            self.assertEqual(len({s.turn_id for s in steps}), 2)             # two turns, one dossier
            self.assertEqual(project.read_ledger()["request"], "add a fan hole on the side")
            self.assertEqual(steps[-1].delta["added"], ["FanHole"])

        def test_an_unclassified_edit_is_escalated_by_its_first_geometry_call(self):
            doc = self.doc_name("IntentEsc")
            project = self.new_project("Esc", chat_id="20261002-101010-c1c1c4")
            script = [calls(call("create_document", name=doc), self.box(doc, "Lid")),
                      calls(call("execute_code", include_screenshot=False, code=self.code(
                          doc, "d.removeObject('Lid')\nd.recompute()"))), say("Removed the lid.")]
            result = self.turn("delete the lid", script, project=project)
            self.assertEqual(self.worker.ledger.intent, "generic_part")      # escalated from "other"
            self.assertEqual((result.ok, result.outcome), (True, "validated"))
            self.assertEqual(project.read_steps()[-1].delta["removed"], ["Lid"])

        def test_a_go_ahead_resumes_the_open_plan_instead_of_starting_a_new_one(self):
            from support import personas

            project = self.new_project("Resume", chat_id="20261002-101010-c1c1c5")
            first = self.turn(personas.prompt_for("asks_the_user"), personas.build("asks_the_user"), project=project)
            self.assertEqual(first.outcome, "needs_input")
            carried = Ledger.from_dict(project.read_ledger())
            self.assertTrue(is_continuation("yes go ahead", carried))
            history = self.history + [Message(role="user", content="yes go ahead")]
            self.turn("yes go ahead", [say("x")] * 5, project=project, history=history, ledger=carried)
            self.assertEqual(self.worker.ledger.intent, "enclosure")          # not re-classified from "yes go ahead"
            self.assertEqual(self.worker.ledger.request, personas.prompt_for("asks_the_user"))


if __name__ == "__main__":
    unittest.main()
