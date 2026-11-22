# SPDX-License-Identifier: Apache-2.0
"""Pure decision-request logic: question/wizard validation, the pending-request registry and the
answer protocol (oscar.askuser). No FreeCAD, no Qt — the thread handshake is exercised for real."""

import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # every suite member isolates itself before importing oscar

from oscar import askuser  # noqa: E402


QUESTION = {
    "title": "Which unit is Base.STL in?",
    "subtitle": "The imported geometry measures 185 x 12 x 200 in the current unit interpretation.",
    "mono_subtitle": False,
    "options": [
        {"label": "Millimetres", "description": "185 x 12 x 200 mm", "likeliest": True},
        {"label": "Centimetres", "description": "1.85 x 0.12 x 2 m"},
        {"label": "Inches", "description": "4.7 x 0.305 x 5.08 m"},
    ],
    "custom_value": {"label": "Another value", "unit": "mm"},
    "allow_free_text": True,
}

WIZARD = {
    "wizard": [
        QUESTION,
        {"title": "Approve the rounded dimensions?",
         "subtitle": "Allowed tolerance: +/-0.1 mm",
         "options": [{"label": "Approve all values", "description": "Use the proposed nominal dimensions",
                      "likeliest": True},
                     {"label": "Review individually", "description": "Review each dimension separately"}]},
    ]
}


class ValidationTest(unittest.TestCase):
    def test_a_well_formed_question_normalizes_with_defaults(self):
        q, err = askuser.validate_question(QUESTION)
        self.assertEqual(err, "")
        self.assertEqual(q["title"], QUESTION["title"])
        self.assertEqual([o["label"] for o in q["options"]], ["Millimetres", "Centimetres", "Inches"])
        self.assertEqual([o["likeliest"] for o in q["options"]], [True, False, False])
        self.assertEqual(q["options"][1]["description"], "1.85 x 0.12 x 2 m")   # carried through
        self.assertTrue(q["allow_free_text"])
        self.assertEqual(q["custom_value"]["unit"], "mm")

    def test_minimal_question_gets_the_defaults(self):
        q, err = askuser.validate_question({
            "title": "Keep the fillet?",
            "options": [{"label": "Yes", "likeliest": True}, {"label": "No"}],
        })
        self.assertEqual(err, "")
        self.assertEqual((q["subtitle"], q["mono_subtitle"], q["allow_free_text"], q["custom_value"]),
                         ("", False, False, None))

    def test_a_well_formed_wizard_passes(self):
        w, err = askuser.validate_wizard(WIZARD)
        self.assertEqual(err, "")
        self.assertEqual(len(w["wizard"]), 2)
        self.assertEqual(w["wizard"][1]["options"][0]["likeliest"], True)

    def test_missing_title_is_rejected(self):
        q, err = askuser.validate_question({"options": QUESTION["options"]})
        self.assertIsNone(q)
        self.assertIn("title", err)

    def test_one_or_five_options_are_rejected(self):
        one = {"title": "t", "options": [{"label": "Only", "likeliest": True}]}
        five = {"title": "t", "options": [{"label": str(i), "likeliest": i == 0} for i in range(5)]}
        for payload in (one, five):
            q, err = askuser.validate_question(payload)
            self.assertIsNone(q)
            self.assertIn("2 to 4", err)

    def test_zero_or_two_likeliest_are_rejected(self):
        none_marked = {"title": "t", "options": [{"label": "a"}, {"label": "b"}]}
        two_marked = {"title": "t", "options": [{"label": "a", "likeliest": True},
                                                {"label": "b", "likeliest": True}]}
        for payload in (none_marked, two_marked):
            q, err = askuser.validate_question(payload)
            self.assertIsNone(q)
            self.assertIn("likeliest", err)

    def test_duplicate_option_labels_are_rejected(self):
        q, err = askuser.validate_question({"title": "t", "options": [{"label": "Same", "likeliest": True},
                                                                      {"label": "Same"}]})
        self.assertIsNone(q)
        self.assertIn("duplicate", err)

    def test_wrong_types_are_rejected_not_repaired(self):
        for bad in ({"title": 7, "options": QUESTION["options"]},
                    {"title": "t", "options": "yes/no"},
                    {"title": "t", "options": [{"label": "a", "likeliest": True}, {"label": 9}]},
                    {"title": "t", "options": [{"label": "a", "likeliest": True}, {"label": "b"}],
                     "allow_free_text": "yes"}):
            q, err = askuser.validate_question(bad)
            self.assertIsNone(q, bad)
            self.assertTrue(err)

    def test_a_wizard_needs_two_to_six_questions(self):
        one = {"wizard": [QUESTION]}
        seven = {"wizard": [QUESTION] * 7}
        for payload in (one, seven):
            w, err = askuser.validate_wizard(payload)
            self.assertIsNone(w)
            self.assertIn("2 to 6", err)

    def test_each_wizard_question_is_validated(self):
        broken = {"wizard": [QUESTION, {"options": QUESTION["options"]}]}   # second lacks a title
        w, err = askuser.validate_wizard(broken)
        self.assertIsNone(w)
        self.assertIn("title", err)


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.reg = askuser.RequestRegistry()

    def test_open_resolve_and_wait_round_trip(self):
        req = self.reg.open(askuser.validate_question(QUESTION)[0], conversation_id="c1", run_id="r1")
        self.assertEqual(req.state, "open")
        ok, err = self.reg.resolve(req.id, {"answer": "Millimetres", "index": 0},
                                   conversation_id="c1", run_id="r1")
        self.assertEqual((ok, err), (True, ""))
        settled = self.reg.wait(req, timeout_s=1.0)
        self.assertEqual(settled.state, "answered")
        self.assertEqual(settled.answer["answer"], "Millimetres")
        self.assertEqual(settled.answer["index"], 0)
        self.assertEqual(settled.answer["request_id"], req.id)
        self.assertEqual((settled.answer["conversation_id"], settled.answer["run_id"]), ("c1", "r1"))
        self.assertGreater(settled.answer["answered_ts"], 0.0)

    def test_a_custom_answer_carries_its_typed_value(self):
        q = askuser.validate_question(QUESTION)[0]
        req = self.reg.open(q, conversation_id="c1", run_id="r1")
        ok, err = self.reg.resolve(req.id, {"answer": "Metres", "index": -1, "value": "m"},
                                   conversation_id="c1", run_id="r1")
        self.assertEqual((ok, err), (True, ""))
        self.assertEqual(req.answer["index"], -1)
        self.assertEqual(req.answer["value"], "m")

    def test_duplicate_submissions_are_rejected(self):
        req = self.reg.open(askuser.validate_question(QUESTION)[0], conversation_id="c1", run_id="r1")
        first = self.reg.resolve(req.id, {"answer": "Inches", "index": 2}, conversation_id="c1", run_id="r1")
        second = self.reg.resolve(req.id, {"answer": "Inches", "index": 2}, conversation_id="c1", run_id="r1")
        self.assertEqual(first[0], True)
        self.assertEqual(second[0], False)
        self.assertIn("already", second[1])

    def test_a_stale_run_cannot_resolve_another_run_request(self):
        req = self.reg.open(askuser.validate_question(QUESTION)[0], conversation_id="c1", run_id="r1")
        ok, err = self.reg.resolve(req.id, {"answer": "Inches", "index": 2},
                                   conversation_id="c1", run_id="OTHER-RUN")
        self.assertFalse(ok)
        self.assertIn("stale", err)
        self.assertEqual(req.state, "open")            # the live request survives the stale attempt

    def test_conversations_are_isolated(self):
        req = self.reg.open(askuser.validate_question(QUESTION)[0], conversation_id="c1", run_id="r1")
        ok, err = self.reg.resolve(req.id, {"answer": "Inches", "index": 2},
                                   conversation_id="OTHER-CHAT", run_id="r1")
        self.assertFalse(ok)
        self.assertIn("stale", err)
        self.assertEqual(req.state, "open")

    def test_unknown_request_ids_are_rejected(self):
        ok, err = self.reg.resolve("nope", {"answer": "x", "index": 0}, conversation_id="c1", run_id="r1")
        self.assertFalse(ok)
        self.assertIn("unknown", err)

    def test_out_of_range_indexes_and_empty_answers_are_rejected(self):
        req = self.reg.open(askuser.validate_question(QUESTION)[0], conversation_id="c1", run_id="r1")
        for bad in ({"answer": "x", "index": 3}, {"answer": "x", "index": -2}, {"answer": "", "index": -1},
                    {"answer": "x"}, {"index": 0}):
            ok, err = self.reg.resolve(req.id, bad, conversation_id="c1", run_id="r1")
            self.assertFalse(ok, bad)
            self.assertEqual(req.state, "open", bad)   # a rejected answer never settles the request

    def test_wizard_answers_must_cover_every_question_in_order(self):
        w = askuser.validate_wizard(WIZARD)[0]
        req = self.reg.open(w, conversation_id="c1", run_id="r1")
        for bad in ({"answers": [{"answer": "Millimetres", "index": 0}]},                      # too short
                    {"answers": [{"answer": "Millimetres", "index": 0},
                                 {"answer": "Approve all values", "index": 0},
                                 {"answer": "x", "index": 0}]}):                                # too long
            ok, err = self.reg.resolve(req.id, bad, conversation_id="c1", run_id="r1")
            self.assertFalse(ok, bad)
        ok, err = self.reg.resolve(req.id, {"answers": [{"answer": "Millimetres", "index": 0},
                                                        {"answer": "Approve all values", "index": 0}]},
                                   conversation_id="c1", run_id="r1")
        self.assertEqual((ok, err), (True, ""))
        self.assertEqual([a["answer"] for a in req.answer["answers"]], ["Millimetres", "Approve all values"])

    def test_cancel_run_releases_the_waiter(self):
        req = self.reg.open(askuser.validate_question(QUESTION)[0], conversation_id="c1", run_id="r1")
        done = []

        def waiter():
            done.append(self.reg.wait(req, timeout_s=30.0).state)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        self.assertEqual(self.reg.cancel_run("r1"), 1)
        t.join(timeout=5)
        self.assertEqual(done, ["cancelled"])

    def test_wait_honours_should_stop(self):
        req = self.reg.open(askuser.validate_question(QUESTION)[0], conversation_id="c1", run_id="r1")
        stop = threading.Event()

        def flip():
            time.sleep(0.05)
            stop.set()

        threading.Thread(target=flip).start()
        settled = self.reg.wait(req, timeout_s=30.0, should_stop=stop.is_set)
        self.assertEqual(settled.state, "cancelled")

    def test_wait_times_out(self):
        req = self.reg.open(askuser.validate_question(QUESTION)[0], conversation_id="c1", run_id="r1",
                            timeout_s=0.05)
        settled = self.reg.wait(req, timeout_s=0.05)
        self.assertEqual(settled.state, "timeout")

    def test_shutdown_releases_every_waiter_and_empties_the_registry(self):
        reqs = [self.reg.open(askuser.validate_question(QUESTION)[0], conversation_id=f"c{i}", run_id=f"r{i}")
                for i in range(3)]
        released = []

        def waiter(r):
            released.append(self.reg.wait(r, timeout_s=30.0).state)

        threads = [threading.Thread(target=waiter, args=(r,)) for r in reqs]
        for t in threads:
            t.start()
        time.sleep(0.05)
        self.assertEqual(self.reg.shutdown(), 3)
        for t in threads:
            t.join(timeout=5)
        self.assertEqual(released, ["cancelled"] * 3)
        self.assertEqual(self.reg.pending(), [])

    def test_pending_lists_open_requests_per_conversation(self):
        self.reg.open(askuser.validate_question(QUESTION)[0], conversation_id="c1", run_id="r1")
        self.reg.open(askuser.validate_question(QUESTION)[0], conversation_id="c2", run_id="r2")
        self.assertEqual([p.conversation_id for p in self.reg.pending()], ["c1", "c2"])
        self.assertEqual([p.conversation_id for p in self.reg.pending(conversation_id="c2")], ["c2"])

    def test_requests_are_serializable_for_persistence(self):
        req = self.reg.open(askuser.validate_wizard(WIZARD)[0], conversation_id="c1", run_id="r1")
        self.reg.resolve(req.id, {"answers": [{"answer": "Millimetres", "index": 0},
                                              {"answer": "Approve all values", "index": 0}]},
                         conversation_id="c1", run_id="r1")
        import json
        blob = json.dumps(req.record())
        back = json.loads(blob)
        self.assertEqual(back["request_id"], req.id)
        self.assertEqual(back["state"], "answered")
        self.assertEqual(len(back["answers"]), 2)
        self.assertEqual(back["payload"]["wizard"][0]["title"], QUESTION["title"])


class RaceTest(unittest.TestCase):
    """The answer/timeout boundary: an answer accepted before the deadline tick can never be
    rejected by the timeout, and an answer that loses the race is reported, never guessed."""

    def setUp(self):
        self.reg = askuser.RequestRegistry()

    def open(self, timeout_s=0.3, run_id="r1"):
        return self.reg.open(askuser.validate_question(QUESTION)[0], conversation_id="c1",
                             run_id=run_id, timeout_s=timeout_s)

    def test_an_answer_during_the_wait_wakes_the_waiter_at_once(self):
        req = self.open(timeout_s=30.0)
        threading.Timer(0.05, lambda: self.reg.resolve(
            req.id, {"answer": "Inches", "index": 2}, conversation_id="c1", run_id="r1")).start()
        settled = self.reg.wait(req, timeout_s=30.0)
        self.assertEqual(settled.state, "answered")
        self.assertEqual(settled.answer["answer"], "Inches")

    def test_answers_at_the_deadline_boundary_are_never_lost(self):
        """Resolve fires from another thread just before each waiter's deadline tick: whichever
        side takes the lock first wins, but an ACCEPTED answer is always delivered (the failure
        this guards against is 'UI answered, agent denied after the timeout')."""
        for _ in range(20):
            req = self.open(timeout_s=0.15)
            accepted = []

            def answer():
                ok, _ = self.reg.resolve(req.id, {"answer": "Millimetres", "index": 0},
                                         conversation_id="c1", run_id="r1")
                accepted.append(ok)

            t = threading.Thread(target=answer)
            t.start()
            settled = self.reg.wait(req, timeout_s=0.15)
            t.join(timeout=5)
            if accepted == [True]:
                self.assertEqual(settled.state, "answered", "an accepted answer must wake the waiter")
                self.assertEqual(settled.answer["answer"], "Millimetres")
            else:
                # the deadline took the lock first: an honest timeout, and the answer was REFUSED
                self.assertEqual(settled.state, "timeout")
                self.assertEqual(accepted, [False])

    def test_resolve_and_cancel_race_keeps_exactly_one_outcome(self):
        for _ in range(10):
            req = self.open(timeout_s=30.0)
            outcomes = []

            def answer():
                outcomes.append(self.reg.resolve(req.id, {"answer": "Inches", "index": 2},
                                                 conversation_id="c1", run_id="r1")[0])

            def cancel():
                outcomes.append(self.reg.cancel_run("r1") > 0)

            threads = [threading.Thread(target=answer), threading.Thread(target=cancel)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)
            settled = self.reg.wait(req, timeout_s=30.0)   # already released; returns at once
            self.assertEqual(sum(1 for o in outcomes if o), 1, "exactly one side may settle")
            self.assertIn(settled.state, ("answered", "cancelled"))

    def test_wait_polls_do_not_overshadow_a_late_answer(self):
        """A slow user answers at 80% of the window: the waiter must return ANSWERED, not time
        out on the strength of an earlier poll."""
        req = self.open(timeout_s=0.5)
        threading.Timer(0.4, lambda: self.reg.resolve(
            req.id, {"answer": "Centimetres", "index": 1}, conversation_id="c1", run_id="r1")).start()
        settled = self.reg.wait(req, timeout_s=0.5)
        self.assertEqual(settled.state, "answered")
        self.assertEqual(settled.answer["answer"], "Centimetres")


class SubmitAnswerTest(unittest.TestCase):
    """The canonical submission API: deterministic reason codes, the ANSWERING transition, the
    retryable invalid answer and the terminal-state guarantee (``answered`` can never be
    overwritten by the deadline, a Stop or a duplicate submission)."""

    def setUp(self):
        self.reg = askuser.RequestRegistry()

    def open(self, run_id="r1", conversation_id="c1"):
        return self.reg.open(askuser.validate_question(QUESTION)[0],
                             conversation_id=conversation_id, run_id=run_id)

    def submit(self, req, answer="Millimetres", index=0, *, conversation_id="c1", run_id="r1"):
        return self.reg.submit_answer(req.id, {"answer": answer, "index": index},
                                      conversation_id=conversation_id, run_id=run_id)

    def test_reason_codes_are_deterministic(self):
        req = self.open()
        self.assertEqual(self.submit(req), (True, ""))
        for kwargs, reason in ((dict(), "already_answered"),                     # duplicate
                               (dict(conversation_id="OTHER"), "stale_request"),  # wrong chat
                               (dict(run_id="OTHER"), "stale_request")):          # wrong run
            ok, err = self.submit(req, answer="Inches", index=2, **kwargs)
            self.assertFalse(ok, reason)
            self.assertTrue(err.startswith(reason), err)
        ok, err = self.reg.submit_answer("nope", {"answer": "x", "index": 0},
                                         conversation_id="c1", run_id="r1")
        self.assertFalse(ok)
        self.assertTrue(err.startswith("unknown_request"), err)
        self.assertEqual(req.state, "answered")       # none of the refusals touched it

    def test_a_timed_out_request_reports_timed_out(self):
        req = self.open()
        self.assertEqual(self.reg.wait(req, timeout_s=0.05).state, "timeout")
        ok, err = self.submit(req, answer="Inches", index=2)
        self.assertFalse(ok)
        self.assertTrue(err.startswith("timed_out"), err)
        self.assertEqual(req.state, "timeout")

    def test_an_invalid_answer_is_rejected_but_retryable(self):
        req = self.open()
        ok, err = self.submit(req, answer="x", index=9)
        self.assertFalse(ok)
        self.assertTrue(err.startswith("invalid_answer"), err)
        self.assertEqual(req.state, "open", "a malformed answer reopens the question, never kills it")
        self.assertEqual(self.submit(req), (True, ""))     # the retry goes through
        self.assertEqual(req.answer["index"], 0)

    def test_an_answered_request_is_terminal(self):
        req = self.open()
        self.assertEqual(self.submit(req), (True, ""))
        for state in (askuser.TIMEOUT, askuser.CANCELLED, askuser.ANSWERED, askuser.REJECTED):
            self.assertFalse(askuser.RequestRegistry._settle(req, state),
                             f"answered must never become {state}")
        self.assertEqual(req.state, "answered")
        self.assertEqual(self.reg.cancel_run("r1"), 0)
        self.assertEqual(self.reg.wait(req).state, "answered")   # its event is already set
        self.assertEqual(self.reg.shutdown(), 0)

    def test_the_answering_transition_beats_a_concurrent_timeout(self):
        """The submission holds the registry lock from the PENDING check to the event set, so a
        waiter whose deadline fires mid-validation can only ever observe ANSWERED."""
        req = self.open()
        entered, release = threading.Event(), threading.Event()
        origin = askuser.RequestRegistry._check_answer

        def slow_check(*args):
            entered.set()
            release.wait(5)
            return origin(*args[-2:])   # (request, answer), bound or unbound

        results = []
        with mock.patch.object(askuser.RequestRegistry, "_check_answer", slow_check):
            submitter = threading.Thread(target=lambda: results.append(self.submit(req)))
            submitter.start()
            self.assertTrue(entered.wait(5))            # the answer is being accepted under the lock
            settled = self.reg.wait(req, timeout_s=0.05)   # its deadline fires mid-validation
            release.set()
            submitter.join(timeout=5)
        self.assertEqual(results, [(True, "")])
        self.assertEqual(settled.state, "answered", "the in-flight answer wins the race")
        self.assertEqual(settled.answer["answer"], "Millimetres")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
