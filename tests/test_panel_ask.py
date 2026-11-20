# SPDX-License-Identifier: Apache-2.0
"""The ask_user decision cards on the panel: surface registration, anchor answers through the
registry, the QuestionsBar interception (option + typed custom), wizard in-place advance, chat
isolation and the settle states (the panel._ask_surface cluster of oscar/panel.py)."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

from test_panel_logic import PanelTestCase  # noqa: E402
from oscar import askuser, tools_ask  # noqa: E402
from oscar.panel import Block  # noqa: E402

QUESTION = {
    "title": "Which unit is Base.STL in?",
    "subtitle": "The geometry measures 185 x 12 x 200.",
    "options": [{"label": "Millimetres", "description": "185 x 12 x 200 mm", "likeliest": True},
                {"label": "Inches", "description": "7.3 x 0.47 x 7.9 in"}],
    "allow_free_text": True,
}

WIZARD = {
    "wizard": [
        QUESTION,
        {"title": "Approve the rounded dimensions?",
         "subtitle": "Allowed tolerance: +/-0.1 mm",
         "options": [{"label": "Approve all values", "likeliest": True},
                     {"label": "Review individually"}]},
    ]
}


class AskCardTest(PanelTestCase):
    def setUp(self):
        super().setUp()
        self._saved_registry = askuser.REGISTRY
        askuser.REGISTRY = askuser.RequestRegistry()      # per-test isolation of the live registry
        self._saved_surface = tools_ask.surface
        self.chat = str(self.p._current_session()["id"])

    def tearDown(self):
        askuser.REGISTRY.shutdown()
        askuser.REGISTRY = self._saved_registry
        tools_ask.surface = self._saved_surface
        super().tearDown()

    def open_ask(self, payload=None, conversation=None, run_id="t1"):
        request = askuser.REGISTRY.open(payload or askuser.validate_question(QUESTION)[0],
                                        conversation_id=conversation or self.chat, run_id=run_id)
        self.p._ask_surface(request)
        return request

    def question_blocks(self):
        return [b for b in self.p.blocks if b.kind == Block.QUESTION]

    def test_panel_registers_as_the_ask_surface(self):
        self.assertEqual(tools_ask.surface, self.p._ask_surface)   # bound methods compare by target
        self.p._shutdown()
        self.assertIsNone(tools_ask.surface)

    def test_a_click_answers_through_the_registry_and_flips_the_card(self):
        request = self.open_ask()
        (block,) = self.question_blocks()
        self.assertEqual(block.detail, "pending")
        self.assertIn(f"oscar://ask/{request.id}/0/1", block.html())
        self.p._on_anchor_text(f"oscar://ask/{request.id}/0/1")
        self.assertEqual(request.state, "answered")
        self.assertEqual(request.answer["index"], 1)
        self.assertEqual(request.answer["answer"], "Inches")
        self.assertEqual(block.detail, "answered")
        self.assertIn("Inches", block.reason)
        self.assertNotIn(request.id, self.p._open_asks)

    def test_double_clicks_and_stale_ids_change_nothing(self):
        request = self.open_ask()
        self.p._on_anchor_text(f"oscar://ask/{request.id}/0/0")
        settled = request.answer
        self.p._on_anchor_text(f"oscar://ask/{request.id}/0/1")     # second click: not open anymore
        self.p._on_anchor_text("oscar://ask/bogus/0/0")             # unknown id
        self.p._on_anchor_text(f"oscar://ask/{request.id}/9/0")     # unknown question index
        self.assertEqual(request.answer, settled)

    def test_the_bar_answers_the_open_card_instead_of_sending(self):
        request = self.open_ask()
        with mock.patch.object(self.p, "_send_or_steer") as send:
            self.p._on_bar_answer("Inches")
            send.assert_not_called()
        self.assertEqual(request.state, "answered")

    def test_a_typed_bar_answer_becomes_the_custom_value(self):
        request = self.open_ask()
        with mock.patch.object(self.p, "_send_or_steer") as send:
            self.p._on_bar_custom("Metres")
            send.assert_not_called()
        self.assertEqual(request.state, "answered")
        self.assertEqual((request.answer["index"], request.answer["value"]), (-1, "Metres"))

    def test_the_bar_falls_through_when_no_ask_is_open(self):
        with mock.patch.object(self.p, "_send_or_steer") as send:
            self.p._on_bar_answer("hello there")
            send.assert_called_once_with("hello there")

    def test_a_bar_click_for_a_settled_decision_flashes_and_never_sends(self):
        """Answered after the request settled elsewhere: report that it can no longer be
        accepted — never leak the click into the chat as a message."""
        request = self.open_ask()
        self.p._on_anchor_text(f"oscar://ask/{request.id}/0/0")
        self.assertEqual(request.state, "answered")
        self.p._bar_ask_id = request.id          # the bar still speaks for the settled decision
        with mock.patch.object(self.p, "_send_or_steer") as send, \
                mock.patch.object(self.p, "_flash_status") as flash:
            self.p._on_bar_answer("Inches")
            send.assert_not_called()
            flash.assert_called_once()
            self.assertIn("no longer open", flash.call_args[0][0])

    def test_a_bar_answer_of_another_chat_flashes_and_never_sends(self):
        request = self.open_ask()
        with mock.patch.object(self.p, "_current_session", return_value={"id": "OTHER-CHAT"}), \
                mock.patch.object(self.p, "_send_or_steer") as send, \
                mock.patch.object(self.p, "_flash_status") as flash:
            self.p._on_bar_answer("Inches")
            send.assert_not_called()
            flash.assert_called_once()
            self.assertIn("Open this chat", flash.call_args[0][0])
        self.assertEqual(request.state, "open", "another chat's click settles nothing")

    def test_an_unknown_bar_label_keeps_the_decision_open(self):
        request = self.open_ask()
        with mock.patch.object(self.p, "_send_or_steer") as send, \
                mock.patch.object(self.p, "_flash_status") as flash:
            self.p._on_bar_answer("Parsecs")
            send.assert_not_called()
            flash.assert_called_once()
        self.assertEqual(request.state, "open")
        self.p._on_anchor_text(f"oscar://ask/{request.id}/0/1")     # still answerable afterwards
        self.assertEqual(request.state, "answered")

    def test_a_typed_answer_when_the_decision_takes_no_text_flashes_and_never_sends(self):
        payload = {k: v for k, v in QUESTION.items() if k != "custom_value"}
        payload["allow_free_text"] = False
        request = self.open_ask(payload=payload)
        with mock.patch.object(self.p, "_send_or_steer") as send, \
                mock.patch.object(self.p, "_flash_status") as flash:
            self.p._on_bar_custom("Metres")
            send.assert_not_called()
            flash.assert_called_once()
        self.assertEqual(request.state, "open")

    def test_the_bar_answers_a_wizard_in_sequence(self):
        request = self.open_ask(payload=askuser.validate_wizard(WIZARD)[0])
        with mock.patch.object(self.p, "_send_or_steer") as send:
            self.p._on_bar_answer("Millimetres")
            self.p._on_bar_answer("Review individually")
            send.assert_not_called()
        self.assertEqual(request.state, "answered")
        self.assertEqual([a["index"] for a in request.answer["answers"]], [0, 1])

    def test_a_timed_out_decision_card_shows_the_real_timeout(self):
        request = self.open_ask()
        request.timeout_s = 1800.0
        with askuser.REGISTRY._lock:
            askuser.RequestRegistry._settle(request, askuser.TIMEOUT)
        self.p._settle_ask(self.p._open_asks[request.id])
        (block,) = self.question_blocks()
        self.assertEqual(block.detail, "timeout")
        self.assertIn("denied after 1800 s", block.html())   # not the renderer's 120 s default

    def test_the_card_stays_pending_when_the_backend_refuses_the_answer(self):
        """The UI/backend contract: the card never shows ANSWERED without the registry's
        acknowledgement — a refused submission keeps the question open and retryable."""
        request = self.open_ask()
        (block,) = self.question_blocks()
        with mock.patch.object(askuser.REGISTRY, "submit_answer",
                               return_value=(False, "already_answered: test double")) as submit, \
                mock.patch.object(self.p, "_flash_status") as flash:
            self.p._on_anchor_text(f"oscar://ask/{request.id}/0/1")
            submit.assert_called_once()
            flash.assert_called_once()
            self.assertIn("could not be submitted", flash.call_args[0][0])
        self.assertEqual(block.detail, "pending", "no ANSWERED before backend acknowledgement")
        self.assertIn(request.id, self.p._open_asks)
        self.assertFalse(self.p._open_asks[request.id]["done"])
        self.assertEqual(request.state, "open")            # the double never touched the registry
        self.p._on_anchor_text(f"oscar://ask/{request.id}/0/1")   # the retry goes through
        self.assertEqual(request.state, "answered")
        self.assertEqual(block.detail, "answered")

    def test_simultaneous_decisions_answer_independently(self):
        """Two open requests of one chat: each answer resolves exactly its own request — a
        half-answered wizard never partially resolves the other one."""
        first = self.open_ask(payload=askuser.validate_wizard(WIZARD)[0], run_id="t1")
        second = self.open_ask(run_id="t2")
        self.p._on_anchor_text(f"oscar://ask/{first.id}/0/0")      # wizard Q1 only
        self.assertEqual(first.state, "open", "a half-answered wizard resolves nothing")
        self.assertEqual(second.state, "open")
        self.p._on_anchor_text(f"oscar://ask/{second.id}/0/1")
        self.assertEqual(second.state, "answered")
        self.assertEqual(first.state, "open", "the wizard was untouched by the other answer")
        self.p._on_anchor_text(f"oscar://ask/{first.id}/1/1")     # wizard Q2 completes it
        self.assertEqual(first.state, "answered")
        self.assertEqual([a["index"] for a in first.answer["answers"]], [0, 1])

    def test_a_wizard_updates_the_same_card_in_place(self):
        request = self.open_ask(payload=askuser.validate_wizard(WIZARD)[0])
        (block,) = self.question_blocks()
        first_bid = block.bid
        self.p._on_anchor_text(f"oscar://ask/{request.id}/0/0")      # answer Q1 (Millimetres)
        (block,) = self.question_blocks()                            # still ONE card
        self.assertEqual(block.bid, first_bid)
        self.assertIn("question 2 of 2", block.text)
        self.assertIn("Approve the rounded dimensions?", block.text)
        self.p._on_anchor_text(f"oscar://ask/{request.id}/1/1")      # answer Q2 (Review individually)
        self.assertEqual(request.state, "answered")
        self.assertEqual([a["index"] for a in request.answer["answers"]], [0, 1])
        self.assertEqual(block.detail, "answered")

    def test_a_card_of_another_chat_is_refused(self):
        with self.assertRaises(RuntimeError):
            self.open_ask(conversation="OTHER-CHAT")

    def test_an_external_settle_flips_the_card(self):
        request = self.open_ask()
        state = self.p._open_asks[request.id]
        askuser.REGISTRY.cancel_run("t1")                            # Stop pressed / turn ended
        self.p._settle_ask(state)
        (block,) = self.question_blocks()
        self.assertEqual(block.detail, "cancelled")
        self.assertNotIn(request.id, self.p._open_asks)

    def test_a_settled_ask_is_remembered_for_the_turn_record(self):
        request = self.open_ask()
        self.p._on_anchor_text(f"oscar://ask/{request.id}/0/0")
        (record,) = self.p._turn_asks
        self.assertEqual((record["request_id"], record["state"]), (request.id, "answered"))
        self.assertEqual(record["payload"]["title"], QUESTION["title"])

    def test_the_restore_card_renders_persisted_decisions(self):
        from oscar import panel_render

        meta = {"v": 1, "request": "rebuild the bracket", "outcome": "answered", "header": "", "steps": [],
                "asks": [{"request_id": "abc123", "state": "answered", "created_ts": 0.0, "resolved_ts": 1.0,
                          "payload": {"title": "Which unit?", "subtitle": "", "options": []},
                          "answers": [{"answer": "Millimetres", "index": 0}]}]}
        html = panel_render.restore_card_html("My chat", 3, [meta])
        self.assertIn("decision:", html)
        self.assertIn("Which unit?", html)
        self.assertIn("Millimetres", html)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
