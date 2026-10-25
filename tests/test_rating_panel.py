"""The sampled rating card in the real panel (oscar/rating_gui.py): a substantial turn that is sampled gets the card;
only a card this panel offered acts on its links; the stars open the dialog and Send posts the turn (prompt, answer,
tool steps; no reasoning rows) to /v1/feedback; Not now puts the card away; the cooldown, a signed-out account and a
small turn get no card. Offscreen QApplication. Skipped when PySide / FreeCAD are unavailable.
"""

import os
import sys
import time
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

_IMPORT_ERROR = ""
try:
    from PySide import QtCore, QtWidgets
    from oscar import params, rating, rating_gui
    from oscar.panel import Block
    from test_panel_flow import FlowTestCase
except Exception as exc:   # pragma: no cover - plain interpreter without FreeCAD / Qt
    _IMPORT_ERROR = str(exc)

    class _Unavailable(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")

    FlowTestCase = _Unavailable

BIG = "Design a wall-mounted enclosure for a Raspberry Pi 4 with vents on both sides and a snap-fit lid. " * 3


class RatingPanelTest(FlowTestCase):
    def setUp(self):
        super().setUp()
        for name, value in ((rating.PREF_LAST, 0), (rating.PREF_COUNT, 0)):
            params.set_int(name, value)
        params.set_string(rating.PREF_DAY, "")
        self.signed_in = True
        for patcher in (
            mock.patch.object(rating_gui.random, "random", return_value=0.0),   # always sampled
            mock.patch.object(self.p, "_account_state", side_effect=lambda: {"signed_in": self.signed_in}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def finish_turn(self, prompt: str = BIG, steps: int = 3) -> list:
        self.begin_turn()
        self.p._turn_request = prompt
        self.p._turn_mutated = False   # what the panel resets when a turn is sent
        for n in range(steps):
            self.rich_step(call_id=f"call_{time.monotonic_ns()}_{n}", title=f"Build part {n}", ordinal=n + 1)
        self.p._on_turn_finished(True, "I built the enclosure.")
        return [block for block in self.p.blocks if block.kind == Block.RATING]

    def dialog(self, stars: int = 4, comment: str = "Vents too small", image: bool = False):
        fake = mock.Mock()
        fake.exec.return_value = QtWidgets.QDialog.Accepted
        fake.stars.return_value = stars
        fake.comment.return_value = comment
        fake.include_image.return_value = image
        return mock.patch.object(rating_gui, "RatingDialog", return_value=fake)

    def wait_for(self, predicate, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            QtCore.QCoreApplication.processEvents()
            if predicate():
                return True
            time.sleep(0.02)
        return predicate()

    def test_a_sampled_turn_gets_a_card_and_send_posts_the_turn(self):
        (card,) = self.finish_turn()
        self.assertIn("How did Oscar do on this one?", card.html())
        posted = mock.Mock()
        posted.authorized_request.return_value = types.SimpleNamespace(status=201)
        with mock.patch("oscar.auth.session.get_session", return_value=posted), self.dialog() as dialog_class:
            self.p._on_anchor_target(f"oscar://rate/{card.bid}/5")   # a block number, not the card's key
            self.p._on_anchor_target("oscar://rate/0123456789abcdef/5")   # not a card this panel offered
            dialog_class.assert_not_called()
            self.p._on_anchor_target(f"oscar://rate/{card.key}/4")
            self.assertTrue(self.wait_for(lambda: card.detail == rating.SENT))
        dialog_class.assert_called_once()
        method, path = posted.authorized_request.call_args.args
        body = posted.authorized_request.call_args.kwargs["json_body"]
        self.assertEqual((method, path, body["kind"], body["stars"]), ("POST", "/v1/feedback", "rating", 4))
        self.assertEqual(body["prompt"], BIG.strip())
        self.assertEqual(body["response"], "I built the enclosure.")
        self.assertEqual([step["title"] for step in body["steps"]], ["Build part 0", "Build part 1", "Build part 2"])
        self.assertEqual(body["text"], "Vents too small")
        self.assertNotIn("image_png", body)
        self.assertTrue(body["context"]["sampled"])
        self.assertIn("Thanks", card.html())
        self.p._on_anchor_target(f"oscar://rate/{card.key}/5")   # sent: the card no longer acts

    def test_a_failed_send_lets_the_user_try_again(self):
        (card,) = self.finish_turn()
        broken = mock.Mock()
        broken.authorized_request.return_value = types.SimpleNamespace(status=503)
        with mock.patch("oscar.auth.session.get_session", return_value=broken), self.dialog():
            self.p._on_anchor_target(f"oscar://rate/{card.key}/2")
            self.assertTrue(self.wait_for(lambda: card.detail == rating.FAILED))
        self.assertIn("couldn’t be sent", card.html())

    def test_not_now_puts_the_card_away(self):
        (card,) = self.finish_turn()
        with self.dialog() as dialog_class:
            self.p._on_anchor_target(f"oscar://rate/{card.key}/later")
        dialog_class.assert_not_called()
        self.assertEqual((card.detail, card.html()), (rating.DISMISSED, ""))

    def test_the_cooldown_a_small_turn_and_a_signed_out_account_get_no_card(self):
        self.assertEqual(len(self.finish_turn()), 1)
        self.assertEqual(len(self.finish_turn()), 1)   # within the cooldown: still only the first card
        params.set_int(rating.PREF_LAST, 0)
        before = len([b for b in self.p.blocks if b.kind == Block.RATING])
        self.assertEqual(len(self.finish_turn(prompt="Make it 2 mm thicker", steps=0)), before)   # small turn
        self.signed_in = False
        self.assertEqual(len(self.finish_turn()), before)


if __name__ == "__main__":
    unittest.main()
