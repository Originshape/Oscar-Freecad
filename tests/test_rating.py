"""Sampled star ratings (oscar/rating.py): which turns are asked about (substantial ones, by chance, with a cooldown
and a daily cap), what a rating sends (prompt, answer, steps, optional picture; never the model's reasoning rows),
the card's HTML and its links. Pure python (stdlib).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

from oscar import rating  # noqa: E402

NOW = 1_800_000_000.0
TODAY = "2027-01-15"
BIG = "Design a wall-mounted enclosure for a Raspberry Pi 4 with vents on both sides, " * 4


def facts(**overrides) -> rating.TurnFacts:
    values = {"prompt": BIG, "response": "Built the enclosure.", "steps": [], "mutated": False, "ok": True,
              "outcome": "answered"}
    values.update(overrides)
    return rating.TurnFacts(**values)


def ask(turn: rating.TurnFacts, **overrides) -> bool:
    values = {"now": NOW, "today": TODAY, "last_asked": 0.0, "asked_day": "", "asked_count": 0, "roll": 0.1}
    values.update(overrides)
    return rating.should_ask(turn, **values)


class WhenToAskTests(unittest.TestCase):
    def test_a_substantial_turn_is_a_long_prompt_several_steps_or_a_change_to_the_model(self):
        self.assertTrue(rating.substantial(facts()))
        short = {"prompt": "Make it 2 mm thicker"}
        self.assertFalse(rating.substantial(facts(**short)))
        self.assertTrue(rating.substantial(facts(**short, steps=[{"tool": "x"}] * rating.MIN_STEPS)))
        self.assertTrue(rating.substantial(facts(**short, mutated=True)))
        self.assertFalse(rating.substantial(facts(ok=False)))
        self.assertFalse(rating.substantial(facts(response="  ")))

    def test_only_by_chance(self):
        self.assertTrue(ask(facts(), roll=rating.CHANCE - 0.01))
        self.assertFalse(ask(facts(), roll=rating.CHANCE))
        self.assertFalse(ask(facts(prompt="hi"), roll=0.0))   # never for a small turn

    def test_never_twice_within_the_cooldown(self):
        self.assertFalse(ask(facts(), last_asked=NOW - rating.COOLDOWN_S + 60))
        self.assertTrue(ask(facts(), last_asked=NOW - rating.COOLDOWN_S - 1))

    def test_at_most_a_few_times_a_day(self):
        self.assertFalse(ask(facts(), asked_day=TODAY, asked_count=rating.DAILY_CAP))
        self.assertTrue(ask(facts(), asked_day="2027-01-14", asked_count=rating.DAILY_CAP))
        self.assertEqual(rating.asked(TODAY, TODAY, 2), (TODAY, 3))
        self.assertEqual(rating.asked(TODAY, "2027-01-14", 3), (TODAY, 1))


class PayloadTests(unittest.TestCase):
    def test_a_rating_carries_the_turn(self):
        steps = [rating.step_record("execute_code", "Sketch the base", True, 2.345),
                 rating.step_record("check_fit", "Check fit: 1 problem found", False, 0.0)]
        body = rating.payload(facts(steps=steps, mutated=True), 4, comment="  Vents too small  ", image_png="iVBOR",
                              context={"model": "Origin1"})
        self.assertEqual((body["kind"], body["stars"], body["text"], body["image_png"]),
                         ("rating", 4, "Vents too small", "iVBOR"))
        self.assertEqual(body["steps"][0], {"tool": "execute_code", "title": "Sketch the base", "ok": True,
                                            "seconds": 2.3})
        self.assertEqual(body["steps"][1], {"tool": "check_fit", "title": "Check fit: 1 problem found", "ok": False})
        self.assertEqual(body["context"], {"sampled": True, "outcome": "answered", "changed_model": True,
                                           "model": "Origin1"})
        self.assertNotIn("thought", str(body).lower())

    def test_no_comment_and_no_picture_are_left_out_and_long_texts_are_cut(self):
        body = rating.payload(facts(prompt="x" * 30_000, response="y" * 200_000), 5)
        self.assertNotIn("text", body)
        self.assertNotIn("image_png", body)
        self.assertEqual((len(body["prompt"]), len(body["response"])), (rating.PROMPT_LIMIT, rating.RESPONSE_LIMIT))

    def test_credentials_in_the_texts_are_redacted(self):
        secret = "sk-ant-api03-" + "a" * 40
        steps = [rating.step_record("execute_code", f"Use key {secret}", True, 1.0)]
        body = rating.payload(facts(prompt=f"My key is {secret}", response=f"Got {secret}", steps=steps), 3,
                              comment=f"It printed {secret}")
        self.assertNotIn(secret, str(body))
        self.assertIn("[redacted credential]", body["prompt"])

    def test_stars_are_one_to_five(self):
        for bad in (0, 6):
            with self.assertRaises(ValueError):
                rating.payload(facts(), bad)


class CardTests(unittest.TestCase):
    def test_the_card_offers_five_stars_and_not_now(self):
        key = rating.new_key()
        offered = rating.card_html(key, rating.OFFERED)
        self.assertIn("How did Oscar do on this one?", offered)
        self.assertEqual(offered.count(f"oscar://rate/{key}/"), 6)
        self.assertIn(f"oscar://rate/{key}/later", offered)
        self.assertIn("couldn’t be sent", rating.card_html(key, rating.FAILED))
        self.assertIn("Thanks", rating.card_html(key, rating.SENT))
        self.assertIn("without the picture", rating.card_html(key, rating.SENT_WITHOUT_PICTURE))
        self.assertEqual(rating.card_html(key, rating.DISMISSED), "")

    def test_cards_have_random_keys(self):
        keys = {rating.new_key() for _ in range(50)}
        self.assertEqual(len(keys), 50)
        self.assertTrue(all(len(key) == 16 for key in keys))

    def test_links_are_parsed_strictly(self):
        key = "0123456789abcdef"
        self.assertEqual(rating.parse_anchor(f"oscar://rate/{key}/4"), (key, 4))
        self.assertEqual(rating.parse_anchor(f"oscar://rate/{key}/later"), (key, None))
        for bad in (f"oscar://rate/{key}/9", "oscar://rate/x/4", f"oscar://rate/{key}", f"oscar://rate/{key}/4/5",
                    "https://example.com", "oscar://rate//4", "oscar://rate/12/4"):   # a block number isn't a key
            self.assertIsNone(rating.parse_anchor(bad), bad)


if __name__ == "__main__":
    unittest.main()
