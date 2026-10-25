"""Searching the panel's chats (oscar.chatsearch): every word as a word prefix, in any order, case- and
accent-insensitive, in the title or in one message. Pure logic; runs under any interpreter."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

from oscar import chatsearch  # noqa: E402

CHATS = [
    ("Raspberry Pi 4 housing", [("user", "A housing for a Pi 4 with vents"), ("assistant", "Built the shell, 2 mm walls.")]),
    ("New chat", [("user", "An M3 bracket, 40 mm long"), ("assistant", "The bracket has two M3 holes.")]),
    ("Résumé holder", [("user", "a stand")]),
]


class FindTest(unittest.TestCase):
    def test_words_are_prefixes_in_any_order(self):
        [found] = chatsearch.find(CHATS, "brack m3")
        self.assertEqual((found.index, found.role), (1, "user"))
        self.assertEqual(found.snippet, "An M3 bracket, 40 mm long")
        self.assertEqual(found.terms, ("brack", "m3"))

    def test_every_word_must_be_in_the_title_or_in_one_message(self):
        self.assertEqual(chatsearch.find(CHATS, "bracket vents"), [])
        self.assertEqual([f.index for f in chatsearch.find(CHATS, "raspberry vents")], [], "title + message do not combine")

    def test_title_matches_come_first_and_case_and_accents_do_not_matter(self):
        self.assertEqual([f.index for f in chatsearch.find(CHATS, "HOUSING")], [0])
        found = chatsearch.find(CHATS, "resume")
        self.assertEqual([(f.index, f.snippet) for f in found], [(2, "")])
        both = [("Shell", [("user", "no match")]), ("Other", [("user", "the shell is thin")])]
        self.assertEqual([f.index for f in chatsearch.find(both, "shell")], [0, 1])

    def test_an_empty_query_finds_nothing_and_results_are_capped(self):
        self.assertEqual(chatsearch.find(CHATS, "  ,. "), [])
        many = [(f"Part {i}", [("user", "bolt")]) for i in range(80)]
        self.assertEqual(len(chatsearch.find(many, "bolt")), chatsearch.MAX_RESULTS)


class SnippetTest(unittest.TestCase):
    def test_a_long_message_is_cut_around_the_first_match(self):
        text = "word " * 40 + "the flange needs six bolt holes " + "more " * 40
        cut = chatsearch.snippet(text, ("flange",))
        self.assertTrue(cut.startswith("…") and cut.endswith("…"), cut)
        self.assertIn("the flange needs", cut)
        self.assertLessEqual(len(cut), chatsearch.SNIPPET_CHARS + 2)

    def test_matched_words_are_marked(self):
        self.assertEqual(chatsearch.mark("An M3 bracket", ("m3", "brack")),
                         [("An ", False), ("M3", True), (" ", False), ("bracket", True)])


if __name__ == "__main__":
    unittest.main()
