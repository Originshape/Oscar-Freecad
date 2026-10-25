"""Regression found by the golden Pi 4 run: a fresh chat inherited the enclosure spec of an EARLIER chat.

``tools_kit_base.load_spec`` fell back to a process-wide cache (keyed by document name, plus a "last spec stored" slot)
whenever the Project had no ``spec.json`` yet. The second Pi 4 build in one session therefore started with the first
build's ``wall_t`` and ``clearance`` instead of the defaults. While a Project is bound it is the only source.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import tools_kit_base as base  # noqa: E402
from oscar.toolctx import ToolContext  # noqa: E402
from support.fakes import temp_project  # noqa: E402


class SpecStoreIsolationTest(unittest.TestCase):
    def setUp(self):
        base.forget_specs()
        self.addCleanup(base.forget_specs)

    def ctx(self, chat_id):
        return ToolContext(project=temp_project(chat_id=chat_id, title="Spec isolation"))

    def test_a_fresh_project_does_not_inherit_the_spec_of_an_earlier_one(self):
        first = self.ctx("20261002-101010-5bec01")
        note = base.store_spec(first, "PiCase", {"wall_t": 2.4, "clearance": 0.35})
        self.assertEqual(note, "")
        self.assertEqual(base.load_spec(first, "PiCase")["wall_t"], 2.4)           # its own project still has it
        second = self.ctx("20261002-101010-5bec02")
        self.assertIsNone(base.load_spec(second, "PiCase"))                        # same document name, another chat
        self.assertIsNone(base.load_spec(second, "OtherDoc"))
        self.assertIsNone(base.load_spec(second))

    def test_the_project_copy_wins_over_the_cache(self):
        ctx = self.ctx("20261002-101010-5bec03")
        base.store_spec(ctx, "PiCase", {"wall_t": 2.4})
        base.store_spec(ToolContext(), "PiCase", {"wall_t": 9.9})                  # a project-less call fills only the cache
        self.assertEqual(base.load_spec(ctx, "PiCase")["wall_t"], 2.4)

    def test_without_a_project_the_cache_still_serves_the_session(self):
        base.store_spec(ToolContext(), "PiCase", {"wall_t": 2.4})
        self.assertEqual(base.load_spec(ToolContext(), "PiCase")["wall_t"], 2.4)
        self.assertEqual(base.load_spec(ToolContext())["wall_t"], 2.4)             # the doc-less "last spec" slot

    def test_a_project_that_cannot_be_read_falls_back_to_the_cache(self):
        class Broken:
            def read_spec(self):
                raise OSError("disk gone")

        base.store_spec(ToolContext(), "PiCase", {"wall_t": 2.4})
        self.assertEqual(base.load_spec(ToolContext(project=Broken()), "PiCase")["wall_t"], 2.4)


if __name__ == "__main__":
    unittest.main()
