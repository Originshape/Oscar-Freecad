"""The shared plumbing of the kit tools (oscar/tools_kit_base.py): formatting, the spec store, ``valid_subset`` (the rescue rule of
``enclosure_spec``), ``merge_validate`` (a raw merge is validated so an unknown key is an error, not silently dropped) and
``ensure_spec``. No document is opened: these are the pure helpers."""

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import tools_kit_base as base  # noqa: E402
from oscar.kit import spec  # noqa: E402
from oscar.toolctx import ToolContext  # noqa: E402
from oscar.tools import ToolError  # noqa: E402

DEFAULT = spec.default_spec().to_dict()


class FakeProject:
    def __init__(self, stored=None, fail=False):
        self.stored, self.fail, self.notes = stored, fail, []

    def read_spec(self):
        return copy.deepcopy(self.stored)

    def save_spec(self, d):
        if self.fail:
            raise RuntimeError("disk full")
        self.stored = copy.deepcopy(d)

    def note(self, kind, text):
        self.notes.append((kind, text))
        return f"N{len(self.notes)}"


class FormatTest(unittest.TestCase):
    def test_fmt(self):
        self.assertEqual(base.fmt(2.4), "2.4")
        self.assertEqual(base.fmt(2.0), "2")
        self.assertEqual(base.fmt(94.9), "94.9")
        self.assertEqual(base.fmt(0.0004), "0")
        self.assertEqual(base.fmt(-0.0), "0")
        self.assertEqual(base.fmt(1 / 3), "0.333")
        self.assertEqual(base.fmt("abc"), "abc")
        self.assertEqual(base.fmt(None), "None")

    def test_limit_list(self):
        self.assertEqual(base.limit_list(["a", "b"]), "a, b")
        self.assertEqual(base.limit_list([str(i) for i in range(9)], 3), "0, 1, 2 and 6 more")
        self.assertEqual(base.limit_list([]), "")

    def test_suggest(self):
        self.assertEqual(base.suggest("wal_t", ["wall_t", "floor_t"]), " (similar: wall_t)")
        self.assertEqual(base.suggest("zzzz", ["wall_t"]), "")

    def test_flatten_nest_and_diff(self):
        d = {"a": {"b": 1, "c": [1, 2]}, "d": 2}
        self.assertEqual(base.flatten(d), {"a.b": 1, "a.c": [1, 2], "d": 2})
        self.assertEqual(base.nest("a.b", 3), {"a": {"b": 3}})
        self.assertEqual(sorted(base.diff(d, {"a": {"b": 2, "c": [1, 2]}, "d": 2, "e": 5})), [("a.b", 1, 2), ("e", None, 5)])
        self.assertEqual(base.diff(d, d), [])


class SpecStoreTest(unittest.TestCase):
    def setUp(self):
        base.forget_specs()
        self.addCleanup(base.forget_specs)

    def test_the_project_is_the_source_of_truth(self):
        project = FakeProject({"board": "rpi4b", "wall_t": 2.4})
        ctx = ToolContext(project=project)
        self.assertEqual(base.load_spec(ctx, "Doc")["wall_t"], 2.4)
        base.store_spec(ctx, "Doc", {"board": "rpi4b", "wall_t": 3.0})
        self.assertEqual(project.stored["wall_t"], 3.0)
        self.assertEqual(base.load_spec(ctx, "Doc")["wall_t"], 3.0)

    def test_without_a_project_the_cache_serves_by_document_then_globally(self):
        ctx = ToolContext()
        self.assertIsNone(base.load_spec(ctx, "A"))
        base.store_spec(ctx, "A", {"wall_t": 2.2})
        self.assertEqual(base.load_spec(ctx, "A")["wall_t"], 2.2)
        self.assertEqual(base.load_spec(ctx, "other")["wall_t"], 2.2)     # a tool call without a doc name still finds the spec
        base.store_spec(ctx, "B", {"wall_t": 2.8})
        self.assertEqual(base.load_spec(ctx, "A")["wall_t"], 2.2)
        self.assertEqual(base.load_spec(ctx, "B")["wall_t"], 2.8)

    def test_loaded_specs_are_copies(self):
        ctx = ToolContext()
        base.store_spec(ctx, "A", {"wall_t": 2.2, "hooks": {"count": 4}})
        got = base.load_spec(ctx, "A")
        got["hooks"]["count"] = 99
        self.assertEqual(base.load_spec(ctx, "A")["hooks"]["count"], 4)

    def test_a_failing_project_write_is_reported_not_raised(self):
        ctx = ToolContext(project=FakeProject(fail=True))
        note = base.store_spec(ctx, "A", {"wall_t": 2.2})
        self.assertIn("could not be written to the project", note)
        self.assertIn("disk full", note)
        self.assertEqual(base.load_spec(ToolContext(), "A")["wall_t"], 2.2)       # the process copy survived

    def test_an_empty_project_spec_counts_as_none(self):
        self.assertIsNone(base.load_spec(ToolContext(project=FakeProject({})), "A"))

    def test_the_cache_is_bounded(self):
        ctx = ToolContext()
        for i in range(base._MAX_CACHE + 10):
            base.store_spec(ctx, f"doc{i}", {"wall_t": 2.0})
        self.assertLessEqual(len(base._SPECS), base._MAX_CACHE + 1)      # +1: the "" slot
        self.assertIsNotNone(base.load_spec(ctx, f"doc{base._MAX_CACHE + 9}"))

    def test_reference_status_prefers_the_project_then_the_ledger(self):
        class Holder:
            def __init__(self, status):
                self.reference_status = status

        self.assertEqual(base.reference_status(ToolContext()), "none")
        self.assertEqual(base.reference_status(ToolContext(ledger=Holder("catalog-nominal"))), "catalog-nominal")
        self.assertEqual(base.reference_status(ToolContext(project=Holder("step-measured"), ledger=Holder("catalog-nominal"))), "step-measured")
        self.assertEqual(base.reference_status(ToolContext(project=Holder(""), ledger=Holder("step-imported"))), "step-imported")

    def test_record_assumption_needs_a_project_and_never_raises(self):
        self.assertEqual(base.record_assumption(ToolContext(), "x"), "")
        project = FakeProject()
        self.assertEqual(base.record_assumption(ToolContext(project=project), "x"), "N1")
        self.assertEqual(project.notes, [("assumption", "x")])


class ValidationHelpersTest(unittest.TestCase):
    def test_merge_validate_accepts_a_good_patch_and_returns_the_normalized_spec(self):
        candidate, errors, _warnings = base.merge_validate(DEFAULT, {"wall_t": "2.4 mm"})
        self.assertEqual(errors, [])
        self.assertEqual(candidate["wall_t"], 2.4)

    def test_merge_validate_reports_unknown_keys_with_a_suggestion(self):
        candidate, errors, _w = base.merge_validate(DEFAULT, {"wal_t": 2.4})
        self.assertIsNone(candidate)
        self.assertIn("wall_t", " ".join(errors))

    def test_merge_validate_reports_rule_violations(self):
        candidate, errors, _w = base.merge_validate(DEFAULT, {"wall_t": 0.5})
        self.assertIsNone(candidate)
        self.assertIn("1.2", " ".join(errors))

    def test_merge_validate_survives_a_patch_of_the_wrong_shape(self):
        candidate, errors, _w = base.merge_validate(DEFAULT, {"hooks": 5})
        self.assertIsNone(candidate)
        self.assertTrue(errors)

    def test_spec_errors_never_raises(self):
        ok, errors, _w = base.spec_errors(object())
        self.assertFalse(ok)
        self.assertTrue(errors)

    def test_valid_subset_keeps_good_leaves_and_drops_bad_ones_with_reasons(self):
        final, kept, dropped = base.valid_subset(DEFAULT, {"wall_t": 0.2, "clearance": 0.4, "hooks": {"count": 2}, "nonsense": 1})
        self.assertEqual((final["wall_t"], final["clearance"], final["hooks"]["count"]), (2.0, 0.4, 2))
        self.assertEqual(sorted(kept), ["clearance", "hooks.count"])
        self.assertEqual(sorted(d.split(":")[0] for d in dropped), ["nonsense", "wall_t"])
        self.assertIn("1.2", next(d for d in dropped if d.startswith("wall_t")))

    def test_valid_subset_with_an_empty_patch_is_the_base(self):
        self.assertEqual(base.valid_subset(DEFAULT, {}), (DEFAULT, [], []))

    def test_valid_subset_does_not_blame_a_leaf_for_an_error_the_base_already_had(self):
        broken = copy.deepcopy(DEFAULT)
        broken["wall_t"] = 0.2
        final, kept, dropped = base.valid_subset(broken, {"clearance": 0.4})
        self.assertEqual(kept, ["clearance"])
        self.assertEqual(dropped, [])
        self.assertEqual(final["wall_t"], 0.2)

    def test_normalized_coerces_numbers_and_ids(self):
        d = base.normalized({"wall_t": "2.4", "board": "RPI4B"})
        self.assertEqual((d["wall_t"], d["board"]), (2.4, "rpi4b"))

    def test_default_spec_dict_validates_for_every_board(self):
        for board in base.known_boards():
            self.assertTrue(spec.validate(base.default_spec_dict(board))[0], board)

    def test_load_board_lists_the_known_ones_for_a_typo(self):
        with self.assertRaises(ToolError) as cm:
            base.load_board("rpi44", "enclosure_spec")
        self.assertIn("enclosure_spec", str(cm.exception))
        self.assertIn("rpi4b", str(cm.exception))
        self.assertEqual(base.load_board("rpi4b", "x")["id"], "rpi4b")

    def test_spec_summary_has_the_numbers_a_model_needs(self):
        text = base.spec_summary(DEFAULT)
        for part in ("board 'rpi4b'", "ABS", "wall 2 mm", "clearance 0.3 mm", "snap_fit lid", "shell 94.1 x 61.94 x 24 mm"):
            self.assertIn(part, text)
        self.assertIn("board 'nonexistent'", base.spec_summary({"board": "nonexistent", "material": "ABS", "wall_t": 2, "clearance": 0.3, "lid": {"style": "screw"}}))


class EnsureSpecTest(unittest.TestCase):
    def setUp(self):
        base.forget_specs()
        self.addCleanup(base.forget_specs)

    def test_an_existing_spec_is_returned_untouched_without_a_note(self):
        project = FakeProject({"board": "rpi4b", "wall_t": 2.4})
        d, note = base.ensure_spec(ToolContext(project=project), "Doc", "build_enclosure")
        self.assertEqual((d["wall_t"], note), (2.4, ""))
        self.assertEqual(project.notes, [])

    def test_a_missing_spec_gets_the_board_defaults_stored_and_recorded(self):
        project = FakeProject()
        d, note = base.ensure_spec(ToolContext(project=project), "Doc", "build_enclosure", board_id="rpi_pico")
        self.assertEqual(d["board"], "rpi_pico")
        self.assertIn("defaults for board 'rpi_pico'", note)
        self.assertEqual(project.stored["board"], "rpi_pico")
        self.assertEqual(project.notes[0][0], "assumption")
        self.assertTrue(spec.validate(d)[0])

    def test_an_unknown_board_is_a_tool_error(self):
        with self.assertRaises(ToolError):
            base.ensure_spec(ToolContext(), "Doc", "add_cutout", board_id="nonexistent")


class ResolveBoardTest(unittest.TestCase):
    """``resolve_board``: a weak model writes the board's name, not its id (review of oscar/tools_kit_base.py)."""

    def test_ids_names_and_aliases_resolve_to_the_catalog_id(self):
        for text in ("rpi4b", "RPI4B", "Raspberry Pi 4 Model B", "raspberry pi 4", "RPi 4", "rpi-4", "  pi4  ", "Pi_4", "raspberry   pi   4b"):
            self.assertEqual(base.resolve_board(text), "rpi4b", text)
        self.assertEqual(base.resolve_board("Raspberry Pi Pico"), "rpi_pico")
        self.assertEqual(base.resolve_board("RPi Pico"), "rpi_pico")
        self.assertEqual(base.resolve_board("Arduino Uno"), "arduino_uno")

    def test_an_unknown_or_partial_name_is_not_guessed(self):
        for text, expected in (("Raspberry Pi", "raspberry pi"), ("rpi9", "rpi9"), ("../etc", "../etc"), ("Pi 4 Model", "pi 4 model")):
            self.assertEqual(base.resolve_board(text), expected, text)

    def test_empty_and_odd_input_never_raises(self):
        self.assertEqual(base.resolve_board(None), "")
        self.assertEqual(base.resolve_board(""), "")
        self.assertEqual(base.resolve_board("   "), "")
        self.assertEqual(base.resolve_board(5), "5")

    def test_a_name_that_two_boards_share_is_left_for_the_validator(self):
        from oscar.refparts import catalog

        fake = [{"id": "a_board", "name": "Alpha", "aliases": ["shared"]}, {"id": "b_board", "name": "Beta", "aliases": ["Shared"]}]
        original = catalog.list_boards
        catalog.list_boards = lambda: fake
        try:
            self.assertEqual(base.resolve_board("shared"), "shared")
            self.assertEqual(base.resolve_board("Alpha"), "a_board")
        finally:
            catalog.list_boards = original


if __name__ == "__main__":
    unittest.main()
