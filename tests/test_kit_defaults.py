"""Board-sized defaults of ``kit.spec.default_spec`` (oscar/kit/_defaults.py; pure, no FreeCAD needed).

Background: the numbers are the Pi 4 / ABS defaults and five of the nine catalogue boards failed validation on
their OWN defaults. The rule now is: the default spec of EVERY catalogue board validates, for every material; the Pi 4 spec is
byte for byte the reference spec; a board that needs adapting changes only the vent / hook / lid-style fields; the result is
deterministic; an unknown board never makes ``default_spec`` raise.
"""

import copy
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.kit import _defaults, materials, spec  # noqa: E402

_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "enclosure")
with open(os.path.join(_FIX, "pi4_spec.json"), encoding="utf-8") as _f:
    CONTRACT = json.load(_f)

BOARDS = spec._known_board_ids()
ADAPTABLE_PREFIXES = ("vents.", "hooks.", "lid.style")


class EveryBoardTest(unittest.TestCase):
    def test_the_catalogue_is_not_empty(self):
        self.assertGreaterEqual(len(BOARDS), 9)
        self.assertIn("rpi4b", BOARDS)

    def test_the_default_spec_of_every_board_validates_for_every_material(self):
        for board in BOARDS:
            for material in materials.names():
                ok, errors, _warnings = spec.validate(spec.default_spec(board, material))
                self.assertTrue(ok, f"{board}/{material}: {errors}")

    def test_defaults_are_buildable_numbers(self):
        for board in BOARDS:
            s = spec.default_spec(board)
            d = spec.derived(s, spec.load_board(board))
            self.assertGreater(d["shell_len"]["value"], 0, board)
            if s.vents.enabled:
                area = spec.vent_area(s, spec.load_board(board))
                span, length, count = spec.vent_extent(s)
                self.assertLessEqual(span, area["v1"] - area["v0"] + 1e-6, board)
                self.assertLessEqual(length, area["u1"] - area["u0"] + 1e-6, board)
                self.assertGreaterEqual(count, 1)
            if s.lid.style == "snap_fit":
                self.assertEqual(len(spec.hook_positions(s, spec.load_board(board))), s.hooks.count, board)

    def test_the_board_id_and_material_are_the_only_other_differences(self):
        base = spec.default_spec("rpi4b", "ABS").to_dict()
        for board in BOARDS:
            d = spec.default_spec(board, "ABS").to_dict()
            for path, (old, new) in _defaults.adapted_fields(board).items():
                self.assertTrue(path.startswith(ADAPTABLE_PREFIXES), f"{board}: {path}")
                self.assertNotEqual(old, new)
            d["board"] = "rpi4b"
            for path in _defaults.adapted_fields(board):
                node, node_base = d, base
                *parents, leaf = path.split(".")
                for key in parents:
                    node, node_base = node[key], node_base[key]
                node[leaf] = node_base[leaf]
            self.assertEqual(d, base, board)

    def test_the_pi4_default_is_the_contract_json(self):
        self.assertEqual(spec.default_spec("rpi4b", "ABS").to_dict(), CONTRACT)
        self.assertEqual(_defaults.adapted_fields("rpi4b"), {})

    def test_boards_that_already_fit_keep_the_contract_numbers(self):
        for board in BOARDS:
            adapted = _defaults.adapted_fields(board)
            if not adapted:
                self.assertEqual({**spec.default_spec(board).to_dict(), "board": "rpi4b"}, CONTRACT, board)

    def test_small_boards_really_are_adapted(self):
        changed = [b for b in BOARDS if _defaults.adapted_fields(b)]
        self.assertGreaterEqual(len(changed), 4, changed)           # zero 2 W, pico, nano, esp32 dev kit at least
        for board in ("rpizero2w", "rpi_pico", "arduino_nano", "esp32_devkitc"):
            self.assertIn(board, changed)
            fields = _defaults.adapted_fields(board)
            self.assertTrue(any(p.startswith("vents.") for p in fields), board)

    def test_the_adapted_vent_array_is_smaller_never_bigger(self):
        base = spec.default_spec("rpi4b").vents
        for board in BOARDS:
            v = spec.default_spec(board).vents
            self.assertLessEqual(v.count, base.count, board)
            self.assertLessEqual(v.slot_l, base.slot_l, board)
            self.assertEqual((v.slot_w, v.pitch), (base.slot_w, base.pitch), board)


class DeterminismTest(unittest.TestCase):
    def test_same_input_same_spec_with_and_without_the_cache(self):
        first = {b: spec.default_spec(b).to_dict() for b in BOARDS}
        _defaults.reset_cache()
        second = {b: spec.default_spec(b).to_dict() for b in BOARDS}
        self.assertEqual(first, second)

    def test_results_are_independent_copies(self):
        a = spec.default_spec("rpi_pico")
        a.hooks.count = 99
        a.vents.count = 99
        b = spec.default_spec("rpi_pico")
        self.assertNotEqual((b.hooks.count, b.vents.count), (99, 99))
        self.assertTrue(spec.validate(b)[0])

    def test_the_cache_is_per_board_and_material(self):
        _defaults.reset_cache()
        spec.default_spec("rpi_pico", "ABS")
        spec.default_spec("rpi_pico", "PLA")
        self.assertEqual(len(_defaults._CACHE), 2)
        spec.default_spec("rpi_pico", "ABS")
        self.assertEqual(len(_defaults._CACHE), 2)


class RobustnessTest(unittest.TestCase):
    def test_an_unknown_board_yields_the_plain_spec_and_never_raises(self):
        s = spec.default_spec("no_such_board")
        self.assertEqual(s.board, "no_such_board")
        self.assertEqual({**s.to_dict(), "board": "rpi4b"}, CONTRACT)
        ok, errors, _w = spec.validate(s)
        self.assertFalse(ok)
        self.assertIn("rpi4b", " ".join(errors))               # the known ids are listed

    def test_board_ids_are_normalised_before_the_lookup(self):
        self.assertEqual(spec.default_spec(" RPI_PICO ").to_dict(), spec.default_spec("rpi_pico").to_dict())

    def test_board_sized_returns_a_spec_that_validates_or_the_input(self):
        s = spec.EnclosureSpec()
        s.board = "rpi_pico"
        sized = _defaults.board_sized(s)
        self.assertTrue(spec.validate(sized)[0])
        pi4 = spec.EnclosureSpec()
        self.assertIs(_defaults.board_sized(pi4), pi4)            # the Pi 4 spec is already right: returned as is

    def test_the_search_shrinks_to_fit_and_falls_back_to_a_friction_lid(self):
        board = copy.deepcopy(spec.load_board("rpi4b"))
        board["outline"]["length"], board["outline"]["width"] = 24.0, 14.0
        d = spec.default_spec("rpi4b").to_dict()
        ok, _errors, _w = spec.validate(d, board)
        self.assertFalse(ok)                                    # the Pi 4 defaults do not fit a 24 x 14 board
        fitted = _defaults._size(copy.deepcopy(d), board)        # the search shrinks vents and hooks until the spec validates
        self.assertTrue(spec.validate(fitted, board)[0])
        self.assertLess(fitted["vents"]["count"], d["vents"]["count"])
        with mock.patch.object(_defaults, "_fit_hooks", return_value=False):      # no hook numbers fit: a friction lid needs none
            fallback = _defaults._size(copy.deepcopy(d), board)
        self.assertEqual(fallback["lid"]["style"], "friction")

    def test_adapted_fields_lists_old_and_new_values(self):
        fields = _defaults.adapted_fields("rpi_pico")
        self.assertIn("vents.count", fields)
        old, new = fields["vents.count"]
        self.assertEqual(old, 9)
        self.assertLess(new, 9)


if __name__ == "__main__":
    unittest.main()
