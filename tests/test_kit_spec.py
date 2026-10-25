"""Tests for oscar.kit.spec / oscar.kit.materials (pure: no FreeCAD needed).

Covers: the default JSON, merge / from_dict coercion, EVERY row of the validation table,
the stage hashes ("an edit changes exactly the stages of its row"), the derived Pi 4 numbers, connector openings, hook
placement and strain, and the materials table. The board is the byte-identical copy of the board JSON in
tests/fixtures/enclosure so the numbers do not move when the board library does.
"""

import ast
import copy
import json
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.kit import materials, spec  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIX = os.path.join(_HERE, "fixtures", "enclosure")
_KIT = os.path.join(_HERE, "..", "oscar", "kit")


def _load(name: str) -> dict:
    with open(os.path.join(_FIX, name), encoding="utf-8") as f:
        return json.load(f)


BOARD = _load("rpi4b_board.json")
DEFAULT = _load("pi4_spec.json")
UNVERIFIED = "board 'rpi4b': dimensions are not yet verified against the drawing"


def patched(**changes) -> dict:
    """The default spec dict with dotted-path changes applied (``patched(**{"hooks.t": 1.5})``)."""
    d = copy.deepcopy(DEFAULT)
    for path, value in changes.items():
        cur = d
        keys = path.split(".")
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = value
    return d


def check(d, board=BOARD):
    return spec.validate(d, board)


class _Base(unittest.TestCase):
    def errors(self, d, board=BOARD) -> list:
        return check(d, board)[1]

    def warnings(self, d, board=BOARD) -> list:
        return check(d, board)[2]

    def assertErr(self, d, *needles, board=BOARD):
        ok, errors, _w = check(d, board)
        self.assertFalse(ok, f"expected an error containing {needles}, got none")
        joined = "\n".join(errors)
        for n in needles:
            self.assertIn(n, joined)
        return errors

    def assertNoErr(self, d, board=BOARD):
        ok, errors, _w = check(d, board)
        self.assertTrue(ok, errors)

    def assertWarn(self, d, *needles, board=BOARD):
        ok, errors, warnings = check(d, board)
        self.assertTrue(ok, errors)
        joined = "\n".join(warnings)
        for n in needles:
            self.assertIn(n, joined)

    def assertNoWarn(self, d, needle, board=BOARD):
        _ok, _e, warnings = check(d, board)
        self.assertFalse([w for w in warnings if needle in w], warnings)


# --------------------------------------------------------------------------------------------------------------------
class MaterialsTest(unittest.TestCase):
    """The material numbers, FDM_DERATE, lookup rules."""

    def test_table_matches_decision_18(self):
        want = {"ABS": (0.008, 1.2, 1.2, 0.020), "ASA": (0.007, 1.2, 1.2, 0.020), "PLA": (0.003, 1.0, 1.0, 0.015),
                "PETG": (0.004, 1.2, 1.2, 0.030), "PC": (0.007, 1.2, 1.2, 0.025), "NYLON": (0.015, 1.2, 1.2, 0.050)}
        self.assertEqual(list(materials.MATERIALS), list(want))
        for name, (shrink, wall, web, strain) in want.items():
            row = materials.MATERIALS[name]
            self.assertEqual((row["shrink"], row["min_wall"], row["min_web"], row["max_strain"]), (shrink, wall, web, strain), name)
            self.assertEqual(set(row), {"shrink", "min_wall", "min_web", "max_strain"})

    def test_fdm_derate_and_limit(self):
        self.assertEqual(materials.FDM_DERATE, 0.5)
        self.assertAlmostEqual(materials.hook_strain_limit("ABS"), 0.010)
        self.assertAlmostEqual(materials.hook_strain_limit("ABS", molded=True), 0.020)
        self.assertAlmostEqual(materials.hook_strain_limit("nylon"), 0.025)

    def test_normalize_get_is_known(self):
        self.assertEqual(materials.normalize(" abs "), "ABS")
        self.assertEqual(materials.normalize(5), 5)
        self.assertTrue(materials.is_known("petg"))
        self.assertFalse(materials.is_known("WOOD"))
        self.assertFalse(materials.is_known(None))
        self.assertEqual(materials.get("abs")["min_wall"], 1.2)
        self.assertEqual(materials.names(), list(materials.MATERIALS))

    def test_get_returns_a_copy(self):
        row = materials.get("ABS")
        row["min_wall"] = 99
        self.assertEqual(materials.MATERIALS["ABS"]["min_wall"], 1.2)

    def test_unknown_material_suggests(self):
        with self.assertRaises(KeyError) as cm:
            materials.get("ABZ")
        self.assertIn("similar: ABS", cm.exception.args[0])
        self.assertIn("known: ABS, ASA, PLA, PETG, PC, NYLON", cm.exception.args[0])
        with self.assertRaises(KeyError):
            materials.get(None)

    def test_shrink_pct(self):
        self.assertEqual(materials.shrink_pct("ABS"), 0.8)
        self.assertEqual(materials.shrink_pct("nylon"), 1.5)


# --------------------------------------------------------------------------------------------------------------------
class PurityTest(unittest.TestCase):
    """Pure modules import no FreeCAD / Qt."""

    def test_no_freecad_or_qt_imports(self):
        banned = {"FreeCAD", "FreeCADGui", "PySide", "PySide2", "PySide6", "Part", "App", "Gui"}
        for fname in ("spec.py", "_validate.py", "materials.py", "checks.py", "params.py", "__init__.py"):
            with open(os.path.join(_KIT, fname), encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in tree.body:   # top-level statements only: lazy imports inside functions are allowed
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                self.assertFalse(set(names) & banned, f"{fname} imports {names} at module level")

    def test_kit_package_is_lazy(self):
        import oscar.kit as kit
        self.assertIs(kit.spec, spec)
        self.assertIs(kit.materials, materials)
        with self.assertRaises(AttributeError):
            kit.no_such_submodule  # noqa: B018
        self.assertIn("spec", dir(kit))


# --------------------------------------------------------------------------------------------------------------------
class DefaultsAndConversionTest(_Base):
    def test_default_spec_equals_the_contract_json(self):
        d = spec.default_spec("rpi4b", "ABS").to_dict()
        self.assertEqual(d, DEFAULT)
        self.assertEqual(list(d), list(DEFAULT), "top-level key order is the contract's")
        for group in ("standoff", "lid", "hooks", "vents", "ports"):
            self.assertEqual(list(d[group]), list(DEFAULT[group]), group)
        self.assertEqual(spec.to_dict(spec.default_spec()), DEFAULT)
        json.dumps(d)   # JSON-able

    def test_roundtrip_and_independence(self):
        s = spec.from_dict(DEFAULT)
        self.assertEqual(s, spec.default_spec())
        self.assertEqual(spec.to_dict(s), DEFAULT)
        a, b = spec.default_spec(), spec.default_spec()
        a.ports.include.append("hdmi0")
        a.colors["shell"][0] = 0.9
        self.assertEqual(b.ports.include, [])
        self.assertEqual(spec.default_spec().to_dict(), DEFAULT, "defaults are not shared mutable state")

    def test_default_spec_board_and_material_normalisation(self):
        s = spec.default_spec(" RPI5 ", "pla")
        self.assertEqual((s.board, s.material), ("rpi5", "PLA"))
        self.assertEqual(s.wall_t, 2.0)

    def test_from_dict_merges_onto_the_defaults_of_the_named_board_and_material(self):
        s = spec.from_dict({"board": "RPi4B", "material": "petg", "wall_t": "2.4 mm", "hooks": {"count": "6"}})
        self.assertEqual((s.board, s.material, s.wall_t, s.hooks.count, s.hooks.w), ("rpi4b", "PETG", 2.4, 6, 6.0))

    def test_from_dict_is_lenient_and_never_raises(self):
        self.assertEqual(spec.from_dict({}), spec.default_spec())
        self.assertEqual(spec.from_dict(None), spec.default_spec())            # type: ignore[arg-type]
        self.assertEqual(spec.from_dict({"nonsense": 1, "hooks": {"zzz": 2}}), spec.default_spec())
        s = spec.from_dict({"wall_t": "thick", "hooks": "many", "board": 7, "material": None})
        self.assertEqual(s.wall_t, "thick")            # kept as given: validate() reports it
        self.assertEqual(s.hooks, spec.default_spec().hooks)
        self.assertEqual((s.board, s.material), (7, None))         # kept as given: validate() reports 'expected text'

    def test_coercions(self):
        s = spec.from_dict({"clearance": "0.35 mm", "hooks": {"ramp_deg": "30 deg", "retention_deg": 45.0},
                            "vents": {"enabled": "yes", "center": [10, "20"], "pattern": " Holes "},
                            "colors": {"shell": ["0.1", 0.2, 0.3]}, "ports": {"include": "HDMI0 ", "exclude": ["Audio"]}})
        self.assertEqual(s.clearance, 0.35)
        self.assertEqual((s.hooks.ramp_deg, s.hooks.retention_deg), (30, 45))
        self.assertIsInstance(s.hooks.retention_deg, int)
        self.assertEqual((s.vents.enabled, s.vents.center, s.vents.pattern), (True, [10.0, 20.0], "holes"))
        self.assertEqual(s.colors["shell"], [0.1, 0.2, 0.3])
        self.assertEqual((s.ports.include, s.ports.exclude), (["hdmi0"], ["audio"]))

    def test_bool_is_not_a_number(self):
        self.assertEqual(spec.from_dict({"wall_t": True}).wall_t, True)
        self.assertErr({"wall_t": True}, "wall_t: expected a number")

    def test_merge_deep_and_pure(self):
        base = {"a": {"x": 1, "y": [1, 2]}, "b": 1, "c": None}
        patch = {"a": {"y": [3]}, "b": None, "c": {"z": 1}, "d": 4}
        out = spec.merge(base, patch)
        self.assertEqual(out, {"a": {"x": 1, "y": [3]}, "b": None, "c": {"z": 1}, "d": 4})
        self.assertEqual(base, {"a": {"x": 1, "y": [1, 2]}, "b": 1, "c": None}, "inputs are not mutated")
        out["a"]["y"].append(9)
        self.assertEqual(patch["a"]["y"], [3], "the result shares nothing with the patch")
        self.assertEqual(spec.merge({"a": 1}, None), {"a": 1})                 # type: ignore[arg-type]
        self.assertEqual(spec.merge(DEFAULT, {"vents": {"center": [1, 2]}})["vents"]["center"], [1, 2])

    def test_a_dict_with_to_dict_input(self):
        d = patched(wall_t=2.4)
        self.assertEqual(spec.to_dict(d), d)
        self.assertIsNot(spec.to_dict(d), d)


# --------------------------------------------------------------------------------------------------------------------
class ValidationTableTest(_Base):
    """One test per row of the validation table, positive and negative, with the boundary values."""

    def test_the_default_spec_has_exactly_one_warning(self):
        ok, errors, warnings = check(spec.default_spec())
        self.assertTrue(ok, errors)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [UNVERIFIED])

    def test_default_through_the_catalog(self):
        cat = spec._catalog()
        if cat is None:
            self.skipTest("refparts.catalog not installed")
        ok, errors, warnings = spec.validate(spec.default_spec())          # board loaded through refparts.catalog
        self.assertTrue(ok, errors)
        self.assertEqual(errors, [])
        self.assertLessEqual(len(warnings), 1)
        self.assertEqual(spec.load_board("rpi4b")["id"], "rpi4b")

    def test_spec_object_and_dict_validate_identically(self):
        self.assertEqual(check(spec.default_spec()), check(DEFAULT))
        d = patched(wall_t=1.0)
        self.assertEqual(check(spec.from_dict(d)), check(d))

    # name / kit_id / version
    def test_name(self):
        for bad in ("1abc", "", "a" * 25, "has space", "dash-ed", "ünï"):
            self.assertErr(patched(name=bad), "name:")
        for good in ("A", "Case_1", "a" * 24):
            self.assertNoErr(patched(name=good))

    def test_kit_id_and_version(self):
        self.assertErr(patched(kit_id="9x"), "kit_id:")
        self.assertErr(patched(kit_id="a" * 17), "kit_id:")
        self.assertNoErr(patched(kit_id="enc_2"))
        self.assertErr(patched(version=2), "version: 2 is not supported")

    # board
    def test_board_unknown_lists_known_ids(self):
        cat = spec._catalog()
        if cat is None:
            self.skipTest("refparts.catalog not installed")
        errors = self.assertErr(patched(board="nosuchboard"), "board: 'nosuchboard' is not in the board library", "known:",
                                "Use find_reference_part.", board=None)
        self.assertIn("rpi4b", "\n".join(errors))

    def test_board_data_for_another_board_is_an_error(self):
        self.assertErr(patched(board="rpi3bplus"), "board: the spec names 'rpi3bplus' but the board data passed is 'rpi4b'")

    def test_board_empty_and_mistyped(self):
        self.assertErr(patched(board=""), "board: is empty")
        self.assertErr(patched(board=7), "board: expected text")

    def test_unverified_board_warns_and_verified_does_not(self):
        self.assertIn(UNVERIFIED, self.warnings(DEFAULT))
        verified = dict(BOARD, verified=True)
        self.assertEqual(self.warnings(DEFAULT, verified), [])

    def test_board_library_missing_skips_board_rules_with_one_warning(self):
        original = spec._catalog
        spec._catalog = lambda: None
        try:
            ok, errors, warnings = spec.validate(patched(**{"standoff.h": 2.0}))      # the microSD rule needs the board: skipped
            self.assertTrue(ok, errors)
            self.assertEqual(len([w for w in warnings if "board library is not available" in w]), 1)
            with self.assertRaises(LookupError):
                spec.load_board("rpi4b")
        finally:
            spec._catalog = original

    def test_broken_board_file_is_reported_not_raised(self):
        original = spec.load_board
        spec.load_board = lambda _id: (_ for _ in ()).throw(ValueError("bad json"))
        try:
            ok, errors, _w = spec.validate(DEFAULT)
        finally:
            spec.load_board = original
        self.assertFalse(ok)
        self.assertIn("could not be read (ValueError: bad json)", "\n".join(errors))

    # material
    def test_material(self):
        self.assertErr(patched(material="WOOD"), "material: 'WOOD' is not supported", "ABS, ASA, PLA, PETG, PC, NYLON")
        for m in materials.MATERIALS:
            self.assertNoErr(patched(material=m))
        self.assertNoErr(patched(material="abs"))            # normalised, not rejected
        self.assertErr(patched(material=3), "material: expected text")

    # wall_t, floor_t, lid_t
    def test_wall_t_boundaries(self):
        self.assertErr(patched(wall_t=1.1), "wall_t: 1.1 is below the minimum 1.2 mm for ABS; use 1.2 or more (default 2.0).")
        screw = {"lid.style": "screw"}          # a snap-fit lid also needs 1.2 mm of wall behind the catch pocket (wall_t >= 1.85)
        self.assertNoErr(patched(wall_t=1.2, **screw))
        self.assertErr(patched(wall_t=1.2), "catch pocket")
        self.assertWarn(patched(wall_t=1.2, **screw), "wall_t: 1.2 is thin for FDM")
        self.assertWarn(patched(wall_t=1.59, **screw), "wall_t")
        self.assertNoWarn(patched(wall_t=1.6, **screw), "wall_t:")
        self.assertNoErr(patched(wall_t=6.0, corner_r=6.0))
        self.assertErr(patched(wall_t=6.1), "wall_t: 6.1 is above the maximum 6.0 mm")

    def test_wall_minimum_follows_the_material(self):
        screw = {"lid.style": "screw"}
        self.assertNoErr(patched(material="PLA", wall_t=1.0, **screw))
        self.assertErr(patched(material="PLA", wall_t=0.9, **screw), "minimum 1.0 mm for PLA")
        self.assertErr(patched(material="ABS", wall_t=1.0, **screw), "minimum 1.2 mm for ABS")

    def test_floor_and_lid_thickness(self):
        for path in ("floor_t", "lid_t"):
            self.assertErr(patched(**{path: 1.1}), f"{path}: 1.1 is below the minimum 1.2")
            self.assertErr(patched(**{path: 6.5}), f"{path}: 6.5 is above the maximum")
            self.assertNoErr(patched(**{path: 1.2}))
            self.assertNoWarn(patched(**{path: 1.2}), f"{path}:")      # only wall_t warns when thin

    # clearance / shrink_comp
    def test_clearance(self):
        self.assertErr(patched(clearance=0.09), "clearance: 0.09 is below the minimum 0.1")
        self.assertWarn(patched(clearance=0.1), "tight for FDM")
        self.assertNoWarn(patched(clearance=0.15), "clearance:")
        self.assertNoErr(patched(clearance=2.0))
        self.assertErr(patched(clearance=2.01), "clearance: 2.01 is above the maximum 2.0")

    def test_shrink_comp(self):
        self.assertErr(patched(shrink_comp="scale"), "'scale' is not supported yet; use 'clearance' or 'none'")
        self.assertErr(patched(shrink_comp="bogus"), "shrink_comp: 'bogus' is not a valid mode")
        self.assertNoErr(patched(shrink_comp="none"))
        self.assertNoErr(patched(shrink_comp="clearance"))
        self.assertNoErr(patched(shrink_comp=" None "))

    # corner_r
    def test_corner_r(self):
        self.assertErr(patched(corner_r=-0.1), "corner_r: -0.1 is negative")
        self.assertNoErr(patched(corner_r=14.0, **{"lid.style": "screw"}))      # (a snap-fit lid has no room for hooks on a 12 mm cavity corner)
        self.assertErr(patched(corner_r=14.0), "hooks.count: only 3 positions are free")
        self.assertErr(patched(corner_r=14.5), "corner_r: 14.5 is above the maximum 14.0")
        self.assertErr(patched(corner_r=3.0, wall_t=4.0), "corner_r: 3.0 is smaller than wall_t 4.0")
        self.assertNoErr(patched(corner_r=2.0, wall_t=2.0))                       # cavity corner radius exactly 0

    # standoff
    def test_standoff_height_microsd_rule_and_message(self):
        errors = self.assertErr(patched(**{"standoff.h": 1.5}), "standoff.h: 1.5 leaves less than 1.0 mm under the board for the microSD socket (underside z -1.5); use 2.5 or more")
        self.assertEqual(len([e for e in errors if e.startswith("standoff.h")]), 1)
        self.assertErr(patched(**{"standoff.h": 2.4}), "use 2.5 or more")
        self.assertNoErr(patched(**{"standoff.h": 2.5}))

    def test_standoff_height_minimum_without_underside_parts(self):
        flat = copy.deepcopy(BOARD)
        flat["height"]["min_bottom_z"] = 0.0
        self.assertErr(patched(**{"standoff.h": 1.9}), "standoff.h: 1.9 is below the minimum 2.0 mm; use 2.0 or more", board=flat)
        self.assertNoErr(patched(**{"standoff.h": 2.0}), board=flat)

    def test_standoff_diameter(self):
        self.assertErr(patched(**{"standoff.d": 3.9}), "standoff.d: 3.9 is below the minimum 4.0")
        self.assertErr(patched(**{"standoff.d": 10.1}), "standoff.d: 10.1 is above the maximum 10.0")
        self.assertNoWarn(patched(**{"standoff.d": 6.2}), "standoff.d")        # equal to the 6.2 pad: fine
        self.assertNoWarn(DEFAULT, "standoff.d")                                # the default 6.0 does not warn
        self.assertWarn(patched(**{"standoff.d": 6.3}), "standoff.d: 6.3 is larger than the board's hole pad 6.2 mm")
        self.assertNoErr(patched(**{"standoff.d": 4.0, "standoff.pilot_d": 2.2}))

    def test_standoff_pilot(self):
        self.assertErr(patched(**{"standoff.pilot_d": 1.4}), "standoff.pilot_d: 1.4 is below the minimum 1.5")
        self.assertErr(patched(**{"standoff.pilot_d": 5.0}), "standoff.pilot_d: 5.0 is too large for a 6.0 mm standoff")
        self.assertNoErr(patched(**{"standoff.pilot_d": 4.9}))
        self.assertWarn(patched(**{"standoff.pilot_d": 2.3}), "standoff.pilot_d: 2.3 is more than 90% of the M2.5 thread")
        self.assertNoWarn(patched(**{"standoff.pilot_d": 2.25}), "standoff.pilot_d")
        self.assertNoWarn(DEFAULT, "standoff.pilot_d")

    # lid
    def test_lid_rows(self):
        self.assertErr(patched(**{"lid.style": "magnet"}), "lid.style: 'magnet' is not a lid style")
        for style in ("snap_fit", "screw", "friction"):
            self.assertNoErr(patched(**{"lid.style": style}))
        self.assertErr(patched(**{"lid.gap": 0.49}), "lid.gap: 0.49 is below the minimum 0.5")
        self.assertNoErr(patched(**{"lid.gap": 0.5}))
        self.assertErr(patched(**{"lid.lip_h": 1.9}), "lid.lip_h: 1.9 is below the minimum 2.0")
        self.assertNoErr(patched(**{"lid.lip_h": 2.0}))
        self.assertErr(patched(**{"lid.lip_t": 0.7}), "lid.lip_t: 0.7 is below the minimum 0.8")
        self.assertNoErr(patched(**{"lid.lip_t": 0.8, "hooks.t": 0.8}))

    def test_lid_fit(self):
        self.assertErr(patched(**{"lid.fit": 0.09}), "lid.fit: 0.09 is below the minimum 0.1")
        self.assertErr(patched(**{"lid.fit": 1.1}), "lid.fit: 1.1 is above the maximum 1.0")
        self.assertWarn(patched(**{"lid.fit": 0.1}), "lid.fit: 0.1 is tight")
        self.assertNoWarn(patched(**{"lid.fit": 0.15}), "lid.fit")

    # hooks
    def test_hook_count(self):
        for bad in (0, 1, 3, 5, 10, -2):
            self.assertErr(patched(**{"hooks.count": bad}), "hooks.count:")
        for good in (2, 4, 6):
            self.assertNoErr(patched(**{"hooks.count": good}))
        self.assertErr(patched(**{"hooks.count": 2.5}), "hooks.count: expected a whole number")

    def test_hook_count_limited_by_free_wall_positions(self):
        errors = self.assertErr(patched(**{"hooks.count": 8}), "hooks.count: only 7 positions are free of port openings on 'rpi4b'; reduce count or move ports")
        self.assertEqual(len(errors), 1)
        self.assertNoErr(patched(**{"hooks.count": 8, "ports.exclude": ["usb_c_power", "hdmi0", "hdmi1", "audio"]}))

    def test_hook_dimensions(self):
        self.assertErr(patched(**{"hooks.w": 0}), "hooks.w: 0.0 must be greater than 0")
        self.assertErr(patched(**{"hooks.t": 0.7}), "hooks.t: 0.7 is below the minimum 0.8")
        self.assertErr(patched(**{"hooks.overhang": 0.3}), "hooks.overhang: 0.3 is outside 0.4..1.5")
        self.assertErr(patched(**{"hooks.overhang": 1.6}), "hooks.overhang: 1.6 is outside 0.4..1.5")
        self.assertErr(patched(**{"hooks.barb_h": 0.5}), "hooks.barb_h: 0.5 is outside 0.6..2.0")
        self.assertErr(patched(**{"hooks.barb_h": 2.1}), "hooks.barb_h: 2.1 is outside 0.6..2.0")
        self.assertNoErr(patched(**{"hooks.barb_h": 0.6}))
        self.assertNoErr(patched(**{"hooks.barb_h": 2.0}))

    def test_hook_overhang_must_exceed_the_fit(self):
        self.assertErr(patched(**{"lid.fit": 0.8}), "hooks.overhang: 0.8 does not exceed lid.fit 0.8; the barb must reach past the gap")
        self.assertErr(patched(**{"lid.fit": 0.9, "hooks.overhang": 0.8}), "does not exceed lid.fit 0.9")

    def test_hook_length(self):
        self.assertErr(patched(**{"hooks.length": 4.9}), "hooks.length: 4.9 is below the minimum 5.0")
        self.assertNoErr(patched(**{"hooks.length": 19.0, "hooks.t": 1.2}))
        self.assertErr(patched(**{"hooks.length": 19.5}), "hooks.length: 19.5 is longer than the 19.0 mm of wall below the skirt; use 19.0 or less")

    def test_hook_window_margin(self):
        self.assertErr(patched(**{"hooks.window_margin": 0.05}), "hooks.window_margin: 0.05 is outside 0.1..0.4")
        self.assertErr(patched(**{"hooks.window_margin": 0.45}), "the retention check lifts the lid 0.5 mm")
        self.assertNoErr(patched(**{"hooks.window_margin": 0.1}))
        self.assertNoErr(patched(**{"hooks.window_margin": 0.4}))

    def test_hook_angles(self):
        self.assertErr(patched(**{"hooks.ramp_deg": 14}), "hooks.ramp_deg: 14 is outside 15..60")
        self.assertErr(patched(**{"hooks.ramp_deg": 61}), "hooks.ramp_deg: 61 is outside 15..60")
        self.assertNoErr(patched(**{"hooks.ramp_deg": 15}))
        self.assertErr(patched(**{"hooks.retention_deg": 29}), "hooks.retention_deg: 29 is outside 30..80")
        self.assertErr(patched(**{"hooks.retention_deg": 81}), "hooks.retention_deg: 81 is outside 30..80")
        self.assertWarn(patched(**{"hooks.retention_deg": 65}), "a re-openable lid needs 60 degrees or less")
        self.assertNoWarn(patched(**{"hooks.retention_deg": 60}), "hooks.retention_deg")
        self.assertNoWarn(DEFAULT, "hooks.retention_deg")

    def test_catch_pocket_wall_rule(self):
        errors = self.assertErr(patched(wall_t=1.5, **{"standoff.d": 6.0}), "the catch pocket (0.65 mm deep) leaves 0.85 mm of wall behind it")
        self.assertIn("1.2", "\n".join(errors))
        self.assertNoErr(patched(wall_t=1.9))                                      # 1.25 behind the pocket
        self.assertErr(patched(wall_t=1.8), "catch pocket")                        # 1.15 behind

    def test_hook_strain_error_and_warning(self):
        self.assertNoWarn(DEFAULT, "strain")                                        # 69 percent
        self.assertWarn(patched(**{"hooks.t": 1.5}), "hooks.length: strain 0.0086 is 86% of the FDM limit 0.0100")
        self.assertNoWarn(patched(**{"hooks.t": 1.35}), "strain")                  # 77 percent
        errors = self.assertErr(patched(**{"hooks.t": 1.8}), "hooks.length: strain 0.0103 exceeds the FDM limit 0.0100 for ABS")
        self.assertIn("1.5 * t * (overhang - fit) / length^2", "\n".join(errors))
        self.assertNoErr(patched(**{"hooks.t": 1.8, "hooks.length": 14.0}))

    def test_hook_strain_limit_follows_the_material(self):
        # PETG 0.030 * 0.5 = 0.015: a thicker beam passes that fails in ABS
        d = patched(**{"hooks.t": 1.8})
        self.assertErr(d, "strain")
        d["material"] = "PETG"
        self.assertNoErr(d)

    def test_hook_rules_apply_only_to_snap_fit(self):
        wild = {"lid.style": "screw", "hooks.count": 3, "hooks.t": 0.1, "hooks.overhang": 9, "hooks.length": 1,
                "hooks.window_margin": 5, "hooks.ramp_deg": 1, "hooks.retention_deg": 99}
        self.assertNoErr(patched(**wild))
        for style in ("friction", "snap_fit"):
            wild["lid.style"] = style
        self.assertErr(patched(**wild), "hooks.")

    # vents
    def test_vent_rows(self):
        self.assertErr(patched(**{"vents.pitch": 3.0}), "vents.pitch: pitch 3.0 - slot_w 2.0 leaves a 1.0 mm web, below the ABS minimum 1.2 mm; use a pitch of 3.2 or more")
        self.assertNoErr(patched(**{"vents.pitch": 3.2}))
        self.assertErr(patched(**{"vents.count": 0}), "vents.count: 0 is outside 1..60")
        self.assertErr(patched(**{"vents.count": 61}), "vents.count: 61 is outside 1..60")
        self.assertErr(patched(**{"vents.part": "base"}), "vents.part: 'base' is not a part")
        self.assertErr(patched(**{"vents.face": "side"}), "vents.face: 'side' is not a face")
        self.assertErr(patched(**{"vents.pattern": "stars"}), "vents.pattern: 'stars' is not a pattern")
        self.assertErr(patched(**{"vents.slot_w": 0.4, "vents.pitch": 3.0}), "vents.slot_w: 0.4 is below the minimum 0.5")
        self.assertErr(patched(**{"vents.slot_l": 0.4}), "vents.slot_l: 0.4 is below the minimum 0.5")
        self.assertErr(patched(**{"vents.pitch": 0.9}), "vents.pitch: 0.9 is below the minimum 1.0")

    def test_minimum_comparisons_tolerate_float_noise(self):
        self.assertEqual(3.3 - 2.1, 1.1999999999999997)               # the trap: a web of exactly 1.2 mm computes as less than 1.2
        self.assertNoErr(patched(**{"vents.pitch": 3.3, "vents.slot_w": 2.1}))
        self.assertNoErr(patched(wall_t=1.95, **{"hooks.overhang": 0.9}))   # pocket 0.75 deep leaves exactly the 1.2 mm minimum
        self.assertErr(patched(**{"vents.pitch": 3.29, "vents.slot_w": 2.1}), "vents.pitch")

    def test_vents_off_skip_every_vent_rule(self):
        self.assertNoErr(patched(**{"vents.enabled": False, "vents.count": 0, "vents.pitch": 0.1, "vents.part": "x"}))

    def test_vent_array_must_fit_the_face(self):
        # the default 9 slots of 24 mm at pitch 4.0 span 34.0 mm across a 57.94 mm wide cavity
        self.assertNoErr(DEFAULT)
        errors = self.assertErr(patched(**{"vents.count": 30}), "vents: 30 slots at pitch 4.0 span 118.0 mm but the lid is only 55.5 mm wide there; reduce count or pitch.")
        self.assertEqual(len([e for e in errors if e.startswith("vents")]), 1)
        self.assertErr(patched(**{"vents.slot_l": 95.0}), "vents.slot_l: 95.0 mm is longer than the 87.7 mm available on the lid top")
        self.assertNoErr(patched(**{"vents.slot_l": 87.7}))
        self.assertNoErr(patched(**{"vents.angle_deg": 45, "vents.count": 14}))                 # 55.2 mm of the 55.5 available
        self.assertErr(patched(**{"vents.angle_deg": 45, "vents.count": 16}), "vents: the rotated array needs 60.8 x 60.8 mm but only 87.7 x 55.5 mm is available on the lid top")
        self.assertErr(patched(**{"vents.center": [0.0, 0.0], "vents.count": 9}), "vents")                 # off-centre: not enough room on one side

    def test_vent_center_inside_the_face_is_accepted(self):
        self.assertNoErr(patched(**{"vents.center": [42.5, 27.5], "vents.count": 5}))

    def test_vents_on_a_wall_use_the_wall_extent(self):
        # a +y wall face: u = x (90.1 mm minus the webs), v = z (cavity height 22 mm minus the webs)
        self.assertNoErr(patched(**{"vents.part": "shell", "vents.face": "+y", "vents.slot_l": 10.0, "vents.count": 4}))
        self.assertErr(patched(**{"vents.part": "shell", "vents.face": "+y", "vents.count": 9}), "vents: 9 slots")

    # ports
    def test_port_ids(self):
        errors = self.assertErr(patched(**{"ports.include": ["usbc"]}), "ports.include: 'usbc' is not a connector of 'rpi4b'", "(similar: usb_c_power)", "connectors: usb_c_power, hdmi0")
        self.assertEqual(len(errors), 1)
        self.assertErr(patched(**{"ports.exclude": ["nope"]}), "ports.exclude: 'nope' is not a connector")
        self.assertErr(patched(**{"ports.overrides": [{"id": "nope", "margin": 0.5}]}), "ports.overrides[0].id: 'nope' is not a connector")
        self.assertErr(patched(**{"ports.overrides": [{"margin": 0.5}]}), "ports.overrides[0]: missing 'id'")
        self.assertErr(patched(**{"ports.overrides": [{"id": "hdmi0", "margin": 2.5}]}), "ports.overrides[0].margin: 2.5 is outside 0..2")
        self.assertNoErr(patched(**{"ports.include": ["gpio"], "ports.exclude": ["audio"], "ports.overrides": [{"id": "hdmi0", "margin": 0.8}]}))

    def test_include_and_exclude_overlap_warns(self):
        self.assertWarn(patched(**{"ports.include": ["audio"], "ports.exclude": ["audio"]}), "'audio' is in both include and exclude; exclude wins")

    def test_port_margin(self):
        self.assertErr(patched(**{"ports.margin": -0.1}), "ports.margin: -0.1 is negative")
        self.assertErr(patched(**{"ports.margin": 2.1}), "ports.margin: 2.1 is above the maximum 2.0")
        self.assertWarn(patched(**{"ports.margin": 0.1}), "ports.margin: 0.1 is tight")
        self.assertNoWarn(patched(**{"ports.margin": 0.2}), "ports.margin")
        self.assertNoErr(patched(**{"ports.margin": 0.0}))
        self.assertErr(patched(**{"ports.gap": -1}), "ports.gap: -1.0 is negative")

    # cutouts
    def test_cutouts(self):
        good = {"part": "shell", "face": "-x", "u": 28.0, "v": 3.0, "w": 5.0, "h": 4.0}
        self.assertNoErr(patched(cutouts=[good]))
        self.assertNoErr(patched(cutouts=[dict(good, r=1.0, margin=0.2, through=False, id="cut1")]))
        for key in ("part", "face", "u", "v", "w", "h"):
            bad = {k: v for k, v in good.items() if k != key}
            self.assertErr(patched(cutouts=[bad]), f"cutouts[0]: missing '{key}'")
        self.assertErr(patched(cutouts=[dict(good, w=0.05)]), "cutouts[0].w: 0.05 is below the minimum 0.1")
        self.assertErr(patched(cutouts=[dict(good, h=0.0)]), "cutouts[0].h")
        self.assertErr(patched(cutouts=[dict(good, part="base")]), "cutouts[0].part: 'base' is not a part")
        self.assertErr(patched(cutouts=[dict(good, face="left")]), "cutouts[0].face: 'left' is not a face")
        self.assertErr(patched(cutouts=["nope"]), "cutouts[0]: expected an object")
        self.assertErr(patched(cutouts={"a": 1}), "cutouts: expected a list of objects")
        self.assertErr(patched(cutouts=[dict(good, wide=3)]), "cutouts[0].wide: unknown key 'wide'")

    # lid_lift, orientation, colors
    def test_lid_lift(self):
        self.assertErr(patched(lid_lift=-0.1), "lid_lift: -0.1 is outside 0..100")
        self.assertErr(patched(lid_lift=100.5), "lid_lift: 100.5 is outside 0..100")
        self.assertNoErr(patched(lid_lift=0.0))
        self.assertNoErr(patched(lid_lift=15))
        self.assertNoErr(patched(lid_lift=100))

    def test_print_orientation(self):
        self.assertErr(patched(**{"print_orientation.shell": "on_side"}), "print_orientation.shell: 'on_side' is not a shell orientation")
        self.assertErr(patched(**{"print_orientation.lid": "floor_down"}), "print_orientation.lid: 'floor_down' is not a lid orientation")
        self.assertNoErr(patched(**{"print_orientation.shell": "as_modeled", "print_orientation.lid": "as_modeled"}))

    def test_colors(self):
        self.assertErr(patched(**{"colors.shell": [1.1, 0.2, 0.3]}), "colors.shell: channel 1.1 is outside 0..1")
        self.assertErr(patched(**{"colors.lid": [0.1, -0.2, 0.3]}), "colors.lid: channel -0.2 is outside 0..1")
        self.assertErr(patched(**{"colors.ref": [0.1, 0.2]}), "colors.ref: expected three numbers")
        self.assertNoErr(patched(**{"colors.shell": [0, 1, 0.5]}))

    # unknown keys, types, garbage
    def test_unknown_keys_suggest_similar(self):
        d = patched()
        d["wall_th"] = 2.0
        self.assertErr(d, "wall_th: unknown key 'wall_th' (similar: wall_t)")
        d = patched()
        d["hooks"]["lenght"] = 12.0
        self.assertErr(d, "hooks.lenght: unknown key 'lenght' (similar: length)")
        d = patched()
        d["zzzzzz"] = 1
        self.assertErr(d, "zzzzzz: unknown key 'zzzzzz'; known keys: version, kit_id")
        d = patched(**{"ports.overrides": [{"id": "hdmi0", "marginn": 1}]})
        self.assertErr(d, "ports.overrides[0].marginn: unknown key 'marginn' (similar: margin)")

    def test_wrong_group_shape_is_reported(self):
        d = patched()
        d["hooks"] = 4
        self.assertErr(d, "hooks: expected an object with the keys count, w")
        d = patched()
        d["ports"] = ["hdmi0"]
        self.assertErr(d, "ports: expected an object")

    def test_type_errors_name_the_path_and_never_raise(self):
        self.assertErr(patched(wall_t="abc"), "wall_t: expected a number")
        self.assertErr(patched(wall_t=None), "wall_t: expected a number")
        self.assertErr(patched(wall_t=[2]), "wall_t: expected a number")
        self.assertErr(patched(wall_t=float("nan")), "wall_t: expected a number")
        self.assertErr(patched(wall_t=float("inf")), "wall_t: expected a number")
        self.assertErr(patched(**{"hooks.count": "four"}), "hooks.count: expected a whole number")
        self.assertErr(patched(**{"vents.enabled": "maybe"}), "vents.enabled: expected true or false")
        self.assertErr(patched(**{"ports.include": 5}), "ports.include: expected a list of connector ids")
        self.assertErr(patched(**{"vents.center": [1]}), "vents.center: expected null or a [u, v] pair")
        self.assertErr(patched(name=None), "name: expected text")

    def test_numeric_strings_are_accepted(self):
        self.assertNoErr(patched(wall_t="2.4 mm", clearance="0.35", **{"hooks.ramp_deg": "30 deg"}))

    def test_a_non_object_spec_is_an_error_not_a_crash(self):
        for bad in (None, 5, "spec", [1]):
            ok, errors, warnings = spec.validate(bad)           # type: ignore[arg-type]
            self.assertFalse(ok)
            self.assertIn("spec: expected an object", errors[0])

    def test_validate_never_raises_on_garbage(self):
        rng = random.Random(7)
        values = [None, True, False, 0, -1, 3, 2.5, 1e9, -1e9, float("nan"), "", "abs", "2.4 mm", [], [1, 2], ["a"], {}, {"a": 1},
                  [[1, 2], [3]], "scale", "snap_fit", "top", 10 ** 30]
        paths = list(spec._LEAF_KINDS) + ["standoff", "lid", "hooks", "vents", "ports", "colors", "print_orientation"]
        for _ in range(400):
            d = copy.deepcopy(DEFAULT)
            for _k in range(rng.randint(1, 4)):
                path = rng.choice(paths)
                value = rng.choice(values)
                cur = d
                keys = path.split(".")
                for k in keys[:-1]:
                    if not isinstance(cur.get(k), dict):
                        cur[k] = {}
                    cur = cur[k]
                cur[keys[-1]] = value
            ok, errors, warnings = spec.validate(d, BOARD)
            self.assertIsInstance(ok, bool)
            self.assertEqual(ok, not errors)
            self.assertTrue(all(isinstance(m, str) and m for m in errors + warnings))
            # the sibling functions are equally robust on whatever from_dict made of it
            s = spec.from_dict(d)
            spec.spec_hash(s)
            spec.to_dict(s)

    def test_messages_follow_the_style(self):
        errs = self.errors(patched(wall_t=0.5, clearance=9, **{"hooks.count": 3, "standoff.h": 1.0}))
        self.assertGreaterEqual(len(errs), 4)
        for e in errs:
            self.assertRegex(e, r"^[A-Za-z_.\[\]0-9]+: .+")
            self.assertFalse(e.startswith("Traceback"))

    def test_all_errors_are_listed_at_once(self):
        errs = self.errors(patched(wall_t=0.5, floor_t=0.5, lid_t=0.5))
        self.assertEqual(len([e for e in errs if e.split(":")[0] in ("wall_t", "floor_t", "lid_t")]), 3)


# --------------------------------------------------------------------------------------------------------------------
class DerivedValuesTest(unittest.TestCase):
    """The default Pi 4 numbers and the derived formulas."""

    def setUp(self):
        self.dv = spec.derived(spec.default_spec(), BOARD)

    def val(self, name):
        return self.dv[name]["value"]

    def test_pi4_default_numbers(self):
        for name, want in {"clearance_eff": 0.64, "margin_nx": 2.8, "margin_px": 2.3, "margin_ny": 1.3, "margin_py": 0.64,
                           "cav_x0": -2.8, "cav_x1": 87.3, "cav_y0": -1.3, "cav_y1": 56.64, "cav_len": 90.1, "cav_wid": 57.94,
                           "shell_len": 94.1, "shell_wid": 61.94, "z_floor_top": -4.0, "z_floor_bot": -6.0, "z_cavity_top": 18.0,
                           "shell_h": 24.0, "lid_len": 97.0, "lid_wid": 64.84}.items():
            self.assertAlmostEqual(self.val(name), want, places=6, msg=name)

    def test_shape_of_the_result(self):
        self.assertEqual(list(self.dv), [n for n, _f, _u in spec.DERIVED_FORMULAS])
        for name, row in self.dv.items():
            self.assertEqual(set(row), {"value", "expr", "unit"})
            self.assertEqual(row["unit"], "mm")
            self.assertTrue(row["expr"].startswith("="), name)
        self.assertEqual(self.dv["clearance_eff"]["expr"], "=clearance + shrink_pct/100*board_len/2")
        self.assertEqual(self.dv["margin_nx"]["expr"], "=max(clearance_eff; pr_nx + port_gap)")

    def test_derived_depends_on_the_material_and_shrink_mode(self):
        pla = spec.derived(spec.default_spec("rpi4b", "PLA"), BOARD)
        self.assertAlmostEqual(pla["clearance_eff"]["value"], 0.3 + 0.3 / 100 * 85 / 2)
        s = spec.default_spec()
        s.shrink_comp = "none"
        self.assertAlmostEqual(spec.derived(s, BOARD)["clearance_eff"]["value"], 0.3)

    def test_margins_follow_the_connector_protrusions(self):
        # pr_nx 2.5 (microSD), pr_px 2.0 (USB / Ethernet), pr_ny 1.0 (-y ports), pr_py 0
        self.assertAlmostEqual(self.val("margin_nx"), 2.8)            # 2.5 + port_gap 0.3
        self.assertAlmostEqual(self.val("margin_px"), 2.3)
        self.assertAlmostEqual(self.val("margin_ny"), 1.3)
        self.assertAlmostEqual(self.val("margin_py"), 0.64)           # no +y connector: the clearance wins
        s = spec.default_spec()
        s.ports.exclude = ["usb3_pair", "usb2_pair", "ethernet"]
        self.assertAlmostEqual(spec.derived(s, BOARD)["margin_px"]["value"], 0.64)
        s.ports.auto = False
        dv = spec.derived(s, BOARD)
        self.assertAlmostEqual(dv["margin_nx"]["value"], 0.64)
        self.assertAlmostEqual(dv["cav_len"]["value"], 85 + 2 * 0.64)

    def test_editing_wall_and_clearance_moves_the_outer_size(self):
        s = spec.default_spec()
        s.wall_t, s.clearance = 2.4, 0.35
        dv = spec.derived(s, BOARD)
        # the EditabilityTest: new bbox 94.9 x 62.79 x 24.0
        self.assertAlmostEqual(dv["shell_len"]["value"], 94.9, places=6)
        self.assertAlmostEqual(dv["shell_wid"]["value"], 62.79, places=6)
        self.assertAlmostEqual(dv["shell_h"]["value"], 24.0, places=6)

    def test_input_params_follow_decision_22_sources(self):
        rows = {r["name"]: r for r in spec.input_params(spec.default_spec(), BOARD)}
        self.assertEqual(rows["board_len"]["source"], "datasheet")
        self.assertEqual(rows["pcb_t"]["source"], "assumption")
        self.assertEqual(rows["tallest_z"]["source"], "assumption")        # community -> assumption
        self.assertEqual((rows["board_len"]["value"], rows["board_w"]["value"]), (85.0, 56.0))
        self.assertEqual((rows["hole_off"]["value"], rows["hole_dx"]["value"], rows["hole_dy"]["value"], rows["hole_d"]["value"]), (3.5, 58.0, 49.0, 2.75))
        self.assertEqual((rows["tallest_z"]["value"], rows["bottom_z"]["value"]), (17.0, -1.5))
        self.assertEqual((rows["pr_nx"]["value"], rows["pr_px"]["value"], rows["pr_ny"]["value"], rows["pr_py"]["value"]), (2.5, 2.0, 1.0, 0.0))
        self.assertEqual(rows["shrink_pct"]["value"], 0.8)
        self.assertEqual(rows["shrink_pct"]["unit"], "")
        measured = copy.deepcopy(BOARD)
        measured["outline"]["confidence"] = "measured"
        self.assertEqual({r["name"]: r for r in spec.input_params(spec.default_spec(), measured)}["board_len"]["source"], "step_measurement")

    def test_param_rows_order_and_formulas(self):
        rows = spec.param_rows(spec.default_spec(), BOARD)
        names = [r["name"] for r in rows]
        self.assertEqual(names[:3], ["board_len", "board_w", "pcb_t"])
        self.assertEqual(len(names), len(set(names)))
        derived_rows = [r for r in rows if r["source"] == "derived"]
        self.assertEqual([r["name"] for r in derived_rows], [n for n, _f, _u in spec.DERIVED_FORMULAS])
        self.assertTrue(all(r["formula"] and r["formula"].startswith("=") for r in derived_rows))
        self.assertTrue(all(r["formula"] is None for r in rows if r["source"] != "derived"))
        self.assertEqual(names.index("shrink_pct"), names.index("clearance") + 1)
        self.assertEqual(spec.INPUT_PARAM_ORDER, tuple(r["name"] for r in rows if r["source"] != "derived"))

    def test_incomplete_data_raises_value_error(self):
        with self.assertRaises(ValueError):
            spec.derived(spec.default_spec(), {"id": "x"})
        s = spec.default_spec()
        s.material = "WOOD"
        with self.assertRaises(ValueError):
            spec.derived(s, BOARD)
        s = spec.default_spec()
        s.wall_t = "thick"
        with self.assertRaises(ValueError):
            spec.input_params(s, BOARD)

    def test_param_field_maps(self):
        self.assertEqual(spec.PARAM_OF_FIELD["hooks.length"], "hook_len")
        self.assertEqual(spec.FIELD_OF_PARAM["lid_fit"], "lid.fit")
        for name, path in spec.FIELD_OF_PARAM.items():
            self.assertIn(path, spec._LEAF_KINDS, name)


class EvalFormulaTest(unittest.TestCase):
    def test_operators_and_functions(self):
        env = {"a": 3.0, "b": 4.0}
        for text, want in {"=a + b*2": 11.0, "a - b": -1.0, "=-(a + b)": -7.0, "=+a": 3.0, "=a**2": 9.0, "=b % 3": 1.0, "=b/a*3": 4.0,
                           "=max(a; b)": 4.0, "=min(a; b; 1)": 1.0, "=abs(a - b)": 1.0, "=ceil(2.1)": 3.0, "=floor(2.9)": 2.0,
                           "=sqrt(16)": 4.0, "=round(2.6)": 3.0, "=(a + b) * (a - b)": -7.0, "=1e2": 100.0}.items():
            self.assertAlmostEqual(spec.eval_formula(text, env), want, msg=text)

    def test_rejects_everything_else_with_value_error(self):
        env = {"a": 3.0}
        for bad in ("=zzz", "=a +", "=__import__('os')", "=a.real", "=(lambda: 1)()", "=[1, 2]", "='x'", "=a if a else 1", "=a < 2",
                    "=open('f')", "=max(key=1)", "=a[0]", "=True", "=1/0", "=a % 0", "=sqrt(-1)", "=10**400", "=9.0**9**9",
                    "=max()", "=ceil(1; 2)", "", "=", "=(" * 300, "\x00", "=a;;b"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                spec.eval_formula(bad, env)

    def test_non_finite_results_are_refused(self):
        with self.assertRaises(ValueError):
            spec.eval_formula("=1e308*10", {})

    def test_every_derived_formula_evaluates_on_the_defaults(self):
        env = {r["name"]: r["value"] for r in spec.input_params(spec.default_spec(), BOARD)}
        for name, formula, _unit in spec.DERIVED_FORMULAS:
            env[name] = spec.eval_formula(formula, env)
        self.assertAlmostEqual(env["lid_wid"], 64.84)


# --------------------------------------------------------------------------------------------------------------------
class OpeningsTest(unittest.TestCase):
    """Connector openings (7.1 `opening`, 7.1a plug sizes) and the user cut-outs."""

    def by_id(self, s=None, **kw):
        return {o["id"]: o for o in spec.port_openings(s or spec.default_spec(), BOARD, **kw)}

    def test_active_connectors(self):
        s = spec.default_spec()
        ids = [c["id"] for c in spec.active_connectors(s, BOARD)]
        self.assertEqual(ids, ["usb_c_power", "hdmi0", "hdmi1", "audio", "ethernet", "usb3_pair", "usb2_pair", "microsd"])
        s.ports.exclude = ["audio", "hdmi1"]
        self.assertNotIn("audio", [c["id"] for c in spec.active_connectors(s, BOARD)])
        s.ports.include = ["gpio"]                      # a connector without a wall opening can be forced
        self.assertIn("gpio", [c["id"] for c in spec.active_connectors(s, BOARD)])
        s.ports.auto = False
        self.assertEqual([c["id"] for c in spec.active_connectors(s, BOARD)], ["gpio"])
        s.ports.exclude = ["gpio"]                      # exclude always wins
        self.assertEqual(spec.active_connectors(s, BOARD), [])

    def test_receptacle_geometry_matches_the_derived_openings_of_7_1(self):
        c = {c["id"]: c for c in BOARD["connectors"]}
        o = spec.connector_opening(c["usb_c_power"], BOARD)
        self.assertEqual((o["face"], o["u"], o["w"], round(o["v"], 6), round(o["h"], 6)), ("-y", 11.2, 9.0, 3.05, 3.3))
        o = spec.connector_opening(c["hdmi0"], BOARD)
        self.assertEqual((o["u"], round(o["w"], 6), round(o["v"], 6), round(o["h"], 6)), (26.0, 6.7, 2.95, 3.1))
        o = spec.connector_opening(c["audio"], BOARD)
        self.assertEqual((o["u"], o["w"], round(o["v"], 6), o["h"]), (53.5, 7.0, 4.4, 6.0))
        o = spec.connector_opening(c["ethernet"], BOARD)
        self.assertEqual((o["face"], o["u"], o["w"], round(o["v"], 6), round(o["h"], 6)), ("+x", 45.75, 16.0, 8.15, 13.5))
        o = spec.connector_opening(c["usb3_pair"], BOARD)
        self.assertEqual((o["u"], round(o["w"], 6), round(o["v"], 6), round(o["h"], 6)), (27.0, 13.1, 9.2, 15.6))
        o = spec.connector_opening(c["microsd"], BOARD)
        self.assertEqual((o["face"], o["u"], o["w"], o["v"], o["h"]), ("-x", 28.0, 12.0, -0.75, 1.5))
        o = spec.connector_opening(c["gpio"], BOARD)
        self.assertEqual(o["face"], "top")
        self.assertEqual((round(o["u"], 6), round(o["v"], 6)), (32.5, 52.5))
        for name, pr in {"usb_c_power": 0.5, "audio": 1.0, "ethernet": 2.0, "usb3_pair": 2.0, "microsd": 2.5, "gpio": 0.0}.items():
            self.assertAlmostEqual(spec.connector_opening(c[name], BOARD)["protrusion"], pr, msg=name)

    def test_plug_clearance_sizes_of_7_1a(self):
        o = self.by_id()
        self.assertEqual((round(o["usb_c_power"]["w"], 6), round(o["usb_c_power"]["h"], 6)), (12.0, 6.5))
        self.assertEqual((round(o["hdmi0"]["w"], 6), round(o["hdmi0"]["h"], 6)), (11.0, 7.0))
        self.assertEqual((round(o["audio"]["w"], 6), round(o["audio"]["h"], 6)), (9.0, 9.0))
        self.assertEqual((round(o["ethernet"]["w"], 6), round(o["ethernet"]["h"], 6)), (17.0, 14.5))      # 16 + 2 * 0.5, 13.5 + 2 * 0.5
        self.assertEqual((round(o["usb3_pair"]["w"], 6), round(o["usb3_pair"]["h"], 6)), (14.1, 16.6))
        self.assertEqual((o["usb_c_power"]["u"], round(o["usb_c_power"]["v"], 6)), (11.2, 3.05))
        self.assertEqual((o["hdmi1"]["u"], round(o["hdmi1"]["v"], 6)), (39.5, 2.95))
        self.assertEqual((o["audio"]["u"], round(o["audio"]["v"], 6)), (53.5, 4.4))

    def test_plug_clearance_off_uses_receptacle_plus_margin(self):
        s = spec.default_spec()
        s.ports.plug_clearance = False
        o = self.by_id(s)
        self.assertAlmostEqual(o["usb_c_power"]["w"], 9.0 + 2 * 0.4)
        self.assertAlmostEqual(o["usb_c_power"]["h"], 3.3 + 2 * 0.4)
        self.assertAlmostEqual(o["audio"]["w"], 7.0 + 2 * 0.5)

    def test_connector_plug_override_wins_over_the_kind_default(self):
        c = copy.deepcopy(BOARD["connectors"][0])
        c["plug_w"], c["plug_h"] = 13.0, 8.0
        o = spec.connector_opening(c, BOARD, margin=0.4)
        self.assertEqual((o["w"], o["h"]), (13.0, 8.0))
        self.assertEqual(spec.connector_opening(c, BOARD, margin=0.4, plug=False)["w"], 9.0 + 0.8)

    def test_margin_is_a_floor_not_an_override(self):
        s = spec.default_spec()
        c = {c["id"]: c for c in BOARD["connectors"]}
        self.assertEqual(spec.connector_margin(s, c["usb_c_power"]), 0.4)           # max(0.4, hint 0.4)
        self.assertEqual(spec.connector_margin(s, c["audio"]), 0.5)                  # hint 0.5 beats the spec's 0.4
        s.ports.margin = 0.9
        self.assertEqual(spec.connector_margin(s, c["audio"]), 0.9)
        s.ports.overrides = [{"id": "audio", "margin": 0.2}]
        self.assertEqual(spec.connector_margin(s, c["audio"]), 0.2)                  # an explicit override can go lower

    def test_pcb_thickness_shift_moves_only_connectors_above_the_pcb_top(self):
        base, shifted = self.by_id(), self.by_id(pcb_t=1.6)
        self.assertAlmostEqual(shifted["usb_c_power"]["v"], base["usb_c_power"]["v"] + 0.2)
        self.assertAlmostEqual(shifted["ethernet"]["v"], base["ethernet"]["v"] + 0.2)
        self.assertEqual(shifted["microsd"]["v"], base["microsd"]["v"])               # below the board: unchanged
        self.assertEqual(self.by_id(pcb_t=1.4)["audio"]["v"], base["audio"]["v"])

    def test_webs_between_adjacent_minus_y_openings(self):
        o = self.by_id()
        ordered = [o["usb_c_power"], o["hdmi0"], o["hdmi1"], o["audio"]]
        webs = [(b["u"] - b["w"] / 2) - (a["u"] + a["w"] / 2) for a, b in zip(ordered, ordered[1:])]
        for got, want in zip(webs, (3.3, 2.5, 4.0)):
            self.assertAlmostEqual(got, want, places=6)
        self.assertAlmostEqual(ordered[-1]["u"] + ordered[-1]["w"] / 2, 58.0)         # the free span of the wall starts at x = 58

    def test_shell_cutouts_and_wall_openings(self):
        s = spec.default_spec()
        s.cutouts = [{"part": "shell", "face": "+y", "u": 40.0, "v": 5.0, "w": 10.0, "h": 4.0, "margin": 0.5, "id": "fan"},
                     {"part": "lid", "face": "top", "u": 1, "v": 1, "w": 2, "h": 2},
                     {"part": "shell", "face": "left", "u": 1, "v": 1, "w": 2, "h": 2},
                     "junk", {"part": "shell", "face": "-x", "u": "x", "v": 1, "w": 2, "h": 2}]
        cuts = spec.shell_cutout_openings(s)
        self.assertEqual(len(cuts), 1)
        self.assertEqual((cuts[0]["id"], cuts[0]["face"], cuts[0]["w"], cuts[0]["h"]), ("fan", "+y", 11.0, 5.0))
        self.assertEqual(len(spec.wall_openings(s, BOARD)), 8 + 1)
        self.assertEqual([o["id"] for o in spec.wall_openings(s, BOARD, "+y")], ["fan"])
        self.assertEqual(len(spec.wall_openings(s, BOARD, "-y")), 4)


# --------------------------------------------------------------------------------------------------------------------
class SnapFitTest(unittest.TestCase):
    """strain, pocket depth and the placement of the hooks."""

    def test_default_strain_and_pocket(self):
        s = spec.default_spec()
        self.assertAlmostEqual(spec.hook_strain(s), 1.5 * 1.2 * 0.55 / 144)
        self.assertAlmostEqual(spec.hook_strain(s), 0.006875)
        self.assertAlmostEqual(spec.hook_strain_limit(s), 0.010)
        self.assertAlmostEqual(spec.hook_strain_limit(s, molded=True), 0.020)
        self.assertAlmostEqual(spec.catch_pocket_depth(s), 0.65)
        self.assertAlmostEqual(s.wall_t - spec.catch_pocket_depth(s), 1.35)
        s.hooks.length = 0
        self.assertEqual(spec.hook_strain(s), math.inf)

    def test_default_positions_on_the_pi4(self):
        pos = spec.hook_positions(spec.default_spec(), BOARD)
        self.assertEqual([p["wall"] for p in pos], ["+y", "+y", "-y", "-y"])
        self.assertAlmostEqual(pos[0]["u"], 19.725, places=3)           # 25 % and 75 % of the cavity length (x 19.7 and 64.8)
        self.assertAlmostEqual(pos[1]["u"], 64.775, places=3)
        self.assertTrue(all(p["w"] == 6.0 for p in pos))
        minus_y = [p["u"] for p in pos if p["wall"] == "-y"]
        self.assertTrue(all(u >= 58.0 + 1.0 + 3.0 for u in minus_y), minus_y)        # beyond the audio opening plus 1 mm + w / 2
        self.assertTrue(all(u <= 87.3 - 1.0 - 3.0 for u in minus_y), minus_y)        # inside the straight part of the wall
        self.assertGreaterEqual(minus_y[1] - minus_y[0], 6.0 + 1.0)

    def test_positions_clear_every_opening_for_many_configurations(self):
        rng = random.Random(3)
        ids = [c["id"] for c in BOARD["connectors"]]
        for _ in range(60):
            s = spec.default_spec()
            s.hooks.count = rng.choice([2, 4, 6])
            s.hooks.w = rng.choice([4.0, 6.0, 8.0])
            s.ports.exclude = rng.sample(ids, rng.randint(0, 5))
            for p in spec.hook_positions(s, BOARD):
                for o in spec.wall_openings(s, BOARD, p["wall"]):
                    gap = abs(p["u"] - o["u"]) - o["w"] / 2 - p["w"] / 2
                    self.assertGreaterEqual(gap, 1.0 - 1e-6, (p, o))

    def test_positions_per_wall_and_count(self):
        s = spec.default_spec()
        s.hooks.count = 2
        pos = spec.hook_positions(s, BOARD)
        self.assertEqual([p["wall"] for p in pos], ["+y", "-y"])
        self.assertAlmostEqual(pos[0]["u"], -2.8 + 0.5 * 90.1, places=3)           # one hook per wall: the middle
        s.hooks.count = 6
        self.assertEqual([p["wall"] for p in spec.hook_positions(s, BOARD)].count("+y"), 3)

    def test_positions_without_openings_on_minus_y(self):
        s = spec.default_spec()
        s.ports.exclude = ["usb_c_power", "hdmi0", "hdmi1", "audio"]
        pos = spec.hook_positions(s, BOARD)
        minus = [p["u"] for p in pos if p["wall"] == "-y"]
        self.assertEqual(len(minus), 2)
        self.assertAlmostEqual(minus[0], 19.725, places=3)

    def test_a_user_cutout_blocks_the_hook_beside_it(self):
        s = spec.default_spec()
        s.cutouts = [{"part": "shell", "face": "+y", "u": 19.7, "v": 5.0, "w": 10.0, "h": 4.0}]
        plus = [p["u"] for p in spec.hook_positions(s, BOARD) if p["wall"] == "+y"]
        self.assertEqual(len(plus), 2)
        self.assertTrue(all(abs(u - 19.7) >= 5.0 + 1.0 + 3.0 - 1e-6 for u in plus), plus)

    def test_not_snap_fit_has_no_hooks(self):
        s = spec.default_spec()
        s.lid.style = "screw"
        self.assertEqual(spec.hook_positions(s, BOARD), [])
        s = spec.default_spec()
        s.hooks.count = 0
        self.assertEqual(spec.hook_positions(s, BOARD), [])

    def test_vent_helpers(self):
        s = spec.default_spec()
        span, length, count = spec.vent_extent(s)
        self.assertEqual((span, length, count), (34.0, 24.0, 9))
        s.vents.pattern = "holes"
        self.assertEqual(spec.vent_extent(s)[1], 2.0)
        area = spec.vent_area(spec.default_spec(), BOARD)
        self.assertEqual((area["part"], area["face"]), ("lid", "top"))
        self.assertAlmostEqual(area["cu"], (-2.8 + 87.3) / 2)
        self.assertAlmostEqual(area["u0"], -2.8 + 1.2)
        self.assertAlmostEqual(area["v1"], 56.64 - 1.2)
        wall = spec.vent_area(spec.default_spec(), BOARD, "shell", "+y")
        self.assertAlmostEqual(wall["v0"], -4.0 + 1.2)
        self.assertAlmostEqual(wall["v1"], 18.0 - 1.2)


# --------------------------------------------------------------------------------------------------------------------
# the fields each stage depends on, written out from the table (NOT derived from STAGE_FIELDS)
ALL = set(spec.STAGES)
NOT_REF = ALL - {"reference"}
DOWNSTREAM_OF_LID = {"params", "lid", "vents", "hooks", "assembly"}
EXPECTED_STAGES = {
    "version": {"params"}, "kit_id": {"params"}, "name": set(),
    "board": ALL, "material": NOT_REF, "wall_t": NOT_REF, "floor_t": NOT_REF, "lid_t": DOWNSTREAM_OF_LID,
    "clearance": NOT_REF, "shrink_comp": NOT_REF, "corner_r": NOT_REF,
    "standoff.h": NOT_REF, "standoff.d": {"params", "standoffs"}, "standoff.pilot_d": {"params", "standoffs"},
    "lid.style": DOWNSTREAM_OF_LID, "lid.gap": NOT_REF, "lid.lip_h": DOWNSTREAM_OF_LID, "lid.lip_t": DOWNSTREAM_OF_LID,
    "lid.fit": DOWNSTREAM_OF_LID,
    "hooks.count": {"params", "hooks"}, "hooks.w": {"params", "hooks"}, "hooks.t": {"params", "hooks"},
    "hooks.overhang": {"params", "hooks"}, "hooks.length": {"params", "hooks"}, "hooks.barb_h": {"params", "hooks"},
    "hooks.ramp_deg": {"params", "hooks"}, "hooks.retention_deg": {"params", "hooks"}, "hooks.window_margin": {"params", "hooks"},
    "vents.enabled": {"params", "vents"}, "vents.part": {"params", "vents"}, "vents.face": {"params", "vents"},
    "vents.pattern": {"params", "vents"}, "vents.count": {"params", "vents"}, "vents.slot_w": {"params", "vents"},
    "vents.slot_l": {"params", "vents"}, "vents.pitch": {"params", "vents"}, "vents.center": {"params", "vents"},
    "vents.angle_deg": {"params", "vents"},
    # ports.auto is not a row of the validation table, but it decides which connectors protrude into the cavity margins
    "ports.auto": NOT_REF, "ports.include": NOT_REF, "ports.exclude": NOT_REF, "ports.margin": NOT_REF, "ports.gap": NOT_REF,
    "ports.plug_clearance": NOT_REF, "ports.overrides": NOT_REF,
    "lid_lift": {"params", "assembly"},
    "print_orientation.shell": set(), "print_orientation.lid": set(),
    "colors.shell": {"assembly"}, "colors.lid": {"assembly"}, "colors.ref": {"assembly"},
}
EDITS = {
    "version": 2, "kit_id": "enc2", "name": "Other", "board": "rpi5", "material": "PLA", "shrink_comp": "none",
    "lid.style": "screw", "vents.part": "shell", "vents.face": "bottom", "vents.pattern": "holes", "vents.enabled": False,
    "vents.center": [10.0, 20.0], "ports.auto": False, "ports.plug_clearance": False, "ports.include": ["gpio"],
    "ports.exclude": ["audio"], "ports.overrides": [{"id": "audio", "margin": 0.9}],
    "print_orientation.shell": "as_modeled", "print_orientation.lid": "as_modeled",
    "colors.shell": [0.9, 0.9, 0.9], "colors.lid": [0.1, 0.1, 0.1], "colors.ref": [0.5, 0.5, 0.5],
}


def _edited(path: str) -> dict:
    d = copy.deepcopy(DEFAULT)
    cur = d
    keys = path.split(".")
    for k in keys[:-1]:
        cur = cur[k]
    old = cur[keys[-1]]
    cur[keys[-1]] = EDITS[path] if path in EDITS else (old + 1 if isinstance(old, int) and not isinstance(old, bool) else old + 0.37)
    return d


class StageHashTest(unittest.TestCase):
    """spec_hash: stable, canonical, and an edit reaches exactly the stages of its row."""

    def changed_stages(self, edited: dict) -> set:
        base = spec.default_spec()
        s = spec.from_dict(edited)
        return {st for st in spec.STAGES if spec.spec_hash(base, st) != spec.spec_hash(s, st)}

    def test_every_leaf_field_is_covered_by_the_expectation_table(self):
        leaves = set(spec._LEAF_KINDS) - {"cutouts"}
        self.assertEqual(leaves, set(EXPECTED_STAGES), "a new spec field needs a decision about the stages it affects")

    def test_each_field_edit_changes_exactly_the_stages_of_its_row(self):
        for path, want in EXPECTED_STAGES.items():
            got = self.changed_stages(_edited(path))
            self.assertEqual(got, want, f"editing {path}: stages changed {sorted(got)}, expected {sorted(want)}")

    def test_shell_and_lid_cutouts_reach_their_owning_stage_and_dependents(self):
        entry = {"part": "shell", "face": "+y", "u": 40.0, "v": 5.0, "w": 10.0, "h": 4.0}
        self.assertEqual(self.changed_stages(dict(DEFAULT, cutouts=[entry])), {"params", "ports", "hooks"})
        self.assertEqual(self.changed_stages(dict(DEFAULT, cutouts=[dict(entry, part="lid", face="top")])), {"params", "vents"})
        both = [entry, dict(entry, part="lid", face="top")]
        self.assertEqual(self.changed_stages(dict(DEFAULT, cutouts=both)), {"params", "ports", "hooks", "vents"})

    def test_stages_affected_by_agrees_with_the_table(self):
        for path, want in EXPECTED_STAGES.items():
            self.assertEqual(set(spec.stages_affected_by(path)), want, path)
        self.assertEqual(set(spec.stages_affected_by("cutouts")), {"params", "ports", "hooks", "vents"})
        self.assertEqual(spec.stages_affected_by("hooks"), ["params", "hooks"])           # a whole group
        self.assertEqual(spec.stages_affected_by("nothing"), [])

    def test_hash_format_and_stability(self):
        s = spec.default_spec()
        h = spec.spec_hash(s)
        self.assertRegex(h, r"^[0-9a-f]{12}$")
        self.assertEqual(h, spec.spec_hash(spec.default_spec()))
        self.assertEqual(h, spec.spec_hash(DEFAULT))
        self.assertEqual(h, spec.spec_hash(spec.from_dict(json.loads(json.dumps(DEFAULT)))))
        for st in spec.STAGES:
            self.assertRegex(spec.spec_hash(s, st), r"^[0-9a-f]{12}$")
        self.assertEqual(len({spec.spec_hash(s, st) for st in spec.STAGES} | {h}), len(spec.STAGES) + 1, "no two stage hashes collide")

    def test_golden_hashes_are_the_same_on_every_machine_and_run(self):
        s = spec.default_spec()
        golden = {None: "517358e00075", "params": "0ad3c57856ed", "reference": "0d1edca4e3cc", "shell": "640374fd8f4f",
                  "standoffs": "615a0fdc83e7", "ports": "9389eebc7652", "lid": "7261006147dc", "vents": "89dd3b4e8c1b",
                  "hooks": "a0bb1c71be79", "assembly": "415908ca5295"}
        for stage, want in golden.items():
            self.assertEqual(spec.spec_hash(s, stage), want, f"stage {stage}: the hash format changed; rebuilt documents would all look stale")

    def test_canonical_numbers_and_key_order(self):
        base = spec.spec_hash(DEFAULT)
        reordered = dict(reversed(list(copy.deepcopy(DEFAULT).items())))
        self.assertEqual(spec.spec_hash(reordered), base)
        self.assertEqual(spec.spec_hash(patched(wall_t=2)), base)                         # 2 == 2.0
        self.assertEqual(spec.spec_hash(patched(wall_t="2.0 mm")), base)
        self.assertEqual(spec.spec_hash(patched(wall_t=2.0 + 1e-12)), base)               # float noise
        self.assertNotEqual(spec.spec_hash(patched(wall_t=2.0 + 1e-6)), base)
        self.assertNotEqual(spec.spec_hash(patched(**{"ports.include": ["hdmi0"]})), base)

    def test_unrelated_stages_do_not_change(self):
        s = spec.default_spec()
        t = spec.default_spec()
        t.hooks.length = 13.0
        for st in spec.STAGES:
            same = spec.spec_hash(s, st) == spec.spec_hash(t, st)
            self.assertEqual(same, st not in ("params", "hooks"), st)

    def test_unknown_stage_is_a_value_error(self):
        with self.assertRaises(ValueError) as cm:
            spec.spec_hash(spec.default_spec(), "paint")
        self.assertIn("params, reference, shell", str(cm.exception))

    def test_stage_table_is_consistent(self):
        self.assertEqual(spec.STAGES, ("params", "reference", "shell", "standoffs", "ports", "lid", "vents", "hooks", "assembly"))
        self.assertEqual(set(spec.STAGE_FIELDS), set(spec.STAGES))
        self.assertEqual(set(spec.STAGE_DEPS), set(spec.STAGES))
        self.assertEqual(spec.STAGE_DEPS["hooks"], ("shell", "lid", "ports"))
        self.assertEqual(spec.STAGE_DEPS["vents"], ("shell", "lid"))
        order = {s: i for i, s in enumerate(spec.STAGES)}
        for stage, deps in spec.STAGE_DEPS.items():
            self.assertTrue(all(order[d] < order[stage] for d in deps), f"{stage} depends on a later stage")
        self.assertNotIn("name", spec.STAGE_FIELDS["params"])
        self.assertNotIn("colors", spec.STAGE_FIELDS["params"])
        self.assertNotIn("print_orientation", spec.STAGE_FIELDS["params"])


class RobustnessTest(unittest.TestCase):
    """Malformed input never crashes the helpers the tools call on whatever the model sent."""

    def test_hash_of_malformed_cutouts_does_not_crash_and_differs(self):
        base = spec.spec_hash(DEFAULT, "ports")
        for bad in (None, True, 3, "x", {"a": 1}, float("inf")):
            h = spec.spec_hash(dict(DEFAULT, cutouts=bad), "ports")
            self.assertRegex(h, r"^[0-9a-f]{12}$")
        self.assertNotEqual(spec.spec_hash(dict(DEFAULT, cutouts="x"), "ports"), base)

    def test_hook_positions_refuses_absurd_counts(self):
        s = spec.default_spec()
        for count in (0, 1, 65, 10 ** 12, -4, "four", None, 2.5):
            s.hooks.count = count
            self.assertEqual(spec.hook_positions(s, BOARD), [], count)

    def test_helpers_on_bad_boards_raise_value_error_not_key_error(self):
        for fn in (spec.derived, spec.param_rows, spec.input_params):
            with self.assertRaises(ValueError):
                fn(spec.default_spec(), {"id": "empty"})


class CatalogIntegrationTest(unittest.TestCase):
    """The kit reads boards through refparts.catalog: same numbers as the fixture, and every board is usable."""

    def setUp(self):
        self.cat = spec._catalog()
        if self.cat is None:
            self.skipTest("refparts.catalog not installed")

    def test_rpi4b_from_the_catalog_matches_the_fixture_numbers(self):
        board = spec.load_board("rpi4b")
        s = spec.default_spec()
        self.assertEqual(spec.derived(s, board), spec.derived(s, BOARD))
        self.assertEqual(spec.hook_positions(s, board), spec.hook_positions(s, BOARD))
        self.assertEqual([o["id"] for o in spec.port_openings(s, board)], [o["id"] for o in spec.port_openings(s, BOARD)])
        self.assertEqual(spec.validate(s, board), spec.validate(s, BOARD))

    def test_unknown_board_raises_key_error_with_the_known_ids(self):
        with self.assertRaises(KeyError):
            spec.load_board("no_such_board")
        self.assertIn("rpi4b", spec._known_board_ids())

    def test_every_board_in_the_library_works_with_the_kit_helpers(self):
        for info in self.cat.list_boards():
            board = self.cat.load_board(info["id"])
            s = spec.default_spec(board["id"])
            ok, errors, warnings = spec.validate(s, board)          # may be invalid for a small board, must never raise
            self.assertEqual(ok, not errors, board["id"])
            dv = spec.derived(s, board)
            self.assertGreater(dv["shell_len"]["value"], board["outline"]["length"], board["id"])
            self.assertGreater(dv["shell_h"]["value"], 0, board["id"])
            spec.port_openings(s, board)
            spec.hook_positions(s, board)
            spec.param_rows(s, board)


if __name__ == "__main__":
    unittest.main()
