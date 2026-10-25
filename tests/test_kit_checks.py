"""Tests for oscar.kit.checks.dfm_rows (pure: no FreeCAD, no document).

One row per DFM id with PASS / WARN / FAIL / INFO, the thresholds at their boundaries, the FDM strain
derate, the rim-web notch warning of the default Pi 4 (`usb3_pair` / `usb2_pair`), applicability for non-snap lids and
the "never raises, never mutates, pure" guarantees. (fit_rows and spec_coverage belong to the builder stage.)
"""

import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.kit import checks, spec  # noqa: E402

_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "enclosure")


def _load(name: str) -> dict:
    with open(os.path.join(_FIX, name), encoding="utf-8") as f:
        return json.load(f)


BOARD = _load("rpi4b_board.json")
DEFAULT = _load("pi4_spec.json")


def patched(**changes) -> dict:
    d = copy.deepcopy(DEFAULT)
    for path, value in changes.items():
        cur = d
        keys = path.split(".")
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = value
    return d


def rows_of(d=None, board=BOARD) -> dict:
    out = checks.dfm_rows(DEFAULT if d is None else d, board)
    return {r["id"]: r for r in out}


class DfmBaseTest(unittest.TestCase):
    def status(self, rid: str, d=None, board=BOARD) -> str:
        return rows_of(d, board)[rid]["status"]

    def detail(self, rid: str, d=None, board=BOARD) -> str:
        return rows_of(d, board)[rid]["detail"]


class DefaultRowsTest(DfmBaseTest):
    def test_one_row_per_id_in_order_with_the_documented_shape(self):
        rows = checks.dfm_rows(spec.default_spec(), BOARD)
        self.assertEqual([r["id"] for r in rows], list(checks.DFM_IDS))
        self.assertEqual(len(set(checks.DFM_IDS)), len(checks.DFM_IDS))
        for r in rows:
            self.assertEqual(set(r), {"id", "status", "detail"})
            self.assertIn(r["status"], checks.STATUSES)
            self.assertTrue(r["detail"])
        for rid in ("dfm.min_wall", "dfm.min_web", "dfm.hook_strain", "dfm.pocket_wall", "dfm.retention_angle", "dfm.port_web",
                    "dfm.rim_web", "dfm.standoff_h", "dfm.overhang", "dfm.shrink"):      # every id
            self.assertIn(rid, checks.DFM_IDS)

    def test_pi4_defaults(self):
        got = {r["id"]: r["status"] for r in checks.dfm_rows(DEFAULT, BOARD)}
        self.assertEqual(got, {"dfm.min_wall": "PASS", "dfm.min_web": "PASS", "dfm.hook_strain": "PASS", "dfm.hook_strain_molded": "INFO",
                               "dfm.pocket_wall": "PASS", "dfm.retention_angle": "PASS", "dfm.port_web": "PASS", "dfm.rim_web": "WARN",
                               "dfm.standoff_h": "PASS", "dfm.overhang": "PASS", "dfm.shrink": "INFO"})
        self.assertNotIn("FAIL", got.values())

    def test_spec_object_dict_and_default_all_work(self):
        self.assertEqual(checks.dfm_rows(spec.default_spec(), BOARD), checks.dfm_rows(DEFAULT, BOARD))
        self.assertEqual(checks.dfm_rows(spec.from_dict({}), BOARD), checks.dfm_rows(DEFAULT, BOARD))

    def test_pure_deterministic_and_does_not_mutate(self):
        d, b = copy.deepcopy(DEFAULT), copy.deepcopy(BOARD)
        first = checks.dfm_rows(d, b)
        self.assertEqual(checks.dfm_rows(d, b), first)
        self.assertEqual(d, DEFAULT)
        self.assertEqual(b, BOARD)
        s = spec.default_spec()
        before = s.to_dict()
        checks.dfm_rows(s, BOARD)
        self.assertEqual(s.to_dict(), before)


class MinWallTest(DfmBaseTest):
    def test_boundaries(self):
        screw = {"lid.style": "screw"}
        self.assertEqual(self.status("dfm.min_wall", patched(wall_t=2.0)), "PASS")
        self.assertEqual(self.status("dfm.min_wall", patched(wall_t=1.6, **screw)), "PASS")
        self.assertEqual(self.status("dfm.min_wall", patched(wall_t=1.59, **screw)), "WARN")          # thin for FDM
        self.assertEqual(self.status("dfm.min_wall", patched(wall_t=1.2, **screw)), "WARN")          # exactly the ABS minimum
        self.assertEqual(self.status("dfm.min_wall", patched(wall_t=1.19, **screw)), "FAIL")

    def test_floor_and_lid_count_and_are_named(self):
        self.assertEqual(self.status("dfm.min_wall", patched(floor_t=1.0)), "FAIL")
        self.assertIn("floor_t 1.0", self.detail("dfm.min_wall", patched(floor_t=1.0)))
        d = self.detail("dfm.min_wall", patched(lid_t=1.1, floor_t=1.1))
        self.assertIn("floor_t 1.1", d)
        self.assertIn("lid_t 1.1", d)
        self.assertNotIn("wall_t", d)

    def test_follows_the_material(self):
        self.assertEqual(self.status("dfm.min_wall", patched(material="PLA", wall_t=1.0, **{"lid.style": "screw"})), "WARN")
        self.assertEqual(self.status("dfm.min_wall", patched(material="ABS", wall_t=1.0, **{"lid.style": "screw"})), "FAIL")
        self.assertIn("ABS minimum 1.2", self.detail("dfm.min_wall"))


class MinWebTest(DfmBaseTest):
    def test_boundaries(self):
        self.assertEqual(self.status("dfm.min_web", patched(**{"vents.pitch": 3.2})), "PASS")          # web 1.2 = the minimum
        self.assertEqual(self.status("dfm.min_web", patched(**{"vents.pitch": 3.19})), "FAIL")
        self.assertIn("vent web 1.19 mm (pitch 3.19 - slot_w 2.0)", self.detail("dfm.min_web", patched(**{"vents.pitch": 3.19})))

    def test_a_web_of_exactly_the_minimum_passes_despite_float_noise(self):
        self.assertEqual(self.status("dfm.min_web", patched(**{"vents.pitch": 3.3, "vents.slot_w": 2.1})), "PASS")    # 3.3 - 2.1 = 1.1999999999999997
        self.assertEqual(self.status("dfm.pocket_wall", patched(wall_t=1.95, **{"hooks.overhang": 0.9})), "PASS")

    def test_disabled_vents_are_not_applicable(self):
        r = rows_of(patched(**{"vents.enabled": False}))["dfm.min_web"]
        self.assertEqual(r["status"], "INFO")
        self.assertIn("not applicable", r["detail"])

    def test_pla_has_a_smaller_minimum(self):
        self.assertEqual(self.status("dfm.min_web", patched(material="PLA", **{"vents.pitch": 3.0})), "PASS")
        self.assertEqual(self.status("dfm.min_web", patched(material="ABS", **{"vents.pitch": 3.0})), "FAIL")


class HookStrainTest(DfmBaseTest):
    def test_default_is_69_percent_of_the_fdm_limit(self):
        r = rows_of()["dfm.hook_strain"]
        self.assertEqual(r["status"], "PASS")
        self.assertIn("strain 0.0069 against the FDM limit 0.0100 (69%)", r["detail"])
        self.assertIn("deflection 0.55 mm", r["detail"])

    def test_warn_above_80_and_fail_above_100_percent(self):
        self.assertEqual(self.status("dfm.hook_strain", patched(**{"hooks.t": 1.35})), "PASS")          # 77 %
        self.assertEqual(self.status("dfm.hook_strain", patched(**{"hooks.t": 1.4})), "WARN")           # 80.2 %
        self.assertEqual(self.status("dfm.hook_strain", patched(**{"hooks.t": 1.74})), "WARN")          # 99.7 %
        self.assertEqual(self.status("dfm.hook_strain", patched(**{"hooks.t": 1.75})), "FAIL")          # 100.3 %
        self.assertEqual(self.status("dfm.hook_strain", patched(**{"hooks.length": 8.0})), "FAIL")      # shorter beam, much more strain

    def test_the_formula_uses_overhang_minus_fit_not_the_overhang(self):
        # the earlier y = overhang form would be 0.0100 (100 percent): the current form is 0.0069
        d = rows_of()["dfm.hook_strain"]["detail"]
        self.assertNotIn("(100%)", d)

    def test_limit_follows_the_material_with_the_fdm_derate(self):
        self.assertIn("FDM limit 0.0150", self.detail("dfm.hook_strain", patched(material="PETG")))
        self.assertIn("FDM limit 0.0075", self.detail("dfm.hook_strain", patched(material="PLA")))
        self.assertEqual(self.status("dfm.hook_strain", patched(material="PETG", **{"hooks.t": 1.8})), "PASS")

    def test_barb_that_does_not_reach_past_the_fit_fails(self):
        self.assertEqual(self.status("dfm.hook_strain", patched(**{"lid.fit": 0.9})), "FAIL")
        self.assertEqual(self.status("dfm.hook_strain", patched(**{"hooks.length": 0})), "FAIL")

    def test_the_moulded_limit_is_an_info_row(self):
        r = rows_of()["dfm.hook_strain_molded"]
        self.assertEqual(r["status"], "INFO")
        self.assertIn("injection-moulded limit for ABS is 0.0200", r["detail"])
        self.assertIn("34%", r["detail"])
        self.assertEqual(self.status("dfm.hook_strain_molded", patched(**{"hooks.t": 5.0})), "INFO")   # never judged

    def test_non_snap_lids_have_no_hook_rows(self):
        for style in ("screw", "friction"):
            rows = rows_of(patched(**{"lid.style": style}))
            for rid in ("dfm.hook_strain", "dfm.hook_strain_molded", "dfm.pocket_wall", "dfm.retention_angle"):
                self.assertEqual(rows[rid]["status"], "INFO", rid)
                self.assertIn("not applicable", rows[rid]["detail"])


class PocketAndRetentionTest(DfmBaseTest):
    def test_pocket_wall(self):
        r = rows_of()["dfm.pocket_wall"]
        self.assertEqual(r["status"], "PASS")
        self.assertIn("1.35 mm of wall remain behind the 0.65 mm catch pocket (minimum 1.2 mm)", r["detail"])
        self.assertEqual(self.status("dfm.pocket_wall", patched(wall_t=1.9)), "PASS")            # 1.25
        self.assertEqual(self.status("dfm.pocket_wall", patched(wall_t=1.8)), "FAIL")            # 1.15
        self.assertEqual(self.status("dfm.pocket_wall", patched(**{"hooks.overhang": 1.2})), "FAIL")   # pocket 1.05 deep: 0.95 mm remain

    def test_retention_angle(self):
        self.assertEqual(self.status("dfm.retention_angle", patched(**{"hooks.retention_deg": 60})), "PASS")
        self.assertEqual(self.status("dfm.retention_angle", patched(**{"hooks.retention_deg": 61})), "WARN")
        self.assertIn("60 degrees or less", self.detail("dfm.retention_angle", patched(**{"hooks.retention_deg": 75})))
        self.assertEqual(self.status("dfm.retention_angle", patched(**{"hooks.retention_deg": 90})), "WARN")


class PortWebTest(DfmBaseTest):
    def test_default_narrowest_web_is_between_the_two_hdmi_ports(self):
        r = rows_of()["dfm.port_web"]
        self.assertEqual(r["status"], "PASS")
        self.assertIn("narrowest web 2.5 mm between 'hdmi0' and 'hdmi1' on the -y wall (minimum 1.2 mm)", r["detail"])

    def test_overlapping_user_cutout_fails(self):
        d = patched(cutouts=[{"part": "shell", "face": "-y", "u": 32.5, "v": 3.0, "w": 6.0, "h": 6.0}])
        r = rows_of(d)["dfm.port_web"]
        self.assertEqual(r["status"], "FAIL")
        self.assertIn("overlap", r["detail"])

    def test_thin_web_next_to_a_user_cutout_fails_and_a_wide_one_passes(self):
        # hdmi0 spans u 20.5..31.5; a 6 mm cutout centred at 35.0 starts at 32.0: a 0.5 mm web
        thin = patched(cutouts=[{"part": "shell", "face": "-y", "u": 35.0, "v": 3.0, "w": 6.0, "h": 6.0, "id": "x"}])
        r = rows_of(thin)["dfm.port_web"]
        self.assertEqual(r["status"], "FAIL")
        self.assertIn("web 0.5 mm", r["detail"])
        self.assertIn("minimum 1.2 mm for ABS", r["detail"])
        # without hdmi1 and the jack, a 4 mm cutout starting at 34.0 leaves a 2.5 mm web to hdmi0 (right edge 31.5)
        ok = patched(**{"ports.exclude": ["hdmi1", "audio"]}, cutouts=[{"part": "shell", "face": "-y", "u": 36.0, "v": 3.0, "w": 4.0, "h": 4.0}])
        self.assertEqual(self.status("dfm.port_web", ok), "PASS")

    def test_vertical_separation_counts_as_web(self):
        # same u range, 3 mm above the HDMI opening: the horizontal gap is negative but the vertical web is large
        d = patched(**{"ports.exclude": ["usb_c_power", "hdmi1", "audio"]},
                    cutouts=[{"part": "shell", "face": "-y", "u": 26.0, "v": 12.0, "w": 6.0, "h": 3.0}])
        self.assertEqual(self.status("dfm.port_web", d), "PASS")

    def test_no_wall_with_two_openings_is_not_applicable(self):
        d = patched(**{"ports.auto": False})
        r = rows_of(d)["dfm.port_web"]
        self.assertEqual(r["status"], "INFO")
        self.assertIn("not applicable", r["detail"])


class RimWebTest(DfmBaseTest):
    def test_default_usb_pairs_merge_with_the_rim(self):
        r = rows_of()["dfm.rim_web"]
        self.assertEqual(r["status"], "WARN")
        self.assertIn("opening 'usb3_pair' merged with the rim (web 0.5 < 1.2)", r["detail"])
        self.assertIn("opening 'usb2_pair' merged with the rim (web 0.5 < 1.2)", r["detail"])
        self.assertNotIn("ethernet", r["detail"])           # top at 15.4: a 2.6 mm web
        self.assertIn("notch", r["detail"])

    def test_a_taller_cavity_or_excluded_ports_remove_the_warning(self):
        self.assertEqual(self.status("dfm.rim_web", patched(**{"lid.gap": 2.0})), "PASS")           # rim 19.0: web 1.5
        self.assertEqual(self.status("dfm.rim_web", patched(**{"lid.gap": 1.7})), "PASS")           # web 1.2 exactly
        self.assertEqual(self.status("dfm.rim_web", patched(**{"lid.gap": 1.6})), "WARN")
        self.assertEqual(self.status("dfm.rim_web", patched(**{"ports.exclude": ["usb3_pair", "usb2_pair"]})), "PASS")
        self.assertIn("keeps at least 2.6 mm below the rim ('ethernet'; minimum 1.2 mm)",
                      self.detail("dfm.rim_web", patched(**{"ports.exclude": ["usb3_pair", "usb2_pair"]})))

    def test_no_openings_is_not_applicable(self):
        self.assertEqual(self.status("dfm.rim_web", patched(**{"ports.auto": False})), "INFO")


class StandoffTest(DfmBaseTest):
    def test_default_clears_the_microsd_socket(self):
        r = rows_of()["dfm.standoff_h"]
        self.assertEqual(r["status"], "PASS")
        self.assertIn("standoff 4.0 mm leaves 2.5 mm under the lowest underside part (z -1.5); minimum standoff 2.5 mm", r["detail"])

    def test_boundaries(self):
        self.assertEqual(self.status("dfm.standoff_h", patched(**{"standoff.h": 2.5})), "PASS")
        r = rows_of(patched(**{"standoff.h": 2.4}))["dfm.standoff_h"]
        self.assertEqual(r["status"], "FAIL")
        self.assertIn("needs 2.5 mm or more", r["detail"])
        self.assertIn("leaves 0.9 mm", r["detail"])

    def test_board_without_underside_parts_needs_two_mm(self):
        flat = copy.deepcopy(BOARD)
        flat["height"]["min_bottom_z"] = 0.0
        self.assertEqual(self.status("dfm.standoff_h", patched(**{"standoff.h": 2.0}), flat), "PASS")
        self.assertEqual(self.status("dfm.standoff_h", patched(**{"standoff.h": 1.9}), flat), "FAIL")
        self.assertIn("under the board", self.detail("dfm.standoff_h", patched(**{"standoff.h": 3.0}), flat))


class OverhangAndShrinkTest(DfmBaseTest):
    def test_default_prints_without_supports(self):
        r = rows_of()["dfm.overhang"]
        self.assertEqual(r["status"], "PASS")
        self.assertIn("the lid prints top_down, the shell floor_down", r["detail"])

    def test_warn_only_never_fail(self):
        for d in (patched(**{"print_orientation.lid": "as_modeled"}),
                  patched(**{"print_orientation.lid": "as_modeled", "hooks.ramp_deg": 50}),
                  patched(**{"hooks.retention_deg": 70})):
            self.assertEqual(self.status("dfm.overhang", d), "WARN")
        self.assertIn("bridges over the skirt", self.detail("dfm.overhang", patched(**{"print_orientation.lid": "as_modeled"})))
        self.assertIn("lead-in ramp (50 degrees)", self.detail("dfm.overhang", patched(**{"print_orientation.lid": "as_modeled", "hooks.ramp_deg": 50})))
        self.assertEqual(self.status("dfm.overhang", patched(**{"hooks.retention_deg": 45})), "PASS")
        self.assertEqual(self.status("dfm.overhang", patched(**{"lid.style": "screw", "hooks.retention_deg": 70})), "PASS")

    def test_shrink_is_always_info(self):
        self.assertEqual(rows_of()["dfm.shrink"], {"id": "dfm.shrink", "status": "INFO", "detail": "ABS shrinkage 0.8% compensated by clearance only"})
        self.assertIn("PLA shrinkage 0.3% compensated by clearance only", self.detail("dfm.shrink", patched(material="PLA")))
        self.assertIn("is not compensated (shrink_comp 'none')", self.detail("dfm.shrink", patched(shrink_comp="none")))
        self.assertIn("'scale' is not supported yet", self.detail("dfm.shrink", patched(shrink_comp="scale")))
        for m in ("ABS", "ASA", "PLA", "PETG", "PC", "NYLON"):
            self.assertEqual(self.status("dfm.shrink", patched(material=m)), "INFO")


class RobustnessTest(DfmBaseTest):
    def test_unknown_material_degrades_to_warnings(self):
        rows = checks.dfm_rows(patched(material="WOOD"), BOARD)
        self.assertEqual([r["id"] for r in rows], list(checks.DFM_IDS))
        self.assertTrue(all(r["status"] == "WARN" for r in rows))
        self.assertIn("could not be evaluated", rows[0]["detail"])
        self.assertIn("WOOD", rows[0]["detail"])

    def test_incomplete_board_degrades_to_warnings_never_raises(self):
        for bad in ({}, {"id": "x"}, {"id": "x", "outline": {}}):
            rows = checks.dfm_rows(DEFAULT, bad)
            self.assertEqual(len(rows), len(checks.DFM_IDS))
            self.assertTrue(all(r["status"] == "WARN" for r in rows), bad)

    def test_garbage_specs_never_raise(self):
        for d in (patched(wall_t="thick"), patched(**{"hooks.count": None}), patched(vents=None), {"lid": 3}, {}, patched(**{"vents.pitch": 0}),
                  patched(**{"hooks.length": "x"}), patched(cutouts=[{"part": "shell"}, 7, None])):
            rows = checks.dfm_rows(d, BOARD)
            self.assertEqual(len(rows), len(checks.DFM_IDS))
            self.assertTrue(all(r["status"] in checks.STATUSES for r in rows))

    def test_every_other_board_in_the_library_evaluates(self):
        cat = spec._catalog()
        if cat is None:
            self.skipTest("refparts.catalog not installed")
        for info in cat.list_boards():
            board = cat.load_board(info["id"])
            rows = checks.dfm_rows(spec.default_spec(board["id"]), board)
            self.assertEqual(len(rows), len(checks.DFM_IDS), board["id"])
            self.assertFalse([r for r in rows if "could not be evaluated" in r["detail"]], board["id"])


if __name__ == "__main__":
    unittest.main()
