"""Tests of the reference-part catalog and the board library (``oscar.refparts.catalog``).

Pure data and logic: no FreeCAD, no network. Covers the shipped data (the Pi 4 file is pinned, every other
board is honest and unverified), alias matching, the derived openings (7.1 / 7.1a numbers), and the board validator with
deliberately broken boards, one per rule.
"""

import copy
import json
import math
import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.refparts import catalog  # noqa: E402

_PI4_JSON = json.dumps(catalog.load_board("rpi4b"))      # read before any test patches the data directory
BOARD_IDS = ["rpi4b", "rpi3bplus", "rpi5", "rpizero2w", "rpi_pico", "arduino_uno", "arduino_nano", "esp32_devkitc", "jetson_nano"]


def good() -> dict:
    """A fresh, valid Pi 4 board dict to break."""
    return catalog.load_board("rpi4b")


def conn(board: dict, cid: str) -> dict:
    return next(c for c in board["connectors"] if c["id"] == cid)


def check(board) -> tuple:
    ok, errors, warnings = catalog.validate_board(board)
    assert isinstance(ok, bool) and isinstance(errors, list) and isinstance(warnings, list)
    assert ok == (not errors)
    return ok, errors, warnings


class _Base(unittest.TestCase):
    def assertError(self, board, *fragments):
        ok, errors, _w = check(board)
        self.assertFalse(ok, "expected the board to be rejected")
        text = "\n".join(errors)
        for fragment in fragments:
            self.assertIn(fragment, text)
        return errors

    def assertWarning(self, board, *fragments):
        ok, errors, warnings = check(board)
        self.assertTrue(ok, f"unexpected errors: {errors}")
        text = "\n".join(warnings)
        for fragment in fragments:
            self.assertIn(fragment, text)
        return warnings


# --- the shipped data ---------------------------------------------------------

class CatalogDataTest(unittest.TestCase):
    def test_catalog_file_validates(self):
        ok, errors, warnings = catalog.validate_catalog()
        self.assertTrue(ok, errors)
        self.assertEqual(errors, [])

    def test_allow_hosts_are_exactly_the_contract_list(self):
        # migration (STEP sourcing): the two verified vendor hosts were added; pip.raspberrypi.com redirects to pip-assets.raspberrypi.com
        self.assertEqual(catalog.allow_hosts(), ["raw.githubusercontent.com", "github.com", "datasheets.raspberrypi.com",
                                                  "www.raspberrypi.com", "pip.raspberrypi.com", "pip-assets.raspberrypi.com"])

    def test_entries_are_the_nine_boards_in_order(self):
        self.assertEqual([e["id"] for e in catalog.list_entries()], BOARD_IDS)
        self.assertEqual([b["id"] for b in catalog.list_boards()], BOARD_IDS)

    def test_every_entry_has_the_contract_fields(self):
        for entry in catalog.list_entries():
            for key in ("id", "name", "aliases", "kind", "board", "vendor_urls", "datasheet_urls", "step_urls", "license",
                        "last_verified", "verified"):
                self.assertIn(key, entry, f"{entry['id']} lacks {key}")
            self.assertEqual(entry["board"], f"boards/{entry['id']}.json")

    def test_no_step_url_is_invented_and_nothing_is_marked_verified(self):
        # migration (STEP sourcing): exactly the two files that were downloaded and verified on 2026-10-02 are listed; nothing else
        hosts = set(catalog.allow_hosts())
        for entry in catalog.list_entries():
            self.assertFalse(entry["verified"], entry["id"])
            listed = entry["id"] in ("rpi4b", "rpi5")
            self.assertEqual(entry["last_verified"], "2026-10-02" if listed else "", entry["id"])
            self.assertEqual(len(entry["step_urls"]), 1 if listed else 0, entry["id"])
            for step in entry["step_urls"]:   # any URL must be https on an allowlisted host
                host = step["url"].split("/")[2]
                self.assertTrue(step["url"].startswith("https://") and host in hosts, step)

    def test_pi4_entry_matches_the_contract(self):
        entry = catalog.get("rpi4b")
        self.assertEqual(entry["name"], "Raspberry Pi 4 Model B")
        self.assertEqual(entry["kind"], "sbc")
        self.assertEqual(entry["aliases"], ["raspberry pi 4", "raspberry pi 4 model b", "raspberry pi 4b", "rpi 4", "rpi4", "rpi4b",
                                            "pi 4", "pi4"])
        self.assertEqual(entry["vendor_urls"], ["https://www.raspberrypi.com/products/raspberry-pi-4-model-b/"])
        self.assertEqual(entry["datasheet_urls"],
                         ["https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-mechanical-drawing.pdf"])

    def test_get_is_case_insensitive_and_unknown_lists_the_known_ids(self):
        self.assertEqual(catalog.get(" RPI4B ")["id"], "rpi4b")
        with self.assertRaises(KeyError) as cm:
            catalog.get("nope")
        message = str(cm.exception)
        self.assertIn("nope", message)
        for board_id in BOARD_IDS:
            self.assertIn(board_id, message)
        for bad in ("", None):
            with self.assertRaises(KeyError):
                catalog.get(bad)

    def test_callers_get_copies(self):
        entry = catalog.get("rpi4b")
        entry["aliases"].append("zzz")
        entry["step_urls"].append({"url": "x"})
        self.assertNotIn("zzz", catalog.get("rpi4b")["aliases"])
        self.assertEqual(len(catalog.get("rpi4b")["step_urls"]), 1)
        listed = catalog.list_entries()
        listed[0]["name"] = "changed"
        self.assertEqual(catalog.list_entries()[0]["name"], "Raspberry Pi 4 Model B")
        hosts = catalog.allow_hosts()
        hosts.append("evil.example.com")
        self.assertNotIn("evil.example.com", catalog.allow_hosts())

    def test_list_boards_shape(self):
        for item in catalog.list_boards():
            self.assertEqual(sorted(item), ["aliases", "id", "name"])
            self.assertEqual(item["aliases"], catalog.get(item["id"])["aliases"])


class DataHygieneTest(unittest.TestCase):
    """The JSON files themselves: strict, no duplicate keys (they would silently override each other), no orphans."""

    @staticmethod
    def strict_load(path):
        def no_duplicates(pairs):
            keys = [k for k, _v in pairs]
            dupes = {k for k in keys if keys.count(k) > 1}
            if dupes:
                raise ValueError(f"{os.path.basename(path)}: duplicate keys {sorted(dupes)}")
            return dict(pairs)

        def no_constants(name):
            raise ValueError(f"{os.path.basename(path)}: {name} is not valid JSON")

        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh, object_pairs_hook=no_duplicates, parse_constant=no_constants)

    def test_every_data_file_is_strict_json_without_duplicate_keys(self):
        paths = [catalog.CATALOG_PATH] + [os.path.join(catalog.DATA_DIR, "boards", n)
                                          for n in sorted(os.listdir(os.path.join(catalog.DATA_DIR, "boards"))) if n.endswith(".json")]
        self.assertEqual(len(paths), 1 + len(BOARD_IDS))
        for path in paths:
            self.strict_load(path)

    def test_no_orphan_board_files_and_ids_equal_file_stems(self):
        listed = sorted(n[:-5] for n in os.listdir(os.path.join(catalog.DATA_DIR, "boards")) if n.endswith(".json"))
        self.assertEqual(listed, sorted(BOARD_IDS))
        for board_id in BOARD_IDS:
            raw = self.strict_load(os.path.join(catalog.DATA_DIR, "boards", f"{board_id}.json"))
            self.assertEqual(raw["id"], board_id)
            self.assertEqual(raw["schema"], 1)

    def test_the_pi4_file_derived_height_is_the_contract_one(self):
        raw = self.strict_load(os.path.join(catalog.DATA_DIR, "boards", "rpi4b.json"))
        self.assertEqual(raw["height"], {"max_top_z": 17.0, "min_bottom_z": -1.5})
        self.assertEqual(max(c["box"]["z"][1] for c in raw["connectors"]), 17.0)
        self.assertEqual(min(c["box"]["z"][0] for c in raw["connectors"]), -1.5)


# --- matching -----------------------------------------------------------------

class MatchTest(unittest.TestCase):
    def first(self, text):
        found = catalog.match(text)
        return found[0]["id"] if found else None

    def test_pi4_spellings(self):
        for text in ("Raspberry Pi 4 Model B", "raspberry pi 4", "Raspberry Pi 4B", "rpi4", "RPi4", "rpi 4", "rpi4b", "Pi 4", "pi4",
                     "  PI-4  "):
            self.assertEqual(self.first(text), "rpi4b", text)

    def test_a_whole_request_that_mentions_the_part(self):
        text = "Design a snap-fit enclosure for my Raspberry Pi 4 Model B with vents and a lid."
        found = catalog.match(text)
        self.assertEqual(found[0]["id"], "rpi4b")
        self.assertEqual([e["id"] for e in found], ["rpi4b"])

    def test_other_boards(self):
        expected = {
            "raspberry pi 3b+": "rpi3bplus", "Raspberry Pi 3 Model B+": "rpi3bplus", "rpi3b+": "rpi3bplus",
            "raspberry pi 5": "rpi5", "RPi 5": "rpi5", "pi5": "rpi5",
            "raspberry pi zero 2 w": "rpizero2w", "pi zero 2w": "rpizero2w",
            "raspberry pi pico": "rpi_pico", "pico": "rpi_pico",
            "arduino uno": "arduino_uno", "Arduino Uno Rev3": "arduino_uno", "arduino nano": "arduino_nano",
            "esp32 devkitc": "esp32_devkitc", "ESP32-DevKitC V4": "esp32_devkitc",
            "jetson nano": "jetson_nano", "NVIDIA Jetson Nano Developer Kit": "jetson_nano",
        }
        for text, board_id in expected.items():
            self.assertEqual(self.first(text), board_id, text)

    def test_words_never_match_partially(self):
        self.assertEqual(catalog.match("raspberry pi 400"), [])
        self.assertEqual(catalog.match("pi 40"), [])
        self.assertNotIn("rpi4b", [e["id"] for e in catalog.match("raspberry pi 5")])
        self.assertNotIn("rpi5", [e["id"] for e in catalog.match("raspberry pi 4")])

    def test_unrelated_empty_and_odd_input_match_nothing(self):
        for text in ("a plain box with a lid", "", "   ", None, 42, "!!!", "x"):
            self.assertEqual(catalog.match(text), [], repr(text))

    def test_fragment_of_a_name_is_ambiguous_but_deterministic(self):
        found = [e["id"] for e in catalog.match("arduino")]
        self.assertEqual(found, ["arduino_uno", "arduino_nano"])
        self.assertEqual(found, [e["id"] for e in catalog.match("arduino")])

    def test_longer_alias_wins_inside_a_request(self):
        scored = catalog.match_scored("I own a raspberry pi 3 model b plus and also a pi 4")
        self.assertEqual(scored[0][1]["id"], "rpi3bplus")
        self.assertIn("rpi4b", [e["id"] for _s, e in scored])
        scores = [s for s, _e in scored]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_ties_keep_catalog_order_and_limit_caps_the_list(self):
        ids = [e["id"] for e in catalog.match("raspberry pi 4 and raspberry pi 5")]
        self.assertEqual(ids[:2], ["rpi4b", "rpi5"])
        self.assertEqual([e["id"] for e in catalog.match("raspberry pi 4 and raspberry pi 5", limit=1)], ["rpi4b"])

    def test_plus_sign_reads_as_the_word_plus(self):
        self.assertEqual(self.first("pi 3b+"), "rpi3bplus")
        self.assertEqual(self.first("pi 3b plus"), "rpi3bplus")

    def test_results_are_copies(self):
        found = catalog.match("pi4")
        found[0]["aliases"].clear()
        self.assertTrue(catalog.match("pi4")[0]["aliases"])

    def test_every_alias_of_every_entry_finds_its_own_entry(self):
        for entry in catalog.list_entries():
            for alias in entry["aliases"]:
                ids = [e["id"] for e in catalog.match(alias)]
                self.assertIn(entry["id"], ids, alias)


# --- boards and derived openings ---------------------------------------------

class BoardLibraryTest(unittest.TestCase):
    def test_every_board_passes_the_validator_without_warnings(self):
        for board_id in BOARD_IDS:
            board = catalog.load_board(board_id)
            ok, errors, warnings = catalog.validate_board(board)
            self.assertTrue(ok, f"{board_id}: {errors}")
            self.assertEqual(warnings, [], board_id)

    def test_every_board_is_unverified_and_honest(self):
        for board_id in BOARD_IDS:
            board = catalog.load_board(board_id)
            self.assertEqual(board["id"], board_id)
            self.assertIs(board["verified"], False, board_id)
            nodes = [board["outline"], board["pcb"], board["mounting_holes"]] + board["connectors"]
            for node in nodes:
                self.assertIn(node["confidence"], catalog.CONFIDENCE_LEVELS)
                if node["confidence"] == "measured":   # migration: only the official Pi 5 STEP was measured (2026-10-02), and it says so
                    self.assertEqual(board_id, "rpi5")
                    self.assertIn("official Raspberry Pi 5 STEP", node["source"])
                    self.assertEqual(board["step_cross_check"]["trust"], "official")
                self.assertTrue(node["source"].strip())
                if node["confidence"] == "datasheet":
                    self.assertTrue(node["source_url"].startswith("https://"))
            self.assertEqual(board["pcb"]["confidence"] in ("assumed", "community", "approximate") or board_id == "rpi5", True)
            self.assertTrue(board["verify_against"].startswith("https://"))

    def test_only_the_pi_family_claims_datasheet_outlines(self):
        datasheet = [b for b in BOARD_IDS if catalog.load_board(b)["outline"]["confidence"] == "datasheet"]
        self.assertEqual(datasheet, ["rpi4b", "rpi3bplus", "rpi5"])
        for board_id in BOARD_IDS:
            if board_id not in datasheet:
                self.assertEqual(catalog.load_board(board_id)["outline"]["confidence"], "community")

    def test_unknown_board_lists_the_known_ids(self):
        with self.assertRaises(KeyError) as cm:
            catalog.load_board("pi99")
        for board_id in BOARD_IDS:
            self.assertIn(board_id, str(cm.exception))
        with self.assertRaises(KeyError):
            catalog.load_board("")
        self.assertEqual(catalog.load_board(" RPi4B ")["id"], "rpi4b")

    def test_load_board_returns_independent_copies(self):
        a = catalog.load_board("rpi4b")
        a["connectors"][0]["box"]["x"][0] = 99.0
        a["outline"]["length"] = 1.0
        b = catalog.load_board("rpi4b")
        self.assertEqual(b["connectors"][0]["box"]["x"][0], 6.7)
        self.assertEqual(b["outline"]["length"], 85.0)

    def test_connectors_serialise_and_copy_like_plain_dicts(self):
        board = catalog.load_board("rpi4b")
        self.assertEqual(json.loads(json.dumps(board)), json.loads(json.dumps(copy.deepcopy(board))))
        c = copy.deepcopy(board["connectors"][0])
        self.assertEqual(c["id"], "usb_c_power")
        self.assertEqual(catalog.opening(c, board=board)["protrusion"], 0.5)
        ethernet = conn(board, "ethernet")
        for clone in (copy.deepcopy(ethernet), copy.copy(ethernet), copy.deepcopy(board)["connectors"][4]):
            self.assertEqual(catalog.opening(clone)["protrusion"], 2.0)       # the clone still knows its board's length

    def test_load_board_is_thread_safe(self):
        results, errors = [], []

        def work():
            try:
                results.append(catalog.load_board("rpi4b")["height"])
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(results, [{"max_top_z": 17.0, "min_bottom_z": -1.5}] * 8)


class Pi4DataTest(unittest.TestCase):
    """The Pi 4 file is pinned: outline, holes, every connector box, the height."""

    EXPECTED = {   # id: (face, x, y, z, margin_hint, confidence)
        "usb_c_power": ("-y", (6.7, 15.7), (-0.5, 6.9), (1.4, 4.7), 0.4, "community"),
        "hdmi0": ("-y", (22.65, 29.35), (-0.5, 5.5), (1.4, 4.5), 0.4, "community"),
        "hdmi1": ("-y", (36.15, 42.85), (-0.5, 5.5), (1.4, 4.5), 0.4, "community"),
        "audio": ("-y", (50.0, 57.0), (-1.0, 12.0), (1.4, 7.4), 0.5, "community"),
        "ethernet": ("+x", (65.8, 87.0), (37.75, 53.75), (1.4, 14.9), 0.5, "community"),
        "usb3_pair": ("+x", (69.5, 87.0), (20.45, 33.55), (1.4, 17.0), 0.5, "community"),
        "usb2_pair": ("+x", (69.5, 87.0), (2.45, 15.55), (1.4, 17.0), 0.5, "community"),
        "microsd": ("-x", (-2.5, 12.0), (22.0, 34.0), (-1.5, 0.0), 0.6, "approximate"),
        "gpio": ("top", (7.1, 57.9), (49.96, 55.04), (1.4, 9.9), 0.0, "community"),
    }

    def setUp(self):
        self.board = catalog.load_board("rpi4b")

    def test_identity_outline_pcb_and_holes(self):
        b = self.board
        self.assertEqual((b["schema"], b["id"], b["name"], b["kind"]), (1, "rpi4b", "Raspberry Pi 4 Model B", "sbc"))
        self.assertEqual(b["frame"], {"origin": "pcb_bottom_lower_left", "x": "long_edge", "y": "short_edge",
                                      "z": "up_through_pcb", "units": "mm"})
        o = b["outline"]
        self.assertEqual((o["shape"], o["length"], o["width"], o["corner_radius"], o["confidence"]),
                         ("rounded_rect", 85.0, 56.0, 3.0, "datasheet"))
        self.assertEqual((b["pcb"]["thickness"], b["pcb"]["confidence"]), (1.4, "assumed"))
        h = b["mounting_holes"]
        self.assertEqual((h["diameter"], h["pad_diameter"], h["thread"], h["confidence"]), (2.75, 6.2, "M2.5", "datasheet"))
        self.assertEqual(h["positions"], [[3.5, 3.5], [61.5, 3.5], [3.5, 52.5], [61.5, 52.5]])
        self.assertEqual(h["pattern"], {"offset": [3.5, 3.5], "dx": 58.0, "dy": 49.0})
        self.assertIs(b["verified"], False)
        self.assertEqual(b["height"], {"max_top_z": 17.0, "min_bottom_z": -1.5})
        self.assertEqual(b["verify_against"], "https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-mechanical-drawing.pdf")

    def test_connector_boxes_faces_and_confidences(self):
        self.assertEqual([c["id"] for c in self.board["connectors"]], list(self.EXPECTED))
        for c in self.board["connectors"]:
            face, x, y, z, hint, confidence = self.EXPECTED[c["id"]]
            self.assertEqual(c["face"], face, c["id"])
            self.assertEqual((tuple(c["box"]["x"]), tuple(c["box"]["y"]), tuple(c["box"]["z"])), (x, y, z), c["id"])
            self.assertEqual((c["margin_hint"], c["confidence"]), (hint, confidence), c["id"])
            self.assertEqual(c["cutout"], c["id"] != "gpio")
            self.assertTrue(c["source"] and c["source_url"].startswith("https://"))

    def test_derived_receptacle_openings_of_section_7_1(self):
        def got(cid):
            o = catalog.opening(conn(self.board, cid))
            return tuple(o[k] for k in ("face", "u", "v", "w", "h", "protrusion"))

        expected = {
            "usb_c_power": ("-y", 11.2, 3.05, 9.0, 3.3, 0.5),
            "hdmi0": ("-y", 26.0, 2.95, 6.7, 3.1, 0.5),
            "hdmi1": ("-y", 39.5, 2.95, 6.7, 3.1, 0.5),
            "audio": ("-y", 53.5, 4.4, 7.0, 6.0, 1.0),
            "ethernet": ("+x", 45.75, 8.15, 16.0, 13.5, 2.0),
            "usb3_pair": ("+x", 27.0, 9.2, 13.1, 15.6, 2.0),
            "usb2_pair": ("+x", 9.0, 9.2, 13.1, 15.6, 2.0),
            "microsd": ("-x", 28.0, -0.75, 12.0, 1.5, 2.5),
            "gpio": ("top", 32.5, 52.5, 50.8, 5.08, 0.0),
        }
        for cid, want in expected.items():
            have = got(cid)
            self.assertEqual(have[0], want[0], cid)
            for a, b in zip(have[1:], want[1:]):
                self.assertAlmostEqual(a, b, places=4, msg=cid)

    def test_plug_clearance_numbers_of_section_7_1a(self):
        usb_c = catalog.opening(conn(self.board, "usb_c_power"), margin=0.4)
        self.assertEqual((usb_c["w"], usb_c["h"], usb_c["u"], usb_c["v"]), (12.0, 6.5, 11.2, 3.05))
        for cid, u in (("hdmi0", 26.0), ("hdmi1", 39.5)):
            o = catalog.opening(conn(self.board, cid), margin=0.4)
            self.assertEqual((o["w"], o["h"], o["u"], o["v"]), (11.0, 7.0, u, 2.95))
        audio = catalog.opening(conn(self.board, "audio"), margin=0.5)
        self.assertEqual((audio["w"], audio["h"], audio["u"], audio["v"]), (9.0, 9.0, 53.5, 4.4))

        def span(cid, margin):
            o = catalog.opening(conn(self.board, cid), margin=margin)
            return o["u"] - o["w"] / 2, o["u"] + o["w"] / 2

        webs = [span("hdmi0", 0.4)[0] - span("usb_c_power", 0.4)[1],
                span("hdmi1", 0.4)[0] - span("hdmi0", 0.4)[1],
                span("audio", 0.5)[0] - span("hdmi1", 0.4)[1]]
        self.assertEqual([round(w, 3) for w in webs], [3.3, 2.5, 4.0])
        self.assertEqual(round(span("audio", 0.5)[1], 6), 58.0)

    def test_receptacle_only_openings_are_unaffected_by_plug_data(self):
        self.assertEqual(catalog.opening(conn(self.board, "usb_c_power"), margin=None)["w"], 9.0)
        o = catalog.opening(conn(self.board, "usb_c_power"), margin=0.4, plug=False)
        self.assertEqual((o["w"], o["h"]), (9.8, 4.1))     # the add_cutout example
        rj45 = catalog.opening(conn(self.board, "ethernet"), margin=0.5)
        self.assertEqual((rj45["w"], rj45["h"]), (17.0, 14.5))    # receptacles that hold their plug have no plug size
        self.assertEqual(catalog.opening(conn(self.board, "microsd"), margin=0.6)["w"], 13.2)


class OpeningTest(unittest.TestCase):
    def setUp(self):
        self.board = good()

    def test_plug_defaults_are_the_contract_values(self):
        self.assertEqual(catalog.PLUG_DEFAULTS, {"usb_c": (12.0, 6.5), "micro_hdmi": (11.0, 7.0), "audio_jack": (9.0, 9.0)})
        self.assertEqual(catalog.plug_size({"kind": "usb_c"}), (12.0, 6.5))
        self.assertEqual(catalog.plug_size({"kind": "rj45"}), (0.0, 0.0))
        self.assertEqual(catalog.plug_size({}), (0.0, 0.0))

    def test_per_connector_plug_overrides_win(self):
        c = copy.deepcopy(conn(self.board, "usb_c_power"))
        c["plug_w"], c["plug_h"] = 14.0, 8.0
        o = catalog.opening(c, margin=0.4, board=self.board)
        self.assertEqual((o["w"], o["h"]), (14.0, 8.0))
        self.assertEqual(catalog.plug_size(c), (14.0, 8.0))
        c["plug_h"] = None     # a half override keeps the default for the other axis
        self.assertEqual(catalog.plug_size(c), (14.0, 6.5))

    def test_a_small_plug_never_shrinks_the_opening(self):
        c = copy.deepcopy(conn(self.board, "usb_c_power"))
        c["plug_w"], c["plug_h"] = 5.0, 2.0
        o = catalog.opening(c, margin=0.4, board=self.board)
        self.assertEqual((o["w"], o["h"]), (9.8, 4.1))

    def test_margin_zero_is_allowed_and_negative_is_not(self):
        o = catalog.opening(conn(self.board, "ethernet"), margin=0)
        self.assertEqual((o["w"], o["h"]), (16.0, 13.5))
        for bad in (-0.1, "x", float("nan"), True):
            with self.assertRaises(ValueError):
                catalog.opening(conn(self.board, "ethernet"), margin=bad)

    def test_plus_faces_need_the_board_size(self):
        bare = copy.deepcopy(dict(conn(self.board, "ethernet")))
        plain = json.loads(json.dumps(bare))                       # a plain dict that does not remember its board
        with self.assertRaises(ValueError) as cm:
            catalog.opening(plain)
        self.assertIn("board=", str(cm.exception))
        self.assertEqual(catalog.opening(plain, board=self.board)["protrusion"], 2.0)
        self.assertEqual(catalog.opening(conn(self.board, "ethernet"))["protrusion"], 2.0)   # load_board's connectors carry it
        self.assertEqual(catalog.opening(plain, board={"outline": {"length": 90.0, "width": 56.0}})["protrusion"], 0.0)

    def test_plus_y_and_minus_x_protrusion(self):
        wall = {"id": "x", "name": "x", "kind": "k", "face": "+y", "box": {"x": [10, 20], "y": [50, 59], "z": [1, 5]}}
        board = {"outline": {"length": 85.0, "width": 56.0}}
        o = catalog.opening(wall, board=board)
        self.assertEqual((o["face"], o["u"], o["v"], o["w"], o["h"], o["protrusion"]), ("+y", 15.0, 3.0, 10.0, 4.0, 3.0))
        side = {"id": "y", "name": "y", "kind": "k", "face": "-x", "box": {"x": [-4, 6], "y": [10, 30], "z": [0, 2]}}
        o = catalog.opening(side)
        self.assertEqual((o["u"], o["w"], o["protrusion"]), (20.0, 20.0, 4.0))
        inside = {"id": "z", "name": "z", "kind": "k", "face": "-y", "box": {"x": [1, 3], "y": [2, 5], "z": [0, 1]}}
        self.assertEqual(catalog.opening(inside)["protrusion"], 0.0)

    def test_bad_input_raises_value_error_naming_the_connector(self):
        with self.assertRaises(ValueError):
            catalog.opening("usb")
        with self.assertRaises(ValueError) as cm:
            catalog.opening({"id": "bad", "face": "left", "box": {}})
        self.assertIn("'bad'", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            catalog.opening({"id": "nobox", "face": "-y"})
        self.assertIn("'nobox'", str(cm.exception))
        with self.assertRaises(ValueError):
            catalog.opening({"id": "b", "face": "-y", "box": {"x": [1, "a"], "y": [0, 1], "z": [0, 1]}})

    def test_results_are_rounded_and_have_exactly_the_contract_keys(self):
        o = catalog.opening(conn(self.board, "hdmi0"))
        self.assertEqual(sorted(o), ["face", "h", "protrusion", "u", "v", "w"])
        self.assertEqual(o["w"], 6.7)
        self.assertEqual(o["u"], 26.0)


# --- the validator, with deliberately broken boards ---------------------------

class ValidatorBasicsTest(_Base):
    def test_the_pi4_file_is_clean(self):
        self.assertEqual(check(good()), (True, [], []))

    def test_never_raises_on_garbage(self):
        for garbage in (None, [], "board", 42, {}, {"connectors": "x"}, {"outline": 3, "pcb": [], "mounting_holes": "z"}):
            ok, errors, _w = check(garbage)
            self.assertFalse(ok, repr(garbage))
            self.assertTrue(errors)
        self.assertEqual(check({})[1][0].split(":")[0], "board")

    def test_every_required_top_level_key_is_enforced(self):
        for key in catalog._REQUIRED_BOARD_KEYS:
            board = good()
            del board[key]
            self.assertError(board, f"'{key}'")

    def test_null_values_are_errors_not_silence(self):
        for key in catalog._REQUIRED_BOARD_KEYS:
            board = good()
            board[key] = None
            ok, errors, _w = check(board)
            self.assertFalse(ok, key)

    def test_replacing_each_top_level_value_with_junk_is_rejected_without_raising(self):
        for key in catalog._REQUIRED_BOARD_KEYS:
            for junk in ({"__junk__": 1}, [3], 12345):
                board = good()
                board[key] = junk
                ok, errors, _w = check(board)
                self.assertFalse(ok, f"{key}={junk!r} was accepted")

    def test_schema_id_name_kind_frame_and_flags(self):
        b = good()
        b["schema"] = 2
        self.assertError(b, "schema: 2 is not supported")
        b = good()
        b["id"] = "Pi 4"
        self.assertError(b, "id: 'Pi 4'")
        b = good()
        b["name"] = "  "
        self.assertError(b, "name:")
        b = good()
        b["kind"] = "laptop"
        self.assertError(b, "kind: 'laptop'")
        b = good()
        b["aliases"] = ["ok", ""]
        self.assertError(b, "aliases:")
        b = good()
        b["frame"] = dict(b["frame"], origin="board_centre")
        self.assertError(b, "frame:")
        b = good()
        b["verified"] = "yes"
        self.assertError(b, "verified: must be true or false")
        b = good()
        b["verify_against"] = ""
        self.assertError(b, "verify_against")
        b = good()
        b["omitted"] = "all of it"
        self.assertError(b, "omitted:")

    def test_messages_follow_the_error_style(self):
        b = good()
        conn(b, "hdmi0")["box"]["y"] = [-9.0, 5.5]
        for message in check(b)[1]:
            self.assertIn(":", message)
            self.assertIn("hdmi0", message)
            self.assertFalse(message.endswith("."), message)


class ProvenanceRuleTest(_Base):
    """Rule 1."""

    def test_confidence_must_be_in_the_vocabulary(self):
        b = good()
        b["outline"]["confidence"] = "certain"
        self.assertError(b, "outline: confidence 'certain'")
        b = good()
        conn(b, "audio")["confidence"] = "guess"
        self.assertError(b, "connector 'audio': confidence 'guess'")
        b = good()
        del b["pcb"]["confidence"]
        self.assertError(b, "pcb: confidence None")

    def test_every_vocabulary_level_is_accepted_when_it_has_its_evidence(self):
        for level in catalog.CONFIDENCE_LEVELS:
            b = good()
            conn(b, "audio")["confidence"] = level
            self.assertEqual(check(b)[0], True, level)

    def test_source_must_be_non_empty(self):
        for path in ("outline", "pcb", "mounting_holes"):
            b = good()
            b[path]["source"] = "  "
            self.assertError(b, f"{path}: 'source' is missing or empty")
        b = good()
        del conn(b, "ethernet")["source"]
        self.assertError(b, "connector 'ethernet': 'source' is missing")
        b = good()
        conn(b, "ethernet")["source"] = 5
        self.assertError(b, "connector 'ethernet'")

    def test_datasheet_and_measured_need_a_source_url(self):
        for path in ("outline", "mounting_holes"):
            b = good()
            del b[path]["source_url"]
            self.assertError(b, f"{path}: confidence 'datasheet' requires a 'source_url'")
        for level in ("datasheet", "measured"):
            b = good()
            c = conn(b, "audio")
            c["confidence"] = level
            c["source_url"] = ""
            self.assertError(b, f"connector 'audio': confidence '{level}' requires a 'source_url'")
        b = good()
        c = conn(b, "audio")
        c["confidence"] = "community"
        del c["source_url"]
        self.assertEqual(check(b)[0], True)

    def test_connector_ids_must_be_unique_and_well_formed(self):
        b = good()
        conn(b, "hdmi1")["id"] = "hdmi0"
        self.assertError(b, "connector 'hdmi0': id is used twice")
        b = good()
        conn(b, "hdmi1")["id"] = "HDMI 1"
        self.assertError(b, "id must match [a-z0-9_]+")
        b = good()
        del conn(b, "hdmi1")["id"]
        self.assertError(b, "missing required key 'id'")

    def test_connector_required_keys(self):
        for key in catalog._REQUIRED_CONNECTOR_KEYS:
            b = good()
            del conn(b, "audio")[key]
            ok, errors, _w = check(b)
            self.assertFalse(ok, key)

    def test_connector_field_types(self):
        b = good()
        conn(b, "audio")["margin_hint"] = -1
        self.assertError(b, "connector 'audio': margin_hint -1")
        b = good()
        conn(b, "audio")["cutout"] = "yes"
        self.assertError(b, "connector 'audio': cutout must be true or false")
        b = good()
        conn(b, "audio")["note"] = 7
        self.assertError(b, "connector 'audio': note must be a string")
        b = good()
        conn(b, "audio")["kind"] = ""
        self.assertError(b, "connector 'audio': kind")

    def test_verified_true_with_weak_confidence_warns(self):
        b = good()
        b["verified"] = True
        self.assertWarning(b, "verified: true but", "pcb", "connector 'usb_c_power'")

    def test_verified_true_is_quiet_when_everything_is_datasheet_or_measured(self):
        b = good()
        b["verified"] = True
        b["pcb"]["confidence"] = "measured"
        b["pcb"]["source_url"] = "https://example.com/step"
        for c in b["connectors"]:
            c["confidence"] = "measured"
            c["source_url"] = "https://example.com/step"
        self.assertEqual(check(b), (True, [], []))


class OutlinePcbRuleTest(_Base):
    """Rule 2."""

    def test_length_must_be_at_least_width_and_positive(self):
        b = good()
        b["outline"]["length"], b["outline"]["width"] = 40.0, 56.0
        self.assertError(b, "outline: length 40.0 and width 56.0 must satisfy length >= width > 0")
        b = good()
        b["outline"]["width"] = 0
        self.assertError(b, "outline: length 85.0 and width 0.0 must satisfy")
        b = good()
        b["outline"]["length"] = -85.0
        self.assertError(b, "must satisfy length >= width > 0")

    def test_a_square_board_is_allowed(self):
        b = good()
        b["outline"]["length"] = b["outline"]["width"] = 56.0
        ok, errors, _w = check(b)
        self.assertFalse(any("must satisfy" in e for e in errors))

    def test_length_and_width_must_be_numbers(self):
        for bad in ("85", None, True, float("nan"), float("inf")):
            b = good()
            b["outline"]["length"] = bad
            self.assertError(b, "outline: length and width must be numbers")

    def test_corner_radius_range(self):
        b = good()
        b["outline"]["corner_radius"] = 28.5
        self.assertError(b, "outline: corner_radius 28.5 is outside 0..28.0")
        b = good()
        b["outline"]["corner_radius"] = -0.1
        self.assertError(b, "outline: corner_radius -0.1")
        b = good()
        b["outline"]["corner_radius"] = "3"
        self.assertError(b, "outline: corner_radius must be a number")
        for ok_radius in (0, 28.0):
            b = good()
            b["outline"]["corner_radius"] = ok_radius
            ok, errors, _w = check(b)
            self.assertFalse(any("corner_radius" in e for e in errors), ok_radius)

    def test_outline_shape(self):
        b = good()
        b["outline"]["shape"] = "ellipse"
        self.assertError(b, "outline: shape 'ellipse' is not supported")

    def test_pcb_thickness_bounds(self):
        for bad in (0.3, 3.3, 0, -1.6, "1.6", None):
            b = good()
            b["pcb"]["thickness"] = bad
            self.assertError(b, "pcb: thickness")
        for ok_value in (0.4, 1.6, 3.2):
            b = good()
            b["pcb"]["thickness"] = ok_value
            self.assertEqual(check(b)[0], True, ok_value)


class MountingHoleRuleTest(_Base):
    """Rules 3 and 9."""

    def test_at_least_two_holes(self):
        b = good()
        b["mounting_holes"]["pattern"] = None
        del b["mounting_holes"]["pattern"]
        b["mounting_holes"]["positions"] = [[3.5, 3.5]]
        self.assertError(b, "mounting_holes: 1 hole(s) listed; at least 2")
        b["mounting_holes"]["positions"] = []
        self.assertError(b, "at least 2")
        b["mounting_holes"]["positions"] = [[3.5, 3.5], [61.5, 3.5]]
        self.assertEqual(check(b)[0], True)

    def test_hole_must_be_inside_the_outline_by_the_pad_radius(self):
        b = good()
        del b["mounting_holes"]["pattern"]
        b["mounting_holes"]["positions"] = [[2.0, 3.5], [61.5, 3.5], [3.5, 52.5], [61.5, 52.5]]
        self.assertError(b, "hole 1 at (2.0, 3.5) is closer than the pad radius 3.1 mm to the board edge")
        b["mounting_holes"]["positions"] = [[3.1, 3.5], [61.5, 3.5], [3.5, 52.5], [61.5, 52.5]]     # exactly the pad radius: fine
        self.assertEqual(check(b)[0], True)
        b["mounting_holes"]["positions"] = [[3.5, 3.5], [61.5, 3.5], [3.5, 52.5], [83.0, 52.5]]
        self.assertError(b, "hole 4 at (83.0, 52.5)")
        b["mounting_holes"]["positions"] = [[3.5, 3.5], [61.5, 3.5], [3.5, 54.0], [61.5, 52.5]]
        self.assertError(b, "hole 3 at (3.5, 54.0)")

    def test_positions_must_equal_the_pattern_expansion(self):
        b = good()
        b["mounting_holes"]["positions"][3] = [61.5, 52.6]          # off by 0.1 mm
        self.assertError(b, "positions do not equal the pattern expansion")
        b = good()
        b["mounting_holes"]["positions"][3] = [61.505, 52.495]       # within 0.01 mm: accepted
        self.assertEqual(check(b)[0], True)
        b = good()
        b["mounting_holes"]["pattern"]["dx"] = 57.0
        self.assertError(b, "positions do not equal the pattern expansion")
        b = good()
        b["mounting_holes"]["positions"].append([30.0, 30.0])
        self.assertError(b, "positions do not equal the pattern expansion")

    def test_malformed_pattern_and_positions(self):
        b = good()
        b["mounting_holes"]["pattern"] = {"offset": [3.5], "dx": 58, "dy": 49}
        self.assertError(b, "pattern must be")
        b = good()
        b["mounting_holes"]["positions"] = [[3.5, "x"], [61.5, 3.5]]
        self.assertError(b, "positions must be a list of [x, y] numbers")
        b = good()
        b["mounting_holes"]["positions"] = "four"
        self.assertError(b, "positions must be a list")

    def test_a_board_without_a_pattern_is_fine(self):
        b = good()
        del b["mounting_holes"]["pattern"]
        self.assertEqual(check(b)[0], True)

    def test_duplicate_holes(self):
        b = good()
        del b["mounting_holes"]["pattern"]
        b["mounting_holes"]["positions"] = [[3.5, 3.5], [3.5, 3.5], [61.5, 52.5]]
        self.assertError(b, "holes 1 and 2 are at the same position (3.5, 3.5)")

    def test_diameter_pad_and_thread(self):
        b = good()
        b["mounting_holes"]["diameter"] = 0
        self.assertError(b, "mounting_holes: diameter 0")
        b = good()
        b["mounting_holes"]["pad_diameter"] = 2.0
        self.assertError(b, "pad_diameter 2.0 is smaller than the hole diameter 2.75")
        b = good()
        b["mounting_holes"]["pad_diameter"] = None
        self.assertError(b, "pad_diameter None must be a positive number")
        b = good()
        b["mounting_holes"]["thread"] = 2.5
        self.assertError(b, "thread must be a string")

    def test_overlapping_pads_warn(self):
        b = good()
        del b["mounting_holes"]["pattern"]
        b["mounting_holes"]["positions"] = [[30.0, 30.0], [34.0, 30.0], [3.5, 52.5], [61.5, 3.5]]     # 4.0 mm apart, pads 6.2 mm
        self.assertWarning(b, "the pads of holes 1 and 2 overlap")

    def test_a_pad_may_not_overlap_a_connector_in_plan_view(self):
        b = good()
        del b["mounting_holes"]["pattern"]
        b["mounting_holes"]["positions"] = [[3.5, 3.5], [61.5, 3.5], [3.5, 52.5], [75.0, 10.0]]    # inside the USB 2.0 stack
        errors = self.assertError(b, "connector 'usb2_pair': overlaps the pad of mounting hole 4 at (75.0, 10.0)")
        self.assertIn("pad radius 3.1 mm", "\n".join(errors))

    def test_a_pad_touching_a_connector_exactly_is_allowed(self):
        b = good()
        del b["mounting_holes"]["pattern"]
        # pad radius 3.1: centre 3.1 mm from the gpio box's lower edge (y 49.96) touches it without overlapping
        b["mounting_holes"]["positions"] = [[3.5, 3.5], [61.5, 3.5], [3.5, 52.5], [30.0, 49.96 - 3.1]]
        self.assertFalse(any("overlaps the pad" in e for e in check(b)[1]))


class ConnectorBoxRuleTest(_Base):
    """Rules 4, 5, 6, 7."""

    def test_box_axes_must_be_ordered(self):
        for axis in "xyz":
            b = good()
            c = conn(b, "audio")
            lo, hi = c["box"][axis]
            c["box"][axis] = [hi, lo]
            self.assertError(b, f"connector 'audio': box {axis}")
        b = good()
        conn(b, "audio")["box"]["z"] = [4.0, 4.0]
        self.assertError(b, "is empty; z0 must be below z1")

    def test_malformed_boxes(self):
        for bad in (None, "box", {"x": [1, 2]}, {"x": [1, 2], "y": [1, 2], "z": [1]}, {"x": [1, 2], "y": [1, 2], "z": ["a", "b"]},
                    {"x": [1, 2], "y": [1, 2], "z": [None, 3]}):
            b = good()
            conn(b, "audio")["box"] = bad
            self.assertError(b, "connector 'audio': box must be")

    def test_face_must_be_a_known_face(self):
        for bad in ("left", None, "", "+z"):
            b = good()
            conn(b, "audio")["face"] = bad
            self.assertError(b, "connector 'audio': face")

    def test_top_and_bottom_faces_are_valid(self):
        b = good()
        c = conn(b, "gpio")
        c["cutout"] = True       # a lid opening above the header
        self.assertEqual(check(b)[0], True)
        c["face"] = "bottom"
        self.assertEqual(check(b)[0], True)

    def test_wall_box_must_stay_inside_the_outline_along_the_wall(self):
        b = good()
        conn(b, "audio")["box"]["x"] = [80.0, 90.0]
        self.assertError(b, "connector 'audio': box x 80.0..90.0 leaves the outline 0..85.0 along the '-y' wall")
        b = good()
        conn(b, "ethernet")["box"]["y"] = [45.0, 58.0]
        self.assertError(b, "connector 'ethernet': box y 45.0..58.0 leaves the outline 0..56.0 along the '+x' wall")
        b = good()
        conn(b, "audio")["box"]["x"] = [-1.0, 6.0]
        self.assertError(b, "box x -1.0..6.0 leaves the outline")

    def test_wall_box_must_reach_into_the_board(self):
        b = good()
        conn(b, "audio")["box"]["y"] = [-1.0, 1.5]
        self.assertError(b, "connector 'audio': box reaches only 1.5 mm into the board from face '-y'")
        b = good()
        conn(b, "audio")["box"]["y"] = [-1.0, 2.0]       # exactly 2 mm: fine
        self.assertEqual(check(b)[0], True)
        b = good()
        conn(b, "ethernet")["box"]["x"] = [84.0, 87.0]
        self.assertError(b, "connector 'ethernet': box reaches only 1.0 mm into the board from face '+x'")

    def test_protrusion_over_six_mm_is_a_wrong_sign(self):
        b = good()
        conn(b, "audio")["box"]["y"] = [-9.0, 12.0]
        errors = self.assertError(b, "connector 'audio': protrusion 9.0 mm on face '-y'; probably a wrong sign")
        self.assertEqual(len([e for e in errors if "protrusion" in e]), 1)
        b = good()
        conn(b, "audio")["box"]["y"] = [-6.0, 12.0]       # exactly 6 mm: fine
        self.assertEqual(check(b)[0], True)
        b = good()
        conn(b, "ethernet")["box"]["x"] = [65.8, 92.5]
        self.assertError(b, "connector 'ethernet': protrusion 7.5 mm on face '+x'")

    def test_top_and_bottom_boxes_must_lie_inside_the_outline(self):
        b = good()
        conn(b, "gpio")["box"]["x"] = [7.1, 90.0]
        self.assertError(b, "connector 'gpio': a 'top' box must lie inside the outline")
        b = good()
        conn(b, "gpio")["box"]["y"] = [49.96, 57.0]
        self.assertError(b, "connector 'gpio': a 'top' box must lie inside the outline")

    def test_wrong_face_for_the_box_position_warns(self):
        b = good()
        conn(b, "audio")["face"] = "+y"           # the audio jack sits at the -y edge
        ok, errors, warnings = check(b)
        self.assertTrue(any("nearer the opposite edge" in w for w in warnings) or not ok)

    def test_protruding_connector_without_a_cutout_warns(self):
        b = good()
        conn(b, "usb_c_power")["box"]["y"] = [-2.0, 6.9]
        conn(b, "usb_c_power")["cutout"] = False
        self.assertWarning(b, "connector 'usb_c_power': protrudes 2.0 mm past the '-y' edge but has no cutout")

    def test_overlapping_boxes_are_an_error_naming_both(self):
        b = good()
        conn(b, "hdmi1")["box"]["x"] = [28.0, 34.7]          # overlaps hdmi0 (22.65..29.35) in x, y and z
        self.assertError(b, "connectors 'hdmi0' and 'hdmi1' overlap")
        b = good()
        conn(b, "hdmi1")["box"]["x"] = [29.35, 36.05]         # touching faces are not an overlap
        self.assertFalse(any("overlap:" in e for e in check(b)[1]))

    def test_overlap_in_two_axes_only_is_fine(self):
        b = good()
        c = conn(b, "hdmi1")
        c["box"] = {"x": [22.65, 29.35], "y": [-0.5, 5.5], "z": [4.5, 6.0]}      # stacked above hdmi0: shares x and y, not z
        self.assertFalse(any("overlap:" in e for e in check(b)[1]))

    def test_same_wall_openings_closer_than_point_four_warn(self):
        b = good()
        conn(b, "hdmi1")["box"]["x"] = [29.55, 36.25]          # 0.2 mm after hdmi0
        self.assertWarning(b, "connectors 'hdmi0' and 'hdmi1' are on the same '-y' wall and closer than 0.4 mm")
        b = good()
        conn(b, "hdmi1")["box"]["x"] = [29.75, 36.45]          # exactly 0.4 mm: quiet
        self.assertEqual(check(b)[2], [])

    def test_a_connector_appears_once_with_exactly_one_face(self):
        b = good()
        twin = copy.deepcopy(dict(conn(b, "audio")))
        twin["id"] = "audio_top"
        twin["face"] = "top"
        twin["box"] = {"x": [50.0, 57.0], "y": [20.0, 25.0], "z": [1.4, 7.4]}
        b["connectors"].append(twin)
        self.assertError(b, "connector 'audio_top': name '3.5 mm audio / composite jack' is already used by 'audio' (face '-y')",
                         "exactly one face")

    def test_connectors_must_be_a_non_empty_list_of_objects(self):
        b = good()
        b["connectors"] = []
        self.assertError(b, "connectors: must be a non-empty list")
        b = good()
        b["connectors"].append("usb")
        self.assertError(b, "connectors[9]: must be an object")


class HeightRuleTest(_Base):
    """Rule 8."""

    def test_max_top_z_must_match_the_boxes(self):
        b = good()
        b["height"]["max_top_z"] = 17.5
        self.assertError(b, "height.max_top_z 17.5 != 17.0 (max of connector tops)")

    def test_min_bottom_z_must_match_the_boxes(self):
        b = good()
        b["height"]["min_bottom_z"] = 0.0
        self.assertError(b, "height.min_bottom_z 0.0 != -1.5 (min of connector bottoms)")

    def test_tolerance_is_one_hundredth_of_a_millimetre(self):
        b = good()
        b["height"]["max_top_z"] = 17.009
        self.assertEqual(check(b)[0], True)
        b["height"]["max_top_z"] = 17.02
        self.assertFalse(check(b)[0])

    def test_height_follows_a_changed_box(self):
        b = good()
        conn(b, "usb3_pair")["box"]["z"] = [1.4, 18.0]
        conn(b, "usb2_pair")["box"]["z"] = [1.4, 18.0]
        self.assertError(b, "height.max_top_z 17.0 != 18.0")
        b["height"]["max_top_z"] = 18.0
        self.assertEqual(check(b)[0], True)

    def test_malformed_height(self):
        b = good()
        b["height"] = {"max_top_z": "tall", "min_bottom_z": 0}
        self.assertError(b, "height: max_top_z and min_bottom_z must be numbers")
        b = good()
        b["height"] = [17.0, -1.5]
        self.assertError(b, "height: must be")


class PlugKeysTest(_Base):
    """The optional plug_w / plug_h are accepted and validated."""

    def test_valid_plug_sizes_pass(self):
        b = good()
        conn(b, "usb_c_power")["plug_w"] = 13.5
        conn(b, "usb_c_power")["plug_h"] = 7.0
        self.assertEqual(check(b), (True, [], []))

    def test_invalid_plug_sizes_are_rejected(self):
        for bad in (0, -3.0, "wide", None, True):
            b = good()
            conn(b, "usb_c_power")["plug_w"] = bad
            self.assertError(b, "connector 'usb_c_power': plug_w")
            b = good()
            conn(b, "usb_c_power")["plug_h"] = bad
            self.assertError(b, "connector 'usb_c_power': plug_h")


class FakeFileDataTest(unittest.TestCase):
    """Catalog and board files read from a temporary data directory (corrupt, mismatched)."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="refparts_data_")
        os.makedirs(os.path.join(self.dir, "boards"))
        self.patches = [mock.patch.object(catalog, "DATA_DIR", self.dir),
                        mock.patch.object(catalog, "CATALOG_PATH", os.path.join(self.dir, "catalog.json"))]
        for p in self.patches:
            p.start()
        catalog._reset_cache()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for p in self.patches:
            p.stop()
        catalog._reset_cache()

    def write(self, rel, data):
        path = os.path.join(self.dir, rel)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(data if isinstance(data, str) else json.dumps(data))

    def small_catalog(self, **entry):
        base = {"id": "demo", "name": "Demo", "aliases": ["demo board"], "kind": "sbc", "board": "boards/demo.json",
                "vendor_urls": [], "datasheet_urls": [], "step_urls": [], "license": "test", "last_verified": "", "verified": False}
        base.update(entry)
        return {"schema": 1, "allow_hosts": ["example.com"], "entries": [base]}

    def test_unreadable_catalog_raises_value_error(self):
        self.write("catalog.json", "{ not json")
        with self.assertRaises(ValueError) as cm:
            catalog.list_entries()
        self.assertIn("cannot be read", str(cm.exception))

    def test_missing_catalog_raises_value_error(self):
        with self.assertRaises(ValueError):
            catalog.list_entries()

    def test_catalog_that_is_not_an_object_raises(self):
        self.write("catalog.json", "[1, 2]")
        with self.assertRaises(ValueError) as cm:
            catalog.allow_hosts()
        self.assertIn("not a JSON object", str(cm.exception))

    def test_board_file_with_another_id_is_refused(self):
        self.write("catalog.json", self.small_catalog())
        self.write("boards/demo.json", {"id": "other"})
        with self.assertRaises(ValueError) as cm:
            catalog.load_board("demo")
        self.assertIn("has id 'other', expected 'demo'", str(cm.exception))

    def test_missing_board_file_raises_value_error(self):
        self.write("catalog.json", self.small_catalog())
        with self.assertRaises(ValueError):
            catalog.load_board("demo")

    def test_entry_without_a_board_is_not_a_board(self):
        self.write("catalog.json", self.small_catalog(board=""))
        self.assertEqual(catalog.list_boards(), [])
        with self.assertRaises(KeyError):
            catalog.load_board("demo")
        self.assertEqual(catalog.get("demo")["name"], "Demo")

    def test_connectors_of_a_loaded_board_resolve_without_a_second_argument(self):
        board = good_dict()
        board["id"] = "demo"
        self.write("catalog.json", self.small_catalog())
        self.write("boards/demo.json", board)
        loaded = catalog.load_board("demo")
        self.assertEqual(catalog.opening(next(c for c in loaded["connectors"] if c["id"] == "ethernet"))["protrusion"], 2.0)


def good_dict() -> dict:
    return json.loads(_PI4_JSON)


class CatalogValidatorTest(unittest.TestCase):
    def cat(self):
        return json.loads(json.dumps({"schema": 1, "allow_hosts": ["raw.githubusercontent.com", "datasheets.raspberrypi.com",
                                                                      "pip.raspberrypi.com", "pip-assets.raspberrypi.com"],
                                      "entries": catalog.list_entries()}))

    def errors(self, cat):
        ok, errors, _w = catalog.validate_catalog(cat)
        self.assertFalse(ok)
        return "\n".join(errors)

    def test_the_shipped_catalog_is_clean_and_a_copy_validates_too(self):
        self.assertEqual(catalog.validate_catalog(self.cat())[:2], (True, []))

    def test_garbage_never_raises(self):
        for garbage in (None, [], "x", 3, {}, {"entries": "x"}, {"schema": 1, "entries": [3], "allow_hosts": []}):
            ok, errors, _w = catalog.validate_catalog(garbage) if garbage is not None else catalog.validate_catalog({})
            self.assertFalse(ok)

    def test_schema_and_hosts(self):
        c = self.cat()
        c["schema"] = 3
        self.assertIn("schema 3 is not supported", self.errors(c))
        c = self.cat()
        c["allow_hosts"] = ["Raw.GitHub.com", "https://x.com/"]
        self.assertIn("allow_hosts: must be a list of lower-case host names", self.errors(c))

    def test_ids_and_aliases(self):
        c = self.cat()
        c["entries"][1]["id"] = "rpi4b"
        self.assertIn("entry 'rpi4b': id is used twice", self.errors(c))
        c = self.cat()
        c["entries"][1]["aliases"].append("Pi 4")
        self.assertIn("alias 'Pi 4' is also an alias of 'rpi4b'", self.errors(c))
        c = self.cat()
        c["entries"][0]["id"] = "Pi-4"
        self.assertIn("id must match", self.errors(c))
        c = self.cat()
        c["entries"][0]["aliases"] = []
        self.assertIn("aliases must be a non-empty list", self.errors(c))
        c = self.cat()
        c["entries"][0]["kind"] = "toaster"
        self.assertIn("kind 'toaster'", self.errors(c))

    def test_board_paths(self):
        c = self.cat()
        c["entries"][0]["board"] = "../secrets.json"
        self.assertIn("must be a path like 'boards/<id>.json'", self.errors(c))
        c = self.cat()
        c["entries"][0]["board"] = "boards/rpi5.json"
        self.assertIn("must be named after the entry id", self.errors(c))
        c = self.cat()
        c["entries"][0]["id"] = "ghost"
        c["entries"][0]["board"] = "boards/ghost.json"
        self.assertIn("does not exist", self.errors(c))

    def test_urls_must_be_https(self):
        c = self.cat()
        c["entries"][0]["vendor_urls"] = ["http://www.raspberrypi.com/x"]
        self.assertIn("vendor_urls must be a list of https URLs", self.errors(c))
        c = self.cat()
        c["entries"][0]["datasheet_urls"] = "https://x.com"
        self.assertIn("datasheet_urls", self.errors(c))
        c = self.cat()
        c["entries"][0]["step_urls"] = "x"
        self.assertIn("step_urls must be a list", self.errors(c))

    def step(self, **kw):
        base = {"url": "https://raw.githubusercontent.com/o/r/main/pi4.step", "mirror": False, "expected_sha256": "",
                "expected_bbox": [85.0, 56.0, 17.0], "license": "CC-BY-4.0"}
        base.update(kw)
        return base

    def test_a_good_step_url_validates_with_a_warning_about_the_missing_hash(self):
        c = self.cat()
        c["entries"][0]["step_urls"] = [self.step()]
        ok, errors, warnings = catalog.validate_catalog(c)
        self.assertTrue(ok, errors)
        self.assertTrue(any("no expected_sha256 pinned" in w for w in warnings))
        c["entries"][0]["step_urls"] = [self.step(expected_sha256="a" * 64)]
        self.assertEqual(catalog.validate_catalog(c)[2], [])

    def test_step_url_rules(self):
        cases = [
            (self.step(url="https://evil.example.com/pi4.step"), "host 'evil.example.com' is not in allow_hosts"),
            (self.step(url="http://raw.githubusercontent.com/o/r/pi4.step"), "must be an https URL"),
            (self.step(url="https://raw.githubusercontent.com@evil.example.com/x"), "must be an https URL"),
            (self.step(mirror="no"), "mirror must be true or false"),
            (self.step(expected_sha256="XYZ"), "expected_sha256 must be"),
            (self.step(expected_sha256="A" * 64), "expected_sha256 must be"),
            (self.step(expected_bbox=[85.0, 56.0]), "expected_bbox must be"),
            (self.step(expected_bbox=[85.0, 0, 17.0]), "expected_bbox must be"),
            (self.step(license=""), "license must be a non-empty string"),
            ("https://raw.githubusercontent.com/x", "must be an object"),
        ]
        for step, fragment in cases:
            c = self.cat()
            c["entries"][0]["step_urls"] = [step]
            self.assertIn(fragment, self.errors(c))

    def test_licence_and_dates_and_flags(self):
        c = self.cat()
        c["entries"][0]["license"] = ""
        self.assertIn("license must be a non-empty string", self.errors(c))
        c = self.cat()
        c["entries"][0]["last_verified"] = "yesterday"
        self.assertIn("last_verified must be", self.errors(c))
        c = self.cat()
        c["entries"][0]["verified"] = "true"
        self.assertIn("verified must be true or false", self.errors(c))
        c = self.cat()
        c["entries"][0]["verified"] = True
        c["entries"][0]["last_verified"] = ""      # migration: the shipped Pi 4 entry now carries a verification date
        ok, _e, warnings = catalog.validate_catalog(c)
        self.assertTrue(any("verified is true but last_verified is empty" in w for w in warnings))


class HelperTest(unittest.TestCase):
    def test_number_helper_rejects_bool_nan_and_text(self):
        self.assertEqual(catalog._num(3), 3.0)
        for bad in (True, "3", None, float("nan"), float("inf"), [], {}):
            self.assertIsNone(catalog._num(bad))
        self.assertTrue(math.isfinite(catalog._num(-0.5)))

    def test_constants(self):
        self.assertEqual(catalog.FACES, ("-x", "+x", "-y", "+y", "top", "bottom"))
        self.assertEqual(catalog.CONFIDENCE_LEVELS, ("measured", "datasheet", "community", "approximate", "assumed"))


if __name__ == "__main__":
    unittest.main()
