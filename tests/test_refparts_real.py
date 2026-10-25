"""OPT-IN: the two REAL registered STEP files (never committed, skipped unless they exist).

Set ``OSCAR_REAL_STEP_DIR`` to a folder that holds the two files downloaded on 2026-10-02 for verification::

    rp5.zip    https://pip.raspberrypi.com/documents/RP-010083-CA        (official Raspberry Pi 5, redirects to pip-assets.raspberrypi.com)
    pi4.step   the community Raspberry Pi 4 Model B v4.step               (multigamesystem/MGS-CAD-Files, licence unspecified)

then run this module under ``freecadcmd`` (OSCAR_TEST_MODULES=test_refparts_real). It checks the catalog pins against the real bytes (size,
sha256, ISO-10303-21 magic, zip integrity and member), runs them through the real ``fetch`` / ``netguard.download_verified`` / ZIP
extraction (a fake ``requests.get`` serves the local bytes: no network), imports them with the app's importer, measures them and
cross-checks the result with ``boards/rpi4b.json`` and ``boards/rpi5.json``. The Pi 5 import takes about 200 s (2689 solids).
Nothing here runs by default: without the folder every test is skipped.
"""

import hashlib
import os
import sys
import unittest
import zipfile
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402
import requests  # noqa: E402

from oscar.refparts import catalog, fetch, importref, measure  # noqa: E402

REAL_DIR = os.environ.get("OSCAR_REAL_STEP_DIR", "")
ZIP_PATH = os.path.join(REAL_DIR, "rp5.zip")
PI4_PATH = os.path.join(REAL_DIR, "pi4.step")
HAVE_FILES = bool(REAL_DIR) and os.path.isfile(ZIP_PATH) and os.path.isfile(PI4_PATH)
SKIP = "set OSCAR_REAL_STEP_DIR to a folder with rp5.zip and pi4.step (the real files, never committed) to run this"


def sha_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StreamResponse:
    """A response that streams a local file in 1 MB chunks (the 13 MB / 17 MB real files never sit in memory twice)."""

    def __init__(self, path="", *, status=200, url="", headers=None):
        self.status_code, self.url, self._path = status, url, path
        self.headers = {"Content-Length": str(os.path.getsize(path) if path else 0), **(headers or {})}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)

    def iter_content(self, chunk_size=1):
        if self._path:
            with open(self._path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    yield chunk


@unittest.skipUnless(HAVE_FILES, SKIP)
class RealFilesTest(unittest.TestCase):
    def setUp(self):
        fetch.forget_all()
        self.dest = isolation.fresh_dir("real_dl_")
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(fetch.ENV_LOCAL_DIR, None)

    def serve(self, routes):
        def get(url, **kwargs):
            return routes[url]()
        return mock.patch.object(requests, "get", get)

    def test_pins_match_the_real_bytes(self):
        pi5, pi4 = catalog.get("rpi5")["step_urls"][0], catalog.get("rpi4b")["step_urls"][0]
        self.assertEqual((os.path.getsize(ZIP_PATH), sha_of(ZIP_PATH)), (pi5["size"], pi5["expected_sha256"]))
        self.assertEqual((os.path.getsize(PI4_PATH), sha_of(PI4_PATH)), (pi4["size"], pi4["expected_sha256"]))
        with zipfile.ZipFile(ZIP_PATH) as zf:
            self.assertIsNone(zf.testzip())
            self.assertEqual(sorted(i.filename for i in zf.infolist()), sorted([pi5["member"], "LICENSE.txt"]))
            self.assertEqual(zf.getinfo(pi5["member"]).file_size, pi5["member_bytes"])
            self.assertIn(b"MIT License", zf.read("LICENSE.txt"))
            self.assertEqual(zf.read(pi5["member"])[:13], b"ISO-10303-21;")
        with open(PI4_PATH, "rb") as fh:
            self.assertEqual(fh.read(13), b"ISO-10303-21;")

    def test_the_official_archive_goes_through_netguard_and_extracts_its_member(self):
        step = catalog.get("rpi5")["step_urls"][0]
        final = "https://pip-assets.raspberrypi.com/categories/892-raspberry-pi-5/documents/RP-010083-CA-1-rpi-5%203D%20STEP%20-%20No%20Graphics%20small%20file.zip"
        routes = {step["url"]: lambda: StreamResponse(status=302, url=step["url"], headers={"Location": final}),
                  final: lambda: StreamResponse(ZIP_PATH, url=final)}
        with self.serve(routes):
            result = fetch.fetch(catalog.get("rpi5"), confirm=lambda s, d: True, dest_dir=self.dest, reason="real-file check")
        self.assertEqual(result.status, "obtained", result.note)
        self.assertEqual((result.pin, result.trust, result.warnings), ("match", "official", []))
        self.assertEqual((result.sha256, result.bytes), (step["member_sha256"], step["member_bytes"]))
        self.assertEqual(os.path.basename(result.path), step["member"])
        self.assertEqual(result.archive["sha256"], step["expected_sha256"])

    def test_the_community_file_downloads_and_matches_its_pin(self):
        step = catalog.get("rpi4b")["step_urls"][0]
        with self.serve({step["url"]: lambda: StreamResponse(PI4_PATH, url=step["url"])}):
            result = fetch.fetch(catalog.get("rpi4b"), confirm=lambda s, d: True, dest_dir=self.dest, reason="real-file check")
        self.assertEqual((result.status, result.pin, result.trust, result.sha256), ("obtained", "match", "community", step["expected_sha256"]), result.note)

    def import_and_measure(self, path, ident):
        board = catalog.load_board(ident)
        step = catalog.get(ident)["step_urls"][0]
        doc = App.newDocument(f"Real{ident}")
        self.addCleanup(lambda: App.closeDocument(doc.Name))
        summary = importref.import_reference(doc, path, ref_id=ident, board=board, expected_bbox=step["expected_bbox"], sha256=sha_of(path))
        m = measure.measure_object(doc.getObject(summary["object"]), board=board, expected_bbox=step["expected_bbox"], top_n=30)
        return summary, m

    def centre(self, conn_box):
        return sum(conn_box["y"]) / 2.0

    def test_pi4_community_model_agrees_with_the_drawing_data(self):
        summary, m = self.import_and_measure(PI4_PATH, "rpi4b")
        board = catalog.load_board("rpi4b")
        self.assertEqual(summary["identity"]["status"], "match")
        self.assertEqual(summary["solids"], 109)
        self.assertEqual([round(v, 2) for v in m.bbox["size"]], [88.99, 58.4, 19.9])
        self.assertEqual((round(m.pcb["length"], 2), round(m.pcb["width"], 2), round(m.pcb["thickness"], 2), round(m.pcb["corner_radius"], 2)), (85.0, 56.0, 1.6, 3.0))
        self.assertEqual((m.mounting["pattern"]["dx"], m.mounting["pattern"]["dy"], m.mounting["pattern"]["offset"]), (58.0, 49.0, [3.5, 3.5]))
        self.assertEqual(m.mounting["d"], 6.0)
        by_id = {c["matches"]: c for c in m.connectors if c.get("matches")}
        for conn in board["connectors"]:
            if conn["id"] in ("usb_c_power", "hdmi0", "hdmi1", "audio", "ethernet", "usb3_pair", "usb2_pair"):
                cand = by_id[conn["id"]]
                axis = conn["box"]["x"] if conn["face"] in ("-y", "+y") else conn["box"]["y"]
                limit = 0.55 if conn["id"] == "audio" else 0.3          # the audio jack is 0.5 mm off (recorded in step_cross_check)
                self.assertLess(abs(cand["offset"] - sum(axis) / 2.0), limit, f"{conn['id']}: centre differs from the drawing-based box by more than {limit} mm")
        board_after = catalog.load_board("rpi4b")
        self.assertEqual((board_after["pcb"]["thickness"], board_after["pcb"]["confidence"]), (1.4, "assumed"), "the community model never overwrites the drawing")

    def test_pi5_official_model_agrees_with_the_measured_board_data(self):
        member = os.path.join(self.dest, "rpi-5b_no_graphics.step")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extract("rpi-5b_no_graphics.step", self.dest)
        summary, m = self.import_and_measure(member, "rpi5")
        board = catalog.load_board("rpi5")
        self.assertEqual(summary["identity"]["status"], "match")
        self.assertEqual(summary["solids"], 2689)
        self.assertEqual([round(v, 2) for v in m.bbox["size"]], [88.5, 57.2, 18.98])
        self.assertEqual((round(m.pcb["length"], 3), round(m.pcb["width"], 3), round(m.pcb["thickness"], 2)), (85.0, 56.0, board["pcb"]["thickness"]))
        self.assertEqual((m.mounting["pattern"]["dx"], m.mounting["pattern"]["dy"], m.mounting["pattern"]["offset"]), (58.0, 49.0, [3.5, 3.5]))
        self.assertEqual(m.mounting["d"], board["mounting_holes"]["diameter"])
        wall = [c for c in m.connectors if c["side"] == "+x"]
        offsets = sorted({round(c["offset"], 2) for c in wall if c["width"] > 10})
        measured = {c["id"]: c for c in board["connectors"] if c["confidence"] == "measured" and c["face"] == "+x"}
        self.assertEqual(offsets, sorted(round(self.centre(c["box"]), 2) for c in measured.values()))
        usb_c = next(c for c in m.connectors if c["side"] == "-y" and c["width"] > 8)
        box = measured_box = board["connectors"][0]["box"]
        self.assertLess(abs(usb_c["offset"] - sum(box["x"]) / 2.0), 0.05)
        self.assertEqual(measured_box, {"x": [6.73, 15.67], "y": [-1.2, 6.12], "z": [0.31, 4.57]})
        self.assertFalse([c for c in m.connectors if c["side"] == "-y" and 20 < c["offset"] < 45],
                         "the official 'No Graphics small file' model has no micro-HDMI shell: hdmi0 / hdmi1 must stay drawing-based")


if __name__ == "__main__":
    unittest.main()
