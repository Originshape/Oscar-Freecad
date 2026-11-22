"""Tests of ``oscar.refparts.fetch``: catalog -> ONE allowlisted download -> verification -> import-ready.

No FreeCAD document is touched. The network is a fake ``requests.get`` (through the real ``netguard.download_verified``) or a stub
downloader; the catalog is patched to list a STEP URL for the Pi 4 (the shipped catalog lists none: no URL is invented). Covers the
four statuses, the offline mirror, the allowlist, the pinned hash, ZIP archives, the one-attempt-per-turn rule and the registries.
"""

import copy
import hashlib
import io
import json
import os
import sys
import unittest
import zipfile
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import requests  # noqa: E402

from oscar import netguard  # noqa: E402
from oscar.refparts import catalog, fetch  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "refparts")
STEP = open(os.path.join(FIXTURES, "pi4_like.step"), "rb").read()
STEP_SHA = hashlib.sha256(STEP).hexdigest()
STL_ASCII = b"solid cube\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid cube\n"
HTML = b"<!DOCTYPE html><html><body>Sign in to download</body></html>"
URL = "https://raw.githubusercontent.com/example/pi/main/pi4.step"


def real_catalog() -> dict:
    return copy.deepcopy(catalog._catalog())


class FakeResponse:
    def __init__(self, body=b"", *, headers=None, url="", status=200):
        self.status_code, self.headers, self.url, self._body = status, dict(headers or {}), url, body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)

    def iter_content(self, chunk_size=1):
        if self._body:
            yield self._body


class FakeGet:
    def __init__(self, response=None, raises=None):
        self.response, self.raises, self.calls = response, raises, []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.raises is not None:
            raise self.raises
        return self.response(url) if callable(self.response) else self.response


class Base(unittest.TestCase):
    """A patched catalog (the Pi 4 lists ``URL``), clean registries and a fresh download directory for every test."""

    pinned = ""
    step_urls = None

    def setUp(self):
        fetch.forget_all()
        cat = real_catalog()
        rpi4 = next(e for e in cat["entries"] if e["id"] == "rpi4b")
        rpi4["step_urls"] = self.step_urls if self.step_urls is not None else [
            {"url": URL, "mirror": False, "expected_sha256": self.pinned, "expected_bbox": [89.5, 57.0, 18.5], "license": "CC-BY test"}]
        patcher = mock.patch.object(catalog, "_catalog", return_value=cat)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(fetch.ENV_LOCAL_DIR, None)
        self.dest = isolation.fresh_dir("fetch_dl_")
        self.entry = catalog.get("rpi4b")
        self.asked = []

    def confirm(self, answer=True):
        def ask(summary, detail):
            self.asked.append((summary, detail))
            return answer
        return ask

    def run_fetch(self, body=STEP, *, answer=True, entry="default", **kwargs):
        net = FakeGet(FakeResponse(body, headers={"Content-Length": str(len(body))}, url=kwargs.pop("final_url", URL)))
        with mock.patch.object(requests, "get", net):
            result = fetch.fetch(self.entry if entry == "default" else entry, confirm=self.confirm(answer), dest_dir=self.dest,
                                 reason="reference geometry", **kwargs)
        self.net = net
        return result


class SlugAndNamesTest(unittest.TestCase):
    def test_slug_id(self):
        self.assertEqual(fetch.slug_id("Pi 4 (v2).step"), "pi_4_v2")
        self.assertEqual(fetch.slug_id("/tmp/My Board.STEP"), "my_board")
        self.assertEqual(fetch.slug_id("raspberry pi 4"), "raspberry_pi_4")
        self.assertEqual(fetch.slug_id(""), "part")
        self.assertEqual(fetch.slug_id("???", "x"), "x")
        self.assertLessEqual(len(fetch.slug_id("a" * 100)), 32)

    def test_ref_object_name(self):
        self.assertEqual(fetch.ref_object_name("rpi4b"), "Ref_RPI4B")
        self.assertEqual(fetch.ref_object_name("pi-4 v2"), "Ref_PI_4_V2")
        self.assertEqual(fetch.ref_object_name(""), "Ref_PART")


class ResolveEntryTest(Base):
    def test_explicit_id_wins_and_is_case_insensitive(self):
        entry, others = fetch.resolve_entry("RPI4B", "an arduino", "")
        self.assertEqual((entry["id"], others), ("rpi4b", []))

    def test_unknown_id_names_the_known_ones(self):
        with self.assertRaises(KeyError) as cm:
            fetch.resolve_entry("rpi99")
        self.assertIn("rpi4b", cm.exception.args[0])

    def test_hint_then_request(self):
        self.assertEqual(fetch.resolve_entry("", "a Raspberry Pi 4 Model B", "")[0]["id"], "rpi4b")
        self.assertEqual(fetch.resolve_entry("", "", "design a case for the arduino uno")[0]["id"], "arduino_uno")
        self.assertEqual(fetch.resolve_entry("", "gibberish", "Pi 4 case")[0]["id"], "rpi4b")

    def test_nothing_named(self):
        self.assertEqual(fetch.resolve_entry("", "", "a hammer"), (None, []))
        self.assertEqual(fetch.resolve_entry(), (None, []))

    def test_candidates_are_the_other_matches(self):
        entry, others = fetch.resolve_entry("", "raspberry pi", "")
        self.assertIsNotNone(entry)
        self.assertIsInstance(others, list)


class EntrySummaryTest(Base):
    def test_pi4_summary_is_honest(self):
        info = fetch.entry_summary(self.entry)
        self.assertEqual((info["id"], info["board"], info["connectors"]), ("rpi4b", True, 9))
        self.assertFalse(info["verified"])
        self.assertEqual(info["weakest"], "assumed")
        self.assertEqual(info["confidence"]["community"], 8)
        self.assertEqual(info["step_urls"], 1)

    def test_entry_without_a_board(self):
        info = fetch.entry_summary({"id": "x", "name": "X", "kind": "mcu", "step_urls": [], "datasheet_urls": []})
        self.assertEqual((info["board"], info["connectors"], info["weakest"]), (False, 0, ""))

    def test_unreadable_board_is_reported_as_no_board(self):
        info = fetch.entry_summary({"id": "ghost", "name": "G", "board": "boards/ghost.json"})
        self.assertFalse(info["board"])


class MirrorTest(Base):
    def test_no_env_means_no_mirror(self):
        self.assertEqual((fetch.local_mirror_dir(), fetch.find_local_mirror("rpi4b")), ("", ""))

    def test_finds_step_then_other_extensions(self):
        folder = isolation.fresh_dir("mirror_")
        os.environ[fetch.ENV_LOCAL_DIR] = folder
        self.assertEqual(fetch.find_local_mirror("rpi4b"), "")
        for name in ("rpi4b.stl", "rpi4b.stp"):
            open(os.path.join(folder, name), "wb").write(STL_ASCII)
        self.assertTrue(fetch.find_local_mirror("rpi4b").endswith("rpi4b.stp"))
        open(os.path.join(folder, "rpi4b.step"), "wb").write(STEP)
        self.assertTrue(fetch.find_local_mirror("RPI4B").endswith("rpi4b.step"))

    def test_ids_with_path_characters_never_escape_the_folder(self):
        folder = isolation.fresh_dir("mirror_")
        os.environ[fetch.ENV_LOCAL_DIR] = folder
        outside = os.path.join(os.path.dirname(folder), "evil_mirror.step")
        open(outside, "wb").write(STEP)
        self.addCleanup(os.remove, outside)
        for bad in ("../evil_mirror", "a/b", "", "..", "a b"):
            self.assertEqual(fetch.find_local_mirror(bad), "", bad)

    def test_env_is_read_at_call_time_and_expands_user(self):
        os.environ[fetch.ENV_LOCAL_DIR] = "~/somewhere"
        self.assertEqual(fetch.local_mirror_dir(), os.path.expanduser("~/somewhere"))
        os.environ[fetch.ENV_LOCAL_DIR] = "  "
        self.assertEqual(fetch.local_mirror_dir(), "")

    def test_mirror_file_is_verified_like_a_download(self):
        folder = isolation.fresh_dir("mirror_")
        os.environ[fetch.ENV_LOCAL_DIR] = folder
        open(os.path.join(folder, "rpi4b.step"), "wb").write(STEP)
        result = fetch.fetch(self.entry, confirm=self.confirm(False), dest_dir=self.dest)
        self.assertEqual((result.status, result.origin, result.detected_type), ("obtained", "local_mirror", "step"))
        self.assertEqual((result.sha256, result.bytes), (STEP_SHA, len(STEP)))
        self.assertEqual((self.asked, result.url, result.license), ([], "", ""), "no dialog, no network, no download licence for a local file")
        self.assertEqual(fetch.obtained_file("rpi4b")["path"], result.path)

    def test_a_bad_mirror_file_fails_and_never_falls_back_to_the_network(self):
        folder = isolation.fresh_dir("mirror_")
        os.environ[fetch.ENV_LOCAL_DIR] = folder
        open(os.path.join(folder, "rpi4b.step"), "wb").write(HTML)
        net = FakeGet(FakeResponse(STEP))
        with mock.patch.object(requests, "get", net):
            result = fetch.fetch(self.entry, confirm=self.confirm(True), dest_dir=self.dest)
        self.assertEqual(result.status, "failed")
        self.assertIn("HTML", result.note)
        self.assertIn("board data is used instead", result.note)
        self.assertEqual(net.calls, [])
        self.assertEqual(fetch.obtained_file("rpi4b"), {})

    def test_mirror_beats_an_explicit_url_and_allow_download_false(self):
        folder = isolation.fresh_dir("mirror_")
        os.environ[fetch.ENV_LOCAL_DIR] = folder
        open(os.path.join(folder, "rpi4b.step"), "wb").write(STEP)
        result = fetch.fetch(self.entry, url=URL, allow_download=False, confirm=None, dest_dir=self.dest)
        self.assertEqual((result.status, result.origin), ("obtained", "local_mirror"))

    def test_catalog_pinned_hash_applies_to_the_mirror_file(self):
        folder = isolation.fresh_dir("mirror_")
        os.environ[fetch.ENV_LOCAL_DIR] = folder
        open(os.path.join(folder, "rpi4b.step"), "wb").write(STEP)
        self.entry["step_urls"][0]["expected_sha256"] = "0" * 64
        result = fetch.fetch(self.entry, dest_dir=self.dest)
        self.assertEqual(result.status, "failed")
        self.assertIn("hash mismatch", result.note)


class VerifyFileTest(unittest.TestCase):
    def write(self, data: bytes, name="x.step") -> str:
        path = os.path.join(isolation.fresh_dir("verify_"), name)
        open(path, "wb").write(data)
        return path

    def test_good_step(self):
        facts, error = fetch.verify_file(self.write(STEP))
        self.assertEqual((error, facts["detected_type"], facts["sha256"], facts["bytes"]), ("", "step", STEP_SHA, len(STEP)))

    def test_html_lfs_and_empty_are_rejected_with_a_hint(self):
        for data, word in ((HTML, "web page"), (b"version https://git-lfs.github.com/spec/v1\noid sha256:ab\nsize 5\n", "git-LFS"), (b"", "empty")):
            facts, error = fetch.verify_file(self.write(data))
            self.assertIn(word, error)
            self.assertEqual(error.split(":")[0], "file rejected")

    def test_wrong_type_names_what_it_is_and_what_was_expected(self):
        _facts, error = fetch.verify_file(self.write(b"%PDF-1.4\n%%EOF\n", "x.pdf"))
        self.assertIn("a PDF document", error)
        self.assertIn("a STEP file", error)

    def test_pinned_hash_mismatch_and_match(self):
        path = self.write(STEP)
        self.assertEqual(fetch.verify_file(path, expected_sha256=STEP_SHA)[1], "")
        self.assertEqual(fetch.verify_file(path, expected_sha256=STEP_SHA.upper())[1], "")
        self.assertIn("hash mismatch", fetch.verify_file(path, expected_sha256="1" * 64)[1])

    def test_size_limit_and_unreadable_file(self):
        self.assertIn("too large", fetch.verify_file(self.write(STEP), max_bytes=100)[1])
        facts, error = fetch.verify_file("/no/such/file.step")
        self.assertEqual(facts, {})
        self.assertIn("cannot be read", error)

    def test_expect_narrows_the_accepted_types(self):
        path = self.write(STL_ASCII, "x.stl")
        self.assertEqual(fetch.verify_file(path)[1], "")
        self.assertIn("expected a STEP file", fetch.verify_file(path, expect=("step",))[1])


class CheckImportableTest(unittest.TestCase):
    def write(self, data: bytes, name: str) -> str:
        path = os.path.join(isolation.fresh_dir("imp_"), name)
        open(path, "wb").write(data)
        return path

    def test_step_and_stl(self):
        kind, facts, error = fetch.check_importable(os.path.join(FIXTURES, "pi4_like.step"))
        self.assertEqual((kind, error, facts["detected_type"]), ("brep", "", "step"))
        kind, _facts, error = fetch.check_importable(self.write(STL_ASCII, "cube.stl"))
        self.assertEqual((kind, error), ("mesh", ""))

    def test_relative_missing_and_unsupported(self):
        self.assertIn("must be absolute", fetch.check_importable("pi4.step")[2])
        self.assertIn("file not found", fetch.check_importable("/no/such/pi4.step")[2])
        self.assertIn("unsupported reference file type '.docx'", fetch.check_importable(self.write(b"x", "a.docx"))[2])
        self.assertIn("unsupported reference file type '(no extension)'", fetch.check_importable(self.write(b"x", "noext"))[2])

    def test_zip_is_refused_with_advice(self):
        error = fetch.check_importable(self.write(b"PK\x03\x04", "pi.zip"))[2]
        self.assertIn("ZIP archive", error)
        self.assertIn("unzip", error)

    def test_extension_and_content_must_agree(self):
        error = fetch.check_importable(self.write(STL_ASCII, "fake.step"))[2]
        self.assertIn("holds an STL mesh", error)
        self.assertIn("fake.step", error)
        self.assertIn("web page", fetch.check_importable(self.write(HTML, "login.step"))[2])
        self.assertIn("empty", fetch.check_importable(self.write(b"", "zero.step"))[2])

    def test_obj_and_ply_are_not_importable_references(self):
        self.assertIn("unsupported", fetch.check_importable(self.write(b"v 0 0 0\n", "a.obj"))[2])
        self.assertIn("unsupported", fetch.check_importable(self.write(b"ply\n", "a.ply"))[2])


class UrlCheckTest(Base):
    def test_allowlisted_https(self):
        self.assertEqual(fetch.check_url(URL), "")
        self.assertEqual(fetch.check_url("https://datasheets.raspberrypi.com/x.step"), "")

    def test_other_hosts_http_and_junk_are_refused(self):
        self.assertIn("not on the allowlist", fetch.check_url("https://grabcad.com/library/pi.step"))
        self.assertIn("not https", fetch.check_url("http://raw.githubusercontent.com/x/y/main/a.step"))
        self.assertIn("only http(s)", fetch.check_url("ftp://raw.githubusercontent.com/a.step"))
        self.assertIn("only http(s)", fetch.check_url(""))
        self.assertIn("not on the allowlist", fetch.check_url("https://raw.githubusercontent.com.evil.com/a.step"))
        self.assertIn("not on the allowlist", fetch.check_url("https://user@evil.com/raw.githubusercontent.com/a.step"))


class ConfirmReceivesTheUrlTest(Base):
    """Review finding (PR #3): the per-host grant was recovered from the dialog text by regex. fetch passes the URL structurally."""

    def test_a_callback_that_accepts_url_gets_it(self):
        got = {}

        def ask(summary, detail, url=""):
            got.update(url=url)
            return False

        with mock.patch.object(requests, "get", FakeGet(FakeResponse(STEP, url=URL))):
            fetch.fetch(self.entry, confirm=ask, dest_dir=self.dest, reason="r")
        self.assertEqual(got["url"], URL)

    def test_a_kwargs_callback_gets_it_too(self):
        got = {}

        def ask(summary, detail, **kwargs):
            got.update(kwargs)
            return False

        with mock.patch.object(requests, "get", FakeGet(FakeResponse(STEP, url=URL))):
            fetch.fetch(self.entry, confirm=ask, dest_dir=self.dest, reason="r")
        self.assertEqual(got, {"url": URL})

    def test_a_two_argument_callback_still_works(self):
        result = self.run_fetch(answer=False)
        self.assertEqual(result.status, "declined")
        self.assertEqual(len(self.asked), 1)

    def test_accepts_url_rules(self):
        self.assertTrue(fetch._accepts_url(lambda summary, detail, url="": True))
        self.assertTrue(fetch._accepts_url(lambda *a, **k: True))
        self.assertFalse(fetch._accepts_url(lambda summary, detail: True))
        self.assertFalse(fetch._accepts_url(42))


class HeavyModelTest(Base):
    """The official Pi 5 model unpacks to 77.6 MB / 2689 solids and freezes the GUI for ~200 s on import. A heavy catalog STEP is
    opt-in: the (measured) board data is used unless ``allow_heavy`` is passed."""

    step_urls = [{"url": URL, "mirror": False, "trust": "official", "size": 13_270_704, "member_bytes": 77_627_469,
                  "expected_sha256": "", "expected_bbox": [88.5, 57.2, 18.98], "license": "vendor-published"}]

    def test_a_heavy_model_is_not_downloaded_by_default(self):
        result = self.run_fetch()
        self.assertEqual(result.status, "nominal")
        self.assertEqual(self.net.calls, [])                       # nothing was fetched
        self.assertEqual(self.asked, [])                           # and the user was not even asked
        self.assertIn("allow_heavy=true", result.note)
        self.assertIn("measured from that official model", result.note)

    def test_allow_heavy_downloads_it_anyway(self):
        result = self.run_fetch(allow_heavy=True)
        self.assertNotEqual(result.note.count("NOT downloaded"), 1)
        self.assertTrue(self.net.calls)                            # the normal path (permission, download) ran
        self.assertTrue(self.asked)

    def test_a_user_supplied_url_is_never_second_guessed(self):
        result = self.run_fetch(url=URL)
        self.assertNotIn("NOT downloaded automatically", result.note)

    def test_is_heavy_rules(self):
        self.assertTrue(fetch.is_heavy({"heavy": True}))
        self.assertTrue(fetch.is_heavy({"member_bytes": fetch.HEAVY_IMPORT_BYTES}))
        self.assertTrue(fetch.is_heavy({"size": fetch.HEAVY_IMPORT_BYTES + 1}))
        self.assertFalse(fetch.is_heavy({"member_bytes": 17_535_443}))
        self.assertFalse(fetch.is_heavy({"member_bytes": "not a number"}))
        self.assertFalse(fetch.is_heavy({}))
        self.assertFalse(fetch.is_heavy(None))


class LightModelStillDownloadsTest(Base):
    """The community Pi 4 model (17.5 MB, ~8 s import) is below the limit and downloads as before."""

    step_urls = [{"url": URL, "mirror": False, "trust": "community", "size": 17_535_443, "expected_sha256": "",
                  "expected_bbox": [89.0, 58.4, 19.9], "license": "unspecified"}]

    def test_a_light_model_goes_through_the_permission_dialog(self):
        result = self.run_fetch()
        self.assertTrue(self.asked)
        self.assertTrue(self.net.calls)
        self.assertNotIn("allow_heavy", result.note)


class FetchStatusTest(Base):
    def test_nominal_when_the_catalog_lists_no_step_file(self):
        self.step_urls = []
        self.setUp()
        result = fetch.fetch(self.entry, confirm=self.confirm(True), dest_dir=self.dest)
        self.assertEqual((result.status, result.has_board, result.origin, result.path), ("nominal", True, "none", ""))
        self.assertIn("board data", result.note)
        self.assertEqual((self.asked, os.listdir(self.dest)), ([], []))

    def test_no_board_and_no_file_is_a_failure_not_nominal(self):
        entry = {"id": "motor_x", "name": "Motor X", "kind": "hardware", "step_urls": []}
        result = fetch.fetch(entry, confirm=self.confirm(True), dest_dir=self.dest)
        self.assertEqual((result.status, result.has_board), ("failed", False))
        self.assertIn("attach", result.note)

    def test_obtained_download_is_verified_and_registered(self):
        result = self.run_fetch()
        self.assertEqual(result.status, "obtained", result.note)
        self.assertEqual((result.origin, result.detected_type, result.sha256, result.bytes), ("download", "step", STEP_SHA, len(STEP)))
        self.assertEqual(result.path, os.path.join(self.dest, "pi4.step"))
        self.assertEqual(result.expected_bbox, [89.5, 57.0, 18.5])
        self.assertEqual(result.license, "CC-BY test")
        self.assertEqual(fetch.obtained_file("rpi4b")["sha256"], STEP_SHA)
        self.assertEqual(len(self.net.calls), 1)
        self.assertEqual(self.net.calls[0][0], URL)

    def test_the_permission_text_names_url_part_reason_and_the_safeguards(self):
        net = FakeGet(FakeResponse(STEP, url=URL))
        with mock.patch.object(requests, "get", net):
            fetch.fetch(self.entry, confirm=self.confirm(True), dest_dir=self.dest, reason="so the case fits")
        (summary, detail), = self.asked
        self.assertIn(URL, summary)
        self.assertIn("Raspberry Pi 4 Model B", summary)
        self.assertIn("so the case fits", detail)
        self.assertIn("raw.githubusercontent.com", detail)
        self.assertIn("at most one download", detail)

    def test_declined_by_the_user_makes_no_request_and_says_what_to_do(self):
        result = self.run_fetch(answer=False)
        self.assertEqual(result.status, "declined")
        self.assertEqual(self.net.calls, [])
        self.assertIn("declined", result.note)
        self.assertIn("attach", result.note)
        self.assertIn(URL, result.note)
        self.assertEqual(os.listdir(self.dest), [])
        self.assertEqual(fetch.obtained_file("rpi4b"), {})

    def test_no_confirm_channel_means_declined(self):
        net = FakeGet(FakeResponse(STEP))
        with mock.patch.object(requests, "get", net):
            result = fetch.fetch(self.entry, confirm=None, dest_dir=self.dest)
        self.assertEqual((result.status, net.calls), ("declined", []))

    def test_allow_download_false_is_declined_without_a_dialog(self):
        result = self.run_fetch(allow_download=False)
        self.assertEqual(result.status, "declined")
        self.assertIn("allow_download=false", result.note)
        self.assertEqual((self.asked, self.net.calls), ([], []))

    def test_http_error_is_failed_with_the_nominal_fallback_wording(self):
        net = FakeGet(FakeResponse(b"", status=404, url=URL))
        with mock.patch.object(requests, "get", net):
            result = fetch.fetch(self.entry, confirm=self.confirm(True), dest_dir=self.dest)
        self.assertEqual(result.status, "failed")
        self.assertIn("404", result.note)
        self.assertIn("board data is used instead", result.note)
        self.assertEqual(os.listdir(self.dest), [])

    def test_a_403_tells_the_model_to_ask_the_user_to_attach(self):
        net = FakeGet(FakeResponse(b"", status=403, url=URL))
        with mock.patch.object(requests, "get", net):
            result = fetch.fetch(self.entry, confirm=self.confirm(True), dest_dir=self.dest)
        self.assertEqual(result.status, "failed")
        self.assertIn("attach", result.note)

    def test_connection_error_is_failed(self):
        with mock.patch.object(requests, "get", FakeGet(raises=requests.ConnectionError("no route"))):
            result = fetch.fetch(self.entry, confirm=self.confirm(True), dest_dir=self.dest)
        self.assertEqual(result.status, "failed")
        self.assertIn("no route", result.note)

    def test_html_page_instead_of_a_file_fails_and_keeps_the_type_facts(self):
        result = self.run_fetch(HTML)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.detected_type, "html")
        self.assertIn("web page", result.note)
        self.assertEqual(os.listdir(self.dest), [])

    def test_a_pdf_is_not_a_reference_file(self):
        result = self.run_fetch(b"%PDF-1.4\n" + b"x" * 200)
        self.assertEqual((result.status, result.detected_type), ("failed", "pdf"))
        self.assertIn("expected", result.note)

    def test_redirect_off_the_allowlist_is_refused(self):
        result = self.run_fetch(final_url="https://evil.example.com/pi4.step")
        self.assertEqual(result.status, "failed")
        self.assertIn("redirected to 'evil.example.com'", result.note)

    def test_stop_during_the_download(self):
        net = FakeGet(FakeResponse(STEP, url=URL))
        with mock.patch.object(requests, "get", net):
            result = fetch.fetch(self.entry, confirm=self.confirm(True), dest_dir=self.dest, should_stop=lambda: True)
        self.assertEqual(result.status, "failed")
        self.assertIn("stopped", result.note)
        self.assertEqual(os.listdir(self.dest), [])

    def test_a_downloader_that_raises_is_never_visible_to_the_caller(self):
        # download_verified never raises; a replacement that does would still be a bug of the replacement: fetch must not hide it silently
        def boom(*a, **k):
            raise RuntimeError("bug in the downloader")
        with self.assertRaises(RuntimeError):
            fetch.fetch(self.entry, confirm=self.confirm(True), dest_dir=self.dest, downloader=boom)


class PinnedHashTest(Base):
    pinned = STEP_SHA

    def test_matching_pin_is_obtained(self):
        self.assertEqual(self.run_fetch().status, "obtained")

    def test_mismatching_pin_fails_and_does_not_register_the_file(self):
        self.entry["step_urls"][0]["expected_sha256"] = "a" * 64
        result = self.run_fetch()
        self.assertEqual(result.status, "failed")
        self.assertIn("hash mismatch", result.note)
        self.assertIn("aaaaaaaaaaaa", result.note)
        self.assertEqual(fetch.obtained_file("rpi4b"), {})
        self.assertEqual(result.sha256, STEP_SHA, "the hash of what arrived is kept for the message")


class ExplicitUrlTest(Base):
    def test_off_allowlist_url_is_refused_before_any_dialog_or_request(self):
        result = self.run_fetch(url="https://grabcad.com/library/pi.step")
        self.assertEqual(result.status, "failed")
        self.assertIn("not on the allowlist", result.note)
        self.assertEqual((self.asked, self.net.calls), ([], []))

    def test_a_refusal_before_the_network_costs_no_attempt(self):
        self.run_fetch(url="https://grabcad.com/library/pi.step", turn_id="T1")
        again = self.run_fetch(turn_id="T1")
        self.assertEqual(again.status, "obtained", again.note)

    def test_url_only_reference_gets_a_slug_id(self):
        result = self.run_fetch(entry=None, url="https://raw.githubusercontent.com/o/r/main/My Board.step".replace(" ", "%20"))
        self.assertEqual(result.status, "obtained", result.note)
        self.assertEqual(result.ref_id, "my_board")
        self.assertFalse(result.has_board)
        self.assertEqual(fetch.obtained_file("my_board")["path"], result.path)

    def test_explicit_url_is_used_instead_of_the_catalog_url(self):
        other = "https://raw.githubusercontent.com/other/repo/main/better.step"
        result = self.run_fetch(url=other)
        self.assertEqual(result.status, "obtained", result.note)
        self.assertEqual(self.net.calls[0][0], other)
        self.assertEqual(result.expected_bbox, [89.5, 57.0, 18.5], "the catalog's expectation still applies to this part")

    def test_a_catalog_url_given_explicitly_keeps_its_pin(self):
        self.entry["step_urls"][0]["expected_sha256"] = "b" * 64
        result = self.run_fetch(url=URL)
        self.assertEqual(result.status, "failed")
        self.assertIn("hash mismatch", result.note)

    def test_github_blob_url_is_rewritten_by_the_downloader(self):
        blob = "https://github.com/example/pi/blob/main/pi4.step"
        result = self.run_fetch(url=blob)
        self.assertEqual(result.status, "obtained", result.note)
        self.assertEqual(self.net.calls[0][0], "https://raw.githubusercontent.com/example/pi/main/pi4.step")


class ZipTest(Base):
    def make_zip(self, members: dict) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, data in members.items():
                zf.writestr(name, data)
        return buf.getvalue()

    def test_zip_member_is_extracted_verified_and_registered(self):
        body = self.make_zip({"models/pi4.step": STEP, "readme.txt": b"hi", "models/pi4.stl": STL_ASCII})
        result = self.run_fetch(body, url="https://raw.githubusercontent.com/o/r/main/pi.zip", final_url="https://raw.githubusercontent.com/o/r/main/pi.zip")
        self.assertEqual(result.status, "obtained", result.note)
        self.assertTrue(result.path.endswith("pi4.step"), "STEP is preferred over STL")
        self.assertEqual(result.sha256, STEP_SHA)
        self.assertEqual(result.archive["member"], "pi4.step")
        self.assertEqual(result.archive["sha256"], hashlib.sha256(body).hexdigest())
        self.assertIn("extracted", result.note)

    def test_zip_without_cad_is_failed(self):
        result = self.run_fetch(self.make_zip({"readme.txt": b"hi", "a.pdf": b"%PDF-1"}), url="https://raw.githubusercontent.com/o/r/main/pi.zip",
                                final_url="https://raw.githubusercontent.com/o/r/main/pi.zip")
        self.assertEqual(result.status, "failed")
        self.assertIn("no STEP", result.note)

    def test_zip_slip_is_refused(self):
        import uuid

        evil = f"zipslip_{uuid.uuid4().hex[:8]}.step"
        result = self.run_fetch(self.make_zip({f"../{evil}": STEP}), url="https://raw.githubusercontent.com/o/r/main/pi.zip",
                                final_url="https://raw.githubusercontent.com/o/r/main/pi.zip")
        self.assertEqual(result.status, "failed")
        self.assertIn("unsafe", result.note)
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(self.dest), evil)))

    def test_pin_applies_to_the_archive_not_the_member(self):
        body = self.make_zip({"pi4.step": STEP})
        self.entry["step_urls"][0]["expected_sha256"] = hashlib.sha256(body).hexdigest()
        result = self.run_fetch(body)
        self.assertEqual(result.status, "obtained", result.note)
        self.entry["step_urls"][0]["expected_sha256"] = "c" * 64
        fetch.forget_all()
        self.assertEqual(self.run_fetch(body).status, "failed")

    def test_pick_cad_member_ranking(self):
        folder = isolation.fresh_dir("pick_")
        names = {"a.stl": b"x" * 10, "b.step": b"x" * 10, "big.step": b"x" * 100, "pi4_model.step": b"x" * 5, "c.brep": b"x"}
        paths = {n: os.path.join(folder, n) for n in names}
        for n, data in names.items():
            open(paths[n], "wb").write(data)
        self.assertEqual(fetch.pick_cad_member(paths.values()), paths["big.step"])
        self.assertEqual(fetch.pick_cad_member(paths.values(), "Pi4 model"), paths["pi4_model.step"])
        self.assertEqual(fetch.pick_cad_member([paths["a.stl"], paths["c.brep"]]), paths["c.brep"])
        self.assertEqual(fetch.pick_cad_member([os.path.join(folder, "x.txt")]), "")
        self.assertEqual(fetch.pick_cad_member([]), "")


class AttemptRulesTest(Base):
    def test_a_failed_attempt_ends_the_downloads_of_the_turn(self):
        first = self.run_fetch(HTML, turn_id="T1")
        self.assertEqual(first.status, "failed")
        net = FakeGet(FakeResponse(STEP, url=URL))
        with mock.patch.object(requests, "get", net):
            second = fetch.fetch(self.entry, url="https://raw.githubusercontent.com/o/r/main/other.step", confirm=self.confirm(True),
                                 dest_dir=self.dest, turn_id="T1")
        self.assertEqual(second.status, "failed")
        self.assertIn("only one download attempt", second.note)
        self.assertEqual(net.calls, [], "no second request after a failure")

    def test_the_same_call_again_returns_the_first_answer(self):
        first = self.run_fetch(HTML, turn_id="T1")
        second = self.run_fetch(STEP, turn_id="T1")
        self.assertEqual((second.status, second.cached), ("failed", True))
        self.assertTrue(second.note.startswith("already attempted in this turn"))
        self.assertEqual(second.note.split(": ", 1)[1], first.note)
        self.assertEqual(self.net.calls, [])

    def test_a_new_turn_starts_fresh(self):
        self.run_fetch(HTML, turn_id="T1")
        self.assertEqual(self.run_fetch(STEP, turn_id="T2").status, "obtained")

    def test_declined_is_remembered_for_the_turn(self):
        self.run_fetch(answer=False, turn_id="T1")
        self.assertEqual(len(self.asked), 1)
        again = self.run_fetch(answer=True, turn_id="T1")
        self.assertEqual((again.status, again.cached, len(self.asked)), ("declined", True, 1))

    def test_successes_are_limited_but_not_to_one(self):
        for n in range(fetch.MAX_ATTEMPTS_PER_TURN):
            url = f"https://raw.githubusercontent.com/o/r/main/p{n}.step"
            self.assertEqual(self.run_fetch(url=url, turn_id="T1", entry=None).status, "obtained")
        extra = self.run_fetch(url="https://raw.githubusercontent.com/o/r/main/extra.step", turn_id="T1", entry=None)
        self.assertEqual(extra.status, "failed")
        self.assertIn("downloads were already made", extra.note)

    def test_calls_without_a_turn_id_are_never_limited(self):
        for _ in range(5):
            self.assertEqual(self.run_fetch(STEP).status, "obtained")

    def test_budget_function(self):
        self.assertEqual(fetch.attempt_budget("T9"), "")
        self.assertEqual(fetch.attempt_budget(""), "")
        fetch.record_attempt("T9", "k", fetch.SourceResult(status="failed"))
        self.assertIn("only one download attempt", fetch.attempt_budget("T9"))

    def test_turn_table_is_bounded(self):
        for n in range(fetch._TURNS_KEPT + 10):
            fetch.record_attempt(f"T{n}", "k", fetch.SourceResult(status="obtained"))
        self.assertLessEqual(len(fetch._ATTEMPTS), fetch._TURNS_KEPT)
        self.assertEqual(fetch.attempt_budget("T0"), "", "the oldest turn was forgotten")


class RegistryTest(Base):
    def test_obtained_file_requires_the_file_to_exist(self):
        path = os.path.join(isolation.fresh_dir("reg_"), "a.step")
        open(path, "wb").write(STEP)
        fetch.register_obtained("RPI4B", path, sha256=STEP_SHA)
        self.assertEqual(fetch.obtained_file("rpi4b")["path"], path)
        os.remove(path)
        self.assertEqual(fetch.obtained_file("rpi4b"), {})
        self.assertEqual(fetch.latest_obtained(), {})

    def test_latest_and_by_sha(self):
        folder = isolation.fresh_dir("reg_")
        a, b = os.path.join(folder, "a.step"), os.path.join(folder, "b.step")
        open(a, "wb").write(STEP)
        open(b, "wb").write(STEP + b" ")
        fetch.register_obtained("one", a, sha256="1" * 64)
        fetch.register_obtained("two", b, sha256="2" * 64)
        self.assertEqual(fetch.latest_obtained()["ref_id"], "two")
        self.assertEqual(fetch.obtained_by_sha("1" * 64)["ref_id"], "one")
        self.assertEqual(fetch.obtained_by_sha("3" * 64), {})
        self.assertEqual(fetch.obtained_by_sha(""), {})

    def test_registry_is_thread_safe(self):
        import threading

        path = os.path.join(isolation.fresh_dir("reg_"), "a.step")
        open(path, "wb").write(STEP)

        def work(n):
            for i in range(50):
                fetch.register_obtained(f"id{n}_{i}", path, sha256=str(n))
                fetch.obtained_file(f"id{n}_{i}")
                fetch.latest_obtained()
                fetch.record_attempt(f"T{n}", "k", fetch.SourceResult(status="obtained"))
                fetch.attempt_budget(f"T{n}")

        threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len([k for k in fetch._OBTAINED if k.startswith("id")]), 300)


class SourceResultTest(unittest.TestCase):
    def test_as_dict_round_trips_through_json(self):
        result = fetch.SourceResult(status="obtained", ref_id="rpi4b", sha256="a" * 64, archive={"member": "x.step"})
        self.assertEqual(result.provenance, "")
        data = json.loads(json.dumps(result.as_dict()))
        self.assertEqual(data["status"], "obtained")
        self.assertEqual(fetch.SourceResult(**data), result)

    def test_statuses_are_the_contract_enum(self):
        self.assertEqual(fetch.STATUSES, ("obtained", "declined", "failed", "nominal"))


class ShippedCatalogTest(unittest.TestCase):
    def test_no_url_is_listed_that_was_not_verified(self):
        # a STEP URL is listed only with the evidence of a real download (see StepRegistryTest in test_refparts_registry)
        for entry in catalog.list_entries():
            if entry["id"] not in ("rpi4b", "rpi5"):
                self.assertEqual(entry["step_urls"], [], f"{entry['id']}: no STEP URL may be listed before it is verified (decision 19)")
            self.assertFalse(entry["verified"], entry["id"])

    def test_every_shipped_entry_with_a_board_is_nominal_without_a_network(self):
        for entry in catalog.list_entries():
            if entry["step_urls"]:                 # migration: the two verified sources are downloads, not nominal (tested in test_refparts_registry)
                continue
            with mock.patch.object(requests, "get", side_effect=AssertionError("no network")):
                result = fetch.fetch(entry, confirm=lambda s, d: self.fail("no dialog"), dest_dir=isolation.fresh_dir("nom_"))
            self.assertEqual((result.status, result.has_board), ("nominal" if entry.get("board") else "failed", bool(entry.get("board"))), entry["id"])


class DefaultDownloaderTest(Base):
    def test_default_downloader_is_netguard(self):
        # fetch() with no downloader goes through netguard.download_verified (the FakeGet tests above rely on it); pin the wiring
        seen = {}
        real = netguard.download_verified

        def spy(url, dest, filename="", **kwargs):
            seen.update(kwargs, url=url)
            return real(url, dest, filename, **kwargs)

        with mock.patch.object(netguard, "download_verified", spy), mock.patch.object(requests, "get", FakeGet(FakeResponse(STEP, url=URL))):
            result = fetch.fetch(self.entry, confirm=self.confirm(True), dest_dir=self.dest)
        self.assertEqual(result.status, "obtained")
        self.assertEqual(seen["allowed_hosts"], catalog.allow_hosts())
        self.assertEqual(set(seen["expect"]), {"step", "stl", "brep", "iges", "zip"})
        self.assertEqual(seen["max_bytes"], fetch.MAX_REFERENCE_BYTES)
        self.assertEqual(seen["url"], URL)


if __name__ == "__main__":
    unittest.main()
