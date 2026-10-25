"""STEP sourcing registry: the two verified catalog sources, trust levels and the hash policy (``refparts.catalog`` / ``fetch`` / ``identity``).

On 2026-10-02 exactly two files were downloaded for verification and registered: the official Raspberry Pi 5 STEP (a ZIP behind a
redirect, trust ``official``) and a community Raspberry Pi 4 Model B STEP (trust ``community``). The real files are NOT committed and
never touched here (the opt-in ``test_refparts_real`` runs them). These tests cover the registry data (every evidence field), the
trust wording, the board cross-check records, and the policy that a vendor-side change of a pinned file degrades to a warning while its
bounding box / outline identity still passes (and is refused when it does not). The network is a fake ``requests.get`` behind the real
``netguard.download_verified`` (SSRF guard, allowlist on every redirect hop, ZIP extraction); the file bytes are the Pi-4-like fixture.
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

from oscar.refparts import catalog, fetch, identity  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "refparts")
STEP = open(os.path.join(FIXTURES, "pi4_like.step"), "rb").read()
CHANGED = STEP + b"\n/* revised by the vendor */\n"
PI5_URL = "https://pip.raspberrypi.com/documents/RP-010083-CA"
PI5_FINAL = "https://pip-assets.raspberrypi.com/categories/892-raspberry-pi-5/documents/RP-010083-CA-1-rpi-5%203D%20STEP%20-%20No%20Graphics%20small%20file.zip"
PI4_URL = ("https://raw.githubusercontent.com/multigamesystem/MGS-CAD-Files/main/STEP%20files%20with%20images/"
           "Raspberry%20Pi%204%20Model%20B%20v4.step")
MEMBER = "rpi-5b_no_graphics.step"
SHA64 = "0123456789abcdef" * 4


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_zip(member_bytes: bytes = STEP, member: str = MEMBER, extra=(("LICENSE.txt", b"MIT test licence"),)) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, member_bytes)
        for name, data in extra:
            zf.writestr(name, data)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, body=b"", *, status=200, url="", headers=None):
        self.status_code, self.url, self._body = status, url, body
        self.headers = {"Content-Length": str(len(body)), **(headers or {})}

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
        if self._body:
            yield self._body


class Net:
    """A fake ``requests.get``: ``routes`` maps a URL to a ``FakeResponse`` (or a callable returning one); records every URL asked for."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, url, **kwargs):
        self.calls.append(url)
        if url not in self.routes:
            raise AssertionError(f"unexpected request to {url}")
        route = self.routes[url]
        return route() if callable(route) else route


def pi5_routes(archive: bytes) -> dict:
    return {PI5_URL: FakeResponse(status=302, url=PI5_URL, headers={"Location": PI5_FINAL}),
            PI5_FINAL: FakeResponse(archive, url=PI5_FINAL, headers={"Content-Type": "application/zip"})}


class CatalogCopy(unittest.TestCase):
    """A deep copy of the shipped catalog installed as the live one; ``self.entry(id)`` edits the copy."""

    def setUp(self):
        fetch.forget_all()
        self.cat = copy.deepcopy(catalog._catalog())
        patcher = mock.patch.object(catalog, "_catalog", return_value=self.cat)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(fetch.ENV_LOCAL_DIR, None)
        self.dest = isolation.fresh_dir("registry_dl_")
        self.asked = []

    def entry(self, ident):
        return next(e for e in self.cat["entries"] if e["id"] == ident)

    def pin_pi5(self, archive: bytes, member: bytes = STEP, **overrides):
        step = self.entry("rpi5")["step_urls"][0]
        step.update(expected_sha256=sha(archive), size=len(archive), member_sha256=sha(member), member_bytes=len(member), **overrides)
        return step

    def confirm(self, answer=True):
        def ask(summary, detail):
            self.asked.append((summary, detail))
            return answer
        return ask

    def fetch(self, ident, routes, **kwargs):
        net = Net(routes)
        with mock.patch.object(requests, "get", net):
            kwargs.setdefault("allow_heavy", True)   # the official Pi 5 archive is heavy (opt-in); these tests exercise the download path
            result = fetch.fetch(catalog.get(ident), confirm=self.confirm(True), dest_dir=self.dest, reason="reference geometry", **kwargs)
        self.net = net
        return result


# --- the shipped data --------------------------------------------------------------------------------------------

class ShippedSourcesTest(unittest.TestCase):
    def step(self, ident):
        steps = catalog.get(ident)["step_urls"]
        self.assertEqual(len(steps), 1, ident)
        return steps[0]

    def test_exactly_two_sources_are_listed_nothing_for_third_party_aggregators(self):
        listed = {e["id"]: e["step_urls"] for e in catalog.list_entries() if e["step_urls"]}
        self.assertEqual(sorted(listed), ["rpi4b", "rpi5"])
        for ident in ("arduino_uno", "arduino_nano", "esp32_devkitc", "rpi_pico", "rpi3bplus", "rpizero2w", "jetson_nano"):
            self.assertEqual(catalog.get(ident)["step_urls"], [], ident)
        text = json.dumps(catalog.list_entries()).lower() + " ".join(catalog.allow_hosts())
        for banned in ("snapeda", "step.parts", "grabcad", "traceparts"):
            self.assertNotIn(banned, text)

    def test_the_catalog_validates_without_a_single_warning(self):
        ok, errors, warnings = catalog.validate_catalog()
        self.assertEqual((ok, errors, warnings), (True, [], []))

    def test_pi5_official_source_has_all_the_evidence(self):
        step = self.step("rpi5")
        self.assertEqual(step["url"], PI5_URL)
        self.assertEqual((step["trust"], step["mirror"], step["on_hash_mismatch"], step["last_verified"]), ("official", False, "identity", "2026-10-02"))
        self.assertEqual((step["size"], step["member"], step["member_bytes"]), (13270704, MEMBER, 77627469))
        self.assertEqual(step["expected_sha256"], "6841637b4cfa97637bf34b529bfe896f24d9634b2c40b57a156781f55d88f06f")
        self.assertEqual(step["member_sha256"], "78164070c1cc7ab854382950d2e85926c6e35432f1e5a146b733397a7c6afb44")
        self.assertEqual(step["expected_bbox"], [88.5, 57.2, 18.98])
        self.assertIn("vendor-published, see vendor terms", step["license"])
        self.assertIn("MIT", step["license"])
        self.assertIn("pip-assets.raspberrypi.com", step["note"])
        hosts = catalog.allow_hosts()
        self.assertIn("pip.raspberrypi.com", hosts)
        self.assertIn("pip-assets.raspberrypi.com", hosts, "the documented redirect target must be allowlisted")

    def test_pi4_community_source_never_claims_more_than_it_knows(self):
        step = self.step("rpi4b")
        self.assertEqual(step["url"], PI4_URL)
        self.assertEqual((step["trust"], step["on_hash_mismatch"], step["last_verified"], step["size"]), ("community", "identity", "2026-10-02", 17535443))
        self.assertEqual(step["expected_sha256"], "fd75897c59e79106ada16c3c744c212b92e60f1992b8b42349ddc2f2b3e90ce4")
        self.assertEqual(step["expected_bbox"], [88.99, 58.4, 19.9])
        self.assertEqual(step["license"], "unspecified - community model, accuracy unverified: cross-checked against the official drawing, see notes")
        self.assertNotIn("member", step)
        self.assertIn("no official Pi 4 STEP exists", step["note"])

    def test_the_entries_are_not_marked_verified_as_boards(self):
        for ident in ("rpi4b", "rpi5"):
            entry = catalog.get(ident)
            self.assertFalse(entry["verified"], "the board data is not verified against the drawing")
            self.assertEqual(entry["last_verified"], "2026-10-02")

    def test_pi4_board_keeps_the_drawing_values_and_records_the_cross_check(self):
        board = catalog.load_board("rpi4b")
        self.assertEqual((board["outline"]["length"], board["outline"]["width"], board["outline"]["corner_radius"], board["outline"]["confidence"]),
                         (85.0, 56.0, 3.0, "datasheet"))
        holes = board["mounting_holes"]
        self.assertEqual((holes["diameter"], holes["pad_diameter"], holes["pattern"], holes["confidence"]),
                         (2.75, 6.2, {"offset": [3.5, 3.5], "dx": 58.0, "dy": 49.0}, "datasheet"))
        self.assertEqual((board["pcb"]["thickness"], board["pcb"]["confidence"]), (1.4, "assumed"), "the community 1.6 never overwrites it")
        self.assertEqual(board["connectors"][0]["box"], {"x": [6.7, 15.7], "y": [-0.5, 6.9], "z": [1.4, 4.7]})
        self.assertNotIn("measured", {c["confidence"] for c in board["connectors"]})
        cross = board["step_cross_check"]
        self.assertEqual((cross["trust"], cross["date"]), ("community", "2026-10-02"))
        text = " ".join(cross["discrepancies"])
        for fact in ("1.6", "6.0", "88.99 x 58.4 x 19.9"):
            self.assertIn(fact, text)
        self.assertIn("RECORDED ONLY", cross["result"])

    def test_pi5_board_is_measured_only_where_the_model_is_unambiguous(self):
        board = catalog.load_board("rpi5")
        measured = {c["id"]: c for c in board["connectors"] if c["confidence"] == "measured"}
        self.assertEqual(sorted(measured), ["ethernet", "gpio", "usb2_pair", "usb3_pair", "usb_c_power"])
        self.assertEqual((board["pcb"]["thickness"], board["pcb"]["confidence"]), (1.28, "measured"))
        centre = lambda c: round(sum(c["box"]["y"]) / 2, 2)  # noqa: E731
        self.assertEqual([centre(measured[i]) for i in ("ethernet", "usb3_pair", "usb2_pair")], [10.25, 29.0, 47.0])
        self.assertEqual(measured["usb_c_power"]["box"], {"x": [6.73, 15.67], "y": [-1.2, 6.12], "z": [0.31, 4.57]})
        for ident in ("hdmi0", "hdmi1", "microsd"):                 # not (reliably) in the official model: drawing-based values stay
            conn = next(c for c in board["connectors"] if c["id"] == ident)
            self.assertNotEqual(conn["confidence"], "measured", ident)
            self.assertIn("cross-check 2026-10-02", conn["note"])
        for conn in measured.values():
            self.assertTrue(conn["source"].startswith("measured from the official Raspberry Pi 5 STEP"), conn["id"])
            self.assertEqual(conn["source_url"], PI5_URL)
        self.assertIn("unverified", measured["usb3_pair"]["note"], "which stack is USB 3.0 cannot be read from the unnamed solids")
        self.assertEqual((board["outline"]["confidence"], board["mounting_holes"]["confidence"]), ("datasheet", "datasheet"))
        self.assertEqual(board["height"], {"max_top_z": 17.5, "min_bottom_z": -1.5})
        self.assertEqual(catalog.validate_board(board)[:2], (True, []))
        self.assertEqual(board["step_cross_check"]["trust"], "official")
        self.assertTrue(any("ethernet" in d for d in board["step_cross_check"]["discrepancies"]))


# --- trust helpers and validator rules ----------------------------------------------------------------------------------

class TrustHelpersTest(unittest.TestCase):
    def test_trust_of_defaults_to_the_safe_reading(self):
        self.assertEqual(catalog.trust_of({"trust": "official"}), "official")
        for step in ({}, {"trust": "community"}, {"trust": "vendor"}, {"trust": None}, None, "x"):
            self.assertEqual(catalog.trust_of(step), "community", repr(step))

    def test_trust_label_wording(self):
        self.assertEqual(catalog.trust_label("community"), "community, unverified accuracy")
        self.assertEqual(catalog.trust_label("official"), "official, vendor-published")
        self.assertEqual(catalog.trust_label("official", changed=True), "official, vendor-published, file changed since verified")
        self.assertEqual(catalog.trust_label("community", changed=True), "community, unverified accuracy, file changed since verified")
        self.assertEqual(catalog.trust_label("nonsense"), "community, unverified accuracy")

    def test_step_for_prefers_url_then_hash_then_first(self):
        entry = {"step_urls": [{"url": "https://a", "expected_sha256": "1" * 64}, {"url": "https://b", "expected_sha256": "2" * 64, "member_sha256": "3" * 64}]}
        self.assertEqual(catalog.step_for(entry, url="https://b")["url"], "https://b")
        self.assertEqual(catalog.step_for(entry, sha256="3" * 64)["url"], "https://b")
        self.assertEqual(catalog.step_for(entry, sha256=("2" * 64).upper())["url"], "https://b")
        self.assertEqual(catalog.step_for(entry, sha256="9" * 64)["url"], "https://a")
        self.assertEqual(catalog.step_for({"step_urls": []}), {})
        self.assertEqual(catalog.step_for({}, url="x"), {})
        self.assertEqual(catalog.step_for(None), {})


class StepFieldValidatorTest(unittest.TestCase):
    def cat(self, **extra):
        cat = json.loads(json.dumps({"schema": 1, "allow_hosts": catalog.allow_hosts(), "entries": catalog.list_entries()}))
        cat["entries"][0]["step_urls"][0].update(extra)
        return cat

    def errors(self, **extra):
        ok, errors, _w = catalog.validate_catalog(self.cat(**extra))
        self.assertFalse(ok, f"{extra} must be refused")
        return "\n".join(errors)

    def test_the_shipped_items_pass(self):
        self.assertEqual(catalog.validate_catalog(self.cat())[:2], (True, []))

    def test_every_new_field_is_checked(self):
        for extra, fragment in [
            ({"trust": "vendor"}, "trust 'vendor' must be one of official, community"),
            ({"size": 0}, "size must be a positive integer"),
            ({"size": "13"}, "size must be a positive integer"),
            ({"size": True}, "size must be a positive integer"),
            ({"member": "a/b.step", "member_sha256": SHA64}, "member must be one plain file name"),
            ({"member": ".."}, "member must be one plain file name"),
            ({"member": "x.step", "member_sha256": "XYZ"}, "member_sha256 must be 64 lower-case hex characters"),
            ({"member_sha256": SHA64}, "need a 'member' name"),
            ({"member_bytes": 5}, "need a 'member' name"),
            ({"on_hash_mismatch": "ignore"}, "on_hash_mismatch 'ignore' must be one of reject, identity"),
            ({"on_hash_mismatch": "identity", "expected_bbox": []}, "needs expected_bbox"),
            ({"last_verified": "yesterday"}, "last_verified must be YYYY-MM-DD"),
            ({"note": 5}, "note must be a string"),
        ]:
            self.assertIn(fragment, self.errors(**extra), extra)

    def test_a_legacy_item_without_the_new_fields_stays_valid(self):
        cat = self.cat()
        cat["entries"][0]["step_urls"] = [{"url": PI4_URL, "mirror": False, "expected_sha256": "", "expected_bbox": [], "license": "x"}]
        self.assertTrue(catalog.validate_catalog(cat)[0])

    def test_the_cross_check_record_is_validated(self):
        board = catalog.load_board("rpi4b")
        self.assertEqual(catalog.validate_board(board)[:2], (True, []))
        for mutate in (lambda c: c.pop("source"), lambda c: c.update(trust="vendor"), lambda c: c.update(date="today"),
                       lambda c: c.update(discrepancies="x"), lambda c: c.update(result="")):
            broken = copy.deepcopy(board)
            mutate(broken["step_cross_check"])
            ok, errors, _w = catalog.validate_board(broken)
            self.assertFalse(ok)
            self.assertTrue(any(e.startswith("step_cross_check:") for e in errors), errors)
        broken = copy.deepcopy(board)
        broken["step_cross_check"] = "fine"
        self.assertFalse(catalog.validate_board(broken)[0])


# --- the pure policy -----------------------------------------------------------------------------------------------------

class PinVerdictTest(unittest.TestCase):
    def ident(self, measured, expected=(88.5, 57.2, 18.98)):
        return identity.bbox_identity(measured, expected)

    def test_a_matching_or_missing_pin_accepts_without_looking_at_identity(self):
        for pin in ("match", "unpinned", "", None):
            self.assertEqual(identity.pin_verdict(pin, [self.ident((1, 1, 1))]), ("accepted", ""), pin)

    def test_a_changed_file_whose_geometry_still_matches_is_a_warning(self):
        verdict, text = identity.pin_verdict("mismatch", [self.ident((88.4, 57.0, 19.0))])
        self.assertEqual(verdict, "accepted_with_warning")
        self.assertIn("vendor changed it", text)
        self.assertIn("bounding box / outline still match", text)

    def test_a_changed_file_with_a_different_part_is_rejected_and_names_the_check(self):
        verdict, text = identity.pin_verdict("mismatch", [self.ident((120.0, 57.0, 19.0))])
        self.assertEqual(verdict, "rejected")
        self.assertIn("geometry no longer matches", text)
        self.assertIn("does not match the expected 88.5 x 57.2 x 18.98", text)

    def test_any_failing_check_rejects_even_when_another_passes(self):
        good, bad = self.ident((88.5, 57.2, 19.0)), self.ident((88.5, 20.0, 19.0))
        self.assertEqual(identity.pin_verdict("mismatch", [good, bad])[0], "rejected")
        self.assertEqual(identity.pin_verdict("mismatch", [bad, good])[0], "rejected")

    def test_nothing_checkable_is_rejected(self):
        unchecked = identity.bbox_identity((88.0, 57.0, 19.0), ())
        self.assertEqual(unchecked.status, "unchecked")
        for checks in ([], None, [unchecked], [None, "x", 3]):
            verdict, text = identity.pin_verdict("mismatch", checks)
            self.assertEqual(verdict, "rejected", checks)
            self.assertIn("nothing could confirm", text)

    def test_the_dict_form_of_measure_is_understood(self):
        ok = {"ok": True, "status": "match", "detail": "fits"}
        bad = {"ok": False, "status": "mismatch", "detail": "wrong size"}
        self.assertEqual(identity.pin_verdict("mismatch", [ok])[0], "accepted_with_warning")
        verdict, text = identity.pin_verdict("mismatch", [ok, bad])
        self.assertEqual(verdict, "rejected")
        self.assertIn("wrong size", text)

    def test_boundary_of_the_bounding_box_tolerance(self):
        expected = (88.5, 57.2, 18.98)
        tol = max(1.5, 0.08 * 18.98)                                    # the smallest side: 1.5 mm
        self.assertEqual(identity.pin_verdict("mismatch", [self.ident((88.5, 57.2, 18.98 + tol - 0.01), expected)])[0], "accepted_with_warning")
        self.assertEqual(identity.pin_verdict("mismatch", [self.ident((88.5, 57.2, 18.98 + tol + 0.05), expected)])[0], "rejected")
        self.assertEqual(identity.pin_verdict("mismatch", [self.ident((88.5, 57.2, 18.98 - tol - 0.05), expected)])[0], "rejected")


# --- fetch: pins, trust, redirects and archives ----------------------------------------------------------------------------

class PolicyHelpersTest(CatalogCopy):
    def test_hash_policy_needs_both_the_flag_and_a_signature(self):
        self.assertEqual(fetch.hash_policy({"on_hash_mismatch": "identity", "expected_bbox": [1, 2, 3]}), "identity")
        self.assertEqual(fetch.hash_policy({"on_hash_mismatch": "identity", "expected_bbox": []}), "reject")
        self.assertEqual(fetch.hash_policy({"expected_bbox": [1, 2, 3]}), "reject")
        self.assertEqual(fetch.hash_policy({"on_hash_mismatch": "reject", "expected_bbox": [1, 2, 3]}), "reject")
        self.assertEqual(fetch.hash_policy(None), "reject")

    def test_pin_state_uses_the_member_hash_of_an_archive_item(self):
        entry = catalog.get("rpi5")
        step = entry["step_urls"][0]
        self.assertEqual(fetch.pin_state(entry, step["member_sha256"]), "match")
        self.assertEqual(fetch.pin_state(entry, step["member_sha256"].upper()), "match")
        self.assertEqual(fetch.pin_state(entry, "0" * 64), "mismatch")
        self.assertEqual(fetch.pin_state(catalog.get("rpi4b"), catalog.get("rpi4b")["step_urls"][0]["expected_sha256"]), "match")
        self.assertEqual(fetch.pin_state(catalog.get("arduino_uno"), "0" * 64), "", "an entry without a STEP source pins nothing")
        self.assertEqual(fetch.pin_state(None, "0" * 64), "")
        self.entry("rpi4b")["step_urls"][0]["expected_sha256"] = ""
        self.assertEqual(fetch.pin_state(catalog.get("rpi4b"), "0" * 64), "unpinned")

    def test_trust_info(self):
        entry = catalog.get("rpi5")
        good = fetch.trust_info(entry, entry["step_urls"][0]["member_sha256"])
        self.assertEqual((good["trust"], good["label"], good["pin"], good["policy"]), ("official", "official, vendor-published", "match", "identity"))
        self.assertEqual(good["last_verified"], "2026-10-02")
        changed = fetch.trust_info(entry, "0" * 64)
        self.assertEqual((changed["pin"], changed["label"]), ("mismatch", "official, vendor-published, file changed since verified"))
        self.assertEqual(fetch.trust_info(catalog.get("rpi4b"), "0" * 64)["label"], "community, unverified accuracy, file changed since verified")
        self.assertEqual(fetch.trust_info(catalog.get("rpi_pico"), "0" * 64), {})
        self.assertEqual(fetch.trust_info(None, ""), {})

    def test_source_result_round_trips_the_new_fields(self):
        result = fetch.SourceResult(status="obtained", ref_id="rpi5", trust="official", trust_label="official, vendor-published",
                                    pin="mismatch", warnings=["w"])
        data = json.loads(json.dumps(result.as_dict()))
        self.assertEqual(fetch.SourceResult(**data), result)
        entry = fetch.entry_summary(catalog.get("rpi5"))
        self.assertEqual((entry["step_urls"], entry["step_trust"]), (1, ["official"]))
        self.assertEqual(fetch.entry_summary(catalog.get("rpi4b"))["step_trust"], ["community"])


class OfficialArchiveTest(CatalogCopy):
    def test_redirect_and_zip_member_extraction_through_netguard(self):
        archive = make_zip()
        self.pin_pi5(archive)
        result = self.fetch("rpi5", pi5_routes(archive))
        self.assertEqual(result.status, "obtained", result.note)
        self.assertEqual(self.net.calls, [PI5_URL, PI5_FINAL], "one request per hop, nothing else")
        self.assertEqual(result.final_url, PI5_FINAL)
        self.assertEqual(os.path.basename(result.path), MEMBER)
        self.assertEqual(open(result.path, "rb").read(), STEP)
        self.assertEqual((result.sha256, result.archive["sha256"], result.archive["member"]), (sha(STEP), sha(archive), MEMBER))
        self.assertEqual((result.trust, result.trust_label, result.pin, result.warnings), ("official", "official, vendor-published", "match", []))
        self.assertIn("vendor-published, see vendor terms", result.license)
        self.assertEqual(result.expected_bbox, [88.5, 57.2, 18.98])
        self.assertEqual(fetch.obtained_file("rpi5")["trust"], "official")
        summary, detail = self.asked[0]
        self.assertIn("• Source: official model", summary)
        self.assertIn("Trust: official, vendor-published.", detail)
        self.assertIn(PI5_URL, summary)

    def test_a_member_pin_mismatch_alone_is_also_a_warning(self):
        archive = make_zip(CHANGED)
        self.pin_pi5(archive, STEP)                                  # the archive hash matches, the member hash is the old one
        result = self.fetch("rpi5", pi5_routes(archive))
        self.assertEqual((result.status, result.pin), ("obtained", "mismatch"), result.note)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("the file has sha256", result.warnings[0])

    def test_a_vendor_side_change_degrades_to_a_warning_when_identity_policy_applies(self):
        archive_old = make_zip()
        self.pin_pi5(archive_old)
        archive_new = make_zip(CHANGED)
        result = self.fetch("rpi5", pi5_routes(archive_new))
        self.assertEqual((result.status, result.pin), ("obtained", "mismatch"), result.note)
        self.assertEqual(open(result.path, "rb").read(), CHANGED)
        self.assertEqual(len(result.warnings), 2, result.warnings)           # the archive and the member both differ
        self.assertIn("the archive has sha256", result.warnings[0])
        self.assertIn("published file changed", result.warnings[0])
        self.assertIn("WARNING:", result.note)
        self.assertEqual(result.trust_label, "official, vendor-published, file changed since verified")
        self.assertEqual(fetch.pin_state(catalog.get("rpi5"), result.sha256), "mismatch")
        self.assertEqual(fetch.obtained_file("rpi5")["pin"], "mismatch")

    def test_the_strict_policy_still_refuses_a_changed_archive(self):
        self.pin_pi5(make_zip(), on_hash_mismatch="reject")
        result = self.fetch("rpi5", pi5_routes(make_zip(CHANGED)))
        self.assertEqual(result.status, "failed")
        self.assertIn("hash mismatch: the archive has sha256", result.note)
        self.assertEqual(fetch.obtained_file("rpi5"), {})

    def test_the_identity_flag_without_a_signature_is_strict(self):
        self.pin_pi5(make_zip(), expected_bbox=[])
        self.entry("rpi5")["step_urls"][0].pop("on_hash_mismatch")            # a validator-clean item cannot have both; the runtime still copes
        result = self.fetch("rpi5", pi5_routes(make_zip(CHANGED)))
        self.assertEqual(result.status, "failed")

    def test_the_catalog_member_name_is_preferred_over_other_cad_files(self):
        archive = make_zip(STEP, extra=(("LICENSE.txt", b"x"), ("zzz_big_other.step", STEP + b"\n" * 5000)))
        self.pin_pi5(archive)
        result = self.fetch("rpi5", pi5_routes(archive))
        self.assertEqual((result.status, os.path.basename(result.path)), ("obtained", MEMBER), result.note)

    def test_a_renamed_member_falls_back_to_the_best_cad_file_and_is_judged_by_identity(self):
        archive = make_zip(STEP, member="renamed_by_vendor.step")
        self.pin_pi5(archive)
        result = self.fetch("rpi5", pi5_routes(archive))
        self.assertEqual((result.status, os.path.basename(result.path)), ("obtained", "renamed_by_vendor.step"), result.note)
        self.assertEqual(result.pin, "match", "the archive hash matches and the member pin is only judged for the named member")

    def test_an_archive_without_cad_is_refused(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("LICENSE.txt", b"x")
        archive = buf.getvalue()
        self.pin_pi5(archive)
        result = self.fetch("rpi5", pi5_routes(archive))
        self.assertEqual(result.status, "failed")
        self.assertIn("no STEP", result.note)

    def test_a_declined_download_makes_no_request_and_still_names_the_source(self):
        net = Net({})
        with mock.patch.object(requests, "get", net):
            result = fetch.fetch(catalog.get("rpi5"), confirm=self.confirm(False), dest_dir=self.dest, reason="r", allow_heavy=True)
        self.assertEqual((result.status, result.trust, net.calls), ("declined", "official", []))
        self.assertTrue(result.has_board)


class RedirectGuardTest(CatalogCopy):
    def test_a_redirect_off_the_allowlist_is_refused(self):
        routes = {PI5_URL: FakeResponse(status=302, url=PI5_URL, headers={"Location": "https://evil.example.com/pi5.zip"})}
        result = self.fetch("rpi5", routes)
        self.assertEqual(result.status, "failed")
        self.assertIn("not on the allowlist", result.note)
        self.assertEqual(self.net.calls, [PI5_URL])

    def test_a_redirect_to_an_internal_address_is_refused_before_any_request(self):
        for target in ("http://169.254.169.254/latest/meta-data", "http://127.0.0.1:8080/pi5.zip", "https://192.168.1.10/pi5.zip",
                       "http://localhost/pi5.zip", "http://2130706433/pi5.zip"):
            fetch.forget_all()
            routes = {PI5_URL: FakeResponse(status=302, url=PI5_URL, headers={"Location": target})}
            result = self.fetch("rpi5", routes)
            self.assertEqual(result.status, "failed", target)
            self.assertEqual(self.net.calls, [PI5_URL], target)
            self.assertTrue("internal" in result.note or "private" in result.note or "loopback" in result.note or "metadata" in result.note
                            or "allowlist" in result.note, result.note)

    def test_the_download_url_is_never_a_non_public_host_even_if_the_catalog_said_so(self):
        self.entry("rpi5")["step_urls"][0]["url"] = "https://pip.raspberrypi.com/documents/RP-010083-CA"
        self.cat["allow_hosts"].append("127.0.0.1")
        self.entry("rpi5")["step_urls"][0]["url"] = "https://127.0.0.1/pi5.zip"
        result = self.fetch("rpi5", {})
        self.assertEqual(result.status, "failed")
        self.assertEqual(self.net.calls, [])

    def test_too_many_redirects_stop(self):
        routes = {PI5_URL: FakeResponse(status=302, url=PI5_URL, headers={"Location": PI5_URL})}
        result = self.fetch("rpi5", routes)
        self.assertEqual(result.status, "failed")
        self.assertIn("redirects", result.note)
        self.assertLessEqual(len(self.net.calls), 7)


class CommunityMirrorTest(CatalogCopy):
    """The offline mirror (``OSCAR_REFPART_LOCAL_DIR``) holds the Pi-4-like fixture; the shipped Pi 4 item pins the real community file."""

    def setUp(self):
        super().setUp()
        os.environ[fetch.ENV_LOCAL_DIR] = os.path.join(FIXTURES, "mirror")

    def test_the_fixture_is_a_changed_file_that_degrades_to_a_warning(self):
        result = fetch.fetch(catalog.get("rpi4b"), dest_dir=self.dest)
        self.assertEqual((result.status, result.origin, result.pin), ("obtained", "local_mirror", "mismatch"), result.note)
        self.assertEqual((result.trust, result.trust_label), ("community", "community, unverified accuracy, file changed since verified"))
        self.assertIn("WARNING:", result.note)
        self.assertEqual(result.expected_bbox, [88.99, 58.4, 19.9])

    def test_a_matching_pin_is_a_match(self):
        self.entry("rpi4b")["step_urls"][0]["expected_sha256"] = sha(STEP)
        result = fetch.fetch(catalog.get("rpi4b"), dest_dir=self.dest)
        self.assertEqual((result.status, result.pin, result.warnings), ("obtained", "match", []))
        self.assertEqual(result.trust_label, "community, unverified accuracy")

    def test_the_strict_policy_refuses_the_mirror_file(self):
        self.entry("rpi4b")["step_urls"][0]["on_hash_mismatch"] = "reject"
        result = fetch.fetch(catalog.get("rpi4b"), dest_dir=self.dest)
        self.assertEqual(result.status, "failed")
        self.assertIn("hash mismatch", result.note)
        self.assertEqual(fetch.obtained_file("rpi4b"), {})

    def test_a_part_without_a_source_has_no_trust(self):
        os.environ[fetch.ENV_LOCAL_DIR] = os.path.join(FIXTURES, "mirror")
        entry = catalog.get("rpi4b")
        entry["step_urls"] = []
        result = fetch.fetch(entry, dest_dir=self.dest)
        self.assertEqual((result.status, result.trust, result.trust_label, result.pin), ("obtained", "", "", ""))


class DirectDownloadTest(CatalogCopy):
    def test_the_community_file_downloads_and_matches_its_pin(self):
        self.entry("rpi4b")["step_urls"][0].update(expected_sha256=sha(STEP), size=len(STEP))
        net_routes = {PI4_URL: FakeResponse(STEP, url=PI4_URL)}
        result = self.fetch("rpi4b", net_routes)
        self.assertEqual((result.status, result.pin, result.trust), ("obtained", "match", "community"), result.note)
        self.assertEqual(self.net.calls, [PI4_URL])
        self.assertIn("unspecified - community model", result.license)
        self.assertIn("Source: community model", self.asked[0][0])

    def test_a_changed_community_file_is_a_warning_and_html_is_never_accepted(self):
        result = self.fetch("rpi4b", {PI4_URL: FakeResponse(CHANGED, url=PI4_URL)})
        self.assertEqual((result.status, result.pin), ("obtained", "mismatch"))
        fetch.forget_all()
        html = b"<!DOCTYPE html><html><body>Not found</body></html>"
        failed = self.fetch("rpi4b", {PI4_URL: FakeResponse(html, url=PI4_URL)})
        self.assertEqual(failed.status, "failed", "a vendor page instead of a file never passes, pin policy or not")


if __name__ == "__main__":
    unittest.main()
