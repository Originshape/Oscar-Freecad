"""The four reference-part tools (oscar/tools_ref.py): registration against the matrix, their schemas, and each
handler through ``tools.run_tool`` with a real Project, a real Ledger and real FreeCAD documents under freecadcmd.

The network is a fake ``requests.get`` (through the real ``netguard.download_verified``); the permission dialog is a patched
``netguard.request_permission``; the catalog is patched to list a STEP URL where a download is the point of the test (the shipped
catalog lists none). The Pi-4-like STEP fixtures come from ``tests/fixtures/refparts``.
"""

import copy
import hashlib
import json
import os
import re
import shutil
import sys
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402
import requests  # noqa: E402

from oscar import netguard, tools, tools_ref  # noqa: E402
from oscar.flow import Ledger  # noqa: E402
from oscar.project import ProjectError  # noqa: E402
from oscar.refparts import catalog, fetch  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402
from support.fakes import temp_project  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "refparts")
STEP_PATH = os.path.join(FIXTURES, "pi4_like.step")
OFFSET_PATH = os.path.join(FIXTURES, "pi4_like_offset.step")
MIRROR = os.path.join(FIXTURES, "mirror")
STEP = open(STEP_PATH, "rb").read()
STEP_SHA = hashlib.sha256(STEP).hexdigest()
HTML = b"<!DOCTYPE html><html><body>Sign in</body></html>"
URL = "https://raw.githubusercontent.com/example/pi/main/pi4.step"
REQUEST = "Design a snap-fit enclosure for the Raspberry Pi 4 Model B with port cutouts, vent slots and a snap-fit lid in ABS."
ALL_PHASES = ("INTAKE", "SOURCE", "PLAN", "PARAMS", "BUILD", "VERIFY", "DELIVER")


class FakeResponse:
    def __init__(self, body=b"", *, status=200, url=URL):
        self.status_code, self.url, self._body = status, url, body
        self.headers = {"Content-Length": str(len(body))}

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


class Base(unittest.TestCase):
    """Isolated catalog / registries / env, a real Project, a Ledger for the Pi 4 request and a context for ``run_tool``."""

    listed_url = None            # when set, the catalog lists this STEP URL for the Pi 4
    ship_steps = False           # True keeps the STEP sources the shipped catalog lists (Pi 4 community, Pi 5 official); the default
                                 # strips them so the nominal / mirror tests keep describing "a part without a STEP source"
    mirror = False               # OSCAR_REFPART_LOCAL_DIR points at the fixture mirror
    with_ledger = True
    with_project = True

    def setUp(self):
        fetch.forget_all()
        netguard.reset_turn()
        cat = copy.deepcopy(catalog._catalog())
        if not self.ship_steps:
            for entry in cat["entries"]:
                entry["step_urls"] = []
        if self.listed_url:
            next(e for e in cat["entries"] if e["id"] == "rpi4b")["step_urls"] = [
                {"url": self.listed_url, "mirror": False, "expected_sha256": "", "expected_bbox": [], "license": "test licence"}]
        self.cat = cat
        patcher = mock.patch.object(catalog, "_catalog", return_value=cat)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(fetch.ENV_LOCAL_DIR, None)
        os.environ.pop("OSCAR_ALLOW_NET", None)
        if self.mirror:
            os.environ[fetch.ENV_LOCAL_DIR] = MIRROR
        self.permission = mock.patch.object(netguard, "request_permission", return_value="deny")
        self.ask = self.permission.start()
        self.addCleanup(self.permission.stop)
        self.net = mock.patch.object(requests, "get", side_effect=AssertionError("unexpected network access"))
        self.net_mock = self.net.start()
        self.addCleanup(self.net.stop)
        self.dl = isolation.fresh_dir("tr_dl_")
        dl = mock.patch.dict(os.environ, {"OSCAR_DOWNLOAD_DIR": self.dl})
        dl.start()
        self.addCleanup(dl.stop)
        self.project = temp_project(title="Pi 4 case") if self.with_project else None
        self.ledger = Ledger.from_request(REQUEST) if self.with_ledger else None
        self.turn = "T" + uuid.uuid4().hex[:8]
        self.docs = []
        self.addCleanup(self.close_docs)

    def close_docs(self):
        for name in self.docs:
            if name in App.listDocuments():
                App.closeDocument(name)

    def new_doc(self, prefix="PiCase"):
        doc = App.newDocument(f"{prefix}{uuid.uuid4().hex[:5]}")
        self.docs.append(doc.Name)
        return doc

    def ctx(self, **kw):
        values = dict(project=self.project, ledger=self.ledger, turn_id=self.turn)
        values.update(kw)
        return ToolContext(**values)

    def call(self, tool_name, ctx=None, **args):
        """``(text, is_error)`` of ``tools.run_tool`` under the test context (``self.outcome`` keeps the last ToolOutcome)."""
        ctx = ctx or self.ctx()
        with use_context(ctx):
            text, _img, is_error = tools.run_tool(tool_name, args)
        self.outcome = ctx.outcome
        return text, is_error

    def ok(self, tool_name, **args):
        text, is_error = self.call(tool_name, **args)
        self.assertFalse(is_error, text)
        return text

    def fails(self, tool_name, *fragments, **args):
        text, is_error = self.call(tool_name, **args)
        self.assertTrue(is_error, text)
        for fragment in fragments:
            self.assertIn(fragment, text)
        return text

    def allow_download(self, body=STEP, status=200, url=URL):
        self.ask.return_value = "allow"
        self.net_mock.side_effect = lambda u, **kw: FakeResponse(body, status=status, url=url)

    def entries(self):
        return self.project.read_provenance()["entries"]


class ScopedConfirmTest(unittest.TestCase):
    """Review finding (PR #3): the grant host must come from the structured URL; with no host at all nothing is asked (never an
    unscoped 'Always allow')."""

    def ask(self, summary, detail="d", url=""):
        from oscar import netguard

        asked = {}

        def fake(summary, detail="", *, tool="", host=""):
            asked.update(tool=tool, host=host)
            return "allow"

        with mock.patch.object(netguard, "request_permission", side_effect=fake):
            return tools_ref._confirm_scoped(summary, detail, url), asked

    def test_the_structured_url_scopes_the_grant(self):
        ok, asked = self.ask("anything without a link", url="https://Pip.RaspberryPi.com/documents/RP-010083-CA")
        self.assertTrue(ok)
        self.assertEqual(asked, {"tool": "source_reference_part", "host": "pip.raspberrypi.com"})

    def test_an_older_caller_falls_back_to_the_url_in_the_text(self):
        ok, asked = self.ask("Oscar wants to download:\n\u2022 https://example.com/a.step\n\u2022 Part: x")
        self.assertTrue(ok)
        self.assertEqual(asked["host"], "example.com")

    def test_no_host_means_no_question_and_no_grant(self):
        from oscar import netguard

        with mock.patch.object(netguard, "request_permission") as ask:
            self.assertFalse(tools_ref._confirm_scoped("no link here", "d", ""))
        ask.assert_not_called()


class RegistrationTest(unittest.TestCase):
    def meta(self, name):
        entry = tools.tool_meta(name)
        self.assertIsNotNone(entry, f"{name} is not registered")
        return entry

    def test_matrix_flags_of_6_3(self):
        expected = {
            "find_reference_part": ("source", False, False, ("INTAKE", "SOURCE", "PLAN")),
            "source_reference_part": ("source", True, False, ("INTAKE", "SOURCE", "PLAN")),
            "import_reference_part": ("source", True, True, ("SOURCE",)),
            "measure_reference_part": ("source", True, False, ("SOURCE", "PARAMS")),
        }
        for name, (category, mutating, geometry, phases) in expected.items():
            entry = self.meta(name)
            self.assertEqual((entry.category, entry.mutating, entry.geometry, tuple(entry.phases)), (category, mutating, geometry, phases), name)

    def test_the_two_lite_tools(self):
        for name in ("find_reference_part", "source_reference_part"):
            self.assertIn(name, tools.LITE_TOOLS)
        self.assertNotIn("import_reference_part", tools.LITE_TOOLS)

    def test_schemas_follow_6_6(self):
        find = self.meta("find_reference_part").parameters
        self.assertEqual((find["required"], sorted(find["properties"])), (["query"], ["limit", "query"]))
        self.assertEqual({k: find["properties"]["limit"][k] for k in ("type", "default", "minimum", "maximum")},
                         {"type": "integer", "default": 5, "minimum": 1, "maximum": 10})
        source = self.meta("source_reference_part").parameters
        self.assertEqual((source["required"], sorted(source["properties"])), (["reason"], ["allow_download", "allow_heavy", "id", "reason", "url"]))
        self.assertEqual(source["properties"]["allow_download"], {"type": "boolean", "default": True})
        self.assertEqual({k: source["properties"]["allow_heavy"][k] for k in ("type", "default")}, {"type": "boolean", "default": False})
        imp = self.meta("import_reference_part").parameters
        self.assertEqual((imp["required"], sorted(imp["properties"])), (["doc_name"], ["doc_name", "id", "name", "path"]))
        measure = self.meta("measure_reference_part").parameters
        self.assertEqual((measure["required"], sorted(measure["properties"])), (["doc_name"], ["doc_name", "object", "top_n", "write_params"]))
        self.assertEqual(measure["properties"]["write_params"], {"type": "boolean", "default": False})
        self.assertEqual({k: measure["properties"]["top_n"][k] for k in ("default", "minimum", "maximum")}, {"default": 8, "minimum": 1, "maximum": 30})

    def test_descriptions_say_the_things_the_model_must_know(self):
        self.assertIn("Never invent a URL", self.meta("source_reference_part").description)
        self.assertIn("at most one", self.meta("source_reference_part").description)
        self.assertIn("nominal", self.meta("source_reference_part").description)
        self.assertIn("hidden App::Part named Ref_<ID>", self.meta("import_reference_part").description)
        self.assertIn("step_measurement", self.meta("measure_reference_part").description)
        self.assertIn("needs no network", self.meta("find_reference_part").description)

    def test_schemas_use_no_union_keywords(self):
        for name in ("find_reference_part", "source_reference_part", "import_reference_part", "measure_reference_part"):
            text = json.dumps(self.meta(name).parameters)
            for banned in ("oneOf", "anyOf", "allOf", "$ref"):
                self.assertNotIn(banned, text, name)

    def test_mutating_tools_get_the_universal_step_title(self):
        specs = {s.name: s for s in tools.specs(names=["find_reference_part", "source_reference_part", "import_reference_part", "measure_reference_part"])}
        self.assertNotIn("step_title", specs["find_reference_part"].parameters["properties"])
        for name in ("source_reference_part", "import_reference_part", "measure_reference_part"):
            self.assertIn("step_title", specs[name].parameters["properties"], name)

    def test_row_titles(self):
        self.assertEqual(tools.derive_title("source_reference_part", {"id": "rpi4b"}), "Source reference rpi4b")
        self.assertEqual(tools.derive_title("import_reference_part", {"path": "/tmp/pi.step"}), "Import reference pi.step")
        self.assertEqual(tools.derive_title("measure_reference_part", {"object": "Ref_Pi"}), "Measure reference Ref_Pi")

    def test_visible_in_the_phases_of_the_matrix(self):
        for phase, visible in (("SOURCE", 4), ("INTAKE", 2), ("PLAN", 2), ("PARAMS", 1), ("BUILD", 0)):
            names = {s.name for s in tools.specs(phases=[phase]) if s.name.endswith("_reference_part")}
            self.assertEqual(len(names), visible, (phase, names))


class FindTest(Base):
    def test_pi4_query(self):
        text = self.ok("find_reference_part", query="raspberry pi 4 model b")
        self.assertIn("1 catalog match for 'raspberry pi 4 model b':", text)
        self.assertIn("1. rpi4b - Raspberry Pi 4 Model B [sbc]:", text)
        self.assertIn("board data (9 connectors; 2 datasheet, 8 community, 1 approximate, 1 assumed)", text)
        self.assertIn("STEP download: none listed (nominal board data only)", text)
        self.assertIn("NOT verified against the drawing", text)
        self.assertIn("datasheets.raspberrypi.com", text)
        self.assertIn('source_reference_part(id="rpi4b"', text)
        self.assertEqual(self.net_mock.call_count, 0)

    def test_aliases_and_a_whole_request(self):
        for query in ("pi4", "RPi 4", "Raspberry Pi 4B", "a case for the raspberry pi 4 model b please"):
            self.assertIn("1. rpi4b", self.ok("find_reference_part", query=query), query)

    def test_a_listed_step_url_is_counted(self):
        next(e for e in catalog._catalog()["entries"] if e["id"] == "rpi4b")["step_urls"] = [
            {"url": URL, "mirror": False, "expected_sha256": "", "expected_bbox": [], "license": "x"}]
        self.assertIn("STEP download: 1 source listed", self.ok("find_reference_part", query="pi 4"))

    def test_several_matches_and_the_limit(self):
        text = self.ok("find_reference_part", query="raspberry pi", limit=2)
        lines = [l for l in text.splitlines() if re.match(r"^\d+\. ", l)]
        self.assertEqual(len(lines), 2)
        self.assertIn("(showing 2)", text)
        self.assertEqual(len([l for l in self.ok("find_reference_part", query="raspberry pi", limit=10).splitlines() if re.match(r"^\d+\. ", l)]),
                         len(catalog.match("raspberry pi")))

    def test_no_match_lists_the_catalog_and_the_ways_forward(self):
        text = self.ok("find_reference_part", query="nema 17 stepper motor")
        self.assertIn("No catalog entry matches 'nema 17 stepper motor'", text)
        self.assertIn("rpi4b (Raspberry Pi 4 Model B)", text)
        self.assertIn("attach its STEP or STL file", text)
        self.assertIn("never invent a URL", text)

    def test_empty_query_and_bad_limits(self):
        self.fails("find_reference_part", "'query' is empty", query="   ")
        self.fails("find_reference_part", "missing required parameter", **{})
        self.fails("find_reference_part", "'limit' must be between 1 and 10", query="pi", limit=11)
        self.fails("find_reference_part", "'limit' must be between 1 and 10", query="pi", limit=0)

    def test_it_is_not_a_step_record(self):
        self.ok("find_reference_part", query="pi 4")
        self.assertEqual(self.project.read_steps(), [])
        self.assertFalse(self.outcome.mutating)


class SourceObtainedTest(Base):
    mirror = True

    def test_result_block_follows_the_contract(self):
        text = self.ok("source_reference_part", id="rpi4b", reason="reference geometry for the housing")
        lines = text.splitlines()
        self.assertEqual(lines[0], "SOURCE RESULT: status=obtained")
        self.assertEqual(lines[1], "reference_status: none")
        self.assertEqual(lines[2], "reference: rpi4b (Raspberry Pi 4 Model B)")
        self.assertRegex(lines[3], r"^file: .*/refs/408a9ff6c848_rpi4b\.step \(sha256 [0-9a-f]{64}, 77\.8 KB, step\)$")
        self.assertEqual(lines[4], "provenance: E1")
        self.assertTrue(lines[5].startswith("note: STEP file taken from the local mirror"))
        self.assertIn('import_reference_part(doc_name=..., id="rpi4b")', lines[5])
        self.assertEqual(len(lines), 6)

    def test_the_file_is_copied_into_the_project_and_hashed(self):
        text = self.ok("source_reference_part", id="rpi4b", reason="r")
        path = re.search(r"^file: (\S+) ", text, re.M).group(1)
        self.assertTrue(path.startswith(self.project.path("refs")))
        self.assertEqual(open(path, "rb").read(), STEP)
        entry = self.entries()[0]
        self.assertEqual((entry["kind"], entry["status"], entry["detected_type"], entry["sha256"], entry["bytes"]), ("download", "obtained", "step", STEP_SHA, len(STEP)))
        self.assertEqual((entry["local_path"], entry["confidence"], entry["reference_status"]), ("refs/408a9ff6c848_rpi4b.step", "community", ""))
        self.assertTrue(entry["source"].startswith("local-mirror:rpi4b"))

    def test_the_project_reference_is_named_but_the_status_stays_none(self):
        self.ok("source_reference_part", id="rpi4b", reason="r")
        ref = self.project.read_provenance()["reference"]
        self.assertEqual((ref["id"], ref["name"], ref["status"], ref["entry"]), ("rpi4b", "Raspberry Pi 4 Model B", "none", "E1"))
        self.assertEqual(self.project.reference_status, "none")

    def test_the_hand_over_to_the_ledger(self):
        self.ok("source_reference_part", id="rpi4b", reason="r")
        result = self.outcome.extra["reference_result"]
        self.assertEqual((result["status"], result["reference_status"], result["id"], result["sha256"], result["origin"]), ("obtained", "none", "rpi4b", STEP_SHA, "local_mirror"))
        self.assertTrue(result["has_board"])
        self.assertEqual(self.ledger.step("S2").status, "done")
        self.assertIn(f"sha256 {STEP_SHA[:12]}", self.ledger.step("S2").note)
        self.assertEqual(self.ledger.reference_status, "none")
        self.assertEqual(self.ledger.step("S3").status, "pending")

    def test_no_dialog_and_no_network_for_a_local_file(self):
        self.ok("source_reference_part", id="rpi4b", reason="r")
        self.assertEqual((self.ask.call_count, self.net_mock.call_count), (0, 0))

    def test_it_is_one_step_record(self):
        self.ok("source_reference_part", id="rpi4b", reason="r")
        steps = self.project.read_steps()
        self.assertEqual([(s.tool, s.ok) for s in steps], [("source_reference_part", True)])
        self.assertEqual(steps[0].replay["kind"], "none", "downloads are not replayed")

    def test_the_id_comes_from_the_request_when_omitted(self):
        text = self.ok("source_reference_part", reason="source the part named in the request")
        self.assertIn("reference: rpi4b (Raspberry Pi 4 Model B)", text)

    def test_ids_are_case_insensitive(self):
        self.assertIn("status=obtained", self.ok("source_reference_part", id="RPI4B", reason="r"))

    def test_a_second_call_is_harmless(self):
        self.ok("source_reference_part", id="rpi4b", reason="r")
        again = self.ok("source_reference_part", id="rpi4b", reason="again")
        self.assertIn("status=obtained", again)
        self.assertIn("provenance: E1", again)
        self.assertEqual(len(self.entries()), 1, "an identical repeat is not recorded twice")
        self.assertEqual(len(os.listdir(self.project.path("refs"))), 1, "the same bytes are stored once")

    def test_without_a_project_it_still_works(self):
        text, err = self.call("source_reference_part", ctx=ToolContext(), id="rpi4b", reason="r")
        self.assertFalse(err, text)
        self.assertIn("provenance: none (no project is bound to this chat", text)
        self.assertRegex(text, r"file: .*rpi4b\.step \(sha256")

    def test_a_failing_project_never_fails_the_tool(self):
        with mock.patch.object(self.project, "provenance_add", side_effect=ProjectError("disk full")):
            text = self.ok("source_reference_part", id="rpi4b", reason="r")
        self.assertIn("status=obtained", text)
        self.assertIn("provenance: none (disk full)", text)
        with mock.patch.object(self.project, "provenance_add", side_effect=RuntimeError("boom")):
            self.assertIn("provenance: none (RuntimeError: boom)", self.ok("source_reference_part", id="rpi4b", reason="r"))


class SourceNominalTest(Base):
    def test_nominal_result_block_and_provenance(self):
        text = self.ok("source_reference_part", id="rpi4b", reason="reference geometry for the housing")
        lines = text.splitlines()
        self.assertEqual(lines[:5], ["SOURCE RESULT: status=nominal", "reference_status: catalog-nominal", "reference: rpi4b (Raspberry Pi 4 Model B)",
                                     "file: none", "provenance: E1"])
        self.assertIn("skip the import and measurement", lines[5])
        entry = self.entries()[0]
        self.assertEqual((entry["kind"], entry["status"], entry["confidence"], entry["reference_status"], entry["local_path"], entry["sha256"]),
                         ("reference", "nominal", "datasheet", "catalog-nominal", "", ""))
        self.assertIn("Board outline 85 x 56 mm, mounting holes 58 x 49 mm at (3.5, 3.5), connector positions", entry["claim"])
        self.assertTrue(entry["source"].startswith("catalog:rpi4b board.json v1"))
        self.assertEqual(entry["url"], "https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-mechanical-drawing.pdf")
        self.assertIn("No STEP file was obtained; nominal published dimensions are used.", entry["note"])
        self.assertIn("Weakest board data: assumed", entry["note"])
        self.assertEqual(self.project.reference_status, "catalog-nominal")
        self.assertEqual(self.net_mock.call_count, 0)

    def test_the_ledger_skips_the_import_step(self):
        self.ok("source_reference_part", id="rpi4b", reason="r")
        self.assertEqual(self.ledger.reference_status, "catalog-nominal")
        self.assertEqual((self.ledger.step("S2").status, self.ledger.step("S3").status), ("done", "skipped"))
        self.assertEqual(self.outcome.extra["reference_result"]["status"], "nominal")

    def test_other_boards_are_nominal_too(self):
        for ident in ("rpi5", "arduino_uno", "esp32_devkitc", "jetson_nano"):
            text = self.ok("source_reference_part", id=ident, reason="r")
            self.assertIn("status=nominal", text, ident)
            self.assertIn("reference_status: catalog-nominal", text, ident)

    def test_repeating_the_call_does_not_repeat_the_provenance_entry(self):
        self.ok("source_reference_part", id="rpi4b", reason="r")
        again = self.ok("source_reference_part", id="rpi4b", reason="r again")
        self.assertIn("provenance: E1", again)
        self.assertEqual(len(self.entries()), 1)
        self.ok("source_reference_part", id="rpi5", reason="another board")
        self.assertEqual(len(self.entries()), 2, "a different part is a new entry")

    def test_a_stronger_status_is_never_downgraded(self):
        self.project.set_reference("rpi4b", "Raspberry Pi 4 Model B", "step-measured", "")
        text = self.ok("source_reference_part", id="rpi4b", reason="r")
        self.assertIn("reference_status: step-measured", text)
        self.assertEqual(self.project.reference_status, "step-measured")


class SourceDownloadTest(Base):
    listed_url = URL

    def test_the_user_is_asked_with_the_url_part_and_reason_then_the_file_is_obtained(self):
        self.allow_download()
        text = self.ok("source_reference_part", id="rpi4b", reason="so the case fits the real board")
        self.assertIn("status=obtained", text)
        summary = self.ask.call_args.args[0]
        detail = self.ask.call_args.kwargs["detail"]
        self.assertIn(URL, summary)
        self.assertIn("Raspberry Pi 4 Model B", summary)
        self.assertIn("so the case fits the real board", detail)
        self.assertEqual(self.net_mock.call_count, 1)
        self.assertEqual(os.listdir(self.dl), ["pi4.step"])
        entry = self.entries()[0]
        self.assertEqual((entry["url"], entry["sha256"], entry["status"]), (URL, STEP_SHA, "obtained"))
        self.assertEqual(entry["source"], "catalog:rpi4b download from raw.githubusercontent.com")
        self.assertIn("test licence", entry["note"])
        self.assertEqual(self.ledger.step("S2").status, "done")

    def test_always_allow_is_honoured(self):
        self.allow_download()
        self.ask.return_value = "always"
        self.assertIn("status=obtained", self.ok("source_reference_part", id="rpi4b", reason="r"))

    def test_declined_falls_back_to_the_nominal_data_and_records_it(self):
        text = self.ok("source_reference_part", id="rpi4b", reason="r")
        lines = text.splitlines()
        self.assertEqual(lines[:2], ["SOURCE RESULT: status=declined", "reference_status: catalog-nominal"])
        self.assertIn("file: none", text)
        self.assertIn("declined the download", text)
        self.assertIn("continue with the nominal data", text)
        entry = self.entries()[0]
        self.assertEqual((entry["status"], entry["reference_status"], entry["kind"]), ("declined", "catalog-nominal", "reference"))
        self.assertEqual(self.net_mock.call_count, 0)
        self.assertEqual(self.project.reference_status, "catalog-nominal")
        self.assertEqual((self.ledger.reference_status, self.ledger.step("S2").status, self.ledger.step("S3").status),
                         ("catalog-nominal", "done", "skipped"))
        self.assertIn("nominal data used", self.ledger.step("S3").note)

    def test_a_404_fails_and_falls_back(self):
        self.allow_download(b"", status=404)
        text = self.ok("source_reference_part", id="rpi4b", reason="r")
        self.assertEqual(text.splitlines()[:2], ["SOURCE RESULT: status=failed", "reference_status: catalog-nominal"])
        self.assertIn("404", text)
        self.assertEqual(self.entries()[0]["status"], "failed")
        self.assertEqual(os.listdir(self.dl), [])

    def test_an_html_login_page_is_never_taken_for_a_step_file(self):
        self.allow_download(HTML)
        text = self.ok("source_reference_part", id="rpi4b", reason="r")
        self.assertIn("status=failed", text)
        self.assertIn("web page", text)
        self.assertEqual(os.listdir(self.dl), [])
        self.assertEqual(fetch.obtained_file("rpi4b"), {})

    def test_one_attempt_per_turn_whatever_the_model_tries_next(self):
        self.allow_download(b"", status=500)
        self.ok("source_reference_part", id="rpi4b", reason="r")
        calls = self.net_mock.call_count
        again = self.ok("source_reference_part", id="rpi4b", reason="try again")
        self.assertIn("already attempted in this turn", again)
        self.assertEqual(len(self.entries()), 1, "the cached answer is not recorded a second time")
        self.assertIn("provenance: E1", again)
        other = self.ok("source_reference_part", url="https://raw.githubusercontent.com/x/y/main/other.step", reason="another URL")
        self.assertIn("only one download attempt", other)
        self.assertEqual(self.net_mock.call_count, calls, "no further request")

    def test_a_new_turn_may_try_again(self):
        self.allow_download(b"", status=500)
        self.ok("source_reference_part", id="rpi4b", reason="r")
        self.turn = "T" + uuid.uuid4().hex[:8]
        self.allow_download()
        self.assertIn("status=obtained", self.ok("source_reference_part", id="rpi4b", reason="r"))

    def test_allow_download_false_never_asks(self):
        text = self.ok("source_reference_part", id="rpi4b", reason="r", allow_download=False)
        self.assertIn("status=declined", text)
        self.assertEqual((self.ask.call_count, self.net_mock.call_count), (0, 0))

    def test_the_download_dir_follows_the_environment(self):
        self.allow_download()
        self.ok("source_reference_part", id="rpi4b", reason="r")
        self.assertTrue(os.path.isfile(os.path.join(self.dl, "pi4.step")))


class SourceUrlTest(Base):
    with_ledger = False

    def test_an_official_url_without_an_id(self):
        self.allow_download(url="https://raw.githubusercontent.com/o/r/main/my_board.step")
        text = self.ok("source_reference_part", url="https://raw.githubusercontent.com/o/r/main/my_board.step", reason="the vendor file")
        self.assertIn("status=obtained", text)
        self.assertIn("reference: my_board (my_board)", text)
        self.assertTrue(self.entries()[0]["source"].startswith("url:my_board"))
        self.assertEqual(self.entries()[0]["confidence"], "community")
        self.assertEqual(fetch.obtained_file("my_board")["sha256"], STEP_SHA)

    def test_a_url_off_the_allowlist_is_a_failure_without_dialog_or_request(self):
        text = self.ok("source_reference_part", url="https://grabcad.com/library/pi4.step", reason="r")
        self.assertIn("status=failed", text)
        self.assertIn("not on the allowlist", text)
        self.assertIn("reference_status: none", text, "no board data for an unknown part")
        self.assertEqual((self.ask.call_count, self.net_mock.call_count), (0, 0))
        self.assertEqual(self.outcome.extra["reference_result"]["has_board"], False)

    def test_the_same_url_with_a_known_id_falls_back_to_the_nominal_board(self):
        text = self.ok("source_reference_part", id="rpi4b", url="https://grabcad.com/library/pi4.step", reason="r")
        self.assertIn("status=failed", text)
        self.assertIn("reference_status: catalog-nominal", text)
        self.assertEqual(self.entries()[0]["status"], "failed")

    def test_a_declined_unknown_part(self):
        text = self.ok("source_reference_part", url="https://raw.githubusercontent.com/o/r/main/thing.step", reason="r")
        self.assertIn("status=declined", text)
        self.assertIn("reference_status: none", text)
        self.assertIn("Ask the user to attach the file", text)
        self.assertEqual(self.project.reference_status, "none")


class SourceLedgerResolutionTest(Base):
    def test_a_url_without_an_id_is_matched_to_the_part_of_the_request(self):
        self.allow_download(url="https://raw.githubusercontent.com/o/r/main/my_board.step")
        text = self.ok("source_reference_part", url="https://raw.githubusercontent.com/o/r/main/my_board.step", reason="r")
        self.assertIn("reference: rpi4b (Raspberry Pi 4 Model B)", text)
        self.assertIn("status=obtained", text)

    def test_a_refused_url_for_the_part_of_the_request_falls_back_to_the_nominal_board(self):
        text = self.ok("source_reference_part", url="https://grabcad.com/library/pi4.step", reason="r")
        self.assertIn("status=failed", text)
        self.assertIn("reference_status: catalog-nominal", text)
        self.assertEqual(self.ledger.reference_status, "catalog-nominal")
        self.assertEqual(self.ledger.step("S2").status, "done")

    def test_an_unknown_part_blocks_the_source_step_until_the_model_resolves_it(self):
        ledger = Ledger.from_request("Design an enclosure around my custom sensor module with a snap-fit lid.")
        self.assertTrue(ledger.needs_reference, ledger.reference_hint)
        text = self.call("source_reference_part", ctx=self.ctx(ledger=ledger), url="https://grabcad.com/library/x.step", reason="r")[0]
        self.assertIn("status=failed", text)
        self.assertIn("reference_status: none", text)
        self.assertEqual(ledger.step("S2").status, "blocked")
        self.assertEqual(ledger.reference_status, "none")


class SourceArgumentsTest(Base):
    def test_empty_reason_is_refused(self):
        self.fails("source_reference_part", "'reason' is empty", id="rpi4b", reason="  ")
        self.fails("source_reference_part", "missing required parameter", id="rpi4b")

    def test_unknown_id_names_the_catalog(self):
        self.fails("source_reference_part", "unknown reference part 'rpi99'", "rpi4b", "find_reference_part", id="rpi99", reason="r")

    def test_non_http_url(self):
        self.fails("source_reference_part", "must be an http(s) URL", url="ftp://x/y.step", reason="r")

    def test_nothing_named_and_nothing_in_the_request(self):
        ctx = self.ctx(ledger=Ledger.from_request("Design a bracket 40 mm wide"))
        text, err = self.call("source_reference_part", ctx=ctx, reason="r")
        self.assertTrue(err)
        self.assertIn("does not name a catalog part", text)
        self.assertIn("rpi4b (Raspberry Pi 4 Model B)", text)
        self.assertIn("never invent a URL", text)

    def test_without_a_ledger_nothing_can_be_resolved(self):
        text, err = self.call("source_reference_part", ctx=self.ctx(ledger=None), reason="r")
        self.assertTrue(err)
        self.assertIn("find_reference_part", text)

    def test_unexpected_parameters_are_rejected_by_the_registry(self):
        self.fails("source_reference_part", "unknown parameter", reason="r", mirror=True)


class ImportBase(Base):
    mirror = True

    def source(self):
        return self.ok("source_reference_part", id="rpi4b", reason="r")

    def imported(self, doc, **args):
        args.setdefault("id", "rpi4b")
        return self.ok("import_reference_part", doc_name=doc.Name, **args)


class ImportTest(ImportBase):
    def test_import_after_source_builds_the_hidden_reference(self):
        self.source()
        doc = self.new_doc()
        text = self.imported(doc)
        part = doc.getObject("Ref_RPI4B")
        self.assertEqual((part.TypeId, len(part.Group), part.Visibility), ("App::Part", 10, False))
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith(f"Imported Ref_RPI4B (hidden App::Part) in '{doc.Name}': 10 solids, overall bounding box 89.5 x 57 x 18.5 mm"))
        self.assertIn("Largest solids: PCB 85 x 56 x 1.4;", text)
        self.assertIn("Frame: already in the board frame", text)
        self.assertIn("Identity: footprint 85 x 56 mm is consistent with the Raspberry Pi 4 Model B outline", text)
        self.assertIn("reference: rpi4b (step-imported); provenance: E2", text)
        self.assertIn(f'Next: measure_reference_part(doc_name="{doc.Name}", write_params=true)', text)
        self.assertLess(len(text), 1500, "bounded summary")

    def test_it_is_a_geometry_step_with_health_and_a_delta(self):
        self.source()
        doc = self.new_doc()
        self.imported(doc)
        out = self.outcome
        self.assertTrue(out.geometry and out.mutating and out.handler_ok)
        self.assertTrue(out.ok, out.text)
        self.assertIn("Ref_RPI4B", out.delta["added"])
        self.assertIn("[health]", out.text)
        steps = self.project.read_steps()
        self.assertEqual([s.tool for s in steps], ["source_reference_part", "import_reference_part"])
        self.assertEqual(steps[1].replay["kind"], "tool")
        self.assertTrue(steps[1].checkpoint, "a geometry step is checkpointed")

    def test_provenance_status_and_ledger(self):
        self.source()
        doc = self.new_doc()
        self.imported(doc)
        entry = self.entries()[1]
        self.assertEqual((entry["kind"], entry["status"], entry["reference_status"], entry["sha256"], entry["detected_type"]),
                         ("reference", "obtained", "step-imported", STEP_SHA, "step"))
        self.assertIn("Imported rpi4b.step as Ref_RPI4B: 10 solids", entry["claim"])
        self.assertEqual(self.project.reference_status, "step-imported")
        self.assertEqual(self.ledger.reference_status, "step-imported")
        self.assertEqual(self.ledger.step("S3").stages_done, ["import"])

    def test_the_object_survives_the_checkpoint_restore_names(self):
        self.source()
        doc = self.new_doc()
        self.imported(doc)
        self.assertEqual(sorted(o.Label for o in doc.getObject("Ref_RPI4B").Group), sorted(["PCB"] + [c["id"] for c in catalog.load_board("rpi4b")["connectors"]]))

    def test_the_offset_file_is_normalized_by_the_tool(self):
        doc = self.new_doc()
        text = self.ok("import_reference_part", doc_name=doc.Name, path=OFFSET_PATH, id="rpi4b")
        self.assertIn("normalized into the board frame", text)
        self.assertIn("moved by 42.50, 28.00, 0.70 mm", text)
        self.assertIn("overall bounding box 89.5 x 57 x 18.5 mm (x -2.5..87, y -1..56, z -1.5..17)", text)

    def test_import_by_id_without_a_prior_source_call_uses_the_mirror(self):
        doc = self.new_doc()
        text = self.imported(doc)
        self.assertIn("Imported Ref_RPI4B", text)

    def test_import_by_id_in_a_fresh_process_uses_the_project_copy(self):
        self.source()
        os.environ.pop(fetch.ENV_LOCAL_DIR)
        fetch.forget_all()                                 # a resumed chat: only the project's provenance remembers the file
        doc = self.new_doc()
        self.assertIn("Imported Ref_RPI4B", self.imported(doc))

    def test_neither_id_nor_path_uses_the_latest_obtained_file(self):
        self.source()
        doc = self.new_doc()
        text = self.ok("import_reference_part", doc_name=doc.Name)
        self.assertIn("Imported Ref_RPI4B", text)

    def test_the_file_line_of_the_source_result_passed_back_as_path_is_not_an_attachment(self):
        text = self.source()
        path = re.search(r"^file: (\S+) ", text, re.M).group(1)
        doc = self.new_doc()
        self.ok("import_reference_part", doc_name=doc.Name, path=path)
        self.assertEqual(self.ledger.reference_status, "step-imported")
        self.assertEqual(self.project.reference_status, "step-imported")
        self.assertEqual(self.outcome.args.get("path"), "", "the recorded call is the id-based one")
        self.assertEqual(self.outcome.args.get("id"), "rpi4b")
        self.assertEqual(self.entries()[1]["kind"], "reference")

    def test_a_user_attachment_is_user_attached(self):
        folder = isolation.fresh_dir("att_")
        path = os.path.join(folder, "Pi4 board v2.step")
        shutil.copyfile(STEP_PATH, path)
        doc = self.new_doc()
        text = self.ok("import_reference_part", doc_name=doc.Name, path=path)
        self.assertIn("Imported Ref_PI4_BOARD_V2", text)
        self.assertIn("reference: pi4_board_v2 (user-attached)", text)
        entry = self.entries()[0]
        self.assertEqual((entry["kind"], entry["status"], entry["reference_status"], entry["source"]), ("attachment", "attached", "user-attached", "user attachment: Pi4 board v2.step"))
        self.assertRegex(entry["local_path"], rf"^refs/{STEP_SHA[:12]}_Pi4.board.v2\.step$")
        self.assertTrue(os.path.isfile(self.project.path(entry["local_path"])))
        self.assertEqual((self.ledger.reference_status, self.project.reference_status), ("user-attached", "user-attached"))
        self.assertEqual(self.outcome.args["path"], path, "the ledger reads the path")

    def test_an_attachment_with_a_catalog_id_gets_the_board_data_for_orientation(self):
        folder = isolation.fresh_dir("att_")
        path = os.path.join(folder, "pi.step")
        shutil.copyfile(OFFSET_PATH, path)
        doc = self.new_doc()
        text = self.ok("import_reference_part", doc_name=doc.Name, path=path, id="rpi4b")
        self.assertIn("Ref_RPI4B", text)
        self.assertIn("overall bounding box 89.5 x 57 x 18.5 mm (x -2.5..87", text)

    def test_a_custom_object_name(self):
        self.source()
        doc = self.new_doc()
        self.assertIn("Imported Ref_BOARD_A", self.imported(doc, name="Ref_BOARD_A"))
        self.assertIsNotNone(doc.getObject("Ref_BOARD_A"))
        self.assertIsNone(doc.getObject("Ref_RPI4B"))

    def test_the_wrong_part_is_reported_but_not_refused(self):
        folder = isolation.fresh_dir("att_")
        path = os.path.join(folder, "pico.step")
        shutil.copyfile(STEP_PATH, path)
        doc = self.new_doc()
        text = self.ok("import_reference_part", doc_name=doc.Name, path=path, id="rpi_pico")
        self.assertIn("WARNING identity", text)
        self.assertIn("probably a different part", text)
        self.assertIn("confirm", text)
        self.assertIsNotNone(doc.getObject("Ref_RPI_PICO"))
        self.assertIn("different part", self.entries()[0]["note"])

    def test_without_a_project_or_ledger(self):
        doc = self.new_doc()
        text, err = self.call("import_reference_part", ctx=ToolContext(), doc_name=doc.Name, id="rpi4b")
        self.assertFalse(err, text)
        self.assertIn("provenance: none (no project is bound", text)

    def test_a_project_failure_does_not_fail_the_import(self):
        self.source()
        doc = self.new_doc()
        with mock.patch.object(self.project, "provenance_add", side_effect=ProjectError("disk full")):
            text = self.imported(doc)
        self.assertIn("provenance: none (disk full)", text)
        self.assertIsNotNone(doc.getObject("Ref_RPI4B"))

    def test_stl_attachment(self):
        import Mesh
        import Part

        path = os.path.join(isolation.fresh_dir("att_"), "board.stl")
        Mesh.Mesh(Part.makeBox(85, 56, 1.4).tessellate(0.1)).write(path)
        doc = self.new_doc()
        text = self.ok("import_reference_part", doc_name=doc.Name, path=path)
        self.assertIn("Imported the mesh as Ref_BOARD", text)
        self.assertIn("12 triangles", text)
        self.assertIn("(a mesh is only bounded, not measured)", text)
        self.assertIn("mesh reference", text)
        self.assertEqual(self.entries()[0]["detected_type"], "stl")


class ImportErrorsTest(ImportBase):
    def test_document_not_open_lists_the_open_ones(self):
        self.new_doc("Other")
        self.source()
        self.fails("import_reference_part", "Document 'Nope' is not open", "Open documents:", doc_name="Nope", id="rpi4b")

    def test_empty_document_name(self):
        self.fails("import_reference_part", "missing required parameter", id="rpi4b")

    def test_no_file_for_the_id_says_what_to_do(self):
        os.environ.pop(fetch.ENV_LOCAL_DIR)
        doc = self.new_doc()
        text = self.fails("import_reference_part", "no file was obtained for 'rpi4b'", doc_name=doc.Name, id="rpi4b")
        self.assertIn("source_reference_part", text)
        self.assertIn("skip the import and measurement", text)
        self.assertIn("path=", text)

    def test_neither_id_nor_path_and_nothing_obtained(self):
        os.environ.pop(fetch.ENV_LOCAL_DIR)
        doc = self.new_doc()
        self.fails("import_reference_part", "neither 'id' nor 'path'", doc_name=doc.Name)

    def test_bad_paths(self):
        doc = self.new_doc()
        self.fails("import_reference_part", "must be absolute", doc_name=doc.Name, path="pi4.step")
        self.fails("import_reference_part", "file not found", doc_name=doc.Name, path="/no/such/pi4.step")
        folder = isolation.fresh_dir("bad_")
        html = os.path.join(folder, "login.step")
        open(html, "wb").write(HTML)
        self.fails("import_reference_part", "web page", doc_name=doc.Name, path=html)
        zipped = os.path.join(folder, "pi.zip")
        open(zipped, "wb").write(b"PK\x03\x04")
        self.fails("import_reference_part", "ZIP archive", "unzip", doc_name=doc.Name, path=zipped)
        doc_file = os.path.join(folder, "notes.docx")
        open(doc_file, "wb").write(b"x")
        self.fails("import_reference_part", "unsupported reference file type", doc_name=doc.Name, path=doc_file)
        self.assertEqual([o.Name for o in doc.Objects], [])

    def test_taken_name_and_invalid_name(self):
        self.source()
        doc = self.new_doc()
        self.imported(doc)
        before = sorted(o.Name for o in doc.Objects)
        self.fails("import_reference_part", "'Ref_RPI4B' already exists", "name='Ref_<other>'", doc_name=doc.Name, id="rpi4b")
        self.fails("import_reference_part", "invalid object name", doc_name=doc.Name, id="rpi4b", name="not valid!")
        self.assertEqual(sorted(o.Name for o in doc.Objects), before)

    def test_a_failed_handler_ticks_nothing_and_records_nothing_as_imported(self):
        self.source()
        doc = self.new_doc()
        self.imported(doc)
        provenance = len(self.entries())
        self.fails("import_reference_part", doc_name=doc.Name, id="rpi4b")
        self.assertEqual(len(self.entries()), provenance)
        self.assertEqual(self.project.reference_status, "step-imported")

    def test_unknown_parameter(self):
        doc = self.new_doc()
        self.fails("import_reference_part", "unknown parameter", doc_name=doc.Name, id="rpi4b", normalize=False)




class EndToEndTest(ImportBase):
    def test_the_golden_sequence_of_the_pi4_scenario(self):
        self.ok("find_reference_part", query="raspberry pi 4 model b")
        self.ok("source_reference_part", id="rpi4b", reason="reference geometry for the housing")
        doc = self.new_doc("PiCase")
        self.ok("import_reference_part", doc_name=doc.Name, id="rpi4b")
        self.ok("measure_reference_part", doc_name=doc.Name, write_params=True)
        self.assertEqual(self.ledger.reference_status, "step-measured")
        self.assertEqual([self.ledger.step(s).status for s in ("S2", "S3")], ["done", "done"])
        self.assertEqual(self.project.reference_status, "step-measured")
        prov = self.project.read_provenance()
        self.assertEqual([e["kind"] for e in prov["entries"]], ["download", "reference", "measurement"])
        self.assertEqual(prov["reference"], {"id": "rpi4b", "name": "Raspberry Pi 4 Model B", "status": "step-measured", "entry": "E3"})
        self.assertRegex(prov["entries"][0]["sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(os.path.isfile(self.project.path(prov["entries"][0]["local_path"])))
        self.assertEqual([s.tool for s in self.project.read_steps()], ["source_reference_part", "import_reference_part", "measure_reference_part"])
        self.assertEqual(doc.getObject("Params").Name, "Params")
        self.assertIsNone(doc.getObject("Params001"))

    def test_the_nominal_variant(self):
        os.environ.pop(fetch.ENV_LOCAL_DIR)
        self.ok("find_reference_part", query="raspberry pi 4 model b")
        text = self.ok("source_reference_part", id="rpi4b", reason="reference geometry for the housing")
        self.assertIn("status=nominal", text)
        self.assertEqual(self.ledger.reference_status, "catalog-nominal")
        self.assertEqual([self.ledger.step(s).status for s in ("S2", "S3")], ["done", "skipped"])

    def test_replay_style_calls_without_a_context_work(self):
        doc = self.new_doc()
        text, err = self.call("import_reference_part", ctx=ToolContext(), doc_name=doc.Name, id="rpi4b")
        self.assertFalse(err, text)
        text, err = self.call("measure_reference_part", ctx=ToolContext(), doc_name=doc.Name)
        self.assertFalse(err, text)
        self.assertIn("PCB: 85 x 56 x 1.4 mm", text)

    def test_an_offset_attachment_then_measure_matches_the_catalog_file(self):
        folder = isolation.fresh_dir("att_")
        path = os.path.join(folder, "pi.step")
        shutil.copyfile(OFFSET_PATH, path)
        doc = self.new_doc()
        self.ok("import_reference_part", doc_name=doc.Name, path=path, id="rpi4b")
        text = self.ok("measure_reference_part", doc_name=doc.Name, write_params=True)
        self.assertIn("usb_c_power -> usb_c_power", text)
        self.assertIn("-x offset 28 width 12 height 1.5 z -1.5..0 overhang 2.5 (microsd -> microsd)", text)


PI4_URL = ("https://raw.githubusercontent.com/multigamesystem/MGS-CAD-Files/main/STEP%20files%20with%20images/"
           "Raspberry%20Pi%204%20Model%20B%20v4.step")
PI5_URL = "https://pip.raspberrypi.com/documents/RP-010083-CA"
PI5_FINAL = "https://pip-assets.raspberrypi.com/categories/892-raspberry-pi-5/documents/RP-010083-CA-1-rpi-5%203D%20STEP%20-%20No%20Graphics%20small%20file.zip"
COMMUNITY = "community, unverified accuracy"


def _zip_of(member_bytes=STEP, member="rpi-5b_no_graphics.step"):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, member_bytes)
        zf.writestr("LICENSE.txt", b"MIT test licence")
    return buf.getvalue()


class ShippedSourcesToolTest(Base):
    """The two registered sources (Pi 4 community, Pi 5 official) through the tools: trust level in the result, the provenance and the Build Summary.

    The real files are not used: the network is a fake that serves the Pi-4-like fixture; the catalog copy pins that fixture's hash where a
    pin match is the point and keeps the real (different) pin where a vendor-side change is the point."""

    ship_steps = True

    def pin_pi4(self):
        step = self.cat_entry("rpi4b")["step_urls"][0]
        step.update(expected_sha256=STEP_SHA, size=len(STEP))

    def cat_entry(self, ident):
        return next(e for e in self.cat["entries"] if e["id"] == ident)

    def serve_pi4(self, body=STEP):
        self.ask.return_value = "allow"
        self.net_mock.side_effect = lambda u, **kw: FakeResponse(body, url=u)

    def serve_pi5(self, archive):
        self.ask.return_value = "allow"

        def route(u, **kw):
            if u == PI5_URL:
                r = FakeResponse(b"", status=302, url=u)
                r.headers["Location"] = PI5_FINAL
                return r
            self.assertEqual(u, PI5_FINAL)
            return FakeResponse(archive, url=u)
        self.net_mock.side_effect = route

    def reference_entry(self):
        data = self.project.read_provenance()
        return next(e for e in data["entries"] if e["id"] == data["reference"]["entry"])

    def test_find_names_the_trust_of_each_source(self):
        text = self.ok("find_reference_part", query="raspberry pi 4 model b")
        self.assertIn("STEP download: 1 source listed (community trust)", text)
        self.assertIn("STEP download: 1 source listed (official trust)", self.ok("find_reference_part", query="raspberry pi 5"))
        self.assertEqual(self.net_mock.call_count, 0)

    def test_the_permission_dialog_and_result_state_the_community_trust(self):
        self.pin_pi4()
        self.serve_pi4()
        text = self.ok("source_reference_part", id="rpi4b", reason="reference geometry for the housing")
        summary = self.ask.call_args[0][0]
        self.assertIn("Source: community model", summary)
        self.assertIn(f"Trust: {COMMUNITY}.", self.ask.call_args[1]["detail"])
        lines = text.splitlines()
        self.assertEqual(lines[0], "SOURCE RESULT: status=obtained")
        trust = next(l for l in lines if l.startswith("trust:"))
        self.assertEqual(trust, f"trust: {COMMUNITY}; licence: unspecified - community model, accuracy unverified: cross-checked against "
                                "the official drawing, see notes")
        self.assertLess(lines.index(trust), next(i for i, l in enumerate(lines) if l.startswith("note:")))
        entry = self.entries()[0]
        self.assertEqual(entry["confidence"], "community")
        self.assertIn(f"[{COMMUNITY}]", entry["claim"])
        extra = self.outcome.extra["reference_result"]
        self.assertEqual((extra["trust"], extra["trust_label"], extra["pin"]), ("community", COMMUNITY, "match"))
        self.assertEqual(self.net_mock.call_count, 1)

    def test_a_community_model_is_never_overstated_in_provenance_or_the_build_summary(self):
        self.pin_pi4()
        self.serve_pi4()
        self.ok("source_reference_part", id="rpi4b", reason="r")
        doc = self.new_doc()
        text = self.ok("import_reference_part", doc_name=doc.Name, id="rpi4b")
        self.assertIn(f"Trust: {COMMUNITY}; say so when you report dimensions", text)
        entry = self.reference_entry()
        self.assertEqual((entry["reference_status"], entry["confidence"]), ("step-imported", "community"))
        self.assertTrue(entry["source"].startswith(f"step-imported ({COMMUNITY}): import_reference_part:"), entry["source"])
        self.assertIn(f"[{COMMUNITY}]", entry["claim"])
        self.assertEqual(doc.getObject("Ref_RPI4B").OscarRefTrust, COMMUNITY)
        summary = self.project.build_summary(None, "answered", None).markdown
        self.assertIn(f"status **step-imported** - source: step-imported ({COMMUNITY})", summary)
        measured = self.ok("measure_reference_part", doc_name=doc.Name)
        self.assertIn("reference_status: step-measured", measured)
        entry = self.reference_entry()
        self.assertEqual(entry["reference_status"], "step-measured")
        self.assertTrue(entry["source"].startswith(f"step-measured ({COMMUNITY}): measure_reference_part on Ref_RPI4B"), entry["source"])
        self.assertIn(f"[{COMMUNITY}]", entry["claim"])
        summary = self.project.build_summary(None, "answered", None).markdown
        self.assertIn(f"status **step-measured** - source: step-measured ({COMMUNITY})", summary)

    def test_an_official_archive_is_obtained_through_the_redirect_and_stated_as_official(self):
        archive = _zip_of()
        step = self.cat_entry("rpi5")["step_urls"][0]
        step.update(expected_sha256=hashlib.sha256(archive).hexdigest(), size=len(archive), member_sha256=STEP_SHA, member_bytes=len(STEP))
        self.serve_pi5(archive)
        text = self.ok("source_reference_part", id="rpi5", reason="reference geometry for the housing")
        self.assertIn("status=obtained", text)
        self.assertIn("trust: official, vendor-published; licence: vendor-published, see vendor terms", text)
        self.assertRegex(text, r"file: .*rpi-5b_no_graphics\.step \(sha256 [0-9a-f]{64}")
        self.assertEqual([c.args[0] for c in self.net_mock.call_args_list], [PI5_URL, PI5_FINAL])
        entry = self.entries()[0]
        self.assertEqual(entry["confidence"], "datasheet")
        self.assertIn("pip-assets.raspberrypi.com", entry["source"])
        self.assertIn("[official, vendor-published]", entry["claim"])
        doc = self.new_doc()
        self.ok("import_reference_part", doc_name=doc.Name, id="rpi5")
        ref = self.reference_entry()
        self.assertTrue(ref["source"].startswith("step-imported (official, vendor-published): import_reference_part:"), ref["source"])
        self.assertEqual(ref["confidence"], "datasheet")

    def test_a_changed_vendor_file_whose_outline_still_matches_is_imported_with_a_warning(self):
        os.environ[fetch.ENV_LOCAL_DIR] = MIRROR            # the fixture is NOT the file the catalog pins: the hash differs
        text = self.ok("source_reference_part", id="rpi4b", reason="r")
        self.assertIn(f"trust: {COMMUNITY}, file changed since verified", text)
        self.assertIn("WARNING: the file has sha256", text)
        doc = self.new_doc()
        imported = self.ok("import_reference_part", doc_name=doc.Name, id="rpi4b")
        self.assertIn("vendor changed it", imported)
        self.assertIn(f"Trust: {COMMUNITY}, file changed since verified", imported)
        self.assertIsNotNone(doc.getObject("Ref_RPI4B"))
        entry = self.reference_entry()
        self.assertTrue(entry["source"].startswith(f"step-imported ({COMMUNITY}, file changed since verified):"), entry["source"])
        self.assertEqual(entry["confidence"], "community")
        self.assertIn("vendor changed it", entry["note"])
        self.assertEqual(doc.getObject("Ref_RPI4B").OscarRefTrust, f"{COMMUNITY}, file changed since verified")

    def test_a_changed_vendor_file_with_another_outline_is_refused_and_the_document_is_unchanged(self):
        os.environ[fetch.ENV_LOCAL_DIR] = MIRROR
        self.cat_entry("rpi4b")["step_urls"][0]["expected_bbox"] = [150.0, 120.0, 40.0]
        self.ok("source_reference_part", id="rpi4b", reason="r")
        doc = self.new_doc()
        before = [o.Name for o in doc.Objects]
        self.fails("import_reference_part", "was not imported", "geometry no longer matches", "does not match the expected 150 x 120 x 40",
                   doc_name=doc.Name, id="rpi4b")
        self.assertEqual([o.Name for o in doc.Objects], before)
        self.assertEqual(len(self.entries()), 1, "no provenance for an import that did not happen")
        self.assertEqual(self.project.reference_status, "none")

    def test_a_user_attachment_is_never_labelled_and_never_judged_by_the_catalog_pin(self):
        self.cat_entry("rpi4b")["step_urls"][0]["expected_bbox"] = [150.0, 120.0, 40.0]       # would refuse a changed catalog file
        doc = self.new_doc()
        text = self.ok("import_reference_part", doc_name=doc.Name, path=OFFSET_PATH, id="rpi4b")
        self.assertNotIn("Trust:", text)
        self.assertIn("(user-attached)", text)
        entry = self.reference_entry()
        self.assertEqual((entry["reference_status"], entry["source"].startswith("user attachment:")), ("user-attached", True))
        self.assertFalse(hasattr(doc.getObject("Ref_RPI4B"), "OscarRefTrust"))
        measured = self.ok("measure_reference_part", doc_name=doc.Name)
        self.assertNotIn(COMMUNITY, measured)
        self.assertNotIn(COMMUNITY, self.reference_entry()["source"])

    def test_a_big_step_gets_more_gui_time_but_never_more_than_twenty_minutes(self):
        from oscar import tools_ref

        self.assertEqual(tools_ref._import_timeout(0), 240.0)
        self.assertEqual(tools_ref._import_timeout(17_535_443), 240.0)            # the community Pi 4 file: 165 s of budget < the 240 s floor
        self.assertAlmostEqual(tools_ref._import_timeout(77_627_469), 60.0 + 6.0 * 77.627469, places=3)     # the official Pi 5 file
        self.assertGreater(tools_ref._import_timeout(77_627_469), 500.0)
        self.assertEqual(tools_ref._import_timeout(10**12), 1200.0)
        self.assertEqual(tools_ref._import_timeout(None), 240.0)
        self.assertEqual(tools_ref._import_timeout(-5), 240.0)
        os.environ[fetch.ENV_LOCAL_DIR] = MIRROR
        seen = []
        real = tools_ref._gui_call
        with mock.patch.object(tools_ref, "_gui_call", lambda fn, op, timeout=240.0: (seen.append((op, timeout)), real(fn, op, timeout))[1]):
            doc = self.new_doc()
            self.ok("import_reference_part", doc_name=doc.Name, id="rpi4b")
        self.assertEqual(seen, [("import_reference_part", 240.0)])


if __name__ == "__main__":
    unittest.main()
