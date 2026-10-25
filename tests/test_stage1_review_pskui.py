"""Review of projects, part sourcing, the plan builder, the enclosure kit and the UI: regression tests for the defects
the review found by adversarial probing. Each test names the defect it pins. Pure python except where a test says so."""

import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import project_support as S  # noqa: E402
from oscar import chats, paths  # noqa: E402
from oscar import panel_render as pr  # noqa: E402
from oscar import project as P  # noqa: E402
from oscar.kit import spec as kit_spec  # noqa: E402
from oscar.providers import Message, ToolCall  # noqa: E402
from oscar.refparts import catalog  # noqa: E402


def _read(path, mode="r"):
    """The whole content of a file (closed again)."""
    if "b" in mode:
        with open(path, mode) as f:
            return f.read()
    with open(path, mode, encoding="utf-8") as f:
        return f.read()


class FallbackRestoreTest(unittest.TestCase):
    """A chat whose project had to move to the fallback folder (an unwritable ~/Documents) must find that
    project again after a restart instead of starting a fresh manifest on top of it."""

    def setUp(self):
        self.blocker = S.new_temp_file(S.sandbox_dir(), "iamafile")
        self.default_base = os.path.join(self.blocker, "projects")          # makedirs under a FILE: always an OSError
        self.fallback = os.path.join(S.sandbox_dir(), "fallback")
        env = {k: v for k, v in os.environ.items() if k != "OSCAR_PROJECTS_DIR"}
        patches = [mock.patch.dict(os.environ, env, clear=True),
                   mock.patch.object(paths, "_default_projects_dir", return_value=self.default_base),
                   mock.patch.object(paths, "fallback_projects_dir", return_value=self.fallback)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(setattr, paths, "PROJECTS_FALLBACK", False)

    def _first_session(self):
        first = P.Project.for_chat(S.CHAT_ID, "Pi 4 housing", snapshot_fn=S.tiny_png_snapshot)
        first.begin_turn(S.TURN, "build", work=True)               # materializes: re-routes into the fallback
        S.record(first, geometry=False, doc="")
        first.set_reference("rpi4b", "Raspberry Pi 4 Model B", "catalog-nominal", "E1")
        return first

    def test_for_chat_finds_the_project_that_lives_in_the_fallback_folder(self):
        first = self._first_session()
        self.assertTrue(first.root.startswith(self.fallback), first.root)
        again = P.Project.for_chat(S.CHAT_ID, "Pi 4 housing", snapshot_fn=S.tiny_png_snapshot)
        self.assertEqual(again.root, first.root)
        self.assertEqual(again.manifest["reference"]["id"], "rpi4b")
        self.assertGreaterEqual(again.manifest["counters"]["step"], 1)

    def test_a_renamed_chat_is_still_found_in_the_fallback_folder(self):
        first = self._first_session()
        again = P.Project.for_chat(S.CHAT_ID, "A completely different title", snapshot_fn=S.tiny_png_snapshot)
        self.assertEqual(again.root, first.root)

    def test_rerouting_adopts_the_manifest_that_is_already_in_the_fallback_folder(self):
        first = self._first_session()
        # an object built for the unwritable default path (what the panel's probe-free for_chat gives when the
        # folder search found nothing): ensure_dir() re-routes into the fallback where project.json already exists
        stale = P.Project(os.path.join(self.default_base, os.path.basename(first.root)), chat_id=S.CHAT_ID,
                          title="Pi 4 housing", snapshot_fn=S.tiny_png_snapshot)
        stale.ensure_dir()
        self.assertEqual(stale.root, first.root)
        self.assertEqual(stale.manifest["reference"]["id"], "rpi4b")
        self.assertGreaterEqual(stale.manifest["counters"]["step"], 1)
        stale.note("assumption", "later note")                      # a write must not wipe the stored reference
        on_disk = json.loads(_read(os.path.join(first.root, "project.json")))
        self.assertEqual(on_disk["reference"]["id"], "rpi4b")
        self.assertIn("not writable", on_disk["storage_note"])


class LoneSurrogateTest(unittest.TestCase):
    """A model can emit a lone UTF-16 surrogate (``\\ud83d`` without its pair) in a tool argument. It must never
    stop NOTES.md, notes.json, provenance.json or any other project file from being written again."""

    BAD = json.loads('"x\\ud83d y"')

    def setUp(self):
        self.proj = S.make_project(create=True)

    def test_a_step_with_a_surrogate_in_its_title_and_arguments_does_not_freeze_notes_md(self):
        rec = S.record(self.proj, tool="create_object", geometry=False, title=self.BAD, args={"k": self.BAD},
                       text=self.BAD)
        self.assertEqual(rec.index, 1)
        self.assertTrue(os.path.isfile(self.proj.path("NOTES.md")))
        S.record(self.proj, tool="create_object", geometry=False, title="next", call_id="c2")
        notes = _read(self.proj.path("NOTES.md"))
        self.assertIn("next", notes, "NOTES.md must keep being regenerated after the bad step")
        self.assertEqual([s.index for s in self.proj.read_steps()], [1, 2])

    def test_note_and_provenance_accept_a_surrogate(self):
        self.assertEqual(self.proj.note("assumption", self.BAD), "N1")
        entry = self.proj.provenance_add(kind="reference", claim=self.BAD, source="catalog", status="nominal")
        self.assertEqual(entry, "E1")
        self.assertEqual(len(self.proj.read_notes()), 1)
        self.assertEqual(len(self.proj.read_provenance()["entries"]), 1)

    def test_json_files_round_trip_the_surrogate_and_text_files_replace_it(self):
        from oscar import project_util as U

        path = os.path.join(S.sandbox_dir(), "x.json")
        U._atomic_write_json(path, {"a": self.BAD, "b": "ü ☃"})
        self.assertEqual(json.loads(_read(path)), {"a": self.BAD, "b": "ü ☃"})
        text_path = os.path.join(S.sandbox_dir(), "x.md")
        U._atomic_write_text(text_path, self.BAD)
        self.assertEqual(_read(text_path), "x? y")

    def test_ordinary_unicode_is_still_written_as_is(self):
        from oscar import project_util as U

        path = os.path.join(S.sandbox_dir(), "u.json")
        U._atomic_write_json(path, {"t": "Gehäuse ☃"})
        self.assertIn("Gehäuse ☃", _read(path))


class ProvenanceInputTest(unittest.TestCase):
    """provenance_add raises ProjectError (never a bare ValueError) for bad input and bounds what it stores."""

    def setUp(self):
        self.proj = S.make_project(create=True)
        self.source = os.path.join(S.sandbox_dir(), "pi4.step")
        with open(self.source, "wb") as f:
            f.write(b"ISO-10303-21;\nDATA;\n" * 100)

    def add(self, **kw):
        base = dict(kind="reference", claim="c", source="catalog", status="nominal")
        base.update(kw)
        return self.proj.provenance_add(**base)

    def test_bytes_must_be_a_non_negative_integer(self):
        for bad in (float("nan"), float("inf"), "abc", -5, [1], {"a": 1}):
            with self.assertRaises(P.ProjectError, msg=repr(bad)):
                self.add(bytes=bad)
        self.assertEqual(self.add(bytes="12"), "E1")
        self.assertEqual(self.proj.read_provenance()["entries"][0]["bytes"], 12)
        self.assertEqual(len(self.proj.read_provenance()["entries"]), 1, "a refused call records nothing")

    def test_claim_and_note_are_bounded(self):
        self.add(claim="x" * 100000, note="y" * 100000)
        entry = self.proj.read_provenance()["entries"][0]
        self.assertEqual((len(entry["claim"]), len(entry["note"])), (2000, 2000))

    def test_a_local_file_is_copied_into_refs_byte_for_byte(self):
        entry_id = self.add(kind="download", status="obtained", local_path=self.source, detected_type="step")
        entry = [e for e in self.proj.read_provenance()["entries"] if e["id"] == entry_id][0]
        copy = self.proj.path(*entry["local_path"].split("/"))
        self.assertEqual(_read(copy, "rb"), _read(self.source, "rb"))
        self.assertEqual(entry["bytes"], os.path.getsize(self.source))
        leftovers = [n for n in os.listdir(self.proj.path("refs")) if ".tmp-" in n]
        self.assertEqual(leftovers, [])

    def test_a_second_add_of_the_same_file_reuses_the_copy(self):
        self.add(kind="download", status="obtained", local_path=self.source)
        self.add(kind="download", status="obtained", local_path=self.source)
        self.assertEqual(len(os.listdir(self.proj.path("refs"))), 1)

    def test_a_failed_copy_leaves_no_temp_file(self):
        from oscar import project_util as U

        dest = os.path.join(S.sandbox_dir(), "out.bin")
        with self.assertRaises(OSError):
            U._copy_file_atomic(os.path.join(S.sandbox_dir(), "does-not-exist"), dest)
        self.assertFalse([n for n in os.listdir(os.path.dirname(dest)) if ".tmp-" in n])
        self.assertFalse(os.path.exists(dest))


class ChatsRobustnessTest(unittest.TestCase):
    """One damaged chat file must never take the panel's chat list or a chat restore down."""

    def setUp(self):
        self.sandbox = tempfile.mkdtemp(prefix="chats_review_", dir=isolation.sandbox_root())
        original = chats.chats_dir
        chats.chats_dir = lambda: self.sandbox
        self.addCleanup(setattr, chats, "chats_dir", original)

    def _write(self, name, payload):
        with open(os.path.join(self.sandbox, name), "w", encoding="utf-8") as f:
            f.write(payload if isinstance(payload, str) else json.dumps(payload))

    def test_load_skips_a_damaged_message_and_keeps_the_rest(self):
        self._write("20260101-000000-abcdef.json", {
            "id": "20260101-000000-abcdef", "title": "t", "messages": [
                {"role": "user", "content": "hello"},
                None,
                "not a message",
                {"role": "assistant", "content": "x", "tool_calls": [{"name": "no_id"}]},
                {"role": "assistant", "content": "fine"}]})
        loaded = chats.load("20260101-000000-abcdef")
        self.assertIsNotNone(loaded)
        self.assertEqual([m.content for m in loaded["history"]], ["hello", "fine"])

    def test_load_tolerates_a_messages_value_that_is_not_a_list(self):
        for value in (None, "oops", 5, {"a": 1}):
            self._write("20260101-000000-abcdef.json", {"id": "20260101-000000-abcdef", "title": "t", "messages": value})
            loaded = chats.load("20260101-000000-abcdef")
            self.assertIsNotNone(loaded, repr(value))
            self.assertEqual(loaded["history"], [])

    def test_list_sessions_skips_files_whose_body_is_not_an_object(self):
        self._write("20260101-000000-aaaaaa.json", [])
        self._write("20260101-000000-bbbbbb.json", "null")
        self._write("20260101-000000-cccccc.json", {"id": "20260101-000000-cccccc", "title": "ok", "updated": None,
                                                       "messages": None})
        self._write("20260101-000000-dddddd.json", {"id": "20260101-000000-dddddd", "title": "fine",
                                                       "updated": "2026-01-02T00:00:00", "messages": [{}]})
        listed = chats.list_sessions()
        ids = {s["id"] for s in listed}
        self.assertIn("20260101-000000-cccccc", ids)
        self.assertIn("20260101-000000-dddddd", ids)
        self.assertNotIn("20260101-000000-aaaaaa", ids)
        self.assertEqual(listed[0]["id"], "20260101-000000-dddddd")   # newest first even with an empty/None date

    def test_a_good_chat_still_round_trips(self):
        session = chats.new_session("round trip")
        session["history"] = [Message(role="user", content="a"),
                              Message(role="assistant", content="b",
                                      tool_calls=[ToolCall(id="c1", name="x", arguments={"k": 1})])]
        chats.save(session)
        loaded = chats.load(session["id"])
        self.assertEqual(len(loaded["history"]), 2)
        self.assertEqual(loaded["history"][1].tool_calls[0].arguments, {"k": 1})


class KitSpecExtremeValuesTest(unittest.TestCase):
    """validate() never raises, and an absurd value is an error instead of a crash or a silent pass."""

    def setUp(self):
        self.board = catalog.load_board("rpi4b")

    def _spec(self, **path_values):
        d = kit_spec.default_spec().to_dict()
        for path, value in path_values.items():
            node = d
            parts = path.split("__")
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = value
        return d

    def test_a_hook_beam_of_1e308_mm_does_not_overflow(self):
        ok, errors, _ = kit_spec.validate(self._spec(hooks__length=1e308), self.board)
        self.assertFalse(ok)
        self.assertTrue(any("hooks.length" in e for e in errors), errors)

    def test_hook_strain_of_an_absurd_beam_is_finite_and_tiny(self):
        spec = kit_spec.from_dict(self._spec(hooks__length=1e308))
        self.assertEqual(kit_spec.hook_strain(spec), 0.0)

    def test_a_value_that_overflows_a_derived_dimension_is_reported_not_accepted(self):
        for path in ("lid__lip_t", "ports__gap"):
            ok, errors, _ = kit_spec.validate(self._spec(**{path: 1e308}), self.board)
            self.assertFalse(ok, path)
            self.assertTrue(any("not a finite number" in e for e in errors), (path, errors))

    def test_non_finite_and_wrong_typed_numbers_never_raise(self):
        garbage = [None, "", "abc", [], {}, True, float("nan"), float("inf"), -1, 0, 1e308, "1e999", [1]]
        for path in ("wall_t", "floor_t", "clearance", "corner_r", "standoff__h", "standoff__d", "lid__fit",
                     "hooks__count", "hooks__w", "hooks__overhang", "vents__count", "vents__pitch", "ports__margin"):
            for value in garbage:
                result = kit_spec.validate(self._spec(**{path: value}), self.board)
                self.assertEqual(len(result), 3, (path, value))

    def test_the_default_spec_is_still_valid_with_exactly_one_warning(self):
        ok, errors, warnings = kit_spec.validate(kit_spec.default_spec().to_dict(), self.board)
        self.assertTrue(ok, errors)
        self.assertEqual(len(warnings), 1, warnings)


class FileLinkClassificationTest(unittest.TestCase):
    """A link in model text must never launch an application, and on macOS an application is a DIRECTORY."""

    def test_a_bundle_directory_is_revealed_not_opened(self):
        for bundle in ("/Applications/Calculator.app", "/p/Thing.workflow", "/p/installer.pkg", "/p/x.saver/",
                       "C:\\p\\Tool.APP"):
            self.assertEqual(pr.classify_file_link(bundle, True), "reveal", bundle)

    def test_an_ordinary_directory_is_opened(self):
        for folder in ("/Users/me/Documents/Oscar/Projects/pi4-3f9a1c", "/p/exports", "/p/v1.2", "/"):
            self.assertEqual(pr.classify_file_link(folder, True), "open", folder)

    def test_the_file_rules_are_unchanged(self):
        self.assertEqual(pr.classify_file_link("/p/a.png"), "open")
        self.assertEqual(pr.classify_file_link("/p/Tool.app"), "reveal")
        self.assertEqual(pr.classify_file_link("/p/rebuild.py", False), "reveal")


class MarkdownLinkRegexTest(unittest.TestCase):
    """The link pattern must stay linear on a run of unclosed brackets (the GUI thread renders model text)."""

    def test_a_long_run_of_unclosed_brackets_renders_quickly(self):
        import time

        started = time.monotonic()
        html = pr.md_to_html("[" * 60000 + "](")
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertIn("[[[[", html)

    def test_many_unterminated_link_starts_render_quickly(self):
        import time

        started = time.monotonic()
        pr.md_to_html("[a](http://x" * 15000)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_ordinary_links_still_work_and_an_overlong_label_is_plain_text(self):
        html = pr.md_to_html("see [the doc](https://example.com/a?b=1) and [x](file:///tmp/a.stl)")
        self.assertIn("<a href='https://example.com/a?b=1'>the doc</a>", html)
        self.assertIn("<a href='file:///tmp/a.stl'>x</a>", html)
        long_label = pr.md_to_html("[" + "a" * 300 + "](https://example.com)")
        self.assertNotIn("<a href='https://example.com'>" + "a" * 300, long_label)


class DerivedExpressionTest(unittest.TestCase):
    """A derived-parameter formula is validated with the SAME alphabet FreeCAD's expression parser accepts."""

    @staticmethod
    def _validate(expr):
        from oscar.plan import schema

        mod = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(mod, "tests", "fixtures", "plans", "plate_4holes.json"), encoding="utf-8") as f:
            plan = json.load(f)
        plan["parameters"]["derived_x"] = {"expr": expr}
        return schema.validate(plan)

    def test_only_ascii_digits_are_numbers(self):
        for expr in ("\u0663 * 2", "2 * \u0966", "\uff13"):          # Arabic-Indic, Devanagari, full-width digits
            ok, errors, _ = self._validate(expr)
            self.assertFalse(ok, repr(expr))
            self.assertTrue(any("derived_x" in e for e in errors), errors)

    def test_ordinary_numbers_still_pass(self):
        for expr in ("3 * 2", "1.5e2", ".5", "5.", "2*pi", "abs(-3) + sqrt(4)"):
            self.assertTrue(self._validate(expr)[0], expr)

    def test_hostile_formulas_are_errors_not_crashes(self):
        for expr in ("__import__('os')", "a.b", "1;2", "(" * 3000 + "1" + ")" * 3000, "1+" * 5000 + "1", "x" * 100000, "\u0000"):
            ok, errors, _ = self._validate(expr)
            self.assertFalse(ok, repr(expr)[:40])
            self.assertTrue(errors)


class ModuleSanityTest(unittest.TestCase):
    def test_the_math_module_is_not_shadowed(self):
        self.assertTrue(math.isfinite(kit_spec.hook_strain(kit_spec.default_spec())))


if __name__ == "__main__":
    unittest.main()
