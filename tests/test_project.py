"""Project (oscar/project.py) file logic: paths, layout, lazy creation, atomic writes, step records, script
capture, provenance, notes, ledger / spec / validation files, thread safety, the hook skip rule and budget,
checkpoint pruning, finalize / verify_artifacts. No FreeCAD document is opened here (``doc=""`` or fakes):
the real-document behaviour is in test_project_gui.py, the generated text in test_project_docs.py."""

import json
import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import project_support as S  # noqa: E402
from oscar import paths  # noqa: E402
from oscar import project as P  # noqa: E402
from oscar import project_gui as gui  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402


class PathsTest(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(paths.slugify("Raspberry Pi 4 housing"), "raspberry-pi-4-housing")
        self.assertEqual(paths.slugify("  --Hello,   WORLD!!  "), "hello-world")
        self.assertEqual(paths.slugify(""), "project")
        self.assertEqual(paths.slugify("!!!"), "project")
        self.assertEqual(paths.slugify(None), "project")
        self.assertEqual(paths.slugify("Gehäuse für den Pi"), "gehause-fur-den-pi")

    def test_slugify_truncates_without_a_trailing_dash(self):
        slug = paths.slugify("abcdefghij klmnopqrstuvw xyz 0123456789", 12)
        self.assertLessEqual(len(slug), 12)
        self.assertFalse(slug.endswith("-"))
        self.assertEqual(paths.slugify("a" * 100), "a" * 32)

    def test_project_dirname_uses_the_hex_tail_of_the_chat_id(self):
        self.assertEqual(paths.project_dirname("20261001-213400-3f9a1c", "Raspberry Pi 4 housing"),
                         "raspberry-pi-4-housing-3f9a1c")

    def test_project_dirname_hashes_an_id_without_a_hex_tail(self):
        import hashlib

        name = paths.project_dirname("legacy chat id", "Bracket")
        self.assertEqual(name, "bracket-" + hashlib.sha1(b"legacy chat id").hexdigest()[:6])
        self.assertEqual(paths.project_dirname("x-ABCDEF", "T"), "t-" + hashlib.sha1(b"x-ABCDEF").hexdigest()[:6])
        self.assertEqual(paths.project_dirname("a-12345", "T")[:2], "t-")   # 5 hex chars: not a tail

    def test_projects_dir_reads_the_environment_at_call_time(self):
        target = os.path.join(S.sandbox_dir(), "late", "projects")
        with mock.patch.dict(os.environ, {"OSCAR_PROJECTS_DIR": target}):
            self.assertEqual(paths.projects_dir(), target)
            self.assertTrue(os.path.isdir(target))

    def test_probe_false_does_no_disk_work(self):
        target = os.path.join(S.sandbox_dir(), "never")
        with mock.patch.dict(os.environ, {"OSCAR_PROJECTS_DIR": target}):
            self.assertEqual(paths.projects_dir(probe=False), target)
        self.assertFalse(os.path.exists(target))

    def test_an_unwritable_environment_override_is_not_rescued_by_the_fallback(self):
        blocker = S.new_temp_file(S.sandbox_dir(), "iamafile")
        with mock.patch.dict(os.environ, {"OSCAR_PROJECTS_DIR": os.path.join(blocker, "sub")}):
            with self.assertRaises(OSError):
                paths.projects_dir()

    def test_unwritable_default_falls_back_and_sets_the_flag(self):
        blocker = S.new_temp_file(S.sandbox_dir(), "iamafile")
        fallback = os.path.join(S.sandbox_dir(), "fallback")
        env = {k: v for k, v in os.environ.items() if k != "OSCAR_PROJECTS_DIR"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(paths, "_default_projects_dir", return_value=os.path.join(blocker, "x")), \
                mock.patch.object(paths, "fallback_projects_dir", return_value=fallback):
            try:
                self.assertEqual(paths.projects_dir(), fallback)
                self.assertTrue(paths.PROJECTS_FALLBACK)
                self.assertTrue(os.path.isdir(fallback))
                good = os.path.join(S.sandbox_dir(), "good")
                with mock.patch.object(paths, "_default_projects_dir", return_value=good):
                    self.assertEqual(paths.projects_dir(), good)
                    self.assertFalse(paths.PROJECTS_FALLBACK)   # a successful probe resets the flag
            finally:
                paths.PROJECTS_FALLBACK = False

    def test_cache_dir_is_read_at_call_time_and_created(self):
        base = os.path.join(S.sandbox_dir(), "cache")
        with mock.patch.dict(os.environ, {"OSCAR_CACHE_DIR": base}):
            self.assertEqual(paths.cache_dir("headless", "x"), os.path.join(base, "headless", "x"))
            self.assertTrue(os.path.isdir(os.path.join(base, "headless", "x")))


class LayoutTest(unittest.TestCase):
    def test_for_chat_is_lazy_and_writes_nothing(self):
        root = S.temp_root()
        proj = S.make_project(root)
        self.assertFalse(os.path.exists(proj.root))
        self.assertEqual(os.path.basename(proj.root), "pi-4-housing-abc123")
        self.assertEqual(proj.read_steps(), [])
        self.assertEqual(proj.artifact_list(), [])
        self.assertEqual(proj.last_step_index(), 0)
        self.assertEqual(os.listdir(root), [])

    def test_create_true_materializes_the_manifest(self):
        proj = S.make_project(create=True)
        manifest = json.load(open(proj.path("project.json"), encoding="utf-8"))
        self.assertEqual(manifest["schema"], 1)
        self.assertEqual(manifest["chat_id"], S.CHAT_ID)
        self.assertEqual(manifest["title"], "Pi 4 housing")
        self.assertEqual(manifest["id"], "pi-4-housing-abc123")
        self.assertEqual(manifest["counters"], {"step": 0, "export": 0})
        self.assertEqual(manifest["reference"]["status"], "none")
        self.assertEqual(manifest["docs"], [])
        self.assertEqual(manifest["storage_note"], "")

    def test_a_question_turn_never_creates_the_folder(self):
        proj = S.make_project()
        proj.begin_turn(S.TURN, "what is a fillet?", work=False)
        proj.end_turn(S.TURN, "answered", rounds=1)
        self.assertFalse(os.path.exists(proj.root))

    def test_a_work_turn_materializes_and_flushes_the_buffered_turn_start(self):
        proj = S.make_project()
        proj.begin_turn(S.TURN, "build a housing", doc_names=["A"], work=True)
        self.assertTrue(os.path.isfile(proj.path("project.json")))
        first = json.loads(S.read_lines(proj)[0])
        self.assertEqual((first["type"], first["turn_id"], first["request"], first["docs"]),
                         ("turn_start", S.TURN, "build a housing", ["A"]))

    def test_buffered_turn_start_lines_are_written_in_order(self):
        proj = S.make_project()
        proj.begin_turn("T20261001-000001-0001", "first", work=False)
        proj.end_turn("T20261001-000001-0001", "answered")
        proj.begin_turn("T20261001-000002-0002", "second", work=True)
        kinds = [(json.loads(l)["type"], json.loads(l).get("request", json.loads(l).get("outcome")))
                 for l in S.read_lines(proj)]
        self.assertEqual(kinds, [("turn_start", "first"), ("turn_end", "answered"), ("turn_start", "second")])

    def test_for_chat_reopens_an_existing_project(self):
        root = S.temp_root()
        first = S.make_project(root, create=True)
        S.record(first)
        again = S.make_project(root)
        self.assertEqual(again.root, first.root)
        self.assertEqual(again.last_step_index(), 1)
        self.assertEqual(again.manifest["counters"]["step"], 1)

    def test_a_renamed_chat_keeps_its_folder(self):
        root = S.temp_root()
        first = S.make_project(root, title="Old name", create=True)
        again = S.make_project(root, title="A brand new title")
        self.assertEqual(again.root, first.root)

    def test_two_chats_get_two_folders(self):
        root = S.temp_root()
        a = S.make_project(root, chat_id="20261001-000000-aaaaaa", title="Same", create=True)
        b = S.make_project(root, chat_id="20261001-000000-bbbbbb", title="Same", create=True)
        self.assertNotEqual(a.root, b.root)

    def test_open_requires_a_manifest(self):
        with self.assertRaises(P.ProjectError):
            P.Project.open(S.temp_root())
        proj = S.make_project(create=True)
        self.assertEqual(P.Project.open(proj.root).manifest["chat_id"], S.CHAT_ID)

    def test_path_and_rel(self):
        proj = S.make_project(create=True)
        self.assertEqual(proj.rel(proj.path("snapshots", "001_a.png")), "snapshots/001_a.png")
        with self.assertRaises(P.ProjectError):
            proj.rel(os.path.join(os.path.dirname(proj.root), "elsewhere.txt"))

    def test_unwritable_root_falls_back_with_a_storage_note(self):
        blocker = S.new_temp_file(S.sandbox_dir(), "iamafile")
        fallback = os.path.join(S.sandbox_dir(), "fallback")
        proj = P.Project(os.path.join(blocker, "projects", "x-abc123"), chat_id=S.CHAT_ID, title="x")
        try:
            with mock.patch.object(paths, "fallback_projects_dir", return_value=fallback):
                proj.ensure_dir()
            self.assertEqual(proj.root, os.path.join(fallback, "x-abc123"))
            self.assertIn("not writable", proj.storage_note)
            self.assertTrue(paths.PROJECTS_FALLBACK)
            self.assertEqual(json.load(open(proj.path("project.json")))["storage_note"], proj.storage_note)
        finally:
            paths.PROJECTS_FALLBACK = False

    def test_unwritable_root_and_fallback_raise_a_project_error(self):
        blocker = S.new_temp_file(S.sandbox_dir(), "iamafile")
        proj = P.Project(os.path.join(blocker, "projects", "x-abc123"), chat_id=S.CHAT_ID, title="x")
        with mock.patch.object(paths, "fallback_projects_dir", return_value=os.path.join(blocker, "fb")):
            with self.assertRaises(P.ProjectError) as caught:
                proj.ensure_dir()
        self.assertIn("OSCAR_PROJECTS_DIR", str(caught.exception))

    def test_artifact_list_ignores_backups_and_temp_files(self):
        proj = S.make_project(create=True)
        S.new_temp_file(proj.root, "model.FCStd", b"x")
        S.new_temp_file(proj.root, "model.20261001.FCBak", b"x")
        S.new_temp_file(proj.path("exports"), "pi-shell.step", b"x")
        S.new_temp_file(proj.path("exports"), "pi-shell.step.tmp-1-ab", b"x")
        S.new_temp_file(proj.path("exports"), ".hidden", b"x")
        S.new_temp_file(proj.path("docs"), "Other.FCStd", b"x")
        S.new_temp_file(proj.path("docs"), "Other.1.FCBak", b"x")
        for i in range(1, 9):
            S.new_temp_file(proj.path("snapshots"), f"{i:03d}_x.png", b"x")
        listing = proj.artifact_list()
        names = [e["name"] for e in listing]
        self.assertIn("model.FCStd", names)
        self.assertIn("docs/Other.FCStd", names)
        self.assertIn("exports/pi-shell.step", names)
        self.assertEqual([n for n in names if "FCBak" in n or ".tmp-" in n or "hidden" in n], [])
        self.assertEqual(len([n for n in names if n.startswith("snapshots/")]), 6)   # the newest six
        self.assertTrue(all(os.path.isabs(e["path"]) for e in listing))
        kinds = {e["name"]: e["kind"] for e in listing}
        self.assertEqual((kinds["model.FCStd"], kinds["exports/pi-shell.step"]), ("fcstd", "export"))

    def test_export_path_is_unique_and_thread_safe(self):
        proj = S.make_project(create=True)
        got, lock = [], threading.Lock()

        def grab():
            path = proj.export_path("pi shell", "STEP")
            with lock:
                got.append(path)

        threads = [threading.Thread(target=grab) for _ in range(12)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(set(got)), 12)
        self.assertTrue(all(p.startswith(proj.path("exports") + os.sep) and p.endswith(".step") for p in got))
        self.assertIn(proj.path("exports", "pi-shell.step"), got)
        self.assertIn(proj.path("exports", "pi-shell-2.step"), got)
        self.assertEqual(proj.manifest["counters"]["export"], 12)


class AtomicWriteTest(unittest.TestCase):
    def test_no_temp_file_is_left_behind(self):
        directory = S.sandbox_dir()
        target = os.path.join(directory, "a.json")
        P._atomic_write_json(target, {"x": 1})
        self.assertEqual(os.listdir(directory), ["a.json"])
        self.assertEqual(json.load(open(target)), {"x": 1})

    def test_a_failed_replace_removes_the_temp_file_and_keeps_the_old_content(self):
        directory = S.sandbox_dir()
        target = os.path.join(directory, "a.txt")
        P._atomic_write_text(target, "old")
        with mock.patch("os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                P._atomic_write_text(target, "new")
        self.assertEqual(os.listdir(directory), ["a.txt"])
        self.assertEqual(open(target).read(), "old")

    def test_a_reader_never_sees_a_partial_file(self):
        directory = S.sandbox_dir()
        target = os.path.join(directory, "big.json")
        P._atomic_write_json(target, {"n": 0, "pad": "x" * 200_000})
        stop, bad = threading.Event(), []

        def reader():
            while not stop.is_set():
                try:
                    with open(target, encoding="utf-8") as handle:
                        json.load(handle)
                except ValueError as exc:
                    bad.append(str(exc))
                # A real reader (the panel, a second process) reads and goes away. Windows cannot replace a file while ANY
                # handle is open, so a reader that re-opens it the instant it closes it would starve every writer for ever
                # (no retry can win against a handle that is always open); a short pause models real readers.
                time.sleep(0.002)

        thread = threading.Thread(target=reader)
        thread.start()
        for n in range(1, 60):
            P._atomic_write_json(target, {"n": n, "pad": "x" * 200_000})
        stop.set()
        thread.join()
        self.assertEqual(bad, [])

    def test_save_copy_uses_a_temp_name_that_ends_in_fcstd(self):
        """FreeCAD appends .FCStd to any other name and writes THERE: a '.tmp' name would never exist."""
        directory = S.sandbox_dir()
        dest = os.path.join(directory, "model.FCStd")
        doc = S.FakeDoc()
        self.assertEqual(gui.save_copy(doc, dest), (True, ""))
        self.assertEqual(len(doc.saved_to), 1)
        self.assertTrue(doc.saved_to[0].endswith(".FCStd"))
        self.assertNotEqual(doc.saved_to[0], dest)
        self.assertIn(".tmp-", os.path.basename(doc.saved_to[0]))
        self.assertEqual(os.listdir(directory), ["model.FCStd"])   # the temp is gone, nothing else appeared
        import zipfile

        self.assertTrue(zipfile.is_zipfile(dest))

    def test_save_copy_overwrites_atomically_without_a_backup_file(self):
        directory = S.sandbox_dir()
        dest = os.path.join(directory, "model.FCStd")
        doc = S.FakeDoc()
        gui.save_copy(doc, dest)
        gui.save_copy(doc, dest)
        self.assertEqual(os.listdir(directory), ["model.FCStd"])

    def test_save_copy_reports_a_save_that_wrote_nothing(self):
        directory = S.sandbox_dir()
        ok, error = gui.save_copy(S.FakeDoc(write=False), os.path.join(directory, "model.FCStd"))
        self.assertFalse(ok)
        self.assertIn("did not produce", error)
        self.assertEqual(os.listdir(directory), [])

    def test_save_copy_rejects_a_file_that_is_not_a_zip_and_cleans_up(self):
        directory = S.sandbox_dir()
        dest = os.path.join(directory, "model.FCStd")
        ok, _ = gui.save_copy(S.FakeDoc(garbage=True), dest)
        self.assertFalse(ok)
        self.assertEqual(os.listdir(directory), [])

    def test_save_copy_creates_the_parent_folder(self):
        dest = os.path.join(S.sandbox_dir(), "docs", "Other.FCStd")
        self.assertTrue(gui.save_copy(S.FakeDoc("Other"), dest)[0])


class StepRecordTest(unittest.TestCase):
    def setUp(self):
        self.proj = S.make_project()

    def test_the_contract_example(self):
        """The documented layout, verbatim."""
        proj = self.proj
        proj.begin_turn("T20261001-000001-0001", "build a housing", doc_names=[])
        rec = proj.on_tool_result(turn_id="T20261001-000001-0001", round=0, call_id="c1", tool="execute_code",
                                  title="Make a box", args={"code": "print(1)"}, text="ok", ok=True, dur_s=0.1,
                                  doc="", delta={"added": ["Box"], "removed": [], "changed": []},
                                  health=None, mutating=True, geometry=True)
        self.assertEqual(rec.index, 1)
        self.assertEqual(rec.script, "scripts/001_make-a-box.py")

    def test_read_tools_are_not_recorded(self):
        self.assertIsNone(S.record(self.proj, tool="get_objects", mutating=False, geometry=False))
        self.assertFalse(os.path.exists(self.proj.root))

    def test_a_step_line_has_every_field(self):
        rec = S.record(self.proj, health={"valid": True, "errors": 0, "warnings": 0, "line": "[health] ok",
                                          "findings": [{"code": "X"}] * 9})
        line = json.loads(S.read_lines(self.proj)[-1])
        self.assertEqual(line["type"], "step")
        for key in ("index", "ts", "turn_id", "round", "call_id", "tool", "title", "args", "ok", "result_head",
                    "result_chars", "dur_s", "doc", "delta", "health", "snapshot", "snapshot_note", "checkpoint",
                    "checkpoint_note", "script", "replay"):
            self.assertIn(key, line)
        self.assertNotIn("superseded", line)
        self.assertEqual(len(line["health"]["findings"]), 5)   # at most five findings are stored
        self.assertEqual(rec.index, line["index"])
        self.assertEqual(self.proj.read_steps()[0].tool, "execute_code")

    def test_geometry_step_without_a_document(self):
        rec = S.record(self.proj)
        self.assertEqual((rec.snapshot, rec.snapshot_note, rec.checkpoint, rec.checkpoint_note),
                         ("", "no document", "", "no document"))

    def test_result_head_title_and_arguments(self):
        text = "r" * 1000
        rec = S.record(self.proj, text=text, title="  A   very " + "long " * 40 + "title", args={"code": "x" * 250_000, "n": 3})
        self.assertEqual((len(rec.result_head), rec.result_chars), (600, 1000))
        self.assertLessEqual(len(rec.title), 80)
        self.assertNotIn("  ", rec.title)
        self.assertTrue(rec.args["code"].endswith("...[truncated 50000 chars]"))
        self.assertEqual(len(rec.args["code"]), 200_000 + len("...[truncated 50000 chars]"))
        self.assertEqual(rec.args["n"], 3)

    def test_empty_title_falls_back_to_the_tool_name(self):
        self.assertEqual(S.record(self.proj, title="").title, "execute_code")

    def test_unserialisable_arguments_do_not_break_the_record(self):
        rec = S.record(self.proj, args={"obj": object(), "nan": float("nan"), "set": {2, 1}, "t": (1, 2), 5: "int key"})
        self.assertEqual(rec.args["nan"], "nan")
        self.assertEqual(rec.args["set"], [1, 2])
        self.assertEqual(rec.args["t"], [1, 2])
        self.assertEqual(rec.args["5"], "int key")
        json.loads(S.read_lines(self.proj)[-1])

    def test_unicode_and_lone_surrogates_are_written(self):
        rec = S.record(self.proj, text="Gehäuse \ud800 done", args={"code": "print('Gehäuse')"})
        self.assertGreater(rec.index, 0)
        self.assertEqual(len(self.proj.read_steps()), 1)

    def test_indexes_are_monotonic_and_survive_a_reopen(self):
        S.record(self.proj)
        S.record(self.proj)
        again = S.make_project(os.path.dirname(self.proj.root))
        self.assertEqual(S.record(again).index, 3)

    def test_an_index_is_never_reused_after_a_crash_between_journal_and_manifest(self):
        S.record(self.proj)
        S.record(self.proj)
        manifest = json.load(open(self.proj.path("project.json")))
        manifest["counters"]["step"] = 1   # the manifest write of step 2 was lost
        json.dump(manifest, open(self.proj.path("project.json"), "w"))
        again = S.make_project(os.path.dirname(self.proj.root))
        self.assertEqual(S.record(again).index, 3)

    def test_a_torn_last_line_is_repaired_and_does_not_swallow_the_next_record(self):
        S.record(self.proj)
        with open(self.proj.path("steps.jsonl"), "ab") as f:
            f.write(b'{"type":"step","index":2,"tool":"exec')   # crash mid-write: no newline
        again = S.make_project(os.path.dirname(self.proj.root))
        self.assertEqual(again.last_step_index(), 1)   # the torn line is skipped by readers
        rec = S.record(again)
        self.assertEqual(rec.index, 2)
        lines = S.read_lines(again)
        self.assertEqual(len(lines), 3)
        self.assertRaises(ValueError, json.loads, lines[1])   # the corrupt one stayed on its own line
        self.assertEqual(json.loads(lines[2])["index"], 2)
        self.assertEqual([s.index for s in again.read_steps()], [1, 2])

    def test_readers_skip_garbage_lines(self):
        S.record(self.proj)
        with open(self.proj.path("steps.jsonl"), "ab") as f:
            f.write(b"\nnot json\n[1,2]\n{\"type\":\"step\"}\n{\"type\":\"mystery\"}\n")
        steps = self.proj.read_steps()
        self.assertEqual([s.index for s in steps][:1], [1])

    def test_rewind_lines_supersede_later_steps(self):
        for _ in range(4):
            S.record(self.proj)
        self.assertEqual(self.proj.last_geometry_step(), 4)
        with self.proj._lock:
            self.proj._journal_line({"type": "rewind", "ts": "x", "to_step": 2, "restored_checkpoint": "002"})
        self.assertEqual([s.index for s in self.proj.read_steps()], [1, 2])
        everything = self.proj.read_steps(include_superseded=True)
        self.assertEqual([(s.index, s.superseded) for s in everything], [(1, False), (2, False), (3, True), (4, True)])
        self.assertEqual(self.proj.last_geometry_step(), 2)
        self.assertEqual(self.proj.last_step_index(), 4)   # indexes are never reused
        self.assertEqual(S.record(self.proj).index, 5)

    def test_a_later_step_is_not_superseded_by_an_earlier_rewind(self):
        for _ in range(3):
            S.record(self.proj)
        with self.proj._lock:
            self.proj._journal_line({"type": "rewind", "ts": "x", "to_step": 1, "restored_checkpoint": "001"})
        S.record(self.proj)   # step 4 comes after the rewind line
        self.assertEqual([s.index for s in self.proj.read_steps()], [1, 4])

    def test_last_geometry_step_ignores_failed_and_non_geometry_steps(self):
        S.record(self.proj)
        S.record(self.proj, ok=False)
        S.record(self.proj, tool="record_note_like", geometry=False)
        self.assertEqual(self.proj.last_geometry_step(), 1)
        self.assertEqual(self.proj.manifest["last_geometry_step"], 1)

    def test_concurrent_hooks_never_interleave_or_reuse_an_index(self):
        errors = []

        def worker(n):
            try:
                for k in range(15):
                    S.record(self.proj, call_id=f"t{n}-{k}", title=f"step {n}-{k}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        lines = [json.loads(l) for l in S.read_lines(self.proj)]
        indexes = sorted(l["index"] for l in lines if l["type"] == "step")
        self.assertEqual(indexes, list(range(1, 121)))
        self.assertEqual(len(self.proj.read_steps()), 120)
        self.assertEqual(self.proj.manifest["counters"]["step"], 120)

    def test_the_hook_never_raises_into_the_agent(self):
        with mock.patch.object(P.Project, "_record", side_effect=RuntimeError("boom")):
            rec = S.record(self.proj)
        self.assertEqual(rec.index, -1)
        self.assertIn("project error", rec.snapshot_note)
        self.assertIn("boom", rec.snapshot_note)

    def test_end_turn_line(self):
        self.proj.begin_turn(S.TURN, "x", work=True)
        self.proj.end_turn(S.TURN, "validated", rounds=7)
        last = json.loads(S.read_lines(self.proj)[-1])
        self.assertEqual((last["type"], last["turn_id"], last["outcome"], last["rounds"]),
                         ("turn_end", S.TURN, "validated", 7))


class ScriptCaptureTest(unittest.TestCase):
    def setUp(self):
        self.proj = S.make_project()

    def test_script_file_has_a_three_line_header_and_the_code_verbatim(self):
        code = "import Part\nbox = Part.makeBox(1, 2, 3)\nprint('Gehäuse')"
        rec = S.record(self.proj, args={"code": code}, title="Bottom shell with standoffs", doc="")
        text = open(self.proj.path(rec.script), encoding="utf-8").read()
        lines = text.split("\n")
        self.assertEqual(rec.script, "scripts/001_bottom-shell-with-standoffs.py")
        self.assertTrue(lines[0].startswith("# Oscar step 001: Bottom shell with standoffs"))
        self.assertTrue(lines[1].startswith(f"# turn {S.TURN}  tool execute_code  doc "))
        self.assertTrue(lines[2].startswith("# Run through oscar.cad.scripting.run_script"))
        self.assertEqual("\n".join(lines[3:]).rstrip("\n"), code)
        self.assertTrue(text.endswith("\n"))
        compile(text, "script", "exec")

    def test_headless_scripts_are_captured_too(self):
        rec = S.record(self.proj, tool="execute_code_headless", geometry=False, args={"code": "print(2)"})
        self.assertEqual(rec.replay, {"kind": "script", "file": rec.script, "headless": True, "gui_only": False})

    def test_failed_and_empty_scripts_are_not_captured(self):
        self.assertEqual(S.record(self.proj, ok=False).script, "")
        self.assertEqual(S.record(self.proj, args={"code": "   \n"}).script, "")
        self.assertEqual(S.record(self.proj, args={}).script, "")
        self.assertEqual(S.record(self.proj, args={"code": 5}).script, "")
        self.assertFalse(os.path.isdir(self.proj.path("scripts")))

    def test_other_tools_do_not_write_scripts(self):
        rec = S.record(self.proj, tool="create_object", args={"code": "ignored", "obj_type": "Part::Box"})
        self.assertEqual(rec.script, "")

    def test_gui_only_lint(self):
        lint = P._lint_gui_only
        self.assertTrue(lint("o = App.ActiveDocument.Box\no.ViewObject.ShapeColor = (1, 0, 0)"))
        self.assertTrue(lint("import FreeCADGui\nFreeCADGui.SendMsgToActiveView('ViewFit')"))
        self.assertTrue(lint("from FreeCADGui import something"))
        self.assertTrue(lint("Gui.ActiveDocument.ActiveView.fitAll()"))
        self.assertFalse(lint("o = App.ActiveDocument.Box\nprint(o.Shape.Volume)"))
        self.assertFalse(lint("vo = getattr(obj, 'ViewObject', None)\nif vo is not None:\n    vo.ShapeColor = (1, 0, 0)"))
        self.assertFalse(lint("if App.GuiUp:\n    obj.ViewObject.Visibility = False"))
        self.assertFalse(lint("if obj.ViewObject:\n    obj.ViewObject.Visibility = False"))
        self.assertFalse(lint("try:\n    obj.ViewObject.Visibility = False\nexcept Exception:\n    pass"))
        self.assertFalse(lint("def broken(:\n"))   # a script that does not parse is not flagged

    def test_gui_only_scripts_are_marked_and_listed_as_unresolved(self):
        rec = S.record(self.proj, args={"code": "obj = App.ActiveDocument.Box\nobj.ViewObject.ShapeColor = (1, 0, 0)"},
                       title="Colour the box")
        self.assertTrue(rec.replay["gui_only"])
        notes = open(self.proj.path("NOTES.md"), encoding="utf-8").read()
        self.assertIn("not headless-replayable", notes)
        self.assertIn("Colour the box", notes)

    def test_headless_scripts_are_never_gui_only(self):
        rec = S.record(self.proj, tool="execute_code_headless", geometry=False,
                       args={"code": "obj.ViewObject.ShapeColor = 1"})
        self.assertFalse(rec.replay["gui_only"])


class ReplayEntryTest(unittest.TestCase):
    def setUp(self):
        self.proj = S.make_project()

    def test_replayable_tools_are_recorded_as_tool_replays(self):
        for tool in sorted(P.REPLAYABLE):
            rec = S.record(self.proj, tool=tool, args={"doc_name": "D", "k": 1})
            self.assertEqual(rec.replay, {"kind": "tool", "name": tool, "args": {"doc_name": "D", "k": 1}}, tool)

    def test_failed_calls_are_not_replayed(self):
        self.assertEqual(S.record(self.proj, tool="create_object", ok=False).replay["kind"], "none")

    def test_not_replayable_tools_say_why(self):
        for tool in ("execute_code_async", "source_reference_part", "run_fem_analysis", "save_document",
                     "export_model", "export_rebuild_script", "some_new_tool"):
            rec = S.record(self.proj, tool=tool, geometry=False)
            self.assertEqual(rec.replay["kind"], "none", tool)
            self.assertTrue(rec.replay["why"], tool)

    def test_create_document_records_the_document_name_from_the_result(self):
        rec = S.record(self.proj, tool="create_document", geometry=False, args={"name": "PiCase"},
                       text="Document 'PiCase1' created and active.")
        self.assertEqual(rec.doc, "PiCase1")
        self.assertEqual(self.proj.manifest["docs"], ["PiCase1"])


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.proj = S.make_project()
        self.good = dict(kind="reference", claim="Board outline 85 x 56 mm", source="catalog:rpi4b board.json v1",
                         status="nominal", confidence="datasheet")

    def test_ids_and_the_stored_entry(self):
        first = self.proj.provenance_add(**self.good, url="https://example.org/d.pdf", note="no STEP")
        second = self.proj.provenance_add(**self.good)
        self.assertEqual((first, second), ("E1", "E2"))
        data = self.proj.read_provenance()
        self.assertEqual(data["schema"], 1)
        entry = data["entries"][0]
        self.assertEqual((entry["id"], entry["kind"], entry["status"], entry["confidence"], entry["url"], entry["note"]),
                         ("E1", "reference", "nominal", "datasheet", "https://example.org/d.pdf", "no STEP"))
        self.assertEqual(entry["detected_type"], "none")
        self.assertEqual(entry["reference_status"], "")

    def test_invalid_enums_list_the_accepted_values(self):
        for field, value, expected in (("kind", "rumour", "reference"), ("status", "fine", "nominal"),
                                       ("confidence", "sure", "measured"), ("detected_type", "docx", "step"),
                                       ("reference_status", "perfect", "catalog-nominal")):
            with self.assertRaises(P.ProjectError) as caught:
                self.proj.provenance_add(**{**self.good, field: value})
            self.assertIn(expected, str(caught.exception), field)
            self.assertIn(field, str(caught.exception))

    def test_missing_kind_and_status_are_errors(self):
        for field in ("kind", "status"):
            bad = {k: v for k, v in self.good.items() if k != field}
            with self.assertRaises(P.ProjectError):
                self.proj.provenance_add(**bad)

    def test_source_must_be_a_non_empty_string(self):
        for bad in ("", "   ", None, 5):
            with self.assertRaises(P.ProjectError):
                self.proj.provenance_add(**{**self.good, "source": bad})
        with self.assertRaises(P.ProjectError):
            self.proj.provenance_add(**{k: v for k, v in self.good.items() if k != "source"})

    def test_unknown_fields_are_rejected(self):
        with self.assertRaises(P.ProjectError) as caught:
            self.proj.provenance_add(**self.good, colour="red")
        self.assertIn("colour", str(caught.exception))
        self.assertIn("accepted", str(caught.exception))

    def test_nothing_is_written_for_an_invalid_call(self):
        with self.assertRaises(P.ProjectError):
            self.proj.provenance_add(**{**self.good, "status": "fine"})
        self.assertFalse(os.path.exists(self.proj.path("provenance.json")))

    def test_confidence_defaults_to_assumed(self):
        good = {k: v for k, v in self.good.items() if k != "confidence"}
        self.proj.provenance_add(**good)
        self.assertEqual(self.proj.read_provenance()["entries"][0]["confidence"], "assumed")

    def test_local_file_is_copied_hashed_and_made_relative(self):
        import hashlib

        src = S.new_temp_file(S.sandbox_dir(), "Pi 4 (rev1).step", b"ISO-10303-21;\nDATA;")
        digest = hashlib.sha256(b"ISO-10303-21;\nDATA;").hexdigest()
        entry_id = self.proj.provenance_add(**self.good, local_path=src, detected_type="step")
        entry = self.proj.read_provenance()["entries"][0]
        self.assertEqual(entry["sha256"], digest)
        self.assertEqual(entry["bytes"], 19)
        self.assertTrue(entry["local_path"].startswith(f"refs/{digest[:12]}_"))
        self.assertTrue(entry["local_path"].endswith(".step"))
        self.assertEqual(open(self.proj.path(entry["local_path"]), "rb").read(), b"ISO-10303-21;\nDATA;")
        self.assertEqual(entry_id, "E1")

    def test_a_wrong_sha256_or_a_missing_file_is_rejected(self):
        src = S.new_temp_file(S.sandbox_dir(), "a.step", b"abc")
        with self.assertRaises(P.ProjectError) as caught:
            self.proj.provenance_add(**self.good, local_path=src, sha256="0" * 64)
        self.assertIn("does not match", str(caught.exception))
        with self.assertRaises(P.ProjectError):
            self.proj.provenance_add(**self.good, local_path=os.path.join(S.sandbox_dir(), "gone.step"))
        with self.assertRaises(P.ProjectError):
            self.proj.provenance_add(**self.good, local_path="relative/path.step")

    def test_reference_status_updates_the_manifest_and_provenance(self):
        self.assertEqual(self.proj.reference_status, "none")
        entry_id = self.proj.provenance_add(**self.good, reference_status="catalog-nominal")
        self.assertEqual(self.proj.reference_status, "catalog-nominal")
        data = self.proj.read_provenance()
        self.assertEqual(data["reference"]["status"], "catalog-nominal")
        self.assertEqual(data["reference"]["entry"], entry_id)
        self.assertEqual(json.load(open(self.proj.path("project.json")))["reference"]["status"], "catalog-nominal")

    def test_set_reference(self):
        self.proj.set_reference("rpi4b", "Raspberry Pi 4 Model B", "step-imported", "E3")
        self.assertEqual(self.proj.reference_status, "step-imported")
        self.assertEqual(self.proj.read_provenance()["reference"],
                         {"id": "rpi4b", "name": "Raspberry Pi 4 Model B", "status": "step-imported", "entry": "E3"})
        with self.assertRaises(P.ProjectError) as caught:
            self.proj.set_reference("x", "y", "unknown-status", "E1")
        self.assertIn("catalog-nominal", str(caught.exception))
        self.assertEqual(self.proj.reference_status, "step-imported")

    def test_reference_status_property_tolerates_a_corrupt_manifest_value(self):
        self.proj.ensure_dir()
        self.proj.manifest["reference"]["status"] = "bogus"
        self.assertEqual(self.proj.reference_status, "none")

    def test_the_enum_constants(self):
        self.assertEqual(P.REFERENCE_STATUSES, ("none", "catalog-nominal", "step-imported", "step-measured", "user-attached"))
        self.assertEqual(P.PROVENANCE_STATUSES, ("obtained", "declined", "failed", "nominal", "attached"))
        self.assertEqual(P.CONFIDENCE_LEVELS, ("measured", "datasheet", "community", "approximate", "assumed"))
        self.assertEqual(P.NOTE_KINDS, ("plan", "assumption", "provenance", "validation", "unresolved", "decision"))


class NotesTest(unittest.TestCase):
    def setUp(self):
        self.proj = S.make_project()

    def test_ids_kinds_refs_and_step(self):
        S.record(self.proj)
        first = self.proj.note("assumption", "wall 2 mm", refs=["S5", "R2"])
        second = self.proj.note("decision", "use a cap lid")
        self.assertEqual((first, second), ("N1", "N2"))
        entries = self.proj.read_notes()
        self.assertEqual((entries[0]["kind"], entries[0]["refs"], entries[0]["step"], entries[0]["text"]),
                         ("assumption", ["S5", "R2"], 1, "wall 2 mm"))
        self.assertEqual(json.load(open(self.proj.path("notes.json")))["schema"], 1)

    def test_text_is_clipped_to_2000_characters(self):
        self.proj.note("unresolved", "x" * 5000)
        self.assertEqual(len(self.proj.read_notes()[0]["text"]), 2000)

    def test_invalid_kind_and_empty_text(self):
        with self.assertRaises(P.ProjectError) as caught:
            self.proj.note("rumour", "x")
        self.assertIn("assumption", str(caught.exception))
        with self.assertRaises(P.ProjectError):
            self.proj.note("plan", "   ")
        self.assertFalse(os.path.exists(self.proj.path("notes.json")))

    def test_a_note_regenerates_notes_md(self):
        self.proj.note("assumption", "the board is 85 x 56 mm")
        notes = open(self.proj.path("NOTES.md"), encoding="utf-8").read()
        self.assertIn("the board is 85 x 56 mm", notes)

    def test_concurrent_notes_get_distinct_ids(self):
        ids, lock = [], threading.Lock()

        def add():
            note_id = self.proj.note("plan", "p")
            with lock:
                ids.append(note_id)

        threads = [threading.Thread(target=add) for _ in range(10)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(sorted(ids, key=lambda s: int(s[1:])), [f"N{i}" for i in range(1, 11)])


class LedgerSpecValidationFilesTest(unittest.TestCase):
    def setUp(self):
        self.proj = S.make_project()

    def test_defaults_before_anything_is_written(self):
        self.assertIsNone(self.proj.read_ledger())
        self.assertEqual(self.proj.read_params(), {"params": []})
        self.assertEqual(self.proj.read_provenance()["entries"], [])
        self.assertEqual(self.proj.read_provenance()["reference"]["status"], "none")
        self.assertEqual(self.proj.read_notes(), [])
        self.assertIsNone(self.proj.read_spec())
        self.assertIsNone(self.proj.read_validation())

    def test_readers_tolerate_corrupt_files(self):
        self.proj.ensure_dir()
        for name in ("ledger.json", "params.json", "provenance.json", "notes.json", "spec.json", "validation.json"):
            with open(self.proj.path(name), "w") as f:
                f.write("{ torn")
        self.assertIsNone(self.proj.read_ledger())
        self.assertEqual(self.proj.read_params(), {"params": []})
        self.assertEqual(self.proj.read_provenance()["entries"], [])
        self.assertEqual(self.proj.read_notes(), [])
        self.assertIsNone(self.proj.read_spec())
        self.assertIsNone(self.proj.read_validation())

    def test_ledger_roundtrip_accepts_a_dict_or_an_object_with_to_dict(self):
        class Ledger:
            def to_dict(self):
                return {"intent": "enclosure", "steps": []}

        self.proj.save_ledger({"intent": "bracket"})
        self.assertEqual(self.proj.read_ledger(), {"intent": "bracket"})
        self.proj.save_ledger(Ledger())
        self.assertEqual(self.proj.read_ledger()["intent"], "enclosure")
        with self.assertRaises(P.ProjectError):
            self.proj.save_ledger(42)

    def test_archive_ledger_moves_the_file(self):
        self.proj.archive_ledger("T1")   # nothing to archive: a no-op
        self.proj.save_ledger({"intent": "bracket"})
        self.proj.archive_ledger("T20261001-000001-0001")
        self.assertIsNone(self.proj.read_ledger())
        self.assertEqual(json.load(open(self.proj.path("ledgers", "T20261001-000001-0001.json")))["intent"], "bracket")

    def test_spec_roundtrip(self):
        self.proj.save_spec({"name": "pi4-case", "material": "ABS"})
        self.assertEqual(self.proj.read_spec(), {"name": "pi4-case", "material": "ABS"})
        with self.assertRaises(P.ProjectError):
            self.proj.save_spec("not a dict")

    def test_record_validation_fills_defaults(self):
        S.record(self.proj)
        S.record(self.proj)
        self.proj.record_validation({"ok": False, "ran": ["health"], "text": "never stored"})
        data = self.proj.read_validation()
        self.assertEqual(data["schema"], 1)
        self.assertEqual(data["after_step"], 2)   # defaults to the newest geometry step
        self.assertFalse(data["ok"])
        self.assertEqual(data["ran"], ["health"])
        for key in ("health", "gate_errors", "fit_rows", "dfm", "coverage", "not_validated", "ts", "doc"):
            self.assertIn(key, data)
        self.assertNotIn("text", data)

    def test_record_validation_accepts_a_validation_result_object(self):
        from support.fakes import fake_validation

        S.record(self.proj)
        self.proj.record_validation(fake_validation(ok=True))
        data = self.proj.read_validation()
        self.assertEqual(data["ran"], ["health"])
        self.assertEqual(data["not_validated"], ["step_collision", "physical"])
        self.assertTrue(data["ok"])


class HookSkipRuleTest(unittest.TestCase):
    """GUI hooks are skipped (with a visible note) after a dispatch timeout, after Stop and when the budget is gone."""

    def setUp(self):
        self.gui = S.FakeGui().install()
        self.addCleanup(self.gui.uninstall)
        self.fake = S.FakeDoc("Doc")
        patcher = mock.patch.object(gui, "resolve_doc", lambda name: self.fake if name in ("", "Doc") else None)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("has_geometry",):
            p = mock.patch.object(gui, name, lambda doc: True)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(gui, "object_names", lambda doc: (["Box"], ["Box"]))
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(gui, "param_table", lambda doc: {"sheet": "", "rows": []})
        p.start()
        self.addCleanup(p.stop)
        self.proj = S.make_project()

    def test_a_healthy_geometry_step_gets_autosave_checkpoint_and_snapshot(self):
        rec = S.record(self.proj, doc="Doc", title="Make a box")
        self.assertEqual(rec.checkpoint, "001")
        self.assertEqual(rec.snapshot, "snapshots/001_make-a-box.png")
        self.assertEqual((rec.checkpoint_note, rec.snapshot_note), ("", ""))
        self.assertTrue(os.path.isfile(self.proj.path("model.FCStd")))
        self.assertTrue(os.path.isfile(self.proj.path("checkpoints", "001.FCStd")))
        side = json.load(open(self.proj.path("checkpoints", "001.json")))
        self.assertEqual((side["step"], side["doc"], side["objects"]), (1, "Doc", ["Box"]))
        self.assertEqual(self.proj.manifest["active_doc"], "Doc")
        self.assertEqual(self.proj.manifest["docs"], ["Doc"])
        self.assertTrue(all(p.endswith(".FCStd") for p in self.fake.saved_to))

    def test_a_dispatch_timeout_in_the_handler_skips_every_gui_hook(self):
        before = len(self.gui.ops)
        rec = S.record(self.proj, doc="Doc", ok=True,
                       text="'execute_code' did not complete within 60s (the GUI thread stayed busy)")
        self.assertEqual(rec.checkpoint_note, "skipped (GUI busy)")
        self.assertEqual(rec.snapshot_note, "skipped (GUI busy)")
        self.assertEqual((rec.checkpoint, rec.snapshot), ("", ""))
        self.assertEqual(len(self.gui.ops), before)   # not a single dispatch
        self.assertEqual(self.fake.saved_to, [])

    def test_stop_skips_every_gui_hook(self):
        ctx = ToolContext(project=self.proj, should_stop=lambda: True)
        with use_context(ctx):
            rec = S.record(self.proj, doc="Doc")
        self.assertEqual(rec.checkpoint_note, "skipped (stopped)")
        self.assertEqual(rec.snapshot_note, "skipped (stopped)")
        self.assertEqual(self.gui.ops, [])

    def test_the_hook_budget_is_shared_by_all_gui_hooks(self):
        self.proj.HOOK_BUDGET_S = 1.0
        slow = S.FakeGui(delay=1.3).install()
        self.addCleanup(slow.uninstall)
        started = time.monotonic()
        rec = S.record(self.proj, doc="Doc")
        elapsed = time.monotonic() - started
        self.assertEqual(rec.checkpoint_note, "skipped (GUI busy)")
        self.assertEqual(rec.snapshot_note, "skipped (GUI busy)")
        self.assertLess(elapsed, 3.0)
        self.assertEqual(slow.ops, ["project_autosave"])   # the first hook spent the whole budget

    def test_each_dispatch_passes_should_stop(self):
        seen = []

        def spy(fn, timeout=60.0, op="", should_stop=None):
            seen.append((op, should_stop))
            return self.gui.dispatch(fn, timeout, op)

        with mock.patch.object(gui, "load_dispatch", lambda: spy):
            ctx = ToolContext(project=self.proj, should_stop=lambda: False)
            with use_context(ctx):
                S.record(self.proj, doc="Doc")
        self.assertTrue(seen)
        self.assertTrue(all(callable(stop) for _, stop in seen))

    def test_a_failed_handler_takes_no_autosave_checkpoint_or_snapshot(self):
        rec = S.record(self.proj, doc="Doc", ok=False, text="failed")
        self.assertEqual((rec.checkpoint, rec.snapshot, rec.snapshot_note), ("", "", "step failed"))
        self.assertEqual(self.gui.ops, [])
        self.assertEqual(rec.replay["kind"], "none")

    def test_a_health_failure_still_takes_every_artifact(self):
        """ok is the HANDLER result: a call that left new health errors is still a full step."""
        rec = S.record(self.proj, doc="Doc", ok=True, health={"valid": False, "errors": 1, "warnings": 0,
                                                              "line": "[health] 1 error", "findings": []})
        self.assertEqual((rec.checkpoint, rec.snapshot), ("001", "snapshots/001_make-a-box.png"))
        self.assertFalse(rec.health["valid"])
        self.assertEqual(rec.replay["kind"], "script")

    def test_an_unchanged_step_skips_the_autosave_when_a_model_file_exists(self):
        S.record(self.proj, doc="Doc")
        saves = len(self.fake.saved_to)
        rec = S.record(self.proj, doc="Doc", delta={"added": [], "removed": [], "changed": []})
        self.assertEqual((rec.checkpoint, rec.checkpoint_note, rec.snapshot_note), ("", "unchanged", "unchanged"))
        self.assertEqual(len(self.fake.saved_to), saves)

    def test_an_empty_delta_still_saves_when_no_model_file_exists_yet(self):
        rec = S.record(self.proj, doc="Doc", delta={"added": [], "removed": [], "changed": []})
        self.assertEqual(rec.checkpoint, "001")

    def test_a_save_that_fails_is_a_visible_note_not_an_exception(self):
        self.fake.write = False
        rec = S.record(self.proj, doc="Doc")
        self.assertEqual(rec.checkpoint, "")
        self.assertIn("autosave failed", rec.checkpoint_note)
        self.assertGreater(rec.index, 0)

    def test_a_missing_document_is_reported(self):
        rec = S.record(self.proj, doc="Nope")
        self.assertEqual((rec.checkpoint, rec.checkpoint_note), ("", "no document"))

    def test_a_checkpoint_above_the_file_limit_is_skipped_with_a_note(self):
        self.proj.CHECKPOINT_MAX_BYTES = 10
        rec = S.record(self.proj, doc="Doc")
        self.assertEqual(rec.checkpoint, "")
        self.assertTrue(rec.checkpoint_note.startswith("skipped:"), rec.checkpoint_note)
        self.assertIn("per-file limit", rec.checkpoint_note)
        self.assertTrue(os.path.isfile(self.proj.path("model.FCStd")))   # the autosave itself is still there

    def test_a_second_document_goes_to_docs_not_model_fcstd(self):
        other = S.FakeDoc("Other")
        with mock.patch.object(gui, "resolve_doc", lambda name: {"Doc": self.fake, "Other": other}.get(name)):
            S.record(self.proj, doc="Doc")
            S.record(self.proj, doc="Other")
        self.assertTrue(os.path.isfile(self.proj.path("model.FCStd")))
        self.assertTrue(os.path.isfile(self.proj.path("docs", "Other.FCStd")))
        self.assertEqual(self.proj.manifest["active_doc"], "Doc")
        self.assertEqual(self.proj.manifest["docs"], ["Doc", "Other"])
        names = [e["name"] for e in self.proj.artifact_list()]
        self.assertIn("docs/Other.FCStd", names)

    def test_snapshot_failure_notes(self):
        for outcome, note in (((False, "blank frame"), "blank frame"),
                              ((False, "no 3D view (headless session)"), "no 3D view (headless session)")):
            proj = S.make_project(chat_id=f"20261001-000000-{abs(hash(note)) % 0xffffff:06x}",
                                  snapshot_fn=lambda *a, _o=outcome: _o)
            rec = S.record(proj, doc="Doc")
            self.assertEqual((rec.snapshot, rec.snapshot_note), ("", note))
        proj = S.make_project(chat_id="20261001-000000-aa0001", snapshot_fn=lambda *a: (True, ""))   # claims success, writes nothing
        self.assertEqual(S.record(proj, doc="Doc").snapshot_note, "snapshot produced no image")

    def test_empty_document_gets_no_snapshot(self):
        with mock.patch.object(gui, "has_geometry", lambda doc: False):
            rec = S.record(self.proj, doc="Doc")
        self.assertEqual((rec.snapshot, rec.snapshot_note), ("", "empty document"))
        self.assertEqual(rec.checkpoint, "001")   # the checkpoint is independent of the picture

    def test_the_snapshot_function_gets_the_documented_arguments(self):
        calls = []

        def snap(doc_name, path, width, height):
            calls.append((doc_name, os.path.basename(path), width, height))
            return S.tiny_png_snapshot(doc_name, path, width, height)

        proj = S.make_project(chat_id="20261001-000000-aa0002", snapshot_fn=snap)
        S.record(proj, doc="Doc")
        self.assertEqual(calls[0][0], "Doc")
        self.assertEqual(calls[0][2:], (480, 360))
        self.assertTrue(calls[0][1].endswith(".png"))
        self.assertIn(".tmp-", calls[0][1])   # the picture is written to a temp name and replaced atomically
        self.assertEqual(sorted(os.listdir(proj.path("snapshots"))), ["001_make-a-box.png"])


class LockDisciplineTest(unittest.TestCase):
    """No lock across a dispatch; readers are lock-free."""

    def test_the_gui_thread_can_read_and_take_the_lock_while_a_hook_waits_on_it(self):
        proj = S.make_project()
        fake = S.FakeDoc("Doc")
        timings = []

        def gui_side():
            started = time.monotonic()
            got_lock = proj._lock.acquire(timeout=1.0)   # would fail if the worker held the lock across the dispatch
            if got_lock:
                proj._lock.release()
            proj.read_steps()
            proj.artifact_list()
            proj.last_step_index()
            proj.last_geometry_step()
            proj.read_params()
            _ = proj.reference_status
            timings.append(time.monotonic() - started)
            return got_lock

        fake_gui = S.FakeGui(before_task=gui_side).install()
        self.addCleanup(fake_gui.uninstall)
        with mock.patch.object(gui, "resolve_doc", lambda name: fake), \
                mock.patch.object(gui, "has_geometry", lambda doc: True), \
                mock.patch.object(gui, "object_names", lambda doc: (["Box"], ["Box"])), \
                mock.patch.object(gui, "param_table", lambda doc: {"sheet": "", "rows": []}):
            worker = threading.Thread(target=lambda: S.record(proj, doc="Doc"))
            worker.start()
            worker.join(30)
        self.assertFalse(worker.is_alive())
        self.assertTrue(fake_gui.observations, "the GUI thread never ran a task")
        self.assertTrue(all(fake_gui.observations), "the Project lock was held while the worker waited on the GUI thread")
        self.assertLess(max(timings), 0.5)
        self.assertGreaterEqual(len(fake_gui.ops), 3)   # autosave + snapshot (+ checkpoint through the file copy)

    def test_reads_do_not_wait_for_a_writer_that_holds_the_lock(self):
        proj = S.make_project(create=True)
        S.record(proj)
        done = []
        with proj._lock:   # a writer in the middle of a publish, in another thread's view
            def reader():
                proj.read_steps()
                proj.artifact_list()
                proj.read_ledger()
                proj.read_notes()
                proj.read_spec()
                proj.read_validation()
                proj.read_provenance()
                proj.last_step_index()
                done.append(True)

            thread = threading.Thread(target=reader)
            thread.start()
            thread.join(5)
        self.assertEqual(done, [True])


class CheckpointPruningTest(unittest.TestCase):
    def setUp(self):
        self.proj = S.make_project(create=True)
        os.makedirs(self.proj.path("checkpoints"), exist_ok=True)

    def _geometry(self, index, size=10):
        S.new_temp_file(self.proj.path("checkpoints"), f"{index:03d}.FCStd", b"x" * size)
        with open(self.proj.path("checkpoints", f"{index:03d}.json"), "w") as f:
            json.dump({"step": index, "doc": "D", "objects": [], "ts": "2026-01-01T00:00:%02d" % (index % 60)}, f)

    def _turn(self, n, after_step=0, size=10):
        stem = f"T{n:04d}"
        S.new_temp_file(self.proj.path("checkpoints"), f"{stem}.FCStd", b"x" * size)
        with open(self.proj.path("checkpoints", f"{stem}.json"), "w") as f:
            json.dump({"turn": f"T-x-{stem}", "doc": "D", "after_step": after_step, "objects": [],
                       "ts": f"2026-01-01T00:{n // 60:02d}:{n % 60:02d}"}, f)

    def _ids(self):
        return sorted(os.path.splitext(n)[0] for n in os.listdir(self.proj.path("checkpoints")) if n.endswith(".FCStd"))

    def test_at_most_60_geometry_checkpoints_and_the_keep_rules(self):
        for i in range(1, 101):
            self._geometry(i)
        with self.proj._lock:
            self.proj._prune_checkpoints()
        kept = {int(i) for i in self._ids()}
        self.assertLessEqual(len(kept), 60)
        self.assertTrue(set(range(81, 101)) <= kept)                 # the newest 20 are never pruned
        self.assertTrue({5, 10, 15, 20, 25, 30} <= kept)             # every 5th is never pruned (<= the cap of 60)
        self.assertNotIn(1, kept)                                    # the oldest unprotected go first
        self.assertNotIn(2, kept)

    def test_turn_checkpoints_keep_the_newest_ten(self):
        for n in range(1, 16):
            self._turn(n)
        with self.proj._lock:
            self.proj._prune_checkpoints()
        self.assertEqual(self._ids(), [f"T{n:04d}" for n in range(6, 16)])

    def test_the_checkpoint_a_surviving_turn_refers_to_is_kept(self):
        for i in range(1, 80):
            self._geometry(i)
        self._turn(1, after_step=3)   # step 3 is the baseline of a surviving turn checkpoint
        with self.proj._lock:
            self.proj._prune_checkpoints()
        self.assertIn("003", self._ids())

    def test_the_byte_budget_bounds_the_folder_oldest_first(self):
        self.proj.CHECKPOINT_BUDGET_BYTES = 100
        for i in range(1, 31):
            self._geometry(i, size=10)   # 300 bytes in total
        with self.proj._lock:
            self.proj._prune_checkpoints()
        total = sum(os.path.getsize(self.proj.path("checkpoints", n)) for n in os.listdir(self.proj.path("checkpoints")) if n.endswith(".FCStd"))
        self.assertLessEqual(total, 100)
        self.assertIn("030", self._ids())                      # the newest always stays
        self.assertNotIn("001", self._ids())

    def test_the_rescue_copy_and_temp_files_are_never_pruned_or_counted(self):
        S.new_temp_file(self.proj.path("checkpoints"), "_rescue.FCStd", b"x" * 1000)
        S.new_temp_file(self.proj.path("checkpoints"), "007.tmp-1-ab.FCStd", b"x" * 1000)
        self.proj.CHECKPOINT_BUDGET_BYTES = 50
        self._geometry(1)
        with self.proj._lock:
            self.proj._prune_checkpoints()
        self.assertIn("_rescue", self._ids())
        self.assertIn("007.tmp-1-ab", self._ids())
        self.assertIn("001", self._ids())

    def test_end_turn_prunes(self):
        for n in range(1, 14):
            self._turn(n)
        self.proj.end_turn(S.TURN, "answered")
        self.assertEqual(len(self._ids()), 10)

    def test_pruning_an_empty_folder_is_a_no_op(self):
        with self.proj._lock:
            self.proj._prune_checkpoints()
        self.assertEqual(self._ids(), [])


class VerifyAndFinalizeTest(unittest.TestCase):
    def setUp(self):
        self.proj = S.make_project()

    def test_verify_artifacts_always_requires_the_core_files(self):
        self.proj.ensure_dir()
        self.assertEqual(self.proj.verify_artifacts(), ["steps.jsonl", "ledger.json", "NOTES.md"])

    def test_geometry_steps_require_model_and_rebuild(self):
        S.record(self.proj)   # doc="" -> no autosave happened
        self.proj.save_ledger({"intent": "other"})
        missing = self.proj.verify_artifacts()
        self.assertEqual(missing, ["model.FCStd"])   # NOTES.md and rebuild.py were generated by the hook

    def test_provenance_is_required_when_the_ledger_needs_a_reference(self):
        S.record(self.proj)
        self.proj.save_ledger({"intent": "other", "needs_reference": True})
        self.assertIn("provenance.json", self.proj.verify_artifacts())
        self.proj.provenance_add(kind="reference", claim="c", source="s", status="nominal")
        self.assertNotIn("provenance.json", self.proj.verify_artifacts())

    def test_an_exportable_intent_requires_an_export(self):
        S.record(self.proj)
        self.proj.save_ledger({"intent": "enclosure"})
        self.assertIn("exports/*", self.proj.verify_artifacts())
        S.new_temp_file(self.proj.path("exports"), "a.step")
        self.assertNotIn("exports/*", self.proj.verify_artifacts())
        self.proj.save_ledger({"intent": "modify"})
        self.assertNotIn("exports/*", self.proj.verify_artifacts())

    def test_parameter_tools_require_params_json(self):
        S.record(self.proj, tool="set_parameters", geometry=True, args={"doc_name": "D", "params": []})
        self.proj.save_ledger({"intent": "other"})
        self.assertIn("params.json", self.proj.verify_artifacts())

    def test_finalize_writes_the_documents_and_the_manifest_without_exporting(self):
        S.record(self.proj, tool="create_object", args={"doc_name": "D", "obj_type": "Part::Box"})
        summary = self.proj.finalize(ledger={"intent": "enclosure", "request": "build a case", "steps": [],
                                             "requirements": []}, outcome="validated", validation=None, export=False)
        self.assertTrue(summary.markdown.startswith("## Build Summary"))
        self.assertEqual(summary.data["outcome"], "validated")
        for name in ("NOTES.md", "rebuild.py", "ledger.json"):
            self.assertTrue(os.path.isfile(self.proj.path(name)), name)
        final = json.load(open(self.proj.path("project.json")))["finalized"]
        self.assertEqual(final["outcome"], "validated")
        self.assertIn("ts", final)
        self.assertFalse(os.path.isdir(self.proj.path("exports")) and os.listdir(self.proj.path("exports")))

    def test_finalize_accepts_a_ledger_object_with_to_dict(self):
        class Ledger:
            def to_dict(self):
                return {"intent": "other", "steps": [], "requirements": [], "request": "hello"}

        summary = self.proj.finalize(ledger=Ledger(), outcome="answered", validation=None)
        self.assertIn("hello", open(self.proj.path("NOTES.md"), encoding="utf-8").read())
        self.assertEqual(summary.data["outcome"], "answered")

    def test_an_auto_export_failure_becomes_an_unresolved_note_never_an_exception(self):
        self.proj.manifest["active_doc"] = "Gone"
        self.proj.ensure_dir()
        with mock.patch("oscar.project.export_tips", side_effect=RuntimeError("no writer")):
            self.proj.finalize(ledger={"intent": "enclosure", "steps": [], "requirements": []}, outcome="validated",
                               validation=None)
        notes = self.proj.read_notes()
        self.assertEqual(notes[-1]["kind"], "unresolved")
        self.assertIn("auto-export failed", notes[-1]["text"])
        self.assertIn("auto-export failed", open(self.proj.path("NOTES.md"), encoding="utf-8").read())

    def test_auto_export_runs_only_for_the_exporting_intents_and_only_when_exports_is_empty(self):
        self.proj.manifest["active_doc"] = "Doc"
        self.proj.ensure_dir()
        calls = []

        class Result:
            errors = ["bad.stl: boom"]

        def fake_export(project, doc, **kw):
            calls.append((doc, kw["formats"]))
            return Result()

        with mock.patch("oscar.project.export_tips", fake_export):
            self.proj.finalize(ledger={"intent": "inspect", "steps": []}, outcome="answered", validation=None)
            self.assertEqual(calls, [])
            self.proj.finalize(ledger={"intent": "enclosure", "steps": []}, outcome="validated", validation=None)
            self.assertEqual(calls, [("Doc", ("step", "stl"))])
            self.assertEqual(self.proj.read_notes()[-1]["text"], "auto-export problem: bad.stl: boom")
            S.new_temp_file(self.proj.path("exports"), "x.step")
            self.proj.finalize(ledger={"intent": "enclosure", "steps": []}, outcome="validated", validation=None)
            self.assertEqual(len(calls), 1)   # exports/ already holds a file
            self.proj.finalize(ledger={"intent": "inspect", "steps": []}, outcome="answered", validation=None, export=True)
            self.assertEqual(len(calls), 2)   # export=True forces it


class LifecycleNeverRaisesTest(unittest.TestCase):
    """The agent must never die because the dossier failed: storage failures are logged, only an
    invalid ARGUMENT raises ``ProjectError``."""

    def unwritable(self):
        blocker = S.new_temp_file(S.sandbox_dir(), "iamafile")
        proj = P.Project(os.path.join(blocker, "projects", "x-abc123"), chat_id=S.CHAT_ID, title="x")
        patcher = mock.patch.object(paths, "fallback_projects_dir", return_value=os.path.join(blocker, "fb"))
        patcher.start()
        self.addCleanup(patcher.stop)
        return proj

    def test_storage_error_is_a_project_error(self):
        self.assertTrue(issubclass(P.ProjectStorageError, P.ProjectError))
        with self.assertRaises(P.ProjectStorageError):
            self.unwritable().ensure_dir()

    def test_begin_turn_on_an_unwritable_disk_does_not_raise(self):
        proj = self.unwritable()
        with mock.patch("oscar.project.log_event") as log:
            self.assertIsNone(proj.begin_turn(S.TURN, "build", work=True))
        self.assertTrue(any(c.args[0] == "project_error" and c.kwargs["where"] == "begin_turn" for c in log.call_args_list))
        self.assertEqual(proj._turn_id, S.TURN)   # the turn is still tracked in memory

    def test_end_turn_with_a_full_disk_does_not_raise(self):
        proj = S.make_project(create=True)
        with mock.patch.object(P.Project, "_append_raw", side_effect=OSError("disk full")), \
                mock.patch("oscar.project.log_event") as log:
            self.assertIsNone(proj.end_turn(S.TURN, "answered", rounds=2))
        self.assertTrue(any(c.kwargs.get("where") == "end_turn" for c in log.call_args_list))

    def test_save_ledger_logs_disk_errors_but_still_rejects_a_bad_argument(self):
        proj = S.make_project(create=True)
        with mock.patch("oscar.project._atomic_write_json", side_effect=OSError("disk full")), \
                mock.patch("oscar.project.log_event") as log:
            self.assertIsNone(proj.save_ledger({"intent": "other"}))
        self.assertTrue(any(c.kwargs.get("where") == "save_ledger" for c in log.call_args_list))
        with self.assertRaises(P.ProjectError):
            proj.save_ledger(42)

    def test_archive_ledger_and_record_validation_do_not_raise_on_disk_errors(self):
        proj = S.make_project(create=True)
        proj.save_ledger({"intent": "other"})
        with mock.patch("oscar.project.os.replace", side_effect=OSError("denied")), mock.patch("oscar.project.log_event"):
            self.assertIsNone(proj.archive_ledger("T1"))
        with mock.patch("oscar.project._atomic_write_json", side_effect=OSError("denied")), mock.patch("oscar.project.log_event"):
            self.assertIsNone(proj.record_validation({"ok": True, "ran": ["health"]}))

    def test_ensure_turn_checkpoint_on_an_unwritable_disk_returns_empty(self):
        proj = self.unwritable()
        proj.begin_turn(S.TURN, "x")
        with mock.patch("oscar.project.log_event"):
            self.assertEqual(proj.ensure_turn_checkpoint("Doc"), "")

    def test_a_gui_timeout_does_not_disable_the_turn_checkpoint_for_the_whole_turn(self):
        """Review finding (PR #3): one transient GUI timeout used to cache '' for the (turn, doc) and silently
        remove the rewind safety net for the rest of the turn; the next call must capture again."""
        proj = S.make_project(create=True)
        proj.begin_turn(S.TURN, "x")
        suffix = "T" + S.TURN[-4:]
        timeout = (None, "'project_turn_checkpoint' did not complete within 20s")
        ok = ({"id": suffix, "doc": "Doc", "existing": True}, "")
        with mock.patch("oscar.project.gui.gui_call", side_effect=[timeout, ok]) as call, \
                mock.patch("oscar.project.log_event"):
            self.assertEqual(proj.ensure_turn_checkpoint("Doc"), "")        # busy GUI: nothing captured ...
            self.assertEqual(proj.ensure_turn_checkpoint("Doc"), suffix)    # ... but the next call retries and succeeds
            self.assertEqual(proj.ensure_turn_checkpoint("Doc"), suffix)    # and a success IS cached
        self.assertEqual(call.call_count, 2)

    def test_transient_checkpoint_failures_stop_retrying_after_the_cap(self):
        proj = S.make_project(create=True)
        proj.begin_turn(S.TURN, "x")
        timeout = (None, "'project_turn_checkpoint' did not complete within 20s")
        with mock.patch("oscar.project.gui.gui_call", return_value=timeout) as call, mock.patch("oscar.project.log_event"):
            for _ in range(proj.TURN_CKPT_RETRIES + 3):
                self.assertEqual(proj.ensure_turn_checkpoint("Doc"), "")
        self.assertEqual(call.call_count, proj.TURN_CKPT_RETRIES)           # a permanently failing capture is bounded

    def test_a_save_failure_note_is_transient_too(self):
        proj = S.make_project(create=True)
        proj.begin_turn(S.TURN, "x")
        suffix = "T" + S.TURN[-4:]
        failed = ({"note": "save failed", "doc": "Doc"}, "")
        ok = ({"id": suffix, "doc": "Doc", "existing": True}, "")
        with mock.patch("oscar.project.gui.gui_call", side_effect=[failed, ok]), mock.patch("oscar.project.log_event"):
            self.assertEqual(proj.ensure_turn_checkpoint("Doc"), "")
            self.assertEqual(proj.ensure_turn_checkpoint("Doc"), suffix)

    def test_finalize_still_returns_the_summary_when_the_files_cannot_be_written(self):
        proj = S.make_project(create=True)
        with mock.patch.object(P.Project, "_finalize_files", side_effect=OSError("disk full")), \
                mock.patch("oscar.project.log_event") as log:
            summary = proj.finalize(ledger={"intent": "other", "steps": [], "requirements": []}, outcome="answered",
                                    validation=None)
        self.assertTrue(summary.markdown.startswith("## Build Summary"))
        self.assertTrue(any(c.kwargs.get("where") == "finalize" for c in log.call_args_list))

    def test_finalize_rejects_a_wrongly_typed_argument_before_writing_anything(self):
        proj = S.make_project()
        with self.assertRaises(P.ProjectError):
            proj.finalize(ledger=42, outcome="answered", validation=None)
        with self.assertRaises(P.ProjectError):
            proj.finalize(ledger=None, outcome="answered", validation="not a dict")
        self.assertFalse(os.path.exists(proj.root))

    def test_the_hook_survives_an_unwritable_disk(self):
        proj = self.unwritable()
        with mock.patch("oscar.project.log_event"):
            rec = S.record(proj)
        self.assertEqual(rec.index, -1)
        self.assertIn("project error", rec.snapshot_note)


class PureImportTest(unittest.TestCase):
    """project.py and its private modules are PURE: FreeCAD / Qt only inside functions."""

    def test_no_top_level_freecad_or_qt_import(self):
        import ast

        base = os.path.join(os.path.dirname(__file__), "..", "oscar")
        banned = {"FreeCAD", "FreeCADGui", "PySide", "PySide2", "PySide6", "Part", "Mesh"}
        for name in ("project.py", "project_docs.py", "project_export.py", "project_gui.py", "project_replay.py",
                     "project_summary.py", "project_util.py", "paths.py", "chats.py"):
            tree = ast.parse(open(os.path.join(base, name), encoding="utf-8").read())
            for node in tree.body:   # module level only: imports inside functions are the point
                modules = []
                if isinstance(node, ast.Import):
                    modules = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    modules = [(node.module or "").split(".")[0]]
                self.assertEqual(banned & set(modules), set(), f"{name} imports {modules} at module level")


if __name__ == "__main__":
    unittest.main()
