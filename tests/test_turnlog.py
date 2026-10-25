"""oscar.turnlog: where the log lives, what every line carries, context, rotation,
tail, and the guarantees that logging never raises and never leaks keys.

Pure python: runs under the bundled interpreter and under freecadcmd alike.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

from oscar import turnlog  # noqa: E402


def _lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class LogCase(unittest.TestCase):
    """A private log directory per test, the context cleared before and after."""

    def setUp(self):
        self.dir = isolation.fresh_dir("turnlog_")
        patcher = mock.patch.dict(os.environ, {"OSCAR_LOG_DIR": self.dir})
        patcher.start()
        self.addCleanup(patcher.stop)
        turnlog.clear_context()
        self.addCleanup(turnlog.clear_context)
        self.path = os.path.join(self.dir, "agent.log")


class LocationTest(LogCase):
    def test_override_is_used_and_read_at_call_time(self):
        turnlog.log_event("one")
        other = isolation.fresh_dir("turnlog_other_")
        with mock.patch.dict(os.environ, {"OSCAR_LOG_DIR": other}):
            self.assertEqual(turnlog.log_path(), os.path.join(other, "agent.log"))
            turnlog.log_event("two")
        self.assertEqual([e["kind"] for e in _lines(self.path)], ["one"])
        self.assertEqual([e["kind"] for e in _lines(os.path.join(other, "agent.log"))], ["two"])

    def test_directory_is_created(self):
        nested = os.path.join(self.dir, "a", "b")
        with mock.patch.dict(os.environ, {"OSCAR_LOG_DIR": nested}):
            turnlog.log_event("x")
            self.assertTrue(os.path.isfile(os.path.join(nested, "agent.log")))

    def test_under_test_without_override_redirects_to_a_temp_dir(self):
        env = {k: v for k, v in os.environ.items() if k != "OSCAR_LOG_DIR"}
        with mock.patch.dict(os.environ, env, clear=True):
            expect = os.path.join(tempfile.gettempdir(), f"oscar-test-logs-{os.getpid()}")
            self.addCleanup(shutil.rmtree, expect, ignore_errors=True)
            self.assertEqual(turnlog.log_path(), os.path.join(expect, "agent.log"))
            turnlog.log_event("redirected", n=1)
            self.assertEqual(turnlog.tail(1)[0]["kind"], "redirected")   # tail reads the redirected file
            self.assertTrue(os.path.isfile(os.path.join(expect, "agent.log")))

    def test_real_log_is_never_used_under_unittest(self):
        env = {k: v for k, v in os.environ.items() if k != "OSCAR_LOG_DIR"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.addCleanup(shutil.rmtree, os.path.join(tempfile.gettempdir(), f"oscar-test-logs-{os.getpid()}"),
                            ignore_errors=True)
            with mock.patch.object(turnlog, "_default_dir", side_effect=AssertionError("real log dir touched")):
                path = turnlog.log_path()            # would raise if it asked for the real directory
                turnlog.log_event("safe")
            self.assertTrue(path.startswith(tempfile.gettempdir()))

    def test_a_live_gui_session_is_never_redirected(self):
        fake = types.SimpleNamespace(GuiUp=1)
        with mock.patch.dict(sys.modules, {"FreeCAD": fake}):
            self.assertFalse(turnlog.under_test())
        with mock.patch.dict(sys.modules, {"FreeCAD": types.SimpleNamespace(GuiUp=0)}):
            self.assertTrue(turnlog.under_test())

    def test_pytest_marker_counts_as_a_test_run(self):
        with mock.patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "x::y"}):
            with mock.patch.dict(sys.modules, {"FreeCAD": types.SimpleNamespace(GuiUp=1)}):
                self.assertTrue(turnlog.under_test())

    def test_default_directory_is_the_app_support_dir(self):
        fake = types.SimpleNamespace(getUserAppDataDir=lambda: "/data/fc/")
        with mock.patch.dict(sys.modules, {"FreeCAD": fake}):
            self.assertEqual(turnlog._default_dir(), os.path.join("/data/fc/", "Oscar", "logs"))
        with mock.patch.dict(sys.modules, {"FreeCAD": None}):    # import raises ImportError
            self.assertTrue(turnlog._default_dir().endswith(os.path.join("Oscar", "logs")))


class LineShapeTest(LogCase):
    def test_every_line_carries_ts_kind_and_src(self):
        turnlog.log_event("hello", n=3, name="x")
        (entry,) = _lines(self.path)
        self.assertEqual((entry["kind"], entry["src"], entry["n"], entry["name"]), ("hello", "test", 3, "x"))
        self.assertRegex(entry["ts"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$")
        self.assertEqual(list(entry)[:3], ["ts", "kind", "src"])

    def test_src_is_app_outside_a_test_run(self):
        with mock.patch.object(turnlog, "under_test", return_value=False):
            turnlog.log_event("prod")
        self.assertEqual(_lines(self.path)[0]["src"], "app")

    def test_fields_cannot_override_the_reserved_keys(self):
        turnlog.log_event("real", ts="evil", src="evil", extra=1)
        (entry,) = _lines(self.path)
        self.assertNotEqual(entry["ts"], "evil")
        self.assertEqual((entry["kind"], entry["src"], entry["extra"]), ("real", "test", 1))

    def test_unserialisable_values_are_stringified(self):
        turnlog.log_event("odd", obj=object(), path=os.path, tup=(1, 2), nested={"a": {1, 2}})
        (entry,) = _lines(self.path)
        self.assertIn("object", entry["obj"])
        self.assertEqual(entry["tup"], [1, 2])

    def test_non_ascii_survives_a_round_trip(self):
        turnlog.log_event("uni", text="Gehaeuse \u00e4\u00f6\u00fc \u2713")
        self.assertEqual(turnlog.tail(1)[0]["text"], "Gehaeuse \u00e4\u00f6\u00fc \u2713")

    def test_keys_are_masked_even_when_nested(self):
        turnlog.log_event("auth", api_key="sk-secret", Authorization="Bearer x", profile={"refresh_token": "r", "model": "m"},
                          items=[{"password": "p"}], empty_key="", api_token_count=3)
        (entry,) = _lines(self.path)
        self.assertEqual(entry["api_key"], "[redacted]")
        self.assertEqual(entry["Authorization"], "[redacted]")
        self.assertEqual(entry["profile"], {"refresh_token": "[redacted]", "model": "m"})
        self.assertEqual(entry["items"], [{"password": "[redacted]"}])
        self.assertNotIn("sk-secret", _text(self.path))
        self.assertEqual(entry["api_token_count"], 3)

    def test_a_blank_key_is_not_reported_as_redacted(self):
        turnlog.log_event("auth", profile={"api_key": ""})
        self.assertEqual(_lines(self.path)[0]["profile"], {"api_key": ""})

    def test_long_strings_are_clipped(self):
        big = "x" * (turnlog.MAX_VALUE_CHARS + 500)
        turnlog.log_event("big", code=big, short="ok")
        (entry,) = _lines(self.path)
        self.assertTrue(entry["code"].startswith("x" * turnlog.MAX_VALUE_CHARS))
        self.assertIn("+500 chars", entry["code"])
        self.assertEqual(entry["short"], "ok")
        self.assertLess(os.path.getsize(self.path), turnlog.MAX_VALUE_CHARS + 600)


class ContextTest(LogCase):
    def test_context_is_merged_into_every_event(self):
        turnlog.set_context(turn_id="T1", chat_id="c1", model="m")
        turnlog.log_event("a")
        turnlog.log_event("b", n=1)
        rows = _lines(self.path)
        self.assertEqual([(r["turn_id"], r["chat_id"], r["model"]) for r in rows], [("T1", "c1", "m")] * 2)
        self.assertEqual(rows[1]["n"], 1)

    def test_set_context_accumulates_and_none_removes(self):
        turnlog.set_context(turn_id="T1", model="m")
        turnlog.set_context(chat_id="c1", model=None)
        self.assertEqual(turnlog.get_context(), {"turn_id": "T1", "chat_id": "c1"})

    def test_clear_context_stops_the_merge(self):
        turnlog.set_context(turn_id="T1")
        turnlog.log_event("in")
        turnlog.clear_context()
        turnlog.log_event("out")
        inside, outside = _lines(self.path)
        self.assertEqual(inside["turn_id"], "T1")
        self.assertNotIn("turn_id", outside)
        self.assertEqual(turnlog.get_context(), {})

    def test_event_fields_win_over_the_context(self):
        turnlog.set_context(turn_id="ctx")
        turnlog.log_event("x", turn_id="own")
        self.assertEqual(_lines(self.path)[0]["turn_id"], "own")

    def test_get_context_returns_a_copy(self):
        turnlog.set_context(a=1)
        snap = turnlog.get_context()
        snap["a"] = 2
        self.assertEqual(turnlog.get_context(), {"a": 1})

    def test_context_values_are_scrubbed_too(self):
        turnlog.set_context(api_key="sk-leak")
        turnlog.log_event("x")
        self.assertNotIn("sk-leak", _text(self.path))


class RotationTest(LogCase):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(turnlog, "ROTATE_BYTES", 400)
        p.start()
        self.addCleanup(p.stop)

    def _fill(self, count, tag="e"):
        for i in range(count):
            turnlog.log_event(tag, i=i, pad="p" * 60)

    def test_rotates_when_the_size_limit_is_reached(self):
        self._fill(12)
        self.assertTrue(os.path.isfile(self.path + ".1"))
        self.assertLess(os.path.getsize(self.path), 400 + 200)

    def test_keeps_exactly_three_generations(self):
        self._fill(200)
        names = sorted(n for n in os.listdir(self.dir))
        self.assertEqual(names, ["agent.log", "agent.log.1", "agent.log.2", "agent.log.3"])

    def test_newest_events_are_in_the_live_file_and_oldest_are_dropped(self):
        self._fill(200)
        newest = _lines(self.path)[-1]["i"]
        self.assertEqual(newest, 199)
        first_kept = _lines(self.path + ".3")[0]["i"]
        self.assertGreater(first_kept, 0)   # event 0 fell off the end

    def test_generations_are_ordered_oldest_to_newest(self):
        self._fill(60)
        gens = [_lines(self.path + f".{i}") for i in (3, 2, 1)] + [_lines(self.path)]
        flat = [e["i"] for g in gens for e in g]
        self.assertEqual(flat, sorted(flat))

    def test_tail_continues_into_rotated_files(self):
        self._fill(10)                                   # about four lines per generation
        self.assertTrue(os.path.isfile(self.path + ".1"))
        self.assertEqual([e["i"] for e in turnlog.tail(10)], list(range(10)))
        self.assertEqual([e["i"] for e in turnlog.tail(6)], list(range(4, 10)))

    def test_rotation_failure_does_not_lose_the_event(self):
        with open(self.path, "w", encoding="utf-8") as f:     # already over the (patched) limit
            f.write(json.dumps({"kind": "old", "i": 1, "pad": "p" * 500}) + "\n")
        with mock.patch("os.replace", side_effect=OSError("locked")):
            turnlog.log_event("after", i=999)
        self.assertEqual(turnlog.tail(1)[0]["i"], 999)
        self.assertEqual([e["kind"] for e in turnlog.tail(5)], ["old", "after"])

    def test_small_files_are_not_rotated(self):
        with mock.patch.object(turnlog, "ROTATE_BYTES", 10 * 1024 * 1024):
            self._fill(50)
        self.assertFalse(os.path.exists(self.path + ".1"))


class TailTest(LogCase):
    def test_empty_when_nothing_was_logged(self):
        self.assertEqual(turnlog.tail(10), [])

    def test_returns_the_most_recent_oldest_first(self):
        for i in range(10):
            turnlog.log_event("n", i=i)
        self.assertEqual([e["i"] for e in turnlog.tail(3)], [7, 8, 9])
        self.assertEqual(len(turnlog.tail()), 10)

    def test_zero_or_negative_limit(self):
        turnlog.log_event("x")
        self.assertEqual(turnlog.tail(0), [])
        self.assertEqual(turnlog.tail(-5), [])

    def test_garbage_lines_are_skipped_not_fatal(self):
        turnlog.log_event("good", i=1)
        with open(self.path, "ab") as f:
            f.write(b"this is not json\n\n\xff\xfe broken \n[1, 2]\n")
        turnlog.log_event("good", i=2)
        self.assertEqual([e["i"] for e in turnlog.tail(10)], [1, 2])

    def test_large_file_is_read_from_the_end(self):
        with open(self.path, "w", encoding="utf-8") as f:
            for i in range(5000):
                f.write(json.dumps({"kind": "bulk", "i": i, "pad": "z" * 100}) + "\n")
        got = turnlog.tail(3)
        self.assertEqual([e["i"] for e in got], [4997, 4998, 4999])

    def test_a_partial_first_line_in_the_window_is_dropped(self):
        with open(self.path, "w", encoding="utf-8") as f:
            for i in range(2000):
                f.write(json.dumps({"kind": "bulk", "i": i}) + "\n")
        for entry in turnlog.tail(1500):
            self.assertEqual(entry["kind"], "bulk")


class RobustnessTest(LogCase):
    def test_log_event_never_raises_when_the_directory_is_unusable(self):
        blocker = os.path.join(self.dir, "afile")
        with open(blocker, "w") as f:
            f.write("x")
        with mock.patch.object(turnlog, "_WARNED", True), \
                mock.patch.dict(os.environ, {"OSCAR_LOG_DIR": os.path.join(blocker, "sub")}):
            turnlog.log_event("lost")                    # the parent is a file: makedirs fails
            self.assertEqual(turnlog.tail(5), [])

    def test_the_failure_is_reported_once_on_stderr(self):
        blocker = os.path.join(self.dir, "afile")
        with open(blocker, "w") as f:
            f.write("x")
        err = []
        with mock.patch.object(turnlog, "_WARNED", False), \
                mock.patch.object(sys, "stderr", types.SimpleNamespace(write=err.append, flush=lambda: None)), \
                mock.patch.dict(os.environ, {"OSCAR_LOG_DIR": os.path.join(blocker, "sub")}):
            for _ in range(5):
                turnlog.log_event("lost")
        self.assertEqual(len(err), 1)
        self.assertIn("turnlog", err[0])

    def test_a_non_string_kind_is_logged_as_a_string(self):
        turnlog.log_event(42)
        self.assertEqual(_lines(self.path)[0]["kind"], "42")

    def test_concurrent_writers_never_interleave_lines(self):
        def writer(tag):
            for i in range(60):
                turnlog.log_event("w", tag=tag, i=i, pad="q" * 80)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        rows = _lines(self.path)           # json.loads on every line: an interleaved line would raise
        self.assertEqual(len(rows), 8 * 60)
        for tag in range(8):
            self.assertEqual([r["i"] for r in rows if r["tag"] == tag], list(range(60)))

    def test_context_changes_race_safely_with_logging(self):
        stop = threading.Event()

        def flip():
            n = 0
            while not stop.is_set():
                turnlog.set_context(turn_id=f"T{n}")
                n += 1
                turnlog.clear_context()

        t = threading.Thread(target=flip)
        t.start()
        try:
            for i in range(200):
                turnlog.log_event("race", i=i)
        finally:
            stop.set()
            t.join(5)
        self.assertEqual(len(_lines(self.path)), 200)


class AgentContractTest(LogCase):
    """The names the agent relies on: ``set_context`` and ``clear_context`` exist and are callable
    with exactly the keyword arguments agent.py passes."""

    def test_agent_call_shape(self):
        turnlog.set_context(turn_id="T20261001-214012-9f3a", chat_id="abc123", model="claude")
        turnlog.log_event("turn_start", intent="enclosure")
        row = turnlog.tail(1)[0]
        self.assertEqual((row["turn_id"], row["chat_id"], row["model"], row["intent"]),
                         ("T20261001-214012-9f3a", "abc123", "claude", "enclosure"))
        turnlog.clear_context()
        turnlog.log_event("turn_end")
        self.assertNotIn("turn_id", turnlog.tail(1)[0])


if __name__ == "__main__":
    unittest.main()
