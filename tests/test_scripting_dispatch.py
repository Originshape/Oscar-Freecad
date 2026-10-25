"""The GUI-thread dispatcher (``oscar.cad.dispatch``: abandoned requests, the ``should_stop`` poll, the STILL RUNNING
timeout text) and the crash-isolated headless helper (``oscar.cad.headless``).

The dispatcher is exercised through its QUEUED path with a stand-in for the GUI thread (a thread that drains the
queue), because under ``freecadcmd`` the real dispatcher runs everything inline. The headless tests start a real
``freecadcmd`` helper process."""

import os
import signal
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.cad import dispatch, headless  # noqa: E402


# ---------------------------------------------------------------------------------------------- dispatcher


class QueuedDispatchCase(unittest.TestCase):
    """Runs ``dispatch.dispatch`` through the queue. ``self.gui`` decides what "wake the GUI thread" does."""

    def setUp(self):
        self._threads: list[threading.Thread] = []
        self._patches = [
            mock.patch.object(dispatch, "_run_inline", return_value=False),
            mock.patch.object(dispatch, "init", lambda: None),
            mock.patch.object(dispatch, "_wake", self._wake),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)
        self.gui = "drain"     # "drain": a GUI thread that drains at once; "busy": nobody drains
        self.drain_started = threading.Event()
        self.addCleanup(self._join)

    def _wake(self):
        if self.gui == "drain":
            t = threading.Thread(target=self._drain_thread, daemon=True)
            self._threads.append(t)
            t.start()

    def _drain_thread(self):
        self.drain_started.set()
        dispatch.drain_queue()

    def _join(self):
        for t in self._threads:
            t.join(5)
        self.assertFalse(any(t.is_alive() for t in self._threads), "a drain thread is stuck")
        dispatch.drain_queue()   # nothing may linger for the next test


class QueuedDispatchTest(QueuedDispatchCase):
    def test_result_comes_back_through_the_queue(self):
        threads = []
        result, failed = dispatch.dispatch(lambda: threads.append(threading.current_thread().name) or 42, timeout=5)
        self.assertEqual((result, failed), (42, False))
        self.assertNotEqual(threads[0], threading.current_thread().name)    # it really ran on the "GUI" thread

    def test_an_exception_comes_back_as_text_and_the_drain_survives(self):
        result, failed = dispatch.dispatch(lambda: 1 / 0, timeout=5)
        self.assertEqual((result, failed), ("ZeroDivisionError: division by zero", True))
        self.assertEqual(dispatch.dispatch(lambda: "next", timeout=5), ("next", False))

    def test_system_exit_in_a_task_does_not_kill_the_drain(self):
        def quit_():
            raise SystemExit(3)

        result, failed = dispatch.dispatch(quit_, timeout=5)
        self.assertTrue(failed)
        self.assertEqual(dispatch.dispatch(lambda: 1, timeout=5), (1, False))

    def test_concurrent_callers_get_their_own_results(self):
        out = {}

        def call(i):
            out[i] = dispatch.dispatch(lambda: i * i, timeout=5)

        workers = [threading.Thread(target=call, args=(i,)) for i in range(8)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(10)
        self.assertEqual(out, {i: (i * i, False) for i in range(8)})

    def test_timeout_before_the_task_started_cancels_it(self):
        self.gui = "busy"
        ran = []
        t0 = time.monotonic()
        result, failed = dispatch.dispatch(lambda: ran.append(1), timeout=0.3, op="create_object")
        self.assertLess(time.monotonic() - t0, 3)
        self.assertTrue(failed)
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["still_running"])
        self.assertTrue(result["error"].startswith("'create_object' did not complete within 0.3s (the GUI thread stayed busy)"))
        self.assertIn("never started and has been cancelled", result["error"])
        self.assertNotIn("STILL RUNNING", result["error"])
        dispatch.drain_queue()    # the GUI thread finally gets to it: the abandoned task never runs
        self.assertEqual(ran, [])

    def test_timeout_of_a_running_task_says_still_running_and_the_task_finishes(self):
        release = threading.Event()
        finished = threading.Event()

        def long_task():
            release.wait(10)
            finished.set()
            return "late"

        result, failed = dispatch.dispatch(long_task, timeout=0.4, op="execute_code")
        self.assertTrue(failed)
        self.assertTrue(result["timed_out"] and result["still_running"])
        self.assertIn("did not complete within 0.4s (the GUI thread stayed busy)", result["error"])
        self.assertIn("STILL RUNNING on the GUI thread and cannot be cancelled: do not re-run it", result["error"])
        self.assertFalse(finished.is_set())
        release.set()
        self.assertTrue(finished.wait(5))     # it was NOT cancelled

    def test_should_stop_returns_stopped_quickly_and_abandons_a_queued_task(self):
        self.gui = "busy"
        flag = threading.Event()
        threading.Timer(0.3, flag.set).start()
        ran = []
        t0 = time.monotonic()
        result, failed = dispatch.dispatch(lambda: ran.append(1), timeout=30, op="x", should_stop=flag.is_set)
        self.assertLess(time.monotonic() - t0, 3)
        self.assertEqual((result, failed), ({"success": False, "error": "stopped", "stopped": True}, True))
        dispatch.drain_queue()
        self.assertEqual(ran, [])

    def test_should_stop_does_not_cancel_a_task_that_already_started(self):
        release, finished = threading.Event(), threading.Event()
        flag = threading.Event()

        def task():
            flag.set()
            release.wait(10)
            finished.set()

        result, failed = dispatch.dispatch(task, timeout=30, should_stop=flag.is_set)
        self.assertEqual((result, failed), ({"success": False, "error": "stopped", "stopped": True}, True))
        release.set()
        self.assertTrue(finished.wait(5))

    def test_a_should_stop_that_raises_counts_as_false(self):
        def boom():
            raise RuntimeError("x")

        self.assertEqual(dispatch.dispatch(lambda: "ok", timeout=5, should_stop=boom), ("ok", False))

    def test_a_timeout_in_one_call_never_poisons_another(self):
        self.gui = "busy"
        dispatch.dispatch(lambda: "lost", timeout=0.2)
        self.gui = "drain"
        self.assertEqual(dispatch.dispatch(lambda: "fine", timeout=5), ("fine", False))

    def test_no_timeout_means_wait_until_done(self):
        self.assertEqual(dispatch.dispatch(lambda: time.sleep(0.4) or "done", timeout=None), ("done", False))


class DrainQueueTest(unittest.TestCase):
    def test_an_abandoned_request_is_dropped_without_running_and_released(self):
        ran = []
        request = dispatch._Request(lambda: ran.append(1))
        request.abandoned = True
        dispatch._pending.put(request)
        dispatch.drain_queue()
        self.assertEqual(ran, [])
        self.assertTrue(request.done.is_set())
        self.assertFalse(request.started)

    def test_a_wanted_request_runs_once_and_is_marked_started(self):
        request = dispatch._Request(lambda: "r")
        dispatch._pending.put(request)
        dispatch.drain_queue()
        dispatch.drain_queue()
        self.assertEqual((request.result, request.failed, request.started), ("r", False, True))

    def test_request_has_the_documented_slots(self):
        self.assertTrue({"abandoned", "started", "done", "reentrant"} <= set(dispatch._Request.__slots__))
        self.assertFalse(dispatch._Request(lambda: 1).reentrant)


class DrainReentrancyTest(unittest.TestCase):
    """P0 #6: a nested event loop inside a running GUI task (updateGui, processEvents, a progress bar, a modal dialog)
    must not run another queued task inside it: a failing script's rollback would delete that task's work."""

    def setUp(self):
        self._empty_queue()
        self.addCleanup(self._empty_queue)
        self.order: list[str] = []

    @staticmethod
    def _empty_queue():
        while not dispatch._pending.empty():
            dispatch._pending.get_nowait().done.set()
        dispatch._TLS.depth = 0

    def _queue(self, label: str, reentrant: bool = False) -> "dispatch._Request":
        request = dispatch._Request(lambda: self.order.append(label) or label, reentrant)
        dispatch._pending.put(request)
        return request

    def _outer(self, reentrant: bool = False) -> "dispatch._Request":
        def outer():
            self.order.append("outer-start")
            self._queue("inner")
            dispatch.drain_queue()          # what a nested event loop does when the wake signal arrives
            self.order.append("outer-end")
            return "outer"

        request = dispatch._Request(outer, reentrant)
        dispatch._pending.put(request)
        return request

    def test_a_nested_drain_leaves_the_queue_to_the_outer_loop(self):
        outer = self._outer()
        dispatch.drain_queue()
        self.assertEqual(self.order, ["outer-start", "outer-end", "inner"])
        self.assertEqual((outer.result, outer.failed), ("outer", False))
        self.assertEqual(dispatch._depth(), 0)

    def test_a_reentrant_task_still_lets_queued_tasks_run_inside_it(self):
        self._outer(reentrant=True)
        dispatch.drain_queue()
        self.assertEqual(self.order, ["outer-start", "inner", "outer-end"])

    def test_the_depth_is_restored_after_a_task_that_raised(self):
        def boom():
            raise SystemExit(4)

        dispatch._pending.put(dispatch._Request(boom))
        dispatch.drain_queue()
        self.assertEqual(dispatch._depth(), 0)
        self._queue("after")
        dispatch.drain_queue()
        self.assertEqual(self.order, ["after"])

    def test_the_depth_belongs_to_the_thread_that_runs_the_task(self):
        started, release = threading.Event(), threading.Event()

        def long_task():
            started.set()
            release.wait(10)

        dispatch._pending.put(dispatch._Request(long_task))
        busy = threading.Thread(target=dispatch.drain_queue, daemon=True)
        busy.start()
        self.assertTrue(started.wait(5))
        other = self._queue("other thread")
        dispatch.drain_queue()                 # another stand-in GUI thread is not inside a task
        self.assertTrue(other.done.is_set())
        release.set()
        busy.join(5)
        self.assertEqual(self.order, ["other thread"])

    def test_a_task_queued_during_an_inline_task_runs_after_it(self):
        def inline_task():
            self.order.append("inline-start")
            self._queue("queued")
            dispatch.drain_queue()             # the nested event loop refuses to run it here
            self.order.append("inline-end")
            return "inline"

        waker = mock.Mock()
        waker.rouse.emit.side_effect = dispatch.drain_queue    # the re-wake drains at once, as the event loop would
        with mock.patch.object(dispatch, "_run_inline", return_value=True), \
                mock.patch.object(dispatch, "_waker", waker):
            self.assertEqual(dispatch.dispatch(inline_task), ("inline", False))
        self.assertEqual(self.order, ["inline-start", "inline-end", "queued"])
        waker.rouse.emit.assert_called_once()

    def test_an_inline_reentrant_task_does_not_block_the_queue(self):
        def dialog():
            self._queue("while the dialog is open")
            dispatch.drain_queue()
            return "answer"

        with mock.patch.object(dispatch, "_run_inline", return_value=True):
            self.assertEqual(dispatch.dispatch(dialog, reentrant=True), ("answer", False))
        self.assertEqual(self.order, ["while the dialog is open"])

    def test_the_rewake_never_raises(self):
        waker = mock.Mock()
        waker.rouse.emit.side_effect = RuntimeError("Internal C++ object already deleted")

        def task():
            self._queue("left for later")
            return 1

        with mock.patch.object(dispatch, "_run_inline", return_value=True), \
                mock.patch.object(dispatch, "_waker", waker):
            self.assertEqual(dispatch.dispatch(task), (1, False))
        dispatch.drain_queue()
        self.assertEqual(self.order, ["left for later"])


class InlineDispatchTest(unittest.TestCase):
    """Console sessions (no QApplication) and the GUI thread itself run the task inline."""

    def test_inline_runs_at_once_and_reports_exceptions(self):
        self.assertEqual(dispatch.dispatch(lambda: 7), (7, False))
        result, failed = dispatch.dispatch(lambda: {}["k"], op="probe")
        self.assertTrue(failed)
        self.assertIn("KeyError", result)

    def test_inline_honours_a_should_stop_that_is_already_true(self):
        ran = []
        result, failed = dispatch.dispatch(lambda: ran.append(1), should_stop=lambda: True)
        self.assertEqual((result, failed), ({"success": False, "error": "stopped", "stopped": True}, True))
        self.assertEqual(ran, [])
        self.assertEqual(dispatch.dispatch(lambda: 1, should_stop=lambda: False), (1, False))

    def test_timeout_text_helper(self):
        text = dispatch._timeout_result("probe", 60.0, started=True)
        self.assertEqual(text["error"].split(". ")[0], "'probe' did not complete within 60s (the GUI thread stayed busy)")
        self.assertTrue(text["still_running"])
        self.assertFalse(dispatch._timeout_result("probe", 2.5, started=False)["still_running"])


# ---------------------------------------------------------------------------------------------- headless: pure helpers


class HeadlessHelpersTest(unittest.TestCase):
    def test_clean_drops_startup_noise_but_keeps_user_lines(self):
        noise = ('Error: Failed to open library "/Library/Frameworks/3DconnexionNavlib.framework/3DconnexionNavlib"! '
                 "Error: dlopen(...)")
        text = "\n".join(["FreeCAD 1.1.3, Libs: 1.1.3R", "(C) 2001-2026 Juergen Riegel", "Importing project files", noise,
                          "", "   ", "FreeCAD says hi", "result = 5", "Recompute......", "\t\t(33 %)\t\t(66 %)",
                          "(100 %)", "(50%) done", "Recompute the model now"])
        self.assertEqual(headless._clean(text), "FreeCAD says hi\nresult = 5\n(50%) done\nRecompute the model now")

    def test_clean_handles_bytes_and_empty(self):
        self.assertEqual(headless._clean(b"a\nb"), "a\nb")
        self.assertEqual(headless._clean(None), "")
        self.assertEqual(headless._clean(""), "")
        self.assertEqual(headless._clean(b"\xff\xfeok"), "��ok")

    def test_crash_detection_and_signal_names(self):
        self.assertTrue(headless._is_crash(-11))
        self.assertTrue(headless._is_crash(-6))
        self.assertFalse(headless._is_crash(0))
        self.assertFalse(headless._is_crash(1))
        self.assertFalse(headless._is_crash(3))
        self.assertTrue(headless._is_crash(0xC0000005))
        self.assertEqual(headless._signal_text(-11), "signal 11 (SIGSEGV)")
        self.assertEqual(headless._signal_text(-6), "signal 6 (SIGABRT)")
        self.assertEqual(headless._signal_text(-250), "signal 250")
        self.assertEqual(headless._signal_text(0xC0000005), "status 0xC0000005")

    def test_failure_text_for_a_crash_mentions_signal_and_advice(self):
        text = headless._failure_text("x", -11, None, had_output=True)
        self.assertIn("CRASHED (killed by signal 11 (SIGSEGV))", text)
        self.assertIn("Your session is unaffected", text)
        self.assertNotIn("nothing was printed", text)
        self.assertIn("nothing was printed before the crash", headless._failure_text("x", -11, None, had_output=False))

    def test_failure_text_for_an_exit_status_and_an_exception(self):
        self.assertIn("exited with status 3", headless._failure_text("x", 3, None, True))
        info = {"type": "NameError", "mro": ["NameError", "Exception"], "message": "name 'Part' is not defined",
                "lines": [2], "names": ["<module>"], "name": "Part"}
        text = headless._failure_text("import FreeCAD\nPart.makeBox(1, 1, 1)\n", 1, info, True)
        self.assertTrue(text.startswith("NameError: name 'Part' is not defined\n  at script line 2: Part.makeBox(1, 1, 1)"))
        self.assertIn("execute_code_headless needs its own 'import Part'", text)

    def test_a_syntax_error_report_uses_the_syntax_line(self):
        info = {"type": "SyntaxError", "mro": ["SyntaxError"], "message": "'(' was never closed", "lines": [], "names": [],
                "syntax_line": 2, "name": None}
        self.assertIn("at script line 2: y = (", headless._format_exception("x = 1\ny = (\n", info))

    def test_split_error_removes_only_the_report_line(self):
        report = headless._ERR_MARK + '{"type": "ValueError", "message": "m", "lines": [1]}'
        rest, info = headless._split_error("warning one\n" + report + "\nwarning two\n")
        self.assertEqual(rest, "warning one\nwarning two\n")
        self.assertEqual(info["type"], "ValueError")
        self.assertEqual(headless._split_error(""), ("", None))
        self.assertEqual(headless._split_error(None), ("", None))
        self.assertEqual(headless._split_error(b"x")[0], "x")
        rest, info = headless._split_error(headless._ERR_MARK + "{not json\nkept")
        self.assertEqual((rest, info), ("kept", None))
        self.assertEqual(headless._split_error(headless._ERR_MARK + "[1, 2]")[1], None)

    def test_bootstrap_is_standalone_and_names_the_script(self):
        compile(headless._BOOTSTRAP, "boot", "exec")
        self.assertNotIn("oscar", headless._BOOTSTRAP)
        self.assertIn("line_buffering=True", headless._BOOTSTRAP)
        self.assertIn(headless.SCRIPT_NAME, headless._BOOTSTRAP)
        self.assertLess(len("exec(open('/x/_oscar_headless_boot.py', encoding='utf-8').read())"), 400)   # freecadcmd -c limit

    def test_invalid_timeouts_and_missing_engine(self):
        for bad in (0, -1, float("nan"), float("inf")):
            self.assertEqual(headless.run("pass", bad), {"success": False, "error": "timeout must be a positive finite number"})
        with mock.patch.object(headless, "find_console", return_value=None):
            self.assertEqual(headless.run("pass"), {"success": False, "error": "no headless engine binary found"})

    def test_script_directory_follows_the_cache_env_at_call_time(self):
        root = isolation.fresh_dir("hl_")
        with mock.patch.dict(os.environ, {"OSCAR_CACHE_DIR": root}):
            path = headless._script_dir()
        self.assertEqual(path, os.path.join(root, "headless"))
        self.assertTrue(os.path.isdir(path))

    def test_boot_file_is_written_once_and_rewritten_when_it_changed(self):
        d = isolation.fresh_dir("hl_")
        path = headless._boot_path(d)
        self.assertEqual(open(path, encoding="utf-8").read(), headless._BOOTSTRAP)
        mtime = os.stat(path).st_mtime_ns
        self.assertEqual(headless._boot_path(d), path)
        self.assertEqual(os.stat(path).st_mtime_ns, mtime)              # unchanged content: not rewritten
        with open(path, "w", encoding="utf-8") as f:
            f.write("tampered")
        headless._boot_path(d)
        self.assertEqual(open(path, encoding="utf-8").read(), headless._BOOTSTRAP)
        self.assertEqual([n for n in os.listdir(d) if ".tmp-" in n], [])

    def test_an_unwritable_script_directory_is_reported_not_raised(self):
        with mock.patch.object(headless, "find_console", return_value=["/bin/true"]), \
                mock.patch.object(headless, "_script_dir", side_effect=PermissionError("denied")):
            r = headless.run("pass")
        self.assertFalse(r["success"])
        self.assertIn("could not prepare the headless script", r["error"])


# ---------------------------------------------------------------------------------------------- headless: real helper


@unittest.skipIf(headless.find_console() is None, "no freecadcmd binary")
class HeadlessRunTest(unittest.TestCase):
    def leftovers(self):
        d = headless._script_dir()
        return [n for n in os.listdir(d) if n.startswith("headless_")]

    def test_success_returns_output_without_noise(self):
        r = headless.run("print('hello')\nprint('FreeCAD says hi')\nprint(__name__)")
        self.assertTrue(r["success"], r)
        self.assertEqual((r["returncode"], r["crashed"]), (0, False))
        self.assertEqual(r["output"], "hello\nFreeCAD says hi\n__main__")
        self.assertNotIn("error", r)

    def test_scripts_live_in_the_cache_dir_and_are_removed_afterwards(self):
        before = set(self.leftovers())
        r = headless.run("import os\nprint(os.path.basename(os.path.dirname(__file__)))")
        self.assertEqual(r["output"], "headless")
        self.assertEqual(set(self.leftovers()), before)
        self.assertTrue(os.path.isfile(os.path.join(headless._script_dir(), headless._BOOT_FILE)))

    def test_an_exception_is_formatted_like_execute_code_errors(self):
        r = headless.run("import FreeCAD\nprint('before the error')\n\ndef build():\n    return Part.makeBox(1, 1, 1)\n\nbuild()\n")
        self.assertFalse(r["success"])
        self.assertEqual((r["returncode"], r["crashed"]), (1, False))
        self.assertEqual(r["output"], "before the error")               # printed output survives an exception
        self.assertIn("NameError: name 'Part' is not defined", r["error"])
        self.assertIn("at script line 5: return Part.makeBox(1, 1, 1)", r["error"])
        self.assertIn("call chain:\n  line 7", r["error"])
        self.assertIn("execute_code_headless needs its own 'import Part'", r["error"])
        self.assertNotIn("OSCAR-HEADLESS-ERROR", r["output"])

    def test_a_syntax_error_names_the_line(self):
        r = headless.run("x = 1\ny = (\n")
        self.assertFalse(r["success"])
        self.assertIn("SyntaxError", r["error"])
        self.assertIn("at script line 2: y = (", r["error"])

    def test_sys_exit_keeps_the_output_and_reports_the_status(self):
        r = headless.run("print('x')\nraise SystemExit(3)")
        self.assertEqual((r["success"], r["returncode"], r["crashed"], r["output"]), (False, 3, False, "x"))
        self.assertIn("exited with status 3", r["error"])
        self.assertTrue(headless.run("import sys\nsys.exit(0)")["success"])

    @unittest.skipIf(sys.platform == "win32", "signals are POSIX only")
    def test_a_native_crash_is_surfaced_with_the_output_printed_before_it(self):
        r = headless.run("import os, signal\nprint('partial output')\nos.kill(os.getpid(), signal.SIGSEGV)")
        self.assertFalse(r["success"])
        self.assertTrue(r["crashed"])
        self.assertEqual(r["returncode"], -signal.SIGSEGV)
        self.assertEqual(r["output"], "partial output")                 # line-buffered: nothing lost
        self.assertIn("CRASHED (killed by signal 11 (SIGSEGV))", r["error"])

    def test_timeout_kills_the_helper_and_keeps_partial_output(self):
        t0 = time.monotonic()
        r = headless.run("import time\nprint('start')\ntime.sleep(60)", timeout=3)
        self.assertLess(time.monotonic() - t0, 30)
        self.assertEqual((r["success"], r["timed_out"], r["crashed"]), (False, True, False))
        self.assertEqual(r["output"], "start")
        self.assertIn("timed out after 3s", r["error"])

    def test_non_ascii_output_and_code_roundtrip(self):
        r = headless.run("print('Δ 25 mm · ø')")
        self.assertTrue(r["success"], r)
        self.assertEqual(r["output"], "Δ 25 mm · ø")

    def test_the_helper_is_a_separate_process_with_its_own_documents(self):
        r = headless.run("import FreeCAD\nprint(len(FreeCAD.listDocuments()))")
        self.assertEqual(r["output"], "0")


@unittest.skipIf(headless.find_console() is None, "no freecadcmd binary")
class HeadlessNetworkTest(unittest.TestCase):
    """P0 #2: the helper guards its whole process unless the network was granted for the turn."""

    CONNECT = "import socket\nsocket.create_connection(('203.0.113.9', 9), timeout=0.3)"

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in ("OSCAR_ALLOW_NET", "OSCAR_NET_AUDIT", headless._ENV_NET):
            os.environ.pop(key, None)

    def test_a_connection_is_blocked_without_a_grant(self):
        r = headless.run(self.CONNECT, allow_network=False)
        self.assertFalse(r["success"])
        self.assertIn("PermissionError: network access blocked", r["error"])
        self.assertIn("at script line 2", r["error"])

    def test_with_a_grant_the_attempt_is_an_ordinary_network_error(self):
        r = headless.run(self.CONNECT, allow_network=True)
        self.assertFalse(r["success"])
        self.assertNotIn("network access blocked", r["error"])
        self.assertRegex(r["error"], r"(TimeoutError|OSError|ConnectionError|timed out)")

    def test_a_thread_inside_the_helper_is_blocked_too(self):
        code = ("import socket, threading\nbox = {}\n"
                "def work():\n    try:\n        socket.gethostbyname('example.com')\n"
                "    except BaseException as e:\n        box['e'] = e\n"
                "t = threading.Thread(target=work)\nt.start()\nt.join(30)\nraise box['e']")
        r = headless.run(code, allow_network=False)
        self.assertIn("network access blocked", r["error"])

    def test_the_script_cannot_reach_the_guard_through_main(self):
        # the bootstrap runs in __main__ next to the script: the guard is a closure, not a name to rebind
        code = ("import __main__\nprint(sorted(n for n in vars(__main__) if 'guard' in n.lower()))\n"
                "import socket\nsocket.gethostbyname('example.com')")
        r = headless.run(code, allow_network=False)
        self.assertEqual(r["output"], "[]")
        self.assertIn("network access blocked", r["error"])

    def test_loopback_and_plain_geometry_stay_allowed(self):
        code = ("import socket\ntry:\n    socket.create_connection(('127.0.0.1', 9), timeout=0.3)\nexcept OSError as e:\n"
                "    print(type(e).__name__ != 'PermissionError')\n"
                "import Part\nprint(round(Part.makeBox(1, 2, 3).Volume, 3))")
        r = headless.run(code, allow_network=False)
        self.assertTrue(r["success"], r)
        self.assertEqual(r["output"], "True\n6.0")

    @unittest.skipIf(os.name == "nt", "POSIX program path")
    def test_starting_a_program_follows_the_turns_grant(self):
        code = "import subprocess\nsubprocess.run(['/bin/echo', 'hi'], check=True)"
        with mock.patch("oscar.netguard.network_granted", return_value=False):
            denied = headless.run(code)
        self.assertIn("network access blocked", denied.get("error", ""))
        with mock.patch("oscar.netguard.network_granted", return_value=True):
            granted = headless.run(code)
        self.assertTrue(granted["success"], granted)

    @unittest.skipIf(os.name == "nt", "POSIX program path")
    def test_the_kill_switch_reaches_the_helper(self):
        os.environ["OSCAR_NET_AUDIT"] = "0"
        r = headless.run("import subprocess\nsubprocess.run(['/bin/echo', 'hi'], check=True)", allow_network=False)
        self.assertTrue(r["success"], r)


if __name__ == "__main__":
    unittest.main()
