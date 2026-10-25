"""Stop plumbing of the engine: the Stop button of a turn reaches the engine methods that can
block for a long time, so queued GUI work is abandoned instead of leaving the worker thread blocked for up to the
timeout (600 s for ``apply_feature_plan``).

* ``dispatch.bound_should_stop`` / ``dispatch_stoppable`` / ``failure_text``;
* the CadEngine methods that use it (reload, create / edit / delete) and the ones that must NOT
  (reads that explain errors, the turn snapshot / revert that run on the GUI thread);

Runs under ``freecadcmd`` (inline dispatch) plus a queued-path stand-in for the GUI thread."""

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

import FreeCAD as App  # noqa: E402

from oscar import tools  # noqa: E402
from oscar.cad import dispatch  # noqa: E402
from oscar.cad.engine import CadEngine  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402

STOPPED = {"success": False, "error": "stopped", "stopped": True}
SH = "/bin/sh"


def _tool(name):
    """Absolute path of a coreutils program (the PATH of a freecadcmd child may be empty)."""
    for folder in ("/bin", "/usr/bin"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return ""


SLEEP, TOUCH = _tool("sleep"), _tool("touch")


class DispatchHelpersTest(unittest.TestCase):
    def test_without_a_bound_context_the_poll_never_fires(self):
        poll = dispatch.bound_should_stop()
        self.assertFalse(poll())

    def test_a_bound_context_hands_over_its_own_stop_signal(self):
        flag = threading.Event()
        with use_context(ToolContext(should_stop=flag.is_set)):
            poll = dispatch.bound_should_stop()
            self.assertEqual(poll, flag.is_set)
            self.assertFalse(poll())
            flag.set()
            self.assertTrue(poll())

    def test_the_context_is_per_thread(self):
        seen = {}
        with use_context(ToolContext(should_stop=lambda: True)):
            t = threading.Thread(target=lambda: seen.update(poll=dispatch.bound_should_stop()))
            t.start()
            t.join(5)
        self.assertFalse(seen["poll"]())          # another thread has no turn: it is never stopped by this one

    def test_failure_text_reads_every_failure_shape(self):
        self.assertEqual(dispatch.failure_text(STOPPED), "stopped")
        self.assertEqual(dispatch.failure_text({"error": "'op' did not complete within 5s"}),
                         "'op' did not complete within 5s")
        self.assertEqual(dispatch.failure_text("KeyError: 'x'"), "KeyError: 'x'")
        self.assertEqual(dispatch.failure_text({}), "the GUI thread did not answer")
        self.assertEqual(dispatch.failure_text(None, "fallback"), "fallback")
        self.assertEqual(dispatch.failure_text("", "fallback"), "fallback")

    def test_dispatch_stoppable_runs_normally_and_refuses_after_stop(self):
        self.assertEqual(dispatch.dispatch_stoppable(lambda: 5, timeout=5, op="probe"), (5, False))
        ran = []
        with use_context(ToolContext(should_stop=lambda: True)):
            self.assertEqual(dispatch.dispatch_stoppable(lambda: ran.append(1), op="probe"), (STOPPED, True))
        self.assertEqual(ran, [])


class RunStoppableTest(unittest.TestCase):
    """``dispatch.run_stoppable``: blocking non-GUI work (a worker subprocess call) that Stop may abandon."""

    def test_the_value_comes_back(self):
        self.assertEqual(dispatch.run_stoppable(lambda: {"ok": 1}), {"ok": 1})
        self.assertIsNone(dispatch.run_stoppable(lambda: None))

    def test_an_exception_is_re_raised_in_the_caller(self):
        def boom():
            raise KeyError("worker")

        with self.assertRaises(KeyError):
            dispatch.run_stoppable(boom)

    def test_stop_abandons_the_work_and_the_helper_finishes_on_its_own(self):
        release, finished = threading.Event(), threading.Event()

        def work():
            release.wait(10)
            finished.set()
            return "late"

        flag = threading.Event()
        threading.Timer(0.3, flag.set).start()
        t0 = time.monotonic()
        result = dispatch.run_stoppable(work, should_stop=flag.is_set)
        self.assertEqual(result, {"error": "stopped"})
        self.assertLess(time.monotonic() - t0, 5)
        release.set()
        self.assertTrue(finished.wait(5))          # nothing was killed: the discarded result just lands nowhere

    def test_the_bound_context_is_the_default_signal(self):
        release = threading.Event()
        self.addCleanup(release.set)
        with use_context(ToolContext(should_stop=lambda: True)):
            self.assertEqual(dispatch.run_stoppable(lambda: release.wait(10)), {"error": "stopped"})

    def test_a_stop_probe_that_raises_does_not_abandon_healthy_work(self):
        def boom():
            raise RuntimeError("probe")

        self.assertEqual(dispatch.run_stoppable(lambda: time.sleep(0.4) or "done", should_stop=boom), "done")

    def test_the_helper_thread_is_a_daemon(self):
        seen = {}
        dispatch.run_stoppable(lambda: seen.update(daemon=threading.current_thread().daemon))
        self.assertTrue(seen["daemon"])


class QueuedStopTest(unittest.TestCase):
    """The queued path: nobody drains, Stop turns true after 0.3 s, the queued task never runs."""

    def setUp(self):
        for p in (mock.patch.object(dispatch, "_run_inline", return_value=False),
                  mock.patch.object(dispatch, "init", lambda: None),
                  mock.patch.object(dispatch, "_wake", lambda: None)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(dispatch.drain_queue)

    def test_a_queued_engine_style_call_is_abandoned_on_stop(self):
        flag = threading.Event()
        threading.Timer(0.3, flag.set).start()
        ran = []
        t0 = time.monotonic()
        with use_context(ToolContext(should_stop=flag.is_set)):
            result, failed = dispatch.dispatch_stoppable(lambda: ran.append(1), timeout=600.0, op="plan_apply")
        self.assertLess(time.monotonic() - t0, 5)
        self.assertEqual((result, failed), (STOPPED, True))
        dispatch.drain_queue()
        self.assertEqual(ran, [])                 # it never started: the GUI thread does not run it later

    def test_a_context_without_a_stop_keeps_the_normal_timeout(self):
        t0 = time.monotonic()
        with use_context(ToolContext()):
            result, failed = dispatch.dispatch_stoppable(lambda: None, timeout=0.4, op="probe")
        self.assertTrue(failed)
        self.assertIn("did not complete within", result["error"])
        self.assertGreaterEqual(time.monotonic() - t0, 0.35)


if __name__ == "__main__":
    unittest.main()
