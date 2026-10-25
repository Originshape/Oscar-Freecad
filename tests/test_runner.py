"""tests/run.py: path discovery, module selection (env, argv, ``fast`` / ``pure``), the guards (zero
tests, no isolated home), exit codes, UTF-8 safety, and the ``os._exit`` ending that survives freecadcmd's teardown abort.

Pure python (the runner is exercised against throw-away test directories), so it runs in the fast CI job.
"""

import contextlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_PY = os.path.join(TESTS_DIR, "run.py")

_spec = importlib.util.spec_from_file_location("oscar_runner_under_test", RUN_PY)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)     # __name__ != "__main__": importing must not run anything


def make_tests_dir(files):
    """A throw-away tests directory: run.py + support/ markers, plus ``files`` ({name: source})."""
    root = tempfile.mkdtemp(prefix="runner_")
    shutil.copy(RUN_PY, os.path.join(root, "run.py"))
    os.makedirs(os.path.join(root, "support"))
    open(os.path.join(root, "support", "__init__.py"), "w").close()
    for name, source in files.items():
        with open(os.path.join(root, name), "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(source))
    return root


PASSING = """
    import unittest
    class A(unittest.TestCase):
        def test_ok(self):
            pass
        def test_ok2(self):
            pass
    """
FAILING = """
    import unittest
    class B(unittest.TestCase):
        def test_bad(self):
            self.fail("on purpose")
    """
SKIPPED = """
    import unittest
    @unittest.skip("later stage")
    class C(unittest.TestCase):
        def test_x(self):
            pass
    """
CLASS_SKIPPED = """
    import unittest
    class D(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            raise unittest.SkipTest("Qt unavailable")
        def test_x(self):
            pass
    """
BROKEN = "import module_that_does_not_exist_anywhere\n"
NONASCII = """
    import unittest
    class D(unittest.TestCase):
        def test_gehaeuse(self):
            print("Geh\\u00e4use \\u2713")
            self.assertEqual("\\u00fc", "\\u00fc")
    """


@contextlib.contextmanager
def preserved_imports(root):
    """Everything ``main`` imports from the throw-away tests dir (``test_rn_*`` and its own empty ``support`` package,
    which discovery imports) must not leak into this process: other tests need the REAL ``support`` package."""
    before = dict(sys.modules)
    path = list(sys.path)
    try:
        yield
    finally:
        root_real = os.path.realpath(root)
        for name, mod in list(sys.modules.items()):
            origin = getattr(mod, "__file__", None) or ""
            spec_paths = list(getattr(getattr(mod, "__spec__", None), "submodule_search_locations", None) or [])
            if name not in before and any(os.path.realpath(p).startswith(root_real) for p in [origin, *spec_paths] if p):
                del sys.modules[name]
        for name, mod in before.items():
            if sys.modules.get(name) is not mod:
                sys.modules[name] = mod
        sys.path[:] = path


class Case(unittest.TestCase):
    def make_dir(self, files):
        root = make_tests_dir(files)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        return root

    def run_main(self, files, argv=(), env=None, **kw):
        """(exit code, captured stderr) of ``main`` against a throw-away tests dir."""
        root = self.make_dir(files)
        environ = {"FREECAD_USER_HOME": "/isolated/home"}
        environ.update(env or {})
        err = io.StringIO()
        with preserved_imports(root), mock.patch.object(sys, "stderr", err), mock.patch.object(sys, "stdout", io.StringIO()):
            code = runner.main(argv=list(argv), environ=environ, tests_dir=root, **kw)
        return code, err.getvalue()


class LocateTest(Case):
    def test_file_hint_wins(self):
        self.assertEqual(runner.locate_tests_dir(file_hint=RUN_PY, argv=["x"], cwd="/nowhere"), TESTS_DIR)

    def test_argv0_is_used_without_a_file_hint(self):
        self.assertEqual(runner.locate_tests_dir(file_hint=None, argv=[RUN_PY], cwd="/nowhere"), TESTS_DIR)

    def test_argv0_is_ignored_unless_it_is_run_py(self):
        with self.assertRaises(RuntimeError):
            runner.locate_tests_dir(file_hint=None, argv=["/usr/bin/freecadcmd"], cwd="/nowhere")

    def test_cwd_is_the_last_resort_both_inside_tests_and_at_the_module_root(self):
        self.assertEqual(runner.locate_tests_dir(file_hint=None, argv=[], cwd=TESTS_DIR), TESTS_DIR)
        self.assertEqual(runner.locate_tests_dir(file_hint=None, argv=[], cwd=os.path.dirname(TESTS_DIR)), TESTS_DIR)

    def test_a_directory_without_support_is_not_the_tests_dir(self):
        bare = tempfile.mkdtemp(prefix="runner_bare_")
        self.addCleanup(shutil.rmtree, bare, ignore_errors=True)
        shutil.copy(RUN_PY, os.path.join(bare, "run.py"))
        with self.assertRaisesRegex(RuntimeError, "cannot locate"):
            runner.locate_tests_dir(file_hint=os.path.join(bare, "run.py"), argv=[], cwd=bare)

    def test_a_bogus_file_hint_falls_through_to_the_next_candidate(self):
        self.assertEqual(runner.locate_tests_dir(file_hint="<string>", argv=[RUN_PY], cwd="/nowhere"), TESTS_DIR)


class SelectionTest(Case):
    def dirs(self, names):
        return self.make_dir({f"{n}.py": PASSING for n in names})

    def test_nothing_selected_means_discover_everything(self):
        self.assertEqual(runner.select_names(self.dirs([]), [], {}), ([], False))

    def test_env_and_argv_are_combined_in_order_without_duplicates(self):
        names, _ = runner.select_names(self.dirs([]), ["test_c", "test_a"], {"OSCAR_TEST_MODULES": "test_a,test_b,,"})
        self.assertEqual(names, ["test_a", "test_b", "test_c"])

    def test_paths_and_extensions_are_normalised_and_class_names_kept(self):
        names, _ = runner.select_names(self.dirs([]), ["tests/test_x.py", "test_y.GateTest", "test_z.GateTest.test_one"], {})
        self.assertEqual(names, ["test_x", "test_y.GateTest", "test_z.GateTest.test_one"])

    def test_windows_style_paths(self):
        names, _ = runner.select_names(self.dirs([]), ["tests\\test_x.py"], {})
        self.assertEqual(names, ["test_x"])

    def test_other_flags_are_ignored_and_list_is_reported(self):
        self.assertEqual(runner.select_names(self.dirs([]), ["-v", "--list", "test_a"], {}), (["test_a"], True))

    def test_fast_expands_to_the_existing_fast_modules_only(self):
        have = [m for m in runner.FAST_MODULES[:3]]
        names, _ = runner.select_names(self.dirs(have), [], {"OSCAR_TEST_MODULES": "fast"})
        self.assertEqual(names, have)
        names, _ = runner.select_names(self.dirs(have), ["--fast", "test_extra"], {})
        self.assertEqual(names, have + ["test_extra"])

    def test_pure_expands_to_the_existing_pure_modules_only(self):
        have = list(runner.PURE_MODULES[:2])
        names, _ = runner.select_names(self.dirs(have), ["--pure"], {})
        self.assertEqual(names, have)

    def test_fast_plus_an_explicit_module_does_not_duplicate_it(self):
        first = runner.FAST_MODULES[0]
        names, _ = runner.select_names(self.dirs([first]), [first], {"OSCAR_TEST_MODULES": "fast"})
        self.assertEqual(names, [first])

    def test_group_definitions_are_sane(self):
        for group in (runner.FAST_MODULES, runner.PURE_MODULES):
            self.assertEqual(len(group), len(set(group)))
            self.assertTrue(all(n.startswith("test_") for n in group))
        self.assertTrue(set(runner.PURE_MODULES) >= {"test_suite_guard", "test_turnlog", "test_runner"})
        self.assertTrue(set(runner.FAST_MODULES) >= {"test_suite_guard", "test_flow", "test_agent_loop", "test_providers",
                                                     "test_netguard", "test_project", "test_prompt", "test_context"})

    def test_the_real_pure_and_fast_modules_that_exist_are_discoverable(self):
        for group in (runner.FAST_MODULES, runner.PURE_MODULES):
            for name in group:
                path = os.path.join(TESTS_DIR, name + ".py")
                if os.path.isfile(path):
                    self.assertTrue(name.startswith("test_"))


class MainTest(Case):
    def test_all_green_exits_zero_and_reports_the_count(self):
        code, err = self.run_main({"test_rn_a.py": PASSING})
        self.assertEqual(code, runner.EXIT_OK)
        self.assertIn("OSCAR_TESTS: 2 tests selected", err)
        self.assertIn("ran 2, failures 0, errors 0", err)

    def test_a_failing_test_exits_one(self):
        code, err = self.run_main({"test_rn_a.py": PASSING, "test_rn_b.py": FAILING})
        self.assertEqual(code, runner.EXIT_FAILED)
        self.assertIn("failures 1", err)
        self.assertIn("on purpose", err)

    def test_an_import_error_in_a_test_module_is_a_failure_not_a_silent_omission(self):
        code, err = self.run_main({"test_rn_a.py": PASSING, "test_rn_broken.py": BROKEN})
        self.assertEqual(code, runner.EXIT_FAILED)
        self.assertIn("errors 1", err)
        self.assertIn("module_that_does_not_exist_anywhere", err)

    def test_zero_collected_tests_fail_with_exit_two(self):
        code, err = self.run_main({"helper.py": "x = 1\n"})
        self.assertEqual(code, runner.EXIT_NO_TESTS)
        self.assertIn("no tests were collected", err)

    def test_a_selection_that_matches_nothing_fails(self):
        code, err = self.run_main({"test_rn_a.py": PASSING}, argv=["test_rn_a.NoSuchClass"])
        self.assertNotEqual(code, runner.EXIT_OK)

    def test_selecting_a_missing_module_fails_loudly(self):
        code, err = self.run_main({"test_rn_a.py": PASSING}, argv=["test_rn_missing"])
        self.assertEqual(code, runner.EXIT_FAILED)

    def test_only_skipped_tests_still_count_as_collected_and_pass(self):
        code, err = self.run_main({"test_rn_s.py": SKIPPED})
        self.assertEqual(code, runner.EXIT_OK)
        self.assertIn("skipped 1", err)

    def test_a_module_whose_classes_all_skip_in_setupclass_passes(self):
        """CI regression: the panel modules skip every class (no PySide in the bundled interpreter); run one module at a time
        that is 'ran 0, skipped N' and used to be reported as 'no tests ran'."""
        code, err = self.run_main({"test_rn_c.py": CLASS_SKIPPED})
        self.assertEqual(code, runner.EXIT_OK)
        self.assertIn("all 1 selected test classes were skipped", err)
        self.assertNotIn("no tests ran", err)

    def test_selection_by_env_restricts_the_run(self):
        code, err = self.run_main({"test_rn_a.py": PASSING, "test_rn_b.py": FAILING}, env={"OSCAR_TEST_MODULES": "test_rn_a"})
        self.assertEqual(code, runner.EXIT_OK)
        self.assertIn("2 tests selected (test_rn_a)", err)

    def test_selection_by_argv_restricts_the_run(self):
        code, err = self.run_main({"test_rn_a.py": PASSING, "test_rn_b.py": FAILING}, argv=["tests/test_rn_a.py"])
        self.assertEqual(code, runner.EXIT_OK)

    def test_a_single_test_by_dotted_name(self):
        code, err = self.run_main({"test_rn_a.py": PASSING}, argv=["test_rn_a.A.test_ok"])
        self.assertEqual(code, runner.EXIT_OK)
        self.assertIn("1 tests selected", err)

    def test_list_prints_the_ids_and_runs_nothing(self):
        root = self.make_dir({"test_rn_a.py": PASSING, "test_rn_b.py": FAILING})
        out = io.StringIO()
        with preserved_imports(root), mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", io.StringIO()):
            code = runner.main(argv=["--list"], environ={"FREECAD_USER_HOME": "x"}, tests_dir=root)
        self.assertEqual(code, runner.EXIT_OK)                 # the failing test was not run
        self.assertIn("test_rn_a.A.test_ok", out.getvalue())
        self.assertIn("test_rn_b.B.test_bad", out.getvalue())

    def test_refuses_without_an_isolated_home(self):
        code, err = self.run_main({"test_rn_a.py": PASSING}, env={"FREECAD_USER_HOME": ""})
        self.assertEqual(code, runner.EXIT_REFUSED)
        self.assertIn("FREECAD_USER_HOME", err)
        self.assertNotIn("ran 2", err)

    def test_the_real_home_override(self):
        code, _ = self.run_main({"test_rn_a.py": PASSING}, env={"FREECAD_USER_HOME": "", "OSCAR_TEST_ALLOW_REAL_HOME": "1"})
        self.assertEqual(code, runner.EXIT_OK)
        code, _ = self.run_main({"test_rn_a.py": PASSING}, env={"FREECAD_USER_HOME": "", "OSCAR_TEST_ALLOW_REAL_HOME": "0"})
        self.assertEqual(code, runner.EXIT_REFUSED)

    def test_a_crash_inside_the_runner_is_a_failure_with_a_traceback(self):
        root = self.make_dir({"test_rn_a.py": PASSING})
        err = io.StringIO()
        with preserved_imports(root), mock.patch.object(sys, "stderr", err), \
                mock.patch.object(runner, "build_suite", side_effect=RuntimeError("kaput")):
            code = runner.main(argv=[], environ={"FREECAD_USER_HOME": "x"}, tests_dir=root)
        self.assertEqual(code, runner.EXIT_FAILED)
        self.assertIn("runner crashed", err.getvalue())
        self.assertIn("kaput", err.getvalue())

    def test_non_ascii_output_survives(self):
        code, err = self.run_main({"test_rn_u.py": NONASCII})
        self.assertEqual(code, runner.EXIT_OK)

    def test_main_never_raises_when_the_tests_dir_cannot_be_found(self):
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err), mock.patch.object(runner, "locate_tests_dir", side_effect=RuntimeError("nowhere")):
            self.assertEqual(runner.main(argv=[], environ={"FREECAD_USER_HOME": "x"}), runner.EXIT_FAILED)

    def test_argv_is_only_trusted_when_it_is_run_py(self):
        root = self.make_dir({"test_rn_a.py": PASSING, "test_rn_b.py": FAILING})
        with preserved_imports(root), mock.patch.object(sys, "argv", ["freecadcmd", "-c", "import sys; sys.argv=[]; test_rn_b"]), \
                mock.patch.object(sys, "stderr", io.StringIO()):
            code = runner.main(environ={"FREECAD_USER_HOME": "x", "OSCAR_TEST_MODULES": "test_rn_a"}, tests_dir=root)
        self.assertEqual(code, runner.EXIT_OK)            # the freecadcmd argv was ignored, not taken for module names


class SigpipeTest(unittest.TestCase):
    def test_ignore_sigpipe_sets_the_disposition_and_is_idempotent(self):
        import signal
        if not hasattr(signal, "SIGPIPE"):
            self.skipTest("no SIGPIPE on this platform")
        old = signal.getsignal(signal.SIGPIPE)
        self.addCleanup(signal.signal, signal.SIGPIPE, old)
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        self.assertTrue(runner.ignore_sigpipe())
        self.assertEqual(signal.getsignal(signal.SIGPIPE), signal.SIG_IGN)
        self.assertFalse(runner.ignore_sigpipe())             # already ignored: nothing to do

    def test_a_write_to_a_closed_socket_raises_instead_of_killing_the_process(self):
        import socket
        import signal
        if not hasattr(signal, "SIGPIPE"):
            self.skipTest("no SIGPIPE on this platform")
        old = signal.getsignal(signal.SIGPIPE)
        self.addCleanup(signal.signal, signal.SIGPIPE, old)
        runner.ignore_sigpipe()
        a, b = socket.socketpair()
        b.close()
        with self.assertRaises((BrokenPipeError, ConnectionResetError)):
            for _ in range(50):
                a.send(b"x" * 1024)
        a.close()

    def test_a_thread_cannot_change_signal_handlers_and_that_is_not_an_error(self):
        import threading
        result = []
        t = threading.Thread(target=lambda: result.append(runner.ignore_sigpipe()))
        t.start()
        t.join()
        self.assertEqual(len(result), 1)                      # returned (False) instead of raising ValueError


class SourceTest(unittest.TestCase):
    def test_run_py_is_pure_ascii(self):
        """freecadcmd reads ``exec(open(path).read())`` with the locale encoding (ASCII in a sandboxed shell)."""
        with open(RUN_PY, "rb") as f:
            f.read().decode("ascii")

    def test_the_main_guard_keeps_the_import_side_effect_free(self):
        self.assertNotEqual(runner.__name__, "__main__")
        self.assertTrue(callable(runner.main) and callable(runner.finish))


def _plain_python():
    """A real python interpreter (``sys.executable`` is freecadcmd under the recipe)."""
    exe = sys.executable or ""
    if os.path.basename(exe).lower().startswith("python"):
        return exe
    for cand in ("python", "python3", "python.exe"):
        path = os.path.join(os.path.dirname(exe), cand)
        if exe and os.path.isfile(path):
            return path
    return None


@unittest.skipUnless(_plain_python(), "no plain python interpreter next to sys.executable")
class SubprocessTest(Case):
    """The real entry point: ``python run.py`` ends with ``os._exit(code)`` and the exit status is the verdict."""

    def go(self, files, *args, env=None, locale=None):
        root = self.make_dir(files)
        environ = {k: v for k, v in os.environ.items() if k not in ("FREECAD_USER_HOME", "OSCAR_TEST_ALLOW_REAL_HOME",
                                                                    "OSCAR_TEST_MODULES", "PYTHONUTF8")}
        environ.update({"OSCAR_TEST_ALLOW_REAL_HOME": "1"})
        if locale:
            environ.update({"LC_ALL": locale, "LANG": locale})
        environ.update(env or {})
        proc = subprocess.run([_plain_python(), os.path.join(root, "run.py"), *args], cwd=tempfile.gettempdir(), env=environ,
                              capture_output=True, timeout=120)
        return proc.returncode, proc.stderr.decode("utf-8", "replace"), proc.stdout.decode("utf-8", "replace")

    def test_exit_codes_through_a_real_process(self):
        self.assertEqual(self.go({"test_a.py": PASSING})[0], 0)
        self.assertEqual(self.go({"test_a.py": PASSING, "test_b.py": FAILING})[0], 1)
        self.assertEqual(self.go({"helper.py": "x = 1\n"})[0], 2)
        self.assertEqual(self.go({"test_a.py": PASSING}, env={"OSCAR_TEST_ALLOW_REAL_HOME": "0"})[0], 3)

    def test_the_count_line_is_printed(self):
        code, err, _ = self.go({"test_a.py": PASSING})
        self.assertIn("OSCAR_TESTS: 2 tests selected", err)

    def test_the_runner_finds_its_own_directory_from_any_cwd(self):
        code, err, _ = self.go({"test_a.py": PASSING})
        self.assertEqual(code, 0)           # cwd is the system temp dir, not the tests dir

    def test_argv_names_work_in_a_real_process(self):
        code, err, _ = self.go({"test_a.py": PASSING, "test_b.py": FAILING}, "test_a")
        self.assertEqual(code, 0)
        code, err, _ = self.go({"test_a.py": PASSING, "test_b.py": FAILING}, "test_b")
        self.assertEqual(code, 1)

    def test_non_ascii_output_under_the_c_locale(self):
        code, err, out = self.go({"test_u.py": NONASCII}, locale="C")
        self.assertEqual(code, 0, err)

    def test_a_hard_exit_survives_a_non_daemon_thread(self):
        thread_test = """
            import threading, time, unittest
            class T(unittest.TestCase):
                def test_leaves_a_thread_behind(self):
                    threading.Thread(target=lambda: time.sleep(600)).start()
            """
        started = __import__("time").monotonic()
        code, err, _ = self.go({"test_t.py": thread_test})
        self.assertEqual(code, 0)
        self.assertLess(__import__("time").monotonic() - started, 60)    # os._exit does not wait for the thread


if __name__ == "__main__":
    unittest.main()
