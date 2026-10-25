"""Test runner for console-mode execution.

The bundled interpreter cannot ``import FreeCAD`` directly (FreeCAD.so sits in Resources/lib, off
sys.path), so the suite runs under the console binary instead. ``scripts/test.sh {fast|full|gui}`` wraps
the whole recipe (isolated user home, OSCAR_* sandbox, the right ``-c`` string); by hand:

    <app>/Contents/Resources/bin/freecadcmd -c \\
      "import sys; sys.argv=['run.py']; __file__='<mod>/tests/run.py'; exec(open(__file__, encoding='utf-8').read())"

freecadcmd defines no ``__file__``, hands over its own argv and reads files with the locale encoding, so
the recipe sets all three; this file is pure ASCII for the same reason and finds its own directory from
``__file__``, ``sys.argv[0]`` or the working directory (in that order).

Selection (all combinable, in this order of appearance):

* no names: every ``test_*.py`` next to this file is discovered (new files need no registration);
* names from ``OSCAR_TEST_MODULES`` (comma separated: freecadcmd -c dies above ~400 characters of argv)
  and from argv: ``test_flow``, ``test_flow.GateTest``, ``test_flow.GateTest.test_x``, ``tests/test_flow.py``;
* the token ``fast`` (or ``--fast``) selects ``FAST_MODULES`` and ``pure`` (or ``--pure``) selects ``PURE_MODULES``
  (those that exist): the pure ones need no FreeCAD, so the fast CI job runs them on a plain python;
* ``--list`` prints the selected test ids and exits without running them.

Guards: it FAILS (exit 2) when zero tests were collected, and refuses to run (exit 3) when
``FREECAD_USER_HOME`` is unset, because the installed app's user Mod dir may be a symlink to the main
checkout and ``import oscar`` would then silently test the wrong code; set ``OSCAR_TEST_ALLOW_REAL_HOME=1``
to override (CI does). Exit code: 0 all passed, 1 failures/errors, 2 nothing collected, 3 refused. The
process ends with ``os._exit(code)`` after flushing stdout/stderr, because freecadcmd aborts at teardown
(code 134, recursive_mutex) and the real status must survive. It also ignores SIGPIPE (freecadcmd leaves it at the
default, so a write to a closed socket would kill the run with exit code 141).
"""

import os
import signal
import sys
import time
import traceback
import unittest

# "fast" = flow, agent loop, providers, tools, netguard, project, prompt, context (+ the support and guard tests).
# Names that do not exist yet (a later stage's file) are skipped, so the set can list what is coming.
FAST_MODULES = (
    "test_stage0", "test_suite_guard", "test_turnlog", "test_runner", "test_scripted_server", "test_personas",
    "test_flow", "test_flow_guard", "test_loopcore", "test_agent_loop", "test_agent_loop_surface", "test_oscar_regressions",
    "test_providers", "test_provider_faults", "test_provider_catalogue", "test_models",
    "test_tools_and_params", "test_tool_registry", "test_netguard",
    "test_project", "test_prompt", "test_context", "test_context_guard",
)

# Modules whose code under test is pure python: they must pass on an interpreter that cannot
# import FreeCAD, which is exactly how the fast CI job runs them (a module that stops being pure fails there).
PURE_MODULES = (
    "test_suite_guard", "test_turnlog", "test_runner", "test_flow", "test_flow_guard", "test_loopcore",
    "test_prompt", "test_context", "test_kit_spec", "test_refparts_identity", "test_panel_render", "test_project",
    # the OriginShape account: the sign-in core, the P0 script and network gates, the OriginShape provider
    "test_auth_basics", "test_auth_device", "test_auth_fake", "test_auth_keystore", "test_auth_loopback",
    "test_auth_redaction", "test_auth_session", "test_netguard_threads", "test_codeguard", "test_provider_originshape",
    # the workspace: connect mode's protocol client and loop
    "test_connect",
    # what the model may see: credentials redacted, untrusted content labelled
    "test_context_guard",
    # sampled star ratings of turns
    "test_rating",
    # is there a newer Oscar? (the public update check)
    "test_updates",
    # the pen's region line and the chat search
    "test_markup", "test_chatsearch",
)

EXIT_OK, EXIT_FAILED, EXIT_NO_TESTS, EXIT_REFUSED = 0, 1, 2, 3


def locate_tests_dir(file_hint=None, argv=None, cwd=None):
    """Directory holding this runner: ``file_hint`` (``__file__``), then ``argv[0]``, then the working directory
    (or its ``tests`` subdirectory). A candidate counts only when it contains ``run.py`` and ``support/``."""
    argv = sys.argv if argv is None else argv
    cwd = os.getcwd() if cwd is None else cwd
    candidates = []
    if file_hint:
        candidates.append(os.path.dirname(os.path.abspath(file_hint)))
    if argv and argv[0] and os.path.basename(argv[0]) == "run.py":
        candidates.append(os.path.dirname(os.path.abspath(argv[0])))
    candidates += [cwd, os.path.join(cwd, "tests")]
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "run.py")) and os.path.isdir(os.path.join(cand, "support")):
            return cand
    raise RuntimeError("cannot locate the tests directory (tried: %s)" % ", ".join(candidates))


def _normalise(name):
    """``tests/test_x.py`` -> ``test_x``; ``test_x.Case.test`` is kept as is."""
    name = name.strip().replace("\\", "/")
    if name.endswith(".py"):
        name = name[:-3]
    return name.rsplit("/", 1)[-1] if "/" in name else name


def select_names(tests_dir, argv, environ):
    """The module / class / test names asked for, ``fast`` expanded, duplicates removed, order kept.
    Returns ``(names, list_only)``; an empty ``names`` means "discover everything"."""
    raw = [n for n in (environ.get("OSCAR_TEST_MODULES") or "").split(",") if n.strip()]
    list_only = False
    for arg in argv:
        if arg == "--list":
            list_only = True
        elif arg == "--fast":
            raw.append("fast")
        elif arg == "--pure":
            raw.append("pure")
        elif not arg.startswith("-"):
            raw.append(arg)
    names = []
    groups = {"fast": FAST_MODULES, "pure": PURE_MODULES}
    for item in raw:
        if item.strip() in groups:
            expanded = [m for m in groups[item.strip()] if os.path.isfile(os.path.join(tests_dir, m + ".py"))]
        else:
            expanded = [_normalise(item)]
        for n in expanded:
            if n and n not in names:
                names.append(n)
    return names, list_only


def build_suite(tests_dir, names):
    """The selected ``unittest.TestSuite`` (a module that fails to import becomes a failing test, never a
    silent omission)."""
    loader = unittest.TestLoader()
    if names:
        return unittest.TestSuite(loader.loadTestsFromName(n) for n in names)
    return loader.discover(tests_dir, pattern="test_*.py", top_level_dir=tests_dir)


def iter_tests(suite):
    """Every leaf test of ``suite``, flattened."""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item


def _say(text, stream=None):
    stream = stream or sys.stderr
    try:
        stream.write(text + "\n")
        stream.flush()
    except Exception:  # noqa: BLE001 - a broken console must not hide the result
        pass


def ignore_sigpipe():
    """Make a write to a closed socket raise ``BrokenPipeError`` instead of killing the process.

    A normal python start-up sets SIGPIPE to ignore; the embedded interpreter of freecadcmd leaves it at the
    default, so one test that writes to a connection its peer already closed (a Stop in the middle of a stream, a
    server handler answering a cancelled request) ends the WHOLE run with exit code 141 and no report.
    Returns True when the disposition was changed."""
    sigpipe = getattr(signal, "SIGPIPE", None)
    if sigpipe is None:  # Windows
        return False
    try:
        if signal.getsignal(sigpipe) == signal.SIG_IGN:
            return False
        signal.signal(sigpipe, signal.SIG_IGN)
        return True
    except (ValueError, OSError):  # not the main thread / not allowed
        return False


def _prepare_streams():
    """UTF-8 stdout/stderr (test names and messages may be non-ASCII under an ASCII locale) and UTF-8
    children (the geometry tests spawn the bundled python)."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - freecadcmd's console object may not support it
            pass


def main(argv=None, environ=None, tests_dir=None, file_hint=None, stream=None):
    """Run the selected tests; return the exit code (see the module docstring). Never raises."""
    environ = os.environ if environ is None else environ
    if argv is None:
        # freecadcmd hands over ITS argv unless the recipe replaced it: only trust a real "run.py" argv
        argv = sys.argv[1:] if sys.argv and os.path.basename(sys.argv[0]) == "run.py" else []
    argv = list(argv)
    try:
        tests_dir = tests_dir or locate_tests_dir(file_hint)
        mod_dir = os.path.dirname(tests_dir)
        for path in (mod_dir, tests_dir):
            if path not in sys.path:
                sys.path.insert(0, path)
        if not environ.get("FREECAD_USER_HOME") and environ.get("OSCAR_TEST_ALLOW_REAL_HOME") != "1":
            _say("OSCAR_TESTS: refusing to run: FREECAD_USER_HOME is not set, so 'import oscar' may resolve to the "
                 "installed (symlinked) checkout instead of this one. Use scripts/test.sh, or set "
                 "OSCAR_TEST_ALLOW_REAL_HOME=1 to override.")
            return EXIT_REFUSED
        names, list_only = select_names(tests_dir, argv, environ)
        suite = build_suite(tests_dir, names)
        count = suite.countTestCases()
        scope = ", ".join(names) if names else "all test_*.py"
        _say("OSCAR_TESTS: %d tests selected (%s) in %s" % (count, scope, tests_dir))
        if count == 0:
            _say("OSCAR_TESTS: FAILED: no tests were collected")
            return EXIT_NO_TESTS
        if list_only:
            for test in iter_tests(suite):
                _say(test.id(), sys.stdout)
            return EXIT_OK
        started = time.time()
        result = unittest.TextTestRunner(stream=stream or sys.stderr, verbosity=2).run(suite)
        _say("OSCAR_TESTS: ran %d, failures %d, errors %d, skipped %d in %.1fs" % (
            result.testsRun, len(result.failures), len(result.errors), len(result.skipped), time.time() - started))
        if result.testsRun == 0 and not result.skipped:
            _say("OSCAR_TESTS: FAILED: no tests ran")
            return EXIT_NO_TESTS
        if result.testsRun == 0:
            # every selected test was skipped (a whole class skips in setUpClass, e.g. the panel tests when the bundled
            # interpreter has no PySide): collected and honestly skipped, not "nothing ran"
            _say("OSCAR_TESTS: note: all %d selected test classes were skipped" % len(result.skipped))
        return EXIT_OK if result.wasSuccessful() else EXIT_FAILED
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_FAILED
    except BaseException:  # noqa: BLE001 - report, never swallow into a green exit
        _say("OSCAR_TESTS: runner crashed:\n" + traceback.format_exc())
        return EXIT_FAILED


def finish(code):
    """Flush the console streams and end the process with ``code`` (``os._exit``: freecadcmd aborts at normal
    teardown with exit code 134 after the tests are done, which would hide the real status)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    os._exit(code)


if __name__ == "__main__":
    ignore_sigpipe()
    _prepare_streams()
    try:
        _hint = __file__
    except NameError:  # freecadcmd -c without the __file__ recipe
        _hint = None
    finish(main(file_hint=_hint))
