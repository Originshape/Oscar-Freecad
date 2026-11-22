"""The network gate's runtime guard and static scan beyond the script's own thread (P0 #2), on plain Python.

* threads a guarded script starts inherit its guard (``threading.Thread``, thread pools, ``_thread``); threads that
  were already running, and threads ordinary code starts, do not;
* code compiled from a script (``<execute_code...>``) is judged wherever it runs later (a callback on another thread);
* DNS lookups (``gethostbyname`` / ``gethostbyaddr`` / ``getnameinfo``), ``sendmsg``, ``_posixsubprocess.fork_exec`` and
  Windows' ``os.startfile`` are judged like connects and spawns;
* the static scan asks about computed imports and the modules that bypass the runtime hook.

The guard is driven through ``netguard._audit`` with the argument shapes CPython uses, so nothing touches the network.
No FreeCAD and no Qt: ``netguard`` imports without them.
"""

import _thread
import concurrent.futures
import os
import subprocess
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import netguard  # noqa: E402

REMOTE = ("203.0.113.9", 9)     # TEST-NET-3: never routed, and the guard is called directly anyway


def connect_event() -> None:
    """What ``socket.create_connection(REMOTE)`` raises in the audit hook, without a socket."""
    netguard._audit("socket.connect", (None, REMOTE))


def outcome(fn) -> "BaseException | None":
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - the test inspects it
        return exc
    return None


def script_function(source: str, name: str, filename: str = "<execute_code:99>", **names):
    """A function compiled as model-written code (the file name is what the frame fallback looks for)."""
    namespace = {"netguard": netguard, "REMOTE": REMOTE, "threading": threading, **names}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - building the test's "script"
    return namespace[name]


class _GuardCase(unittest.TestCase):
    def setUp(self):
        netguard.reset_turn()
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in ("OSCAR_ALLOW_NET", "OSCAR_NET_AUDIT"):
            os.environ.pop(key, None)
        # no saved grant, whether or not a settings store can be imported here
        grants = mock.patch.object(netguard, "has_grant", return_value=False)
        grants.start()
        self.addCleanup(grants.stop)
        self.addCleanup(netguard.reset_turn)

    def in_thread(self, fn) -> "BaseException | None":
        """Run ``fn`` on a new ``threading.Thread`` started HERE; its exception (or None)."""
        box: dict = {}
        worker = threading.Thread(target=lambda: box.update(exc=outcome(fn)))
        worker.start()
        worker.join(10)
        self.assertFalse(worker.is_alive())
        return box.get("exc")

    def assertBlocked(self, exc):
        self.assertIsInstance(exc, PermissionError)
        self.assertIn("network access blocked", str(exc))


class ThreadInheritanceTest(_GuardCase):
    def test_a_thread_started_inside_the_guard_is_guarded(self):
        with netguard.script_guard():
            exc = self.in_thread(connect_event)
        self.assertBlocked(exc)

    def test_a_thread_started_before_the_guard_is_not(self):
        go, box = threading.Event(), {}

        def waiting():
            go.wait(10)
            box["exc"] = outcome(connect_event)

        worker = threading.Thread(target=waiting)
        worker.start()                          # started by ordinary code ...
        with netguard.script_guard():
            go.set()                            # ... and only working while a script runs elsewhere
            worker.join(10)
        self.assertIsNone(box.get("exc", "never ran"))

    def test_threads_started_by_ordinary_code_are_untouched(self):
        self.assertIsNone(self.in_thread(connect_event))
        worker = threading.Thread(target=lambda: None)
        worker.start()
        worker.join(5)
        self.assertFalse(getattr(worker, "_oscar_guarded", False))

    def test_a_thread_pool_created_inside_the_guard_is_guarded(self):
        with netguard.script_guard():
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(connect_event) for _ in range(3)]
                errors = [f.exception(10) for f in futures]
        for exc in errors:
            self.assertBlocked(exc)

    def test_a_raw_thread_started_inside_the_guard_is_guarded(self):
        done, box = threading.Event(), {}

        def body():
            box["exc"] = outcome(connect_event)
            done.set()

        with netguard.script_guard():
            _thread.start_new_thread(body, ())
        self.assertTrue(done.wait(10))
        self.assertBlocked(box["exc"])

    def test_a_granted_turn_lets_guarded_threads_through(self):
        netguard._TURN_DECISION["download"] = "allow"
        with netguard.script_guard():
            self.assertIsNone(self.in_thread(connect_event))

    def test_the_kill_switch_also_switches_inheritance_off(self):
        os.environ["OSCAR_NET_AUDIT"] = "0"
        with netguard.script_guard():
            self.assertIsNone(self.in_thread(connect_event))

    def test_the_wrappers_are_installed_once_and_keep_the_originals(self):
        with netguard.script_guard():
            pass
        self.assertFalse(netguard.install_audit_hook())
        start = threading.Thread.start
        self.assertTrue(getattr(start, "oscar_guard", False))
        self.assertFalse(getattr(start.__wrapped__, "oscar_guard", False))
        self.assertTrue(getattr(_thread.start_new_thread, "oscar_guard", False))

    def test_the_wrapper_never_breaks_a_thread_start(self):
        with netguard.script_guard(), mock.patch.object(netguard, "_guarded_callable", side_effect=RuntimeError("bug")):
            self.assertIsNone(self.in_thread(lambda: None))


class ScriptFrameFallbackTest(_GuardCase):
    """A callback compiled from script code runs later outside the script's guard (a Qt timer, a FeaturePython
    ``execute()``, a document observer): the frame on the stack still marks it as the script's."""

    def test_script_code_is_judged_on_a_thread_without_a_guard(self):
        callback = script_function("def callback():\n    netguard._audit('socket.connect', (None, REMOTE))\n", "callback")
        self.assertBlocked(self.in_thread(callback))
        self.assertIsNone(self.in_thread(connect_event))          # the same event from ordinary code passes

    def test_every_script_file_name_counts_and_others_do_not(self):
        source = "def callback():\n    netguard._audit('socket.gethostbyname', ('example.com',))\n"
        for filename in ("<execute_code>", "<execute_code:7>", "<execute_code_async>", "<execute_code_headless>"):
            with self.subTest(filename):
                self.assertBlocked(outcome(script_function(source, "callback", filename)))
        for filename in ("<string>", "macro.py", "<stdin>"):
            with self.subTest(filename):
                self.assertIsNone(outcome(script_function(source, "callback", filename)))

    def test_a_thread_started_from_script_code_is_guarded(self):
        starter = script_function("def start(target):\n    t = threading.Thread(target=target)\n    t.start()\n"
                                  "    return t\n", "start")
        box = {}
        worker = starter(lambda: box.update(exc=outcome(connect_event)))   # plain code, started by the script
        worker.join(10)
        self.assertBlocked(box["exc"])

    def test_frequent_os_events_never_walk_the_stack(self):
        with mock.patch.object(netguard, "_script_code_on_stack", side_effect=AssertionError("walked")) as walk:
            for event, args in (("os.listdir", (".",)), ("open", ("x", "r", 0)), ("os.remove", ("x", -1)),
                                ("socket.__new__", (None, 2, 1, 0)), ("socket.bind", (None, ("0.0.0.0", 0)))):
                netguard._audit(event, args)
        walk.assert_not_called()


class LookupAndSendEventsTest(_GuardCase):
    def guarded(self, event, args):
        with netguard.script_guard():
            return outcome(lambda: netguard._audit(event, args))

    def test_dns_lookups_of_other_hosts_are_blocked(self):
        for event, args in (("socket.gethostbyname", ("example.com",)),
                            ("socket.gethostbyaddr", ("93.184.216.34",)),
                            ("socket.getaddrinfo", ("example.com", 443, 0, 0, 0)),
                            ("socket.getaddrinfo", (b"secret.attacker.example", None, 0, 0, 0)),
                            ("socket.getnameinfo", (("93.184.216.34", 53),))):
            with self.subTest(event=event, args=args):
                self.assertBlocked(self.guarded(event, args))

    def test_local_lookups_pass(self):
        for event, args in (("socket.gethostbyname", ("localhost",)),
                            ("socket.gethostbyname", ("LOCALHOST.",)),
                            ("socket.gethostbyaddr", ("127.0.0.1",)),
                            ("socket.getaddrinfo", (None, 80, 0, 0, 0)),
                            ("socket.getaddrinfo", (b"localhost", 80, 0, 0, 0)),
                            ("socket.getaddrinfo", ("127.0.0.53", 53, 0, 0, 0)),
                            ("socket.getnameinfo", (("127.0.0.1", 80),)),
                            ("socket.getnameinfo", (("::1", 80, 0, 0),))):
            with self.subTest(event=event, args=args):
                self.assertIsNone(self.guarded(event, args))

    def test_sendmsg_to_another_host_is_blocked(self):
        self.assertBlocked(self.guarded("socket.sendmsg", (None, REMOTE)))
        self.assertIsNone(self.guarded("socket.sendmsg", (None, ("127.0.0.1", 9))))
        self.assertIsNone(self.guarded("socket.sendmsg", (None, None)))        # a connected socket: judged at connect
        self.assertIsNone(self.guarded("socket.sendmsg", (None, "/tmp/sock")))  # AF_UNIX

    def test_outside_the_guard_nothing_is_judged(self):
        for event, args in (("socket.gethostbyname", ("example.com",)), ("socket.sendmsg", (None, REMOTE)),
                            ("os.startfile", ("https://example.com/?q=1", "open"))):
            self.assertIsNone(outcome(lambda: netguard._audit(event, args)))

    def test_handing_a_file_or_url_to_another_program_is_a_spawn(self):
        # Windows: os.startfile (what webbrowser.open uses there) starts the program registered for the file or URL
        self.assertBlocked(self.guarded("os.startfile", ("https://example.com/?q=secret", "open")))
        self.assertBlocked(self.guarded("os.startfile/2", ("C:\\Users\\me\\tool.exe", "open", "", None, 1)))
        netguard._TURN_DECISION["download"] = "allow"
        self.assertIsNone(self.guarded("os.startfile", ("https://example.com/", "open")))


def _fork_exec_is_audited() -> bool:
    """Whether this Python raises ``_posixsubprocess.fork_exec`` (3.14 does; 3.13.1 and FreeCAD's 3.11 do not), asked of
    a child interpreter so this process gets no extra audit hook."""
    if sys.version_info < (3, 12):
        return False      # no such event yet (and under freecadcmd sys.executable is not a plain interpreter)
    probe = ("import subprocess, sys\nseen = []\nsys.addaudithook(lambda e, a: seen.append(e))\n"
             "subprocess.run(['/bin/echo'], capture_output=True)\nprint('_posixsubprocess.fork_exec' in seen)")
    try:
        out = subprocess.run([sys.executable, "-I", "-B", "-c", probe], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.stdout.strip() == "True"


@unittest.skipIf(os.name == "nt", "POSIX program paths")
class ForkExecTest(_GuardCase):
    """``_posixsubprocess.fork_exec`` (newer Pythons raise it after ``subprocess.Popen``, and alone for a multiprocessing
    "spawn") is judged on its own: the program that really runs, never the argv[0] it is given."""

    PYTHON = os.fsencode(os.path.realpath(sys.executable))

    def fork_exec(self, executables, argv, env=None):
        with netguard.script_guard():
            return outcome(lambda: netguard._audit("_posixsubprocess.fork_exec", (tuple(executables), list(argv), env)))

    def test_a_harmless_system_program_passes(self):
        self.assertIsNone(self.fork_exec([b"/bin/echo"], [b"/bin/echo", b"hi"]))
        # the parent's PATH search: the first candidate that exists is the one that runs
        self.assertIsNone(self.fork_exec([b"/nonexistent/dir/echo", b"/bin/echo"], ["echo", "hi"]))

    def test_an_interpreter_is_blocked_without_a_grant(self):
        spawn = [self.PYTHON, "-B", "-c", "from multiprocessing.spawn import spawn_main; spawn_main()"]
        self.assertBlocked(self.fork_exec([self.PYTHON], spawn))
        netguard._TURN_DECISION["download"] = "allow"
        self.assertIsNone(self.fork_exec([self.PYTHON], spawn))

    def test_argv0_cannot_disguise_the_program(self):
        self.assertBlocked(self.fork_exec([self.PYTHON], ["echo", "hi"]))

    def test_a_network_tool_and_an_unresolvable_program_are_blocked(self):
        self.assertBlocked(self.fork_exec([b"/usr/bin/curl"], ["curl", "-s", "http://203.0.113.9/"]))
        self.assertBlocked(self.fork_exec([b"./cat"], ["cat"]))
        self.assertBlocked(self.fork_exec([], ["cat"]))

    def test_the_environment_is_not_part_of_the_command(self):
        env = [b"HTTPS_PROXY=https://proxy.example:8080", b"PATH=/usr/bin:/bin"]
        self.assertIsNone(self.fork_exec([b"/bin/echo"], [b"/bin/echo", b"hi"], env))

    def test_the_hook_lets_the_event_through_to_the_guard(self):
        with mock.patch.object(netguard, "_audit") as audit:
            netguard._audit_hook("_posixsubprocess.fork_exec", ((b"/bin/echo",), [b"echo"], None))
        audit.assert_called_once()

    def test_a_real_harmless_subprocess_still_runs_inside_the_guard(self):
        with netguard.script_guard():
            self.assertEqual(subprocess.run(["echo", "hi"], capture_output=True).returncode, 0)

    def test_a_real_multiprocessing_spawn_is_blocked(self):
        if not _fork_exec_is_audited():
            self.skipTest("this Python raises no event for a multiprocessing spawn (the static scan asks instead)")
        import multiprocessing

        process = multiprocessing.get_context("spawn").Process(target=print)
        with netguard.script_guard():
            with self.assertRaises(PermissionError):
                process.start()


class StaticScanTest(unittest.TestCase):
    """``analyze`` asks about what the runtime hook cannot see or judge (P0 #2e)."""

    def reasons(self, code):
        return netguard.analyze(code)["reasons"]

    def test_computed_module_names_are_asked_about(self):
        for code in ("import importlib\nm = importlib.import_module('so' + 'cket')",
                     "name = 'socket'\nm = __import__(name)",
                     "import importlib\nm = importlib.import_module(f'{base}.request')",
                     "import importlib.util\nspec = importlib.util.find_spec(x)",
                     "spec.loader.exec_module(module)",
                     "import runpy\nrunpy.run_path(path)",
                     "import importlib\nm = importlib.import_module(name=x)"):
            with self.subTest(code):
                verdict = netguard.analyze(code)
                self.assertTrue(verdict["network"], code)
                self.assertTrue(any(r.startswith("dynamic import of a computed module name") for r in verdict["reasons"]),
                                verdict["reasons"])

    def test_modules_that_bypass_the_runtime_hook_are_asked_about(self):
        cases = {
            "from PySide import QtNetwork": "Qt network module: QtNetwork",
            "from PySide6.QtNetwork import QNetworkAccessManager": "Qt network module: QtNetwork",
            "import PySide2.QtWebSockets": "Qt network module: QtWebSockets",
            "from PySide6 import QtWebEngineWidgets": "Qt web engine: QtWebEngineWidgets",
            "import ctypes": "native code access: ctypes",
            "import ctypes.util": "native code access: ctypes",
            "from cffi import FFI": "native code access: cffi",
            "import _socket": "low-level network module: _socket",
            "import _ssl": "low-level network module: _ssl",
            "import multiprocessing": "child processes: multiprocessing",
            "from multiprocessing import Pool": "child processes: multiprocessing",
            "import _posixsubprocess": "child processes: _posixsubprocess",
            "import _winapi": "child processes: _winapi",
            "from concurrent.futures import ProcessPoolExecutor": "child processes: ProcessPoolExecutor",
            "import importlib\nimportlib.import_module('ctypes')": "native code access: ctypes",
            "lib = sys.modules['x'].QtNetwork": "Qt network module: QtNetwork",
        }
        for code, reason in cases.items():
            with self.subTest(code):
                self.assertTrue(netguard.analyze(code)["network"], code)
                self.assertIn(reason, self.reasons(code))

    def test_ordinary_scripts_stay_quiet(self):
        for code in ("import importlib\nm = importlib.import_module('json')",
                     "m = __import__('math')",
                     "import asyncio\nimport http\nimport urllib_like_name\n",
                     "exec('x = 1')\nv = eval('2+2')",
                     "from PySide import QtCore, QtGui\nfrom FreeCAD import Base",
                     "import FreeCAD, Part\nbox = Part.makeBox(1, 2, 3)\nprint(box.Volume)",
                     "import concurrent.futures\npool = concurrent.futures.ThreadPoolExecutor(2)",
                     "this is ( not python"):
            with self.subTest(code):
                self.assertFalse(netguard.analyze(code)["network"], netguard.analyze(code))

    def test_the_summary_names_the_reason(self):
        needs, summary = netguard.needs_permission("import importlib\nm = importlib.import_module('so' + 'cket')")
        self.assertTrue(needs)
        self.assertIn("dynamic import of a computed module name", summary)


def _qt_importable() -> bool:
    try:
        from PySide import QtWidgets  # noqa: F401
    except Exception:  # noqa: BLE001 - a plain interpreter
        return False
    return True


@unittest.skipIf(_qt_importable(), "PySide is importable here (the dialog itself is covered by test_codeguard)")
class AskUserWithoutQtTest(unittest.TestCase):
    def test_no_qt_means_no_dialog(self):
        waits = []
        with mock.patch.object(netguard, "on_permission_wait", waits.append):
            answer = netguard.ask_user("t", "x", choices=(netguard.Choice("ok", "OK"),), default="ok",
                                       object_name="probe", op="probe")
        self.assertEqual(answer, netguard.NO_QT)
        self.assertEqual(waits, [])


if __name__ == "__main__":
    unittest.main()
