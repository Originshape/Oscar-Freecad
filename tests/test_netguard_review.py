"""Regression tests for the PR #3 review findings on the network gate.

* a script must not escape the in-process audit hook by spawning a child interpreter (``python3 -c "import socket..."``);
* the SSRF check must connect to the addresses it VALIDATED (DNS pinning), so a rebinding host cannot answer the first lookup
  with a public address and the second with an internal one;
* ``download_file`` / ``source_reference_part`` ask per host (a scoped grant), not with the unscoped broad grant.
"""

import os
import socket
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from support import isolation  # noqa: E402

isolation.isolate_process()

from oscar import netguard  # noqa: E402


class SpawnGuardTest(unittest.TestCase):
    """Review finding (medium): ``subprocess.run(["python3", "-c", "import socket; ..."])`` was neither caught by the static
    analyzer nor by the runtime hook, and the child ran with full network access."""

    def setUp(self):
        netguard.reset_turn()
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in ("OSCAR_ALLOW_NET", "OSCAR_NET_AUDIT"):
            os.environ.pop(key, None)
        self.addCleanup(netguard.reset_turn)

    def test_a_spawned_interpreter_is_blocked_at_runtime(self):
        import subprocess

        with netguard.script_guard():
            for argv in ([sys.executable, "-c", "print(1)"], ["python3", "-c", "import socket"],
                         ["sh", "-c", "echo hi"], ["/bin/bash", "-c", "true"], ["node", "-e", "1"]):
                with self.assertRaises(PermissionError, msg=argv):
                    subprocess.run(argv)

    def test_shell_commands_with_metacharacters_and_forks_are_blocked(self):
        with netguard.script_guard():
            with self.assertRaises(PermissionError):
                os.system("echo hi; python3 -c 'import socket'")
            if hasattr(os, "fork"):
                with self.assertRaises(PermissionError):
                    os.fork()   # the guard raises BEFORE forking

    @unittest.skipIf(os.name == "nt", "echo / ls are not programs on Windows")
    def test_harmless_local_programs_still_run(self):
        import subprocess

        with netguard.script_guard():
            self.assertEqual(subprocess.run(["echo", "hi"], capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run(["ls", "."], capture_output=True).returncode, 0)

    def test_a_network_grant_lifts_the_block_for_spawns(self):
        import subprocess

        os.environ["OSCAR_ALLOW_NET"] = "1"
        with netguard.script_guard():
            self.assertEqual(subprocess.run([sys.executable, "-c", "pass"]).returncode, 0)

    def test_the_kill_switch_unblocks(self):
        import subprocess

        os.environ["OSCAR_NET_AUDIT"] = "0"
        with netguard.script_guard():
            self.assertEqual(subprocess.run([sys.executable, "-c", "pass"]).returncode, 0)

    def test_other_threads_are_not_affected(self):
        import subprocess

        out = {}
        go = threading.Event()

        def other():
            go.wait(20)
            out["rc"] = subprocess.run([sys.executable, "-c", "pass"]).returncode

        t = threading.Thread(target=other)
        t.start()                       # started by ordinary code before the script, so not the script's thread
        with netguard.script_guard():
            go.set()
            t.join(20)
        self.assertEqual(out.get("rc"), 0)

    def test_a_thread_the_script_starts_is_guarded_too(self):
        import subprocess

        out = {}

        def other():
            try:
                subprocess.run([sys.executable, "-c", "pass"])
            except PermissionError as exc:
                out["exc"] = exc

        with netguard.script_guard():
            t = threading.Thread(target=other)
            t.start()
            t.join(20)
        self.assertIsInstance(out.get("exc"), PermissionError)

    def test_the_static_analysis_flags_a_child_interpreter_up_front(self):
        for code in ("import subprocess\nsubprocess.run(['python3', '-c', 'import socket'])",
                     "import subprocess, sys\nsubprocess.run([sys.executable, '-c', 'x'])",
                     "import os\nos.system('bash -c \"echo hi\"')"):
            verdict = netguard.analyze(code)
            self.assertTrue(verdict["network"], code)
            self.assertTrue(any("child interpreter" in r for r in verdict["reasons"]), code)

    def test_a_harmless_subprocess_is_not_a_network_intent(self):
        self.assertFalse(netguard.analyze("import subprocess\nsubprocess.run(['echo', 'hi'])")["network"])
        self.assertFalse(netguard.analyze("# python3 is mentioned only in a comment\nx = 1")["network"])

    def test_program_name_extraction(self):
        self.assertEqual(netguard._spawn_program("subprocess.Popen", ("python3", ["python3", "-c", "x"], None, None)), "python3")
        self.assertEqual(netguard._spawn_program("subprocess.Popen", (None, ["C:\\Tools\\Python.EXE", "-c"], None, None)), "python")
        self.assertEqual(netguard._spawn_program("os.system", ("echo hi",)), "echo")
        self.assertEqual(netguard._spawn_program("os.system", ("echo hi && curl x",)), "")
        self.assertEqual(netguard._spawn_program("os.fork", ()), "")


@unittest.skipIf(os.name == "nt", "POSIX shell scripts and system directories")
class PlantedProgramTest(unittest.TestCase):
    """Review finding (medium): the spawn allowlist matched the program NAME only, so a script could plant an executable called
    ``cat`` (in its working directory or at the front of PATH) and have it allowed. The program is now resolved the way the OS
    would, and only a binary in a system directory the user cannot write is trusted."""

    def setUp(self):
        netguard.reset_turn()
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in ("OSCAR_ALLOW_NET", "OSCAR_NET_AUDIT"):
            os.environ.pop(key, None)
        self.addCleanup(netguard.reset_turn)
        self.tmp = isolation.fresh_dir("planted_")
        self.planted = os.path.join(self.tmp, "cat")
        with open(self.planted, "w") as handle:
            handle.write("#!/bin/sh\nexit 0\n")
        os.chmod(self.planted, 0o755)

    def test_a_planted_program_first_on_the_call_path_is_blocked(self):
        import subprocess

        env = {"PATH": self.tmp + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin")}
        with netguard.script_guard():
            with self.assertRaises(PermissionError):
                subprocess.run(["cat"], env=env)

    def test_a_planted_program_first_on_this_process_path_is_blocked(self):
        import subprocess

        os.environ["PATH"] = self.tmp + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin")
        with netguard.script_guard():
            with self.assertRaises(PermissionError):
                subprocess.run(["cat"])

    def test_a_planted_program_in_the_working_directory_is_blocked(self):
        import subprocess

        before = os.getcwd()
        os.chdir(self.tmp)
        try:
            with netguard.script_guard():
                with self.assertRaises(PermissionError):
                    subprocess.run(["./cat"])
                with self.assertRaises(PermissionError):
                    subprocess.run(["cat"], cwd=self.tmp, executable="./cat")
        finally:
            os.chdir(before)

    def test_a_dot_entry_in_path_does_not_make_the_working_directory_trusted(self):
        import subprocess

        before = os.getcwd()
        os.chdir(self.tmp)
        os.environ["PATH"] = "." + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin")
        try:
            with netguard.script_guard():
                with self.assertRaises(PermissionError):
                    subprocess.run(["cat"])
        finally:
            os.chdir(before)

    def test_the_real_system_program_still_runs(self):
        import shutil
        import subprocess

        real = shutil.which("echo")
        self.assertTrue(real and netguard._is_system_binary(real), real)
        with netguard.script_guard():
            self.assertEqual(subprocess.run(["echo", "hi"], capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run([real, "hi"], capture_output=True).returncode, 0)

    def test_is_system_binary_rules(self):
        self.assertFalse(netguard._is_system_binary(""))
        self.assertFalse(netguard._is_system_binary("cat"))                           # not absolute
        self.assertFalse(netguard._is_system_binary(self.planted))                    # a user-writable directory
        self.assertFalse(netguard._is_system_binary("/usr/local/nonexistent/cat"))    # not a system directory
        link = os.path.join(self.tmp, "link")
        os.symlink("/bin/echo", link)
        self.assertFalse(netguard._is_system_binary(link))                            # the lookup path is outside the system dirs

    def test_spawn_executable_resolution(self):
        env = {"PATH": self.tmp}
        self.assertEqual(netguard._spawn_executable("subprocess.Popen", (None, ["cat"], None, env)), self.planted)
        self.assertEqual(netguard._spawn_executable("subprocess.Popen", ("./cat", ["cat"], self.tmp, None)), self.planted)
        self.assertEqual(netguard._spawn_executable("os.posix_spawn", ("cat", ["cat"], {"PATH": self.tmp})), self.planted)
        self.assertEqual(netguard._spawn_executable("os.fork", ()), "")


class _ListeningServer:
    def __enter__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        return self

    def __exit__(self, *exc):
        self.sock.close()


class DnsPinTest(unittest.TestCase):
    """Review finding (low): the host was resolved once for the SSRF check and again by ``requests`` at connect time."""

    def setUp(self):
        netguard._install_pin_hook()
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("OSCAR_DNS_PIN", None)

    @staticmethod
    def connect(host, port):
        from urllib3.util import connection

        return connection.create_connection((host, port), 2.0)

    def test_a_pinned_host_connects_to_the_validated_address_without_resolving(self):
        real = socket.getaddrinfo

        def no_dns_for_the_pinned_name(host, *a, **k):
            if host == "rebind.example":
                raise AssertionError("the pinned host name was resolved again")
            return real(host, *a, **k)

        with _ListeningServer() as server, mock.patch("socket.getaddrinfo", side_effect=no_dns_for_the_pinned_name):
            with netguard.pinned_dns("rebind.example", ["127.0.0.1"]):
                sock = self.connect("rebind.example", server.port)
            sock.close()

    def test_without_a_pin_the_name_is_resolved_as_usual(self):
        with mock.patch("socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
            with self.assertRaises(OSError):
                self.connect("rebind.example", 9)

    def test_the_pin_is_scoped_to_the_block_and_the_thread(self):
        seen = {}

        def other():
            try:
                self.connect("rebind.example", 9)
            except OSError as exc:
                seen["err"] = exc

        real = socket.getaddrinfo

        def only_the_pinned_name_fails(host, *a, **k):
            if host == "rebind.example":
                raise socket.gaierror("no such host")
            return real(host, *a, **k)

        with mock.patch("socket.getaddrinfo", side_effect=only_the_pinned_name_fails):
            with _ListeningServer() as server:
                with netguard.pinned_dns("rebind.example", ["127.0.0.1"]):
                    t = threading.Thread(target=other)
                    t.start()
                    t.join(10)
                    self.assertIn("err", seen)                                  # another thread has no pin
                    self.connect("rebind.example", server.port).close()          # this one does
                with self.assertRaises(OSError):
                    self.connect("rebind.example", server.port)                 # and it ends with the block

    def test_pins_nest_and_restore(self):
        with netguard.pinned_dns("a.example", ["127.0.0.1"]):
            with netguard.pinned_dns("b.example", ["127.0.0.1"]):
                self.assertEqual(sorted(netguard._PIN_TLS.pins), ["a.example", "b.example"])
            self.assertEqual(sorted(netguard._PIN_TLS.pins), ["a.example"])
        self.assertIsNone(netguard._PIN_TLS.pins)

    def test_ip_literals_empty_lists_and_the_kill_switch_do_not_pin(self):
        with netguard.pinned_dns("93.184.216.34", ["93.184.216.34"]):
            self.assertIsNone(getattr(netguard._PIN_TLS, "pins", None))
        with netguard.pinned_dns("x.example", []):
            self.assertIsNone(getattr(netguard._PIN_TLS, "pins", None))
        os.environ["OSCAR_DNS_PIN"] = "0"
        with netguard.pinned_dns("x.example", ["93.184.216.34"]):
            self.assertIsNone(getattr(netguard._PIN_TLS, "pins", None))

    def test_resolve_public_returns_the_validated_addresses_in_order(self):
        infos = [(0, 0, 0, "", ("93.184.216.34", 0)), (0, 0, 0, "", ("2606:2800:220:1:248:1893:25c8:1946", 0)),
                 (0, 0, 0, "", ("93.184.216.34", 0))]
        with mock.patch("socket.getaddrinfo", return_value=infos):
            why, addresses = netguard._resolve_public("https://example.com/x")
        self.assertEqual(why, "")
        self.assertEqual(addresses, ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"])

    def test_one_internal_answer_refuses_the_host(self):
        infos = [(0, 0, 0, "", ("93.184.216.34", 0)), (0, 0, 0, "", ("10.0.0.5", 0))]
        with mock.patch("socket.getaddrinfo", return_value=infos):
            why, addresses = netguard._resolve_public("https://rebind.example/x")
        self.assertIn("private-network", why)
        self.assertEqual(addresses, [])

    def test_check_public_url_keeps_its_contract(self):
        self.assertEqual(netguard.check_public_url("https://93.184.216.34/x"), "")
        self.assertIn("loopback", netguard.check_public_url("http://127.0.0.1/x"))
        self.assertIn("internal host name", netguard.check_public_url("http://printer.local/x"))


class FetchPinsEveryHopTest(unittest.TestCase):
    """``_fetch_checked`` resolves ONCE per hop and connects to exactly what it validated, whatever the resolver says next."""

    class Resp:
        def __init__(self, code, location=None, url=""):
            self.status_code = code
            self.headers = {"Location": location} if location else {}
            self.url = url

        def close(self):
            pass

    def test_each_hop_is_pinned_to_its_own_validated_addresses_and_unpinned_after(self):
        answers = {"first.example": "93.184.216.34", "second.example": "93.184.216.35"}
        calls = []

        def fake_getaddrinfo(host, *a, **k):
            calls.append(host)
            return [(0, 0, 0, "", (answers[host] if len(calls) <= 2 else "10.0.0.5", 0))]   # a rebinding answer AFTER the checks

        class FakeRequests:
            def __init__(self):
                self.pins = []

            def get(self, url, **kwargs):
                self.pins.append(dict(getattr(netguard._PIN_TLS, "pins", None) or {}))
                if "first.example" in url:
                    return FetchPinsEveryHopTest.Resp(302, "https://second.example/file.step")
                return FetchPinsEveryHopTest.Resp(200, url=url)

        fake = FakeRequests()
        with mock.patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
            resp = netguard._fetch_checked(fake, "https://first.example/a", timeout=5, headers={}, allowed=None)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(fake.pins, [{"first.example": ["93.184.216.34"]}, {"second.example": ["93.184.216.35"]}])
        self.assertEqual(calls, ["first.example", "second.example"])              # each name resolved once; nothing else resolved
        self.assertIsNone(netguard._PIN_TLS.pins)


class ScopedPermissionTest(unittest.TestCase):
    """``download_file`` and ``source_reference_part`` ask PER HOST, so 'Always allow' is this host for this tool."""

    def test_download_file_asks_for_its_host(self):
        from oscar import tools

        asked = {}

        def fake_permission(summary, detail="", *, tool="", host=""):
            asked.update(tool=tool, host=host)
            return "deny"

        netguard.reset_turn()
        tools.reset_turn_circuits()
        with mock.patch.object(netguard, "request_permission", side_effect=fake_permission):
            text, _image, is_error = tools.run_tool("download_file", {"url": "https://Datasheets.RaspberryPi.com/x.zip", "reason": "test"})
        self.assertTrue(is_error)
        self.assertEqual(asked, {"tool": "download_file", "host": "datasheets.raspberrypi.com"})
        tools.reset_turn_circuits()


if __name__ == "__main__":
    unittest.main()
