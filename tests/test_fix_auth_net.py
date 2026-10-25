"""Regression tests for the auth-net review fixes (auth-net-01/03/08/09/10/11/13).

The MCP bridge (oscar/cad/server.py) is exercised end to end: the real server
is started on an ephemeral loopback port with a stub engine and hit with
crafted HTTP requests. netguard's download()/analyze()/permission dialog are
tested with a mocked requests.get and a stub dialog; no real network and no
GUI is needed.
"""

import http.client
import os
import socket
import sys
import tempfile
import time
import types
import unittest
import xmlrpc.client
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

# Adapted: the suite's isolation header (test_suite_guard NO_ISOLATION) - temp settings / projects / logs / cache /
# downloads BEFORE importing oscar. The original per-module temp settings file is only a fallback now.
isolation.isolate_process()
os.environ.setdefault(
    "OSCAR_AI_SETTINGS",
    os.path.join(tempfile.mkdtemp(prefix="oscar_fix_auth_net_"), "settings.json"),
)

if "FreeCADGui" not in sys.modules:
    try:
        import FreeCADGui  # noqa: F401
    except Exception:
        sys.modules["FreeCADGui"] = types.ModuleType("FreeCADGui")

from oscar import netguard  # noqa: E402
from oscar.cad import server  # noqa: E402


class _StubEngine:
    """Records every call so a rejected request can prove it never ran."""

    def __init__(self):
        self.calls = []

    def execute_code(self, code):
        self.calls.append(code)
        return {"success": True, "output": "ran"}

    def control_chars(self):
        return {"out": "a\x1bb\x00c\x07d", "n": 1,
                "nested": ["\x1b", ["t\x00"]], "key\x1b": "v"}

    def odd_code_points(self):
        return {"s": "x\ud800y\ufffez\uffff"}

    def plain_text(self):
        return {"out": "cube \U0001F4E6 and café – ok\n\ttabbed"}

    def boom(self):
        raise RuntimeError("bad \x1b thing")


_CALL = ('<?xml version="1.0"?><methodCall><methodName>execute_code'
         '</methodName><params><param><value><string>print(1)</string>'
         '</value></param></params></methodCall>')


class _BridgeTestCase(unittest.TestCase):
    def setUp(self):
        self.engine = _StubEngine()
        msg = server.start(self.engine, 0)
        self.assertIn("listening", msg)
        self.port = server.port()
        self.addCleanup(server.stop)

    def post(self, headers, body=_CALL, path="/"):
        """POST with exactly the given headers (Host is not added for us)."""
        data = body.encode()
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.putrequest("POST", path, skip_host=True,
                            skip_accept_encoding=True)
            for key, value in headers.items():
                conn.putheader(key, value)
            if "Content-Length" not in headers:
                conn.putheader("Content-Length", str(len(data)))
            conn.endheaders(data)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    @property
    def host(self):
        return f"127.0.0.1:{self.port}"


class BridgeRequestGuardTest(_BridgeTestCase):
    """auth-net-01: any web page could POST to the unauthenticated bridge
    (CSRF with a simple text/plain request, or DNS rebinding) and run code."""

    def assertRejected(self, headers, expect, body=_CALL):
        status, _ = self.post(headers, body)
        self.assertEqual(status, expect)
        self.assertEqual(self.engine.calls, [], "rejected request still ran")

    def test_drive_by_post_from_a_web_page_is_refused(self):
        self.assertRejected(
            {"Host": f"rebind.evil.example:{self.port}",
             "Origin": "https://evil.example",
             "Content-Type": "text/plain"}, 403)

    def test_any_origin_header_is_refused(self):
        for origin in ("https://evil.example", "null",
                       f"http://localhost:{self.port}"):
            with self.subTest(origin=origin):
                self.assertRejected(
                    {"Host": self.host, "Origin": origin,
                     "Content-Type": "text/xml"}, 403)

    def test_non_loopback_host_is_refused(self):
        for host in (f"evil.example:{self.port}", f"127.0.0.1.evil.example:{self.port}",
                     "127.0.0.1:1", "127.0.0.1", "[::1", ""):
            with self.subTest(host=host):
                self.assertRejected(
                    {"Host": host, "Content-Type": "text/xml"}, 403)

    def test_content_type_must_be_text_xml(self):
        for ctype in ("text/plain", "application/x-www-form-urlencoded",
                      "application/json", "multipart/form-data; boundary=x"):
            with self.subTest(content_type=ctype):
                self.assertRejected(
                    {"Host": self.host, "Content-Type": ctype}, 415)
        # no Content-Type at all
        self.assertRejected({"Host": self.host}, 415)

    def test_oversize_request_is_refused_without_reading_it(self):
        with mock.patch.object(server, "MAX_REQUEST_BYTES", 1000, create=True):
            # repeated: the unread body must not turn the 413 into a reset
            for _ in range(8):
                self.assertRejected(
                    {"Host": self.host, "Content-Type": "text/xml"}, 413,
                    body="x" * 5000)
            # a declared size alone is enough, however little is sent
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            try:
                conn.putrequest("POST", "/", skip_host=True)
                conn.putheader("Host", self.host)
                conn.putheader("Content-Type", "text/xml")
                conn.putheader("Content-Length", "99999999")
                conn.endheaders(b"<x/>")
                self.assertEqual(conn.getresponse().status, 413)
            finally:
                conn.close()

    def test_missing_or_bad_content_length_is_refused(self):
        for length in ("abc", "-5"):
            with self.subTest(length=length):
                status, _ = self.post(
                    {"Host": self.host, "Content-Type": "text/xml",
                     "Content-Length": length}, body="")
                self.assertIn(status, (411, 413))
        self.assertEqual(self.engine.calls, [])

    def test_legitimate_clients_still_work(self):
        # what xmlrpc.client (the MCP clients, tests/gui_smoke.py) sends
        for url in (f"http://127.0.0.1:{self.port}",
                    f"http://localhost:{self.port}",
                    f"http://127.0.0.1:{self.port}/RPC2"):
            with self.subTest(url=url):
                proxy = xmlrpc.client.ServerProxy(url, allow_none=True)
                self.assertEqual(proxy.execute_code("x")["output"], "ran")
        self.assertEqual(self.engine.calls, ["x", "x", "x"])

    def test_header_variants_clients_use_are_accepted(self):
        for headers, path in (
                ({"Host": self.host,
                  "Content-Type": "text/xml; charset=utf-8"}, "/"),
                ({"Host": f"localhost:{self.port}",
                  "Content-Type": "TEXT/XML"}, "/RPC2"),
                ({"Host": f"[::1]:{self.port}",
                  "Content-Type": "text/xml"}, "/"),
                ({"Content-Type": "text/xml"}, "/"),  # HTTP/1.0-style, no Host
        ):
            with self.subTest(headers=headers, path=path):
                status, body = self.post(headers, path=path)
                self.assertEqual(status, 200)
                self.assertIn(b"ran", body)

    def test_non_loopback_bind_is_refused(self):
        server.stop()
        for host in ("0.0.0.0", "192.0.2.1", ""):
            with self.subTest(host=host):
                msg = server.start(_StubEngine(), 0, host=host)
                self.assertFalse(server.is_running(), msg)
                self.assertIn("loopback", msg)


class BridgeXmlSafetyTest(_BridgeTestCase):
    """auth-net-10: control characters in a result made the HTTP 200 body
    ill-formed XML, so the client lost the result of a call that had run."""

    def proxy(self):
        return xmlrpc.client.ServerProxy(
            f"http://127.0.0.1:{self.port}", allow_none=True)

    def test_control_characters_in_a_result_do_not_break_the_reply(self):
        res = self.proxy().control_chars()
        self.assertEqual(res["out"], "a\ufffdb\ufffdc\ufffdd")
        self.assertEqual(res["n"], 1)
        self.assertEqual(res["nested"], ["\ufffd", ["t\ufffd"]])
        self.assertEqual(res["key\ufffd"], "v")

    def test_lone_surrogates_and_noncharacters_do_not_break_the_reply(self):
        self.assertEqual(self.proxy().odd_code_points(),
                         {"s": "x\ufffdy\ufffdz\ufffd"})

    def test_error_text_with_control_characters_is_a_clean_fault(self):
        with self.assertRaises(xmlrpc.client.Fault) as cm:
            self.proxy().boom()
        self.assertIn("RuntimeError", cm.exception.faultString)
        self.assertIn("bad \ufffd thing", cm.exception.faultString)

    def test_ordinary_text_passes_through_untouched(self):
        res = self.proxy().plain_text()
        self.assertEqual(
            res["out"], "cube \U0001F4E6 and café – ok\n\ttabbed")

    def test_unknown_method_is_still_a_fault(self):
        with self.assertRaises(xmlrpc.client.Fault):
            self.proxy().no_such_method()


class BridgeRestartTest(unittest.TestCase):
    """auth-net-13: stop() returned before the socket was released, so a
    quick restart on the same port failed with 'Address already in use'."""

    def tearDown(self):
        server.stop()

    def test_restart_on_the_same_port_right_after_stop(self):
        self.assertIn("listening", server.start(_StubEngine(), 0))
        port = server.port()
        self.assertEqual(server.stop(), "bridge stopped")
        msg = server.start(_StubEngine(), port)
        self.assertIn("listening", msg)
        self.assertEqual(server.port(), port)
        proxy = xmlrpc.client.ServerProxy(f"http://127.0.0.1:{port}")
        self.assertEqual(proxy.execute_code("again")["output"], "ran")

    def test_start_on_a_port_someone_else_holds_reports_it(self):
        blocker = socket.socket()
        self.addCleanup(blocker.close)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        msg = server.start(_StubEngine(), blocker.getsockname()[1])
        self.assertIn("could not bind", msg)
        self.assertFalse(server.is_running())

    def test_stop_when_not_running(self):
        self.assertEqual(server.stop(), "bridge was not running")

    def test_windows_does_not_share_the_port(self):
        # SO_REUSEADDR lets a second instance bind a live port on Windows.
        self.assertEqual(server._ThreadingXMLRPCServer.allow_reuse_address,
                         os.name != "nt")


class _FakeResp:
    def __init__(self, chunks=(b"abc",), fail=False, headers=None):
        self.headers = {"Content-Length": "3"} if headers is None else headers
        self.chunks = chunks
        self.fail = fail

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield from self.chunks
        if self.fail:
            import requests

            raise requests.exceptions.ChunkedEncodingError("connection broke")


class _DownloadCase(unittest.TestCase):
    """These tests are about file NAMES and partial files, with made-up hosts such as ``h``. Adapted for the merged
    netguard: ``download`` now runs the SSRF guard (``check_public_url``) first, which refuses a dotless host name
    as an internal one, so the guard is stubbed here (its own behaviour is pinned in test_netguard.py)."""

    def setUp(self):
        self.dest = tempfile.mkdtemp(prefix="ng_fix_dl_")
        patcher = mock.patch.object(netguard, "check_public_url", lambda url: "")
        patcher.start()
        self.addCleanup(patcher.stop)

    def fetch(self, url, filename="", resp=None, dest=None):
        import requests

        with mock.patch.object(requests, "get",
                               lambda u, **kw: resp or _FakeResp()):
            return netguard.download(url, dest or self.dest, filename)

    def listing(self):
        found = []
        for root, _dirs, files in os.walk(self.dest):
            found += [os.path.relpath(os.path.join(root, f), self.dest)
                      for f in files]
        return found


class DownloadFilenameSafetyTest(_DownloadCase):
    """auth-net-03: only '/' and '..' were rejected, so on Windows a drive or
    backslash path in the model-supplied name escaped the download folder."""

    UNSAFE = ("..\\x", "..\\..\\x.bat", "C:\\Users\\u\\Startup\\x.bat",
              "C:y.bat", "\\\\srv\\share\\x", "a/b", "a\\b", "../evil",
              ".hidden", "x\x00y", "tab\tname.stl", "CON", "nul.txt",
              "com1.stl", "LPT9", "trailing.", "trailing ", "foo:bar.txt",
              "a|b.stl", "what?.stl", " ")

    def test_unsafe_explicit_names_are_refused_and_nothing_is_written(self):
        for name in self.UNSAFE:
            with self.subTest(filename=name):
                path, error = self.fetch("https://h/x.stl", name)
                self.assertEqual(path, "")
                self.assertIn("unsafe filename", error)
        self.assertEqual(self.listing(), [])
        self.assertEqual(os.listdir(os.path.dirname(self.dest)).count("x"), 0)

    def test_an_ordinary_name_still_works(self):
        for name in ("ok name.stl", "bracket-v2_final.STEP", "a.b.c.obj"):
            with self.subTest(filename=name):
                path, error = self.fetch("https://h/x.stl", name)
                self.assertEqual(error, "")
                self.assertEqual(os.path.basename(path), name)
                self.assertTrue(os.path.isfile(path))

    @unittest.skipUnless(hasattr(os, "symlink"), "no symlinks")
    def test_a_link_inside_the_folder_cannot_redirect_the_write(self):
        outside = tempfile.mkdtemp(prefix="ng_fix_out_")
        target = os.path.join(outside, "victim.txt")
        with open(target, "w") as f:
            f.write("original")
        try:
            os.symlink(target, os.path.join(self.dest, "link.stl"))
        except OSError:
            self.skipTest("cannot create symlinks here")
        path, error = self.fetch("https://h/x.stl", "link.stl")
        self.assertEqual(path, "")
        self.assertIn("unsafe filename", error)
        with open(target) as f:
            self.assertEqual(f.read(), "original")

    def test_a_download_folder_that_is_itself_a_link_is_fine(self):
        if not hasattr(os, "symlink"):
            self.skipTest("no symlinks")
        real = tempfile.mkdtemp(prefix="ng_fix_real_")
        link = os.path.join(tempfile.mkdtemp(prefix="ng_fix_ln_"), "dl")
        try:
            os.symlink(real, link)
        except OSError:
            self.skipTest("cannot create symlinks here")
        path, error = self.fetch("https://h/x.stl", "ok.stl", dest=link)
        self.assertEqual(error, "")
        self.assertTrue(os.path.isfile(os.path.join(real, "ok.stl")))


class DownloadResilienceTest(_DownloadCase):
    """tools-12 extras: a broken stream, a bogus Content-Length or an
    unwritable folder must end as an error string, never a stray exception
    or a truncated file left under the final name."""

    def test_midstream_failure_leaves_no_partial_file(self):
        path, error = self.fetch(
            "https://h/x.step", resp=_FakeResp(chunks=(b"abc",), fail=True))
        self.assertEqual(path, "")
        self.assertIn("download failed", error)
        self.assertEqual(self.listing(), [])

    def test_non_numeric_content_length_is_tolerated(self):
        path, error = self.fetch(
            "https://h/x.step", resp=_FakeResp(headers={"Content-Length": "abc"}))
        self.assertEqual(error, "")
        self.assertTrue(os.path.isfile(path))

    def test_oversize_stream_leaves_no_file(self):
        import requests

        resp = _FakeResp(chunks=(b"x" * 10, b"x" * 10), headers={})
        with mock.patch.object(requests, "get", lambda u, **kw: resp):
            path, error = netguard.download(
                "https://h/x.step", self.dest, max_bytes=15)
        self.assertEqual(path, "")
        self.assertIn("cap", error)
        self.assertEqual(self.listing(), [])

    def test_unwritable_destination_is_an_error_not_an_exception(self):
        blocker = os.path.join(self.dest, "i_am_a_file")
        with open(blocker, "w") as f:
            f.write("x")
        path, error = self.fetch("https://h/x.stl", dest=blocker)
        self.assertEqual(path, "")
        self.assertTrue(error)


class DownloadDerivedNameTest(_DownloadCase):
    """auth-net-11: the default name came from the whole URL, so a query
    string became part of the extension ('bracket.stl_dl_1') and the
    importer, which picks its reader from the extension, refused the file."""

    def name_for(self, url):
        path, error = self.fetch(url)
        self.assertEqual(error, "", url)
        self.assertEqual(os.path.dirname(path), self.dest)
        return os.path.basename(path)

    def test_query_and_fragment_are_not_part_of_the_name(self):
        cases = {
            "https://h/files/bracket.stl?dl=1": "bracket.stl",
            "https://github.com/u/r/blob/main/part.stl?raw=true#x":
                "part.stl",
            "https://h/a/b.step?X-Amz-Signature=abc&x=1": "b.step",
            "https://h/a/b.step#frag": "b.step",
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(self.name_for(url), expected)

    def test_percent_encoding_is_decoded_then_sanitised(self):
        self.assertEqual(self.name_for("https://h/a%20b.stl"), "a_b.stl")
        self.assertEqual(self.name_for("https://h/x/..%5C..%5Cevil.bat"),
                         "evil.bat")
        self.assertEqual(self.name_for("https://h/a%00b.stl"), "a_b.stl")

    def test_long_names_keep_their_extension(self):
        name = self.name_for("https://h/" + "a" * 200 + ".step")
        self.assertTrue(name.endswith(".step"), name)
        self.assertLessEqual(len(name), 130)

    def test_names_without_a_usable_path_still_get_a_sane_one(self):
        self.assertEqual(self.name_for("https://example.com"), "download")
        self.assertEqual(self.name_for("https://example.com/"), "download")
        self.assertEqual(self.name_for("https://h/dir/"), "dir")
        self.assertEqual(self.name_for("https://h/.."), "download")
        self.assertEqual(self.name_for("https://h/.hidden.stl"), "hidden.stl")

    def test_a_reserved_device_name_is_not_used_bare(self):
        name = self.name_for("https://h/CON")
        self.assertNotEqual(name.upper(), "CON")
        self.assertTrue(name.endswith("CON"))

    def test_dot_runs_inside_a_name_are_collapsed_not_rejected(self):
        # net-build-03: only an explicit name may be refused for '..'; a name
        # taken from the URL is sanitised, so a harmless URL still downloads.
        cases = {
            "https://h/files/bracket..v2.stl": "bracket.v2.stl",
            "https://h/a...stl": "a.stl",
            "https://h/my..part.step?dl=1": "my.part.step",
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(self.name_for(url), expected)

    def test_trimming_a_long_stem_cannot_leave_a_dot_run(self):
        # The 100-char stem cut can land on a '.', which meets the extension's.
        name = self.name_for("https://h/" + "a" * 99 + "." + "b" * 5 + ".stl")
        self.assertNotIn("..", name)
        self.assertTrue(name.endswith(".stl"), name)

    def test_a_derived_name_always_passes_the_explicit_name_check(self):
        # "Always sanitised, never rejected" as an invariant, not just cases.
        urls = ("https://h/..", "https://h/...", "https://h/a....b",
                "https://h/%2e%2e", "https://h/x/..%5C..%5Cevil.bat",
                "https://h/CON.", "https://h/nul..txt", "https://h/a b..c",
                "https://h/.a.", "https://h/" + "." * 300 + "x",
                "https://h/" + "a." * 80 + "stl", "http://[::1/x.stl")
        for url in urls:
            with self.subTest(url=url):
                name = netguard._derived_filename(url)
                self.assertTrue(name)
                self.assertFalse(netguard._unsafe_filename(name), name)

    def test_a_malformed_url_is_an_error_string_not_an_exception(self):
        # urlsplit() raises ValueError on a bracketed host that never closes.
        for url in ("http://[::1/x.stl", "https://[bad/x.stl",
                    "http://h]/x.stl"):
            with self.subTest(url=url):
                path, error = self.fetch(url)
                self.assertIsInstance(path, str)
                self.assertIsInstance(error, str)
                if not error:
                    self.assertEqual(os.path.basename(path), "download")


class AnalyzeImportBypassTest(unittest.TestCase):
    """auth-net-09: the import scan only saw one module at the start of a
    line, so common forms went unflagged and the user was never asked."""

    BYPASSES = {
        "multi-module import": "import os, socket\ns = socket.create_connection((h, 443))",
        "multi-module import, no call": "import os, requests",
        "from-import of a submodule": "from http import client\nc = client.HTTPSConnection(h)",
        "urllib3": "import urllib3\nhttp = urllib3.PoolManager()",
        "ssl": "import ssl\nctx = ssl.create_default_context()",
        "multi-module smtplib": "import os, smtplib\nsmtplib.SMTP(h)",
        "xmlrpc.client": "import xmlrpc.client",
        "from xmlrpc import client": "from xmlrpc import client",
        "asyncio streams": "import asyncio\nasyncio.open_connection(h, 1)",
        "after a semicolon": "x = 1; import socket",
        "inside a function": "def f():\n    import socket\n    return 1",
        "after a colon": "try: import socket\nexcept ImportError: pass",
        "parenthesised from-import": "from http import (\n    client,\n)",
        "websockets": "import websockets",
        "pycurl": "import pycurl",
        "continued import": "import os, \\\n    socket",
        "alias": "import socket as s",
        "relative import without a module": "from . import socket",
        "relative import of a network module": "from .socket import create_connection",
        # P0 #2: what the runtime hook cannot see, or cannot judge
        "computed import_module": "import importlib\nm = importlib.import_module('so' + 'cket')",
        "computed __import__": "name = 'socket'\nm = __import__(name)",
        "Qt network module": "from PySide import QtNetwork\nmanager = QtNetwork.QNetworkAccessManager()",
        "ctypes": "import ctypes\nlibc = ctypes.CDLL(None)",
        "raw _socket": "import _socket",
        "multiprocessing": "import multiprocessing\nmultiprocessing.Process(target=print).start()",
    }

    def test_each_form_is_flagged(self):
        for label, code in self.BYPASSES.items():
            with self.subTest(label):
                self.assertTrue(netguard.analyze(code)["network"], code)
                allowed, _msg = netguard.gate_code(code, "execute_code")
                self.assertFalse(allowed, "console session must deny")

    def test_reason_names_the_module_back_to_the_user(self):
        cases = {
            "import os, socket": "import socket",
            "from http import client": "import http.client",
            "import urllib3": "import urllib3",
            "from xmlrpc import client": "import xmlrpc.client",
            "import os, requests": "import requests",
        }
        for code, reason in cases.items():
            with self.subTest(code):
                self.assertIn(reason, netguard.analyze(code)["reasons"])

    def test_unparsable_snippet_is_still_scanned(self):
        code = "import os, socket\nthis is ( not python\n"
        self.assertTrue(netguard.analyze(code)["network"])
        self.assertTrue(netguard.analyze(
            "x = (\nimport socket; from http import client")["network"])

    def test_plain_cad_scripts_stay_quiet(self):
        samples = (
            "import FreeCAD, Part, Draft\ndoc = FreeCAD.newDocument('x')\n"
            "box = Part.makeBox(1, 2, 3)\nprint(box.Volume)",
            "import os, sys, math, json\nfrom FreeCAD import Base, Vector\n"
            "v = Vector(1, 2, 3)",
            "import asyncio\nimport http\nimport urllib_like_name\n",
            "# import socket is mentioned only in a comment\nx = 1",
            "ssl_depth = 3\nsocketry = 2\nsmtp_style = 1\nprint(ssl_depth)",
            "from FreeCAD import Base\ndef start_server_label():\n    return 1",
        )
        for code in samples:
            with self.subTest(code[:30]):
                verdict = netguard.analyze(code)
                self.assertFalse(verdict["network"], verdict)


def _offscreen_qapp():
    try:
        from PySide import QtWidgets
    except Exception:
        raise unittest.SkipTest("PySide unavailable")
    app = QtWidgets.QApplication.instance()
    if app is None:
        # a stray test must never open a real window on the desktop
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtWidgets.QApplication(getattr(sys, "argv", ["tests"])[:1])
    return app


class PermissionDialogTimeoutTest(unittest.TestCase):
    """auth-net-08: the dialog advertised an auto-deny that never closed it,
    so it sat on screen after the agent had already been told 'deny' and a
    late 'Always allow' click was dropped."""

    def setUp(self):
        from PySide import QtCore, QtWidgets

        from oscar import params

        self.QtCore, self.QtWidgets, self.params = QtCore, QtWidgets, params
        self._timers = []
        self.app = _offscreen_qapp()
        os.environ.pop("OSCAR_ALLOW_NET", None)
        params.set_bool("AllowNetwork", False)
        netguard.reset_turn()
        self.main = QtWidgets.QMainWindow()
        self.main.show()
        gui = sys.modules["FreeCADGui"]
        patcher = mock.patch.object(
            gui, "getMainWindow", lambda: self.main, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for timer in self._timers:
            timer.stop()  # a stray timer would hit the NEXT test's dialog
        for box in self._boxes():
            box.reject()
        self.main.close()
        self.main.deleteLater()
        self.params.set_bool("AllowNetwork", False)
        self.params.clear_net_grants()
        netguard.reset_turn()

    def _boxes(self):
        return [w for w in self.app.topLevelWidgets()
                if isinstance(w, self.QtWidgets.QMessageBox) and w.isVisible()]

    def _after(self, ms, fn):
        timer = self.QtCore.QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(fn)
        timer.start(ms)
        self._timers.append(timer)

    def _click(self, text):
        for box in self._boxes():
            for button in box.buttons():
                if button.text() == text:
                    button.click()
                    return True
        return False

    def _safety_valve(self):
        """Never hang the suite: a dialog that outlives the test is closed."""
        tripped = []
        self._after(1500, lambda: (tripped.append(1),
                                   [b.reject() for b in self._boxes()]))
        return tripped

    def test_dialog_closes_itself_and_denies_after_the_timeout(self):
        valve = self._safety_valve()
        started = time.monotonic()
        with mock.patch.object(netguard, "DENY_TIMEOUT_S", 0.3):
            verdict = netguard.request_permission("wants the network", "x")
        elapsed = time.monotonic() - started
        self.assertEqual(verdict, "deny")
        self.assertEqual(valve, [], "the dialog had to be closed by force")
        self.assertLess(elapsed, 1.2)
        self.assertEqual(self._boxes(), [])

    def test_a_request_that_ran_out_of_time_never_opens_a_dialog(self):
        from oscar.cad import dispatch

        shown = []
        valve = self._safety_valve()

        def late_dispatch(fn, timeout=60.0, op="gui", **_kw):   # reentrant=True: the dialog holds no transaction
            # the GUI thread was busy: the dialog request is only served
            # after its own deadline has passed
            time.sleep(0.45)
            self._after(0, lambda: shown.extend(self._boxes()))
            return fn(), False

        with mock.patch.object(netguard, "DENY_TIMEOUT_S", 0.2), \
                mock.patch.object(dispatch, "dispatch", late_dispatch):
            verdict = netguard.request_permission("wants the network", "x")
            self.app.processEvents()
        self.assertEqual(verdict, "deny")
        self.assertEqual(valve, [], "a stale dialog was shown")
        self.assertEqual(shown, [])

    def test_a_click_that_lands_after_the_agent_gave_up_is_not_lost(self):
        from oscar.cad import dispatch

        self._safety_valve()

        def gave_up(fn, timeout=60.0, op="gui", **_kw):
            self._after(150, lambda: self._click("Always allow"))
            fn()
            return {"success": False, "error": "timed out"}, True

        with mock.patch.object(dispatch, "dispatch", gave_up):
            verdict = netguard.request_permission("wants the network", "x")
        self.assertEqual(verdict, "deny")  # the agent was already told no
        # migration (F-26): 'Always allow' stores a visible per-(tool, host) grant, never the legacy global flag
        self.assertTrue(netguard.has_grant(netguard.UNSCOPED_TOOL, netguard.UNSCOPED_HOST))
        self.assertFalse(self.params.get_bool("AllowNetwork", False))

    def test_clicks_still_work_normally(self):
        self._safety_valve()
        self._after(150, lambda: self._click("Allow once"))
        self.assertEqual(
            netguard.request_permission("wants the network", "x"), "allow")
        self.assertEqual(netguard.list_grants(), [])
        netguard.reset_turn()
        self._after(150, lambda: self._click("Always allow"))
        self.assertEqual(
            netguard.request_permission("wants the network", "x"), "always")
        # migration (F-26): a stored, revocable grant instead of the legacy global flag
        self.assertTrue(netguard.has_grant(netguard.UNSCOPED_TOOL, netguard.UNSCOPED_HOST))
        self.assertFalse(self.params.get_bool("AllowNetwork", False))


if __name__ == "__main__":
    unittest.main()
