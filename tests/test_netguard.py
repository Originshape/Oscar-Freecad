"""Network permission gate tests (netguard). Console sessions have no
human to ask — the gate must DENY there — so the full deny path is
testable headlessly; the GUI dialog itself is interactive/manual."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)
if "FreeCADGui" not in sys.modules:
    import types

    try:
        import FreeCADGui  # noqa: F401
    except Exception:
        sys.modules["FreeCADGui"] = types.ModuleType("FreeCADGui")

from oscar import netguard, tools  # noqa: E402

# console tests must never see the allow override
os.environ.pop("OSCAR_ALLOW_NET", None)
os.environ.pop("OSCAR_NET_TEST_ALWAYS", None)


class AnalyzeTest(unittest.TestCase):
    def test_requests_import_flagged(self):
        verdict = netguard.analyze(
            "import requests\nrequests.get('https://x.example/file.step')")
        self.assertTrue(verdict["network"])
        self.assertTrue(any("requests" in r for r in verdict["reasons"]))

    def test_urlopen_flagged(self):
        self.assertTrue(netguard.analyze(
            "from urllib.request import urlopen")["network"])

    def test_pip_install_flagged(self):
        self.assertTrue(netguard.analyze(
            "subprocess.run('pip install numpy')")["network"])

    def test_socket_flagged(self):
        self.assertTrue(netguard.analyze("s = socket.socket()")["network"])

    def test_plain_code_not_flagged(self):
        code = "import Part\nb = Part.makeBox(1,2,3)\nprint(b.Volume)"
        self.assertFalse(netguard.analyze(code)["network"])
        allowed, _msg = netguard.gate_code(code, "execute_code")
        self.assertTrue(allowed)

    def test_comments_ignored(self):
        self.assertFalse(netguard.analyze(
            "# maybe requests.get(url) someday\nx = 1")["network"])

    def test_urls_collected_and_localhost_noted(self):
        verdict = netguard.analyze(
            "open('http://127.0.0.1:8080/a')  # local")
        self.assertTrue(verdict["network"])
        self.assertTrue(any("local" in r for r in verdict["reasons"]))


class ConsoleDenyTest(unittest.TestCase):
    """Console sessions: nobody to ask → deny, with a model-facing
    message naming the alternative (download_file / user-provided file)."""

    def test_gate_denies_network_code_in_console(self):
        allowed, message = netguard.gate_code(
            "import requests\nrequests.get('https://example.com/pi4.step')",
            "execute_code")
        self.assertFalse(allowed)
        self.assertIn("DECLINED", message)
        self.assertIn("download_file", message)

    def test_execute_code_tool_blocked(self):
        text, _img, _is_error = tools.run_tool(
            "execute_code",
            {"code": "import requests\nprint(requests.get('https://x.io').status)"})
        self.assertIn("DECLINED", text)
        self.assertIn("download_file", text)

    def test_headless_variant_blocked(self):
        text, _img, _is_error = tools.run_tool(
            "execute_code_headless",
            {"code": "import urllib.request\nurllib.request.urlretrieve("
                     "'https://x.io/m.step', '/tmp/m.step')"})
        self.assertIn("DECLINED", text)

    def test_async_variant_blocked(self):
        text, _img, _is_error = tools.run_tool(
            "execute_code_async",
            {"code": "import socket\nsocket.socket()"})
        self.assertIn("DECLINED", text)

    def test_env_override_allows_unattended_runs(self):
        os.environ["OSCAR_ALLOW_NET"] = "1"
        try:
            allowed, _msg = netguard.gate_code("import requests", "test")
            self.assertTrue(allowed)
        finally:
            os.environ.pop("OSCAR_ALLOW_NET", None)


class DownloadToolTest(unittest.TestCase):
    def test_import_file_refuses_urls(self):
        text, _img, _is_error = tools.run_tool(
            "import_file",
            {"path": "https://example.com/pi4.step"})
        self.assertIn("download_file", text)

    def test_download_tool_denies_in_console(self):
        text, _img, _is_error = tools.run_tool(
            "download_file",
            {"url": "https://example.com/pi4.step",
             "reason": "test request"})
        self.assertIn("DECLINED", text)

    def test_download_url_validation(self):
        _path, error = netguard.download("ftp://example.com/x", "/tmp/x")
        self.assertIn("http(s)", error)
        _path, error = netguard.download("https://example.com/x",
                                         "/tmp/netguard_test",
                                         filename="../evil")
        self.assertIn("unsafe filename", error)


class OrientationTest(unittest.TestCase):
    """The fixes that make the download-and-build flow actually move:
    an explained wait, a logged decision trail, and an agent that knows
    the playbook instead of stalling."""

    def test_permission_decisions_logged(self):
        from oscar import turnlog

        netguard.reset_turn()  # don't inherit a cached decision
        netguard.gate_code("import requests", "execute_code")
        newest = turnlog.tail(1)[0]
        self.assertEqual(newest["kind"], "net_permission")
        self.assertEqual(newest["decision"], "deny_console")

    def test_wait_hook_semantics(self):
        seen = []
        netguard.on_permission_wait = seen.append
        try:
            # the hook mechanism dispatches to the registered listener
            netguard._announce_wait(True)
            netguard._announce_wait(False)
            self.assertEqual(seen, [True, False])
            # console denies never open a dialog → no spurious wait status
            seen.clear()
            netguard.request_permission("test summary")
            self.assertEqual(seen, [])
        finally:
            netguard.on_permission_wait = None

    def test_tool_calls_logged(self):
        from oscar import turnlog

        tools.run_tool("list_documents", {})
        self.assertTrue(any(e["kind"] == "tool" and e.get("name") ==
                            "list_documents" for e in turnlog.tail(20)))

    def test_agent_kows_the_sourcing_playbook(self):
        from oscar import agent

        self.assertIn("HOUSING/ENCLOSURE", agent.STRATEGY)
        self.assertIn("download_file", agent.STRATEGY)
        # enough rounds for a real build; the last one asks for a summary
        self.assertGreaterEqual(agent.MAX_TOOL_ROUNDS, 24)


class DownloadCircuitTest(unittest.TestCase):
    """The retry storm from the 403 incident: two strikes and the tool
    refuses instantly with the ask-the-user-attach instruction."""

    def setUp(self):
        tools.reset_turn_circuits()
        netguard.reset_turn()
        os.environ.pop("OSCAR_ALLOW_NET", None)

    def tearDown(self):
        tools.reset_turn_circuits()
        netguard.reset_turn()

    def _call(self, url="https://10.255.255.1/x.step"):
        try:
            text, _img, _err = tools.run_tool(
                "download_file", {"url": url, "reason": "test"})
            return text
        except Exception as e:  # ToolError from the reworked registry
            return str(e)

    def test_two_strikes_open_the_circuit(self):
        first = self._call()
        second = self._call()
        self.assertIn("ATTACH", first)
        self.assertIn("circuit OPEN", second)
        # third call refuses instantly — no permission prompt path at all
        third = self._call()
        self.assertIn("circuit OPEN", third)
        self.assertIn("Do not call download_file", third)

    def test_circuit_resets_next_turn(self):
        self._call()
        self._call()
        tools.reset_turn_circuits()
        again = self._call()
        self.assertNotIn("circuit OPEN", again)

    def test_permission_decision_cached_within_turn(self):
        # console deny once → the whole turn reuses it, instantly
        self.assertEqual(netguard.request_permission("x"), "deny")
        self.assertEqual(netguard._TURN_DECISION.get("download"), "deny")
        self.assertEqual(netguard.request_permission("x"), "deny")

    def test_download_sends_browser_user_agent(self):
        import urllib.request
        from unittest import mock

        import requests as _requests

        captured = {}

        class FakeResp:
            headers = {"Content-Length": "3"}
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def raise_for_status(self):
                pass
            def iter_content(self, chunk_size):
                return iter([b"abc"])

        def fake_get(url, **kwargs):
            captured.update(kwargs)
            return FakeResp()

        with mock.patch.object(_requests, "get", fake_get):
            import tempfile

            path, error = netguard.download(
                "https://example.com/p.stl",
                tempfile.mkdtemp(prefix="ng_ua_"))
        self.assertEqual(error, "")
        self.assertIn("Mozilla/5.0", captured["headers"]["User-Agent"])

    def test_unknown_tool_lists_real_tools(self):
        text, _img, _err = tools.run_tool("create_boxen", {})
        self.assertIn("Available tools", text)
        self.assertIn("create_object", text)


# --- verified downloads, file naming, zip extraction ----------------------------------------------------------

import hashlib  # noqa: E402
import io  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
import zipfile  # noqa: E402
from unittest import mock  # noqa: E402

import requests  # noqa: E402

STEP = (b"ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'2;1');\nENDSEC;\nDATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\nENDSEC;\n"
        b"END-ISO-10303-21;\n")
ASCII_STL = b"solid cube\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid cube\n"
HTML = b"<!DOCTYPE html><html><head><title>Sign in</title></head><body>Please sign in to download</body></html>"
LFS = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"cd" * 32 + b"\nsize 48211\n"


def zip_of(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


class FakeResponse:
    """What ``requests.get(..., stream=True)`` returns, as far as ``download_verified`` uses it."""

    def __init__(self, body=b"", *, chunks=None, headers=None, url="", status=200, fail_after=None):
        self.status_code = status
        self.headers = dict(headers or {})
        self.url = url
        self._chunks = chunks if chunks is not None else ([body] if body else [])
        self._fail_after = fail_after
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)

    def iter_content(self, chunk_size=1):
        for chunk in self._chunks:
            yield chunk
        if self._fail_after is not None:
            raise self._fail_after


class FakeGet:
    """A ``requests.get`` stand-in: records every call, answers with ``response`` (or a callable of the URL), or raises ``raises``."""

    def __init__(self, response=None, raises=None):
        self.response, self.raises, self.calls = response, raises, []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.raises is not None:
            raise self.raises
        return self.response(url) if callable(self.response) else self.response

    def patch(self):
        return mock.patch.object(requests, "get", self)


def _fake_getaddrinfo(host, *args, **kwargs):
    """DNS without the network: ``localhost`` is this computer, any other name a public address (``requests.get`` is
    faked too, so nothing connects). A real lookup made these tests hang where DNS doesn't answer."""
    local = str(host).lower().rstrip(".")
    address = "127.0.0.1" if local == "localhost" or local.endswith(".localhost") else "93.184.216.34"
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]


class DownloadVerifiedBase(unittest.TestCase):
    def setUp(self):
        self.dest = isolation.fresh_dir("ng_dl_")
        dns = mock.patch.object(socket, "getaddrinfo", _fake_getaddrinfo)
        dns.start()
        self.addCleanup(dns.stop)

    def files(self):
        return sorted(os.listdir(self.dest))

    def get(self, url, response=None, raises=None, **kwargs):
        net = FakeGet(response, raises)
        with net.patch():
            result = netguard.download_verified(url, self.dest, **kwargs)
        self.net = net
        return result

    def assertFailed(self, result, *fragments):
        self.assertFalse(result.ok, result)
        self.assertEqual(result.path, "")
        self.assertTrue(result.error)
        for fragment in fragments:
            self.assertIn(fragment, result.error)
        self.assertEqual(self.files(), [], "a failed download must leave nothing behind")


class DownloadVerifiedTest(DownloadVerifiedBase):
    def test_success_reports_path_size_hash_type_and_final_url(self):
        result = self.get("https://example.com/files/pi4.step", FakeResponse(STEP, headers={"Content-Length": str(len(STEP))},
                                                                             url="https://cdn.example.com/files/pi4.step"))
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.error, "")
        self.assertEqual(result.path, os.path.join(self.dest, "pi4.step"))
        self.assertEqual(result.bytes, len(STEP))
        self.assertEqual(result.sha256, hashlib.sha256(STEP).hexdigest())
        self.assertEqual(result.detected_type, "step")
        self.assertEqual(result.final_url, "https://cdn.example.com/files/pi4.step")
        with open(result.path, "rb") as fh:
            self.assertEqual(fh.read(), STEP)
        self.assertEqual(self.files(), ["pi4.step"])

    def test_result_fields_and_dict(self):
        result = self.get("https://example.com/a.step", FakeResponse(STEP))
        self.assertEqual(sorted(result.as_dict()), ["bytes", "detected_type", "error", "final_url", "path", "sha256"])
        self.assertEqual(result.as_dict()["path"], result.path)
        empty = netguard.DownloadResult()
        self.assertFalse(empty.ok)
        self.assertEqual((empty.path, empty.bytes, empty.sha256, empty.detected_type, empty.final_url, empty.error), ("", 0, "", "", "", ""))

    def test_request_shape_browser_headers_stream_and_timeouts(self):
        self.get("https://example.com/a.step", FakeResponse(STEP), timeout=30)
        url, kwargs = self.net.calls[0]
        self.assertEqual(url, "https://example.com/a.step")
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["timeout"], (15, 30))
        self.assertIn("Mozilla/5.0", kwargs["headers"]["User-Agent"])

    def test_the_response_is_always_closed(self):
        resp = FakeResponse(HTML)
        self.get("https://example.com/a.step", resp)
        self.assertTrue(resp.closed)

    def test_a_partial_file_is_never_visible_under_the_final_name(self):
        seen = []

        def chunks():
            yield STEP[:20]
            seen.append(self.files())
            yield STEP[20:]

        result = self.get("https://example.com/pi4.step", FakeResponse(chunks=chunks()))
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(seen[0]), 1)
        self.assertTrue(seen[0][0].startswith("pi4.step.tmp-"), seen[0])
        self.assertEqual(self.files(), ["pi4.step"])

    # --- naming ---

    def test_query_and_fragment_are_stripped_from_the_name(self):
        result = self.get("https://example.com/dl/board.step?token=abc&dl=1#frag", FakeResponse(STEP))
        self.assertEqual(os.path.basename(result.path), "board.step")

    def test_percent_encoding_and_odd_characters_are_made_safe(self):
        result = self.get("https://example.com/a%20b/My%20Part%20(v2).step", FakeResponse(STEP))
        self.assertEqual(os.path.basename(result.path), "My_Part__v2_.step")

    def test_content_disposition_wins_over_the_url(self):
        resp = FakeResponse(STEP, headers={"Content-Disposition": 'attachment; filename="Pi4 Board.STEP"'})
        result = self.get("https://example.com/download?id=77", resp)
        self.assertEqual(os.path.basename(result.path), "Pi4_Board.STEP")

    def test_content_disposition_rfc5987_wins_over_the_plain_name(self):
        resp = FakeResponse(STEP, headers={"content-disposition": "attachment; filename=\"fallback.txt\"; filename*=UTF-8''my%20part.step"})
        result = self.get("https://example.com/x", resp)
        self.assertEqual(os.path.basename(result.path), "my_part.step")

    def test_content_disposition_cannot_choose_a_directory(self):
        resp = FakeResponse(STEP, headers={"Content-Disposition": 'attachment; filename="../../evil.step"'})
        result = self.get("https://example.com/x", resp)
        self.assertEqual(result.path, os.path.join(self.dest, "evil.step"))
        self.assertEqual(self.files(), ["evil.step"])
        resp = FakeResponse(STEP, headers={"Content-Disposition": 'attachment; filename="..\\..\\evil2.step"'})
        result = self.get("https://example.com/x", resp)
        self.assertEqual(os.path.dirname(result.path), self.dest)

    def test_the_extension_follows_the_content(self):
        self.assertEqual(os.path.basename(self.get("https://example.com/download", FakeResponse(STEP)).path), "download.step")
        self.assertEqual(os.path.basename(self.get("https://example.com/get.php?id=3", FakeResponse(ASCII_STL)).path), "get.stl")
        self.assertEqual(os.path.basename(self.get("https://example.com/model.bin", FakeResponse(zip_of({"a.step": STEP}))).path),
                         "model.zip")
        self.assertEqual(os.path.basename(self.get("https://example.com/part.stp", FakeResponse(STEP)).path), "part.stp")

    def test_an_unrelated_extension_is_kept_and_the_right_one_appended(self):
        result = self.get("https://example.com/notes.v2", FakeResponse(STEP))
        self.assertEqual(os.path.basename(result.path), "notes.v2.step")

    def test_a_bare_host_gets_a_default_name(self):
        result = self.get("https://example.com/", FakeResponse(STEP))
        self.assertEqual(os.path.basename(result.path), "download.step")

    def test_long_names_are_clipped_but_keep_their_extension(self):
        result = self.get("https://example.com/" + "a" * 300 + ".step", FakeResponse(STEP))
        name = os.path.basename(result.path)
        self.assertLessEqual(len(name), 120)
        self.assertTrue(name.endswith(".step"))

    def test_an_explicit_filename_is_used_as_given(self):
        result = self.get("https://example.com/whatever", FakeResponse(STEP), filename="my board.step")
        self.assertEqual(os.path.basename(result.path), "my board.step")
        result = self.get("https://example.com/whatever", FakeResponse(ASCII_STL), filename="bare")
        self.assertEqual(os.path.basename(result.path), "bare")

    def test_unsafe_explicit_filenames_are_refused_before_any_request(self):
        for bad in ("../x.step", "a/b.step", "..", ".hidden", "a\\b.step", "a..b.step", "x\n.step"):
            net = FakeGet(FakeResponse(STEP))
            with net.patch():
                result = netguard.download_verified("https://example.com/x", self.dest, bad)
            self.assertIn("unsafe filename", result.error, bad)
            self.assertEqual(net.calls, [], bad)
            self.assertEqual(self.files(), [])

    def test_names_that_are_unsafe_on_windows_are_refused_before_any_request(self):
        for bad in ("..\\x", "C:\\Users\\u\\x.bat", "C:y.bat", "CON", "nul.txt", "com1.stl", "trailing.", "trailing ",
                    "a|b.stl", "what?.stl", "tab\tname.stl", " "):
            net = FakeGet(FakeResponse(STEP))
            with net.patch():
                result = netguard.download_verified("https://example.com/x", self.dest, bad)
            self.assertIn("unsafe filename", result.error, repr(bad))
            self.assertEqual(net.calls, [], repr(bad))
        self.assertEqual(self.files(), [])

    @unittest.skipUnless(hasattr(os, "symlink"), "no symlinks")
    def test_a_link_in_the_folder_cannot_redirect_an_explicit_name_out_of_it(self):
        outside = isolation.fresh_dir("ng_out_")
        victim = os.path.join(outside, "victim.step")
        with open(victim, "w") as f:
            f.write("original")
        try:
            os.symlink(victim, os.path.join(self.dest, "link.step"))
        except OSError:
            self.skipTest("cannot create symlinks here")
        net = FakeGet(FakeResponse(STEP))
        with net.patch():
            result = netguard.download_verified("https://example.com/x", self.dest, "link.step")
        self.assertIn("unsafe filename", result.error)
        self.assertEqual(net.calls, [])                                  # refused before any request
        with open(victim) as f:
            self.assertEqual(f.read(), "original")

    def test_a_name_taken_from_the_url_is_always_sanitised_never_refused(self):
        cases = {
            "https://example.com/x/..%5C..%5Cevil.step": "evil.step",    # a decoded backslash splits like a separator
            "https://example.com/a%00b.step": "a_b.step",
            "https://example.com/files/bracket..v2.step": "bracket.v2.step",
            "https://example.com/.hidden.step": "hidden.step",
            "https://example.com/CON": "_CON.step",                      # a Windows device name is never used bare
            "https://example.com/nul..step": "_nul.step",
            "https://example.com/dir/": "dir.step",
        }
        for url, expected in cases.items():
            for name in self.files():
                os.remove(os.path.join(self.dest, name))
            result = self.get(url, FakeResponse(STEP))
            self.assertTrue(result.ok, (url, result))
            self.assertEqual(os.path.basename(result.path), expected, url)
            self.assertEqual(os.path.dirname(result.path), self.dest, url)

    def test_a_malformed_url_is_an_error_not_an_exception(self):
        for url in ("http://[::1/x.step", "https://[bad/x.step", "http://h]/x.step"):
            result = self.get(url, FakeResponse(STEP))
            self.assertFalse(result.ok, url)
            self.assertTrue(result.error, url)
            self.assertEqual(self.files(), [], url)

    def test_a_taken_name_is_never_overwritten_with_different_bytes(self):
        first = self.get("https://example.com/a.step", FakeResponse(STEP))
        other = STEP + b"! different\n"
        second = self.get("https://example.com/a.step", FakeResponse(other))
        self.assertEqual(os.path.basename(first.path), "a.step")
        self.assertEqual(os.path.basename(second.path), f"a-{hashlib.sha256(other).hexdigest()[:8]}.step")
        with open(first.path, "rb") as fh:
            self.assertEqual(fh.read(), STEP)
        with open(second.path, "rb") as fh:
            self.assertEqual(fh.read(), other)
        again = self.get("https://example.com/a.step", FakeResponse(other))
        self.assertEqual(again.path, second.path)
        self.assertEqual(self.get("https://example.com/a.step", FakeResponse(STEP)).path, first.path)
        self.assertEqual(len(self.files()), 2)

    # --- GitHub blob pages ---

    def test_a_github_blob_url_is_fetched_as_the_raw_file(self):
        result = self.get("https://github.com/acme/boards/blob/main/cad/pi4.step?raw=true#L1", FakeResponse(STEP))
        self.assertTrue(result.ok, result.error)
        self.assertEqual(self.net.calls[0][0], "https://raw.githubusercontent.com/acme/boards/main/cad/pi4.step")
        self.assertEqual(os.path.basename(result.path), "pi4.step")

    def test_github_raw_url_rewrites_only_blob_pages(self):
        raw = netguard.github_raw_url
        self.assertEqual(raw("https://github.com/o/r/blob/v1.2/dir/sub/file.stl"),
                         "https://raw.githubusercontent.com/o/r/v1.2/dir/sub/file.stl")
        self.assertEqual(raw("https://www.github.com/o/r/blob/main/a.step"), "https://raw.githubusercontent.com/o/r/main/a.step")
        self.assertEqual(raw("HTTPS://GitHub.com/o/r/blob/main/a.step"), "https://raw.githubusercontent.com/o/r/main/a.step")
        for unchanged in ("https://github.com/o/r/tree/main/dir", "https://github.com/o/r", "https://github.com/o/r/releases/download/v1/a.step",
                          "https://raw.githubusercontent.com/o/r/main/a.step", "https://gitlab.com/o/r/blob/main/a.step",
                          "https://example.com/blob/main/a.step", "https://github.com.evil.example/o/r/blob/main/a.step", "", "not a url"):
            self.assertEqual(raw(unchanged), unchanged)
        self.assertEqual(raw(None), None)

    # --- content checks ---

    def test_an_html_page_is_never_accepted_as_a_cad_file(self):
        result = self.get("https://example.com/model.step", FakeResponse(HTML, headers={"Content-Type": "text/html; charset=utf-8"}))
        self.assertFailed(result, "HTML web page", "raw")
        self.assertEqual(result.detected_type, "html")
        self.assertEqual(result.bytes, len(HTML))
        self.assertEqual(result.sha256, hashlib.sha256(HTML).hexdigest())

    def test_an_html_content_type_with_an_unrecognisable_body_is_html(self):
        result = self.get("https://example.com/model.step", FakeResponse(b"just some words", headers={"Content-Type": "text/html"}))
        self.assertFailed(result, "HTML")
        self.assertEqual(result.detected_type, "html")

    def test_a_git_lfs_pointer_is_rejected_with_the_media_url_hint(self):
        result = self.get("https://raw.githubusercontent.com/o/r/main/pi4.step", FakeResponse(LFS))
        self.assertFailed(result, "git-LFS pointer", "media.githubusercontent.com")
        self.assertEqual(result.detected_type, "lfs_pointer")

    def test_a_zero_byte_download_is_an_error_and_leaves_nothing(self):
        for resp in (FakeResponse(chunks=[]), FakeResponse(chunks=[b"", b""], headers={"Content-Length": "0"})):
            result = self.get("https://example.com/a.step", resp)
            self.assertFailed(result, "empty (0 bytes)")
            self.assertEqual((result.bytes, result.detected_type), (0, "empty"))

    def test_a_short_read_against_content_length_is_an_error(self):
        result = self.get("https://example.com/a.step", FakeResponse(STEP[:40], headers={"Content-Length": str(len(STEP))}))
        self.assertFailed(result, f"received 40 of {len(STEP)} bytes")
        self.assertEqual(result.bytes, 40)

    def test_a_compressed_body_is_not_compared_with_content_length(self):
        result = self.get("https://example.com/a.step", FakeResponse(STEP, headers={"Content-Length": "5", "Content-Encoding": "gzip"}))
        self.assertTrue(result.ok, result.error)

    def test_expected_types(self):
        result = self.get("https://example.com/a.step", FakeResponse(ASCII_STL), expect="step")
        self.assertFailed(result, "expected a STEP file but received an STL mesh")
        self.assertEqual(result.detected_type, "stl")
        result = self.get("https://example.com/a.zip", FakeResponse(zip_of({"a.step": STEP})), expect=("step", "zip"))
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.detected_type, "zip")
        result = self.get("https://example.com/b.step", FakeResponse(STEP), expect=["STEP"])
        self.assertTrue(result.ok, result.error)
        result = self.get("https://example.com/c.step", FakeResponse(b"abc"), expect="step")
        self.assertIn("expected a STEP file", result.error)
        self.assertEqual(result.detected_type, "unknown")

    def test_without_expect_any_other_content_is_accepted_and_reported(self):
        result = self.get("https://example.com/p.stl", FakeResponse(b"abc"))      # the legacy behaviour the older tests rely on
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.detected_type, "unknown")
        self.assertEqual(os.path.basename(result.path), "p.stl")
        result = self.get("https://example.com/doc.pdf", FakeResponse(b"%PDF-1.7\n1 0 obj"))
        self.assertEqual((result.ok, result.detected_type), (True, "pdf"))

    # --- limits ---

    def test_a_declared_size_over_the_cap_is_refused_without_writing(self):
        resp = FakeResponse(STEP, headers={"Content-Length": "10000"})
        result = self.get("https://example.com/a.step", resp, max_bytes=1000)
        self.assertFailed(result, "download too large", "cap")
        self.assertIn("MB", result.error)

    def test_a_stream_over_the_cap_is_aborted_and_cleaned_up(self):
        result = self.get("https://example.com/a.step", FakeResponse(chunks=[b"a" * 60, b"a" * 60, b"a" * 60]), max_bytes=100)
        self.assertFailed(result, "download exceeded", "aborted")

    def test_a_stream_exactly_at_the_cap_is_fine(self):
        result = self.get("https://example.com/a.step", FakeResponse(STEP), max_bytes=len(STEP))
        self.assertTrue(result.ok, result.error)
        result = self.get("https://example.com/b.step", FakeResponse(STEP), max_bytes=len(STEP) - 1)
        self.assertIn("download exceeded", result.error)

    def test_stop_request_between_chunks_aborts_and_cleans_up(self):
        polls = []

        def should_stop():
            polls.append(1)
            return len(polls) >= 2

        result = self.get("https://example.com/a.step", FakeResponse(chunks=[STEP[:30], STEP[30:60], STEP[60:]]), should_stop=should_stop)
        self.assertFailed(result, "download stopped")

    # --- transport faults ---

    def test_http_403_and_401_tell_the_model_to_ask_the_user_to_attach_the_file(self):
        for code in (401, 403):
            result = self.get("https://example.com/a.step", FakeResponse(status=code))
            self.assertFailed(result, f"HTTP {code}", "Do NOT retry", "attach it to this chat")

    def test_other_http_errors_name_the_exception(self):
        result = self.get("https://example.com/a.step", FakeResponse(status=404))
        self.assertFailed(result, "download failed: HTTPError")
        result = self.get("https://example.com/a.step", FakeResponse(status=503))
        self.assertFailed(result, "503")

    def test_connection_and_timeout_errors_are_returned_not_raised(self):
        for exc in (requests.exceptions.ConnectionError("no route"), requests.exceptions.ConnectTimeout("slow"),
                    requests.exceptions.ReadTimeout("slower"), requests.exceptions.InvalidURL("bad url"),
                    requests.exceptions.SSLError("certificate verify failed")):
            result = self.get("https://example.com/a.step", raises=exc)
            self.assertFailed(result, f"download failed: {type(exc).__name__}")

    def test_an_unexpected_exception_is_returned_not_raised(self):
        result = self.get("https://example.com/a.step", raises=RuntimeError("boom"))
        self.assertFailed(result, "download failed: RuntimeError: boom")

    def test_a_connection_cut_mid_stream_leaves_no_partial_file(self):
        resp = FakeResponse(chunks=[STEP[:50]], fail_after=requests.exceptions.ChunkedEncodingError("connection broken"))
        result = self.get("https://example.com/a.step", resp)
        self.assertFailed(result, "ChunkedEncodingError")
        resp = FakeResponse(chunks=[STEP[:50]], fail_after=ConnectionResetError("reset by peer"))
        result = self.get("https://example.com/a.step", resp)
        self.assertFalse(result.ok)
        self.assertEqual(self.files(), [])

    def test_an_unwritable_destination_is_an_error_not_an_exception(self):
        blocker = os.path.join(self.dest, "iam_a_file")
        with open(blocker, "w") as fh:
            fh.write("x")
        net = FakeGet(FakeResponse(STEP))
        with net.patch():
            result = netguard.download_verified("https://example.com/a.step", blocker)
        self.assertFalse(result.ok)
        self.assertIn("cannot write into", result.error)
        self.assertEqual(self.files(), ["iam_a_file"])

    def test_the_destination_directory_is_created(self):
        nested = os.path.join(self.dest, "new", "deeper")
        net = FakeGet(FakeResponse(STEP))
        with net.patch():
            result = netguard.download_verified("https://example.com/a.step", nested)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(os.listdir(nested), ["a.step"])

    # --- URL policy ---

    def test_only_http_urls_are_fetched(self):
        for bad in ("ftp://example.com/a.step", "file:///etc/passwd", "javascript:alert(1)", "example.com/a.step", "", None, 42):
            net = FakeGet(FakeResponse(STEP))
            with net.patch():
                result = netguard.download_verified(bad, self.dest)
            self.assertIn("only http(s) URLs are supported", result.error, repr(bad))
            self.assertEqual(net.calls, [], repr(bad))

    def test_allowed_hosts_restrict_the_url_and_the_final_url(self):
        allowed = ["raw.githubusercontent.com", "datasheets.raspberrypi.com"]
        net = FakeGet(FakeResponse(STEP))
        with net.patch():
            result = netguard.download_verified("https://evil.example.com/pi4.step", self.dest, allowed_hosts=allowed)
        self.assertIn("host 'evil.example.com' is not on the allowlist", result.error)
        self.assertEqual(net.calls, [])
        redirected = FakeResponse(STEP, url="https://evil.example.com/landing")
        result = self.get("https://raw.githubusercontent.com/o/r/main/pi4.step", redirected, allowed_hosts=allowed)
        self.assertFailed(result, "redirected to 'evil.example.com'")
        ok = FakeResponse(STEP, url="https://datasheets.raspberrypi.com/rpi4/pi4.step")
        result = self.get("https://raw.githubusercontent.com/o/r/main/pi4.step", ok, allowed_hosts=[h.upper() for h in allowed])
        self.assertTrue(result.ok, result.error)

    def test_the_allowlist_is_checked_after_the_blob_rewrite(self):
        result = self.get("https://github.com/o/r/blob/main/pi4.step", FakeResponse(STEP), allowed_hosts=["raw.githubusercontent.com"])
        self.assertTrue(result.ok, result.error)

    # --- the legacy tuple function ---

    def test_legacy_download_delegates_and_keeps_its_tuple_shape(self):
        net = FakeGet(FakeResponse(STEP))
        with net.patch():
            path, error = netguard.download("https://example.com/p.step", self.dest)
        self.assertEqual((os.path.basename(path), error), ("p.step", ""))
        net = FakeGet(FakeResponse(status=403))
        with net.patch():
            path, error = netguard.download("https://example.com/q.step", self.dest)
        self.assertEqual(path, "")
        self.assertIn("HTTP 403", error)
        path, error = netguard.download("ftp://example.com/q.step", self.dest)
        self.assertEqual((path, "only http(s)" in error), ("", True))
        net = FakeGet(FakeResponse(HTML))
        with net.patch():
            path, error = netguard.download("https://example.com/r.step", self.dest)
        self.assertEqual(path, "")
        self.assertIn("HTML", error)

    def test_legacy_download_keeps_its_signature_and_limits(self):
        self.assertEqual(netguard.MAX_DOWNLOAD_BYTES, 200 * 1024 * 1024)
        net = FakeGet(FakeResponse(STEP, headers={"Content-Length": "5000000"}))
        with net.patch():
            path, error = netguard.download("https://example.com/p.step", self.dest, "p.step", 1000, 5.0)
        self.assertEqual(path, "")
        self.assertIn("download too large", error)

    def test_concurrent_downloads_do_not_trample_each_other(self):
        bodies = {f"https://example.com/part{i}.step": STEP + f"#{i}\n".encode() for i in range(6)}
        net = FakeGet(lambda url: FakeResponse(bodies[url]))
        results = {}

        def work(url):
            results[url] = netguard.download_verified(url, self.dest)

        with net.patch():
            threads = [threading.Thread(target=work, args=(u,)) for u in bodies]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        for url, body in bodies.items():
            self.assertTrue(results[url].ok, results[url].error)
            with open(results[url].path, "rb") as fh:
                self.assertEqual(fh.read(), body)
        self.assertEqual(len(self.files()), 6)


class DefaultDownloadDirTest(unittest.TestCase):
    def test_env_var_wins_and_is_read_at_call_time(self):
        with mock.patch.dict(os.environ, {"OSCAR_DOWNLOAD_DIR": "/tmp/oscar_dl_a"}):
            self.assertEqual(netguard.default_download_dir(), os.path.abspath("/tmp/oscar_dl_a"))
            os.environ["OSCAR_DOWNLOAD_DIR"] = "/tmp/oscar_dl_b"
            self.assertEqual(netguard.default_download_dir(), os.path.abspath("/tmp/oscar_dl_b"))

    def test_default_is_downloads_oscar_under_home(self):
        env = {k: v for k, v in os.environ.items() if k != "OSCAR_DOWNLOAD_DIR"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(netguard.default_download_dir(), os.path.join(os.path.expanduser("~"), "Downloads", "Oscar"))
        with mock.patch.dict(os.environ, dict(env, OSCAR_DOWNLOAD_DIR=""), clear=True):
            self.assertEqual(netguard.default_download_dir(), os.path.join(os.path.expanduser("~"), "Downloads", "Oscar"))

    def test_tilde_is_expanded_and_the_path_is_absolute(self):
        with mock.patch.dict(os.environ, {"OSCAR_DOWNLOAD_DIR": "~/oscar_downloads"}):
            self.assertEqual(netguard.default_download_dir(), os.path.join(os.path.expanduser("~"), "oscar_downloads"))
        with mock.patch.dict(os.environ, {"OSCAR_DOWNLOAD_DIR": "relative/dir"}):
            self.assertTrue(os.path.isabs(netguard.default_download_dir()))

    def test_the_directory_is_not_created_by_asking(self):
        target = os.path.join(isolation.fresh_dir("ng_dd_"), "later")
        with mock.patch.dict(os.environ, {"OSCAR_DOWNLOAD_DIR": target}):
            self.assertEqual(netguard.default_download_dir(), target)
        self.assertFalse(os.path.exists(target))

    def test_the_test_sandbox_redirects_downloads(self):
        self.assertTrue(os.environ.get("OSCAR_DOWNLOAD_DIR"))
        self.assertEqual(netguard.default_download_dir(), os.path.abspath(os.environ["OSCAR_DOWNLOAD_DIR"]))


class ContentDispositionTest(unittest.TestCase):
    def name(self, header):
        return netguard.filename_from_content_disposition(header)

    def test_plain_quoted_and_token_forms(self):
        self.assertEqual(self.name('attachment; filename="a.step"'), "a.step")
        self.assertEqual(self.name("attachment; filename=a.step"), "a.step")
        self.assertEqual(self.name('attachment; filename="a b.step"; size=3'), "a b.step")
        self.assertEqual(self.name('ATTACHMENT; FILENAME="Upper.STEP"'), "Upper.STEP")
        self.assertEqual(self.name('attachment; filename="a\\"b.step"'), 'a"b.step')

    def test_rfc5987_form(self):
        self.assertEqual(self.name("attachment; filename*=UTF-8''%E2%82%AC%20part.step"), "\u20ac part.step")
        self.assertEqual(self.name("attachment; filename*=utf-8''plain.step"), "plain.step")
        self.assertEqual(self.name("attachment; filename*=bogus-charset''a%20b.step"), "a b.step")
        self.assertEqual(self.name("attachment; filename=\"old.txt\"; filename*=UTF-8''new.step"), "new.step")

    def test_nothing_or_nothing_useful(self):
        for header in ("", None, "inline", "attachment", 'attachment; filename=""', "attachment; size=3"):
            self.assertEqual(self.name(header), "", repr(header))

    def test_only_the_last_path_component_is_returned(self):
        self.assertEqual(self.name('attachment; filename="../../etc/passwd"'), "passwd")
        self.assertEqual(self.name('attachment; filename="C:\\Users\\x\\a.step"'), "a.step")
        self.assertEqual(self.name('attachment; filename="/abs/path/b.step"'), "b.step")


def _zip_with(members, **kw):
    """Write a zip of ``members`` ({name: bytes or ZipInfo-tuple}) to a temp file and return its path."""
    path = os.path.join(isolation.fresh_dir("ng_zip_"), "archive.zip")
    with zipfile.ZipFile(path, "w", **kw) as zf:
        for name, data in members.items():
            if isinstance(data, tuple):                    # (ZipInfo attrs, bytes)
                info = zipfile.ZipInfo(name)
                info.external_attr = data[0]
                zf.writestr(info, data[1])
            else:
                zf.writestr(name, data)
    return path


class ExtractZipTest(unittest.TestCase):
    def setUp(self):
        self.base = isolation.fresh_dir("ng_x_")
        self.dest = os.path.join(self.base, "out")

    def tree(self, root=None):
        found = []
        for folder, _dirs, names in os.walk(root or self.base):
            found.extend(os.path.relpath(os.path.join(folder, n), self.base) for n in names)
        return sorted(found)

    def test_extracts_files_and_lists_the_cad_ones(self):
        path = _zip_with({"pi4/Model.STEP": STEP, "pi4/readme.txt": b"hi", "pi4/sub/mesh.stl": ASCII_STL, "top.step": STEP,
                          "pi4/": b"", "__MACOSX/pi4/._Model.STEP": b"junk", ".DS_Store": b"x", "pi4/.DS_Store": b"x"})
        result = netguard.extract_zip(path, self.dest)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.error, "")
        self.assertEqual(self.tree(self.dest), sorted(os.path.relpath(os.path.join(self.dest, p), self.base)
                                                      for p in ("pi4/Model.STEP", "pi4/readme.txt", "pi4/sub/mesh.stl", "top.step")))
        self.assertEqual(result.cad_files, sorted(os.path.normpath(os.path.join(result.dest_dir, p))
                                                  for p in ("pi4/Model.STEP", "pi4/sub/mesh.stl", "top.step")))
        self.assertEqual(sorted(result.files), sorted(os.path.normpath(os.path.join(result.dest_dir, p))
                                                      for p in ("pi4/Model.STEP", "pi4/readme.txt", "pi4/sub/mesh.stl", "top.step")))
        self.assertEqual(result.total_bytes, len(STEP) * 2 + 2 + len(ASCII_STL))
        self.assertEqual(result.skipped, [])
        with open(os.path.join(self.dest, "pi4", "Model.STEP"), "rb") as fh:
            self.assertEqual(fh.read(), STEP)
        self.assertFalse([n for n in os.listdir(os.path.join(self.dest, "pi4")) if ".tmp-" in n])

    def test_zip_slip_is_refused_and_nothing_is_written(self):
        outside = os.path.join(self.base, "evil.txt")
        for name in ("../evil.txt", "a/../../evil.txt", "/abs_evil.txt", "..\\evil.txt", "C:\\evil.txt", "c:/evil.txt", "ok/../../../evil.txt",
                     "a\nb.step", "bell\x07.step"):
            path = _zip_with({"fine.step": STEP, name: b"pwned"})
            result = netguard.extract_zip(path, self.dest)
            self.assertFalse(result.ok, repr(name))
            self.assertIn("unsafe member path", result.error, repr(name))
            self.assertEqual((result.files, result.cad_files, result.total_bytes), ([], [], 0), repr(name))
            self.assertFalse(os.path.exists(self.dest), repr(name))
            self.assertFalse(os.path.exists(outside), repr(name))
            self.assertFalse(os.path.exists("/abs_evil.txt"))

    def test_member_through_a_symlink_in_the_destination_is_refused_and_rolled_back(self):
        elsewhere = os.path.join(self.base, "elsewhere")
        os.makedirs(elsewhere)
        os.makedirs(self.dest)
        os.symlink(elsewhere, os.path.join(self.dest, "link"))
        path = _zip_with({"ok.step": STEP, "link/evil.txt": b"pwned"})
        result = netguard.extract_zip(path, self.dest)
        self.assertIn("resolves outside the destination", result.error)
        self.assertEqual(result.files, [])
        self.assertEqual(os.listdir(elsewhere), [])
        self.assertFalse(os.path.exists(os.path.join(self.dest, "ok.step")), "files written before the refusal are rolled back")

    def test_too_many_members(self):
        path = _zip_with({f"f{i}.txt": b"x" for i in range(6)})
        result = netguard.extract_zip(path, self.dest, max_members=5)
        self.assertIn("6 members exceed the limit of 5", result.error)
        self.assertFalse(os.path.exists(self.dest))
        self.assertTrue(netguard.extract_zip(path, self.dest, max_members=6).ok)

    def test_declared_size_over_the_cap_is_a_zip_bomb(self):
        path = _zip_with({"big.bin": b"0" * 5000}, compression=zipfile.ZIP_DEFLATED)
        self.assertLess(os.path.getsize(path), 1000)                    # tiny on disk, large when expanded
        result = netguard.extract_zip(path, self.dest, max_total_bytes=1000)
        self.assertIn("possible zip bomb", result.error)
        self.assertIn("would expand", result.error)
        self.assertFalse(os.path.exists(self.dest))
        self.assertTrue(netguard.extract_zip(path, self.dest, max_total_bytes=5000).ok)

    def test_the_cap_is_enforced_on_the_bytes_actually_written(self):
        path = _zip_with({"a.bin": b"0" * 3000, "b.bin": b"1" * 3000})
        real_infolist = zipfile.ZipFile.infolist

        def understated(self):
            infos = real_infolist(self)
            for info in infos:
                info.file_size = 1                 # a header that lies about the size
            return infos

        with mock.patch.object(zipfile.ZipFile, "infolist", understated), \
                mock.patch.object(zipfile.ZipFile, "open", lambda zf, info, *a, **k: io.BytesIO(b"x" * 3000)):
            result = netguard.extract_zip(path, self.dest, max_total_bytes=4000)
        self.assertIn("exceeds the", result.error)
        self.assertEqual((result.files, result.total_bytes), ([], 0))
        self.assertEqual(self.tree(), [], "nothing is left behind after the cap trips")

    def test_a_corrupt_member_aborts_and_removes_what_was_written(self):
        path = _zip_with({"a.step": b"A" * 64, "b.step": b"BBBBBBBBBBBBBBBB" * 4})
        data = bytearray(open(path, "rb").read())
        data[data.index(b"BBBBBBBB")] ^= 0x01
        with open(path, "wb") as fh:
            fh.write(bytes(data))
        result = netguard.extract_zip(path, self.dest)
        self.assertIn("not a valid ZIP archive", result.error)
        self.assertEqual((result.files, result.cad_files, result.total_bytes), ([], [], 0))
        self.assertEqual(self.tree(self.dest), [])

    def test_symlink_members_are_skipped_and_reported(self):
        path = _zip_with({"real.step": STEP, "link.step": (0o120777 << 16, b"/etc/passwd")})
        result = netguard.extract_zip(path, self.dest)
        self.assertTrue(result.ok, result.error)
        self.assertEqual([os.path.basename(f) for f in result.files], ["real.step"])
        self.assertEqual(result.skipped, ["link.step: symbolic link not extracted"])
        self.assertFalse(os.path.lexists(os.path.join(self.dest, "link.step")))

    def test_encrypted_members_are_skipped_and_reported(self):
        path = _zip_with({"secret.step": b"x" * 20, "open.step": STEP})
        data = bytearray(open(path, "rb").read())
        central = data.index(b"PK\x01\x02")                          # the first central-directory record is secret.step
        data[central + 8] |= 0x01                                     # general-purpose flag bit 0: encrypted
        with open(path, "wb") as fh:
            fh.write(bytes(data))
        result = netguard.extract_zip(path, self.dest)
        self.assertTrue(result.ok, result.error)
        self.assertEqual([os.path.basename(f) for f in result.files], ["open.step"])
        self.assertEqual(result.skipped, ["secret.step: encrypted member not extracted"])

    def test_not_a_zip_and_missing_files(self):
        bogus = os.path.join(self.base, "bogus.zip")
        with open(bogus, "wb") as fh:
            fh.write(b"this is not a zip")
        result = netguard.extract_zip(bogus, self.dest)
        self.assertIn("not a valid ZIP archive", result.error)
        result = netguard.extract_zip(os.path.join(self.base, "missing.zip"), self.dest)
        self.assertIn("zip extraction failed", result.error)
        self.assertFalse(os.path.exists(self.dest))

    def test_an_empty_archive_is_ok_with_no_files(self):
        path = _zip_with({})
        result = netguard.extract_zip(path, self.dest)
        self.assertTrue(result.ok, result.error)
        self.assertEqual((result.files, result.cad_files), ([], []))

    def test_extracting_into_an_existing_directory_keeps_other_files(self):
        os.makedirs(self.dest)
        with open(os.path.join(self.dest, "keep.txt"), "w") as fh:
            fh.write("keep")
        result = netguard.extract_zip(_zip_with({"new.step": STEP}), self.dest)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(sorted(os.listdir(self.dest)), ["keep.txt", "new.step"])

    def test_a_refused_archive_does_not_delete_files_that_were_already_there(self):
        os.makedirs(self.dest)
        with open(os.path.join(self.dest, "keep.txt"), "w") as fh:
            fh.write("keep")
        result = netguard.extract_zip(_zip_with({"new.step": STEP, "../x.txt": b"y"}), self.dest)
        self.assertFalse(result.ok)
        self.assertEqual(os.listdir(self.dest), ["keep.txt"])

    def test_safe_member_parts(self):
        parts = netguard._safe_member_parts
        self.assertEqual(parts("a/b/c.step"), ["a", "b", "c.step"])
        self.assertEqual(parts("./a//b.step"), ["a", "b.step"])
        self.assertEqual(parts("dir/"), ["dir"])
        self.assertEqual(parts("a\\b.step"), ["a", "b.step"])
        for bad in ("", "../a", "a/../b", "/a", "\\a", "C:/a", "d:\\a", "a\x00b", "a\tb"):
            self.assertIsNone(parts(bad), repr(bad))

    def test_downloaded_zip_then_extracted_end_to_end(self):
        dest = isolation.fresh_dir("ng_e2e_")
        body = zip_of({"Raspberry-Pi-4/Pi4.step": STEP, "Raspberry-Pi-4/LICENSE.txt": b"CC-BY"})
        net = FakeGet(FakeResponse(body, headers={"Content-Length": str(len(body))}))
        with net.patch():
            downloaded = netguard.download_verified("https://example.com/files/pi4", dest, expect=("step", "zip"))
        self.assertTrue(downloaded.ok, downloaded.error)
        self.assertEqual((os.path.basename(downloaded.path), downloaded.detected_type), ("pi4.zip", "zip"))
        extracted = netguard.extract_zip(downloaded.path, os.path.join(dest, "pi4"))
        self.assertTrue(extracted.ok, extracted.error)
        self.assertEqual([os.path.basename(f) for f in extracted.cad_files], ["Pi4.step"])


class PermissionDialogTest(unittest.TestCase):
    """The permission dialog (a QMessageBox built on the GUI thread) is built from fakes: it carries the object name
    ``oscarPermission`` that ``ui_rebrand`` styles and the GUI smoke test looks for, shows the summary and detail, defaults to Deny and
    maps the clicked button to the answer."""

    class FakeBox:
        Warning, NoButton, YesRole, AcceptRole, NoRole = 1, 0, 2, 3, 4
        instances = []
        click = "deny"

        def __init__(self, icon, title, text, buttons, parent):
            self.icon, self.title, self.text, self.parent = icon, title, text, parent
            self.name, self.informative, self.buttons, self.default, self.modal, self.shown = "", "", {}, None, None, False
            self._clicked = None
            type(self).instances.append(self)

        def setObjectName(self, name):
            self.name = name

        def objectName(self):
            return self.name

        def setInformativeText(self, text):
            self.informative = text

        def addButton(self, label, role):
            button = object()
            self.buttons[label] = button
            return button

        def setDefaultButton(self, button):
            self.default = button

        def setModal(self, value):
            self.modal = value

        def show(self):
            self.shown = True

        def raise_(self):
            pass

        def activateWindow(self):
            pass

        def exec(self):
            self._clicked = self.buttons[{"deny": "Deny", "once": "Allow once", "always": "Always allow"}[type(self).click]]

        def clickedButton(self):
            return self._clicked

    class FakeApp:
        window = object()

        @staticmethod
        def instance():
            return PermissionDialogTest.FakeApp

        @staticmethod
        def activeWindow():
            return PermissionDialogTest.FakeApp.window

    def setUp(self):
        from unittest import mock as _mock

        from PySide import QtWidgets
        from oscar import params as _params
        from oscar.cad import dispatch

        netguard.reset_turn()
        self.FakeBox.instances = []
        self.waits = []
        patches = [
            _mock.patch.object(QtWidgets, "QMessageBox", self.FakeBox),
            _mock.patch.object(QtWidgets, "QApplication", self.FakeApp),
            _mock.patch.object(sys.modules["FreeCADGui"], "getMainWindow", create=True, return_value=object()),
            _mock.patch.object(dispatch, "dispatch", side_effect=lambda fn, timeout=0, op="", **kw: (fn(), False)),
            _mock.patch.object(_params, "get_bool", return_value=False),
            _mock.patch.object(_params, "set_bool"),
            _mock.patch.object(netguard, "on_permission_wait", self.waits.append),
        ]
        self.set_bool = patches[5].start()
        for patch in patches[:5] + patches[6:]:
            patch.start()
        for patch in patches:
            self.addCleanup(patch.stop)
        self.addCleanup(netguard.reset_turn)
        self.addCleanup(_params.clear_net_grants)
        _params.clear_net_grants()

    def ask(self, click="deny", detail="Reason: test"):
        self.FakeBox.click = click
        netguard.reset_turn()
        return netguard.request_permission("Oscar wants to download a file:\n\u2022 https://example.com/a.step", detail=detail)

    def test_the_dialog_has_the_object_name_the_rebrand_relies_on(self):
        self.assertEqual(self.ask("once"), "allow")
        (box,) = self.FakeBox.instances
        self.assertEqual(box.objectName(), "oscarPermission")

    def test_text_title_parent_and_modality(self):
        self.ask()
        (box,) = self.FakeBox.instances
        self.assertIn("https://example.com/a.step", box.text)
        self.assertEqual(box.informative, "Reason: test")
        self.assertIn("network access", box.title)
        self.assertIs(box.parent, self.FakeApp.window)
        self.assertTrue(box.modal and box.shown)

    def test_three_buttons_and_deny_is_the_default(self):
        self.ask()
        (box,) = self.FakeBox.instances
        self.assertEqual(sorted(box.buttons), ["Allow once", "Always allow", "Deny"])
        self.assertIs(box.default, box.buttons["Deny"])

    def test_every_answer_maps_to_a_verdict(self):
        self.assertEqual(self.ask("deny"), "deny")
        self.assertEqual(self.ask("once"), "allow")
        self.set_bool.assert_not_called()
        self.assertEqual(self.ask("always"), "always")
        self.set_bool.assert_not_called()       # migration (F-26): no legacy global flag; a visible per-(tool, host) grant instead
        self.assertEqual([(g["tool"], g["host"]) for g in netguard.list_grants()], [("network", "*")])
        self.assertEqual({b.name for b in self.FakeBox.instances}, {"oscarPermission"})

    def test_the_wait_hook_brackets_the_dialog(self):
        self.ask()
        self.assertEqual(self.waits[0], True)
        self.assertEqual(self.waits[-1], False)

    def test_no_detail_means_no_informative_text(self):
        self.ask(detail="")
        self.assertEqual(self.FakeBox.instances[0].informative, "")


class SsrfGuardTest(DownloadVerifiedBase):
    """download_verified never fetches (or follows a redirect to) an internal address."""

    BAD = ("http://127.0.0.1/a.step", "http://localhost:8080/a.step", "http://10.0.0.5/a.step", "http://192.168.1.1/a.step",
           "http://172.16.3.4/a.step", "http://169.254.169.254/latest/meta-data", "http://[::1]/a.step", "http://[fd00::1]/a",
           "http://[::ffff:127.0.0.1]/a", "http://2130706433/a.step", "http://0x7f.1/a.step", "http://printer.local/a.step",
           "http://metadata.google.internal/x", "http://intranet/a.step", "http://100.64.0.1/a", "http://0.0.0.0/a")

    def test_internal_urls_are_refused_before_any_request(self):
        for url in self.BAD:
            with self.subTest(url):
                result = self.get(url, FakeResponse(STEP))
                self.assertFailed(result, "download refused")
                self.assertEqual(self.net.calls, [])

    def test_a_non_http_scheme_is_refused(self):
        for url in ("file:///etc/passwd", "ftp://example.com/a.step"):
            self.assertFailed(self.get(url, FakeResponse(STEP)), "http(s)")

    def test_a_host_name_resolving_to_a_private_address_is_refused(self):
        import socket
        with mock.patch.object(socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("10.1.2.3", 0))]):
            result = self.get("https://rebind.example.com/a.step", FakeResponse(STEP))
        self.assertFailed(result, "resolves to 10.1.2.3")
        self.assertEqual(self.net.calls, [])

    def test_a_public_address_passes(self):
        import socket
        with mock.patch.object(socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 0))]):
            self.assertTrue(self.get("https://example.com/a.step", FakeResponse(STEP)).ok)

    def test_every_redirect_hop_is_validated(self):
        def answer(url):
            if url.startswith("https://example.com/"):
                return FakeResponse(b"", status=302, headers={"Location": "http://169.254.169.254/latest/meta-data"})
            return FakeResponse(STEP)
        result = self.get("https://example.com/a.step", answer)
        self.assertFailed(result, "download refused", "metadata")
        self.assertEqual([c[0] for c in self.net.calls], ["https://example.com/a.step"])
        self.assertIs(self.net.calls[0][1]["allow_redirects"], False)

    def test_a_relative_redirect_is_followed_and_a_loop_is_capped(self):
        def answer(url):
            if url.endswith("/start"):
                return FakeResponse(b"", status=301, headers={"Location": "/final.step"})
            return FakeResponse(STEP)
        self.assertTrue(self.get("https://example.com/start", answer).ok)
        self.assertEqual(self.net.calls[-1][0], "https://example.com/final.step")
        for name in self.files():
            os.remove(os.path.join(self.dest, name))
        loop = self.get("https://example.com/a", lambda url: FakeResponse(b"", status=302, headers={"Location": url}))
        self.assertFailed(loop, "redirects")
        self.assertEqual(len(self.net.calls), netguard.MAX_REDIRECTS + 1)

    def test_the_final_reported_url_is_checked_too(self):
        result = self.get("https://example.com/a.step", FakeResponse(STEP, url="http://10.0.0.9/a.step"))
        self.assertFailed(result, "download refused")

    def test_ssrf_check_comes_before_the_allowlist(self):
        result = self.get("http://127.0.0.1/a.step", FakeResponse(STEP), allowed_hosts=["127.0.0.1"])
        self.assertFailed(result, "internal addresses")


class AnalyzeImportFormsTest(unittest.TestCase):
    """The parse sees every import form the old line scan missed; the line scan stays for snippets that do not parse."""

    def reasons(self, code):
        return netguard.analyze(code)["reasons"]

    def test_forms_the_line_scan_alone_would_miss(self):
        for code in ("import os, socket", "x = 1; import socket", "try: import socket\nexcept ImportError: pass",
                     "def f():\n    import socket", "from http import client", "import os, \\\n    socket",
                     "from http import (\n    client,\n)", "import ssl", "import urllib3", "import xmlrpc.client"):
            self.assertTrue(netguard.analyze(code)["network"], code)

    def test_the_most_specific_module_is_named_back(self):
        self.assertIn("import xmlrpc.client", self.reasons("from xmlrpc import client"))
        self.assertNotIn("import xmlrpc", self.reasons("from xmlrpc import client"))
        self.assertEqual(self.reasons("import xmlrpc.server"), ["import xmlrpc"])
        self.assertIn("import http.client", self.reasons("from http import client"))

    def test_a_snippet_that_does_not_parse_is_still_scanned(self):
        self.assertTrue(netguard.analyze("import os, socket\nthis is ( not python\n")["network"])
        self.assertTrue(netguard.analyze("x = (\nimport socket; from http import client")["network"])

    def test_new_fetch_calls_are_flagged_and_plain_names_are_not(self):
        for code in ("sock = create_connection(addr)", "ip = getaddrinfo(h, 80)", "c = HTTPSConnection(h)",
                     "m = SMTP(h)", "http = urllib3.PoolManager()"):
            self.assertTrue(any(r.startswith("fetch call") for r in self.reasons(code)), code)
        for code in ("ssl_depth = 3\nsocketry = 2", "def start_server_label():\n    return 1", "import os, sys, math, json"):
            self.assertFalse(netguard.analyze(code)["network"], code)


class AnalyzeDynamicExecTest(unittest.TestCase):
    def test_exec_of_a_network_string_is_flagged(self):
        for code in ("exec('import socket')", "eval(\"__import__('urllib.request')\")",
                     "exec(compile('import requests', 'x', 'exec'))"):
            verdict = netguard.analyze(code)
            self.assertTrue(verdict["network"], code)
            self.assertTrue(any("dynamic code" in r for r in verdict["reasons"]), code)

    def test_exec_without_network_markers_is_not(self):
        self.assertFalse(netguard.analyze("exec('x = 1')\nv = eval('2+2')")["network"])


class ScriptGuardTest(unittest.TestCase):
    def setUp(self):
        netguard.reset_turn()
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in ("OSCAR_ALLOW_NET", "OSCAR_NET_AUDIT"):
            os.environ.pop(key, None)
        self.addCleanup(netguard.reset_turn)

    @staticmethod
    def connect():
        import socket
        socket.create_connection(("203.0.113.9", 9), timeout=0.2)

    def test_installed_at_most_once(self):
        with netguard.script_guard():
            pass
        self.assertFalse(netguard.install_audit_hook())

    def test_blocks_on_the_guarded_thread_only(self):
        seen = {}
        go = threading.Event()

        def other():
            go.wait(10)
            try:
                self.connect()
            except Exception as exc:  # noqa: BLE001
                seen["exc"] = exc
        worker = threading.Thread(target=other)
        worker.start()                  # a thread ordinary code started BEFORE the script: never the script's
        with netguard.script_guard():
            with self.assertRaises(PermissionError):
                self.connect()
            go.set()                    # it works while the script runs
            worker.join(10)
        self.assertIn("exc", seen)
        self.assertNotIsInstance(seen.get("exc"), PermissionError)   # an ordinary network failure, not our block
        with self.assertRaises(OSError) as ctx:                      # and the guard is gone after the context
            self.connect()
        self.assertNotIsInstance(ctx.exception, PermissionError)

    def test_a_thread_the_script_starts_inherits_the_guard(self):
        seen = {}

        def other():
            try:
                self.connect()
            except Exception as exc:  # noqa: BLE001
                seen["exc"] = exc
        with netguard.script_guard():
            worker = threading.Thread(target=other)
            worker.start()
            worker.join(10)
        self.assertIsInstance(seen.get("exc"), PermissionError)

    def test_a_turn_grant_lifts_the_block(self):
        netguard._TURN_DECISION["download"] = "allow"
        with netguard.script_guard():
            with self.assertRaises(OSError) as ctx:
                self.connect()
        self.assertNotIsInstance(ctx.exception, PermissionError)

    def test_a_denied_turn_stays_blocked_and_kill_switch_unblocks(self):
        netguard._TURN_DECISION["download"] = "deny"
        with netguard.script_guard():
            with self.assertRaises(PermissionError):
                self.connect()
        os.environ["OSCAR_NET_AUDIT"] = "0"
        with netguard.script_guard():
            with self.assertRaises(OSError) as ctx:
                self.connect()
        self.assertNotIsInstance(ctx.exception, PermissionError)

    def test_network_command_lines_are_blocked_other_commands_are_not(self):
        import subprocess
        with netguard.script_guard():
            with self.assertRaises(PermissionError):
                subprocess.run(["curl", "-s", "http://203.0.113.9/"])
            with self.assertRaises(PermissionError):
                os.system("wget http://203.0.113.9/x")
            if os.name != "nt":     # `echo` is a shell built-in on Windows, not a program the guard can resolve
                self.assertEqual(subprocess.run(["echo", "hi"], capture_output=True).returncode, 0)

    def test_reentrant(self):
        with netguard.script_guard():
            with netguard.script_guard():
                pass
            with self.assertRaises(PermissionError):
                self.connect()


if __name__ == "__main__":
    unittest.main()
