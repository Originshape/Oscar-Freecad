"""``fetch_url`` and ``web_search`` (oscar.tools_web): permission per (tool, host), SSRF guard, caps, extraction, provenance.

Never touches the real network: a local HTTP server stands in for the web and ``netguard.check_public_url`` is wrapped so the
loopback names of that server pass while every other address still goes through the REAL guard."""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import context_guard, netguard, params, tools, tools_web  # noqa: E402
from oscar.tools import ToolError  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402

os.environ.pop("OSCAR_ALLOW_NET", None)
os.environ.pop(tools_web.SEARCH_URL_ENV, None)

HTML = """<html><head><title>Raspberry Pi 4 &amp; friends</title><style>.a{color:red}</style>
<script>var secret = 'SCRIPT_TEXT';</script></head><body>
<nav><a href="/x">NAV_TEXT</a></nav>
<h1>Mechanical drawing</h1><p>Board is 85 mm x 56 mm.</p><ul><li>Hole one</li><li>Hole two</li></ul>
<script>other()</script><p>Caf&eacute;   spacing</p></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    hits: list = []

    def log_message(self, *args):
        pass

    def _send(self, code, ctype, body, extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except ConnectionError:      # the client stopped reading at its size cap
            pass

    def do_GET(self):  # noqa: N802
        type(self).hits.append(self.path)
        port = self.server.server_address[1]
        path = self.path.split("?")[0]
        if path == "/page.html":
            self._send(200, "text/html; charset=utf-8", HTML.encode())
        elif path == "/plain.txt":
            self._send(200, "text/plain", b"plain datasheet text")
        elif path == "/data.json":
            self._send(200, "application/json", b'{"a": 1}')
        elif path == "/file.step":
            self._send(200, "application/octet-stream", b"ISO-10303-21;")
        elif path == "/nul.txt":
            self._send(200, "text/plain", b"abc\x00\x01\x02binary")
        elif path == "/big.txt":
            self._send(200, "text/plain", b"x" * (tools_web.MAX_BYTES + 10))
        elif path == "/stream-big.txt":      # no Content-Length: the cap must hold on the stream itself
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            try:
                for _ in range(40):
                    self.wfile.write(b"z" * 65536)
            except ConnectionError:
                pass
        elif path == "/long.txt":
            self._send(200, "text/plain", b"y" * 30000)
        elif path == "/empty.html":
            self._send(200, "text/html", b"<html><script>x()</script></html>")
        elif path == "/forbidden":
            self._send(403, "text/html", b"no")
        elif path == "/same":
            self._send(302, "text/plain", b"", [("Location", "/page.html")])
        elif path == "/other-host":
            self._send(302, "text/plain", b"", [("Location", f"http://localhost:{port}/page.html")])
        elif path == "/internal":
            self._send(302, "text/plain", b"", [("Location", "http://169.254.169.254/latest/meta-data/")])
        elif path == "/loop":
            self._send(302, "text/plain", b"", [("Location", "/loop")])
        elif path == "/search":
            query = self.path.split("q=", 1)[-1].split("&")[0]
            results = [{"title": "Pi 4 mechanical drawing " + query, "url": "https://datasheets.raspberrypi.com/rpi4/x.pdf",
                        "content": "Official  drawing\nof the board"},
                       {"title": "bad scheme", "url": "javascript:alert(1)", "content": "x"},
                       {"title": "Second", "url": "https://github.com/raspberrypi/hats", "content": "y" * 900}]
            self._send(200, "application/json", json.dumps({"results": results}).encode())
        elif path == "/nojson/search":
            self._send(200, "text/html", b"<html>json disabled</html>")
        elif path == "/badshape/search":
            self._send(200, "application/json", b'{"nope": 1}')
        else:
            self._send(404, "text/plain", b"missing")


def _local_check(real):
    def check(url):
        host = (tools_web._host_of(url))
        if host in ("127.0.0.1", "localhost"):
            return ""
        return real(url)
    return check


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _Handler.hits = []
        os.environ.pop("OSCAR_ALLOW_NET", None)
        os.environ.pop(tools_web.SEARCH_URL_ENV, None)
        params.clear_net_grants()
        params.set_string(tools_web.ALLOWLIST_SETTING, "")
        params.set_string(tools_web.SEARCH_URL_SETTING, "")
        netguard.reset_turn()
        self.questions = []
        self.verdicts = {}
        real = netguard.check_public_url
        for patcher in (mock.patch.object(netguard, "check_public_url", _local_check(real)),
                        mock.patch.object(netguard, "request_permission", self._permission)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(params.clear_net_grants)
        self.addCleanup(netguard.reset_turn)
        self.addCleanup(params.set_string, tools_web.SEARCH_URL_SETTING, "")
        self.addCleanup(params.set_string, tools_web.ALLOWLIST_SETTING, "")

    def _permission(self, summary, detail="", *, tool="", host=""):
        self.questions.append({"summary": summary, "detail": detail, "tool": tool, "host": host})
        return self.verdicts.get(host, self.verdicts.get("*", "allow"))

    def fetch(self, path, **kw):
        return tools_web._fetch_url(url=self.base + path if path.startswith("/") else path, **kw)


class FetchUrlTest(_Base):
    def test_html_becomes_text_without_script_style_nav(self):
        text = self.fetch("/page.html", purpose="board outline")
        body = text.split("\n\n", 1)[1]
        self.assertIn("Raspberry Pi 4 & friends", body)         # the title, entities decoded
        self.assertIn("Board is 85 mm x 56 mm.", body)
        self.assertIn("Caf\u00e9 spacing", body)                # whitespace collapsed
        for dropped in ("SCRIPT_TEXT", "NAV_TEXT", "color:red", "other()"):
            self.assertNotIn(dropped, body)
        self.assertNotIn("\n\n\n", body)

    def test_reply_reports_final_url_type_size_and_hash(self):
        text = self.fetch("/page.html", purpose="x")
        lines = text.split("\n")
        self.assertEqual(lines[0], f"FETCHED {self.base}/page.html")
        self.assertIn(f"final_url: {self.base}/page.html", lines)
        self.assertIn("content_type: text/html", lines)
        self.assertTrue(any(line.startswith("bytes: ") for line in lines))
        self.assertTrue(any(len(line) == 72 and line.startswith("sha256: ") for line in lines))

    def test_final_url_after_a_same_host_redirect(self):
        text = self.fetch("/same", purpose="x")
        self.assertIn(f"final_url: {self.base}/page.html", text)
        self.assertEqual(len(self.questions), 1, "a redirect inside the approved host asks nothing new")

    def test_plain_and_json_are_accepted(self):
        self.assertIn("plain datasheet text", self.fetch("/plain.txt", purpose="x"))
        self.assertIn("content_type: application/json", self.fetch("/data.json", purpose="x"))

    def test_binary_content_type_is_refused(self):
        with self.assertRaises(ToolError) as ctx:
            self.fetch("/file.step", purpose="x")
        self.assertIn("application/octet-stream", str(ctx.exception))
        self.assertIn("source_reference_part", str(ctx.exception))

    def test_binary_body_labelled_text_is_refused(self):
        with self.assertRaises(ToolError) as ctx:
            self.fetch("/nul.txt", purpose="x")
        self.assertIn("binary", str(ctx.exception))

    def test_size_cap_two_megabytes(self):
        self.assertEqual(tools_web.MAX_BYTES, 2 * 1024 * 1024)
        with self.assertRaises(ToolError) as ctx:
            self.fetch("/big.txt", purpose="x")
        self.assertIn("2 MB", str(ctx.exception))

    def test_size_cap_holds_on_a_stream_without_content_length(self):
        with self.assertRaises(ToolError) as ctx:
            self.fetch("/stream-big.txt", purpose="x")
        self.assertIn("larger than the 2 MB limit", str(ctx.exception))

    def test_text_is_clipped_to_max_chars_and_says_so(self):
        text = self.fetch("/long.txt", purpose="x", max_chars=1000)
        self.assertIn("text: 1000 of 30000 chars", text)
        note, page = text.split("\n\n", 1)[1].split("\n", 1)
        self.assertEqual(note, context_guard.UNTRUSTED_NOTE.format(source="this web page"))   # data, not instructions
        self.assertEqual(len(page), 1000)
        wide = self.fetch("/long.txt", purpose="x")                       # default 20000
        self.assertIn("text: 20000 of 30000 chars", wide)
        low = self.fetch("/long.txt", purpose="x", max_chars=1)           # clamped up to the minimum
        self.assertIn(f"text: {tools_web.MIN_MAX_CHARS} of 30000", low)

    def test_bad_arguments(self):
        for bad in ("ftp://example.com/x", "example.com/x", "", "file:///etc/passwd"):
            with self.assertRaises(ToolError):
                tools_web._fetch_url(url=bad, purpose="x")
        with self.assertRaises(ToolError):
            self.fetch("/page.html", purpose="x", max_chars="many")
        self.assertEqual(self.questions, [], "no question for a bad argument")

    def test_empty_text_is_an_error(self):
        with self.assertRaises(ToolError) as ctx:
            self.fetch("/empty.html", purpose="x")
        self.assertIn("no readable text", str(ctx.exception))

    def test_http_error_is_reported_with_the_paste_advice(self):
        with self.assertRaises(ToolError) as ctx:
            self.fetch("/forbidden", purpose="x")
        self.assertIn("HTTP 403", str(ctx.exception))
        self.assertIn("paste", str(ctx.exception))
        with self.assertRaises(ToolError) as ctx:
            self.fetch("/missing", purpose="x")
        self.assertIn("HTTP 404", str(ctx.exception))


class FetchUrlPermissionTest(_Base):
    def test_asks_per_tool_and_host_with_purpose_in_the_detail(self):
        self.fetch("/page.html", purpose="the board outline")
        (question,) = self.questions
        self.assertEqual((question["tool"], question["host"]), ("fetch_url", "127.0.0.1"))
        self.assertEqual(question["detail"], "Reason: the board outline")
        self.assertIn(f"{self.base}/page.html", question["summary"])

    def test_declined_never_touches_the_network(self):
        self.verdicts["*"] = "deny"
        with self.assertRaises(ToolError) as ctx:
            self.fetch("/page.html", purpose="x")
        self.assertIn("DECLINED", str(ctx.exception))
        self.assertEqual(_Handler.hits, [])

    def test_unknown_host_gets_the_clearer_warning_vendor_host_does_not(self):
        self.fetch("/page.html", purpose="x")
        self.assertIn("not on the list of known vendor sites", self.questions[0]["summary"])
        params.set_string(tools_web.ALLOWLIST_SETTING, "127.0.0.1")        # the user extends the list
        self.fetch("/page.html", purpose="x")
        self.assertIn("known vendor site", self.questions[1]["summary"])
        self.assertNotIn("not on the list", self.questions[1]["summary"])

    def test_the_default_allowlist_is_exactly_the_documented_vendors_and_never_skips_the_question(self):
        self.assertEqual(set(tools_web.DEFAULT_ALLOWLIST), {
            "raspberrypi.com", "datasheets.raspberrypi.com", "pip.raspberrypi.com", "github.com", "raw.githubusercontent.com",
            "docs.arduino.cc", "docs.espressif.com", "grabcad.com"})
        self.assertTrue(tools_web.known_host("datasheets.raspberrypi.com"))
        self.assertFalse(tools_web.known_host("www.raspberrypi.com"))      # exact match only
        params.set_string(tools_web.ALLOWLIST_SETTING, "Example.ORG, https://docs.vendor.io/path  x.net")
        self.assertEqual(tools_web.user_allowlist(), ["example.org", "docs.vendor.io", "x.net"])
        self.assertTrue(tools_web.known_host("docs.vendor.io"))
        self.fetch("/page.html", purpose="x")
        self.assertEqual(len(self.questions), 1)                           # even a listed host is asked

    def test_redirect_to_another_host_asks_again(self):
        text = self.fetch("/other-host", purpose="x")
        self.assertIn("final_url: http://localhost:", text)
        self.assertEqual([q["host"] for q in self.questions], ["127.0.0.1", "localhost"])
        self.assertIn("redirected from '127.0.0.1'", self.questions[1]["summary"])

    def test_redirect_to_another_host_declined_stops_there(self):
        self.verdicts["localhost"] = "deny"
        with self.assertRaises(ToolError) as ctx:
            self.fetch("/other-host", purpose="x")
        self.assertIn("'localhost'", str(ctx.exception))
        self.assertEqual(_Handler.hits, ["/other-host"], "the second host was never contacted")

    def test_grant_for_one_host_does_not_cover_another_host_or_tool(self):
        netguard.add_grant("fetch_url", "127.0.0.1")
        self.assertTrue(netguard.has_grant("fetch_url", "127.0.0.1"))
        self.assertFalse(netguard.has_grant("web_search", "127.0.0.1"))
        self.assertFalse(netguard.has_grant("fetch_url", "localhost"))

    def test_audit_event_names_the_host_never_the_query(self):
        events = []
        with mock.patch("oscar.turnlog.log_event", lambda name, **f: events.append((name, f))):
            self.fetch("/page.html?secret=token123&k=v", purpose="x")
        uses = [f for name, f in events if name == "net_use"]
        self.assertEqual(len(uses), 1)
        self.assertEqual((uses[0]["tool"], uses[0]["host"], uses[0]["outcome"]), ("fetch_url", "127.0.0.1", "ok"))
        self.assertNotIn("token123", json.dumps(events, default=str))
        self.assertNotIn("/page.html", json.dumps(uses))


class FetchUrlSsrfTest(unittest.TestCase):
    """The REAL guard (no wrapper): internal targets are refused before any question is asked."""

    def setUp(self):
        params.clear_net_grants()
        netguard.reset_turn()
        patcher = mock.patch.object(netguard, "request_permission", return_value="allow")
        self.ask = patcher.start()
        self.addCleanup(patcher.stop)

    def test_internal_urls_are_refused_without_asking_or_connecting(self):
        for url in ("http://127.0.0.1:9/x", "http://localhost/x", "http://10.0.0.5/x", "http://192.168.1.1/",
                    "http://169.254.169.254/latest/meta-data/", "http://[::1]/", "http://printer.local/", "http://2130706433/",
                    "http://intranet/"):
            with mock.patch("requests.get") as get, self.assertRaises(ToolError) as ctx:
                tools_web._fetch_url(url=url, purpose="x")
            self.assertIn("refused", str(ctx.exception), url)
            get.assert_not_called()
        self.ask.assert_not_called()

    def test_a_hostname_that_resolves_to_a_private_address_is_refused(self):
        with mock.patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("10.1.2.3", 0))]), \
                mock.patch("requests.get") as get, self.assertRaises(ToolError) as ctx:
            tools_web._fetch_url(url="http://docs.example.com/a", purpose="x")
        self.assertIn("private-network", str(ctx.exception))
        get.assert_not_called()

    def test_redirect_into_the_internal_network_is_refused(self):
        class Resp:
            status_code = 302
            headers = {"Location": "http://169.254.169.254/latest/"}
            url = "https://docs.example.com/a"

            def close(self):
                pass

        with mock.patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 0))]), \
                mock.patch("requests.get", return_value=Resp()) as get, self.assertRaises(ToolError) as ctx:
            tools_web._fetch_url(url="https://docs.example.com/a", purpose="x")
        self.assertIn("redirect", str(ctx.exception))
        self.assertEqual(get.call_count, 1)

    def test_every_hop_connects_only_to_the_addresses_it_validated(self):
        """Review finding (PR #3): fetch_url validated each hop but connected with a plain requests.get, so a rebinding host could
        answer the check with a public address and the connect with an internal one."""
        answers = {"docs.example.com": "93.184.216.34", "second.example": "93.184.216.35"}
        seen = []

        def fake_getaddrinfo(host, *a, **k):
            return [(2, 1, 6, "", (answers.get(host, "10.0.0.5"), 0))]

        class Redirect:
            status_code = 302
            headers = {"Location": "https://second.example/b"}
            url = "https://docs.example.com/a"

            def close(self):
                pass

        class NotFound:
            status_code = 404
            headers = {}
            url = "https://second.example/b"

            def close(self):
                pass

        def fake_get(url, **kwargs):
            seen.append(dict(getattr(netguard._PIN_TLS, "pins", None) or {}))
            return Redirect() if "docs.example.com" in url else NotFound()

        with mock.patch("socket.getaddrinfo", side_effect=fake_getaddrinfo), mock.patch("requests.get", side_effect=fake_get), \
                self.assertRaises(ToolError):
            tools_web._fetch_url(url="https://docs.example.com/a", purpose="x")
        self.assertEqual(seen, [{"docs.example.com": ["93.184.216.34"]}, {"second.example": ["93.184.216.35"]}])
        self.assertIsNone(netguard._PIN_TLS.pins)                       # and nothing stays pinned afterwards

    def test_https_to_http_downgrade_is_refused(self):
        class Resp:
            status_code = 301
            headers = {"Location": "http://docs.example.com/a"}
            url = "https://docs.example.com/a"

            def close(self):
                pass

        with mock.patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 0))]), \
                mock.patch("requests.get", return_value=Resp()), self.assertRaises(ToolError) as ctx:
            tools_web._fetch_url(url="https://docs.example.com/a", purpose="x")
        self.assertIn("https to http", str(ctx.exception))


class RedirectLoopTest(_Base):
    def test_redirect_limit(self):
        with self.assertRaises(ToolError) as ctx:
            self.fetch("/loop", purpose="x")
        self.assertIn("redirects", str(ctx.exception))
        self.assertEqual(len(_Handler.hits), netguard.MAX_REDIRECTS + 1)


class ProvenanceTest(_Base):
    class FakeProject:
        def __init__(self, fail=False):
            self.calls, self.fail = [], fail

        def provenance_add(self, **fields):
            if self.fail:
                raise RuntimeError("disk full")
            self.calls.append(fields)
            return f"E{len(self.calls)}"

    def run_with(self, project, path="/page.html", purpose="board outline"):
        with use_context(ToolContext(project=project)):
            return self.fetch(path, purpose=purpose)

    def test_a_fetch_is_recorded_with_purpose_url_hash_and_time(self):
        project = self.FakeProject()
        text = self.run_with(project)
        (fields,) = project.calls
        from oscar import project as _project

        self.assertEqual(fields["kind"], "web_page" if "web_page" in _project.PROVENANCE_KINDS else "datasheet")
        self.assertEqual(fields["claim"], "board outline")
        self.assertEqual(fields["url"], f"{self.base}/page.html")
        self.assertEqual(fields["source"], "web:127.0.0.1")
        self.assertEqual(len(fields["sha256"]), 64)
        self.assertIn(f"sha256: {fields['sha256']}", text)
        self.assertRegex(fields["note"], r"fetched_at \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d")
        self.assertIn("provenance: E1", text)

    def test_a_project_failure_never_fails_the_fetch(self):
        text = self.run_with(self.FakeProject(fail=True))
        self.assertIn("provenance: not recorded", text)
        self.assertIn("Board is 85 mm", text)

    def test_no_project_says_so(self):
        self.assertIn("not recorded (no project bound)", self.run_with(None))

    def test_a_refused_fetch_records_nothing(self):
        project = self.FakeProject()
        with self.assertRaises(ToolError):
            self.run_with(project, "/file.step")
        self.assertEqual(project.calls, [])

    def test_a_real_project_records_the_entry_in_provenance_json(self):
        import tempfile

        from oscar.project import Project

        root = tempfile.mkdtemp(prefix="web_prov_")
        project = Project.for_chat("20261001-000000-abcdef", "Web test", projects_root=root, create=True)
        text = self.run_with(project)
        self.assertRegex(text, r"provenance: E\d+")
        with open(project.path("provenance.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        entries = data.get("entries", data) if isinstance(data, dict) else data
        self.assertIn(f"{self.base}/page.html", json.dumps(entries))


class WebSearchTest(_Base):
    def configure(self, path=""):
        params.set_string(tools_web.SEARCH_URL_SETTING, self.base + path)

    def test_not_configured_says_exactly_how_and_asks_nothing(self):
        with self.assertRaises(ToolError) as ctx:
            tools_web._web_search(query="raspberry pi 4 drawing")
        message = str(ctx.exception)
        self.assertIn("Search backend URL", message)
        self.assertIn(tools_web.SEARCH_URL_ENV, message)
        self.assertIn("SearXNG", message)
        self.assertIn("paste", message)
        self.assertEqual(self.questions, [])
        self.assertEqual(_Handler.hits, [])

    def test_env_overrides_the_setting(self):
        params.set_string(tools_web.SEARCH_URL_SETTING, "http://settings.example")
        with mock.patch.dict(os.environ, {tools_web.SEARCH_URL_ENV: self.base}):
            self.assertEqual(tools_web.search_backend(), self.base)
            self.assertIn("SEARCH RESULTS", tools_web._web_search(query="pi"))

    def test_results_are_compact_title_url_snippet(self):
        self.configure()
        text = tools_web._web_search(query="pi 4")
        head, note, body = text.split("\n", 2)
        self.assertEqual(head, "SEARCH RESULTS (2) for 'pi 4'")           # the javascript: URL was dropped
        self.assertEqual(note, context_guard.UNTRUSTED_NOTE.format(source="web search results"))   # data, not instructions
        results = json.loads(body)
        self.assertEqual(set(results[0]), {"title", "url", "snippet"})
        self.assertEqual(results[0]["snippet"], "Official drawing of the board")
        self.assertTrue(results[1]["snippet"].endswith("\u2026"))
        self.assertLessEqual(len(results[1]["snippet"]), 300)
        self.assertTrue(_Handler.hits[0].startswith("/search?"))
        self.assertIn("format=json", _Handler.hits[0])

    def test_max_results_is_clamped_and_applied(self):
        self.configure()
        self.assertTrue(tools_web._web_search(query="pi", max_results=1).startswith("SEARCH RESULTS (1)"))
        self.assertTrue(tools_web._web_search(query="pi", max_results=0).startswith("SEARCH RESULTS (1)"))
        self.assertTrue(tools_web._web_search(query="pi", max_results=99).startswith("SEARCH RESULTS (2)"))
        with self.assertRaises(ToolError):
            tools_web._web_search(query="pi", max_results="lots")

    def test_empty_query_and_bad_backend(self):
        self.configure()
        with self.assertRaises(ToolError):
            tools_web._web_search(query="  ")
        params.set_string(tools_web.SEARCH_URL_SETTING, "ftp://host")
        with self.assertRaises(ToolError) as ctx:
            tools_web._web_search(query="pi")
        self.assertIn("not an http(s) URL", str(ctx.exception))

    def test_permission_is_per_tool_and_host(self):
        self.configure()
        tools_web._web_search(query="pi")
        (question,) = self.questions
        self.assertEqual((question["tool"], question["host"]), ("web_search", "127.0.0.1"))
        self.verdicts["*"] = "deny"
        netguard.reset_turn()
        with self.assertRaises(ToolError) as ctx:
            tools_web._web_search(query="pi")
        self.assertIn("DECLINED", str(ctx.exception))

    def test_backend_that_is_not_json_or_not_searxng(self):
        self.configure("/nojson/search")
        with self.assertRaises(ToolError) as ctx:
            tools_web._web_search(query="pi")
        self.assertIn("application/json", str(ctx.exception))
        self.configure("/badshape/search")
        with self.assertRaises(ToolError) as ctx:
            tools_web._web_search(query="pi")
        self.assertIn("results", str(ctx.exception))

    def test_query_is_url_encoded(self):
        self.configure()
        tools_web._web_search(query="a&b=c d")
        self.assertIn("q=a%26b%3Dc%20d", _Handler.hits[0])

    def test_a_redirect_away_from_the_backend_is_checked_like_any_fetch(self):
        class Resp:
            status_code = 302
            headers = {"Location": "http://169.254.169.254/"}
            url = "http://search.lan/search"

            def close(self):
                pass

        params.set_string(tools_web.SEARCH_URL_SETTING, "http://search.lan:8080")
        with mock.patch("requests.get", return_value=Resp()), self.assertRaises(ToolError) as ctx:
            tools_web._web_search(query="pi")           # the user's own (private) backend is reachable, its redirect is not
        self.assertIn("redirect", str(ctx.exception))


class RegistrationTest(unittest.TestCase):
    def test_both_tools_are_registered_with_the_matrix_flags(self):
        by_name = {t.name: t for t in tools.get_tools()}
        for name in ("web_search", "fetch_url"):
            entry = by_name[name]
            self.assertEqual(entry.category, "net")
            self.assertFalse(entry.mutating)
            self.assertFalse(entry.geometry)
            self.assertEqual(tuple(entry.phases), ("INTAKE", "SOURCE", "PLAN"))
        self.assertIn("tools_web", tools._TOOL_MODULES)

    def test_schemas_match_the_contract(self):
        by_name = {t.name: t for t in tools.get_tools()}
        fetch = by_name["fetch_url"].parameters
        self.assertEqual(fetch["required"], ["url"])
        self.assertEqual(set(fetch["properties"]), {"url", "purpose", "max_chars"})
        self.assertEqual(fetch["properties"]["max_chars"]["maximum"], 20000)
        search = by_name["web_search"].parameters
        self.assertEqual(search["required"], ["query"])
        self.assertEqual(search["properties"]["max_results"], {"type": "integer", "default": 5, "minimum": 1, "maximum": 10})

    def test_run_tool_reports_the_not_configured_error(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(tools_web.SEARCH_URL_ENV, None)
            params.set_string(tools_web.SEARCH_URL_SETTING, "")
            text, _image, is_error = tools.run_tool("web_search", {"query": "pi"})
        self.assertTrue(is_error)
        self.assertIn("no search backend is configured", text)


class HtmlToTextTest(unittest.TestCase):
    def test_blocks_become_lines_and_nested_skips_balance(self):
        text = tools_web.html_to_text("<div>a<nav><div>x</div>NAVTEXT</nav>b</div><p>c</p><p>d")
        self.assertEqual(text.split("\n"), ["ab", "c", "d"])         # one line per block, nothing from the skipped nav
        self.assertNotIn("NAVTEXT", text)

    def test_an_unclosed_head_does_not_hide_the_body(self):
        self.assertIn("visible", tools_web.html_to_text("<html><head><title>T</title><body><p>visible</p>"))

    def test_malformed_and_empty_input(self):
        self.assertEqual(tools_web.html_to_text(""), "")
        self.assertIn("ok", tools_web.html_to_text("<p>ok</b></i><<>"))


if __name__ == "__main__":
    unittest.main()
