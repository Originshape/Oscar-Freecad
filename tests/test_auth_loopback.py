"""oscar.auth.loopback: the browser sign-in (authorization code + PKCE + a 127.0.0.1 redirect) against the fake
OriginShape API. The test plays the browser: it GETs the authorize URL and follows the redirect to the listener.

Covered: success (the fake verifies PKCE for real), the authorize URL, state mismatch, access_denied, a failed code
exchange, timeout, cancel, exactly one callback, nothing but GET /callback served, the listener always closed, and
no request line ever printed. Pure python (stdlib + requests).
"""

import io
import os
import socket
import sys
import threading
import time
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.auth import discovery, loopback, pkce  # noqa: E402
from oscar.auth.transport import (AuthError, AuthUnavailable, SignInCancelled, SignInDeclined,  # noqa: E402
                                  SignInTimedOut, TokenSet)
from support.fake_originshape import FakeOriginShape, follow_authorize  # noqa: E402


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _closed_port() -> int:
    """A loopback port nothing listens on (bound, then released)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LoopbackCase(unittest.TestCase):
    def setUp(self):
        discovery.clear_cache()
        self.addCleanup(discovery.clear_cache)
        self.fake = FakeOriginShape()
        self.addCleanup(self.fake.close)
        self.opened: list[str] = []
        self.flow = loopback.LoopbackSignIn(self.fake.url, open_browser=self.opened.append, device_name="Test Box")
        self.addCleanup(self.flow.close)

    def browse_in_background(self, url: str | None = None) -> dict:
        """Play the browser on a thread (the callback answers only once wait() has decided)."""
        out: dict = {}
        target = url or self.flow.prepare()

        def run():
            try:
                out["response"] = follow_authorize(target)
            except Exception as e:  # noqa: BLE001 - the assertion happens in the test
                out["error"] = e

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        out["thread"] = thread
        return out

    def finish(self, browser: dict) -> requests.Response:
        browser["thread"].join(10)
        self.assertNotIn("error", browser, browser.get("error"))
        return browser["response"]

    def assertResult(self, response: requests.Response, result: str) -> None:
        """The callback sends the browser on to the app's result page, naming only the outcome."""
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["Location"], f"{self.flow.api_url}/device/done?result={result}")
        self.assertEqual(response.content, b"")


class SuccessTest(LoopbackCase):
    def test_a_full_sign_in(self):
        url = self.flow.prepare()
        browser = self.browse_in_background(url)
        tokens = self.flow.wait(threading.Event(), timeout=10)
        page = self.finish(browser)
        self.assertIsInstance(tokens, TokenSet)
        self.assertTrue(tokens.access_token.startswith("fake-at-"))
        self.assertTrue(tokens.refresh_token.startswith("fake-rt-"))
        self.assertEqual(tokens.expires_in, 900)
        self.assertEqual(self.fake.counts["token:authorization_code"], 1)
        self.assertEqual(self.fake.grant_of(tokens.refresh_token).device_name, "Test Box")
        # the tab ends on the app, not on 127.0.0.1: a 303 that carries the outcome, never the code or the state;
        # never cached, and no Referer (this address holds the code) goes with it
        self.assertResult(page, "signed-in")
        location = page.headers["Location"]
        self.assertNotIn(self.flow._state, location)
        self.assertNotIn("code", location)
        self.assertEqual(page.headers["Cache-Control"], "no-store")
        self.assertEqual(page.headers["Referrer-Policy"], "no-referrer")
        self.assertFalse(_port_open(self.flow.port), "the listener is closed after the sign-in")

    def test_the_authorize_url(self):
        url = self.flow.prepare()
        self.assertEqual(self.flow.prepare(), url, "prepare() is idempotent")
        parts = urlsplit(url)
        self.assertEqual(f"{parts.scheme}://{parts.netloc}{parts.path}", self.fake.url + "/oauth/authorize")
        query = {k: v[0] for k, v in parse_qs(parts.query).items()}
        self.assertEqual(query["response_type"], "code")
        self.assertEqual(query["client_id"], "oscar-freecad")
        self.assertEqual(query["redirect_uri"], f"http://127.0.0.1:{self.flow.port}/callback")
        self.assertEqual(query["scope"], "oscar:chat oscar:account offline_access")
        self.assertEqual(query["code_challenge_method"], "S256")
        self.assertEqual(query["code_challenge"], pkce.challenge(self.flow._verifier))
        self.assertEqual(query["state"], self.flow._state)
        self.assertEqual(query["device_name"], "Test Box")
        self.assertNotIn(self.flow._verifier, url, "the verifier never travels through the browser")
        self.assertEqual(self.fake.calls, [], "prepare() makes no request (safe on the GUI thread)")

    def test_the_result_page_is_on_the_app_the_server_names(self):
        app = self.fake.url.replace("127.0.0.1", "localhost")   # a local stack: the web app apart from the API
        self.fake.metadata_overrides = {"originshape": {**self.fake._metadata()["originshape"], "app_origin": app}}
        browser = self.browse_in_background()
        self.assertTrue(self.flow.wait(threading.Event(), timeout=10).access_token)
        page = self.finish(browser)
        self.assertEqual(page.status_code, 303)
        self.assertEqual(page.headers["Location"], f"{app}/device/done?result=signed-in")

    def test_the_discovered_authorize_endpoint_is_used_when_cached(self):
        web = self.fake.url.replace("127.0.0.1", "localhost") + "/oauth/authorize"   # another origin, same fake
        self.fake.metadata_overrides = {"authorization_endpoint": web}
        discovery.fetch(self.fake.url)
        url = self.flow.prepare()
        self.assertTrue(url.startswith(web + "?"), url)
        browser = self.browse_in_background(url)
        self.assertTrue(self.flow.wait(threading.Event(), timeout=10).access_token)
        self.assertResult(self.finish(browser), "signed-in")

    def test_open_uses_the_given_opener_and_reports_failure(self):
        self.assertTrue(self.flow.open())
        self.assertEqual(self.opened, [self.flow.authorize_url])

        def broken(url):
            raise RuntimeError("no browser")

        flow = loopback.LoopbackSignIn(self.fake.url, open_browser=broken)
        self.addCleanup(flow.close)
        self.assertFalse(flow.open())
        flow2 = loopback.LoopbackSignIn(self.fake.url, open_browser=lambda url: False)
        self.addCleanup(flow2.close)
        self.assertFalse(flow2.open())

    def test_the_default_device_name_is_a_short_hostname(self):
        with mock.patch.object(socket, "gethostname", return_value="Ada's-MacBook-Pro.local"):
            self.assertEqual(loopback.short_hostname(), "Ada's-MacBook-Pro")
        with mock.patch.object(socket, "gethostname", return_value="<script>.example"):
            self.assertEqual(loopback.short_hostname(), "script")
        with mock.patch.object(socket, "gethostname", side_effect=OSError):
            self.assertEqual(loopback.short_hostname(), "FreeCAD")
        self.assertEqual(loopback.clean_device_name("x" * 100), "x" * 64)

    def test_nothing_is_printed(self):
        stderr, stdout = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "stderr", stderr), mock.patch.object(sys, "stdout", stdout):
            browser = self.browse_in_background()
            self.flow.wait(threading.Event(), timeout=10)
            self.finish(browser)
        self.assertEqual(stderr.getvalue() + stdout.getvalue(), "", "a request line (with the code) was printed")


class FailureTest(LoopbackCase):
    def test_a_state_mismatch_is_refused_and_no_code_is_exchanged(self):
        self.fake.authorize_state = "forged-state"
        browser = self.browse_in_background()
        with self.assertRaises(AuthError) as ctx:
            self.flow.wait(threading.Event(), timeout=10)
        self.assertEqual(ctx.exception.code, "state_mismatch")
        self.assertResult(self.finish(browser), "mismatch")
        self.assertEqual(self.fake.counts["token:authorization_code"], 0)
        self.assertFalse(_port_open(self.flow.port))

    def test_a_missing_state_is_a_mismatch(self):
        self.flow.prepare()
        browser = self.browse_in_background(f"http://127.0.0.1:{self.flow.port}/callback?code=abcdefghijkl")
        with self.assertRaises(AuthError) as ctx:
            self.flow.wait(threading.Event(), timeout=10)
        self.assertEqual(ctx.exception.code, "state_mismatch")
        browser["thread"].join(10)

    def test_access_denied_is_a_clean_decline(self):
        self.fake.authorize_decision = "deny"
        browser = self.browse_in_background()
        with self.assertRaises(SignInDeclined) as ctx:
            self.flow.wait(threading.Event(), timeout=10)
        self.assertEqual(ctx.exception.code, "declined")
        self.assertIn("declined", str(ctx.exception))
        self.assertResult(self.finish(browser), "declined")
        self.assertEqual(self.fake.counts["token:authorization_code"], 0)

    def test_a_suspended_account_is_told_so_not_declined(self):
        """The consent page sends a suspended account back with access_denied / account_suspended (audit 2026-10-28:
        the page was a dead end and Oscar waited out its timeout)."""
        self.flow.prepare()
        url = (f"http://127.0.0.1:{self.flow.port}/callback?error=access_denied&error_description=account_suspended"
               f"&state={self.flow._state}")
        browser = self.browse_in_background(url)
        with self.assertRaises(AuthError) as ctx:
            self.flow.wait(threading.Event(), timeout=10)
        self.assertEqual(ctx.exception.code, "suspended")
        self.assertIn("suspended", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, SignInDeclined)
        browser["thread"].join(10)

    def test_another_authorize_error_is_reported_by_code(self):
        self.flow.prepare()
        url = (f"http://127.0.0.1:{self.flow.port}/callback?error=invalid_request&error_description=bad"
               f"&state={self.flow._state}")
        browser = self.browse_in_background(url)
        with self.assertRaises(AuthError) as ctx:
            self.flow.wait(threading.Event(), timeout=10)
        self.assertEqual(ctx.exception.code, "invalid_request")
        browser["thread"].join(10)

    def test_a_repeated_parameter_is_malformed(self):
        self.flow.prepare()
        state = self.flow._state
        url = f"http://127.0.0.1:{self.flow.port}/callback?code=abcdefghijkl&state={state}&state={state}"
        browser = self.browse_in_background(url)
        with self.assertRaises(AuthError) as ctx:
            self.flow.wait(threading.Event(), timeout=10)
        self.assertEqual(ctx.exception.code, "bad_response")
        browser["thread"].join(10)

    def test_a_refused_code_exchange_fails_the_sign_in_and_the_page_says_so(self):
        self.fake.fail("/oauth/token", 400, {"error": "invalid_grant", "error_description": "code expired"})
        browser = self.browse_in_background()
        with self.assertRaises(AuthError) as ctx:
            self.flow.wait(threading.Event(), timeout=10)
        self.assertEqual(ctx.exception.code, "invalid_grant")
        # the reason (an expired code) stays in FreeCAD; the browser learns only that it failed
        self.assertResult(self.finish(browser), "failed")

    def test_timeout(self):
        start = time.monotonic()
        with self.assertRaises(SignInTimedOut):
            self.flow.wait(threading.Event(), timeout=0.3)
        self.assertLess(time.monotonic() - start, 2.0)
        self.assertFalse(_port_open(self.flow.port), "the listener is closed after a timeout")

    def test_an_unreachable_server_ends_the_wait_at_once(self):
        flow = loopback.LoopbackSignIn(f"http://127.0.0.1:{_closed_port()}", open_browser=lambda url: True)
        self.addCleanup(flow.close)
        flow.prepare()
        port = flow.port
        start = time.monotonic()
        with self.assertRaises(AuthUnavailable):
            flow.wait(threading.Event(), timeout=30)
        self.assertLess(time.monotonic() - start, 5.0)
        self.assertFalse(_port_open(port), "the listener is closed when the server cannot be reached")

    def test_cancel(self):
        cancelled = threading.Event()
        self.flow.prepare()
        port = self.flow.port
        self.assertTrue(_port_open(port))
        threading.Timer(0.2, cancelled.set).start()
        start = time.monotonic()
        with self.assertRaises(SignInCancelled):
            self.flow.wait(cancelled, timeout=30)
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertFalse(_port_open(port), "the listener is closed after a cancel")

    def test_cancel_during_the_code_exchange_revokes_the_new_grant(self):
        cancelled = threading.Event()
        self.fake.delays["/oauth/token"] = 0.4
        browser = self.browse_in_background()

        def cancel_once_exchanging():
            deadline = time.monotonic() + 5
            while not self.fake.calls_to("/oauth/token") and time.monotonic() < deadline:
                time.sleep(0.01)
            cancelled.set()

        threading.Thread(target=cancel_once_exchanging, daemon=True).start()
        with self.assertRaises(SignInCancelled):
            self.flow.wait(cancelled, timeout=10)
        browser["thread"].join(10)
        self.assertEqual(self.fake.counts["revoke"], 1)
        self.assertTrue(all(grant.revoked for grant in self.fake.grants.values()))

    def test_a_finished_flow_cannot_be_reused(self):
        with self.assertRaises(SignInTimedOut):
            self.flow.wait(None, timeout=0.05)
        with self.assertRaises(AuthError) as ctx:
            self.flow.prepare()
        self.assertEqual(ctx.exception.code, "finished")
        with self.assertRaises(AuthError):
            self.flow.wait(threading.Event(), timeout=1)

    def test_close_without_waiting_releases_the_port(self):
        self.flow.prepare()
        port = self.flow.port
        self.assertTrue(_port_open(port))
        self.flow.close()
        self.assertFalse(_port_open(port))
        self.flow.close()   # idempotent

    def test_a_listener_nobody_waits_for_closes_itself(self):
        with mock.patch.object(loopback, "LISTENER_MAX_S", 0.2):
            flow = loopback.LoopbackSignIn(self.fake.url, open_browser=lambda url: True)
            flow.prepare()
            port = flow.port
            deadline = time.monotonic() + 3
            while _port_open(port) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(_port_open(port))
            with self.assertRaises(AuthError) as ctx:
                flow.wait(threading.Event(), timeout=5)
            self.assertEqual(ctx.exception.code, "finished")

    def test_a_wait_in_progress_ends_when_the_listener_closes_itself(self):
        with mock.patch.object(loopback, "LISTENER_MAX_S", 0.3):
            flow = loopback.LoopbackSignIn(self.fake.url, open_browser=lambda url: True)
            flow.prepare()
            start = time.monotonic()
            with self.assertRaises(SignInCancelled):
                flow.wait(threading.Event(), timeout=30)
            self.assertLess(time.monotonic() - start, 3.0)


class ListenerSurfaceTest(LoopbackCase):
    def test_only_get_callback_is_served(self):
        self.flow.prepare()
        base = f"http://127.0.0.1:{self.flow.port}"
        for path in ("/", "/favicon.ico", "/callback/x", "/callbackx", "/oauth/token", "/../etc/passwd"):
            self.assertEqual(requests.get(base + path, timeout=5).status_code, 404, path)
        for method in ("POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"):
            self.assertEqual(requests.request(method, base + "/callback", timeout=5).status_code, 404, method)
        # another Host (a DNS-rebinding page) is not answered either
        response = requests.get(base + "/callback?code=x&state=y", headers={"Host": "evil.example.com"}, timeout=5)
        self.assertEqual(response.status_code, 404)
        # none of these used up the one callback: the real sign-in still works
        browser = self.browse_in_background()
        self.assertTrue(self.flow.wait(threading.Event(), timeout=10).access_token)
        self.assertResult(self.finish(browser), "signed-in")

    def test_exactly_one_callback_is_accepted(self):
        url = self.flow.prepare()
        first = self.browse_in_background(url)
        deadline = time.monotonic() + 5
        while self.flow._accepting and time.monotonic() < deadline:   # the first answer is in, wait() has not run
            time.sleep(0.01)
        second = requests.get(f"http://127.0.0.1:{self.flow.port}/callback?code=other-code-1&state={self.flow._state}",
                              allow_redirects=False, timeout=5)
        self.assertResult(second, "already-used")
        tokens = self.flow.wait(threading.Event(), timeout=10)
        self.assertResult(self.finish(first), "signed-in")
        self.assertTrue(tokens.access_token)
        self.assertEqual(self.fake.counts["token:authorization_code"], 1)

    def test_the_listener_binds_loopback_only(self):
        self.flow.prepare()
        thread_names = [t.name for t in threading.enumerate() if t.name == "oscar-signin-listener"]
        self.assertTrue(thread_names)
        self.assertTrue(all(t.daemon for t in threading.enumerate() if t.name == "oscar-signin-listener"))
        self.assertTrue(self.flow.redirect_uri.startswith("http://127.0.0.1:"))


if __name__ == "__main__":
    unittest.main()
