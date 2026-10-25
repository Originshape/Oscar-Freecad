"""No secret in any exception text, repr or log line, anywhere in oscar.auth.

The fake server is made hostile: its error answers ECHO what it was sent (the code, the PKCE verifier, the refresh
token, the device code, the API key), it returns malformed tokens, or it vanishes mid-request. Every exception is
collected with its whole chain (``str``, ``repr``, ``args``, ``__cause__``, ``__context__``) and, at the end, every
secret the fake ever issued or received is searched for in all of it and in the agent log. Also: turnlog masks the
new sign-in field names. Pure python (stdlib + requests).
"""

import json
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

from oscar import turnlog  # noqa: E402
from oscar.auth import account, device, discovery, keystore, loopback, session, transport  # noqa: E402
from oscar.auth.transport import AuthError, TokenSet  # noqa: E402
from support.fake_originshape import FakeOriginShape, follow_authorize  # noqa: E402


def _chain_text(error: BaseException) -> str:
    """Everything an exception can show: its text, repr and args, through ``__cause__`` and ``__context__``."""
    parts: list[str] = []
    seen: set[int] = set()
    stack = [error]
    while stack:
        item = stack.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        parts += [str(item), repr(item), repr(getattr(item, "args", ())), str(getattr(item, "code", ""))]
        stack += [item.__cause__, item.__context__]
    return "\n".join(parts)


class RedactionCase(unittest.TestCase):
    def setUp(self):
        for reset in (discovery.clear_cache, session.reset_sessions, keystore.reset_memory, account.clear_model_cache):
            reset()
            self.addCleanup(reset)
        self.log_dir = isolation.fresh_dir("auth_log_")
        patcher = mock.patch.dict(os.environ, {keystore.ENV_STORE: "memory", "OSCAR_LOG_DIR": self.log_dir})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.fake = FakeOriginShape()
        self.addCleanup(self.fake.close)
        self.texts: list[str] = []          # every exception chain and repr produced below
        self.sent: list[str] = []           # secrets the plugin sent (echoed back by the hostile routes)

    def expect_error(self, call) -> BaseException:
        try:
            call()
        except AuthError as e:
            self.texts.append(_chain_text(e))
            return e
        self.fail("an AuthError was expected")

    def echo(self, route: str, status: int = 400, error: str = "server_echo") -> None:
        """Make ``route`` answer an OAuth error whose description repeats everything it received."""
        def answer(call):
            received = dict(call.form or {})
            if call.headers.get("authorization"):
                received["authorization"] = call.headers["authorization"]
            self.sent += [v for v in received.values() if isinstance(v, str)]
            return self.fake._json(status, {"error": error, "error_description": json.dumps(received),
                                            "type": "error", "message": json.dumps(received)})

        patcher = mock.patch.object(self.fake, route, answer)
        patcher.start()
        self.addCleanup(patcher.stop)

    def all_secrets(self) -> list[str]:
        fake = self.fake
        with fake.lock:
            found = list(fake._refresh) + list(fake._access) + list(fake._codes) + list(fake.api_keys)
            found += [d.device_code for d in fake.devices.values()]
        found += [s for s in self.sent if len(s) >= 8 and s not in ("oscar-freecad",) and "://" not in s
                  and not s.startswith(("authorization_code", "refresh_token", "urn:"))]
        return [s for s in found if s]

    def assert_no_secret_anywhere(self) -> None:
        secrets = self.all_secrets()
        self.assertTrue(secrets, "the scenario produced no secrets to look for")
        log = json.dumps(turnlog.tail(1000))
        for secret in secrets:
            bare = secret.split(" ", 1)[-1]   # "Bearer <token>" -> the token
            for text in self.texts:
                self.assertNotIn(bare, text)
            self.assertNotIn(bare, log)


class FlowRedactionTest(RedactionCase):
    def test_browser_sign_in_with_a_server_that_echoes_the_code_and_the_verifier(self):
        self.echo("route_token")
        flow = loopback.LoopbackSignIn(self.fake.url, open_browser=lambda url: True)
        self.addCleanup(flow.close)
        url = flow.prepare()
        thread = threading.Thread(target=lambda: follow_authorize(url), daemon=True)
        thread.start()
        error = self.expect_error(lambda: flow.wait(threading.Event(), timeout=10))
        thread.join(10)
        self.assertEqual(error.code, "server_echo")
        self.sent.append(flow._verifier)
        self.texts.append(repr(flow))
        self.assertIn(flow._verifier, json.dumps(self.sent), "the hostile server did receive the verifier")
        self.assert_no_secret_anywhere()

    def test_device_sign_in_with_a_server_that_echoes_the_device_code(self):
        flow = device.DeviceSignIn(self.fake.url, sleep=lambda cancelled, seconds: cancelled.is_set())
        code = flow.start()
        self.texts += [repr(flow), repr(code), str(code)]
        self.echo("route_token")
        self.expect_error(lambda: flow.poll(threading.Event()))
        self.assert_no_secret_anywhere()

    def test_refresh_errors_that_echo_the_refresh_token(self):
        auth = session.get_session(self.fake.url)
        data = self.fake.issue_tokens()
        auth.complete_sign_in(TokenSet(data["access_token"], data["refresh_token"], 900,
                                       received_at=time.monotonic() - 900))
        self.texts.append(repr(auth))
        self.echo("route_token", error="invalid_request")
        self.expect_error(auth.access_token)
        self.echo("route_token", status=400, error="invalid_grant")   # -> signed out
        self.expect_error(auth.access_token)
        self.assertFalse(auth.has_session())
        self.assert_no_secret_anywhere()

    def test_an_api_key_the_server_echoes(self):
        key = "os_live_" + "Q" * 20 + "-secret-key-material"
        self.sent.append(key)
        self.echo("route_me", status=401)
        auth = session.get_session(self.fake.url)
        self.expect_error(lambda: auth.use_api_key(key))
        self.echo("route_me", status=500)
        self.expect_error(lambda: auth.use_api_key(key))
        self.echo("route_me", status=418)
        self.expect_error(lambda: auth.use_api_key(key))
        self.assert_no_secret_anywhere()

    def test_malformed_tokens_from_the_server_are_never_quoted(self):
        bad_access = "fake-at-SECRET-WITH-SPACE inside"
        self.sent.append("fake-at-SECRET-WITH-SPACE")

        def answer(call):
            return self.fake._json(200, {"access_token": bad_access, "token_type": "Bearer", "expires_in": 900,
                                         "refresh_token": "fake-rt-SECRET\nNEWLINE"})

        self.sent.append("fake-rt-SECRET")
        auth = session.get_session(self.fake.url)
        data = self.fake.issue_tokens()
        auth.complete_sign_in(TokenSet(data["access_token"], data["refresh_token"], 900,
                                       received_at=time.monotonic() - 900))
        with mock.patch.object(self.fake, "route_token", answer):
            self.expect_error(auth.access_token)
        self.assert_no_secret_anywhere()

    def test_the_network_vanishing_mid_request(self):
        auth = session.get_session(self.fake.url)
        data = self.fake.issue_tokens()
        auth.complete_sign_in(TokenSet(data["access_token"], data["refresh_token"], 900,
                                       received_at=time.monotonic() - 900))
        self.fake.fail("/oauth/token", 0)
        self.expect_error(auth.access_token)
        discovery.fetch(self.fake.url)
        self.fake.close()
        self.expect_error(auth.access_token)
        self.expect_error(lambda: auth.account(force=True))
        self.assert_no_secret_anywhere()


class HappyPathRedactionTest(RedactionCase):
    def test_a_whole_successful_session_leaves_no_secret_in_the_log_or_a_repr(self):
        flow = loopback.LoopbackSignIn(self.fake.url, open_browser=lambda url: True)
        url = flow.prepare()
        thread = threading.Thread(target=lambda: follow_authorize(url), daemon=True)
        thread.start()
        tokens = flow.wait(threading.Event(), timeout=10)
        thread.join(10)
        self.sent.append(flow._verifier)
        auth = session.get_session(self.fake.url)
        auth.complete_sign_in(tokens)
        auth.invalidate()
        auth.access_token()                     # a successful refresh (rotation)
        auth.account(force=True)
        account.fetch_models(self.fake.url)
        code_flow = device.DeviceSignIn(self.fake.url, sleep=lambda cancelled, seconds: cancelled.is_set())
        self.fake.device_auto, self.fake.enforce_interval = "approve", False
        code_flow.start()
        auth.complete_sign_in(code_flow.poll(threading.Event()))
        auth.use_api_key(self.fake.add_api_key())
        auth.sign_out()
        self.texts += [repr(flow), repr(code_flow), repr(auth), repr(tokens), repr(auth.cached_snapshot())]
        events = turnlog.tail(1000)
        self.assertTrue(any(e.get("kind") == "auth" for e in events), "the flows did log (safely)")
        self.assert_no_secret_anywhere()


class TransportRedactionTest(RedactionCase):
    def test_an_invalid_header_value_is_never_quoted(self):
        secret = "fake-at-HEADER-SECRET-123456"
        for value in (f"Bearer {secret}\nX-Injected: 1", f"Bearer {secret}\r", f" Bearer {secret}"):
            error = self.expect_error(lambda: transport.request("GET", self.fake.url + "/v1/me",
                                                                headers={"Authorization": value}))
            if isinstance(error, transport.AuthUnavailable):
                self.assertIsNone(error.__context__, "the requests exception is not even chained")
        for text in self.texts:
            self.assertNotIn(secret, text)

    def test_a_url_with_credentials_is_never_quoted(self):
        error = self.expect_error(lambda: session.get_session("https://user:pa55-SECRET@example.com"))
        self.assertNotIn("pa55-SECRET", self.texts[-1])
        self.assertEqual(error.code, "invalid_url")

    def test_reprs_hide_every_secret(self):
        tokens = TokenSet("fake-at-REPR-SECRET-1", "fake-rt-REPR-SECRET-2")
        cred = keystore.StoredCredential("oauth", "fake-rt-REPR-SECRET-3", "a@b.c")
        reply = transport.Reply(200, {}, b'{"access_token": "fake-at-REPR-SECRET-4"}')
        text = " ".join([repr(tokens), str(tokens), repr(cred), str(cred), repr(reply)])
        for n in range(1, 5):
            self.assertNotIn(f"REPR-SECRET-{n}", text)

    def test_keystore_errors_never_carry_the_blob(self):
        import subprocess

        cred = keystore.StoredCredential("oauth", "fake-rt-KEYCHAIN-SECRET-5")

        def failing(argv, **kw):
            return subprocess.CompletedProcess(argv, 51, "", f"security: failed for {kw.get('input')}")

        backend = keystore.MacKeychainBackend(run=failing)
        error = self.expect_error(lambda: backend.write("originshape:0123456789ab", cred))
        self.assertNotIn("KEYCHAIN-SECRET-5", self.texts[-1])
        self.assertNotIn(keystore.encode_blob(cred), self.texts[-1])
        self.assertIn("51", str(error))


class TurnlogKeysTest(RedactionCase):
    def test_the_sign_in_field_names_are_masked(self):
        values = {"code_verifier": "verifier-SECRET-1", "device_code": "device-SECRET-2",
                  "authorization_code": "code-SECRET-3", "refresh": "refresh-SECRET-4", "user_code": "BCDF-GHJK",
                  "refresh_token": "refresh-SECRET-5", "access_token": "access-SECRET-6"}
        turnlog.log_event("auth_probe", **values, nested={"device_code": "device-SECRET-7"}, outcome="ok")
        entry = turnlog.tail(1)[-1]
        for key in values:
            self.assertEqual(entry[key], "[redacted]", key)
        self.assertEqual(entry["nested"], {"device_code": "[redacted]"})
        self.assertEqual(entry["outcome"], "ok")
        text = json.dumps(entry)
        for n in range(1, 8):
            self.assertNotIn(f"SECRET-{n}", text)
        self.assertNotIn("BCDF-GHJK", text)


if __name__ == "__main__":
    unittest.main()
