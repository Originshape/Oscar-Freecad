"""oscar.auth.session (and account.fetch_me / fetch_models): the session of one API URL against the fake OriginShape API.

Covered: 10 threads -> exactly one refresh; proactive renewal 60 s before expiry; the rotated refresh token is
persisted (a restart can still refresh, with no reuse detected); invalid_grant -> signed out, store cleared,
on_change told; a transient error keeps the session; API-key mode (checked with /v1/me first); sign_out revokes;
account() caching and its 401 -> refresh -> retry; on_change payloads carry no secrets and a failing hook is harmless.
Pure python (stdlib + requests); the keystore runs in memory.
"""

import json
import os
import sys
import threading
import time
import unittest
from unittest import mock

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.auth import account, discovery, keystore, session  # noqa: E402
from oscar.auth.keystore import KeystoreError, StoredCredential  # noqa: E402
from oscar.auth.transport import AuthError, AuthUnavailable, SignedOut, TokenSet  # noqa: E402
from support.fake_originshape import FakeOriginShape, envelope  # noqa: E402


class SessionCase(unittest.TestCase):
    def setUp(self):
        for reset in (discovery.clear_cache, session.reset_sessions, keystore.reset_memory, account.clear_model_cache):
            reset()
            self.addCleanup(reset)
        patcher = mock.patch.dict(os.environ, {keystore.ENV_STORE: "memory"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.fake = FakeOriginShape()
        self.addCleanup(self.fake.close)
        self.events: list[dict] = []
        self.raw_events: list[str] = []

        def hook(payload: str) -> None:
            self.raw_events.append(payload)
            self.events.append(json.loads(payload))

        hook_patch = mock.patch.object(session, "on_change", hook)
        hook_patch.start()
        self.addCleanup(hook_patch.stop)
        self.session = session.get_session(self.fake.url)

    def tokens(self, *, age: float = 0.0, refresh: bool = True) -> TokenSet:
        """Tokens of a fresh grant at the fake, received ``age`` seconds ago."""
        data = self.fake.issue_tokens(device_name="Test Box")
        return TokenSet(data["access_token"], data["refresh_token"] if refresh else "", data["expires_in"],
                        data["scope"], received_at=time.monotonic() - age)

    def sign_in(self, **kw) -> TokenSet:
        tokens = self.tokens(**kw)
        self.session.complete_sign_in(tokens)
        self.events.clear()
        self.raw_events.clear()
        return tokens

    def stored(self) -> StoredCredential | None:
        return keystore.load(self.session.api_url)

    def restart(self) -> session.AuthSession:
        """A new process: memory is gone, the keystore is not."""
        session.reset_sessions()
        discovery.clear_cache()
        self.session = session.get_session(self.fake.url)
        return self.session

    def event_names(self) -> list[str]:
        return [e["event"] for e in self.events]


class RegistryTest(SessionCase):
    def test_one_session_per_api_url(self):
        self.assertIs(session.get_session(self.fake.url + "/"), self.session)
        other = session.get_session("https://app.originshape.design")
        self.assertIsNot(other, self.session)
        # The retired address (never served) is the same session as the app's, however it reaches get_session.
        self.assertIs(session.get_session("https://API.originshape.design/"), other)
        with mock.patch.dict(os.environ, {"OSCAR_API_URL": self.fake.url}):
            self.assertIs(session.get_session(), self.session)
        self.assertNotIn("fake-", repr(self.session))


class SignedOutTest(SessionCase):
    def test_nothing_works_and_nothing_is_asked(self):
        self.assertFalse(self.session.has_session())
        self.assertEqual(self.session.kind, "")
        self.assertIsNone(self.session.cached_snapshot())
        for call in (self.session.access_token, self.session.auth_header, self.session.account):
            with self.assertRaises(SignedOut) as ctx:
                call()
            self.assertEqual(ctx.exception.code, "signed_out")
        self.assertFalse(self.session.invalidate())
        self.assertEqual(self.fake.calls, [])


class SignInTest(SessionCase):
    def test_complete_sign_in(self):
        tokens = self.tokens()
        self.session.complete_sign_in(tokens)
        self.assertTrue(self.session.has_session())
        self.assertEqual(self.session.kind, "oauth")
        self.assertEqual(self.session.access_token(), tokens.access_token)
        self.assertEqual(self.session.auth_header(), {"Authorization": f"Bearer {tokens.access_token}"})
        self.assertEqual(self.fake.calls, [], "a fresh access token needs no request")
        self.assertEqual(self.stored(), StoredCredential("oauth", tokens.refresh_token))
        self.assertEqual(self.event_names(), ["signed_in"])
        self.assertEqual(self.events[0]["kind"], "oauth")
        self.assertTrue(self.events[0]["signed_in"])
        for raw in self.raw_events:
            self.assertNotIn(tokens.access_token, raw)
            self.assertNotIn(tokens.refresh_token, raw)

    def test_a_stored_sign_in_is_used_after_a_restart(self):
        self.sign_in()
        self.restart()
        self.assertTrue(self.session.has_session())
        token = self.session.access_token()   # nothing in memory: one refresh
        self.assertTrue(token.startswith("fake-at-"))
        self.assertEqual(self.fake.counts["token:refresh_token"], 1)

    def test_a_sign_in_without_a_refresh_token_lasts_until_the_access_token_expires(self):
        tokens = self.tokens(refresh=False)
        self.session.complete_sign_in(tokens)
        self.assertIsNone(self.stored())
        self.assertEqual(self.session.access_token(), tokens.access_token)
        self.session.invalidate(tokens.access_token)
        with self.assertRaises(SignedOut):
            self.session.access_token()
        self.assertFalse(self.session.has_session())
        self.assertEqual(self.event_names()[-1], "signed_out")

    def test_a_keystore_failure_keeps_the_session_and_says_why(self):
        with mock.patch.object(keystore, "save", side_effect=KeystoreError("The macOS keychain did not save it.")):
            tokens = self.sign_in()
        self.assertTrue(self.session.has_session())
        self.assertEqual(self.session.access_token(), tokens.access_token)
        self.assertIn("keychain", self.session.last_store_error)
        self.assertNotIn(tokens.refresh_token, self.session.last_store_error)

    def test_a_malformed_token_is_refused(self):
        with self.assertRaises(AuthError):
            self.session.complete_sign_in(TokenSet("bad token with spaces", "fake-rt-abcdefgh"))
        self.assertFalse(self.session.has_session())


class RefreshTest(SessionCase):
    def test_ten_threads_get_exactly_one_refresh(self):
        old = self.sign_in(age=890)   # 10 s left: inside the renewal margin
        self.fake.delays["/oauth/token"] = 0.3   # keep the refresh in flight while the others arrive
        barrier = threading.Barrier(10)
        results: list = []
        lock = threading.Lock()

        def worker():
            barrier.wait()
            try:
                value = self.session.access_token()
            except Exception as e:  # noqa: BLE001
                value = e
            with lock:
                results.append(value)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual(len(results), 10)
        self.assertEqual(len(set(results)), 1, results)
        self.assertNotEqual(results[0], old.access_token)
        self.assertEqual(self.fake.counts["token:refresh_token"], 1)

    def test_ten_threads_on_an_ended_grant_send_one_refresh_and_sign_out_once(self):
        tokens = self.sign_in(age=900)
        self.fake.grant_of(tokens.refresh_token).revoked = True   # revoked from the website meanwhile
        self.fake.delays["/oauth/token"] = 0.3   # the nine others queue behind the one refresh in flight
        barrier = threading.Barrier(10)
        results: list = []
        lock = threading.Lock()

        def worker():
            barrier.wait()
            try:
                value = self.session.access_token()
            except Exception as e:  # noqa: BLE001
                value = e
            with lock:
                results.append(value)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual(len(results), 10)
        self.assertTrue(all(isinstance(r, SignedOut) for r in results), results)
        self.assertEqual(self.fake.counts["token:refresh_token"], 1, "the dead refresh token is sent once")
        self.assertEqual(self.event_names(), ["signed_out"])
        self.assertIsNone(self.stored())

    def test_a_new_sign_in_while_the_old_grant_fails_to_refresh_wins(self):
        old = self.sign_in(age=900)
        self.fake.grant_of(old.refresh_token).revoked = True
        self.fake.delays["/oauth/token"] = 0.3   # the doomed refresh is in flight while the user signs in again
        outcome: dict = {}

        def renew():
            try:
                outcome["token"] = self.session.access_token()
            except Exception as e:  # noqa: BLE001
                outcome["error"] = e

        thread = threading.Thread(target=renew, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5
        while not self.fake.calls_to("/oauth/token") and time.monotonic() < deadline:
            time.sleep(0.01)
        fresh = self.tokens()
        self.session.complete_sign_in(fresh)
        thread.join(5)
        self.assertEqual(outcome.get("token"), fresh.access_token, outcome)
        self.assertTrue(self.session.has_session())
        self.assertEqual(self.stored().secret, fresh.refresh_token)
        self.assertNotIn("signed_out", self.event_names())

    def test_renewal_happens_60_seconds_before_expiry(self):
        fresh = self.sign_in(age=900 - 70)   # 70 s left: still used
        self.assertEqual(self.session.access_token(), fresh.access_token)
        self.assertEqual(self.fake.counts["token:refresh_token"], 0)
        dying = self.sign_in(age=900 - 55)   # 55 s left: renewed first
        self.assertNotEqual(self.session.access_token(), dying.access_token)
        self.assertEqual(self.fake.counts["token:refresh_token"], 1)

    def test_a_tiny_lifetime_renews_at_half_life_not_on_every_call(self):
        data = self.fake.issue_tokens()
        self.session.complete_sign_in(TokenSet(data["access_token"], data["refresh_token"], 30))
        for _ in range(5):
            self.assertEqual(self.session.access_token(), data["access_token"])
        self.assertEqual(self.fake.counts["token:refresh_token"], 0)

    def test_the_rotated_refresh_token_is_persisted(self):
        old = self.sign_in(age=900)
        self.session.access_token()
        stored = self.stored()
        self.assertNotEqual(stored.secret, old.refresh_token)
        self.assertEqual(self.fake.refresh_state(old.refresh_token), "used")
        self.assertEqual(self.fake.refresh_state(stored.secret), "active")
        form = self.fake.calls_to("/oauth/token")[-1].form
        self.assertEqual(form, {"grant_type": "refresh_token", "client_id": "oscar-freecad",
                                "refresh_token": old.refresh_token})
        self.restart()   # the next process uses the rotated token: no reuse, the grant survives
        self.assertTrue(self.session.access_token())
        self.assertEqual(self.fake.refresh_state(stored.secret), "used")
        self.assertFalse(any(grant.revoked for grant in self.fake.grants.values()))

    def test_invalid_grant_signs_out_clears_the_store_and_tells_the_ui(self):
        tokens = self.sign_in(age=900)
        self.fake.grant_of(tokens.refresh_token).revoked = True   # e.g. revoked from the website
        with self.assertRaises(SignedOut) as ctx:
            self.session.access_token()
        self.assertIn("sign in again", str(ctx.exception))
        self.assertFalse(self.session.has_session())
        self.assertIsNone(self.stored())
        self.assertEqual(self.event_names(), ["signed_out"])
        self.assertEqual(self.events[0]["reason"], "expired")
        self.assertFalse(self.events[0]["signed_in"])
        with self.assertRaises(SignedOut):   # and it stays signed out without asking again
            self.session.access_token()
        self.assertEqual(self.fake.counts["token:refresh_token"], 1)

    def test_reuse_of_a_rotated_token_elsewhere_ends_the_grant_here_too(self):
        old = self.sign_in(age=900)
        self.session.access_token()   # rotates
        stolen = self.fake.calls_to("/oauth/token")   # an attacker replays the old token
        self.assertEqual(len(stolen), 1)
        requests.post(self.fake.url + "/oauth/token", timeout=5, data={
            "grant_type": "refresh_token", "client_id": "oscar-freecad", "refresh_token": old.refresh_token})
        self.session.invalidate()
        with self.assertRaises(SignedOut):
            self.session.access_token()
        self.assertIsNone(self.stored())

    def test_a_transient_failure_keeps_the_session(self):
        tokens = self.sign_in(age=900)
        self.fake.fail("/oauth/token", 503)
        with self.assertRaises(AuthUnavailable) as ctx:
            self.session.access_token()
        self.assertEqual(ctx.exception.status, 503)
        self.assertTrue(self.session.has_session())
        self.assertEqual(self.stored().secret, tokens.refresh_token)
        self.assertEqual(self.events, [])
        self.assertTrue(self.session.access_token(), "the next try works")

    def test_an_unreachable_server_keeps_the_session(self):
        tokens = self.sign_in(age=900)
        discovery.fetch(self.fake.url)
        self.fake.close()
        with self.assertRaises(AuthUnavailable):
            self.session.access_token()
        self.assertTrue(self.session.has_session())
        self.assertEqual(self.stored().secret, tokens.refresh_token)

    def test_another_refusal_is_an_error_but_not_a_sign_out(self):
        self.sign_in(age=900)
        self.fake.fail("/oauth/token", 401, {"error": "invalid_client"})
        with self.assertRaises(AuthError) as ctx:
            self.session.access_token()
        self.assertNotIsInstance(ctx.exception, SignedOut)
        self.assertEqual(ctx.exception.code, "invalid_client")
        self.assertTrue(self.session.has_session())

    def test_invalidate(self):
        tokens = self.sign_in()
        self.assertTrue(self.session.invalidate("some-other-token"))
        self.assertEqual(self.session.access_token(), tokens.access_token, "a token renewed meanwhile is kept")
        self.assertTrue(self.session.invalidate(tokens.access_token))
        renewed = self.session.access_token()
        self.assertNotEqual(renewed, tokens.access_token)
        self.assertEqual(self.fake.counts["token:refresh_token"], 1)

    def test_signing_out_while_a_refresh_is_in_flight_drops_its_result(self):
        self.sign_in(age=900)
        self.fake.reply_delays["/oauth/token"] = 0.4   # the server rotates at once, the answer comes late
        outcome: dict = {}

        def renew():
            try:
                outcome["token"] = self.session.access_token()
            except Exception as e:  # noqa: BLE001
                outcome["error"] = e

        thread = threading.Thread(target=renew, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5
        while not self.fake.counts["token:refresh_token"] and time.monotonic() < deadline:
            time.sleep(0.01)
        self.session.sign_out()
        thread.join(5)
        self.assertIsInstance(outcome.get("error"), SignedOut)
        self.assertFalse(self.session.has_session())
        self.assertIsNone(self.stored())
        self.assertEqual(self.fake.counts["revoke"], 2, "the old token at sign-out, the late new one after")
        self.assertTrue(all(grant.revoked for grant in self.fake.grants.values()))


class ApiKeyTest(SessionCase):
    def test_a_malformed_key_is_refused_without_a_request(self):
        for bad in ("", "sk-ant-not-ours-123456", "os_live_ has spaces", "short", None, "os_live_x\ny-zzzzzz"):
            with self.assertRaises(AuthError) as ctx:
                self.session.use_api_key(bad)
            self.assertEqual(ctx.exception.code, "invalid_key", bad)
            self.assertIsNone(ctx.exception.__context__, "nothing chained (it could quote the value)")
        self.assertEqual(self.fake.calls, [])

    def test_a_key_the_server_refuses_is_not_kept(self):
        key = "os_live_" + "x" * 43
        with self.assertRaises(AuthError) as ctx:
            self.session.use_api_key(key)
        self.assertEqual((ctx.exception.code, ctx.exception.status), ("invalid_key", 401))
        self.assertNotIn(key, str(ctx.exception))
        self.assertFalse(self.session.has_session())
        self.assertIsNone(self.stored())
        self.assertEqual(self.events, [])

    def test_an_accepted_key(self):
        key = self.fake.add_api_key()
        snapshot = self.session.use_api_key("  " + key + "\n")
        self.assertEqual(snapshot.email, "ada@example.com")
        self.assertEqual(self.session.kind, "api_key")
        self.assertEqual(self.session.access_token(), key)
        self.assertEqual(self.session.auth_header(), {"Authorization": f"Bearer {key}"})
        self.assertEqual(self.stored(), StoredCredential("api_key", key, "ada@example.com"))
        self.assertEqual(self.event_names(), ["signed_in"])
        self.assertEqual(self.events[0]["snapshot"]["email"], "ada@example.com")
        self.assertNotIn(key, self.raw_events[0])
        self.assertIs(self.session.account(), snapshot, "the check's snapshot is reused")
        self.assertFalse(self.session.invalidate(key), "a refresh cannot help an API key")
        self.assertEqual(self.session.access_token(), key)
        self.assertEqual(self.fake.calls_to("/v1/me")[0].headers["authorization"], f"Bearer {key}")
        self.assertEqual(self.fake.counts["token:refresh_token"], 0)

    def test_a_key_revoked_later_signs_out(self):
        key = self.fake.add_api_key()
        self.session.use_api_key(key)
        self.fake.api_keys.discard(key)
        with self.assertRaises(SignedOut):
            self.session.account(force=True)
        self.assertFalse(self.session.has_session())
        self.assertIsNone(self.stored())
        self.assertEqual(self.event_names()[-1], "signed_out")

    def test_a_stale_401_never_ends_a_newer_sign_in(self):
        """The old key's 401 arrives after the user signed in again: the new sign-in stays (memory and keystore)."""
        old = self.fake.add_api_key()
        self.session.use_api_key(old)
        self.fake.api_keys.discard(old)   # revoked on the website
        new = self.fake.add_api_key()
        real = session.request

        def request(method, url, **kw):
            reply = real(method, url, **kw)
            if (kw.get("headers") or {}).get("Authorization") == f"Bearer {old}":
                self.session.use_api_key(new)   # signed in again while the old key's answer was on its way
            return reply

        with mock.patch.object(session, "request", request), self.assertRaises(SignedOut):
            self.session.account(force=True)
        self.assertEqual(self.session.access_token(), new)
        self.assertEqual(self.stored().secret, new)
        self.assertNotIn("signed_out", self.event_names())

    def test_unavailable_while_checking(self):
        self.fake.fail("/v1/me", 503)
        with self.assertRaises(AuthUnavailable):
            self.session.use_api_key(self.fake.add_api_key())
        self.assertFalse(self.session.has_session())


class SignOutTest(SessionCase):
    def test_sign_out_revokes_then_forgets(self):
        tokens = self.sign_in()
        discovery.fetch(self.fake.url)
        self.session.sign_out()
        self.assertEqual(self.fake.counts["revoke"], 1)
        form = self.fake.calls_to("/oauth/revoke")[0].form
        self.assertEqual(form, {"token": tokens.refresh_token, "token_type_hint": "refresh_token",
                                "client_id": "oscar-freecad"})
        self.assertTrue(self.fake.grant_of(tokens.refresh_token).revoked)
        self.assertFalse(self.session.has_session())
        self.assertIsNone(self.stored())
        self.assertEqual(self.event_names(), ["signed_out"])
        self.assertEqual(self.events[0]["reason"], "signed_out")
        with self.assertRaises(SignedOut):
            self.session.access_token()

    def test_sign_out_without_cached_metadata_uses_the_standard_revoke_path(self):
        self.sign_in()
        self.session.sign_out()
        self.assertEqual(self.fake.counts["discovery"], 0, "no slow discovery on the way out")
        self.assertEqual(self.fake.counts["revoke"], 1)

    def test_sign_out_works_offline(self):
        self.sign_in()
        self.fake.close()
        start = time.monotonic()
        self.session.sign_out()
        self.assertLess(time.monotonic() - start, 5)
        self.assertFalse(self.session.has_session())
        self.assertIsNone(self.stored())

    def test_an_api_key_is_only_forgotten(self):
        self.session.use_api_key(self.fake.add_api_key())
        self.session.sign_out()
        self.assertEqual(self.fake.counts["revoke"], 0)
        self.assertFalse(self.session.has_session())
        self.assertIsNone(self.stored())


class AccountTest(SessionCase):
    def test_the_snapshot_is_fetched_and_cached(self):
        tokens = self.sign_in()
        snapshot = self.session.account()
        self.assertEqual((snapshot.email, snapshot.status, snapshot.credits_available), ("ada@example.com", "approved", 42.5))
        self.assertEqual(self.fake.calls_to("/v1/me")[0].headers["authorization"], f"Bearer {tokens.access_token}")
        self.assertIs(self.session.account(), snapshot)
        self.assertIs(self.session.cached_snapshot(), snapshot)
        self.assertEqual(self.fake.counts["me"], 1)
        self.assertIsNot(self.session.account(force=True), snapshot)
        self.assertEqual(self.fake.counts["me"], 2)
        with mock.patch.object(session, "SNAPSHOT_TTL_S", 0.0):
            self.session.account()
        self.assertEqual(self.fake.counts["me"], 3)
        self.assertEqual(self.event_names(), ["account"], "unchanged snapshots do not notify again")
        self.assertEqual(self.stored().email, "ada@example.com", "the email is kept with the sign-in")

    def test_changes_are_announced(self):
        self.sign_in()
        self.session.account()
        self.fake.set_account(balance=0.0, low=True)
        snapshot = self.session.account(force=True)
        self.assertTrue(snapshot.out_of_credits)
        self.assertEqual(self.event_names(), ["account", "account"])
        self.assertEqual(self.events[-1]["snapshot"]["credits_available"], 0.0)
        self.fake.set_account(status="waitlisted")
        self.assertTrue(self.session.account(force=True).waitlisted)

    def test_a_401_refreshes_once_and_retries(self):
        tokens = self.sign_in()
        self.fake.reject_token(tokens.access_token)   # the server no longer takes this access token
        snapshot = self.session.account()
        self.assertEqual(snapshot.email, "ada@example.com")
        self.assertEqual(self.fake.counts["token:refresh_token"], 1)
        self.assertEqual(self.fake.counts["me"], 2)
        auths = [c.headers["authorization"] for c in self.fake.calls_to("/v1/me")]
        self.assertEqual(auths[0], f"Bearer {tokens.access_token}")
        self.assertNotEqual(auths[1], auths[0])
        self.assertTrue(self.session.has_session())

    def test_a_second_401_signs_out(self):
        self.sign_in()
        self.fake.fail("/v1/me", 401, times=2)
        with self.assertRaises(SignedOut) as ctx:
            self.session.account()
        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(self.fake.counts["token:refresh_token"], 1)
        self.assertFalse(self.session.has_session())
        self.assertIsNone(self.stored())

    def test_server_trouble_keeps_the_session(self):
        self.sign_in()
        self.fake.fail("/v1/me", 503)
        with self.assertRaises(AuthUnavailable):
            self.session.account()
        self.fake.fail("/v1/me", 404, envelope("not_found", "gone"))
        with self.assertRaises(AuthError) as ctx:
            self.session.account()
        self.assertEqual(ctx.exception.code, "not_found", "the envelope's code reaches the caller")
        self.fake.fail("/v1/me", 501, envelope("llm_not_configured", "no key"))
        with self.assertRaises(AuthError) as ctx:
            self.session.account()
        self.assertNotIsInstance(ctx.exception, AuthUnavailable)
        self.assertEqual(str(ctx.exception), "The OriginShape server has no AI model configured.")
        self.assertEqual(self.fake.counts["fault"], 3, "a 501 is not retried")
        self.assertTrue(self.session.has_session())
        self.assertEqual(self.session.account().email, "ada@example.com")

    def test_fetch_me_and_fetch_models(self):
        self.sign_in()
        with mock.patch.dict(os.environ, {"OSCAR_API_URL": self.fake.url}):
            self.assertEqual(account.fetch_me().email, "ada@example.com")
            self.assertIsNone(account.cached_models())
            self.fake.models["data"][1]["display_name"] = "Oscar (quick)"
            catalog = account.fetch_models()
            self.assertEqual(catalog.default, "oscar-default")
            self.assertIs(account.cached_models(), catalog)
            self.assertEqual(account.cached_model("oscar-fast").display_name, "Oscar (quick)")
            self.assertEqual(account.cached_model("oscar-default").display_name, "Oscar (best)")
        self.assertEqual(account.cached_model("oscar-fast", self.fake.url).display_name, "Oscar (quick)")
        self.assertIsNone(account.cached_model("oscar-fast", "https://app.originshape.design"),
                          "another server's catalog is not used")
        self.fake.fail("/v1/models", 503)
        with self.assertRaises(AuthUnavailable):
            account.fetch_models(self.fake.url)


class HookTest(SessionCase):
    def test_a_failing_hook_never_breaks_signing_in_or_out(self):
        def broken(payload):
            raise RuntimeError("the window is gone")

        with mock.patch.object(session, "on_change", broken):
            self.sign_in()
            self.assertTrue(self.session.has_session())
            self.session.sign_out()
            self.assertFalse(self.session.has_session())

    def test_the_hook_may_call_back_into_the_session(self):
        seen: list = []

        def reentrant(payload):
            seen.append((json.loads(payload)["event"], self.session.has_session()))

        tokens = self.sign_in(age=900)
        self.fake.grant_of(tokens.refresh_token).revoked = True
        with mock.patch.object(session, "on_change", reentrant):
            with self.assertRaises(SignedOut):
                self.session.access_token()
        self.assertEqual(seen, [("signed_out", False)])


if __name__ == "__main__":
    unittest.main()
