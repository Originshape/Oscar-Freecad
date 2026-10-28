"""oscar.auth basics: PKCE, the API URL and its rules (config), discovery validation and caching, and the parsing of
``/v1/me`` (AccountSnapshot) and ``/v1/models`` (ModelCatalog, cached_model).

Pure python (stdlib + requests), against a real HTTP fake (support.fake_originshape): runs on the CI pure job and
under freecadcmd alike.
"""

import json
import os
import re
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import oscar  # noqa: E402
from oscar import models  # noqa: E402
from oscar.auth import account, config, discovery, pkce, transport  # noqa: E402
from oscar.auth.transport import (AuthError, AuthUnavailable, InvalidServerURL, ServerMetadataError,  # noqa: E402
                                  TokenSet)
from support.fake_originshape import FakeOriginShape  # noqa: E402

ME_EXAMPLE = {
    "user": {"id": "u1", "email": "a@b.com", "name": "A", "avatar_url": None, "status": "approved", "is_admin": False,
             "created_at": "2026-10-01T00:00:00Z"},
    "org": {"id": "o1", "plan": "trial", "plan_name": "Trial", "subscription_status": None, "current_period_end": None},
    "credits": {"balance": 42.5, "available": 42.5, "low": False},
    "links": {"app": "http://localhost:3210", "billing": "http://localhost:3210/billing",
              "manage": "http://localhost:3210/settings", "waitlist": "http://localhost:3210/waitlist",
              "download": "http://localhost:3210/download"},
    "plugin": {"latest_version": "0.1.0", "min_version": "0.1.0"},
}


def _stub_params(server: str = "", settings: str = "", custom: bool = True) -> types.SimpleNamespace:
    """A stand-in for ``oscar.params`` (the real one imports FreeCAD); ``custom``: the *Use a custom server* switch."""
    return types.SimpleNamespace(get_string=lambda name, default="": server if name == "OriginShapeServer" else default,
                                 get_bool=lambda name, default=False: custom if name == "OriginShapeCustomServer"
                                 else default,
                                 settings_path=lambda: settings)


class PkceTest(unittest.TestCase):
    def test_the_rfc_7636_example_challenge(self):
        self.assertEqual(pkce.challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
                         "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM")

    def test_verifiers_are_long_random_and_unreserved(self):
        seen = {pkce.new_verifier() for _ in range(50)}
        self.assertEqual(len(seen), 50)
        for verifier in seen:
            self.assertEqual(len(verifier), pkce.VERIFIER_LENGTH)
            self.assertTrue(pkce.is_valid_verifier(verifier))
        self.assertEqual(len(pkce.new_verifier(43)), 43)
        self.assertEqual(len(pkce.new_verifier(128)), 128)
        for bad in (42, 129):
            with self.assertRaises(ValueError):
                pkce.new_verifier(bad)

    def test_a_bad_verifier_is_refused_without_being_quoted(self):
        bad = "short-secret-verifier"
        with self.assertRaises(ValueError) as ctx:
            pkce.challenge(bad)
        self.assertNotIn(bad, str(ctx.exception))
        self.assertFalse(pkce.is_valid_verifier("x" * 42))
        self.assertFalse(pkce.is_valid_verifier("x" * 50 + " "))
        self.assertFalse(pkce.is_valid_verifier(None))

    def test_state_comparison(self):
        state = pkce.new_state()
        self.assertGreaterEqual(len(state), 43)
        self.assertNotEqual(state, pkce.new_state())
        self.assertTrue(pkce.states_match(state, state))
        self.assertFalse(pkce.states_match(state, state[:-1]))
        self.assertFalse(pkce.states_match(state, None))
        self.assertFalse(pkce.states_match(state, "\u00e9t\u00e9"))   # non-ASCII from a forged request never raises
        self.assertFalse(pkce.states_match("", ""))


class ConfigTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(config.ENV_API_URL, None)
        params = mock.patch.object(config, "_params", lambda: None)
        params.start()
        self.addCleanup(params.stop)

    def test_the_default_is_the_app_which_serves_the_api_too(self):
        self.assertEqual(config.api_url(), "https://app.originshape.design")

    def test_the_retired_api_address_reads_as_the_app(self):
        # api.originshape.design was never served; a Server field copied from early documentation still works.
        for stored in ("https://api.originshape.design", "HTTPS://API.originshape.design:443/"):
            with mock.patch.object(config, "_params", lambda stored=stored: _stub_params(stored)):
                self.assertEqual(config.api_url(), "https://app.originshape.design", stored)
        os.environ[config.ENV_API_URL] = "https://api.originshape.design"
        self.assertEqual(config.api_url(), "https://app.originshape.design")
        # Every cache keyed by a server URL (sessions, model lists, sign-in metadata) keys it under the served one.
        self.assertEqual(config.normalise_url("https://api.originshape.design/"), "https://app.originshape.design")

    def test_the_environment_wins_and_is_read_at_call_time(self):
        os.environ[config.ENV_API_URL] = "http://127.0.0.1:8210/"
        self.assertEqual(config.api_url(), "http://127.0.0.1:8210")
        os.environ[config.ENV_API_URL] = "https://staging.example.com"
        self.assertEqual(config.api_url(), "https://staging.example.com")
        os.environ[config.ENV_API_URL] = "   "
        self.assertEqual(config.api_url(), "https://app.originshape.design")

    def test_the_server_preference_is_used_when_the_environment_is_unset(self):
        with mock.patch.object(config, "_params", lambda: _stub_params("http://localhost:8210/")):
            self.assertEqual(config.api_url(), "http://localhost:8210")
            os.environ[config.ENV_API_URL] = "https://env.example.com"
            self.assertEqual(config.api_url(), "https://env.example.com")

    def test_the_server_preference_applies_only_while_the_custom_server_switch_is_on(self):
        # A local server typed in once for testing is kept, but sign-in goes to the app until the switch is on.
        with mock.patch.object(config, "_params", lambda: _stub_params("http://localhost:8210/", custom=False)):
            self.assertEqual(config.configured_url(), "https://app.originshape.design")
            self.assertEqual(config.api_url(), "https://app.originshape.design")
            os.environ[config.ENV_API_URL] = "https://env.example.com"
            self.assertEqual(config.api_url(), "https://env.example.com")   # the environment still wins

    def test_an_unreadable_preference_falls_back_to_the_default(self):
        def broken(name, default=""):
            raise OSError("locked")

        for stub in (types.SimpleNamespace(get_string=broken, get_bool=lambda name, default=False: True),
                     types.SimpleNamespace(get_string=lambda name, default="": "http://localhost:8210",
                                           get_bool=broken)):
            with mock.patch.object(config, "_params", lambda stub=stub: stub):
                self.assertEqual(config.api_url(), "https://app.originshape.design")

    def test_normalisation(self):
        cases = {
            "https://APP.OriginShape.design/": "https://app.originshape.design",
            "HTTPS://app.originshape.design:443": "https://app.originshape.design",
            "https://example.com/originshape/api//": "https://example.com/originshape/api",
            "http://localhost:80/": "http://localhost",
            "http://[::1]:8210/": "http://[::1]:8210",
            " https://example.com:8443 ": "https://example.com:8443",
        }
        for raw, expected in cases.items():
            self.assertEqual(config.normalise_url(raw), expected, raw)

    def test_plain_http_is_allowed_for_loopback_hosts_only(self):
        for url in ("http://localhost:8210", "http://127.0.0.1:9", "http://[::1]:1234", "http://127.0.0.5"):
            self.assertTrue(config.normalise_url(url).startswith("http://"), url)
        for url in ("http://example.com", "http://10.0.0.5:8210", "http://app.originshape.design",
                    "http://localhost.example.com"):
            with self.assertRaises(InvalidServerURL) as ctx:
                config.normalise_url(url)
            self.assertEqual(ctx.exception.code, "insecure_url")
            self.assertIn("https", str(ctx.exception))

    def test_malformed_urls_are_refused_without_echoing_credentials(self):
        for url in ("", "   ", "ftp://example.com", "example.com", "https://", "https://exa mple.com",
                    "https://example.com:99999", "http://[::1", "https://example.com/?x=1", "https://example.com/#f",
                    "https://user:hunter2-secret@example.com", None, 42):
            with self.assertRaises(InvalidServerURL, msg=repr(url)) as ctx:
                config.normalise_url(url)
            self.assertNotIn("hunter2", str(ctx.exception))

    def test_an_invalid_configured_url_raises_on_use(self):
        os.environ[config.ENV_API_URL] = "http://evil.example.com"
        with self.assertRaises(InvalidServerURL):
            config.api_url()

    def test_browser_urls_keep_the_query_and_origins_compare(self):
        self.assertEqual(config.browser_url("https://App.example.com/oauth/authorize?x=1#frag", what="x"),
                         "https://app.example.com/oauth/authorize?x=1")
        self.assertEqual(config.browser_url("https://app.example.com", what="x"), "https://app.example.com/")
        self.assertTrue(config.same_origin("https://a.example.com/x", "https://A.example.com:443/y"))
        self.assertFalse(config.same_origin("https://a.example.com", "http://a.example.com"))
        self.assertFalse(config.same_origin("http://127.0.0.1:1", "http://localhost:1"))
        self.assertFalse(config.same_origin("http://127.0.0.1:1", "http://127.0.0.1:2"))

    def test_settings_dir(self):
        with mock.patch.object(config, "_params", lambda: _stub_params(settings="/tmp/x/oscar_settings.json")):
            self.assertEqual(config.settings_dir(), os.path.abspath("/tmp/x"))
        os.environ["OSCAR_AI_SETTINGS"] = "/tmp/y/settings.json"
        self.assertEqual(config.settings_dir(), os.path.abspath("/tmp/y"))
        del os.environ["OSCAR_AI_SETTINGS"]
        self.assertIsNone(config.settings_dir())


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        discovery.clear_cache()
        self.addCleanup(discovery.clear_cache)
        self.fake = FakeOriginShape()
        self.addCleanup(self.fake.close)

    def test_valid_metadata(self):
        meta = discovery.fetch(self.fake.url)
        url = self.fake.url
        self.assertEqual(meta.issuer, url)
        self.assertEqual(meta.token_endpoint, url + "/oauth/token")
        self.assertEqual(meta.device_authorization_endpoint, url + "/oauth/device_authorization")
        self.assertEqual(meta.revocation_endpoint, url + "/oauth/revoke")
        self.assertEqual(meta.authorization_endpoint, url + "/oauth/authorize")
        self.assertEqual(meta.billing_url, url + "/billing")
        self.assertEqual(meta.gateway_base_url, url + "/v1/llm/anthropic")
        call = self.fake.calls[-1]
        self.assertEqual(call.path, "/.well-known/oauth-authorization-server")
        self.assertTrue(call.headers["user-agent"].startswith("Oscar-FreeCAD/"))

    def test_the_answer_is_cached_per_url_for_an_hour(self):
        discovery.fetch(self.fake.url)
        discovery.fetch(self.fake.url + "/")   # the same server, spelled differently
        self.assertEqual(self.fake.counts["discovery"], 1)
        self.assertIsNotNone(discovery.cached(self.fake.url))
        discovery.fetch(self.fake.url, force=True)
        self.assertEqual(self.fake.counts["discovery"], 2)
        now = discovery._clock()
        with mock.patch.object(discovery, "_clock", lambda: now + discovery.CACHE_TTL_S + 1):
            self.assertIsNone(discovery.cached(self.fake.url))
            discovery.fetch(self.fake.url)
        self.assertEqual(self.fake.counts["discovery"], 3)

    def test_cached_never_touches_the_network(self):
        self.assertIsNone(discovery.cached(self.fake.url))
        self.assertEqual(self.fake.calls, [])

    def assert_refused(self, overrides, *, word=""):
        discovery.clear_cache()
        self.fake.metadata_overrides = overrides
        with self.assertRaises(ServerMetadataError) as ctx:
            discovery.fetch(self.fake.url)
        if word:
            self.assertIn(word, str(ctx.exception))
        self.assertIsNone(discovery.cached(self.fake.url), "invalid metadata is never cached")

    def test_the_issuer_must_be_the_api_url(self):
        self.assert_refused({"issuer": "https://evil.example.com"}, word="issuer")
        self.assert_refused({"issuer": None}, word="issuer")
        self.assert_refused({"issuer": self.fake.url.replace("127.0.0.1", "localhost")}, word="issuer")

    def test_token_endpoints_must_stay_on_the_api_origin(self):
        other = self.fake.url.replace("127.0.0.1", "localhost")
        for key in ("token_endpoint", "device_authorization_endpoint", "revocation_endpoint", "jwks_uri"):
            self.assert_refused({key: other + "/oauth/token"}, word="another server")
            self.assert_refused({key: "https://evil.example.com/steal"}, word="another server")
            self.assert_refused({key: "http://evil.example.com/steal"}, word="another server")
        self.assert_refused({"token_endpoint": None}, word="token_endpoint")

    def test_optional_endpoints_may_be_missing(self):
        self.fake.metadata_overrides = {"device_authorization_endpoint": None, "revocation_endpoint": None,
                                        "jwks_uri": None, "originshape": None}
        meta = discovery.fetch(self.fake.url)
        self.assertEqual((meta.device_authorization_endpoint, meta.revocation_endpoint, meta.jwks_uri), ("", "", ""))
        self.assertEqual(meta.billing_url, "")

    def test_the_authorization_endpoint_may_be_on_another_https_or_loopback_origin(self):
        web = self.fake.url.replace("127.0.0.1", "localhost") + "/oauth/authorize"
        self.fake.metadata_overrides = {"authorization_endpoint": web}
        self.assertEqual(discovery.fetch(self.fake.url).authorization_endpoint, web)
        discovery.clear_cache()
        self.fake.metadata_overrides = {"authorization_endpoint": "https://app.example.com/oauth/authorize"}
        self.assertEqual(discovery.fetch(self.fake.url).authorization_endpoint, "https://app.example.com/oauth/authorize")
        self.assert_refused({"authorization_endpoint": "http://app.example.com/oauth/authorize"}, word="authorization")
        self.assert_refused({"authorization_endpoint": "javascript:alert(1)"}, word="authorization")
        self.assert_refused({"authorization_endpoint": None}, word="authorization")

    def test_pkce_s256_is_required(self):
        self.assert_refused({"code_challenge_methods_supported": ["plain"]}, word="S256")

    def test_bad_links_are_dropped_not_fatal(self):
        self.fake.metadata_overrides = {"originshape": {"billing_url": "javascript:alert(1)",
                                                        "manage_url": "http://evil.example.com/x",
                                                        "download_url": "https://dl.example.com/oscar",
                                                        "gateway_base_url": "https://evil.example.com/v1"}}
        meta = discovery.fetch(self.fake.url)
        self.assertEqual((meta.billing_url, meta.manage_url, meta.gateway_base_url), ("", "", ""))
        self.assertEqual(meta.download_url, "https://dl.example.com/oscar")

    def test_server_failures(self):
        self.fake.fail("/.well-known/oauth-authorization-server", 404, {"detail": "nope"})
        with self.assertRaises(ServerMetadataError) as ctx:
            discovery.fetch(self.fake.url)
        self.assertEqual(ctx.exception.status, 404)
        self.fake.fail("/.well-known/oauth-authorization-server", 503)
        with self.assertRaises(AuthUnavailable):
            discovery.fetch(self.fake.url)
        self.fake.fail("/.well-known/oauth-authorization-server", 200, "<html>a captive portal</html>")
        with self.assertRaises(ServerMetadataError):
            discovery.fetch(self.fake.url)
        self.fake.fail("/.well-known/oauth-authorization-server", 0)   # the connection drops
        with self.assertRaises(AuthUnavailable):
            discovery.fetch(self.fake.url)
        self.assertEqual(discovery.fetch(self.fake.url).issuer, self.fake.url)

    def test_an_unreachable_server_is_unavailable(self):
        url = self.fake.url
        self.fake.close()
        with self.assertRaises(AuthUnavailable) as ctx:
            discovery.fetch(url)
        self.assertIn("127.0.0.1", str(ctx.exception))

    def test_redirects_are_never_followed(self):
        self.fake.fail("/.well-known/oauth-authorization-server", 307, "",
                       headers={"Location": "https://evil.example.com/.well-known/oauth-authorization-server"})
        with self.assertRaises(ServerMetadataError) as ctx:
            discovery.fetch(self.fake.url)
        self.assertEqual(ctx.exception.status, 307)


class TransportTest(unittest.TestCase):
    def test_safe_text_masks_secrets_and_token_shapes(self):
        secret = "fake-rt-SECRETSECRETSECRET"
        text = (f"refresh {secret} was reused; jwt eyJhbGciOiJFUzI1NiJ9.eyJzdWIiOiIxIn0.c2ln key os_live_abcdefghijk "
                f"blob {'A' * 40} encoded {secret.replace('-', '%2D')}\n\x00tail")
        safe = transport.safe_text(text, secret)
        for leak in (secret, "eyJhbGciOiJFUzI1NiJ9", "os_live_abcdefghijk", "A" * 40):
            self.assertNotIn(leak, safe)
        self.assertNotIn("\x00", safe)
        self.assertEqual(transport.safe_text("ab " * 200, limit=10), "ab ab ab a")
        self.assertEqual(transport.safe_text("x" * 500), "[redacted]")   # one long token-shaped run
        self.assertEqual(transport.safe_text(None), "")

    def test_clean_token(self):
        self.assertEqual(transport.clean_token("  abc.def_ghi-jkl  ", "token"), "abc.def_ghi-jkl")
        for bad in ("", "short", "has space inside", "line\nbreak-token", 'quote"token-value', None, 12345678, "x" * 9000):
            with self.assertRaises(AuthError) as ctx:
                transport.clean_token(bad, "refresh token")
            self.assertEqual(ctx.exception.code, "bad_token")
            self.assertIn("refresh token", str(ctx.exception))

    def test_token_set_repr_hides_the_tokens(self):
        tokens = TokenSet("fake-at-ACCESS-SECRET", "fake-rt-REFRESH-SECRET", 900, "oscar:chat")
        text = repr(tokens) + str(tokens)
        self.assertNotIn("ACCESS-SECRET", text)
        self.assertNotIn("REFRESH-SECRET", text)
        self.assertAlmostEqual(tokens.expires_at - tokens.received_at, 900)

    def test_token_set_parsing(self):
        def reply(status, body):
            return transport.Reply(status, {}, body if isinstance(body, bytes) else json.dumps(body).encode())

        tokens = transport.token_set(reply(200, {"access_token": "abcdefgh.ij", "token_type": "bearer",
                                                 "expires_in": 900, "refresh_token": "r" * 43, "scope": "oscar:chat"}))
        self.assertEqual((tokens.access_token, tokens.expires_in, tokens.scope), ("abcdefgh.ij", 900, "oscar:chat"))
        self.assertEqual(transport.token_set(reply(200, {"access_token": "abcdefgh.ij"})).expires_in, 300)
        for odd in (b"NaN", b"Infinity", b"-5", b"true", b'"900"'):   # json.loads accepts NaN and Infinity
            body = b'{"access_token": "abcdefgh.ij", "expires_in": ' + odd + b"}"
            self.assertEqual(transport.token_set(reply(200, body)).expires_in, 300, odd)
        self.assertEqual(transport.token_set(reply(200, {"access_token": "abcdefgh.ij", "expires_in": 10 ** 9}))
                         .expires_in, 86400)
        for body in ({"access_token": "abcdefgh.ij", "token_type": "mac"}, {"token_type": "Bearer"}, b"not json",
                     {"access_token": "has space token"}):
            with self.assertRaises(AuthError):
                transport.token_set(reply(200, body))
        cases = {"invalid_grant": "invalid_grant", "invalid_client": "invalid_client", "weird_error": "weird_error"}
        for error, code in cases.items():
            with self.assertRaises(AuthError) as ctx:
                transport.token_set(reply(400, {"error": error}))
            self.assertEqual(ctx.exception.code, code)
        with self.assertRaises(transport.SignInDeclined):
            transport.token_set(reply(400, {"error": "access_denied"}))
        with self.assertRaises(transport.SignInExpired):
            transport.token_set(reply(400, {"error": "expired_token"}))
        with self.assertRaises(AuthUnavailable):
            transport.token_set(reply(503, {"error": "temporarily_unavailable"}))

    def test_oauth_error_reads_both_error_shapes(self):
        body = {"type": "error", "error": {"type": "permission_error", "code": "waitlisted", "message": "On the list."}}
        self.assertEqual(transport.oauth_error(transport.Reply(403, {}, json.dumps(body).encode())),
                         ("waitlisted", "On the list."))
        self.assertEqual(transport.oauth_error(transport.Reply(400, {}, b'{"error": "Bad Code!"}')), ("", ""))
        self.assertEqual(transport.oauth_error(transport.Reply(500, {}, b"<html>")), ("", ""))

    def test_error_details_takes_text_bytes_or_json(self):
        body = {"type": "error", "error": {"type": "api_error", "code": "llm_not_configured", "message": "No model."}}
        for form in (body, json.dumps(body), json.dumps(body).encode()):
            self.assertEqual(transport.error_details(form), ("llm_not_configured", "No model."))
        self.assertEqual(transport.error_details('{"error": "invalid_grant", "error_description": "used"}'),
                         ("invalid_grant", "used"))
        for junk in (None, "", "<html>", b"\xff\xfe", "[1, 2]", 42):
            self.assertEqual(transport.error_details(junk), ("", ""), junk)
        secret = "fake-rt-ECHOED-SECRET-123"
        self.assertNotIn(secret, transport.error_details({"error": "x", "error_description": secret}, secret)[1])

    def test_501_is_permanent_and_llm_not_configured_is_explained(self):
        for status in (408, 425, 429, 500, 502, 503, 504, 529):
            self.assertTrue(transport.is_unavailable(status), status)
        for status in (200, 400, 401, 402, 403, 404, 501):
            self.assertFalse(transport.is_unavailable(status), status)

        def reply(status, body):
            return transport.Reply(status, {}, json.dumps(body).encode())

        missing = transport.api_error(reply(501, {"type": "error", "error": {
            "type": "api_error", "code": "llm_not_configured", "message": "No AI model is configured."}}), "x")
        self.assertNotIsInstance(missing, AuthUnavailable, "never 'try again in a moment'")
        self.assertEqual((missing.code, missing.status), ("llm_not_configured", 501))
        self.assertEqual(str(missing), "The OriginShape server has no AI model configured.")
        self.assertEqual(transport.LLM_NOT_CONFIGURED_MESSAGE, str(missing))
        other_501 = transport.api_error(transport.Reply(501, {}, b"Not Implemented"), "the account")
        self.assertEqual((type(other_501), other_501.code), (AuthError, "http_error"))
        self.assertIn("HTTP 501", str(other_501))
        self.assertIsInstance(transport.api_error(reply(503, {}), "the account"), AuthUnavailable)
        waitlisted = transport.api_error(reply(403, {"type": "error", "error": {"code": "waitlisted"}}), "the account")
        self.assertEqual((waitlisted.code, waitlisted.status), ("waitlisted", 403))
        self.assertEqual(str(waitlisted), "OriginShape did not return the account (HTTP 403).")

    def test_an_oversized_answer_is_refused(self):
        with FakeOriginShape() as fake:
            fake.fail("/v1/me", 200, "x" * (transport.MAX_BODY_BYTES + 10))
            with self.assertRaises(AuthError) as ctx:
                transport.request("GET", fake.url + "/v1/me")
            self.assertEqual(ctx.exception.code, "bad_response")


class AccountParsingTest(unittest.TestCase):
    def test_an_account_is_parsed(self):
        snap = account.parse_me(ME_EXAMPLE, fetched_at=123.0)
        self.assertEqual((snap.status, snap.email, snap.name, snap.is_admin), ("approved", "a@b.com", "A", False))
        self.assertEqual((snap.plan, snap.plan_name), ("trial", "Trial"))
        self.assertEqual((snap.credits_balance, snap.credits_available, snap.credits_low), (42.5, 42.5, False))
        self.assertEqual(snap.link_billing, "http://localhost:3210/billing")
        self.assertEqual(snap.links["waitlist"], "http://localhost:3210/waitlist")
        self.assertEqual((snap.plugin_latest_version, snap.plugin_min_version), ("0.1.0", "0.1.0"))
        self.assertEqual(snap.fetched_at, 123.0)
        self.assertTrue(snap.approved)
        self.assertFalse(snap.waitlisted or snap.suspended or snap.out_of_credits)

    def test_states(self):
        waitlisted = account.parse_me({**ME_EXAMPLE, "user": {**ME_EXAMPLE["user"], "status": "Waitlisted"}})
        self.assertTrue(waitlisted.waitlisted)
        empty = account.parse_me({**ME_EXAMPLE, "credits": {"balance": 0, "available": -1.5, "low": True}})
        self.assertTrue(empty.out_of_credits)
        self.assertTrue(empty.credits_low)

    def test_missing_or_mistyped_fields_become_defaults(self):
        snap = account.parse_me({"user": {"email": 5, "status": None, "is_admin": "yes"},
                                 "credits": {"balance": "42", "available": float("nan"), "low": 1},
                                 "links": ["not", "a", "dict"], "org": None})
        self.assertEqual((snap.email, snap.status, snap.is_admin), ("", "", False))
        self.assertEqual((snap.credits_balance, snap.credits_available, snap.credits_low), (0.0, 0.0, False))
        self.assertEqual(snap.link_app, "")
        for bad in (None, [], {"user": "x"}, {"credits": {}}):
            with self.assertRaises(AuthError):
                account.parse_me(bad)

    def test_links_must_be_safe_to_open(self):
        links = {"app": "javascript:alert(1)", "billing": "http://evil.example.com/pay",
                 "manage": "https://app.example.com/settings", "waitlist": "file:///etc/passwd", "download": 7}
        snap = account.parse_me({**ME_EXAMPLE, "links": links})
        self.assertEqual(snap.links, {"app": "", "billing": "", "manage": "https://app.example.com/settings",
                                      "waitlist": "", "download": ""})

    def test_dict_round_trip_and_comparison(self):
        snap = account.parse_me(ME_EXAMPLE, fetched_at=1.0)
        again = account.AccountSnapshot.from_dict(snap.to_dict())
        self.assertEqual(again, snap)
        later = account.parse_me(ME_EXAMPLE, fetched_at=2.0)
        self.assertTrue(snap.same_state(later))
        self.assertFalse(snap.same_state(None))
        richer = account.parse_me({**ME_EXAMPLE, "credits": {"balance": 41.0, "available": 41.0, "low": False}})
        self.assertFalse(snap.same_state(richer))
        self.assertEqual(account.AccountSnapshot.from_dict("junk"), account.AccountSnapshot())


class PluginVersionTest(unittest.TestCase):
    """This Oscar against the server's ``/v1/me`` ``plugin`` versions."""

    @staticmethod
    def versions(latest: str, minimum: str) -> dict:
        return {"plugin_latest_version": latest, "plugin_min_version": minimum}

    def test_versions_are_read_as_numbers(self):
        self.assertEqual(account.version_key("0.2.0"), (0, 2, 0, 0))
        self.assertEqual(account.version_key(" v1.10 "), (1, 10, 0, 0))
        self.assertEqual(account.version_key("0.3.0-rc1"), (0, 3, 0, 0))
        self.assertLess(account.version_key("0.9.0"), account.version_key("0.10.0"))
        for bad in ("", "latest", "x1.0", None, 3):
            self.assertIsNone(account.version_key(bad), bad)

    def test_what_the_server_says_about_this_oscar(self):
        cases = ((("0.2.0", "0.2.0"), ""), (("0.3.0", "0.1.0"), "available"), (("0.3.0", "0.3"), "required"),
                 (("0.1.0", "0.1.0"), ""),          # a server whose numbers lag behind this Oscar: nothing to say
                 (("", ""), ""), (("soon", "?"), ""))
        for (latest, minimum), expected in cases:
            self.assertEqual(account.plugin_update(self.versions(latest, minimum), "0.2.0"), expected,
                             (latest, minimum))
        self.assertEqual(account.plugin_update(None), "")
        self.assertEqual(account.plugin_update(self.versions("0.3.0", "0.1.0"), "dev"), "")
        parsed = account.parse_me({**ME_EXAMPLE, "plugin": {"latest_version": "0.4.0", "min_version": "0.3.0"}})
        self.assertEqual(account.plugin_update(parsed, "0.2.0"), "required")
        self.assertEqual(account.plugin_update(parsed.to_dict(), "0.3.1"), "available")

    def test_an_oscar_older_than_the_minimum_is_refused_with_a_message(self):
        versions = self.versions("0.4.0", "0.3.0")
        message = account.outdated_message(versions, "0.2.0")
        self.assertIn("This Oscar (0.2.0)", message)
        self.assertIn("needs Oscar 0.3.0 or newer", message)
        self.assertEqual(account.outdated_message(versions, "0.3.0"), "")
        with self.assertRaises(AuthError) as caught:
            account.require_supported(versions, "0.2.0")
        self.assertEqual((caught.exception.code, str(caught.exception)), (account.PLUGIN_OUTDATED, message))
        account.require_supported(versions, "0.3.0")

    def test_package_xml_and_the_code_name_the_same_version(self):
        # The server compares the version Oscar reports (hello, the user agent) with the one FreeCAD's Addon
        # Manager installed: they must be one number.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "package.xml"), encoding="utf-8") as handle:
            declared = re.search(r"<version>\s*([^<\s]+)\s*</version>", handle.read()).group(1)
        self.assertEqual(declared, oscar.__version__)
        self.assertIsNotNone(account.version_key(oscar.__version__))


class ModelCatalogTest(unittest.TestCase):
    def setUp(self):
        account.clear_model_cache()
        self.addCleanup(account.clear_model_cache)

    @staticmethod
    def cache(data: dict) -> account.ModelCatalog:
        """``data`` as the cached ``/v1/models`` of the configured server (what ``fetch_models`` leaves)."""
        catalog = account.parse_models(data)
        with account._CATALOG_LOCK:
            account._CATALOGS[config.api_url()] = catalog
        return catalog

    def test_the_server_describes_its_models(self):
        catalog = account.parse_models({"default": "origin1", "data": [
            {"id": "extra-model", "display_name": "Extra Model", "description": "A model the server added.",
             "credit_multiplier": 1},
            {"id": "origin1", "description": 7, "credit_multiplier": "lots"}]})
        design, plain = catalog.get("extra-model"), catalog.get("origin1")
        self.assertEqual((design.description, design.credit_multiplier), ("A model the server added.", 1.0))
        self.assertEqual((plain.description, plain.credit_multiplier), ("", 0.0))

    def test_a_cached_model_by_its_alias(self):
        self.assertIsNone(account.cached_model("extra-model"))   # nothing fetched yet
        self.cache({"data": [{"id": "extra-model", "display_name": "Extra Model"}]})
        self.assertEqual(account.cached_model("extra-model").display_name, "Extra Model")
        self.assertIsNone(account.cached_model("origin1-str"))
        self.assertTrue(account.is_model_id("origin1-str"))
        for bad in ("", "has space", None, "x" * 101):
            self.assertFalse(account.is_model_id(bad), bad)

    def test_a_model_the_server_added_gets_its_capabilities_from_the_list(self):
        profile = types.SimpleNamespace(provider="originshape", base_url="", model="extra-model", caps_override={})
        self.assertEqual(models.resolve_caps(profile).source, "default")   # unknown until the server lists it
        self.cache({"data": [
            {"id": "extra-model", "context_window": 200000, "max_output_tokens": 32000,
             "capabilities": {"vision": False, "tools": True, "prompt_caching": True}},
            {"id": "extra-adaptive", "context_window": 400000,
             "capabilities": {"vision": True, "reasoning": "adaptive"}},
            {"id": "extra-budget", "capabilities": {"reasoning": "budget"}}]})
        caps = models.resolve_caps(profile)
        self.assertEqual((caps.vision, caps.context_window, caps.max_output, caps.prompt_cache, caps.reasoning_control),
                         (False, 200000, 32000, True, "none"))   # no reasoning declared: the server sets thinking
        self.assertEqual(caps.source, "catalog")

        def resolved(alias):
            return models.resolve_caps(types.SimpleNamespace(provider="originshape", base_url="", model=alias,
                                                             caps_override={}))

        # Thinking as the server declares it, with the server's window.
        adaptive = resolved("extra-adaptive")
        self.assertEqual((adaptive.reasoning_control, adaptive.reasoning_replay, adaptive.context_window,
                          adaptive.vision), ("anthropic_adaptive", True, 400000, True))
        self.assertEqual(resolved("extra-budget").reasoning_control, "anthropic_budget")
        # The lineup has no rows of its own: an alias this list does not name has the defaults.
        self.assertEqual(resolved("origin1").source, "default")

    def test_a_model_list_is_parsed(self):
        data = {"default": "oscar-default", "data": [
            {"id": "oscar-default", "display_name": "Oscar (best)", "context_window": 200000,
             "max_output_tokens": 64000,
             "capabilities": {"vision": True, "tools": True, "reasoning": "adaptive", "prompt_caching": True,
                              "nested": {"dropped": True}}},
            {"id": "bad id with spaces"}, "junk",
            {"id": "oscar-fast"}]}
        catalog = account.parse_models(data)
        self.assertEqual(catalog.default, "oscar-default")
        self.assertEqual([m.id for m in catalog.models], ["oscar-default", "oscar-fast"])
        best = catalog.get("oscar-default")
        self.assertEqual((best.context_window, best.max_output_tokens), (200000, 64000))
        self.assertEqual(best.capabilities, {"vision": True, "tools": True, "reasoning": "adaptive",
                                             "prompt_caching": True})
        self.assertEqual(account.parse_models({"default": "missing", "data": []}).default, "origin1")
        with self.assertRaises(AuthError):
            account.parse_models({"models": []})

    def test_the_lineup_before_the_server_answers(self):
        self.assertEqual(account.LINEUP, ("origin1", "origin1-super"))


if __name__ == "__main__":
    unittest.main()
