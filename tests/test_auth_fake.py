"""support.fake_originshape: the fake OriginShape API behaves as its docstring says, so the auth and provider tests can
rely on it. Raw HTTP only (requests): the routes, the protocol checks (PKCE, exact redirect URI, code reuse, refresh
rotation and reuse detection) and the gateway: the server's checks in its order (403 waitlisted / suspended,
501 llm_not_configured, 400 model_unavailable, 402 insufficient_credits), then the scripted replies (SSE or JSON by
default, typed errors, a scripted Round, recording).

Pure python (stdlib + requests); ``support.scripted_llm`` (for the Round case) imports ``oscar.providers``.
"""

import base64
import hashlib
import json
import os
import sys
import unittest
from urllib.parse import parse_qs, urlencode, urlsplit

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from support.fake_originshape import FakeOriginShape, gateway_error, raw_reply, sse_text  # noqa: E402

VERIFIER = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
CHALLENGE = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
REDIRECT = "http://127.0.0.1:53111/callback"


def _sse_events(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


class FakeCase(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOriginShape()
        self.addCleanup(self.fake.close)
        self.url = self.fake.url

    def authorize(self, **overrides) -> requests.Response:
        params = {"response_type": "code", "client_id": "oscar-freecad", "redirect_uri": REDIRECT, "state": "st-1",
                  "scope": "oscar:chat oscar:account offline_access", "code_challenge": CHALLENGE,
                  "code_challenge_method": "S256", "device_name": "Box"}
        params.update(overrides)
        query = urlencode({k: v for k, v in params.items() if v is not None})
        return requests.get(f"{self.url}/oauth/authorize?{query}", allow_redirects=False, timeout=5)

    def code(self) -> str:
        location = self.authorize().headers["Location"]
        return parse_qs(urlsplit(location).query)["code"][0]

    def token(self, **form) -> requests.Response:
        form.setdefault("client_id", "oscar-freecad")
        return requests.post(self.url + "/oauth/token", data=form, timeout=5)

    def exchange(self, code: str, verifier: str = VERIFIER, redirect: str = REDIRECT) -> requests.Response:
        return self.token(grant_type="authorization_code", code=code, redirect_uri=redirect, code_verifier=verifier)


class RoutesTest(FakeCase):
    def test_discovery_and_unknown_routes(self):
        meta = requests.get(self.url + "/.well-known/oauth-authorization-server", timeout=5).json()
        self.assertEqual(meta["issuer"], self.url)
        self.assertEqual(meta["code_challenge_methods_supported"], ["S256"])
        self.fake.metadata_overrides = {"jwks_uri": None, "issuer": "https://other.example.com"}
        meta = requests.get(self.url + "/.well-known/oauth-authorization-server", timeout=5).json()
        self.assertNotIn("jwks_uri", meta)
        self.assertEqual(meta["issuer"], "https://other.example.com")
        self.assertEqual(requests.get(self.url + "/.well-known/jwks.json", timeout=5).json(), {"keys": []})
        for path in ("/", "/v1/unknown", "/oauth/authorize/x", "/v1/llm/anthropic/v1/messages"):
            answer = requests.get(self.url + path, timeout=5)
            self.assertEqual(answer.status_code, 404, path)
            self.assertEqual(answer.json()["error"]["code"], "not_found")
        self.assertEqual(self.fake.counts["not_found"], 4)

    def test_calls_are_recorded(self):
        requests.post(self.url + "/oauth/revoke", data={"token": "x" * 20, "client_id": "oscar-freecad"},
                      headers={"X-Probe": "1"}, timeout=5)
        call = self.fake.calls[-1]
        self.assertEqual((call.method, call.path, call.form["token"]), ("POST", "/oauth/revoke", "x" * 20))
        self.assertEqual(call.headers["x-probe"], "1")


class AuthorizeTest(FakeCase):
    def test_an_approving_user_is_sent_back_with_a_code_and_the_state(self):
        answer = self.authorize()
        self.assertEqual(answer.status_code, 302)
        location = urlsplit(answer.headers["Location"])
        self.assertEqual(f"{location.scheme}://{location.netloc}{location.path}", REDIRECT)
        query = parse_qs(location.query)
        self.assertEqual(query["state"], ["st-1"])
        self.assertTrue(query["code"][0].startswith("fake-code-"))

    def test_deny_and_a_forged_state(self):
        self.fake.authorize_decision = "deny"
        self.assertEqual(parse_qs(urlsplit(self.authorize().headers["Location"]).query),
                         {"error": ["access_denied"], "state": ["st-1"]})
        self.fake.authorize_decision, self.fake.authorize_state = "approve", "forged"
        self.assertEqual(parse_qs(urlsplit(self.authorize().headers["Location"]).query)["state"], ["forged"])

    def test_a_bad_client_or_redirect_uri_is_never_redirected(self):
        for overrides in ({"client_id": "someone-else"}, {"redirect_uri": "https://evil.example.com/callback"},
                          {"redirect_uri": "http://127.0.0.1:53111/other"}, {"redirect_uri": None}):
            answer = self.authorize(**overrides)
            self.assertEqual(answer.status_code, 400, overrides)
            self.assertNotIn("Location", answer.headers)

    def test_other_problems_are_redirected_as_invalid_request(self):
        for overrides in ({"code_challenge": None}, {"code_challenge": "short"}, {"code_challenge_method": "plain"},
                          {"response_type": "token"}, {"scope": "oscar:chat admin"}):
            query = parse_qs(urlsplit(self.authorize(**overrides).headers["Location"]).query)
            self.assertEqual(query["error"], ["invalid_request"], overrides)


class TokenTest(FakeCase):
    def test_pkce_is_verified_for_real(self):
        self.assertEqual(self.exchange(self.code(), verifier="x" * 43).json()["error"], "invalid_grant")
        self.assertEqual(self.exchange(self.code(), redirect="http://127.0.0.1:9/callback").json()["error"],
                         "invalid_grant")
        answer = self.exchange(self.code())
        self.assertEqual(answer.status_code, 200)
        body = answer.json()
        self.assertEqual((body["token_type"], body["expires_in"]), ("Bearer", 900))
        self.assertEqual(self.fake.grant_of(body["refresh_token"]).device_name, "Box")
        digest = base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).rstrip(b"=").decode()
        self.assertEqual(digest, CHALLENGE)

    def test_a_code_used_twice_revokes_its_grant(self):
        code = self.code()
        first = self.exchange(code).json()
        self.assertEqual(self.exchange(code).json()["error"], "invalid_grant")
        self.assertTrue(self.fake.grant_of(first["refresh_token"]).revoked)
        me = requests.get(self.url + "/v1/me", headers={"Authorization": f"Bearer {first['access_token']}"}, timeout=5)
        self.assertEqual(me.status_code, 401)

    def test_refresh_rotates_and_reuse_revokes_the_grant(self):
        first = self.fake.issue_tokens()
        second = self.token(grant_type="refresh_token", refresh_token=first["refresh_token"]).json()
        self.assertNotEqual(second["refresh_token"], first["refresh_token"])
        self.assertEqual(self.fake.refresh_state(first["refresh_token"]), "used")
        reused = self.token(grant_type="refresh_token", refresh_token=first["refresh_token"])
        self.assertEqual((reused.status_code, reused.json()["error"]), (400, "invalid_grant"))
        self.assertEqual(self.fake.refresh_state(second["refresh_token"]), "revoked")
        self.assertEqual(self.token(grant_type="refresh_token", refresh_token=second["refresh_token"]).json()["error"],
                         "invalid_grant")

    def test_client_and_grant_type_checks(self):
        self.assertEqual(self.token(grant_type="password").json()["error"], "unsupported_grant_type")
        answer = self.token(grant_type="refresh_token", refresh_token="x" * 20, client_id="other")
        self.assertEqual((answer.status_code, answer.json()["error"]), (401, "invalid_client"))

    def test_json_bodies_are_accepted(self):
        tokens = self.fake.issue_tokens()
        answer = requests.post(self.url + "/oauth/token", timeout=5, json={
            "grant_type": "refresh_token", "client_id": "oscar-freecad", "refresh_token": tokens["refresh_token"]})
        self.assertEqual(answer.status_code, 200)

    def test_scripted_failures_and_a_dropped_connection(self):
        self.fake.fail("/oauth/token", 503, times=2)
        self.assertEqual(self.token(grant_type="password").status_code, 503)
        self.assertEqual(self.token(grant_type="password").json()["error"], "temporarily_unavailable")
        self.assertEqual(self.token(grant_type="password").status_code, 400)
        self.fake.fail("/oauth/token", 0)
        with self.assertRaises(requests.ConnectionError):
            self.token(grant_type="password")


class AccountRoutesTest(FakeCase):
    def test_me_and_models_need_a_valid_bearer(self):
        tokens = self.fake.issue_tokens()
        key = self.fake.add_api_key()
        for headers in ({"Authorization": f"Bearer {tokens['access_token']}"}, {"Authorization": f"Bearer {key}"},
                        {"x-api-key": key}):
            self.assertEqual(requests.get(self.url + "/v1/me", headers=headers, timeout=5).status_code, 200)
            self.assertEqual(requests.get(self.url + "/v1/models", headers=headers, timeout=5).json()["default"],
                             "oscar-default")
        for headers in ({}, {"Authorization": "Bearer nope-nope-nope"}, {"Authorization": f"Basic {key}"}):
            answer = requests.get(self.url + "/v1/me", headers=headers, timeout=5)
            self.assertEqual(answer.status_code, 401)
            self.assertEqual(answer.json()["error"]["type"], "authentication_error")

    def test_set_account_and_reject_token(self):
        tokens = self.fake.issue_tokens()
        auth = {"Authorization": f"Bearer {tokens['access_token']}"}
        self.fake.set_account(status="waitlisted", balance=3.5, low=True, email="b@example.com", plan="pro")
        me = requests.get(self.url + "/v1/me", headers=auth, timeout=5).json()
        self.assertEqual((me["user"]["status"], me["user"]["email"], me["org"]["plan"]),
                         ("waitlisted", "b@example.com", "pro"))
        self.assertEqual(me["credits"], {"balance": 3.5, "available": 3.5, "low": True})
        self.fake.reject_token(tokens["access_token"])
        self.assertEqual(requests.get(self.url + "/v1/me", headers=auth, timeout=5).status_code, 401)
        self.assertEqual(requests.get(self.url + "/v1/me", headers=auth, timeout=5).status_code, 200, "once only")

    def test_an_expired_access_token_is_refused(self):
        self.fake.access_ttl = 0
        tokens = self.fake.issue_tokens()
        answer = requests.get(self.url + "/v1/me", headers={"Authorization": f"Bearer {tokens['access_token']}"},
                              timeout=5)
        self.assertEqual(answer.status_code, 401)


class GatewayCase(FakeCase):
    def post(self, body=None, headers=None) -> requests.Response:
        return requests.post(self.fake.gateway_url, json=body or {"model": "oscar-default", "stream": True},
                             headers=headers if headers is not None else self.auth, timeout=5)

    def setUp(self):
        super().setUp()
        self.auth = {"Authorization": f"Bearer {self.fake.issue_tokens()['access_token']}",
                     "X-Oscar-Session": "chat-1", "anthropic-version": "2023-06-01"}


class GatewayTest(GatewayCase):
    def test_the_default_answer_is_an_anthropic_stream(self):
        answer = self.post({"model": "oscar-default", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(answer.status_code, 200)
        self.assertTrue(answer.headers["Content-Type"].startswith("text/event-stream"))
        events = _sse_events(answer.text)
        self.assertEqual([e["type"] for e in events][0], "message_start")
        self.assertEqual(events[-1]["type"], "message_stop")
        text = "".join(e["delta"]["text"] for e in events if e["type"] == "content_block_delta")
        self.assertEqual(text, "Hello from the fake OriginShape gateway.")
        recorded = self.fake.gateway_requests[-1]
        self.assertEqual(recorded.path, "/v1/llm/anthropic/v1/messages")
        self.assertEqual(recorded.headers["x-oscar-session"], "chat-1")
        self.assertEqual(recorded.json["messages"][0]["content"], "hi")

    def test_scripted_replies_in_order(self):
        self.fake.gateway = [
            gateway_error(402, "insufficient_credits", "You are out of credits.", action_url=self.url + "/billing"),
            gateway_error(429, "rate_limited", "Slow down.", headers={"Retry-After": "7"}),
            "custom words",
            sse_text("with usage", usage={"input_tokens": 5, "cache_read_input_tokens": 3, "output_tokens": 2}),
            raw_reply('{"type":"error","error":{"type":"invalid_request_error","message":"temperature is deprecated"}}',
                      status=400, content_type="application/json"),
        ]
        paid = self.post()
        self.assertEqual(paid.status_code, 402)
        self.assertEqual(paid.json()["error"], {"type": "billing_error", "code": "insufficient_credits",
                                                "message": "You are out of credits.",
                                                "action_url": self.url + "/billing"})
        limited = self.post()
        self.assertEqual((limited.status_code, limited.headers["Retry-After"]), (429, "7"))
        self.assertEqual(limited.json()["error"]["type"], "rate_limit_error")
        self.assertIn("custom words", self.post().text)
        usage = _sse_events(self.post().text)[0]["message"]["usage"]
        self.assertEqual(usage, {"input_tokens": 5, "cache_read_input_tokens": 3, "output_tokens": 1})
        upstream = self.post()
        self.assertEqual(upstream.status_code, 400)
        self.assertIn("temperature is deprecated", upstream.text)
        self.assertIn("Hello", self.post().text, "the default once the script is used up")
        self.assertEqual(self.fake.counts["gateway"], 6)

    def test_a_scripted_round(self):
        from support.scripted_llm import say   # imports oscar.providers: pure, but only needed here

        self.fake.gateway = [say("a scripted round")]
        events = _sse_events(self.post().text)
        self.assertIn("a scripted round", json.dumps(events))
        self.assertEqual(events[-1]["type"], "message_stop")

    def test_a_missing_or_bad_bearer_is_401(self):
        self.assertEqual(self.post(headers={}).status_code, 401)
        answer = self.post(headers={"Authorization": "Bearer fake-at-unknown-token"})
        self.assertEqual(answer.status_code, 401)
        self.assertEqual(answer.json()["error"]["code"], "unauthorized")
        self.assertEqual(len(self.fake.gateway_requests), 2, "refused requests are recorded too")

    def test_the_answer_without_streaming_is_json_naming_the_alias(self):
        answer = self.post({"model": "oscar-fast", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(answer.status_code, 200)
        self.assertTrue(answer.headers["Content-Type"].startswith("application/json"))
        body = answer.json()
        self.assertEqual((body["type"], body["model"], body["stop_reason"]), ("message", "oscar-fast", "end_turn"))
        self.assertEqual(body["content"][0]["text"], "Hello from the fake OriginShape gateway.")
        streamed = _sse_events(self.post({"model": "oscar-fast", "stream": True}).text)
        self.assertEqual(streamed[0]["message"]["model"], "oscar-fast")


class GatewayPolicyTest(GatewayCase):
    """The server's own checks, in its order, before any scripted reply is served."""

    def error(self, answer: requests.Response) -> tuple:
        return answer.status_code, answer.json()["error"]["code"], answer.json()["error"]["type"]

    def test_waitlisted_and_suspended_accounts_are_403(self):
        self.fake.gateway = ["never served"]
        self.fake.set_account(status="waitlisted")
        answer = self.post()
        self.assertEqual(self.error(answer), (403, "waitlisted", "permission_error"))
        self.assertEqual(answer.json()["error"]["action_url"], self.url + "/waitlist")
        self.fake.set_account(status="suspended")
        self.assertEqual(self.error(self.post()), (403, "suspended", "permission_error"))
        self.fake.enforce_account = False
        self.assertIn("never served", self.post().text)

    def test_a_server_without_an_ai_model_is_501_llm_not_configured(self):
        self.fake.llm_configured = False
        answer = self.post()
        self.assertEqual(self.error(answer), (501, "llm_not_configured", "api_error"))
        self.assertEqual(answer.json()["error"]["message"], "The OriginShape server has no AI model configured.")
        self.fake.set_account(status="waitlisted")   # authorization is checked first
        self.assertEqual(self.error(self.post())[1], "waitlisted")

    def test_unknown_models_and_bodies_are_400(self):
        answer = self.post({"model": "not-an-alias", "stream": True})   # the wire must carry an alias
        self.assertEqual(self.error(answer), (400, "model_unavailable", "invalid_request_error"))
        self.assertIn("oscar-default", answer.json()["error"]["message"])
        not_json = requests.post(self.fake.gateway_url, data=b"model=oscar-default", headers=self.auth, timeout=5)
        self.assertEqual(self.error(not_json)[:2], (400, "invalid_request"))
        self.fake.enforce_models = False
        self.assertEqual(self.post({"model": "anything", "stream": True}).status_code, 200)

    def test_no_credits_left_is_402_with_the_billing_link(self):
        self.fake.set_account(balance=0.0)
        answer = self.post()
        self.assertEqual(self.error(answer), (402, "insufficient_credits", "billing_error"))
        self.assertEqual(answer.json()["error"]["action_url"], self.url + "/billing")
        self.fake.set_account(balance=0.5)
        self.assertEqual(self.post().status_code, 200)


class DeviceRoutesTest(FakeCase):
    def test_the_state_machine(self):
        start = requests.post(self.url + "/oauth/device_authorization", timeout=5,
                              data={"client_id": "oscar-freecad", "device_name": "Box"}).json()
        self.assertRegex(start["user_code"], r"^[BCDFGHJKLMNPQRSTVWXZ]{4}-[BCDFGHJKLMNPQRSTVWXZ]{4}$")
        self.assertEqual((start["interval"], start["expires_in"]), (5, 600))

        def poll():
            return self.token(grant_type="urn:ietf:params:oauth:grant-type:device_code",
                              device_code=start["device_code"]).json()

        self.fake.enforce_interval = False
        self.assertEqual(poll()["error"], "authorization_pending")
        self.fake.enforce_interval = True   # the next poll comes well within 5 s of the last one
        self.assertEqual(poll()["error"], "slow_down")
        self.assertEqual(self.fake.devices[start["user_code"]].interval, 10)
        self.fake.enforce_interval = False
        self.fake.approve_device(start["user_code"])
        self.assertIn("access_token", poll())
        self.assertEqual(poll()["error"], "invalid_grant", "a device code works once")
        other = requests.post(self.url + "/oauth/device_authorization", timeout=5,
                              data={"client_id": "oscar-freecad"}).json()
        self.fake.deny_device(other["user_code"])
        self.assertEqual(self.token(grant_type="urn:ietf:params:oauth:grant-type:device_code",
                                    device_code=other["device_code"]).json()["error"], "access_denied")
        late = requests.post(self.url + "/oauth/device_authorization", timeout=5,
                             data={"client_id": "oscar-freecad"}).json()
        self.fake.expire_device()
        self.assertEqual(self.token(grant_type="urn:ietf:params:oauth:grant-type:device_code",
                                    device_code=late["device_code"]).json()["error"], "expired_token")


if __name__ == "__main__":
    unittest.main()
