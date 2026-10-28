"""OriginShapeProvider against the fake OriginShape API (tests/support/fake_originshape.py), over real HTTP.

Covered: an alias's body equals its real model's body except ``model``; the request goes to the API's gateway with a
Bearer token and never ``x-api-key``; a 401 renews the token once and resends; 402 / 403 / 429 / 501 map to typed
errors (and 429 is retried within the budget only); upstream errors passed through are never read as the account;
``request_context`` headers; no token ever appears in an error; the ChatLoop hooks change nothing for the other
providers. Pure python (stdlib + requests): the keystore runs in memory and retry sleeps are recorded, not slept.
"""

import hashlib
import json
import os
import sys
import threading
import time
import types
import unittest
import uuid
from unittest import mock

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import __version__, models, provider_chat, provider_originshape, provider_probe, providers  # noqa: E402
from oscar.auth import account, discovery, keystore, session  # noqa: E402
from oscar.auth.transport import LLM_NOT_CONFIGURED_MESSAGE, TokenSet  # noqa: E402
from oscar.provider_transport import ProviderError  # noqa: E402
from support.fake_originshape import (GATEWAY_PATH, FakeOriginShape, envelope, gateway_error, raw_reply,  # noqa: E402
                                      sse_text)

Message, ToolSpec = providers.Message, providers.ToolSpec
USER = [Message(role="user", content="Make a 20 mm cube.")]
TOOLS = [ToolSpec("create_object", "Create an object.", {"type": "object", "properties": {"kind": {"type": "string"}}}),
         ToolSpec("get_objects", "List the objects.", {"type": "object", "properties": {}})]
ALIASES = ("origin1", "origin1-super", "oscar-default", "oscar-fast")
CHAT_ID = "20261007-101010-a1b2c3"
TURN_ID = "T20261007-101011-9f3e"


def bare_profile(provider: str, model: str, base_url: str = "") -> types.SimpleNamespace:
    """A profile as the provider reads it (``oscar.params`` imports FreeCAD, so the pure suite cannot build one)."""
    return types.SimpleNamespace(name="t", provider=provider, base_url=base_url, model=model, api_key="",
                                 temperature=1.0, max_tokens=32000, reasoning="auto", caps_override={})


def forget_learned(*profiles) -> None:
    """Drop what the providers learned (adaptations, persisted capability facts) about ``profiles``."""
    providers._LEARNED.clear()
    store = models._params()
    for profile in profiles:
        key = models.caps_key(profile)
        with models._MIRROR_LOCK:
            models._LEARNED.pop(key, None)
            models._RUNTIME.pop(key, None)
        if store is not None:
            store.set_model_caps(key, {})


def collect(events) -> dict:
    state = {"text": "", "kinds": [], "error": None, "done": None}
    for kind, payload in events:
        state["kinds"].append(kind)
        if kind == "text":
            state["text"] += payload
        elif kind == "reset":
            state["text"] = ""
        elif kind == "error":
            state["error"] = payload
        elif kind == "done":
            state["done"] = payload
    return state


class _Case(unittest.TestCase):
    """A fake API on 127.0.0.1 that ``OSCAR_API_URL`` points at, a memory keystore, no real retry sleeps."""

    def setUp(self):
        for reset in (discovery.clear_cache, session.reset_sessions, keystore.reset_memory, account.clear_model_cache):
            reset()
            self.addCleanup(reset)
        self.fake = FakeOriginShape()
        self.addCleanup(self.fake.close)
        env = mock.patch.dict(os.environ, {"OSCAR_API_URL": self.fake.url, keystore.ENV_STORE: "memory"})
        env.start()
        self.addCleanup(env.stop)
        self.sleeps: list[tuple[float, tuple]] = []

        def fake_sleep(loop, delay, call, should_stop, retry):
            self.sleeps.append((round(delay, 3), tuple(retry)))
            return iter(())

        for patcher in (mock.patch.object(provider_chat.ChatLoop, "_sleep", fake_sleep),
                        mock.patch.object(provider_chat, "log_event", lambda *a, **k: None)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.profiles = [bare_profile("originshape", alias) for alias in ALIASES]
        forget_learned(*self.profiles)
        self.addCleanup(lambda: forget_learned(*self.profiles))

    # -- helpers ----------------------------------------------------------------------------------------------

    def sign_in(self, *, age: float = 0.0) -> TokenSet:
        """An OAuth sign-in of a fresh grant at the fake; its access token was received ``age`` seconds ago."""
        data = self.fake.issue_tokens()
        tokens = TokenSet(data["access_token"], data["refresh_token"], data["expires_in"], data["scope"],
                          received_at=time.monotonic() - age)
        session.get_session(self.fake.url).complete_sign_in(tokens)
        return tokens

    def provider(self, model: str = "oscar-default") -> provider_originshape.OriginShapeProvider:
        profile = bare_profile("originshape", model)
        if profile.model not in ALIASES:
            self.profiles.append(profile)
            forget_learned(profile)
        return providers.make_provider(profile)

    def chat(self, prov, *, tools=(), **opts) -> dict:
        return collect(prov.chat("SYSTEM", USER, list(tools), None, **opts))

    def gateway_headers(self, n: int = -1) -> dict:
        return self.fake.gateway_requests[n].headers


class FactoryTest(_Case):
    def test_make_provider_builds_it_without_network_or_keystore(self):
        with mock.patch.object(requests, "post", side_effect=AssertionError("network")), \
                mock.patch.object(requests, "get", side_effect=AssertionError("network")), \
                mock.patch.object(keystore, "load", side_effect=AssertionError("keystore")):
            prov = providers.make_provider(bare_profile("originshape", "oscar-default"))
        self.assertIsInstance(prov, provider_originshape.OriginShapeProvider)
        self.assertIsInstance(prov, providers.AnthropicProvider)
        self.assertEqual((prov.name, prov.kind), ("originshape", "originshape"))
        self.assertEqual(providers.COMING_SOON_KINDS, {})

    def test_the_endpoint_is_always_the_api_gateway(self):
        self.assertEqual(self.provider()._endpoint(), self.fake.url + GATEWAY_PATH)
        with mock.patch.dict(os.environ, {"OSCAR_API_URL": "https://api.example.test/"}):
            self.assertEqual(self.provider()._endpoint(), "https://api.example.test" + GATEWAY_PATH)

    def test_one_server_per_provider_even_if_the_setting_changes_mid_turn(self):
        prov = self.provider()
        first = prov._endpoint()
        with mock.patch.dict(os.environ, {"OSCAR_API_URL": "https://elsewhere.example.test"}):
            self.assertEqual(prov._endpoint(), first)
            self.assertTrue(self.provider()._endpoint().startswith("https://elsewhere.example.test/"))

    def test_an_insecure_server_url_is_refused_before_anything_is_sent(self):
        self.sign_in()
        with mock.patch.dict(os.environ, {"OSCAR_API_URL": "http://api.example.test"}):
            state = self.chat(self.provider())
        error = state["error"]
        self.assertIsInstance(error, ProviderError)
        self.assertEqual((error.kind, error.code, error.retriable), ("auth", "insecure_url", False))
        self.assertEqual(self.fake.gateway_requests, [])


class AliasBodyTest(_Case):
    """(a) An alias's body follows what the server's list declares for it, with the account's rules: the alias as
    ``model``, no ``temperature``, thinking switched off as low effort, never a forced tool choice."""

    def setUp(self):
        super().setUp()
        self.sign_in()
        account.fetch_models(self.fake.url)   # the aliases' capabilities come from the server

    OPTIONS = [{}, {"reasoning": "off"}, {"reasoning": "low"}, {"reasoning": "medium"}, {"reasoning": "high"},
               {"tool_choice": "required"}, {"tool_choice": "create_object"}, {"tool_choice": "none"},
               {"tool_choice": "auto", "max_tokens": 12000}, {"parallel_tool_calls": False},
               {"tool_choice": "required", "reasoning": "medium"}]

    def test_the_accounts_rules_hold_for_every_alias_and_option(self):
        for alias in ALIASES:
            for opts in self.OPTIONS:
                for tools in ([], TOOLS):
                    with self.subTest(alias=alias, opts=opts, tools=bool(tools)):
                        body = self.provider(alias)._body("SYSTEM", USER, tools, **opts)
                        self.assertEqual(body["model"], alias)
                        self.assertNotIn("temperature", body)
                        self.assertNotIn((body.get("tool_choice") or {}).get("type"), ("any", "tool"))
                        self.assertIn((body.get("thinking") or {}).get("type"), (None, "adaptive"))

    def test_thinking_follows_the_requested_effort(self):
        prov = self.provider("origin1")
        high = prov._body("SYSTEM", USER, [], reasoning="high")
        self.assertEqual((high["thinking"], high["output_config"]), ({"type": "adaptive"}, {"effort": "high"}))
        off = prov._body("SYSTEM", USER, [], reasoning="off")
        self.assertNotIn("thinking", off)
        self.assertEqual(off["output_config"], {"effort": "low"})
        required = prov._body("SYSTEM", USER, TOOLS, tool_choice="required", parallel_tool_calls=False)
        self.assertEqual(required["tool_choice"], {"type": "auto", "disable_parallel_tool_use": True})

    def test_the_wire_carries_the_alias_and_the_accounts_body(self):
        for alias in ALIASES:
            with self.subTest(alias=alias):
                state = self.chat(self.provider(alias), tools=TOOLS, reasoning="medium", tool_choice="required")
                self.assertIsNone(state["error"])
                expected = self.provider(alias)._body("SYSTEM", USER, TOOLS, reasoning="medium", tool_choice="required")
                self.assertEqual(self.fake.gateway_requests[-1].json, expected)

    def test_capabilities_of_an_alias_are_what_the_server_declares(self):
        for alias in ALIASES:
            with self.subTest(alias=alias):
                caps = models.resolve_caps(bare_profile("originshape", alias))
                self.assertEqual(caps.source, "catalog")
                self.assertEqual((caps.reasoning_control, caps.vision, caps.context_window, caps.max_output,
                                  caps.prompt_cache), ("anthropic_adaptive", True, 1000000, 64000, True))
                self.assertEqual(models.host_class(bare_profile("originshape", alias)), "originshape")
        origin1 = next(m for m in self.fake.models["data"] if m["id"] == "origin1")
        origin1.update(capabilities={"reasoning": "budget", "vision": False}, max_output_tokens=32000)
        account.fetch_models(self.fake.url)
        caps = models.resolve_caps(bare_profile("originshape", "origin1"))
        self.assertEqual((caps.reasoning_control, caps.vision, caps.max_output), ("anthropic_budget", False, 32000))

    def test_the_output_cap_follows_the_model_list(self):
        """The gateway would silently lower a max_tokens above the model's own cap (audit 2026-10-28)."""
        origin1 = next(m for m in self.fake.models["data"] if m["id"] == "origin1")
        origin1["max_output_tokens"] = 16000
        account.fetch_models(self.fake.url)
        prov = self.provider("origin1")
        self.assertEqual(prov._max_tokens(), 16000)
        self.assertEqual(prov._max_tokens(8000), 8000)
        self.assertEqual(self.provider("oscar-fast")._max_tokens(), 16000, "a retired alias: its tier's cap")
        self.assertEqual(self.provider("origin1-super")._max_tokens(), 32000, "the profile's cap below the model's")

    def test_without_the_model_list_the_defaults_apply_and_no_request_is_made(self):
        account.clear_model_cache()
        before = self.fake.counts["models"]
        body = self.provider("origin1")._body("SYSTEM", USER, [])
        self.assertEqual(self.fake.counts["models"], before, "no request while the body is built")
        self.assertEqual(models.resolve_caps(bare_profile("originshape", "origin1")).source, "default")
        self.assertNotIn("temperature", body)

    def test_the_model_list_answers_the_aliases_without_a_request(self):
        profile = bare_profile("originshape", "oscar-default")
        with mock.patch.object(requests, "get", side_effect=AssertionError("network")), \
                mock.patch.object(requests, "post", side_effect=AssertionError("network")):
            self.assertEqual(provider_probe.list_models(profile), (["origin1", "origin1-super"], ""))
            self.assertEqual(provider_probe.probe(profile), {})


class HeadersTest(_Case):
    """(b) Bearer present, x-api-key absent, the API's host; (e) request_context headers."""

    def test_a_signed_in_request(self):
        tokens = self.sign_in()
        state = self.chat(self.provider())
        self.assertEqual(state["text"], "Hello from the fake OriginShape gateway.")
        call = self.fake.gateway_requests[0]
        self.assertEqual(call.path, GATEWAY_PATH)
        headers = call.headers
        self.assertEqual(headers["authorization"], f"Bearer {tokens.access_token}")
        self.assertNotIn("x-api-key", headers)
        self.assertEqual(headers["host"], self.fake.url.split("://", 1)[1])
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(headers["user-agent"], f"Oscar-FreeCAD/{__version__}")
        self.assertEqual(headers["x-oscar-client"], f"freecad/{__version__}")
        for name in ("x-oscar-session", "x-oscar-project", "x-oscar-turn"):
            self.assertNotIn(name, headers)

    def test_an_api_key_is_sent_as_the_bearer(self):
        key = self.fake.add_api_key()
        session.get_session(self.fake.url).use_api_key(key)
        self.assertIsNone(self.chat(self.provider())["error"])
        self.assertEqual(self.gateway_headers()["authorization"], f"Bearer {key}")
        self.assertNotIn("x-api-key", self.gateway_headers())

    def test_request_context_headers(self):
        self.sign_in()
        prov = self.provider()
        prov.request_context = {"session": CHAT_ID, "turn": TURN_ID}
        self.chat(prov)
        headers = self.gateway_headers()
        self.assertEqual(headers["x-oscar-session"], CHAT_ID)
        self.assertEqual(headers["x-oscar-project"], hashlib.sha256(f"project:{CHAT_ID}".encode()).hexdigest()[:16])
        self.assertEqual(headers["x-oscar-turn"], TURN_ID)
        prov.request_context = {"session": CHAT_ID, "turn": "T2"}     # per round: read at every attempt
        self.chat(prov)
        self.assertEqual(self.gateway_headers()["x-oscar-turn"], "T2")

    def test_unsafe_context_values_are_left_out(self):
        for bad in ("a b", "x\r\nInjected: 1", "é", "x" * 200, 42, None, ""):
            with self.subTest(bad=bad):
                self.assertEqual(provider_originshape.context_headers({"session": bad, "turn": bad}), {})
        self.assertEqual(provider_originshape.context_headers("not a mapping"), {})
        self.assertEqual(provider_originshape.context_headers({"turn": TURN_ID}), {"x-oscar-turn": TURN_ID})

    def test_the_token_is_read_on_the_reader_thread(self):
        self.sign_in()
        seen = []
        real = session.AuthSession.access_token

        def spy(auth_session):
            seen.append(threading.current_thread().name)
            return real(auth_session)

        with mock.patch.object(session.AuthSession, "access_token", spy):
            self.assertIsNone(self.chat(self.provider())["error"])
        self.assertEqual(seen, ["oscar-reader-originshape"])     # a renewal never blocks the caller (Stop works)

    def test_stop_during_a_slow_renewal_sends_nothing(self):
        self.sign_in(age=900.0)
        self.fake.reply_delays["/oauth/token"] = 1.0
        started = time.monotonic()
        with self.assertRaises(providers.StopRequested):
            collect(self.provider().chat("SYSTEM", USER, [], lambda: time.monotonic() - started > 0.3))
        self.assertLess(time.monotonic() - started, 2.5)              # the caller left at once (+ the 1 s reader join)
        deadline = time.monotonic() + 5.0
        while self.fake.counts["token:refresh_token"] == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(1.2)                                                # the renewal's answer arrives after the stop
        self.assertEqual(self.fake.gateway_requests, [])

    def test_the_shared_default_context_is_read_only(self):
        with self.assertRaises(TypeError):
            providers.BaseProvider.request_context["session"] = "x"
        prov = self.provider()
        prov.request_context = {"session": CHAT_ID}
        self.assertEqual(dict(providers.BaseProvider.request_context), {})
        self.assertEqual(dict(self.provider().request_context), {})


class ReauthenticationTest(_Case):
    """(c) A 401 renews the token once and resends; it is not an attempt."""

    def test_a_refused_token_is_renewed_and_the_request_sent_again(self):
        tokens = self.sign_in()
        self.fake.reject_token(tokens.access_token)
        state = self.chat(self.provider())
        self.assertIsNone(state["error"])
        self.assertEqual(state["text"], "Hello from the fake OriginShape gateway.")
        self.assertEqual(self.fake.counts["token:refresh_token"], 1)
        first, second = (c.headers["authorization"] for c in self.fake.gateway_requests)
        self.assertEqual(first, f"Bearer {tokens.access_token}")
        self.assertNotEqual(second, first)
        self.assertEqual(second, f"Bearer {session.get_session(self.fake.url).access_token()}")
        self.assertEqual(self.sleeps, [])   # not an attempt: no backoff

    def test_a_second_401_is_final_and_says_signed_out(self):
        self.sign_in()
        self.fake.fail(GATEWAY_PATH, 401, envelope("unauthorized", "Invalid token."), times=2)
        error = self.chat(self.provider())["error"]
        self.assertEqual((error.kind, error.code, error.status, error.retriable), ("auth", "signed_out", 401, False))
        first, second = (c.headers["authorization"] for c in self.fake.calls_to(GATEWAY_PATH))
        self.assertNotEqual(first, second)                      # the resend carried the renewed token
        self.assertEqual(self.fake.counts["token:refresh_token"], 1)

    def test_an_api_key_is_not_renewed(self):
        session.get_session(self.fake.url).use_api_key(self.fake.add_api_key())
        self.fake.fail(GATEWAY_PATH, 401, envelope("unauthorized", "Invalid API key."))
        error = self.chat(self.provider())["error"]
        self.assertEqual((error.kind, error.code), ("auth", "signed_out"))
        self.assertEqual(len(self.fake.calls_to(GATEWAY_PATH)), 1)
        self.assertEqual(self.fake.counts["token:refresh_token"], 0)

    def test_an_expired_token_is_renewed_before_the_request(self):
        tokens = self.sign_in(age=890.0)                     # 10 s left: inside the 60 s margin
        self.assertIsNone(self.chat(self.provider())["error"])
        self.assertEqual(self.fake.counts["token:refresh_token"], 1)
        self.assertNotEqual(self.gateway_headers()["authorization"], f"Bearer {tokens.access_token}")
        self.assertEqual(len(self.fake.gateway_requests), 1)

    def test_not_signed_in_sends_nothing(self):
        error = self.chat(self.provider())["error"]
        self.assertIsInstance(error, ProviderError)
        self.assertEqual((error.kind, error.code, error.retriable), ("auth", "signed_out", False))
        self.assertEqual(self.fake.gateway_requests, [])

    def test_a_grant_the_server_ended_signs_out_and_sends_nothing(self):
        self.sign_in(age=900.0)
        self.fake.fail("/oauth/token", 400, {"error": "invalid_grant", "error_description": "revoked"})
        error = self.chat(self.provider())["error"]
        self.assertEqual((error.kind, error.code), ("auth", "signed_out"))
        self.assertEqual(self.fake.gateway_requests, [])
        self.assertFalse(session.get_session(self.fake.url).has_session())

    def test_an_unreachable_token_endpoint_is_retried_like_a_transport_error(self):
        self.sign_in(age=900.0)
        self.fake.fail("/oauth/token", 503)
        state = self.chat(self.provider())
        self.assertIsNone(state["error"])
        self.assertEqual(len(self.sleeps), 1)
        self.assertEqual(len(self.fake.gateway_requests), 1)
        self.assertTrue(session.get_session(self.fake.url).has_session())


class AccountErrorTest(_Case):
    """(d) 402 / 403 / 429 / 501 and what is retried."""

    def setUp(self):
        super().setUp()
        self.sign_in()

    def final(self, **opts) -> ProviderError:
        error = self.chat(self.provider(), **opts)["error"]
        self.assertIsInstance(error, ProviderError)
        return error

    def test_no_credits(self):
        self.fake.set_account(balance=0)
        error = self.final()
        self.assertEqual((error.status, error.kind, error.code, error.retriable),
                         (402, "account", "insufficient_credits", False))
        self.assertIn("credits", str(error))
        self.assertEqual(error.data["code"], "insufficient_credits")
        self.assertEqual((len(self.fake.gateway_requests), self.sleeps), (1, []))

    def test_waitlisted_and_suspended(self):
        for status in ("waitlisted", "suspended"):
            with self.subTest(status=status):
                self.fake.set_account(status=status)
                error = self.final()
                self.assertEqual((error.status, error.kind, error.code, error.retriable),
                                 (403, "account", status, False))
        self.assertEqual(self.sleeps, [])

    def test_another_403_keeps_its_code(self):
        self.fake.gateway = [gateway_error(403, "forbidden", "This token cannot chat.")]
        error = self.final()
        self.assertEqual((error.kind, error.code), ("account", "forbidden"))
        self.assertIn("This token cannot chat.", str(error))

    def test_a_rate_limit_within_the_budget_is_retried_after_retry_after(self):
        self.fake.gateway = [gateway_error(429, "rate_limited", "Too many requests.", headers={"Retry-After": "2"}),
                             sse_text("after the wait")]
        state = self.chat(self.provider())
        self.assertIsNone(state["error"])
        self.assertEqual(state["text"], "after the wait")
        self.assertEqual(self.sleeps, [(2.0, (2, 5))])

    def test_a_rate_limit_longer_than_the_budget_is_final_at_once(self):
        self.fake.gateway = [gateway_error(429, "rate_limited", "Too many requests.", headers={"Retry-After": "120"})]
        error = self.final()
        self.assertEqual((error.status, error.kind, error.code, error.retriable, error.retry_after),
                         (429, "account", "rate_limited", False, 120.0))
        self.assertIn("120 s", str(error))
        self.assertEqual((len(self.fake.gateway_requests), self.sleeps), (1, []))

    def test_a_rate_limit_without_advice_uses_the_normal_retries(self):
        self.fake.gateway = [gateway_error(429, "rate_limited", "Too many requests.") for _ in range(5)]
        error = self.final()
        self.assertEqual((error.code, error.retry_after), ("rate_limited", None))
        self.assertEqual(len(self.fake.gateway_requests), 5)
        self.assertEqual(len(self.sleeps), 4)

    def test_a_server_without_an_ai_model_is_said_plainly_and_never_retried(self):
        self.fake.llm_configured = False
        error = self.final()
        self.assertEqual(str(error), LLM_NOT_CONFIGURED_MESSAGE)
        self.assertEqual((error.status, error.code, error.retriable), (501, "llm_not_configured", False))
        self.assertEqual((len(self.fake.gateway_requests), self.sleeps), (1, []))

    def test_a_long_chat_with_credits_left_is_not_called_out_of_credits(self):
        """The gateway holds credits for the whole chat before it runs; a 402 with credits left names the chat length
        (audit 2026-10-28: a 50-credit account was told "no credits left" once a chat reached ~0.7 MB)."""
        message = ("This chat is too long for the credits left: it needs about 58.20 credits set aside and 12.40 are "
                   "available. Start a new chat, or add credits.")
        body = json.dumps({"type": "error", "error": {
            "type": "invalid_request_error", "code": "insufficient_credits", "message": message,
            "details": [{"field": "available_credits", "message": "12.40"},
                        {"field": "required_credits", "message": "58.20"}]}})
        self.fake.gateway = [raw_reply(body, status=402, content_type="application/json")]
        error = self.final()
        self.assertEqual((error.status, error.kind, error.code, error.retriable),
                         (402, "account", "chat_too_long_for_credits", False))
        self.assertEqual(str(error), f"{message} (HTTP 402)")
        self.assertEqual(error.data["details"]["required_credits"], "58.20")

    def test_the_daily_limit_says_when_it_resets_and_is_not_retried(self):
        self.fake.gateway = [gateway_error(429, "rate_limited", "You've reached today's usage limit. It resets at "
                                           "midnight UTC.", headers={"Retry-After": "12345"})]
        error = self.final()
        self.assertEqual((error.code, error.retriable, error.retry_after), ("daily_limit", False, 12345.0))
        self.assertIn("today's usage limit", str(error))
        self.assertIn("in about 3 h 26 min", str(error))
        self.assertNotIn("limiting how many requests", str(error))
        self.assertEqual((len(self.fake.gateway_requests), self.sleeps), (1, []))

    def test_one_model_without_a_provider_is_named_not_the_whole_server(self):
        self.fake.gateway = [gateway_error(501, "llm_not_configured", "This model is not available on this server. "
                                           "Pick another model.")]
        error = self.final()
        self.assertEqual((error.code, error.retriable), ("llm_not_configured", False))
        self.assertTrue(str(error).startswith("This model is not available on this server."))

    def test_other_api_errors_show_their_message_not_the_json(self):
        self.fake.gateway = [gateway_error(413, "request_too_large", "The request is larger than 32 MB.")]
        error = self.final()
        self.assertEqual(str(error), "The request is too large for OriginShape (HTTP 413): The request is larger than 32 MB.")
        self.assertEqual(error.code, "request_too_large")

    def test_a_model_the_server_does_not_offer(self):
        error = self.chat(self.provider("oscar-unknown"))["error"]
        self.assertEqual((error.status, error.code, error.retriable), (400, "model_unavailable", False))
        self.assertIn("'oscar-unknown'", str(error))

    def test_server_errors_are_retried(self):
        self.fake.gateway = [gateway_error(502, "upstream_error", "Upstream hiccup."), sse_text("fine")]
        state = self.chat(self.provider())
        self.assertEqual((state["error"], state["text"]), (None, "fine"))
        self.assertEqual(len(self.sleeps), 1)

    def test_upstream_errors_passed_through_are_never_the_account(self):
        """A body in Anthropic's shape (no ``code``) is about the SERVER's key or billing, not the user's."""
        cases = [(401, "authentication_error", "invalid x-api-key"),
                 (402, "billing_error", "Your credit balance is too low to access the API."),
                 (403, "permission_error", "Your API key does not have permission.")]
        for status, etype, message in cases:
            with self.subTest(status=status):
                body = json.dumps({"type": "error", "error": {"type": etype, "message": message}})
                self.fake.gateway = [raw_reply(body, status=status, content_type="application/json")]
                refreshes = self.fake.counts["token:refresh_token"]
                before = len(self.fake.gateway_requests)
                error = self.final()
                self.assertEqual((error.kind, error.code, error.status), ("http", "upstream_refused", status))
                self.assertIn(message, str(error))
                self.assertEqual(len(self.fake.gateway_requests) - before, 1)    # no resend, no renewal
                self.assertEqual(self.fake.counts["token:refresh_token"], refreshes)

    def test_an_unreachable_server_says_so_after_the_retries(self):
        self.fake.close()
        error = self.final()
        self.assertEqual((error.kind, error.code), ("transport", "unreachable"))
        self.assertIn("Cannot reach OriginShape", str(error))
        self.assertIn("request failed after 5 attempts", str(error))
        self.assertEqual(len(self.sleeps), 4)


class NoSecretsTest(_Case):
    """(f) No token, malformed or echoed back, ever appears in an error."""

    def test_a_malformed_token_never_reaches_a_header_or_a_message(self):
        bad = "fake-at-SECRET value\r\nInjected: yes"
        with mock.patch.object(session.AuthSession, "access_token", return_value=bad):
            error = self.chat(self.provider())["error"]
        self.assertIsInstance(error, ProviderError)
        self.assertEqual((error.kind, error.code), ("auth", "signed_out"))
        for part in ("SECRET", "Injected", bad):
            self.assertNotIn(part, str(error))
        self.assertEqual(self.fake.gateway_requests, [])

    def test_a_token_echoed_in_an_error_body_is_scrubbed(self):
        tokens = self.sign_in()
        token = tokens.access_token
        for status, code in ((400, "invalid_request"), (403, "forbidden"), (404, "not_found"), (422, "invalid_request"),
                             (403, ""), (400, "")):
            with self.subTest(status=status, code=code):
                error = {"type": "invalid_request_error", "message": f"Bad header Bearer {token} ({token})"}
                if code:
                    error["code"] = code
                self.fake.gateway = [raw_reply(json.dumps({"type": "error", "error": error}), status=status,
                                               content_type="application/json")]
                final = self.chat(self.provider())["error"]
                self.assertIsInstance(final, ProviderError)
                self.assertNotIn(token, str(final))
                self.assertNotIn(token, json.dumps(final.data))
                self.assertNotIn(token[8:30], str(final))

    def test_a_refresh_token_never_appears_in_an_error(self):
        tokens = self.sign_in(age=900.0)
        self.fake.fail("/oauth/token", 400, {"error": "invalid_request",
                                             "error_description": f"bad refresh_token {tokens.refresh_token}"})
        error = self.chat(self.provider())["error"]
        self.assertIsInstance(error, ProviderError)
        self.assertNotIn(tokens.refresh_token, str(error))
        self.assertEqual(error.kind, "auth")


class OtherProvidersUnchangedTest(unittest.TestCase):
    """The new ChatLoop hooks are no-ops for every provider that does not sign in."""

    def setUp(self):
        self.profile = bare_profile("anthropic", "claude-sonnet-4-5", base_url=f"https://h{uuid.uuid4().hex[:8]}.test")
        forget_learned(self.profile)
        self.addCleanup(forget_learned, self.profile)
        patcher = mock.patch.object(provider_chat.ChatLoop, "_sleep", lambda *a, **k: iter(()))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_401_is_not_resent_and_reads_as_before(self):
        calls = []

        class Reply:
            status_code, text, headers = 401, '{"type":"error","error":{"type":"authentication_error"}}', {}

            def close(self):
                pass

        def post(url, headers=None, json=None, stream=True, timeout=None):
            calls.append(headers)
            return Reply()

        prov = providers.AnthropicProvider(self.profile)
        self.assertFalse(prov._reauthenticate("anything"))
        with mock.patch.object(providers.requests, "post", post):
            error = collect(prov.chat("s", USER, []))["error"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["x-api-key"], "")
        self.assertTrue(str(error).startswith("HTTP 401 unauthorized (check the API key)"))
        self.assertEqual((error.code, error.data, error.retry_after), ("", None, None))

    def test_the_default_http_error_is_the_shared_one(self):
        prov = providers.AnthropicProvider(self.profile)
        error = prov._http_error(429, "slow down", {"Retry-After": "300"})
        self.assertEqual((str(error), error.retriable, error.retry_after), ("HTTP 429 rate limited: slow down", True, None))
        self.assertEqual(prov._upstream_model(), "claude-sonnet-4-5")


if __name__ == "__main__":
    unittest.main()
