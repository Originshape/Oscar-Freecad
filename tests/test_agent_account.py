"""AgentWorker and the OriginShape account, over the REAL OriginShapeProvider against the fake OriginShape API.

Covered: a turn that fails for an account reason (no credits, waitlisted, signed out) emits ``account_problem`` (JSON)
just BEFORE ``turn_outcome`` / ``turn_finished``, with ``first_request`` telling whether anything reached the model;
the ``"Provider error: ..."`` text stays as it was; a 501 (the server has no AI model) is a plain error, not an account
problem; ``session_id`` and the turn id reach the provider as a NEW ``request_context`` per turn and per round, so the
``x-oscar-*`` headers of every request name the chat and the turn. Needs FreeCAD (``oscar.agent`` imports it).
"""

import hashlib
import json
import os
import sys
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

try:
    from oscar import agent, params, provider_chat, providers  # noqa: E402
    from oscar.auth import account, discovery, keystore, session  # noqa: E402
    from oscar.auth.transport import LLM_NOT_CONFIGURED_MESSAGE, TokenSet  # noqa: E402
    from oscar.flow import AgentConfig  # noqa: E402
    from oscar.provider_transport import ProviderError  # noqa: E402
    from oscar.providers import Message  # noqa: E402
    from support.fake_originshape import FakeOriginShape, gateway_error, sse_text  # noqa: E402
    from support.scripted_llm import call, calls  # noqa: E402
    from support.signal_recorder import SignalRecorder  # noqa: E402
    from test_agent_loop import FakeTools  # noqa: E402
    _IMPORT_ERROR = None
except ImportError as e:  # FreeCAD / PySide missing: the whole module skips
    _IMPORT_ERROR = e

CHAT_ID = "20261007-101010-a1b2c3"


class _AccountTurnCase(unittest.TestCase):
    """A fake OriginShape API that ``OSCAR_API_URL`` points at, a memory keystore, and no real retry sleeps."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"FreeCAD / Qt unavailable: {_IMPORT_ERROR}")
        for reset in (discovery.clear_cache, session.reset_sessions, keystore.reset_memory, account.clear_model_cache):
            reset()
            self.addCleanup(reset)
        self.addCleanup(setattr, session, "on_change", session.on_change)
        session.on_change = None
        self.fake = FakeOriginShape()
        self.addCleanup(self.fake.close)
        env = mock.patch.dict(os.environ, {"OSCAR_API_URL": self.fake.url, keystore.ENV_STORE: "memory"})
        env.start()
        self.addCleanup(env.stop)

        def no_sleep(loop, delay, call_state, should_stop, retry):
            return iter(())

        patcher = mock.patch.object(provider_chat.ChatLoop, "_sleep", no_sleep)
        patcher.start()
        self.addCleanup(patcher.stop)
        providers._LEARNED.clear()
        self.addCleanup(providers._LEARNED.clear)

    def sign_in(self) -> None:
        data = self.fake.issue_tokens()
        session.get_session(self.fake.url).complete_sign_in(
            TokenSet(data["access_token"], data["refresh_token"], data["expires_in"], data["scope"]))

    def profile(self):
        return params.account_profile(f"OriginShape {uuid.uuid4().hex[:6]}")

    def worker(self, *, provider=None, **kw):
        profile = self.profile()
        history = [Message(role="user", content="Make a 20 mm cube.")]
        # flow=False: no plan, gate or baseline snapshot; what is tested here does not depend on them
        worker = agent.AgentWorker(history, provider=provider or providers.make_provider(profile), profile=profile,
                                   config=AgentConfig(), validator=False, flow=False, **kw)
        self.events = []
        # the recorder knows the long-standing signals; account_problem is recorded into the same ordered list
        self.rec = SignalRecorder(worker)
        self.events = self.rec.events
        worker.account_problem.connect(lambda payload: self.events.append(("account_problem", (payload,))))
        self.history = history
        return worker

    def run_turn(self, **kw):
        worker = self.worker(**kw)
        tools = FakeTools()
        with mock.patch.object(agent._tools, "run_tool", lambda n, a: tools(n, a)):
            result = worker.run_turn()
        self.worker_ = worker
        return result

    def problem(self) -> dict:
        payloads = [json.loads(args[0]) for name, args in self.events if name == "account_problem"]
        self.assertEqual(len(payloads), 1, self.rec.names())
        return payloads[0]

    def assert_problem_comes_last_but_two(self) -> None:
        names = [name for name, _args in self.events]
        self.assertEqual(names[-3:], ["account_problem", "turn_outcome", "turn_finished"], names)


class AccountProblemTest(_AccountTurnCase):
    def test_no_credits_emits_account_problem_just_before_turn_outcome(self):
        self.sign_in()
        self.fake.set_account(balance=0)
        result = self.run_turn()
        self.assertFalse(result.ok)
        self.assertTrue(result.text.startswith("Provider error: "), result.text)    # the pinned text format
        self.assert_problem_comes_last_but_two()
        problem = self.problem()
        self.assertEqual((problem["code"], problem["kind"], problem["status"]),
                         ("insufficient_credits", "account", 402))
        self.assertTrue(problem["first_request"])
        self.assertEqual(problem["data"].get("code"), "insufficient_credits")
        self.assertIn("credits", problem["message"])

    def test_a_waitlisted_account(self):
        self.sign_in()
        self.fake.set_account(status="waitlisted")
        self.run_turn()
        self.assert_problem_comes_last_but_two()
        problem = self.problem()
        self.assertEqual((problem["code"], problem["kind"], problem["status"]), ("waitlisted", "account", 403))

    def test_signed_out_fails_before_anything_is_sent(self):
        result = self.run_turn()
        self.assertFalse(result.ok)
        self.assertEqual(self.fake.gateway_requests, [])
        problem = self.problem()
        self.assertEqual((problem["code"], problem["kind"]), ("signed_out", "auth"))
        self.assertTrue(problem["first_request"])

    def test_a_failure_after_a_tool_round_is_not_the_first_request(self):
        self.sign_in()
        self.fake.gateway = [calls(call("create_object", doc_name="D", obj_type="Part::Box")),
                             gateway_error(402, "insufficient_credits", "You have no credits left.")]
        self.run_turn()
        self.assertEqual(len(self.fake.gateway_requests), 2)
        problem = self.problem()
        self.assertEqual(problem["code"], "insufficient_credits")
        self.assertFalse(problem["first_request"])

    def test_a_server_without_an_ai_model_is_a_plain_error_never_retried(self):
        self.sign_in()
        self.fake.llm_configured = False
        result = self.run_turn()
        self.assertFalse(result.ok)
        self.assertIn(LLM_NOT_CONFIGURED_MESSAGE, result.text)
        self.assertEqual([n for n, _a in self.events if n == "account_problem"], [])
        self.assertEqual(len(self.fake.gateway_requests), 1)
        self.assertEqual(self.rec.names()[-2:], ["turn_outcome", "turn_finished"])

    def test_a_provider_setup_failure_of_the_account_is_reported_too(self):
        worker = self.worker(provider=None)
        worker.provider = None
        worker._provider_injected = False
        failure = ProviderError("You are not signed in to OriginShape.", kind="auth", code="signed_out")
        with mock.patch.object(agent, "make_provider", side_effect=failure):
            worker.run()
        self.assert_problem_comes_last_but_two()
        self.assertEqual(self.problem()["code"], "signed_out")

    def test_other_provider_errors_emit_nothing(self):
        worker = self.worker(provider=None)
        worker.provider = None
        worker._provider_injected = False
        with mock.patch.object(agent, "make_provider", side_effect=ProviderError("HTTP 500 server error", kind="http")):
            worker.run()
        self.assertEqual([n for n, _a in self.events if n == "account_problem"], [])
        self.assertEqual(self.rec.names()[-2:], ["turn_outcome", "turn_finished"])


class RequestContextTest(_AccountTurnCase):
    def test_every_request_names_the_chat_and_the_turn(self):
        self.sign_in()
        self.fake.gateway = [calls(call("create_object", doc_name="D", obj_type="Part::Box")), sse_text("Done.")]
        result = self.run_turn(session_id=CHAT_ID)
        self.assertTrue(result.ok, result.text)
        requests_ = self.fake.gateway_requests
        self.assertGreaterEqual(len(requests_), 2)
        turn_id = self.worker_._turn_id
        self.assertTrue(turn_id)
        project = hashlib.sha256(("project:" + CHAT_ID).encode()).hexdigest()[:16]
        for request in requests_:
            self.assertEqual(request.headers.get("x-oscar-session"), CHAT_ID)
            self.assertEqual(request.headers.get("x-oscar-project"), project)
            self.assertEqual(request.headers.get("x-oscar-turn"), turn_id)

    def test_the_context_is_a_new_dict_per_round(self):
        self.sign_in()
        self.fake.gateway = [calls(call("create_object", doc_name="D", obj_type="Part::Box")), sse_text("Done.")]
        profile = self.profile()
        provider = providers.make_provider(profile)
        seen = []
        original = provider._headers

        def spy():
            seen.append(provider.request_context)
            return original()

        provider._headers = spy
        self.run_turn(provider=provider, session_id=CHAT_ID)
        rounds = [c.get("round") for c in seen]
        self.assertEqual(rounds[:2], [1, 2])
        self.assertEqual(rounds, sorted(rounds))
        self.assertIsNot(seen[0], seen[1])                               # replaced, never edited in place
        self.assertEqual({c["session"] for c in seen}, {CHAT_ID})
        self.assertEqual(len({c["turn"] for c in seen}), 1)

    def test_without_a_session_id_nothing_names_the_chat(self):
        self.sign_in()
        self.run_turn()
        request = self.fake.gateway_requests[0]
        self.assertNotIn("x-oscar-session", request.headers)
        self.assertNotIn("x-oscar-project", request.headers)
        self.assertTrue(request.headers.get("x-oscar-turn", "").startswith("T"))

    def test_other_providers_simply_carry_the_attribute(self):
        class Plain:
            def chat(self, *a, **k):
                return iter(())

        provider = Plain()
        worker = self.worker(provider=provider, session_id=CHAT_ID)
        worker._turn_id = "T1"
        worker._set_request_context(round=3)
        self.assertEqual(provider.request_context, {"session": CHAT_ID, "turn": "T1", "round": 3})

        class Refusing:
            __slots__ = ()

        worker.provider = Refusing()
        worker._set_request_context()        # a provider that refuses the attribute is fine


if __name__ == "__main__":
    unittest.main()
