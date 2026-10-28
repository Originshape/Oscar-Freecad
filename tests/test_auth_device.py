"""oscar.auth.device: sign-in with a code (RFC 8628) against the fake OriginShape API.

Every token-endpoint state is exercised: authorization_pending, slow_down (+5 s, triggered for real by polling too
fast), access_denied, expired_token, the local deadline, transient failures with back-off, and cancellation (well
within 200 ms). The fake runs protocol time 100x faster (``time_scale``), so a 5 s interval takes 50 ms while the
client still waits "5 s" (the recorded sleeps prove it). Pure python (stdlib + requests).
"""

import os
import sys
import threading
import time
import unittest

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar.auth import device, discovery  # noqa: E402
from oscar.auth.transport import (AuthError, AuthUnavailable, SignInCancelled, SignInDeclined,  # noqa: E402
                                  SignInExpired, TokenSet)
from support.fake_originshape import FakeOriginShape  # noqa: E402

SCALE = 0.01   # real seconds per protocol second, on both sides


class DeviceCase(unittest.TestCase):
    def setUp(self):
        discovery.clear_cache()
        self.addCleanup(discovery.clear_cache)
        self.fake = FakeOriginShape()
        self.addCleanup(self.fake.close)
        # The server's window is 20% shorter than the client's wait: a timer that fires a millisecond early
        # (Windows) must not turn an honest poll into slow_down.
        self.fake.time_scale = SCALE * 0.8
        self.sleeps: list[float] = []

    def scaled_sleep(self, cancelled: threading.Event, seconds: float) -> bool:
        self.sleeps.append(seconds)
        return cancelled.wait(seconds * SCALE)

    def flow(self, **kw) -> device.DeviceSignIn:
        kw.setdefault("sleep", self.scaled_sleep)
        return device.DeviceSignIn(self.fake.url, device_name="Test Box", **kw)


class StartTest(DeviceCase):
    def test_start_returns_what_the_user_needs_and_keeps_the_device_code(self):
        flow = self.flow()
        code = flow.start()
        self.assertRegex(code.user_code, r"^[BCDFGHJKLMNPQRSTVWXZ]{4}-[BCDFGHJKLMNPQRSTVWXZ]{4}$")
        self.assertEqual(code.verification_uri, self.fake.url + "/device")
        self.assertEqual(code.verification_uri_complete, f"{self.fake.url}/device?user_code={code.user_code}")
        self.assertEqual(code.browser_url, code.verification_uri_complete)
        self.assertEqual((code.expires_in, code.interval), (600, 5))
        device_code = self.fake.devices[code.user_code].device_code
        for text in (repr(code), repr(flow), str(code)):
            self.assertNotIn(device_code, text)
        self.assertNotIn("device_code", {f for f in code.__dataclass_fields__})
        form = self.fake.calls_to("/oauth/device_authorization")[0].form
        self.assertEqual(form, {"client_id": "oscar-freecad", "scope": "oscar:chat oscar:account offline_access",
                                "device_name": "Test Box"})

    def test_a_server_without_the_device_flow(self):
        self.fake.metadata_overrides = {"device_authorization_endpoint": None}
        with self.assertRaises(AuthError) as ctx:
            self.flow().start()
        self.assertEqual(ctx.exception.code, "unsupported")

    def test_errors_from_the_device_endpoint(self):
        self.fake.fail("/oauth/device_authorization", 503)
        with self.assertRaises(AuthUnavailable):
            self.flow().start()
        self.fake.fail("/oauth/device_authorization", 401, {"error": "invalid_client"})
        with self.assertRaises(AuthError) as ctx:
            self.flow().start()
        self.assertEqual(ctx.exception.code, "invalid_client")

    def answer(self, **body) -> AuthError | device.DeviceCode:
        payload = {"device_code": "fake-dc-abcdefghijklmnop", "user_code": "BCDF-GHJK",
                   "verification_uri": self.fake.url + "/device", "expires_in": 600, "interval": 5}
        payload.update(body)
        self.fake.fail("/oauth/device_authorization", 200, {k: v for k, v in payload.items() if v is not None})
        try:
            return self.flow().start()
        except AuthError as e:
            return e

    def test_what_the_server_sends_is_validated(self):
        for bad in ({"verification_uri": "http://evil.example.com/device"}, {"verification_uri": None},
                    {"verification_uri": "javascript:alert(1)"}, {"user_code": "<script>"}, {"user_code": None},
                    {"device_code": "has spaces in it"}, {"device_code": None}):
            self.assertIsInstance(self.answer(**bad), AuthError, bad)
        code = self.answer(verification_uri_complete="javascript:alert(1)")
        self.assertEqual(code.verification_uri_complete, "")
        self.assertEqual(code.browser_url, self.fake.url + "/device")
        clamped = self.answer(interval=0, expires_in=10 ** 9, user_code="bcdf-ghjk")
        self.assertEqual((clamped.interval, clamped.expires_in, clamped.user_code), (1, 3600, "BCDF-GHJK"))
        defaults = self.answer(interval="soon", expires_in=None)
        self.assertEqual((defaults.interval, defaults.expires_in), (5, 600))
        odd = self.answer(interval=float("nan"), expires_in=float("inf"))   # json.loads accepts NaN and Infinity
        self.assertEqual((odd.interval, odd.expires_in), (5, 600))


class PollTest(DeviceCase):
    def test_pending_then_approved(self):
        self.fake.device_auto, self.fake.device_auto_after = "approve", 2
        flow = self.flow()
        code = flow.start()
        tokens = flow.poll(threading.Event())
        self.assertIsInstance(tokens, TokenSet)
        self.assertTrue(tokens.refresh_token.startswith("fake-rt-"))
        self.assertEqual(self.fake.counts["token:device_code"], 3)   # pending, pending, approved
        self.assertEqual(self.sleeps, [5, 5, 5], "one request per interval")
        self.assertEqual(self.fake.devices[code.user_code].status, "exchanged")
        self.assertEqual(self.fake.grant_of(tokens.refresh_token).device_name, "Test Box")
        form = self.fake.calls_to("/oauth/token")[0].form
        self.assertEqual(form["grant_type"], "urn:ietf:params:oauth:grant-type:device_code")
        self.assertEqual(form["client_id"], "oscar-freecad")

    def test_approved_by_the_user_while_polling(self):
        flow = self.flow()
        flow.start()
        threading.Timer(0.15, self.fake.approve_device).start()
        self.assertTrue(flow.poll(threading.Event()).access_token)
        self.assertGreaterEqual(self.fake.counts["token:device_code"], 2)

    def test_polling_too_fast_gets_slow_down_and_the_interval_grows_by_five(self):
        self.fake.device_auto, self.fake.device_auto_after = "approve", 0

        def impatient(cancelled, seconds):
            self.sleeps.append(seconds)
            return cancelled.wait(0 if len(self.sleeps) == 1 else seconds * SCALE)   # the first wait is skipped

        flow = self.flow(sleep=impatient)
        code = flow.start()
        self.assertTrue(flow.poll(threading.Event()).access_token)
        self.assertEqual(self.sleeps, [5, 10])
        self.assertEqual(self.fake.devices[code.user_code].interval, 10)

    def test_every_slow_down_adds_five_seconds(self):
        self.fake.device_auto, self.fake.device_auto_after = "approve", 0
        self.fake.fail("/oauth/token", 400, {"error": "slow_down"}, times=2)
        flow = self.flow()
        flow.start()
        self.assertTrue(flow.poll(threading.Event()).access_token)
        self.assertEqual(self.sleeps, [5, 10, 15])

    def test_access_denied(self):
        self.fake.device_auto = "deny"
        flow = self.flow()
        flow.start()
        with self.assertRaises(SignInDeclined):
            flow.poll(threading.Event())

    def test_denied_by_the_user(self):
        flow = self.flow()
        flow.start()
        self.fake.deny_device()
        with self.assertRaises(SignInDeclined):
            flow.poll(threading.Event())

    def test_a_suspended_account_is_told_so(self):
        flow = self.flow()
        flow.start()
        self.fake.fail("/oauth/token", 400, {"error": "access_denied", "error_description": "account_suspended"})
        with self.assertRaises(AuthError) as ctx:
            flow.poll(threading.Event())
        self.assertEqual(ctx.exception.code, "suspended")
        self.assertNotIsInstance(ctx.exception, SignInDeclined)

    def test_expired_token(self):
        flow = self.flow()
        flow.start()
        self.fake.expire_device()
        with self.assertRaises(SignInExpired) as ctx:
            flow.poll(threading.Event())
        self.assertEqual(ctx.exception.code, "expired")
        self.assertEqual(self.fake.counts["token:device_code"], 1)

    def test_the_local_deadline_ends_polling(self):
        self.fake.enforce_interval = False
        self.fake.device_expires_in = 12
        now = [1000.0]

        def fake_sleep(cancelled, seconds):
            self.sleeps.append(seconds)
            now[0] += seconds
            return cancelled.is_set()

        flow = self.flow(sleep=fake_sleep, clock=lambda: now[0])
        flow.start()
        with self.assertRaises(SignInExpired):
            flow.poll(threading.Event())
        self.assertEqual(self.sleeps, [5, 5, 2])
        self.assertEqual(self.fake.counts["token:device_code"], 2, "no request once the code has expired")

    def test_cancel_is_quick(self):
        self.fake.time_scale = 1.0
        flow = device.DeviceSignIn(self.fake.url)   # the real sleep: waits on the event
        flow.start()
        cancelled = threading.Event()
        result: dict = {}

        def run():
            try:
                flow.poll(cancelled)
            except Exception as e:  # noqa: BLE001
                result["error"] = e
            result["at"] = time.monotonic()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        time.sleep(0.1)
        cancelled_at = time.monotonic()
        cancelled.set()
        thread.join(5)
        self.assertIsInstance(result.get("error"), SignInCancelled)
        self.assertLess(result["at"] - cancelled_at, 0.2)
        self.assertEqual(self.fake.counts["token:device_code"], 0)

    def test_cancel_during_the_exchange_revokes_the_new_grant(self):
        self.fake.device_auto, self.fake.device_auto_after = "approve", 0
        self.fake.delays["/oauth/token"] = 0.3
        flow = self.flow()
        flow.start()
        cancelled = threading.Event()

        def cancel_once_polling():
            deadline = time.monotonic() + 5
            while not self.fake.calls_to("/oauth/token") and time.monotonic() < deadline:
                time.sleep(0.01)
            cancelled.set()

        threading.Thread(target=cancel_once_polling, daemon=True).start()
        with self.assertRaises(SignInCancelled):
            flow.poll(cancelled)
        self.assertEqual(self.fake.counts["revoke"], 1)
        self.assertTrue(all(grant.revoked for grant in self.fake.grants.values()))

    def test_transient_failures_back_off_and_recover(self):
        self.fake.device_auto, self.fake.device_auto_after = "approve", 0
        self.fake.fail("/oauth/token", 503, times=1)
        self.fake.fail("/oauth/token", 0, times=1)   # a dropped connection
        flow = self.flow()
        flow.start()
        self.assertTrue(flow.poll(threading.Event()).access_token)
        self.assertEqual(self.sleeps, [5, 10, 20])

    def test_three_failures_in_a_row_are_unavailable(self):
        self.fake.fail("/oauth/token", 0, times=3)
        flow = self.flow()
        flow.start()
        with self.assertRaises(AuthUnavailable):
            flow.poll(threading.Event())
        self.fake.fail("/oauth/token", 502, times=3)
        flow = self.flow()
        flow.start()
        with self.assertRaises(AuthUnavailable) as ctx:
            flow.poll(threading.Event())
        self.assertEqual(ctx.exception.status, 502)

    def test_an_unexpected_error_ends_the_flow(self):
        self.fake.fail("/oauth/token", 400, {"error": "invalid_grant"})
        flow = self.flow()
        flow.start()
        with self.assertRaises(AuthError) as ctx:
            flow.poll(threading.Event())
        self.assertEqual(ctx.exception.code, "invalid_grant")

    def test_poll_needs_start_and_a_code_works_once(self):
        flow = self.flow()
        with self.assertRaises(AuthError) as ctx:
            flow.poll(threading.Event())
        self.assertEqual(ctx.exception.code, "not_started")
        self.fake.device_auto = "approve"
        code = flow.start()
        flow.poll(threading.Event())
        with self.assertRaises(AuthError):
            flow.poll(threading.Event())
        device_code = self.fake.devices[code.user_code].device_code   # the server side refuses a second exchange
        answer = requests.post(self.fake.url + "/oauth/token", timeout=5, data={
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code", "client_id": "oscar-freecad",
            "device_code": device_code})
        self.assertEqual((answer.status_code, answer.json()["error"]), (400, "invalid_grant"))


if __name__ == "__main__":
    unittest.main()
