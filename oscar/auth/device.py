"""Sign in with a code (OAuth 2.0 Device Authorization Grant, RFC 8628), for when the browser cannot reach FreeCAD.

``DeviceSignIn.start()`` asks the server for a code pair and returns the half the user needs (``DeviceCode``: the
short ``user_code`` and the page to type it on). The other half, the device code, is a secret: it stays inside the
object and is only ever sent to the token endpoint. ``poll(cancelled)`` then asks the token endpoint until the user
decides, following RFC 8628:

* ``authorization_pending``: keep waiting, one request per ``interval``.
* ``slow_down``: the interval grows by 5 seconds for the rest of the flow.
* ``access_denied`` raises ``SignInDeclined``; ``expired_token`` (or the local deadline) raises ``SignInExpired``.
* A network failure, 429 or 5xx is retried with a growing delay; ``MAX_TRANSIENT_FAILURES`` in a row raise
  ``AuthUnavailable``.

Both calls block, so they run on a worker thread. Waiting happens on ``cancelled``, so setting it ends ``poll``
at once, except during the one request in flight, which is bounded by ``POLL_TIMEOUT``.
"""

import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from .. import turnlog as _turnlog
from . import config, discovery, transport
from .loopback import clean_device_name, short_hostname
from .transport import (CLIENT_ID, DEVICE_GRANT, SCOPE, AuthError, AuthUnavailable, InvalidServerURL, SignInCancelled,
                        SignInDeclined, SignInExpired, TokenSet)

DEFAULT_INTERVAL_S = 5
SLOW_DOWN_STEP_S = 5
MAX_INTERVAL_S = 60
MAX_TRANSIENT_FAILURES = 3
POLL_TIMEOUT: tuple[float, float] = (5.0, 10.0)
_USER_CODE_RE = re.compile(r"[A-Za-z0-9]{4,8}(?:-[A-Za-z0-9]{4,8}){0,2}")

Sleep = Callable[[threading.Event, float], bool]


def _wait_on(cancelled: threading.Event, seconds: float) -> bool:
    """The default sleep: wait on ``cancelled``; True when it was set (so cancelling needs no polling)."""
    return cancelled.wait(max(0.0, seconds))


@dataclass(frozen=True)
class DeviceCode:
    """What the user needs: type ``user_code`` at ``verification_uri`` (or open ``verification_uri_complete``, which
    carries the code). ``expires_in`` and ``interval`` are in seconds."""

    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int

    @property
    def browser_url(self) -> str:
        """The page to open for the user: the one that already carries the code when the server offers it."""
        return self.verification_uri_complete or self.verification_uri


def _int(value: Any, default: int, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default   # json.loads accepts NaN and Infinity
    return int(min(max(value, low), high))


class DeviceSignIn:
    """One sign-in with a code: ``start()`` then ``poll(cancelled)``; single use."""

    def __init__(self, api_url: str | None = None, *, device_name: str | None = None, sleep: Sleep | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.api_url = config.normalise_url(api_url) if api_url else config.api_url()
        self.device_name = clean_device_name(device_name) if device_name else short_hostname()
        self.code: DeviceCode | None = None
        self._sleep = sleep or _wait_on
        self._clock = clock
        self._device_code = ""     # the secret half: only ever sent to the token endpoint
        self._token_endpoint = ""
        self._revocation_endpoint = ""
        self._interval = DEFAULT_INTERVAL_S
        self._deadline = 0.0
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        user_code = self.code.user_code if self.code else ""
        return f"DeviceSignIn(api_url={self.api_url!r}, user_code={user_code!r})"

    def start(self) -> DeviceCode:
        """Ask the server for a code. Raises ``AuthUnavailable`` (network, 5xx) or ``AuthError``."""
        meta = discovery.fetch(self.api_url)
        if not meta.device_authorization_endpoint:
            raise AuthError("This OriginShape server does not offer sign-in with a code.", code="unsupported")
        reply = transport.request("POST", meta.device_authorization_endpoint,
                                  data={"client_id": CLIENT_ID, "scope": SCOPE, "device_name": self.device_name})
        if reply.status != 200:
            raise transport.token_error(reply)
        data = reply.json()
        if not isinstance(data, dict):
            raise AuthError("OriginShape answered with something Oscar cannot read.", code="bad_response")
        device_code = transport.clean_token(data.get("device_code"), "device code")
        user_code = data.get("user_code")
        if not isinstance(user_code, str) or not _USER_CODE_RE.fullmatch(user_code.strip()):
            raise AuthError("OriginShape sent a sign-in code Oscar cannot show.", code="bad_response")
        verification_uri = self._page_url(data.get("verification_uri"))
        if not verification_uri:
            raise AuthError("OriginShape sent no usable page to enter the code on (it must be https).",
                            code="bad_response")
        complete = self._page_url(data.get("verification_uri_complete"))
        expires_in = _int(data.get("expires_in"), 600, 1, 3600)
        interval = _int(data.get("interval"), DEFAULT_INTERVAL_S, 1, MAX_INTERVAL_S)
        code = DeviceCode(user_code.strip().upper(), verification_uri, complete, expires_in, interval)
        with self._lock:
            self._device_code = device_code
            self._token_endpoint = meta.token_endpoint
            self._revocation_endpoint = meta.revocation_endpoint
            self._interval = interval
            self._deadline = self._clock() + expires_in
            self.code = code
        _log("started")
        return code

    @staticmethod
    def _page_url(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return ""
        try:
            return config.browser_url(value, what="verification page")
        except InvalidServerURL:
            return ""

    def poll(self, cancelled: threading.Event | None = None) -> TokenSet:
        """Wait for the user's decision and return the ``TokenSet``. Raises ``SignInCancelled``,
        ``SignInDeclined``, ``SignInExpired``, ``AuthUnavailable`` or ``AuthError``."""
        cancelled = cancelled if cancelled is not None else threading.Event()
        if not self._lock.acquire(blocking=False):
            raise AuthError("This sign-in is already being completed.", code="busy")
        try:
            outcome = "error"
            try:
                tokens = self._poll(cancelled)
                outcome = "ok"
                return tokens
            except AuthError as e:
                outcome = e.code
                raise
            finally:
                _log(outcome)
        finally:
            self._lock.release()

    def _poll(self, cancelled: threading.Event) -> TokenSet:
        if not self._device_code:
            raise AuthError("Start the sign-in first (or start a new one: each code works once).", code="not_started")
        interval = float(self._interval)
        delay = interval
        failures = 0
        while True:
            remaining = self._deadline - self._clock()
            if remaining <= 0:
                raise SignInExpired("The code expired before it was approved. Start again to get a new one.")
            if self._sleep(cancelled, min(delay, remaining)):
                raise SignInCancelled("Sign-in was cancelled.")
            if self._clock() >= self._deadline:
                raise SignInExpired("The code expired before it was approved. Start again to get a new one.")
            try:
                reply = transport.request("POST", self._token_endpoint, timeout=POLL_TIMEOUT, data={
                    "grant_type": DEVICE_GRANT, "client_id": CLIENT_ID, "device_code": self._device_code})
            except AuthUnavailable:
                reply = None
            if reply is not None and reply.status == 200:
                tokens = transport.token_set(reply, self._device_code)
                self._device_code = ""   # exchanged: the server answers invalid_grant from now on
                if cancelled.is_set():   # the user gave up during the exchange: leave no connected device
                    transport.revoke_quietly(self._revocation_endpoint, tokens.refresh_token)
                    raise SignInCancelled("Sign-in was cancelled.")
                return tokens
            if cancelled.is_set():
                raise SignInCancelled("Sign-in was cancelled.")
            if reply is None or transport.is_unavailable(reply.status):
                failures += 1
                if failures >= MAX_TRANSIENT_FAILURES:
                    if reply is None:
                        raise AuthUnavailable("Could not reach the OriginShape server to finish signing in. Check "
                                              "your connection and try again.")
                    raise transport.unavailable(reply)
                delay = min(interval * (2 ** failures), MAX_INTERVAL_S)   # RFC 8628 3.5: back off on errors
                continue
            failures = 0
            code, _ = transport.oauth_error(reply, self._device_code)
            if code == "authorization_pending":
                delay = interval
                continue
            if code == "slow_down":
                interval = min(interval + SLOW_DOWN_STEP_S, MAX_INTERVAL_S)
                delay = interval
                continue
            if code == "access_denied":
                raise SignInDeclined("The sign-in request was declined.", status=reply.status)
            if code == "expired_token":
                raise SignInExpired("The code expired before it was approved. Start again to get a new one.",
                                    status=reply.status)
            raise transport.token_error(reply, self._device_code)


def _log(outcome: str) -> None:
    _turnlog.log_event("auth", event="sign_in", method="code", outcome=outcome)
