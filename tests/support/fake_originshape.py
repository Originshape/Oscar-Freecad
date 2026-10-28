"""tests/support/fake_originshape.py - FakeOriginShape: a router-style fake of the OriginShape API on 127.0.0.1.

A helper module with no test case on purpose. It speaks the OriginShape API with REAL checks: PKCE S256, exact
redirect URIs, refresh-token rotation with reuse detection, and the RFC 8628 device state machine with interval
enforcement. So the plugin's auth package and the OriginShape provider are tested against the protocol, not against
mocks of themselves.
It imports nothing from ``oscar`` (``support.server`` is imported lazily, only to render a scripted ``Round``).

Usage::

    from support.fake_originshape import FakeOriginShape, follow_authorize, gateway_error, message_json, sse_text

    with FakeOriginShape() as fake:                  # binds 127.0.0.1:<free port>, serves on daemon threads
        os.environ["OSCAR_API_URL"] = fake.url       # fake.url is the API URL and the OAuth issuer
        grant = fake.issue_tokens()                  # a signed-in grant without running a flow (token JSON dict)
        key = fake.add_api_key()                     # a valid personal API key (os_live_...)
        ...
        fake.calls                                   # every request in order (RecordedCall)
        fake.counts["token:refresh_token"]           # per-route counters (COUNTER KEYS below)

ROUTES
  GET  /.well-known/oauth-authorization-server   the RFC 8414 metadata; ``metadata_overrides`` is merged over it
  GET  /.well-known/jwks.json                     {"keys": []}
  GET  /oauth/authorize                           a user who decides at once: 302 to ``redirect_uri?code=..&state=..``
                                                  (or ``?error=access_denied&state=..``). Unknown client or a redirect
                                                  URI that is not http://127.0.0.1|[::1]:<port>/callback: a 400 page,
                                                  never a redirect. Bad PKCE / response_type / scope:
                                                  error=invalid_request.
  POST /oauth/token                               form-encoded (JSON accepted):
      authorization_code  unused, unexpired (``code_ttl``), same client, exact redirect_uri,
                          BASE64URL(SHA256(code_verifier)) == code_challenge. A code used twice revokes its grant.
      refresh_token       rotates (old one marked used). Reusing a used token revokes the whole grant -> invalid_grant.
      device_code         authorization_pending | slow_down (polled faster than the interval: interval += 5) |
                          access_denied | expired_token | tokens once, then invalid_grant.
  POST /oauth/device_authorization                {device_code, user_code (XXXX-XXXX), verification_uri[_complete],
                                                  expires_in, interval}
  POST /oauth/revoke                              revokes the grant of a refresh token; always 200
  GET  /v1/me                                     ``me`` (the /v1/me shape) for a valid bearer, else 401 envelope
  GET  /v1/models                                 ``models`` (the /v1/models shape) for a valid bearer, else 401 envelope
  GET  /v1/plugin                                 ``plugin`` (latest / min version, download link): public, no bearer
  POST /v1/llm/anthropic/v1/messages              the gateway (below): the checks, then the scripted replies
  anything else                                   404 with the API's error envelope

A valid bearer is an issued, unexpired access token whose grant is not revoked, or a key in ``api_keys`` (as
``Authorization: Bearer`` or ``x-api-key``).

SCRIPTING (plain attributes; change them at any time)
  authorize_decision = "approve" | "deny"     what the simulated user answers on /oauth/authorize
  authorize_state = None | str                replace the state in that redirect (a forged or stale callback)
  access_ttl = 900                            expires_in of new access tokens (seconds, also enforced)
  code_ttl = 60                               authorization codes expire after this (seconds)
  device_interval = 5, device_expires_in = 600   the device authorization answer
  time_scale = 1.0                            real seconds per protocol second for the device interval and expiry
                                              (0.01 runs a 5 s interval in 50 ms; the answer still says 5)
  enforce_interval = True                     answer slow_down when polled faster than the interval
  device_auto = None | "approve" | "deny"     decide automatically after ``device_auto_after`` pending polls
  me = {...}                                  the /v1/me body; ``set_account(status=, balance=, available=, low=,
                                              email=, plan=, plan_name=)`` edits the usual fields (the gateway
                                              obeys them, see GATEWAY)
  models = {...}                              the /v1/models body
  metadata_overrides = {}                     merged over the discovery document (a value of None deletes the key)
  delays = {"/oauth/token": 0.3}              sleep BEFORE handling a path (to overlap concurrent requests)
  reply_delays = {"/oauth/token": 0.3}        sleep AFTER handling, before answering (the server acted; the
                                              client is still waiting)
  fail(path, status, body=None, *, method=None, headers=None, times=1)
                                              answer the next ``times`` matching requests with this status, before
                                              any route logic (the gateway's checks included); status 0 drops the
                                              connection without an answer (a network failure). ``body=None`` is
                                              an RFC 6749 error under /oauth, else an ``upstream_error`` envelope
  reject_token(token)                         the next bearer check of this access token fails with 401 (once)
  approve_device(user_code=None) / deny_device(...) / expire_device(...)   decide a pending device code (None =
                                              the latest one)

GATEWAY  (POST /v1/llm/anthropic/v1/messages)
  This fake's own checks (the answers the client must handle), each with the API's error envelope; the first that
  fails answers:
    1. bearer (above)                         401 unauthorized
    2. ``me`` user status (``enforce_account``)   403 waitlisted (action_url = links.waitlist) | 403 suspended
    3. ``llm_configured = False``             501 llm_not_configured ("The OriginShape server has no AI model
                                              configured."): a server without an AI model. Never retried.
    4. a JSON body whose ``model`` is an id in ``models`` (``enforce_models``)   400 invalid_request |
                                              400 model_unavailable
    5. ``me`` credits available > 0 (``enforce_account``)   402 insufficient_credits (action_url = links.billing)
  enforce_account = True, enforce_models = True, llm_configured = True   switch those checks
  gateway = []                                replies served in order once the checks pass; when empty, a short
                                              answer: ``sse_text()`` for ``"stream": true``, else ``message_json()``
                                              (the ``model`` in it is the alias the request named)
    items: ``sse_text(text, usage=...)``; ``message_json(text, usage=...)``; ``gateway_error(status, code,
    message, *, error_type=, action_url=, headers=)`` (the API's error envelope, e.g. 429 rate_limited with
    ``headers={"Retry-After": "2"}``); ``raw_reply(body, status=, content_type=, headers=)``; a ``str`` (=
    ``sse_text(str)``); a ``support.scripted_llm.Round`` (rendered by ``support.server.render_anthropic``)
  gateway_requests                            RecordedCall list (path, lower-case headers, json), refused ones too
  gateway_url                                 fake.url + "/v1/llm/anthropic/v1/messages"

STATE you can assert on
  calls, calls_to(path), gateway_requests, counts, grants (id -> Grant), revoked_tokens, api_keys,
  devices (user_code -> Device), refresh_state(token) -> "active" | "used" | "revoked" | "unknown",
  grant_of(refresh_token) -> Grant | None

COUNTER KEYS
  "discovery", "jwks", "authorize", "token:authorization_code", "token:refresh_token", "token:device_code",
  "token:other", "device_authorization", "revoke", "me", "models", "gateway", "unauthorized", "not_found", "fault"
"""

import base64
import collections
import copy
import hashlib
import json
import re
import secrets
import socketserver
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

import requests

CLIENT_ID = "oscar-freecad"
SCOPE = "oscar:chat oscar:account offline_access"
SCOPES = frozenset(SCOPE.split())
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
GATEWAY_PATH = "/v1/llm/anthropic/v1/messages"
USER_CODE_ALPHABET = "BCDFGHJKLMNPQRSTVWXZ"
_REDIRECT_RE = re.compile(r"http://(?:127\.0\.0\.1|\[::1\]):\d{1,5}/callback")
_CHALLENGE_RE = re.compile(r"[A-Za-z0-9_-]{43}")
_VERIFIER_RE = re.compile(r"[A-Za-z0-9\-._~]{43,128}")
_ERROR_TYPES = {
    "unauthorized": "authentication_error", "forbidden": "permission_error", "waitlisted": "permission_error",
    "suspended": "permission_error", "insufficient_credits": "billing_error", "rate_limited": "rate_limit_error",
    "not_found": "not_found_error", "invalid_request": "invalid_request_error",
    "model_unavailable": "invalid_request_error", "conflict": "invalid_request_error", "upstream_error": "api_error",
    "llm_not_configured": "api_error",
}
_REASONS = {200: "OK", 302: "Found", 400: "Bad Request", 401: "Unauthorized", 402: "Payment Required",
            403: "Forbidden", 404: "Not Found", 409: "Conflict", 429: "Too Many Requests",
            500: "Internal Server Error", 501: "Not Implemented", 502: "Bad Gateway", 503: "Service Unavailable",
            529: "Overloaded"}
LLM_NOT_CONFIGURED_MESSAGE = "The OriginShape server has no AI model configured."


# -- records and state ------------------------------------------------------------------------------------

@dataclass
class RecordedCall:
    """One request the fake received."""

    n: int
    method: str
    path: str          # without the query
    query: dict
    headers: dict      # lower-case names
    form: dict         # the parsed form body ({} for JSON or empty bodies)
    json: Any          # the parsed JSON body (None for form or empty bodies)
    raw: bytes


@dataclass
class Grant:
    """A "connected device": what a successful sign-in creates."""

    id: str
    client_id: str
    device_name: str
    scope: str
    revoked: bool = False


@dataclass
class Device:
    """One device authorization (RFC 8628)."""

    device_code: str
    user_code: str
    scope: str
    device_name: str
    interval: int
    issued_at: float
    expires_at: float
    last_poll: float
    status: str = "pending"   # pending | approved | denied | exchanged
    polls: int = 0


@dataclass
class _Code:
    code: str
    client_id: str
    redirect_uri: str
    challenge: str
    scope: str
    device_name: str
    expires_at: float
    used: bool = False
    grant_id: str = ""


@dataclass
class _Refresh:
    grant_id: str
    used: bool = False


@dataclass
class _Fault:
    path: str
    status: int
    body: bytes
    headers: dict
    method: str | None
    times: int


# -- gateway replies --------------------------------------------------------------------------------------

@dataclass
class GatewayReply:
    """A pre-built gateway answer."""

    status: int = 200
    body: bytes = b""
    content_type: str = "text/event-stream"
    headers: dict = field(default_factory=dict)


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def sse_text(text: str = "Hello from the fake OriginShape gateway.", *, usage: dict | None = None,
             model: str = "oscar-default", stop_reason: str = "end_turn") -> GatewayReply:
    """An Anthropic Messages stream that answers ``text`` (``usage`` feeds message_start / message_delta)."""
    usage = dict(usage or {"input_tokens": 12, "output_tokens": 7})
    start_usage = {k: v for k, v in usage.items() if k != "output_tokens"}
    start_usage.setdefault("output_tokens", 1)
    body = b"".join([
        _sse("message_start", {"type": "message_start", "message": {
            "id": "msg_" + secrets.token_hex(6), "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None, "usage": start_usage}}),
        _sse("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
        _sse("ping", {"type": "ping"}),
        _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": text}}),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                               "usage": {"output_tokens": usage.get("output_tokens", 1)}}),
        _sse("message_stop", {"type": "message_stop"}),
    ])
    return GatewayReply(200, body)


def message_json(text: str = "Hello from the fake OriginShape gateway.", *, usage: dict | None = None,
                 model: str = "oscar-default", stop_reason: str = "end_turn") -> GatewayReply:
    """A non-streaming Anthropic Messages answer (``application/json``)."""
    body = {"id": "msg_" + secrets.token_hex(6), "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}], "stop_reason": stop_reason, "stop_sequence": None,
            "usage": dict(usage or {"input_tokens": 12, "output_tokens": 7})}
    return GatewayReply(200, json.dumps(body).encode("utf-8"), "application/json")


def envelope(code: str, message: str = "", *, error_type: str | None = None, action_url: str | None = None) -> dict:
    """The API's error body."""
    error = {"type": error_type or _ERROR_TYPES.get(code, "api_error"), "code": code, "message": message or code}
    if action_url:
        error["action_url"] = action_url
    return {"type": "error", "error": error}


def gateway_error(status: int, code: str, message: str = "", *, error_type: str | None = None,
                  action_url: str | None = None, headers: dict | None = None) -> GatewayReply:
    """An HTTP error with the spec's error envelope, e.g. ``gateway_error(402, "insufficient_credits", "...")``."""
    body = json.dumps(envelope(code, message, error_type=error_type, action_url=action_url)).encode("utf-8")
    return GatewayReply(status, body, "application/json", dict(headers or {}))


def raw_reply(body: bytes | str, *, status: int = 200, content_type: str = "text/event-stream",
              headers: dict | None = None) -> GatewayReply:
    """Any body, sent as it is (e.g. an upstream error passed through verbatim)."""
    data = body.encode("utf-8") if isinstance(body, str) else bytes(body)
    return GatewayReply(status, data, content_type, dict(headers or {}))


def follow_authorize(authorize_url: str, *, timeout: float = 10.0) -> requests.Response:
    """Play the browser: GET the authorize URL, follow its ONE redirect (to the loopback callback) and return the
    callback's response (the plugin's page). Raises AssertionError when the authorize step does not redirect."""
    first = requests.get(authorize_url, allow_redirects=False, timeout=timeout)
    location = first.headers.get("Location")
    if first.status_code not in (301, 302, 303, 307, 308) or not location:
        raise AssertionError(f"the authorize step answered HTTP {first.status_code} without a redirect")
    return requests.get(location, allow_redirects=False, timeout=timeout)


def default_me(app: str) -> dict:
    """A /v1/me body (an approved trial account with 42.5 credits)."""
    return {
        "user": {"id": "user-1", "email": "ada@example.com", "name": "Ada", "avatar_url": None, "status": "approved",
                 "is_admin": False, "created_at": "2026-10-01T00:00:00Z"},
        "org": {"id": "org-1", "plan": "trial", "plan_name": "Trial", "subscription_status": None,
                "current_period_end": None},
        "credits": {"balance": 42.5, "available": 42.5, "low": False},
        "links": {"app": app, "billing": app + "/billing", "manage": app + "/settings", "waitlist": app + "/waitlist",
                  "download": app + "/download"},
        "plugin": {"latest_version": "0.1.0", "min_version": "0.1.0"},
    }


# Aliases the real gateway still serves but never lists in /v1/models (gateway/models_catalog.py): the retired
# spellings. The fake's gateway accepts them too, so tests with an old profile keep working the way the server does.
HIDDEN_MODELS = ("oscar-default", "oscar-fast")


def default_models() -> dict:
    """A /v1/models body as the real server sends it (gateway/models_catalog.py): the lineup only, ``origin1`` as the
    default, a 1M context window and 64000 output tokens (audit 2026-10-28: this fake listed the retired aliases, so
    tests could not see what a retired name gets)."""
    caps = {"vision": True, "tools": True, "reasoning": "adaptive", "prompt_caching": True}
    names = {"origin1": "Origin1", "origin1-super": "Origin1 Super"}
    return {"default": "origin1", "data": [
        {"id": alias, "display_name": names[alias], "context_window": 1000000, "max_output_tokens": 64000,
         "capabilities": dict(caps)} for alias in names]}


# -- the server -------------------------------------------------------------------------------------------

class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self) -> None:   # TCPServer's bind: HTTPServer's would do a reverse DNS lookup
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = host, port


class FakeOriginShape:
    """The fake API. See the module docstring for routes, scripting and state."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 0, start: bool = True):
        self.lock = threading.RLock()
        self.clock = time.monotonic
        self._host, self._port = host, port
        self._httpd: _Server | None = None
        self._thread: threading.Thread | None = None
        # scripting
        self.authorize_decision = "approve"
        self.authorize_state: str | None = None
        self.access_ttl = 900
        self.code_ttl = 60
        self.device_interval = 5
        self.device_expires_in = 600
        self.time_scale = 1.0
        self.enforce_interval = True
        self.device_auto: str | None = None
        self.device_auto_after = 0
        self.llm_configured = True
        self.enforce_account = True
        self.enforce_models = True
        self.metadata_overrides: dict = {}
        self.delays: dict[str, float] = {}
        self.reply_delays: dict[str, float] = {}
        self.gateway: list = []
        self.me: dict = {}
        self.models: dict = default_models()
        self.plugin: dict = {"latest_version": "0.1.0", "min_version": "0.1.0", "download_url": ""}
        # state
        self.calls: list[RecordedCall] = []
        self.gateway_requests: list[RecordedCall] = []
        self.counts: collections.Counter = collections.Counter()
        self.grants: dict[str, Grant] = {}
        self.devices: dict[str, Device] = {}
        self.revoked_tokens: list[str] = []
        self.api_keys: set[str] = set()
        self._codes: dict[str, _Code] = {}
        self._refresh: dict[str, _Refresh] = {}
        self._access: dict[str, tuple[str, float]] = {}
        self._faults: list[_Fault] = []
        self._rejects: set[str] = set()
        if start:
            self.start()

    # lifecycle ------------------------------------------------------------------------------------------

    def start(self) -> "FakeOriginShape":
        """Bind and serve on a daemon thread (idempotent)."""
        if self._httpd is not None:
            return self
        owner = self

        class Handler(_Handler):
            fake = owner

        httpd = _Server((self._host, self._port), Handler)
        self._port = httpd.server_address[1]
        self._httpd = httpd
        if not self.me:
            self.me = default_me(self.url)
        self._thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True,
                                        name="fake-originshape")
        self._thread.start()
        return self

    def close(self) -> None:
        """Stop serving and close the socket (idempotent). ``url`` stays valid, and now refuses connections."""
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            try:
                httpd.shutdown()
            finally:
                httpd.server_close()
        if self._thread is not None:
            self._thread.join(2.0)
            self._thread = None

    shutdown = close

    def __enter__(self) -> "FakeOriginShape":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        """``http://127.0.0.1:<port>``: the API URL and the OAuth issuer."""
        return f"http://{self._host}:{self._port}"

    @property
    def gateway_url(self) -> str:
        return self.url + GATEWAY_PATH

    # helpers for tests ----------------------------------------------------------------------------------

    def issue_tokens(self, *, device_name: str = "test", scope: str = SCOPE) -> dict:
        """A new grant and its token JSON (what /oauth/token answers), without running a flow."""
        with self.lock:
            return self._tokens(self._new_grant(device_name, scope))

    def add_api_key(self, key: str | None = None) -> str:
        """Register (and return) a personal API key the fake accepts as a bearer."""
        key = key or "os_live_" + secrets.token_urlsafe(32)
        with self.lock:
            self.api_keys.add(key)
        return key

    def set_account(self, *, status: str | None = None, email: str | None = None, balance: float | None = None,
                    available: float | None = None, low: bool | None = None, plan: str | None = None,
                    plan_name: str | None = None) -> None:
        """Edit the usual /v1/me fields (``available`` follows ``balance`` unless given)."""
        with self.lock:
            if status is not None:
                self.me["user"]["status"] = status
            if email is not None:
                self.me["user"]["email"] = email
            if balance is not None:
                self.me["credits"]["balance"] = balance
                self.me["credits"]["available"] = balance if available is None else available
            elif available is not None:
                self.me["credits"]["available"] = available
            if low is not None:
                self.me["credits"]["low"] = low
            if plan is not None:
                self.me["org"]["plan"] = plan
            if plan_name is not None:
                self.me["org"]["plan_name"] = plan_name

    def fail(self, path: str, status: int, body: Any = None, *, method: str | None = None,
             headers: dict | None = None, times: int = 1) -> None:
        """Answer the next ``times`` requests to ``path`` (and ``method``, when given) with ``status`` and ``body``
        (dict -> JSON, str/bytes as is, None -> an error body in the path's style). Status 0 drops the connection."""
        if body is None:
            body = ({"error": "temporarily_unavailable", "error_description": "scripted failure"}
                    if path.startswith("/oauth") else envelope("upstream_error", "scripted failure"))
        data = (json.dumps(body).encode("utf-8") if isinstance(body, (dict, list))
                else body.encode("utf-8") if isinstance(body, str) else bytes(body))
        with self.lock:
            self._faults.append(_Fault(path, status, data, dict(headers or {}), method, times))

    def reject_token(self, token: str) -> None:
        """Make the next bearer check of ``token`` fail with 401 (a server-side invalidation the client must
        recover from by refreshing)."""
        with self.lock:
            self._rejects.add(token)

    def approve_device(self, user_code: str | None = None) -> None:
        self._decide(user_code, "approved")

    def deny_device(self, user_code: str | None = None) -> None:
        self._decide(user_code, "denied")

    def expire_device(self, user_code: str | None = None) -> None:
        with self.lock:
            device = self._device(user_code)
            device.expires_at = self.clock() - 1.0

    def refresh_state(self, token: str) -> str:
        """``"active"``, ``"used"`` (rotated away) or ``"unknown"``; a token of a revoked grant is ``"revoked"``."""
        with self.lock:
            entry = self._refresh.get(token)
            if entry is None:
                return "unknown"
            if self.grants[entry.grant_id].revoked:
                return "revoked"
            return "used" if entry.used else "active"

    def grant_of(self, refresh_token: str) -> Grant | None:
        with self.lock:
            entry = self._refresh.get(refresh_token)
            return self.grants.get(entry.grant_id) if entry is not None else None

    def calls_to(self, path: str) -> list[RecordedCall]:
        with self.lock:
            return [c for c in self.calls if c.path == path]

    # internals (called with ``lock`` held) --------------------------------------------------------------

    def _device(self, user_code: str | None) -> Device:
        if user_code is None:
            if not self.devices:
                raise AssertionError("no device authorization was requested")
            return list(self.devices.values())[-1]
        return self.devices[user_code]

    def _decide(self, user_code: str | None, status: str) -> None:
        with self.lock:
            device = self._device(user_code)
            if device.status == "pending":
                device.status = status

    def _new_grant(self, device_name: str, scope: str) -> Grant:
        grant = Grant("grant-" + secrets.token_hex(6), CLIENT_ID, device_name, scope)
        self.grants[grant.id] = grant
        return grant

    def _tokens(self, grant: Grant) -> dict:
        access = "fake-at-" + secrets.token_urlsafe(24)
        refresh = "fake-rt-" + secrets.token_urlsafe(32)
        self._access[access] = (grant.id, self.clock() + self.access_ttl)
        self._refresh[refresh] = _Refresh(grant.id)
        return {"access_token": access, "token_type": "Bearer", "expires_in": self.access_ttl,
                "refresh_token": refresh, "scope": grant.scope}

    def _bearer(self, headers: dict) -> bool:
        auth = headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else headers.get("x-api-key", "").strip()
        if not token:
            return False
        if token in self.api_keys:
            return True
        if token in self._rejects:
            self._rejects.discard(token)
            return False
        entry = self._access.get(token)
        if entry is None:
            return False
        grant_id, expires_at = entry
        return self.clock() < expires_at and not self.grants[grant_id].revoked

    def _take_fault(self, method: str, path: str) -> _Fault | None:
        for fault in self._faults:
            if fault.path == path and (fault.method is None or fault.method == method):
                fault.times -= 1
                if fault.times <= 0:
                    self._faults.remove(fault)
                return fault
        return None

    def _metadata(self) -> dict:
        url = self.url
        data = {
            "issuer": url,
            "authorization_endpoint": url + "/oauth/authorize",
            "token_endpoint": url + "/oauth/token",
            "device_authorization_endpoint": url + "/oauth/device_authorization",
            "revocation_endpoint": url + "/oauth/revoke",
            "jwks_uri": url + "/.well-known/jwks.json",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token", DEVICE_GRANT],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "scopes_supported": sorted(SCOPES),
            "originshape": {"app_origin": url, "gateway_base_url": url + "/v1/llm/anthropic",
                            "manage_url": url + "/settings", "billing_url": url + "/billing",
                            "download_url": url + "/download"},
        }
        for key, value in self.metadata_overrides.items():
            if value is None:
                data.pop(key, None)
            else:
                data[key] = value
        return data

    # routes: each returns (status, headers, body, content type) -----------------------------------------

    def _json(self, status: int, body: Any, headers: dict | None = None) -> tuple:
        return status, dict(headers or {}), json.dumps(body).encode("utf-8"), "application/json"

    def _oauth_error(self, code: str, description: str = "", status: int = 400) -> tuple:
        body = {"error": code}
        if description:
            body["error_description"] = description
        return self._json(status, body)

    def route_discovery(self, call: RecordedCall) -> tuple:
        self.counts["discovery"] += 1
        return self._json(200, self._metadata())

    def route_jwks(self, call: RecordedCall) -> tuple:
        self.counts["jwks"] += 1
        return self._json(200, {"keys": []})

    def route_authorize(self, call: RecordedCall) -> tuple:
        self.counts["authorize"] += 1
        q = call.query
        redirect_uri = q.get("redirect_uri", "")
        if q.get("client_id") != CLIENT_ID or not _REDIRECT_RE.fullmatch(redirect_uri):
            body = b"<!doctype html><title>Sign-in error</title><p>Unknown client or redirect URI.</p>"
            return 400, {}, body, "text/html; charset=utf-8"
        state = self.authorize_state if self.authorize_state is not None else q.get("state")

        def back(**params: str) -> tuple:
            if state is not None:
                params["state"] = state
            return 302, {"Location": f"{redirect_uri}?{urlencode(params)}"}, b"", "text/plain"

        scope = q.get("scope") or SCOPE
        if (q.get("response_type") != "code" or q.get("code_challenge_method") != "S256"
                or not _CHALLENGE_RE.fullmatch(q.get("code_challenge", "")) or not set(scope.split()) <= SCOPES):
            return back(error="invalid_request", error_description="bad authorization request")
        if self.authorize_decision == "deny":
            return back(error="access_denied")
        code = "fake-code-" + secrets.token_urlsafe(16)
        self._codes[code] = _Code(code, CLIENT_ID, redirect_uri, q["code_challenge"], scope,
                                  q.get("device_name", ""), self.clock() + self.code_ttl)
        return back(code=code)

    def route_token(self, call: RecordedCall) -> tuple:
        form = call.form or (call.json if isinstance(call.json, dict) else {})
        grant_type = form.get("grant_type", "")
        if grant_type == "authorization_code":
            self.counts["token:authorization_code"] += 1
        elif grant_type == "refresh_token":
            self.counts["token:refresh_token"] += 1
        elif grant_type == DEVICE_GRANT:
            self.counts["token:device_code"] += 1
        else:
            self.counts["token:other"] += 1
            return self._oauth_error("unsupported_grant_type")
        if form.get("client_id") != CLIENT_ID:
            return self._oauth_error("invalid_client", status=401)
        if grant_type == "authorization_code":
            return self._grant_code(form)
        if grant_type == "refresh_token":
            return self._grant_refresh(form)
        return self._grant_device(form)

    def _grant_code(self, form: dict) -> tuple:
        entry = self._codes.get(form.get("code", ""))
        if entry is None:
            return self._oauth_error("invalid_grant", "unknown code")
        if entry.used:   # a code used twice revokes the grant created from it
            if entry.grant_id in self.grants:
                self.grants[entry.grant_id].revoked = True
            return self._oauth_error("invalid_grant", "code already used")
        if self.clock() >= entry.expires_at:
            return self._oauth_error("invalid_grant", "code expired")
        if form.get("redirect_uri") != entry.redirect_uri:
            return self._oauth_error("invalid_grant", "redirect_uri mismatch")
        verifier = form.get("code_verifier", "")
        if not _VERIFIER_RE.fullmatch(verifier):
            return self._oauth_error("invalid_grant", "bad code_verifier")
        digest = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode()
        if digest != entry.challenge:
            return self._oauth_error("invalid_grant", "PKCE verification failed")
        entry.used = True
        grant = self._new_grant(entry.device_name, entry.scope)
        entry.grant_id = grant.id
        return self._json(200, self._tokens(grant))

    def _grant_refresh(self, form: dict) -> tuple:
        entry = self._refresh.get(form.get("refresh_token", ""))
        if entry is None:
            return self._oauth_error("invalid_grant", "unknown refresh token")
        grant = self.grants[entry.grant_id]
        if grant.revoked:
            return self._oauth_error("invalid_grant", "grant revoked")
        if entry.used:   # reuse of a rotated token: theft detection revokes the whole grant
            grant.revoked = True
            return self._oauth_error("invalid_grant", "refresh token reused")
        entry.used = True
        return self._json(200, self._tokens(grant))

    def _grant_device(self, form: dict) -> tuple:
        device = next((d for d in self.devices.values() if d.device_code == form.get("device_code")), None)
        if device is None or device.status == "exchanged":
            return self._oauth_error("invalid_grant", "unknown or used device code")
        now = self.clock()
        if now >= device.expires_at:
            return self._oauth_error("expired_token")
        if device.status == "denied":
            return self._oauth_error("access_denied")
        too_fast = self.enforce_interval and now - device.last_poll < device.interval * self.time_scale
        device.last_poll = now
        if too_fast:
            device.interval += 5
            return self._oauth_error("slow_down")
        if device.status == "pending":
            device.polls += 1
            if self.device_auto in ("approve", "deny") and device.polls > self.device_auto_after:
                device.status = "approved" if self.device_auto == "approve" else "denied"
            if device.status == "denied":
                return self._oauth_error("access_denied")
            if device.status == "pending":
                return self._oauth_error("authorization_pending")
        device.status = "exchanged"
        return self._json(200, self._tokens(self._new_grant(device.device_name, device.scope)))

    def route_device_authorization(self, call: RecordedCall) -> tuple:
        self.counts["device_authorization"] += 1
        form = call.form or (call.json if isinstance(call.json, dict) else {})
        if form.get("client_id") != CLIENT_ID:
            return self._oauth_error("invalid_client", status=401)
        scope = form.get("scope") or SCOPE
        if not set(scope.split()) <= SCOPES:
            return self._oauth_error("invalid_scope")
        user_code = ""
        while not user_code or user_code in self.devices:
            letters = "".join(secrets.choice(USER_CODE_ALPHABET) for _ in range(8))
            user_code = f"{letters[:4]}-{letters[4:]}"
        now = self.clock()
        device = Device("fake-dc-" + secrets.token_urlsafe(24), user_code, scope, form.get("device_name", ""),
                        self.device_interval, now, now + self.device_expires_in * self.time_scale, now)
        self.devices[user_code] = device
        return self._json(200, {
            "device_code": device.device_code, "user_code": user_code, "verification_uri": self.url + "/device",
            "verification_uri_complete": f"{self.url}/device?user_code={user_code}",
            "expires_in": self.device_expires_in, "interval": self.device_interval})

    def route_revoke(self, call: RecordedCall) -> tuple:
        self.counts["revoke"] += 1
        form = call.form or (call.json if isinstance(call.json, dict) else {})
        token = form.get("token", "")
        entry = self._refresh.get(token)
        if entry is not None:
            self.grants[entry.grant_id].revoked = True
            self.revoked_tokens.append(token)
        return self._json(200, {})

    def _unauthorized(self) -> tuple:
        self.counts["unauthorized"] += 1
        return self._json(401, envelope("unauthorized", "Sign in again."))

    def route_me(self, call: RecordedCall) -> tuple:
        self.counts["me"] += 1
        if not self._bearer(call.headers):
            return self._unauthorized()
        return self._json(200, copy.deepcopy(self.me))

    def route_models(self, call: RecordedCall) -> tuple:
        self.counts["models"] += 1
        if not self._bearer(call.headers):
            return self._unauthorized()
        return self._json(200, copy.deepcopy(self.models))

    def route_plugin(self, call: RecordedCall) -> tuple:
        self.counts["plugin"] += 1
        body = copy.deepcopy(self.plugin)
        body["download_url"] = body["download_url"] or self.url + "/download"
        return self._json(200, body)

    def _gateway_refusal(self, call: RecordedCall) -> tuple | None:
        """The answer of the first server check that fails (after the bearer), or None."""
        user, links = self.me.get("user") or {}, self.me.get("links") or {}
        status = user.get("status")
        if self.enforce_account and status == "waitlisted":
            return self._json(403, envelope("waitlisted", "Your account is on the waitlist.",
                                            action_url=links.get("waitlist")))
        if self.enforce_account and status == "suspended":
            return self._json(403, envelope("suspended", "Your account is suspended."))
        if not self.llm_configured:
            return self._json(501, envelope("llm_not_configured", LLM_NOT_CONFIGURED_MESSAGE))
        if self.enforce_models:
            if not isinstance(call.json, dict):
                return self._json(400, envelope("invalid_request", "The request body is not a JSON object."))
            ids = [m.get("id") for m in self.models.get("data", []) if isinstance(m, dict)]
            if call.json.get("model") not in ids and call.json.get("model") not in HIDDEN_MODELS:
                return self._json(400, envelope("model_unavailable", f"Use one of: {', '.join(map(str, ids))}."))
        available = (self.me.get("credits") or {}).get("available", 0)
        if self.enforce_account and isinstance(available, (int, float)) and available <= 0:
            return self._json(402, envelope("insufficient_credits", "You have no credits left.",
                                            action_url=links.get("billing")))
        return None

    @staticmethod
    def _reply_model(call: RecordedCall) -> str:
        alias = call.json.get("model") if isinstance(call.json, dict) else None
        return alias if isinstance(alias, str) and alias else "oscar-default"

    def route_gateway(self, call: RecordedCall) -> tuple:
        self.counts["gateway"] += 1
        self.gateway_requests.append(call)
        if not self._bearer(call.headers):
            return self._unauthorized()
        refusal = self._gateway_refusal(call)
        if refusal is not None:
            return refusal
        model = self._reply_model(call)
        if self.gateway:
            item = self.gateway.pop(0)
        else:
            streaming = isinstance(call.json, dict) and call.json.get("stream") is True
            item = sse_text(model=model) if streaming else message_json(model=model)
        if isinstance(item, str):
            item = sse_text(item, model=model)
        if not isinstance(item, GatewayReply) and hasattr(item, "events"):   # a support.scripted_llm.Round
            from support.server import render_anthropic   # lazy: it imports oscar

            chunks, terminal = render_anthropic(item, model)
            item = GatewayReply(200, b"".join(chunks + terminal))
        if not isinstance(item, GatewayReply):
            raise TypeError(f"cannot serve a {type(item).__name__} from the fake gateway")
        headers = {"Cache-Control": "no-cache", **item.headers}
        return item.status, headers, item.body, item.content_type

    def route_for(self, method: str, path: str) -> str | None:
        """The route of a path that ``ROUTES`` can't name exactly (one with an id in it); a subclass adds them
        (``fake_workspace.FakeWorkspace``). None: 404."""
        return None

    ROUTES = {
        ("GET", "/.well-known/oauth-authorization-server"): "route_discovery",
        ("GET", "/.well-known/jwks.json"): "route_jwks",
        ("GET", "/oauth/authorize"): "route_authorize",
        ("POST", "/oauth/token"): "route_token",
        ("POST", "/oauth/device_authorization"): "route_device_authorization",
        ("POST", "/oauth/revoke"): "route_revoke",
        ("GET", "/v1/me"): "route_me",
        ("GET", "/v1/models"): "route_models",
        ("GET", "/v1/plugin"): "route_plugin",
        ("POST", GATEWAY_PATH): "route_gateway",
    }


class _Handler(BaseHTTPRequestHandler):
    fake: FakeOriginShape
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server's signature
        pass

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        fake = self.fake
        parts = urlsplit(self.path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        headers = {k.lower(): v for k, v in self.headers.items()}
        form: dict = {}
        body_json: Any = None
        if raw:
            if "json" in headers.get("content-type", ""):
                try:
                    body_json = json.loads(raw.decode("utf-8"))
                except ValueError:
                    body_json = None
            else:
                form = dict(parse_qsl(raw.decode("utf-8", "replace"), keep_blank_values=True))
        with fake.lock:
            call = RecordedCall(len(fake.calls) + 1, method, parts.path,
                                dict(parse_qsl(parts.query, keep_blank_values=True)), headers, form, body_json, raw)
            fake.calls.append(call)
        delay = fake.delays.get(parts.path)
        if delay:
            time.sleep(delay)
        with fake.lock:
            fault = fake._take_fault(method, parts.path)
            if fault is not None:
                fake.counts["fault"] += 1
                answer = (fault.status, fault.headers, fault.body, "application/json")
            else:
                route = FakeOriginShape.ROUTES.get((method, parts.path)) or fake.route_for(method, parts.path)
                if route is None:
                    fake.counts["not_found"] += 1
                    answer = fake._json(404, envelope("not_found", "Not found."))
                else:
                    try:
                        answer = getattr(fake, route)(call)
                    except Exception as e:  # noqa: BLE001 - a fake bug is answered, never a stderr traceback
                        answer = fake._json(500, envelope("upstream_error", f"fake error: {type(e).__name__}: {e}"))
        status, extra, body, content_type = answer
        reply_delay = fake.reply_delays.get(parts.path)
        if reply_delay:
            time.sleep(reply_delay)
        if status == 0:   # a scripted network failure: drop the connection without an answer
            self.close_connection = True
            return
        try:
            self.send_response(status, _REASONS.get(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            for name, value in extra.items():
                self.send_header(name, str(value))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        except OSError:
            pass
        self.close_connection = True
