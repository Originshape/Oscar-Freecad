"""HTTP plumbing shared by ``oscar.auth``: the error types, ``TokenSet``, and the one ``request()`` every call uses.

Why a single door: every auth request needs the same guarantees.

* An explicit timeout, so no request can hang a worker thread forever.
* The ``Oscar-FreeCAD/<version>`` user agent.
* No redirects. ``requests`` re-sends a form body (a refresh token, a code) to wherever a 307/308 points, and a
  token must only ever reach the server it was meant for.
* A capped body: an auth answer is a few hundred bytes, so a flood is refused instead of being buffered.
* Failures that are safe to show and log. ``requests`` quotes header values and URLs in its exception messages, and
  a header can hold a token, so a transport failure becomes ``AuthUnavailable`` built from the host and the
  exception TYPE only. It is raised after the ``except`` block, so the original is not even chained as
  ``__context__``.

Text the server sends (an ``error_description``) is clipped and scrubbed by ``safe_text`` before it may appear in a
message. No Qt and no FreeCAD; ``requests`` is the only third-party import.

``download()`` is the same door for a file (a workspace attachment): the body streams to disk instead of memory,
capped at the size the server announced, and only a complete file with the expected checksum is kept.
"""

import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlsplit

import requests

from .. import __version__
from .. import context_guard as _context_guard

CLIENT_ID = "oscar-freecad"
SCOPE = "oscar:chat oscar:account offline_access"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
DEFAULT_TIMEOUT: tuple[float, float] = (5.0, 15.0)   # (connect, read) seconds
REVOKE_TIMEOUT: tuple[float, float] = (2.0, 3.0)     # best effort on the way out: never keep the user waiting
MAX_BODY_BYTES = 1_000_000

TimeoutSpec = float | tuple[float, float]


# -- errors -----------------------------------------------------------------------------------------------

class AuthError(Exception):
    """A sign-in or account failure whose message is safe to show and to log.

    The message never contains a token, a code, a verifier or an API key. ``code`` is a short reason for the UI
    (``"declined"``, ``"signed_out"``, ``"invalid_grant"`` ...). ``status`` is the HTTP status when there was one
    and 0 otherwise."""

    code = "auth_error"

    def __init__(self, message: str, *, code: str = "", status: int = 0):
        super().__init__(message)
        if code:
            self.code = code
        self.status = status


class SignedOut(AuthError):
    """No usable session: never signed in, signed out, or the server ended the grant (``invalid_grant``)."""

    code = "signed_out"


class AuthUnavailable(AuthError):
    """The server could not be reached or is failing (network error, 429, 5xx). A session is kept."""

    code = "unavailable"


class AccountSuspended(AuthError):
    """The account is suspended: a sign-in is refused with ``access_denied`` / ``account_suspended``, a refresh with
    ``invalid_grant`` and ``"reason": "account_suspended"``. Signing in again cannot help."""

    code = "suspended"


SUSPENDED_MESSAGE = ("Your OriginShape account is suspended, so Oscar can't use it. Signing in again won't help: "
                     "contact OriginShape support to find out why.")


def says_suspended(reply_or_body: Any, description: str = "") -> bool:
    """True when an OAuth error answer says the account is suspended (``"reason": "account_suspended"`` in the body,
    or ``account_suspended`` as the description: the browser's redirect and the device poll carry it there)."""
    data = reply_or_body.json() if isinstance(reply_or_body, Reply) else reply_or_body
    reason = data.get("reason") if isinstance(data, dict) else ""
    return reason == "account_suspended" or (description or "").strip() == "account_suspended"


class SignInDeclined(AuthError):
    """The user answered no (``access_denied``)."""

    code = "declined"


class SignInExpired(AuthError):
    """The sign-in request (a device code) expired before it was approved."""

    code = "expired"


class SignInCancelled(AuthError):
    """The caller set the ``cancelled`` event."""

    code = "cancelled"


class SignInTimedOut(AuthError):
    """No answer from the browser within the time allowed."""

    code = "timeout"


class InvalidServerURL(AuthError):
    """A server URL Oscar refuses to use: malformed, carrying credentials, or plain http to a non-loopback host
    (``code`` is ``"insecure_url"`` then)."""

    code = "invalid_url"


class ServerMetadataError(AuthError):
    """The server's sign-in metadata is missing or failed validation."""

    code = "bad_metadata"


# -- tokens -----------------------------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class TokenSet:
    """One token endpoint answer. ``repr`` never shows a token, so an accidental log line stays safe."""

    access_token: str
    refresh_token: str = ""
    expires_in: int = 900
    scope: str = ""
    token_type: str = "Bearer"
    received_at: float = field(default_factory=time.monotonic)   # monotonic: a wall-clock jump cannot fake freshness

    def __post_init__(self) -> None:
        # Tokens stay out of the model's context, whatever a script prints (context_guard).
        _context_guard.remember(self.access_token)
        _context_guard.remember(self.refresh_token)

    @property
    def expires_at(self) -> float:
        """``time.monotonic()`` value at which the access token expires."""
        return self.received_at + self.expires_in

    def __repr__(self) -> str:
        refresh = "<hidden>" if self.refresh_token else "<none>"
        return (f"TokenSet(access_token=<hidden>, refresh_token={refresh}, expires_in={self.expires_in}, "
                f"scope={self.scope!r})")


# RFC 6750 b64token: what an Authorization header can carry without quoting. JWTs, opaque tokens and os_live_ keys
# all fit; a space, a quote or a newline never does (requests would quote such a value in its exception message).
_TOKEN_RE = re.compile(r"[A-Za-z0-9\-._~+/]+=*")


def clean_token(value: Any, what: str, *, error: str = "") -> str:
    """``value`` without surrounding whitespace, or ``AuthError`` (``code="bad_token"``) whose message names only
    ``what``: an empty value, one longer than 8 KB, or one with a character a header cannot carry is refused, so a
    malformed token can never reach a header or an exception text."""
    if isinstance(value, str):
        token = value.strip()
        if 8 <= len(token) <= 8192 and _TOKEN_RE.fullmatch(token):
            return token
    raise AuthError(error or f"OriginShape sent a malformed {what}.", code="bad_token")


# -- safe text --------------------------------------------------------------------------------------------

_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}(?:\.[A-Za-z0-9_-]*)?")
_KEY_RE = re.compile(r"os_[a-z]+_[A-Za-z0-9_-]{6,}")
_LONG_RE = re.compile(r"[A-Za-z0-9_-]{32,}")   # opaque tokens, codes and verifiers are long base64url runs


def safe_text(text: Any, *secrets: str, limit: int = 200) -> str:
    """Server-provided text made safe for a message: printable, single-spaced, every ``secrets`` value (raw and
    URL-encoded) and anything shaped like a token masked, then clipped to ``limit`` characters."""
    if not isinstance(text, str):
        return ""
    out = " ".join("".join(ch if ch.isprintable() else " " for ch in text).split())
    for secret in secrets:
        if isinstance(secret, str) and len(secret) >= 4:
            out = out.replace(secret, "[redacted]").replace(quote(secret, safe=""), "[redacted]")
    for pattern in (_JWT_RE, _KEY_RE, _LONG_RE):
        out = pattern.sub("[redacted]", out)
    return out[:limit]


# -- requests ---------------------------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class Reply:
    """A finished response: status, headers (lower-case names) and the body bytes."""

    status: int
    headers: dict
    body: bytes

    def json(self) -> Any:
        """The body as JSON, or None when it is not JSON."""
        try:
            return json.loads(self.body.decode("utf-8"))
        except ValueError:   # UnicodeDecodeError and JSONDecodeError are both ValueErrors
            return None

    def __repr__(self) -> str:
        return f"Reply(status={self.status}, {len(self.body)} bytes)"


def user_agent() -> str:
    return f"Oscar-FreeCAD/{__version__}"


def host_of(url: str) -> str:
    """``host[:port]`` of ``url`` for messages (never the path, the query or a user name)."""
    try:
        parts = urlsplit(url)
        host, port = parts.hostname or "?", parts.port
    except ValueError:
        return "?"
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{port}" if port else host


def _read_capped(response: requests.Response) -> bytes | None:
    """The body, or None when it is larger than ``MAX_BODY_BYTES``."""
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(65536):
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def request(method: str, url: str, *, timeout: TimeoutSpec = DEFAULT_TIMEOUT, headers: dict[str, str] | None = None,
            data: dict[str, str] | None = None, json_body: Any = None, content: bytes | None = None,
            content_type: str = "application/octet-stream") -> Reply:
    """One HTTP exchange with the guarantees in the module docstring. Raises ``AuthUnavailable`` for a transport
    failure and ``AuthError`` for an oversized answer; every HTTP status (including 4xx/5xx) is returned.
    ``content`` sends raw bytes as the body (a file upload), typed ``content_type``; it excludes ``data`` and
    ``json_body``."""
    if content is not None and (data is not None or json_body is not None):
        raise ValueError("content excludes data and json_body")
    hdrs = {"User-Agent": user_agent(), "Accept": "application/json"}
    if content is not None:
        hdrs["Content-Type"] = content_type
    hdrs.update(headers or {})
    failure = ""
    status, reply_headers, body = 0, {}, None
    response = None
    try:
        response = requests.request(method, url, headers=hdrs, data=content if content is not None else data,
                                    json=json_body, timeout=timeout, allow_redirects=False, stream=True)
        status = response.status_code
        reply_headers = {k.lower(): v for k, v in response.headers.items()}
        body = _read_capped(response)
    except requests.RequestException as e:   # includes InvalidHeader / InvalidURL, whose text quotes the value
        failure = type(e).__name__
    except (OSError, ValueError) as e:      # socket / urllib3 errors that escape requests' wrapping, odd URLs
        failure = type(e).__name__
    finally:
        if response is not None:
            response.close()
    if failure:
        raise AuthUnavailable(f"Could not reach the OriginShape server at {host_of(url)} ({failure}).")
    if body is None:
        raise AuthError("The OriginShape server sent an unexpectedly large answer.", code="bad_response", status=status)
    return Reply(status, reply_headers, body)


@dataclass(frozen=True)
class Saved:
    """A file ``download()`` wrote: its size and sha256 (hex)."""

    size: int
    sha256: str


def download(url: str, dest: str, *, max_bytes: int, sha256: str = "", timeout: TimeoutSpec = DEFAULT_TIMEOUT,
             headers: dict[str, str] | None = None) -> tuple[Reply, Saved | None]:
    """GET ``url`` into the file ``dest`` (its folder must exist). For a 200 the body streams to a temporary file
    beside ``dest``, which replaces ``dest`` only once complete, at most ``max_bytes`` long and, when ``sha256`` is
    given, matching it: returns ``(reply without body, Saved)``. Any other status returns ``(reply, None)`` with the
    (capped) body, as ``request`` does, and leaves no file behind. Raises ``AuthUnavailable`` for a transport failure
    and ``AuthError`` (``code`` ``too_large`` or ``corrupt``) for a file that is not what was announced."""
    hdrs = {"User-Agent": user_agent(), "Accept": "*/*"}
    hdrs.update(headers or {})
    failure, problem = "", None
    status, reply_headers, body, saved = 0, {}, b"", None
    response = None
    part = ""
    try:
        response = requests.get(url, headers=hdrs, timeout=timeout, allow_redirects=False, stream=True)
        status = response.status_code
        reply_headers = {k.lower(): v for k, v in response.headers.items()}
        if status != 200:
            body = _read_capped(response)
        else:
            digest, total = hashlib.sha256(), 0
            handle, part = tempfile.mkstemp(prefix=".part-", dir=os.path.dirname(os.path.abspath(dest)))
            with os.fdopen(handle, "wb") as out:
                for chunk in response.iter_content(65536):
                    total += len(chunk)
                    if total > max_bytes:
                        problem = AuthError("The attached file is larger than OriginShape said it would be.",
                                            code="too_large", status=status)
                        break
                    digest.update(chunk)
                    out.write(chunk)
            if problem is None and sha256 and digest.hexdigest() != sha256.lower():
                problem = AuthError("The attached file arrived damaged (its checksum doesn’t match).",
                                    code="corrupt", status=status)
            if problem is None:
                os.replace(part, dest)
                part = ""
                saved = Saved(total, digest.hexdigest())
    except requests.RequestException as e:
        failure = type(e).__name__
    except (OSError, ValueError) as e:
        failure = type(e).__name__
    finally:
        if response is not None:
            response.close()
        if part:
            try:
                os.remove(part)
            except OSError:
                pass
    if failure:
        raise AuthUnavailable(f"Could not download from the OriginShape server at {host_of(url)} ({failure}).")
    if problem is not None:
        raise problem
    if body is None:
        raise AuthError("The OriginShape server sent an unexpectedly large answer.", code="bad_response", status=status)
    return Reply(status, reply_headers, body), saved


def is_unavailable(status: int) -> bool:
    """True for statuses that mean "try again later": the request itself was fine. 501 is not one of them: the
    server does not offer what was asked (``llm_not_configured``), and asking again changes nothing."""
    return status in (408, 425, 429) or (status >= 500 and status != 501)


def unavailable(reply: Reply) -> AuthUnavailable:
    return AuthUnavailable(f"The OriginShape server is unavailable right now (HTTP {reply.status}). Try again in a "
                           "moment.", status=reply.status)


# -- error bodies -----------------------------------------------------------------------------------------

_CODE_RE = re.compile(r"[a-z0-9_.\-]{1,64}")

# A server without an AI model configured answers a chat request with 501 ``llm_not_configured``. It is permanent
# (never retried), and the user is told plainly instead of seeing "HTTP 501".
LLM_NOT_CONFIGURED = "llm_not_configured"
LLM_NOT_CONFIGURED_MESSAGE = "The OriginShape server has no AI model configured."


def error_code(value: Any) -> str:
    """``value`` as an error code (lower case, ``[a-z0-9_.-]``, at most 64 characters), else ``""``."""
    if isinstance(value, str) and _CODE_RE.fullmatch(value.strip().lower()):
        return value.strip().lower()
    return ""


def error_details(body: Any, *secrets: str) -> tuple[str, str]:
    """``(code, description)`` of an error body: the API's envelope (``{"type": "error", "error": {"type", "code",
    "message"}}``, the OriginShape API) or RFC 6749 section 5.2 (``{"error", "error_description"}``). ``body`` is the raw
    text or bytes, or the decoded JSON. The description is scrubbed with ``secrets``; ``("", "")`` when the body is
    neither shape. The OriginShape provider can use it on the gateway's error bodies too."""
    data = body
    if isinstance(body, (bytes, bytearray)):
        body = bytes(body).decode("utf-8", "replace")
    if isinstance(body, str):
        try:
            data = json.loads(body)
        except ValueError:
            return "", ""
    if not isinstance(data, dict):
        return "", ""
    err, desc = data.get("error"), data.get("error_description")
    if isinstance(err, dict):
        desc = err.get("message")
        err = err.get("code") or err.get("type")
    return error_code(err), safe_text(desc, *secrets)


def oauth_error(reply: Reply, *secrets: str) -> tuple[str, str]:
    """``error_details`` of a reply's body."""
    return error_details(reply.json(), *secrets)


def api_error(reply: Reply, what: str, *secrets: str) -> AuthError:
    """The exception for an API answer (``/v1/me``, ``/v1/models`` ...) that is not 200. ``what`` names the thing
    asked for in the message ("the account"). 408/425/429/5xx (but 501) give ``AuthUnavailable``; 501
    ``llm_not_configured`` gives ``LLM_NOT_CONFIGURED_MESSAGE``; anything else an ``AuthError`` whose ``code`` is the
    envelope's code (``"waitlisted"``, ``"not_found"`` ...) or ``"http_error"``."""
    if is_unavailable(reply.status):
        return unavailable(reply)
    code, _desc = oauth_error(reply, *secrets)
    if code == LLM_NOT_CONFIGURED:
        return AuthError(LLM_NOT_CONFIGURED_MESSAGE, code=code, status=reply.status)
    return AuthError(f"OriginShape did not return {what} (HTTP {reply.status}).", code=code or "http_error",
                     status=reply.status)


_REFUSED = ("OriginShape does not recognise this version of Oscar. Update Oscar, or check the server URL in its "
            "settings.")


def token_error(reply: Reply, *secrets: str) -> AuthError:
    """The exception for a token endpoint answer that is not 200 (sign-in flows; refresh handles its own)."""
    if is_unavailable(reply.status):
        return unavailable(reply)
    code, desc = oauth_error(reply, *secrets)
    if code == "invalid_grant":
        return AuthError("The sign-in was not accepted: the code expired or was already used. Please sign in again.",
                         code=code, status=reply.status)
    if code == "access_denied":
        if says_suspended(reply, desc):
            return AccountSuspended(SUSPENDED_MESSAGE, status=reply.status)
        return SignInDeclined("The sign-in request was declined.", status=reply.status)
    if code == "expired_token":
        return SignInExpired("The sign-in request expired. Please start again.", status=reply.status)
    if code in ("invalid_client", "unauthorized_client"):
        return AuthError(_REFUSED, code=code, status=reply.status)
    detail = f"{code}: {desc}" if code and desc else (code or f"HTTP {reply.status}")
    return AuthError(f"The OriginShape server refused the sign-in ({detail}).", code=code or "token_error",
                     status=reply.status)


def token_set(reply: Reply, *secrets: str) -> TokenSet:
    """The ``TokenSet`` of a token endpoint answer; ``token_error`` for any other status. ``secrets`` (the code, the
    verifier, the refresh token that was sent) are scrubbed from any server text that reaches a message."""
    if reply.status != 200:
        raise token_error(reply, *secrets)
    data = reply.json()
    if not isinstance(data, dict):
        raise AuthError("OriginShape answered the sign-in with something Oscar cannot read.", code="bad_response",
                        status=reply.status)
    token_type = data.get("token_type") or "Bearer"
    if not isinstance(token_type, str) or token_type.lower() != "bearer":
        raise AuthError("OriginShape issued a kind of token Oscar cannot use.", code="bad_response",
                        status=reply.status)
    access = clean_token(data.get("access_token"), "access token")
    refresh_raw = data.get("refresh_token")
    refresh = clean_token(refresh_raw, "refresh token") if refresh_raw not in (None, "") else ""
    expires_in = data.get("expires_in")
    if (isinstance(expires_in, bool) or not isinstance(expires_in, (int, float)) or not math.isfinite(expires_in)
            or expires_in <= 0):   # json.loads accepts NaN and Infinity
        expires_in = 300   # missing or odd: assume short so the token is renewed early rather than late
    scope = safe_text(data.get("scope"), limit=200) if isinstance(data.get("scope"), str) else ""
    return TokenSet(access, refresh, int(min(expires_in, 86400)), scope, "Bearer")


def revoke_quietly(endpoint: str, token: str, *, timeout: TimeoutSpec = REVOKE_TIMEOUT) -> bool:
    """Best-effort RFC 7009 revocation of a refresh token; True when the server answered 200. Never raises."""
    if not endpoint or not token:
        return False
    try:
        reply = request("POST", endpoint, timeout=timeout,
                        data={"token": token, "token_type_hint": "refresh_token", "client_id": CLIENT_ID})
    except AuthError:
        return False
    return reply.status == 200
