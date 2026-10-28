"""OriginShape: the account's models through the OriginShape API, paid with the user's account.

``OriginShapeProvider`` is ``AnthropicProvider`` (same Messages wire, stream parser and 400 adaptations) with three
differences.

* **Where requests go.** ``_endpoint()`` is always ``{api_url}`` + ``GATEWAY_PATH``, with the API URL from
  ``oscar.auth.config`` (``OSCAR_API_URL``, the Server preference while *Use a custom server* is on, or the public
  API). It is resolved once per provider (one turn), so the token and the endpoint of every attempt belong to the same
  server, and it never falls back to another provider's API: the account's token must only ever reach the
  OriginShape API.
* **How it signs in.** ``_headers()`` runs on the reader thread, once per attempt, and asks the ``AuthSession`` of that
  URL for a bearer: the in-memory access token (renewed when it is about to expire) or the personal API key. A renewal
  blocks the reader thread only, so Stop keeps working. The token is validated before it goes into a header, because
  ``requests`` quotes a malformed header value in its exception text. ``x-api-key`` is never sent. ``request_context``
  adds ``x-oscar-session``, ``x-oscar-project`` (a hash of the chat id, never the project's name) and ``x-oscar-turn``.
  After a 401 the gateway answered itself, ``_reauthenticate`` forgets the refused token and the request goes once more.
* **What the model is.** The profile names an alias (``origin1`` / ``origin1-super``, or a retired ``oscar-*``) and
  the wire carries it as is. Its capabilities are what the server's ``/v1/models`` declares (cached by the account hub
  when it refreshes, and for a workspace job's model; the defaults until then). ``_body`` adds three rules that hold
  for any model an alias stands for: no ``temperature``, thinking switched off as low effort, and no forced tool
  choice.

Errors (``ProviderError.kind`` / ``.code``; ``.data`` is ``{"code", "type", "message", "upstream"}`` from the body).
The API's own errors carry an error ``code``. An error body without one was passed through from further up the
request and is never read as the user's account.

===========  ==========================================  =========  ===============================
status       the answer                                  kind       code
===========  ==========================================  =========  ===============================
401          unauthorized, after the one renewal         auth       signed_out
402          insufficient_credits                        account    insufficient_credits
403          waitlisted / suspended / another code       account    waitlisted / suspended / <code>
429          rate_limited: retried, final when its       account    rate_limited (``retry_after``)
             Retry-After exceeds what the call may wait
501          llm_not_configured: never retried           http       llm_not_configured
400          model_unavailable                           http       model_unavailable
401/402/403  an error passed through, without a code     http       upstream_refused
(none)       no answer after every retry                 transport  unreachable
===========  ==========================================  =========  ===============================

Any other status keeps the shared ``http_error`` wording and retry rule, with the gateway's ``code`` attached. Before
anything is sent: no sign-in, or one the server has ended, is ``kind="auth", code="signed_out"``; a server URL Oscar
refuses is ``kind="auth"`` with ``code`` ``invalid_url`` / ``insecure_url``; a token endpoint that cannot be reached is
retried like a transport failure (``kind="transport", code="unavailable"``). Every message is safe to show and log:
server text is clipped and scrubbed, and no token ever appears in one.
"""

import hashlib
import json
import math
import re
from collections.abc import Mapping

from . import __version__
from .auth import config as _config
from .auth import session as _session
from .auth.transport import (LLM_NOT_CONFIGURED, LLM_NOT_CONFIGURED_MESSAGE, AuthError, AuthUnavailable, SignedOut,
                             clean_token, error_code, host_of, safe_text)
from .provider_chat import ChatLoop
from .provider_transport import ProviderError, http_error, is_quota_error, retry_after_seconds
from .providers import AnthropicProvider

GATEWAY_PATH = "/v1/llm/anthropic/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
_CONTEXT_VALUE = re.compile(r"[A-Za-z0-9._:\-]{1,128}")   # chat and turn ids; anything else stays out of the headers


def project_id(chat_id: str) -> str:
    """The ``x-oscar-project`` value, ``sha256("project:" + chat_id)[:16]``: a hash, because the project folder's
    name carries a slug of the user's first prompt."""
    return hashlib.sha256(("project:" + chat_id).encode("utf-8")).hexdigest()[:16]


def _context_value(value: object) -> str:
    return value if isinstance(value, str) and _CONTEXT_VALUE.fullmatch(value) else ""


def context_headers(context: object) -> dict[str, str]:
    """The ``x-oscar-*`` headers for a ``request_context`` (``{"session": <chat id>, "turn": <turn id>}``). A value that
    is not a short id (letters, digits, ``._:-``) is left out, so it can neither be sent nor break the request."""
    if not isinstance(context, Mapping):
        return {}
    headers: dict[str, str] = {}
    chat = _context_value(context.get("session"))
    if chat:
        headers["x-oscar-session"] = chat
        headers["x-oscar-project"] = project_id(chat)
    turn = _context_value(context.get("turn"))
    if turn:
        headers["x-oscar-turn"] = turn
    return headers


def _error_body(text: str, *secrets: str) -> tuple[str, str, str]:
    """``(code, type, message)`` of an error body in the API's envelope or in the Messages wire's shape.
    ``code`` is only ever OriginShape's own, since errors passed through have none. The message is printable,
    clipped, and scrubbed of ``secrets`` and of anything shaped like a token."""
    try:
        data = json.loads(text) if text else None
    except ValueError:
        data = None
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return "", "", ""
    return error_code(error.get("code")), error_code(error.get("type")), safe_text(error.get("message"), *secrets)


def _error_details(text: str) -> dict[str, str]:
    """``{field: message}`` of the envelope's ``details`` list (a 402 names ``available_credits`` and
    ``required_credits`` there when a long chat needs more credits set aside than the account has)."""
    try:
        data = json.loads(text) if text else None
    except ValueError:
        return {}
    error = data.get("error") if isinstance(data, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    return {str(d.get("field")): safe_text(d.get("message")) for d in details or () if isinstance(d, dict)}


def _credits(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _wait_text(seconds: "float | None") -> str:
    """``in 41 s`` for a short wait, ``in about 3 h 25 min`` for a long one (a daily limit resets at midnight UTC)."""
    if not seconds:
        return "in a moment"
    if seconds < 600:
        return f"in {math.ceil(seconds)} s"
    minutes = math.ceil(seconds / 60)
    hours, minutes = divmod(minutes, 60)
    return f"in about {hours} h {minutes} min" if hours else f"in about {minutes} min"


def _auth_failure(error: AuthError) -> ProviderError:
    """The provider error for a sign-in that cannot give a token (an ``AuthError`` message is safe by contract)."""
    if isinstance(error, SignedOut):
        if error.code == "suspended":   # the server ended the grant because the account is suspended
            return ProviderError(str(error), status=error.status, kind="account", code="suspended")
        return ProviderError(str(error), status=error.status, kind="auth", code="signed_out")
    if isinstance(error, AuthUnavailable):   # the token endpoint is unreachable or failing: like a transport error
        return ProviderError(str(error), status=error.status, retriable=True, kind="transport", code="unavailable")
    return ProviderError(str(error), status=error.status, kind="auth", code=error.code)


class OriginShapeProvider(AnthropicProvider):
    """The account's models through the OriginShape API with the account's sign-in (see the module docstring)."""

    name = "originshape"
    kind = "originshape"

    def __init__(self, profile):  # oscar.params.Profile; no network and no keystore here (make_provider may run anywhere)
        super().__init__(profile)
        self._api = ""          # the API root, resolved on first use and then kept: one server per provider (turn)
        self._sent_token = ""   # the bearer of the latest attempt: written on the reader thread before the request,
                                # read on the consumer thread once that attempt has ended

    # -- where, and as whom ---------------------------------------------------------------------------------

    def _api_url(self) -> str:
        """The API root; ``ProviderError`` (``kind="auth"``) for a URL tokens must not be sent to."""
        if not self._api:
            failure: AuthError | None = None
            try:
                self._api = _config.api_url()
            except AuthError as e:
                failure = e
            if failure is not None:   # raised outside the handler: nothing chained
                raise _auth_failure(failure)
        return self._api

    def _session(self) -> "_session.AuthSession":
        return _session.get_session(self._api_url())

    def _bearer(self) -> str:
        """A bearer valid for this attempt. Blocking when the access token must be renewed (reader thread)."""
        session = self._session()
        token, failure = "", None
        try:
            token = session.access_token()
        except AuthError as e:
            failure = _auth_failure(e)
        except Exception as e:  # noqa: BLE001 - a bug in the sign-in code: its text could quote anything
            failure = ProviderError(f"Oscar could not read the OriginShape sign-in ({type(e).__name__}).", kind="auth",
                                    code="auth_error")
        if failure is not None:
            raise failure
        try:
            return clean_token(token, "access token")
        except AuthError:
            pass
        raise ProviderError("OriginShape returned an access token Oscar cannot use. Please sign in again.", kind="auth",
                            code="signed_out")

    def _endpoint(self) -> str:
        return self._api_url() + GATEWAY_PATH

    def _headers(self) -> dict[str, str]:
        token = self._bearer()
        self._sent_token = token
        headers = {
            "authorization": f"Bearer {token}",
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            "user-agent": f"Oscar-FreeCAD/{__version__}",
            "x-oscar-client": f"freecad/{__version__}",
        }
        headers.update(context_headers(getattr(self, "request_context", None)))
        return headers

    def _body(self, system, messages, tools, **opts):
        """``AnthropicProvider``'s body with the account's rules: no ``temperature`` (the account's models set their
        own sampling), thinking switched off as low effort, and no forced tool choice (the agent's nudge text carries
        the instruction)."""
        body = super()._body(system, messages, tools, **opts)
        body.pop("temperature", None)
        if (body.get("thinking") or {}).get("type") in ("disabled", "between_tools"):
            del body["thinking"]
            if self._learned.get("output_config_ok", True):
                body["output_config"] = {"effort": "low"}
        choice = body.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") in ("any", "tool"):
            kept = {k: v for k, v in choice.items() if k == "disable_parallel_tool_use"}
            body["tool_choice"] = {"type": "auto", **kept}
        return body

    def _max_tokens(self, per_call: int | None = None) -> int:
        """The output cap, also bounded by the model's own in ``/v1/models`` (``max_output_tokens``): the gateway would
        lower a larger one silently."""
        value = super()._max_tokens(per_call)
        try:
            from .auth import account

            info = account.cached_model(self.profile.model or "")
        except Exception:  # noqa: BLE001 - no settings file or an unusable server URL: the profile's cap stands
            info = None
        cap = info.max_output_tokens if info is not None else 0
        return min(value, cap) if cap else value

    # -- errors ---------------------------------------------------------------------------------------------

    def _reauthenticate(self, text: str) -> bool:
        """After a 401: forget the refused token, so the next attempt renews it (``_headers``, reader thread). True
        when a renewal can help: an OAuth sign-in with a refresh token, refused by the API itself (a 401 passed through
        from further up has no code, and no sign-in of ours can fix it). No network here."""
        code, etype, _message = _error_body(text)
        if etype and not code:
            return False
        try:
            return bool(self._session().invalidate(self._sent_token or None))
        except Exception:  # noqa: BLE001 - an unusable server URL or a sign-in bug: the 401 itself is reported
            return False

    def _http_error(self, status: int, text: str, headers) -> ProviderError:
        token = self._sent_token
        code, etype, message = _error_body(text, token)
        upstream = bool(etype) and not code
        data = {"code": code, "type": etype, "message": message, "upstream": upstream}
        detail = f": {message}" if message else "."
        if upstream and status in (401, 402, 403):
            return ProviderError(f"OriginShape could not run the request (HTTP {status}, {etype}){detail}",
                                 status=status, kind="http", code="upstream_refused", data=data)
        if status == 401:
            return ProviderError("OriginShape did not accept your sign-in (HTTP 401). Please sign in again.",
                                 status=401, kind="auth", code="signed_out", data=data)
        if status == 402:
            details = _error_details(text)
            if _credits(details.get("available_credits", "")) > 0 and message:
                # credits are left, but not enough to set aside for a request this long: a new chat helps, buying may not
                return ProviderError(f"{message} (HTTP 402)", status=402, kind="account", code="chat_too_long_for_credits",
                                     data={**data, "details": details})
            return ProviderError("You have no OriginShape credits left (HTTP 402). Add credits or choose a plan to "
                                 "continue.",
                                 status=402, kind="account", code="insufficient_credits", data=data)
        if status == 403:
            if code == "waitlisted":
                return ProviderError("Your OriginShape account is still on the waitlist (HTTP 403).", status=403,
                                     kind="account", code="waitlisted", data=data)
            if code == "suspended":
                return ProviderError("Your OriginShape account is suspended (HTTP 403).", status=403, kind="account",
                                     code="suspended", data=data)
            named = f", {code}" if code else ""
            return ProviderError(f"OriginShape refused the request (HTTP 403{named}){detail}", status=403,
                                 kind="account", code=code or "forbidden", data=data)
        if status == 429 and not upstream and not is_quota_error(text):
            wait = retry_after_seconds(headers)
            if "usage limit" in message.lower():   # the account's daily limit (gateway/budget.py), not a request rate
                return ProviderError(f"{message} (HTTP 429) Try again {_wait_text(wait)}.", status=429, kind="account",
                                     code="daily_limit", data=data, retry_after=wait)
            return ProviderError(f"OriginShape is limiting how many requests your account can send (HTTP 429). "
                                 f"Try again {_wait_text(wait)}.", status=429, retriable=True, kind="account",
                                 code="rate_limited", data=data, retry_after=wait)
        if status == 501 and code == LLM_NOT_CONFIGURED:   # a server without an AI model: asking again changes nothing
            # the same code covers one model whose provider has no key ("This model is not available…"): say that
            said = f"{message} (HTTP 501)" if message and message != LLM_NOT_CONFIGURED_MESSAGE else LLM_NOT_CONFIGURED_MESSAGE
            return ProviderError(said, status=501, kind="http", code=LLM_NOT_CONFIGURED, data=data)
        if status == 400 and code == "model_unavailable":
            model = self.profile.model or ""
            return ProviderError(f"OriginShape does not offer the model {model!r} (HTTP 400){detail}", status=400,
                                 kind="http", code=code, data=data)
        error = http_error(status, safe_text(text, token, limit=2000))
        if code and message:   # the API's own envelope: its message, not the raw JSON
            what = "The request is too large for OriginShape" if status == 413 else "OriginShape could not run the request"
            error = ProviderError(f"{what} (HTTP {status}): {message}", status=status, retriable=error.retriable,
                                  kind="http")
        error.code, error.data = code, data
        return error

    def _exhausted(self, kind: str, error, attempts: int) -> Exception:
        final = ChatLoop._exhausted(kind, error, attempts)
        if kind != "exc":
            return final
        where = host_of(self._api) if self._api else "the server"
        return ProviderError(f"Cannot reach OriginShape ({where}): the request failed after {attempts} attempts "
                             f"({type(error).__name__}). Check your internet connection and try again.",
                             retriable=True, kind="transport", code="unreachable")
