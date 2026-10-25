"""The server's OAuth metadata (RFC 8414): ``GET {api_url}/.well-known/oauth-authorization-server``.

It decides where tokens go, so it is validated before use:

* ``issuer`` must equal the API URL (both normalised).
* The token, device authorization, revocation and JWKS endpoints must be on the API URL's own origin. A refresh
  token, a device code or a PKCE verifier is only ever sent to our API, whatever the document says.
* The authorization endpoint may be on another origin (it is the web app, opened in the browser), but it must be
  https, or http on a loopback host.
* The ``originshape`` links (manage, billing, download ...) are kept only when they pass the same browser rule.

Answers are cached per API URL for an hour. ``cached()`` never touches the network, which is what
``LoopbackSignIn.prepare`` (GUI thread) uses; ``fetch()`` blocks for up to ``TIMEOUT_S``.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any

from . import config, transport
from .transport import InvalidServerURL, ServerMetadataError

WELL_KNOWN_PATH = "/.well-known/oauth-authorization-server"
CACHE_TTL_S = 3600.0
TIMEOUT_S = 8.0

_LOCK = threading.Lock()
_CACHE: dict[str, "ServerMetadata"] = {}
_clock = time.monotonic   # tests replace it to age the cache


@dataclass(frozen=True)
class ServerMetadata:
    """The validated metadata of one API URL. Optional endpoints the server does not offer are ``""``."""

    api_url: str
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    device_authorization_endpoint: str = ""
    revocation_endpoint: str = ""
    jwks_uri: str = ""
    app_origin: str = ""
    gateway_base_url: str = ""
    manage_url: str = ""
    billing_url: str = ""
    download_url: str = ""
    fetched_at: float = 0.0   # time.monotonic() of the fetch


def _base(api_url: str | None) -> str:
    return config.normalise_url(api_url) if api_url else config.api_url()


def cached(api_url: str | None = None) -> ServerMetadata | None:
    """The cached metadata when it is younger than ``CACHE_TTL_S``, else None. Never touches the network."""
    base = _base(api_url)
    with _LOCK:
        meta = _CACHE.get(base)
    if meta is not None and _clock() - meta.fetched_at < CACHE_TTL_S:
        return meta
    return None


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def fetch(api_url: str | None = None, *, force: bool = False) -> ServerMetadata:
    """The metadata of ``api_url`` (default ``config.api_url()``): from the cache unless ``force``, else fetched
    (``TIMEOUT_S`` timeout) and validated. Raises ``AuthUnavailable`` (network, 429, 5xx) or
    ``ServerMetadataError`` (not an OriginShape server, or metadata Oscar must not trust)."""
    base = _base(api_url)
    if not force:
        hit = cached(base)
        if hit is not None:
            return hit
    reply = transport.request("GET", base + WELL_KNOWN_PATH, timeout=TIMEOUT_S)
    if transport.is_unavailable(reply.status):
        raise transport.unavailable(reply)
    if reply.status != 200:
        raise ServerMetadataError(f"{transport.host_of(base)} did not answer like an OriginShape server (HTTP "
                                  f"{reply.status} for its sign-in metadata). Check the server URL.",
                                  status=reply.status)
    meta = validate(base, reply.json(), fetched_at=_clock())
    with _LOCK:
        _CACHE[base] = meta
    return meta


_LABELS = {"token_endpoint": "token requests", "device_authorization_endpoint": "sign-in codes",
           "revocation_endpoint": "sign-out requests", "jwks_uri": "its signing keys"}


def _api_endpoint(api_url: str, data: dict, key: str, *, required: bool = False) -> str:
    value = data.get(key)
    if value in (None, ""):
        if required:
            raise ServerMetadataError(f"The server's sign-in metadata has no {key}.")
        return ""
    url = ""
    if isinstance(value, str):
        try:
            url = config.browser_url(value, what=key)
        except InvalidServerURL:
            url = ""
    if not url or not config.same_origin(url, api_url):
        raise ServerMetadataError(f"The server's sign-in metadata sends {_LABELS[key]} to another server; Oscar only "
                                  f"sends tokens to {transport.host_of(api_url)}.")
    return url


def _link(extra: dict, key: str) -> str:
    value = extra.get(key)
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        return config.browser_url(value, what=key)
    except InvalidServerURL:
        return ""


def validate(api_url: str, data: Any, *, fetched_at: float = 0.0) -> ServerMetadata:
    """``ServerMetadata`` from a decoded document, or ``ServerMetadataError`` (see the module docstring)."""
    if not isinstance(data, dict):
        raise ServerMetadataError("The server's sign-in metadata is not a JSON object.")
    issuer = ""
    if isinstance(data.get("issuer"), str):
        try:
            issuer = config.normalise_url(data["issuer"], what="issuer")
        except InvalidServerURL:
            issuer = ""
    if issuer != api_url:
        raise ServerMetadataError("The server's sign-in metadata names a different issuer than the server URL, so "
                                  "Oscar will not sign in there.")
    token = _api_endpoint(api_url, data, "token_endpoint", required=True)
    device = _api_endpoint(api_url, data, "device_authorization_endpoint")
    revoke = _api_endpoint(api_url, data, "revocation_endpoint")
    jwks = _api_endpoint(api_url, data, "jwks_uri")
    authorize = data.get("authorization_endpoint")
    authorize_url = ""
    if isinstance(authorize, str):
        try:
            authorize_url = config.browser_url(authorize, what="authorization endpoint")
        except InvalidServerURL:
            authorize_url = ""
    if not authorize_url:
        raise ServerMetadataError("The server's sign-in metadata has no usable authorization endpoint (it must be "
                                  "https).")
    methods = data.get("code_challenge_methods_supported")
    if isinstance(methods, list) and "S256" not in methods:
        raise ServerMetadataError("The server does not offer PKCE with S256, which Oscar requires to sign in.")
    extra = data.get("originshape") if isinstance(data.get("originshape"), dict) else {}
    gateway = _link(extra, "gateway_base_url")
    if gateway and not config.same_origin(gateway, api_url):
        gateway = ""   # informational only: the provider always uses its own path on the API URL
    return ServerMetadata(
        api_url=api_url, issuer=issuer, authorization_endpoint=authorize_url, token_endpoint=token,
        device_authorization_endpoint=device, revocation_endpoint=revoke, jwks_uri=jwks,
        app_origin=_link(extra, "app_origin"), gateway_base_url=gateway, manage_url=_link(extra, "manage_url"),
        billing_url=_link(extra, "billing_url"), download_url=_link(extra, "download_url"), fetched_at=fetched_at)
