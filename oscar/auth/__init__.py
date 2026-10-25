"""OriginShape sign-in for Oscar.

Pure Python: the standard library and ``requests``, no Qt. ``oscar.params``, which imports FreeCAD, is imported
lazily and only to read the Server preference and to find the settings folder.

* ``config``: ``api_url()``, from ``OSCAR_API_URL``, the ``OriginShapeServer`` preference (only while the
  ``OriginShapeCustomServer`` switch is on), or the public API.
* ``discovery``: the server's OAuth metadata, validated so tokens only ever go to the API's own origin.
* ``pkce``: the verifier, the S256 challenge and the state.
* ``loopback``: ``LoopbackSignIn``, the browser sign-in with a ``127.0.0.1`` redirect.
* ``device``: ``DeviceSignIn``, sign-in with a code (RFC 8628).
* ``keystore``: where the sign-in is kept (the OS keychain, a 0600 file, or memory).
* ``session``: ``AuthSession`` / ``get_session()``, with access tokens on demand, API keys, sign-out and ``/v1/me``.
* ``account``: ``AccountSnapshot``, ``fetch_me()``, ``fetch_models()``, ``cached_model()``, and
  ``plugin_update()`` (this Oscar against the server's latest and minimum versions).
* ``transport``: the shared HTTP door, the error types, ``TokenSet``, and ``error_details()`` /
  ``LLM_NOT_CONFIGURED_MESSAGE`` for the API's error envelope.

The Qt side learns about changes through ``oscar.auth.session.on_change`` (a callable taking one JSON string, or
None). Assign it on that module: a name imported from here would be a copy the session never reads.
"""

from . import keystore
from .account import LINEUP, AccountSnapshot, ModelCatalog, ModelInfo, cached_model, fetch_me, fetch_models
from .config import DEFAULT_API_URL, api_url, normalise_url
from .device import DeviceCode, DeviceSignIn
from .discovery import ServerMetadata
from .keystore import KeystoreError, KeystoreUnavailable, StoredCredential
from .loopback import LoopbackSignIn
from .session import AuthSession, get_session
from .transport import (CLIENT_ID, LLM_NOT_CONFIGURED, LLM_NOT_CONFIGURED_MESSAGE, SCOPE, AuthError, AuthUnavailable,
                        InvalidServerURL, ServerMetadataError, SignedOut, SignInCancelled, SignInDeclined, SignInExpired,
                        SignInTimedOut, TokenSet, error_details)

__all__ = [
    "AccountSnapshot", "AuthError", "AuthSession", "AuthUnavailable", "CLIENT_ID", "DEFAULT_API_URL", "DeviceCode",
    "DeviceSignIn", "InvalidServerURL", "KeystoreError", "KeystoreUnavailable", "LINEUP", "LLM_NOT_CONFIGURED",
    "LLM_NOT_CONFIGURED_MESSAGE", "LoopbackSignIn", "ModelCatalog", "ModelInfo", "SCOPE",
    "ServerMetadata", "ServerMetadataError", "SignInCancelled", "SignInDeclined", "SignInExpired", "SignInTimedOut",
    "SignedOut", "StoredCredential", "TokenSet", "api_url", "cached_model", "error_details", "fetch_me", "fetch_models",
    "get_session", "keystore", "normalise_url",
]
