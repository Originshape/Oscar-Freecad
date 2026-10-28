"""The OriginShape session of one API URL: who is signed in, and a valid access token on demand.

``get_session(api_url=None)`` returns the process-wide ``AuthSession`` of that API URL (default
``config.api_url()``). A session is either an OAuth sign-in (a refresh token in the keystore, a short-lived access
token in memory only) or a personal API key (in the keystore, sent as the bearer token).

* ``access_token()`` returns the in-memory token while it has more than ``REFRESH_MARGIN_S`` to live. Otherwise it
  refreshes, single-flight: one thread renews and the others wait for that one result. The rotated refresh token is
  saved through the keystore (an atomic replace). ``invalid_grant`` means the server ended the grant (revoked,
  expired, or a reused token): the sign-in is forgotten and ``SignedOut`` raised. Network trouble raises
  ``AuthUnavailable`` and keeps the session.
* ``invalidate(rejected)`` after a 401 forgets the access token, so the next ``access_token()`` refreshes.
* ``complete_sign_in(tokens)`` (after ``LoopbackSignIn`` / ``DeviceSignIn``), ``use_api_key(key)`` (checked with
  ``/v1/me`` before it is kept), ``sign_out()`` (best-effort revocation with a short timeout, then forget).
* ``account(force=False)`` is the ``/v1/me`` snapshot, cached for ``SNAPSHOT_TTL_S``; a 401 refreshes and retries
  once. ``cached_snapshot()`` never touches the network.

The module-level ``on_change`` hook (a callable or None, the same shape as ``netguard.on_permission_wait``) is
called with one JSON string whenever the sign-in state or the snapshot changes. It is called on whichever thread
made the change and never under a lock; an exception in it is swallowed. Assign it on THIS module
(``oscar.auth.session.on_change = hub.changed.emit``).

Network work blocks, so it belongs on a worker thread. ``has_session()`` and ``cached_snapshot()`` are local; the
first ``has_session()`` reads the keystore once (on macOS a short ``security`` subprocess).
"""

import json
import threading
import time
from typing import Any, Callable

from .. import turnlog as _turnlog
from . import config, discovery, keystore
from .account import AccountSnapshot, parse_me
from .keystore import KeystoreError, StoredCredential
from .transport import (CLIENT_ID, SUSPENDED_MESSAGE, AuthError, Reply, Saved, SignedOut, TimeoutSpec, TokenSet,
                        api_error, clean_token, download, is_unavailable, oauth_error, request, revoke_quietly,
                        says_suspended, token_set, unavailable)

on_change: Callable[[str], Any] | None = None

REFRESH_MARGIN_S = 60.0       # renew this long before expiry, so a request never leaves with a dying token
SNAPSHOT_TTL_S = 30.0         # account(force=False) reuses a snapshot this young
ME_TIMEOUT: tuple[float, float] = (5.0, 8.0)
REFRESH_TIMEOUT: tuple[float, float] = (5.0, 15.0)
DOWNLOAD_TIMEOUT: tuple[float, float] = (5.0, 60.0)   # read timeout: the longest wait between two chunks

_SESSIONS: dict[str, "AuthSession"] = {}
_SESSIONS_LOCK = threading.Lock()


def get_session(api_url: str | None = None) -> "AuthSession":
    """The session of ``api_url`` (default ``config.api_url()``, read now). Cheap: no I/O until it is used."""
    url = config.normalise_url(api_url) if api_url else config.api_url()
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(url)
        if session is None:
            session = _SESSIONS[url] = AuthSession(url)
        return session


def reset_sessions() -> None:
    """Forget every in-memory session (tests). Stored sign-ins are untouched."""
    with _SESSIONS_LOCK:
        _SESSIONS.clear()


def _notify(payload: dict) -> None:
    hook = on_change
    if hook is None:
        return
    try:
        hook(json.dumps(payload))
    except Exception:  # noqa: BLE001 - a closed window must never break signing in or out
        pass


def _log(event: str, **fields: Any) -> None:
    _turnlog.log_event("auth", event=event, **fields)


class AuthSession:
    """One API URL's sign-in. Thread-safe; see the module docstring."""

    def __init__(self, api_url: str):
        self.api_url = config.normalise_url(api_url)
        self.last_store_error = ""   # why the last keystore write failed ("" = it worked), for the UI
        self._state = threading.Lock()        # guards the fields below; never held during I/O or callbacks
        self._refreshing = threading.Lock()   # single flight: one refresh at a time
        self._storing = threading.Lock()      # one keystore write at a time, always of the latest state
        self._loaded = False
        self._cred: StoredCredential | None = None
        self._access = ""
        self._access_expires = 0.0            # time.monotonic()
        self._margin = REFRESH_MARGIN_S
        self._snapshot: AccountSnapshot | None = None
        self._snapshot_at = 0.0               # time.monotonic() of the last /v1/me
        self._generation = 0                  # bumped by every sign-in and sign-out: late results of an older
                                              # session (a refresh that finishes after sign-out) are dropped
        self._ended = "expired"               # why the server last ended the grant: "expired" or "suspended"

    def __repr__(self) -> str:
        with self._state:
            kind = self._cred.kind if self._cred is not None else ("oauth" if self._access else "")
        return f"AuthSession(api_url={self.api_url!r}, kind={kind!r})"

    # -- local state (no network) -------------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        with self._state:
            if self._loaded:
                return
            generation = self._generation
        failure = ""
        try:
            stored = keystore.load(self.api_url)
        except KeystoreError as e:
            stored, failure = None, e.code
        with self._state:
            if not self._loaded and self._generation == generation:   # no sign-in/out happened meanwhile
                self._cred, self._loaded = stored, True
        if failure:
            _log("store", outcome=failure)

    def has_session(self) -> bool:
        """True when signed in (OAuth or API key). Local: it never asks the server."""
        self._ensure_loaded()
        with self._state:
            return self._cred is not None or bool(self._access)

    @property
    def kind(self) -> str:
        """``"oauth"``, ``"api_key"`` or ``""`` (signed out)."""
        self._ensure_loaded()
        with self._state:
            if self._cred is not None:
                return self._cred.kind
            return "oauth" if self._access else ""

    @property
    def email(self) -> str:
        """The signed-in email as last seen (snapshot or stored), ``""`` when unknown."""
        self._ensure_loaded()
        with self._state:
            if self._snapshot is not None and self._snapshot.email:
                return self._snapshot.email
            return self._cred.email if self._cred is not None else ""

    def cached_snapshot(self) -> AccountSnapshot | None:
        """The last ``/v1/me`` snapshot of this sign-in, however old, or None. No network."""
        with self._state:
            return self._snapshot

    # -- tokens -------------------------------------------------------------------------------------------

    def _valid_token(self) -> str:
        with self._state:
            if self._cred is not None and self._cred.kind == "api_key":
                return self._cred.secret
            if self._access and time.monotonic() < self._access_expires - self._margin:
                return self._access
        return ""

    def _set_access(self, tokens: TokenSet) -> None:
        """Under ``_state``: keep the access token. A very short lifetime renews at half-life, not at -60 s, so a
        server with a tiny TTL cannot cause a refresh per request."""
        self._access = tokens.access_token
        self._access_expires = tokens.expires_at
        self._margin = min(REFRESH_MARGIN_S, tokens.expires_in / 2.0)

    def access_token(self) -> str:
        """A bearer token valid for at least ``REFRESH_MARGIN_S`` (or the API key). May refresh (blocking).
        Raises ``SignedOut``, ``AuthUnavailable`` (session kept) or ``AuthError``."""
        self._ensure_loaded()
        token = self._valid_token()
        if token:
            return token
        with self._refreshing:
            token = self._valid_token()   # the thread we waited for has just renewed it
            if token:
                return token
            with self._state:
                cred, generation, had_token = self._cred, self._generation, bool(self._access)
            if cred is None and not had_token:
                raise SignedOut("You are not signed in to OriginShape.")
            if cred is not None:
                with keystore.refresh_lock(self.api_url):   # another FreeCAD may share this stored sign-in
                    cred = self._adopt_stored(cred, generation)
                    tokens = self._refresh(cred, generation)
                    if tokens is None:   # refused: a FreeCAD that renewed meanwhile left its rotated token in the store
                        newer = self._adopt_stored(cred, generation)
                        if newer is not cred:
                            tokens = self._refresh(newer, generation)
                if tokens is not None:
                    return tokens.access_token
            # The grant ended (invalid_grant), or a sign-in without a refresh token ran out. Dropped from memory
            # while the refresh lock is still held, so the threads queued behind this one find "signed out" instead
            # of sending the dead refresh token again.
            dropped = self._drop(generation)
            if not dropped:
                token = self._valid_token()   # a new sign-in arrived while the old grant was failing: use it
                if token:
                    return token
        if dropped:   # the keystore and on_change OUTSIDE the lock: on_change may call straight back in
            self._announce_signed_out(self._ended)
        if self._ended == "suspended":
            raise SignedOut(SUSPENDED_MESSAGE, code="suspended")
        raise SignedOut("Your OriginShape sign-in has expired or was revoked. Please sign in again.")

    def auth_header(self) -> dict[str, str]:
        """``{"Authorization": "Bearer <token>"}`` (see ``access_token``)."""
        return {"Authorization": f"Bearer {self.access_token()}"}

    def invalidate(self, rejected: str | None = None) -> bool:
        """After a 401: treat the in-memory access token as expired, so the next ``access_token()`` refreshes (or,
        without a refresh token, signs out and says so). With ``rejected`` (the token the server refused), a token
        another thread has already renewed is kept. True when a refresh can help (an OAuth sign-in with a refresh
        token), False for an API key or no session."""
        with self._state:
            if rejected is None or rejected == self._access:
                self._access_expires = 0.0
            return self._cred is not None and self._cred.kind == "oauth"

    def _adopt_stored(self, cred: StoredCredential, generation: int) -> StoredCredential:
        """The stored OAuth credential when another process has rotated it since this one read it (the store holds a
        different refresh token), else ``cred``. Sending our older, already used token would make the server end the
        sign-in for every FreeCAD."""
        try:
            stored = keystore.load(self.api_url)
        except KeystoreError:
            return cred
        if stored is None or stored.kind != "oauth" or not stored.secret or stored.secret == cred.secret:
            return cred
        with self._state:
            if self._generation != generation or self._cred is not cred:
                return cred
            self._cred = stored
        _log("refresh", outcome="adopted_stored")
        return stored

    def _refresh(self, cred: StoredCredential, generation: int) -> TokenSet | None:
        """Renew with the refresh token (under ``_refreshing``). None when the server ended the grant
        (``invalid_grant``); the caller forgets the sign-in."""
        meta = discovery.fetch(self.api_url)
        reply = request("POST", meta.token_endpoint, timeout=REFRESH_TIMEOUT,
                        data={"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": cred.secret})
        if reply.status != 200:
            code, _desc = oauth_error(reply, cred.secret)
            _log("refresh", outcome=code or f"http_{reply.status}")
            if code == "invalid_grant":
                self._ended = "suspended" if says_suspended(reply) else "expired"
                return None
            if is_unavailable(reply.status):
                raise unavailable(reply)
            raise AuthError(f"OriginShape did not renew the sign-in ({code or f'HTTP {reply.status}'}).",
                            code=code or "refresh_failed", status=reply.status)
        tokens = token_set(reply, cred.secret)
        rotated = StoredCredential("oauth", tokens.refresh_token, cred.email) if tokens.refresh_token else cred
        with self._state:
            current = self._generation == generation
            if current:
                self._cred = rotated
                self._set_access(tokens)
        if not current:   # signed out (or in again) while renewing: this grant's new token must not linger
            revoke_quietly(meta.revocation_endpoint, tokens.refresh_token)
            raise SignedOut("You are no longer signed in to OriginShape.")
        if rotated is not cred:
            self._sync_store()
        _log("refresh", outcome="ok")
        return tokens

    # -- signing in and out -------------------------------------------------------------------------------

    def complete_sign_in(self, tokens: TokenSet, *, email: str = "") -> None:
        """Adopt the tokens of a finished browser or code sign-in. The refresh token is stored (when the server sent
        one); the access token stays in memory. Call it on the worker that finished the flow: the keystore may run
        a short subprocess."""
        clean_token(tokens.access_token, "access token")
        refresh = clean_token(tokens.refresh_token, "refresh token") if tokens.refresh_token else ""
        with self._state:
            self._generation += 1
            self._loaded = True
            self._cred = StoredCredential("oauth", refresh, email) if refresh else None
            self._set_access(tokens)
            self._snapshot, self._snapshot_at = None, 0.0
        self._sync_store()
        _log("signed_in", method="oauth")
        self._changed("signed_in")

    def use_api_key(self, key: str) -> AccountSnapshot:
        """Sign in with a personal API key (``os_live_...``). It is checked with ``GET /v1/me`` first and kept only
        when the server accepts it. Returns the snapshot. Raises ``AuthError`` (``code="invalid_key"``) for a key the
        server refuses, ``AuthUnavailable`` when it cannot be checked."""
        message = "That is not an OriginShape API key (they start with os_live_)."
        try:
            key = clean_token(key, "API key")
        except AuthError:
            key = ""   # raised below, outside the handler: one code for every malformed key, nothing chained
        if not key.startswith("os_live_"):   # the server reads only os_live_ as a key (security/tokens.py); any other
            raise AuthError(message, code="invalid_key")   # os_ value would be checked as a JWT and refused
        reply = request("GET", self.api_url + "/v1/me", timeout=ME_TIMEOUT, headers={"Authorization": f"Bearer {key}"})
        if reply.status == 401:
            raise AuthError("OriginShape did not accept this API key. Check it on the website, or create a new one.",
                            code="invalid_key", status=401)
        if reply.status != 200:
            raise api_error(reply, "the account of this API key", key)
        snapshot = parse_me(reply.json())
        with self._state:
            self._generation += 1
            self._loaded = True
            self._cred = StoredCredential("api_key", key, snapshot.email)
            self._access, self._access_expires = "", 0.0
            self._snapshot, self._snapshot_at = snapshot, time.monotonic()
        self._sync_store()
        _log("signed_in", method="api_key")
        self._changed("signed_in")
        return snapshot

    def sign_out(self) -> None:
        """Revoke the refresh token (best effort, ``transport.REVOKE_TIMEOUT``), then forget the sign-in here and in
        the keystore. An API key is only forgotten: keys are revoked on the website."""
        self._ensure_loaded()
        with self._state:
            cred = self._cred
        if cred is not None and cred.kind == "oauth":
            meta = discovery.cached(self.api_url)   # no slow discovery on the way out
            endpoint = meta.revocation_endpoint if meta is not None else self.api_url + "/oauth/revoke"
            revoked = revoke_quietly(endpoint, cred.secret)
            _log("revoke", outcome="ok" if revoked else "failed")
        self._forget(None, reason="signed_out")

    def _forget(self, generation: int | None, *, reason: str) -> bool:
        """Drop the sign-in (only if ``generation`` is still current, when given), clear the keystore and tell
        ``on_change``. False when a newer sign-in or sign-out got there first."""
        if not self._drop(generation):
            return False
        self._announce_signed_out(reason)
        return True

    def _drop(self, generation: int | None) -> bool:
        """The in-memory half of ``_forget``: no I/O, no callback, so it may run under ``_refreshing``."""
        with self._state:
            if generation is not None and generation != self._generation:
                return False
            self._generation += 1
            self._loaded = True
            self._cred = None
            self._access, self._access_expires = "", 0.0
            self._snapshot, self._snapshot_at = None, 0.0
        return True

    def _announce_signed_out(self, reason: str) -> None:
        """The rest of ``_forget``: make the keystore match (empty), log, tell ``on_change``. Never under a lock."""
        self._sync_store()
        _log("signed_out", reason=reason)
        self._changed("signed_out", reason=reason)

    def _sync_store(self) -> None:
        """Make the keystore match memory. The state is read INSIDE ``_storing``, so whichever write runs last wrote
        the newest state: concurrent sign-in, rotation and sign-out cannot leave an older entry behind."""
        failure = ""
        with self._storing:
            with self._state:
                cred = self._cred
            try:
                if cred is None:
                    keystore.delete(self.api_url)
                else:
                    keystore.save(self.api_url, cred)
            except KeystoreError as e:
                failure = str(e)
            self.last_store_error = failure
        if failure:
            _log("store", outcome="failed")

    # -- the account --------------------------------------------------------------------------------------

    def authorized_request(self, method: str, path: str, *, timeout: TimeoutSpec = ME_TIMEOUT,
                           json_body: Any = None, content: bytes | None = None,
                           content_type: str = "application/octet-stream") -> Reply:
        """``method {api_url}{path}`` with this session's bearer token. A 401 forgets the access token and retries
        once with a renewed one. A second 401, or a 401 for an API key, means the server no longer accepts this
        sign-in: it is forgotten (a newer sign-in made meanwhile is kept) and ``SignedOut`` raised. Any other status
        is returned as it is. ``content`` uploads raw bytes (``transport.request``)."""
        if not path.startswith("/"):
            raise ValueError("path must start with '/'")
        url = self.api_url + path
        body = {"json_body": json_body, "content": content, "content_type": content_type}
        return self._with_bearer(lambda auth: (request(method, url, timeout=timeout, headers=auth, **body), None))[0]

    def authorized_download(self, path: str, dest: str, *, max_bytes: int, sha256: str = "",
                            timeout: TimeoutSpec = DOWNLOAD_TIMEOUT) -> tuple[Reply, Saved | None]:
        """``GET {api_url}{path}`` into the file ``dest`` (``transport.download``), with the same 401 handling as
        ``authorized_request``."""
        if not path.startswith("/"):
            raise ValueError("path must start with '/'")
        url = self.api_url + path
        return self._with_bearer(lambda auth: download(url, dest, max_bytes=max_bytes, sha256=sha256,
                                                       timeout=timeout, headers=auth))

    def _with_bearer(self, send: Callable[[dict[str, str]], tuple[Reply, Any]]) -> tuple[Reply, Any]:
        """``send(headers)`` with this session's bearer token; it returns ``(reply, anything)``. A 401 forgets the
        access token and sends once more with a renewed one; a second 401 forgets the sign-in and raises
        ``SignedOut``."""
        with self._state:   # read BEFORE the request: a sign-in made while it is on its way must survive its 401
            generation = self._generation
        token = self.access_token()
        result = send({"Authorization": f"Bearer {token}"})
        if result[0].status != 401:
            return result
        if self.invalidate(token):
            token = self.access_token()   # SignedOut here when the grant itself is gone
            result = send({"Authorization": f"Bearer {token}"})
            if result[0].status != 401:
                return result
        self._forget(generation, reason="rejected")
        raise SignedOut("OriginShape no longer accepts this sign-in. Please sign in again.", status=401)

    def account(self, force: bool = False) -> AccountSnapshot:
        """The ``/v1/me`` snapshot: the cached one when younger than ``SNAPSHOT_TTL_S`` (unless ``force``), else
        fetched. Raises ``SignedOut``, ``AuthUnavailable`` (session kept) or ``AuthError``."""
        with self._state:
            snapshot, at = self._snapshot, self._snapshot_at
        if snapshot is not None and not force and time.monotonic() - at < SNAPSHOT_TTL_S:
            return snapshot
        self._ensure_loaded()
        with self._state:
            generation = self._generation
        reply = self.authorized_request("GET", "/v1/me", timeout=ME_TIMEOUT)
        if reply.status != 200:
            raise api_error(reply, "the account")
        snapshot = parse_me(reply.json())
        with self._state:
            if self._generation != generation:   # signed out, or in as someone else, while this was loading
                raise AuthError("The OriginShape sign-in changed while the account was loading. Try again.",
                                code="changed")
            previous = self._snapshot
            self._snapshot, self._snapshot_at = snapshot, time.monotonic()
            cred = self._cred
            renamed = cred is not None and bool(snapshot.email) and snapshot.email != cred.email
            if renamed:   # keep the email with the stored sign-in, so it can be shown before the next fetch
                self._cred = StoredCredential(cred.kind, cred.secret, snapshot.email)
        if renamed:
            self._sync_store()
        if not snapshot.same_state(previous):
            self._changed("account")
        return snapshot

    def _changed(self, event: str, **extra: Any) -> None:
        with self._state:
            snapshot, cred = self._snapshot, self._cred
            kind = cred.kind if cred is not None else ("oauth" if self._access else "")
            email = snapshot.email if snapshot is not None and snapshot.email else (cred.email if cred else "")
        _notify({"event": event, "api_url": self.api_url, "signed_in": bool(kind), "kind": kind, "email": email,
                 "snapshot": snapshot.to_dict() if snapshot is not None else None, **extra})
