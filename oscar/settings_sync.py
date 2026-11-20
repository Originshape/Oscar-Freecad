"""Oscar's AI model settings, kept in step with the OriginShape account (the website's Settings → AI model; issue #86,
contract ``docs/architecture/ai-settings-sync.md`` in the app repository). Pure Python: the sync (``Syncer``, run on a
daemon thread by ``request``) and the Test connection the website can ask for (``run_profile_test``). The pieces:

* ``settings_wire``: profile ids, the active profile, the gate that says whether a profile may be used
  (``unusable_reason``) and the wire format: what OriginShape sees of a profile, never a key or a piece of one;
* ``settings_merge``: the three-way merge, how a pull is applied here, and the settings dialog's Save;
* ``keyguard``: the one policy for keys and pieces of keys in text that leaves this computer.

The names other modules use are re-exported here.

**When** (``request``; signed in with OriginShape only, never with a personal API key; signed out nothing is sent):
when Oscar's panel starts, after a sign-in, when the settings dialog opens, after a Save there and a profile picked in
the panel (``prefer="local"``), before each chat turn when the last pull is more than a minute old (waiting for it up
to two seconds: ``pull_before_turn``), and before the website's Test in FreeCAD runs. A sync asked for while one runs
runs once more after it. A failure is logged (never with a settings value) and the next trigger tries again; it never
blocks a chat or the dialog for longer than that wait. ``OSCAR_SETTINGS_SYNC=0`` turns all of it off.

**One sync.** ``GET /v1/me/ai-settings``; in one locked transaction of the settings file, ``settings_merge.plan_sync``
merges and writes; then ``PUT`` with the version it merged against when the result differs from the server's. A 409
is merged again with the server's current settings and pushed once more. A 400 is shown with the field the server
names. A pull that would remove more than half of this computer's profiles is not applied (``blocked``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit

from . import keyguard as _keyguard
from .auth.transport import AuthError, AuthUnavailable, SignedOut, api_error, error_details, safe_text
# Re-exported: the names params, settings_ui, panel, providers, provider_probe, connect_gui and the tests use.
from .settings_merge import (DERIVED_FIELDS, Entry, Merged, Plan, adopt_remote_ids, apply_merge,  # noqa: F401
                             confirm_push, merge, merge_saved, plan_sync, settle_pending, stored_from_wire,
                             usable_snapshot)
from .settings_wire import (ACCOUNT_PROVIDER, MAX_PROFILES, NEEDS_CONFIRMATION_TEXT, OVERRIDE_KEYS,  # noqa: F401
                            PENDING_KEY, PROVIDERS, STATE_KEY, WIRE_FIELDS, active_index, approval_for, display_host,
                            endpoint, endpoint_of, ensure_ids, field_of, local_secrets, local_settings, masked_fields,
                            needs_confirmation, new_profile_id, normalise_remote, overrides_of, pending_active,
                            pending_text, publishable_url, same_settings, set_active, syncable, to_wire,
                            unusable_reason, valid_id, wire_profile)

ROUTE = "/v1/me/ai-settings"
ENV = "OSCAR_SETTINGS_SYNC"      # "0": never sync (and don't offer the website's Test in FreeCAD)
PULL_AGE_S = 60.0                # before a chat turn: pull when the last pull is older than this
TURN_WAIT_S = 2.0                # ... and wait for it this long at most
REQUEST_TIMEOUT: tuple[float, float] = (5.0, 15.0)
TEST_MESSAGE_LIMIT = 500
TEST_TIMEOUT_S = 150.0           # the server expires a test 3 minutes after FreeCAD claimed it
TEST_PULL_WAIT_S = 5.0           # before the website's test: the latest settings, waited for this long at most
UNKNOWN_PROFILE_TEXT = ("This profile isn't in Oscar's settings in FreeCAD (it was removed there, or FreeCAD hasn't "
                        "synced it yet). Open Oscar's settings in FreeCAD, then try again.")

on_change: Callable[[str], Any] | None = None   # (any thread) JSON ``{"event": "status" | "changed", ...}``


def account_tag(api_url: str, email: str) -> str:
    """Which account a snapshot belongs to, without keeping the email in the settings file."""
    return hashlib.sha256(f"{api_url}\n{(email or '').strip().casefold()}".encode("utf-8")).hexdigest()[:32]


def _log(kind: str, **fields: Any) -> None:
    try:
        from . import turnlog

        turnlog.log_event(kind, **fields)
    except Exception:  # noqa: BLE001 - logging never breaks a sync
        pass


def _default_session() -> Any:
    from .auth import config, session

    try:
        return session.get_session(config.api_url())
    except Exception:  # noqa: BLE001 - an unusable server URL: nothing to sync with
        return None


class ParamsStore:
    """The settings file (``oscar.params``), read and written in one locked transaction."""

    def transaction(self, mutate: Callable[[dict], tuple[bool, Any]]) -> Any:
        from . import params

        return params.update_state(mutate)

    def limits(self, profile: dict) -> tuple[float, int]:
        from types import SimpleNamespace

        from . import params

        return params.effective_limits(SimpleNamespace(**profile))

    def profile(self, profile_id: str) -> Any:
        from . import params

        return params.profile_by_id(profile_id)


class SyncConflict(AuthError):
    code = "conflict"


def refusal(reply, payload: dict) -> tuple[str, str, str]:
    """``(code, field, text)`` of a 400 to a push: the server's ``details`` name the field (``profiles.1.base_url``)
    and say what is wrong, never with a value. ``text`` names the profile by its name for the dialog."""
    code, description = error_details(reply.body)
    data = reply.json()
    error = data.get("error") if isinstance(data, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    first = details[0] if isinstance(details, list) and details and isinstance(details[0], dict) else {}
    field_name = first.get("field") if isinstance(first.get("field"), str) else ""
    message = safe_text(first.get("message"), limit=300) or description or f"HTTP {reply.status}"
    where = field_name
    match = re.fullmatch(r"profiles\.(\d+)(?:\.(.+))?", field_name)
    profiles = payload.get("profiles") or []
    if match and int(match.group(1)) < len(profiles):
        name = profiles[int(match.group(1))].get("name") or "a profile"
        where = f"the profile “{name}”" + (f", {match.group(2).replace('_', ' ')}" if match.group(2) else "")
    elif field_name:
        where = field_name.replace("_", " ")
    text = f"{where}: {message}" if where else message
    return code, field_name[:120], text if text.endswith(".") else text + "."


def _conflict_current(reply) -> dict | None:
    """``details.current`` of a 409 (in the error envelope, or beside it)."""
    data = reply.json()
    if not isinstance(data, dict):
        return None
    for holder in (data.get("error"), data):
        details = holder.get("details") if isinstance(holder, dict) else None
        if isinstance(details, dict) and isinstance(details.get("current"), dict):
            return details["current"]
    return None


_STATUS_TEXT = {
    "signed_out": "Not synced: signed out.",
    "api_key": "Not synced: an OriginShape API key can't sync settings. Sign in with OriginShape instead.",
}


def _local_only_text(plan: Plan) -> str:
    """What stays on this computer only after a sync (a provider OriginShape doesn't know, past the 50 it keeps)."""
    notes = []
    if plan.unknown_provider:
        n = plan.unknown_provider
        notes.append(f"{n} profile{'' if n == 1 else 's'} with a provider OriginShape doesn't know stay"
                     f"{'s' if n == 1 else ''} on this computer only.")
    others = plan.local_only - plan.unknown_provider
    if others > 0:
        notes.append(f"OriginShape keeps {MAX_PROFILES} profiles: {others} more stay"
                     f"{'s' if others == 1 else ''} on this computer only.")
    return " ".join(notes)


class Syncer:
    """Syncs this computer's AI model settings with the account (see the module docstring). ``sync`` is blocking and
    never raises; ``status`` says how the last one went."""

    def __init__(self, *, session_for: Callable[[], Any] | None = None, store: Any = None,
                 notify: Callable[[str, dict], None] | None = None, log: Callable[..., None] | None = None) -> None:
        self._session_for = session_for or _default_session
        self.store = store or ParamsStore()
        self._notify = notify or _notify_hook
        self._log = log or _log
        self._lock = threading.Lock()
        self.last_pull = float("-inf")       # time.monotonic() of the last pull that worked
        self.last_attempt = float("-inf")    # ... and of the last sync started, whatever came of it
        self.status: dict = {"state": "idle", "message": "", "at": 0.0, "version": 0}

    def _say(self, state: str, message: str = "", **extra: Any) -> dict:
        self.status = {"state": state, "message": message, "at": time.time(),
                       "version": extra.get("version", self.status.get("version", 0))}
        self._notify("status", dict(self.status))
        return dict(self.status)

    def sync(self, prefer: str = "remote", reason: str = "") -> dict:
        """One sync (``prefer``: ``"remote"`` for a pull, ``"local"`` after a Save). Returns ``status``."""
        with self._lock:
            self.last_attempt = time.monotonic()
            try:
                return self._sync(prefer, reason)
            except SignedOut:
                return self._say("signed_out", _STATUS_TEXT["signed_out"])
            except AuthUnavailable as e:
                self._log("ai_settings_sync", outcome="unavailable", reason=reason, status=e.status)
                return self._say("unavailable", f"Not synced: {e}")
            except AuthError as e:
                self._log("ai_settings_sync", outcome="failed", reason=reason, code=e.code, status=e.status)
                return self._say("error", f"Not synced: {e}")
            except OSError as e:
                self._log("ai_settings_sync", outcome="failed", reason=reason, error=type(e).__name__)
                return self._say("error", "Not synced: Oscar's settings file can't be read right now.")
            except Exception as e:  # noqa: BLE001 - a bug: its text could quote a setting, so only its type
                self._log("ai_settings_sync", outcome="failed", reason=reason, error=type(e).__name__)
                return self._say("error", f"Not synced: unexpected {type(e).__name__}.")

    def _session(self) -> Any:
        session = self._session_for()
        if session is None or not session.has_session():
            return None
        return session

    def _pull(self, session: Any) -> dict:
        reply = session.authorized_request("GET", ROUTE, timeout=REQUEST_TIMEOUT)
        if reply.status == 404:
            raise AuthError("the OriginShape server doesn't sync AI model settings yet.", code="not_supported",
                            status=404)
        if reply.status != 200:
            raise api_error(reply, "the AI model settings")
        try:
            remote = normalise_remote(reply.json())
        except ValueError:
            raise AuthError("OriginShape sent AI model settings Oscar can't read.", code="bad_response",
                            status=reply.status) from None
        self.last_pull = time.monotonic()
        return remote

    def _plan(self, remote: dict, prefer: str, account: str) -> Plan:
        plan = self.store.transaction(lambda state: plan_sync(state, remote, prefer=prefer, account=account,
                                                              limits=self.store.limits))
        if plan.masked and plan.push:   # names of the fields only, never their values
            self._log("ai_settings_masked", fields=sorted(plan.masked), profiles=max(plan.masked.values()))
        if plan.changed:
            self._notify("changed", {"version": remote["version"], "updated_by": remote.get("updated_by")})
        return plan

    def _sync(self, prefer: str, reason: str) -> dict:
        session = self._session()
        if session is None:
            return self._say("signed_out", _STATUS_TEXT["signed_out"])
        if session.kind != "oauth":
            return self._say("api_key", _STATUS_TEXT["api_key"])
        self._say("syncing", "Syncing with your OriginShape account…")
        email = session.email or session.account().email
        account = account_tag(session.api_url, email)
        remote = self._pull(session)
        plan = self._plan(remote, prefer, account)
        pulled = plan.changed
        for attempt in range(2):
            if plan.blocked:
                self._log("ai_settings_sync", outcome="blocked", reason=reason, removals=plan.removals)
                return self._say("blocked", plan.blocked, version=remote["version"])
            if not plan.push:
                break
            reply = session.authorized_request("PUT", ROUTE, timeout=REQUEST_TIMEOUT, json_body=plan.payload)
            if 200 <= reply.status < 300:
                data = reply.json()
                version = data.get("version") if isinstance(data, dict) else None
                if isinstance(version, bool) or not isinstance(version, int):
                    version = plan.version + 1
                pushed = plan
                self.store.transaction(lambda state: confirm_push(state, pushed, version))
                self._log("ai_settings_sync", outcome="pushed", reason=reason, version=version, pulled=pulled,
                          profiles=len(plan.payload["profiles"]), adopted=plan.adopted, local_only=plan.local_only)
                return self._say("synced", _local_only_text(plan), version=version)
            if reply.status == 409 and attempt == 0:
                current = _conflict_current(reply)
                try:
                    remote = normalise_remote(current) if current is not None else self._pull(session)
                except ValueError:
                    remote = self._pull(session)
                plan = self._plan(remote, prefer, account)
                pulled = pulled or plan.changed
                continue
            if reply.status == 409:
                raise SyncConflict("the settings changed on the website at the same moment. Oscar tries again "
                                   "later.", status=409)
            if reply.status == 400:
                code, field_name, text = refusal(reply, plan.payload)
                self._log("ai_settings_refused", reason=reason, code=code, field=field_name)
                raise AuthError(f"OriginShape refused these settings: {text}", code=code or "invalid_request",
                                status=400)
            raise api_error(reply, "the AI model settings")
        self._log("ai_settings_sync", outcome="pulled" if pulled else "unchanged", reason=reason,
                  version=remote["version"], profiles=len(remote["profiles"]), adopted=plan.adopted)
        return self._say("synced", _local_only_text(plan), version=remote["version"])


def _notify_hook(event: str, data: dict) -> None:
    hook = on_change
    if hook is None:
        return
    try:
        hook(json.dumps({"event": event, **data}))
    except Exception:  # noqa: BLE001 - a closed window must never break a sync
        pass


# -- the process-wide syncer and its thread ----------------------------------------------------------------------

_LOCK = threading.Lock()
_SYNCER: Syncer | None = None
_THREAD: threading.Thread | None = None
_AGAIN: tuple[str, str] | None = None     # (prefer, reason) of a sync asked for while one ran


def enabled() -> bool:
    """False when ``OSCAR_SETTINGS_SYNC=0`` is set (tests, or a user who wants these settings on this computer only)."""
    return (os.environ.get(ENV) or "").strip() != "0"


def syncer() -> Syncer:
    """The process-wide ``Syncer`` (the configured server, the settings file)."""
    global _SYNCER
    with _LOCK:
        if _SYNCER is None:
            _SYNCER = Syncer()
        return _SYNCER


def reset(replacement: Syncer | None = None) -> None:
    """Use ``replacement`` (tests), or a fresh default ``Syncer`` on next use."""
    global _SYNCER, _AGAIN
    with _LOCK:
        _SYNCER, _AGAIN = replacement, None


def status() -> dict:
    """How the last sync went: ``{"state": "idle" | "syncing" | "synced" | "blocked" | "signed_out" | "api_key" |
    "unavailable" | "error", "message", "at", "version"}``."""
    return dict(syncer().status)


def request(reason: str, *, prefer: str = "remote") -> None:
    """Start a sync on a daemon thread and return at once; asked while one runs, it runs once more after it (a Save's
    ``prefer="local"`` is kept). Never raises."""
    global _THREAD, _AGAIN
    if not enabled():
        return
    try:
        target = syncer()
        with _LOCK:
            if _THREAD is not None and _THREAD.is_alive():
                keep = "local" if prefer == "local" or (_AGAIN is not None and _AGAIN[0] == "local") else "remote"
                _AGAIN = (keep, reason)
                return
            _THREAD = threading.Thread(target=_run, args=(target, prefer, reason), name="oscar-settings-sync",
                                       daemon=True)
            _THREAD.start()
    except Exception as e:  # noqa: BLE001 - a trigger never breaks its caller
        _log("ai_settings_sync", outcome="not_started", reason=reason, error=type(e).__name__)


def _run(target: Syncer, prefer: str, reason: str) -> None:
    global _THREAD, _AGAIN
    while True:
        try:
            target.sync(prefer=prefer, reason=reason)
        except Exception:  # noqa: BLE001 - sync never raises; belt and braces for the thread
            pass
        with _LOCK:
            if _AGAIN is None:
                _THREAD = None
                return
            (prefer, reason), _AGAIN = _AGAIN, None


def _stale(max_age: float) -> bool:
    """A pull is due: on, and neither a pull nor an attempt (signed out, offline) within ``max_age`` seconds. Cheap:
    no thread, no keychain, no network."""
    if not enabled():
        return False
    target = syncer()
    return time.monotonic() - max(target.last_pull, target.last_attempt) >= max_age


def pull_if_stale(max_age: float = PULL_AGE_S) -> bool:
    """Start a sync when one is due (``_stale``); True when one started (or was queued). Never blocks."""
    try:
        if not _stale(max_age):
            return False
    except Exception:  # noqa: BLE001
        return False
    request("chat")
    return True


def pull_before_turn(max_age: float = PULL_AGE_S, wait: float = TURN_WAIT_S) -> bool:
    """Before a chat turn (the user's or a workspace task's): when a sync is due, run one and wait for it up to ``wait``
    seconds, then go on with what the settings file holds. True when one was started."""
    if not pull_if_stale(max_age):
        return False
    wait_idle(wait)
    return True


def wait_idle(timeout: float = 10.0) -> bool:
    """Wait for the sync thread to finish (up to ``timeout``); True when none runs."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _LOCK:
            thread = _THREAD
        if thread is None or not thread.is_alive():
            return True
        thread.join(min(0.05, max(0.0, deadline - time.monotonic())))
    with _LOCK:
        return _THREAD is None or not _THREAD.is_alive()


# -- the Test connection the website asks for --------------------------------------------------------------------


def scrub_test_message(text: str, profile: Any) -> str:
    """A test report as it may leave this computer: ``keyguard.scrub`` (the one policy: every key by its shape, the
    profile's key by value and any piece of it, a key a provider echoed masked) and a local base URL's secret parts
    replaced; at most ``TEST_MESSAGE_LIMIT`` characters."""
    key = field_of(profile, "api_key", "")
    text = _keyguard.scrub(text or "", (key,) if isinstance(key, str) else ())
    url = field_of(profile, "base_url", "") or ""
    if isinstance(url, str) and url.strip() and not publishable_url(url.strip()):
        text = text.replace(url.strip(), "[local base URL]")
        try:
            parts = urlsplit(url.strip())
            secret_parts = [p for p in (parts.username, parts.password, parts.query, parts.fragment) if p]
        except ValueError:
            secret_parts = []
        for part in sorted(secret_parts, key=len, reverse=True):
            if len(part) >= 3:
                text = text.replace(part, _keyguard.REDACTED)
    text = text.strip()
    return text if len(text) <= TEST_MESSAGE_LIMIT else text[:TEST_MESSAGE_LIMIT - 1] + "…"


def report_checks(report: Any) -> dict:
    """``{"reply", "tools", "image"}`` of a ``provider_check.ConnectionReport``: True / False, or None when the step
    was not run or did not decide."""
    lines = list(getattr(report, "lines", None) or ())
    connect = next((line for line in lines if getattr(line, "step", "") == "connect"), None)
    reply = connect.ok if connect is not None and connect.ok in (True, False) else None
    if reply is not True:
        return {"reply": reply, "tools": None, "image": None}
    tools_line = next((line for line in lines if getattr(line, "step", "") == "tools"), None)
    tools = True if getattr(report, "tools", "") else (tools_line.ok if tools_line is not None
                                                      and tools_line.ok in (True, False) else None)
    vision = getattr(report, "vision", None)
    return {"reply": True, "tools": tools, "image": vision if vision in (True, False) else None}


_NO_CHECKS = {"reply": None, "tools": None, "image": None}


def run_profile_test(profile_id: str, *, store: Any = None, run: Callable[..., Any] | None = None,
                     should_stop: Callable[[], bool] | None = None,
                     timeout: float = TEST_TIMEOUT_S) -> tuple[bool, str, dict]:
    """The website's **Test in FreeCAD**: the dialog's Test connection for the local profile ``profile_id``, with its
    local key. ``(passed, message, checks)``; the message is scrubbed (``scrub_test_message``). A profile this
    computer may not use (``unusable_reason``) is not tested. Blocking (up to ``timeout`` seconds); never raises."""
    try:
        profile = (store or ParamsStore()).profile(profile_id) if valid_id(profile_id) else None
    except Exception as e:  # noqa: BLE001 - an unreadable settings file
        _log("ai_settings_test", outcome="unreadable", error=type(e).__name__)
        return False, "Oscar couldn't read its settings in FreeCAD. Try again in a moment.", dict(_NO_CHECKS)
    if profile is None:
        _log("ai_settings_test", outcome="unknown_profile")
        return False, UNKNOWN_PROFILE_TEXT, dict(_NO_CHECKS)
    if unusable_reason(profile):   # created or redirected on the website: nothing is sent until confirmed here
        _log("ai_settings_test", outcome="needs_confirmation")
        return False, NEEDS_CONFIRMATION_TEXT, dict(_NO_CHECKS)
    if run is None:
        from .provider_check import run_connection_test as run
    deadline = time.monotonic() + timeout

    def stop() -> bool:
        return time.monotonic() > deadline or bool(should_stop is not None and should_stop())

    try:
        report = run(profile, should_stop=stop)
    except Exception as e:  # noqa: BLE001 - StopRequested (too long), or a provider bug
        if type(e).__name__ == "StopRequested":
            _log("ai_settings_test", outcome="stopped")
            return False, "The test took too long in FreeCAD and was stopped.", dict(_NO_CHECKS)
        _log("ai_settings_test", outcome="error", error=type(e).__name__)
        return False, f"FreeCAD couldn't run the test ({type(e).__name__}).", dict(_NO_CHECKS)
    text = report.summary() or "(no result)"
    if not report.ok:
        text += "\n✗ This model cannot be used as it is."
    checks = report_checks(report)
    _log("ai_settings_test", outcome="passed" if report.ok else "failed", **{k: v for k, v in checks.items()})
    return bool(report.ok), scrub_test_message(text, profile), checks


def run_website_test(profile_id: str, **kwargs: Any) -> tuple[bool, str, dict]:
    """``run_profile_test`` on the latest settings: a sync first (waited for up to ``TEST_PULL_WAIT_S``), so a profile
    the website just changed is tested as it is now, or refused until confirmed here."""
    if enabled():
        request("profile_test")
        wait_idle(TEST_PULL_WAIT_S)
    return run_profile_test(profile_id, **kwargs)
