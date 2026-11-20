"""Oscar's AI model settings, kept in step with the OriginShape account (the website's Settings → AI model).

The contract is ``docs/architecture/ai-settings-sync.md`` in the OriginShape app repository. This module is pure Python
(no FreeCAD, no Qt): the profile ids, the wire format, the three-way merge, the sync itself (``Syncer``, run on a daemon
thread by ``request``) and the Test connection the website can ask for (``run_profile_test``). ``params`` keeps the
settings file and calls the id helpers here; ``settings_ui`` and ``panel`` start syncs and show what happened.

**Provider keys never leave this computer.** For each profile OriginShape receives exactly its ``id``, name, provider,
base URL, ``base_url_local``, model, reasoning, Advanced overrides (the five keys of ``caps_override`` the dialog sets)
and ``has_key`` (a key for it is saved in FreeCAD); never the key, part of it or a hash of it. A base URL that is not a
plain http(s) URL (a user name or password, a query, a fragment, something shaped like a credential, or not an
http(s) URL at all) goes as ``""`` with ``base_url_local: true`` and the real one stays here; an empty one is an
unfinished profile. A key found in a name or model is masked, and so is anything shaped like a credential
(``context_guard``). The notes, ``model_caps``, the preferences, other ``caps_override`` keys, ``temperature`` /
``max_tokens`` and the network grants stay local, and a pull never touches them.

**When** (``request``; signed in with OriginShape only, never with a personal API key; signed out nothing is sent):
when Oscar's panel starts, after a sign-in, when the settings dialog opens, after a Save there (and after a profile
is picked in the panel), and before a chat turn when the last pull is more than a minute old (``pull_if_stale``).
Every sync runs on one daemon thread; a sync asked for while one runs runs once more after it. A failure is logged
(never with a settings value) and the next trigger tries again; it never blocks a chat or the dialog.

**How.** ``GET /v1/me/ai-settings``, then a three-way merge with the snapshot of the last sync (``ai_sync`` in the
settings file), per profile and per field: a field changed only on the server is taken, one changed only here is kept
and pushed, one changed on both sides goes to the server's value on a pull and to ours after a Save
(``prefer="local"``). A profile deleted on the server is deleted here, with its key, unless it changed here since the
snapshot. ``PUT`` sends the result with the version it was merged against; a 409 is merged again with the server's
current settings and pushed once more. A pull never replaces a key, ``model_caps`` or the preferences, and never
replaces a local base URL by ``""``.

Two device-specific fields, ``has_key`` and the base URL (one FreeCAD may hold a key or a local URL another one does
not), are compared with what THIS FreeCAD last had accepted by the server (``ai_sync.device``), so two computers never
undo each other's values on every sync.

**A key stays with the server it was entered for.** When a pull changes the provider or the base URL of a profile that
has a key (a new URL, another provider, or a website URL replacing one kept in FreeCAD), the key is set aside
(``Profile.held_key``) and the profile is not used at all (no chat, no Test connection, no model list, a website test
fails) until the user decides in the settings dialog: use the key with the new address, or remove it (typing another
key decides too). The key stays in the settings file meanwhile and comes back by itself if a later pull points the
profile back where it was. Otherwise a stolen website session could point a profile at its own server and receive the
key with the next chat. A change made in FreeCAD's own dialog needs no confirmation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit

from . import context_guard as _context_guard
from .auth.transport import AuthError, AuthUnavailable, SignedOut, api_error, error_details, safe_text

ROUTE = "/v1/me/ai-settings"
ENV = "OSCAR_SETTINGS_SYNC"      # "0": never sync (and don't offer the website's Test in FreeCAD)
STATE_KEY = "ai_sync"            # the snapshot of the last sync in oscar_settings.json
PULL_AGE_S = 60.0                # before a chat turn: pull when the last pull is older than this
REQUEST_TIMEOUT: tuple[float, float] = (5.0, 15.0)
ACCOUNT_PROVIDER = "originshape"
REASONING_LEVELS = ("auto", "off", "low", "medium", "high")
OVERRIDE_KEYS = ("max_output", "temperature", "context_window", "tools", "ollama_native")
MAX_OUTPUT_RANGE = (256, 1_000_000)          # the ranges of settings_model's Advanced overrides (and the contract's)
TEMPERATURE_RANGE = (0.0, 2.0)
CONTEXT_WINDOW_RANGE = (1024, 10_000_000)
NAME_LIMIT, URL_LIMIT, MODEL_LIMIT = 80, 500, 200
MIN_MASKED_KEY = 6               # a shorter "key" (a local server's placeholder) is not looked for in names
TEST_MESSAGE_LIMIT = 500
TEST_TIMEOUT_S = 150.0           # the server expires a test 3 minutes after FreeCAD claimed it
NEEDS_CONFIRMATION_TEXT = "Confirm the new address in Oscar's settings in FreeCAD before its key is used."
UNKNOWN_PROFILE_TEXT = ("This profile isn't in Oscar's settings in FreeCAD (it was removed there, or FreeCAD hasn't "
                        "synced it yet). Open Oscar's settings in FreeCAD, then try again.")
_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_REGULAR = ("name", "provider", "model", "reasoning", *(f"overrides.{key}" for key in OVERRIDE_KEYS))
_ENDPOINT_FIELDS = ("provider", "base_url", "model", "overrides.max_output", "overrides.temperature")

on_change: Callable[[str], Any] | None = None   # (any thread) JSON ``{"event": "status" | "changed", ...}``


# -- profile ids -------------------------------------------------------------------------------------------------


def new_profile_id() -> str:
    """A new stable profile id (32 hex characters)."""
    return uuid.uuid4().hex


def valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_ID_RE.fullmatch(value))


def _derived_id(name: str, seen: int) -> str:
    """The id of a profile from before ids: derived from its name (and how many earlier profiles had that name), so
    every read of the same file gives the same id without writing it; the next write keeps it."""
    digest = hashlib.sha256(f"{seen}\n{name}".encode("utf-8")).hexdigest()
    return "p" + digest[:31]


def ensure_ids(state: dict) -> bool:
    """Give every profile of a settings ``state`` a valid, unique ``id`` (in place); True when one was added.

    A profile without one (a file from before ids, or one an older Oscar saved) gets back the id the last sync knew it
    by (the same name in the ``ai_sync`` snapshot), else an id derived from its name: the same on every read."""
    profiles = [p for p in state.get("profiles") or () if isinstance(p, dict)]
    taken = {p["id"] for p in profiles if valid_id(p.get("id"))}
    known: dict[str, str] = {}
    snapshot = state.get(STATE_KEY)
    remote = snapshot.get("remote") if isinstance(snapshot, dict) else None
    for item in (remote.get("profiles") if isinstance(remote, dict) else None) or ():
        if isinstance(item, dict) and valid_id(item.get("id")) and isinstance(item.get("name"), str):
            known.setdefault(item["name"].casefold(), item["id"])
    seen: set[str] = set()
    names: dict[str, int] = {}
    changed = False
    for profile in profiles:
        pid = profile.get("id")
        if valid_id(pid) and pid not in seen:
            seen.add(pid)
            continue
        name = profile.get("name") if isinstance(profile.get("name"), str) else ""
        candidate = known.get(name.casefold())
        if candidate is None or candidate in taken or candidate in seen:
            count = names.get(name, 0)
            candidate = _derived_id(name, count)
            while candidate in taken or candidate in seen:
                count += 1
                candidate = _derived_id(name, count)
            names[name] = count + 1
        profile["id"] = candidate
        seen.add(candidate)
        changed = True
    return changed


def active_index(state: dict) -> int:
    """The index of the active profile of a settings ``state`` (-1 without profiles).

    ``active_profile_id`` names it, and ``active_profile`` (its name, kept for Oscars from before ids) must agree: an
    older Oscar switches profiles by name only, so a name that names another profile wins. A stale or missing choice
    means the first profile."""
    profiles = state.get("profiles") or []
    if not profiles:
        return -1
    name = state.get("active_profile") if isinstance(state.get("active_profile"), str) else ""
    pid = state.get("active_profile_id")
    by_id = next((i for i, p in enumerate(profiles) if valid_id(pid) and p.get("id") == pid), -1)
    if by_id >= 0 and (not name or profiles[by_id].get("name") == name):
        return by_id
    by_name = next((i for i, p in enumerate(profiles) if name and p.get("name") == name), -1)
    if by_name >= 0:
        return by_name
    return by_id if by_id >= 0 else 0


def set_active(state: dict, index: int) -> bool:
    """Make profile ``index`` the active one (its id and its name); True when that changed the state."""
    profiles = state.get("profiles") or []
    if 0 <= index < len(profiles):
        wanted = (profiles[index].get("id"), profiles[index].get("name") or "")
    else:
        wanted = (None, "")
    if (state.get("active_profile_id"), state.get("active_profile")) == wanted:
        return False
    state["active_profile_id"], state["active_profile"] = wanted
    return True


# -- the wire format (what OriginShape sees of a profile) -------------------------------------------------------


def publishable_url(url: Any) -> bool:
    """True for a base URL OriginShape may see (and accepts): an http(s) URL with a host, no user name or password, no
    query and no fragment, at most ``URL_LIMIT`` characters, without spaces, control characters or backslashes, and
    nothing in it shaped like a credential."""
    if not isinstance(url, str) or not url or len(url) > URL_LIMIT or "?" in url or "#" in url or "\\" in url:
        return False
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in url):
        return False
    try:
        parts = urlsplit(url)
        host = parts.hostname
        _port = parts.port
    except ValueError:
        return False
    return (parts.scheme.lower() in ("http", "https") and bool(host) and "@" not in parts.netloc
            and _context_guard.redact(url) == url)


def endpoint(provider: Any, base_url: Any) -> tuple[str, str]:
    """``(provider, base URL)`` as a key's destination, for comparing two of them: the URL's scheme and host in lower
    case, without a trailing ``/``. Any other difference (another host, port, path, user) is another destination."""
    url = base_url.strip() if isinstance(base_url, str) else ""
    try:
        parts = urlsplit(url)
        url = parts._replace(scheme=parts.scheme.lower(), netloc=parts.netloc.lower()).geturl()
    except ValueError:
        pass
    return (provider if isinstance(provider, str) else ""), url.rstrip("/")


def display_host(base_url: Any) -> str:
    """The host (and port) of a base URL for a message (never its user, password, path or query)."""
    try:
        parts = urlsplit(base_url.strip() if isinstance(base_url, str) else "")
        host = parts.hostname or ""
        return f"{host}:{parts.port}" if host and parts.port else host
    except ValueError:
        return ""


def needs_confirmation(profile: Any) -> bool:
    """True while a profile's key is set aside because a sync changed its provider or base URL (``held_key``): the
    profile is not used (no chat, no Test connection, no model list) until the user confirms in the settings dialog."""
    held = profile.get("held_key") if isinstance(profile, dict) else getattr(profile, "held_key", None)
    return isinstance(held, dict) and isinstance(held.get("key"), str) and bool(held["key"].strip())


def confirmation_text(profile: Any) -> str:
    """Why a profile waiting for confirmation is not used, and what to do (no key, no URL secrets in it)."""
    name = profile.get("name") if isinstance(profile, dict) else getattr(profile, "name", "")
    url = profile.get("base_url") if isinstance(profile, dict) else getattr(profile, "base_url", "")
    where = display_host(url) or "its new address"
    return (f"The provider or base URL of the profile “{name or 'Profile'}” was changed on your OriginShape account. "
            f"Oscar doesn't use it, or send its API key to {where}, until you confirm the change in Oscar's settings "
            "(⚙ → AI model): use your key there, or remove the key.")


def _text(value: Any, limit: int) -> str:
    return value[:limit] if isinstance(value, str) else ""


def _masked(text: str, secrets: tuple[str, ...]) -> str:
    """``text`` without any of ``secrets``, without anything shaped like a credential or a piece of one
    (``context_guard``), and without NUL characters (the server refuses a request that has any of them)."""
    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, _context_guard.REDACTED)
    return _context_guard.redact_fragments(_context_guard.redact(text)).replace("\x00", "")


def _number(value: Any, low: float, high: float, cast) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return None
    return cast(value) if low <= value <= high else None


def overrides_of(caps: Any) -> dict:
    """The contract's ``overrides`` from a profile's ``caps_override`` (or from what the server sent): each of the five
    keys, ``None`` for "no override". What the dialog stores as "automatic" (``tools`` ``auto``, ``ollama_native``
    true, 0) is ``None`` too, so both sides compare equal."""
    caps = caps if isinstance(caps, dict) else {}
    temperature = _number(caps.get("temperature"), *TEMPERATURE_RANGE, float)
    return {
        "max_output": _number(caps.get("max_output"), *MAX_OUTPUT_RANGE, int),
        "temperature": round(temperature, 3) if temperature is not None else None,
        "context_window": _number(caps.get("context_window"), *CONTEXT_WINDOW_RANGE, int),
        "tools": caps.get("tools") if caps.get("tools") in ("native", "text") else None,
        "ollama_native": False if caps.get("ollama_native") is False else None,
    }


ACCOUNT_WIRE_MODELS = ("origin1", "origin1-super")   # the account's models the server takes in a profile


def _account_model(model: str) -> str:
    from .auth.account import RETIRED_ALIASES

    return RETIRED_ALIASES.get(model.strip(), model.strip())


def _account_wire_model(model: str) -> str:
    """The account profile's model as the server takes it: one of ``ACCOUNT_WIRE_MODELS`` (a retired alias as its
    tier; an alias the server added since this list as the default, so the rest of the settings still sync)."""
    model = _account_model(model)
    return model if model in ACCOUNT_WIRE_MODELS else ACCOUNT_WIRE_MODELS[0]


WIRE_FIELDS = ("id", "name", "provider", "base_url", "base_url_local", "model", "reasoning", "overrides",
               "has_key")   # exactly what a profile carries on the wire (the server refuses any other field)


def masked_fields(profile: dict, secrets: tuple[str, ...] = ()) -> list[str]:
    """The free-text fields of a stored profile that ``to_wire`` masks (a key or a credential-shaped value in them):
    their names only, for the log."""
    secrets = tuple(s for s in secrets if isinstance(s, str) and len(s) >= MIN_MASKED_KEY)
    found = []
    for name, limit in (("name", 4 * NAME_LIMIT), ("model", MODEL_LIMIT)):
        text = _text(profile.get(name), limit)
        if _masked(text, secrets) != text.replace("\x00", ""):
            found.append(name)
    return found


def to_wire(profile: dict, secrets: tuple[str, ...] = ()) -> dict:
    """What OriginShape receives of a stored profile (a ``params.Profile`` as a dict): exactly ``WIRE_FIELDS``; never
    ``api_key`` or ``held_key``; a base URL only when ``publishable_url`` (an empty one is an unfinished profile; any
    other goes as ``""`` with ``base_url_local``); every ``secrets`` value and anything credential-shaped masked in the
    free text."""
    secrets = tuple(s for s in secrets if isinstance(s, str) and len(s) >= MIN_MASKED_KEY)
    provider = _text(profile.get("provider"), 64)
    url = profile.get("base_url").strip() if isinstance(profile.get("base_url"), str) else ""
    if provider == ACCOUNT_PROVIDER or not url:
        base_url, local = "", False
    elif publishable_url(url) and not any(s in url for s in secrets):
        base_url, local = url, False
    else:
        base_url, local = "", True
    model = _masked(_text(profile.get("model"), MODEL_LIMIT).strip(), secrets)
    if provider == ACCOUNT_PROVIDER:
        model = _account_wire_model(model)
    name = " ".join(_masked(_text(profile.get("name"), 4 * NAME_LIMIT), secrets).split())[:NAME_LIMIT] or "Profile"
    reasoning = profile.get("reasoning") if profile.get("reasoning") in REASONING_LEVELS else "auto"
    api_key = profile.get("api_key")
    return {
        "id": profile.get("id"),
        "name": name,
        "provider": provider,
        "base_url": base_url,
        "base_url_local": local,
        "model": model[:MODEL_LIMIT],
        "reasoning": reasoning,
        "overrides": overrides_of(profile.get("caps_override")),
        # a key set aside until the user confirms a changed provider or URL is still a key saved in FreeCAD
        "has_key": (isinstance(api_key, str) and bool(api_key.strip())) or needs_confirmation(profile),
    }


def wire_profile(profile: dict) -> dict:
    """A wire profile with exactly ``WIRE_FIELDS`` (and the five override keys), whatever else ``profile`` holds."""
    out = {name: profile.get(name) for name in WIRE_FIELDS}
    out["overrides"] = {key: (profile.get("overrides") or {}).get(key) for key in OVERRIDE_KEYS}
    return out


def _local_secrets(state: dict) -> tuple[str, ...]:
    secrets = []
    for profile in state.get("profiles") or ():
        if not isinstance(profile, dict):
            continue
        for value in (profile.get("api_key"), (profile.get("held_key") or {}).get("key")
                      if isinstance(profile.get("held_key"), dict) else None):
            if isinstance(value, str) and value.strip():
                secrets.append(value.strip())
    return tuple(sorted(set(secrets), key=len, reverse=True))


def local_settings(state: dict) -> dict:
    """This computer's settings in the wire format: ``{"active_profile_id", "profiles"}`` (keys masked, see
    ``to_wire``)."""
    secrets = _local_secrets(state)
    profiles = [to_wire(p, secrets) for p in state.get("profiles") or () if isinstance(p, dict) and valid_id(p.get("id"))]
    index = active_index(state)
    active = state["profiles"][index].get("id") if index >= 0 else None
    return {"active_profile_id": active if valid_id(active) else None, "profiles": profiles}


def normalise_remote(data: Any) -> dict:
    """``AiSettings`` as the server sent it, made comparable with ``local_settings``: ``{"version", "updated_at",
    "updated_by", "active_profile_id", "profiles"}``. Profiles without a usable id are left out. Raises ``ValueError``
    for an answer that is not an AiSettings object."""
    if not isinstance(data, dict) or not isinstance(data.get("profiles", []), list):
        raise ValueError("not an AiSettings object")
    version = data.get("version", 0)
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("version")
    profiles, seen = [], set()
    for item in data.get("profiles") or ():
        if not isinstance(item, dict) or not valid_id(item.get("id")) or item["id"] in seen:
            continue
        seen.add(item["id"])
        provider = _text(item.get("provider"), 64)
        model = _text(item.get("model"), MODEL_LIMIT).strip()
        local = item.get("base_url_local") is True
        profiles.append({
            "id": item["id"],
            "name": " ".join(_text(item.get("name"), 4 * NAME_LIMIT).split())[:NAME_LIMIT] or "Profile",
            "provider": provider,
            "base_url": "" if local or provider == ACCOUNT_PROVIDER else _text(item.get("base_url"), URL_LIMIT).strip(),
            "base_url_local": local and provider != ACCOUNT_PROVIDER,
            "model": _account_model(model) if provider == ACCOUNT_PROVIDER else model,
            "reasoning": item.get("reasoning") if item.get("reasoning") in REASONING_LEVELS else "auto",
            "overrides": overrides_of(item.get("overrides")),
            "has_key": item.get("has_key") is True,
        })
    active = data.get("active_profile_id")
    return {"version": version, "updated_at": data.get("updated_at"), "updated_by": data.get("updated_by"),
            "active_profile_id": active if valid_id(active) and active in seen else None, "profiles": profiles}


def same_settings(a: dict, b: dict) -> bool:
    """The same active profile and the same profiles, in the same order (versions and timestamps aside)."""
    return a.get("active_profile_id") == b.get("active_profile_id") and a.get("profiles") == b.get("profiles")


# -- the three-way merge -----------------------------------------------------------------------------------------


def _get(profile: dict, name: str) -> Any:
    if name.startswith("overrides."):
        return (profile.get("overrides") or {}).get(name[len("overrides."):])
    return profile.get(name)


def _url(profile: dict) -> tuple[str, bool]:
    return profile.get("base_url") or "", bool(profile.get("base_url_local"))


def _device(profile: dict) -> dict:
    return {"has_key": bool(profile.get("has_key")), "url": list(_url(profile))}


@dataclass
class Entry:
    """One profile of a merge: ``wire`` goes to the server, ``apply`` (field -> value) and ``url`` (a real base URL
    the server took, or None to keep ours) change the local one; ``new`` makes it here from ``wire``."""

    id: str
    wire: dict
    apply: dict = field(default_factory=dict)
    url: str | None = None
    new: bool = False


@dataclass
class Merged:
    entries: list[Entry]
    active: str | None

    def settings(self) -> dict:
        return {"active_profile_id": self.active, "profiles": [entry.wire for entry in self.entries]}


def _changed_here(local: dict, base: dict, device: dict | None) -> bool:
    """True when ``local`` differs from the profile at the last sync (``base``; its key and URL from ``device``)."""
    if any(_get(local, name) != _get(base, name) for name in _REGULAR):
        return True
    if device is not None:
        return local["has_key"] != device.get("has_key") or list(_url(local)) != list(device.get("url") or [])
    return local["has_key"] != base["has_key"] or _url(local) != _url(base)


def _changed_there(remote: dict, base: dict) -> bool:
    return (any(_get(remote, name) != _get(base, name) for name in _REGULAR) or _url(remote) != _url(base)
            or remote["has_key"] != base["has_key"])


def _merge_profile(base: dict | None, local: dict, remote: dict, device: dict | None, prefer: str) -> Entry:
    wire: dict[str, Any] = {"id": local["id"], "overrides": {}}
    apply: dict[str, Any] = {}
    for name in _REGULAR:
        ours, theirs = _get(local, name), _get(remote, name)
        if ours == theirs:
            value = ours
        elif base is not None and ours == _get(base, name):
            value = theirs                    # changed only on the server
        elif base is not None and theirs == _get(base, name):
            value = ours                      # changed only here
        else:
            value = theirs if prefer == "remote" else ours   # both (or no snapshot): the newer write
        if name.startswith("overrides."):
            wire["overrides"][name[len("overrides."):]] = value
        else:
            wire[name] = value
        if value != ours:
            apply[name] = value
    # the base URL: compared with what THIS FreeCAD had at the last sync (another one may keep a local URL)
    ours_url, theirs_url = _url(local), _url(remote)
    ours_changed = device is None or list(ours_url) != list(device.get("url") or [])
    theirs_changed = base is None or theirs_url != _url(base)
    url = None
    if ours_url == theirs_url:
        sent = ours_url
    elif ours_changed and (not theirs_changed or prefer == "local"):
        sent = ours_url
    else:
        sent = theirs_url
        if theirs_changed and not theirs_url[1] and (theirs_url[0] or not ours_url[1]):
            url = theirs_url[0]               # a real URL set elsewhere replaces ours ("" never replaces a local one)
    wire["base_url"], wire["base_url_local"] = sent
    # has_key: "a FreeCAD holds a key". True while this one does; false when this one just dropped its key; else what
    # the server has (another FreeCAD may hold one)
    if local["has_key"]:
        wire["has_key"] = True
    elif device is not None and device.get("has_key"):
        wire["has_key"] = False
    else:
        wire["has_key"] = remote["has_key"]
    return Entry(local["id"], {key: wire[key] for key in local}, apply, url)


def _order(local_ids: list[str], remote_ids: list[str], prefer: str) -> list[str]:
    """The preferred side's order, with the other side's extra profiles after their predecessor there."""
    first, second = (remote_ids, local_ids) if prefer == "remote" else (local_ids, remote_ids)
    result = list(first)
    for i, pid in enumerate(second):
        if pid in result:
            continue
        before = next((second[j] for j in range(i - 1, -1, -1) if second[j] in result), None)
        result.insert(result.index(before) + 1 if before is not None else 0, pid)
    return result


def _numbered(name: str, taken: set[str]) -> str:
    number = 2
    while True:
        suffix = f" {number}"
        candidate = name[:NAME_LIMIT - len(suffix)] + suffix
        if candidate.casefold() not in taken:
            return candidate
        number += 1


def _unique_names(entries: list[Entry], keep: set[str]) -> None:
    """Names unique (ignoring case), as the server requires: the profiles in ``keep`` keep theirs first."""
    taken: set[str] = set()
    for entry in sorted(entries, key=lambda e: e.id not in keep):   # stable: ``keep`` first, then in order
        name = entry.wire["name"]
        if name.casefold() in taken:
            name = _numbered(name, taken)
            entry.wire["name"] = name
            if not entry.new:
                entry.apply["name"] = name
        taken.add(name.casefold())


def merge(base: dict | None, local: dict, remote: dict, *, device: dict | None = None,
          prefer: str = "remote") -> Merged:
    """Three-way merge of the wire settings ``local`` (this computer) and ``remote`` (the server) against ``base`` (the
    server's settings at the last sync; None for a first sync with this account). ``device`` is this FreeCAD's
    ``has_key`` and base URL per profile at the last sync. ``prefer`` (``"remote"`` on a pull, ``"local"`` after a Save)
    decides a field both sides changed."""
    device = device or {}
    by_base = {p["id"]: p for p in base["profiles"]} if base is not None else {}
    by_local = {p["id"]: p for p in local["profiles"]}
    by_remote = {p["id"]: p for p in remote["profiles"]}
    entries: list[Entry] = []
    for pid in _order(list(by_local), list(by_remote), prefer):
        b, ours, theirs = by_base.get(pid), by_local.get(pid), by_remote.get(pid)
        if ours is not None and theirs is not None:
            entries.append(_merge_profile(b, ours, theirs, device.get(pid), prefer))
        elif ours is not None:
            if b is not None and not _changed_here(ours, b, device.get(pid)):
                continue                      # deleted on the server, unchanged here: deleted here too
            entries.append(Entry(pid, copy.deepcopy(ours)))   # new here, or changed here since: kept (and pushed)
        else:
            if b is not None and (prefer == "local" or not _changed_there(theirs, b)):
                continue                      # deleted here: the server deletes it too
            entries.append(Entry(pid, copy.deepcopy(theirs), new=True))
    side = by_remote if prefer == "remote" else by_local
    _unique_names(entries, {e.id for e in entries if e.id in side and e.wire["name"] == side[e.id]["name"]})
    ids = [e.id for e in entries]
    ours_active, theirs_active = local.get("active_profile_id"), remote.get("active_profile_id")
    base_active = base.get("active_profile_id") if base is not None else object()
    if ours_active == theirs_active:
        active = ours_active
    elif ours_active == base_active:
        active = theirs_active
    elif theirs_active == base_active:
        active = ours_active
    else:
        active = theirs_active if prefer == "remote" else ours_active
    if active not in ids:
        active = next((a for a in (ours_active, theirs_active) if a in ids), ids[0] if ids else None)
    return Merged(entries, active)


def adopt_remote_ids(state: dict, remote: dict) -> int:
    """First sync with an account: a local profile the server has under another id (the same name, or the only
    account profile on each side) takes the server's id, so it is merged instead of doubled. Returns how many."""
    profiles = [p for p in state.get("profiles") or () if isinstance(p, dict)]
    remote_ids = {p["id"] for p in remote["profiles"]}
    local_ids = {p.get("id") for p in profiles}
    claimed = {pid for pid in local_ids if pid in remote_ids}
    by_name = {}
    for item in remote["profiles"]:
        by_name.setdefault(item["name"].casefold(), item["id"])
    mapping: dict[str, str] = {}
    for profile in profiles:
        if profile.get("id") in remote_ids:
            continue
        name = profile.get("name") if isinstance(profile.get("name"), str) else ""
        rid = by_name.get(" ".join(name.split()).casefold())
        if rid is not None and rid not in claimed and rid not in local_ids:
            mapping[profile["id"]] = rid
            claimed.add(rid)
    mine = [p for p in profiles if p.get("provider") == ACCOUNT_PROVIDER and p.get("id") not in remote_ids
            and p.get("id") not in mapping]
    theirs = [p["id"] for p in remote["profiles"] if p["provider"] == ACCOUNT_PROVIDER and p["id"] not in claimed]
    if len(mine) == 1 and len(theirs) == 1 and theirs[0] not in local_ids:
        mapping[mine[0]["id"]] = theirs[0]
    for profile in profiles:
        new = mapping.get(profile.get("id"))
        if new is not None:
            if state.get("active_profile_id") == profile["id"]:
                state["active_profile_id"] = new
            profile["id"] = new
    return len(mapping)


# -- applying a merge to the settings file -----------------------------------------------------------------------


def _set_field(profile: dict, name: str, value: Any) -> None:
    if name.startswith("overrides."):
        key = name[len("overrides."):]
        caps = dict(profile.get("caps_override")) if isinstance(profile.get("caps_override"), dict) else {}
        if value is None:
            caps.pop(key, None)
        else:
            caps[key] = value
        profile["caps_override"] = caps
    else:
        profile[name] = value


def stored_from_wire(wire: dict) -> dict:
    """A new local profile (``params.Profile`` as a dict) from one the server has: no key, and no base URL when the
    real one is kept on another computer."""
    return {
        "name": wire["name"], "provider": wire["provider"],
        "base_url": "" if wire["base_url_local"] else wire["base_url"], "model": wire["model"], "api_key": "",
        "temperature": 0.2, "max_tokens": 4096, "notes": "", "reasoning": wire["reasoning"],
        "caps_override": {k: v for k, v in wire["overrides"].items() if v is not None}, "id": wire["id"],
        "held_key": {},
    }


def _guard_key(before: dict, profile: dict) -> None:
    """A key goes only where it was entered for. When a pull changed the provider or the base URL of a profile with a
    key, the key is set aside (``held_key``: the key and where it was entered for) and the profile waits for the user
    to confirm (``needs_confirmation``); it comes back by itself when the profile points there again."""
    old, new = endpoint(before.get("provider"), before.get("base_url")), \
        endpoint(profile.get("provider"), profile.get("base_url"))
    key = profile.get("api_key") if isinstance(profile.get("api_key"), str) else ""
    held = profile.get("held_key") if isinstance(profile.get("held_key"), dict) else {}
    if key.strip() and old != new:
        profile["held_key"] = {"key": key, "provider": before.get("provider") or "",
                               "base_url": before.get("base_url") or ""}
        profile["api_key"] = ""
    elif not key.strip() and held.get("key") and endpoint(held.get("provider"), held.get("base_url")) == new:
        profile["api_key"], profile["held_key"] = held["key"], {}


def apply_merge(state: dict, merged: Merged, limits: Callable[[dict], tuple[float, int]] | None = None) -> None:
    """Make the settings ``state``'s profiles and active profile what ``merged`` says (in place). Keys, local base
    URLs and every local-only field are kept; ``limits(profile)`` recomputes the stored temperature and output cap of a
    profile whose provider, URL, model or overrides changed."""
    by_id = {p.get("id"): p for p in state.get("profiles") or () if isinstance(p, dict)}
    result = []
    for entry in merged.entries:
        current = by_id.get(entry.id)
        if current is None:
            profile = stored_from_wire(entry.wire)
            touched = True
        else:
            profile = copy.deepcopy(current)
            for name, value in entry.apply.items():
                _set_field(profile, name, value)
            if entry.url is not None:
                profile["base_url"] = entry.url
            _guard_key(current, profile)
            touched = any(name in entry.apply for name in _ENDPOINT_FIELDS) or entry.url is not None \
                or any(name.startswith("overrides.") for name in entry.apply)
        if touched and limits is not None:
            try:
                profile["temperature"], profile["max_tokens"] = limits(profile)
            except Exception:  # noqa: BLE001 - the stored values are recomputed at every turn anyway
                pass
        result.append(profile)
    state["profiles"] = result
    set_active(state, next((i for i, p in enumerate(result) if p.get("id") == merged.active), 0 if result else -1))


def _profile_view(state: dict) -> str:
    index = active_index(state)
    return json.dumps([state.get("profiles") or [], index], sort_keys=True, default=str)


def usable_snapshot(snapshot: Any, account: str, remote_version: int) -> dict | None:
    """The snapshot of the last sync when it can be the merge's base: same account, well formed, and not newer than
    the server (a server whose settings were reset, or an account deleted and made again, starts over: nothing is
    deleted here because the server forgot it)."""
    if not isinstance(snapshot, dict) or snapshot.get("account") != account:
        return None
    version, remote = snapshot.get("version"), snapshot.get("remote")
    if isinstance(version, bool) or not isinstance(version, int) or version > remote_version \
            or not isinstance(remote, dict):
        return None
    try:
        base = normalise_remote({**remote, "version": version})
    except ValueError:
        return None
    device = snapshot.get("device") if isinstance(snapshot.get("device"), dict) else {}
    return {"remote": base, "device": {k: v for k, v in device.items() if isinstance(v, dict)}}


@dataclass
class Plan:
    """What one merge decided: ``payload`` (the PUT body) when ``push``; ``changed`` when the local profiles or the
    active profile changed; ``device`` is this FreeCAD's key and URL record once the server has the payload."""

    payload: dict
    push: bool
    changed: bool
    device: dict
    version: int
    account: str
    adopted: int = 0
    masked: dict = field(default_factory=dict)   # free-text field -> how many profiles had a credential masked in it


def plan_sync(state: dict, remote: dict, *, prefer: str, account: str,
              limits: Callable[[dict], tuple[float, int]] | None = None) -> tuple[bool, Plan]:
    """Merge the server's ``remote`` settings into the settings ``state`` (in place); ``(write, Plan)``. The
    snapshot becomes ``remote`` (the next merge's base), so what is still to be pushed stays a local change."""
    ensure_ids(state)
    before = _profile_view(state)
    old_snapshot = copy.deepcopy(state.get(STATE_KEY))
    snap = usable_snapshot(old_snapshot, account, remote["version"])
    adopted = adopt_remote_ids(state, remote) if snap is None else 0
    device = snap["device"] if snap is not None else {}
    merged = merge(snap["remote"] if snap is not None else None, local_settings(state), remote, device=device,
                   prefer=prefer)
    before_keys = {p["id"]: p["has_key"] for p in local_settings(state)["profiles"]}
    apply_merge(state, merged, limits)
    after = local_settings(state)["profiles"]
    for entry in merged.entries:   # a key the merge set aside (or gave back) changes what the server is told
        key = next((p["has_key"] for p in after if p["id"] == entry.id), None)
        if key is not None and entry.id in before_keys and key != before_keys[entry.id]:
            entry.wire["has_key"] = key
    settings = merged.settings()
    payload = {"version": remote["version"], "active_profile_id": settings["active_profile_id"],
               "profiles": [wire_profile(p) for p in settings["profiles"]]}
    push = not same_settings(payload, remote)
    now = {p["id"]: _device(p) for p in after}
    masked: dict[str, int] = {}
    secrets = _local_secrets(state)
    for profile in state["profiles"]:
        for name in masked_fields(profile, secrets):
            masked[name] = masked.get(name, 0) + 1
    state[STATE_KEY] = {"account": account, "version": remote["version"],
                        "remote": {"active_profile_id": remote["active_profile_id"],
                                   "profiles": copy.deepcopy(remote["profiles"])},
                        "device": device if push else now}
    changed = _profile_view(state) != before
    plan = Plan(payload, push, changed, now, remote["version"], account, adopted, masked)
    return changed or state[STATE_KEY] != old_snapshot, plan


def confirm_push(state: dict, plan: Plan, version: int) -> tuple[bool, None]:
    """The server took ``plan.payload`` as ``version``: it is the next merge's base, with this FreeCAD's record."""
    state[STATE_KEY] = {"account": plan.account, "version": version,
                        "remote": {"active_profile_id": plan.payload["active_profile_id"],
                                   "profiles": copy.deepcopy(plan.payload["profiles"])},
                        "device": plan.device}
    return True, None


# -- the dialog's Save: its edits merged with what a pull wrote meanwhile ---------------------------------------


def _merge_value(base: Any, ours: Any, theirs: Any) -> Any:
    if isinstance(base, dict) and isinstance(ours, dict) and isinstance(theirs, dict):
        out = {}
        for key in [*theirs, *(k for k in ours if k not in theirs), *(k for k in base if k not in theirs and k not in ours)]:
            missing = object()
            value = _merge_value(base.get(key, missing), ours.get(key, missing), theirs.get(key, missing))
            if value is not missing:
                out[key] = value
        return out
    return theirs if ours == base else ours


DERIVED_FIELDS = ("temperature", "max_tokens")   # recomputed from provider, model and overrides: not an edit


def _edited(profile: dict, base: dict) -> bool:
    """True when the dialog changed ``profile`` (a stored dict) since it loaded ``base``."""
    keys = (set(profile) | set(base)) - set(DERIVED_FIELDS)
    return any(profile.get(key) != base.get(key) for key in keys)


def merge_saved(base: dict, ours: dict, theirs: dict) -> dict:
    """The settings dialog's Save. ``base`` is what it loaded, ``ours`` what it saves, ``theirs`` what the file holds
    now (a pull may have changed it while the dialog was open); each ``{"profiles": [stored dicts],
    "active_profile_id"}``. Per profile and field, what the dialog changed wins and the rest is the file's: a profile
    the dialog removed goes, one a pull added or changed stays."""
    by_base = {p["id"]: p for p in base["profiles"]}
    by_ours = {p["id"]: p for p in ours["profiles"]}
    by_theirs = {p["id"]: p for p in theirs["profiles"]}
    profiles: list[dict] = []
    for pid in _order(list(by_theirs), list(by_ours), "local"):
        b, o, t = by_base.get(pid), by_ours.get(pid), by_theirs.get(pid)
        if o is not None and t is not None:
            merged = _merge_value(b, o, t) if b is not None else copy.deepcopy(o)
            if b is not None and o.get("api_key") and o.get("api_key") != b.get("api_key"):
                merged["held_key"] = {}      # a key typed in the dialog decides for a profile a pull put on hold
            profiles.append(merged)
        elif o is not None:
            if b is None or _edited(o, b):   # new in the dialog, or edited there after a pull removed it
                profiles.append(copy.deepcopy(o))
        elif b is None:                      # a pull added it while the dialog was open
            profiles.append(copy.deepcopy(t))
    taken: set[str] = set()
    mine = {pid for pid in by_ours}
    for profile in sorted(profiles, key=lambda p: p["id"] not in mine):
        name = profile.get("name") if isinstance(profile.get("name"), str) else "Profile"
        if name.casefold() in taken:
            profile["name"] = name = _numbered(name, taken)
        taken.add(name.casefold())
    ids = [p["id"] for p in profiles]
    active = ours.get("active_profile_id")
    if active == base.get("active_profile_id") and theirs.get("active_profile_id") in ids:
        active = theirs.get("active_profile_id")
    if active not in ids:
        active = next((a for a in (theirs.get("active_profile_id"),) if a in ids), ids[0] if ids else None)
    return {"profiles": profiles, "active_profile_id": active}


# -- the sync ----------------------------------------------------------------------------------------------------


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
        self.status: dict = {"state": "idle", "message": "", "at": 0.0, "version": 0}

    def _say(self, state: str, message: str = "", **extra: Any) -> dict:
        self.status = {"state": state, "message": message, "at": time.time(),
                       "version": extra.get("version", self.status.get("version", 0))}
        self._notify("status", dict(self.status))
        return dict(self.status)

    def sync(self, prefer: str = "remote", reason: str = "") -> dict:
        """One sync (``prefer``: ``"remote"`` for a pull, ``"local"`` after a Save). Returns ``status``."""
        with self._lock:
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
            if not plan.push:
                break
            reply = session.authorized_request("PUT", ROUTE, timeout=REQUEST_TIMEOUT, json_body=plan.payload)
            if 200 <= reply.status < 300:
                data = reply.json()
                version = data.get("version") if isinstance(data, dict) else None
                if isinstance(version, bool) or not isinstance(version, int):
                    version = plan.version + 1
                self.store.transaction(lambda state: confirm_push(state, plan, version))
                self._log("ai_settings_sync", outcome="pushed", reason=reason, version=version, pulled=pulled,
                          profiles=len(plan.payload["profiles"]), adopted=plan.adopted)
                return self._say("synced", version=version)
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
        return self._say("synced", version=remote["version"])


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
    """How the last sync went: ``{"state": "idle" | "syncing" | "synced" | "signed_out" | "api_key" | "unavailable" |
    "error", "message", "at", "version"}``."""
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


def pull_if_stale(max_age: float = PULL_AGE_S) -> bool:
    """Before a chat turn: start a sync when the last pull is older than ``max_age`` seconds. True when one started
    (or was queued). Never blocks."""
    try:
        if not enabled() or time.monotonic() - syncer().last_pull < max_age:
            return False
    except Exception:  # noqa: BLE001
        return False
    request("chat")
    return True


def wait_idle(timeout: float = 10.0) -> bool:
    """Wait for the sync thread to finish (tests); True when none runs."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _LOCK:
            thread = _THREAD
        if thread is None or not thread.is_alive():
            return True
        thread.join(0.05)
    return False


# -- the Test connection the website asks for --------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z0-9_\-]{4,}")


def scrub_test_message(text: str, profile: Any) -> str:
    """A test report as it may leave this computer: every credential ``context_guard`` knows or recognises, the
    profile's key (and its held key) and any run of it (6+ characters, or 4+ with a digit), a key a provider echoed
    masked or a word that starts like a key (``context_guard.redact_fragments``: ``sk-proj-****…Ab9Z``, ``****abcd``)
    and a local base URL's secret parts replaced; at most ``TEST_MESSAGE_LIMIT`` characters."""
    text = _context_guard.redact(text or "")
    held = getattr(profile, "held_key", None)
    secrets = [s.strip() for s in (getattr(profile, "api_key", ""), held.get("key") if isinstance(held, dict) else "")
               if isinstance(s, str) and s.strip()]
    for secret in sorted(secrets, key=len, reverse=True):
        if len(secret) >= 4:
            text = text.replace(secret, _context_guard.REDACTED)
    text = _context_guard.redact_fragments(text)

    def fragment(match: re.Match) -> str:
        word = match.group(0)
        long_enough = len(word) >= 6 or any(ch.isdigit() for ch in word)
        return _context_guard.REDACTED if long_enough and any(word in s for s in secrets) else word

    text = _WORD.sub(fragment, text)
    url = getattr(profile, "base_url", "") or ""
    if isinstance(url, str) and url.strip() and not publishable_url(url.strip()):
        text = text.replace(url.strip(), "[local base URL]")
        try:
            parts = urlsplit(url.strip())
            secret_parts = [p for p in (parts.username, parts.password, parts.query, parts.fragment) if p]
        except ValueError:
            secret_parts = []
        for part in sorted(secret_parts, key=len, reverse=True):
            if len(part) >= 3:
                text = text.replace(part, _context_guard.REDACTED)
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
    local key. ``(passed, message, checks)``; the message is scrubbed (``scrub_test_message``). Blocking (up to
    ``timeout`` seconds); never raises."""
    try:
        profile = (store or ParamsStore()).profile(profile_id) if valid_id(profile_id) else None
    except Exception as e:  # noqa: BLE001 - an unreadable settings file
        _log("ai_settings_test", outcome="unreadable", error=type(e).__name__)
        return False, "Oscar couldn't read its settings in FreeCAD. Try again in a moment.", dict(_NO_CHECKS)
    if profile is None:
        _log("ai_settings_test", outcome="unknown_profile")
        return False, UNKNOWN_PROFILE_TEXT, dict(_NO_CHECKS)
    if needs_confirmation(profile):   # its provider or URL was changed elsewhere: nothing is sent until confirmed here
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
