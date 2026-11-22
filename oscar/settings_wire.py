"""The AI model settings as the settings file holds them and as OriginShape sees them (issue #86; the contract is
``docs/architecture/ai-settings-sync.md`` in the app repository). Pure Python: profile ids, the active profile, the
gate that says whether a profile may be used, and the wire format. ``settings_merge`` merges, ``settings_sync`` syncs;
``params`` keeps the file.

**Ids.** Every profile has a stable ``id``. A profile from before ids gets back the id the last sync knew it by (its
name in the ``ai_sync`` snapshot), else one derived from its position and name, the same on every read (reading never
writes; the next write keeps it). A name that holds a key, or anything shaped like one, is never hashed into an id:
such a profile's id comes from its position and provider only.

**The active profile** is ``active_profile_id`` plus its name in ``active_profile`` (what an Oscar from before ids
reads; a name that names another profile wins: an older Oscar switches by name). ``active_pending`` is the website's
choice this computer does not follow yet (a profile to confirm, or one whose key is on another computer).

**The gate** (``unusable_reason``): a profile is used (chat, Test connection, model list, metadata probe, the website's
Test in FreeCAD) only at the provider and base URL the user approved at this computer (``Profile.approved``). A profile
made or edited in FreeCAD is approved as it is; one a sync created, or whose provider or base URL a sync changed, waits
until the user confirms that exact address in the settings dialog. The account profile (``originshape``) has a fixed
endpoint and needs none. So a stolen website session can neither send a key to its own server nor have prompts sent
there. A profile without ``approved`` (a 0.2.1 settings file, or one an older Oscar saved) is approved for the address
it has here when it is read (``ensure_approvals``, on every read and before a pull is applied), never for one a pull
gives it.

**The wire format** (``to_wire``): exactly ``WIRE_FIELDS``; never a key, a piece of one or a hash of one. A base URL
that is not a plain http(s) URL (a user, password, query, fragment, anything key-shaped, ``..`` or ``xxxx`` in it) goes
as ``""`` with ``base_url_local``; an empty one is an unfinished profile. Names and models go through ``keyguard``. The
notes, ``model_caps``, the preferences, other ``caps_override`` keys and ``temperature`` / ``max_tokens`` stay local.
Profiles whose provider OriginShape doesn't know, and profiles the website removed that still wait for the user's
decision (``removed_remotely``), are not synced (``syncable``).
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any
from urllib.parse import urlsplit

from . import keyguard as _keyguard

STATE_KEY = "ai_sync"            # the snapshot of the last sync in oscar_settings.json
PENDING_KEY = "active_pending"   # {"id", "why": "confirm" | "no_key"}: the website's active profile, not followed yet
ACCOUNT_PROVIDER = "originshape"
PROVIDERS = (ACCOUNT_PROVIDER, "anthropic", "openai_compat")
ACCOUNT_WIRE_MODELS = ("origin1", "origin1-super")   # the account's models the server takes in a profile
REASONING_LEVELS = ("auto", "off", "low", "medium", "high")
OVERRIDE_KEYS = ("max_output", "temperature", "context_window", "tools", "ollama_native")
MAX_OUTPUT_RANGE = (256, 1_000_000)          # the ranges of settings_model's Advanced overrides (and the contract's)
TEMPERATURE_RANGE = (0.0, 2.0)
CONTEXT_WINDOW_RANGE = (1024, 10_000_000)
NAME_LIMIT, URL_LIMIT, MODEL_LIMIT = 80, 500, 200
MAX_PROFILES = 50
WIRE_FIELDS = ("id", "name", "provider", "base_url", "base_url_local", "model", "reasoning", "overrides",
               "has_key")   # exactly what a profile carries on the wire (the server refuses any other field)
NEEDS_CONFIRMATION_TEXT = "Confirm the new address in Oscar's settings in FreeCAD before its key is used."
_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def field_of(profile: Any, name: str, default: Any = None) -> Any:
    """A field of a profile, a ``params.Profile`` or its stored dict alike."""
    if isinstance(profile, dict):
        return profile.get(name, default)
    return getattr(profile, name, default)


def _text(value: Any, limit: int) -> str:
    return value[:limit] if isinstance(value, str) else ""


# -- ids ---------------------------------------------------------------------------------------------------------


def new_profile_id() -> str:
    """A new stable profile id (32 hex characters)."""
    return uuid.uuid4().hex


def valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_ID_RE.fullmatch(value))


def local_secrets(state: dict) -> tuple[str, ...]:
    """Every provider key the settings ``state`` holds (for ``keyguard``)."""
    keys = {p.get("api_key").strip() for p in state.get("profiles") or ()
            if isinstance(p, dict) and isinstance(p.get("api_key"), str) and p.get("api_key").strip()}
    return tuple(sorted(keys, key=len, reverse=True))


def _derived_id(source: str, seen: int) -> str:
    return "p" + hashlib.sha256(f"{seen}\n{source}".encode("utf-8")).hexdigest()[:31]


def ensure_ids(state: dict) -> bool:
    """Give every profile of a settings ``state`` a valid, unique ``id`` (in place); True when one was added. See the
    module docstring: the snapshot's id for the same name, else one derived from the name, or from the position and
    provider when the name holds a key or anything shaped like one."""
    profiles = [p for p in state.get("profiles") or () if isinstance(p, dict)]
    taken = {p["id"] for p in profiles if valid_id(p.get("id"))}
    known: dict[str, str] = {}
    snapshot = state.get(STATE_KEY)
    remote = snapshot.get("remote") if isinstance(snapshot, dict) else None
    for item in (remote.get("profiles") if isinstance(remote, dict) else None) or ():
        if isinstance(item, dict) and valid_id(item.get("id")) and isinstance(item.get("name"), str):
            known.setdefault(item["name"].casefold(), item["id"])
    secrets = local_secrets(state)
    seen: set[str] = set()
    counts: dict[str, int] = {}
    changed = False
    for index, profile in enumerate(profiles):
        pid = profile.get("id")
        if valid_id(pid) and pid not in seen:
            seen.add(pid)
            continue
        name = profile.get("name") if isinstance(profile.get("name"), str) else ""
        candidate = known.get(name.casefold())
        if candidate is None or candidate in taken or candidate in seen:
            safe = not _keyguard.looks_like_a_key(name, secrets)
            source = f"name\n{name}" if safe else f"position\n{index}\n{profile.get('provider')}"
            count = counts.get(source, 0)
            candidate = _derived_id(source, count)
            while candidate in taken or candidate in seen:
                count += 1
                candidate = _derived_id(source, count)
            counts[source] = count + 1
        profile["id"] = candidate
        seen.add(candidate)
        changed = True
    return changed


def ensure_approvals(state: dict) -> bool:
    """Give every profile of a settings ``state`` its ``approved`` (in place); True when one was added. A profile
    without one (a file from before approvals, or one an Oscar that doesn't know them saved) is approved for the
    address it has HERE, at this moment: run on every read of the settings file and before anything from the server is
    applied, so a provider or base URL the server changes is never approved by default. A value that isn't a dict is
    "not approved"."""
    changed = False
    for profile in state.get("profiles") or ():
        if not isinstance(profile, dict):
            continue
        if profile.get("approved") is None:
            profile["approved"] = approval_for(profile)
            changed = True
        elif not isinstance(profile.get("approved"), dict):
            profile["approved"] = {}
            changed = True
    return changed


# -- the active profile ------------------------------------------------------------------------------------------


def active_index(state: dict) -> int:
    """The index of the active profile of a settings ``state`` (-1 without profiles). See the module docstring."""
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


def pending_active(state: dict) -> dict | None:
    """``{"id", "why", "profile"}`` of the website's active profile this computer doesn't follow yet, or None."""
    pending = state.get(PENDING_KEY)
    if not isinstance(pending, dict) or not valid_id(pending.get("id")):
        return None
    profile = next((p for p in state.get("profiles") or () if isinstance(p, dict) and p.get("id") == pending["id"]),
                   None)
    if profile is None:
        return None
    return {"id": pending["id"], "why": pending.get("why") if pending.get("why") in ("confirm", "no_key") else "confirm",
            "profile": profile}


def pending_text(pending: dict | None) -> str:
    """The notice for ``pending_active``: what the website picked and why this computer doesn't use it yet."""
    if not pending:
        return ""
    profile = pending["profile"]
    name = field_of(profile, "name") or "A profile"
    if pending["why"] == "no_key":
        return f"{name} is active on another computer; its key isn't on this one. Add it in Oscar's settings to use it."
    where = display_host(field_of(profile, "base_url")) or field_of(profile, "provider") or "its address"
    return f"The website picked {name} ({where}). Confirm it in Oscar's settings to use it."


# -- endpoints and the gate ----------------------------------------------------------------------------------------


def endpoint(provider: Any, base_url: Any) -> tuple[str, str]:
    """``(provider, base URL)`` as a destination, for comparing two of them: the URL's scheme and host in lower case,
    without a trailing ``/``. Any other difference (another host, port, path, user) is another destination."""
    url = base_url.strip() if isinstance(base_url, str) else ""
    try:
        parts = urlsplit(url)
        url = parts._replace(scheme=parts.scheme.lower(), netloc=parts.netloc.lower()).geturl()
    except ValueError:
        pass
    return (provider if isinstance(provider, str) else ""), url.rstrip("/")


def endpoint_of(profile: Any) -> tuple[str, str]:
    return endpoint(field_of(profile, "provider"), field_of(profile, "base_url"))


def approval_for(profile: Any) -> dict:
    """What ``Profile.approved`` holds once the user approved the profile as it is now."""
    provider, url = field_of(profile, "provider") or "", field_of(profile, "base_url") or ""
    return {"provider": provider if isinstance(provider, str) else "", "base_url": url if isinstance(url, str) else ""}


def display_host(base_url: Any) -> str:
    """The host (and port) of a base URL for a message (never its user, password, path or query)."""
    try:
        parts = urlsplit(base_url.strip() if isinstance(base_url, str) else "")
        host = parts.hostname or ""
        return f"{host}:{parts.port}" if host and parts.port else host
    except ValueError:
        return ""


def unusable_reason(profile: Any) -> str:
    """``""`` when ``profile`` may be used now, else why not and what to do (the one gate: ``providers.make_provider``,
    ``provider_probe``, the panel's send, the website's test and the dialog all ask it). See the module docstring."""
    if profile is None or field_of(profile, "provider") == ACCOUNT_PROVIDER:
        return ""
    approved = field_of(profile, "approved")
    if approved is None:      # never set (ensure_approvals runs on every read): approved where it is, as made here
        return ""
    name = field_of(profile, "name") or "Profile"
    where = display_host(field_of(profile, "base_url")) or "its address"
    if isinstance(approved, dict) and endpoint(approved.get("provider"), approved.get("base_url")) \
            == endpoint_of(profile):
        return ""
    if isinstance(approved, dict) and approved.get("provider"):
        return (f"The provider or base URL of the profile “{name}” was changed on your OriginShape account. Oscar "
                f"doesn't use it, or send anything to {where}, until you confirm the new address in Oscar's settings "
                "(⚙ → AI model).")
    return (f"The profile “{name}” was made on your OriginShape account. Oscar doesn't use it, or send anything to "
            f"{where}, until you confirm its address in Oscar's settings (⚙ → AI model).")


def needs_confirmation(profile: Any) -> bool:
    return bool(unusable_reason(profile))


def syncable(profile: Any) -> bool:
    """A profile OriginShape takes and the user hasn't been asked about removing (see the module docstring)."""
    return field_of(profile, "provider") in PROVIDERS and not field_of(profile, "removed_remotely")


# -- the wire format ---------------------------------------------------------------------------------------------


def publishable_url(url: Any) -> bool:
    """True for a base URL OriginShape may see (and accepts): an http(s) URL with a host, no user name or password, no
    query and no fragment, at most ``URL_LIMIT`` characters, without spaces, control characters or backslashes, and
    nothing in it that is or looks like a piece of a key (``..``, ``xxxx``, a key's shape)."""
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
            and not _keyguard.looks_like_a_key(url))


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


def account_model(model: str) -> str:
    from .auth.account import RETIRED_ALIASES

    return RETIRED_ALIASES.get(model.strip(), model.strip())


def _account_wire_model(model: str) -> str:
    """The account profile's model as the server takes it: one of ``ACCOUNT_WIRE_MODELS`` (a retired alias as its
    tier; an alias the server added since this list as the default, so the rest of the settings still sync)."""
    model = account_model(model)
    return model if model in ACCOUNT_WIRE_MODELS else ACCOUNT_WIRE_MODELS[0]


def outgoing(text: str, secrets: tuple[str, ...] = ()) -> str:
    """Free text as it may go to OriginShape: ``keyguard.scrub`` (the one policy), without NUL characters."""
    return _keyguard.scrub(text, secrets).replace("\x00", "")


def masked_fields(profile: dict, secrets: tuple[str, ...] = ()) -> list[str]:
    """The free-text fields of a stored profile that ``to_wire`` masks: their names only, for the log."""
    found = []
    for name, limit in (("name", 4 * NAME_LIMIT), ("model", MODEL_LIMIT)):
        text = _text(profile.get(name), limit)
        if outgoing(text, secrets) != text.replace("\x00", ""):
            found.append(name)
    return found


def to_wire(profile: dict, secrets: tuple[str, ...] = ()) -> dict:
    """What OriginShape receives of a stored profile (a ``params.Profile`` as a dict): see the module docstring."""
    provider = _text(profile.get("provider"), 64)
    url = profile.get("base_url").strip() if isinstance(profile.get("base_url"), str) else ""
    if provider == ACCOUNT_PROVIDER or not url:
        base_url, local = "", False
    elif publishable_url(url) and not _keyguard.looks_like_a_key(url, secrets):
        base_url, local = url, False
    else:
        base_url, local = "", True
    model = outgoing(_text(profile.get("model"), MODEL_LIMIT).strip(), secrets)
    if provider == ACCOUNT_PROVIDER:
        model = _account_wire_model(model)
    name = " ".join(outgoing(_text(profile.get("name"), 4 * NAME_LIMIT), secrets).split())[:NAME_LIMIT] or "Profile"
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
        "has_key": isinstance(api_key, str) and bool(api_key.strip()),
    }


def wire_profile(profile: dict) -> dict:
    """A wire profile with exactly ``WIRE_FIELDS`` (and the five override keys), whatever else ``profile`` holds."""
    out = {name: profile.get(name) for name in WIRE_FIELDS}
    out["overrides"] = {key: (profile.get("overrides") or {}).get(key) for key in OVERRIDE_KEYS}
    return out


def local_settings(state: dict) -> dict:
    """This computer's settings in the wire format: ``{"active_profile_id", "profiles", "excluded"}``. ``excluded``
    lists the ids of the profiles that are not synced (``syncable``). A name masked into one another profile has is
    told apart by its id. The active profile is the website's pending choice while there is one (it is not a change
    made here), and the snapshot's when the profile in use here is not synced."""
    secrets = local_secrets(state)
    profiles, excluded, names = [], [], set()
    for profile in state.get("profiles") or ():
        if not isinstance(profile, dict) or not valid_id(profile.get("id")):
            continue
        if not syncable(profile):
            excluded.append(profile["id"])
            continue
        wire = to_wire(profile, secrets)
        if wire["name"].casefold() in names and masked_fields(profile, secrets):
            wire["name"] = f"{wire['name'][:NAME_LIMIT - 9]} ({profile['id'][:6]})"
        names.add(wire["name"].casefold())
        profiles.append(wire)
    ids = {p["id"] for p in profiles}
    pending = pending_active(state)
    index = active_index(state)
    active = pending["id"] if pending else (state["profiles"][index].get("id") if index >= 0 else None)
    if active not in ids:
        snapshot = state.get(STATE_KEY)
        remote = snapshot.get("remote") if isinstance(snapshot, dict) else None
        before = remote.get("active_profile_id") if isinstance(remote, dict) else None
        active = before if before in ids else (profiles[0]["id"] if profiles else None)
    return {"active_profile_id": active, "profiles": profiles, "excluded": excluded}


def normalise_remote(data: Any) -> dict:
    """``AiSettings`` as the server sent it, made comparable with ``local_settings``: ``{"version", "updated_at",
    "updated_by", "active_profile_id", "profiles"}``. Profiles without a usable id are left out, and so is anything
    but ``WIRE_FIELDS`` (an older server's ``notes``). Raises ``ValueError`` for an answer that is not AiSettings."""
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
            "model": account_model(model) if provider == ACCOUNT_PROVIDER else model,
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
