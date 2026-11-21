"""The three-way merge of the AI model settings (issue #86): this computer's against the server's, with the snapshot of
the last sync as the base, and the settings dialog's Save against what a sync wrote while it was open. Pure Python, on
the wire format of ``settings_wire``.

**Per profile and per field.** A field changed only on the server is taken, one changed only here is kept (and
pushed); one changed on both sides goes to the server on a pull and to us after a Save (``prefer="local"``). A
profile new on either side is kept; names are made unique, as the server requires.

**Deletions.** A profile deleted here goes from the server unless it changed there and this is a pull. A profile
deleted on the server goes here only when it has no key and didn't change here since the last sync; one with a key is
kept, marked ``removed_remotely`` (not synced) until the user removes or keeps it in the settings dialog. A pull that
would remove more than half of this computer's profiles is not applied at all (``Plan.blocked``).

**What a pull never does here:** replace a key, a local base URL (by ``""``), the notes or anything else local; make a
profile it created or redirected usable (``Profile.approved`` stays as the user left it, see ``settings_wire``); switch
chat to a profile this computer can't use yet (it becomes ``active_pending`` instead).

Two device-specific fields, ``has_key`` and the base URL (one FreeCAD may hold a key or a local URL another one does
not), are compared with what THIS FreeCAD last had accepted by the server (``ai_sync.device``), so two computers never
undo each other on every sync. ``has_key`` is "a FreeCAD holds a key": true while this one does.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .settings_wire import (ACCOUNT_PROVIDER, MAX_PROFILES, NAME_LIMIT, OVERRIDE_KEYS, PENDING_KEY, STATE_KEY,
                            active_index, approval_for, ensure_approvals, ensure_ids, local_secrets, local_settings,
                            masked_fields, normalise_remote,
                            pending_active, same_settings, set_active, syncable, unusable_reason, wire_profile)

_REGULAR = ("name", "provider", "model", "reasoning", *(f"overrides.{key}" for key in OVERRIDE_KEYS))
_RECOMPUTE_FIELDS = ("provider", "model", "overrides.max_output", "overrides.temperature")
DERIVED_FIELDS = ("temperature", "max_tokens")   # recomputed from provider, model and overrides: not an edit
ATOMIC_FIELDS = ("approved",)                    # merged whole: an approval is of one exact endpoint


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
    """One profile of a merge. ``wire`` goes to the server when ``push``; ``apply`` (field -> value) and ``url`` (a
    real base URL the server took, or None to keep ours) change the local profile; ``new`` makes it here from ``wire``;
    ``mark_removed`` keeps a profile the server deleted and asks the user; ``keep_local`` leaves a local profile that
    isn't synced alone (the server's copy is sent back as it is)."""

    id: str
    wire: dict
    apply: dict = field(default_factory=dict)
    url: str | None = None
    new: bool = False
    push: bool = True
    mark_removed: bool = False
    keep_local: bool = False


@dataclass
class Merged:
    entries: list[Entry]
    active: str | None
    removals: int = 0          # profiles here that the server's settings remove (deleted, or marked removed)

    def settings(self) -> dict:
        return {"active_profile_id": self.active, "profiles": [e.wire for e in self.entries if e.push]}


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


def numbered(name: str, taken: set[str]) -> str:
    """``name 2``, ``name 3`` ... the first one not in ``taken`` (casefolded), within the server's name limit."""
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
    for entry in sorted((e for e in entries if e.push), key=lambda e: e.id not in keep):
        name = entry.wire["name"]
        if name.casefold() in taken:
            name = numbered(name, taken)
            entry.wire["name"] = name
            if not entry.new:
                entry.apply["name"] = name
        taken.add(name.casefold())


def merge(base: dict | None, local: dict, remote: dict, *, device: dict | None = None, prefer: str = "remote",
          excluded: Any = ()) -> Merged:
    """Three-way merge of the wire settings ``local`` (this computer) and ``remote`` (the server) against ``base`` (the
    server's settings at the last sync; None for a first sync with this account). ``device`` is this FreeCAD's
    ``has_key`` and base URL per profile at the last sync; ``excluded`` the ids of local profiles that are not synced.
    See the module docstring."""
    device = device or {}
    excluded = set(excluded or ())
    by_base = {p["id"]: p for p in base["profiles"]} if base is not None else {}
    by_local = {p["id"]: p for p in local["profiles"]}
    by_remote = {p["id"]: p for p in remote["profiles"]}
    entries: list[Entry] = []
    removals = 0
    for pid in _order(list(by_local), list(by_remote), prefer):
        b, ours, theirs = by_base.get(pid), by_local.get(pid), by_remote.get(pid)
        if ours is not None and theirs is not None:
            entries.append(_merge_profile(b, ours, theirs, device.get(pid), prefer))
        elif ours is not None:
            if b is not None and not _changed_here(ours, b, device.get(pid)):   # deleted on the server
                removals += 1
                if ours["has_key"]:   # its key goes only when the user says so: kept, marked, not synced
                    entries.append(Entry(pid, copy.deepcopy(ours), push=False, mark_removed=True))
                continue
            entries.append(Entry(pid, copy.deepcopy(ours)))   # new here, or changed here since: kept (and pushed)
        elif pid in excluded:          # a local profile that isn't synced: the server's copy stays as it is
            entries.append(Entry(pid, copy.deepcopy(theirs), keep_local=True))
        else:
            if b is not None and (prefer == "local" or not _changed_there(theirs, b)):
                continue                      # deleted here: the server deletes it too
            entries.append(Entry(pid, copy.deepcopy(theirs), new=True))
    side = by_remote if prefer == "remote" else by_local
    _unique_names(entries, {e.id for e in entries if e.id in side and e.wire["name"] == side[e.id]["name"]})
    pushed = [e for e in entries if e.push]
    for entry in reversed(pushed):      # at most MAX_PROFILES: what the server has first, the rest stays local
        if len([e for e in entries if e.push]) <= MAX_PROFILES:
            break
        if entry.id not in by_remote:
            entry.push = False
    ids = [e.id for e in entries if e.push]
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
    return Merged(entries, active, removals)


def adopt_remote_ids(state: dict, remote: dict) -> int:
    """First sync with an account: a local profile the server has under another id (the same name, or the only
    account profile on each side) takes the server's id, so it is merged instead of doubled. Returns how many."""
    profiles = [p for p in state.get("profiles") or () if isinstance(p, dict)]
    remote_ids = {p["id"] for p in remote["profiles"]}
    local_ids = {p.get("id") for p in profiles}
    claimed = {pid for pid in local_ids if pid in remote_ids}
    by_name: dict[str, str] = {}
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
    """A new local profile (``params.Profile`` as a dict) from one the server has: no key, no base URL when the real
    one is kept on another computer, and not approved (the user confirms its address before it is used)."""
    return {
        "name": wire["name"], "provider": wire["provider"],
        "base_url": "" if wire["base_url_local"] else wire["base_url"], "model": wire["model"], "api_key": "",
        "temperature": 0.2, "max_tokens": 4096, "notes": "", "reasoning": wire["reasoning"],
        "caps_override": {k: v for k, v in wire["overrides"].items() if v is not None}, "id": wire["id"],
        "approved": {}, "removed_remotely": False,
    }


def _with_unseen(result: list[dict], before: list[dict], seen: set[str]) -> list[dict]:
    """``result`` with the local profiles the merge didn't see because they are not synced (``syncable``: a provider
    OriginShape doesn't know, a removal waiting for the user) back where they were. A synced profile the merge left out
    was deleted."""
    out = list(result)
    for i, profile in enumerate(before):
        if not isinstance(profile, dict) or profile.get("id") in seen or syncable(profile):
            continue
        ids = [p.get("id") for p in out]
        prev = next((before[j].get("id") for j in range(i - 1, -1, -1) if before[j].get("id") in ids), None)
        out.insert(ids.index(prev) + 1 if prev is not None else 0, profile)
    return out


def _usable_here(profile: dict, remote_has_key: bool) -> str:
    """Why chat can't switch to ``profile`` now: ``"confirm"`` (its address isn't approved here), ``"no_key"`` (its
    key is on another computer), or ``""``."""
    if unusable_reason(profile):
        return "confirm"
    key = profile.get("api_key") if isinstance(profile.get("api_key"), str) else ""
    if remote_has_key and not key.strip() and profile.get("provider") != ACCOUNT_PROVIDER:
        return "no_key"
    return ""


def apply_merge(state: dict, merged: Merged, remote: dict | None = None,
                limits: Callable[[dict], tuple[float, int]] | None = None) -> None:
    """Make the settings ``state`` what ``merged`` says (in place); see the module docstring. ``limits(profile)``
    recomputes the stored temperature and output cap of a profile whose provider, model or overrides changed."""
    before = [p for p in state.get("profiles") or () if isinstance(p, dict)]
    by_id = {p.get("id"): p for p in before}
    index = active_index(state)
    current_active = before[index].get("id") if index >= 0 else None
    result: list[dict] = []
    seen: set[str] = set()
    for entry in merged.entries:
        seen.add(entry.id)
        current = by_id.get(entry.id)
        if entry.keep_local:
            if current is not None:
                result.append(current)
            continue
        if current is None:
            profile, touched = stored_from_wire(entry.wire), True
        else:
            profile = copy.deepcopy(current)
            if profile.get("approved") is None:   # approved where it is now, never where the server moves it
                profile["approved"] = approval_for(current)
            for name, value in entry.apply.items():
                _set_field(profile, name, value)
            if entry.url is not None:
                profile["base_url"] = entry.url
            if entry.mark_removed:
                profile["removed_remotely"] = True
            touched = entry.url is not None or any(name in entry.apply for name in _RECOMPUTE_FIELDS) \
                or any(name.startswith("overrides.") for name in entry.apply)
        if touched and limits is not None:
            try:
                profile["temperature"], profile["max_tokens"] = limits(profile)
            except Exception:  # noqa: BLE001 - the stored values are recomputed at every turn anyway
                pass
        result.append(profile)
    result = _with_unseen(result, before, seen)
    state["profiles"] = result
    ids = [p.get("id") for p in result]
    remote_keys = {p["id"]: p["has_key"] for p in (remote or {}).get("profiles") or ()}
    target = merged.active if merged.active in ids else None
    why = _usable_here(result[ids.index(target)], remote_keys.get(target, False)) if target else ""
    keep = None
    if why and target != current_active:   # a switch chat can't follow yet: it keeps what it used
        keep = current_active if current_active in ids else None
        if keep is None:                    # that one is gone: the first it can use (the account's first)
            usable = sorted((p for p in result if not _usable_here(p, False) and p.get("id") != target),
                            key=lambda p: p.get("provider") != ACCOUNT_PROVIDER)
            keep = usable[0].get("id") if usable else None
    if keep is not None:
        state[PENDING_KEY] = {"id": target, "why": why}
        set_active(state, ids.index(keep))
    else:   # the profile in use itself changed: it stays in use, and the gate says what to confirm
        state.pop(PENDING_KEY, None)
        set_active(state, ids.index(target) if target else (0 if result else -1))


def _profile_view(state: dict) -> str:
    index = active_index(state)
    return json.dumps([state.get("profiles") or [], index, state.get(PENDING_KEY)], sort_keys=True, default=str)


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
    active profile changed; ``device`` is this FreeCAD's key and URL record once the server has the payload;
    ``blocked`` says why nothing was applied; ``local_only`` counts the profiles that stay on this computer only."""

    payload: dict
    push: bool
    changed: bool
    device: dict
    version: int
    account: str
    adopted: int = 0
    masked: dict = field(default_factory=dict)   # free-text field -> how many profiles had a key masked in it
    blocked: str = ""
    removals: int = 0
    local_only: int = 0
    unknown_provider: int = 0


def blocked_text(removals: int, total: int) -> str:
    return (f"Not synced: the settings on your OriginShape account would remove {removals} of your {total} profiles "
            "here, so Oscar didn't apply them and nothing was removed. Check Settings → AI model on the website; to "
            "remove them here too, remove them in Oscar's settings.")


def plan_sync(state: dict, remote: dict, *, prefer: str, account: str,
              limits: Callable[[dict], tuple[float, int]] | None = None) -> tuple[bool, Plan]:
    """Merge the server's ``remote`` settings into the settings ``state`` (in place); ``(write, Plan)``. The snapshot
    becomes ``remote`` (the next merge's base), so what is still to be pushed stays a local change."""
    ensure_ids(state)
    ensure_approvals(state)   # before anything of the server's: a profile without approval is approved where it is
    pristine = copy.deepcopy(state)
    before = _profile_view(state)
    old_snapshot = copy.deepcopy(state.get(STATE_KEY))
    snap = usable_snapshot(old_snapshot, account, remote["version"])
    adopted = adopt_remote_ids(state, remote) if snap is None else 0
    device = snap["device"] if snap is not None else {}
    local = local_settings(state)
    merged = merge(snap["remote"] if snap is not None else None, local, remote, device=device, prefer=prefer,
                   excluded=local["excluded"])
    total = len(state.get("profiles") or ())
    if merged.removals and merged.removals * 2 > total:
        state.clear()
        state.update(pristine)            # nothing of this pull is applied
        return False, Plan({}, False, False, {}, remote["version"], account, blocked=blocked_text(merged.removals,
                                                                                                  total),
                           removals=merged.removals)
    apply_merge(state, merged, remote, limits)
    after = local_settings(state)
    settings = merged.settings()
    payload = {"version": remote["version"], "active_profile_id": settings["active_profile_id"],
               "profiles": [wire_profile(p) for p in settings["profiles"]]}
    push = not same_settings(payload, remote)
    now = {p["id"]: _device(p) for p in after["profiles"]}
    state[STATE_KEY] = {"account": account, "version": remote["version"],
                        "remote": {"active_profile_id": remote["active_profile_id"],
                                   "profiles": copy.deepcopy(remote["profiles"])},
                        "device": device if push else now}
    masked: dict[str, int] = {}
    secrets = local_secrets(state)
    for profile in state["profiles"]:
        for name in masked_fields(profile, secrets):
            masked[name] = masked.get(name, 0) + 1
    unknown = sum(1 for p in state["profiles"] if isinstance(p, dict) and p.get("id") in after["excluded"]
                  and not p.get("removed_remotely"))
    local_only = unknown + sum(1 for e in merged.entries if not e.push and not e.mark_removed and not e.keep_local)
    changed = _profile_view(state) != before
    plan = Plan(payload, push, changed, now, remote["version"], account, adopted, masked,
                removals=merged.removals, local_only=local_only, unknown_provider=unknown)
    return changed or state[STATE_KEY] != old_snapshot, plan


def confirm_push(state: dict, plan: Plan, version: int) -> tuple[bool, None]:
    """The server took ``plan.payload`` as ``version``: it is the next merge's base, with this FreeCAD's record."""
    state[STATE_KEY] = {"account": plan.account, "version": version,
                        "remote": {"active_profile_id": plan.payload["active_profile_id"],
                                   "profiles": copy.deepcopy(plan.payload["profiles"])},
                        "device": plan.device}
    return True, None


# -- the dialog's Save: its edits merged with what a pull wrote meanwhile ---------------------------------------


def _merge_value(base: Any, ours: Any, theirs: Any, atomic: tuple[str, ...] = ()) -> Any:
    if isinstance(base, dict) and isinstance(ours, dict) and isinstance(theirs, dict):
        out = {}
        for key in [*theirs, *(k for k in ours if k not in theirs), *(k for k in base if k not in theirs and k not in ours)]:
            missing = object()
            b, o, t = base.get(key, missing), ours.get(key, missing), theirs.get(key, missing)
            value = (t if o == b else o) if key in atomic else _merge_value(b, o, t)
            if value is not missing:
                out[key] = value
        return out
    return theirs if ours == base else ours


def _edited(profile: dict, base: dict) -> bool:
    """True when the dialog changed ``profile`` (a stored dict) since it loaded ``base``."""
    keys = (set(profile) | set(base)) - set(DERIVED_FIELDS)
    return any(profile.get(key) != base.get(key) for key in keys)


def merge_saved(base: dict, ours: dict, theirs: dict) -> dict:
    """The settings dialog's Save. ``base`` is what it loaded, ``ours`` what it saves, ``theirs`` what the file holds
    now (a pull may have changed it while the dialog was open); each ``{"profiles": [stored dicts],
    "active_profile_id"}``. Per profile and field, what the dialog changed wins and the rest is the file's: a profile
    the dialog removed goes, one a pull added or changed stays. ``approved`` is merged whole, so an approval given in
    the dialog is of the endpoint the dialog showed: when a pull moved the profile meanwhile, it waits for approval
    again (with the key typed in the dialog unused until then)."""
    by_base = {p["id"]: p for p in base["profiles"]}
    by_ours = {p["id"]: p for p in ours["profiles"]}
    by_theirs = {p["id"]: p for p in theirs["profiles"]}
    profiles: list[dict] = []
    for pid in _order(list(by_theirs), list(by_ours), "local"):
        b, o, t = by_base.get(pid), by_ours.get(pid), by_theirs.get(pid)
        if o is not None and t is not None:
            profiles.append(_merge_value(b, o, t, ATOMIC_FIELDS) if b is not None else copy.deepcopy(o))
        elif o is not None:
            if b is None or _edited(o, b):   # new in the dialog, or edited there after a pull removed it
                profiles.append(copy.deepcopy(o))
        elif b is None:                      # a pull added it while the dialog was open
            profiles.append(copy.deepcopy(t))
    taken: set[str] = set()
    mine = set(by_ours)
    for profile in sorted(profiles, key=lambda p: p["id"] not in mine):
        name = profile.get("name") if isinstance(profile.get("name"), str) else "Profile"
        if name.casefold() in taken:
            profile["name"] = name = numbered(name, taken)
        taken.add(name.casefold())
    ids = [p["id"] for p in profiles]
    active = ours.get("active_profile_id")
    if active == base.get("active_profile_id") and theirs.get("active_profile_id") in ids:
        active = theirs.get("active_profile_id")
    if active not in ids:
        active = next((a for a in (theirs.get("active_profile_id"),) if a in ids), ids[0] if ids else None)
    return {"profiles": profiles, "active_profile_id": active}


def settle_pending(state: dict, chosen_here: bool) -> bool:
    """After a Save or a profile picked in FreeCAD: a choice made here ends the website's pending one; else the
    pending profile becomes active once this computer can use it (approved, with its key). True when it changed."""
    pending = pending_active(state)
    if pending is None:
        return state.pop(PENDING_KEY, None) is not None
    if chosen_here:
        state.pop(PENDING_KEY, None)
        return True
    if _usable_here(pending["profile"], pending["why"] == "no_key"):
        return False
    state.pop(PENDING_KEY, None)
    ids = [p.get("id") for p in state.get("profiles") or ()]
    set_active(state, ids.index(pending["id"]))
    return True
