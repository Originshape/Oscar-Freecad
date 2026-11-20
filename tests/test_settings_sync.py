"""oscar.settings_sync: the AI model settings synced with the OriginShape account (issue #86), against the fake API.

Covered: profile ids (derived the same on every read for a file from before ids, unique, found again by name from the
snapshot) and the active profile (by id, a name an older Oscar wrote wins); the wire format (no key, no part of one,
anywhere in the serialized payload; a base URL with a user, password, query or fragment stays local; ``has_key``; the
Advanced overrides); the three-way merge per profile and field (a change on one side, on both sides for a pull and for
a Save, a deletion against an edit, new profiles on both sides with the same name, the active profile, two FreeCADs
that hold different keys and URLs); a first sync with an account (ids adopted by name and for the account profile), a
server that forgot its settings, another account; a pull never replacing a key, ``model_caps``, the preferences or a
local base URL, and a key set aside when the account moves its profile to another host; the ``Syncer`` against the
fake (signed out and API-key sign-ins send nothing, a first push, a pull, nothing to push, a 409 merged and pushed
once more, a second 409, a server without the route); the dialog's Save merge; the thread (``request``,
``pull_if_stale``) and the off switch. Connect: ``hello`` announces ``ai_settings`` only with a handler, a
``profile_test`` runs and its result is posted, an unknown profile and a failing handler; the message is scrubbed of the
key (plain, masked, in pieces) and of a local base URL's secrets, and clipped to 500 characters.
Pure python (stdlib + requests); the keystore runs in memory.
"""
import copy
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import connect, settings_sync  # noqa: E402
from oscar.auth import account, discovery, keystore, session  # noqa: E402
from oscar.auth.transport import TokenSet  # noqa: E402
from support.fake_workspace import FakeWorkspace  # noqa: E402

KEY = "sk-ant-api03-" + "Zq7xT2wPk9" * 5
OPENAI_KEY = "sk-proj-" + "a1B2c3D4e5" * 5
LOCAL_KEY = "lm-studio-local-7"
NO_OVERRIDES = {"max_output": None, "temperature": None, "context_window": None, "tools": None, "ollama_native": None}


def stored(name, pid, *, provider="anthropic", base_url="https://api.anthropic.com", model="claude-sonnet-4-5",
           api_key="", notes="", reasoning="auto", caps=None, **extra):
    """A profile as the settings file holds it (``params.Profile`` as a dict)."""
    return {"name": name, "provider": provider, "base_url": base_url, "model": model, "api_key": api_key,
            "temperature": 0.2, "max_tokens": 32000, "notes": notes, "reasoning": reasoning,
            "caps_override": dict(caps or {}), "id": pid, "held_key": {}, **extra}


def account_profile(pid="acct", name="OriginShape"):
    return stored(name, pid, provider="originshape", base_url="", model="origin1")


def wire(name, pid, *, provider="anthropic", base_url="https://api.anthropic.com", model="claude-sonnet-4-5",
         has_key=False, local=False, notes="", reasoning="auto", overrides=None):
    return {"id": pid, "name": name, "provider": provider, "base_url": "" if local else base_url,
            "base_url_local": local, "model": model, "reasoning": reasoning,
            "overrides": {**NO_OVERRIDES, **(overrides or {})}, "notes": notes, "has_key": has_key}


def settings(*profiles, active=None, version=0):
    return {"version": version, "updated_at": None, "updated_by": None,
            "active_profile_id": active if active is not None else (profiles[0]["id"] if profiles else None),
            "profiles": [copy.deepcopy(p) for p in profiles]}


def state_of(*profiles, active=None, **extra):
    profiles = [copy.deepcopy(p) for p in profiles]
    active_profile = next((p for p in profiles if p["id"] == active), profiles[0] if profiles else None)
    return {"profiles": profiles, "active_profile": active_profile["name"] if active_profile else "",
            "active_profile_id": active_profile["id"] if active_profile else None,
            "prefs": {"DockArea": 2, "ConnectWorkspace": True},
            "model_caps": {"anthropic|https://api.anthropic.com|claude-sonnet-4-5": {"learned": {"tools": "native"}}},
            "net_grants": [{"tool": "fetch_url", "host": "a.example.com", "granted_at": ""}], **extra}


def substrings(secret: str, size: int = 8):
    return {secret[i:i + size] for i in range(len(secret) - size + 1)}


class MemoryStore:
    """The settings file in memory, with the same transaction contract as ``params.update_state``."""

    def __init__(self, state):
        self.state = copy.deepcopy(state)
        self.writes = 0
        self.lock = threading.Lock()

    def transaction(self, mutate):
        with self.lock:
            state = copy.deepcopy(self.state)
            settings_sync.ensure_ids(state)
            write, result = mutate(state)
            if write:
                self.state = state
                self.writes += 1
            return result

    @staticmethod
    def limits(profile):
        return 0.2, 8192

    def profile(self, profile_id):
        item = next((p for p in self.state["profiles"] if p.get("id") == profile_id), None)
        return SimpleNamespace(**item) if item is not None else None

    def by_id(self, pid):
        return next((p for p in self.state["profiles"] if p["id"] == pid), None)

    def names(self):
        return [p["name"] for p in self.state["profiles"]]

    def active(self):
        return self.state["profiles"][settings_sync.active_index(self.state)]["id"]


# -- ids and the active profile ------------------------------------------------------------------------------------


class IdsTest(unittest.TestCase):
    def test_a_file_from_before_ids_gets_the_same_ids_on_every_read(self):
        old = {"profiles": [{"name": "Work"}, {"name": "Home"}, {"name": "Work"}], "active_profile": "Home"}
        first, second = copy.deepcopy(old), copy.deepcopy(old)
        self.assertTrue(settings_sync.ensure_ids(first))
        settings_sync.ensure_ids(second)
        ids = [p["id"] for p in first["profiles"]]
        self.assertEqual(ids, [p["id"] for p in second["profiles"]])
        self.assertEqual(len(set(ids)), 3, "a repeated name still gets its own id")
        self.assertTrue(all(settings_sync.valid_id(pid) for pid in ids))
        self.assertFalse(settings_sync.ensure_ids(first), "ids once given are kept")

    def test_duplicate_and_invalid_ids_are_replaced(self):
        state = {"profiles": [{"name": "A", "id": "x"}, {"name": "B", "id": "x"}, {"name": "C", "id": "bad id!"}]}
        settings_sync.ensure_ids(state)
        ids = [p["id"] for p in state["profiles"]]
        self.assertEqual(ids[0], "x")
        self.assertEqual(len(set(ids)), 3)

    def test_ids_an_older_oscar_dropped_are_found_again_by_name(self):
        state = {"profiles": [{"name": "Work"}, {"name": "New"}],
                 settings_sync.STATE_KEY: {"remote": {"profiles": [{"id": "srv-work", "name": "work"}]}}}
        settings_sync.ensure_ids(state)
        self.assertEqual(state["profiles"][0]["id"], "srv-work")
        self.assertNotEqual(state["profiles"][1]["id"], "srv-work")

    def test_the_active_profile_follows_its_id_and_an_older_oscars_choice_by_name(self):
        state = {"profiles": [{"id": "a", "name": "A"}, {"id": "b", "name": "B renamed"}],
                 "active_profile_id": "b", "active_profile": "B renamed"}
        self.assertEqual(settings_sync.active_index(state), 1)
        state["active_profile"] = "A"                  # an older Oscar switched to A by name
        self.assertEqual(settings_sync.active_index(state), 0)
        state["active_profile"] = "gone"                # a name nobody has: the id still counts
        self.assertEqual(settings_sync.active_index(state), 1)
        state.update(active_profile_id="zzz", active_profile="")
        self.assertEqual(settings_sync.active_index(state), 0)
        self.assertEqual(settings_sync.active_index({"profiles": []}), -1)
        self.assertTrue(settings_sync.set_active(state, 1))
        self.assertEqual((state["active_profile_id"], state["active_profile"]), ("b", "B renamed"))
        self.assertFalse(settings_sync.set_active(state, 1))


# -- what OriginShape sees -------------------------------------------------------------------------------------------


class WireTest(unittest.TestCase):
    def test_no_key_and_no_part_of_one_is_in_the_payload(self):
        state = state_of(
            stored("Claude", "p1", api_key=KEY, notes=f"my key is {KEY} (do not share)"),
            stored(f"OpenAI {OPENAI_KEY}", "p2", provider="openai_compat", base_url="https://api.openai.com/v1",
                   model="gpt-4o", api_key=OPENAI_KEY),
            stored("LM Studio", "p3", provider="openai_compat", base_url="http://localhost:1234/v1", model="qwen",
                   api_key=LOCAL_KEY, notes=LOCAL_KEY),
            stored("Held", "p4", held_key={"key": KEY[::-1], "provider": "anthropic", "base_url": ""}),
        )
        payload = {"version": 0, **settings_sync.local_settings(state)}
        text = json.dumps(payload)
        for secret in (KEY, OPENAI_KEY, LOCAL_KEY, KEY[::-1]):
            self.assertNotIn(secret, text)
            for piece in substrings(secret):
                self.assertNotIn(piece, text, f"a part of a key: {piece!r}")
        self.assertNotIn("api_key", text)
        self.assertNotIn("held_key", text)
        self.assertEqual([p["has_key"] for p in payload["profiles"]], [True, True, True, False])
        self.assertIn("[redacted credential]", payload["profiles"][0]["notes"])
        for p in payload["profiles"]:
            self.assertEqual(set(p), {"id", "name", "provider", "base_url", "base_url_local", "model", "reasoning",
                                      "overrides", "notes", "has_key"})

    def test_a_base_url_that_carries_a_credential_stays_on_this_computer(self):
        local = ("https://user:pass@proxy.example.com/v1", "https://api.example.com/v1?key=abc123",
                 "https://api.example.com/v1#token", "localhost:11434/v1", "", "ftp://example.com",
                 f"https://gw.example.com/{KEY}/v1", "https://" + "a" * 600 + ".com")
        for url in local:
            with self.subTest(url=url):
                w = settings_sync.to_wire(stored("X", "p", provider="openai_compat", base_url=url, api_key=KEY),
                                          (KEY,))
                self.assertEqual((w["base_url"], w["base_url_local"]), ("", True))
        w = settings_sync.to_wire(stored("X", "p", provider="openai_compat", base_url="http://localhost:11434/v1"))
        self.assertEqual((w["base_url"], w["base_url_local"]), ("http://localhost:11434/v1", False))
        w = settings_sync.to_wire(account_profile())
        self.assertEqual((w["base_url"], w["base_url_local"], w["model"]), ("", False, "origin1"))
        w = settings_sync.to_wire(dict(account_profile(), model="oscar-default"))
        self.assertEqual(w["model"], "origin1-super", "a retired alias goes as its tier")

    def test_the_advanced_overrides_and_only_them(self):
        caps = {"max_output": 4096, "temperature": 0.33333, "context_window": 5, "tools": "text",
                "ollama_native": False, "vision": False}
        w = settings_sync.to_wire(stored("X", "p", caps=caps, reasoning="high"))
        self.assertEqual(w["overrides"], {"max_output": 4096, "temperature": 0.333, "context_window": None,
                                          "tools": "text", "ollama_native": False})
        self.assertEqual(w["reasoning"], "high")
        self.assertEqual(settings_sync.overrides_of({"tools": "auto", "ollama_native": True, "max_output": True}),
                         NO_OVERRIDES)

    def test_what_the_server_sends_is_read_the_same_way(self):
        remote = settings_sync.normalise_remote({"version": 3, "active_profile_id": "nobody", "profiles": [
            {"id": "a", "name": "  A  ", "provider": "anthropic", "base_url": "https://x.example", "base_url_local": False,
             "model": "m", "reasoning": "weird", "overrides": {"tools": "auto", "max_output": 512}, "notes": "n",
             "has_key": True},
            {"id": "bad id", "name": "dropped"}, {"id": "a", "name": "duplicate"}]})
        self.assertEqual(remote["version"], 3)
        self.assertIsNone(remote["active_profile_id"])
        self.assertEqual(len(remote["profiles"]), 1)
        p = remote["profiles"][0]
        self.assertEqual((p["name"], p["reasoning"], p["overrides"]["tools"], p["overrides"]["max_output"]),
                         ("A", "auto", None, 512))
        with self.assertRaises(ValueError):
            settings_sync.normalise_remote({"version": -1, "profiles": []})
        with self.assertRaises(ValueError):
            settings_sync.normalise_remote(["not", "settings"])


# -- the three-way merge ---------------------------------------------------------------------------------------------


class MergeTest(unittest.TestCase):
    def local(self, *profiles, active=None):
        return {"active_profile_id": active or profiles[0]["id"], "profiles": [copy.deepcopy(p) for p in profiles]}

    def test_a_change_on_one_side_is_taken(self):
        base = settings(wire("A", "a", model="m1"), wire("B", "b"))
        local = self.local(wire("A", "a", model="m1"), wire("B renamed here", "b"))
        remote = settings(wire("A", "a", model="m2"), wire("B", "b"))
        merged = settings_sync.merge(base, local, remote)
        a, b = merged.entries
        self.assertEqual((a.wire["model"], a.apply), ("m2", {"model": "m2"}))           # the server's change, applied
        self.assertEqual((b.wire["name"], b.apply), ("B renamed here", {}))             # ours, pushed

    def test_a_field_both_changed_goes_to_the_server_on_a_pull_and_to_us_after_a_save(self):
        base = settings(wire("A", "a", model="m1", overrides={"max_output": 1024}))
        local = self.local(wire("A", "a", model="mine", overrides={"tools": "text", "max_output": 1024}))
        remote = settings(wire("A", "a", model="theirs", overrides={"max_output": 2048}))
        pull = settings_sync.merge(base, local, remote, prefer="remote").entries[0]
        save = settings_sync.merge(base, local, remote, prefer="local").entries[0]
        self.assertEqual(pull.wire["model"], "theirs")
        self.assertEqual(save.wire["model"], "mine")
        for entry in (pull, save):   # per override: each side's own change is kept whatever is preferred
            self.assertEqual((entry.wire["overrides"]["tools"], entry.wire["overrides"]["max_output"]), ("text", 2048))

    def test_a_profile_deleted_on_the_server_goes_unless_it_changed_here(self):
        base = settings(wire("A", "a"), wire("B", "b"), wire("C", "c"))
        device = {"a": {"has_key": False, "url": ["https://api.anthropic.com", False]},
                  "b": {"has_key": False, "url": ["https://api.anthropic.com", False]},
                  "c": {"has_key": False, "url": ["https://api.anthropic.com", False]}}
        local = self.local(wire("A", "a"), wire("B", "b", model="edited here"), wire("C", "c", has_key=True))
        remote = settings(wire("A", "a"))
        merged = settings_sync.merge(base, local, remote, device=device)
        self.assertEqual([e.id for e in merged.entries], ["a", "b", "c"])
        self.assertEqual(merged.entries[1].wire["model"], "edited here")
        local = self.local(wire("A", "a"), wire("B", "b"), wire("C", "c"))
        self.assertEqual([e.id for e in settings_sync.merge(base, local, remote, device=device).entries], ["a"])

    def test_a_profile_deleted_here_goes_from_the_server_unless_it_changed_there_and_we_pull(self):
        base = settings(wire("A", "a"), wire("B", "b"))
        local = self.local(wire("A", "a"))
        self.assertEqual([e.id for e in settings_sync.merge(base, local, settings(wire("A", "a"), wire("B", "b")))
                          .entries], ["a"])
        edited = settings(wire("A", "a"), wire("B", "b", model="edited on the web"))
        pull = settings_sync.merge(base, local, edited, prefer="remote")
        self.assertEqual([(e.id, e.new) for e in pull.entries], [("a", False), ("b", True)])
        save = settings_sync.merge(base, local, edited, prefer="local")
        self.assertEqual([e.id for e in save.entries], ["a"])

    def test_new_profiles_on_both_sides_are_kept_and_names_stay_unique(self):
        base = settings(wire("A", "a"))
        local = self.local(wire("A", "a"), wire("Work", "mine"))
        remote = settings(wire("A", "a"), wire("work", "theirs"))
        merged = settings_sync.merge(base, local, remote)
        names = {e.id: e.wire["name"] for e in merged.entries}
        self.assertEqual(names, {"a": "A", "theirs": "work", "mine": "Work 2"})
        self.assertEqual([e.apply.get("name") for e in merged.entries if e.id == "mine"], ["Work 2"])
        self.assertTrue(next(e for e in merged.entries if e.id == "theirs").new)

    def test_the_active_profile(self):
        base = settings(wire("A", "a"), wire("B", "b"), wire("C", "c"), active="a")
        both = (wire("A", "a"), wire("B", "b"), wire("C", "c"))
        cases = [("a", "b", "remote", "b"), ("b", "a", "remote", "b"), ("b", "c", "remote", "c"),
                 ("b", "c", "local", "b"), ("a", "a", "remote", "a")]
        for ours, theirs, prefer, expected in cases:
            with self.subTest(ours=ours, theirs=theirs, prefer=prefer):
                merged = settings_sync.merge(base, self.local(*both, active=ours), settings(*both, active=theirs),
                                             prefer=prefer)
                self.assertEqual(merged.active, expected)
        gone = settings_sync.merge(base, self.local(wire("A", "a"), active="a"), settings(wire("A", "a"), active="a"))
        self.assertEqual(gone.active, "a")

    def test_two_freecads_with_different_keys_and_urls_never_undo_each_other(self):
        # this FreeCAD has no key and a plain URL; the other one has a key and a URL with a password in it
        remote = settings(wire("P", "p", has_key=True, local=True))
        local = self.local(wire("P", "p", has_key=False, base_url="https://proxy.example/v1"))
        first = settings_sync.merge(None, local, remote)
        self.assertEqual((first.entries[0].wire["has_key"], first.entries[0].wire["base_url_local"]), (True, True))
        self.assertIsNone(first.entries[0].url, '"" + local never replaces our URL')
        device = {"p": {"has_key": False, "url": ["https://proxy.example/v1", False]}}
        again = settings_sync.merge(remote, local, remote, device=device)
        self.assertEqual(again.settings()["profiles"], remote["profiles"], "nothing to push")
        # a key added here goes out; a real URL set on the website replaces ours
        local_key = self.local(wire("P", "p", has_key=True, base_url="https://proxy.example/v1"))
        web = settings(wire("P", "p", has_key=True, base_url="https://new.example/v1"))
        merged = settings_sync.merge(remote, local_key, web, device=device).entries[0]
        self.assertEqual((merged.wire["has_key"], merged.url), (True, "https://new.example/v1"))


# -- a sync's plan, on the settings file's state ----------------------------------------------------------------------


TAG = settings_sync.account_tag("http://api.test", "ada@example.com")


def plan(state, remote, prefer="remote", account=TAG):
    write, result = settings_sync.plan_sync(state, settings_sync.normalise_remote(remote), prefer=prefer,
                                            account=account, limits=MemoryStore.limits)
    return result


def confirmed(state, result, version=None):
    settings_sync.confirm_push(state, result, version if version is not None else result.version + 1)
    return {"version": version if version is not None else result.version + 1, **result.payload}


class PlanTest(unittest.TestCase):
    def test_a_first_sync_pushes_everything_and_keeps_the_keys(self):
        state = state_of(account_profile(), stored("Claude", "p1", api_key=KEY), active="p1")
        result = plan(state, settings())
        self.assertTrue(result.push)
        self.assertFalse(result.changed)
        self.assertEqual([p["id"] for p in result.payload["profiles"]], ["acct", "p1"])
        self.assertEqual(result.payload["active_profile_id"], "p1")
        self.assertNotIn(KEY, json.dumps(result.payload))
        self.assertEqual(state["profiles"][1]["api_key"], KEY)

    def test_a_first_sync_with_an_account_adopts_its_ids_by_name_and_for_the_account_profile(self):
        state = state_of(account_profile("mine-acct", "My account"), stored("Claude", "mine-c", api_key=KEY),
                         stored("Only here", "mine-x"), active="mine-c")
        remote = settings(wire("OriginShape", "srv-acct", provider="originshape", base_url="", model="origin1-super"),
                          wire("claude", "srv-c", model="claude-opus-4-1", has_key=True),
                          wire("Web only", "srv-w"), active="srv-w", version=7)
        result = plan(state, remote)
        by_id = {p["id"]: p for p in state["profiles"]}
        self.assertEqual(result.adopted, 2)
        self.assertEqual(set(by_id), {"srv-acct", "srv-c", "mine-x", "srv-w"})
        self.assertEqual(by_id["srv-c"]["api_key"], KEY, "the key stays with the profile it was saved for")
        self.assertEqual(by_id["srv-c"]["model"], "claude-opus-4-1", "no snapshot: the server's value on a pull")
        self.assertEqual(by_id["srv-acct"]["model"], "origin1-super")
        self.assertEqual(by_id["srv-w"]["api_key"], "")
        self.assertEqual(state["active_profile_id"], "srv-w")
        self.assertTrue(result.changed)
        self.assertTrue(result.push)   # "Only here" goes to the server

    def test_a_pull_replaces_no_key_model_caps_preference_or_local_url(self):
        state = state_of(stored("Proxy", "p", provider="openai_compat", base_url="https://u:pw@proxy.example/v1",
                                api_key=KEY), stored("Claude", "c", api_key=OPENAI_KEY))
        before = copy.deepcopy(state)
        result = plan(state, settings())
        confirmed(state, result, 1)
        web = settings(wire("Proxy", "p", provider="openai_compat", local=True, has_key=True, model="llama"),
                       wire("Claude", "c", has_key=False, model="claude-opus-4-1"), active="c", version=2)
        result = plan(state, web)
        p, c = state["profiles"]
        self.assertEqual((p["base_url"], p["api_key"], p["model"]), ("https://u:pw@proxy.example/v1", KEY, "llama"))
        self.assertEqual((c["api_key"], c["model"]), (OPENAI_KEY, "claude-opus-4-1"))
        for key in ("model_caps", "prefs", "net_grants"):
            self.assertEqual(state[key], before[key])
        self.assertEqual(state["active_profile_id"], "c")
        self.assertTrue(result.push, "has_key: our keys are still here, the server is told again")

    def test_a_profile_deleted_on_the_website_goes_here_with_its_key(self):
        state = state_of(stored("Keep", "k"), stored("Gone", "g", api_key=KEY))
        confirmed(state, plan(state, settings()), 1)
        current = state[settings_sync.STATE_KEY]["remote"]
        web = {"version": 2, "active_profile_id": "k", "profiles": [current["profiles"][0]]}
        result = plan(state, web)
        self.assertEqual([p["id"] for p in state["profiles"]], ["k"])
        self.assertNotIn(KEY, json.dumps(state))
        self.assertTrue(result.changed)
        self.assertFalse(result.push)

    def test_a_server_that_forgot_its_settings_or_another_account_deletes_nothing_here(self):
        state = state_of(stored("A", "a"), stored("B", "b"))
        confirmed(state, plan(state, settings()), 5)
        for remote, account in ((settings(version=0), TAG), (settings(version=9), "another-account")):
            with self.subTest(account=account, version=remote["version"]):
                copy_state = copy.deepcopy(state)
                result = plan(copy_state, remote, account=account)
                self.assertEqual([p["id"] for p in copy_state["profiles"]], ["a", "b"])
                self.assertEqual([p["id"] for p in result.payload["profiles"]], ["a", "b"])

    def test_nothing_to_push_when_both_sides_agree(self):
        state = state_of(stored("A", "a", api_key=KEY))
        sent = confirmed(state, plan(state, settings()), 1)
        result = plan(state, sent)
        self.assertFalse(result.push)
        self.assertFalse(result.changed)

    def test_a_key_is_set_aside_when_the_account_moves_its_profile_to_another_host(self):
        state = state_of(stored("Claude", "c", api_key=KEY))
        sent = confirmed(state, plan(state, settings()), 1)
        evil = copy.deepcopy(sent)
        evil["version"] = 2
        evil["profiles"][0].update(provider="openai_compat", base_url="https://collect.example.net/v1")
        result = plan(state, evil)
        profile = state["profiles"][0]
        self.assertEqual((profile["provider"], profile["base_url"], profile["api_key"]),
                         ("openai_compat", "https://collect.example.net/v1", ""))
        self.assertEqual(profile["held_key"], {"key": KEY, "provider": "anthropic",
                                               "base_url": "https://api.anthropic.com"})
        self.assertFalse(result.payload["profiles"][0]["has_key"])
        confirmed(state, result, 3)
        back = copy.deepcopy(evil)
        back["version"] = 4
        back["profiles"][0].update(provider="anthropic", base_url="https://api.anthropic.com/", model="m2")
        plan(state, back)
        profile = state["profiles"][0]
        self.assertEqual((profile["api_key"], profile["held_key"], profile["model"]), (KEY, {}, "m2"))

    def test_a_model_change_on_the_same_host_keeps_the_key(self):
        state = state_of(stored("Claude", "c", api_key=KEY))
        sent = confirmed(state, plan(state, settings()), 1)
        sent["version"] = 2
        sent["profiles"][0].update(model="claude-opus-4-1", base_url="https://api.anthropic.com/v1")
        plan(state, sent)
        self.assertEqual((state["profiles"][0]["api_key"], state["profiles"][0]["model"]), (KEY, "claude-opus-4-1"))


class MergeSavedTest(unittest.TestCase):
    """The settings dialog's Save while a pull changed the file under it."""

    def test_the_dialogs_changes_and_the_pulls_are_both_kept(self):
        base = {"profiles": [stored("A", "a", model="m1"), stored("B", "b"), stored("C", "c")],
                "active_profile_id": "a"}
        ours = copy.deepcopy(base)
        ours["profiles"][0]["name"] = "A renamed"
        ours["profiles"][0]["max_tokens"] = 1          # recomputed, not an edit
        ours["profiles"].append(stored("Mine", "m"))
        del ours["profiles"][2]                          # removed in the dialog
        theirs = copy.deepcopy(base)
        theirs["profiles"][0]["model"] = "m2"            # a pull
        theirs["profiles"][0]["caps_override"] = {"max_output": 2048}
        del theirs["profiles"][1]                        # deleted on the website, untouched in the dialog
        theirs["profiles"].append(stored("Web", "w"))
        theirs["active_profile_id"] = "w"
        merged = settings_sync.merge_saved(base, ours, theirs)
        by_id = {p["id"]: p for p in merged["profiles"]}
        self.assertEqual(set(by_id), {"a", "m", "w"})
        self.assertEqual((by_id["a"]["name"], by_id["a"]["model"], by_id["a"]["caps_override"]),
                         ("A renamed", "m2", {"max_output": 2048}))
        self.assertEqual(merged["active_profile_id"], "w")


# -- the Syncer against the fake API ----------------------------------------------------------------------------------


class _FakeCase(unittest.TestCase):
    def setUp(self):
        for reset in (discovery.clear_cache, session.reset_sessions, keystore.reset_memory, account.clear_model_cache):
            reset()
            self.addCleanup(reset)
        patcher = mock.patch.dict(os.environ, {keystore.ENV_STORE: "memory", settings_sync.ENV: "1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.fake = FakeWorkspace()
        self.addCleanup(self.fake.close)
        self.session = session.get_session(self.fake.url)
        self.events: list[tuple[str, dict]] = []

    def sign_in(self):
        data = self.fake.issue_tokens(device_name="Test Box")
        self.session.complete_sign_in(TokenSet(data["access_token"], data["refresh_token"], data["expires_in"],
                                               data["scope"], received_at=time.monotonic()))

    def syncer(self, store):
        return settings_sync.Syncer(session_for=lambda: self.session, store=store,
                                    notify=lambda event, data: self.events.append((event, data)))

    def puts(self):
        return [json.loads(raw) for raw in self.fake.ai_settings_puts]


class SyncerTest(_FakeCase):
    def test_signed_out_nothing_is_sent(self):
        store = MemoryStore(state_of(stored("A", "a", api_key=KEY)))
        status = self.syncer(store).sync()
        self.assertEqual(status["state"], "signed_out")
        self.assertEqual(self.fake.calls_to(settings_sync.ROUTE), [])
        self.assertEqual(store.writes, 0)

    def test_a_personal_api_key_does_not_sync(self):
        self.session.use_api_key(self.fake.add_api_key())
        status = self.syncer(MemoryStore(state_of(stored("A", "a")))).sync()
        self.assertEqual(status["state"], "api_key")
        self.assertEqual(self.fake.calls_to(settings_sync.ROUTE), [])

    def test_a_first_sync_pushes_the_profiles_without_keys(self):
        self.sign_in()
        store = MemoryStore(state_of(account_profile(), stored("Claude", "c", api_key=KEY, notes=KEY),
                                     stored("Proxy", "p", provider="openai_compat",
                                            base_url="https://me:secret-pw@proxy.example/v1?k=v", api_key=OPENAI_KEY),
                                     active="c"))
        status = self.syncer(store).sync()
        self.assertEqual(status["state"], "synced", status)
        self.assertEqual(status["version"], 1)
        self.assertEqual(self.fake.ai_settings_refusals, [])
        raw = b"".join(self.fake.ai_settings_puts).decode("utf-8")
        for secret in (KEY, OPENAI_KEY, "secret-pw"):
            self.assertNotIn(secret, raw)
            for piece in substrings(secret):
                self.assertNotIn(piece, raw)
        stored_there = self.fake.ai_settings
        self.assertEqual(stored_there["active_profile_id"], "c")
        self.assertEqual([(p["id"], p["has_key"], p["base_url_local"]) for p in stored_there["profiles"]],
                         [("acct", False, False), ("c", True, False), ("p", True, True)])
        self.assertEqual(store.state[settings_sync.STATE_KEY]["version"], 1)
        self.assertEqual(store.by_id("c")["api_key"], KEY)

    def test_a_website_edit_is_pulled_and_nothing_is_pushed_back(self):
        self.sign_in()
        store = MemoryStore(state_of(stored("Claude", "c", api_key=KEY), stored("Local", "l", provider="openai_compat",
                                                                                base_url="http://localhost:11434/v1")))
        syncer = self.syncer(store)
        syncer.sync()

        def edit(settings_now):
            settings_now["profiles"][0]["model"] = "claude-opus-4-1"
            settings_now["profiles"][1]["name"] = "Ollama"
            settings_now["active_profile_id"] = "l"
        self.fake.web_edit(edit)
        self.events.clear()
        status = syncer.sync()
        self.assertEqual(status, {**status, "state": "synced", "version": 2})
        self.assertEqual(store.by_id("c")["model"], "claude-opus-4-1")
        self.assertEqual(store.by_id("c")["api_key"], KEY)
        self.assertEqual(store.names(), ["Claude", "Ollama"])
        self.assertEqual(store.active(), "l")
        self.assertEqual(len(self.fake.ai_settings_puts), 1, "nothing to push back")
        self.assertIn("changed", [event for event, _data in self.events])
        self.assertEqual(syncer.sync()["state"], "synced")
        self.assertEqual(len(self.fake.ai_settings_puts), 1)

    def test_a_409_is_merged_with_the_current_settings_and_pushed_once_more(self):
        self.sign_in()
        store = MemoryStore(state_of(stored("A", "a"), stored("B", "b")))
        syncer = self.syncer(store)
        syncer.sync()
        for n, (beside, without) in enumerate(((False, False), (True, False), (False, True))):
            with self.subTest(beside=beside, without=without):
                name, model = f"A here {n}", f"from the web {n}"
                store.state["profiles"][0]["name"] = name      # a local change ...

                def web(fake, model=model):                    # ... and a website write between our GET and PUT
                    fake.web_edit(lambda s: s["profiles"][1].update(model=model))
                self.fake.conflict_details_beside, self.fake.conflict_without_details = beside, without
                self.fake.before_ai_put = web
                puts = len(self.fake.ai_settings_puts)
                status = syncer.sync(prefer="local")
                self.assertEqual(status["state"], "synced", status)
                self.assertEqual(len(self.fake.ai_settings_puts) - puts, 2)
                there = {p["id"]: p for p in self.fake.ai_settings["profiles"]}
                self.assertEqual((there["a"]["name"], there["b"]["model"]), (name, model))
                self.assertEqual(store.by_id("b")["model"], model)

    def test_a_second_409_waits_for_the_next_trigger(self):
        self.sign_in()
        store = MemoryStore(state_of(stored("A", "a")))
        syncer = self.syncer(store)
        syncer.sync()
        store.state["profiles"][0]["name"] = "changed here"

        def web_twice(fake):
            fake.web_edit(lambda s: s["profiles"][0].update(model="w1"))
            fake.before_ai_put = lambda f: f.web_edit(lambda s: s["profiles"][0].update(notes="w2"))
        self.fake.before_ai_put = web_twice
        puts = len(self.fake.ai_settings_puts)
        status = syncer.sync()
        self.assertEqual(status["state"], "error")
        self.assertIn("same moment", status["message"])
        self.assertEqual(len(self.fake.ai_settings_puts) - puts, 2)
        self.assertEqual(syncer.sync()["state"], "synced")     # the next trigger
        self.assertEqual(self.fake.ai_settings["profiles"][0]["name"], "changed here")

    def test_failures_are_reported_and_logged_without_settings_values(self):
        self.sign_in()
        logged = []
        syncer = settings_sync.Syncer(session_for=lambda: self.session, store=MemoryStore(state_of(stored(KEY, "a"))),
                                      notify=lambda *a: None, log=lambda kind, **fields: logged.append((kind, fields)))
        self.fake.fail(settings_sync.ROUTE, 404, method="GET")
        self.assertEqual(syncer.sync()["state"], "error")
        self.fake.fail(settings_sync.ROUTE, 503, method="GET")
        self.assertEqual(syncer.sync()["state"], "unavailable")
        self.fake.fail(settings_sync.ROUTE, 400, {"type": "error", "error": {"code": "invalid_request",
                                                                               "message": "names must be unique"}},
                       method="PUT")
        status = syncer.sync()
        self.assertEqual(status["state"], "error")
        self.assertIn("names must be unique", status["message"])
        self.assertNotIn(KEY, json.dumps(logged))
        self.assertTrue(all(kind == "ai_settings_sync" for kind, _ in logged))
        self.assertEqual(syncer.sync()["state"], "synced")

    def test_an_unreadable_settings_file_changes_nothing(self):
        self.sign_in()

        class Locked(MemoryStore):
            def transaction(self, mutate):
                raise OSError(13, "locked")
        status = self.syncer(Locked(state_of(stored("A", "a")))).sync()
        self.assertEqual(status["state"], "error")
        self.assertEqual(self.fake.ai_settings_puts, [])


class ThreadTest(_FakeCase):
    def test_request_runs_in_the_background_and_pull_if_stale_waits_a_minute(self):
        self.sign_in()
        store = MemoryStore(state_of(stored("A", "a")))
        syncer = self.syncer(store)
        settings_sync.reset(syncer)
        self.addCleanup(settings_sync.reset)
        settings_sync.request("test")
        settings_sync.request("again", prefer="local")       # coalesced: runs once more after
        self.assertTrue(settings_sync.wait_idle())
        self.assertEqual(settings_sync.status()["state"], "synced")
        self.assertFalse(settings_sync.pull_if_stale())
        with mock.patch.object(settings_sync, "request") as started:
            syncer.last_pull -= settings_sync.PULL_AGE_S + 1
            self.assertTrue(settings_sync.pull_if_stale())
            started.assert_called_once_with("chat")

    def test_switched_off_nothing_starts(self):
        self.sign_in()
        syncer = self.syncer(MemoryStore(state_of(stored("A", "a"))))
        settings_sync.reset(syncer)
        self.addCleanup(settings_sync.reset)
        with mock.patch.dict(os.environ, {settings_sync.ENV: "0"}):
            settings_sync.request("test")
            self.assertFalse(settings_sync.pull_if_stale())
        self.assertTrue(settings_sync.wait_idle())
        self.assertEqual(self.fake.calls_to(settings_sync.ROUTE), [])


# -- the website's Test in FreeCAD ------------------------------------------------------------------------------------


class Line(SimpleNamespace):
    pass


class Report:
    """A ``provider_check.ConnectionReport`` stand-in."""

    def __init__(self, ok, lines, tools="", vision=None):
        self.ok, self.lines, self.tools, self.vision = ok, lines, tools, vision

    def summary(self):
        marks = {True: "✓", False: "✗", None: "-"}
        return "\n".join(f"{marks[line.ok]} {line.text}" for line in self.lines)


def profile_store():
    return MemoryStore(state_of(stored("Claude", "c", api_key=KEY),
                                stored("Proxy", "p", provider="openai_compat",
                                       base_url="https://bob:hunter2pw@proxy.example/v1?token=q9q9q9", api_key=KEY)))


class ProfileTestTest(unittest.TestCase):
    def store(self):
        return profile_store()

    def test_the_message_is_scrubbed_of_the_key_and_clipped(self):
        masked = KEY[:7] + "*" * 20 + KEY[-4:]
        text = (f"Anthropic / claude: error 401 invalid x-api-key {KEY}; Incorrect API key provided: {masked}; "
                f"the key starts {KEY[:16]} and ends {KEY[-10:]}; ") * 3
        report = Report(False, [Line(step="connect", ok=False, text=text)])
        passed, message, checks = settings_sync.run_profile_test("c", store=self.store(),
                                                                 run=lambda profile, should_stop: report)
        self.assertFalse(passed)
        self.assertLessEqual(len(message), 500)
        for piece in substrings(KEY, 6):
            self.assertNotIn(piece, message)
        self.assertNotIn("*****", message)
        self.assertEqual(checks, {"reply": False, "tools": None, "image": None})

    def test_a_local_base_urls_secrets_are_scrubbed(self):
        report = Report(False, [Line(step="connect", ok=False,
                                     text="could not reach https://bob:hunter2pw@proxy.example/v1?token=q9q9q9 "
                                          "(user bob, password hunter2pw, token=q9q9q9)")])
        _passed, message, _checks = settings_sync.run_profile_test("p", store=self.store(),
                                                                   run=lambda profile, should_stop: report)
        for secret in ("hunter2pw", "q9q9q9", "bob:"):
            self.assertNotIn(secret, message)

    def test_a_passing_test_says_which_checks_passed(self):
        lines = [Line(step="connect", ok=True, text="Anthropic / claude: “ready”"),
                 Line(step="tools", ok=True, text="tool calling works natively"),
                 Line(step="vision", ok=None, text="images accepted but not confirmed")]
        seen = []

        def run(profile, should_stop):
            seen.append((profile.id, profile.api_key, should_stop()))
            return Report(True, lines, tools="native", vision=None)
        passed, message, checks = settings_sync.run_profile_test("c", store=self.store(), run=run)
        self.assertTrue(passed)
        self.assertEqual(seen, [("c", KEY, False)], "the local profile, with its local key")
        self.assertEqual(checks, {"reply": True, "tools": True, "image": None})
        self.assertIn("tool calling works natively", message)

    def test_an_unknown_profile_a_failing_test_and_a_test_that_takes_too_long(self):
        passed, message, checks = settings_sync.run_profile_test("nobody", store=self.store(),
                                                                 run=lambda *a, **k: self.fail("not run"))
        self.assertEqual((passed, message, checks),
                         (False, settings_sync.UNKNOWN_PROFILE_TEXT, {"reply": None, "tools": None, "image": None}))

        def boom(profile, should_stop):
            raise RuntimeError(f"provider bug quoting {KEY}")
        passed, message, _ = settings_sync.run_profile_test("c", store=self.store(), run=boom)
        self.assertEqual((passed, message), (False, "FreeCAD couldn't run the test (RuntimeError)."))

        class StopRequested(Exception):
            pass

        def slow(profile, should_stop):
            self.assertTrue(should_stop())
            raise StopRequested()
        passed, message, _ = settings_sync.run_profile_test("c", store=self.store(), run=slow, timeout=-1)
        self.assertEqual((passed, message), (False, "The test took too long in FreeCAD and was stopped."))


class ConnectProfileTestTest(_FakeCase):
    def setUp(self):
        super().setUp()
        for name, value in {"BACKOFF_S": (0.01, 0.01, 0.01), "SIGNED_OUT_RETRY_S": 0.02, "KEEPALIVE_S": 0.05}.items():
            fast = mock.patch.object(connect, name, value)
            fast.start()
            self.addCleanup(fast.stop)
        self.sign_in()
        self.files_dir = tempfile.mkdtemp(prefix="oscar-sync-connect-")
        self.addCleanup(shutil.rmtree, self.files_dir, True)

    def loop(self, **extra):
        loop = connect.Loop(lambda: self.session, start_turn=lambda job, files: self.fail("no turn"),
                            is_idle=lambda: True, on_state=lambda *a: None, versions=connect.versions("1.1.3"),
                            files_dir=lambda: self.files_dir, **extra)
        self.addCleanup(loop.stop, 2.0)
        return loop

    def wait_until(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not predicate():
            time.sleep(0.01)
        return predicate()

    def test_only_an_oscar_that_can_test_profiles_says_so(self):
        self.loop().start()
        self.assertTrue(self.wait_until(lambda: self.fake.hellos))
        self.assertNotIn("ai_settings", self.fake.hellos[-1].get("capabilities") or [])
        self.fake.hellos.clear()
        self.loop(test_profile=lambda test: (True, "", {})).start()
        self.assertTrue(self.wait_until(lambda: self.fake.hellos))
        self.assertIn("ai_settings", self.fake.hellos[-1]["capabilities"])

    def test_a_profile_test_runs_here_and_only_its_outcome_goes_back(self):
        asked = []

        def test_profile(test):
            asked.append(test)
            report = Report(True, [Line(step="connect", ok=True, text=f"ready ({KEY})"),
                                   Line(step="tools", ok=True, text="tool calling works natively")],
                            tools="native", vision=True)
            return settings_sync.run_profile_test(test.profile_id, store=profile_store(),
                                                  run=lambda profile, should_stop: report)
        known = self.fake.add_profile_test("c")
        unknown = self.fake.add_profile_test("nobody")
        self.loop(test_profile=test_profile).start()
        self.assertTrue(self.wait_until(lambda: len(self.fake.profile_test_results) == 2))
        self.assertEqual([(t.id, t.profile_id) for t in asked], [(known, "c"), (unknown, "nobody")])
        good, bad = self.fake.profile_test_results[known], self.fake.profile_test_results[unknown]
        self.assertEqual((good["passed"], good["checks"]), (True, {"reply": True, "tools": True, "image": True}))
        self.assertNotIn(KEY, json.dumps(good))
        self.assertEqual((bad["passed"], bad["message"]), (False, settings_sync.UNKNOWN_PROFILE_TEXT))
        self.assertEqual(bad["checks"], {"reply": None, "tools": None, "image": None})

    def test_a_handler_that_fails_still_answers(self):
        def broken(test):
            raise ValueError(f"oops {KEY}")
        test_id = self.fake.add_profile_test("c")
        self.loop(test_profile=broken).start()
        self.assertTrue(self.wait_until(lambda: test_id in self.fake.profile_test_results))
        body = self.fake.profile_test_results[test_id]
        self.assertEqual((body["passed"], body["message"]), (False, "FreeCAD couldn’t run the test (ValueError)."))

    def test_a_profile_test_job_is_parsed(self):
        test = connect.ProfileTest.parse({"id": "0b7c0a3e-1111-4222-8333-444455556666", "profile_id": "p-1"})
        self.assertEqual(test.profile_id, "p-1")
        for bad in ({"id": "../x", "profile_id": "p"}, {"id": "t"}, {"id": "t", "profile_id": ""}, "nope"):
            with self.subTest(bad=bad), self.assertRaises(connect.ConnectError):
                connect.ProfileTest.parse(bad)


if __name__ == "__main__":
    unittest.main()
