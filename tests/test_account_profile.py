"""The OriginShape account in Oscar's settings (the data side): the first preset, the account profile,
the new-install seed, ``ensure_account_profile`` for users who sign in, and the predefined limits of the aliases.

``oscar.params`` imports FreeCAD, so this module runs under freecadcmd; every test has its own settings file.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import params  # noqa: E402
from oscar.params import Profile  # noqa: E402

WORK = {"name": "Work", "provider": "anthropic", "base_url": "https://api.anthropic.com", "model": "claude-sonnet-4-5",
        "api_key": "sk-REAL"}
LOCAL = {"name": "Local", "provider": "openai_compat", "base_url": "http://localhost:11434/v1", "model": "qwen3:8b"}


class _Settings(unittest.TestCase):
    """Each test gets its own settings file (the env var is read per call) and no legacy migration."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="oscar_account_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.settings = os.path.join(self.tmp, "settings.json")
        patcher = mock.patch.dict(os.environ, {"OSCAR_AI_SETTINGS": self.settings})
        patcher.start()
        self.addCleanup(patcher.stop)
        migrated = params._MIGRATED
        params._MIGRATED = True
        self.addCleanup(setattr, params, "_MIGRATED", migrated)

    def put(self, profiles, active=""):
        raw = json.dumps({"profiles": profiles, "active_profile": active, "prefs": {"DockArea": 1}}).encode()
        with open(self.settings, "wb") as f:
            f.write(raw)
        return raw

    def raw(self) -> bytes:
        with open(self.settings, "rb") as f:
            return f.read()

    def stored(self) -> dict:
        return json.loads(self.raw())


class PresetTest(unittest.TestCase):
    def test_the_account_preset_comes_first(self):
        preset = params.PROVIDER_PRESETS[0]
        self.assertIs(preset, params.ORIGINSHAPE_PRESET)
        self.assertEqual({k: preset[k] for k in ("provider", "base_url", "default_model", "models", "needs_key",
                                                 "account", "max_tokens", "temperature")},
                         {"provider": "originshape", "base_url": "", "default_model": "origin1",
                          "models": ["origin1", "origin1-super"], "needs_key": False, "account": True,
                          "max_tokens": 32000, "temperature": 1.0})
        self.assertEqual([p["label"] for p in params.PROVIDER_PRESETS if p.get("account")], ["OriginShape (your account)"])
        self.assertEqual((params.ACCOUNT_PROVIDER, params.ACCOUNT_PROFILE_NAME), ("originshape", "OriginShape"))
        self.assertEqual(params.COMING_SOON_PROVIDERS, [])

    def test_presets_are_found_by_kind(self):
        self.assertIs(params.preset_for("originshape"), params.ORIGINSHAPE_PRESET)
        self.assertIs(params.preset_for("originshape", "http://hand.edited/v1"), params.ORIGINSHAPE_PRESET)
        self.assertEqual(params.preset_for("anthropic")["label"], "Anthropic (Claude)")
        self.assertEqual(params.preset_for("no_such_kind")["label"], "Custom (OpenAI-compatible)")

    def test_the_aliases_have_predefined_limits(self):
        self.assertEqual(params.MODEL_LIMITS[:2], [("origin1", 32000), ("oscar-", 32000)])
        for model in params.ACCOUNT_MODELS:
            with self.subTest(model=model):
                self.assertEqual(params.predefined_limits("originshape", "", model), (1.0, 32000))
        self.assertEqual(params.effective_limits(params.account_profile()), (1.0, 32000))
        override = params.account_profile()
        override.caps_override = {"max_output": 8000}
        self.assertEqual(params.effective_limits(override), (1.0, 8000))

    def test_the_account_profile(self):
        profile = params.account_profile()
        self.assertEqual((profile.name, profile.provider, profile.base_url, profile.model, profile.api_key),
                         ("OriginShape", "originshape", "", "origin1", ""))
        self.assertEqual(params.account_profile("Mine").name, "Mine")
        self.assertTrue(params.is_account_profile(profile))
        self.assertFalse(params.is_account_profile(Profile(name="x")))
        self.assertFalse(params.is_account_profile(None))

    def test_the_model_lineup(self):
        self.assertEqual(params.ACCOUNT_MODEL_CHOICES, ("origin1", "origin1-super"))
        self.assertEqual(params.ACCOUNT_MODELS, ("origin1", "origin1-super", "oscar-default", "oscar-fast"))
        self.assertEqual(params.LEGACY_ACCOUNT_MODELS, {"oscar-default": "origin1-super", "oscar-fast": "origin1"})
        for alias, current in (("origin1", "origin1"), ("origin1-super", "origin1-super"), (" oscar-fast ", "origin1"),
                               ("oscar-default", "origin1-super"), ("origin2", "origin2"), ("", ""), (None, "")):
            with self.subTest(alias=alias):
                self.assertEqual(params.account_model_alias(alias), current)

    def test_the_model_names_the_user_sees(self):
        def account(model):
            return Profile(name="a", provider="originshape", base_url="", model=model)
        for model, name in (("origin1", "Origin1"), ("origin1-super", "Origin1 Super"), ("oscar-fast", "Origin1"),
                            ("oscar-default", "Origin1 Super"), ("origin2", "origin2")):
            with self.subTest(model=model):
                self.assertEqual(params.model_display_name(account(model)), name)
        self.assertEqual(params.model_display_name(Profile(name="w", model="claude-sonnet-4-5")), "claude-sonnet-4-5")
        self.assertEqual(params.model_display_name(None), "")
        # every account model is shown by its OriginShape name and nothing else
        self.assertEqual({params.model_display_name(account(model)) for model in params.ACCOUNT_MODELS},
                         {"Origin1", "Origin1 Super"})

    def test_the_profile_defaults_are_unchanged(self):
        p = Profile()
        self.assertEqual((p.provider, p.base_url, p.model, p.temperature, p.max_tokens),
                         ("anthropic", "https://api.anthropic.com", "claude-sonnet-4-5", 0.2, 4096))


class SeedTest(_Settings):
    def test_a_new_install_starts_on_the_account(self):
        profiles = params.load_profiles()
        self.assertEqual([(p.name, p.provider) for p in profiles], [("OriginShape", "originshape")])
        self.assertEqual(params.active_profile_name(), "OriginShape")
        self.assertTrue(params.is_account_profile(params.active_profile()))
        seeded = self.stored()["profiles"][0]
        self.assertEqual((seeded["api_key"], seeded["base_url"], seeded["model"]), ("", "", "origin1"))

    def test_a_file_without_profiles_is_seeded_but_keeps_its_prefs(self):
        self.put([])
        params.load_profiles()
        self.assertEqual([p["provider"] for p in self.stored()["profiles"]], ["originshape"])
        self.assertEqual(self.stored()["prefs"], {"DockArea": 1})

    def test_existing_users_keep_their_profiles_untouched(self):
        raw = self.put([WORK, LOCAL], active="Local")
        self.assertEqual([p.name for p in params.load_profiles()], ["Work", "Local"])
        self.assertEqual(params.active_profile_name(), "Local")
        self.assertEqual(self.raw(), raw)                         # no upgrade write

    def test_a_retired_alias_is_read_as_its_tier_and_saved_on_the_next_save(self):
        old_default = {"name": "OriginShape", "provider": "originshape", "model": "oscar-default", "base_url": ""}
        old_fast = {"name": "Fast", "provider": "originshape", "model": "oscar-fast", "base_url": ""}
        own = dict(WORK, model="oscar-default")      # the user's own provider: its model id is never touched
        raw = self.put([old_default, old_fast, own], active="OriginShape")
        loaded = params.load_profiles()
        self.assertEqual([p.model for p in loaded], ["origin1-super", "origin1", "oscar-default"])
        self.assertEqual(params.active_profile().model, "origin1-super")
        self.assertEqual(self.raw(), raw)                         # reading never writes
        params.save_profiles(loaded)
        self.assertEqual([p["model"] for p in self.stored()["profiles"]], ["origin1-super", "origin1", "oscar-default"])
        self.assertEqual(self.stored()["profiles"][2]["api_key"], "sk-REAL")
        self.assertTrue(os.path.exists(self.settings + ".bak"))    # the previous file is kept

    def test_an_account_profile_round_trips(self):
        params.save_profiles([params.account_profile(), Profile(**WORK)])
        loaded = params.load_profiles()
        self.assertEqual([(p.name, p.provider, p.base_url) for p in loaded],
                         [("OriginShape", "originshape", ""), ("Work", "anthropic", "https://api.anthropic.com")])


class EnsureAccountProfileTest(_Settings):
    def test_signing_in_adds_and_activates_it_and_keeps_the_others(self):
        self.put([WORK, LOCAL], active="Local")
        self.assertEqual(params.ensure_account_profile(activate=True), "OriginShape")
        stored = self.stored()
        self.assertEqual([p["name"] for p in stored["profiles"]], ["OriginShape", "Work", "Local"])
        self.assertEqual(stored["profiles"][1]["api_key"], "sk-REAL")
        self.assertEqual(stored["active_profile"], "OriginShape")
        self.assertEqual(stored["prefs"], {"DockArea": 1})
        self.assertTrue(os.path.exists(self.settings + ".bak"))    # the profiles changed: the previous file is kept

    def test_it_is_idempotent_and_writes_nothing_when_nothing_changes(self):
        self.put([WORK], active="Work")
        params.ensure_account_profile(activate=True)
        raw = self.raw()
        self.assertEqual(params.ensure_account_profile(activate=True), "OriginShape")
        self.assertEqual(params.ensure_account_profile(activate=False), "OriginShape")
        self.assertEqual(self.raw(), raw)
        self.assertEqual(sum(p["provider"] == "originshape" for p in self.stored()["profiles"]), 1)

    def test_an_existing_account_profile_is_used_whatever_its_name(self):
        self.put([WORK, {"name": "My account", "provider": "originshape", "model": "oscar-fast", "base_url": ""}],
                 active="Work")
        self.assertEqual(params.ensure_account_profile(activate=True), "My account")
        self.assertEqual([p["name"] for p in self.stored()["profiles"]], ["Work", "My account"])
        self.assertEqual(params.active_profile_name(), "My account")

    def test_a_taken_name_gets_a_number(self):
        self.put([dict(WORK, name="OriginShape"), dict(LOCAL, name="originshape 2")], active="OriginShape")
        self.assertEqual(params.ensure_account_profile(activate=True), "OriginShape 3")
        self.assertEqual(params.active_profile().provider, "originshape")

    def test_without_activate_the_profile_in_use_stays_in_use(self):
        for active in ("Local", "", "Gone"):     # "" and a stale name both mean "the first profile"
            with self.subTest(active=active):
                self.put([WORK, LOCAL], active=active)
                in_use = params.active_profile_name()
                self.assertEqual(params.ensure_account_profile(activate=False), "OriginShape")
                self.assertEqual(params.active_profile_name(), in_use)
                self.assertEqual(params.load_profiles()[0].name, "OriginShape")

    def test_an_empty_file_gets_it_as_the_only_profile(self):
        self.put([])
        self.assertEqual(params.ensure_account_profile(activate=False), "OriginShape")
        self.assertEqual(params.active_profile_name(), "OriginShape")


if __name__ == "__main__":
    unittest.main()
