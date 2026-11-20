"""The AI model settings sync (issue #86) on the real settings file and in the settings dialog.

Covered: profile ids in ``oscar_settings.json`` (a file from before ids reads with stable ids and is not written by a
read; the next write keeps them; a new profile gets its own; a rename keeps the active profile); backward
compatibility (an older Oscar reads a newer file, and when it saves without the ids and switches profiles by name, the
ids come back from the snapshot and its choice counts); the dialog's Save merged with what a sync wrote meanwhile; the
sync through ``params`` against the fake API (keys, ``model_caps``, preferences and network grants untouched, the
snapshot in the file, a website edit pulled); the Test in FreeCAD reading the profile with its key from the file; and
the dialog: the sync line, a pull reloading the tab (or offering Reload over unsaved changes, Save keeping both), a
key set aside offered back, Duplicate making a new id.
``oscar.params`` imports FreeCAD, so this module runs under freecadcmd; every test has its own settings file.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from dataclasses import asdict
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import params, settings_sync  # noqa: E402
from oscar.auth import account, discovery, keystore, session  # noqa: E402
from oscar.auth.transport import TokenSet  # noqa: E402
from oscar.params import Profile  # noqa: E402
from support.fake_workspace import FakeWorkspace  # noqa: E402

KEY = "sk-ant-api03-" + "Zq7xT2wPk9" * 5
WORK = {"name": "Work", "provider": "anthropic", "base_url": "https://api.anthropic.com", "model": "claude-sonnet-4-5",
        "api_key": KEY}
LOCAL = {"name": "Local", "provider": "openai_compat", "base_url": "http://localhost:11434/v1", "model": "qwen3:8b"}
OLD_FIELDS = ("name", "provider", "base_url", "model", "api_key", "temperature", "max_tokens", "notes", "reasoning",
              "caps_override")   # the Profile of Oscar 0.2.1


class _Settings(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="oscar_sync_file_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.settings = os.path.join(self.tmp, "settings.json")
        patcher = mock.patch.dict(os.environ, {"OSCAR_AI_SETTINGS": self.settings})
        patcher.start()
        self.addCleanup(patcher.stop)
        migrated = params._MIGRATED
        params._MIGRATED = True
        self.addCleanup(setattr, params, "_MIGRATED", migrated)

    def put(self, state: dict) -> bytes:
        raw = json.dumps(state).encode()
        with open(self.settings, "wb") as f:
            f.write(raw)
        return raw

    def raw(self) -> bytes:
        with open(self.settings, "rb") as f:
            return f.read()

    def stored(self) -> dict:
        return json.loads(self.raw())


class IdsTest(_Settings):
    def test_a_file_from_before_ids_reads_with_stable_ids_and_the_next_write_keeps_them(self):
        raw = self.put({"profiles": [WORK, LOCAL], "active_profile": "Local", "prefs": {"DockArea": 1},
                        "model_caps": {"k": {"learned": {"tools": "text"}}}})
        ids = [p.id for p in params.load_profiles()]
        self.assertEqual(ids, [p.id for p in params.load_profiles()])
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual(params.active_profile().id, ids[1])
        self.assertEqual(self.raw(), raw, "reading never writes")
        params.set_bool("PanelAutoOpen", True)
        stored = self.stored()
        self.assertEqual([p["id"] for p in stored["profiles"]], ids)
        self.assertEqual(stored["profiles"][0]["api_key"], KEY)
        self.assertEqual((stored["prefs"]["DockArea"], stored["model_caps"]), (1, {"k": {"learned": {"tools": "text"}}}))

    def test_new_profiles_get_their_own_id(self):
        a, b = Profile(name="A"), Profile(name="B")
        self.assertNotEqual(a.id, b.id)
        self.assertTrue(settings_sync.valid_id(a.id))
        self.assertTrue(settings_sync.valid_id(Profile(name="C", id="not valid!").id))
        self.assertEqual(Profile(name="D", approved="junk").approved, {}, "unreadable: not approved")
        made_here = Profile(name="E", provider="openai_compat", base_url="https://e.example/v1")
        self.assertEqual(made_here.approved, {"provider": "openai_compat", "base_url": "https://e.example/v1"})
        self.assertFalse(settings_sync.needs_confirmation(made_here), "a profile made in FreeCAD is approved")

    def test_a_rename_keeps_the_active_profile(self):
        work, local = Profile(**WORK), Profile(**LOCAL)
        params.save_profiles([work, local], active_id=local.id)
        self.assertEqual((params.active_profile_name(), params.active_profile_id()), ("Local", local.id))
        local.name = "Ollama"
        params.save_profiles([work, local])
        self.assertEqual((params.active_profile_name(), params.active_profile_id()), ("Ollama", local.id))
        params.set_active_profile("Work")
        stored = self.stored()
        self.assertEqual((stored["active_profile"], stored["active_profile_id"]), ("Work", work.id))
        self.assertEqual(params.profile_by_id(work.id).api_key, KEY)
        self.assertIsNone(params.profile_by_id("nobody"))


class BackwardCompatibilityTest(_Settings):
    def newer_file(self) -> dict:
        work, local = Profile(**WORK, id="id-work"), Profile(**LOCAL, id="id-local")
        params.save_profiles([work, local], active_id="id-local")
        params.update_state(lambda state: (True, state.update({settings_sync.STATE_KEY: {
            "account": "x", "version": 3, "device": {},
            "remote": {"active_profile_id": "id-local", "profiles": [settings_sync.to_wire(asdict(work)),
                                                                     settings_sync.to_wire(asdict(local))]}}})))
        return self.stored()

    def test_an_older_oscar_reads_a_newer_file(self):
        state = self.newer_file()
        # what Oscar 0.2.1 does with it: _well_formed, then Profile(**{only the fields it knows})
        self.assertTrue(params._well_formed(state))
        old = [{k: v for k, v in p.items() if k in OLD_FIELDS} for p in state["profiles"]]
        self.assertEqual([(p["name"], p["api_key"]) for p in old], [("Work", KEY), ("Local", "")])
        self.assertEqual(state["active_profile"], "Local", "the name an older Oscar reads is kept up to date")

    def test_what_an_older_oscar_saved_is_understood(self):
        state = self.newer_file()
        # Oscar 0.2.1 saves: its own profile dicts (no ids), a new profile, the active profile by name; it keeps the
        # top-level keys it does not know
        state["profiles"] = [{k: v for k, v in p.items() if k in OLD_FIELDS} for p in state["profiles"]]
        state["profiles"].append({**{k: v for k, v in asdict(Profile(name="Third")).items() if k in OLD_FIELDS}})
        state["active_profile"] = "Work"
        self.put(state)
        loaded = params.load_profiles()
        self.assertEqual([p.id for p in loaded[:2]], ["id-work", "id-local"], "found again by name")
        self.assertTrue(settings_sync.valid_id(loaded[2].id))
        self.assertEqual((params.active_profile_name(), params.active_profile_id()), ("Work", "id-work"))
        self.assertEqual(loaded[0].api_key, KEY)


class SaveMergeTest(_Settings):
    def test_the_dialogs_save_keeps_what_a_sync_wrote_meanwhile(self):
        work, local = Profile(**WORK), Profile(**LOCAL)
        params.save_profiles([work, local], active_id=work.id)
        profiles, from_file, base = params.load_profiles_for_edit()
        self.assertTrue(from_file)
        self.assertEqual(base["active_profile_id"], work.id)

        def pull(state):   # a sync, while the dialog is open: Local's model changes, a profile is added
            state["profiles"][1]["model"] = "llama3.3"
            state["profiles"].append(asdict(Profile(name="From the web", id="web-1")))
            return True, None
        params.update_state(pull)
        profiles[0].name = "Work (renamed)"
        params.save_profiles(profiles, active_id=profiles[0].id, base=base)
        stored = self.stored()
        self.assertEqual([(p["name"], p["model"]) for p in stored["profiles"]],
                         [("Work (renamed)", "claude-sonnet-4-5"), ("Local", "llama3.3"),
                          ("From the web", "claude-sonnet-4-5")])
        self.assertEqual(stored["profiles"][0]["api_key"], KEY)
        self.assertEqual(stored["active_profile_id"], work.id)


    def test_a_profile_deleted_by_a_sync_stays_deleted_when_the_dialog_did_not_touch_it(self):
        old_file = {"profiles": [dict(WORK), dict(LOCAL, model="oscar-default", provider="originshape", base_url="",
                                                  name="Account")], "active_profile": "Work"}
        self.put(old_file)                                   # a file from before ids (and before approvals)
        profiles, _from_file, base = params.load_profiles_for_edit()
        params.update_state(lambda state: (True, state["profiles"].pop(0)))   # a sync deleted Work
        params.save_profiles(profiles, active_id=profiles[1].id, base=base)
        stored = self.stored()
        self.assertEqual([p["name"] for p in stored["profiles"]], ["Account"])
        self.assertEqual(stored["profiles"][0]["model"], "origin1-super", "a retired alias is saved as its tier")
        self.assertEqual(stored["active_profile"], "Account")


class ConfirmationTest(_Settings):
    """A profile a sync made, or whose provider or base URL a sync changed, is not used until confirmed here."""

    def moved(self, **extra):
        return Profile(**{**WORK, "base_url": "https://collect.example.net/v1",
                          "approved": {"provider": "anthropic", "base_url": "https://api.anthropic.com"}, **extra})

    def made(self, **extra):
        return Profile(**{"name": "Web made", "provider": "openai_compat", "base_url": "https://chat.evil.example/v1",
                          "model": "m", "approved": {}, **extra})

    def test_no_provider_is_made_for_it(self):
        from oscar import providers
        from oscar.provider_transport import ProviderError

        for profile in (self.moved(), self.made(), self.moved(api_key="")):
            with self.subTest(profile.name, keyed=bool(profile.api_key)), self.assertRaises(ProviderError) as caught:
                providers.make_provider(profile)
            self.assertEqual(caught.exception.code, "endpoint_changed")
            self.assertNotIn(KEY, str(caught.exception))
        providers.make_provider(Profile(**WORK))   # a profile nobody changed elsewhere
        providers.make_provider(params.account_profile())

    def test_no_test_connection_no_model_list_no_probe(self):
        from oscar import provider_check, provider_probe

        def nothing(*args, **kwargs):
            raise AssertionError("no request for a profile waiting for confirmation")
        for profile in (self.moved(), self.made()):
            with self.subTest(profile.name):
                with mock.patch.object(provider_probe, "_PROBES", {"anthropic": nothing, "openai": nothing,
                                                                   "ollama": nothing, "openrouter": nothing}):
                    report = provider_check.run_connection_test(profile)
                    self.assertEqual(provider_probe.probe(profile), {})
                self.assertFalse(report.ok)
                self.assertEqual([(line.step, line.ok) for line in report.lines], [("connect", False)])
                self.assertIn("confirm", report.error)
                models, error = provider_probe.list_models(profile)
                self.assertEqual(models, [])
                self.assertIn("confirm", error)

    def test_the_panel_refuses_to_send(self):
        try:
            from oscar import panel
        except ImportError as e:
            self.skipTest(f"the panel needs FreeCAD's GUI modules: {e}")
        for profile in (self.moved(), self.made()):
            self.assertEqual(panel.OscarPanel._send_blocker(None, profile), "endpoint")

    def test_a_change_made_in_this_freecad_needs_no_confirmation(self):
        work = Profile(**WORK)
        params.save_profiles([work])
        work.base_url, work.provider = "https://proxy.example/v1", "openai_compat"
        work.approved = settings_sync.approval_for(work)   # what the dialog does for an edited address
        params.save_profiles([work])
        saved = params.profile_by_id(work.id)
        self.assertEqual(saved.api_key, KEY)
        self.assertFalse(settings_sync.needs_confirmation(saved))

    def test_a_pull_through_the_settings_file_makes_it_wait(self):
        work = Profile(**WORK)
        params.save_profiles([work])
        tag = settings_sync.account_tag("http://api.test", "a@example.com")
        local = settings_sync.local_settings(json.loads(self.raw()))
        local.pop("excluded")
        params.update_state(lambda state: settings_sync.plan_sync(
            state, settings_sync.normalise_remote({"version": 1, **local}), prefer="remote", account=tag))
        web = json.loads(json.dumps(local))
        web["profiles"][0]["base_url"] = "https://collect.example.net/v1"
        params.update_state(lambda state: settings_sync.plan_sync(
            state, settings_sync.normalise_remote({"version": 2, **web}), prefer="remote", account=tag,
            limits=settings_sync.ParamsStore().limits))
        saved = params.profile_by_id(work.id)
        self.assertEqual((saved.base_url, saved.api_key), ("https://collect.example.net/v1", KEY))
        self.assertTrue(settings_sync.needs_confirmation(saved))

    def test_the_websites_pick_is_shown_and_a_pick_here_ends_it(self):
        work, made = Profile(**WORK), self.made()
        params.save_profiles([work, made], active_id=work.id)
        params.update_state(lambda state: (True, state.update({settings_sync.PENDING_KEY: {"id": made.id,
                                                                                           "why": "confirm"}})))
        self.assertEqual(params.pending_notice(), "The website picked Web made (chat.evil.example). Confirm it in "
                                                  "Oscar's settings to use it.")
        self.assertEqual(params.active_profile().id, work.id)
        params.set_active_profile("Work")
        self.assertEqual(params.pending_notice(), "")


class _FakeSession(_Settings):
    def setUp(self):
        super().setUp()
        for reset in (discovery.clear_cache, session.reset_sessions, keystore.reset_memory, account.clear_model_cache):
            reset()
            self.addCleanup(reset)
        patcher = mock.patch.dict(os.environ, {keystore.ENV_STORE: "memory", settings_sync.ENV: "1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.fake = FakeWorkspace()
        self.addCleanup(self.fake.close)
        self.session = session.get_session(self.fake.url)
        data = self.fake.issue_tokens()
        self.session.complete_sign_in(TokenSet(data["access_token"], data["refresh_token"], data["expires_in"],
                                               data["scope"], received_at=time.monotonic()))


class ParamsSyncTest(_FakeSession):
    def test_a_sync_through_the_settings_file(self):
        params.save_profiles([params.account_profile(), Profile(**WORK), Profile(**LOCAL)])
        params.update_model_caps("k|u|m", lambda entry: {"learned": {"vision": True}})
        params.add_net_grant("fetch_url", "a.example.com")
        params.set_bool("ConnectWorkspace", True)
        before = self.stored()
        syncer = settings_sync.Syncer(session_for=lambda: self.session, notify=lambda *a: None)
        self.assertEqual(syncer.sync()["state"], "synced")
        self.assertNotIn(KEY, b"".join(self.fake.ai_settings_puts).decode())
        stored = self.stored()
        self.assertEqual(stored[settings_sync.STATE_KEY]["version"], 1)
        self.assertEqual(stored["profiles"], before["profiles"])
        self.fake.web_edit(lambda s: (s["profiles"][1].update(model="claude-opus-4-1", name="Work (web)"),
                                      s.update(active_profile_id=s["profiles"][2]["id"])))
        self.assertEqual(syncer.sync()["state"], "synced")
        stored = self.stored()
        work = stored["profiles"][1]
        self.assertEqual((work["name"], work["model"], work["api_key"]), ("Work (web)", "claude-opus-4-1", KEY))
        self.assertEqual(params.active_profile_name(), "Local")
        for key in ("model_caps", "prefs", "net_grants"):
            self.assertEqual(stored[key], before[key], key)
        self.assertEqual(len(self.fake.ai_settings_puts), 1)

    def test_a_0_2_1_file_never_approves_an_address_the_website_sets(self):
        from oscar import providers
        from oscar.provider_transport import ProviderError

        self.put({"profiles": [WORK, dict(LOCAL)], "active_profile": "Work", "prefs": {}})   # no ids, no approvals
        syncer = settings_sync.Syncer(session_for=lambda: self.session, notify=lambda *a: None)
        self.assertEqual(syncer.sync()["state"], "synced")
        work_id, local_id = (p.id for p in params.load_profiles())

        def moved(s):
            s["profiles"][0]["base_url"] = "https://attacker.example/v1"   # Work: keyed, another server
            s["profiles"][1]["provider"] = "anthropic"                      # Local: keyless, another provider
        self.fake.web_edit(moved)
        self.assertEqual(syncer.sync()["state"], "synced")
        for pid in (work_id, local_id):
            profile = params.profile_by_id(pid)
            with self.subTest(profile.name):
                self.assertTrue(settings_sync.unusable_reason(profile))
                with self.assertRaises(ProviderError):
                    providers.make_provider(profile)
                passed, message, _ = settings_sync.run_profile_test(pid, run=lambda *a, **k: self.fail("sent"))
                self.assertEqual((passed, message), (False, settings_sync.NEEDS_CONFIRMATION_TEXT))
        self.assertEqual(params.profile_by_id(work_id).api_key, KEY)

        def back(s):
            s["profiles"][0]["base_url"] = WORK["base_url"]
            s["profiles"][1]["provider"] = LOCAL["provider"]
        self.fake.web_edit(back)
        self.assertEqual(syncer.sync()["state"], "synced")
        for pid in (work_id, local_id):
            profile = params.profile_by_id(pid)
            self.assertEqual(settings_sync.unusable_reason(profile), "")
            providers.make_provider(profile)

    def test_the_website_test_reads_the_profile_with_its_key(self):
        work = Profile(**WORK)
        params.save_profiles([work])
        seen = []

        class Report:
            ok, tools, vision = True, "native", True
            lines = [type("L", (), {"step": "connect", "ok": True, "text": f"ready {KEY}"})()]

            def summary(self):
                return f"✓ ready {KEY}"

        def run(profile, should_stop):
            seen.append((profile.id, profile.api_key))
            return Report()
        passed, message, checks = settings_sync.run_profile_test(work.id, run=run)
        self.assertEqual(seen, [(work.id, KEY)])
        self.assertTrue(passed)
        self.assertNotIn(KEY, message)
        self.assertEqual(checks["reply"], True)


# -- the dialog ------------------------------------------------------------------------------------------------------

try:
    from PySide import QtWidgets  # noqa: E402

    from oscar import settings_ui  # noqa: E402
    _QT_ERROR = None
except ImportError as e:   # FreeCAD / PySide missing: the dialog tests skip
    _QT_ERROR = e


class DialogTest(_Settings):
    @classmethod
    def setUpClass(cls):
        if _QT_ERROR is not None:
            raise unittest.SkipTest(f"Qt unavailable: {_QT_ERROR}")
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            cls.app = QtWidgets.QApplication(sys.argv[:1])

    def setUp(self):
        super().setUp()
        for reset in (discovery.clear_cache, session.reset_sessions, keystore.reset_memory):
            reset()
            self.addCleanup(reset)
        self.addCleanup(setattr, session, "on_change", session.on_change)
        self.addCleanup(setattr, settings_sync, "on_change", settings_sync.on_change)
        settings_ui.reset_account_hub()
        self.addCleanup(settings_ui.reset_account_hub)

    def dialog(self, profiles, active=None):
        params.save_profiles(profiles, active_id=active)
        dialog = settings_ui.SettingsDialog()
        dialog._open_url = lambda url: True
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(dialog._cancel_all_jobs)
        return dialog

    def pulled(self, dialog, mutate):
        """A sync changed the file, and says so (as the hub does on the GUI thread)."""
        params.update_state(lambda state: (True, mutate(state)))
        dialog._hub.settings_synced.emit(json.dumps({"event": "changed", "version": 2}))
        QtWidgets.QApplication.processEvents()

    def test_the_sync_line(self):
        d = self.dialog([params.account_profile()])
        deadline = time.monotonic() + 5
        while not d._hub.state.get("known") and time.monotonic() < deadline:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.01)
        d._show_sync_state()
        self.assertTrue(d.findChild(QtWidgets.QLabel, "oscarSyncStatus").text().startswith("Not synced"))
        with mock.patch.dict(os.environ, {settings_sync.ENV: "0"}):
            d._show_sync_state()
            self.assertIn("OSCAR_SETTINGS_SYNC=0", d.sync_label.text())
        with mock.patch.dict(d._hub.state, {"known": True, "signed_in": True}), \
                mock.patch.object(settings_sync, "status", return_value={"state": "synced", "message": ""}), \
                mock.patch.dict(os.environ, {settings_sync.ENV: "1"}):
            d._show_sync_state()
            self.assertEqual(d.sync_label.text(), "Synced with your OriginShape account. Your API keys stay on this "
                                                  "computer.")

    def test_a_pull_reloads_the_tab_when_nothing_is_unsaved(self):
        work, local = Profile(**WORK), Profile(**LOCAL)
        d = self.dialog([params.account_profile(), work, local], active=work.id)
        self.pulled(d, lambda state: state["profiles"][2].update(name="Ollama", model="llama3.3"))
        self.assertEqual([d.list.item(i).text() for i in range(d.list.count())], ["Work", "Ollama"])
        self.assertIn("Updated from your OriginShape account.", d.sync_label.text())
        self.assertFalse(d.sync_reload_btn.isVisibleTo(d))
        self.assertEqual(d._active.id, work.id)

    def test_a_pull_over_unsaved_changes_offers_reload_and_save_keeps_both(self):
        work, local = Profile(**WORK), Profile(**LOCAL)
        d = self.dialog([params.account_profile(), work, local], active=work.id)
        d.list.setCurrentRow(0)
        d.name_edit.setText("Work (edited here)")
        d._sync_current_name()
        self.pulled(d, lambda state: state["profiles"][2].update(model="llama3.3"))
        self.assertTrue(d.sync_reload_btn.isVisibleTo(d))
        self.assertIn("changed on your OriginShape account", d.sync_label.text())
        d._save()
        stored = self.stored()
        self.assertEqual([(p["name"], p["model"]) for p in stored["profiles"][1:]],
                         [("Work (edited here)", "claude-sonnet-4-5"), ("Local", "llama3.3")])
        self.assertEqual(stored["profiles"][1]["api_key"], KEY)

    def test_reload_drops_the_unsaved_changes(self):
        work, local = Profile(**WORK), Profile(**LOCAL)
        d = self.dialog([params.account_profile(), work, local], active=work.id)
        d.list.setCurrentRow(0)
        d.name_edit.setText("Work (edited here)")
        d._sync_current_name()
        self.pulled(d, lambda state: state["profiles"][2].update(model="llama3.3"))
        d.sync_reload_btn.click()
        self.assertEqual([p.name for p in d.profiles], ["OriginShape", "Work", "Local"])
        self.assertEqual(d.profiles[2].model, "llama3.3")
        self.assertFalse(d.sync_reload_btn.isVisibleTo(d))

    def moved(self):
        return Profile(name="Moved", provider="openai_compat", base_url="https://other.example/v1", model="m",
                       api_key=KEY, approved={"provider": "anthropic", "base_url": "https://api.anthropic.com"})

    def wait_test(self, d):
        deadline = time.monotonic() + 5
        while "test" in d._jobs and time.monotonic() < deadline:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.01)
        QtWidgets.QApplication.processEvents()
        d._drain_mailbox()

    def test_a_profile_changed_elsewhere_waits_and_use_my_key_confirms_it(self):
        moved = self.moved()
        d = self.dialog([params.account_profile(), moved], active=moved.id)
        d.list.setCurrentRow(0)
        self.assertTrue(d.pending_row.isVisibleTo(d))
        self.assertIn("it was api.anthropic.com", d.pending_label.text())
        self.assertEqual(d.confirm_btn.text(), "Use my key with other.example")
        self.assertTrue(d.remove_key_btn.isVisibleTo(d))
        d._test()                                   # the dialog's Test connection sends nothing meanwhile
        self.wait_test(d)
        self.assertIn("confirm", d.test_label.text())
        d.confirm_btn.click()
        self.assertFalse(d.pending_row.isVisibleTo(d))
        d._save()
        saved = params.profile_by_id(moved.id)
        self.assertEqual((saved.api_key, saved.base_url), (KEY, "https://other.example/v1"))
        self.assertEqual(saved.approved, {"provider": "openai_compat", "base_url": "https://other.example/v1"})
        self.assertFalse(settings_sync.needs_confirmation(saved))

    def test_remove_the_key_keeps_the_address_waiting(self):
        moved = self.moved()
        d = self.dialog([params.account_profile(), moved], active=moved.id)
        d.list.setCurrentRow(0)
        d.remove_key_btn.click()
        self.assertTrue(d.pending_row.isVisibleTo(d))
        self.assertEqual(d.confirm_btn.text(), "Use other.example")
        self.assertFalse(d.remove_key_btn.isVisibleTo(d))
        d._save()
        saved = params.profile_by_id(moved.id)
        self.assertEqual(saved.api_key, "")
        self.assertTrue(settings_sync.needs_confirmation(saved), "without its key it still goes nowhere unconfirmed")
        self.assertNotIn(KEY, self.raw().decode())

    def test_typing_a_key_approves_nothing_editing_the_address_does(self):
        moved = self.moved()
        d = self.dialog([params.account_profile(), moved], active=moved.id)
        d.list.setCurrentRow(0)
        d.key_edit.setText("sk-new-key-for-the-new-server")
        d._save()
        saved = params.profile_by_id(moved.id)
        self.assertEqual(saved.api_key, "sk-new-key-for-the-new-server")
        self.assertTrue(settings_sync.needs_confirmation(saved))
        d = settings_ui.SettingsDialog()
        self.addCleanup(d.deleteLater)
        d.list.setCurrentRow(0)
        d.base_edit.setText("https://my-own-proxy.example/v1")
        d._save()
        self.assertFalse(settings_sync.needs_confirmation(params.profile_by_id(moved.id)))

    def test_a_profile_removed_on_the_website_is_removed_or_kept_here(self):
        work, gone = Profile(**WORK), Profile(**{**LOCAL, "name": "Gone", "api_key": "sk-local-12345678",
                                                 "removed_remotely": True})
        d = self.dialog([params.account_profile(), work, gone], active=work.id)
        d.list.setCurrentRow(1)
        self.assertTrue(d.keep_btn.isVisibleTo(d) and d.remove_here_btn.isVisibleTo(d))
        self.assertIn("removed on your OriginShape account", d.pending_label.text())
        d.keep_btn.click()
        d._save()
        self.assertFalse(params.profile_by_id(gone.id).removed_remotely)
        params.update_state(lambda state: (True, state["profiles"][2].update(removed_remotely=True)))
        d = settings_ui.SettingsDialog()
        self.addCleanup(d.deleteLater)
        d.list.setCurrentRow(1)
        d.remove_here_btn.click()
        d._save()
        self.assertIsNone(params.profile_by_id(gone.id))
        self.assertNotIn("sk-local-12345678", self.raw().decode())

    def test_duplicate_makes_a_profile_of_its_own(self):
        work = Profile(**WORK)
        d = self.dialog([params.account_profile(), work], active=work.id)
        d.list.setCurrentRow(0)
        d._duplicate()
        clone = d.profiles[-1]
        self.assertNotEqual(clone.id, work.id)
        self.assertEqual(clone.name, "Work copy")
        d._save()
        self.assertEqual(params.active_profile_id(), work.id)
        self.assertEqual(len({p["id"] for p in self.stored()["profiles"]}), 3)


if __name__ == "__main__":
    unittest.main()
