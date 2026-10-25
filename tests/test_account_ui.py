"""The settings dialog's OriginShape account, against the fake OriginShape API (tests/support/fake_originshape.py).

Covered: the layout (three tabs, every part named, readable helper colours, the account card taking only the room of
the page shown, the Developer section with the Server field); the AI model tab (the account or your own provider
mapped onto the active profile, the account's model picked by name, the account profile never in the own-provider
list, "Add" starts from Anthropic, temperature and max tokens behind "Advanced overrides"); the Scripts switch
persisting ``AllowCodeExecution``; the settings-wipe guard (a dialog opened while the file was unreadable never
saves); the Server field; and every account action on worker threads: the browser sign-in (the test plays the
browser), cancelling it and closing the dialog (the 127.0.0.1 listener closes), sign-in with a code, a personal API
key, sign-out, the waitlist, and ``start_account_action`` (the panel's links and the Sign in command).
Needs FreeCAD and Qt (offscreen); the keystore runs in memory.
"""

import json
import os
import socket
import sys
import threading
import time
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

try:
    from PySide import QtCore, QtWidgets  # noqa: E402

    from oscar import params, settings_ui  # noqa: E402
    from oscar.auth import account, discovery, keystore, session  # noqa: E402
    from oscar.auth.transport import TokenSet  # noqa: E402
    from oscar.params import Profile  # noqa: E402
    from support.fake_originshape import FakeOriginShape, follow_authorize  # noqa: E402
    _IMPORT_ERROR = None
except ImportError as e:  # FreeCAD / PySide missing: the whole module skips
    _IMPORT_ERROR = e


def _closed_port() -> int:
    """A loopback port nothing listens on (bound, then released)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _own_provider(name="Mine"):
    return Profile(name=name, provider="openai_compat", base_url="http://llm.example.test/v1", model="m-own",
                   api_key="k-own")


class _DialogCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # never open a window on the user's screen
            cls.app = QtWidgets.QApplication(sys.argv[:1])

    def setUp(self):
        backup = params.backup_state()
        self.addCleanup(params.restore_state, backup)
        for reset in (discovery.clear_cache, session.reset_sessions, keystore.reset_memory, account.clear_model_cache):
            reset()
            self.addCleanup(reset)
        self.addCleanup(setattr, session, "on_change", session.on_change)
        settings_ui.reset_account_hub()
        self.addCleanup(settings_ui.reset_account_hub)
        self.fake = FakeOriginShape()
        self.addCleanup(self.fake.close)
        env = mock.patch.dict(os.environ, {"OSCAR_API_URL": self.fake.url, keystore.ENV_STORE: "memory"})
        env.start()
        self.addCleanup(env.stop)
        self.opened: list[str] = []      # what the dialog asked the browser to open

    def open_url(self, url: str) -> bool:
        self.opened.append(url)
        return True

    def dialog(self, profiles=None, active=None):
        profiles = profiles or [params.account_profile()]
        params.save_profiles(profiles)
        params.set_active_profile(active or profiles[0].name)
        dialog = settings_ui.SettingsDialog()
        dialog._open_url = self.open_url
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(dialog._cancel_all_jobs)
        return dialog

    def wait(self, predicate, timeout=15.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            QtWidgets.QApplication.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        QtWidgets.QApplication.processEvents()
        return predicate()

    def sign_in_directly(self) -> None:
        data = self.fake.issue_tokens()
        session.get_session(self.fake.url).complete_sign_in(
            TokenSet(data["access_token"], data["refresh_token"], data["expires_in"], data["scope"]))

    def play_browser(self) -> threading.Thread:
        """The user's browser: follow the authorize page's redirect to Oscar's loopback listener."""
        url = self.opened[-1]
        thread = threading.Thread(target=follow_authorize, args=(url,), daemon=True)
        thread.start()
        return thread


class LayoutTest(_DialogCase):
    def test_three_tabs_and_every_part_is_named(self):
        d = self.dialog()
        self.assertEqual(d.windowTitle(), "Oscar — Settings")
        self.assertEqual([d.tabs.tabText(i).replace("&&", "&") for i in range(d.tabs.count())],
                         ["Account", "AI model", "Tools & permissions"])
        account_tab, model_tab, tools_tab = (d.tabs.widget(i) for i in range(3))
        box = d.findChild(QtWidgets.QGroupBox, "oscarAccount")
        self.assertEqual(box.title(), "OriginShape account")
        self.assertTrue(account_tab.isAncestorOf(box))
        self.assertTrue(account_tab.isAncestorOf(d.server_edit))
        self.assertTrue(account_tab.isAncestorOf(d.connect_check))
        for widget in (d.use_account_radio, d.use_own_radio, d.account_model_combo, d.byok_box, d.list, d.test_btn):
            self.assertTrue(model_tab.isAncestorOf(widget), widget.objectName())
        for name in ("oscarNetPermissions", "oscarScripts", "oscarWebAccess"):
            self.assertTrue(tools_tab.isAncestorOf(d.findChild(QtWidgets.QGroupBox, name)), name)
        for kind, name in ((QtWidgets.QPushButton, "oscarAccountSignIn"),
                           (QtWidgets.QPushButton, "oscarAccountSignInCode"),
                           (QtWidgets.QPushButton, "oscarAccountUseKey"), (QtWidgets.QPushButton, "oscarAccountManage"),
                           (QtWidgets.QPushButton, "oscarAccountBilling"),
                           (QtWidgets.QPushButton, "oscarAccountSignOut"), (QtWidgets.QLabel, "oscarAccountEmail"),
                           (QtWidgets.QLabel, "oscarAccountPlan"), (QtWidgets.QLabel, "oscarAccountCredits"),
                           (QtWidgets.QLabel, "oscarAccountStatus"), (QtWidgets.QLineEdit, "oscarAccountServer"),
                           (QtWidgets.QCheckBox, "oscarAccountServerCustom"),
                           (QtWidgets.QWidget, "oscarAccountDeveloper"),
                           (QtWidgets.QCheckBox, "oscarConnectWorkspace"), (QtWidgets.QLabel, "oscarConnectStatus"),
                           (QtWidgets.QRadioButton, "oscarUseAccount"), (QtWidgets.QRadioButton, "oscarUseOwnProvider"),
                           (QtWidgets.QComboBox, "oscarAccountModel"), (QtWidgets.QComboBox, "oscarOwnProfileInUse"),
                           (QtWidgets.QWidget, "oscarByokSection"),
                           (QtWidgets.QGroupBox, "oscarScripts"), (QtWidgets.QCheckBox, "oscarAllowCode"),
                           (QtWidgets.QGroupBox, "oscarNetPermissions")):
            self.assertIsNotNone(d.findChild(kind, name), name)
        self.assertEqual(d.signin_btn.text(), "Sign in with OriginShape")
        self.assertEqual(d.signin_code_btn.text(), "Sign in with a code")
        self.assertEqual(d.use_key_btn.text(), "Use an OriginShape API key")
        self.assertTrue(d.use_key_btn.isHidden())   # personal keys are off on the server for now
        for button in (d.signin_btn, d.signin_code_btn, d.use_key_btn, d.signout_btn, d.account_manage_btn,
                       d.account_billing_btn, d.account_test_btn, d.test_btn):
            self.assertFalse(button.autoDefault(), button.objectName())      # Enter keeps meaning Save

    def test_the_tab_it_opens_on(self):
        self.assertEqual(self.dialog().tabs.currentIndex(), 0)                         # the account
        mine = self.dialog([params.account_profile(), _own_provider()], active="Mine")
        self.assertEqual(mine.tabs.currentIndex(), 1)                                  # your own provider: AI model

    def test_what_the_server_says_is_shown_as_plain_text(self):
        d = self.dialog()
        for label in (d.account_status, d.account_email, d.account_plan, d.account_credits, d.account_code_label,
                      d.ai_account_state, d.account_test_label, d.server_problem):
            self.assertEqual(label.textFormat(), QtCore.Qt.PlainText, label.objectName())

    def test_helper_text_is_readable_on_a_dark_and_a_light_palette(self):
        """Helper text used the palette's Mid role, which nearly vanished on dark themes: it is now the text colour
        blended 30 % towards the window, so it keeps a real contrast on both."""
        from PySide import QtGui

        def contrast(a, b):
            def lum(c):
                def ch(v):
                    v /= 255.0
                    return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
                return 0.2126 * ch(c.red()) + 0.7152 * ch(c.green()) + 0.0722 * ch(c.blue())
            hi, lo = sorted((lum(a), lum(b)), reverse=True)
            return (hi + 0.05) / (lo + 0.05)

        for window, text in (("#333333", "#dcdcdc"), ("#efefef", "#000000")):
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor(window))
            palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(text))
            palette.setColor(QtGui.QPalette.Mid, QtGui.QColor(window).darker(110))
            for tone in ("muted", "error", "ok"):
                css = settings_ui.tone_color(palette, tone)
                color = QtGui.QColor(*(int(v) for v in css[4:-1].split(","))) if css.startswith("rgb(") \
                    else QtGui.QColor(css)
                self.assertGreaterEqual(contrast(color, QtGui.QColor(window)), 4.5, (window, tone, css))
        d = self.dialog()
        self.assertNotIn("palette(mid)", d.connect_status.styleSheet())
        self.assertEqual(d.connect_status.property("oscarTone"), "muted")

    def test_signed_out_the_three_ways_in_are_offered(self):
        d = self.dialog()
        self.assertTrue(d._hub.state["known"])                   # the memory keystore is read at once
        self.assertEqual(d.account_stack.currentIndex(), 0)
        for button in (d.signin_btn, d.signin_code_btn, d.use_key_btn):
            self.assertTrue(button.isEnabled(), button.objectName())
        self.assertEqual(d.account_changed, False)
        self.assertIn("not signed in", d.ai_account_state.text())
        self.assertFalse(d.goto_account_btn.isHidden())

    def test_only_the_shown_account_page_takes_room(self):
        """The stack used to be as tall as the signed-in page, which left a large empty band around the signed-out
        text."""
        d = self.dialog()
        out, signed_in = d.account_stack.widget(0), d.account_stack.widget(1)
        self.assertEqual(out.sizePolicy().verticalPolicy(), QtWidgets.QSizePolicy.Preferred)
        self.assertEqual(signed_in.sizePolicy().verticalPolicy(), QtWidgets.QSizePolicy.Ignored)

    def test_the_developer_section_holds_the_server_and_opens_for_a_custom_one(self):
        os.environ.pop("OSCAR_API_URL")          # the setUp patch restores it
        d = self.dialog()
        self.assertFalse(d.server_toggle.isChecked())
        self.assertTrue(d.server_box.isHidden())
        self.assertFalse(d.server_custom.isChecked())                    # off by default: the app's server
        self.assertFalse(d.server_edit.isEnabled())
        d.server_custom.setChecked(True)
        d.server_edit.setText("http://example.com")
        self.assertFalse(d._apply_server_field())
        self.assertTrue(d.server_toggle.isChecked())                     # the reason is shown next to the field
        self.assertFalse(d.server_box.isHidden())
        self.assertFalse(d.server_problem.isHidden())
        self.assertIn("https", d.server_problem.text())
        params.set_bool("OriginShapeCustomServer", False)
        params.set_string("OriginShapeServer", "http://localhost:8210")
        self.assertFalse(self.dialog().server_toggle.isChecked())        # remembered but not used: nothing to flag
        params.set_bool("OriginShapeCustomServer", True)
        self.assertTrue(self.dialog().server_toggle.isChecked())         # not the default server: never a surprise


class ModelChoiceTest(_DialogCase):
    """The AI model tab: the account or your own provider, mapped onto the active profile."""

    def test_the_account_profile_is_never_listed_with_your_own_providers(self):
        d = self.dialog([params.account_profile(), _own_provider()])
        self.assertEqual([d.list.item(i).text() for i in range(d.list.count())], ["Mine"])
        self.assertEqual([p.name for p in d.profiles], ["OriginShape", "Mine"])
        self.assertTrue(d.use_account_radio.isChecked())
        self.assertTrue(d.byok_box.isHidden())                   # the profile editor belongs to "your own provider"
        self.assertFalse(d.account_details.isHidden())

    def test_the_account_model_is_picked_by_name(self):
        d = self.dialog()
        combo = d.account_model_combo
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["Origin1", "Origin1 Super"])
        self.assertEqual([combo.itemData(i) for i in range(combo.count())], list(params.ACCOUNT_MODEL_CHOICES))
        self.assertEqual(combo.currentData(), "origin1")
        self.assertEqual(d.account_model_hint.text(), "Recommended for most work.")
        combo.setCurrentIndex(combo.findData("origin1-super"))
        self.assertIn("more credits than Origin1", d.account_model_hint.text())
        d._save()
        stored = params.load_profiles()[0]
        self.assertEqual((stored.provider, stored.model, stored.base_url, stored.api_key),
                         ("originshape", "origin1-super", "", ""))

    def test_the_picker_lists_the_servers_models_in_its_words(self):
        self.sign_in_directly()
        with self.fake.lock:
            self.fake.models = {"default": "origin1", "data": [
                {"id": "origin1", "display_name": "Origin1", "description": "Fast and capable."},
                {"id": "origin1-super", "display_name": "Origin1 Super", "description": "The most capable model."},
                {"id": "extra-model", "display_name": "Extra Model", "description": "A model the server added."}]}
        account.fetch_models(self.fake.url)
        d = self.dialog()
        combo = d.account_model_combo
        self.assertEqual([combo.itemText(i) for i in range(combo.count())],
                         ["Origin1", "Origin1 Super", "Extra Model"])
        self.assertEqual(d.account_model_hint.text(), "Fast and capable.")
        combo.setCurrentIndex(combo.findData("extra-model"))
        self.assertEqual(d.account_model_hint.text(), "A model the server added.")
        d._save()
        stored = params.load_profiles()[0]
        self.assertEqual((stored.model, params.model_display_name(stored)), ("extra-model", "Extra Model"))

    def test_a_retired_alias_is_shown_as_its_tier_and_saving_keeps_working(self):
        for legacy, current, name in (("oscar-default", "origin1-super", "Origin1 Super"),
                                      ("oscar-fast", "origin1", "Origin1")):
            with self.subTest(legacy=legacy):
                profile = params.account_profile()
                profile.model = legacy                              # an old settings file: read as its tier
                d = self.dialog([profile])
                combo = d.account_model_combo
                self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["Origin1", "Origin1 Super"])
                self.assertEqual((combo.currentText(), combo.currentData()), (name, current))
                d._save()
                self.assertEqual(params.load_profiles()[0].model, current)
                d.profiles[0].model = legacy                        # a profile held in memory with the old alias
                d._show_account_model()
                self.assertEqual((combo.count(), combo.currentText()), (2, name))
                d._save()                                           # still saves: the server accepts the old alias
                self.assertEqual(params.load_profiles()[0].model, current)

    def test_the_model_tab_shows_the_accounts_names_only(self):
        for model in params.ACCOUNT_MODELS:
            with self.subTest(model=model):
                profile = params.account_profile()
                profile.model = model
                d = self.dialog([profile])
                combo = d.account_model_combo
                shown = [combo.itemText(i) for i in range(combo.count())] + [
                    str(combo.itemData(i, QtCore.Qt.ToolTipRole)) for i in range(combo.count())]
                shown += [d.account_model_hint.text()] + [label.text() for label in d.account_details.findChildren(
                    QtWidgets.QLabel)] + [b.text() for b in d.findChildren(QtWidgets.QRadioButton)]
                self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["Origin1", "Origin1 Super"])
                text = " ".join(shown).lower()
                for word in ("oscar-default", "oscar-fast"):     # a retired alias is shown by its tier's name
                    self.assertNotIn(word, text)

    def test_an_alias_the_dialog_does_not_know_is_kept(self):
        profile = params.account_profile()
        profile.model = "oscar-next"
        d = self.dialog([profile])
        self.assertEqual(d.account_model_combo.currentData(), "oscar-next")
        d._save()
        self.assertEqual(params.load_profiles()[0].model, "oscar-next")

    def test_choosing_your_own_provider_makes_it_active_and_the_account_back(self):
        d = self.dialog([params.account_profile(), _own_provider()])
        d.use_own_radio.click()
        self.assertIs(d._active, d.profiles[1])
        self.assertFalse(d.byok_box.isHidden())
        self.assertEqual(d.own_active_combo.currentText(), "Mine")
        self.assertEqual(d.key_edit.text(), "k-own")
        d._save()
        self.assertEqual(params.active_profile_name(), "Mine")
        again = self.dialog(params.load_profiles(), active="Mine")
        self.assertTrue(again.use_own_radio.isChecked())
        again.use_account_radio.click()
        again._save()
        self.assertEqual(params.active_profile().provider, "originshape")
        self.assertEqual(next(p for p in params.load_profiles() if p.name == "Mine").api_key, "k-own")

    def test_the_profile_in_use_menu_picks_among_your_profiles(self):
        d = self.dialog([params.account_profile(), _own_provider("A"), _own_provider("B")], active="A")
        self.assertEqual([d.own_active_combo.itemText(i) for i in range(d.own_active_combo.count())], ["A", "B"])
        d.own_active_combo.setCurrentIndex(1)
        d.own_active_combo.activated.emit(1)
        self.assertEqual(d._active.name, "B")
        self.assertEqual(d.name_edit.text(), "B")                # the editor follows
        d.list.setCurrentRow(0)                                  # looking at another profile changes nothing
        d._save()
        self.assertEqual(params.active_profile_name(), "B")

    def test_your_own_provider_without_a_profile_starts_one_that_the_account_drops_again(self):
        d = self.dialog()
        d.use_own_radio.click()
        self.assertEqual(len(d.profiles), 2)
        added = d.profiles[1]
        self.assertEqual(added.provider, "anthropic")
        self.assertIs(d._active, added)
        self.assertFalse(d.base_edit.isHidden())
        d.use_account_radio.click()                              # no key was entered: nothing to keep
        self.assertEqual([p.provider for p in d.profiles], ["originshape"])
        d.use_own_radio.click()
        d.key_edit.setText("sk-new")
        d.use_account_radio.click()                              # a key was entered: it is kept
        self.assertEqual(len(d.profiles), 2)
        self.assertEqual(d.profiles[1].api_key, "sk-new")

    def test_choosing_the_account_adds_its_profile_without_a_url_or_key(self):
        d = self.dialog([_own_provider()])
        self.assertTrue(d.use_own_radio.isChecked())
        d.base_edit.setText("http://should-stay-with-mine.test")
        d.use_account_radio.click()
        d._save()
        stored = params.load_profiles()
        self.assertEqual([(p.provider, p.base_url, p.api_key, p.model) for p in stored],
                         [("originshape", "", "", "origin1"),
                          ("openai_compat", "http://should-stay-with-mine.test", "k-own", "m-own")])
        self.assertEqual(params.active_profile().provider, "originshape")

    def test_another_providers_profile_keeps_its_model_even_when_it_reads_like_an_alias(self):
        gateway = Profile(name="Gateway", provider="anthropic", base_url="https://gateway.example.test/v1/llm/anthropic",
                          model="oscar-fast", api_key="k-gateway")
        d = self.dialog([params.account_profile(), gateway])
        d.list.setCurrentRow(0)
        self.assertTrue(d.model_combo.isEditable())
        self.assertEqual(d.model_combo.currentText(), "oscar-fast")
        self.assertEqual(d.key_edit.text(), "k-gateway")
        openai = next(i for i, p in enumerate(params.PROVIDER_PRESETS) if p["label"] == "OpenAI")
        d.preset_combo.setCurrentIndex(d.preset_combo.findData(openai))
        self.assertEqual(d.model_combo.currentText(), params.PROVIDER_PRESETS[openai]["default_model"])
        self.assertEqual(d.key_edit.text(), "")                  # the key belonged to the other provider

    def test_add_starts_from_anthropic(self):
        d = self.dialog([params.account_profile(), _own_provider()], active="Mine")
        d.list.setCurrentRow(0)
        d.preset_combo.setCurrentIndex(d.preset_combo.findData(next(
            i for i, p in enumerate(params.PROVIDER_PRESETS) if p["label"].startswith("Ollama"))))
        d._add()
        added = d.profiles[-1]
        self.assertEqual(d._selected(), added)
        self.assertTrue(d._active.name == "Mine")               # adding a profile does not switch to it

    def test_temperature_and_max_tokens_are_not_shown_as_settings(self):
        d = self.dialog([params.account_profile(), _own_provider()], active="Mine")
        self.assertTrue(d.temp_spin.isHidden())
        self.assertTrue(d.max_tokens_spin.isHidden())
        self.assertTrue(d.adv_box.isHidden())                    # the overrides live behind "Advanced overrides"
        self.assertTrue(d.adv_max_out.specialValueText().startswith("Automatic ("))

    def test_the_key_note_says_where_each_secret_lives(self):
        d = self.dialog()
        note = d.findChild(QtWidgets.QLabel, "oscarKeyStorageNote").text()
        self.assertIn("settings file on this computer", note)
        self.assertIn("keychain", note)
        self.assertNotIn("plain text", note)


class ScriptsSwitchTest(_DialogCase):
    def test_off_by_default_and_saved_with_save(self):
        d = self.dialog()
        self.assertFalse(d.allow_code_check.isChecked())
        self.assertIn("you approve each one", d.allow_code_check.text())
        d.allow_code_check.setChecked(True)
        d._save()
        self.assertTrue(params.get_bool("AllowCodeExecution", False))
        again = self.dialog(params.load_profiles())
        self.assertTrue(again.allow_code_check.isChecked())
        again.allow_code_check.setChecked(False)
        again._save()
        self.assertFalse(params.get_bool("AllowCodeExecution", True))

    def test_cancel_changes_nothing(self):
        d = self.dialog()
        d.allow_code_check.setChecked(True)
        d.reject()
        self.assertFalse(params.get_bool("AllowCodeExecution", False))


class WipeGuardTest(_DialogCase):
    """P0 #5: a dialog opened while the settings file could not be read holds an in-memory starter profile; saving it
    would replace the user's profiles, so it refuses."""

    def test_a_dialog_opened_on_an_unreadable_file_never_saves(self):
        params.save_profiles([_own_provider("Work")])
        path = params.settings_path()
        with open(path, "rb") as f:
            raw = f.read()
        real_open, left = open, {"n": 99}

        def flaky(name, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if os.fspath(name) == path and "r" in mode and "+" not in mode and left["n"] > 0:
                left["n"] -= 1
                raise PermissionError(13, "sharing violation")
            return real_open(name, *args, **kwargs)

        with mock.patch.object(params, "open", flaky, create=True), mock.patch("time.sleep", lambda s: None):
            d = settings_ui.SettingsDialog()
        self.addCleanup(d.deleteLater)
        self.assertFalse(d._from_file)
        d._save()
        with open(path, "rb") as f:
            self.assertEqual(f.read(), raw)
        self.assertFalse(d.save_label.isHidden())
        self.assertIn("could not read its settings file", d.save_label.text())
        self.assertNotEqual(d.result(), QtWidgets.QDialog.Accepted)
        self.assertEqual([(p.name, p.api_key) for p in params.load_profiles()], [("Work", "k-own")])


class ServerFieldTest(_DialogCase):
    def test_an_unsafe_server_is_refused_and_a_local_one_applies_at_once(self):
        os.environ.pop("OSCAR_API_URL")          # the setUp patch restores it
        d = self.dialog()
        d.server_custom.setChecked(True)
        self.assertTrue(d.account_changed)        # the switch alone changes the server (here: still the app)
        d.account_changed = False
        self.assertTrue(d.server_edit.isEnabled())
        self.assertEqual(d.server_edit.text(), "")
        d.server_edit.setText("http://example.com")
        self.assertFalse(d._apply_server_field())
        self.assertIn("https", d.account_status.text())
        self.assertEqual(params.get_string("OriginShapeServer", ""), "")
        self.assertFalse(d.account_changed)
        d.server_edit.setText("http://LOCALHOST:8210/")
        self.assertTrue(d._apply_server_field())
        self.assertEqual(params.get_string("OriginShapeServer", ""), "http://localhost:8210")
        self.assertEqual(d.server_edit.text(), "http://localhost:8210")
        self.assertTrue(d.account_changed)
        self.assertEqual(d._hub.state["api_url"], "http://localhost:8210")

    def test_a_refused_server_blocks_save(self):
        os.environ.pop("OSCAR_API_URL")
        d = self.dialog()
        d.server_custom.setChecked(True)
        d.server_edit.setText("ftp://nowhere")
        d._save()
        self.assertFalse(d.save_label.isHidden())
        self.assertNotEqual(d.result(), QtWidgets.QDialog.Accepted)

    def test_the_custom_server_applies_only_while_its_switch_is_on(self):
        os.environ.pop("OSCAR_API_URL")
        params.set_string("OriginShapeServer", "http://localhost:8210")   # left over from testing a local stack
        d = self.dialog()
        self.assertFalse(d.server_custom.isChecked())
        self.assertFalse(d.server_edit.isEnabled())
        self.assertEqual(d.server_edit.text(), "http://localhost:8210")    # kept for next time
        self.assertEqual(d._hub.state["api_url"], "https://app.originshape.design")
        self.assertTrue(d._apply_server_field())                            # a sign-in goes to the app
        d.server_custom.setChecked(True)
        self.assertTrue(params.get_bool("OriginShapeCustomServer", False))
        self.assertTrue(d.server_edit.isEnabled())
        self.assertEqual(d._hub.state["api_url"], "http://localhost:8210")
        self.assertEqual(d.account_status.text(), "Server: http://localhost:8210")
        self.assertTrue(d.account_changed)
        d.server_custom.setChecked(False)
        self.assertFalse(params.get_bool("OriginShapeCustomServer", True))
        self.assertEqual(params.get_string("OriginShapeServer", ""), "http://localhost:8210")
        self.assertEqual(d._hub.state["api_url"], "https://app.originshape.design")
        self.assertEqual(d.account_status.text(), "Server: https://app.originshape.design")

    def test_the_environment_variable_wins(self):
        d = self.dialog()
        self.assertFalse(d.server_edit.isEnabled())
        self.assertTrue(d.server_custom.isChecked())
        self.assertFalse(d.server_custom.isEnabled())
        self.assertEqual(d.server_edit.text(), self.fake.url)


class BrowserSignInTest(_DialogCase):
    def test_signing_in_with_the_browser(self):
        d = self.dialog([_own_provider()])
        d.signin_btn.click()
        self.assertEqual(len(self.opened), 1)
        authorize = self.opened[0]
        self.assertTrue(authorize.startswith(self.fake.url + "/oauth/authorize?"), authorize)
        query = parse_qs(urlsplit(authorize).query)
        self.assertTrue(query["redirect_uri"][0].startswith("http://127.0.0.1:"))
        self.assertEqual(d.signin_btn.text(), "Cancel sign-in")
        self.assertFalse(d.signin_code_btn.isEnabled())
        self.play_browser()
        self.assertTrue(self.wait(lambda: not d._account_busy() and d._hub.state["signed_in"]), d.account_status.text())
        self.assertTrue(session.get_session(self.fake.url).has_session())
        self.assertTrue(self.wait(lambda: d.account_email.text() == "ada@example.com"), d.account_email.text())
        self.assertEqual(d.account_stack.currentIndex(), 1)
        self.assertEqual(d.account_plan.text(), "Trial")
        self.assertEqual(d.account_credits.text(), "42.5 credits")
        self.assertTrue(d.account_changed)
        self.assertIn("Signed in.", d.account_status.text())
        # the account profile was added and made active; the user's own profile is untouched
        self.assertTrue(params.is_account_profile(d._active))
        self.assertIn(d._active, d.profiles)
        self.assertEqual(params.active_profile().provider, "originshape")
        self.assertIn("Mine", [p.name for p in params.load_profiles()])
        self.assertEqual(next(p for p in params.load_profiles() if p.name == "Mine").api_key, "k-own")
        d._save()                                           # Save keeps the account profile active
        self.assertEqual(params.active_profile().provider, "originshape")

    def test_a_declined_sign_in_says_so(self):
        self.fake.authorize_decision = "deny"
        d = self.dialog()
        d.signin_btn.click()
        self.play_browser()
        self.assertTrue(self.wait(lambda: not d._account_busy()))
        self.assertIn("Not signed in", d.account_status.text())
        self.assertFalse(d._hub.state["signed_in"])
        self.assertFalse(d.account_changed)
        self.assertEqual(d.signin_btn.text(), "Sign in with OriginShape")

    def test_a_second_click_cancels(self):
        d = self.dialog()
        d.signin_btn.click()
        d.signin_btn.click()
        self.assertFalse(d._account_busy())
        self.assertEqual(d.account_status.text(), "Sign-in cancelled.")
        self.assertEqual(d.signin_btn.text(), "Sign in with OriginShape")

    def test_closing_the_dialog_closes_the_listener(self):
        d = self.dialog()
        d.signin_btn.click()
        port = urlsplit(parse_qs(urlsplit(self.opened[0]).query)["redirect_uri"][0]).port
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass                                            # listening while the sign-in waits
        d.reject()

        def closed():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    return False
            except OSError:
                return True

        self.assertTrue(self.wait(closed, timeout=5.0))
        self.assertFalse(session.get_session(self.fake.url).has_session())

    def test_an_unreachable_custom_server_ends_the_sign_in_and_names_its_setting(self):
        os.environ.pop("OSCAR_API_URL")          # the setUp patch restores it
        params.set_string("OriginShapeServer", f"http://127.0.0.1:{_closed_port()}")
        params.set_bool("OriginShapeCustomServer", True)
        d = self.dialog()
        d.signin_btn.click()
        self.assertTrue(self.wait(lambda: not d._account_busy(), timeout=5.0), d.account_status.text())
        text = d.account_status.text()
        self.assertIn("Not signed in: Could not reach", text)
        self.assertIn("turn off Use a custom server", text)
        self.assertIn("https://app.originshape.design", text)
        self.assertEqual(d.signin_btn.text(), "Sign in with OriginShape")

    def test_start_account_action_is_what_the_panel_links_use(self):
        d = self.dialog()
        d.start_account_action("signin")
        self.assertEqual(len(self.opened), 1)
        self.assertTrue(d._account_busy())
        d.start_account_action("device")                    # one account action at a time
        self.assertEqual(len(self.opened), 1)
        d._cancel_all_jobs()
        self.sign_in_directly()
        signed_in = self.dialog()
        signed_in.start_account_action("signin")            # already signed in: nothing to do
        self.assertEqual(len(self.opened), 1)


class CodeAndKeySignInTest(_DialogCase):
    def test_signing_in_with_a_code(self):
        self.fake.device_interval = 1
        d = self.dialog()
        d.signin_code_btn.click()
        self.assertTrue(self.wait(lambda: not d.account_code_label.isHidden()), d.account_status.text())
        code = d.account_code_label.text()
        self.assertIn(code, self.fake.devices)
        self.assertIn("user_code=", self.opened[-1])            # the page that already carries the code
        self.assertEqual(d.signin_code_btn.text(), "Cancel")
        self.fake.approve_device()
        self.assertTrue(self.wait(lambda: d._hub.state["signed_in"] and not d._account_busy()), d.account_status.text())
        self.assertTrue(d.account_code_label.isHidden())
        self.assertTrue(d.account_changed)
        self.assertEqual(params.active_profile().provider, "originshape")

    def test_no_code_from_an_unreachable_server_says_why(self):
        os.environ["OSCAR_API_URL"] = f"http://127.0.0.1:{_closed_port()}"   # the setUp patch restores it
        d = self.dialog()
        d.signin_code_btn.click()
        self.assertTrue(self.wait(lambda: not d._account_busy()), d.account_status.text())
        text = d.account_status.text()
        self.assertIn("Could not get a sign-in code: Could not reach", text)
        self.assertIn("OSCAR_API_URL", text)
        self.assertTrue(d.account_code_label.isHidden())
        self.assertEqual(d.signin_code_btn.text(), "Sign in with a code")

    def test_a_personal_api_key(self):
        d = self.dialog()
        d.use_key_btn.click()
        self.assertFalse(d.account_key_row.isHidden())
        self.assertEqual(d.account_key_edit.echoMode(), QtWidgets.QLineEdit.Password)
        d.account_key_edit.setText("nope")
        d.account_key_btn.click()
        self.assertTrue(self.wait(lambda: not d._account_busy()))
        self.assertIn("os_live_", d.account_status.text())
        self.assertFalse(d._hub.state["signed_in"])
        key = self.fake.add_api_key()
        d.account_key_edit.setText(key)
        d.account_key_btn.click()
        self.assertTrue(self.wait(lambda: d._hub.state["signed_in"] and not d._account_busy()), d.account_status.text())
        self.assertEqual(d._hub.state["kind"], "api_key")
        self.assertEqual(d.account_key_edit.text(), "")          # the key lives in the keystore only
        self.assertIn("personal API key", d.account_plan.text())
        self.assertNotIn(key, d.account_status.text())
        self.assertNotIn(key, json.dumps(d._hub.state))

    def test_a_refused_key_stays_editable(self):
        d = self.dialog()
        d.use_key_btn.click()
        d.account_key_edit.setText("os_live_" + "x" * 43)       # well formed, unknown to the server
        d.account_key_btn.click()
        self.assertTrue(self.wait(lambda: not d._account_busy()))
        self.assertIn("did not accept", d.account_status.text())
        self.assertEqual(d.account_key_edit.text(), "os_live_" + "x" * 43)


class SignedInTest(_DialogCase):
    def test_signing_out(self):
        self.sign_in_directly()
        d = self.dialog()
        self.assertTrue(self.wait(lambda: d.account_email.text() == "ada@example.com"))
        self.assertEqual(d.account_stack.currentIndex(), 1)
        d.signout_btn.click()
        self.assertTrue(self.wait(lambda: not d._account_busy() and not d._hub.state["signed_in"]))
        self.assertEqual(d.account_stack.currentIndex(), 0)
        self.assertEqual(self.fake.counts["revoke"], 1)
        self.assertTrue(d.account_changed)
        self.assertEqual(d.account_status.text(), "Signed out.")
        self.assertFalse(session.get_session(self.fake.url).has_session())

    def test_a_waitlisted_account(self):
        self.fake.set_account(status="waitlisted")
        self.sign_in_directly()
        d = self.dialog()
        self.assertTrue(self.wait(lambda: not d.account_waitlist.isHidden()))
        self.assertTrue(d.account_credits.isHidden())
        self.assertTrue(d._account_form.labelForField(d.account_credits).isHidden())
        self.assertTrue(d.account_billing_btn.isHidden())                # nothing to buy before approval
        self.assertIn("waitlist", d.ai_account_state.text())
        self.assertFalse(d.goto_account_btn.isHidden())

    def test_a_suspended_account(self):
        self.fake.set_account(status="suspended")
        self.sign_in_directly()
        d = self.dialog()
        self.assertTrue(self.wait(lambda: not d.account_notice.isHidden()))
        self.assertIn("suspended", d.account_notice.text())
        self.assertEqual(d.account_notice.property("oscarTone"), "error")
        self.assertTrue(d.account_billing_btn.isHidden())

    def test_no_credits_left(self):
        self.fake.set_account(balance=0)
        self.sign_in_directly()
        d = self.dialog()
        self.assertTrue(self.wait(lambda: d.account_credits.text() == "0 credits"), d.account_credits.text())
        self.assertFalse(d.account_notice.isHidden())
        self.assertIn("out of credits", d.account_notice.text())
        self.assertFalse(d.account_billing_btn.isHidden())
        self.assertTrue(d.account_billing_btn.property("oscarPrimary"))  # buying credits is the thing to do
        d.account_billing_btn.click()
        self.assertEqual(self.opened, [self.fake.me["links"]["billing"]])

    def test_credits_are_shown_on_both_tabs(self):
        self.sign_in_directly()
        d = self.dialog()
        self.assertTrue(self.wait(lambda: d.account_credits.text() == "42.5 credits"), d.account_credits.text())
        self.assertTrue(d.account_notice.isHidden())
        self.assertFalse(d.account_billing_btn.property("oscarPrimary"))
        self.assertEqual(d.ai_account_state.text(), "Signed in as ada@example.com · 42.5 credits available")
        self.assertTrue(d.goto_account_btn.isHidden())

    def test_an_account_that_cannot_be_loaded_says_so_until_it_can(self):
        self.sign_in_directly()
        self.fake.fail("/v1/me", 503, times=1)
        d = self.dialog()
        self.assertTrue(self.wait(lambda: d.account_status.text().startswith("Could not load your account")),
                        d.account_status.text())
        self.assertEqual(d.account_stack.currentIndex(), 1)       # still signed in: the session is kept
        d._hub.refresh(force=True)
        self.assertTrue(self.wait(lambda: d.account_email.text() == "ada@example.com"))
        self.assertTrue(self.wait(lambda: d.account_status.isHidden()), d.account_status.text())

    def test_manage_account_opens_the_accounts_own_page(self):
        self.sign_in_directly()
        d = self.dialog()
        self.assertTrue(self.wait(lambda: bool(d._hub.state.get("snapshot"))))
        d.account_manage_btn.click()
        self.assertEqual(self.opened, [self.fake.me["links"]["manage"]])

    def test_the_account_is_loaded_on_a_worker_thread(self):
        self.sign_in_directly()
        self.fake.delays = {"/v1/me": 2.0}
        started = time.monotonic()
        d = self.dialog()
        self.assertLess(time.monotonic() - started, 1.5)         # the dialog opened without waiting for /v1/me
        self.assertEqual(d.account_email.text(), "(loading…)")
        self.assertTrue(self.wait(lambda: d.account_email.text() == "ada@example.com"))


if __name__ == "__main__":
    unittest.main()
