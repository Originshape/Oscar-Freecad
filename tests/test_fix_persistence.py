"""Regression tests for the persistence review fixes.

persistence-01 (Remove clobbers the neighbour), -04 (renaming the active profile
switched it), -05 (duplicate profile names), -07 (Test / Load models froze the GUI),
-08 (settings file wiped on a read problem), -09 (non-UTF-8 attachments),
-10 (one malformed chat file locked the panel out), -13 (Test showed
"exceptions must derive from BaseException"), -14 (every OpenAI-compatible
profile shown as "OpenAI").

The settings dialog runs against a real offscreen QApplication; no network,
every settings file / chat directory lives in a temp dir.

The dialog's background work is a dict of jobs by kind (``dialog._jobs``) delivered
through a mailbox drained by a GUI timer; ``_cancel_job`` takes the kind; the
connection test is ``provider_check`` (a fake provider's ``chat`` takes the extra
options it passes).
"""

import glob
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (suite guard)

if "FreeCADGui" not in sys.modules:
    try:
        import FreeCADGui  # noqa: F401
    except Exception:
        sys.modules["FreeCADGui"] = types.ModuleType("FreeCADGui")

import FreeCAD  # noqa: E402,F401

from oscar import attachments, chats, params  # noqa: E402
from oscar.params import Profile  # noqa: E402

try:
    from PySide import QtCore, QtGui, QtWidgets
except Exception:  # pragma: no cover - GUI-less interpreter
    QtCore = QtGui = QtWidgets = None


def _preset_index(label_part):
    for i, preset in enumerate(params.PROVIDER_PRESETS):
        if label_part in preset["label"]:
            return i
    raise KeyError(label_part)


class _IsolatedSettings(unittest.TestCase):
    """Each test gets its own settings file (the env var is read per call)."""

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="oscar_persist_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.settings = os.path.join(self.tmp, "settings.json")
        patcher = mock.patch.dict(os.environ,
                                  {"OSCAR_AI_SETTINGS": self.settings})
        patcher.start()
        self.addCleanup(patcher.stop)
        migrated = params._MIGRATED
        params._MIGRATED = True  # skip the one-time import of old user parameters
        self.addCleanup(setattr, params, "_MIGRATED", migrated)


class _DialogCase(_IsolatedSettings):
    def setUp(self):
        super().setUp()
        if QtWidgets is None:
            self.skipTest("PySide unavailable")
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            # a stray test must never open a real window on the desktop
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            self.app = QtWidgets.QApplication(getattr(sys, "argv", ["tests"])[:1])

    @staticmethod
    def _close(dialog):
        try:
            dialog.reject()
        except RuntimeError:  # the C++ dialog was deleted by the test
            pass

    def _dialog(self, profiles, active=None):
        params.save_profiles(profiles)
        params.set_active_profile(active or profiles[0].name)
        from oscar.settings_ui import SettingsDialog

        dialog = SettingsDialog()
        self.addCleanup(self._close, dialog)
        return dialog

    def _spin(self, condition, timeout=5.0):
        end = time.time() + timeout
        while not condition() and time.time() < end:
            self.app.processEvents()
            time.sleep(0.005)
        return condition()

    def _wait(self, dialog):
        """Until the dialog's background job (if any) has been delivered."""
        self.assertTrue(self._spin(lambda: not dialog._jobs),
                        "background job never finished")

    def _pump(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.app.processEvents()
            time.sleep(0.005)


# ---------------------------------------------------------------------------
# persistence-01: Remove


class RemoveProfileTest(_DialogCase):
    def test_remove_keeps_every_other_profile_intact(self):
        for index in (0, 1, 2):
            with self.subTest(removed=index):
                profiles = [Profile(name=n, api_key=f"key-{n}", model=f"model-{n}")
                            for n in "ABC"]
                dialog = self._dialog(profiles)
                dialog.list.setCurrentRow(index)
                dialog._remove()
                expected = [(p.name, p.api_key, p.model)
                            for i, p in enumerate(profiles) if i != index]
                self.assertEqual(
                    [(p.name, p.api_key, p.model) for p in dialog.profiles],
                    expected)
                dialog._save()
                self.assertEqual(
                    [(p.name, p.api_key, p.model) for p in params.load_profiles()],
                    expected)


# ---------------------------------------------------------------------------
# persistence-05 / -04: names and the active profile


class ProfileNamesTest(_DialogCase):
    def _unique(self, dialog):
        names = [p.name.casefold() for p in dialog.profiles]
        self.assertEqual(len(set(names)), len(names), names)

    def test_duplicate_twice_from_the_same_source_keeps_names_unique(self):
        dialog = self._dialog([Profile(name="A", api_key="kA"),
                               Profile(name="B", api_key="kB"),
                               Profile(name="C")])
        dialog.list.setCurrentRow(1)
        dialog._duplicate()
        dialog.list.setCurrentRow(1)
        dialog._duplicate()
        self._unique(dialog)
        # the freshly created clone is the one shown and edited
        row = dialog.list.currentRow()
        self.assertEqual(row, len(dialog.profiles) - 1)
        self.assertEqual(dialog.name_edit.text(), dialog.profiles[row].name)
        self.assertEqual(dialog.key_edit.text(), "kB")

    def test_rename_to_an_existing_name_is_made_unique(self):
        dialog = self._dialog([Profile(name="A"), Profile(name="B")])
        dialog.list.setCurrentRow(0)
        dialog.name_edit.setText("B")
        dialog.name_edit.editingFinished.emit()
        self._unique(dialog)
        for row, profile in enumerate(dialog.profiles):
            self.assertEqual(dialog.list.item(row).text(), profile.name)

    def test_rename_on_save_is_made_unique(self):
        dialog = self._dialog([Profile(name="A"), Profile(name="B")])
        dialog.list.setCurrentRow(0)
        dialog.name_edit.setText("B")  # no editingFinished: only stashed by Save
        dialog._save()
        names = [p.name for p in params.load_profiles()]
        self.assertEqual(len(set(names)), 2, names)

    def test_add_after_remove_does_not_reuse_a_name(self):
        dialog = self._dialog([Profile(name="Profile 1"), Profile(name="Profile 2"),
                               Profile(name="Profile 3")])
        dialog.list.setCurrentRow(0)
        dialog._remove()
        dialog._add()
        self._unique(dialog)
        row = dialog.list.currentRow()
        self.assertEqual(row, len(dialog.profiles) - 1)
        self.assertEqual(dialog.name_edit.text(), dialog.profiles[row].name)

    def test_names_already_duplicated_on_disk_are_fixed_on_save(self):
        dialog = self._dialog([Profile(name="B copy", api_key="k1"),
                               Profile(name="B copy", api_key="k2")])
        dialog._save()
        saved = params.load_profiles()
        self.assertEqual(len({p.name for p in saved}), 2)
        self.assertEqual([p.api_key for p in saved], ["k1", "k2"])


class ActiveProfileTest(_DialogCase):
    def test_renaming_the_active_profile_keeps_it_active(self):
        dialog = self._dialog([Profile(name="A", api_key="kA"),
                               Profile(name="B", api_key="kB")], active="B")
        dialog.list.setCurrentRow(1)
        dialog.name_edit.setText("B-renamed")
        dialog.name_edit.editingFinished.emit()
        dialog._save()
        self.assertEqual(params.active_profile_name(), "B-renamed")
        self.assertEqual(params.active_profile().api_key, "kB")

    def test_renaming_via_save_alone_keeps_it_active(self):
        dialog = self._dialog([Profile(name="A"), Profile(name="B")], active="B")
        dialog.list.setCurrentRow(1)
        dialog.name_edit.setText("B2")
        dialog._save()
        self.assertEqual(params.active_profile_name(), "B2")

    def test_removing_the_active_profile_falls_back_to_the_first(self):
        dialog = self._dialog([Profile(name="A"), Profile(name="B"),
                               Profile(name="C")], active="B")
        dialog.list.setCurrentRow(1)
        dialog._remove()
        dialog._save()
        self.assertEqual(params.active_profile_name(), "A")

    def test_untouched_active_profile_stays(self):
        dialog = self._dialog([Profile(name="A"), Profile(name="B")], active="B")
        dialog._save()
        self.assertEqual(params.active_profile_name(), "B")


# ---------------------------------------------------------------------------
# persistence-14: the provider combo


class PresetLabelTest(_DialogCase):
    def test_each_profile_shows_its_own_provider(self):
        cases = [
            ("anthropic", "https://api.anthropic.com", "Anthropic (Claude)"),
            ("openai_compat", "https://api.openai.com/v1", "OpenAI"),
            ("openai_compat",
             "https://generativelanguage.googleapis.com/v1beta/openai",
             "Google Gemini"),
            ("openai_compat", "http://localhost:11434/v1", "Ollama (local)"),
            ("openai_compat", "https://my.proxy.example/v1",
             "Custom (OpenAI-compatible)"),
        ]
        profiles = [Profile(name=f"p{i}", provider=prov, base_url=url)
                    for i, (prov, url, _label) in enumerate(cases)]
        dialog = self._dialog(profiles)
        for row, (_prov, _url, label) in enumerate(cases):
            dialog.list.setCurrentRow(row)
            self.assertEqual(dialog.preset_combo.currentText(), label, row)
        # visiting every row stored nothing different
        dialog._save()
        saved = params.load_profiles()
        self.assertEqual([(p.provider, p.base_url) for p in saved],
                         [(prov, url) for prov, url, _ in cases])


# ---------------------------------------------------------------------------
# persistence-13: error events from the provider


class _FakeProvider:
    name = "fake"

    def __init__(self, events):
        self.events = events

    def chat(self, system, messages, tools, should_stop=None, **opts):
        yield from self.events


class TestConnectionErrorTest(_DialogCase):
    def _label_for(self, events):
        dialog = self._dialog([Profile(name="P", provider="openai_compat",
                                       base_url="http://localhost:1/v1",
                                       model="m")])
        with mock.patch("oscar.providers.make_provider",
                        lambda profile: _FakeProvider(events)):
            dialog._test()
            self._wait(dialog)
        return dialog.test_label.text()

    def test_dict_payload_shows_the_server_message(self):
        label = self._label_for([("error", {"code": "model_not_found",
                                            "message": "no such model: m"})])
        self.assertTrue(label.startswith("✗"), label)
        self.assertIn("no such model: m", label)
        self.assertNotIn("BaseException", label)

    def test_nested_failure_event_shows_the_server_message(self):
        label = self._label_for([("error", {
            "type": "response.failed",
            "response": {"error": {"message": "quota exceeded"}}})])
        self.assertIn("quota exceeded", label)
        self.assertNotIn("BaseException", label)

    def test_payload_without_a_message_is_still_readable(self):
        label = self._label_for([("error", {"type": "weird", "code": 7})])
        self.assertTrue(label.startswith("✗"), label)
        self.assertNotIn("BaseException", label)
        self.assertIn("weird", label)

    def test_exception_payload_keeps_its_text(self):
        label = self._label_for([("error", RuntimeError("HTTP 401: bad key"))])
        self.assertIn("HTTP 401: bad key", label)

    def test_success_label(self):
        label = self._label_for([("text", "rea"), ("text", "dy")])
        self.assertTrue(label.startswith("✓"), label)
        self.assertIn("ready", label)


# ---------------------------------------------------------------------------
# persistence-07: the dialog must not freeze the GUI thread


class AsyncDialogTest(_DialogCase):
    def setUp(self):
        super().setUp()
        self.release = threading.Event()
        self.started = threading.Event()
        self.thread_errors = []
        hook = mock.patch.object(threading, "excepthook",
                                 lambda args: self.thread_errors.append(args))
        hook.start()
        self.addCleanup(hook.stop)
        self.addCleanup(self.release.set)  # never leave a stub thread parked


    def _ticks(self):
        ticks = []
        timer = QtCore.QTimer()
        timer.setInterval(10)
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start()
        self.addCleanup(timer.stop)
        return ticks


    def test_test_connection_runs_off_the_gui_thread(self):
        threads = []
        outer = self

        class Gated:
            name = "gated"

            def chat(self, system, messages, tools, should_stop=None, **opts):
                threads.append(threading.current_thread())
                outer.started.set()
                outer.release.wait(1.5)
                yield "text", "ready"

        dialog = self._dialog([Profile(name="P", provider="openai_compat",
                                       base_url="http://localhost:1/v1",
                                       model="m")])
        ticks = self._ticks()
        with mock.patch("oscar.providers.make_provider", lambda p: Gated()):
            began = time.time()
            dialog._test()
            self.assertLess(time.time() - began, 0.3)
            self.assertEqual(dialog.test_label.text(), "Testing…")
            self._pump(0.15)
            self.assertGreaterEqual(len(ticks), 3)
            self.release.set()
            self._wait(dialog)
        self.assertIsNot(threads[0], threading.main_thread())
        self.assertTrue(dialog.test_label.text().startswith("✓"),
                        dialog.test_label.text())
        self.assertIn("ready", dialog.test_label.text())

    def test_load_models_runs_off_the_gui_thread(self):
        def slow_models(profile):
            self.started.set()
            self.release.wait(1.5)
            return ["m1", "m2"], ""

        dialog = self._dialog([Profile(name="P", provider="openai_compat",
                                       base_url="http://localhost:1/v1",
                                       model="m1")])
        ticks = self._ticks()
        with mock.patch("oscar.providers.list_models", slow_models):
            began = time.time()
            dialog._load_models()
            self.assertLess(time.time() - began, 0.3)
            self._pump(0.15)
            self.assertGreaterEqual(len(ticks), 3)
            self.release.set()
            self._wait(dialog)
        items = [dialog.model_combo.itemText(i)
                 for i in range(dialog.model_combo.count())]
        self.assertEqual(items, ["m1", "m2"])
        self.assertIn("2 models loaded", dialog.test_label.text())

    def test_models_for_another_row_are_not_applied(self):
        def slow_models(profile):
            self.started.set()
            self.release.wait(1.5)
            return ["x1", "x2"], ""

        dialog = self._dialog([
            Profile(name="P", provider="openai_compat",
                    base_url="http://localhost:1/v1", model="m1"),
            Profile(name="Q", provider="openai_compat",
                    base_url="http://localhost:1/v1", model="q-model")])
        with mock.patch("oscar.providers.list_models", slow_models):
            dialog._load_models()
            self.assertTrue(self.started.wait(1.0))
            dialog.list.setCurrentRow(1)
            self.release.set()
            self._pump(0.2)
        self.assertEqual(dialog.model_combo.currentText(), "q-model")
        self.assertEqual(dialog.test_label.text(), "")

    def test_buttons_are_locked_while_a_job_runs(self):
        dialog = self._dialog([Profile(name="P", provider="openai_compat",
                                       base_url="http://localhost:1/v1",
                                       model="m1")])

        def slow_models(profile):
            self.started.set()
            self.release.wait(1.5)
            return ["m1"], ""

        with mock.patch("oscar.providers.list_models", slow_models):
            dialog._load_models()
            self.assertTrue(self.started.wait(1.0))
            self.assertFalse(dialog.test_btn.isEnabled())
            self.release.set()
            self._wait(dialog)
        self.assertTrue(dialog.test_btn.isEnabled())
        self.assertTrue(dialog.load_btn.isEnabled())


# ---------------------------------------------------------------------------
# persistence-10: a malformed chat file must never lock the panel out


class ChatsRobustnessTest(unittest.TestCase):
    def setUp(self):
        self.sandbox = tempfile.mkdtemp(prefix="chats_fix_")
        self.addCleanup(shutil.rmtree, self.sandbox, True)
        original = chats.chats_dir
        chats.chats_dir = lambda: self.sandbox
        self.addCleanup(setattr, chats, "chats_dir", original)

    def _write(self, name, text):
        with open(os.path.join(self.sandbox, name), "w", encoding="utf-8") as f:
            f.write(text)

    MALFORMED = {
        "list": "[]",
        "str": '"x"',
        "null": "null",
        "msgstr": '{"id":"a","title":"x","messages":["hi"]}',
        "tcnoid": '{"id":"a","messages":[{"role":"assistant",'
                  '"tool_calls":[{"name":"n","arguments":{}}]}]}',
        "tcstr": '{"id":"a","messages":[{"role":"assistant","tool_calls":["x"]}]}',
        "msgsdict": '{"id":"a","messages":{}}',
        "resultstr": '{"id":"a","messages":[{"role":"user","tool_results":["x"]}]}',
    }

    def test_malformed_files_never_raise(self):
        good = chats.new_session("good")
        chats.save(good)
        for label, text in self.MALFORMED.items():
            self._write(f"{label}.json", text)
        # the shapes list_sessions can read but load() must refuse
        self._write("msgsnull.json", '{"id":"a","title":"t","messages":null}')
        self._write("updnull.json",
                    '{"id":"a","title":"t","updated":null,"messages":[]}')
        self._write("updint.json",
                    '{"id":"a","title":5,"updated":7,"messages":[]}')
        sessions = chats.list_sessions()  # must not raise (nor sort-crash)
        self.assertIn(good["id"], [s["id"] for s in sessions])
        for meta in sessions:
            self.assertIsInstance(meta["updated"], str)
            self.assertIsInstance(meta["title"], str)
            self.assertIsInstance(meta["messages"], int)
            chats.load(meta["id"])  # the panel loads every listed meta
        for label in list(self.MALFORMED) + ["msgsnull", "updnull", "updint"]:
            chats.load(label)  # never raises
        loaded = chats.load(good["id"])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["title"], "good")

    # a chat whose "messages" value is not a list holds no messages at all: our loader returns the (empty) session
    # (pinned by test_stage1_review_pskui; the panel skips history-less chats), where the original returned None
    LENIENT = ("msgsdict",)

    def test_structurally_invalid_chats_load_as_none(self):
        for label, text in self.MALFORMED.items():
            if label in self.LENIENT:
                continue
            self._write(f"{label}.json", text)
            self.assertIsNone(chats.load(label), label)

    def test_a_non_list_messages_value_loads_as_an_empty_chat(self):
        for label in self.LENIENT:
            self._write(f"{label}.json", self.MALFORMED[label])
            loaded = chats.load(label)
            self.assertIsNotNone(loaded, label)
            self.assertEqual(loaded["history"], [], label)

    def test_a_chat_without_an_id_loads_under_its_filename(self):
        self._write("noid.json", '{"title":"x","messages":[]}')
        loaded = chats.load("noid")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["id"], "noid")
        self.assertEqual([s["id"] for s in chats.list_sessions()], ["noid"])

    def test_a_copied_file_does_not_shadow_the_original(self):
        session = chats.new_session("orig")
        chats.save(session)
        shutil.copy(os.path.join(self.sandbox, f"{session['id']}.json"),
                    os.path.join(self.sandbox, "copy.json"))
        ids = [s["id"] for s in chats.list_sessions()]
        self.assertEqual(sorted(ids), sorted([session["id"], "copy"]))
        self.assertEqual(chats.load("copy")["id"], "copy")
        self.assertEqual(chats.load(session["id"])["id"], session["id"])

    def test_a_hostile_id_in_the_file_cannot_escape_the_directory(self):
        self._write("a.json",
                    '{"id":"../../evil","title":"T","updated":"2026","messages":[]}')
        self.assertEqual([s["id"] for s in chats.list_sessions()], ["a"])
        self.assertEqual(chats.load("a")["id"], "a")

    def test_valid_roundtrip_is_unchanged(self):
        session = chats.new_session("keep")
        chats.save(session)
        loaded = chats.load(session["id"])
        self.assertEqual(loaded["id"], session["id"])
        self.assertEqual(loaded["title"], "keep")


# ---------------------------------------------------------------------------
# persistence-08: the settings file


class SettingsFileTest(_IsolatedSettings):
    GOOD = {"profiles": [{"name": "Work", "provider": "anthropic",
                          "api_key": "sk-REAL", "model": "m"}],
            "active_profile": "Work", "prefs": {"BridgeAutoStart": True}}

    def _put(self, raw):
        with open(self.settings, "wb") as f:
            f.write(raw)
        return raw

    def _bytes(self):
        with open(self.settings, "rb") as f:
            return f.read()

    def _backups(self):
        return sorted(glob.glob(self.settings + ".corrupt-*"))

    def test_utf8_bom_file_loads_and_is_not_overwritten(self):
        raw = self._put(b"\xef\xbb\xbf" + json.dumps(self.GOOD).encode())
        profiles = params.load_profiles()
        self.assertEqual([(p.name, p.api_key) for p in profiles],
                         [("Work", "sk-REAL")])
        self.assertEqual(params.active_profile_name(), "Work")
        self.assertEqual(self._bytes(), raw)
        self.assertEqual(self._backups(), [])

    def test_a_write_after_a_bom_file_keeps_the_profiles(self):
        self._put(b"\xef\xbb\xbf" + json.dumps(self.GOOD).encode())
        params.set_bool("PanelAutoOpen", False)
        self.assertEqual(params.load_profiles()[0].api_key, "sk-REAL")
        self.assertFalse(params.get_bool("PanelAutoOpen", True))

    def test_truncated_file_is_backed_up_before_the_defaults_replace_it(self):
        raw = self._put(json.dumps(self.GOOD).encode()[:-10])
        profiles = params.load_profiles()
        self.assertEqual([(p.name, p.provider) for p in profiles], [("OriginShape", "originshape")])   # the new-install seed
        backups = self._backups()
        self.assertEqual(len(backups), 1, backups)
        with open(backups[0], "rb") as f:
            self.assertEqual(f.read(), raw)
        # and the live file is valid again
        self.assertEqual(params.load_profiles()[0].name, "OriginShape")
        self.assertEqual(params.active_profile_name(), "OriginShape")
        self.assertEqual(len(self._backups()), 1)

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_corrupt_backup_is_owner_only_like_the_live_file(self):
        # the unparseable file still holds the API keys in clear text
        self._put(json.dumps(self.GOOD).encode()[:-10])
        previous = os.umask(0o022)      # the usual default: files come out 0644
        try:
            params.set_bool("PanelAutoOpen", False)
        finally:
            os.umask(previous)
        backups = self._backups()
        self.assertEqual(len(backups), 1, backups)
        with open(backups[0], "rb") as f:
            self.assertIn(b"sk-REAL", f.read())
        self.assertEqual(stat.S_IMODE(os.stat(backups[0]).st_mode) & 0o077, 0)
        self.assertEqual(stat.S_IMODE(os.stat(self.settings).st_mode) & 0o077, 0)

    def test_repeated_reads_of_a_corrupt_file_make_no_backups(self):
        self._put(b"{not json")
        for _ in range(3):
            params.get_bool("PanelAutoOpen", True)
        self.assertEqual(self._backups(), [])

    def test_wrong_shapes_never_raise(self):
        shapes = [
            b"[]", b'"x"', b"null", b"5",
            b'{"profiles": null}',
            b'{"profiles": {}, "prefs": {}}',
            b'{"profiles": [1, "x", null]}',
            b'{"prefs": null, "profiles": []}',
            b'{"prefs": [], "profiles": []}',
            b'{"profiles": [{"name":"A"}], "active_profile": 5, "prefs": {}}',
        ]
        for raw in shapes:
            with self.subTest(raw=raw):
                self._put(raw)
                params.get_bool("BridgeAutoStart", True)
                params.get_int("DockArea", 2)
                params.get_string("x", "d")
                params.active_profile_name()
                params.active_profile()
                self.assertGreaterEqual(len(params.load_profiles()), 1)

    def test_wrong_shaped_file_keeps_what_is_valid_and_is_backed_up(self):
        raw = self._put(json.dumps({"profiles": self.GOOD["profiles"],
                                    "prefs": None}).encode())
        self.assertEqual(params.load_profiles()[0].api_key, "sk-REAL")
        self.assertFalse(params.get_bool("BridgeAutoStart", True))  # the addon default is off
        params.set_bool("PanelAutoOpen", False)
        self.assertEqual(params.load_profiles()[0].api_key, "sk-REAL")
        backups = self._backups()
        self.assertEqual(len(backups), 1)
        with open(backups[0], "rb") as f:
            self.assertEqual(f.read(), raw)

    def test_unreadable_file_is_never_replaced(self):
        raw = self._put(json.dumps(self.GOOD).encode())
        settings = self.settings
        real_open = open

        def flaky_open(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if os.fspath(path) == settings and "r" in mode and "+" not in mode:
                raise PermissionError(13, "sharing violation")
            return real_open(path, *args, **kwargs)

        with mock.patch.object(params, "open", flaky_open, create=True), \
                mock.patch("time.sleep", lambda s: None):
            profiles = params.load_profiles()
            self.assertGreaterEqual(len(profiles), 1)  # an in-memory seed
            with self.assertRaises(OSError):
                params.set_bool("PanelAutoOpen", False)
            self.assertFalse(params.get_bool("BridgeAutoStart", True))  # the addon default is off
        self.assertEqual(self._bytes(), raw)
        self.assertEqual(self._backups(), [])
        self.assertEqual(params.load_profiles()[0].api_key, "sk-REAL")

    def test_a_transient_read_error_is_retried(self):
        raw = self._put(json.dumps(self.GOOD).encode())
        settings = self.settings
        real_open = open
        failures = {"left": 1}

        def once_flaky(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if (os.fspath(path) == settings and "r" in mode
                    and failures["left"]):
                failures["left"] -= 1
                raise PermissionError(13, "sharing violation")
            return real_open(path, *args, **kwargs)

        with mock.patch.object(params, "open", once_flaky, create=True), \
                mock.patch("time.sleep", lambda s: None):
            profiles = params.load_profiles()
        self.assertEqual([p.api_key for p in profiles], ["sk-REAL"])
        self.assertEqual(self._bytes(), raw)

    def test_normal_roundtrip_still_works(self):
        params.save_profiles([Profile(name="a", api_key="k")])
        params.set_int("DockArea", 4)
        self.assertEqual(params.load_profiles()[0].api_key, "k")
        self.assertEqual(params.get_int("DockArea", 2), 4)
        self.assertEqual(self._backups(), [])


class SettingsWipeRaceTest(_IsolatedSettings):
    """P0 #5 (9.2.5): a read that FAILS and a later read that SUCCEEDS must never let the defaults (or a seed) replace
    the file. Writers read strictly; the dialog learns when its profiles did not come from the file; a profile change
    keeps the previous file as ``.bak``."""

    GOOD, _put, _bytes, _backups = SettingsFileTest.GOOD, SettingsFileTest._put, SettingsFileTest._bytes, SettingsFileTest._backups

    def _flaky(self, failures, partial=False):
        settings, real_open, left = self.settings, open, {"n": failures}

        def fake(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if os.fspath(path) == settings and "r" in mode and "+" not in mode and left["n"] > 0:
                left["n"] -= 1
                if partial:                                # a half-synced file: the first half of the bytes
                    with real_open(path, *args, **kwargs) as f:
                        text = f.read()
                    return io.StringIO(text[: len(text) // 2]) if "b" not in mode else io.BytesIO(text[: len(text) // 2])
                raise PermissionError(13, "sharing violation")
            return real_open(path, *args, **kwargs)
        return fake

    def _assert_intact(self, raw):
        self.assertEqual(self._bytes(), raw)
        self.assertEqual([(p.name, p.api_key) for p in params.load_profiles()], [("Work", "sk-REAL")])

    def test_a_failed_read_then_a_good_one_never_seeds_over_the_profiles(self):
        for failures in (3, 6):          # 3 = the first whole read; 6 = the seed transaction's own read too
            with self.subTest(failures=failures):
                raw = self._put(json.dumps(self.GOOD).encode())
                with mock.patch.object(params, "open", self._flaky(failures), create=True), \
                        mock.patch("time.sleep", lambda s: None):
                    params.load_profiles()
                self._assert_intact(raw)

    def test_a_pref_write_after_a_failed_read_raises_and_writes_nothing(self):
        for write in (lambda: params.set_bool("PanelAutoOpen", True), lambda: params.set_int("DockArea", 1),
                      lambda: params.set_string("x", "y"), lambda: params.set_active_profile("Work"),
                      lambda: params.update_model_caps("k|u|m", lambda e: {"learned": {"tools": "text"}}),
                      lambda: params.add_net_grant("fetch_url", "a.example.com"),
                      lambda: params.save_profiles([Profile(name="Other")]),
                      lambda: params.ensure_account_profile(activate=True)):
            raw = self._put(json.dumps(self.GOOD).encode())
            with mock.patch.object(params, "open", self._flaky(3), create=True), \
                    mock.patch("time.sleep", lambda s: None):
                with self.assertRaises(params.SettingsUnavailable):
                    write()
            self._assert_intact(raw)

    def test_settings_unavailable_is_an_oserror_with_a_safe_message(self):
        self._put(json.dumps(self.GOOD).encode())
        with mock.patch.object(params, "open", self._flaky(3), create=True), mock.patch("time.sleep", lambda s: None):
            with self.assertRaises(OSError) as caught:
                params.set_bool("PanelAutoOpen", True)
        self.assertIsInstance(caught.exception, params.SettingsUnavailable)
        self.assertIn("nothing was changed", str(caught.exception))
        self.assertNotIn("sk-REAL", str(caught.exception))

    def test_a_half_read_file_is_read_again_not_replaced(self):
        raw = self._put(json.dumps(self.GOOD).encode())
        with mock.patch.object(params, "open", self._flaky(1, partial=True), create=True):
            self.assertEqual([p.api_key for p in params.load_profiles()], ["sk-REAL"])
        self._assert_intact(raw)
        with mock.patch.object(params, "open", self._flaky(1, partial=True), create=True):
            params.set_string("x", "y")
        self.assertEqual(params.load_profiles()[0].api_key, "sk-REAL")
        self.assertEqual(params.get_string("x", ""), "y")
        self.assertEqual(self._backups(), [])

    def test_the_dialog_is_told_when_its_profiles_did_not_come_from_the_file(self):
        self._put(json.dumps(self.GOOD).encode())
        with mock.patch.object(params, "open", self._flaky(99), create=True), \
                mock.patch("time.sleep", lambda s: None):
            profiles, from_file = params.load_profiles_checked()
        self.assertFalse(from_file)
        self.assertEqual([p.provider for p in profiles], ["originshape"])     # an in-memory starter only
        self.assertEqual(params.load_profiles_checked(), (params.load_profiles(), True))

    def test_a_missing_file_is_seeded_once_and_reads_as_from_file(self):
        profiles, from_file = params.load_profiles_checked()
        self.assertTrue(from_file)
        self.assertEqual([(p.name, p.provider, p.base_url, p.api_key) for p in profiles],
                         [("OriginShape", "originshape", "", "")])
        self.assertTrue(os.path.exists(self.settings))
        self.assertEqual(params.load_profiles_checked(), (profiles, True))

    def test_migration_runs_only_for_a_missing_file(self):
        raw = self._put(json.dumps(self.GOOD).encode())
        params._MIGRATED = False
        with mock.patch.object(params, "_migrate_from_user_parameters") as migrate, \
                mock.patch.object(params, "open", self._flaky(3), create=True), mock.patch("time.sleep", lambda s: None):
            params.get_bool("PanelAutoOpen", False)      # a read that failed is not a first run
        migrate.assert_not_called()
        self.assertEqual(self._bytes(), raw)
        os.unlink(self.settings)
        with mock.patch.object(params, "_migrate_from_user_parameters") as migrate:
            params.get_bool("PanelAutoOpen", False)
        migrate.assert_called_once()

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_a_profile_change_keeps_the_previous_file_as_bak(self):
        raw = self._put(json.dumps(self.GOOD).encode())
        params.save_profiles([Profile(name="Other", api_key="sk-OTHER")])
        with open(self.settings + ".bak", "rb") as f:
            self.assertEqual(f.read(), raw)
        self.assertEqual(stat.S_IMODE(os.stat(self.settings + ".bak").st_mode) & 0o077, 0)
        params.set_bool("PanelAutoOpen", True)                    # a pref write does not churn the backup
        with open(self.settings + ".bak", "rb") as f:
            self.assertEqual(f.read(), raw)


# ---------------------------------------------------------------------------
# persistence-09: attachment encodings


class ReadTextEncodingTest(unittest.TestCase):
    def _path(self, raw, suffix=".txt"):
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        self.addCleanup(os.unlink, path)
        return path

    def _read(self, raw, **kwargs):
        return attachments.read_text(self._path(raw), **kwargs)

    TEXT = "angle 90° tol ±0.1 µm Ø12"

    def test_utf8(self):
        self.assertEqual(self._read(self.TEXT.encode("utf-8")), (self.TEXT, False))

    def test_utf8_bom_is_stripped(self):
        content, _ = self._read(b"\xef\xbb\xbf" + self.TEXT.encode("utf-8"))
        self.assertEqual(content, self.TEXT)

    def test_cp1252_keeps_the_symbols(self):
        content, truncated = self._read(self.TEXT.encode("cp1252"))
        self.assertEqual(content, self.TEXT)
        self.assertFalse(truncated)

    def test_utf16_with_bom(self):
        for label, raw in (
                ("le", b"\xff\xfe" + self.TEXT.encode("utf-16-le")),
                ("be", b"\xfe\xff" + self.TEXT.encode("utf-16-be"))):
            with self.subTest(label):
                self.assertEqual(self._read(raw), (self.TEXT, False))

    def test_utf16_without_bom(self):
        plain = "width,height\n10,20\n"
        self.assertEqual(self._read(plain.encode("utf-16-le")), (plain, False))
        self.assertEqual(self._read(plain.encode("utf-16-be")), (plain, False))

    def test_crlf_and_cr_become_newlines(self):
        content, _ = self._read(b"a\r\nb\rc\n")
        self.assertEqual(content, "a\nb\nc\n")

    def test_truncation_counts_characters(self):
        content, truncated = self._read("é".encode("utf-8") * 50, limit=10)
        self.assertEqual((content, truncated), ("é" * 10, True))

    def test_a_multibyte_character_cut_by_the_read_limit_is_not_mojibake(self):
        raw = ("a" + "é" * 100).encode("utf-8")  # the byte window ends mid-é
        content, truncated = self._read(raw, limit=10)
        self.assertEqual(content, "a" + "é" * 9)
        self.assertTrue(truncated)

    def test_undecodable_bytes_fall_back_without_raising(self):
        content, _ = self._read(b"ok \x81\x8d\x8f\x90 end")
        self.assertTrue(content.startswith("ok "))
        self.assertTrue(content.endswith(" end"))

    def test_binary_file_is_not_inlined(self):
        raw = b"\x00\x01\x02binary\x00\xff\xfe\x00" * 20
        path = self._path(raw, ".log")
        self.assertEqual(attachments.read_text(path), ("", False))
        block = attachments.prompt_block(attachments.make(path))
        self.assertNotIn("\x00", block)
        self.assertIn("binary", block)
        self.assertNotIn("empty or unreadable", block)

    def test_empty_file_keeps_its_message(self):
        block = attachments.prompt_block(attachments.make(self._path(b"")))
        self.assertIn("empty or unreadable", block)

    def test_cp1252_csv_reaches_the_prompt_intact(self):
        path = self._path("part,tol\nshaft,±0.1 µm\nØ12,90°\n".encode("cp1252"),
                          ".csv")
        block = attachments.prompt_block(attachments.make(path))
        self.assertIn("±0.1 µm", block)
        self.assertIn("Ø12,90°", block)
        self.assertNotIn("�", block)


if __name__ == "__main__":
    unittest.main()
