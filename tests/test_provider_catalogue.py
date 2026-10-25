"""Tests for the provider configuration: the preset catalogue, profile fields and Advanced overrides, background
jobs, the connection test, and the settings-dialog selection bug fix (clicking a profile showed only its name)."""

import base64
import json
import os
import struct
import sys
import threading
import time
import unittest
import uuid
import zlib
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import models as _models  # noqa: E402
from oscar import params as _params  # noqa: E402
from oscar import provider_check, settings_model  # noqa: E402
from oscar.params import Profile  # noqa: E402


class _FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self, decode_unicode=True):
        yield from self._lines


class SettingsSelectionRegressionTest(unittest.TestCase):
    """The bug: clicking a profile in the list repopulated only the name;
    model, key and base URL came up empty (editor state was stashed into
    the WRONG profile on row change). Requires a QApplication."""

    @classmethod
    def setUpClass(cls):
        try:
            from PySide import QtWidgets

        except ImportError:
            raise unittest.SkipTest("PySide unavailable")
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            import sys as _sys

            argv = getattr(_sys, "argv", [])[:1]
            cls.app = QtWidgets.QApplication(argv)
            cls._owns_app = True

    def test_selecting_profile_populates_all_fields(self):
        profile = Profile(name="Primary", provider="openai_compat",
                          base_url="http://localhost:8931/v1",
                          model="mock-model", api_key="key-123")
        _params.save_profiles([profile])
        _params.set_active_profile("Primary")
        from oscar.settings_ui import SettingsDialog

        dialog = SettingsDialog()
        dialog._load_editor(dialog.profiles[0], 0)
        # select a DIFFERENT row (add one first) then come back — the
        # exact interaction that used to blank the fields
        dialog._add()
        dialog.list.setCurrentRow(0)
        dialog._load_editor(dialog.profiles[0], 0)
        self.assertEqual(dialog.name_edit.text(), "Primary")
        self.assertEqual(dialog.base_edit.text(), "http://localhost:8931/v1")
        self.assertEqual(dialog.model_combo.currentText(), "mock-model")
        self.assertEqual(dialog.key_edit.text(), "key-123")
        # and switching away did not corrupt the original profile
        self.assertEqual(dialog.profiles[0].api_key, "key-123")
        self.assertEqual(dialog.profiles[0].model, "mock-model")


class ParamsAdditionsTest(unittest.TestCase):
    """``Profile`` fields, ``effective_limits`` and ``update_profile_tokens``."""

    def setUp(self):
        self._backup = _params.backup_state()
        self.addCleanup(_params.restore_state, self._backup)

    def test_new_profile_fields_default_and_roundtrip(self):
        p = Profile(name="r1")
        self.assertEqual((p.reasoning, p.caps_override), ("auto", {}))
        q = Profile(name="r2", reasoning="high", caps_override={"tools": "text", "context_window": 8192})
        _params.save_profiles([p, q])
        loaded = {x.name: x for x in _params.load_profiles()}
        self.assertEqual((loaded["r2"].reasoning, loaded["r2"].caps_override), ("high", {"tools": "text", "context_window": 8192}))
        self.assertEqual(loaded["r1"].reasoning, "auto")

    def test_the_overrides_dict_is_not_shared_between_profiles(self):
        a, b = Profile(name="a"), Profile(name="b")
        a.caps_override["x"] = 1
        self.assertEqual(b.caps_override, {})

    def test_a_hand_edited_settings_file_cannot_break_profiles(self):
        self.assertEqual(Profile(name="x", reasoning="turbo").reasoning, "auto")
        self.assertEqual(Profile(name="x", caps_override="oops").caps_override, {})
        self.assertEqual(Profile(name="x", reasoning="off").reasoning, "off")

    def test_old_settings_without_the_new_keys_and_unknown_future_keys_load(self):
        _params.save_profiles([Profile(name="old")])
        state = json.loads(_params.backup_state())
        for item in state["profiles"]:
            item.pop("reasoning", None)
            item.pop("caps_override", None)
            item["from_the_future"] = 1
        state.pop("model_caps", None)
        _params.restore_state(json.dumps(state))
        profile = _params.load_profiles()[0]
        self.assertEqual((profile.reasoning, profile.caps_override), ("auto", {}))
        _params.set_model_caps("k|u|m", {"learned": {"vision": False}})        # the key reappears on first use
        self.assertEqual(_params.get_model_caps("k|u|m"), {"learned": {"vision": False}})
        _params.set_model_caps("k|u|m", {})

    def test_effective_limits_defaults_to_the_predefined_limits(self):
        p = Profile(name="l", provider="anthropic", base_url="https://api.anthropic.com", model="claude-sonnet-4-5")
        self.assertEqual(_params.effective_limits(p), _params.predefined_limits(p.provider, p.base_url, p.model))

    def test_effective_limits_applies_valid_overrides(self):
        p = Profile(name="l", caps_override={"max_output": 2048, "temperature": 0.7})
        self.assertEqual(_params.effective_limits(p), (0.7, 2048))
        p = Profile(name="l", caps_override={"max_output": 100.0, "temperature": 0})
        self.assertEqual(_params.effective_limits(p), (0.0, 100))

    def test_effective_limits_ignores_invalid_overrides(self):
        base = _params.predefined_limits("anthropic", "https://api.anthropic.com", "claude-sonnet-4-5")
        for override in ({"max_output": 0}, {"max_output": -5}, {"max_output": "big"}, {"max_output": True},
                         {"temperature": 2.5}, {"temperature": -0.1}, {"temperature": "hot"}, {"temperature": False}, {}):
            with self.subTest(override=override):
                p = Profile(name="l", provider="anthropic", base_url="https://api.anthropic.com",
                            model="claude-sonnet-4-5", caps_override=override)
                self.assertEqual(_params.effective_limits(p), base)

    def test_effective_limits_accepts_bare_objects(self):
        class Bare:
            pass

        temperature, max_tokens = _params.effective_limits(Bare())
        self.assertIsInstance(max_tokens, int)
        self.assertGreater(max_tokens, 0)


# ---------------------------------------------------------------------------------------------------------------
# Advanced overrides, background jobs, the connection test, the dialog that must not freeze


def _unique(**extra):
    return Profile(name="adv", provider="openai_compat", base_url="https://llm.example.com/v1",
                   model=f"m-{uuid.uuid4().hex[:8]}", **extra)


class AdvancedOverridesTest(unittest.TestCase):
    def test_defaults_read_as_no_override(self):
        self.assertEqual(settings_model.read_advanced(_unique()), {
            "max_output": 0, "temperature": None, "reasoning": "auto", "context_window": 0, "tools": "auto",
            "ollama_native": True})

    def test_apply_then_read_round_trips(self):
        p = _unique()
        values = {"max_output": 12000, "temperature": 0.55, "reasoning": "high", "context_window": 8192, "tools": "text",
                  "ollama_native": False}
        self.assertEqual(settings_model.apply_advanced(p, values), [])
        self.assertEqual(p.caps_override, {"max_output": 12000, "temperature": 0.55, "context_window": 8192, "tools": "text",
                                           "ollama_native": False})
        self.assertEqual(p.reasoning, "high")
        self.assertEqual(settings_model.read_advanced(p), values)

    def test_automatic_values_remove_the_override_and_foreign_keys_survive(self):
        p = _unique(caps_override={"max_output": 9000, "vision": False, "context_window": 4096, "tools": "native",
                                   "temperature": 0.9, "ollama_native": False})
        settings_model.apply_advanced(p, {"max_output": 0, "temperature": None, "context_window": 0, "tools": "auto",
                                          "ollama_native": True, "reasoning": "auto"})
        self.assertEqual(p.caps_override, {"vision": False})

    def test_only_the_fields_present_are_touched(self):
        p = _unique(caps_override={"max_output": 9000}, reasoning="low")
        settings_model.apply_advanced(p, {"context_window": 5000})
        self.assertEqual((p.caps_override, p.reasoning), ({"max_output": 9000, "context_window": 5000}, "low"))

    def test_invalid_values_are_reported_and_not_applied(self):
        p = _unique(caps_override={"max_output": 9000, "context_window": 4096}, reasoning="low")
        problems = settings_model.apply_advanced(p, {"max_output": 10, "context_window": 99999999999, "temperature": 5.0,
                                                     "reasoning": "extreme", "tools": "maybe"})
        self.assertEqual(len(problems), 5)
        self.assertEqual((p.caps_override, p.reasoning), ({"max_output": 9000, "context_window": 4096}, "low"))
        self.assertEqual(settings_model.validate_advanced({"max_output": 256, "context_window": 1024, "temperature": 2.0}), [])
        self.assertTrue(any("max output" in x for x in problems))

    def test_garbage_in_the_stored_override_reads_as_automatic(self):
        p = _unique(caps_override={"max_output": "lots", "context_window": True, "temperature": 99, "tools": "none"})
        self.assertEqual(settings_model.read_advanced(p)["max_output"], 0)
        self.assertEqual(settings_model.read_advanced(p)["context_window"], 0)
        self.assertIsNone(settings_model.read_advanced(p)["temperature"])
        self.assertEqual(settings_model.read_advanced(p)["tools"], "auto")
        broken = _unique()
        broken.caps_override = "junk"
        self.assertEqual(settings_model.apply_advanced(broken, {"max_output": 4096}), [])
        self.assertEqual(broken.caps_override, {"max_output": 4096})

    def test_the_overrides_survive_the_per_turn_recompute_and_the_caps_resolution(self):
        p = _unique()
        settings_model.apply_advanced(p, {"max_output": 12345, "temperature": 0.7, "context_window": 9000, "tools": "text"})
        _params.save_profiles([p])
        stored = _params.load_profiles()[0]
        self.assertEqual(_params.effective_limits(stored), (0.7, 12345))              # what AgentWorker applies each turn
        caps = _models.resolve_caps(stored)
        self.assertEqual((caps.context_window, caps.tools, caps.source), (9000, "text", "override"))

    def test_describe_effective(self):
        p = _unique(reasoning="low")
        settings_model.apply_advanced(p, {"max_output": 5000, "context_window": 9000, "tools": "text", "temperature": 0.4})
        text = settings_model.describe_effective(p)
        for piece in ("context 9000", "output up to 5000", "temperature 0.4", "tools text", "images unknown", "reasoning low",
                      "source: override"):
            self.assertIn(piece, text)
        ollama = Profile(name="o", provider="openai_compat", base_url="http://localhost:11434/v1", model=f"o-{uuid.uuid4().hex[:6]}")
        self.assertIn("context 16384", settings_model.describe_effective(ollama))           # the window the native provider requests
        self.addCleanup(_params.set_model_caps, _models.caps_key(ollama), {})
        with mock.patch.object(_models, "resolve_caps", side_effect=RuntimeError("boom")):
            self.assertEqual(settings_model.describe_effective(p), "effective settings unavailable (RuntimeError)")

    def test_ollama_native_checkbox_value(self):
        p = Profile(name="o", provider="openai_compat", base_url="http://localhost:11434/v1", model="m")
        settings_model.apply_advanced(p, {"ollama_native": False})
        self.assertFalse(_models.ollama_native_wanted(p))
        settings_model.apply_advanced(p, {"ollama_native": True})
        self.assertTrue(_models.ollama_native_wanted(p))


class BackgroundJobTest(unittest.TestCase):
    def collect(self):
        got, event = [], threading.Event()

        def deliver(result):
            got.append((result, threading.current_thread().name))
            event.set()

        return got, event, deliver

    def test_the_result_is_delivered_once_from_the_worker_thread(self):
        got, event, deliver = self.collect()
        job = settings_model.run_in_background(lambda cancelled: 41 + 1, deliver, name="oscar-test-job")
        self.assertTrue(event.wait(5))
        job.join(5)
        self.assertEqual(got[0][0], settings_model.JobResult(True, 42))
        self.assertEqual(got[0][1], "oscar-test-job")
        self.assertNotEqual(got[0][1], threading.current_thread().name)
        self.assertEqual(len(got), 1)

    def test_an_exception_becomes_an_error_sentence(self):
        got, event, deliver = self.collect()
        settings_model.run_in_background(lambda c: 1 / 0, deliver)
        self.assertTrue(event.wait(5))
        self.assertEqual((got[0][0].ok, got[0][0].error), (False, "ZeroDivisionError: division by zero"))

    def test_a_cancelled_job_delivers_nothing_and_sees_the_flag(self):
        got, event, deliver = self.collect()
        started, release = threading.Event(), threading.Event()
        seen = []

        def work(cancelled):
            started.set()
            release.wait(5)
            seen.append(cancelled.is_set())
            return "late"

        job = settings_model.run_in_background(work, deliver)
        self.assertTrue(started.wait(5))
        job.cancel()
        release.set()
        self.assertTrue(job.join(5))
        self.assertEqual((got, seen), ([], [True]))
        self.assertFalse(job.running)

    def test_a_failing_deliver_callback_does_not_kill_the_thread_with_a_traceback(self):
        def deliver(result):
            raise RuntimeError("dialog gone")

        job = settings_model.run_in_background(lambda c: 1, deliver)
        self.assertTrue(job.join(5))


# --- the connection test ----------------------------------------------------------------------------------------


class FakeProvider:
    """``chat`` answers from ``behaviour(profile, system, messages, tools) -> events``; ``vision_enabled`` can flip."""

    name = "fake"

    def __init__(self, profile, behaviour, log):
        self.profile, self.behaviour, self.log = profile, behaviour, log
        self.vision_enabled = True

    def chat(self, system, messages, tools, should_stop=None, **opts):
        self.log.append({"mode": (self.profile.caps_override or {}).get("tools", "auto"), "tools": [t.name for t in tools],
                         "images": bool(messages[-1].images), "system": system, "tool_choice": opts.get("tool_choice")})
        if should_stop is not None and should_stop():
            from oscar.providers import StopRequested

            raise StopRequested()
        yield from self.behaviour(self, system, messages, tools)


def echo_call():
    from oscar.providers import ToolCall

    return ("tool_calls", [ToolCall(id="c1", name="echo", arguments={"text": "oscar-ok"})])


class ConnectionTestTest(unittest.TestCase):
    def setUp(self):
        self.log = []
        self.profile = _unique(api_key="k")
        self.addCleanup(lambda: (_models.learn.__self__ if False else None))
        key = _models.caps_key(self.profile)
        self.addCleanup(lambda: (_models._LEARNED.pop(key, None), _params.set_model_caps(key, {})))
        patcher = mock.patch.object(_models, "probe_metadata", return_value={})
        self.probe = patcher.start()
        self.addCleanup(patcher.stop)

    def run_test(self, behaviour, profile=None, **kw):
        profile = profile or self.profile
        return provider_check.run_connection_test(profile, make=lambda p: FakeProvider(p, behaviour, self.log), **kw)

    @staticmethod
    def talk(provider, system, messages, tools, *, native=True, colour="That is red.", tool_text=None):
        """The behaviour of a capable model unless told otherwise."""
        if tools:                                       # the tool step
            mode = (provider.profile.caps_override or {}).get("tools", "auto")
            if (mode == "text" and tool_text is not False) or (mode != "text" and native):
                yield echo_call()
            else:
                yield ("text", "I would call echo with oscar-ok.")
                yield ("done", "stop")
        elif messages[-1].images:
            yield ("text", colour)
            yield ("done", "stop")
        else:
            yield ("text", "ready")
            yield ("done", "stop")

    def test_a_capable_model_passes_every_step_and_vision_is_remembered(self):
        report = self.run_test(self.talk)
        self.assertTrue(report.ok)
        self.assertEqual((report.tools, report.vision, report.reply), ("native", True, "ready"))
        self.assertEqual([(l.step, l.ok) for l in report.lines], [("connect", True), ("tools", True), ("vision", True)])
        self.assertIs(_models.learned_fields(self.profile)["vision"], True)
        self.assertNotIn("tools", _models.learned_fields(self.profile))              # native needs no note
        self.assertEqual(report.learned, {"vision": True})
        modes = [e["mode"] for e in self.log]
        self.assertEqual(modes[:2], ["auto", "native"])                                  # the probe forces native explicitly
        self.assertEqual([e["tool_choice"] for e in self.log if e["tools"]], ["required"])   # a lazy-but-capable model is not demoted

    def test_the_summary_has_one_marked_line_per_step(self):
        summary = self.run_test(self.talk).summary().splitlines()
        self.assertEqual(len(summary), 3)
        self.assertTrue(summary[0].startswith("\u2713 fake / "))
        self.assertTrue(all(line.startswith("\u2713") for line in summary))

    def test_a_model_that_ignores_native_tools_but_follows_the_text_protocol_is_switched_to_text(self):
        report = self.run_test(lambda *a: self.talk(*a, native=False))
        self.assertTrue(report.ok)
        self.assertEqual(report.tools, "text")
        self.assertEqual(_models.learned_fields(self.profile)["tools"], "text")
        self.assertEqual(report.learned["tools"], "text")
        self.assertIn("text mode", [l for l in report.lines if l.step == "tools"][0].text)
        self.assertEqual([e["mode"] for e in self.log if e["tools"]], ["native", "text"])

    def test_a_model_that_cannot_call_tools_at_all_fails_with_advice(self):
        report = self.run_test(lambda *a: self.talk(*a, native=False, tool_text=False))
        self.assertFalse(report.ok)
        self.assertEqual(report.tools, "")
        line = [l for l in report.lines if l.step == "tools"][0]
        self.assertFalse(line.ok)
        self.assertIn("choose another model", line.text)
        self.assertNotIn("tools", _models.learned_fields(self.profile))

    def test_an_error_on_the_first_request_stops_the_test(self):
        def behaviour(provider, system, messages, tools):
            yield ("error", RuntimeError("HTTP 401 unauthorized (check the API key): bad key"))

        report = self.run_test(behaviour)
        self.assertFalse(report.ok)
        self.assertEqual(len(report.lines), 1)
        self.assertIn("401", report.error)
        self.assertEqual(len(self.log), 1)

    def test_a_provider_that_raises_is_reported_not_propagated(self):
        def behaviour(provider, system, messages, tools):
            raise ValueError("kaboom")
            yield

        report = self.run_test(behaviour)
        self.assertEqual((report.ok, report.error), (False, "ValueError: kaboom"))

    def test_a_model_that_refuses_the_image_is_remembered_as_having_no_vision(self):
        def behaviour(provider, system, messages, tools):
            if not tools and messages[-1].images:
                provider.vision_enabled = False                   # the real provider degraded after the 400 and retried
                yield ("text", "I only see text.")
                yield ("done", "stop")
            else:
                yield from self.talk(provider, system, messages, tools)

        report = self.run_test(behaviour)
        self.assertTrue(report.ok)
        self.assertIs(report.vision, False)
        self.assertIs(_models.learned_fields(self.profile)["vision"], False)             # a tiny valid PNG: unambiguous

    def test_an_image_that_is_accepted_but_not_confirmed_stays_unknown(self):
        report = self.run_test(lambda *a: self.talk(*a, colour="I see a picture."))
        self.assertTrue(report.ok)
        self.assertIsNone(report.vision)
        self.assertNotIn("vision", _models.learned_fields(self.profile))
        self.assertIsNone([l for l in report.lines if l.step == "vision"][0].ok)

    def test_a_known_text_only_model_skips_the_image_step(self):
        self.profile.caps_override = {"vision": False}
        report = self.run_test(self.talk)
        self.assertEqual(report.vision, False)
        self.assertEqual([e for e in self.log if e["images"]], [])

    def test_an_explicit_tool_mode_is_the_only_one_tried(self):
        self.profile.caps_override = {"tools": "text"}
        report = self.run_test(self.talk)
        self.assertEqual(report.tools, "text")
        self.assertEqual([e["mode"] for e in self.log if e["tools"]], ["text"])
        self.assertNotIn("tools", _models.learned_fields(self.profile))                  # the user decided; nothing to learn

    def test_native_working_again_replaces_an_old_text_mode_note(self):
        _models.learn(self.profile, tools="text")
        report = self.run_test(self.talk)
        self.assertEqual(report.tools, "native")
        self.assertEqual(_models.learned_fields(self.profile)["tools"], "native")

    def test_the_probe_step_lists_what_the_server_reported(self):
        self.probe.return_value = {"context_window": 9000, "tools": "native", "reasoning": "stream"}
        report = self.run_test(self.talk)
        self.assertEqual(report.lines[0].step, "metadata")
        self.assertIn("context_window 9000", report.lines[0].text)
        self.assertNotIn("reasoning", report.lines[0].text)
        self.probe.assert_called_once()
        self.assertTrue(self.probe.call_args.kwargs["refresh"])
        self.probe.reset_mock()
        self.run_test(self.talk, probe=False)
        self.probe.assert_not_called()

    def test_stop_is_propagated(self):
        from oscar.providers import StopRequested

        with self.assertRaises(StopRequested):
            self.run_test(self.talk, should_stop=lambda: True)

    def test_the_test_image_is_a_valid_png(self):
        data = provider_check.tiny_png()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        pos, chunks = 8, []
        while pos < len(data):
            length, kind = struct.unpack(">I4s", data[pos:pos + 8])
            body = data[pos + 8:pos + 8 + length]
            (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
            self.assertEqual(crc, zlib.crc32(kind + body) & 0xFFFFFFFF, kind)
            chunks.append((kind, body))
            pos += 12 + length
        self.assertEqual([k for k, _ in chunks], [b"IHDR", b"IDAT", b"IEND"])
        self.assertEqual(struct.unpack(">II", chunks[0][1][:8]), (64, 64))
        self.assertEqual(len(zlib.decompress(chunks[1][1])), 64 * (1 + 64 * 3))
        self.assertEqual(base64.b64decode(provider_check.tiny_png_b64()), data)


# --- the dialog -------------------------------------------------------------------------------------------------


class _DialogCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from PySide import QtWidgets
        except ImportError:
            raise unittest.SkipTest("PySide unavailable")
        cls.QtWidgets = QtWidgets
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication(getattr(sys, "argv", [])[:1])

    def setUp(self):
        self.backup = _params.backup_state()
        self.addCleanup(_params.restore_state, self.backup)
        self.profile = Profile(name="Primary", provider="openai_compat", base_url="https://llm.example.com/v1",
                               model=f"m-{uuid.uuid4().hex[:8]}", api_key="key-123")
        _params.save_profiles([self.profile])
        _params.set_active_profile("Primary")
        key = _models.caps_key(self.profile)
        self.addCleanup(lambda: (_models._LEARNED.pop(key, None), _params.set_model_caps(key, {})))

    def dialog(self):
        from oscar.settings_ui import SettingsDialog

        dialog = SettingsDialog()
        self.addCleanup(dialog._cancel_all_jobs)
        dialog._load_editor(dialog.profiles[0], 0)
        return dialog

    def wait(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.QtWidgets.QApplication.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        self.QtWidgets.QApplication.processEvents()
        return predicate()


class SettingsAdvancedDialogTest(_DialogCase):
    def test_the_advanced_section_starts_collapsed_and_toggles(self):
        d = self.dialog()
        self.assertTrue(d.adv_box.isHidden())
        d.adv_toggle.setChecked(True)
        self.assertFalse(d.adv_box.isHidden())
        d.adv_toggle.setChecked(False)
        self.assertTrue(d.adv_box.isHidden())

    def test_values_are_stashed_into_the_profile_and_survive_save_and_the_per_turn_recompute(self):
        d = self.dialog()
        d.adv_max_out.setValue(12000)
        d.adv_temp_check.setChecked(True)
        d.adv_temp.setValue(0.65)
        d.adv_reasoning.setCurrentIndex(d.adv_reasoning.findData("high"))
        d.adv_ctx.setValue(8192)
        d.adv_tools.setCurrentIndex(d.adv_tools.findData("text"))
        d._save()
        stored = _params.load_profiles()[0]
        self.assertEqual(stored.caps_override, {"max_output": 12000, "temperature": 0.65, "context_window": 8192, "tools": "text"})
        self.assertEqual(stored.reasoning, "high")
        self.assertEqual(_params.effective_limits(stored), (0.65, 12000))
        self.assertEqual(_models.resolve_caps(stored).context_window, 8192)
        reopened = self.dialog()
        self.assertEqual((reopened.adv_max_out.value(), reopened.adv_ctx.value(), reopened.adv_temp.value()), (12000, 8192, 0.65))
        self.assertTrue(reopened.adv_temp_check.isChecked())
        self.assertEqual((reopened.adv_reasoning.currentData(), reopened.adv_tools.currentData()), ("high", "text"))
        self.assertFalse(reopened.adv_box.isHidden())                                  # a profile with overrides opens expanded

    def test_automatic_everywhere_leaves_no_override(self):
        d = self.dialog()
        d._save()
        stored = _params.load_profiles()[0]
        self.assertEqual((stored.caps_override, stored.reasoning), ({}, "auto"))

    def test_the_predefined_display_shows_what_the_overrides_replace(self):
        d = self.dialog()
        d.adv_max_out.setValue(5000)
        self.assertEqual(d.max_tokens_spin.value(), 5000)
        d.adv_max_out.setValue(0)
        self.assertNotEqual(d.max_tokens_spin.value(), 5000)

    def test_the_effective_line_follows_the_widgets(self):
        d = self.dialog()
        d.adv_ctx.setValue(6000)
        self.assertIn("context 6000", d.adv_effective.text())
        d.adv_tools.setCurrentIndex(d.adv_tools.findData("text"))
        self.assertIn("tools text", d.adv_effective.text())

    def test_the_ollama_checkbox_maps_to_ollama_native(self):
        _params.save_profiles([Profile(name="Primary", provider="openai_compat", base_url="http://localhost:11434/v1", model="qwen2.5:7b")])
        d = self.dialog()
        self.assertTrue(d.adv_ollama.isChecked())
        d.adv_ollama.setChecked(False)
        d._save()
        self.assertEqual(_params.load_profiles()[0].caps_override, {"ollama_native": False})

    def test_changing_the_preset_clears_the_key_and_presets_are_picked_by_base_url(self):
        d = self.dialog()
        self.assertEqual(d.key_edit.text(), "key-123")
        ollama = next(i for i, p in enumerate(_params.PROVIDER_PRESETS) if p["label"].startswith("Ollama"))
        d.preset_combo.setCurrentIndex(d.preset_combo.findData(ollama))          # item data: the preset's index
        self.assertEqual(d.key_edit.text(), "")
        self.assertEqual(d.base_edit.text(), "http://localhost:11434/v1")
        _params.save_profiles([Profile(name="Primary", provider="openai_compat", base_url="https://openrouter.ai/api/v1", model="x/y")])
        d2 = self.dialog()
        self.assertEqual(_params.PROVIDER_PRESETS[d2.preset_combo.currentData()]["label"], "OpenRouter")
        self.assertEqual(d2._preset()["label"], "OpenRouter")


class ComingSoonDialogTest(_DialogCase):
    """An announced provider closes the provider list: visible, greyed out, never selectable. Nothing is announced since
    the OriginShape account became usable, so the mechanism is tested with a dummy entry. The list is the own-provider
    editor's: the account's preset is not in it (the account is chosen with its own radio button), so each item carries
    the index of its preset as data."""

    SOON = {"label": "Example Cloud", "provider": "example_cloud", "note": "Sign in with your Example Cloud account."}

    def announce(self):
        patcher = mock.patch.object(_params, "COMING_SOON_PROVIDERS", [dict(self.SOON)])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_nothing_is_announced_and_the_account_preset_is_not_offered(self):
        combo = self.dialog().preset_combo
        own = [(i, p) for i, p in enumerate(_params.PROVIDER_PRESETS) if not p.get("account")]
        self.assertEqual(_params.COMING_SOON_PROVIDERS, [])
        self.assertEqual(combo.count(), len(own))                                             # no separator either
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], [p["label"] for _i, p in own])
        self.assertEqual([combo.itemData(i) for i in range(combo.count())], [i for i, _p in own])
        self.assertNotIn("OriginShape (your account)", [combo.itemText(i) for i in range(combo.count())])
        self.assertTrue(combo.model().item(0).isEnabled())

    def test_it_is_listed_after_the_presets_and_disabled(self):
        from PySide import QtCore

        self.announce()
        own = [p["label"] for p in _params.PROVIDER_PRESETS if not p.get("account")]
        combo, presets = self.dialog().preset_combo, len(own)
        self.assertEqual(combo.count(), presets + 1 + len(_params.COMING_SOON_PROVIDERS))    # + a separator
        self.assertEqual([combo.itemText(i) for i in range(presets)], own)
        last = combo.count() - 1
        self.assertEqual(combo.itemText(last), "Example Cloud — Coming soon")
        self.assertFalse(combo.model().item(last).isEnabled())
        self.assertIn("Example Cloud account", combo.itemData(last, QtCore.Qt.ToolTipRole))

    def test_the_keyboard_cannot_reach_it(self):
        from PySide import QtCore, QtGui

        self.announce()
        d = self.dialog()
        combo, last_preset = d.preset_combo, len(_params.PROVIDER_PRESETS) - 1
        last_item = combo.findData(last_preset)
        combo.setCurrentIndex(last_item)
        for key in (QtCore.Qt.Key_Down, QtCore.Qt.Key_End, QtCore.Qt.Key_PageDown):
            self.QtWidgets.QApplication.sendEvent(combo, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, QtCore.Qt.NoModifier))
        self.assertEqual(combo.currentIndex(), last_item)
        self.assertEqual(d._preset(), _params.PROVIDER_PRESETS[last_preset])

    def test_a_save_keeps_a_usable_provider(self):
        self.announce()
        d = self.dialog()
        self.assertEqual(d.preset_combo.itemText(d.preset_combo.count() - 1), "Example Cloud — Coming soon")
        d.preset_combo.setCurrentIndex(d.preset_combo.count() - 1)      # only code can do this; the editor must not break
        d._save()
        self.assertEqual(_params.load_profiles()[0].provider, "openai_compat")


class SettingsThreadingTest(_DialogCase):
    """Nothing blocking runs on the GUI thread: every slow call returns control at once and reports through a signal."""

    def test_load_models_returns_at_once_and_fills_the_list_from_the_worker(self):
        release, threads = threading.Event(), []

        def slow_list(profile):
            threads.append(threading.current_thread())
            release.wait(5)
            return ["alpha", "beta", "gamma"], ""

        d = self.dialog()
        with mock.patch("oscar.providers.list_models", slow_list):
            started = time.monotonic()
            d._load_models()
            self.assertLess(time.monotonic() - started, 1.0)                      # control came back while the call is pending
            self.assertEqual(d.test_label.text(), "Loading models…")
            self.assertTrue(d._job_running("models"))
            release.set()
            self.assertTrue(self.wait(lambda: d.model_combo.count() > 1))
        self.assertNotEqual(threads[0], threading.main_thread())
        names = [d.model_combo.itemText(i) for i in range(d.model_combo.count())]
        self.assertEqual(sorted(names)[:3], ["alpha", "beta", "gamma"] if self.profile.model not in names else sorted(names)[:3])
        self.assertIn("models loaded", d.test_label.text())

    def test_the_users_model_stays_selected_when_the_list_does_not_contain_it(self):
        d = self.dialog()
        with mock.patch("oscar.providers.list_models", lambda profile: (["alpha", "beta"], "")):
            d._load_models()
            self.assertTrue(self.wait(lambda: not d._job_running("models")))
        self.assertEqual(d.model_combo.currentText(), self.profile.model)
        self.assertEqual([d.model_combo.itemText(i) for i in range(d.model_combo.count())], [self.profile.model, "alpha", "beta"])
        self.assertIn("kept", d.test_label.text())

    def test_a_failed_listing_shows_the_reason(self):
        d = self.dialog()
        with mock.patch("oscar.providers.list_models", lambda profile: ([], "HTTP 401: bad key")):
            d._load_models()
            self.assertTrue(self.wait(lambda: "Could not load" in d.test_label.text()))
        self.assertIn("HTTP 401: bad key", d.test_label.text())
        with mock.patch("oscar.providers.list_models", side_effect=RuntimeError("boom")):
            d._load_models()
            self.assertTrue(self.wait(lambda: "boom" in d.test_label.text()))

    def test_a_second_click_cancels_the_listing_and_its_late_result_is_dropped(self):
        release = threading.Event()
        d = self.dialog()
        before = d.model_combo.count()
        with mock.patch("oscar.providers.list_models", lambda profile: (release.wait(5), (["x1", "x2"], ""))[1]):
            d._load_models()
            d._load_models()                                                      # the same button cancels
            self.assertEqual(d.test_label.text(), "Model list cancelled.")
            release.set()
            for _ in range(20):
                self.QtWidgets.QApplication.processEvents()
                time.sleep(0.01)
        self.assertEqual(d.model_combo.count(), before)

    def test_the_connection_test_runs_on_a_worker_and_the_button_becomes_cancel(self):
        release, threads, stops = threading.Event(), [], []

        def fake_test(profile, *, should_stop=None, **kw):
            threads.append(threading.current_thread())
            release.wait(5)
            stops.append(should_stop())
            report = provider_check.ConnectionReport(ok=True, tools="native", vision=True, learned={"vision": True})
            report.lines = [provider_check.CheckLine("connect", True, "fake / m: \u201cready\u201d"),
                            provider_check.CheckLine("tools", True, "tool calling works natively")]
            return report

        d = self.dialog()
        with mock.patch.object(provider_check, "run_connection_test", fake_test):
            started = time.monotonic()
            d._test()
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertEqual(d.test_btn.text(), "Cancel test")
            self.assertTrue(d._job_running("test"))
            release.set()
            self.assertTrue(self.wait(lambda: not d._job_running("test")))
        self.assertNotEqual(threads[0], threading.main_thread())
        self.assertEqual(d.test_btn.text(), "Test connection")
        self.assertIn("\u2713 tool calling works natively", d.test_label.text())
        self.assertIn("remembered: vision = True", d.test_label.text())
        self.assertEqual(stops, [False])

    def test_a_failed_test_says_the_model_cannot_be_used(self):
        d = self.dialog()
        report = provider_check.ConnectionReport(ok=False, lines=[provider_check.CheckLine("tools", False, "no tool call came back")])
        with mock.patch.object(provider_check, "run_connection_test", lambda profile, **kw: report):
            d._test()
            self.assertTrue(self.wait(lambda: "cannot be used" in d.test_label.text()))
        self.assertIn("\u2717 no tool call came back", d.test_label.text())

    def test_cancelling_the_test_sets_the_flag_the_worker_polls_and_drops_its_result(self):
        seen, release = [], threading.Event()

        def fake_test(profile, *, should_stop=None, **kw):
            release.wait(5)
            seen.append(should_stop())
            return provider_check.ConnectionReport(ok=True, lines=[provider_check.CheckLine("connect", True, "late")])

        d = self.dialog()
        with mock.patch.object(provider_check, "run_connection_test", fake_test):
            d._test()
            d._test()
            self.assertEqual(d.test_label.text(), "Test cancelled.")
            self.assertEqual(d.test_btn.text(), "Test connection")
            release.set()
            self.assertTrue(self.wait(lambda: bool(seen)))
        for _ in range(10):
            self.QtWidgets.QApplication.processEvents()
        self.assertEqual(seen, [True])
        self.assertEqual(d.test_label.text(), "Test cancelled.")

    def test_a_crashing_test_is_reported_not_raised(self):
        d = self.dialog()
        with mock.patch.object(provider_check, "run_connection_test", side_effect=RuntimeError("kaboom")):
            d._test()
            self.assertTrue(self.wait(lambda: "kaboom" in d.test_label.text()))

    def test_the_test_works_on_a_copy_so_edits_while_it_runs_do_not_leak_into_it(self):
        release, seen = threading.Event(), []

        def fake_test(profile, **kw):
            release.wait(5)
            seen.append(profile.model)
            return provider_check.ConnectionReport(ok=True, lines=[])

        d = self.dialog()
        original = d.model_combo.currentText()
        with mock.patch.object(provider_check, "run_connection_test", fake_test):
            d._test()
            d.model_combo.setEditText("changed-while-running")
            release.set()
            self.assertTrue(self.wait(lambda: bool(seen)))
        self.assertEqual(seen, [original])


    def test_closing_the_dialog_cancels_every_job_and_late_results_are_harmless(self):
        release, flags = threading.Event(), []

        def fake_test(profile, *, should_stop=None, **kw):
            release.wait(5)
            flags.append(should_stop())
            return provider_check.ConnectionReport(ok=True, lines=[])

        d = self.dialog()
        with mock.patch.object(provider_check, "run_connection_test", fake_test):
            d._test()
            d.reject()
            self.assertEqual(d._jobs, {})
            release.set()
            self.assertTrue(self.wait(lambda: bool(flags)))
        self.assertEqual(flags, [True])


    def test_a_new_listing_replaces_a_running_one_of_the_same_kind(self):
        release, calls = threading.Event(), []

        def fake(profile):
            calls.append(profile.base_url)
            release.wait(5)
            return ["only"], ""

        d = self.dialog()
        with mock.patch("oscar.providers.list_models", fake):
            d._load_models()
            first = d._jobs["models"][1]
            d._cancel_job("models")
            d._load_models()
            release.set()
            self.assertTrue(self.wait(lambda: d.model_combo.count() >= 2))
        self.assertTrue(first.cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
