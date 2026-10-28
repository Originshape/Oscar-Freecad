"""Approval for model-written Python (``oscar.codeguard``, P0 #1).

Scripts are off by default (``AllowCodeExecution``) and the three script tools are then not offered to the model; when
on, every script is shown in a dialog first (Run / Run all scripts this turn / Don't run), a declined script is not
asked about again in the turn, console sessions refuse, and ``OSCAR_ALLOW_CODE=1`` runs scripts unasked.

The gate logic runs on plain Python (the dialog replaced by a stub). The real dialog (offscreen Qt, the harness of
``test_fix_auth_net``), ``tools.run_tool`` and the agent's tool lists need FreeCAD and Qt and are skipped without them.
The isolation header sets ``OSCAR_ALLOW_CODE=1`` for the whole suite, so every test here removes it first.
"""

import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import codeguard, netguard, turnlog  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402

_FREECAD_ERROR = ""
try:
    from oscar import params, tools  # noqa: E402
    from oscar.cad import dispatch, scripting  # noqa: E402
except Exception as exc:   # pragma: no cover - plain interpreter without FreeCAD
    _FREECAD_ERROR = str(exc)

SCRIPT = "state['ran'] = 1\nprint('hello')"
OTHER = "state['other'] = 2"


class _GateCase(unittest.TestCase):
    """No environment override; the dialog is a stub that records its calls and answers ``self.answer``."""

    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(codeguard.ENV_ALLOW, None)
        self.asked: list[dict] = []
        self.answer = "run"
        ask = mock.patch.object(netguard, "ask_user", side_effect=self._ask)
        ask.start()
        self.addCleanup(ask.stop)
        self.events: list[tuple[str, dict]] = []
        log = mock.patch.object(turnlog, "log_event", lambda name, **fields: self.events.append((name, fields)))
        log.start()
        self.addCleanup(log.stop)

    def _ask(self, title, text, **kw):
        self.asked.append({"title": title, "text": text, **kw})
        return self.answer

    def switch(self, on: bool):
        """The preference, without touching any settings file."""
        patcher = mock.patch.object(codeguard, "_params", return_value=mock.Mock(get_bool=mock.Mock(return_value=on)))
        patcher.start()
        self.addCleanup(patcher.stop)

    def gate(self, code=SCRIPT, state=None, what="execute_code"):
        if state is None:
            return codeguard.gate_execution(code, what)
        with use_context(ToolContext(state=state)):
            return codeguard.gate_execution(code, what)


class SwitchTest(_GateCase):
    def test_off_by_default(self):
        self.assertFalse(codeguard.enabled())
        self.assertEqual(codeguard.hidden_tools(), codeguard.SCRIPT_TOOLS)
        self.assertEqual(self.gate(), (False, codeguard.OFF_TEXT))
        self.assertEqual(self.asked, [])
        self.assertIn("switched off", codeguard.OFF_TEXT)
        self.assertEqual(self.events[-1][1]["decision"], "off")

    def test_the_preference_turns_scripts_on(self):
        self.switch(True)
        self.assertTrue(codeguard.enabled())
        self.assertEqual(codeguard.hidden_tools(), frozenset())

    def test_an_unreadable_preference_counts_as_off(self):
        broken = mock.Mock(get_bool=mock.Mock(side_effect=OSError("settings file locked")))
        with mock.patch.object(codeguard, "_params", return_value=broken):
            self.assertFalse(codeguard.enabled())
        with mock.patch.object(codeguard, "_params", return_value=None):      # no settings store at all
            self.assertFalse(codeguard.enabled())

    def test_the_environment_override_runs_scripts_unasked(self):
        os.environ[codeguard.ENV_ALLOW] = "1"
        self.assertTrue(codeguard.enabled())
        self.assertEqual(codeguard.hidden_tools(), frozenset())
        self.assertEqual(self.gate(), (True, ""))
        self.assertEqual(self.asked, [])
        self.assertEqual(self.events[-1][1]["decision"], "allow_env_override")
        os.environ[codeguard.ENV_ALLOW] = "0"
        self.assertFalse(codeguard.enabled())

    def test_the_script_tools_are_the_three_execute_tools(self):
        self.assertEqual(codeguard.SCRIPT_TOOLS, {"execute_code", "execute_code_async", "execute_code_headless"})


class AskTest(_GateCase):
    def setUp(self):
        super().setUp()
        self.switch(True)

    def test_run_allows_and_the_dialog_shows_the_code(self):
        self.assertEqual(self.gate(), (True, ""))
        (ask,) = self.asked
        self.assertEqual(ask["detailed"], SCRIPT)
        self.assertEqual(ask["object_name"], "oscarScriptApproval")
        self.assertEqual(ask["default"], "deny")
        self.assertEqual([(c.key, c.label) for c in ask["choices"]],
                         [("run", "Run"), ("turn", "Run all scripts this turn"), ("deny", "Don't run")])
        self.assertIn("execute_code, 2 lines", ask["text"])
        self.assertIn("can read and change your files", ask["text"])
        self.assertEqual(ask["informative"], "")

    def test_a_script_that_uses_the_network_says_so(self):
        self.gate("import requests\nrequests.get('https://example.com')")
        self.assertIn("network", self.asked[0]["informative"])

    def test_a_declined_script_is_refused_again_without_a_second_dialog(self):
        state: dict = {}
        self.answer = "deny"
        self.assertEqual(self.gate(state=state), (False, codeguard.DECLINED_TEXT))
        self.answer = "run"
        self.assertEqual(self.gate(state=state), (False, codeguard.DECLINED_TEXT))
        self.assertEqual(len(self.asked), 1)
        self.assertEqual(self.gate(OTHER, state=state), (True, ""))        # another script is asked about
        self.assertEqual(len(self.asked), 2)
        self.assertEqual(self.gate(state={}), (True, ""))                  # a new turn asks again
        self.assertEqual(len(self.asked), 3)

    def test_run_all_scripts_this_turn_answers_the_rest_of_the_turn(self):
        state: dict = {}
        self.answer = "turn"
        self.assertEqual(self.gate(state=state), (True, ""))
        self.assertEqual(self.gate(OTHER, state=state, what="execute_code_async"), (True, ""))
        self.assertEqual(len(self.asked), 1)
        self.assertEqual(state["code_approval"], "turn")
        self.answer = "deny"
        self.assertEqual(self.gate(state={}), (False, codeguard.DECLINED_TEXT))

    def test_no_answer_refuses_and_is_not_asked_again(self):
        state: dict = {}
        for answer in (netguard.NO_ANSWER, netguard.FAILED):
            with self.subTest(answer=answer):
                self.answer = answer
                code = f"x = {answer!r}"
                self.assertEqual(self.gate(code, state=state), (False, codeguard.NO_ANSWER_TEXT))
                self.assertEqual(self.gate(code, state=state), (False, codeguard.NO_ANSWER_TEXT))   # not "declined"
        self.assertEqual(len(self.asked), 2)

    def test_nobody_to_ask_refuses(self):
        for answer in (netguard.NO_HUMAN, netguard.NO_QT):
            with self.subTest(answer=answer):
                self.answer = answer
                self.assertEqual(self.gate(), (False, codeguard.NO_HUMAN_TEXT))

    def test_the_log_carries_a_hash_never_the_code(self):
        secret = "print('do-not-log-this-script-text')"
        self.answer = "deny"
        self.gate(secret)
        (name, fields), = [e for e in self.events if e[0] == "code_permission"]
        self.assertEqual(fields, {"tool": "execute_code", "decision": "deny", "lines": 1,
                                  "script_hash": codeguard.script_hash(secret)})
        self.assertNotIn("do-not-log", repr(self.events))
        self.assertEqual(len(codeguard.script_hash(secret)), 16)


# ------------------------------------------------------------------------------------------- FreeCAD and Qt


@unittest.skipIf(_FREECAD_ERROR, f"FreeCAD unavailable: {_FREECAD_ERROR}")
class RunToolTest(unittest.TestCase):
    """The tool handlers refuse before anything runs."""

    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(codeguard.ENV_ALLOW, None)
        scripting.reset_state()
        self.addCleanup(scripting.reset_state)
        backup = params.backup_state()
        self.addCleanup(params.restore_state, backup)
        params.set_bool(codeguard.PREF, False)

    def test_switched_off_nothing_runs(self):
        for tool, args in (("execute_code", {"code": SCRIPT, "include_screenshot": False}),
                           ("execute_code_async", {"code": SCRIPT}),
                           ("execute_code_headless", {"code": "print('x')", "timeout": 30})):
            with self.subTest(tool):
                text, _image, is_error = tools.run_tool(tool, args)
                self.assertTrue(is_error)
                self.assertIn("switched off", text)
        self.assertNotIn("ran", scripting.STATE)

    def test_switched_on_in_a_console_session_nothing_runs_and_no_dialog(self):
        from PySide import QtWidgets

        class NoApp:
            @staticmethod
            def instance():
                return None

        params.set_bool(codeguard.PREF, True)
        with mock.patch.object(QtWidgets, "QApplication", NoApp), \
                mock.patch.object(QtWidgets, "QMessageBox", side_effect=AssertionError("a dialog was built")):
            text, _image, is_error = tools.run_tool("execute_code", {"code": SCRIPT, "include_screenshot": False})
        self.assertTrue(is_error)
        self.assertIn("nobody can be asked", text)
        self.assertNotIn("ran", scripting.STATE)

    def test_the_environment_override_runs_the_script(self):
        os.environ[codeguard.ENV_ALLOW] = "1"
        text, _image, is_error = tools.run_tool("execute_code", {"code": SCRIPT, "include_screenshot": False})
        self.assertFalse(is_error, text)
        self.assertEqual(scripting.STATE.get("ran"), 1)


def _offscreen_qapp():
    from PySide import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # a stray test must never open a window on the desktop
        app = QtWidgets.QApplication(getattr(sys, "argv", ["tests"])[:1])
    return app


@unittest.skipIf(_FREECAD_ERROR, f"FreeCAD unavailable: {_FREECAD_ERROR}")
class DialogTest(unittest.TestCase):
    """The real QMessageBox, offscreen: clicks are delivered by timers while ``exec()`` runs."""

    def setUp(self):
        from PySide import QtCore, QtWidgets

        self.QtCore, self.QtWidgets = QtCore, QtWidgets
        self._timers = []
        self.app = _offscreen_qapp()
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(codeguard.ENV_ALLOW, None)
        backup = params.backup_state()
        self.addCleanup(params.restore_state, backup)
        params.set_bool(codeguard.PREF, True)
        self.main = QtWidgets.QMainWindow()
        self.main.show()
        try:
            import FreeCADGui as gui
        except Exception:  # noqa: BLE001
            import types

            gui = sys.modules.setdefault("FreeCADGui", types.ModuleType("FreeCADGui"))
        patcher = mock.patch.object(gui, "getMainWindow", lambda: self.main, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._cleanup)
        self.seen: list = []

    def _cleanup(self):
        for timer in self._timers:
            timer.stop()
        for box in self._boxes():
            box.reject()
        self.main.close()
        self.main.deleteLater()

    def _boxes(self):
        return [w for w in self.app.topLevelWidgets() if isinstance(w, self.QtWidgets.QMessageBox) and w.isVisible()]

    def _after(self, ms, fn):
        timer = self.QtCore.QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(fn)
        timer.start(ms)
        self._timers.append(timer)

    def _click(self, text):
        for box in self._boxes():
            self.seen.append({"name": box.objectName(), "details": box.detailedText(),
                              "default": box.defaultButton().text() if box.defaultButton() else "",
                              "buttons": sorted(b.text() for b in box.buttons())})
            for button in box.buttons():
                if button.text() == text:
                    button.click()
                    return

    def _safety_valve(self, ms: int = 3000):
        """Never hang the suite: a dialog that outlives the test is closed."""
        tripped = []
        self._after(ms, lambda: (tripped.append(1), [b.reject() for b in self._boxes()]))
        return tripped

    def gate(self, code, state):
        with use_context(ToolContext(state=state)):
            return codeguard.gate_execution(code, "execute_code")

    def test_run_shows_the_code_and_allows(self):
        valve = self._safety_valve()
        self._after(100, lambda: self._click("Run"))
        self.assertEqual(self.gate(SCRIPT, {}), (True, ""))
        self.assertEqual(valve, [])
        (box,) = self.seen
        self.assertEqual(box["name"], "oscarScriptApproval")
        self.assertEqual(box["details"], SCRIPT)
        self.assertEqual(box["default"], "Don't run")
        self.assertTrue({"Run", "Run all scripts this turn", "Don't run"} <= set(box["buttons"]))
        self.assertEqual(self._boxes(), [])

    def test_dont_run_refuses_and_the_same_script_is_not_asked_again(self):
        self._safety_valve()
        state: dict = {}
        self._after(100, lambda: self._click("Don't run"))
        self.assertEqual(self.gate(SCRIPT, state), (False, codeguard.DECLINED_TEXT))
        started = time.monotonic()
        self.assertEqual(self.gate(SCRIPT, state), (False, codeguard.DECLINED_TEXT))
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(len(self.seen), 1)

    def test_run_all_scripts_this_turn_lets_the_next_script_through_unasked(self):
        self._safety_valve()
        state: dict = {}
        self._after(100, lambda: self._click("Run all scripts this turn"))
        self.assertEqual(self.gate(SCRIPT, state), (True, ""))
        self.assertEqual(self.gate(OTHER, state), (True, ""))
        self.assertEqual(len(self.seen), 1)

    def test_a_timeout_refuses_and_closes_the_dialog(self):
        # generous margins: the dialog's own 0.3 s timer must win against the valve even on a loaded machine (two
        # suites at once made a 3 s valve / 2 s bound flaky)
        valve = self._safety_valve(15000)
        started = time.monotonic()
        with mock.patch.object(netguard, "DENY_TIMEOUT_S", 0.3):
            verdict = self.gate(SCRIPT, {})
        self.assertEqual(verdict, (False, codeguard.NO_ANSWER_TEXT))
        self.assertEqual(valve, [], "the dialog had to be closed by force")
        self.assertLess(time.monotonic() - started, 12.0)
        self.assertEqual(self._boxes(), [])

    def test_stop_closes_the_open_question_unanswered(self):
        """Stop must not wait out DENY_TIMEOUT_S while an approval is open (audit 2026-10-28: the browser said
        "Stopping…" for two minutes with the approval buttons still live)."""
        valve = self._safety_valve(15000)
        started = time.monotonic()
        self._after(200, netguard.cancel_open_questions)
        self.assertEqual(self.gate(SCRIPT, {}), (False, codeguard.NO_ANSWER_TEXT))
        self.assertEqual(valve, [], "the dialog had to be closed by force")
        self.assertLess(time.monotonic() - started, 10.0)
        self.assertEqual(self._boxes(), [])
        self._after(100, lambda: self._click("Run"))
        self.assertEqual(self.gate(SCRIPT, {}), (True, ""), "a later question is asked as usual")

    def test_a_remote_view_hears_each_question_and_its_answer(self):
        heard: list = []
        hook = mock.patch.object(netguard, "on_approval", heard.append)
        hook.start()
        self.addCleanup(hook.stop)
        self._safety_valve(15000)
        self._after(100, lambda: self._click("Run"))
        self.gate(SCRIPT, {})
        self._after(100, lambda: self._click("Don't run"))
        self.gate(SCRIPT, {})
        with mock.patch.object(netguard, "DENY_TIMEOUT_S", 0.3):
            self.gate(SCRIPT, {})
        states = [item["state"] for item in heard]
        self.assertEqual(states, ["waiting", "approved", "waiting", "declined", "waiting", "no_answer"])
        ids = [item["id"] for item in heard]
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(len(set(ids)), 3)   # one id per question, the same for its answer
        self.assertTrue(all(item["op"] == "code_permission" for item in heard))
        self.assertIn("Python script", heard[0]["text"])

    def test_a_remote_view_can_answer_the_open_question(self):
        heard: list = []
        hook = mock.patch.object(netguard, "on_approval", heard.append)
        hook.start()
        self.addCleanup(hook.stop)
        valve = self._safety_valve(15000)
        taken: list = []

        def answer_from_the_browser():
            approval = heard[-1]
            taken.append(netguard.answer_remote(approval["id"], "nope"))   # not one of its answers
            taken.append(netguard.answer_remote(approval["id"], "turn"))
            taken.append(netguard.answer_remote(approval["id"], "deny"))   # the first answer wins

        self._after(150, answer_from_the_browser)
        state: dict = {}
        self.assertEqual(self.gate(SCRIPT, state), (True, ""))
        self.assertEqual(taken, [False, True, False])
        self.assertEqual(valve, [], "the browser's answer closed the dialog")
        self.assertEqual(self._boxes(), [])
        self.assertEqual([item["state"] for item in heard], ["waiting", "approved"])
        self.assertEqual(heard[1]["by"], "browser")
        self.assertEqual([choice["key"] for choice in heard[0]["choices"]], ["run", "turn", "deny"])
        self.assertEqual(heard[0]["detailed"], SCRIPT)
        self.assertEqual(self.gate(OTHER, state), (True, ""))   # "Run all scripts this turn" counted as a click
        self.assertFalse(netguard.answer_remote(heard[0]["id"], "run"))   # settled: closed to answers

    def test_always_allow_is_never_offered_to_a_remote_view(self):
        heard: list = []
        hook = mock.patch.object(netguard, "on_approval", heard.append)
        hook.start()
        self.addCleanup(hook.stop)
        self._safety_valve(15000)
        taken: list = []
        self._after(150, lambda: taken.extend(
            [netguard.answer_remote(heard[-1]["id"], "always"), netguard.answer_remote(heard[-1]["id"], "deny")]))
        with mock.patch.dict(os.environ, {"OSCAR_ALLOW_NET": ""}):
            self.assertEqual(netguard.request_permission("Download a file from example.com"), "deny")
        self.assertEqual(taken, [False, True])
        self.assertEqual([choice["key"] for choice in heard[0]["choices"]], ["allow", "deny"])
        self.assertEqual((heard[1]["state"], heard[1]["by"]), ("declined", "browser"))

    def test_the_dialog_does_not_block_other_gui_tasks(self):
        calls = []
        real = dispatch.dispatch

        def spy(fn, *args, **kwargs):
            calls.append(kwargs)
            return real(fn, *args, **kwargs)

        self._safety_valve()
        self._after(100, lambda: self._click("Run"))
        with mock.patch.object(dispatch, "dispatch", spy):
            self.gate(SCRIPT, {})
        self.assertEqual([c.get("reentrant") for c in calls], [True])
        self.assertEqual(calls[0].get("op"), "code_permission")


_AGENT_ERROR = _FREECAD_ERROR
if not _AGENT_ERROR:
    try:
        from test_agent_loop import LoopCase  # noqa: E402
        from support.scripted_llm import say  # noqa: E402
    except Exception as exc:   # pragma: no cover
        _AGENT_ERROR = str(exc)
if _AGENT_ERROR:
    LoopCase = unittest.TestCase   # noqa: N816 - placeholder: the class below is skipped

    def say(text):  # noqa: D103 - placeholder
        return text


@unittest.skipIf(_AGENT_ERROR, f"agent unavailable: {_AGENT_ERROR}")
class AgentToolListTest(LoopCase):
    """Switched off, the script tools are not offered: not in the round's specs, not named by the system prompt."""

    def turn_lists(self):
        worker = self.build("hello there", [say("Hi.")])
        worker._begin_turn()
        try:
            names = {s.name for s in worker._round_specs()}
            prompt_names = worker._turn_tool_names("other")
            system = worker._system
        finally:
            worker._teardown("answered", None)
        return names, prompt_names, system

    def test_switched_off_the_script_tools_are_hidden(self):
        with mock.patch.dict(os.environ), mock.patch.object(codeguard, "_params", return_value=None):
            os.environ.pop(codeguard.ENV_ALLOW, None)
            names, prompt_names, system = self.turn_lists()
        self.assertTrue(names, "the turn offers tools")
        self.assertFalse(codeguard.SCRIPT_TOOLS & names)
        self.assertFalse(codeguard.SCRIPT_TOOLS & prompt_names)
        self.assertNotIn("`execute_code`", system)
        self.assertIn("create_object", names)

    def test_switched_on_they_are_offered(self):
        with mock.patch.dict(os.environ, {codeguard.ENV_ALLOW: "1"}):
            names, prompt_names, _system = self.turn_lists()
        self.assertTrue(codeguard.SCRIPT_TOOLS <= names)
        self.assertIn("execute_code", prompt_names)


if __name__ == "__main__":
    unittest.main()
