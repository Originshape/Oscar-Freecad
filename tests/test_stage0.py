"""Acceptance: the shared types and the test support core exist and behave as documented."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from support import isolation  # noqa: E402

isolation.isolate_process()

from oscar import params, providers, stopreason, tools  # noqa: E402
from oscar.providers import Message, ToolCall  # noqa: E402
from oscar.toolctx import ToolContext, ToolOutcome, current_context, use_context  # noqa: E402
from support.scripted_llm import (ScriptedProvider, bad_json, call, calls, from_legacy,  # noqa: E402
                                  nameless, say, think)
from support.signal_recorder import SignalRecorder  # noqa: E402
from support.fakes import fake_validation, tiny_png_snapshot  # noqa: E402


class SharedTypesTest(unittest.TestCase):
    def test_toolcall_and_message_fields(self):
        c = ToolCall(id="1", name="x")
        self.assertEqual((c.args_error, c.raw_args), ("", ""))
        m = Message(role="user", content="hi")
        self.assertFalse(m.synthetic)
        self.assertEqual(m.tag, "")

    def test_profile_fields_roundtrip(self):
        p = params.Profile(name="t", reasoning="low", caps_override={"tools": "text"})
        params.save_profiles([p])
        q = params.load_profiles()[0]
        self.assertEqual((q.reasoning, q.caps_override), ("low", {"tools": "text"}))
        self.assertEqual(params.Profile().reasoning, "auto")

    def test_tool_metadata_defaults_and_decorator(self):
        @tools.tool("zz_probe", "A probe tool used only by the stage 0 test to check metadata.",
                    {"type": "object", "properties": {}}, category="build", geometry=True, phases=("BUILD",))
        def _probe():
            return "ok", None

        try:
            t = tools.tool_meta("zz_probe") if hasattr(tools, "tool_meta") else tools._HANDLERS["zz_probe"]
            self.assertTrue(t.geometry and t.mutating)
            self.assertEqual(t.phases, ("BUILD",))
            self.assertEqual(t.category, "build")
        finally:
            tools._HANDLERS.pop("zz_probe", None)
        # existing tools keep working with defaults
        legacy = tools._HANDLERS["list_documents"]
        self.assertEqual(legacy.phases, tools.ALL_PHASES)
        self.assertFalse(legacy.mutating)

    def test_toolctx(self):
        self.assertFalse(current_context().active)
        ctx = ToolContext(project=object())
        with use_context(ctx):
            self.assertIs(current_context(), ctx)
            self.assertTrue(current_context().active)
        self.assertFalse(current_context().active)
        self.assertEqual(ToolOutcome().step_index, -1)
        o = ToolOutcome()
        self.assertTrue(o.handler_ok and not o.health_failed)

    def test_cache_dir_honours_the_env_var_at_call_time(self):
        import tempfile

        from oscar import paths

        base = tempfile.mkdtemp()
        old = os.environ.get("OSCAR_CACHE_DIR")
        os.environ["OSCAR_CACHE_DIR"] = base
        try:
            path = paths.cache_dir("mesh")
            self.assertEqual(path, os.path.join(base, "mesh"))
            self.assertTrue(os.path.isdir(path))
        finally:
            if old is None:
                os.environ.pop("OSCAR_CACHE_DIR", None)
            else:
                os.environ["OSCAR_CACHE_DIR"] = old

    def test_stopreason(self):
        for raw, norm in (("end_turn", "end_turn"), ("stop", "end_turn"), ("tool_calls", "tool_use"),
                          ("length", "max_tokens"), ("max_tokens", "max_tokens"), ("", "unknown"), (None, "unknown"),
                          ("weird", "unknown"), ("refusal", "refusal")):
            self.assertEqual(stopreason.normalize_stop_reason(raw), norm, raw)
        self.assertTrue(stopreason.is_cut_off("length"))
        self.assertFalse(stopreason.is_cut_off("stop"))


class SupportTest(unittest.TestCase):
    def test_isolation_sets_sandbox_env_and_keeps_home(self):
        root = isolation.sandbox_root()
        for key in ("OSCAR_PROJECTS_DIR", "OSCAR_LOG_DIR", "OSCAR_CACHE_DIR",
                    "OSCAR_DOWNLOAD_DIR"):
            # under the sandbox, or pinned by the caller (scripts/test.sh sets these four variables itself)
            self.assertTrue(os.environ[key].startswith(root) or key in isolation.PINNED, key)
        # HOME must stay the real one: subprocess tests need the user-site python packages
        self.assertFalse(os.environ.get("HOME", "").startswith(root))

    def test_scripted_provider_records_and_is_strict(self):
        p = ScriptedProvider([say("hello"), calls(call("list_documents", "c1"))])
        ev1 = list(p.chat("sys", [Message(role="user", content="q")], [], None, tool_choice="auto"))
        self.assertEqual([k for k, _ in ev1], ["text", "stop_reason", "done"])
        ev2 = list(p.chat("sys", [Message(role="user", content="q2")], []))
        self.assertEqual(ev2[-1][0], "tool_calls")
        self.assertEqual(p.requests[0].kwargs, {"tool_choice": "auto"})
        self.assertEqual(p.requests[1].messages[0].content, "q2")
        p.assert_exhausted()
        with self.assertRaises(AssertionError):
            list(p.chat("sys", [Message(role="user", content="more")], []))

    def test_builders(self):
        self.assertEqual(think(5).events[-1], ("done", "length"))
        r = bad_json("execute_code")
        tc = r.events[-1][1][0]
        self.assertIn("_raw", tc.arguments)
        self.assertTrue(tc.args_error.startswith("truncated"))
        self.assertEqual(nameless().events[-1], ("done", "tool_calls"))
        self.assertEqual(from_legacy("cutoff", "hmm").events[-1], ("done", "length"))
        self.assertEqual(from_legacy("text", "t").events[0], ("text", "t"))

    def test_signal_recorder(self):
        from PySide import QtCore

        class W(QtCore.QObject):
            text_delta = QtCore.Signal(str)
            tool_finished = QtCore.Signal(str, bool, str)
            turn_finished = QtCore.Signal(bool, str)
            plan_changed = QtCore.Signal(str)

        w = W()
        rec = SignalRecorder(w)
        w.text_delta.emit("a")
        w.tool_finished.emit("x()", True, "ok")
        w.plan_changed.emit(json.dumps({"k": 1}))
        w.turn_finished.emit(True, "done")
        self.assertEqual(rec.names(), ["text_delta", "tool_finished", "plan_changed", "turn_finished"])
        self.assertEqual(rec.of("tool_finished"), [("x()", True, "ok")])
        self.assertEqual(rec.json_of("plan_changed"), [{"k": 1}])

    def test_fakes(self):
        import tempfile

        path = os.path.join(tempfile.mkdtemp(), "a.png")
        self.assertEqual(tiny_png_snapshot("D", path, 480, 360), (True, ""))
        self.assertEqual(open(path, "rb").read(8), b"\x89PNG\r\n\x1a\n")
        v = fake_validation(ok=True, fail_rows=[{"id": "x", "status": "FAIL"}])
        self.assertFalse(v.ok)
        self.assertTrue(v.fit_ran)


if __name__ == "__main__":
    unittest.main()
