"""End-to-end test of the panel with a REAL ``AgentWorker`` thread.

The worker runs on a real QThread with a scripted provider, fake tools and a real Project in the test sandbox, so
every signal crosses the thread boundary through Qt's queued connections exactly as in the application, and the
panel's slots, strips, rows, thumbnails and persistence are exercised together. Offscreen QApplication, no network,
nothing shown to the user. Skipped when PySide / FreeCAD are unavailable (plain app python).
"""

import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

_IMPORT_ERROR = ""
try:
    from PySide import QtCore
    from oscar import agent, chats, panel, params
    from oscar.cad import dispatch
    from oscar.panel import Block
    from oscar.toolctx import ToolOutcome, current_context
    from test_panel_flow import FlowTestCase, png
    from test_panel_logic import GOOD_SNAPSHOT
except Exception as exc:   # pragma: no cover - plain interpreter without FreeCAD / Qt
    _IMPORT_ERROR = str(exc)

    class _Unavailable(unittest.TestCase):
        """Base class that skips every test of a module whose Qt / FreeCAD imports failed."""

        @classmethod
        def setUpClass(cls):
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")

    FakeWorker = object
    FlowWorker = object
    PanelTestCase = FlowTestCase = _Unavailable


class EndToEndTest(FlowTestCase):
    def setUp(self):
        super().setUp()
        self.release = threading.Event()
        self.in_tool = threading.Event()
        self.block_on = ""
        engine = mock.Mock()
        engine.begin_turn_snapshot.return_value = dict(GOOD_SNAPSHOT)

        def fake_run_tool(name, args):
            ctx = current_context()
            self.in_tool.set()
            if name == self.block_on:
                self.release.wait(20)
            geometry = name == "create_object"
            shot = png(os.path.join(self.tmp, f"{name}_{int(time.monotonic() * 1000)}.png")) if geometry else ""
            outcome = ToolOutcome(
                name=name, call_id=ctx.call_id, title=args.get("step_title") or f"Step {name}", ok=True, handler_ok=True, text=f"{name} done",
                mutating=True, geometry=geometry, step_index=1 if geometry else -1, snapshot_path=shot,
                args=dict(args), delta={"added": ["Box"], "removed": [], "changed": []} if geometry else {})
            ctx.outcome = outcome
            if ctx.ledger is not None:
                ctx.ledger.on_tool_result(outcome)
            return f"{name} done", None, False

        # The worker thread reaches the "GUI thread" (this test's main thread, spinning the event loop in
        # wait_until) through dispatch; without the wake bridge every GUI hook would wait out its 10 s timeout.
        dispatch.init()
        self.addCleanup(dispatch.shutdown)
        for patcher in (mock.patch.object(params, "active_profile",
                                          return_value=params.Profile(name="t", provider="openai_compat", model="m")),
                        mock.patch("oscar.cad.get_engine", return_value=engine),
                        mock.patch.object(agent._tools, "run_tool", fake_run_tool)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.stop_worker)

    def start_turn(self, text, script, validator=None):
        from support.fakes import fake_validation
        from support.scripted_llm import ScriptedProvider

        provider = ScriptedProvider(script)
        profile = params.Profile(name="t", provider="openai_compat", model="scripted")
        check = validator if validator is not None else (lambda doc, ctx: fake_validation(True))

        def factory(history, **kw):
            return agent.AgentWorker(history, provider=provider, profile=profile, validator=check, **kw)

        with mock.patch.object(panel, "AgentWorker", factory):
            self.p.input.setPlainText(text)
            self.p._send()
        return provider

    def stop_worker(self):
        worker = self.p.worker
        if worker is not None:
            self.release.set()
            worker.stop()
            worker.wait(5000)

    def wait_until(self, condition, seconds=30):
        deadline = time.monotonic() + seconds
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.01)
        self.app.processEvents()
        return condition()

    def test_a_scripted_work_turn_fills_every_widget(self):
        from support.scripted_llm import call, calls, say

        p = self.p
        self.start_turn("Build the smoke test box.", [
            calls(call("create_document", name="D"), text="Setting up a document."),
            calls(call("create_object", doc_name="D", obj_type="Part::Box", step_title="Make the box")),
            say("I created a 10x10x10 mm box.")])
        self.assertTrue(self.wait_until(lambda: p.worker is None), "the turn did not finish")
        card = self.card()
        # exactly the tool rows: plans, validation and the Build Summary never create rows
        rows = [s for s in card.steps if not s.plain and not s.is_thought]
        self.assertEqual([s.view.name for s in rows], ["create_document", "create_object"])
        self.assertEqual(rows[1].view.title, "Make the box")
        self.assertTrue(card.text.startswith("✻ Ran 2 steps"), card.text)
        self.assertTrue(card.text.endswith("✓ validated"), card.text)
        self.assertFalse(card.expanded)
        self.assertEqual(len([s for s in card.steps if s.plain]), 1)       # the pre-tool remark folded into the card
        # plan strip pinned, with the template plan of the intent
        self.assertFalse(p.plan_strip.isHidden())
        self.assertIn("generic_part", p.plan_strip.toPlainText())
        self.assertTrue(p._current_session()["ledger"]["steps"])
        # thumbnail of the geometry step, still resolvable after another render
        self.assertTrue(p.log.has_resource(rows[1].call_id))
        p._render()
        self.assertTrue(p.log.has_resource(rows[1].call_id))
        # artifacts strip; the project folder was written by the worker thread
        self.assertFalse(p.artifacts_strip.isHidden())
        self.assertIn("Oscar project", p.artifacts_strip.text())
        root = p._current_session()["project_dir"]
        self.assertTrue(os.path.isdir(root))
        # the answer once, the Build Summary card once
        plain = p.log.toPlainText()
        self.assertEqual(plain.count("I created a 10x10x10 mm box."), 1)
        self.assertEqual(plain.count("Build Summary"), 1)
        self.assertNotIn("## Build Summary", plain)
        self.assertEqual(p.history[-1].content, "I created a 10x10x10 mm box.")   # the stored answer is the model text alone
        # persisted build state
        saved = chats.load(p._current_session()["id"])
        self.assertEqual(saved["project_dir"], root)
        self.assertEqual(len(saved["blocks_meta"]), 1)
        self.assertTrue(saved["ledger"]["steps"])
        self.assertTrue(p.heartbeat_row.isHidden())
        self.assertFalse(p._heartbeat_timer.isActive())

    def test_the_header_is_working_while_the_worker_runs_and_progress_arrives(self):
        from support.scripted_llm import call, calls, say

        p = self.p
        self.block_on = "create_object"
        self.start_turn("Build the smoke test box.", [
            calls(call("create_object", doc_name="D", obj_type="Part::Box")), say("Done.")])
        self.assertTrue(self.wait_until(self.in_tool.is_set))
        self.assertTrue(self.wait_until(lambda: p._turn_view is not None and p._turn_view.progress is not None))
        card = self.card()
        self.assertTrue(card.live)
        self.assertIn("Working…", p.log.toPlainText())
        self.assertNotIn("✻ Done", p.log.toPlainText())
        self.assertFalse(p.heartbeat_row.isHidden())
        self.assertIn("running create_object", p.heartbeat_label.text())
        self.assertFalse(p.plan_strip.isHidden())                          # the plan was pinned at t=0
        self.release.set()
        self.assertTrue(self.wait_until(lambda: p.worker is None))
        self.assertFalse(self.card().live)

    def test_stop_ends_the_turn_with_a_stopped_badge(self):
        from support.scripted_llm import call, calls, say

        p = self.p
        self.block_on = "create_object"
        self.start_turn("Build the smoke test box.", [
            calls(call("create_object", doc_name="D", obj_type="Part::Box")), say("Done.")])
        self.assertTrue(self.wait_until(self.in_tool.is_set))
        p._stop()
        self.release.set()
        self.assertTrue(self.wait_until(lambda: p.worker is None), "the stopped turn did not end")
        card = self.card()
        self.assertTrue(card.text.endswith("■ stopped"), card.text)
        self.assertEqual(card.header_kind, "error")
        self.assertIn("Stopped.", [b.text for b in p.blocks if b.kind == Block.ERROR])
        self.assertTrue(all(not s.view.running for s in card.steps if s.view is not None))

    def test_a_question_turn_keeps_the_plan_of_the_earlier_work_turn(self):
        from support.scripted_llm import call, calls, say

        p = self.p
        self.start_turn("Build the smoke test box.", [
            calls(call("create_object", doc_name="D", obj_type="Part::Box")), say("Built.")])
        self.assertTrue(self.wait_until(lambda: p.worker is None))
        plan = p._current_session()["ledger"]
        self.assertTrue(plan["steps"])
        self.start_turn("what is a fillet?", [say("A fillet rounds an edge.")])
        self.assertTrue(self.wait_until(lambda: p.worker is None))
        self.assertEqual(p._current_session()["ledger"]["revision"], plan["revision"])   # carried unchanged
        self.assertEqual(p._current_session()["ledger"]["request"], plan["request"])
        self.assertFalse(p.plan_strip.isHidden())
        answers = [b.text for b in p.blocks if b.kind == Block.ASSISTANT]
        self.assertEqual(answers[-1], "A fillet rounds an edge.")

    def test_two_turns_keep_separate_rows_and_thumbnails(self):
        from support.scripted_llm import call, calls, say

        p = self.p
        self.start_turn("Build the smoke test box.", [
            calls(call("create_object", "call_0", doc_name="D", obj_type="Part::Box", step_title="First box")),
            say("One.")])
        self.assertTrue(self.wait_until(lambda: p.worker is None))
        self.start_turn("Add another box.", [
            calls(call("create_object", "call_0", doc_name="D", obj_type="Part::Box", step_title="Second box")),
            say("Two.")])
        self.assertTrue(self.wait_until(lambda: p.worker is None))
        cards = [b for b in p.blocks if b.kind == Block.TOOL and b.steps]
        self.assertEqual(len(cards), 2)
        keys = [c.steps[0].call_id for c in cards]
        self.assertEqual(len(set(keys)), 2)                                 # the provider reused "call_0"
        self.assertEqual([c.steps[0].view.title for c in cards], ["First box", "Second box"])
        self.assertTrue(all(p.log.has_resource(k) for k in keys))


if __name__ == "__main__":
    unittest.main()
