"""Regression tests for the engine review fixes (cad-engine-03/07/08/09/10/11/
12/13, panel-b-02, tools-01/05/09/10/11) and for the second review pass over
them (engine-01/02/03/04/06/07/08).

CadEngine (engine.py), the GUI-thread dispatcher (dispatch.py) and the
property mapper (properties.py) against the real FreeCAD runtime in console
mode: no GUI, no network. Console documents default to UndoMode 0, so every
test that needs the undo stack switches it on explicitly.
"""

import math
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

if "FreeCADGui" not in sys.modules:
    try:
        import FreeCADGui  # noqa: F401
    except Exception:
        sys.modules["FreeCADGui"] = types.ModuleType("FreeCADGui")

import FreeCAD as App  # noqa: E402

from oscar.cad import dispatch, engine as engine_module, get_engine  # noqa: E402
from oscar.cad.properties import apply_properties  # noqa: E402


def _new_doc(name: str, undo: bool = True):
    doc = App.newDocument(name)
    if undo:
        doc.UndoMode = 1
    return doc


def _user_entries(doc, count: int) -> None:
    """GUI-style (non-Oscar) undo entries, as a user's own edits leave."""
    for i in range(count):
        doc.openTransaction(f"user edit {i}")
        doc.addObject("Part::Box", f"U{i}")
        doc.commitTransaction()


class _DocTestCase(unittest.TestCase):
    """Closes every document a test opened."""

    def setUp(self):
        self._before = set(App.listDocuments())
        self.engine = get_engine()

    def tearDown(self):
        for name in list(App.listDocuments()):
            if name not in self._before:
                try:
                    App.closeDocument(name)
                except Exception:
                    pass

    def names(self, doc):
        return [o.Name for o in doc.Objects]


# ---------------------------------------------------------------------------
# cad-engine-03 / cad-engine-13: the dispatcher


class _Drained(unittest.TestCase):
    def setUp(self):
        while not dispatch._pending.empty():
            dispatch._pending.get_nowait()

    def tearDown(self):
        while not dispatch._pending.empty():
            dispatch._pending.get_nowait()


class DispatchExitTest(_Drained):
    """cad-engine-03: a task raising SystemExit/KeyboardInterrupt must come
    back as an error — escaping a Qt slot terminates the whole process."""

    def test_inline_systemexit_is_returned_as_an_error(self):
        def task():
            raise SystemExit(3)

        try:
            result, failed = dispatch.dispatch(task)
        except BaseException as e:  # noqa: BLE001 - this is the defect
            self.fail(f"SystemExit escaped dispatch: {e!r}")
        self.assertTrue(failed)
        self.assertTrue(str(result).startswith("SystemExit"), result)

    def test_inline_keyboardinterrupt_is_returned_as_an_error(self):
        def task():
            raise KeyboardInterrupt()

        try:
            result, failed = dispatch.dispatch(task)
        except BaseException as e:  # noqa: BLE001
            self.fail(f"KeyboardInterrupt escaped dispatch: {e!r}")
        self.assertTrue(failed)
        self.assertIn("KeyboardInterrupt", result)

    def test_drain_queue_survives_systemexit(self):
        """The GUI-thread drain loop is where an escaping SystemExit kills
        the app; it must record the failure and finish the request."""
        def boom():
            raise SystemExit(3)

        later = []
        first = dispatch._Request(boom)
        second = dispatch._Request(lambda: later.append("ran") or "ok")
        dispatch._pending.put(first)
        dispatch._pending.put(second)
        try:
            dispatch.drain_queue()
        except BaseException as e:  # noqa: BLE001
            self.fail(f"SystemExit escaped drain_queue: {e!r}")
        self.assertTrue(first.done.is_set())
        self.assertTrue(first.failed)
        self.assertTrue(str(first.result).startswith("SystemExit"), first.result)
        self.assertEqual(later, ["ran"])  # the loop kept draining

    def test_execute_code_sys_exit_is_an_error_result(self):
        res = get_engine().execute_code("import sys\nsys.exit(2)")
        self.assertFalse(res["success"])
        self.assertIn("SystemExit", res["error"])


class _FakeApp:
    def thread(self):
        return "gui"


class DispatchTimeoutTest(_Drained):
    """cad-engine-13: a request the caller gave up on must never run later."""

    def _gui_less(self, emit):
        """Make dispatch() take the queued path with a stub wake-up."""
        widgets = types.SimpleNamespace(
            QApplication=types.SimpleNamespace(instance=lambda: _FakeApp())
        )
        core = types.SimpleNamespace(
            QThread=types.SimpleNamespace(currentThread=lambda: "worker")
        )
        waker = types.SimpleNamespace(rouse=types.SimpleNamespace(emit=emit))
        for patcher in (
            mock.patch.object(dispatch, "QtWidgets", widgets),
            mock.patch.object(dispatch, "QtCore", core),
            mock.patch.object(dispatch, "_waker", waker),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_timed_out_request_is_cancelled_and_never_runs(self):
        ran = []
        self._gui_less(lambda: None)  # nobody drains: the GUI thread is busy
        result, failed = dispatch.dispatch(
            lambda: ran.append("ran"), timeout=0.1, op="probe"
        )
        self.assertTrue(failed)
        self.assertFalse(result["success"])
        self.assertIn("cancel", result["error"])
        dispatch.drain_queue()  # the GUI thread finally gets round to it
        self.assertEqual(ran, [], "the abandoned task ran after its timeout")

    def test_task_already_running_at_the_timeout_returns_its_real_result(self):
        threads = []

        def emit():
            thread = threading.Thread(target=dispatch.drain_queue)
            threads.append(thread)
            thread.start()

        self._gui_less(emit)

        def slow():
            time.sleep(0.4)
            return {"success": True, "value": 42}

        result, failed = dispatch.dispatch(slow, timeout=0.2, op="probe")
        for thread in threads:
            thread.join(5)
        self.assertFalse(failed)
        self.assertEqual(result, {"success": True, "value": 42})

    def test_task_still_running_after_the_grace_period_is_reported(self):
        threads = []

        def emit():
            thread = threading.Thread(target=dispatch.drain_queue)
            threads.append(thread)
            thread.start()

        self._gui_less(emit)
        with mock.patch.object(dispatch, "_RUNNING_GRACE", 0.05):
            result, failed = dispatch.dispatch(
                lambda: time.sleep(0.45), timeout=0.2, op="probe"
            )
        for thread in threads:
            thread.join(5)
        self.assertTrue(failed)
        self.assertFalse(result["success"])
        self.assertIn("still running", result["error"].lower())   # our text spells it STILL RUNNING


class DispatchReentrancyTest(_DocTestCase):
    """P0 #6: a request another caller queued (the MCP bridge, a background job's commit) while a script runs must not
    run INSIDE the script when the script pumps events: the script's rollback would delete that caller's work."""

    def setUp(self):
        super().setUp()
        from oscar.cad import scripting

        self.scripting = scripting
        self._empty()
        self.addCleanup(self._empty)
        scripting.reset_state()
        self.addCleanup(scripting.reset_state)

    @staticmethod
    def _empty():
        while not dispatch._pending.empty():
            dispatch._pending.get_nowait().done.set()

    def test_a_request_queued_during_a_failing_script_survives_its_rollback(self):
        doc = _new_doc("ReentrantDoc")
        App.setActiveDocument(doc.Name)

        def other_caller():
            doc.addObject("Part::Box", "OtherCaller")
            doc.recompute()
            return {"success": True}

        request = dispatch._Request(engine_module.txn_task(other_caller, "Oscar: create_object", doc.Name))
        dispatch._pending.put(request)
        self.scripting.STATE["pump"] = dispatch.drain_queue          # what the script's processEvents() delivers
        waker = types.SimpleNamespace(rouse=types.SimpleNamespace(emit=dispatch.drain_queue))   # re-wake drains at once
        with mock.patch.object(dispatch, "_run_inline", return_value=True), mock.patch.object(dispatch, "_waker", waker):
            result = self.scripting.run_script(
                "App.ActiveDocument.addObject('Part::Box', 'Mine')\nstate['pump']()\nraise RuntimeError('boom')")
        self.assertFalse(result["success"])
        self.assertIn("ROLLED BACK", result["error"])
        self.assertIsNone(doc.getObject("Mine"))
        self.assertTrue(request.done.is_set())
        self.assertFalse(request.failed, request.result)
        self.assertIsNotNone(doc.getObject("OtherCaller"), "the other caller's work was rolled back with the script")
        self.assertEqual(dispatch._depth(), 0)


# ---------------------------------------------------------------------------
# panel-b-02 / cad-engine-07: revert_turn


class RevertSafetyTest(_DocTestCase):
    """panel-b-02: a revert undoes only what the turn itself did."""

    def test_invalid_snapshots_close_and_undo_nothing(self):
        keep = _new_doc("KeepMe")
        _user_entries(keep, 2)
        before = set(App.listDocuments())
        for bad in ({}, None, {"success": False, "docs": {}},
                    {"success": True, "docs": "nope"}, "snapshot"):
            res = self.engine.revert_turn(bad)
            self.assertFalse(res["success"], bad)
            self.assertIn("error", res)
            self.assertEqual(set(App.listDocuments()), before, bad)
        self.assertEqual(keep.UndoCount, 2)

    def test_snapshot_carries_identity_and_chains_turns(self):
        doc = _new_doc("SnapDoc")
        first = self.engine.begin_turn_snapshot()
        self.assertTrue(first["success"])
        self.assertEqual(first["uids"]["SnapDoc"], doc.Uid)
        self.assertNotIn("end_docs", first)
        second = self.engine.begin_turn_snapshot()
        self.assertGreater(second["seq"], first["seq"])
        # starting the next turn closes out the previous one on the SAME
        # dict object the panel keeps on its card
        self.assertEqual(first["end_docs"]["SnapDoc"], doc.UndoCount)
        self.assertEqual(first["end_uids"]["SnapDoc"], doc.Uid)

    def test_roundtrip_undoes_the_turns_own_steps(self):
        doc = _new_doc("RtDoc")
        _user_entries(doc, 2)
        snap = self.engine.begin_turn_snapshot()
        self.engine.create_object("RtDoc", {"Type": "Part::Box", "Name": "After"})
        res = self.engine.revert_turn(snap)
        self.assertTrue(res["success"], res)
        self.assertEqual(res["undone"], {"RtDoc": 1})
        self.assertNotIn("After", self.names(doc))
        self.assertIn("U0", self.names(doc))

    def test_agent_created_document_is_closed(self):
        snap = self.engine.begin_turn_snapshot()
        name = self.engine.create_document("AgentMade")["document_name"]
        App.getDocument(name).UndoMode = 1
        self.engine.create_object(name, {"Type": "Part::Box", "Name": "Box"})
        res = self.engine.revert_turn(snap)
        self.assertTrue(res["success"], res)
        self.assertIn(name, res["closed_docs"])
        self.assertNotIn(name, App.listDocuments())

    def test_document_created_by_a_script_is_closed(self):
        snap = self.engine.begin_turn_snapshot()
        res = self.engine.execute_code(
            "d = App.newDocument('ScriptMadeDoc')\nd.addObject('Part::Box', 'B')"
        )
        self.assertTrue(res["success"], res)
        res = self.engine.revert_turn(snap)
        self.assertIn("ScriptMadeDoc", res["closed_docs"])
        self.assertNotIn("ScriptMadeDoc", App.listDocuments())

    def test_empty_document_created_in_the_turn_is_closed(self):
        snap = self.engine.begin_turn_snapshot()
        App.newDocument("CreatedByTurnX")
        res = self.engine.revert_turn(snap)
        self.assertIn("CreatedByTurnX", res["closed_docs"])
        self.assertNotIn("CreatedByTurnX", App.listDocuments())

    def test_users_unsaved_document_survives_a_revert(self):
        snap = self.engine.begin_turn_snapshot()
        user = _new_doc("UserBracket")
        user.openTransaction("Pad")
        user.addObject("Part::Box", "Unsaved")
        user.commitTransaction()
        res = self.engine.revert_turn(snap)
        self.assertIn("UserBracket", App.listDocuments())
        self.assertEqual(self.names(user), ["Unsaved"])
        self.assertNotIn("UserBracket", res["closed_docs"])
        self.assertIn("UserBracket", [k["doc"] for k in res["kept_docs"]])

    def test_console_made_document_with_content_survives(self):
        snap = self.engine.begin_turn_snapshot()
        user = App.newDocument("ConsoleMade")
        user.addObject("Part::Box", "Typed")
        res = self.engine.revert_turn(snap)
        self.assertIn("ConsoleMade", App.listDocuments())
        self.assertIn("ConsoleMade", [k["doc"] for k in res["kept_docs"]])

    def test_agent_document_the_user_saved_survives(self):
        snap = self.engine.begin_turn_snapshot()
        name = self.engine.create_document("AgentSaved")["document_name"]
        doc = App.getDocument(name)
        doc.UndoMode = 1
        self.engine.create_object(name, {"Type": "Part::Box", "Name": "Box"})
        path = os.path.join(tempfile.mkdtemp(prefix="oscar_fix_engine_"), "kept.FCStd")
        doc.saveAs(path)
        res = self.engine.revert_turn(snap)
        self.assertIn(name, App.listDocuments())
        self.assertIn(name, [k["doc"] for k in res["kept_docs"]])
        # engine-03: a document of the turn's own that is still open is not
        # a finished revert (the panel would say "Turn reverted")
        self.assertFalse(res["success"], res)
        self.assertTrue(res["partial"])
        self.assertIn(name, res["error"])
        self.assertIn("saved to disk", res["error"])
        self.assertTrue(any(name in w for w in res["warnings"]))

    def test_agent_document_that_cannot_be_closed_is_a_partial_revert(self):
        snap = self.engine.begin_turn_snapshot()
        name = self.engine.create_document("AgentStuck")["document_name"]
        App.getDocument(name).UndoMode = 1
        self.engine.create_object(name, {"Type": "Part::Box", "Name": "Box"})
        with mock.patch.object(App, "closeDocument", side_effect=RuntimeError("busy")):
            res = self.engine.revert_turn(snap)
        self.assertIn(name, App.listDocuments())
        self.assertFalse(res["success"], res)
        self.assertTrue(res["partial"])
        self.assertIn("busy", res["error"])

    def test_agent_document_the_user_edited_is_a_partial_revert(self):
        snap = self.engine.begin_turn_snapshot()
        name = self.engine.create_document("AgentEdited")["document_name"]
        doc = App.getDocument(name)
        doc.UndoMode = 1
        self.engine.create_object(name, {"Type": "Part::Box", "Name": "Box"})
        doc.openTransaction("Move")
        doc.getObject("Box").Length = 33
        doc.commitTransaction()
        res = self.engine.revert_turn(snap)
        self.assertIn(name, App.listDocuments())
        self.assertFalse(res["success"], res)
        self.assertTrue(res["partial"])

    def test_a_document_the_user_made_is_not_a_partial_revert(self):
        # the user's own new document is not part of the turn: leaving it
        # open is the right outcome, not a failure to revert
        snap = self.engine.begin_turn_snapshot()
        user = _new_doc("UserOwn")
        user.openTransaction("Pad")
        user.addObject("Part::Box", "Mine")
        user.commitTransaction()
        res = self.engine.revert_turn(snap)
        self.assertTrue(res["success"], res)
        self.assertFalse(res["partial"])
        self.assertIn("UserOwn", [k["doc"] for k in res["kept_docs"]])

    def test_older_turn_cannot_cascade_into_a_later_turn(self):
        first = self.engine.begin_turn_snapshot()
        name = self.engine.create_document("TwoTurns")["document_name"]
        doc = App.getDocument(name)
        doc.UndoMode = 1
        self.engine.create_object(name, {"Type": "Part::Box", "Name": "T1Box"})
        self.engine.begin_turn_snapshot()  # turn 2 starts
        self.engine.create_object(name, {"Type": "Part::Box", "Name": "T2Box"})
        res = self.engine.revert_turn(first)
        self.assertFalse(res["success"], res)
        self.assertIn(name, App.listDocuments())
        self.assertEqual(self.names(doc), ["T1Box", "T2Box"])
        self.assertTrue(res["conflicts"])

    def test_turns_revert_cleanly_newest_first(self):
        first = self.engine.begin_turn_snapshot()
        name = self.engine.create_document("Chain")["document_name"]
        doc = App.getDocument(name)
        doc.UndoMode = 1
        self.engine.create_object(name, {"Type": "Part::Box", "Name": "T1"})
        second = self.engine.begin_turn_snapshot()
        self.engine.create_object(name, {"Type": "Part::Box", "Name": "T2"})
        third = self.engine.begin_turn_snapshot()
        self.engine.create_object(name, {"Type": "Part::Box", "Name": "T3"})
        self.assertTrue(self.engine.revert_turn(third)["success"])
        self.assertEqual(self.names(doc), ["T1", "T2"])
        self.assertTrue(self.engine.revert_turn(second)["success"])
        self.assertEqual(self.names(doc), ["T1"])
        res = self.engine.revert_turn(first)
        self.assertTrue(res["success"], res)
        self.assertIn(name, res["closed_docs"])

    def test_older_turn_leaves_a_document_a_later_turn_changed(self):
        doc = _new_doc("SharedDoc")
        first = self.engine.begin_turn_snapshot()  # turn 1 only chats
        second = self.engine.begin_turn_snapshot()
        self.engine.create_object("SharedDoc", {"Type": "Part::Box", "Name": "Late"})
        res = self.engine.revert_turn(first)
        self.assertTrue(res["success"], res)  # turn 1 changed nothing to undo
        self.assertEqual(self.names(doc), ["Late"])
        res = self.engine.revert_turn(second)
        self.assertTrue(res["success"], res)
        self.assertEqual(self.names(doc), [])

    def test_a_user_edit_on_top_stops_the_undo(self):
        doc = _new_doc("UserEditsDoc")
        snap = self.engine.begin_turn_snapshot()
        self.engine.create_object("UserEditsDoc", {"Type": "Part::Box", "Name": "TurnBox"})
        doc.openTransaction("Move")
        doc.getObject("TurnBox").Length = 33
        doc.commitTransaction()
        res = self.engine.revert_turn(snap)
        self.assertFalse(res["success"], res)
        self.assertTrue(res["partial"])
        self.assertIn("Move", res["error"])
        self.assertIsNotNone(doc.getObject("TurnBox"))
        self.assertEqual(float(doc.getObject("TurnBox").Length), 33.0)


class CloseTurnDocumentsTest(_DocTestCase):
    """P0 #3: the panel's checkpoint fallback closes the documents a turn created only through the engine's ownership
    rules (``close_turn_documents``), never every document that was not open at turn start."""

    def test_only_the_agents_own_unsaved_document_is_closed(self):
        before = _new_doc("CtdOpenBefore")
        snap = self.engine.begin_turn_snapshot()
        mine = self.engine.create_document("CtdAgentMade")["document_name"]
        App.getDocument(mine).addObject("Part::Box", "Box")
        saved = self.engine.create_document("CtdAgentSaved")["document_name"]
        App.getDocument(saved).addObject("Part::Box", "Box")
        App.getDocument(saved).saveAs(os.path.join(tempfile.mkdtemp(prefix="oscar_ctd_"), "saved.FCStd"))
        user = _new_doc("CtdUserDoc")
        user.openTransaction("Pad")
        user.addObject("Part::Box", "Unsaved")
        user.commitTransaction()
        report = self.engine.close_turn_documents(snap)
        self.assertEqual(report["closed_docs"], [mine])
        self.assertNotIn(mine, App.listDocuments())
        kept = {k["doc"]: k["reason"] for k in report["kept_docs"]}
        self.assertEqual(kept, {saved: "it was saved to disk", user.Name: "it holds changes made outside Oscar"})
        for name in (before.Name, saved, user.Name):
            self.assertIn(name, App.listDocuments())
        self.assertFalse(report["success"])          # the turn's own saved document is still open: said, not hidden
        self.assertTrue(any(saved in w for w in report["warnings"]))

    def test_a_document_of_a_later_turn_is_left_alone(self):
        first = self.engine.begin_turn_snapshot()
        self.engine.begin_turn_snapshot()            # turn 2 starts: turn 1's snapshot is closed out
        later = self.engine.create_document("CtdLaterTurn")["document_name"]
        report = self.engine.close_turn_documents(first)
        self.assertIn(later, App.listDocuments())
        self.assertEqual(report["closed_docs"], [])
        self.assertTrue(report["success"], report)

    def test_a_bad_snapshot_closes_nothing(self):
        made = self.engine.create_document("CtdNoSnapshot")["document_name"]
        for bad in ({}, None, {"success": False, "docs": {}}, {"success": True, "docs": "nope"}, "snapshot"):
            report = self.engine.close_turn_documents(bad)
            self.assertFalse(report["success"], bad)
            self.assertIn("no turn snapshot", report["error"])
            self.assertEqual(report["closed_docs"], [])
        self.assertIn(made, App.listDocuments())


class RevertUndoCapTest(_DocTestCase):
    """cad-engine-07: FreeCAD keeps only 20 undo steps; a revert that cannot
    reach the turn's start must say so instead of reporting success."""

    def _turn(self, doc_name, count, prefix="Turn"):
        for i in range(count):
            res = self.engine.create_object(
                doc_name, {"Type": "Part::Box", "Name": f"{prefix}{i}"}
            )
            self.assertTrue(res["success"], res)

    def test_turn_longer_than_the_history_is_a_partial_revert(self):
        doc = _new_doc("Trunc")
        _user_entries(doc, 10)
        snap = self.engine.begin_turn_snapshot()
        self._turn("Trunc", 25)
        res = self.engine.revert_turn(snap)
        self.assertFalse(res["success"], res)
        self.assertTrue(res["partial"])
        self.assertEqual(res["undone"], {"Trunc": 10})
        self.assertIn("undo history", res["error"])
        left = [n for n in self.names(doc) if n.startswith("Turn")]
        self.assertTrue(left)  # honest: some of the turn is still there

    def test_a_second_revert_of_an_overrun_turn_is_still_partial(self):
        # engine-04: the first click undoes what the history still holds and
        # warns; the card stays revertible, so the NEXT click used to find
        # nothing to pop and report a clean success with the objects there
        doc = _new_doc("TruncTwice")
        _user_entries(doc, 10)
        snap = self.engine.begin_turn_snapshot()
        self._turn("TruncTwice", 25)
        first = self.engine.revert_turn(snap)
        self.assertFalse(first["success"], first)
        left = [n for n in self.names(doc) if n.startswith("Turn")]
        self.assertTrue(left)
        second = self.engine.revert_turn(snap)
        self.assertFalse(second["success"], second)
        self.assertTrue(second["partial"])
        self.assertIn("undo history", second["error"])
        self.assertEqual([n for n in self.names(doc) if n.startswith("Turn")], left)

    def test_a_second_revert_is_clean_once_the_document_matches_again(self):
        doc = _new_doc("TruncHealed")
        _user_entries(doc, 10)
        snap = self.engine.begin_turn_snapshot()
        self._turn("TruncHealed", 25)
        self.assertFalse(self.engine.revert_turn(snap)["success"])
        doc.UndoMode = 0  # history cleared, so the tidying below leaves no step
        for name in [n for n in self.names(doc) if n.startswith("Turn")]:
            doc.removeObject(name)  # the user tidied the leftovers by hand
        res = self.engine.revert_turn(snap)
        self.assertTrue(res["success"], res)

    def test_history_already_full_at_turn_start_is_reported(self):
        doc = _new_doc("Full")
        _user_entries(doc, 20)
        self.assertEqual(doc.UndoCount, 20)
        snap = self.engine.begin_turn_snapshot()
        self._turn("Full", 5)
        res = self.engine.revert_turn(snap)
        self.assertFalse(res["success"], res)
        self.assertTrue(res["partial"])
        self.assertIn("undo history", res["error"])
        self.assertEqual(
            len([n for n in self.names(doc) if n.startswith("Turn")]), 5
        )

    def test_a_turn_that_fits_still_succeeds(self):
        doc = _new_doc("Fits")
        _user_entries(doc, 10)
        snap = self.engine.begin_turn_snapshot()
        self._turn("Fits", 5)
        res = self.engine.revert_turn(snap)
        self.assertTrue(res["success"], res)
        self.assertFalse(res.get("partial"))
        self.assertEqual(len(self.names(doc)), 10)

    def test_a_turn_that_exactly_fills_the_history_is_not_a_false_alarm(self):
        doc = _new_doc("ExactFit")
        _user_entries(doc, 10)
        snap = self.engine.begin_turn_snapshot()
        self._turn("ExactFit", 10)
        self.assertEqual(doc.UndoCount, 20)
        res = self.engine.revert_turn(snap)
        self.assertTrue(res["success"], res)
        self.assertEqual(len(self.names(doc)), 10)

    def test_an_untouched_full_history_elsewhere_is_not_a_warning(self):
        full = _new_doc("OtherFull")
        _user_entries(full, 20)
        _new_doc("Working")
        snap = self.engine.begin_turn_snapshot()
        self.engine.create_object("Working", {"Type": "Part::Box", "Name": "One"})
        res = self.engine.revert_turn(snap)
        self.assertTrue(res["success"], res)
        self.assertFalse(res["warnings"])
        self.assertEqual(len(self.names(full)), 20)


# ---------------------------------------------------------------------------
# tools-01: reload_document


class ReloadDocumentTest(_DocTestCase):
    def _saved(self, name):
        doc = App.newDocument(name)
        doc.addObject("Part::Box", "Box")
        doc.recompute()
        path = os.path.join(tempfile.mkdtemp(prefix="oscar_fix_engine_"), f"{name}.FCStd")
        doc.saveAs(path)
        return path

    def test_reload_reopens_the_document(self):
        path = self._saved("Gear_System")
        res = self.engine.reload_document("Gear_System")
        self.assertTrue(res["success"], res)
        docs = App.listDocuments()
        self.assertIn(res["document_name"], docs)
        self.assertEqual(os.path.normpath(docs[res["document_name"]].FileName),
                         os.path.normpath(path))
        self.assertIsNotNone(docs[res["document_name"]].getObject("Box"))

    def test_reload_accepts_a_near_miss_name(self):
        self._saved("Gear_Near")
        res = self.engine.reload_document("gear_ne")
        self.assertTrue(res["success"], res)
        self.assertEqual(res["document_name"], "Gear_Near")
        self.assertIn("Gear_Near", App.listDocuments())

    def test_failed_reopen_says_the_document_was_closed(self):
        self._saved("Gear_Lost")
        with mock.patch.object(App, "openDocument", side_effect=RuntimeError("boom")):
            res = self.engine.reload_document("Gear_Lost")
        self.assertFalse(res["success"])
        self.assertIn("was closed", res["error"])
        self.assertIn("boom", res["error"])

    # engine-01: the document is the only good copy until the file is known
    # to be readable, and its unsaved edits are the user's to give up.

    def test_a_truncated_file_leaves_the_open_document_alone(self):
        path = self._saved("Gear_Torn")
        doc = App.getDocument("Gear_Torn")
        doc.addObject("Part::Cylinder", "Unsaved")
        with open(path, "r+b") as f:
            f.truncate(50)  # a headless run killed mid-write
        res = self.engine.reload_document("Gear_Torn")
        self.assertFalse(res["success"], res)
        self.assertIn("Gear_Torn", App.listDocuments())
        self.assertIsNotNone(App.getDocument("Gear_Torn").getObject("Unsaved"))
        self.assertNotIn("was closed", res["error"])
        self.assertIn("still open", res["error"])

    def test_a_deleted_file_leaves_the_open_document_alone(self):
        path = self._saved("Gear_Gone")
        os.remove(path)
        res = self.engine.reload_document("Gear_Gone")
        self.assertFalse(res["success"], res)
        self.assertIn("Gear_Gone", App.listDocuments())
        self.assertIsNotNone(App.getDocument("Gear_Gone").getObject("Box"))

    def _gui_modified(self, modified: bool):
        return mock.patch.object(
            sys.modules["FreeCADGui"], "getDocument", create=True,
            side_effect=lambda name: types.SimpleNamespace(Modified=modified),
        )

    def test_unsaved_changes_are_kept_unless_the_caller_says_otherwise(self):
        self._saved("Gear_Dirty")
        App.getDocument("Gear_Dirty").addObject("Part::Cylinder", "Unsaved")
        with self._gui_modified(True):
            res = self.engine.reload_document("Gear_Dirty")
            self.assertFalse(res["success"], res)
            self.assertIn("unsaved", res["error"].lower())
            self.assertIsNotNone(App.getDocument("Gear_Dirty").getObject("Unsaved"))
            res = self.engine.reload_document("Gear_Dirty", discard_unsaved=True)
        self.assertTrue(res["success"], res)
        self.assertIn("discarded", res["warning"])
        self.assertIsNone(App.getDocument("Gear_Dirty").getObject("Unsaved"))

    def test_a_clean_document_reloads_without_a_say_so(self):
        self._saved("Gear_Clean")
        with self._gui_modified(False):
            res = self.engine.reload_document("Gear_Clean")
        self.assertTrue(res["success"], res)
        self.assertNotIn("warning", res)


# ---------------------------------------------------------------------------
# tools-05: Draft polygons and wires


class DraftGeometryTest(_DocTestCase):
    def _box(self, doc_name, spec):
        res = self.engine.create_object(doc_name, spec)
        self.assertTrue(res["success"], res)
        doc = App.getDocument(res["document_name"])
        doc.recompute()
        return doc.getObject(res["object_name"]).Shape.BoundBox

    def test_regular_polygon_honours_radius(self):
        _new_doc("DraftDoc")
        box = self._box("DraftDoc", {
            "Type": "Draft::Polygon", "Name": "Hex",
            "Properties": {"FacesNumber": 6, "Radius": 50},
        })
        self.assertAlmostEqual(box.XLength, 100.0, delta=0.01)
        self.assertAlmostEqual(box.YLength, 86.6, delta=0.1)

    def test_wire_uses_its_points_and_closed_flag(self):
        _new_doc("DraftDoc")
        box = self._box("DraftDoc", {
            "Type": "Draft::Wire", "Name": "Tri",
            "Properties": {
                "Points": [{"x": 0, "y": 0}, {"x": 40, "y": 0}, {"x": 40, "y": 20}],
                "Closed": True,
            },
        })
        self.assertAlmostEqual(box.XLength, 40.0, delta=0.01)
        self.assertAlmostEqual(box.YLength, 20.0, delta=0.01)

    def test_wire_accepts_list_points(self):
        _new_doc("DraftDoc")
        box = self._box("DraftDoc", {
            "Type": "Draft::Wire", "Name": "L",
            "Properties": {"Points": [[0, 0, 0], [30, 0, 0], [30, 10, 0]]},
        })
        self.assertAlmostEqual(box.XLength, 30.0, delta=0.01)
        self.assertAlmostEqual(box.YLength, 10.0, delta=0.01)

    def test_polygon_with_explicit_points_keeps_them(self):
        _new_doc("DraftDoc")
        box = self._box("DraftDoc", {
            "Type": "Draft::Polygon", "Name": "Sq",
            "Properties": {"Points": [
                {"x": 0, "y": 0}, {"x": 30, "y": 0},
                {"x": 30, "y": 30}, {"x": 0, "y": 30},
            ]},
        })
        self.assertAlmostEqual(box.XLength, 30.0, delta=0.01)
        self.assertAlmostEqual(box.YLength, 30.0, delta=0.01)

    def test_wire_needs_at_least_two_points(self):
        _new_doc("DraftDoc")
        res = self.engine.create_object("DraftDoc", {
            "Type": "Draft::Wire", "Name": "Bad", "Properties": {"Points": [[0, 0, 0]]},
        })
        self.assertFalse(res["success"])
        self.assertIn("Points", res["error"])


# ---------------------------------------------------------------------------
# cad-engine-11: edits are atomic


class EditAtomicityTest(_DocTestCase):
    def _each_undo_mode(self):
        for undo in (False, True):
            with self.subTest(undo_mode=int(undo)):
                doc = App.newDocument(f"EditDoc{int(undo)}")
                doc.UndoMode = int(undo)
                self.engine.create_object(doc.Name, {
                    "Type": "Part::Box", "Name": "B4", "Properties": {"Length": 50},
                })
                yield doc

    def test_failed_edit_leaves_earlier_properties_untouched(self):
        for doc in self._each_undo_mode():
            res = self.engine.edit_object(doc.Name, "B4", {"Length": 99, "Bogus": 1})
            self.assertFalse(res["success"])
            self.assertIn("Bogus", res["error"])
            self.assertEqual(float(doc.getObject("B4").Length), 50.0)

    def test_edit_that_invalidates_the_object_is_rolled_back(self):
        for doc in self._each_undo_mode():
            res = self.engine.edit_object(doc.Name, "B4", {"Length": "-5 mm"})
            self.assertFalse(res["success"], res)
            self.assertIn("rolled back", res["error"])
            box = doc.getObject("B4")
            self.assertEqual(float(box.Length), 50.0)
            self.assertNotIn("Invalid", box.State)

    def test_edit_that_repairs_an_invalid_object_succeeds(self):
        doc = _new_doc("RepairDoc")
        self.engine.create_object("RepairDoc", {"Type": "Part::Box", "Name": "B"})
        box = doc.getObject("B")
        box.Length = -5
        doc.recompute()
        self.assertIn("Invalid", box.State)
        res = self.engine.edit_object("RepairDoc", "B", {"Length": 10})
        self.assertTrue(res["success"], res)
        self.assertEqual(float(box.Length), 10.0)

    def test_edit_of_an_already_invalid_object_warns(self):
        doc = _new_doc("StillBadDoc")
        self.engine.create_object("StillBadDoc", {"Type": "Part::Box", "Name": "B"})
        box = doc.getObject("B")
        box.Length = -5
        doc.recompute()
        res = self.engine.edit_object("StillBadDoc", "B", {"Width": 12})
        self.assertTrue(res["success"], res)
        self.assertIn("warning", res)
        self.assertEqual(float(box.Width), 12.0)


class ApplyPropertiesAtomicTest(_DocTestCase):
    def test_a_failing_property_restores_the_earlier_ones(self):
        doc = App.newDocument("AtomicDoc")
        box = doc.addObject("Part::Box", "B")
        with self.assertRaises(ValueError):
            apply_properties(box, {"Length": 77, "Width": 33, "Bogus": 1})
        self.assertEqual(float(box.Length), 10.0)
        self.assertEqual(float(box.Width), 10.0)

    def test_undo_log_records_what_was_applied(self):
        doc = App.newDocument("AtomicDoc2")
        box = doc.addObject("Part::Box", "B")
        log = []
        apply_properties(box, {"Length": 77}, undo_log=log)
        self.assertEqual(len(log), 1)
        self.assertEqual(float(box.Length), 77.0)


# ---------------------------------------------------------------------------
# cad-engine-09: placements


class PlacementTest(_DocTestCase):
    def _box(self, name="PlaceDoc"):
        doc = App.newDocument(name)
        return doc, doc.addObject("Part::Box", "B")

    def test_base_edit_keeps_the_rotation(self):
        _doc, box = self._box()
        box.Placement = App.Placement(App.Vector(1, 2, 3), App.Rotation(App.Vector(1, 0, 0), 45))
        apply_properties(box, {"Placement": {"Base": {"x": 9, "y": 9, "z": 9}}})
        self.assertEqual(tuple(box.Placement.Base), (9, 9, 9))
        self.assertAlmostEqual(box.Placement.Rotation.Angle, math.radians(45), places=6)
        self.assertAlmostEqual(abs(box.Placement.Rotation.Axis.x), 1.0, places=6)

    def test_rotation_edit_keeps_the_position(self):
        _doc, box = self._box()
        box.Placement = App.Placement(App.Vector(5, 6, 7), App.Rotation())
        apply_properties(box, {"Placement": {"Rotation": {
            "Axis": {"x": 0, "y": 0, "z": 1}, "Angle": 90}}})
        self.assertEqual(tuple(box.Placement.Base), (5, 6, 7))
        self.assertAlmostEqual(box.Placement.Rotation.Angle, math.radians(90), places=6)

    def test_empty_parts_leave_the_current_value(self):
        _doc, box = self._box()
        box.Placement = App.Placement(App.Vector(1, 2, 3), App.Rotation(App.Vector(0, 0, 1), 30))
        apply_properties(box, {"Placement": {"Base": {"x": 7, "y": 8, "z": 9}, "Rotation": {}}})
        self.assertEqual(tuple(box.Placement.Base), (7, 8, 9))
        self.assertAlmostEqual(box.Placement.Rotation.Angle, math.radians(30), places=6)

    def test_base_without_z_defaults_to_zero(self):
        _doc, box = self._box()
        apply_properties(box, {"Placement": {"Base": {"x": 10, "y": 5}}})
        self.assertEqual(tuple(box.Placement.Base), (10, 5, 0))

    def test_list_axis_is_honoured(self):
        _doc, box = self._box()
        apply_properties(box, {"Placement": {
            "Base": [3, 4, 5], "Rotation": {"Axis": [1, 0, 0], "Angle": 90}}})
        self.assertEqual(tuple(box.Placement.Base), (3, 4, 5))
        axis = box.Placement.Rotation.Axis
        self.assertAlmostEqual(axis.x, 1.0, places=6)
        self.assertAlmostEqual(axis.z, 0.0, places=6)

    def test_axis_without_z_is_honoured(self):
        _doc, box = self._box()
        apply_properties(box, {"Placement": {
            "Rotation": {"Axis": {"x": 1, "y": 0}, "Angle": 90}}})
        self.assertAlmostEqual(box.Placement.Rotation.Axis.x, 1.0, places=6)

    def test_malformed_placements_are_errors(self):
        _doc, box = self._box()
        box.Placement = App.Placement(App.Vector(1, 1, 1), App.Rotation())
        for bad in (
            {"Base": "oops"},
            {"Base": {"x": "a", "y": 1}},
            {"Rotation": {"Axis": {"x": 0, "y": 0, "z": 0}, "Angle": 45}},
            {"Rotation": {"Axis": [1, 0, 0], "Angle": "ninety"}},
            {"Rotation": {"Axis": [1, 0, 0]}},
            # (a bare {"x", "y", "z"} is NOT malformed here: it is the position-only form, see
            # PropertiesTest.test_bare_xyz_is_a_position_only)
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    apply_properties(box, {"Placement": bad})
                self.assertEqual(tuple(box.Placement.Base), (1, 1, 1))

    def test_bad_placement_edit_reports_failure_and_changes_nothing(self):
        doc = _new_doc("PlaceEditDoc")
        self.engine.create_object("PlaceEditDoc", {"Type": "Part::Box", "Name": "B"})
        box = doc.getObject("B")
        box.Placement = App.Placement(App.Vector(1, 1, 1), App.Rotation())
        res = self.engine.edit_object("PlaceEditDoc", "B", {"Placement": {"Base": "oops"}})
        self.assertFalse(res["success"])
        self.assertEqual(tuple(box.Placement.Base), (1, 1, 1))

    def test_full_placement_still_works_on_creation(self):
        _new_doc("PlaceNewDoc")
        res = self.engine.create_object("PlaceNewDoc", {
            "Type": "Part::Box", "Name": "B",
            "Properties": {"Placement": {
                "Base": {"x": 10, "y": 20, "z": 30},
                "Rotation": {"Axis": {"x": 0, "y": 0, "z": 1}, "Angle": 90}}},
        })
        self.assertTrue(res["success"], res)
        placement = App.getDocument("PlaceNewDoc").getObject("B").Placement
        self.assertEqual(tuple(placement.Base), (10, 20, 30))
        self.assertAlmostEqual(placement.Rotation.Angle, math.radians(90), places=6)


# ---------------------------------------------------------------------------
# cad-engine-10 / cad-engine-12: PartDesign bodies and deletes


class BodyTargetTest(_DocTestCase):
    def _two_bodies(self):
        doc = _new_doc("BodiesDoc")
        for name in ("BodyA", "BodyB"):
            res = self.engine.create_object(
                "BodiesDoc", {"Type": "PartDesign::Body", "Name": name})
            self.assertTrue(res["success"], res)
        return doc

    def test_feature_lands_in_the_most_recent_body(self):
        doc = self._two_bodies()
        res = self.engine.create_object(
            "BodiesDoc", {"Type": "PartDesign::AdditiveBox", "Name": "Feat"})
        self.assertTrue(res["success"], res)
        self.assertEqual([o.Name for o in doc.getObject("BodyB").Group if o.Name == "Feat"],
                         ["Feat"])
        self.assertNotIn("Feat", [o.Name for o in doc.getObject("BodyA").Group])

    def test_feature_can_name_its_body(self):
        doc = self._two_bodies()
        res = self.engine.create_object("BodiesDoc", {
            "Type": "PartDesign::AdditiveBox", "Name": "Feat",
            "Properties": {"Body": "BodyA"},
        })
        self.assertTrue(res["success"], res)
        self.assertIn("Feat", [o.Name for o in doc.getObject("BodyA").Group])
        self.assertNotIn("Feat", [o.Name for o in doc.getObject("BodyB").Group])

    def test_unknown_body_is_an_error_and_leaves_nothing(self):
        doc = self._two_bodies()
        before = self.names(doc)
        res = self.engine.create_object("BodiesDoc", {
            "Type": "PartDesign::AdditiveBox", "Name": "Feat",
            "Properties": {"Body": "NoSuchBody"},
        })
        self.assertFalse(res["success"])
        self.assertIn("NoSuchBody", res["error"])
        self.assertEqual(self.names(doc), before)

    def test_a_new_body_does_not_nest_in_the_first(self):
        doc = self._two_bodies()
        self.assertNotIn("BodyB", [o.Name for o in doc.getObject("BodyA").Group])
        self.assertIsNone(doc.getObject("BodyB").getParentGeoFeatureGroup())

    # engine-07: a Body made through the API is not the GUI's active Body, so
    # the user's active one used to win over the Body the agent just made.

    def _gui_active_body(self, doc, holder: dict):
        """A GUI whose active PartDesign Body is ``holder["name"]``."""
        view = types.SimpleNamespace(
            getActiveObject=lambda kind: doc.getObject(holder["name"]) if kind == "pdbody" else None
        )
        active_doc = types.SimpleNamespace(Document=doc, ActiveView=view)
        return mock.patch.object(
            sys.modules["FreeCADGui"], "ActiveDocument", active_doc, create=True)

    def _agent_body(self, name):
        res = self.engine.create_object(
            "ActiveBodyDoc", {"Type": "PartDesign::Body", "Name": name})
        self.assertTrue(res["success"], res)

    def _feature(self, name):
        res = self.engine.create_object(
            "ActiveBodyDoc", {"Type": "PartDesign::AdditiveBox", "Name": name})
        self.assertTrue(res["success"], res)

    def test_a_body_the_agent_just_made_beats_the_users_active_body(self):
        doc = _new_doc("ActiveBodyDoc")
        doc.addObject("PartDesign::Body", "Bracket")  # the user's part, active in the GUI
        with self._gui_active_body(doc, {"name": "Bracket"}):
            self._agent_body("Plate")
            self._feature("PlateBox")
        self.assertEqual([o.Name for o in doc.getObject("Plate").Group], ["PlateBox"])
        self.assertEqual(doc.getObject("Bracket").Group, [])

    def test_the_users_active_body_is_used_when_the_agent_made_none(self):
        doc = _new_doc("ActiveBodyDoc")
        doc.addObject("PartDesign::Body", "Bracket")
        doc.addObject("PartDesign::Body", "Other")
        with self._gui_active_body(doc, {"name": "Bracket"}):
            self._feature("Hole")
        self.assertEqual([o.Name for o in doc.getObject("Bracket").Group], ["Hole"])

    def test_a_body_the_user_activates_afterwards_wins(self):
        doc = _new_doc("ActiveBodyDoc")
        doc.addObject("PartDesign::Body", "Bracket")
        doc.addObject("PartDesign::Body", "Third")
        active = {"name": "Bracket"}
        with self._gui_active_body(doc, active):
            self._agent_body("Plate")
            active["name"] = "Third"  # the user switched to another Body
            self._feature("Later")
        self.assertEqual([o.Name for o in doc.getObject("Third").Group], ["Later"])
        self.assertEqual(doc.getObject("Plate").Group, [])

    def test_a_body_that_was_deleted_is_not_a_target(self):
        doc = _new_doc("ActiveBodyDoc")
        doc.addObject("PartDesign::Body", "Bracket")
        with self._gui_active_body(doc, {"name": "Bracket"}):
            self._agent_body("Plate")
            self.assertTrue(self.engine.delete_object("ActiveBodyDoc", "Plate")["success"])
            self._feature("Back")
        self.assertEqual([o.Name for o in doc.getObject("Bracket").Group], ["Back"])

    def test_a_feature_still_follows_the_body_that_owns_its_profile(self):
        # what the feature builds on decides first; "newest Body" is only for
        # features that build on nothing
        doc = _new_doc("ActiveBodyDoc")
        self._agent_body("First")
        self._feature("OnFirst")
        self._agent_body("Second")
        res = self.engine.create_object("ActiveBodyDoc", {
            "Type": "PartDesign::AdditiveBox", "Name": "More",
            "Properties": {"BaseFeature": "OnFirst"},
        })
        self.assertTrue(res["success"], res)
        self.assertIn("More", [o.Name for o in doc.getObject("First").Group])


class DeleteObjectTest(_DocTestCase):
    def test_deleting_a_body_removes_its_features(self):
        doc = _new_doc("DelBodyDoc")
        self.engine.create_object("DelBodyDoc", {"Type": "PartDesign::Body", "Name": "Body"})
        self.engine.create_object("DelBodyDoc", {"Type": "PartDesign::AdditiveBox", "Name": "Feat"})
        res = self.engine.delete_object("DelBodyDoc", "Body")
        self.assertTrue(res["success"], res)
        self.assertIn("Feat", res["removed_children"])
        self.assertIsNone(doc.getObject("Feat"))
        self.assertEqual(self.names(doc), [])

    def test_deleting_an_input_of_a_boolean_is_refused(self):
        doc = _new_doc("DelInputDoc")
        for name in ("B5", "B6"):
            self.engine.create_object("DelInputDoc", {"Type": "Part::Box", "Name": name})
        self.engine.create_object("DelInputDoc", {
            "Type": "Part::Cut", "Name": "Cut",
            "Properties": {"Base": "B5", "Tool": "B6"},
        })
        res = self.engine.delete_object("DelInputDoc", "B5")
        self.assertFalse(res["success"])
        self.assertIn("Cut", res["error"])
        self.assertIsNotNone(doc.getObject("B5"))
        self.assertNotIn("Invalid", doc.getObject("Cut").State)
        # deleting the consumer first is the right order
        self.assertTrue(self.engine.delete_object("DelInputDoc", "Cut")["success"])
        self.assertTrue(self.engine.delete_object("DelInputDoc", "B5")["success"])

    def test_force_deletes_a_consumed_object(self):
        doc = _new_doc("DelForceDoc")
        for name in ("B5", "B6"):
            self.engine.create_object("DelForceDoc", {"Type": "Part::Box", "Name": name})
        self.engine.create_object("DelForceDoc", {
            "Type": "Part::Cut", "Name": "Cut",
            "Properties": {"Base": "B5", "Tool": "B6"},
        })
        res = self.engine.delete_object("DelForceDoc", "B5", force=True)
        self.assertTrue(res["success"], res)
        self.assertIsNone(doc.getObject("B5"))

    # engine-02: in a Body the next feature links the previous one (BaseFeature)
    # and the Body relinks the chain when one goes, so that link is no reason
    # to refuse.

    def _body_chain(self, doc_name):
        doc = _new_doc(doc_name)
        self.engine.create_object(doc_name, {"Type": "PartDesign::Body", "Name": "Body"})
        for name in ("F1", "F2", "F3"):
            res = self.engine.create_object(
                doc_name, {"Type": "PartDesign::AdditiveBox", "Name": name})
            self.assertTrue(res["success"], res)
        return doc

    def test_a_middle_body_feature_can_be_deleted(self):
        doc = self._body_chain("DelMidDoc")
        res = self.engine.delete_object("DelMidDoc", "F2")
        self.assertTrue(res["success"], res)
        self.assertIsNone(doc.getObject("F2"))
        self.assertEqual(doc.getObject("F3").BaseFeature.Name, "F1")
        self.assertNotIn("Invalid", doc.getObject("F3").State)

    def test_the_first_body_feature_can_be_deleted(self):
        doc = self._body_chain("DelFirstDoc")
        res = self.engine.delete_object("DelFirstDoc", "F1")
        self.assertTrue(res["success"], res)
        self.assertIsNone(doc.getObject("F1"))
        self.assertIsNone(doc.getObject("F2").BaseFeature)
        self.assertEqual(doc.getObject("F3").BaseFeature.Name, "F2")
        self.assertEqual(doc.getObject("Body").Tip.Name, "F3")

    def test_the_last_body_feature_can_be_deleted_and_the_tip_moves(self):
        doc = self._body_chain("DelLastDoc")
        res = self.engine.delete_object("DelLastDoc", "F3")
        self.assertTrue(res["success"], res)
        self.assertEqual(doc.getObject("Body").Tip.Name, "F2")

    def test_a_feature_something_outside_the_chain_uses_is_still_refused(self):
        doc = self._body_chain("DelExtDoc")
        outside = doc.addObject("Part::FeaturePython", "Outside")
        outside.addProperty("App::PropertyLink", "Ref", "Base", "")
        outside.Ref = doc.getObject("F2")
        res = self.engine.delete_object("DelExtDoc", "F2")
        self.assertFalse(res["success"], res)
        self.assertIn("Outside", res["error"])
        self.assertNotIn("F3", res["error"])  # the chain is not the complaint
        self.assertIsNotNone(doc.getObject("F2"))

    def test_an_object_inside_a_container_can_be_deleted(self):
        doc = _new_doc("DelGroupDoc")
        part = doc.addObject("App::Part", "Assembly")
        box = doc.addObject("Part::Box", "Inside")
        part.addObject(box)
        res = self.engine.delete_object("DelGroupDoc", "Inside")
        self.assertTrue(res["success"], res)
        self.assertIsNone(doc.getObject("Inside"))


# ---------------------------------------------------------------------------
# tools-09: failed GUI tasks must not look like results


class DispatchErrorHandlingTest(_DocTestCase):
    def _failing(self):
        return mock.patch.object(dispatch, "dispatch", return_value=("boom", True))

    def test_screenshot_is_none_when_the_gui_task_failed(self):
        with self._failing():
            self.assertIsNone(self.engine.screenshot(doc_name="Anything"))

    def test_screenshot_is_none_without_a_gui_in_console(self):
        _new_doc("ShotDoc")
        self.assertIsNone(self.engine.screenshot(doc_name="ShotDoc"))

    def test_listings_are_errors_when_the_gui_task_failed(self):
        # engine-06: [] / None read as "open and empty" / "no such object" to
        # the tools, which sent the model off rebuilding a model that is there
        with self._failing():
            for call in (
                lambda: self.engine.list_documents(),
                lambda: self.engine.get_objects("Any"),
                lambda: self.engine.get_object("Any", "Obj"),
            ):
                with self.assertRaises(RuntimeError) as caught:
                    call()
                self.assertIn("boom", str(caught.exception))

    def test_a_busy_gui_thread_is_reported_as_busy(self):
        busy = {"success": False, "error": "'get_objects' was not started within 60s "
                                           "(the GUI thread stayed busy)"}
        with mock.patch.object(dispatch, "dispatch", return_value=(busy, True)):
            with self.assertRaises(RuntimeError) as caught:
                self.engine.get_objects("Any")
        self.assertIn("GUI thread stayed busy", str(caught.exception))

    def test_a_missing_document_or_object_is_still_just_empty(self):
        _new_doc("PresentDoc").addObject("Part::Box", "Box")
        self.assertEqual(self.engine.get_objects("NoSuchDoc"), [])
        self.assertIsNone(self.engine.get_object("NoSuchDoc", "Box"))
        self.assertIsNone(self.engine.get_object("PresentDoc", "NoSuchObject"))
        self.assertEqual([o["Name"] for o in self.engine.get_objects("PresentDoc")], ["Box"])


# ---------------------------------------------------------------------------
# cad-engine-08: BREP import


class BrepImportTest(_DocTestCase):
    def _brep(self, ext):
        import Part

        path = os.path.join(tempfile.mkdtemp(prefix="oscar_fix_engine_"), f"block.{ext}")
        Part.makeBox(10, 20, 30).exportBrep(path)
        return path

    def test_brep_files_import_as_solids(self):
        for ext in ("brep", "brp"):
            with self.subTest(ext=ext):
                name = f"BrepDoc{ext}"
                App.newDocument(name)
                res = self.engine.import_file(self._brep(ext), name)
                self.assertTrue(res["success"], res)
                self.assertEqual(res["count"], 1)
                shape = App.getDocument(name).getObject(res["objects"][0]).Shape
                self.assertAlmostEqual(shape.Volume, 6000.0, places=3)


# ---------------------------------------------------------------------------
# cad-engine-05 / cad-engine-06 (tools-04): FEM — Gmsh meshes and the
# CalculiX run. The real Gmsh/CalculiX binaries ship with the app but are
# not assumed: tests that need them skip, the rest drive stand-ins.

try:
    import ObjectsFem
except Exception:  # FEM workbench not importable in this build
    ObjectsFem = None

_HAVE_GMSH = shutil.which("gmsh") is not None
_HAVE_CCX = shutil.which("ccx") is not None


def _fem_model(doc_name: str, solver: bool = False, material: bool = True,
               analysis_name: str = "Analysis", solver_name: str = "SolverCcx"):
    """A 10 mm Part::Box with an analysis, steel and fixed/force constraints."""
    doc = _new_doc(doc_name)
    box = doc.addObject("Part::Box", "Box")
    box.Length = box.Width = box.Height = 10
    doc.recompute()
    analysis = ObjectsFem.makeAnalysis(doc, analysis_name)
    if solver:
        analysis.addObject(ObjectsFem.makeSolverCalculiXCcxTools(doc, solver_name))
    if material:
        mat = ObjectsFem.makeMaterialSolid(doc, "Steel")
        mat.Material = {
            "Name": "Steel", "YoungsModulus": "210000 MPa",
            "PoissonRatio": "0.3", "Density": "7900 kg/m^3",
        }
        analysis.addObject(mat)
    fixed = ObjectsFem.makeConstraintFixed(doc, "Fix")
    fixed.References = [(box, "Face1")]
    analysis.addObject(fixed)
    force = ObjectsFem.makeConstraintForce(doc, "Force")
    force.References = [(box, "Face2")]
    force.Force = 1000
    force.Direction = (box, ["Edge1"])
    analysis.addObject(force)
    return doc, analysis


def _solvers(analysis):
    return [m for m in analysis.Group
            if getattr(getattr(m, "Proxy", None), "Type", "") == "Fem::SolverCcxTools"]


def _fem_temp_dirs() -> set:
    return {p for p in Path(tempfile.gettempdir()).glob("oscar_fem_*")}


class _FemCase(_DocTestCase):
    """Removes the working directories Gmsh and CalculiX make for themselves."""

    def setUp(self):
        super().setUp()
        tmp = Path(tempfile.gettempdir())
        before = set(tmp.glob("fcfem_*")) | set(tmp.glob("fem_*"))

        def cleanup():
            for path in (set(tmp.glob("fcfem_*")) | set(tmp.glob("fem_*"))) - before:
                shutil.rmtree(path, ignore_errors=True)

        self.addCleanup(cleanup)


@unittest.skipIf(ObjectsFem is None, "FEM workbench not available")
class FemRunTest(_FemCase):
    """run_fem against a solver stand-in: ``ccxtools.FemToolsCcx.run`` returns
    True on success and False on failure, and does its own setup."""

    def setUp(self):
        super().setUp()
        from femtools import ccxtools

        self.ccxtools = ccxtools

    def _patched(self, run_result):
        """No prerequisites problem and a canned ``run()`` outcome; returns
        the run mock."""
        run = mock.Mock(return_value=run_result)
        self.enterContext(mock.patch.object(
            self.ccxtools.FemToolsCcx, "check_prerequisites", return_value=""))
        self.enterContext(mock.patch.object(self.ccxtools.FemToolsCcx, "run", run))
        return run

    def test_failed_solver_run_is_a_failure(self):
        _fem_model("FemFail", solver=True)
        run = self._patched(False)  # CalculiX failed; False == 0 once read as success
        res = self.engine.run_fem("FemFail", "Analysis", 60)
        run.assert_called_once()  # the solver was reached, not a setup error
        self.assertFalse(res["success"], res)
        self.assertIn("CalculiX", res["error"])

    def test_the_solvers_own_error_lines_reach_the_model(self):
        _fem_model("FemFailWhy", solver=True)

        def run(fea):
            fea.ccx_stdout = "setup\n *ERROR in calinput: no material was assigned\n"
            return False

        self.enterContext(mock.patch.object(
            self.ccxtools.FemToolsCcx, "check_prerequisites", return_value=""))
        self.enterContext(mock.patch.object(
            self.ccxtools.FemToolsCcx, "run", autospec=True, side_effect=run))
        res = self.engine.run_fem("FemFailWhy", "Analysis", 60)
        self.assertFalse(res["success"], res)
        self.assertIn("no material was assigned", res["error"])

    def test_successful_solver_run_is_a_success(self):
        _fem_model("FemOk", solver=True)
        self._patched(True)  # used to read as 'solver failed: True'
        res = self.engine.run_fem("FemOk", "Analysis", 60)
        self.assertTrue(res["success"], res)
        self.assertNotIn("error", res)

    def test_a_missing_solver_is_created_once_and_reused(self):
        doc, analysis = _fem_model("FemNoSolver", solver=False)
        self._patched(True)
        res = self.engine.run_fem("FemNoSolver", "Analysis", 60)
        self.assertTrue(res["success"], res)
        self.assertEqual(len(_solvers(analysis)), 1)
        res = self.engine.run_fem("FemNoSolver", "Analysis", 60)
        self.assertTrue(res["success"], res)
        self.assertEqual(len(_solvers(analysis)), 1)
        self.assertEqual(
            [o.TypeId for o in doc.Objects if o.Name.startswith("SolverCcx")],
            ["Fem::FemSolverObjectPython"],
        )

    def test_an_existing_solver_is_not_duplicated(self):
        doc, analysis = _fem_model("FemHasSolver", solver=True, solver_name="MySolver")
        self._patched(True)
        self.assertTrue(self.engine.run_fem("FemHasSolver", "Analysis", 60)["success"])
        self.assertEqual([m.Name for m in _solvers(analysis)], ["MySolver"])

    def test_the_named_analysis_is_solved_when_the_document_has_several(self):
        doc, first = _fem_model("FemTwo", solver=True, analysis_name="First",
                                solver_name="SolverA")
        second = ObjectsFem.makeAnalysis(doc, "Second")
        second.addObject(ObjectsFem.makeSolverCalculiXCcxTools(doc, "SolverB"))
        seen = []

        def run(fea):
            seen.append((fea.analysis.Name, fea.solver.Name))
            return True

        self.enterContext(mock.patch.object(
            self.ccxtools.FemToolsCcx, "check_prerequisites", return_value=""))
        self.enterContext(mock.patch.object(
            self.ccxtools.FemToolsCcx, "run", autospec=True, side_effect=run))
        res = self.engine.run_fem("FemTwo", "Second", 60)
        self.assertTrue(res["success"], res)
        self.assertEqual(seen, [("Second", "SolverB")])

    def test_missing_prerequisites_are_reported_before_the_solver_starts(self):
        _fem_model("FemNoMat", solver=True, material=False)
        run = mock.Mock(return_value=True)
        self.enterContext(mock.patch.object(self.ccxtools.FemToolsCcx, "run", run))
        res = self.engine.run_fem("FemNoMat", "Analysis", 60)
        self.assertFalse(res["success"], res)
        self.assertIn("material", res["error"].lower())
        # the working directory must be set up before the check, or every
        # analysis fails on it regardless of what is actually missing
        self.assertNotIn("working directory", res["error"].lower())
        run.assert_not_called()

    def test_no_unused_temp_directory_is_left_behind(self):
        # engine-08: the old code made an oscar_fem_ directory the
        # solver never used. It only got that far once it had found or made a
        # solver, which it could not (it asked ObjectsFem for a maker name
        # that does not exist), so a pass proved nothing. Give it that name
        # and the old code leaks its directory and fails the run.
        _fem_model("FemTemp", solver=False)
        self.enterContext(mock.patch.object(
            ObjectsFem, "makeSolverCalculixCcxTools",
            ObjectsFem.makeSolverCalculiXCcxTools, create=True))
        self._patched(True)
        before = _fem_temp_dirs()
        res = self.engine.run_fem("FemTemp", "Analysis", 60)
        self.assertEqual(_fem_temp_dirs() - before, set())
        self.assertTrue(res["success"], res)
        self.assertNotIn("oscar_fem_", res["working_dir"])

    @unittest.skipUnless(_HAVE_GMSH and _HAVE_CCX, "Gmsh/CalculiX not installed")
    def test_solved_analysis_reports_displacement_and_stress(self):
        doc, analysis = _fem_model("FemSolve")
        made = self.engine.create_object("FemSolve", {
            "Type": "Fem::FemMeshGmsh", "Name": "Mesh", "Analysis": "Analysis",
            "Properties": {"Shape": "Box", "CharacteristicLengthMax": 5},
        })
        self.assertTrue(made["success"], made)
        res = self.engine.run_fem("FemSolve", "Analysis", 120)
        self.assertTrue(res["success"], res)
        self.assertGreater(res["max_displacement_mm"], 0)
        self.assertGreater(res["max_von_mises_MPa"], 0)
        self.assertGreater(res["node_count"], 0)
        self.assertEqual(len(_solvers(analysis)), 1)

    @unittest.skipUnless(_HAVE_GMSH and _HAVE_CCX, "Gmsh/CalculiX not installed")
    def test_real_run_without_material_fails_cleanly(self):
        _fem_model("FemSolveNoMat", material=False)
        self.engine.create_object("FemSolveNoMat", {
            "Type": "Fem::FemMeshGmsh", "Name": "Mesh", "Analysis": "Analysis",
            "Properties": {"Shape": "Box", "CharacteristicLengthMax": 5},
        })
        res = self.engine.run_fem("FemSolveNoMat", "Analysis", 120)
        self.assertFalse(res["success"], res)
        self.assertIn("material", res["error"].lower())


class _FakeGmshTools:
    """Stands in for femmesh.gmshtools.GmshTools: the 1.1 constructor takes
    only the mesh object, the analysis comes from its parent group."""

    meshed = True
    constructed: list = []

    def __init__(self, obj):
        self.obj = obj
        self.constructed.append((obj, obj.getParentGroup()))

    def create_mesh(self):
        if self.meshed:
            import Fem

            mesh = Fem.FemMesh()
            for x in (0, 1, 2):
                mesh.addNode(float(x), 0.0, 0.0)
            self.obj.FemMesh = mesh


@unittest.skipIf(ObjectsFem is None, "FEM workbench not available")
class FemGmshTest(_FemCase):
    SPELLINGS = ("Fem::FemMeshGmsh", "Fem::MeshGmsh")

    def _model(self, name):
        doc = _new_doc(name)
        box = doc.addObject("Part::Box", "Box")
        box.Length = box.Width = box.Height = 10
        doc.recompute()
        return doc, ObjectsFem.makeAnalysis(doc, "Analysis")

    def _spec(self, type_name, **props):
        return {"Type": type_name, "Name": "Mesh", "Analysis": "Analysis",
                "Properties": {"Shape": "Box", **props}}

    def _fake(self, meshed=True):
        from femmesh import gmshtools

        _FakeGmshTools.meshed = meshed
        _FakeGmshTools.constructed = []
        self.enterContext(mock.patch.object(gmshtools, "GmshTools", _FakeGmshTools))
        return _FakeGmshTools

    @unittest.skipUnless(_HAVE_GMSH, "Gmsh not installed")
    def test_gmsh_mesh_is_generated_on_creation(self):
        for type_name in self.SPELLINGS:
            with self.subTest(type=type_name):
                doc, analysis = self._model("Gmsh" + type_name.split("::")[1])
                res = self.engine.create_object(
                    doc.Name, self._spec(type_name, CharacteristicLengthMax=5))
                self.assertTrue(res["success"], res)
                mesh = doc.getObject(res["object_name"])
                self.assertGreater(mesh.FemMesh.NodeCount, 0)
                self.assertIn(mesh, analysis.Group)
                self.assertEqual(mesh.Shape.Name, "Box")

    def test_both_type_spellings_reach_the_mesher_with_one_argument(self):
        fake = self._fake()
        for type_name in self.SPELLINGS:
            with self.subTest(type=type_name):
                doc, analysis = self._model("GmshArgs" + type_name.split("::")[1])
                fake.constructed.clear()
                res = self.engine.create_object(doc.Name, self._spec(type_name))
                self.assertTrue(res["success"], res)
                mesh = doc.getObject(res["object_name"])
                # added to the analysis BEFORE meshing: GmshTools reads it from there
                self.assertEqual(fake.constructed, [(mesh, analysis)])
                self.assertEqual(analysis.Group.count(mesh), 1)

    def test_the_documented_part_alias_links_the_geometry(self):
        self._fake()
        doc, analysis = self._model("GmshAlias")
        spec = self._spec("Fem::FemMeshGmsh")
        spec["Properties"] = {"Part": "Box"}
        res = self.engine.create_object(doc.Name, spec)
        self.assertTrue(res["success"], res)
        self.assertEqual(doc.getObject(res["object_name"]).Shape.Name, "Box")

    def test_mesh_without_geometry_is_a_clear_error(self):
        fake = self._fake()
        doc, analysis = self._model("GmshNoShape")
        spec = self._spec("Fem::FemMeshGmsh")
        spec["Properties"] = {}
        res = self.engine.create_object(doc.Name, spec)
        self.assertFalse(res["success"], res)
        self.assertIn("Shape", res["error"])
        self.assertEqual(fake.constructed, [])
        self.assertEqual(self.names(doc), ["Box", "Analysis"])

    def test_an_empty_mesh_is_a_failure_and_leaves_nothing_behind(self):
        self._fake(meshed=False)  # gmsh ran but produced no nodes
        doc, analysis = self._model("GmshEmpty")
        res = self.engine.create_object(doc.Name, self._spec("Fem::FemMeshGmsh"))
        self.assertFalse(res["success"], res)
        self.assertIn("no nodes", res["error"].lower())
        self.assertEqual(self.names(doc), ["Box", "Analysis"])
        self.assertEqual(analysis.Group, [])


if __name__ == "__main__":
    unittest.main()
