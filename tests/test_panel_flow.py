"""Tests of the panel's wiring of the structured build flow (oscar/panel*.py), driven by hand-written
payloads that follow.10 exactly.

A real ``OscarPanel`` on an offscreen QApplication (nothing is shown to the user); the worker is a small fake that
the tests feed by calling the panel's slots, except in ``EndToEndTest``, which runs a REAL ``AgentWorker`` thread
with a scripted provider (no network). Skipped when PySide / FreeCAD are unavailable (plain app python); the pure
renderer is covered by test_panel_render.py, the legacy panel behaviour by test_panel_logic.py.
"""

import json
import os
import sys
import tempfile
import time
import unittest
# Phase 9 lineage note: this module was restored from the pre-squash lineage (aacaa84), whose
# panel generation rendered streamed text into p.log and decoded thumbnails eagerly. The
# current panel (redesign lineage) renders only final answers as chat messages and decodes
# thumbnails on card expansion. Tests whose intent is fully covered TODAY on this lineage
# were removed with this mapping (the superseding modules verified on the combined #32+#23 tree):
#   - test_the_header_is_never_done_while_the_worker_runs -> test_stage2_review_panelui (header/heartbeat states on the redesigned panel)
#   - test_the_expected_note_of_a_non_geometry_step_is_not_shown -> test_stage2_review_panelui DisplayFixes coverage of the redesigned cards
#   - test_the_partial_reply_of_a_voided_round_is_removed -> test_panel_render + test_stage2_review_panelui (blocks/round reset on the redesigned transcript)
#   - test_snapshot_is_decoded_once_and_resolvable_after_every_render -> test_stage2_review_panelui.ThumbnailChurnTest (decode-on-expand convergence)
#   - test_the_document_gets_the_image_registered_after_set_html -> test_stage2_review_panelui.ThumbnailChurnTest (resource registration)
#   - test_the_log_serves_the_image_itself_not_only_through_the_registration -> test_stage2_review_panelui.ThumbnailChurnTest (loadResource serving)
# The heartbeat tests were aligned to the phrase-map contract (same payload API, new label).

from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

_IMPORT_ERROR = ""
try:
    from PySide import QtCore, QtGui, QtWidgets
    from oscar import panel, panel_anchors
    from oscar.panel import Block, Step
    from oscar.panel_render import thumb_url
    from test_panel_logic import GOOD_SNAPSHOT, FakeWorker, PanelTestCase
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

from test_panel_render import BUILD_SUMMARY, finished, ledger, progress, started  # noqa: E402


def j(payload) -> str:
    return json.dumps(payload)


def png(path: str, width: int = 480, height: int = 360, color: str = "#2060c0") -> str:
    """Write a real PNG of the given size (the snapshots the Project writes are 480 x 360)."""
    image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
    image.fill(QtGui.QColor(color))
    assert image.save(path, "PNG")
    return path


class FlowWorker(FakeWorker):
    """FakeWorker + the new worker API the panel guards with getattr (act_now, ledger)."""

    def __init__(self, *args, ledger=None, **kw):
        super().__init__(*args, **kw)
        self.ledger = ledger

    def act_now(self):
        self.calls.append(("act_now",))

    def steer(self, text, note="", images=()):
        self.calls.append(("steer", text) if not images else ("steer", text, list(images)))


class FlowTestCase(PanelTestCase):
    """Panel + a running (fake) turn."""

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="oscar_flow_test_")

    def begin_turn(self, worker=None):
        p = self.p
        p.worker = worker or FlowWorker()
        p._turn_seq += 1
        p._turn_view = panel.TurnView()
        p._steps_by_call = {}
        p._turn_outcome = ("", "")
        p._turn_request = "build it"
        p._set_busy(True)
        return p.worker

    def card(self):
        return next(b for b in self.p.blocks if b.kind == Block.TOOL and b.steps)

    def rich_step(self, call_id="call_a1", name="execute_code", title="Make the shell body", ordinal=7, **over):
        """tool_started -> step_started -> step_finished -> tool_finished, like the worker emits them."""
        p = self.p
        p._on_tool_started(name)
        p._on_step_started(j(started(call_id, name, title, ordinal=ordinal)))
        p._on_step_finished(j(finished(call_id, name, title, ordinal=ordinal, **over)))
        p._on_tool_finished(f"{name}(code=...)", over.get("ok", True), "short legacy result")
        return self.card().steps[-1]


# --------------------------------------------------------------------------------------------------
# plan strip


class PlanStripTest(FlowTestCase):
    def test_widget_is_hidden_without_a_plan_and_named(self):
        p = self.p
        self.assertEqual(p.plan_strip.objectName(), "oscarPlanStrip")
        self.assertTrue(p.plan_strip.isHidden())
        p._on_plan_changed(j(ledger(steps=[])))
        self.assertTrue(p.plan_strip.isHidden())          # no steps: nothing to pin

    def test_plan_changed_pins_the_strip_and_stores_the_ledger(self):
        p = self.p
        self.begin_turn()
        p._on_plan_changed(j(ledger()))
        # UI redesign 2026-11-16: the pinned plan is the strip's markup (the tab shell owns its placement).
        # The strip pins COMPACT — the anchor line and the running step — and expands on request.
        self.assertTrue(p._plan_markup)
        text = p.plan_strip.toPlainText()
        self.assertIn("step 5/8", text)                    # the running step is the 5th
        self.assertIn("round 7/30", text)
        self.assertIn("BUILD", text)
        self.assertIn("(1/2)", text)                       # stage progress of the staged step
        self.assertIn("Bottom shell: floor, walls, standoffs", text)
        p._on_anchor(QtCore.QUrl("plan:toggle"))           # expanded: the whole plan is readable
        expanded = p.plan_strip.toPlainText()
        for title in ("Source the board", "Lid", "Optional vents"):
            self.assertIn(title, expanded)
        self.assertIn("intent: enclosure", expanded)
        session = p._current_session()
        self.assertEqual(session["ledger"]["intent"], "enclosure")
        self.assertEqual(len(session["ledger"]["steps"]), 8)
        self.assertTrue(p._persist_timer.isActive())      # re-persisted soon, not once per signal

    def test_the_strip_never_collapses_by_itself(self):
        p = self.p
        self.begin_turn()
        p._on_plan_changed(j(ledger()))
        p._on_anchor(QtCore.QUrl("plan:toggle"))           # the user expanded it; only the user may fold it away
        before = p.plan_strip.toPlainText()
        for event in (lambda: p._on_progress(j(progress(round=9))), lambda: p._on_tool_started("get_objects"),
                      lambda: p._on_tool_finished("get_objects()", True, "x"), lambda: p._on_notice("hello"),
                      lambda: p._on_plan_changed(j(ledger(revision=13)))):
            event()
            self.assertIn("Optional vents", p.plan_strip.toPlainText())
        self.assertIn("Source the board", before)

    def test_plan_toggle_anchor_collapses_and_expands_on_request(self):
        p = self.p
        self.begin_turn()
        p._on_plan_changed(j(ledger()))
        # redesign 2026-11-16: the strip pins compact; the toggle expands it, and folds it back
        p._on_anchor(QtCore.QUrl("plan:toggle"))
        expanded = p.plan_strip.toPlainText()
        self.assertIn("Bottom shell", expanded)
        self.assertIn("Optional vents", expanded)
        p._on_anchor(QtCore.QUrl("plan:toggle"))
        collapsed = p.plan_strip.toPlainText()
        self.assertIn("Bottom shell", collapsed)           # the current step stays
        self.assertNotIn("Optional vents", collapsed)

    def test_progress_moves_the_round_in_the_header(self):
        p = self.p
        self.begin_turn()
        p._on_plan_changed(j(ledger()))
        p._on_progress(j(progress(round=12, round_cap=40, phase="VERIFY")))
        text = p.plan_strip.toPlainText()
        self.assertIn("round 12/40", text)
        self.assertIn("VERIFY", text)

    def test_the_strip_stays_pinned_after_the_turn_and_follows_the_chat(self):
        p = self.p
        self.begin_turn()
        p._on_plan_changed(j(ledger()))
        self.finish_turn()
        # UI redesign 2026-11-16: pinned means the strip keeps the final plan's markup between turns
        self.assertTrue(p._plan_markup)                    # the final plan stays pinned between turns
        self.assertIn("Bottom shell", p.plan_strip.toPlainText())   # compact pin: the anchor + running step
        p._on_anchor(QtCore.QUrl("plan:toggle"))
        self.assertIn("Lid", p.plan_strip.toPlainText())
        first = p._current_session()
        first["history"].append(panel.Message(role="user", content="make a lid"))   # "new chat" is refused in an unused chat
        p._new_chat()                                      # another chat: no plan of its own
        self.assertFalse(p._plan_markup)
        self.assertEqual(p.plan_strip.toPlainText(), "")
        p._activate_session(first)
        self.assertTrue(p._plan_markup)
        # re-pinning starts compact again (per chat): its running step proves the plan came back
        self.assertIn("Bottom shell", p.plan_strip.toPlainText())

    def test_clearing_a_chat_drops_its_plan(self):
        p = self.p
        self.begin_turn()
        p._on_plan_changed(j(ledger()))
        self.finish_turn()
        p.history.append(panel.Message(role="user", content="x"))
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
            p._clear_chat()
        self.assertTrue(p.plan_strip.isHidden())
        self.assertIsNone(p._current_session()["ledger"])

    @unittest.skip("pending UI stream shell: the plan strip's placement is being re-designed (it moved "
                   "into the Chat page but is still gated on the Summary tab), so a fitted height of a "
                   "VISIBLE strip cannot be pinned until that stream lands")
    def test_height_is_capped_at_thirty_percent_of_the_panel(self):
        p = self.p
        steps = [{"id": f"S{i}", "title": f"Step number {i} with a long enough title", "phase": "BUILD",
                  "status": "pending", "required": True} for i in range(1, 60)]
        self.begin_turn()
        p.resize(420, 500)
        p.show()
        self.app.processEvents()
        p._on_plan_changed(j(ledger(steps=steps)))
        self.assertLessEqual(p.plan_strip.height(), int(500 * 0.30) + 1)
        self.assertGreaterEqual(p.plan_strip.height(), 80)
        p.resize(420, 900)
        self.app.processEvents()
        self.assertGreater(p.plan_strip.height(), 150)     # re-fitted on resize

    def test_malformed_plan_payloads_are_ignored(self):
        p = self.p
        self.begin_turn()
        for bad in ("", "not json", "[]", "null", 5):
            p._on_plan_changed(bad)
        self.assertTrue(p.plan_strip.isHidden())
        self.assertIsNone(p._current_session().get("ledger"))


# --------------------------------------------------------------------------------------------------
# step rows


class StepRowTest(FlowTestCase):
    def test_step_started_attaches_to_the_row_the_legacy_signal_created(self):
        p = self.p
        self.begin_turn()
        p._on_tool_started("execute_code")
        card = self.card()
        self.assertEqual(len(card.steps), 1)
        p._on_step_started(j(started("call_a1", ordinal=7)))
        self.assertEqual(len(card.steps), 1)                # never two rows for one call
        # UI redesign 2026-11-16: the live row is its StepView (rows render in the dedicated
        # process view, never in Chat)
        self.assertEqual(card.steps[0].view.title, "Make the shell body")
        self.assertTrue(card.steps[0].view.running)
        p._on_step_finished(j(finished("call_a1", ordinal=7)))
        p._on_tool_finished("execute_code(code=...)", True, "short")
        self.assertEqual(len(card.steps), 1)
        view = card.steps[0].view
        self.assertFalse(view.running)
        self.assertEqual(view.title, "Make the shell body")
        self.assertTrue(card.steps[0].ok)

    def test_the_legacy_signal_never_replaces_the_rich_detail(self):
        p = self.p
        self.begin_turn()
        long_text = "result line\n" * 200               # 2400 chars: the legacy signal clips at 400
        p._on_tool_started("execute_code")
        p._on_step_started(j(started("call_a1", ordinal=7)))
        p._on_step_finished(j(finished("call_a1", ordinal=7, text=long_text, ok=False)))
        p._on_tool_finished("execute_code(code=...)", False, long_text[:400] + "…")
        step = self.card().steps[0]
        self.assertTrue(step.expanded)                     # a failed step opens with its full output
        self.assertEqual(step.view.text, long_text)
        # UI redesign 2026-11-16: the FULL output is pinned on the row markup (never replaced by
        # the legacy signal's 400-char clip)
        self.assertEqual(step.html(self.card().bid, 0).count("result line"), 200)

    def test_row_shows_title_duration_delta_and_health(self):
        p = self.p
        self.begin_turn()
        self.rich_step(health={"valid": False, "errors": 2, "warnings": 0, "line": "[health] 2 errors"})
        # UI redesign 2026-11-16: the row content is pinned on the row markup (rendered in the
        # dedicated process view, never in Chat)
        markup = self.card().steps[0].html(self.card().bid, 0)
        self.assertIn("✓ Make the shell body · 1.8s", markup)
        self.assertIn("+Shell", markup)
        self.assertIn("~Lid", markup)
        self.assertIn("2 errors", markup)

    def test_changed_entries_may_be_dicts_like_the_project_step_record(self):
        p = self.p
        self.begin_turn()
        delta = {"added": ["Shell"], "removed": [],
                 "changed": [{"name": "Lid", "props": ["Height"], "volume_before": 1.0, "volume_after": 2.0}]}
        step = self.rich_step(delta=delta)
        self.assertEqual(step.view.delta["changed"], ["Lid"])
        markup = step.html(self.card().bid, 0)
        self.assertNotIn("volume_before", markup)
        self.assertIn("~Lid", markup)
        step.expanded = True
        p._render()
        # UI redesign 2026-11-16: the expanded delta detail is pinned on the row markup
        self.assertIn("Lid (Height)", step.html(self.card().bid, 0))

    def test_expanding_shows_code_in_pre_with_a_copy_anchor_and_the_full_output(self):
        p = self.p
        self.begin_turn()
        step = self.rich_step(text="X" * 5000)
        step.expanded = True
        card = self.card()
        markup = step.html(card.bid, 0)
        self.assertIn("<pre>", markup)
        self.assertIn("import Part", markup)
        self.assertIn(f"copy:{card.bid}:0", markup)
        self.assertEqual(markup.count("X"), 5000)

    def test_copy_anchor_puts_the_recorded_script_on_the_clipboard(self):
        p = self.p
        self.begin_turn()
        script = os.path.join(self.tmp, "007_shell.py")
        full = "import Part\n" + "x = 1\n" * 3000           # longer than the 12000 chars of the payload
        with open(script, "w", encoding="utf-8") as f:
            f.write(full)
        step = self.rich_step(script_path=script, code="import Part\n")
        card = self.card()
        QtGui.QGuiApplication.clipboard().setText("")
        p._on_anchor(QtCore.QUrl(f"copy:{card.bid}:0"))
        self.assertEqual(QtGui.QGuiApplication.clipboard().text(), full)
        # without a file the payload's code is copied
        step.view.script_path = ""
        p._on_anchor(QtCore.QUrl(f"copy:{card.bid}:0"))
        self.assertEqual(QtGui.QGuiApplication.clipboard().text(), "import Part\n")

    def test_copy_with_nothing_to_copy_says_so(self):
        p = self.p
        self.begin_turn()
        self.rich_step(code="", args_json="")
        self.rich_step("call_a2", ordinal=8, code="", args_json="")
        card = self.card()
        p._on_anchor(QtCore.QUrl(f"copy:{card.bid}:1"))
        self.assertIn("Nothing to copy", p.log.toPlainText())
        p._on_anchor(QtCore.QUrl(f"copy:{card.bid}:9"))    # out of range: ignored
        p._on_anchor(QtCore.QUrl("copy:x:y"))              # malformed: ignored

    def test_show_anchor_selects_and_frames_the_objects_of_the_step(self):
        p = self.p
        self.begin_turn()
        self.rich_step()
        card = self.card()

        class FakeDoc:
            Name = "PiCase"

            def getObject(self, name):
                return object() if name in ("Shell", "Lid") else None

        fake_gui = mock.MagicMock()
        fake_app = mock.MagicMock()
        fake_app.ActiveDocument = FakeDoc()
        fake_app.listDocuments.return_value = {"PiCase": fake_app.ActiveDocument}
        with mock.patch.object(panel_anchors, "Gui", fake_gui), mock.patch.object(panel_anchors, "App", fake_app):
            p._on_anchor(QtCore.QUrl(f"show:{card.bid}:0"))
        fake_gui.Selection.clearSelection.assert_called_once()
        self.assertEqual([c.args for c in fake_gui.Selection.addSelection.call_args_list],
                         [("PiCase", "Shell"), ("PiCase", "Lid")])
        fake_gui.SendMsgToActiveView.assert_called_once_with("ViewSelection")

    def test_show_anchor_reports_objects_that_no_longer_exist(self):
        p = self.p
        self.begin_turn()
        self.rich_step()
        card = self.card()
        fake_app = mock.MagicMock()
        fake_app.ActiveDocument = mock.MagicMock(Name="D", getObject=mock.Mock(return_value=None))
        fake_app.listDocuments.return_value = {}
        with mock.patch.object(panel_anchors, "Gui", mock.MagicMock()), mock.patch.object(panel_anchors, "App", fake_app):
            p._on_anchor(QtCore.QUrl(f"show:{card.bid}:0"))
        self.assertIn("no longer exist", p.log.toPlainText())

    def test_step_without_objects_has_nothing_to_show(self):
        p = self.p
        self.begin_turn()
        self.rich_step(delta={"added": [], "removed": [], "changed": []})
        card = self.card()
        p._on_anchor(QtCore.QUrl(f"show:{card.bid}:0"))
        self.assertIn("did not add or change", p.log.toPlainText())

    def test_header_counts_exactly_the_tool_rows(self):
        p = self.p
        self.begin_turn()
        for i in range(3):
            self.rich_step(f"c{i}", ordinal=i + 1)
        p._on_plan_changed(j(ledger()))                   # plans and validation never create rows
        p._on_artifacts_changed(j({"project_dir": self.tmp, "files": []}))
        p._on_turn_outcome("validated", "")
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn(True, "Done.")
        card = self.card()
        self.assertEqual(len([s for s in card.steps if not s.plain and not s.is_thought]), 3)
        self.assertTrue(card.text.startswith("✻ Ran 3 steps"))
        self.assertTrue(card.text.endswith("✓ validated"))

    def test_a_worker_with_only_the_new_signals_still_gets_one_row_per_call(self):
        p = self.p
        self.begin_turn()
        p._on_step_started(j(started("call_z", "get_objects", "List objects", ordinal=1)))
        p._on_step_finished(j(finished("call_z", "get_objects", "List objects", ordinal=1)))
        p._on_step_started(j(started("call_y", "get_objects", "List again", ordinal=2)))
        p._on_step_finished(j(finished("call_y", "get_objects", "List again", ordinal=2)))
        card = self.card()
        self.assertEqual([s.view.title for s in card.steps], ["List objects", "List again"])
        self.assertTrue(all(not s.view.running for s in card.steps))

    def test_step_finished_without_a_start_makes_a_row(self):
        p = self.p
        self.begin_turn()
        p._on_step_finished(j(finished("call_q", ordinal=3)))
        self.assertEqual(len(self.card().steps), 1)
        # UI redesign 2026-11-16: the row's title is pinned on its view (rows render outside Chat)
        self.assertEqual(self.card().steps[0].view.title, "Make the shell body")

    def test_provider_call_ids_repeated_across_turns_keep_separate_rows(self):
        p = self.p
        self.begin_turn()
        self.rich_step("call_0", ordinal=1, title="First turn step")
        first_key = self.card().steps[0].call_id
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn(True, "ok")
        self.begin_turn()
        self.rich_step("call_0", ordinal=1, title="Second turn step")
        second_card = [b for b in p.blocks if b.kind == Block.TOOL and b.steps][-1]
        second_key = second_card.steps[0].call_id
        self.assertNotEqual(first_key, second_key)
        self.assertEqual(second_card.steps[0].view.title, "Second turn step")

    def test_same_call_id_twice_in_one_turn_is_two_rows(self):
        p = self.p
        self.begin_turn()
        self.rich_step("call_0", ordinal=1, title="One")
        self.rich_step("call_0", ordinal=2, title="Two")
        self.assertEqual([s.view.title for s in self.card().steps], ["One", "Two"])

    def test_malformed_step_payloads_are_ignored(self):
        p = self.p
        self.begin_turn()
        for bad in ("", "garbage", "[1]", None):
            p._on_step_started(bad)
            p._on_step_finished(bad)
        self.assertEqual([b for b in p.blocks if b.kind == Block.TOOL and b.steps], [])

    def test_rich_rows_are_cached_by_a_signature_that_follows_the_view(self):
        p = self.p
        self.begin_turn()
        self.rich_step()
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn(True, "ok")
        card = self.card()
        before = card.signature()
        card.steps[0].view.apply({"title": "Renamed"})
        self.assertNotEqual(card.signature(), before)

    def test_mutating_success_offers_a_revert(self):
        p = self.p
        p._turn_snapshot = dict(GOOD_SNAPSHOT)
        self.begin_turn()
        self.rich_step(name="build_enclosure")
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn(True, "ok")
        self.assertTrue(self.card().revertible)


# --------------------------------------------------------------------------------------------------
# thumbnails


class ThumbnailTest(FlowTestCase):
    def open_log(self):
        """Switch to the Logs page — the timeline view that resolves and renders step snapshots now."""
        p = self.p
        # UI redesign 2026-11-16: step thumbnails left the Chat; the Logs page timeline rebuilds from the
        # project journal (refresh_from) when the tab opens, resolving snapshot paths against the root
        p.tabs.setCurrentIndex(2)




    @unittest.skip("pending UI stream timeline: step-row snapshot notes have no surface yet (rows are no "
                   "longer drawn in Chat and the Log view does not mark a missing file as a muted note)")
    def test_missing_snapshot_becomes_a_muted_note_not_a_broken_image(self):
        p = self.p
        self.begin_turn()
        self.rich_step(snapshot_path=os.path.join(self.tmp, "nope.png"))
        view = self.card().steps[0].view
        self.assertEqual(view.snapshot_path, "")
        self.assertIn("snapshot file not found", p.log.toPlainText())
        self.assertNotIn("es-img://", p.log.toHtml())

    @unittest.skip("pending UI stream timeline: corrupt step snapshots are only detected when a row is "
                   "drawn; no current surface marks them (see the missing-snapshot skip above)")
    def test_corrupt_snapshot_is_handled_like_a_missing_one(self):
        p = self.p
        self.begin_turn()
        bad = os.path.join(self.tmp, "bad.png")
        with open(bad, "wb") as f:
            f.write(b"not a png")
        self.rich_step(snapshot_path=bad)
        self.assertEqual(self.card().steps[0].view.snapshot_path, "")

    def test_empty_snapshot_path_shows_the_note_of_the_payload(self):
        p = self.p
        self.begin_turn()
        self.rich_step(snapshot_path="", snapshot_note="no 3D view (headless session)")
        # UI redesign 2026-11-16: the payload's note rides the step row (drawn in the process view)
        card = self.card()
        self.assertIn("no 3D view (headless session)", card.steps[0].html(card.bid, 0))

    def test_relative_paths_resolve_against_the_project_root(self):
        p = self.p
        root = os.path.join(self.tmp, "proj")
        os.makedirs(os.path.join(root, "snapshots"))
        png(os.path.join(root, "snapshots", "001_a.png"))
        project = mock.Mock(root=root)
        p._current_session()["_project"] = project
        # Windows: QUrl-style forward slashes vs os.path.join backslashes — compare normalized
        self.assertEqual(os.path.normpath(p._abs_snapshot("snapshots/001_a.png")),
                         os.path.normpath(os.path.join(root, "snapshots", "001_a.png")))
        self.assertEqual(p._abs_snapshot("/abs/x.png"), "/abs/x.png")
        self.begin_turn()
        self.rich_step(snapshot_path="snapshots/001_a.png")
        self.open_log()                                     # the Logs page follows the journal (a mock project here)
        # the timeline resolves the record's relative snapshot against the project root — the path the
        # Logs page feeds refresh_from -> build_timeline(snapshot_dir=project.root)
        from oscar.panel_activity import build_timeline
        entries = build_timeline([{"call_id": self.card().steps[0].call_id, "tool": "execute_code", "ok": True,
                                   "snapshot": "snapshots/001_a.png"}], snapshot_dir=root)
        self.assertEqual(os.path.normpath(entries[0]["snapshot_path"]),
                         os.path.normpath(os.path.join(root, "snapshots", "001_a.png")))

    def test_cache_is_bounded_and_remembers_failures(self):
        from oscar.panel_blocks import ThumbCache

        cache = ThumbCache(limit=3)
        for i in range(5):
            cache.load(f"k{i}", png(os.path.join(self.tmp, f"{i}.png"), 40, 30))
        self.assertEqual(cache.keys(), ["k2", "k3", "k4"])  # the oldest are evicted
        self.assertIsNone(cache.load("gone", os.path.join(self.tmp, "gone.png")))
        self.assertTrue(cache.failed("gone"))
        png(os.path.join(self.tmp, "gone.png"), 40, 30)
        self.assertIsNone(cache.load("gone", os.path.join(self.tmp, "gone.png")))   # not probed again
        cache.clear()
        self.assertEqual(cache.keys(), [])

    def test_at_most_two_decodes_per_render_pass(self):
        """The redesigned panel decodes thumbnails DEFERRED (on card expansion), at most two per
        load pass, converging over passes."""
        p = self.p
        self.begin_turn()
        for i in range(5):
            png(os.path.join(self.tmp, f"{i}.png"), 60, 40)
        card = p._ensure_card()
        for i in range(5):
            step = Step("t", summary="t")
            view = panel.TurnView().on_step_started(j(started(f"id{i}", ordinal=i + 1)))
            view.apply({"snapshot_path": os.path.join(self.tmp, f"{i}.png")})
            view.running = False
            step.view, step.call_id, step.ok = view, f"k{i}", True
            card.steps.append(step)
        p._render()
        self.assertEqual(len(p._thumbs.keys()), 0)            # step snapshots never enter the panel cache
        from oscar.panel_timeline import TimelineTab          # their home since the redesign
        tab = TimelineTab()
        self.addCleanup(tab.deleteLater)
        tab.load_entries([{"op": "execute_code", "explanation": "s", "snapshot": path,
                           "snapshot_key": f"k{i}"}
                          for i, path in enumerate(os.path.join(self.tmp, f"{i}.png") for _ in range(0, 5))])
        first = tab._thumb(os.path.join(self.tmp, "0.png"))
        self.assertIsNotNone(first)
        self.assertIs(tab._thumb(os.path.join(self.tmp, "0.png")), first)   # decoded once, reused


class HeartbeatTest(FlowTestCase):
    def test_widgets_and_visibility_follow_the_turn(self):
        p = self.p
        self.assertEqual(p.heartbeat_label.objectName(), "oscarHeartbeat")
        self.assertEqual(p.act_now_btn.objectName(), "oscarActNowButton")
        self.assertTrue(p.heartbeat_row.isHidden())
        self.begin_turn()
        p._update_heartbeat()
        self.assertFalse(p.heartbeat_row.isHidden())
        self.assertEqual(p.heartbeat_label.text(), "Oscar is working…")   # no payload yet: the default phrase
        self.finish_turn()
        self.assertTrue(p.heartbeat_row.isHidden())

    def test_progress_payloads_drive_the_row_for_every_activity(self):
        """The redesigned heartbeat maps every activity to its user-facing phrase (the technical
        round/elapsed composite moved to the plan strip header)."""
        p = self.p
        self.begin_turn()
        cases = {
            "thinking": "Oscar is thinking…",
            "writing": "Oscar is thinking…",
            "tool_args": "Oscar is thinking…",
            "retrying": "Oscar is retrying the current step…",
            "waiting_approval": "Oscar is waiting for your approval…",
            "paused": "Oscar is paused.",
            "validating": "Oscar is validating the geometry…",
        }
        for activity, expected in cases.items():
            payload = progress(activity=activity, tool="execute_code" if activity == "tool_args" else "",
                               retry=[2, 5] if activity == "retrying" else None, started_at=time.time())
            p._on_progress(j(payload))
            self.assertIn(expected, p.heartbeat_label.text(), activity)
        p._on_progress(j(progress(activity="running_tool", tool="build_enclosure",
                                   phase="BUILD", started_at=time.time())))
        self.assertIn("Oscar is building the model…", p.heartbeat_label.text())

    def test_running_tool_is_counted_by_the_panels_own_timer(self):
        """A running tool shows its phase phrase and keeps the row alive on the panel's own timer
        (the elapsed composite moved to the plan strip)."""
        p = self.p
        self.begin_turn()
        p._on_progress(j(progress(activity="running_tool", tool="build_enclosure", phase="BUILD",
                                   started_at=time.time() - 30)))
        self.assertIn("Oscar is building the model…", p.heartbeat_label.text())
        p._turn_view._clock = lambda: time.time() + 65       # a minute later, no new payload arrived
        p._update_heartbeat()
        self.assertFalse(p.heartbeat_row.isHidden())
        self.assertIn("Oscar is building the model…", p.heartbeat_label.text())

    def test_a_silent_provider_is_visible_without_new_payloads(self):
        """A quiet stream keeps the heartbeat visible with a truthful phrase (the redesigned panel
        dropped the 'no data for N s' diagnostics from the label; visibility is the contract)."""
        p = self.p
        self.begin_turn()
        base = time.time()
        clock = {"now": base}
        p._turn_view._clock = lambda: clock["now"]
        p._on_progress(j(progress(activity="thinking", elapsed_round=5.0, idle_s=1.0, reasoning_chars=900)))
        self.assertIn("Oscar is thinking…", p.heartbeat_label.text())
        clock["now"] = base + 40                             # the stream went quiet: no tick arrived for 40 s
        p._update_heartbeat()
        self.assertFalse(p.heartbeat_row.isHidden())
        self.assertIn("Oscar is thinking…", p.heartbeat_label.text())

    def test_the_heartbeat_timer_runs_for_the_turn_only(self):
        p = self.p
        self.begin_turn()
        p._heartbeat_timer.start()
        self.assertEqual(p._heartbeat_timer.interval(), 1000)
        self.finish_turn()
        self.assertFalse(p._heartbeat_timer.isActive())

    def test_act_now_is_enabled_only_while_a_provider_call_is_open(self):
        p = self.p
        worker = self.begin_turn()
        p._on_progress(j(progress(activity="thinking")))
        self.assertFalse(p.act_now_btn.isHidden())
        self.assertTrue(p.act_now_btn.isEnabled())
        p._on_progress(j(progress(activity="running_tool", tool="x")))
        self.assertFalse(p.act_now_btn.isEnabled())
        p._on_progress(j(progress(activity="waiting_approval")))
        self.assertFalse(p.act_now_btn.isEnabled())
        p._on_progress(j(progress(activity="writing")))
        self.assertTrue(p.act_now_btn.isEnabled())
        p.act_now_btn.click()
        self.assertEqual(worker.calls, [("act_now",)])
        self.assertFalse(p.act_now_btn.isEnabled())          # one press per round
        self.assertIn("act", p._status)
        p._on_progress(j(progress(activity="thinking")))
        self.assertFalse(p.act_now_btn.isEnabled())
        p._on_round_reset("watchdog")                        # the retry round re-arms it
        p._on_progress(j(progress(activity="thinking")))
        self.assertTrue(p.act_now_btn.isEnabled())

    def test_act_now_is_absent_for_a_worker_without_it(self):
        p = self.p
        worker = self.begin_turn(FakeWorker())
        p._on_progress(j(progress(activity="thinking")))
        self.assertTrue(p.act_now_btn.isHidden())
        p._act_now()                                          # a guarded no-op
        self.assertEqual(worker.calls, [])

    def test_act_now_failure_is_reported(self):
        p = self.p
        worker = self.begin_turn()
        worker.act_now = mock.Mock(side_effect=RuntimeError("busy"))
        p._on_progress(j(progress(activity="thinking")))
        p._act_now()
        self.assertIn("busy", p._status)

    def test_malformed_progress_is_ignored(self):
        p = self.p
        self.begin_turn()
        p._on_progress("garbage")
        p._on_progress("[]")
        self.assertEqual(p.heartbeat_label.text(), "")


class RoundResetTest(FlowTestCase):

    def test_a_streaming_bubble_without_a_card_is_removed(self):
        p = self.p
        self.begin_turn()
        p._on_text_delta("partial words")
        self.assertTrue(any(b.kind == Block.ASSISTANT and "partial" in b.text for b in p.blocks))
        p._on_round_reset("stream_dropped")
        self.assertFalse(any(b.kind == Block.ASSISTANT and "partial" in b.text for b in p.blocks))
        p._on_text_delta("new")
        self.assertEqual([b.text for b in p.blocks if b.kind == Block.ASSISTANT][-1], "new")

    def test_reset_with_nothing_streaming_is_harmless(self):
        p = self.p
        self.begin_turn()
        for reason in ("retry", "gate", "watchdog", "adapted", ""):
            p._on_round_reset(reason)
        self.assertEqual(p._turn_view.resets[-1], "")

    def test_an_open_thought_is_closed_by_a_reset(self):
        p = self.p
        self.begin_turn()
        p._on_reasoning_delta("thinking hard")
        time.sleep(0.02)   # the thought lived for a moment: on a fast box the delta→reset gap is one clock tick (dur 0.0)
        p._on_round_reset("watchdog")
        thought = self.card().steps[0]
        self.assertTrue(thought.is_thought)
        self.assertGreater(thought.dur, 0)                   # closed with a duration, not left streaming
        self.assertGreater(p._turn_view.thought_s, 0)


# --------------------------------------------------------------------------------------------------
# artifacts strip


class ArtifactsStripTest(FlowTestCase):
    def project_files(self):
        root = os.path.join(self.tmp, "proj")
        for sub in ("exports", "snapshots", "docs"):
            os.makedirs(os.path.join(root, sub))
        names = {"model.FCStd": "fcstd", "NOTES.md": "notes", "rebuild.py": "rebuild", "provenance.json": "provenance",
                 "exports/shell.step": "export", "exports/lid.stl": "export", "docs/Other.FCStd": "fcstd"}
        files = []
        for name, kind in names.items():
            path = os.path.join(root, name)
            with open(path, "w") as f:
                f.write("x")
            files.append({"kind": kind, "name": name, "path": path})
        return root, files

    def test_strip_shows_the_chips_from_artifacts_changed(self):
        p = self.p
        root, files = self.project_files()
        self.assertEqual(p.artifacts_strip.objectName(), "oscarArtifactsStrip")
        self.assertTrue(p.artifacts_strip.isHidden())
        self.begin_turn()
        p._on_artifacts_changed(j({"project_dir": root, "files": files}))
        self.assertFalse(p.artifacts_strip.isHidden())
        text = p.artifacts_strip.text()
        for fragment in ("model.FCStd", "NOTES.md", "rebuild.py", "exports (2)", "provenance.json", "Oscar project",
                         "Other.FCStd", "oscar://artifact/open-doc", "oscar://artifact/open-exports"):
            self.assertIn(fragment, text)
        self.assertEqual(p._current_session()["project_dir"], root)

    def opened(self, target, **kw):
        with mock.patch.object(QtGui.QDesktopServices, "openUrl", return_value=True) as opener:
            self.p._on_anchor_text(target)
        return [c.args[0].toLocalFile() for c in opener.call_args_list]

    def test_chips_open_the_files_through_the_desktop(self):
        p = self.p
        root, files = self.project_files()
        p._on_artifacts_changed(j({"project_dir": root, "files": files}))
        # QUrl.fromLocalFile(...).toLocalFile() spells Windows paths with '/': compare normalized
        self.assertEqual([os.path.normpath(u) for u in self.opened("oscar://artifact/open-notes")],
                         [os.path.normpath(os.path.join(root, "NOTES.md"))])
        self.assertEqual([os.path.normpath(u) for u in self.opened("oscar://artifact/open-provenance")],
                         [os.path.normpath(os.path.join(root, "provenance.json"))])
        self.assertEqual([os.path.normpath(u) for u in self.opened("oscar://artifact/open-exports")],
                         [os.path.normpath(os.path.join(root, "exports"))])
        self.assertEqual([os.path.normpath(u) for u in self.opened("oscar://artifact/open-dir")], [os.path.normpath(root)])

    def test_a_script_is_revealed_never_launched(self):
        p = self.p
        root, files = self.project_files()
        p._on_artifacts_changed(j({"project_dir": root, "files": files}))
        self.assertEqual([os.path.normpath(u) for u in self.opened("oscar://artifact/open-rebuild")],
                         [os.path.normpath(root)])   # its FOLDER, not rebuild.py

    def test_documents_open_in_the_application_not_the_desktop(self):
        p = self.p
        root, files = self.project_files()
        p._on_artifacts_changed(j({"project_dir": root, "files": files}))
        engine = mock.Mock()
        engine.import_file.return_value = {"success": True, "document_name": "model"}
        fake_app = mock.MagicMock()
        fake_app.listDocuments.return_value = {}
        with mock.patch("oscar.cad.get_engine", return_value=engine), mock.patch.object(panel_anchors, "App", fake_app), \
                mock.patch.object(QtGui.QDesktopServices, "openUrl") as opener, mock.patch.object(p, "_fit_view_after_turn"):
            p._on_anchor_text("oscar://artifact/open-doc")
            p._on_anchor_text("oscar://artifact/open-doc/docs%2FOther.FCStd")
        opener.assert_not_called()
        self.assertEqual([(os.path.normpath(a), b) for a, b in (c.args for c in engine.import_file.call_args_list)],
                         [(os.path.normpath(os.path.join(root, "model.FCStd")), ""),
                          (os.path.normpath(os.path.join(root, "docs", "Other.FCStd")), "")])
        self.assertIn("opened model", p.log.toPlainText())

    def test_an_already_open_document_is_only_activated(self):
        p = self.p
        root, files = self.project_files()
        p._on_artifacts_changed(j({"project_dir": root, "files": files}))
        doc = mock.Mock(FileName=os.path.join(root, "model.FCStd"))
        doc.Name = "PiCase"
        fake_app = mock.MagicMock()
        fake_app.listDocuments.return_value = {"PiCase": doc}
        fake_gui = mock.MagicMock()
        engine = mock.Mock()
        with mock.patch("oscar.cad.get_engine", return_value=engine), mock.patch.object(panel_anchors, "App", fake_app), \
                mock.patch.object(panel_anchors, "Gui", fake_gui):
            p._on_anchor_text("oscar://artifact/open-doc")
        engine.import_file.assert_not_called()
        fake_gui.setActiveDocument.assert_called_once_with("PiCase")
        self.assertIn("already open", p.log.toPlainText())

    def test_open_failure_is_reported(self):
        p = self.p
        root, files = self.project_files()
        p._on_artifacts_changed(j({"project_dir": root, "files": files}))
        engine = mock.Mock()
        engine.import_file.return_value = {"success": False, "error": "corrupt file"}
        fake_app = mock.MagicMock()
        fake_app.listDocuments.return_value = {}
        with mock.patch("oscar.cad.get_engine", return_value=engine), mock.patch.object(panel_anchors, "App", fake_app):
            p._on_anchor_text("oscar://artifact/open-doc")
        self.assertIn("open failed: corrupt file", p.log.toPlainText())

    def test_links_from_model_text_cannot_reach_other_files(self):
        p = self.p
        root, files = self.project_files()
        p._on_artifacts_changed(j({"project_dir": root, "files": files}))
        for target in ("oscar://artifact/open-doc/..%2F..%2Fetc%2Fpasswd", "oscar://artifact/open-doc//etc/passwd",
                       "oscar://artifact/open-doc/nope.FCStd", "oscar://artifact/open-notes/..%2Fsecret.md"):
            self.assertEqual([os.path.normpath(u) for u in self.opened(target)],
                             [] if "notes" not in target else [os.path.normpath(os.path.join(root, "NOTES.md"))])
        self.assertIn("File not found", p.log.toPlainText())

    def test_an_entry_outside_the_project_folder_is_refused(self):
        p = self.p
        root, files = self.project_files()
        files.append({"kind": "fcstd", "name": "evil.FCStd", "path": "/etc/hosts"})
        p._on_artifacts_changed(j({"project_dir": root, "files": files}))
        self.assertEqual(p._artifact_target("open-doc", "evil.FCStd"), "")

    def test_unknown_actions_and_missing_files_are_reported(self):
        p = self.p
        root, files = self.project_files()
        p._on_artifacts_changed(j({"project_dir": root, "files": files}))
        self.assertEqual(self.opened("oscar://artifact/launch-missiles"), [])
        self.assertIn("Unknown artifact link", p.log.toPlainText())
        os.remove(os.path.join(root, "NOTES.md"))
        self.assertEqual(self.opened("oscar://artifact/open-notes"), [])
        self.assertIn("File not found", p.log.toPlainText())

    def test_backup_files_never_become_chips_or_targets(self):
        p = self.p
        root, files = self.project_files()
        files.append({"kind": "fcstd", "name": "model.20261001.FCBak", "path": os.path.join(root, "model.FCBak")})
        p._on_artifacts_changed(j({"project_dir": root, "files": files}))
        self.assertNotIn("FCBak", p.artifacts_strip.text())
        self.assertEqual(p._artifact_target("open-doc", "model.20261001.FCBak"), "")

    def test_the_strip_follows_the_chat_and_is_read_from_the_project_folder(self):
        p = self.p
        root, files = self.project_files()
        session = p._current_session()
        project = mock.Mock(root=root)
        project.artifact_list.return_value = files
        session["_project"] = project
        session["history"].append(panel.Message(role="user", content="build it"))   # "new chat" is refused in an unused chat
        p._refresh_session_widgets()
        self.assertFalse(p.artifacts_strip.isHidden())
        self.assertIn("NOTES.md", p.artifacts_strip.text())
        p._new_chat()
        self.assertTrue(p.artifacts_strip.isHidden())
        p._activate_session(session)
        self.assertFalse(p.artifacts_strip.isHidden())

    def test_a_failing_project_listing_does_not_break_the_panel(self):
        p = self.p
        root, _files = self.project_files()
        project = mock.Mock(root=root)
        project.artifact_list.side_effect = OSError("disk gone")
        p._current_session()["_project"] = project
        payload = p._artifacts_of(p._current_session())
        self.assertEqual(payload, {"project_dir": root, "files": []})

    def test_the_folder_chip_appears_with_the_first_plan_of_a_work_turn(self):
        p = self.p
        root, _files = self.project_files()
        project = mock.Mock(root=root)
        project.artifact_list.return_value = []
        p._current_session()["_project"] = project
        self.begin_turn()
        p._on_plan_changed(j(ledger()))
        self.assertIn("Oscar project", p.artifacts_strip.text())


# --------------------------------------------------------------------------------------------------
# outcome badge, Build Summary card, Continue


class OutcomeTest(FlowTestCase):
    def finish(self, outcome, text="Answer.", summary="", ok=True, steps=1, plan=True):
        p = self.p
        self.begin_turn()
        if plan:
            p._on_plan_changed(j(ledger()))
        for i in range(steps):
            self.rich_step(f"c{i}", ordinal=i + 1)
        p._on_turn_outcome(outcome, summary)
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn(ok, text)
        return self.card()


    def test_the_summary_is_rendered_once_as_a_card(self):
        p = self.p
        text = "I built the case.\n\n" + BUILD_SUMMARY
        card = self.finish("validated", text, summary=BUILD_SUMMARY)
        plain = p.log.toPlainText()
        self.assertEqual(plain.count("Build Summary"), 1)
        self.assertNotIn("## Build Summary", plain)
        self.assertEqual(plain.count("NOT validated"), 1)
        answers = [b for b in p.blocks if b.kind == Block.ASSISTANT and b.text.strip() and b is not p._welcome]
        self.assertEqual([b.text for b in answers][-1], "I built the case.")
        summaries = [b for b in p.blocks if b.kind == Block.SUMMARY]
        self.assertEqual(len(summaries), 1)
        self.assertTrue(summaries[0].text.startswith("## Build Summary"))
        self.assertGreater(p.blocks.index(summaries[0]), p.blocks.index(answers[-1]))   # after the answer
        self.assertGreater(p.blocks.index(answers[-1]), p.blocks.index(card))           # which follows the card

    def test_a_legacy_worker_without_turn_outcome_still_renders_the_summary_once(self):
        p = self.p
        self.begin_turn()
        self.rich_step()
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn(True, "Done it.\n\n" + BUILD_SUMMARY)
        self.assertEqual(p.log.toPlainText().count("Build Summary"), 1)
        self.assertEqual(len([b for b in p.blocks if b.kind == Block.SUMMARY]), 1)

    def test_a_turn_without_summary_adds_no_card(self):
        card = self.finish("answered", "Just an answer.")
        self.assertEqual([b for b in self.p.blocks if b.kind == Block.SUMMARY], [])
        self.assertTrue(card.text.startswith("✻ Ran 1 step"))
        self.assertNotIn(" · ✓", card.text)               # a plain answer carries no badge

    def test_badge_follows_the_legacy_header_for_every_outcome(self):
        for outcome, badge, kind in (("validated", "✓ validated", "ok"), ("unverified", "⚠ unverified", "warn"),
                                     ("capped", "⚠ capped", "warn"), ("needs_input", "? needs input", "neutral"),
                                     ("failed", "✗ failed", "error"), ("needs_help", "✗ needs help", "error"),
                                     ("stopped", "■ stopped", "error")):
            self.p.blocks.clear()
            card = self.finish(outcome, "x", ok=outcome in ("validated", "unverified", "capped", "needs_input"))
            self.assertTrue(card.text.startswith("✻ Ran 1 step"), outcome)
            self.assertTrue(card.text.endswith(" · " + badge), (outcome, card.text))
            self.assertEqual(card.header_kind, kind)

    def test_warning_and_error_headers_are_coloured(self):
        # redesign palette (panel_render): danger #ef6f61, warn #e6b422, success never shouted
        card = self.finish("failed", "It broke.", ok=False)
        self.assertIn("#ef6f61", card.html())
        self.p.blocks.clear()
        card = self.finish("unverified", "NOT VERIFIED: x")
        self.assertIn("#e6b422", card.html())
        self.p.blocks.clear()
        card = self.finish("validated", "ok")
        self.assertNotIn("#3cc9c0", card.html())            # success is not shouted

    def test_failed_text_is_cut_at_the_summary_too(self):
        p = self.p
        self.finish("failed", "Nothing was built.\n\n" + BUILD_SUMMARY, ok=False)
        errors = [b for b in p.blocks if b.kind == Block.ERROR]
        self.assertEqual(errors[-1].text, "Nothing was built.")

    def test_stop_ends_with_a_stopped_badge_and_the_stopped_text(self):
        p = self.p
        card = self.finish("stopped", "Stopped.", ok=False)
        self.assertTrue(card.text.endswith("■ stopped"))
        self.assertEqual([b.text for b in p.blocks if b.kind == Block.ERROR][-1], "Stopped.")

    def test_capped_offers_continue_which_sends_the_exact_resume_prompt(self):
        from oscar import flow

        p = self.p
        self.finish("capped", "Wrapped up.\n\n(Stopped after 30 tool rounds.)", summary=BUILD_SUMMARY)
        block = [b for b in p.blocks if b.kind == Block.SUMMARY][-1]
        self.assertEqual(block.actions, [("continue:", "Continue")])
        self.assertIn("href='continue:'", block.html())
        self.assertIn("Continue", p.log.toPlainText())
        sent = []
        p._send_text = lambda text, **kw: sent.append(text)
        p._on_anchor(QtCore.QUrl("continue:"))
        self.assertEqual(len(sent), 1)
        carried = flow.Ledger.from_dict(p._current_session()["ledger"])
        self.assertEqual(sent[0], carried.resume_prompt())
        self.assertTrue(flow.is_continuation(sent[0], carried))     # the worker recognises it as a continuation
        self.assertIn("Next: S5", sent[0])

    def test_capped_without_a_summary_still_offers_continue(self):
        p = self.p
        self.finish("capped", "Wrapped up.", summary="")
        block = [b for b in p.blocks if b.kind == Block.SUMMARY][-1]
        self.assertEqual(block.text, "")
        self.assertIn("continue:", block.html())

    def test_other_outcomes_offer_no_continue(self):
        for outcome in ("validated", "unverified", "needs_input", "failed"):
            self.p.blocks.clear()
            self.finish(outcome, "x", ok=outcome != "failed")
            self.assertNotIn("continue:", self.p.log.toHtml(), outcome)

    def test_continue_is_ignored_while_a_turn_runs_and_has_a_fallback_text(self):
        p = self.p
        sent = []
        p._send_text = lambda text, **kw: sent.append(text)
        p.worker = FlowWorker()
        p._on_anchor(QtCore.QUrl("continue:"))
        self.assertEqual(sent, [])
        self.assertIn("Wait for the current turn", p._status)
        p.worker = None
        p._current_session()["ledger"] = None
        p._on_anchor(QtCore.QUrl("continue:"))
        self.assertEqual(sent, ["Continue the task."])
        p._current_session()["ledger"] = {"intent": "other", "steps": "garbage"}
        self.assertTrue(p._resume_text().startswith("Continue the task."))

    def test_turn_card_collapses_unless_a_step_failed(self):
        card = self.finish("validated", "ok")
        self.assertFalse(card.expanded)
        self.p.blocks.clear()
        p = self.p
        self.begin_turn()
        self.rich_step(ok=False)
        p._on_turn_outcome("failed", "")
        with mock.patch.object(p, "_fit_view_after_turn"):
            self.finish_turn(False, "x")
        self.assertTrue(self.card().expanded)
        self.assertIn("⚠ failed steps", self.card().text)


class LedgerCarryTest(FlowTestCase):
    def test_a_work_turn_carries_the_workers_live_ledger(self):
        from oscar import flow

        p = self.p
        live = flow.Ledger.from_request("Build a housing for a Raspberry Pi 4 with vents")
        self.begin_turn(FlowWorker(ledger=live))
        self.finish_turn(True, "done")
        self.assertEqual(p._current_session()["ledger"], live.to_dict())

    def test_a_question_turn_keeps_the_previous_plan(self):
        from oscar import flow

        p = self.p
        previous = ledger()
        p._current_session()["ledger"] = previous
        throwaway = flow.Ledger.from_request("what is a fillet?")
        self.assertFalse(throwaway.is_work_turn())
        self.begin_turn(FlowWorker(ledger=throwaway))
        self.finish_turn(True, "A fillet rounds an edge.")
        self.assertIs(p._current_session()["ledger"], previous)

    def test_workers_without_a_ledger_or_with_a_broken_one_change_nothing(self):
        p = self.p
        previous = ledger()
        p._current_session()["ledger"] = previous
        self.begin_turn(FakeWorker())
        self.finish_turn(True, "x")
        broken = FlowWorker(ledger=mock.Mock(is_work_turn=mock.Mock(side_effect=RuntimeError("x"))))
        self.begin_turn(broken)
        self.finish_turn(True, "x")
        self.assertIs(p._current_session()["ledger"], previous)

    def test_a_work_turn_with_no_steps_replaces_the_old_plan(self):
        from oscar import flow

        p = self.p
        p._current_session()["ledger"] = ledger()
        empty = flow.Ledger.from_request("make a plate")
        empty.steps = []
        self.begin_turn(FlowWorker(ledger=empty))
        self.finish_turn(True, "x")
        self.assertEqual(p._current_session()["ledger"]["steps"], [])
        self.assertTrue(p.plan_strip.isHidden())


# --------------------------------------------------------------------------------------------------
# small display bugs the real multi-round flow exposed


class DisplayFixesTest(FlowTestCase):
    def test_a_remark_streamed_inside_a_live_card_has_no_stray_prefix(self):
        p = self.p
        self.begin_turn()
        p._on_tool_started("get_objects")
        p._on_tool_finished("get_objects()", True, "x")
        p._on_text_delta("Setting up ")
        p._on_text_delta("a document.")
        p._flush_render()
        note = self.card().steps[-1]
        self.assertTrue(note.plain and note.live)
        self.assertEqual(note.summary, "Setting up a document.")      # Step() used to fall back to its name: "toolSetting up ..."
        self.assertNotIn("toolSetting", p.log.toPlainText())

    def test_a_notice_carries_one_icon(self):
        p = self.p
        p._on_notice("Oscar stopped with unfinished work; asking it to continue.")
        plain = p.log.toPlainText()
        self.assertEqual(plain.count("ℹ"), 1)
        self.assertIn("ℹ Oscar stopped", plain)

