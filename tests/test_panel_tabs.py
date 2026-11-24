"""UI redesign tests: the Chat/Parameters/Logs tab shell, the Parameters view and its
Apply write-path (``tools_kit_params.apply_parameter_edits``), the Logs view, and the in-chat
question cards (``Block.QUESTION`` + ``netguard.panel_surface`` + the ``oscar://answer/`` anchor)
with their states and chat isolation.

There is no Summary tab (shell redesign 2026-11-16): the Build Summary is ONE card inside the
Chat transcript, and the Logs page is the execution view.

A real ``OscarPanel`` on an offscreen QApplication, like test_panel_flow; the FreeCAD
document tests build a small Params sheet with an expression-bound box and apply real
edits through the same undo-transaction path the panel uses.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

_IMPORT_ERROR = ""
try:
    from PySide import QtWidgets
    from oscar import netguard
    from oscar.panel import Block, Step
    from oscar.panel_tabs import ParamsView, LogsView
    from oscar.tools_kit_params import apply_parameter_edits
    from test_panel_logic import PanelTestCase
except Exception as exc:   # pragma: no cover - plain interpreter without FreeCAD / Qt
    _IMPORT_ERROR = str(exc)

    class _Unavailable(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")

    PanelTestCase = _Unavailable


PARAM_TABLE = {
    "sheet": "Params",
    "rows": [
        {"name": "wall_t", "value": 3.0, "unit": "mm", "expr": "", "source": "step_measurement",
         "note": "", "cell": "A1", "bound": 2, "needs_rebuild": False},
        {"name": "plate_w", "value": 60.0, "unit": "mm", "expr": "", "source": "assumption",
         "note": "", "cell": "A2", "bound": 0, "needs_rebuild": False},
        {"name": "height", "value": 12.0, "unit": "mm", "expr": "=wall_t * 4", "source": "",
         "note": "", "cell": "A3", "bound": 1, "needs_rebuild": False},
        {"name": "vent_count", "value": 6, "unit": "", "expr": "", "source": "",
         "note": "", "cell": "A4", "bound": 0, "needs_rebuild": True},
    ],
}


class TabShellTest(PanelTestCase):
    def test_profile_and_model_are_separate_dropdowns(self):
        from oscar.params import Profile

        alpha = Profile(name="Alpha Workspace", provider="custom A", model="alpha-main")
        beta = Profile(name="Beta Cloud", provider="custom B", model="beta-fast")
        self.p._profiles = [alpha, beta]
        self.p._model_catalogs = {
            alpha.name: {"models": ["alpha-main", "alpha-reasoner"], "error": "", "loading": False},
            beta.name: {"models": ["beta-fast", "beta-pro"], "error": "", "loading": False},
        }

        self.assertIsNot(self.p.profile_button, self.p.model_button)
        self.p._rebuild_profile_menu()
        profile_labels = [action.text() for action in self.p.profile_menu.actions()
                          if action.text().endswith(("model", "models", "unavailable", "loading…"))]
        self.assertEqual(profile_labels, ["Alpha Workspace — 2 models", "Beta Cloud — 2 models"])

        self.p._select_chat_model(alpha.name, "alpha-reasoner")
        self.assertEqual(self.p.profile_button.text(), "Alpha Workspace")
        self.assertEqual(self.p.model_button.text(), "alpha-reasoner")
        self.p._rebuild_model_menu()
        model_labels = [action.text() for action in self.p.model_menu.actions() if action.isCheckable()]
        self.assertEqual(model_labels, [
            "alpha-main · custom A · Listed by provider",
            "alpha-reasoner · custom A · Listed by provider",
        ])
        self.assertTrue(self.p.model_menu.actions()[1].isChecked())

    def test_profile_and_model_dropdowns_keep_selections_per_chat(self):
        from oscar.params import Profile

        alpha = Profile(name="Alpha Workspace", provider="custom A", model="alpha-main")
        beta = Profile(name="Beta Cloud", provider="custom B", model="beta-fast")
        self.p._profiles = [alpha, beta]
        self.p._model_catalogs = {
            alpha.name: {"models": ["alpha-main", "alpha-reasoner"], "error": "", "loading": False},
            beta.name: {"models": ["beta-fast", "beta-pro"], "error": "", "loading": False},
        }
        self.p._rebuild_profile_menu()
        profile_actions = [action.text() for action in self.p.profile_menu.actions()
                           if action.text().endswith("models")]
        self.assertEqual(profile_actions, ["Alpha Workspace — 2 models", "Beta Cloud — 2 models"])

        first = self.p._current_session()
        self.p._select_chat_model(alpha.name, "alpha-reasoner")
        from oscar.providers import Message

        first["history"].append(Message(role="user", content="original chat"))
        self.p._new_chat()
        second = self.p._current_session()
        self.p._select_chat_model(beta.name, "beta-pro")
        self.assertEqual((first["profile_name"], first["model"]), (alpha.name, "alpha-reasoner"))
        self.assertEqual((second["profile_name"], second["model"]), (beta.name, "beta-pro"))
        self.assertEqual(self.p.profile_button.text(), "Beta Cloud")
        self.assertEqual(self.p.model_button.text(), "beta-pro")
        self.p._rebuild_model_menu()
        model_labels = [action.text() for action in self.p.model_menu.actions() if action.isCheckable()]
        self.assertEqual(model_labels, [
            "beta-fast · custom B · Listed by provider",
            "beta-pro · custom B · Listed by provider",
        ])

    def test_reference_theme_uses_the_card_chat_palette(self):
        from oscar.panel_theme import TOKENS, panel_qss

        expected = {"bg": "#0f1115", "panel": "#151820", "bg_card": "#1b1f29",
                    "bg_recessed": "#0f1115", "border": "#272c38", "text": "#e6e8ee",
                    "text_2": "#8a90a0", "text_3": "#6f7583", "accent": "#3cc9c0",
                    "success": "#3cc9c0", "attention": "#e6b422", "warn": "#e6b422",
                    "danger": "#ef6f61"}
        for key, value in expected.items():
            self.assertEqual(TOKENS[key], value)
            self.assertIn(value, panel_qss())
        self.assertIn("QToolButton#oscarProfileSelector", panel_qss())
        self.assertIn("QToolButton#oscarModelSelector", panel_qss())
        self.assertIn("QTextBrowser#oscarLog", panel_qss())

    def test_header_has_one_rename_action_and_uses_the_supplied_logo(self):
        import os
        from oscar import paths

        self.assertFalse(hasattr(self.p, "rename_chat_btn"))
        rename_actions = [action for action in self.p.header_menu.actions() if action.text() == "Rename chat…"]
        self.assertEqual(len(rename_actions), 1)
        self.assertTrue(os.path.isfile(paths.icon("OscarMark.png")))
        self.assertFalse(self.p.logo_label.pixmap().isNull())

    def test_compact_composer_and_collapsed_plan_defaults(self):
        self.assertEqual(self.p.input.height(), 38)
        self.p.input.setPlainText("line one\nline two\nline three\nline four")
        self.assertGreater(self.p.input.height(), 38)
        self.assertLessEqual(self.p.input.height(), 76)
        self.p.input.clear()
        self.assertEqual(self.p.input.height(), 38)
        self.assertTrue(self.p._plan_collapsed)
        self.assertEqual(self.p._plan_animation.duration(), 150)

    def test_three_tabs_and_exactly_one_page_visible(self):
        self.assertEqual(self.p.tabs.count(), 3)
        self.assertEqual([self.p.tabs.tabText(i) for i in range(3)], ["Chat", "Parameters", "Logs"])
        pages = {self.p.tabs.widget(i) for i in range(3)}
        self.assertEqual(len(pages), 3)
        for index in range(3):
            self.p.tabs.setCurrentIndex(index)
            self.assertEqual(self.p.tabs.currentIndex(), index)
            for other in range(3):
                self.assertEqual(self.p.tabs.widget(other).isHidden(), other != index)

    def test_no_summary_tab_exists(self):
        labels = [self.p.tabs.tabText(i) for i in range(self.p.tabs.count())]
        self.assertNotIn("Summary", labels)
        self.assertFalse(hasattr(self.p, "summary_view"))

    def test_plan_strip_lives_inside_the_chat_page(self):
        for index in (1, 2, 0):
            self.p.tabs.setCurrentIndex(index)
            self.assertIs(self.p.plan_strip.parentWidget(), self.p._chat_page)
            self.assertIsNot(self.p.log.parentWidget(), self.p)   # the log lives in the Chat page
            self.assertEqual(self.p._chat_page.isHidden(), index != 0)
            self.assertTrue(self.p.plan_strip.isHidden())   # this test has no plan to display

    def test_running_indicator_uses_the_actual_worker_phase(self):
        from types import SimpleNamespace

        self.p.worker = object()
        self.p._turn_view = SimpleNamespace(progress={"activity": "running_tool", "phase": "BUILD"})
        self.p._update_heartbeat()
        self.assertEqual(self.p.heartbeat_label.text(), "Oscar is building the model…")
        self.p._turn_view = SimpleNamespace(progress={"activity": "waiting_approval", "phase": "BUILD"})
        self.p._update_heartbeat()
        self.assertEqual(self.p.heartbeat_label.text(), "Oscar is waiting for your approval…")

    def test_activity_pulse_runs_only_while_the_worker_is_busy(self):
        from PySide import QtCore

        self.p.worker = object()
        self.p._set_busy(True)
        self.assertEqual(self.p.activity_pulse.state(), QtCore.QAbstractAnimation.Running)
        self.p._set_busy(False)
        self.assertEqual(self.p.activity_pulse.state(), QtCore.QAbstractAnimation.Stopped)

    def test_tab_switch_refreshes_the_visible_view(self):
        with mock.patch.object(self.p, "_refresh_params") as params, \
                mock.patch.object(self.p, "_refresh_logs") as logs:
            self.p.tabs.setCurrentIndex(1)
            params.assert_called_once()
            logs.assert_not_called()
            self.p.tabs.setCurrentIndex(2)
            logs.assert_called_once()

    def test_overflow_menu_keeps_the_real_actions(self):
        labels = " ".join(action.text() for action in self.p.header_menu.actions())
        for wanted in ("Delete this chat…", "Open project folder", "Exports folder", "Settings…",
                       "Rename chat…", "Search chats…", "Follow the build"):
            self.assertIn(wanted, labels)


class ParamsViewTest(PanelTestCase):
    def view(self) -> "ParamsView":
        view = ParamsView(on_apply=lambda edits: {"ok": True, "text": "applied", "applied": [e["name"] for e in edits]})
        view.load(PARAM_TABLE)
        return view

    def row_of(self, view: "ParamsView", name: str):
        for r in range(view.table.rowCount()):
            if view.table.item(r, 0).text() == name:
                return r
        return -1

    def test_load_shows_all_rows_with_sources(self):
        view = self.view()
        self.assertEqual(view.table.rowCount(), 4)
        r = self.row_of(view, "wall_t")
        self.assertEqual(view.table.item(r, 1).text(), "3 mm")
        self.assertEqual(view.table.item(r, 4).text(), "Measured")
        self.assertEqual(view.table.item(r, 5).text(), "2×")
        self.assertEqual(self.row_of(view, "height"), 2)

    def test_filter_hides_non_matching_rows(self):
        view = self.view()
        view.search.setText("wall")
        self.assertEqual(view.table.rowCount(), 1)
        self.assertEqual(view.table.item(0, 0).text(), "wall_t")

    def test_safe_rows_editable_derived_and_rebuild_rows_readonly(self):
        from PySide import QtCore

        view = self.view()
        editable = QtCore.Qt.ItemFlag.ItemIsEditable
        self.assertTrue(view.table.item(self.row_of(view, "wall_t"), 1).flags() & editable)
        self.assertFalse(view.table.item(self.row_of(view, "height"), 1).flags() & editable)
        self.assertFalse(view.table.item(self.row_of(view, "vent_count"), 1).flags() & editable)

    def test_pending_edits_drive_apply_and_discard(self):
        view = self.view()
        self.assertFalse(view.apply_btn.isVisibleTo(view))
        view.table.item(self.row_of(view, "wall_t"), 1).setText("4 mm")
        self.assertTrue(view.apply_btn.isVisibleTo(view))
        self.assertEqual(view.apply_btn.text(), "Apply (1)")
        self.assertEqual(view.pending(), {"wall_t": "4 mm"})
        view.apply()
        self.assertEqual(view.pending(), {})   # ok clear
        self.assertFalse(view.apply_btn.isVisibleTo(view))
        view.table.item(self.row_of(view, "plate_w"), 1).setText("61 mm")
        self.assertEqual(view.pending(), {"plate_w": "61 mm"})
        view.discard()
        self.assertEqual(view.pending(), {})
        self.assertEqual(view.table.item(self.row_of(view, "plate_w"), 1).text(), "60 mm")

    def test_failed_apply_keeps_the_pending_edits(self):
        view = ParamsView(on_apply=lambda edits: {"ok": False, "text": "rolled back"})
        view.load(PARAM_TABLE)
        view.table.item(self.row_of(view, "wall_t"), 1).setText("9 mm")
        view.apply()
        self.assertEqual(view.pending(), {"wall_t": "9 mm"})   # the user can fix and retry

    def test_empty_table_shows_the_hint(self):
        view = ParamsView()
        view.load({"sheet": "", "rows": []})
        self.assertTrue(view.hint.isVisibleTo(view))
        self.assertIn("No parameter sheet", view.hint.text())


class ClarificationTurnTest(PanelTestCase):
    def test_needs_input_turn_becomes_inline_choices(self):
        self.p._turn_outcome = ("needs_input", "")
        streamed = self.p._append(Block.ASSISTANT, "The hole depth is unclear.\n\nThrough or blind?\n- Through\n- Blind")
        self.p._current = streamed

        self.p._on_turn_finished(True, streamed.text)

        self.assertNotIn(streamed, self.p.blocks)
        question = next(b for b in self.p.blocks if b.kind == Block.QUESTION)
        self.assertEqual(question.detail, "clarification_pending")
        self.assertEqual([label for _target, label in question.actions], ["Through", "Blind"])
        self.assertIn("Through or blind?", question.html())

    def test_malformed_needs_input_stays_a_plain_assistant_reply(self):
        self.p._turn_outcome = ("needs_input", "")
        self.p._on_turn_finished(True, "Which unit should I use?\n- mm")

        self.assertFalse(any(b.kind == Block.QUESTION for b in self.p.blocks))
        self.assertIn("Which unit should I use?\n- mm",
                      [b.text for b in self.p.blocks if b.kind == Block.ASSISTANT])


class ApplyEditsTest(PanelTestCase):
    """apply_parameter_edits against a real FreeCAD document and expression-bound solid."""

    def setUp(self):
        super().setUp()
        import FreeCAD as App

        self.doc = App.newDocument("ApplyEdits")
        self.addCleanup(lambda: App.closeDocument(self.doc.Name) if self.doc.Name in App.listDocuments() else None)
        self.sheet = self.doc.addObject("Spreadsheet::Sheet", "Params")
        # the kit sheet layout the tools read: values in column B, unit in C, source in D
        self.sheet.set("B1", "3 mm")
        self.sheet.setAlias("B1", "wall")
        self.sheet.set("C1", "mm")
        self.sheet.set("D1", "user_requirement")
        self.doc.recompute()
        self.box = self.doc.addObject("Part::Box", "Box")
        self.box.setExpression("Length", "Params.wall")
        self.doc.recompute()
        self.assertEqual(self.box.Length.Value, 3.0)

    def test_good_edit_applies_recomputes_and_reports(self):
        out = apply_parameter_edits(self.doc.Name, [{"name": "wall", "value": "4 mm"}])
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["applied"], ["wall"])
        self.assertIn("wall", out["text"])
        self.assertIn("Applied", out["text"])
        self.assertEqual(self.sheet.get("wall").Value, 4.0)
        self.assertEqual(self.box.Length.Value, 4.0)

    def test_unknown_name_is_refused_and_changes_nothing(self):
        out = apply_parameter_edits(self.doc.Name, [{"name": "nope", "value": "4 mm"}])
        self.assertFalse(out["ok"])
        self.assertEqual(self.sheet.get("wall").Value, 3.0)

    def test_empty_edits_are_refused(self):
        out = apply_parameter_edits(self.doc.Name, [])
        self.assertFalse(out["ok"])

    def test_bad_value_rolls_the_sheet_back(self):
        out = apply_parameter_edits(self.doc.Name, [{"name": "wall", "value": "not a number"}])
        self.assertFalse(out["ok"], out)
        self.assertEqual(self.sheet.get("wall").Value, 3.0, "the sheet must keep its old value")
        self.doc.recompute()
        self.assertEqual(self.box.Length.Value, 3.0, "the geometry must be unchanged")

    def test_multiple_edits_apply_in_one_call_and_update_bound_geometry(self):
        self.sheet.set("B2", "5 mm")
        self.sheet.setAlias("B2", "width")
        self.sheet.set("C2", "mm")
        self.box.setExpression("Width", "Params.width")
        self.doc.recompute()
        out = apply_parameter_edits(self.doc.Name, [
            {"name": "wall", "value": "4 mm"}, {"name": "width", "value": "7 mm"},
        ])
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["applied"], ["wall", "width"])
        self.assertEqual((self.sheet.get("wall").Value, self.sheet.get("width").Value), (4.0, 7.0))
        self.assertEqual((self.box.Length.Value, self.box.Width.Value), (4.0, 7.0))

    def test_wrong_units_are_refused_before_any_cell_changes(self):
        self.sheet.set("B2", "5 mm")
        self.sheet.setAlias("B2", "width")
        self.sheet.set("C2", "mm")
        out = apply_parameter_edits(self.doc.Name, [
            {"name": "wall", "value": "4 mm"}, {"name": "width", "value": "45 deg"},
        ])
        self.assertFalse(out["ok"])
        self.assertEqual(self.sheet.get("wall").Value, 3.0)

    def test_dependent_rebuild_failure_restores_every_edit(self):
        from types import SimpleNamespace

        calls = []

        def rebuild(_doc, _spec, names, _reference_status):
            calls.append(list(names))
            if len(calls) == 1:
                return [SimpleNamespace(ok=False, stage="shell", message="forced failure")], ""
            return [], ""

        with mock.patch("oscar.tools_kit_params._rebuild_dependents", side_effect=rebuild):
            out = apply_parameter_edits(self.doc.Name, [{"name": "wall", "value": "4 mm"}])
        self.assertFalse(out["ok"], out)
        self.assertIn("all edits were restored", out["text"])
        self.assertEqual(self.sheet.get("wall").Value, 3.0)
        self.assertEqual(self.box.Length.Value, 3.0)
        self.assertEqual(len(calls), 2)

    def test_applied_values_survive_save_and_reopen(self):
        import FreeCAD as App
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "parameter-apply.FCStd")
            out = apply_parameter_edits(self.doc.Name, [{"name": "wall", "value": "6 mm"}])
            self.assertTrue(out["ok"], out)
            self.doc.recompute()
            self.doc.saveAs(path)
            App.closeDocument(self.doc.Name)
            reopened = App.openDocument(path)
            self.doc = reopened
            self.assertEqual(reopened.getObject("Params").get("wall").Value, 6.0)
            self.assertEqual(reopened.getObject("Box").Length.Value, 6.0)

    def test_derived_row_needs_a_formula(self):
        self.sheet.set("B2", "=wall * 2")
        self.sheet.setAlias("B2", "tall")
        self.doc.recompute()
        out = apply_parameter_edits(self.doc.Name, [{"name": "tall", "value": "9 mm"}])
        self.assertFalse(out["ok"])
        self.assertIn("derived", out["text"])


class BuildSummaryAndLogsTest(PanelTestCase):
    """Build Summary stays in Chat; Logs is the separate execution timeline."""

    def test_build_summary_renders_once_in_chat(self):
        self.p._append(Block.ASSISTANT, "I built the bracket.")
        self.p._append(Block.SUMMARY, "## Build Summary\n- Outcome: **validated**")
        self.p._render()
        chat = self.p.log.toPlainText()
        self.assertIn("validated", chat)
        self.assertEqual(chat.count("Build Summary"), 1)

    def test_unchanged_summary_is_not_relaid_out(self):
        self.p._append(Block.SUMMARY, "## Build Summary\n- Outcome: validated")
        self.p._render()
        with mock.patch.object(self.p.log, "setHtml", wraps=self.p.log.setHtml) as set_html:
            self.p._render()
            self.p._render()
        self.assertEqual(set_html.call_count, 0)

    def test_log_content_is_cached_when_no_execution_event_changed(self):
        view = LogsView()
        events = [{"status": "SUCCESS", "timestamp": "10:02:03", "tool": "add_pad",
                   "title": "Base plate"}]
        with mock.patch.object(view.log_view, "setHtml", wraps=view.log_view.setHtml) as set_html:
            view.load_events(events)
            view.load_events(events)
        self.assertEqual(set_html.call_count, 1)

    def test_log_event_keeps_result_and_snapshot_metadata(self):
        view = LogsView()
        event = {
            "id": "chat-a:step-1", "timestamp": "10:02:03", "status": "SUCCESS",
            "tool": "add_pad", "title": "Base plate", "duration": 0.4,
            "input": "length=20 mm", "result": "Created BasePlate; recompute passed.",
            "delta": "+BasePlate", "validation": "valid", "snapshot": "snapshots/001.png",
            "snapshot_key": "call_001",
        }
        view.load_events([event])
        rendered = view.log_view.toPlainText()
        self.assertIn("add_pad", rendered)
        self.assertIn("length=20 mm", rendered)
        self.assertIn("Created BasePlate", rendered)
        self.assertIn("+BasePlate", rendered)
        self.assertIn("valid", rendered)
        self.assertIn("snapshots/001.png", rendered)
        self.assertIn("es-img://step/call_001", view.log_view.toHtml())

    def test_log_snapshot_resolves_from_the_shared_thumbnail_cache(self):
        import os
        import tempfile
        from oscar.panel_blocks import ThumbCache
        from PySide import QtGui

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "snapshot.png")
            image = QtGui.QImage(8, 8, QtGui.QImage.Format_RGB32)
            image.fill(QtGui.QColor("blue"))
            self.assertTrue(image.save(path))
            thumbs = ThumbCache()
            self.assertIsNotNone(thumbs.load("step-log", path))
            view = LogsView()
            view.load_events([{"status": "SUCCESS", "tool": "pad", "snapshot": path,
                               "snapshot_key": "step-log"}], thumbs)
            view.log_view.register_thumbnails(thumbs)
            self.assertTrue(view.log_view.has_resource("step-log"))

    def test_log_events_include_current_chat_workflow_statuses(self):
        from types import SimpleNamespace

        ledger = {"intent": "mesh_reconstruct", "steps": [
            {"id": "M1", "title": "Import and source identity", "status": "blocked",
             "note": "mesh identified but no source hash is on record"},
            {"id": "M14", "title": "Final delivery", "status": "pending", "note": ""},
        ]}
        with mock.patch.object(self.p, "_current_session", return_value={"ledger": ledger}):
            events = self.p._log_events()
        view = LogsView()
        view.load_events(events)
        rendered = view.log_view.toPlainText()
        self.assertIn("M1", rendered)
        self.assertIn("BLOCKED", rendered)
        self.assertIn("no source hash", rendered)
        self.assertIn("M14", rendered)
        self.assertIn("PENDING", rendered)

    def test_log_event_maps_execution_arguments_result_delta_and_validation(self):
        from types import SimpleNamespace

        card = self.p._append(Block.TOOL, "Ran 1 step")
        step = Step("add_pad", "Base plate", ok=True, dur=0.4)
        step.call_id = "call_77"
        step.view = SimpleNamespace(
            started_at=1_700_000_000.0, name="add_pad", title="Base plate", dur_s=0.4,
            args_summary="length=20 mm", args_json="", text="Created BasePlate",
            delta={"added": ["BasePlate"]}, health={"line": "valid"},
            snapshot_path="snapshots/077.png", call_id="call_77",
        )
        card.steps.append(step)
        with mock.patch.object(self.p, "_current_session", return_value=None):
            events = self.p._log_events()
        event = next(item for item in events if item.get("tool") == "add_pad")
        self.assertEqual(event["input"], "length=20 mm")
        self.assertEqual(event["result"], "Created BasePlate")
        self.assertEqual(event["delta"], "added: BasePlate")
        self.assertEqual(event["validation"], "valid")
        self.assertTrue(event["snapshot"].endswith("snapshots/077.png"))

    def test_chat_appends_a_new_message_without_replacing_the_transcript(self):
        self.p._append(Block.USER, "first request")
        self.p._render()
        with mock.patch.object(self.p.log, "setHtml", wraps=self.p.log.setHtml) as set_html:
            self.p._append(Block.ASSISTANT, "second response")
            self.p._render()
        self.assertEqual(set_html.call_count, 0)
        self.assertIn("first request", self.p.log.toPlainText())
        self.assertIn("second response", self.p.log.toPlainText())

    def test_chat_replaces_only_the_streaming_tail(self):
        tail = self.p._append(Block.ASSISTANT, "partial")
        self.p._render()
        with mock.patch.object(self.p.log, "setHtml", wraps=self.p.log.setHtml) as set_html:
            tail.text = "complete"
            self.p._render()
        self.assertEqual(set_html.call_count, 0)
        self.assertIn("complete", self.p.log.toPlainText())
        self.assertNotIn("partial", self.p.log.toPlainText())

    def test_log_contains_execution_events_but_not_chat_or_summary(self):
        from datetime import datetime
        from types import SimpleNamespace

        self.p._append(Block.USER, "build a bracket")
        self.p._append(Block.ASSISTANT, "I built a bracket.")
        card = self.p._append(Block.TOOL, "Ran 2 steps · 1.0s")
        step = Step("add_pad", "Base plate", ok=True, dur=0.4)
        step.view = SimpleNamespace(started_at=1_700_000_000.0)
        card.steps.append(step)
        self.p._append(Block.SUMMARY, "## Build Summary\n- Outcome: validated")
        lines = self.p._log_lines()
        event = next(l for l in lines if "add_pad" in l)
        self.assertIn("SUCCESS", event)
        self.assertTrue(event.startswith(datetime.fromtimestamp(1_700_000_000.0).strftime("%H:%M:%S")))
        self.assertFalse(any("build a bracket" in l for l in lines))
        self.assertFalse(any("I built a bracket" in l for l in lines))
        self.assertFalse(any("Build Summary" in l for l in lines))

    def test_chat_hides_execution_cards_but_keeps_the_build_summary(self):
        self.p._append(Block.USER, "Build a bracket")
        self.p._append(Block.ASSISTANT, "I built a bracket.\n\nIt passed the checks.")
        card = self.p._append(Block.TOOL, "Ran 1 step")
        card.steps.append(Step("add_pad", "Base plate", ok=True, dur=0.4))
        self.p._append(Block.TOOL, "⚠ Could not open the requested file")
        self.p._append(Block.SUMMARY, "## Build Summary\n- Outcome: validated")
        self.p._render()
        chat = self.p.log.toPlainText()
        self.assertIn("Build a bracket", chat)
        self.assertIn("I built a bracket", chat)
        self.assertIn("Could not open the requested file", chat)  # non-execution notices remain conversational
        self.assertNotIn("add_pad", chat)
        # the Build Summary is a card in Chat: no Summary tab exists to hold it
        self.assertEqual(chat.count("Build Summary"), 1)
        self.assertIn("validated", chat)

    def test_log_contains_execution_events_only_and_keeps_the_summary_separate(self):
        self.p._append(Block.USER, "Build a bracket")
        self.p._append(Block.ASSISTANT, "I built a bracket.")
        card = self.p._append(Block.TOOL, "Ran 1 step")
        card.steps.append(Step("add_pad", "Base plate", ok=True, dur=0.4))
        self.p._append(Block.SUMMARY, "## Build Summary\n- Outcome: validated")
        lines = self.p._log_lines()
        self.assertTrue(any("add_pad" in line and "SUCCESS" in line for line in lines))
        joined = "\n".join(lines)
        self.assertNotIn("Build a bracket", joined)
        self.assertNotIn("I built a bracket", joined)
        self.assertNotIn("Build Summary", joined)

    def test_panel_creates_one_authoritative_parameter_table(self):
        from PySide import QtWidgets

        tables = self.p.findChildren(QtWidgets.QTableWidget, "oscarParamsTable")
        self.assertEqual(len(tables), 1)

    def test_logs_tab_does_not_replace_the_chat_build_summary(self):
        self.p._append(Block.SUMMARY, "## Build Summary\n- Outcome: validated")
        self.p._render()
        self.assertIn("validated", self.p.log.toPlainText())
        self.p.tabs.setCurrentIndex(2)
        self.assertEqual(self.p.tabs.tabText(2), "Logs")
        self.assertFalse(hasattr(self.p, "summary_view"))

    def test_logs_host_refresh_reads_project_step_list(self):
        from types import SimpleNamespace

        row = {"index": 1, "ts": "2026-11-16T12:00:00", "tool": "create_object", "ok": True,
               "title": "Created flange", "result_head": "Created flange", "geometry": True,
               "snapshot": "snapshots/001.png", "call_id": "call-001"}
        record = SimpleNamespace(to_line=lambda: row)
        project = SimpleNamespace(root="/tmp/project", read_steps=lambda: [record])
        with mock.patch.object(self.p, "_current_session", return_value={"_project": project}), \
                mock.patch.object(self.p, "_project_of", side_effect=lambda session: session["_project"]):
            self.p._refresh_logs()
        self.assertEqual(len(self.p.logs_view.entries()), 1)
        self.assertEqual(self.p.logs_view.entries()[0]["op"], "create_object")


class _FakeChoice:
    def __init__(self, key, label, role):
        self.key, self.label, self.role = key, label, role
        self.on_click = None
        self.remote = True


class QuestionCardTest(PanelTestCase):
    def open_question(self, qid="7") -> tuple:
        block = self.p._question_append("Run this script?\nThe script writes a file.")
        block.key = qid
        block.detail = "pending"
        block.actions = [("oscar://answer/%s/allow" % qid, "Allow"),
                         ("oscar://answer/%s/deny" % qid, "Deny")]
        settled: list[str] = []

        def settle(key: str) -> None:   # the surface's own guard: the first answer wins
            if not settled:
                settled.append(key)

        choices = [_FakeChoice("allow", "Allow", "AcceptRole"), _FakeChoice("deny", "Deny", "RejectRole")]
        self.p._open_questions[qid] = {"settle": settle, "block": block, "choices": choices}
        return block, settled

    def test_pending_card_renders_the_choice_anchors(self):
        block, _ = self.open_question()
        html = block.html()
        self.assertIn("oscar://answer/7/allow", html)
        self.assertIn("Allow", html)
        self.assertIn("Run this script?", html)

    def test_the_real_surface_settles_and_flips_the_card(self):
        """End to end through ``_question_surface``: the loop runs, the click settles it, the
        card lands in its approved state and the key travels back to netguard."""
        from PySide import QtCore

        info = {"id": 9, "op": "test", "title": "Run this script?", "text": "writes a file", "detailed": ""}
        choices = [_FakeChoice("allow", "Allow", "AcceptRole"), _FakeChoice("deny", "Deny", "RejectRole")]
        QtCore.QTimer.singleShot(80, lambda: self.p._on_question_answer("9/allow"))
        key = self.p._question_surface(info, choices, 2.0, lambda: False, lambda: None, [])
        self.assertEqual(key, "allow")
        block = next(b for b in self.p.blocks if b.kind == Block.QUESTION)
        self.assertEqual(block.detail, "approved")
        self.assertIn("approved", block.html())
        self.assertNotIn("9", self.p._open_questions)

    def test_answer_settles_and_flips_the_card(self):
        block, settled = self.open_question()
        self.p._on_question_answer("7/allow")
        self.assertEqual(settled, ["allow"])
        self.p._settle_question(block, "allow", [_FakeChoice("allow", "Allow", "AcceptRole")], False, 120)
        self.assertEqual(block.detail, "approved")
        self.assertIn("approved", block.html())

    def test_reject_records_the_optional_reason(self):
        block, settled = self.open_question()
        with mock.patch.object(QtWidgets.QInputDialog, "getText", return_value=("too thin", True)):
            self.p._on_question_answer("7/deny")
        self.assertEqual(settled, ["deny"])
        self.assertEqual(block.reason, "too thin")
        self.p._settle_question(block, "deny", [_FakeChoice("deny", "Deny", "RejectRole")], False, 120)
        self.assertEqual(block.detail, "rejected")
        self.assertIn("too thin", block.html())

    def test_answer_of_another_chat_is_refused(self):
        block, settled = self.open_question()
        # simulate: the question's chat is not the shown one → the card is not in self.p.blocks
        saved = self.p.blocks
        self.p.blocks = []
        try:
            self.p._on_question_answer("7/allow")
        finally:
            self.p.blocks = saved
        self.assertEqual(settled, [], "another chat must not answer the question")
        self.assertEqual(block.detail, "pending")

    def test_stale_answer_is_ignored(self):
        _block, settled = self.open_question()
        self.p._on_question_answer("7/allow")
        self.assertEqual(settled, ["allow"])
        self.p._on_question_answer("7/allow")   # the question is closed now
        self.assertEqual(settled, ["allow"], "a settled question stays settled")

    def test_timeout_and_cancelled_states(self):
        block, _ = self.open_question("8")
        self.p._settle_question(block, "", [], False, 120)
        self.assertEqual(block.detail, "timeout")
        self.assertIn("No answer — denied after 120 s", block.html())
        self.p._settle_question(block, "", [], True, 120)
        self.assertEqual(block.detail, "cancelled")
        self.assertIn("the turn was stopped", block.html())

    def test_panel_registers_as_the_question_surface(self):
        self.assertEqual(netguard.panel_surface, self.p._question_surface)   # bound methods compare by target
        self.p._shutdown()
        self.assertIsNone(netguard.panel_surface)

    def test_clarification_option_is_sent_verbatim_and_only_once(self):
        block = self.p._question_append("Which unit should I use?\nThe drawing is ambiguous.")
        block.key = str(block.bid)
        block.detail = "clarification_pending"
        block.actions = [(f"oscar://clarify/{block.key}/0", "Millimetres"),
                         (f"oscar://clarify/{block.key}/1", "Inches")]
        with mock.patch.object(self.p, "_send_text") as send:
            self.p._on_clarification_option(f"{block.key}/1")
            self.p._on_clarification_option(f"{block.key}/1")
        self.assertEqual(block.detail, "answered")
        self.assertEqual(block.reason, "Inches")
        send.assert_called_once_with("Inches")

    def test_clarification_answer_from_another_chat_cannot_be_used(self):
        block = self.p._question_append("Which unit should I use?\nThe drawing is ambiguous.")
        block.key = str(block.bid)
        block.detail = "clarification_pending"
        block.actions = [(f"oscar://clarify/{block.key}/0", "Millimetres")]
        visible = self.p.blocks
        self.p.blocks = []
        try:
            with mock.patch.object(self.p, "_send_text") as send:
                self.p._on_clarification_option(f"{block.key}/0")
        finally:
            self.p.blocks = visible
        self.assertEqual(block.detail, "clarification_pending")
        send.assert_not_called()

    def test_typed_custom_answer_is_preserved_and_new_request_closes_question(self):
        block = self.p._question_append("Which unit should I use?\nThe drawing is ambiguous.")
        block.detail = "clarification_pending"
        with mock.patch("oscar.flow_intent.answers_open_question", return_value=True):
            self.p._settle_typed_clarification("use inches, please", [])
        self.assertEqual((block.detail, block.reason), ("answered", "use inches, please"))
        block.detail = "clarification_pending"
        with mock.patch("oscar.flow_intent.answers_open_question", return_value=False):
            self.p._settle_typed_clarification("Build a gear instead", [])
        self.assertEqual((block.detail, block.reason), ("superseded", "Build a gear instead"))


if __name__ == "__main__":
    unittest.main()
