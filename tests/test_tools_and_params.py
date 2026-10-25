"""Schema and params tests. Runs under the bundled python with a stubbed
FreeCADGui (the console interpreter does not provide it)."""

import json
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Point the settings store at a temp file so tests never touch the user's
# real profiles (their API keys live there).
os.environ["OSCAR_AI_SETTINGS"] = os.path.join(
    tempfile.mkdtemp(prefix="oscar_params_test_"), "settings.json"
)

if "FreeCADGui" not in sys.modules:
    try:
        import FreeCADGui  # noqa: F401
    except Exception:
        sys.modules["FreeCADGui"] = types.ModuleType("FreeCADGui")

import FreeCAD  # noqa: E402

from oscar import tools  # noqa: E402


class ToolSchemaTest(unittest.TestCase):
    def test_every_tool_has_valid_schema(self):
        names = set()
        for tool in tools.get_tools():
            self.assertTrue(tool.name, "tool without a name")
            self.assertNotIn(tool.name, names, f"duplicate tool {tool.name}")
            names.add(tool.name)
            self.assertGreater(len(tool.description), 40, f"{tool.name}: description too thin")
            self.assertEqual(tool.parameters.get("type"), "object", f"{tool.name}: schema root must be object")
            for key in tool.parameters.get("required", []):
                self.assertIn(key, tool.parameters["properties"], f"{tool.name}: required '{key}' missing")
            # JSON-serializable (it crosses the wire)
            json.dumps(tool.parameters)

    # The tools that existed before the structured build flow (names never change).
    EXPECTED_V1 = {
        "create_document", "create_object", "edit_object", "delete_object",
        "execute_code", "execute_code_async", "execute_code_headless",
        "get_async_status", "get_objects", "get_object", "get_view",
        "list_documents", "reload_document", "get_parts_list",
        "insert_part_from_library", "import_file", "run_fem_analysis",
        "apply_feature_plan", "feature_tree_status", "reapply_plan_feature", "download_file",
    }
    # The 22 tools of the build flow, two of them optional (OPTIONAL_TOOL_NAMES).
    NEW_TOOL_NAMES = {
        "plan_set", "plan_update", "record_note", "save_document", "export_model",
        "export_rebuild_script", "validate_model", "check_fit", "find_reference_part",
        "source_reference_part", "import_reference_part", "measure_reference_part",
        "enclosure_spec", "build_enclosure", "add_cutout", "add_vents", "set_parameters",
        "get_parameters", "bind_expression", "set_parameter", "web_search", "fetch_url",
    }

    OPTIONAL_TOOL_NAMES = {"web_search", "fetch_url"}   # always registered (tools_web); web_search says how to configure its backend

    def test_expected_tool_set(self):
        """Strict equality: exactly 43 tools, ``web_search`` / ``fetch_url`` included."""
        names = {t.name for t in tools.get_tools()}
        self.assertEqual(len(self.EXPECTED_V1), 21)
        self.assertEqual(len(self.NEW_TOOL_NAMES), 22)
        self.assertLessEqual(self.EXPECTED_V1, names, "an original tool disappeared")
        self.assertLessEqual(self.OPTIONAL_TOOL_NAMES, names, "web_search and fetch_url are part of the surface")
        expected = self.EXPECTED_V1 | self.NEW_TOOL_NAMES
        self.assertEqual(names, expected, f"missing: {sorted(expected - names)}; unknown: {sorted(names - expected)}")
        self.assertEqual(len(names), 43)
        self.assertEqual(len(tools.get_tools()), len(names), "a tool name is registered twice")

    def test_unknown_tool_is_error(self):
        text, image, is_error = tools.run_tool("no_such_tool", {})
        self.assertTrue(is_error)
        self.assertIn("Unknown tool", text)

    def test_list_documents_runs(self):
        # Executes against the real FreeCAD runtime (GUI dispatch degrades to
        # a direct call when no Qt loop owns the queue in console mode).
        text, image, is_error = tools.run_tool("list_documents", {})
        self.assertFalse(is_error, text)
        docs = json.loads(text)
        self.assertIsInstance(docs, list)


class ImportFileTest(unittest.TestCase):
    def test_import_step_roundtrip(self):
        import Part

        workdir = tempfile.mkdtemp(prefix="oscar_import_test_")
        path = os.path.join(workdir, "block.step")
        Part.makeBox(3, 4, 5).exportStep(path)

        text, _image, is_error = tools.run_tool("import_file", {"path": path})
        self.assertFalse(is_error, text)
        self.assertIn("block.step", text)

        import FreeCAD as App

        found = None
        for doc in App.listDocuments().values():
            for obj in doc.Objects:
                # other tests leave documents with broken shapes around;
                # reading their Volume raises — skip them
                try:
                    shape = getattr(obj, "Shape", None)
                    if shape is not None and abs(shape.Volume - 60.0) < 1e-6:
                        found = obj
                except Exception:
                    continue
        self.assertIsNotNone(found, "imported STEP geometry not found")

    def test_missing_file_is_error(self):
        # Handler failures surface as "Failed…" text the model can read
        # AND as is_error, so the model (and the panel's ✗) know it failed.
        text, _image, is_error = tools.run_tool("import_file", {"path": "/no/such/file.step"})
        self.assertTrue(is_error)
        self.assertIn("Failed to import file", text)
        self.assertIn("file not found", text)

    def test_unsupported_extension_is_error(self):
        fd, path = tempfile.mkstemp(suffix=".weird")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        text, _image, _is_error = tools.run_tool("import_file", {"path": path})
        self.assertIn("Failed to import file", text)
        self.assertIn("unsupported file type", text)


class ParamsTest(unittest.TestCase):
    def test_profile_roundtrip(self):
        from oscar import params

        profiles = [
            params.Profile(name="work-claude", provider="anthropic",
                           base_url="https://api.anthropic.com",
                           model="claude-sonnet-4-5", api_key="sk-1"),
            params.Profile(name="local-ollama", provider="openai_compat",
                           base_url="http://localhost:11434/v1",
                           model="qwen2.5:7b"),
        ]
        params.save_profiles(profiles)
        loaded = params.load_profiles()
        self.assertEqual([p.name for p in loaded], ["work-claude", "local-ollama"])
        self.assertEqual(loaded[1].base_url, "http://localhost:11434/v1")
        self.assertEqual(loaded[0].api_key, "sk-1")
        params.set_active_profile("local-ollama")
        self.assertEqual(params.active_profile().model, "qwen2.5:7b")

    def test_corrupt_settings_file_falls_back(self):
        from oscar import params

        with open(params.settings_path(), "w", encoding="utf-8") as f:
            f.write("{not json")
        profiles = params.load_profiles()
        self.assertGreaterEqual(len(profiles), 1)
        # and the fallback state is writable again
        params.set_bool("PanelAutoOpen", True)
        self.assertTrue(params.get_bool("PanelAutoOpen", False))

    def test_profiles_persist_to_disk_immediately(self):
        # The old parameter backend lost settings unless the app exited
        # cleanly; the file backend must have bytes on disk right away.
        from oscar import params

        params.save_profiles([
            params.Profile(name="keyed", provider="anthropic",
                           api_key="sk-ant-test-123", model="claude-sonnet-4-5")
        ])
        with open(params.settings_path(), "r", encoding="utf-8") as f:
            on_disk = f.read()
        self.assertIn("sk-ant-test-123", on_disk)

    def test_backup_and_restore_state(self):
        from oscar import params

        params.save_profiles([
            params.Profile(name="mine", provider="anthropic", api_key="sk-keep")
        ])
        snapshot = params.backup_state()
        params.save_profiles([
            params.Profile(name="temp", provider="openai_compat", api_key="sk-temp")
        ])
        params.restore_state(snapshot)
        self.assertEqual([p.name for p in params.load_profiles()], ["mine"])
        self.assertEqual(params.load_profiles()[0].api_key, "sk-keep")

    def test_predefined_limits(self):
        from oscar import params

        # preset default for Anthropic: room for thinking + long tool calls
        temp, max_tokens = params.predefined_limits("anthropic", "", "")
        self.assertEqual((temp, max_tokens), (0.2, 32000))
        # every Claude 4+ model accepts 32000; Claude 3.x keep their caps
        _, max_tokens = params.predefined_limits("anthropic", "", "claude-haiku-4-5")
        self.assertEqual(max_tokens, 32000)
        _, max_tokens = params.predefined_limits("anthropic", "", "claude-3-5-haiku-20241022")
        self.assertEqual(max_tokens, 8192)
        # OpenAI preset default and model override
        _, max_tokens = params.predefined_limits("openai_compat", "https://api.openai.com/v1", "")
        self.assertEqual(max_tokens, 16384)
        _, max_tokens = params.predefined_limits(
            "openai_compat", "https://api.openai.com/v1", "gpt-4.1-mini"
        )
        self.assertEqual(max_tokens, 32768)
        # unknown custom endpoint falls back to the generic default
        _, max_tokens = params.predefined_limits("openai_compat", "http://10.0.0.5:8000/v1", "my-model")
        self.assertEqual(max_tokens, 8192)
        # z.ai: GLM always thinks and the thinking counts against the cap
        # (8192 was used up by reasoning alone); the coding-plan endpoint
        # matches the preset by host, and GLM via any host gets the cap
        coding = "https://api.z.ai/api/coding/paas/v4"
        self.assertEqual(params.preset_for("openai_compat", coding)["label"], "Z.ai (GLM)")
        _, max_tokens = params.predefined_limits("openai_compat", coding, "glm-5.3")
        self.assertEqual(max_tokens, 32768)
        _, max_tokens = params.predefined_limits(
            "openai_compat", "https://openrouter.ai/api/v1", "z-ai/glm-5.3")
        self.assertEqual(max_tokens, 32768)
        # exact matches and the generic fallback are unchanged by host matching
        self.assertEqual(params.preset_for("openai_compat", "https://api.deepseek.com/v1")["label"],
                         "DeepSeek")
        self.assertEqual(params.preset_for("openai_compat", "http://10.0.0.5:8000/v1")["label"],
                         "Custom (OpenAI-compatible)")
        self.assertEqual(params.preset_for("anthropic", "https://my-proxy.example/anthropic")["label"],
                         "Anthropic (Claude)")

    def test_presets_carry_predefined_values(self):
        from oscar import params

        for preset in params.PROVIDER_PRESETS:
            self.assertIn("temperature", preset, preset["label"])
            self.assertIn("max_tokens", preset, preset["label"])


class MarkdownTest(unittest.TestCase):
    def _md(self):
        try:
            from oscar.panel import md_to_html
        except Exception:
            self.skipTest("Qt unavailable in this interpreter")
        return md_to_html

    def test_md_to_html(self):
        # panel imports land in a GUI-less interpreter too once FreeCADGui is
        # stubbed — but panel pulls in QtGui; skip gracefully if unavailable.
        md_to_html = self._md()
        html = md_to_html("**bold** and `code` and <script>")
        self.assertIn("<b>bold</b>", html)
        self.assertIn("<code", html)
        self.assertNotIn("<script>", html)
        block = md_to_html("```\nPart.makeBox(1,2,3)\n```")
        self.assertIn("<pre><code>", block)

    def test_markdown_table(self):
        md_to_html = self._md()
        out = md_to_html("| dim | mm |\n|---|---|\n| length | 80 |\n| width | 40 |")
        self.assertIn("<table", out)
        self.assertIn("<th", out)
        self.assertIn("dim</th>", out)
        self.assertIn("length</td>", out)
        self.assertIn("80</td>", out)
        # separator row is metadata, not a data row
        self.assertNotIn("---", out)
        # non-table pipes are left alone
        out = md_to_html("a | b\nplain line")
        self.assertNotIn("<table", out)

    def test_markdown_link(self):
        md_to_html = self._md()
        out = md_to_html("see [docs](https://example.com/x) now")
        self.assertIn("<a href='https://example.com/x'>docs</a>", out)


class EngineWiringTest(unittest.TestCase):
    """Every _engine().… call in a tool handler must name a real engine
    method — the oscar/cad refactor renamed four and they broke silently."""

    def test_handlers_only_call_existing_engine_methods(self):
        import inspect
        import re

        from oscar import cad

        engine = cad.get_engine()
        checked = 0
        for tool in tools.get_tools():
            source = inspect.getsource(tool.handler)
            for name in re.findall(r"_engine\(\)\.(\w+)", source):
                checked += 1
                self.assertTrue(
                    hasattr(engine, name),
                    f"tool '{tool.name}' calls missing engine method '{name}'",
                )
        self.assertGreater(checked, 12)  # the scan actually found calls



class _CloseNewDocuments(unittest.TestCase):
    """Base for tests that create documents: every document the test left open is closed afterwards (a leaked document
    shows up in later tests' workspace snapshots and, in a long run, pushes other documents out of its listing)."""

    def setUp(self):
        import FreeCAD as App

        before = set(App.listDocuments())

        def close_new():
            for name in set(App.listDocuments()) - before:
                try:
                    App.closeDocument(name)
                except Exception:
                    pass

        self.addCleanup(close_new)


class EditObjectTest(_CloseNewDocuments):
    def test_edit_placement_roundtrip(self):
        # Regression: the handler double-wrapped properties, so EVERY
        # edit_object call failed with a bogus 'Properties' attribute error.
        import FreeCAD as App

        doc = App.newDocument("EditTestDoc")
        tools.run_tool("create_object", {
            "doc_name": "EditTestDoc", "obj_type": "Part::Cylinder",
            "obj_name": "Blank", "obj_properties": {"Radius": 30, "Height": 8},
            "include_screenshot": False,
        })
        text, _image, is_error = tools.run_tool("edit_object", {
            "doc_name": "EditTestDoc", "obj_name": "Blank",
            "obj_properties": {
                "Placement": {
                    "Base": {"x": 0, "y": 0, "z": 4},
                    "Rotation": {"Axis": {"x": 0, "y": 0, "z": 1}, "Angle": 15},
                }
            },
            "include_screenshot": False,
        })
        self.assertFalse(is_error, text)
        placement = doc.getObject("Blank").Placement
        self.assertAlmostEqual(placement.Base.z, 4.0, places=5)
        self.assertAlmostEqual(placement.Rotation.Angle * 57.2957795, 15.0, places=3)

    def test_draft_polygon_creates(self):
        # Regression: FreeCAD 1.1.3's make_polygon requires nfaces.
        import FreeCAD as App

        doc = App.newDocument("DraftPolyDoc")
        text, _image, is_error = tools.run_tool("create_object", {
            "doc_name": "DraftPolyDoc", "obj_type": "Draft::Polygon",
            "obj_name": "Tooth", "obj_properties": {"FacesNumber": 12, "Radius": 10},
            "include_screenshot": False,
        })
        if is_error and "Draft" in text:
            self.skipTest("Draft factories unavailable in console mode")
        self.assertFalse(is_error, text)
        # console mode falls back to Part::Feature; the GUI smoke asserts
        # the real Draft::Polygon type.
        self.assertGreaterEqual(len(doc.Objects), 1, text)


class FuzzyDocNameTest(_CloseNewDocuments):
    def test_resolve_doc_tolerates_model_near_misses(self):
        import FreeCAD as App

        from oscar.cad.engine import _resolve_doc

        # close what we open: a lingering unsaved 'Gear_System' makes later
        # tests that create a document of that name get a uniquified one
        for name in ("Gear_System", "GearTrain", "Other"):
            self.addCleanup(App.closeDocument, App.newDocument(name).Name)
        self.assertIsNotNone(_resolve_doc("Gear_System"))       # exact
        self.assertIsNotNone(_resolve_doc("GEAR_SYSTEM"))       # case
        self.assertIsNotNone(_resolve_doc("Gear_Sys"))          # unique prefix
        self.assertIsNone(_resolve_doc("Gear"))                 # ambiguous prefix
        self.assertIsNone(_resolve_doc("Nope_Doc"))

    def test_async_code_warnings(self):
        from oscar.cad.scripting import async_code_warnings

        self.assertIn("ActiveDocument", async_code_warnings("App.ActiveDocument.Box"))
        self.assertIn("FreeCADGui", async_code_warnings("import FreeCADGui"))
        self.assertEqual(async_code_warnings("x = Part.makeBox(1, 2, 3)"), [])


class TurnRevertTest(_CloseNewDocuments):
    def test_snapshot_revert_roundtrip(self):
        import FreeCAD as App

        from oscar import cad

        engine = cad.get_engine()
        doc = App.newDocument("RevertDoc")
        doc.addObject("Part::Box", "Before")
        doc.recompute()
        snapshot = engine.begin_turn_snapshot()
        self.assertIn("RevertDoc", snapshot["docs"])
        doc.addObject("Part::Cylinder", "After")
        doc.recompute()
        if doc.UndoCount == 0:
            self.skipTest("undo stack disabled in console mode (GUI covers it)")
        engine.revert_turn(snapshot)
        self.assertIsNone(doc.getObject("After"))
        self.assertIsNotNone(doc.getObject("Before"))

    def test_revert_closes_documents_created_in_turn(self):
        import FreeCAD as App

        from oscar import cad

        engine = cad.get_engine()
        snapshot = engine.begin_turn_snapshot()
        App.newDocument("CreatedByTurn")
        engine.revert_turn(snapshot)
        self.assertNotIn("CreatedByTurn", App.listDocuments())


class ToolBlockRenderTest(unittest.TestCase):
    def _blocks(self):
        try:
            from oscar.panel import Block, Step
        except Exception:
            self.skipTest("Qt unavailable in this interpreter")
        return Block, Step

    def test_activity_card_collapsed_and_expanded(self):
        Block, Step = self._blocks()
        card = Block(Block.TOOL, "✻ Ran 2 steps · 3.1s")
        card.steps = [
            Step("create_document", summary="create_document(name=Doc)", ok=True, dur=0.2,
                 detail="Document 'Doc' created."),
            Step("import_file", summary="import_file(path=x.stl)", ok=False, dur=2.9,
                 detail="failed hard\nsecond line"),
        ]
        collapsed = card.html()
        self.assertIn(f"href='tool:{card.bid}'", collapsed)
        self.assertIn("✻ Ran 2 steps", collapsed)
        self.assertNotIn("created.", collapsed)  # step details hidden
        card.expanded = True
        expanded = card.html()
        self.assertIn("create_document(name=Doc)", expanded)
        self.assertIn(f"href='step:{card.bid}:0'", expanded)
        self.assertNotIn("Document &#x27;Doc&#x27; created.", expanded)  # step detail still collapsed
        card.steps[0].expanded = True
        self.assertIn("created.", card.html())
        # detail text is escaped
        card.steps[0].detail = "<script>alert(1)</script>"
        self.assertNotIn("<script>alert", card.html())

    def test_thought_steps_inside_card(self):
        # Thoughts are steps: one card per turn, interleaved chronologically.
        Block, Step = self._blocks()
        card = Block(Block.TOOL, "✻ Thought 2.2s · Ran 1 step · 14.0s")
        card.steps = [
            Step(is_thought=True, dur=0.8, detail="first hunch"),
            Step("create_document", summary="create_document(name=D)", ok=True,
                 dur=0.1, detail="Document 'D' created."),
            Step(is_thought=True, dur=1.4, detail="revised plan"),
        ]
        collapsed = card.html()
        self.assertIn("Thought 2.2s", collapsed)
        self.assertNotIn("hunch", collapsed)
        card.expanded = True
        expanded = card.html()
        self.assertIn("Thought · 0.8s", expanded)
        self.assertIn("Thought · 1.4s", expanded)
        self.assertIn("create_document(name=D)", expanded)
        self.assertNotIn("first hunch", expanded)   # thought bodies collapsed
        card.steps[2].expanded = True
        self.assertIn("revised plan", card.html())

    def test_live_thought_shows_tail_and_counter(self):
        Block, Step = self._blocks()
        card = Block(Block.TOOL, "")
        card.expanded = True
        card.t0 = 1.0
        step = Step(is_thought=True, detail="start " + "x" * 3000 + " end")
        step.t0 = 1.0
        card.steps = [step]
        rendered = card.html()
        self.assertIn("Working…", rendered)   # live card header counts up
        self.assertIn("thinking…", rendered)
        self.assertIn("end", rendered)        # recent tail visible
        self.assertNotIn("start", rendered)

    def test_cards_render_in_gray(self):
        # The old CSS palette(mid) form silently fell back to normal text
        # color inside QTextBrowser — muted colors must be explicit.
        Block, Step = self._blocks()
        card = Block(Block.TOOL, "✻ Ran 1 step · 1.0s")
        card.steps = [Step("t", summary="t", ok=True, dur=1.0, detail="d")]
        rendered = card.html()
        self.assertTrue("rgba(" in rendered or "#888888" in rendered)

    def test_plain_step_renders_folded_note(self):
        # Pre-tool remarks fold into the card as static italic notes —
        # not clickable, no timing.
        Block, Step = self._blocks()
        card = Block(Block.TOOL, "✻ Ran 1 step · 1.0s")
        card.expanded = True
        note = Step(summary="Setting up a document.")
        note.plain = True
        card.steps = [note]
        rendered = card.html()
        self.assertIn("Setting up a document.", rendered)
        self.assertNotIn(f"href='step:{card.bid}:0'", rendered)

    def test_live_card_can_be_minimized(self):
        # While a turn runs, the header must honor collapse: the ticking
        # "Working… Ns" line stays, the steps hide.
        Block, Step = self._blocks()
        card = Block(Block.TOOL, "")
        card.t0 = 1.0
        card.expanded = False
        live = Step(is_thought=True, detail="pondering")
        live.t0 = 1.0
        card.steps = [live]
        minimized = card.html()
        self.assertIn("Working…", minimized)
        self.assertIn("▸", minimized)
        self.assertNotIn("pondering", minimized)
        card.expanded = True
        expanded = card.html()
        self.assertIn("▾", expanded)
        self.assertIn("pondering", expanded)

    def test_running_step_shows_spinner(self):
        Block, Step = self._blocks()
        card = Block(Block.TOOL, "Working…")
        card.expanded = True
        card.steps = [Step("get_objects", summary="get_objects …")]
        self.assertIn("◌", card.html())


if __name__ == "__main__":
    unittest.main()
