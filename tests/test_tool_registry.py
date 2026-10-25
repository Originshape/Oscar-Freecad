"""Tool registry: the metadata matrix, ``specs()``, titles, clipping, argument
repair, the tool-module import loop and the ``run_tool`` hook pipeline (health -> project ->
ledger) against fakes. Real-document behaviour of the same pipeline lives in
``test_tool_registry_engine.py``."""

import os
import sys
import types
import unittest
from types import SimpleNamespace
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

from oscar import toolpipe, toolrun, tools  # noqa: E402
from oscar.cad.health_types import (  # noqa: E402
    Finding, HealthReport, ObjectDelta, StateSnapshot)
from oscar.toolctx import ToolContext, ToolOutcome, current_context, use_context  # noqa: E402

# The source of truth: name -> (category, mutating, geometry, phase mask over
# INTAKE SOURCE PLAN PARAMS BUILD VERIFY DELIVER).
MATRIX = {
    "list_documents": ("read", False, False, "XXXXXXX"),
    "get_objects": ("read", False, False, "XXXXXXX"),
    "get_object": ("read", False, False, "XXXXXXX"),
    "get_view": ("read", False, False, "XXXXXXX"),
    "plan_set": ("plan", False, False, "XXXXXXX"),
    "plan_update": ("plan", False, False, "XXXXXXX"),
    "record_note": ("plan", False, False, "XXXXXXX"),
    "create_document": ("session", True, False, "XXXXX--"),
    "reload_document": ("session", True, False, "----XXX"),
    "import_file": ("source", True, True, "-X--X--"),
    "download_file": ("net", False, False, "-X-----"),
    "get_parts_list": ("read", False, False, "-X--X--"),
    "insert_part_from_library": ("build", True, True, "----X--"),
    "create_object": ("build", True, True, "----XX-"),
    "edit_object": ("build", True, True, "----XX-"),
    "delete_object": ("build", True, True, "----XX-"),
    "execute_code": ("build", True, True, "-X-XXX-"),
    "execute_code_async": ("build", True, True, "----XX-"),
    "execute_code_headless": ("build", True, False, "----XX-"),
    "get_async_status": ("read", False, False, "----XX-"),
    "apply_feature_plan": ("build", True, True, "----X--"),
    "reapply_plan_feature": ("build", True, True, "----X--"),
    "feature_tree_status": ("verify", False, False, "----XXX"),
    "run_fem_analysis": ("fem", True, True, "----XX-"),
    "find_reference_part": ("source", False, False, "XXX----"),
    "source_reference_part": ("source", True, False, "XXX----"),
    "import_reference_part": ("source", True, True, "-X-----"),
    "measure_reference_part": ("source", True, False, "-X-X---"),
    "enclosure_spec": ("plan", True, False, "--XXX--"),
    "build_enclosure": ("build", True, True, "----X--"),
    "add_cutout": ("build", True, True, "----XX-"),
    "add_vents": ("build", True, True, "----XX-"),
    "set_parameters": ("params", True, True, "---XXX-"),
    "get_parameters": ("params", False, False, "--XXXXX"),
    "bind_expression": ("params", True, True, "---XXX-"),
    "set_parameter": ("params", True, True, "----XXX"),
    "validate_model": ("verify", False, False, "----XXX"),
    "check_fit": ("verify", False, False, "----XXX"),
    "save_document": ("deliver", True, False, "----XXX"),
    "export_model": ("deliver", True, False, "-----XX"),
    "export_rebuild_script": ("deliver", True, False, "-----XX"),
    "web_search": ("net", False, False, "XXX----"),
    "fetch_url": ("net", False, False, "XXX----"),
}

EXPECTED_V1 = {
    "create_document", "create_object", "edit_object", "delete_object", "execute_code",
    "execute_code_async", "execute_code_headless", "get_async_status", "get_objects", "get_object",
    "get_view", "list_documents", "reload_document", "get_parts_list", "insert_part_from_library",
    "import_file", "run_fem_analysis", "apply_feature_plan", "feature_tree_status",
    "reapply_plan_feature", "download_file",
}


def _mask(phases) -> str:
    return "".join("X" if p in phases else "-" for p in tools.ALL_PHASES)


# --------------------------------------------------------------------------------------------
# Registry: metadata matrix, specs(), lite tools


class MatrixTest(unittest.TestCase):
    def test_matrix_has_the_43_documented_tools(self):
        self.assertEqual(len(MATRIX), 43)
        self.assertEqual(len(EXPECTED_V1 - set(MATRIX)), 0)

    def test_every_registered_tool_follows_the_matrix(self):
        for entry in tools.get_tools():
            self.assertIn(entry.name, MATRIX, f"{entry.name} is registered but not in the matrix")
            category, mutating, geometry, mask = MATRIX[entry.name]
            self.assertEqual(entry.category, category, entry.name)
            self.assertEqual(entry.mutating, mutating, f"{entry.name}: mutating")
            self.assertEqual(entry.geometry, geometry, f"{entry.name}: geometry")
            self.assertEqual(_mask(entry.phases), mask, f"{entry.name}: phase mask")

    def test_the_original_tools_are_all_registered_and_annotated(self):
        names = {t.name for t in tools.get_tools()}
        self.assertLessEqual(EXPECTED_V1, names)
        for name in EXPECTED_V1:
            self.assertIs(tools.tool_meta(name), tools._HANDLERS[name])

    def test_invariants_between_the_flags(self):
        for entry in tools.get_tools():
            self.assertIn(entry.category, tools.CATEGORIES, entry.name)
            self.assertTrue(set(entry.phases) <= set(tools.ALL_PHASES), entry.name)
            self.assertTrue(entry.phases, f"{entry.name}: a tool needs at least one phase")
            if entry.geometry:
                self.assertTrue(entry.mutating, f"{entry.name}: a geometry tool is always mutating")

    def test_mutating_names_and_predicates_agree_with_the_tools(self):
        expected = frozenset(t.name for t in tools.get_tools() if t.mutating)
        self.assertEqual(tools.mutating_names(), expected)
        self.assertIsInstance(tools.mutating_names(), frozenset)
        for entry in tools.get_tools():
            self.assertEqual(tools.is_mutating(entry.name), entry.mutating)
            self.assertEqual(tools.is_geometry(entry.name), entry.geometry)

    def test_unknown_names_are_not_mutating_and_have_no_meta(self):
        self.assertIsNone(tools.tool_meta("no_such_tool"))
        self.assertFalse(tools.is_mutating("no_such_tool"))
        self.assertFalse(tools.is_geometry("no_such_tool"))

    def test_execute_code_headless_is_mutating_but_not_geometry(self):
        self.assertTrue(tools.is_mutating("execute_code_headless"))
        self.assertFalse(tools.is_geometry("execute_code_headless"))
        self.assertTrue(tools.is_geometry("execute_code_async"))

    def test_core_and_lite_tool_sets(self):
        self.assertEqual(tools.CORE_TOOLS, {"plan_set", "plan_update", "record_note", "list_documents",
                                            "get_objects", "get_object", "get_view"})
        self.assertEqual(len(tools.LITE_TOOLS), 19)
        self.assertLessEqual(tools.CORE_TOOLS, tools.LITE_TOOLS)
        self.assertLessEqual({"create_document", "create_object", "edit_object", "execute_code", "import_file",
                              "find_reference_part", "source_reference_part", "build_enclosure",
                              "enclosure_spec", "set_parameter", "validate_model", "check_fit"},
                             tools.LITE_TOOLS)
        for name in tools.CORE_TOOLS & {t.name for t in tools.get_tools()}:
            self.assertEqual(_mask(tools.tool_meta(name).phases), "XXXXXXX", f"{name} is offered in every phase")

    def test_tool_defaults_follow_stage0(self):
        entry = tools.Tool("x", "d" * 50, {"type": "object"}, lambda: ("", None))
        self.assertEqual((entry.category, entry.mutating, entry.geometry), ("read", False, False))
        self.assertEqual(entry.phases, tools.ALL_PHASES)
        geometric = tools.Tool("y", "d" * 50, {"type": "object"}, lambda: ("", None), geometry=True)
        self.assertTrue(geometric.mutating)


class SpecsTest(unittest.TestCase):
    def test_no_argument_returns_every_tool_in_registration_order(self):
        every = [t.name for t in tools.get_tools()]
        self.assertEqual([s.name for s in tools.specs()], every)
        self.assertEqual([s.name for s in tools.specs()], [s.name for s in tools.specs()])

    def test_phase_filter_keeps_tools_whose_mask_intersects(self):
        intake = {s.name for s in tools.specs(phases=["INTAKE"])}
        self.assertIn("list_documents", intake)
        self.assertIn("create_document", intake)
        self.assertNotIn("create_object", intake)
        build = {s.name for s in tools.specs(phases=["BUILD"])}
        self.assertIn("create_object", build)
        self.assertNotIn("create_document", {s.name for s in tools.specs(phases=["DELIVER"])})
        self.assertIn("reload_document", {s.name for s in tools.specs(phases=["DELIVER"])})

    def test_phases_are_cumulative_when_given_together(self):
        early = {s.name for s in tools.specs(phases=["INTAKE", "SOURCE", "PLAN"])}
        everything = {s.name for s in tools.specs(phases=tools.ALL_PHASES)}
        self.assertLessEqual(early, everything)
        self.assertEqual(everything, {s.name for s in tools.specs()})
        both = {s.name for s in tools.specs(phases=["INTAKE", "BUILD"])}
        self.assertEqual(both, {s.name for s in tools.specs(phases=["INTAKE"])}
                         | {s.name for s in tools.specs(phases=["BUILD"])})

    def test_empty_or_missing_phases_never_leave_the_model_without_tools(self):
        total = len(tools.specs())
        self.assertEqual(len(tools.specs(phases=[])), total)
        self.assertEqual(len(tools.specs(phases=None)), total)

    def test_categories_names_and_exclude(self):
        self.assertTrue({s.name for s in tools.specs(categories=["fem"])} <= {
            n for n, m in MATRIX.items() if m[0] == "fem"})
        self.assertEqual([s.name for s in tools.specs(names=["get_view", "list_documents"])],
                         [n for n in (t.name for t in tools.get_tools()) if n in ("get_view", "list_documents")])
        self.assertNotIn("get_view", {s.name for s in tools.specs(exclude=["get_view"])})
        self.assertEqual(tools.specs(names=[]), [])

    def test_intent_category_table(self):
        build_categories = {"session", "read", "build", "params", "source", "verify", "deliver", "plan", "net"}
        for intent in ("enclosure", "fixture", "bracket", "generic_part", "modify", "deliver"):
            self.assertEqual(set(tools.INTENT_CATEGORIES[intent]), build_categories, intent)
        every = set(tools.CATEGORIES)
        self.assertEqual(set(tools.INTENT_CATEGORIES["fem"]), every)
        for intent in ("inspect", "question", "other"):
            self.assertEqual(set(tools.INTENT_CATEGORIES[intent]), every)
        for intent in ("enclosure", "fixture", "bracket"):
            self.assertEqual(len(tools.INTENT_HIDDEN[intent]), 11)
        self.assertNotIn("generic_part", tools.INTENT_HIDDEN)

    def test_enclosure_intent_hides_fem_and_the_kit_hidden_tools(self):
        names = {s.name for s in tools.specs(intent="enclosure")}
        for hidden in tools.INTENT_HIDDEN["enclosure"]:
            self.assertNotIn(hidden, names)
        for name in names:
            self.assertNotEqual(MATRIX[name][0], "fem", name)
        self.assertIn("create_object", names)
        self.assertIn("execute_code", names)

    def test_fem_intent(self):
        fem = {s.name for s in tools.specs(intent="fem")}
        self.assertIn("run_fem_analysis", fem)
        everything = {s.name for s in tools.specs(intent="question")}
        self.assertEqual(everything, {s.name for s in tools.specs()})

    def test_unknown_intent_applies_no_gating(self):
        self.assertEqual(len(tools.specs(intent="no-such-intent")), len(tools.specs()))
        self.assertEqual(len(tools.specs(intent="")), len(tools.specs()))
        self.assertEqual({s.name for s in tools.specs(intent=" Enclosure ")},
                         {s.name for s in tools.specs(intent="enclosure")})

    def test_intent_visibility_is_not_phase_dependent(self):
        # the system prompt is built from intent-only names: they must not change with the phase
        self.assertEqual({s.name for s in tools.specs(intent="enclosure")},
                         {s.name for s in tools.specs(intent="enclosure", phases=None)})

    def test_mutating_tools_offer_step_title_and_readers_do_not(self):
        by_name = {s.name: s for s in tools.specs()}
        for name, spec in by_name.items():
            has = "step_title" in spec.parameters["properties"]
            self.assertEqual(has, tools.is_mutating(name), name)
        self.assertEqual(by_name["create_object"].parameters["properties"]["step_title"],
                         tools.UNIVERSAL_ARGS["step_title"])
        self.assertEqual(tools.UNIVERSAL_ARGS["step_title"]["type"], "string")
        self.assertIn("3-8 word title", tools.UNIVERSAL_ARGS["step_title"]["description"])

    def test_specs_never_modify_the_registered_schemas(self):
        entry = tools.tool_meta("create_object")
        before = repr(entry.parameters)
        tools.specs(lite=True)
        tools.specs()
        self.assertEqual(repr(entry.parameters), before)
        self.assertNotIn("step_title", entry.parameters["properties"])

    def test_lite_specs(self):
        specs = tools.specs(lite=True)
        names = {s.name for s in specs}
        self.assertLessEqual(names, tools.LITE_TOOLS)
        self.assertIn("create_object", names)
        self.assertNotIn("run_fem_analysis", names)
        for spec in specs:
            self.assertLessEqual(len(spec.description), 200, spec.name)
            self.assertGreater(len(spec.description), 10, spec.name)
            for dropped in ("step_title", "view_name", "timeout", "include_screenshot"):
                self.assertNotIn(dropped, spec.parameters["properties"], f"{spec.name}.{dropped}")
            for key in spec.parameters.get("required", []):
                self.assertIn(key, spec.parameters["properties"], spec.name)
        full = tools.tool_meta("execute_code")
        lite = next(s for s in specs if s.name == "execute_code")
        self.assertIn("include_screenshot", full.parameters["properties"])
        self.assertIn("code", lite.parameters["properties"])
        self.assertLess(len(lite.description), len(full.description))

    def test_lite_description_is_the_first_sentence(self):
        entry = tools.Tool("zz_lite", "First sentence here. Second sentence follows. Third.", {"type": "object"},
                           lambda: ("", None))
        spec = entry.spec(lite=True)
        self.assertEqual(spec.description, "First sentence here.")
        self.assertEqual(entry.spec().description, entry.description)


class ModuleLayoutTest(unittest.TestCase):
    """``oscar.tools`` re-exports the registry API and ``run_tool`` from ``oscar.toolrun``."""

    NAMES = ("CORE_TOOLS", "LITE_TOOLS", "INTENT_CATEGORIES", "INTENT_HIDDEN", "get_tools", "tool_meta",
             "is_mutating", "is_geometry", "mutating_names", "specs", "derive_title", "run_tool", "clip_text",
             "_prepare_arguments", "_coerce", "UNIVERSAL_ARGS", "ALL_PHASES", "CATEGORIES", "ToolError",
             "reset_turn_circuits", "VIEW_NAMES", "_HANDLERS", "current_context", "use_context", "ToolContext",
             "ToolOutcome", "Tool", "tool")

    def test_public_names_live_on_oscar_tools(self):
        for name in self.NAMES:
            self.assertTrue(hasattr(tools, name), name)
        self.assertIs(tools.run_tool, toolrun.run_tool)
        self.assertIs(tools.specs, toolrun.specs)

    def test_category_lists_agree(self):
        self.assertEqual(set(tools.CATEGORIES), set(toolrun._ALL_CATEGORIES))

    def test_toolrun_can_be_imported_before_tools(self):
        import importlib

        saved = {k: v for k, v in sys.modules.items() if k == "oscar.toolrun" or k.startswith("oscar.tools")}
        package = sys.modules["oscar"]
        saved_attrs = {k.split(".")[1]: getattr(package, k.split(".")[1], None) for k in saved}
        try:
            for key in saved:
                del sys.modules[key]
                if hasattr(package, key.split(".")[1]):
                    delattr(package, key.split(".")[1])      # ``from . import x`` would find the old module
            fresh = importlib.import_module("oscar.toolrun")
            fresh_tools = sys.modules["oscar.tools"]
            self.assertIs(fresh_tools.run_tool, fresh.run_tool)
            self.assertIsNot(fresh_tools, saved["oscar.tools"])
            self.assertGreaterEqual(len(fresh_tools.get_tools()), 32)
        finally:
            for key in [k for k in sys.modules if k == "oscar.toolrun" or k.startswith("oscar.tools")]:
                del sys.modules[key]
            sys.modules.update(saved)
            for attr, value in saved_attrs.items():
                if value is not None:
                    setattr(package, attr, value)


class ToolModuleLoopTest(unittest.TestCase):
    def test_default_list_is_the_documented_one(self):
        self.assertEqual(tools._TOOL_MODULES,
                         ("tools_flow", "tools_project", "tools_ref", "tools_kit", "tools_verify", "tools_web"))

    def test_a_module_that_is_not_delivered_yet_is_tolerated(self):
        def fake(name):
            if name == "oscar.tools_ref":
                return object()
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)

        with mock.patch.object(tools.importlib, "import_module", side_effect=fake):
            self.assertEqual(tools._import_tool_modules(), ["tools_ref"])

    def test_a_missing_dependency_inside_a_module_is_raised(self):
        def fake(name):
            raise ModuleNotFoundError("No module named 'requests_toolbelt'", name="requests_toolbelt")

        with mock.patch.object(tools.importlib, "import_module", side_effect=fake):
            with self.assertRaises(ModuleNotFoundError):
                tools._import_tool_modules(["tools_kit"])

    def test_any_other_import_failure_propagates(self):
        with mock.patch.object(tools.importlib, "import_module", side_effect=SyntaxError("boom")):
            with self.assertRaises(SyntaxError):
                tools._import_tool_modules(["tools_flow"])

    def test_loop_imports_with_the_package_prefix(self):
        seen = []
        with mock.patch.object(tools.importlib, "import_module", side_effect=lambda n: seen.append(n)):
            tools._import_tool_modules(["tools_flow", "tools_verify"])
        self.assertEqual(seen, ["oscar.tools_flow", "oscar.tools_verify"])


# --------------------------------------------------------------------------------------------
# Titles, clipping, small pure helpers


class DeriveTitleTest(unittest.TestCase):
    CASES = [
        ("create_document", {"name": "PiCase"}, "Create document PiCase"),
        ("create_object", {"obj_type": "Part::Box", "obj_name": "Base"}, "Create Part::Box Base"),
        ("edit_object", {"obj_name": "Base"}, "Edit Base"),
        ("delete_object", {"obj_name": "Base"}, "Delete Base"),
        ("execute_code", {"code": "# Cut the USB-C opening\nx = 1"}, "Cut the USB-C opening"),
        ("execute_code_headless", {"code": "import math\ndef make_lid():\n    pass"}, "Run make_lid"),
        ("execute_code_async", {"code": '"""Build the shell.\nmore text"""\nx = 1'}, "Build the shell."),
        ("execute_code", {"code": "x = 1\ny = 2"}, "Run script (2 lines)"),
        ("execute_code", {"code": "# coding: utf-8\n# ---\nx = 1"}, "Run script (3 lines)"),
        ("import_file", {"path": "/tmp/parts/block.step"}, "Import block.step"),
        ("import_reference_part", {"id": "rpi4b"}, "Import reference rpi4b"),
        ("import_reference_part", {"path": "/tmp/pi.step"}, "Import reference pi.step"),
        ("source_reference_part", {"url": "https://example.com/pi.step"}, "Source reference example.com"),
        ("source_reference_part", {"id": "rpi4b"}, "Source reference rpi4b"),
        ("measure_reference_part", {"object": "Ref_Pi"}, "Measure reference Ref_Pi"),
        ("build_enclosure", {"stage": "shell"}, "Build enclosure: shell"),
        ("build_enclosure", {}, "Build enclosure: next"),
        ("add_cutout", {"w": 9.8, "h": 4.1, "part": "Shell", "face": "-y"}, "Cut 9.8 x 4.1 opening in Shell -y"),
        ("add_vents", {"count": 6, "part": "Lid"}, "Cut 6 vent slots in Lid"),
        ("set_parameters", {"params": [{"name": n} for n in ("a", "b", "c", "d", "e")]},
         "Set parameters: a, b, c, d"),
        ("set_parameter", {"name": "wall_t", "value": "2.4"}, "Set wall_t = 2.4"),
        ("bind_expression", {"object": "Shell", "property": "Length"}, "Bind Shell.Length"),
        ("apply_feature_plan", {}, "Build the feature plan"),
        ("reapply_plan_feature", {"feature_id": "f2"}, "Rebuild plan feature f2"),
        ("insert_part_from_library", {"relative_path": "a/b.FCStd"}, "Insert a/b.FCStd"),
        ("save_document", {"doc_name": "PiCase"}, "Save PiCase"),
        ("export_model", {"formats": ["step", "stl"]}, "Export step+stl"),
        ("list_documents", {}, "list_documents"),
        ("totally_unknown_tool", {"x": 1}, "totally_unknown_tool"),
    ]

    def test_title_table(self):
        for name, args, expected in self.CASES:
            with self.subTest(tool=name, args=args):
                self.assertEqual(tools.derive_title(name, args), expected)

    def test_missing_arguments_never_print_none(self):
        for name in MATRIX:
            title = tools.derive_title(name, {})
            self.assertTrue(title.strip(), name)
            self.assertNotIn("None", title, name)
            self.assertNotIn("  ", title, name)

    def test_bad_argument_shapes_fall_back_to_a_title(self):
        self.assertEqual(tools.derive_title("create_object", None), "Create")
        self.assertEqual(tools.derive_title("set_parameters", {"params": "oops"}), "Set parameters")
        self.assertNotIn("None", tools.derive_title("add_cutout", {"w": "wide"}))
        self.assertEqual(tools.derive_title("list_documents", None), "list_documents")

    def test_title_is_one_line_of_at_most_80_characters(self):
        long_title = tools.derive_title("create_object", {"obj_type": "Part::Box", "obj_name": "x" * 300})
        self.assertLessEqual(len(long_title), 80)
        multi = tools.derive_title("execute_code", {"code": "# first line\n# second"})
        self.assertNotIn("\n", multi)
        self.assertEqual(toolpipe.clean_title("a\n  b\t c"), "a b c")
        self.assertEqual(toolpipe.clean_title(None), "")

    def test_registered_title_callable_wins_over_the_table(self):
        entry = tools.Tool("create_object", "d" * 50, {"type": "object"}, lambda: ("", None),
                           title=lambda a: f"Custom {a.get('obj_name')}")
        self.assertEqual(toolpipe.derive_title("create_object", {"obj_name": "B"}, entry), "Custom B")
        broken = tools.Tool("create_object", "d" * 50, {"type": "object"}, lambda: ("", None),
                            title=lambda a: 1 / 0)
        self.assertEqual(toolpipe.derive_title("create_object", {"obj_type": "T", "obj_name": "B"}, broken),
                         "Create T B")
        empty = tools.Tool("edit_object", "d" * 50, {"type": "object"}, lambda: ("", None), title=lambda a: "  ")
        self.assertEqual(toolpipe.derive_title("edit_object", {"obj_name": "B"}, empty), "Edit B")


class ClipTextTest(unittest.TestCase):
    def test_short_text_is_unchanged(self):
        self.assertEqual(tools.clip_text("abc"), "abc")
        self.assertEqual(tools.clip_text(""), "")
        self.assertEqual(tools.clip_text(None), "")
        exact = "x" * 12000
        self.assertEqual(tools.clip_text(exact), exact)

    def test_long_text_keeps_head_marker_and_tail(self):
        text = "H" * 8000 + "m" * 10000 + "T" * 3500
        clipped = tools.clip_text(text)
        omitted = len(text) - 8000 - 3500
        marker = f"\n...({omitted} chars omitted; call get_objects / get_object for details)...\n"
        self.assertEqual(clipped, "H" * 8000 + marker + "T" * 3500)

    def test_boundary_one_over_the_limit_is_clipped(self):
        clipped = tools.clip_text("a" * 12001)
        self.assertIn("chars omitted", clipped)
        self.assertLess(len(clipped), 12001)

    def test_custom_limit_and_degenerate_limits(self):
        clipped = tools.clip_text("0123456789" * 100, 120)
        self.assertTrue(clipped.startswith("0123456789" * 8))
        self.assertTrue(clipped.endswith("0123456789" * 3))
        self.assertIn("(885 chars omitted;", clipped)
        self.assertIn("chars omitted", tools.clip_text("abcdef", 0))   # a limit below 1 is raised to 1
        self.assertEqual(tools.clip_text(12345), "12345")

    def test_non_string_input_is_converted(self):
        self.assertEqual(tools.clip_text(["a"]), "['a']")


class SmallPureHelpersTest(unittest.TestCase):
    def test_first_sentence(self):
        self.assertEqual(toolpipe.first_sentence("One. Two."), "One.")
        self.assertEqual(toolpipe.first_sentence("Use e.g. a box. Then more."), "Use e.g. a box.")
        self.assertEqual(toolpipe.first_sentence("Pi is 3.14 exactly. Next."), "Pi is 3.14 exactly.")
        self.assertEqual(toolpipe.first_sentence("No terminator here"), "No terminator here")
        self.assertEqual(toolpipe.first_sentence(""), "")
        long = toolpipe.first_sentence("x" * 500, 200)
        self.assertEqual(len(long), 200)
        self.assertTrue(long.endswith("…"))

    def test_similar_options(self):
        views = ["Isometric", "Front", "Top", "Right"]
        self.assertEqual(toolpipe.similar_options("iso", views), ["Isometric"])
        self.assertEqual(toolpipe.similar_options("frnt", views), ["Front"])
        self.assertEqual(toolpipe.similar_options("zzzzzz", views), [])

    def test_dispatch_timeout_detection(self):
        text = "'execute_code' did not complete within 90s (the GUI thread stayed busy)"
        self.assertTrue(toolpipe.is_dispatch_timeout(text, False))
        self.assertTrue(toolpipe.is_dispatch_timeout("Script STILL RUNNING on the GUI thread", False))
        self.assertFalse(toolpipe.is_dispatch_timeout(text, True), "a successful handler is never a timeout")
        self.assertFalse(toolpipe.is_dispatch_timeout("Failed to execute code: NameError", False))

    def test_resolve_doc_name(self):
        docs = ["PiCase", "Gear_System", "gearbox"]
        self.assertEqual(toolpipe.resolve_doc_name("PiCase", docs), "PiCase")
        self.assertEqual(toolpipe.resolve_doc_name("picase", docs), "PiCase")
        self.assertEqual(toolpipe.resolve_doc_name("Gear_S", docs), "Gear_System")
        self.assertEqual(toolpipe.resolve_doc_name("g", docs), "", "an ambiguous prefix resolves to nothing")
        self.assertEqual(toolpipe.resolve_doc_name("", docs), "")
        self.assertEqual(toolpipe.resolve_doc_name(None, docs), "")

    def test_delta_between_ignores_temp_objects_and_pure_visibility(self):
        old = StateSnapshot("D", {"A": {"key": 1, "volume": 5.0, "visible": True},
                                  "_oscar_tmp1": {"key": 1}})
        new = StateSnapshot("D", {"A": {"key": 1, "volume": 5.0, "visible": False},
                                  "B": {"key": 2, "volume": 3.0}, "_oscar_tmp2": {"key": 9}})
        delta = toolpipe.delta_between(old, new)
        self.assertEqual(delta["added"], ["B"])
        self.assertEqual(delta["removed"], [])
        self.assertEqual(delta["changed"], [])
        changed = toolpipe.delta_between(old, StateSnapshot("D", {"A": {"key": 2, "volume": 7.0}}))
        self.assertEqual(changed["changed"][0]["name"], "A")
        self.assertEqual(toolpipe.delta_between(None, new)["added"], ["A", "B"])
        self.assertEqual(toolpipe.delta_size(delta), 1)
        self.assertEqual(toolpipe.delta_names({"added": ["B"], "changed": [{"name": "A"}]}), ["B", "A"])
        self.assertEqual(toolpipe.delta_line({"added": ["B"], "removed": ["C"], "changed": [{
            "name": "A", "volume_before": 1.0, "volume_after": 2.0}]}), "+B -C ~A(vol 1.0->2.0)")

    @staticmethod
    def _solid(shape_hash=1, volume=10.0, bbox=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), placement=(0.0,) * 7, faces=6, expr=""):
        """A snapshot entry of a geometry object: key ("g", hash, placement, bbox, faces, edges)."""
        return {"key": ("g", shape_hash, placement, bbox, faces, 12), "volume": volume, "bbox": list(bbox),
                "placement": placement, "expr": expr}

    def test_delta_between_regenerated_shapes_count_unless_asked_to_ignore_them(self):
        old = StateSnapshot("D", {"Shell": self._solid(shape_hash=1)})
        new = StateSnapshot("D", {"Shell": self._solid(shape_hash=2)})      # the same solid, regenerated
        self.assertEqual([c["name"] for c in toolpipe.delta_between(old, new)["changed"]], ["Shell"])
        ignored = toolpipe.delta_between(old, new, ignore_regenerated=True)
        self.assertEqual(ignored, {"added": [], "removed": [], "changed": []})

    def test_a_real_change_is_never_ignored_as_regenerated(self):
        old = StateSnapshot("D", {"Shell": self._solid()})
        for label, entry in (("volume", self._solid(shape_hash=2, volume=9.5)),
                             ("bbox", self._solid(shape_hash=2, bbox=(0.0, 0.0, 0.0, 1.0, 1.0, 2.0))),
                             ("placement", self._solid(shape_hash=2, placement=(1.0, 0, 0, 0, 0, 0, 1.0))),
                             ("face count", self._solid(shape_hash=2, faces=7)),
                             ("expression", self._solid(shape_hash=2, expr="Params.wall_t"))):
            with self.subTest(label):
                new = StateSnapshot("D", {"Shell": entry})
                delta = toolpipe.delta_between(old, new, ignore_regenerated=True)
                self.assertEqual([c["name"] for c in delta["changed"]], ["Shell"])

    def test_only_geometry_keys_can_be_regenerated(self):
        for old_key, new_key in ((1, 2), (("p", "a"), ("p", "b")), (("g", 1), ("g", 2)),
                                 (("m", 1, 2, None, None), ("m", 1, 2, None, None, 9)),
                                 (("g", 1, None, None, 6, 12), ("m", 2, None, None, 6, 12))):
            with self.subTest(old_key=old_key):
                old = StateSnapshot("D", {"A": {"key": old_key, "volume": 1.0}})
                new = StateSnapshot("D", {"A": {"key": new_key, "volume": 1.0}})
                self.assertEqual(len(toolpipe.delta_between(old, new, ignore_regenerated=True)["changed"]), 1)

    def test_summarize_report(self):
        report = HealthReport(doc="D", ts=0.0, findings=[
            Finding("warning", "W1", "A", "careful"), Finding("error", "E1", "B", "broken")],
            counts={"objects": 2, "solids": 1, "errors": 1, "warnings": 1})
        summary = toolpipe.summarize_report(report)
        self.assertEqual((summary["valid"], summary["errors"], summary["warnings"]), (False, 1, 1))
        self.assertEqual(summary["findings"][0]["code"], "E1", "errors come first")
        self.assertIn("[health] doc 'D'", summary["line"])
        many = HealthReport(doc="D", ts=0.0, findings=[Finding("error", f"E{i}", "O", "m") for i in range(9)])
        self.assertEqual(len(toolpipe.summarize_report(many)["findings"]), 5)


class BuildFailureTextTest(unittest.TestCase):
    """The old "Plan failed at validation: None" bug: the reason lives in ``errors``."""

    def test_errors_list_is_joined_and_never_none(self):
        text = toolpipe._fallback_failure_text(
            {"success": False, "stage": "validation", "errors": ["width missing", "height < 0"]}, "")
        self.assertEqual(text, "Plan failed at 'validation': width missing; height < 0")
        self.assertNotIn("None", text)

    def test_feature_id_error_and_completed_features(self):
        text = toolpipe._fallback_failure_text(
            {"success": False, "feature_id": "f3", "error": "sketch is open", "completed": ["f1", "f2"]}, "")
        self.assertEqual(text, "Plan failed at 'f3': sketch is open\nCompleted features: ['f1', 'f2']")

    def test_no_reason_and_no_place(self):
        self.assertEqual(toolpipe._fallback_failure_text({"success": False}, ""),
                         "Plan failed at 'plan': unknown error")
        self.assertEqual(toolpipe._fallback_failure_text("bad", ""), "Plan failed: bad")

    def test_public_function_uses_the_caller_feature_id_and_never_raises(self):
        text = toolpipe.build_failure_text({"success": False, "errors": ["boom"]}, "f9")
        self.assertIn("'f9'", text)
        self.assertIn("boom", text)
        self.assertNotIn("None", text)
        self.assertTrue(toolpipe.build_failure_text(None))

    def test_builder_formatter_is_preferred_when_it_exists(self):
        from oscar.plan import builder

        with mock.patch.object(builder, "format_build_failure", create=True, return_value="FROM BUILDER"):
            self.assertEqual(toolpipe.build_failure_text({"success": False}), "FROM BUILDER")
        with mock.patch.object(builder, "format_build_failure", create=True, side_effect=RuntimeError("x")):
            self.assertIn("Plan failed", toolpipe.build_failure_text({"success": False, "errors": ["e"]}))


# --------------------------------------------------------------------------------------------
# Argument preparation


class PrepareArgumentsTest(unittest.TestCase):
    def prepare(self, name, args, state=None):
        ctx = ToolContext(state=state if state is not None else {})
        with use_context(ctx):
            return tools._prepare_arguments(tools.tool_meta(name), args), ctx

    def error(self, name, args):
        with self.assertRaises(tools.ToolError) as caught:
            self.prepare(name, args)
        return str(caught.exception)

    def test_enum_miss_is_reported_in_the_fix_it_style_with_a_suggestion(self):
        # The view enum is cosmetic and never errors (see test_view_name_is_resolved_not_rejected in
        # test_fix_runtime), so the fix-it format is pinned on another enum (record_note.kind).
        self.assertEqual(
            self.error("record_note", {"kind": "assumptions", "text": "x"}),
            "Bad arguments for record_note: 'kind' must be one of "
            "plan|assumption|provenance|validation|unresolved|decision, got 'assumptions' (similar: assumption).")

    def test_unknown_view_name_is_resolved_to_isometric_not_rejected(self):
        for spelling, expected in (("iso", "Isometric"), ("Rear view", "Back"), ("zzz", "Isometric")):
            prepared, _ = self.prepare("get_view", {"view_name": spelling})
            self.assertEqual(prepared["view_name"], expected, spelling)

    def test_enum_case_only_miss_is_repaired(self):
        prepared, _ = self.prepare("get_view", {"view_name": "front"})
        self.assertEqual(prepared["view_name"], "Front")

    def test_step_title_is_removed_before_binding_and_kept_on_the_context(self):
        prepared, ctx = self.prepare("create_object", {
            "doc_name": "D", "obj_type": "Part::Box", "obj_name": "B", "step_title": "  Make\nthe base  "})
        self.assertNotIn("step_title", prepared)
        self.assertEqual(ctx.step_title, "Make the base")

    def test_unknown_and_missing_parameters(self):
        message = self.error("get_view", {"foo": 1})
        self.assertIn("unknown parameter(s) 'foo'", message)
        self.assertIn("Accepted parameters:", message)
        message = self.error("create_object", {})
        self.assertIn("missing required parameter(s) 'doc_name', 'obj_type', 'obj_name'", message)
        self.assertIn("obj_properties", self.error("edit_object", {"doc_name": "D", "obj_name": "B", "nope": 1}))
        self.assertIn("expected a JSON object", self.error("get_view", ["a"]))

    def test_booleans_accept_words(self):
        for word, expected in (("false", False), ("no", False), ("0", False), ("off", False),
                               ("true", True), ("Yes", True), ("1", True), ("ON", True), (0, False), (1, True)):
            prepared, _ = self.prepare("get_objects", {"doc_name": "D", "include_screenshot": word})
            self.assertIs(prepared["include_screenshot"], expected, word)
        self.assertIn("must be true or false", self.error(
            "get_objects", {"doc_name": "D", "include_screenshot": "maybe"}))

    def test_json_strings_become_objects(self):
        prepared, _ = self.prepare("edit_object", {"doc_name": "D", "obj_name": "B",
                                                    "obj_properties": '{"Height": 5}'})
        self.assertEqual(prepared["obj_properties"], {"Height": 5})
        self.assertIn("must be a JSON object", self.error(
            "edit_object", {"doc_name": "D", "obj_name": "B", "obj_properties": "[1, 2]"}))

    def test_numbers_integers_bounds(self):
        prepared, _ = self.prepare("get_view", {"width": "640", "height": 480.0})
        self.assertEqual((prepared["width"], prepared["height"]), (640, 480))
        self.assertIn("must be an integer, got 'wide'", self.error("get_view", {"width": "wide"}))
        self.assertIn("must be an integer", self.error("get_view", {"width": 1.5}))
        self.assertIn("'limit' must be between 1 and 500, got 0", self.error("get_objects", {"doc_name": "D", "limit": 0}))
        self.assertIn("'limit' must be between 1 and 500, got 501", self.error("get_objects", {"doc_name": "D", "limit": 501}))
        prepared, _ = self.prepare("get_objects", {"doc_name": "D", "limit": 500, "offset": 0})
        self.assertEqual((prepared["limit"], prepared["offset"]), (500, 0))
        self.assertIn("must be a number", self.error("execute_code", {"code": "x", "timeout": float("nan")}))
        self.assertIn("must be a number", self.error("execute_code", {"code": "x", "timeout": True}))

    def test_all_problems_come_back_in_one_error(self):
        # (an enum problem on a non-view enum plus a type problem; the cosmetic view_name never errors, see above)
        message = self.error("add_cutout", {"doc_name": "D", "part": "shell", "face": "zzz", "u": 1, "v": 1, "w": 1,
                                            "h": "many"})
        self.assertIn("'face'", message)
        self.assertIn("'h'", message)
        self.assertEqual(message.count("Bad arguments for add_cutout"), 1)

    def test_none_passes_for_optional_arguments(self):
        # Reconciled with the merged review fix: an optional argument sent as null is dropped from the prepared call
        # (not an error), so the handler's own default applies instead of being overridden by None.
        prepared, _ = self.prepare("get_view", {"width": None, "view_name": "Top"})
        self.assertNotIn("width", prepared)
        self.assertEqual(prepared["view_name"], "Top")

    def test_numbers_are_accepted_for_string_properties(self):
        prepared, _ = self.prepare("create_document", {"name": 42})
        self.assertEqual(prepared["name"], "42")

    def test_lite_turn_forces_include_screenshot_off(self):
        prepared, _ = self.prepare("create_object", {"doc_name": "D", "obj_type": "Part::Box", "obj_name": "B",
                                                     "include_screenshot": True}, state={"lite": True})
        self.assertIs(prepared["include_screenshot"], False)
        prepared, _ = self.prepare("create_object", {"doc_name": "D", "obj_type": "Part::Box", "obj_name": "B"},
                                   state={"lite": True})
        self.assertIs(prepared["include_screenshot"], False, "also when the model omitted it")
        prepared, _ = self.prepare("create_object", {"doc_name": "D", "obj_type": "Part::Box", "obj_name": "B"})
        self.assertNotIn("include_screenshot", prepared, "a normal turn keeps the handler default")
        prepared, _ = self.prepare("list_documents", {}, state={"lite": True})
        self.assertEqual(prepared, {}, "tools without the switch are untouched")

    def test_array_schema_with_item_types(self):
        schema = {"type": "object", "properties": {"point": {"type": "array", "items": {"type": "number"}}},
                  "required": ["point"]}
        entry = tools.Tool("zz_arr", "d" * 50, schema, lambda point: ("ok", None))
        with use_context(ToolContext()):
            self.assertEqual(tools._prepare_arguments(entry, {"point": "[1, 2.5]"})["point"], [1, 2.5])
            self.assertEqual(tools._prepare_arguments(entry, {"point": ["3", 4]})["point"], [3, 4])
            with self.assertRaises(tools.ToolError) as caught:
                tools._prepare_arguments(entry, {"point": [1, "a"]})
            self.assertIn("'point[1]' must be a number, got 'a'", str(caught.exception))
            with self.assertRaises(tools.ToolError) as caught:
                tools._prepare_arguments(entry, {"point": 5})
            self.assertIn("must be a JSON array", str(caught.exception))

    def test_handler_signature_mismatch_is_a_tool_error(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
        entry = tools.Tool("zz_sig", "d" * 50, schema, lambda a: ("ok", None))
        with use_context(ToolContext()):
            with self.assertRaises(tools.ToolError) as caught:
                tools._prepare_arguments(entry, {"a": "x", "b": "y"})
        self.assertIn("Bad arguments for zz_sig", str(caught.exception))


# --------------------------------------------------------------------------------------------
# run_tool pipeline against fakes


class FakeHealth:
    """Stands in for ``oscar.cad.health``. ``snapshots`` are returned one per
    ``snapshot_all_via_dispatch`` call (before, then after); ``reports`` per doc."""

    def __init__(self, calls, snapshots=None, reports=None, fail_report=False):
        self.calls = calls
        self.snapshots = list(snapshots or [])
        self.reports = dict(reports or {})
        self.fail_report = fail_report
        self.prevs = []

    def snapshot_all_via_dispatch(self, *args, **kwargs):
        self.calls.append("snapshot_all")
        return self.snapshots.pop(0) if self.snapshots else {}

    def health_via_dispatch(self, doc_name, prev=None, **kwargs):
        self.calls.append("health")
        self.prevs.append((doc_name, prev))
        if self.fail_report:
            raise RuntimeError("health exploded")
        return self.reports.get(doc_name or "D")


class FakeProject:
    def __init__(self, calls, raises=False):
        self.calls = calls
        self.raises = raises
        self.records = []
        self.checkpoints = []

    def ensure_turn_checkpoint(self, doc):
        self.calls.append("checkpoint")
        self.checkpoints.append(doc)
        return "T1"

    def on_tool_result(self, **kwargs):
        self.calls.append("project")
        if self.raises:
            raise RuntimeError("project exploded")
        self.records.append(kwargs)
        return SimpleNamespace(index=7, snapshot="snapshots/007_x.png", snapshot_note="", checkpoint="007",
                               script="scripts/007_x.py")

    def path(self, *parts):
        return "/abs/" + "/".join(parts)


class FakeLedger:
    def __init__(self, calls, raises=False):
        self.calls = calls
        self.raises = raises
        self.outcomes = []

    def on_tool_result(self, outcome):
        self.calls.append("ledger")
        if self.raises:
            raise RuntimeError("ledger exploded")
        self.outcomes.append(outcome)


def snap(doc, **objects):
    return StateSnapshot(doc, {n: dict(v) for n, v in objects.items()})


def obj(key=1, volume=10.0):
    return {"key": key, "volume": volume, "bbox": None, "placement": None, "expr": ""}


def report(doc="D", findings=(), added=(), counts=None):
    return HealthReport(doc=doc, ts=0.0, findings=list(findings), delta=ObjectDelta(added=list(added)),
                        counts=counts or {"objects": 3, "solids": 2, "errors": 0, "warnings": 0})


class PipelineBase(unittest.TestCase):
    """Registers throw-away tools and a fake health module; removes them afterwards."""

    def setUp(self):
        self.calls = []
        self.seen = []
        self._saved = dict(tools._HANDLERS)
        self.addCleanup(self._restore)
        self.log = mock.patch.object(tools._turnlog, "log_event").start()
        self.addCleanup(mock.patch.stopall)

        def geo(doc_name="D", size=1, include_screenshot=True):
            self.calls.append("handler")
            self.seen.append({"doc_name": doc_name, "size": size, "include_screenshot": include_screenshot})
            return f"geo ok size={size}", None

        schema = {"type": "object", "properties": {
            "doc_name": {"type": "string"}, "size": {"type": "integer"},
            "include_screenshot": {"type": "boolean"}}}
        tools.tool("zz_geo", "A geometry test tool for the pipeline tests.", schema, category="build",
                   geometry=True, phases=("BUILD",))(geo)
        tools.tool("zz_mut", "A mutating, non-geometry test tool.", {"type": "object", "properties": {}},
                   category="deliver", mutating=True)(lambda: ("mut ok", None))
        tools.tool("zz_read", "A read-only test tool for the pipeline.", {"type": "object", "properties": {}},
                   category="read")(lambda: ("read ok", "IMAGEDATA"))
        tools.tool("zz_fail", "A geometry tool that raises ToolError.",
                   {"type": "object", "properties": {"doc_name": {"type": "string"}}},
                   category="build", geometry=True)(self._fail)
        tools.tool("zz_boom", "A geometry tool that crashes with a plain exception.",
                   {"type": "object", "properties": {}}, category="build", geometry=True)(self._boom)
        tools.tool("zz_big", "A geometry tool with a huge result.",
                   {"type": "object", "properties": {"doc_name": {"type": "string"}}},
                   category="build", geometry=True)(lambda doc_name="D": ("S" * 30000 + "THE END", None))

    @staticmethod
    def _fail(doc_name="D"):
        raise tools.ToolError("it did not work: bad input; use another one")

    @staticmethod
    def _boom():
        raise RuntimeError("kaboom")

    def _restore(self):
        tools._HANDLERS.clear()
        tools._HANDLERS.update(self._saved)

    def use(self, health=None, project=None, ledger=None, **ctx_kwargs):
        ctx = ToolContext(project=project, ledger=ledger, turn_id="T1", round=2, call_id="c1", phase="BUILD",
                          state=ctx_kwargs.pop("state", {}), **ctx_kwargs)
        if health is not None:
            mock.patch.object(toolrun, "_health_api", return_value=health).start()
        return ctx

    def call(self, ctx, name, args=None):
        with use_context(ctx):
            return tools.run_tool(name, args if args is not None else {})

    def events(self, kind):
        return [c for c in self.log.call_args_list if c.args and c.args[0] == kind]


class HookOrderTest(PipelineBase):
    def test_after_the_handler_health_then_project_then_ledger(self):
        health = FakeHealth(self.calls, snapshots=[{"D": snap("D")}, {"D": snap("D", Box=obj())}],
                            reports={"D": report(added=["Box"])})
        project, ledger = FakeProject(self.calls), FakeLedger(self.calls)
        ctx = self.use(health, project, ledger)
        text, image, is_error = self.call(ctx, "zz_geo", {"doc_name": "D"})
        self.assertFalse(is_error)
        self.assertEqual(self.calls, ["snapshot_all", "checkpoint", "handler", "snapshot_all", "health",
                                      "project", "ledger"])
        self.assertEqual(project.checkpoints, ["D"])

    def test_the_outcome_carries_everything_the_hooks_learned(self):
        health = FakeHealth(self.calls, snapshots=[{"D": snap("D")}, {"D": snap("D", Box=obj())}],
                            reports={"D": report(added=["Box"])})
        project, ledger = FakeProject(self.calls), FakeLedger(self.calls)
        ctx = self.use(health, project, ledger)
        text, _, _ = self.call(ctx, "zz_geo", {"doc_name": "D", "size": 3, "step_title": "Make a box"})
        outcome = ctx.outcome
        self.assertIs(ledger.outcomes[0], outcome)
        self.assertEqual((outcome.name, outcome.call_id, outcome.title), ("zz_geo", "c1", "Make a box"))
        self.assertEqual((outcome.ok, outcome.handler_ok, outcome.health_failed), (True, True, False))
        self.assertEqual((outcome.mutating, outcome.geometry, outcome.doc), (True, True, "D"))
        self.assertEqual(outcome.delta["added"], ["Box"])
        self.assertEqual(outcome.health["valid"], True)
        self.assertEqual(outcome.text, text)
        self.assertEqual(outcome.args, {"doc_name": "D", "size": 3})
        self.assertEqual((outcome.step_index, outcome.checkpoint), (7, "007"))
        self.assertEqual(outcome.snapshot_path, "/abs/snapshots/007_x.png")
        self.assertEqual(outcome.script_path, "/abs/scripts/007_x.py")
        self.assertGreaterEqual(outcome.dur_s, 0.0)
        record = project.records[0]
        self.assertEqual((record["turn_id"], record["round"], record["call_id"]), ("T1", 2, "c1"))
        self.assertEqual((record["tool"], record["title"], record["ok"]), ("zz_geo", "Make a box", True))
        self.assertTrue(record["mutating"] and record["geometry"])
        self.assertEqual(record["doc"], "D")
        self.assertEqual(record["delta"]["added"], ["Box"])
        self.assertEqual(record["args"], {"doc_name": "D", "size": 3})

    def test_a_non_geometry_mutating_tool_records_a_step_but_runs_no_health(self):
        health = FakeHealth(self.calls)
        project, ledger = FakeProject(self.calls), FakeLedger(self.calls)
        ctx = self.use(health, project, ledger)
        text, _, is_error = self.call(ctx, "zz_mut")
        self.assertEqual((text, is_error), ("mut ok", False))
        self.assertEqual(self.calls, ["project", "ledger"])
        self.assertFalse(project.records[0]["geometry"])

    def test_read_tools_and_unbound_contexts_run_no_hooks_at_all(self):
        health = FakeHealth(self.calls)
        project, ledger = FakeProject(self.calls), FakeLedger(self.calls)
        ctx = self.use(health, project, ledger)
        text, image, is_error = self.call(ctx, "zz_read")
        self.assertEqual((text, image, is_error), ("read ok", "IMAGEDATA", False))
        self.assertEqual(self.calls, ["ledger"], "the ledger still hears about reads (evidence), nothing else")
        self.calls.clear()
        text, _, _ = tools.run_tool("zz_geo", {})           # no context bound
        self.assertEqual(text, "geo ok size=1")
        self.assertEqual(self.calls, ["handler"])

    def test_unknown_tool_runs_no_hooks_and_fills_an_outcome(self):
        project, ledger = FakeProject(self.calls), FakeLedger(self.calls)
        ctx = self.use(FakeHealth(self.calls), project, ledger)
        text, image, is_error = self.call(ctx, "nope", {"a": 1})
        self.assertTrue(is_error)
        self.assertIn("Unknown tool 'nope'. Available tools:", text)
        self.assertIn("zz_geo", text)
        self.assertEqual(self.calls, [])
        self.assertFalse(ctx.outcome.ok)
        self.assertEqual(ctx.outcome.name, "nope")

    def test_ledger_only_and_project_only_contexts(self):
        health = FakeHealth(self.calls, snapshots=[{"D": snap("D")}, {"D": snap("D")}],
                            reports={"D": report()})
        ledger = FakeLedger(self.calls)
        ctx = self.use(health, None, ledger)
        self.call(ctx, "zz_geo", {"doc_name": "D"})
        self.assertEqual(self.calls, ["snapshot_all", "handler", "snapshot_all", "health", "ledger"])
        self.calls.clear()
        health.snapshots = [{"D": snap("D")}, {"D": snap("D")}]
        project = FakeProject(self.calls)
        self.call(self.use(health, project, None), "zz_geo", {"doc_name": "D"})
        self.assertEqual(self.calls, ["snapshot_all", "checkpoint", "handler", "snapshot_all", "health", "project"])

    def test_a_failing_project_hook_never_changes_the_result_nor_stops_the_ledger(self):
        health = FakeHealth(self.calls, snapshots=[{"D": snap("D")}, {"D": snap("D", B=obj())}],
                            reports={"D": report(added=["B"])})
        project, ledger = FakeProject(self.calls, raises=True), FakeLedger(self.calls)
        text, _, is_error = self.call(self.use(health, project, ledger), "zz_geo", {"doc_name": "D"})
        self.assertFalse(is_error)
        self.assertTrue(text.startswith("geo ok"))
        self.assertIn("ledger", self.calls)
        self.assertEqual(len(self.events("project_error")), 1)

    def test_a_failing_ledger_hook_never_changes_the_result(self):
        health = FakeHealth(self.calls, snapshots=[{"D": snap("D")}, {"D": snap("D")}], reports={"D": report()})
        ledger = FakeLedger(self.calls, raises=True)
        text, _, is_error = self.call(self.use(health, None, ledger), "zz_geo", {"doc_name": "D"})
        self.assertFalse(is_error)
        self.assertEqual(len(self.events("ledger_error")), 1)

    def test_a_failing_health_hook_is_logged_and_the_call_goes_on(self):
        health = FakeHealth(self.calls, snapshots=[{"D": snap("D")}, {"D": snap("D")}], fail_report=True)
        project, ledger = FakeProject(self.calls), FakeLedger(self.calls)
        text, _, is_error = self.call(self.use(health, project, ledger), "zz_geo", {"doc_name": "D"})
        self.assertEqual((text, is_error), ("geo ok size=1", False))
        self.assertEqual(len(self.events("health_error")), 1)
        self.assertEqual(self.calls[-2:], ["project", "ledger"])

    def test_health_module_absent_means_no_health_but_everything_else(self):
        with mock.patch.object(toolrun, "_health_api", return_value=None):
            project, ledger = FakeProject(self.calls), FakeLedger(self.calls)
            ctx = self.use(None, project, ledger)
            text, _, is_error = self.call(ctx, "zz_geo", {"doc_name": "D"})
        self.assertEqual((text, is_error), ("geo ok size=1", False))
        self.assertEqual(self.calls, ["checkpoint", "handler", "project", "ledger"])
        self.assertIsNone(ctx.outcome.health)


class HealthSuffixTest(PipelineBase):
    def geo_run(self, rep, before=None, after=None, project=None, ledger=None, name="zz_geo", state=None,
                args=None):
        before = before if before is not None else {"D": snap("D")}
        after = after if after is not None else {"D": snap("D", Box=obj())}
        health = FakeHealth(self.calls, snapshots=[before, after], reports={"D": rep})
        if project is None and ledger is None:
            project = FakeProject(self.calls)         # hooks only run when a Project or a Ledger is bound
        ctx = self.use(health, project, ledger, state=state if state is not None else {})
        return ctx, self.call(ctx, name, {"doc_name": "D"} if args is None else args)

    def test_success_appends_the_health_line(self):
        rep = report(added=["Box"], counts={"objects": 7, "solids": 3, "errors": 0, "warnings": 0})
        ctx, (text, _, is_error) = self.geo_run(rep)
        self.assertFalse(is_error)
        self.assertEqual(text, "geo ok size=1\n\n[health] doc 'D': 7 objects, 3 solids, VALID · Δ +Box")

    def test_new_error_flips_the_model_verdict_but_not_the_handler_verdict(self):
        rep = report(findings=[Finding("error", "INVALID_SHAPE", "Cut", "isValid() is false; recompute")],
                     added=["Cut"], counts={"objects": 3, "solids": 1, "errors": 1, "warnings": 0})
        project, ledger = FakeProject(self.calls), FakeLedger(self.calls)
        ctx, (text, _, is_error) = self.geo_run(rep, project=project, ledger=ledger)
        self.assertTrue(is_error)
        self.assertIn("\n\nNOT OK: the call ran but the model is in an error state now. Fix it before building on it.\n", text)
        self.assertIn("[health] doc 'D'", text)
        self.assertIn("[INVALID_SHAPE] Cut: isValid() is false; recompute", text)
        outcome = ctx.outcome
        self.assertEqual((outcome.ok, outcome.handler_ok, outcome.health_failed), (False, True, True))
        self.assertEqual(outcome.new_errors, ["isValid() is false; recompute"])
        self.assertTrue(project.records[0]["ok"], "the Project records the step as a handler success")
        self.assertIs(ledger.outcomes[0].handler_ok, True)
        self.assertIs(ledger.outcomes[0].ok, False)

    def test_persisting_errors_are_listed_as_still_failing_and_do_not_flip_by_themselves(self):
        old = Finding("error", "OLD_ERR", "X", "was already broken")
        before = {"D": StateSnapshot("D", {}, finding_keys=[("OLD_ERR", "X")])}
        rep = report(findings=[old], added=["Box"], counts={"objects": 3, "solids": 1, "errors": 1, "warnings": 0})
        ctx, (text, _, is_error) = self.geo_run(rep, before=before)
        self.assertFalse(is_error, "an error that existed before the call is not this call's fault")
        self.assertIn("OPEN PROBLEMS:\n  OLD_ERR X: was already broken", text)
        new = Finding("error", "NEW_ERR", "Y", "freshly broken")
        rep = report(findings=[old, new], added=["Box"], counts={"objects": 3, "solids": 1, "errors": 2, "warnings": 0})
        ctx, (text, _, is_error) = self.geo_run(rep, before=before)
        self.assertTrue(is_error)
        self.assertIn("STILL FAILING (not new):\n  OLD_ERR X: was already broken", text)

    def test_open_problems_block_has_at_most_three_lines_and_includes_warnings(self):
        findings = [Finding("warning", f"W{i}", f"O{i}", f"warn {i}") for i in range(5)]
        before = {"D": StateSnapshot("D", {}, finding_keys=[(f.code, f.object) for f in findings])}
        rep = report(findings=findings, added=["Box"])
        ctx, (text, _, is_error) = self.geo_run(rep, before=before)
        self.assertFalse(is_error)
        block = text.split("OPEN PROBLEMS:\n")[1].splitlines()
        self.assertEqual(len(block), 3)
        self.assertEqual(block[0], "  W0 O0: warn 0")

    def test_problems_from_earlier_calls_persist_through_ctx_state(self):
        warn = Finding("warning", "NO_GEOMETRIC_CHANGE", "Cut001", "volume unchanged by the cut")
        state = {}
        ctx, (text, _, _) = self.geo_run(report(findings=[warn], added=["Box"]), state=state)
        self.assertNotIn("OPEN PROBLEMS", text, "first sighting: not persisting yet")
        self.assertIn("D|NO_GEOMETRIC_CHANGE:Cut001", state["problems"])
        ctx, (text, _, _) = self.geo_run(report(findings=[warn], added=["Box"]), state=state)
        self.assertIn("OPEN PROBLEMS:\n  NO_GEOMETRIC_CHANGE Cut001: volume unchanged by the cut", text)
        ctx, (text, _, _) = self.geo_run(report(added=["Box"]), state=state)
        self.assertNotIn("OPEN PROBLEMS", text)
        self.assertEqual(state["problems"], {}, "replaced by the latest report")

    def test_state_producers_last_health_touched_last_doc(self):
        state = {}
        rep = report(added=["Box"])
        self.geo_run(rep, state=state)
        self.assertEqual(state["last_doc"], "D")
        self.assertEqual(state["last_health"]["D"]["doc"], "D")
        self.assertEqual(state["touched"], {"D": {"Box"}})
        rep2 = report()
        rep2.delta = ObjectDelta(changed=[{"name": "Lid", "props": ["Shape"]}])
        self.geo_run(rep2, before={"D": snap("D", Lid=obj(1))}, after={"D": snap("D", Lid=obj(2))}, state=state)
        self.assertEqual(state["touched"]["D"], {"Box", "Lid"})

    def test_clipping_keeps_the_health_line(self):
        rep = report(added=["Box"])
        health = FakeHealth(self.calls, snapshots=[{"D": snap("D")}, {"D": snap("D", Box=obj())}],
                            reports={"D": rep})
        ctx = self.use(health, FakeProject(self.calls), None)
        text, _, _ = self.call(ctx, "zz_big", {"doc_name": "D"})
        self.assertIn("chars omitted; call get_objects / get_object for details", text)
        self.assertTrue(text.startswith("S" * 8000))
        self.assertIn("THE END\n\n[health] doc 'D'", text)
        self.assertTrue(text.endswith("Δ +Box"))
        self.assertLess(len(text), 12000 + 300)

    def test_clipping_also_applies_without_hooks(self):
        text, _, _ = tools.run_tool("zz_big", {})
        self.assertIn("chars omitted", text)
        self.assertTrue(text.endswith("THE END"))

    def test_failed_handler_with_changes_gets_a_partial_state_line(self):
        rep = report(added=["Half"], counts={"objects": 2, "solids": 1, "errors": 0, "warnings": 0})
        project = FakeProject(self.calls)
        ctx, (text, _, is_error) = self.geo_run(rep, project=project, name="zz_fail")
        self.assertTrue(is_error)
        self.assertTrue(text.startswith("it did not work: bad input; use another one"))
        self.assertIn("PARTIAL STATE: the failed call left changes: +Half. Inspect with get_objects before retrying.", text)
        self.assertFalse(ctx.outcome.handler_ok)
        self.assertFalse(project.records[0]["ok"])

    def test_failed_handler_that_left_nothing_behind_returns_the_bare_error(self):
        ctx, (text, _, is_error) = self.geo_run(report(), before={"D": snap("D", A=obj())},
                                                after={"D": snap("D", A=obj())}, name="zz_fail")
        self.assertEqual((text, is_error), ("it did not work: bad input; use another one", True))
        self.assertNotIn("[health]", text)
        self.assertEqual(self.calls.count("health"), 0, "no trace -> health is not even asked")

    def test_a_failed_call_that_only_regenerated_shapes_reads_as_left_nothing_behind(self):
        """A kit rollback recomputes the touched solids: new shape hashes, same geometry, no PARTIAL STATE."""
        solid = SmallPureHelpersTest._solid
        before = {"D": snap("D", Shell=solid(shape_hash=1), Lid=solid(shape_hash=3))}
        after = {"D": snap("D", Shell=solid(shape_hash=2), Lid=solid(shape_hash=4))}
        ctx, (text, _, is_error) = self.geo_run(report(), before=before, after=after, name="zz_fail")
        self.assertEqual((text, is_error), ("it did not work: bad input; use another one", True))
        self.assertEqual(self.calls.count("health"), 0)

    def test_a_failed_call_lists_only_the_real_changes_in_the_partial_state_line(self):
        solid = SmallPureHelpersTest._solid
        before = {"D": snap("D", Shell=solid(shape_hash=1), Lid=solid(shape_hash=3))}
        after = {"D": snap("D", Shell=solid(shape_hash=2), Lid=solid(shape_hash=4, volume=7.0))}
        rep = HealthReport(doc="D", ts=0.0, findings=[], counts={"objects": 2, "solids": 2, "errors": 0, "warnings": 0},
                           delta=ObjectDelta(changed=[
                               {"name": "Shell", "props": ["Shape"], "volume_before": 10.0, "volume_after": 10.0},
                               {"name": "Lid", "props": ["Shape"], "volume_before": 10.0, "volume_after": 7.0}]))
        ctx, (text, _, is_error) = self.geo_run(rep, before=before, after=after, name="zz_fail")
        self.assertTrue(is_error)
        self.assertIn("PARTIAL STATE: the failed call left changes: ~Lid(vol 10.0->7.0). Inspect", text)
        partial = next(line for line in text.splitlines() if line.startswith("PARTIAL STATE"))
        self.assertNotIn("Shell", partial)
        self.assertEqual(ctx.state["touched"]["D"], {"Lid"}, "the regenerated Shell is not recorded as touched")

    def test_a_successful_call_still_reports_regenerated_dependents_as_changed(self):
        solid = SmallPureHelpersTest._solid
        before = {"D": snap("D", Shell=solid(shape_hash=1))}
        after = {"D": snap("D", Shell=solid(shape_hash=2))}
        rep = HealthReport(doc="D", ts=0.0, findings=[], counts={"objects": 1, "solids": 1, "errors": 0, "warnings": 0},
                           delta=ObjectDelta(changed=[{"name": "Shell", "props": ["Shape"],
                                                       "volume_before": 10.0, "volume_after": 10.0}]))
        ctx, (text, _, is_error) = self.geo_run(rep, before=before, after=after)
        self.assertFalse(is_error)
        self.assertEqual(ctx.state["touched"]["D"], {"Shell"})
        self.assertIn("~Shell", text)

    def test_a_crashing_handler_reports_type_message_and_traceback(self):
        ctx, (text, _, is_error) = self.geo_run(report(), before={"D": snap("D")}, after={"D": snap("D")},
                                                name="zz_boom", args={})
        self.assertTrue(is_error)
        self.assertTrue(text.startswith("RuntimeError: kaboom\n"))
        self.assertIn("Traceback", text)

    def test_bad_arguments_never_reach_the_handler_or_health(self):
        health = FakeHealth(self.calls, snapshots=[{"D": snap("D")}, {"D": snap("D")}], reports={"D": report()})
        project = FakeProject(self.calls)
        ctx = self.use(health, project, None)
        text, _, is_error = self.call(ctx, "zz_geo", {"size": "huge"})
        self.assertTrue(is_error)
        self.assertIn("Bad arguments for zz_geo", text)
        self.assertNotIn("handler", self.calls)
        self.assertNotIn("health", self.calls)
        self.assertEqual(self.calls.count("project"), 1, "the failed attempt is still recorded as a failed step")
        self.assertFalse(project.records[0]["ok"])

    def test_dispatch_timeout_and_stop_skip_health(self):
        timeout = "'execute_code' did not complete within 90s (the GUI thread stayed busy)"

        def slow(doc_name="D"):
            raise tools.ToolError(timeout)

        tools.tool("zz_slow", "A geometry tool whose dispatch times out.",
                   {"type": "object", "properties": {"doc_name": {"type": "string"}}},
                   category="build", geometry=True)(slow)
        health = FakeHealth(self.calls, snapshots=[{"D": snap("D")}, {"D": snap("D", Box=obj())}],
                            reports={"D": report()})
        project = FakeProject(self.calls)
        ctx = self.use(health, project, None)
        text, _, is_error = self.call(ctx, "zz_slow", {"doc_name": "D"})
        self.assertEqual((text, is_error), (timeout, True))
        self.assertNotIn("health", self.calls)
        self.assertEqual(project.records[0]["text"], timeout)
        self.assertEqual(len(self.events("hook_skipped")), 1)
        # a successful handler under Stop: health skipped as well
        self.calls.clear()
        health.snapshots = [{"D": snap("D")}]
        stopped = self.use(health, FakeProject(self.calls), None, should_stop=lambda: True)
        text, _, is_error = self.call(stopped, "zz_geo", {"doc_name": "D"})
        self.assertEqual((text, is_error), ("geo ok size=1", False))
        self.assertNotIn("health", self.calls)

    def test_no_report_means_no_suffix(self):
        health = FakeHealth(self.calls, snapshots=[{"D": snap("D")}, {"D": snap("D")}], reports={})
        ctx = self.use(health, None, FakeLedger(self.calls))
        text, _, is_error = self.call(ctx, "zz_geo", {"doc_name": "D"})
        self.assertEqual((text, is_error), ("geo ok size=1", False))
        self.assertIsNone(ctx.outcome.health)

    def test_without_a_baseline_errors_this_turn_already_reported_are_not_new(self):
        err = Finding("error", "OLD_ERR", "X", "known")
        state = {"problems": {"D|OLD_ERR:X": "OLD_ERR X: known"}}
        rep = report(findings=[err], added=["Box"], counts={"objects": 1, "solids": 1, "errors": 1, "warnings": 0})
        ctx, (text, _, is_error) = self.geo_run(rep, before={}, after={"D": snap("D", Box=obj())}, state=state)
        self.assertFalse(is_error)
        fresh = Finding("error", "FRESH", "Y", "brand new")
        rep = report(findings=[err, fresh], added=["Box"], counts={"objects": 1, "solids": 1, "errors": 2, "warnings": 0})
        ctx, (text, _, is_error) = self.geo_run(rep, before={}, after={"D": snap("D", Box=obj())}, state=state)
        self.assertTrue(is_error)

    def test_a_script_that_touches_another_document_is_attributed_by_delta(self):
        health = FakeHealth(self.calls, snapshots=[{"A": snap("A", X=obj()), "B": snap("B")},
                                                   {"A": snap("A", X=obj()), "B": snap("B", Y=obj(), Z=obj())}],
                            reports={"A": report("A"), "B": report("B", added=["Y", "Z"])})
        ctx = self.use(health, FakeProject(self.calls), None)
        text, _, _ = self.call(ctx, "zz_geo", {})
        # no doc_name was sent: the handler default 'D' is not used for attribution -> largest delta (B)
        self.assertEqual(ctx.outcome.doc, "B")
        self.assertIn("[health] doc 'B'", text)
        self.assertEqual(ctx.state["last_doc"], "B")

    def test_other_documents_with_changes_add_their_own_health_line(self):
        health = FakeHealth(self.calls, snapshots=[{"D": snap("D"), "E": snap("E")},
                                                   {"D": snap("D", Box=obj()), "E": snap("E", Pad=obj())}],
                            reports={"D": report("D", added=["Box"]), "E": report("E", added=["Pad"])})
        ctx = self.use(health, FakeProject(self.calls), None)
        text, _, _ = self.call(ctx, "zz_geo", {"doc_name": "D"})
        self.assertIn("[health] doc 'D'", text)
        self.assertTrue(text.rstrip().endswith("Δ +Pad"))
        self.assertIn("[health] doc 'E'", text)
        self.assertEqual(set(ctx.state["last_health"]), {"D", "E"})
        self.assertEqual(ctx.state["touched"]["E"], {"Pad"})


class StepTitleAndStateTest(PipelineBase):
    def titled(self, args, hint="", ctx_title_tool=None):
        ctx = self.use(None, FakeProject(self.calls), None)
        if hint:
            ctx.state["title_hint"] = hint
        with mock.patch.object(toolrun, "_health_api", return_value=None):
            self.call(ctx, "zz_geo", args)
        return ctx

    def test_precedence_step_title_then_hint_then_tool_title_then_table(self):
        ctx = self.titled({"step_title": "From the model"}, hint="From the handler")
        self.assertEqual(ctx.outcome.title, "From the model")
        ctx = self.titled({}, hint="From the handler")
        self.assertEqual(ctx.outcome.title, "From the handler")
        self.assertEqual(ctx.state.get("title_hint", ""), "", "popped by run_tool")
        tools._HANDLERS["zz_geo"].title = lambda a: f"Custom {a.get('size')}"
        ctx = self.titled({"size": 4})
        self.assertEqual(ctx.outcome.title, "Custom 4")
        tools._HANDLERS["zz_geo"].title = None
        ctx = self.titled({})
        self.assertEqual(ctx.outcome.title, "zz_geo", "no table entry: the tool name")

    def test_step_title_is_cleaned_clipped_and_never_bound_to_the_handler(self):
        ctx = self.titled({"step_title": "x" * 200 + "\nmore"})
        self.assertLessEqual(len(ctx.outcome.title), 80)
        self.assertNotIn("\n", ctx.outcome.title)
        self.assertEqual(self.seen[-1], {"doc_name": "D", "size": 1, "include_screenshot": True})
        self.assertEqual(ctx.step_title, ctx.outcome.title)

    def test_step_title_on_a_read_tool_is_accepted_and_ignored(self):
        text, _, is_error = tools.run_tool("zz_read", {"step_title": "Look around"})
        self.assertEqual((text, is_error), ("read ok", False))

    def test_args_override_is_recorded_but_not_passed_to_the_handler(self):
        def staged(doc_name="D", stage="next"):
            current_context().state["args_override"] = {"stage": "shell"}
            current_context().state["title_hint"] = "Build enclosure: shell"
            current_context().state["extra"] = {"reference_result": {"x": 1}, "param_names": ["wall_t"]}
            return "built", None

        tools.tool("zz_stage", "A stage tool that resolves its own argument.",
                   {"type": "object", "properties": {"doc_name": {"type": "string"}, "stage": {"type": "string"}}},
                   category="build", geometry=True)(staged)
        project = FakeProject(self.calls)
        ctx = self.use(None, project, None)
        with mock.patch.object(toolrun, "_health_api", return_value=None):
            self.call(ctx, "zz_stage", {"doc_name": "D", "stage": "next"})
        self.assertEqual(ctx.outcome.args, {"doc_name": "D", "stage": "shell"})
        self.assertEqual(project.records[0]["args"], {"doc_name": "D", "stage": "shell"})
        self.assertEqual(ctx.outcome.title, "Build enclosure: shell")
        self.assertEqual(ctx.outcome.extra, {"reference_result": {"x": 1}, "param_names": ["wall_t"]})
        for key in ("args_override", "title_hint", "extra"):
            self.assertNotIn(key, ctx.state, f"{key} is popped after the call")

    def test_lite_state_hides_the_screenshot_switch_from_the_handler(self):
        ctx = self.use(None, FakeProject(self.calls), None, state={"lite": True})
        with mock.patch.object(toolrun, "_health_api", return_value=None):
            self.call(ctx, "zz_geo", {"doc_name": "D", "include_screenshot": True})
        self.assertIs(self.seen[-1]["include_screenshot"], False)

    def test_out_of_phase_tools_still_run_and_are_logged(self):
        ctx = self.use(None, FakeProject(self.calls), None)
        ctx.phase = "INTAKE"                       # zz_geo is introduced in BUILD only
        with mock.patch.object(toolrun, "_health_api", return_value=None):
            text, _, is_error = self.call(ctx, "zz_geo", {"doc_name": "D"})
        self.assertEqual((text, is_error), ("geo ok size=1", False))
        events = self.events("tool_out_of_phase")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kwargs, {"name": "zz_geo", "phase": "INTAKE"})
        self.log.reset_mock()
        ctx.phase = "BUILD"
        with mock.patch.object(toolrun, "_health_api", return_value=None):
            self.call(ctx, "zz_geo", {"doc_name": "D"})
        self.assertEqual(self.events("tool_out_of_phase"), [])
        ctx.phase = ""
        with mock.patch.object(toolrun, "_health_api", return_value=None):
            self.call(ctx, "zz_geo", {"doc_name": "D"})
        self.assertEqual(self.events("tool_out_of_phase"), [], "no phase bound: nothing to compare")

    def test_every_call_logs_one_tool_event(self):
        ctx = self.use(None, FakeProject(self.calls), None)
        with mock.patch.object(toolrun, "_health_api", return_value=None):
            self.call(ctx, "zz_geo", {"doc_name": "D"})
        events = self.events("tool")
        self.assertEqual(len(events), 1)
        fields = events[0].kwargs
        self.assertEqual((fields["name"], fields["ok"], fields["phase"], fields["geometry"]), ("zz_geo", True, "BUILD", True))
        self.assertEqual(fields["step_index"], 7)
        self.assertIn("seconds", fields)
        self.assertIn("result_chars", fields)
        self.assertNotIn("error", fields)
        self.log.reset_mock()
        self.call(self.use(None, None, None), "zz_fail", {})
        self.assertIn("error", self.events("tool")[0].kwargs)

    def test_kill_switch_skips_health_and_ledger_but_keeps_project_recording(self):
        health = FakeHealth(self.calls, snapshots=[{"D": snap("D")}, {"D": snap("D")}], reports={"D": report()})
        project, ledger = FakeProject(self.calls), FakeLedger(self.calls)
        ctx = self.use(health, project, ledger)
        with mock.patch.dict(os.environ, {"OSCAR_FLOW": "0"}):
            text, _, is_error = self.call(ctx, "zz_geo", {"doc_name": "D"})
        self.assertEqual((text, is_error), ("geo ok size=1", False))
        self.assertNotIn("health", self.calls)
        self.assertNotIn("snapshot_all", self.calls)
        self.assertNotIn("ledger", self.calls)
        self.assertIn("project", self.calls)
        self.assertIn("checkpoint", self.calls)

    def test_the_turn_flag_switches_the_hooks_off_like_the_environment_does(self):
        """``AgentWorker(flow=False)`` / the OscarFlow pref reach run_tool through ``ctx.state``."""
        health = FakeHealth(self.calls, snapshots=[{"D": snap("D")}, {"D": snap("D")}], reports={"D": report()})
        project, ledger = FakeProject(self.calls), FakeLedger(self.calls)
        ctx = self.use(health, project, ledger, state={"flow_enforcement": False})
        text, _, is_error = self.call(ctx, "zz_geo", {"doc_name": "D"})
        self.assertEqual((text, is_error), ("geo ok size=1", False))
        self.assertNotIn("health", self.calls)
        self.assertNotIn("snapshot_all", self.calls)
        self.assertIn("project", self.calls)
        self.assertIn("checkpoint", self.calls)

    def test_the_flow_flag_defaults_to_on_and_ignores_junk(self):
        self.assertTrue(toolrun._flow_enabled())
        self.assertTrue(toolrun._flow_enabled(None))
        self.assertTrue(toolrun._flow_enabled(ToolContext(state={})))
        self.assertTrue(toolrun._flow_enabled(ToolContext(state={"flow_enforcement": True})))
        self.assertTrue(toolrun._flow_enabled(ToolContext(state={"flow_enforcement": 0})))     # only an explicit False switches it off
        self.assertFalse(toolrun._flow_enabled(ToolContext(state={"flow_enforcement": False})))
        with mock.patch.dict(os.environ, {"OSCAR_FLOW": "0"}):
            self.assertFalse(toolrun._flow_enabled(ToolContext(state={"flow_enforcement": True})))

    def test_none_arguments_mean_no_arguments_and_a_list_is_a_clear_error(self):
        text, _, is_error = tools.run_tool("zz_read", None)
        self.assertEqual((text, is_error), ("read ok", False))
        project = FakeProject(self.calls)
        ctx = self.use(None, project, None)
        text, _, is_error = self.call(ctx, "zz_mut", ["not", "an", "object"])
        self.assertTrue(is_error)
        self.assertIn("expected a JSON object", text)
        self.assertEqual(project.records[0]["args"], {})
        self.assertFalse(project.records[0]["ok"])

    def test_the_outcome_is_replaced_on_every_call(self):
        ctx = self.use(None, FakeProject(self.calls), None)
        self.call(ctx, "zz_mut")
        first = ctx.outcome
        self.assertEqual(first.name, "zz_mut")
        self.call(ctx, "zz_read")
        self.assertIsNot(ctx.outcome, first)
        self.assertEqual(ctx.outcome.name, "zz_read")
        self.assertFalse(ctx.outcome.mutating)
        self.assertEqual(ctx.outcome.image_b64, "IMAGEDATA")

    def test_context_is_per_thread_and_restored(self):
        ctx = self.use(None, FakeProject(self.calls), None)
        self.assertFalse(current_context().active)
        with use_context(ctx):
            self.assertIs(current_context(), ctx)
        self.assertFalse(current_context().active)
        self.assertIsNone(ToolContext().outcome)
        self.assertIsInstance(ToolOutcome(), ToolOutcome)


if __name__ == "__main__":
    unittest.main()
