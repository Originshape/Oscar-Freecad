"""Scripted execution (``oscar.cad.scripting``): atomic ``run_script`` with rollback and an object delta, the fresh seeded
namespace, the injected helpers and ``register_helpers``, the error hint table, the AST pre-flight warnings and the
background jobs. Dispatcher and headless helper: ``test_scripting_dispatch``.

Runs under ``freecadcmd`` (the dispatcher runs inline there; documents have ``UndoMode == 0``, the case the rollback
must survive)."""

import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402

from oscar.cad import dispatch, scripting  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402

HELPER_NAMES = ("kit", "P", "zz_helper", "zz_other")


class ScriptCase(unittest.TestCase):
    """A fresh document per test; every document the test (or its script) left open is closed afterwards."""

    def setUp(self):
        self._before = set(App.listDocuments())
        self.doc = App.newDocument("ST")
        self.dn = self.doc.Name          # another test may have leaked a document called "ST": never hard-code it
        App.setActiveDocument(self.dn)
        scripting.reset_state()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for name in set(App.listDocuments()) - self._before:
            App.closeDocument(name)
        for name in HELPER_NAMES:
            scripting.unregister_helpers(name)
        scripting.reset_state()

    def run_ok(self, code, **kw):
        result = scripting.run_script(code, **kw)
        self.assertTrue(result.get("success"), result)
        return result

    def run_fail(self, code, **kw):
        result = scripting.run_script(code, **kw)
        self.assertFalse(result.get("success"), result)
        return result

    def names(self, doc=None):
        return [o.Name for o in (doc or self.doc).Objects]

    def sheet(self, rows=(("wall_t", "=2 mm"),), name="Params"):
        sh = self.doc.addObject("Spreadsheet::Sheet", name)
        for i, (alias, formula) in enumerate(rows, start=1):
            sh.set(f"A{i}", alias)
            sh.set(f"B{i}", formula)
            sh.setAlias(f"B{i}", alias)
        self.doc.recompute()
        return sh


# ---------------------------------------------------------------------------------------------- run_script: success


class RunScriptSuccessTest(ScriptCase):
    def test_creates_objects_reports_output_and_object_delta(self):
        r = self.run_ok("print('hi')\nb = App.ActiveDocument.addObject('Part::Box', 'Box')\nb.Length = 20\n")
        self.assertEqual(r["message"], "Script executed.\nOutput: hi\nObjects: +Box")
        self.assertEqual(r["deltas"], {self.dn: {"added": ["Box"], "removed": [], "changed": []}})
        self.assertNotIn("lost_geometry", r)
        self.assertAlmostEqual(self.doc.getObject("Box").Shape.Volume, 20 * 10 * 10)   # recomputed although the script did not

    def test_delta_lists_changes_and_losses(self):
        a = self.doc.addObject("Part::Box", "A")
        self.doc.addObject("Part::Box", "Gone")
        self.doc.recompute()
        r = self.run_ok("d = App.ActiveDocument\nd.getObject('A').Length = 30\nd.removeObject('Gone')\n")
        self.assertIn("-Gone", r["message"])
        self.assertIn("~A(vol 1000.0->3000.0)", r["message"])
        self.assertEqual(r["lost_geometry"], {self.dn: ["Gone"]})   # the merge guard builds on this
        del a

    def test_no_delta_line_when_nothing_changed(self):
        r = self.run_ok("x = 1 + 1\nprint(x)")
        self.assertEqual(r["message"], "Script executed.\nOutput: 2")
        self.assertEqual(r["deltas"], {})

    def test_a_document_created_by_the_script_is_all_new(self):
        r = self.run_ok("d = App.newDocument('ST_made')\nd.addObject('Part::Box', 'Made')\nd.recompute()\n")
        self.assertEqual(r["deltas"]["ST_made"]["added"], ["Made"])
        self.assertIn("ST_made", App.listDocuments())

    def test_two_documents_changed_are_labelled(self):
        # the run only snapshots the first _MAX_DOCS open documents (the active one first): documents that other tests
        # leaked in a long suite run could push the second one out of the cap, so this test lifts it
        patcher = mock.patch.object(scripting, "_MAX_DOCS", 10_000)
        patcher.start()
        self.addCleanup(patcher.stop)
        other = App.newDocument("ST_two")
        r = self.run_ok(f"App.getDocument('{self.dn}').addObject('Part::Box', 'A')\n"
                        f"App.getDocument('{other.Name}').addObject('Part::Box', 'B')\n")
        self.assertIn(f"[{self.dn}] +A", r["message"])
        self.assertIn(f"[{other.Name}] +B", r["message"])

    def test_output_is_clipped(self):
        r = self.run_ok("print('x' * 20000)")
        self.assertLess(len(r["message"]), 9000)
        self.assertIn("(output truncated)", r["message"])

    def test_sys_exit_zero_is_success(self):
        self.run_ok("import sys\nsys.exit(0)")
        self.run_ok("raise SystemExit")

    def test_untouched_documents_stay_untouched_and_undo_mode_is_restored(self):
        self.assertEqual(self.doc.UndoMode, 0)
        self.run_ok("App.ActiveDocument.addObject('Part::Box', 'Box')")
        self.assertEqual(self.doc.UndoMode, 0)
        self.assertFalse(self.doc.HasPendingTransaction)   # committed, not left open

    def test_pre_flight_warning_is_attached_to_a_success(self):
        self.doc.addObject("Part::Box", "Box")
        self.doc.recompute()
        r = self.run_ok("App.ActiveDocument.getObject('Box').Shape = __import__('Part').makeSphere(5)\n")
        self.assertEqual(len(r["warnings"]), 1)
        self.assertIn("WARNING (pre-flight): line 1:", r["message"])


# ---------------------------------------------------------------------------------------------- run_script: failure


class RunScriptRollbackTest(ScriptCase):
    def test_error_names_the_line_keeps_output_and_rolls_back_created_objects(self):
        r = self.run_fail("print('before')\nApp.ActiveDocument.addObject('Part::Box', 'Box')\nraise ValueError('nope')\n")
        text = r["error"]
        self.assertIn("ValueError: nope", text)
        self.assertIn("at script line 3: raise ValueError('nope')", text)
        self.assertIn("ROLLED BACK", text)
        self.assertIn("removed 1 created object(s): Box", text)
        self.assertIn("run it again IN FULL", text)
        self.assertIn("Output printed before the error:\nbefore", text)
        self.assertEqual(r["rolled_back"]["removed"], {self.dn: ["Box"]})
        self.assertEqual(self.names(), [])
        self.assertEqual(self.doc.UndoMode, 0)                       # restored
        self.assertFalse(self.doc.HasPendingTransaction)

    def test_a_retry_does_not_produce_numbered_duplicates(self):
        bad = "App.ActiveDocument.addObject('Part::Box', 'Shell')\nraise RuntimeError('x')"
        self.run_fail(bad)
        self.run_ok("App.ActiveDocument.addObject('Part::Box', 'Shell')")
        self.assertEqual(self.names(), ["Shell"])                    # not Shell001

    def test_deleted_objects_and_edited_values_come_back_with_undomode_zero(self):
        box = self.doc.addObject("Part::Box", "Keep")
        box.Length = 11
        victim = self.doc.addObject("Part::Box", "Victim")
        victim.Width = 7
        self.doc.recompute()
        code = ("d = App.ActiveDocument\n"
                "d.getObject('Keep').Length = 99\n"
                "d.removeObject('Victim')\n"
                "d.addObject('Part::Cylinder', 'New')\n"
                "d.recompute()\n"
                "raise KeyError('late')\n")
        r = self.run_fail(code)
        self.assertEqual(self.doc.UndoMode, 0)
        self.assertEqual(sorted(self.names()), ["Keep", "Victim"])
        self.assertEqual(self.doc.getObject("Keep").Length.Value, 11)
        self.assertEqual(self.doc.getObject("Victim").Width.Value, 7)
        self.assertEqual(self.doc.getObject("Victim").Shape.Volume, 10 * 7 * 10)   # recomputed
        self.assertEqual(r["rolled_back"]["removed"], {self.dn: ["New"]})
        self.assertEqual(r["rolled_back"]["restored"], {self.dn: ["Victim"]})
        self.assertIn("restored 1 deleted object(s): Victim", r["error"])
        self.assertEqual(r["rolled_back"]["leftover"], {})

    def test_a_deleted_params_sheet_is_restored_with_its_aliases(self):
        self.sheet([("wall_t", "=2 mm"), ("lid_t", "=3 mm")])
        self.run_fail("d = App.ActiveDocument\nd.removeObject('Params')\nraise ValueError('x')")
        sheet = self.doc.getObject("Params")
        self.assertIsNotNone(sheet)
        self.assertEqual(sheet.getAlias("B1"), "wall_t")
        self.assertEqual(sheet.get("lid_t").Value, 3.0)

    def test_an_edited_expression_and_sheet_cell_revert(self):
        sheet = self.sheet()
        box = self.doc.addObject("Part::Box", "Box")
        box.setExpression("Length", "Params.wall_t * 5")
        self.doc.recompute()
        self.run_fail("d = App.ActiveDocument\nd.getObject('Params').set('B1', '=9 mm')\n"
                      "d.getObject('Box').setExpression('Length', None)\nraise ValueError('x')")
        self.assertEqual(dict(box.ExpressionEngine).get("Length"), "Params.wall_t * 5")
        self.assertEqual(sheet.get("wall_t").Value, 2.0)

    def test_a_document_created_by_the_failed_script_is_closed_and_the_active_one_restored(self):
        r = self.run_fail("d = App.newDocument('ST_ghost')\nd.addObject('Part::Box', 'X')\nraise ValueError('x')")
        self.assertNotIn("ST_ghost", App.listDocuments())
        self.assertEqual(r["rolled_back"]["closed_docs"], ["ST_ghost"])
        self.assertIn("closed the document(s) the script created: ST_ghost", r["error"])
        self.assertEqual(App.ActiveDocument.Name, self.doc.Name)

    def test_a_script_that_closes_a_document_is_warned_about(self):
        victim = App.newDocument("ST_victim")
        r = self.run_fail("App.closeDocument('ST_victim')\nraise ValueError('x')")
        self.assertIn("WARNING: the script closed document(s) ST_victim", r["error"])
        self.assertEqual(r["rolled_back"]["closed_by_script"], ["ST_victim"])
        del victim

    def test_system_exit_nonzero_is_a_failure_with_rollback(self):
        r = self.run_fail("App.ActiveDocument.addObject('Part::Box', 'Box')\nimport sys\nsys.exit(2)")
        self.assertIn("SystemExit: the script called sys.exit(2)", r["error"])
        self.assertEqual(self.names(), [])

    def test_a_syntax_error_never_runs_the_script_and_names_the_line(self):
        r = self.run_fail("x = 1\ny = (\n")
        self.assertIn("SyntaxError", r["error"])
        self.assertIn("script line", r["error"])
        self.assertNotIn("ROLLED BACK", r["error"])
        self.assertNotIn("rolled_back", r)

    def test_call_chain_and_function_names(self):
        code = "def g():\n    raise ValueError('deep')\ndef f():\n    g()\nf()\n"
        text = self.run_fail(code)["error"]
        self.assertIn("at script line 2: raise ValueError('deep')", text)
        self.assertIn("call chain:\n  line 5\n  line 4 in f", text)

    def test_a_failure_does_not_touch_other_documents(self):
        other = App.newDocument("ST_other")
        other.addObject("Part::Box", "Safe")
        self.run_fail("App.getDocument('ST_other').addObject('Part::Box', 'Extra')\nraise ValueError('x')")
        self.assertEqual([o.Name for o in other.Objects], ["Safe"])

    def test_a_script_that_already_ran_in_a_transaction_of_its_own_still_leaves_no_transaction_open(self):
        self.run_fail("d = App.ActiveDocument\nd.openTransaction('mine')\nd.addObject('Part::Box', 'B')\nraise ValueError('x')")
        self.assertFalse(self.doc.HasPendingTransaction)
        self.assertEqual(self.names(), [])


# ---------------------------------------------------------------------------------------------- cancel / timeout


class RunScriptCancelTest(ScriptCase):
    def test_stop_before_the_script_starts_means_it_never_runs(self):
        r = self.run_fail("App.ActiveDocument.addObject('Part::Box', 'Box')", should_stop=lambda: True)
        self.assertEqual(r["error"], "stopped")
        self.assertTrue(r["stopped"])
        self.assertEqual(self.names(), [])

    def test_the_bound_tool_context_supplies_should_stop(self):
        with use_context(ToolContext(should_stop=lambda: True)):
            r = self.run_fail("App.ActiveDocument.addObject('Part::Box', 'Box')")
        self.assertTrue(r["stopped"])
        self.assertEqual(self.names(), [])

    def test_timeout_results_pass_their_flags_through_and_keep_the_output(self):
        timeout = {"success": False, "error": "'execute_code' did not complete within 5s (the GUI thread stayed busy). "
                                              "It is STILL RUNNING on the GUI thread", "timed_out": True, "still_running": True}
        with mock.patch.object(dispatch, "dispatch", return_value=(timeout, True)):
            r = self.run_fail("print('x')")
        self.assertTrue(r["timed_out"] and r["still_running"])
        self.assertIn("STILL RUNNING", r["error"])

    def test_dispatch_receives_the_timeout_op_and_should_stop(self):
        seen = {}

        def fake(fn, timeout=60.0, op="", should_stop=None):
            seen.update(timeout=timeout, op=op, should_stop=should_stop)
            return {"success": False, "error": "x"}, True

        stop = lambda: False  # noqa: E731
        with mock.patch.object(dispatch, "dispatch", side_effect=fake):
            scripting.run_script("pass", timeout=7, should_stop=stop)
        self.assertEqual((seen["timeout"], seen["op"], seen["should_stop"]), (7, "execute_code", stop))


# ---------------------------------------------------------------------------------------------- namespace


class NamespaceTest(ScriptCase):
    def test_every_run_starts_fresh_but_state_persists(self):
        self.run_ok("leaked = 41\nlen = 'shadowed'\nstate['n'] = 5")
        r = self.run_fail("print(leaked)")
        self.assertIn("NameError", r["error"])
        self.assertIn("fresh namespace", r["error"])
        self.assertEqual(self.run_ok("print(len([1, 2]), state['n'])")["message"], "Script executed.\nOutput: 2 5")

    def test_main_guard_and_synthetic_file(self):
        out = self.run_ok("import os\nif __name__ == '__main__':\n    print('main', os.path.basename(__file__), "
                          "os.path.basename(os.path.dirname(__file__)))")["message"]
        self.assertIn("main execute_code.py scripts", out)

    def test_preimported_modules_and_helpers(self):
        out = self.run_ok("print(Part.__name__, Sketcher.__name__, Spreadsheet.__name__, Mesh.__name__, math.pi > 3, "
                          "callable(qv), callable(param), callable(set_param), App is FreeCAD, state is not None)")["message"]
        self.assertIn("Part Sketcher Spreadsheet Mesh True True True True True True", out)

    def test_new_namespace_is_a_copy(self):
        ns = scripting.new_namespace()
        ns["junk"] = 1
        ns["math"] = None
        self.assertNotIn("junk", scripting.new_namespace())
        self.assertIsNotNone(scripting.new_namespace()["math"])
        self.assertIs(scripting.new_namespace()["state"], scripting.STATE)
        self.assertEqual(scripting.new_namespace()["__name__"], "__main__")

    def test_reset_state(self):
        scripting.STATE["a"] = 1
        scripting.reset_state()
        self.assertEqual(scripting.STATE, {})


# ---------------------------------------------------------------------------------------------- helpers


class ParamHelpersTest(ScriptCase):
    def test_qv_reads_quantities_and_plain_numbers(self):
        sheet = self.sheet([("wall_t", "=2 mm"), ("count", "=4")])
        self.assertEqual(scripting.qv(sheet.get("wall_t")), 2.0)
        self.assertEqual(scripting.qv(sheet.get("count")), 4)        # an int has no .Value: the classic crash
        self.assertEqual(scripting.qv("text"), "text")
        self.assertEqual(self.run_ok("print(qv(App.ActiveDocument.Params.get('count')))")["message"], "Script executed.\nOutput: 4")

    def test_param_reads_a_number_and_explains_failures(self):
        self.sheet([("wall_t", "=2 mm"), ("count", "=4"), ("neg", "-1.5 mm")])
        self.assertEqual(scripting.param("wall_t"), 2.0)
        self.assertEqual(scripting.param("count"), 4.0)
        with self.assertRaises(ValueError) as cm:
            scripting.param("missing")
        self.assertIn("param 'missing' is not in sheet 'Params'; known parameters: wall_t, count, neg", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            scripting.param("neg")
        self.assertIn("holds the text '-1.5 mm'", str(cm.exception))
        self.assertIn("=-1.5 mm", str(cm.exception))

    def test_param_without_a_sheet_or_a_document(self):
        with self.assertRaises(ValueError) as cm:
            scripting.param("wall_t")
        self.assertIn("has no 'Params' spreadsheet", str(cm.exception))
        with mock.patch.object(App, "ActiveDocument", None):
            with self.assertRaises(RuntimeError):
                scripting.param("wall_t")

    def test_param_accepts_the_parameters_sheet_and_an_explicit_document(self):
        self.sheet([("a_val", "=1 mm")], name="Parameters")
        self.assertEqual(scripting.param("a_val", self.doc), 1.0)

    def test_set_param_writes_a_formula_recomputes_and_keeps_the_unit(self):
        self.sheet([("wall_t", "=2 mm"), ("off", "=1 mm")])
        box = self.doc.addObject("Part::Box", "Box")
        box.setExpression("Length", "Params.wall_t * 5")
        self.doc.recompute()
        scripting.set_param("wall_t", 3)
        self.assertEqual(box.Length.Value, 15.0)                   # dependents followed
        scripting.set_param("off", -1.5)                           # negative: never stored as text
        self.assertEqual(scripting.param("off"), -1.5)
        self.assertEqual(self.doc.Params.getContents("B2"), "=-1.5 mm")
        scripting.set_param("wall_t", "=4 mm")
        self.assertEqual(scripting.param("wall_t"), 4.0)
        scripting.set_param("wall_t", 6, "mm")
        self.assertEqual(scripting.param("wall_t"), 6.0)

    def test_set_param_rejects_bad_input(self):
        self.sheet([("wall_t", "=2 mm")])
        for bad, text in ((float("nan"), "not a finite number"), (float("inf"), "not a finite number"),
                          ("2 mm", "formula starting with '='")):
            with self.assertRaises(ValueError) as cm:
                scripting.set_param("wall_t", bad)
            self.assertIn(text, str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            scripting.set_param("nope", 1)
        self.assertIn("set_parameters creates new rows", str(cm.exception))
        self.assertEqual(scripting.param("wall_t"), 2.0)           # nothing was written


class RegisterHelpersTest(ScriptCase):
    def test_registered_helper_is_a_global_of_every_later_script(self):
        scripting.register_helpers("zz_helper", {"answer": 42})
        self.assertEqual(self.run_ok("print(zz_helper['answer'])")["message"], "Script executed.\nOutput: 42")
        self.assertEqual(scripting.registered_helpers()["zz_helper"], {"answer": 42})
        self.assertIn("zz_helper", scripting.new_namespace())

    def test_registering_again_replaces_and_unregister_removes(self):
        scripting.register_helpers("zz_helper", 1)
        scripting.register_helpers("zz_helper", 2)
        self.assertEqual(scripting.registered_helpers()["zz_helper"], 2)
        scripting.unregister_helpers("zz_helper")
        scripting.unregister_helpers("zz_helper")                  # no error when it is not registered
        self.assertNotIn("zz_helper", scripting.registered_helpers())
        self.assertIn("NameError", self.run_fail("zz_helper")["error"])

    def test_the_kit_names_are_accepted(self):
        scripting.register_helpers("kit", object())
        scripting.register_helpers("P", object())
        self.assertTrue({"kit", "P"} <= set(scripting.registered_helpers()))

    def test_bad_names_are_rejected_and_nothing_is_registered(self):
        before = scripting.registered_helpers()
        for bad in ("App", "Part", "qv", "param", "state", "math", "commit", "__x", "__name__", "class", "a-b", "", "1x", 5, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                scripting.register_helpers(bad, 1)
        self.assertEqual(scripting.registered_helpers(), before)

    def test_a_script_cannot_poison_the_registry(self):
        scripting.register_helpers("zz_helper", [1])
        self.run_ok("zz_helper = 'overwritten'\nstate['x'] = 1")
        self.assertEqual(scripting.new_namespace()["zz_helper"], [1])

    def test_registered_helpers_is_a_copy(self):
        scripting.register_helpers("zz_helper", 1)
        scripting.registered_helpers().pop("zz_helper")
        self.assertIn("zz_helper", scripting.registered_helpers())


# ---------------------------------------------------------------------------------------------- hints


class HintTableTest(ScriptCase):
    def hint(self, code, exc):
        return scripting._script_error(code, exc)

    def test_int_value_attribute_error(self):
        r = self.run_fail("v = 4\nprint(v.Value)")
        self.assertIn("hint: Spreadsheet.get() returns a plain number", r["error"])
        self.assertIn("qv(x)", r["error"])
        self.assertIn("at script line 2: print(v.Value)", r["error"])
        self.assertIn("hint:", self.run_fail("(2.5).Value")["error"])

    def test_name_error_for_a_preimported_module_in_another_process(self):
        text = scripting.format_script_error("x", "NameError", "name 'Part' is not defined", [1], mro=["NameError"])
        self.assertIn("'Part' is pre-imported in execute_code but not in other processes", text)
        self.assertIn("execute_code_headless needs its own 'import Part'", text)
        math_text = scripting.format_script_error("x", "NameError", "name 'math' is not defined", [1], mro=["NameError"])
        self.assertIn("import math", math_text)

    def test_name_error_for_stdlib_and_for_a_forgotten_variable(self):
        self.assertIn("add 'import os' at the top", self.run_fail("os.getcwd()")["error"])
        self.assertIn("fresh namespace", self.run_fail("undefined_thing + 1")["error"])

    def test_occ_error(self):
        err = type("OCCError", (Exception,), {})("Standard_ConstructionError")
        self.assertIn("OpenCASCADE rejected the operation", scripting._script_error("x", err))
        self.assertIn("OpenCASCADE rejected", scripting._script_error("x", RuntimeError("BRep_API: command not done")))
        self.assertIn("OpenCASCADE rejected", scripting._script_error("x", RuntimeError("Standard_NullObject")))

    def test_import_error_says_what_is_available(self):
        text = self.run_fail("import pandas_not_installed_zz")["error"]
        self.assertIn("'pandas_not_installed_zz' is not installed", text)
        self.assertIn("numpy", text)

    def test_none_lookup_and_document_attribute(self):
        self.assertIn("doc.getObject('Name') is None", self.run_fail("App.ActiveDocument.getObject('Nope').Shape")["error"])
        self.assertIn("internal Name", self.run_fail("App.ActiveDocument.Nope")["error"])

    def test_null_shape(self):
        text = self.run_fail("import Part\nPart.Shape().Volume")["error"]
        self.assertIn("the shape is null or invalid", text)

    def test_at_most_two_hints_and_no_hint_for_unknown_errors(self):
        self.assertEqual(scripting._hints(ValueError("plain")), [])
        self.assertLessEqual(len(scripting._hints_for("AttributeError", ["AttributeError"], "'int' object has no attribute 'Value'")), 2)
        self.assertNotIn("hint:", self.run_fail("raise ValueError('plain')")["error"])

    def test_format_script_error_is_shared_by_the_live_and_the_reported_path(self):
        live = scripting._script_error("a = 1\nb = c\n", NameError("name 'c' is not defined", name="c"))
        reported = scripting.format_script_error("a = 1\nb = c\n", "NameError", "name 'c' is not defined", [2],
                                                 mro=["NameError", "Exception"], missing="c")
        self.assertEqual(live.split("\n")[0], reported.split("\n")[0])
        self.assertEqual(live.split("\n")[-1], reported.split("\n")[-1])
        self.assertIn("at script line 2: b = c", reported)
        self.assertEqual(scripting.format_script_error("a", "ValueError", "m", []), "ValueError: m")
        self.assertEqual(scripting.format_script_error("a", "ValueError", "m", [9]), "ValueError: m")   # line outside the script


# ---------------------------------------------------------------------------------------------- pre-flight


class PreflightTest(unittest.TestCase):
    def warn(self, code):
        return scripting.preflight_warnings(code)

    def test_shape_assigned_on_a_parametric_feature_warns(self):
        (w,) = self.warn("box = doc.getObject('Box')\nbox.Shape = Part.makeSphere(5)\n")
        self.assertIn("line 2:", w)
        self.assertIn("box.Shape = ...", w)
        self.assertIn("discarded when that object recomputes", w)
        self.assertIn("Part::Feature", w)

    def test_a_fresh_plain_feature_may_take_a_shape(self):
        self.assertEqual(self.warn("o = doc.addObject('Part::Feature', 'X')\no.Shape = s\n"), [])
        self.assertEqual(self.warn("doc.addObject('Part::FeaturePython', 'X').Shape = s\n"), [])
        self.assertEqual(self.warn("o = App.ActiveDocument.addObject('Part::Feature', 'X')\no.Shape = a\no.Shape = b\n"), [])

    def test_a_feature_created_with_another_type_warns(self):
        self.assertEqual(len(self.warn("o = doc.addObject('Part::Box', 'X')\no.Shape = s\n")), 1)

    def test_augmented_and_annotated_assignment_and_attribute_chains(self):
        self.assertEqual(len(self.warn("a.Shape += b\n")), 1)
        self.assertEqual(len(self.warn("a.b.Shape = s\n")), 1)

    def test_one_warning_per_line(self):
        self.assertEqual(len(self.warn("a.Shape = b.Shape = s\n")), 1)

    def test_other_attributes_and_code_that_does_not_parse(self):
        self.assertEqual(self.warn("a.Placement = p\nShape = 1\n"), [])
        self.assertEqual(self.warn("def broken(:\n"), [])
        self.assertEqual(self.warn("\x00"), [])
        self.assertEqual(self.warn(""), [])


# ---------------------------------------------------------------------------------------------- background jobs


class JobsTest(ScriptCase):
    def setUp(self):
        super().setUp()
        # A job's commit() dispatches from its worker thread. Another test module may have created a QApplication, which
        # would make the real dispatcher queue the task for a GUI loop that is not running here: run it inline, as in a
        # console session.
        patcher = mock.patch.object(dispatch, "_run_inline", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def wait(self, job_id, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = scripting.job_status(job_id)["job"]
            if job["state"] != "running":
                return job
            time.sleep(0.02)
        self.fail("job did not finish")

    def test_job_runs_in_a_fresh_namespace_with_commit(self):
        scripting.register_helpers("zz_helper", 5)
        started = scripting.start_job("state['v'] = commit(lambda: zz_helper * 2)\nstate['name'] = __name__")
        self.assertTrue(started["success"])
        self.assertIn("NOT complete yet", started["message"])
        job = self.wait(started["job_id"])
        self.assertEqual(job["state"], "done")
        self.assertEqual(scripting.STATE["v"], 10)
        self.assertEqual(scripting.STATE["name"], "__main__")

    def test_failed_job_records_error_and_traceback(self):
        job = self.wait(scripting.start_job("raise ValueError('boom')")["job_id"])
        self.assertEqual(job["state"], "failed")
        self.assertEqual(job["error"], "ValueError: boom")
        self.assertIn("Traceback", job["traceback"])

    def test_system_exit_in_a_job_still_finishes_the_record(self):
        job = self.wait(scripting.start_job("raise SystemExit(3)")["job_id"])
        self.assertEqual(job["state"], "failed")

    def test_commit_is_only_available_inside_a_job(self):
        with self.assertRaises(RuntimeError):
            scripting.commit(lambda: 1)

    def test_a_failing_commit_raises_inside_the_job(self):
        job = self.wait(scripting.start_job("commit(lambda: 1 / 0)")["job_id"])
        self.assertEqual(job["state"], "failed")
        self.assertIn("commit failed", job["error"])

    def test_status_of_unknown_and_all_jobs(self):
        self.assertFalse(scripting.job_status("job-nope")["success"])
        job_id = scripting.start_job("pass")["job_id"]
        self.wait(job_id)
        self.assertIn(job_id, [j["id"] for j in scripting.job_status()["jobs"]])

    def test_dangerous_patterns_in_async_code_are_reported(self):
        self.assertEqual(scripting.async_code_warnings("x = FreeCAD.ActiveDocument"), ["ActiveDocument"])
        self.assertEqual(scripting.async_code_warnings("x = 1"), [])
        r = scripting.start_job("pass  # ActiveDocument")
        self.assertIn("WARNING: this code appears to touch documents/GUI directly", r["message"])
        self.wait(r["job_id"])

    def test_finished_jobs_are_pruned(self):
        ids = [scripting.start_job("pass")["job_id"] for _ in range(scripting._KEEP_FINISHED + 6)]
        for i in ids:
            if i in scripting.JOBS:
                self.wait(i)
        scripting._record("job-prune-trigger", state="done")
        finished = [j for j in scripting.JOBS.values() if j.get("state") in ("done", "failed")]
        self.assertLessEqual(len(finished), scripting._KEEP_FINISHED)


# ------------------------------------------------------------------------------------- network audit hook (netguard)


class _StandInGuiThread:
    """A GUI-thread stand-in started by the TEST (never by a script, so it does not inherit a script's guard):
    ``dispatch(fn, ...)`` runs ``fn`` there and returns ``(result, failed)`` like ``cad.dispatch.dispatch``."""

    def __init__(self):
        import queue
        import threading

        self.ops: list[str] = []
        self._jobs: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="stand-in-gui", daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            item = self._jobs.get()
            if item is None:
                return
            fn, box, done = item
            try:
                box["result"] = fn()
            except BaseException as exc:  # noqa: BLE001 - reported like the dispatcher does
                box["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                done.set()

    def dispatch(self, fn, timeout=60.0, op="", **_kw):
        import threading

        self.ops.append(op)
        box, done = {}, threading.Event()
        self._jobs.put((fn, box, done))
        if not done.wait(timeout):
            return {"success": False, "error": "timed out"}, True
        if "error" in box:
            return box["error"], True
        return box.get("result"), False

    def close(self):
        self._jobs.put(None)
        self._thread.join(5)


class NetworkAuditTest(ScriptCase):
    """``run_script`` executes user code inside ``netguard.script_guard``: network use needs the turn's grant."""

    def setUp(self):
        super().setUp()
        from oscar import netguard
        self.netguard = netguard
        netguard.reset_turn()
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("OSCAR_ALLOW_NET", None)
        os.environ.pop("OSCAR_NET_AUDIT", None)
        self.addCleanup(netguard.reset_turn)

    SCRIPTS = {
        "socket": "import socket\nsocket.create_connection(('203.0.113.9', 9), timeout=0.3)",
        "urllib": "import urllib.request\nurllib.request.urlopen('http://203.0.113.9/x', timeout=0.3)",
        "requests": "import requests\nrequests.get('http://203.0.113.9/x', timeout=0.3)",
        "dynamic_import": "s = __import__('soc' + 'ket')\ns.create_connection(('203.0.113.9', 9), timeout=0.3)",
        "curl": "import subprocess\nsubprocess.run(['curl', '-s', 'http://203.0.113.9/x'])",
        # P0 #2: the worker thread runs only library code (no script frame): it is guarded because the script started it
        "thread": ("import socket\nfrom concurrent.futures import ThreadPoolExecutor\n"
                   "with ThreadPoolExecutor(max_workers=1) as pool:\n"
                   "    pool.submit(socket.create_connection, ('203.0.113.9', 9), 0.3).result()"),
        "dns": "import socket\nsocket.gethostbyname('example.com')",
    }

    def test_a_background_jobs_commit_callback_is_guarded_on_the_gui_thread(self):
        """``commit(fn)`` runs ``fn`` on the GUI thread, which is not the job's (guarded) thread; the callback here is
        library code, so only the guard ``commit`` puts around it can stop it."""
        gui = _StandInGuiThread()
        self.addCleanup(gui.close)
        with mock.patch.object(dispatch, "_run_inline", return_value=False), \
                mock.patch.object(scripting.dispatch, "dispatch", gui.dispatch):
            job_id = scripting.start_job(
                "import functools, socket\n"
                "commit(functools.partial(socket.create_connection, ('203.0.113.9', 9), 0.3))")["job_id"]
            deadline = time.monotonic() + 20
            while scripting.job_status(job_id)["job"]["state"] == "running" and time.monotonic() < deadline:
                time.sleep(0.02)
        job = scripting.job_status(job_id)["job"]
        self.assertEqual(job["state"], "failed", job)
        self.assertIn("network access blocked", job["error"])
        self.assertEqual(gui.ops, ["commit"])

    def test_blocked_without_a_grant(self):
        for name, code in self.SCRIPTS.items():
            with self.subTest(name):
                result = self.run_fail(code)
                self.assertIn("network access blocked", result["error"])

    def test_allowed_with_a_grant_the_attempt_proceeds(self):
        self.netguard._TURN_DECISION["download"] = "allow"
        for name in ("socket", "dynamic_import"):
            with self.subTest(name):
                result = scripting.run_script(self.SCRIPTS[name])
                self.assertNotIn("network access blocked", str(result.get("error", "")))

    def test_loopback_and_plain_scripts_are_untouched(self):
        self.run_ok("import socket\ns = socket.socket()\ns.settimeout(0.3)\ntry:\n    s.connect(('127.0.0.1', 9))\n"
                    "except OSError:\n    pass\nfinally:\n    s.close()")
        self.run_ok("x = 1 + 1")

    def test_kill_switch_disables_the_audit(self):
        os.environ["OSCAR_NET_AUDIT"] = "0"
        result = scripting.run_script(self.SCRIPTS["socket"])
        self.assertNotIn("network access blocked", str(result.get("error", "")))

    def test_guard_is_released_after_the_script(self):
        self.run_fail(self.SCRIPTS["socket"])
        self.assertEqual(getattr(self.netguard._AUDIT_TLS, "depth", 0), 0)
        import socket
        with self.assertRaises(OSError) as ctx:      # outside run_script the hook is inert (a plain network error)
            socket.create_connection(("203.0.113.9", 9), timeout=0.2)
        self.assertNotIsInstance(ctx.exception, PermissionError)


if __name__ == "__main__":
    unittest.main()
