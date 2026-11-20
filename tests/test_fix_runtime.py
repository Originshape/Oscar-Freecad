"""Regression tests for the runtime review fixes (cad-runtime-*, tools-07/08/14,
build-docs-02).

Scripted execution (scripting.py), the headless launcher (headless.py), view
capture (views.py) and the tool argument layer (tools.py). Runs in the console
interpreter: no GUI, no network; the headless tests spawn the real freecadcmd
child and skip when it is absent.
"""

import os
import subprocess
import sys
import tempfile
import types
import unittest
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

import FreeCAD  # noqa: E402

from oscar import tools  # noqa: E402
from oscar.cad import headless, scripting  # noqa: E402


class ScriptExitTest(unittest.TestCase):
    """cad-runtime-01: a script calling sys.exit()/exit() must not take the
    application (or the agent thread) down with it."""

    def test_nonzero_exit_is_an_error_result_not_an_exception(self):
        try:
            res = scripting.run_script("print('before')\nimport sys\nsys.exit(3)")
        except BaseException as e:  # noqa: BLE001 - this is the defect
            self.fail(f"SystemExit escaped run_script: {e!r}")
        self.assertFalse(res["success"])
        self.assertIn("SystemExit", res["error"])
        self.assertIn("before", res["error"])  # output printed so far is kept

    def test_exit_with_message_is_an_error(self):
        res = scripting.run_script("import sys\nsys.exit('bad input')")
        self.assertFalse(res["success"])
        self.assertIn("bad input", res["error"])

    def test_clean_exit_finishes_the_script_normally(self):
        for code in ("print('done')\nexit()", "import sys\nsys.exit(0)",
                     "raise SystemExit"):
            with self.subTest(code=code):
                try:
                    res = scripting.run_script(code)
                except BaseException as e:  # noqa: BLE001
                    self.fail(f"SystemExit escaped run_script: {e!r}")
                self.assertTrue(res["success"], res)

    def test_commit_callback_cannot_raise_systemexit_on_the_gui_thread(self):
        # Drive the real GUI-thread drain loop: whatever the callback does,
        # nothing may escape it, and the script gets a clear message back.
        def drain_dispatch(fn, timeout=60.0, op=""):
            request = scripting.dispatch._Request(fn)
            scripting.dispatch._pending.put(request)
            scripting.dispatch.drain_queue()  # a SystemExit escaping here fails the test
            return request.result, request.failed

        scripting._in_worker.active = True
        try:
            with mock.patch.object(scripting.dispatch, "dispatch", drain_dispatch):
                with self.assertRaises(RuntimeError) as ctx:
                    scripting.commit(lambda: sys.exit(1))
        finally:
            del scripting._in_worker.active
        self.assertIn("exit(1) inside commit()", str(ctx.exception))


class ScriptNamespaceTest(unittest.TestCase):
    """cad-runtime-02: scripts run as __main__, so the common
    ``if __name__ == '__main__': main()`` idiom executes."""

    def test_main_guard_runs(self):
        res = scripting.run_script(
            "def main():\n    print('main ran')\n"
            "if __name__ == '__main__':\n    main()\n"
        )
        self.assertTrue(res["success"], res)
        self.assertIn("main ran", res["message"])


class PlanValidationMessageTest(unittest.TestCase):
    """tools-07: a plan rejected at the validation stage carries its reasons
    in ``errors`` (no ``error`` key) - the tools used to print 'None'."""

    def test_apply_feature_plan_reports_validation_errors(self):
        doc = FreeCAD.newDocument("PlanMsgDoc")
        self.addCleanup(FreeCAD.closeDocument, doc.Name)
        text, _image, is_error = tools.run_tool(
            "apply_feature_plan", {"doc_name": doc.Name, "plan": {"schema_version": 2}}
        )
        self.assertTrue(is_error)
        self.assertIn("pass the WHOLE plan object", text)
        self.assertNotIn(": None", text)

    def test_reapply_plan_feature_reports_validation_errors(self):
        doc = FreeCAD.newDocument("FeatMsgDoc")
        self.addCleanup(FreeCAD.closeDocument, doc.Name)
        text, _image, is_error = tools.run_tool(
            "reapply_plan_feature",
            {"doc_name": doc.Name, "plan": {"schema_version": 2}, "feature_id": "base_pad"},
        )
        self.assertTrue(is_error)
        self.assertIn("pass the WHOLE plan object", text)
        self.assertNotIn(": None", text)


class ConsoleLookupTest(unittest.TestCase):
    """The engine finds FreeCAD's console binary next to the application, wherever its resources are."""

    def _layout(self, name):
        root = tempfile.mkdtemp(prefix="oscar_console_")
        os.makedirs(os.path.join(root, "bin"))
        exe = os.path.join(root, "bin", name)
        open(exe, "w").close()
        return root, exe


    def test_home_bin_is_searched_when_resource_dir_is_elsewhere(self):
        # macOS: getResourceDir() is .../Resources/share/, the binary sits
        # in getHomePath()/bin.
        root, exe = self._layout("freecadcmd")
        elsewhere = tempfile.mkdtemp(prefix="oscar_console_share_")
        with mock.patch.object(FreeCAD, "getResourceDir", return_value=elsewhere), \
                mock.patch.object(FreeCAD, "getHomePath", return_value=root), \
                mock.patch("shutil.which", return_value=None):
            self.assertEqual(headless.find_console(), [exe])


@unittest.skipIf(os.name == "nt", "POSIX process semantics")
class HeadlessProcessTest(unittest.TestCase):
    """cad-runtime-12: temp script files are removed and a timeout kills the
    engine's whole process group, not just the engine."""

    def _home(self):
        # The helper's script directory follows OSCAR_CACHE_DIR (paths.cache_dir) rather than Path.home().
        cache = tempfile.mkdtemp(prefix="oscar_headless_cache_")
        patcher = mock.patch.dict(os.environ, {"OSCAR_CACHE_DIR": cache})
        patcher.start()
        self.addCleanup(patcher.stop)
        return os.path.join(cache, "headless")

    @staticmethod
    def _scripts(directory):
        # The constant bootstrap file stays in the directory by design; only per-run scripts must be removed.
        return [n for n in os.listdir(directory) if n != headless._BOOT_FILE]

    def test_script_file_is_deleted_after_a_run(self):
        directory = self._home()
        with mock.patch.object(headless, "find_console", return_value=["/usr/bin/true"]):
            res = headless.run("print('x')", timeout=20)
        self.assertTrue(res["success"], res)
        self.assertEqual(self._scripts(directory), [])

    def test_script_file_is_deleted_after_a_timeout(self):
        directory = self._home()
        with mock.patch.object(headless, "find_console", return_value=["/bin/sh", "-c", "sleep 30", "engine"]):
            res = headless.run("print('x')", timeout=0.3)
        self.assertTrue(res.get("timed_out"), res)
        self.assertEqual(self._scripts(directory), [])

    def test_timeout_kills_grandchildren(self):
        import signal
        import time

        self._home()
        pid_file = os.path.join(tempfile.mkdtemp(prefix="oscar_headless_pid_"), "pid")
        # sh -c '<script>' $0 -c exec(...): the appended "-c exec(...)" are
        # just positional parameters, so this stands in for the engine.
        fake = ["/bin/sh", "-c", f"sleep 47 & echo $! > {pid_file}; sleep 47", "engine"]
        with mock.patch.object(headless, "find_console", return_value=fake):
            res = headless.run("print('x')", timeout=0.5)
        self.assertTrue(res.get("timed_out"), res)
        with open(pid_file) as f:
            child = int(f.read().strip())
        alive = True
        try:
            for _ in range(40):
                try:
                    os.kill(child, 0)
                except ProcessLookupError:
                    alive = False
                    break
                time.sleep(0.05)
        finally:
            if alive:
                os.kill(child, signal.SIGKILL)
        self.assertFalse(alive, "the grandchild outlived the timeout")

    def test_any_failure_while_waiting_kills_the_engine(self):
        # runtime-tools-01: only TimeoutExpired used to be handled, so any
        # other exception between Popen() and a finished communicate()
        # (OverflowError for a huge timeout, KeyboardInterrupt, ...) left the
        # engine running unseen.
        import signal

        directory = self._home()
        spawned = []
        real_popen = subprocess.Popen

        def spy(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            spawned.append(proc)
            return proc

        fake = ["/bin/sh", "-c", "exec sleep 47", "engine"]
        with mock.patch.object(headless, "find_console", return_value=fake), \
                mock.patch.object(headless.subprocess, "Popen", side_effect=spy), \
                mock.patch.object(real_popen, "communicate",
                                  side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                headless.run("print('x')", timeout=60)
        (engine,) = spawned
        try:
            self.assertIsNotNone(engine.poll(), "the engine outlived the failed wait")
        finally:
            if engine.poll() is None:
                os.killpg(engine.pid, signal.SIGKILL)
                engine.wait()
        self.assertEqual(self._scripts(directory), [])

    def test_unbounded_timeout_does_not_blow_up(self):
        # A model passing 99999999 to mean "no limit" used to hit
        # OverflowError inside communicate().
        self._home()
        with mock.patch.object(headless, "find_console", return_value=["/usr/bin/true"]):
            res = headless.run("print('x')", timeout=99999999)
        self.assertTrue(res["success"], res)


class HeadlessTimeoutClampTest(unittest.TestCase):
    """runtime-tools-01: an absurd timeout is clamped before it reaches
    communicate(), on every platform."""

    def test_huge_timeout_is_clamped(self):
        home = tempfile.mkdtemp(prefix="oscar_headless_home_")
        seen = {}

        def fake_run_process(argv, timeout, **_kw):   # env / code: extra arguments of the cache-dir based launcher
            seen["timeout"] = timeout
            return {"success": True}

        with mock.patch.dict(os.environ, {"OSCAR_CACHE_DIR": home}), \
                mock.patch.object(headless, "find_console", return_value=["engine"]), \
                mock.patch.object(headless, "_run_process", side_effect=fake_run_process):
            headless.run("print('x')", timeout=1e9)
        self.assertLessEqual(seen["timeout"], 86400)
        self.assertGreater(seen["timeout"], 0)


class _RealEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if headless.find_console() is None:
            raise unittest.SkipTest("no headless engine binary available")


class HeadlessOutputTest(_RealEngine):
    """cad-runtime-05: output printed before a failure survives, and
    non-ASCII output works in a Finder-like (no locale) environment."""

    def test_output_before_an_exception_is_preserved(self):
        res = headless.run("print('progress 1')\n1/0", timeout=120)
        self.assertFalse(res["success"], res)
        self.assertIn("progress 1", res["output"])
        # the exception is reported in the model-readable ``error`` (formatted like an execute_code error), not as
        # a raw traceback appended to the output
        self.assertIn("ZeroDivisionError", res["error"])

    def test_output_before_a_crash_is_preserved(self):
        res = headless.run(
            "print('progress 2')\nimport os, signal\nos.kill(os.getpid(), signal.SIGSEGV)",
            timeout=120,
        )
        self.assertFalse(res["success"], res)
        self.assertIn("progress 2", res["output"])

    def test_non_ascii_output_without_a_locale(self):
        # Strip only the locale/encoding variables: an allow-list of
        # PATH/HOME starves a Windows child of SYSTEMROOT/TEMP/USERPROFILE.
        env = {k: v for k, v in os.environ.items()
               if k not in ("LANG", "LANGUAGE", "PYTHONUTF8", "PYTHONIOENCODING")
               and not k.startswith("LC_")}
        with mock.patch.dict(os.environ, env, clear=True):
            res = headless.run("print('Ø10 ✓')", timeout=120)
        self.assertTrue(res["success"], res)
        self.assertIn("Ø10 ✓", res["output"])


class HeadlessFailureReportTest(_RealEngine):
    """cad-runtime-04 (+ tools-13): a failed run says why instead of
    'unknown error'."""

    def test_nonzero_exit_has_an_error(self):
        res = headless.run("import sys\nsys.exit(3)", timeout=120)
        self.assertFalse(res["success"])
        self.assertIn("3", res["error"])

    @unittest.skipIf(os.name == "nt", "POSIX signals")
    def test_native_crash_is_reported_as_a_crash(self):
        res = headless.run("import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)", timeout=120)
        self.assertTrue(res["crashed"], res)
        self.assertIn("crash", res["error"].lower())

    def test_tool_message_is_not_unknown_error(self):
        text, _image, is_error = tools.run_tool(
            "execute_code_headless", {"code": "import sys\nsys.exit(3)"}
        )
        self.assertTrue(is_error)
        self.assertNotIn("unknown error", text)
        self.assertIn("exit", text.lower())
        self.assertIn("3", text)


class HeadlessCleanTest(unittest.TestCase):
    """cad-runtime-11: only real engine noise is filtered from the output."""

    def test_script_output_that_mentions_noise_words_survives(self):
        cleaned = headless._clean(
            "FreeCAD objects created: 3\n"
            "Volume: 1000 mm3 (95.5%) of target\n"
            "LGPL part library loaded\n"
            "plain"
        )
        self.assertEqual(cleaned.splitlines(), [
            "FreeCAD objects created: 3",
            "Volume: 1000 mm3 (95.5%) of target",
            "LGPL part library loaded",
            "plain",
        ])

    def test_engine_noise_is_removed(self):
        cleaned = headless._clean(
            "FreeCAD 1.1.3, Libs: 1.1.3\n"
            "(C) 2001-2026 FreeCAD contributors\n"
            "Recompute......\n"
            "\t\t\t\t\t\t(100 %)\t\n"
            "\t\t(7 %)\t\t\t(15 %)\t\t\t(100 %)\t\t\t\tRecompute......\n"
            "\n"
            "saved"
        )
        self.assertEqual(cleaned, "saved")







class ToolArgumentTest(unittest.TestCase):
    """tools-14: explicit nulls, enum values and JSON-string arrays."""

    def _prepare(self, name, arguments):
        entry = {t.name: t for t in tools.get_tools()}[name]
        return tools._prepare_arguments(entry, arguments)

    def test_null_optional_parameters_fall_back_to_the_handler_default(self):
        prepared = self._prepare(
            "get_objects", {"doc_name": "D", "limit": None}
        )
        self.assertNotIn("limit", prepared)
        prepared = self._prepare("get_view", {"view_name": "", "focus_object": None})
        self.assertEqual(prepared, {})  # "" on an enum is "not given"

    def test_null_required_parameter_is_still_an_error(self):
        with self.assertRaises(tools.ToolError) as ctx:
            self._prepare("get_objects", {"doc_name": None})
        self.assertIn("missing required", str(ctx.exception))




    def test_view_name_is_resolved_not_rejected(self):
        # runtime-tools-02: view_name is cosmetic (screenshot orientation); a
        # misspelling must never refuse a geometry-changing call.
        for given, expected in (
            ("Iso", "Isometric"), ("iso", "Isometric"), ("Isometric view", "Isometric"),
            ("axonometric", "Isometric"), ("Rear", "Back"), ("rear view", "Back"),
            ("  FRONT  ", "Front"), ("top-view", "Top"), ("Bottom", "Bottom"),
            ("nonsense", "Isometric"), ("view", "Isometric"), (7, "Isometric"),
        ):
            with self.subTest(view_name=given):
                prepared = self._prepare("get_view", {"view_name": given})
                self.assertEqual(prepared["view_name"], expected)
        # ... on every tool that carries the parameter
        filler = {"string": "x", "object": {}, "array": [], "integer": 1,
                  "number": 1.0, "boolean": True}
        for entry in tools.get_tools():
            properties = entry.parameters.get("properties", {})
            if "view_name" not in properties:
                continue
            with self.subTest(tool=entry.name):
                # (a required number may carry a minimum)
                arguments = {key: max(filler[properties[key].get("type", "string")], properties[key]["minimum"])
                             if "minimum" in properties[key]
                             else filler[properties[key].get("type", "string")]
                             for key in entry.parameters.get("required", [])}
                arguments["view_name"] = "Iso"
                self.assertEqual(self._prepare(entry.name, arguments)["view_name"], "Isometric")




class _ScratchDoc(unittest.TestCase):
    def setUp(self):
        self.doc = FreeCAD.newDocument("ViewNameDoc")
        self.addCleanup(FreeCAD.closeDocument, self.doc.Name)


class MutatingCallViewNameTest(_ScratchDoc):
    """runtime-tools-02: a cosmetic view_name must not fail the call."""

    def test_create_object_survives_any_view_name(self):
        for index, view_name in enumerate(("Iso", "Rear", "Isometric view", "nonsense")):
            with self.subTest(view_name=view_name):
                text, _image, is_error = tools.run_tool("create_object", {
                    "doc_name": self.doc.Name, "obj_type": "Part::Box",
                    "obj_name": f"B{index}", "view_name": view_name,
                })
                self.assertFalse(is_error, text)
                self.assertIsNotNone(self.doc.getObject(f"B{index}"))

    def test_the_resolved_name_reaches_the_screenshot(self):
        seen = []
        # the tools take pictures through _screenshot_with_reason (image, reason)
        with mock.patch.object(tools, "_screenshot_with_reason",
                               side_effect=lambda view, doc=None: seen.append(view) or (None, "")):
            text, _image, is_error = tools.run_tool("create_object", {
                "doc_name": self.doc.Name, "obj_type": "Part::Box",
                "obj_name": "Shot", "view_name": "Rear",
            })
        self.assertFalse(is_error, text)
        self.assertEqual(seen, ["Back"])

    def test_execute_code_survives_a_bad_view_name(self):
        text, _image, is_error = tools.run_tool("execute_code", {
            "code": "print('ran')", "view_name": "Rear", "include_screenshot": False,
        })
        self.assertFalse(is_error, text)
        self.assertIn("ran", text)


class _FakeEngine:
    """Canned engine results, to check what the tool layer does with them."""

    def __init__(self, **results):
        self.results = results

    def edit_object(self, *args):
        return self.results["edit_object"]

    def delete_object(self, *args, **kwargs):
        return self.results["delete_object"]

    def reload_document(self, *args, **kwargs):
        self.reload_kwargs = kwargs
        return self.results["reload_document"]


class EngineDetailsReachTheModelTest(unittest.TestCase):
    """engine-05: warnings and side effects the engine reports are part of
    the result the model reads."""

    def _run(self, name, arguments, **results):
        with mock.patch.object(tools, "_engine", return_value=_FakeEngine(**results)):
            return tools.run_tool(name, arguments)

    def test_edit_warning_is_reported(self):
        text, _image, is_error = self._run(
            "edit_object",
            {"doc_name": "D", "obj_name": "B", "obj_properties": {"Height": 1},
             "include_screenshot": False},
            edit_object={"success": True, "object_name": "B",
                         "warning": "'B' is still in an invalid state (Invalid)"},
        )
        self.assertFalse(is_error, text)
        self.assertIn("still in an invalid state", text)

    def test_plain_edit_has_no_warning_text(self):
        text, _image, _err = self._run(
            "edit_object",
            {"doc_name": "D", "obj_name": "B", "obj_properties": {"Height": 1},
             "include_screenshot": False},
            edit_object={"success": True, "object_name": "B"},
        )
        self.assertEqual(text, "Object 'B' edited.")

    def test_delete_reports_the_features_removed_with_a_body(self):
        text, _image, is_error = self._run(
            "delete_object",
            {"doc_name": "D", "obj_name": "Body", "include_screenshot": False},
            delete_object={"success": True, "object_name": "Body", "restored": [],
                           "removed_children": ["Pad", "Pocket"]},
        )
        self.assertFalse(is_error, text)
        self.assertIn("Pad", text)
        self.assertIn("Pocket", text)

    def test_reload_reports_discarded_changes(self):
        text, _image, is_error = self._run(
            "reload_document", {"doc_name": "D"},
            reload_document={"success": True, "document_name": "D",
                             "warning": "unsaved changes in the old copy were discarded"},
        )
        self.assertFalse(is_error, text)
        self.assertIn("unsaved changes", text)

    def test_reload_passes_discard_unsaved_through_and_defaults_to_safe(self):
        engine = _FakeEngine(reload_document={"success": True, "document_name": "D"})
        with mock.patch.object(tools, "_engine", return_value=engine):
            tools.run_tool("reload_document", {"doc_name": "D"})
            self.assertEqual(engine.reload_kwargs, {"discard_unsaved": False})
            tools.run_tool("reload_document", {"doc_name": "D", "discard_unsaved": True})
            self.assertEqual(engine.reload_kwargs, {"discard_unsaved": True})


class _BusyEngine:
    """Every GUI read fails the way engine.dispatch reports a timeout."""

    def find_document(self, name):
        return name

    def list_documents(self):
        raise RuntimeError("list_documents failed: GUI thread busy")

    def get_objects(self, doc_name):
        raise RuntimeError("get_objects failed: GUI thread busy")

    def _object_page(self, doc_name, **_kwargs):   # what the get_objects TOOL reads (the paged table)
        raise RuntimeError("get_objects failed: GUI thread busy")

    def get_object(self, doc_name, obj_name, detail=False):
        raise RuntimeError("get_object failed: GUI thread busy")


class EngineReadFailureTest(unittest.TestCase):
    """engine-06: a failed GUI read is an error, not 'empty' or 'not found'."""

    def _run(self, name, arguments):
        with mock.patch.object(tools, "_engine", return_value=_BusyEngine()):
            return tools.run_tool(name, arguments)

    def test_get_objects_failure_is_not_reported_as_an_empty_document(self):
        text, _image, is_error = self._run("get_objects", {"doc_name": "D"})
        self.assertTrue(is_error)
        self.assertIn("GUI thread", text)
        self.assertNotIn("empty", text)

    def test_get_object_failure_is_not_reported_as_not_found(self):
        text, _image, is_error = self._run("get_object", {"doc_name": "D", "obj_name": "B"})
        self.assertTrue(is_error)
        self.assertIn("GUI thread", text)
        self.assertNotIn("not found", text)

    def test_unknown_document_check_survives_a_failed_listing(self):
        class _NoDoc(_BusyEngine):
            def find_document(self, name):
                return None

        with mock.patch.object(tools, "_engine", return_value=_NoDoc()):
            text, _image, is_error = tools.run_tool("get_objects", {"doc_name": "D"})
        self.assertTrue(is_error)
        self.assertIn("GUI thread", text)


class _FakeViewObject:
    def __init__(self, visible):
        self.Visibility = visible


class _FakeObject:
    def __init__(self, name, visible):
        self.Name = name
        self.ViewObject = _FakeViewObject(visible)


class FocusCaptureTest(unittest.TestCase):
    """cad-runtime-14: capture(focus=...) must put a previously hidden target
    (typically the sketch under a Pad) back to hidden afterwards."""

    def _capture(self, objects):
        from oscar.cad import views

        document = types.SimpleNamespace(
            Objects=objects,
            getObject=lambda name: next((o for o in objects if o.Name == name), None),
        )
        seen = {}

        def save_image(path, width, height):
            seen.update({o.Name: o.ViewObject.Visibility for o in objects})

        view = types.SimpleNamespace(
            saveImage=save_image, getCamera=lambda: "cam", setCamera=lambda cam: None,
            getSize=lambda: (800, 600),
        )
        gui = types.SimpleNamespace(
            ActiveDocument=types.SimpleNamespace(Document=document, ActiveView=view),
            getDocument=lambda name: None, updateGui=lambda: None,
            runCommand=lambda name: None,
        )
        # A console session has no 3D view (capture is gated on App.GuiUp) and a blank frame is no picture (the fake
        # saveImage writes no file to inspect): stand in for a GUI whose frame has content.
        with mock.patch.object(views, "Gui", gui), \
                mock.patch.object(views.App, "GuiUp", True), \
                mock.patch.object(views, "is_blank_image", return_value=False):
            ok = views.capture(os.path.join(tempfile.mkdtemp(), "shot.png"), focus="Sketch")
        self.assertTrue(ok)
        return seen, {o.Name: o.ViewObject.Visibility for o in objects}

    def test_hidden_target_is_hidden_again_and_others_are_restored(self):
        objects = [_FakeObject("Sketch", False), _FakeObject("Pad", True), _FakeObject("Other", True)]
        during, after = self._capture(objects)
        self.assertEqual(during, {"Sketch": True, "Pad": False, "Other": False})
        self.assertEqual(after, {"Sketch": False, "Pad": True, "Other": True})

    def test_visible_target_stays_visible(self):
        objects = [_FakeObject("Sketch", True), _FakeObject("Pad", True)]
        _during, after = self._capture(objects)
        self.assertEqual(after, {"Sketch": True, "Pad": True})


class ScriptErrorLineTest(unittest.TestCase):
    """cad-runtime-10: the failing line is looked up in the script that
    defined it, not in whatever script happens to be running now."""

    def test_error_in_a_helper_from_an_earlier_call_names_the_helper_line(self):
        # Every run has a FRESH namespace (deliberate): a helper reaches a later call through the persistent
        # ``state`` dict, which is the way scripts carry things between runs.
        self.addCleanup(scripting.reset_state)
        first = scripting.run_script(
            "import math\ndef helper(area):\n    x = 1\n    return math.sqrt(area / 0.0)\nstate['helper'] = helper\n"
        )
        self.assertTrue(first["success"], first)
        res = scripting.run_script(
            "print('step 1')\nprint('step 2')\nprint('step 3')\nprint('step 4')\nstate['helper'](10)"
        )
        self.assertFalse(res["success"])
        self.assertIn("ZeroDivisionError", res["error"])
        self.assertIn("math.sqrt(area / 0.0)", res["error"])
        self.assertNotIn("print('step 4')", res["error"])

    def test_plain_error_still_names_its_own_line(self):
        res = scripting.run_script("x = 1\ny = 2\nz = x / 0\nprint('never')")
        self.assertIn("at script line 3: z = x / 0", res["error"])

    def test_syntax_error_still_names_its_own_line(self):
        res = scripting.run_script("a = 1\nb = (\nc = 3")
        self.assertFalse(res["success"])
        self.assertIn("SyntaxError", res["error"])
        self.assertIn("at script line", res["error"])


class CommitTransactionTest(unittest.TestCase):
    """cad-runtime-07: document writes a background job applies through
    commit() must enter the undo stack, or "revert turn" leaves them behind."""

    def _wait_for(self, job_id):
        import time

        for _ in range(250):
            job = scripting.job_status(job_id)["job"]
            if job["state"] != "running":
                return job
            time.sleep(0.02)
        self.fail("background job did not finish")

    def test_commit_writes_are_reverted_with_the_turn(self):
        from oscar.cad import engine

        doc = FreeCAD.newDocument("CommitUndoDoc")
        self.addCleanup(FreeCAD.closeDocument, doc.Name)
        FreeCAD.setActiveDocument(doc.Name)
        doc.UndoMode = 1
        doc.addObject("Part::Box", "Base")
        doc.recompute()
        eng = engine.CadEngine()
        snapshot = eng.begin_turn_snapshot()

        # Other modules may have started a QApplication, which would queue
        # commit() for a GUI loop nobody pumps here: run the dispatch inline.
        from oscar.cad import dispatch

        with mock.patch.object(
            scripting.dispatch, "dispatch", lambda fn, timeout=60.0, op="": dispatch._inline(fn)
        ):
            job = scripting.start_job(
                "def apply():\n"
                "    import FreeCAD\n"
                "    FreeCAD.ActiveDocument.addObject('Part::Box', 'FromCommit')\n"
                "commit(apply)\n"
            )
            self.assertEqual(self._wait_for(job["job_id"])["state"], "done")
        self.assertIsNotNone(doc.getObject("FromCommit"))

        eng.revert_turn(snapshot)
        self.assertIsNone(doc.getObject("FromCommit"))
        self.assertIsNotNone(doc.getObject("Base"))


if __name__ == "__main__":
    unittest.main()
