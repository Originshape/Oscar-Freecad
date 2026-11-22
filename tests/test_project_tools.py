"""The four Project tools (oscar/tools_project.py): registration against the matrix, the schemas of
6.6, and each handler through ``tools.run_tool`` with a bound Project (real FreeCAD documents under freecadcmd)."""

import json
import os
import re
import sys
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402

import project_support as S  # noqa: E402
from oscar import project as P  # noqa: E402
from oscar import project_gui as gui  # noqa: E402
from oscar import tools, tools_project  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402

ALL_PHASES = ("INTAKE", "SOURCE", "PLAN", "PARAMS", "BUILD", "VERIFY", "DELIVER")


def unique(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:6]}"


class RegistrationTest(unittest.TestCase):
    def meta(self, name):
        entry = tools.tool_meta(name)
        self.assertIsNotNone(entry, f"{name} is not registered")
        return entry

    def test_the_four_tools_are_registered_with_the_matrix_flags(self):
        expected = {
            "record_note": ("plan", False, False, ALL_PHASES),
            "save_document": ("deliver", True, False, ("BUILD", "VERIFY", "DELIVER")),
            "export_model": ("deliver", True, False, ("VERIFY", "DELIVER")),
            "export_rebuild_script": ("deliver", True, False, ("VERIFY", "DELIVER")),
        }
        for name, (category, mutating, geometry, phases) in expected.items():
            entry = self.meta(name)
            self.assertEqual((entry.category, entry.mutating, entry.geometry, tuple(entry.phases)),
                             (category, mutating, geometry, phases), name)

    def test_record_note_is_a_core_tool_and_not_a_step(self):
        self.assertIn("record_note", tools.CORE_TOOLS)
        self.assertFalse(tools.is_mutating("record_note"))
        self.assertTrue(all(tools.is_mutating(n) for n in ("save_document", "export_model", "export_rebuild_script")))
        self.assertFalse(any(tools.is_geometry(n) for n in ("record_note", "save_document", "export_model", "export_rebuild_script")))

    def test_schemas_follow_the_contract(self):
        note = self.meta("record_note").parameters
        self.assertEqual(note["required"], ["kind", "text"])
        self.assertEqual(note["properties"]["kind"]["enum"], ["plan", "assumption", "provenance", "validation",
                                                              "unresolved", "decision"])
        self.assertEqual(note["properties"]["refs"]["items"], {"type": "string"})
        save = self.meta("save_document").parameters
        self.assertEqual((save["required"], sorted(save["properties"])), (["doc_name"], ["doc_name", "path"]))
        export = self.meta("export_model").parameters
        self.assertEqual(export["required"], ["doc_name"])
        self.assertEqual(sorted(export["properties"]), ["directory", "doc_name", "formats", "objects", "per_part",
                                                        "print_orientation", "stl_tolerance"])
        props = export["properties"]
        self.assertEqual(props["formats"]["items"]["enum"], ["step", "stl", "3mf", "brep"])
        self.assertEqual(props["formats"]["default"], ["step", "stl"])
        self.assertEqual((props["per_part"]["default"], props["print_orientation"]["default"]), (True, True))
        self.assertEqual((props["stl_tolerance"]["default"], props["stl_tolerance"]["minimum"]), (0.05, 0.001))
        rebuild = self.meta("export_rebuild_script").parameters
        self.assertNotIn("required", rebuild)
        self.assertEqual(sorted(rebuild["properties"]), ["doc_name", "run_check"])
        self.assertEqual(rebuild["properties"]["run_check"]["default"], False)

    def test_schemas_use_no_union_keywords(self):
        for name in ("record_note", "save_document", "export_model", "export_rebuild_script"):
            text = json.dumps(self.meta(name).parameters)
            for banned in ("oneOf", "anyOf", "allOf", "$ref"):
                self.assertNotIn(banned, text, name)

    def test_mutating_tools_get_the_universal_step_title(self):
        specs = {s.name: s for s in tools.specs(names=["record_note", "save_document", "export_model", "export_rebuild_script"])}
        self.assertEqual(sorted(specs), ["export_model", "export_rebuild_script", "record_note", "save_document"])
        self.assertNotIn("step_title", specs["record_note"].parameters["properties"])
        for name in ("save_document", "export_model", "export_rebuild_script"):
            self.assertIn("step_title", specs[name].parameters["properties"], name)

    def test_descriptions_are_the_model_facing_text_of_the_contract(self):
        self.assertTrue(self.meta("record_note").description.startswith("Write a note into the project's NOTES.md"))
        self.assertTrue(self.meta("save_document").description.startswith("Save a document to disk as an .FCStd file."))
        self.assertTrue(self.meta("export_model").description.startswith("Export the model for manufacturing: STEP / STL / 3MF / BREP"))
        self.assertTrue(self.meta("export_rebuild_script").description.startswith("Write rebuild.py into the project"))

    def test_row_titles(self):
        self.assertEqual(tools.derive_title("export_model", {"doc_name": "D", "formats": ["step", "stl"]}), "Export step+stl")
        self.assertEqual(tools.derive_title("export_model", {"doc_name": "D"}), "Export step+stl")
        self.assertEqual(tools.derive_title("save_document", {"doc_name": "PiCase"}), "Save PiCase")


class ToolCase(unittest.TestCase):
    """A bound Project, a context and a ``call`` helper that goes through ``run_tool``."""

    def setUp(self):
        self._before = set(App.listDocuments())
        self.proj = S.make_project(chat_id=f"20261001-000000-{uuid.uuid4().hex[:6]}")
        self.ctx = ToolContext(project=self.proj, turn_id="T20261001-000001-0a1b")
        self.proj.begin_turn(self.ctx.turn_id, "do things", work=True)
        self.n = 0

    def tearDown(self):
        S.close_docs([n for n in App.listDocuments() if n not in self._before])

    def call(self, tool_name, /, ctx=None, **args):
        """``run_tool(tool_name, args)`` inside the bound context (``ctx=`` swaps the context)."""
        self.n += 1
        context = ctx or self.ctx
        context.call_id = f"c{self.n}"
        with use_context(context):
            return tools.run_tool(tool_name, args)

    def solid_doc(self):
        doc = App.newDocument(unique("T3T"))
        box = doc.addObject("Part::Box", "Box")
        box.Length, box.Width, box.Height = 20, 10, 5
        cyl = doc.addObject("Part::Cylinder", "Cyl")
        cyl.Radius, cyl.Height = 2, 10
        cyl.Placement.Base = App.Vector(5, 5, -1)
        cut = doc.addObject("Part::Cut", "Cut")
        cut.Base, cut.Tool = box, cyl
        doc.recompute()
        return doc


class RecordNoteToolTest(ToolCase):
    def test_a_note_is_written_to_the_project(self):
        text, image, is_error = self.call("record_note", kind="assumption", text="Wall thickness 2 mm assumed.", refs=["S5", "R2"])
        self.assertFalse(is_error, text)
        self.assertIsNone(image)
        self.assertIn("Note N1 recorded (assumption)", text)
        note = self.proj.read_notes()[0]
        self.assertEqual((note["kind"], note["text"], note["refs"]), ("assumption", "Wall thickness 2 mm assumed.", ["S5", "R2"]))
        self.assertIn("Wall thickness 2 mm assumed.", open(self.proj.path("NOTES.md"), encoding="utf-8").read())

    def test_it_is_not_a_step_record(self):
        self.call("record_note", kind="decision", text="Use a cap lid.")
        self.assertEqual(self.proj.read_steps(), [])

    def test_a_bad_kind_lists_the_accepted_values(self):
        text, _, is_error = self.call("record_note", kind="rumour", text="x")
        self.assertTrue(is_error)
        self.assertIn("kind", text)
        self.assertIn("assumption", text)
        self.assertEqual(self.proj.read_notes(), [])

    def test_empty_text_is_refused(self):
        text, _, is_error = self.call("record_note", kind="plan", text="   ")
        self.assertTrue(is_error)
        self.assertIn("'text' is empty", text)
        text, _, is_error = self.call("record_note", kind="plan", text="")
        self.assertTrue(is_error)
        self.assertEqual(self.proj.read_notes(), [])

    def test_long_text_is_clipped_and_the_model_is_told(self):
        text, _, is_error = self.call("record_note", kind="unresolved", text="y" * 2500)
        self.assertFalse(is_error)
        self.assertIn("clipped to 2000 characters", text)
        self.assertEqual(len(self.proj.read_notes()[0]["text"]), 2000)

    def test_blank_refs_are_dropped(self):
        self.call("record_note", kind="plan", text="p", refs=["S1", " ", ""])
        self.assertEqual(self.proj.read_notes()[0]["refs"], ["S1"])

    def test_without_a_project_the_tool_says_so(self):
        text, _, is_error = self.call("record_note", ctx=ToolContext(), kind="plan", text="p")
        self.assertTrue(is_error)
        self.assertEqual(text, "record_note: no project is bound to this chat (open a chat in the Oscar panel)")


class SaveDocumentToolTest(ToolCase):
    def test_without_a_path_it_saves_into_the_project(self):
        doc = self.solid_doc()
        text, _, is_error = self.call("save_document", doc_name=doc.Name)
        self.assertFalse(is_error, text)
        self.assertIn(f"Saved '{doc.Name}' to {self.proj.path('model.FCStd')}", text)
        self.assertRegex(text, r"\(\d+(\.\d)? (B|KB|MB)\)")
        self.assertTrue(zipfile_ok(self.proj.path("model.FCStd")))
        self.assertEqual(self.proj.manifest["active_doc"], doc.Name)

    def test_the_call_is_a_step_record_that_is_not_replayed_and_takes_no_checkpoint(self):
        doc = self.solid_doc()
        self.call("save_document", doc_name=doc.Name)
        step = self.proj.read_steps()[0]
        self.assertEqual((step.tool, step.title, step.ok, step.geometry), ("save_document", f"Save {doc.Name}", True, False))
        self.assertEqual(step.replay["kind"], "none")
        self.assertEqual((step.checkpoint, step.snapshot), ("", ""))

    def test_a_second_document_goes_to_docs(self):
        first, second = self.solid_doc(), self.solid_doc()
        self.call("save_document", doc_name=first.Name)
        text, _, is_error = self.call("save_document", doc_name=second.Name)
        self.assertFalse(is_error, text)
        self.assertIn(os.path.join("docs", f"{second.Name}.FCStd"), text)

    def test_save_as_writes_a_copy_and_keeps_the_document_unchanged(self):
        doc = self.solid_doc()
        target = os.path.join(S.sandbox_dir(), "out", "My Case.FCStd")
        text, _, is_error = self.call("save_document", doc_name=doc.Name, path=target)
        self.assertFalse(is_error, text)
        self.assertIn(f"Saved '{doc.Name}' to {target}", text)
        self.assertTrue(zipfile_ok(target))
        self.assertEqual(sorted(os.listdir(os.path.dirname(target))), ["My Case.FCStd"])   # no temp or backup files
        self.assertFalse(os.path.exists(self.proj.path("model.FCStd")))   # an explicit save-as does not touch the project folder
        self.assertEqual(doc.FileName, "")                                # saveCopy: the open document keeps its association

    def test_save_as_overwrites_atomically(self):
        doc = self.solid_doc()
        target = os.path.join(S.sandbox_dir(), "a.FCStd")
        self.call("save_document", doc_name=doc.Name, path=target)
        doc.addObject("Part::Sphere", "Ball")
        doc.recompute()
        self.call("save_document", doc_name=doc.Name, path=target)
        self.assertEqual(sorted(os.listdir(os.path.dirname(target))), ["a.FCStd"])
        reopened = App.openDocument(target, True)
        try:
            self.assertIsNotNone(reopened.getObject("Ball"))
        finally:
            App.closeDocument(reopened.Name)

    def test_bad_paths_list_every_problem(self):
        doc = self.solid_doc()
        text, _, is_error = self.call("save_document", doc_name=doc.Name, path="relative/model.fcstd")
        self.assertTrue(is_error)
        self.assertIn("'path' must be absolute", text)
        self.assertIn("'path' must end in .FCStd", text)
        text, _, is_error = self.call("save_document", doc_name=doc.Name, path=os.path.join(S.sandbox_dir(), "model.step"))
        self.assertTrue(is_error)
        self.assertIn("must end in .FCStd", text)
        self.assertNotIn("absolute", text)

    def test_an_unknown_document_lists_the_open_ones(self):
        doc = self.solid_doc()
        text, _, is_error = self.call("save_document", doc_name="NoSuchDocument", path=os.path.join(S.sandbox_dir(), "x.FCStd"))
        self.assertTrue(is_error)
        self.assertIn("is not open", text)
        self.assertIn(doc.Name, text)
        text, _, is_error = self.call("save_document", doc_name="NoSuchDocument")
        self.assertTrue(is_error)
        self.assertIn("was not saved (no document)", text)
        self.assertIn("list_documents", text)

    def test_without_a_project_only_save_as_works(self):
        doc = self.solid_doc()
        bare = ToolContext()
        text, _, is_error = self.call("save_document", ctx=bare, doc_name=doc.Name)
        self.assertTrue(is_error)
        self.assertEqual(text, "save_document: no project is bound to this chat (open a chat in the Oscar panel)")
        target = os.path.join(S.sandbox_dir(), "bare.FCStd")
        text, _, is_error = self.call("save_document", ctx=bare, doc_name=doc.Name, path=target)
        self.assertFalse(is_error, text)
        self.assertTrue(zipfile_ok(target))

    def test_a_failing_save_is_reported_not_raised(self):
        doc = self.solid_doc()
        with mock.patch.object(gui, "save_copy", return_value=(False, "disk full")):
            text, _, is_error = self.call("save_document", doc_name=doc.Name, path=os.path.join(S.sandbox_dir(), "x.FCStd"))
        self.assertTrue(is_error)
        self.assertIn("disk full", text)
        self.assertIn("nothing was written", text)


class ExportModelToolTest(ToolCase):
    LINE = re.compile(r"^exports/[a-z0-9.-]+\.(step|stl|3mf|brep)  \d+(\.\d)? (B|KB|MB)  sha256 [0-9a-f]{12}  \(.+, \d+ solids?\)$")

    def test_the_result_has_one_line_per_file_in_the_contract_format(self):
        doc = self.solid_doc()
        text, _, is_error = self.call("export_model", doc_name=doc.Name)
        self.assertFalse(is_error, text)
        lines = text.split("\n")
        self.assertRegex(lines[0], self.LINE)
        self.assertRegex(lines[1], self.LINE)
        self.assertTrue(lines[0].endswith("(Cut, 1 solid)"), lines[0])
        self.assertEqual(sorted(l.split("  ")[0].rsplit(".", 1)[1] for l in lines[:2]), ["step", "stl"])
        self.assertTrue(lines[-1].startswith("Folder: "))
        for name in os.listdir(self.proj.path("exports")):
            self.assertTrue(os.path.isfile(self.proj.path("exports", name)))

    def test_sha256_in_the_result_is_the_files_hash(self):
        import hashlib

        doc = self.solid_doc()
        text, _, _ = self.call("export_model", doc_name=doc.Name, formats=["step"])
        first = text.split("\n")[0]
        short = re.search(r"sha256 ([0-9a-f]{12})", first).group(1)
        path = self.proj.path(first.split("  ")[0])
        self.assertEqual(hashlib.sha256(open(path, "rb").read()).hexdigest()[:12], short)

    def test_all_four_formats_and_unique_names_on_repeat(self):
        doc = self.solid_doc()
        text, _, is_error = self.call("export_model", doc_name=doc.Name, formats=["step", "stl", "3mf", "brep"])
        self.assertFalse(is_error, text)
        self.assertEqual(len([l for l in text.split("\n") if self.LINE.match(l)]), 4)
        again, _, _ = self.call("export_model", doc_name=doc.Name, formats=["step"])
        self.assertIn("-2.step", again.split("\n")[0])

    def test_objects_per_part_and_the_combined_file(self):
        doc = self.solid_doc()
        text, _, _ = self.call("export_model", doc_name=doc.Name, objects=["Box", "Cyl"], formats=["step"])
        self.assertEqual(len([l for l in text.split("\n") if self.LINE.match(l)]), 2)
        text, _, _ = self.call("export_model", doc_name=doc.Name, objects=["Box", "Cyl"], formats=["step"], per_part=False)
        self.assertIn("(model, 2 solids)", text.split("\n")[0])

    def test_bad_arguments_are_refused_before_any_document_access(self):
        doc = self.solid_doc()
        before = len(os.listdir(self.proj.path("exports"))) if os.path.isdir(self.proj.path("exports")) else 0
        text, _, is_error = self.call("export_model", doc_name=doc.Name, formats=["dwg"])
        self.assertTrue(is_error)
        self.assertIn("formats", text)
        text, _, is_error = self.call("export_model", doc_name=doc.Name, directory="relative")
        self.assertTrue(is_error)
        self.assertIn("'directory' must be an absolute path", text)
        text, _, is_error = self.call("export_model", doc_name=doc.Name, stl_tolerance=0.0001)
        self.assertTrue(is_error)
        text, _, is_error = self.call("export_model", doc_name=doc.Name, stl_tolerance=9)
        self.assertTrue(is_error)
        self.assertIn("outside 0.001..5 mm", text)
        after = len(os.listdir(self.proj.path("exports"))) if os.path.isdir(self.proj.path("exports")) else 0
        self.assertEqual(before, after)

    def test_document_problems_are_model_readable(self):
        doc = self.solid_doc()
        text, _, is_error = self.call("export_model", doc_name="NoSuchDocument")
        self.assertTrue(is_error)
        self.assertTrue(text.startswith("export_model: "))
        self.assertIn("is not open", text)
        text, _, is_error = self.call("export_model", doc_name=doc.Name, objects=["Nope"])
        self.assertTrue(is_error)
        self.assertIn("no such object(s) 'Nope'", text)
        self.assertIn("exportable parts: Cut", text)
        empty = App.newDocument(unique("T3Z"))
        text, _, is_error = self.call("export_model", doc_name=empty.Name)
        self.assertTrue(is_error)
        self.assertIn("no visible solid part to export", text)

    def test_the_call_is_a_recorded_step_that_is_not_replayed(self):
        doc = self.solid_doc()
        self.call("export_model", doc_name=doc.Name, formats=["step", "stl"])
        step = self.proj.read_steps()[0]
        self.assertEqual((step.tool, step.title, step.ok), ("export_model", "Export step+stl", True))
        self.assertEqual(step.replay["kind"], "none")
        self.assertEqual((step.checkpoint, step.snapshot), ("", ""))

    def test_without_a_project_a_directory_is_required(self):
        doc = self.solid_doc()
        bare = ToolContext()
        text, _, is_error = self.call("export_model", ctx=bare, doc_name=doc.Name)
        self.assertTrue(is_error)
        self.assertIn("no project is bound to this chat", text)
        self.assertIn("pass 'directory'", text)
        target = os.path.join(S.sandbox_dir(), "out")
        text, _, is_error = self.call("export_model", ctx=bare, doc_name=doc.Name, directory=target, formats=["stl"])
        self.assertFalse(is_error, text)
        self.assertTrue(text.split("\n")[0].startswith(target + os.sep), text)   # absolute: not inside any project
        self.assertEqual(len([n for n in os.listdir(target) if not n.startswith(".")]), 1)

    def test_partial_success_reports_the_files_and_the_problem(self):
        doc = self.solid_doc()
        real = X_write_one()

        def flaky(shape, fmt, path, tolerance):
            if fmt == "stl":
                raise RuntimeError("mesher exploded")
            return real(shape, fmt, path, tolerance)

        with mock.patch("oscar.project_export._write_one", flaky):
            text, _, is_error = self.call("export_model", doc_name=doc.Name, formats=["step", "stl"])
        self.assertFalse(is_error, text)
        self.assertRegex(text.split("\n")[0], self.LINE)
        self.assertIn("Some files were NOT written", text)
        self.assertIn("mesher exploded", text)

    def test_when_nothing_could_be_written_it_is_an_error(self):
        doc = self.solid_doc()
        with mock.patch("oscar.project_export._write_one", side_effect=RuntimeError("no writer")):
            text, _, is_error = self.call("export_model", doc_name=doc.Name, formats=["step"])
        self.assertTrue(is_error)
        self.assertIn("no file was written", text)
        self.assertIn("no writer", text)

    def test_finalize_uses_the_same_code_path(self):
        doc = self.solid_doc()
        S.record(self.proj, doc=doc.Name)
        self.proj.finalize(ledger={"intent": "enclosure", "steps": [], "requirements": []}, outcome="validated", validation=None)
        names = sorted(os.listdir(self.proj.path("exports")))
        self.assertEqual(names, ["{}-cut.step".format(self.proj.manifest["active_doc"].lower()), "{}-cut.stl".format(self.proj.manifest["active_doc"].lower())])
        self.assertEqual(self.proj.verify_artifacts().count("exports/*"), 0)


def X_write_one():
    from oscar import project_export

    return project_export._write_one


def zipfile_ok(path: str) -> bool:
    import zipfile

    return os.path.isfile(path) and zipfile.is_zipfile(path)


class ExportRebuildScriptToolTest(ToolCase):
    def build(self):
        name = unique("T3B")
        for tool, args in (("create_document", {"name": name}),
                           ("create_object", {"doc_name": name, "obj_type": "Part::Box", "obj_name": "Block",
                                              "obj_properties": {"Length": 12, "Width": 8, "Height": 6},
                                              "include_screenshot": False}),
                           ("execute_code", {"code": "# cyl\nd = App.getDocument(%r)\no = d.addObject('Part::Cylinder', 'Cyl')\n"
                                                     "o.Radius = 3\no.Height = 7\nd.recompute()\n" % name})):
            text, _, is_error = self.call(tool, **args)
            self.assertFalse(is_error, text)
        return name

    def live(self, name):
        return gui.live_volumes(App.getDocument(name))

    def fake_headless(self, name, *, marker=True, volumes=None, error=None, summary_line=True):
        """A stand-in for the headless child that prints what a real rebuild.py prints."""
        seen = {}

        def run(code, timeout):
            seen["code"], seen["timeout"] = code, timeout
            if error:
                return {"success": False, "error": error, "output": ""}
            out = ["PASS step 001 create_document"]
            if summary_line:
                out.append("REBUILD_RESULT " + json.dumps({"docs": {name: volumes if volumes is not None else self.live(name)}}))
            out.append("REBUILD_OK" if marker else "REBUILD_FAILED")
            return {"success": marker, "returncode": 0 if marker else 1, "output": "\n".join(out)}

        return run, seen

    def test_it_writes_rebuild_py_and_reports_the_counts(self):
        name = self.build()
        text, _, is_error = self.call("export_rebuild_script")
        self.assertFalse(is_error, text)
        self.assertTrue(text.startswith(f"rebuild.py written: {self.proj.path('rebuild.py')} (3 replayable step(s))"), text)
        compile(open(self.proj.path("rebuild.py"), encoding="utf-8").read(), "rebuild.py", "exec")
        self.assertEqual(self.proj.read_steps()[-1].tool, "export_rebuild_script")   # the call itself is a step
        self.assertIn(name, open(self.proj.path("rebuild.py"), encoding="utf-8").read())

    def test_not_replayed_steps_are_counted(self):
        self.build()
        self.call("export_rebuild_script")   # recorded as a step with replay "none"
        text, _, _ = self.call("export_rebuild_script")
        self.assertIn("1 recorded step(s) not replayed", text.split("\n")[0])

    def test_without_a_project_the_tool_says_so(self):
        text, _, is_error = self.call("export_rebuild_script", ctx=ToolContext())
        self.assertTrue(is_error)
        self.assertEqual(text, "export_rebuild_script: no project is bound to this chat (open a chat in the Oscar panel)")

    def test_run_check_passes_on_a_matching_rebuild(self):
        name = self.build()
        run, seen = self.fake_headless(name)
        with mock.patch.object(tools_project, "_run_headless", run):
            text, _, is_error = self.call("export_rebuild_script", run_check=True)
        self.assertFalse(is_error, text)
        self.assertIn("Rebuild check: PASS", text)
        self.assertIn("REBUILD_OK, 2 object(s) in 1 document(s)", text)
        self.assertIn(self.proj.path("rebuild.py"), seen["code"])
        self.assertTrue(seen["code"].startswith("exec(open("))
        self.assertEqual(self.proj.manifest["rebuild_check"]["ok"], True)

    def test_a_passing_check_makes_the_summary_say_so(self):
        name = self.build()
        run, _ = self.fake_headless(name)
        with mock.patch.object(tools_project, "_run_headless", run):
            self.call("export_rebuild_script", run_check=True)
        good = self.proj.build_summary(None, "validated", None).data["validated"]
        self.assertIn("rebuild", [g["id"] for g in good])

    def test_run_check_fails_when_the_volumes_differ(self):
        name = self.build()
        wrong = dict(self.live(name))
        wrong["Block"] = wrong["Block"] * 1.01
        run, _ = self.fake_headless(name, volumes=wrong)
        with mock.patch.object(tools_project, "_run_headless", run):
            text, _, is_error = self.call("export_rebuild_script", run_check=True)
        self.assertFalse(is_error, text)   # a failed check is a REPORT, not a failed tool call
        self.assertIn("Rebuild check: FAIL", text)
        self.assertIn("volume of '%s.Block' differs" % name, text)
        self.assertFalse(self.proj.manifest["rebuild_check"]["ok"])
        good = self.proj.build_summary(None, "validated", None).data["validated"]
        self.assertNotIn("rebuild", [g["id"] for g in good])

    def test_run_check_fails_when_objects_are_missing(self):
        name = self.build()
        run, _ = self.fake_headless(name, volumes={"Block": self.live(name)["Block"]})
        with mock.patch.object(tools_project, "_run_headless", run):
            text, _, _ = self.call("export_rebuild_script", run_check=True)
        self.assertIn("Rebuild check: FAIL", text)
        self.assertIn("object names differ", text)
        self.assertIn("missing after rebuild: ['Cyl']", text)

    def test_run_check_fails_without_the_marker(self):
        name = self.build()
        run, _ = self.fake_headless(name, marker=False)
        with mock.patch.object(tools_project, "_run_headless", run):
            text, _, _ = self.call("export_rebuild_script", run_check=True)
        self.assertIn("Rebuild check: FAIL", text)
        self.assertIn("did not print REBUILD_OK", text)

    def test_run_check_reports_an_engine_that_could_not_start(self):
        name = self.build()
        run, _ = self.fake_headless(name, error="no headless engine binary found")
        with mock.patch.object(tools_project, "_run_headless", run):
            text, _, _ = self.call("export_rebuild_script", run_check=True)
        self.assertIn("Rebuild check: FAIL", text)
        self.assertIn("no headless engine binary found", text)

    def test_run_check_without_a_result_line_fails_even_with_the_marker(self):
        name = self.build()
        run, _ = self.fake_headless(name, summary_line=False)
        with mock.patch.object(tools_project, "_run_headless", run):
            text, _, _ = self.call("export_rebuild_script", run_check=True)
        self.assertIn("no REBUILD_RESULT line", text)

    def test_run_check_with_nothing_to_replay_is_an_error(self):
        self.call("record_note", kind="plan", text="p")
        text, _, is_error = self.call("export_rebuild_script", run_check=True)
        self.assertTrue(is_error)
        self.assertIn("no replayable steps", text)

    def test_run_check_for_an_unknown_document_is_an_error(self):
        self.build()
        text, _, is_error = self.call("export_rebuild_script", doc_name="NoSuchDocument", run_check=True)
        self.assertTrue(is_error)
        self.assertIn("is not open", text)

    def test_doc_name_limits_the_comparison_to_that_document(self):
        name = self.build()
        other = self.solid_doc()
        run, _ = self.fake_headless(name)
        with mock.patch.object(tools_project, "_run_headless", run):
            text, _, _ = self.call("export_rebuild_script", doc_name=name, run_check=True)
        self.assertIn("PASS", text, text)
        self.assertNotIn(other.Name, text)

    def test_a_real_headless_run_reproduces_the_model(self):
        """The generated rebuild.py in a fresh freecadcmd through cad.headless.run (a few seconds)."""
        from oscar.cad import headless

        if headless.find_console() is None:
            self.skipTest("no headless engine binary on this machine")
        self.build()
        text, _, is_error = self.call("export_rebuild_script", run_check=True)
        self.assertFalse(is_error, text)
        self.assertIn("Rebuild check: PASS", text)
        self.assertTrue(self.proj.manifest["rebuild_check"]["ok"])

    def test_a_real_headless_run_of_a_corrupted_script_is_reported_as_fail(self):
        from oscar.cad import headless

        if headless.find_console() is None:
            self.skipTest("no headless engine binary on this machine")
        self.build()
        rebuild_text = self.proj.rebuild_script_text().replace("'Length': 12", "'Length': 99")
        with mock.patch.object(P.Project, "rebuild_script_text", lambda self_: rebuild_text):
            text, _, is_error = self.call("export_rebuild_script", run_check=True)
        self.assertFalse(is_error, text)
        self.assertIn("Rebuild check: FAIL", text)
        self.assertIn("volume of", text)


if __name__ == "__main__":
    unittest.main()
