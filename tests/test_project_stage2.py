"""The Project against REAL FreeCAD documents (run under freecadcmd): the geometry predicates of
``cad.health`` (Origin features and ``App::Part`` containers are never model geometry), the ``Params`` mirror and the
rewind re-sync, a real ``HealthReport`` flowing into ``record_validation`` / the Build Summary (and, when
``tools_verify`` exists, its real ``ValidationResult``), restore re-activation with several documents, the
``.FCStd`` temp-name rule against the real ``saveCopy``, and the print orientation of ``export_model`` / the
auto-export of ``finalize`` driven by the kit spec keys ``print_orientation.shell|lid``."""

import json
import os
import sys
import unittest
import uuid
import zipfile
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402
import Part  # noqa: E402

import project_support as S  # noqa: E402
from oscar import project as P  # noqa: E402
from oscar import project_export as X  # noqa: E402
from oscar import project_gui as gui  # noqa: E402
from oscar import tools  # noqa: E402
from oscar.cad import health  # noqa: E402
from oscar.kit import spec as kit_spec  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402
from test_project_gui import build_model  # noqa: E402  (helper only: no test class is imported)


def unique(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:6]}"


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class RealDoc(unittest.TestCase):
    """A Project per test and every document the test created closed afterwards."""

    def setUp(self):
        self._before = set(App.listDocuments())
        self.proj = S.make_project(chat_id=f"20261001-000000-{uuid.uuid4().hex[:6]}")

    def tearDown(self):
        S.close_docs([n for n in App.listDocuments() if n not in self._before])


# ---------------------------------------------------------------------------
# Origin features and containers

class GeometryPredicatesTest(RealDoc):
    """Origin features have an infinite bounding box and a garbage Volume, a part inside an ``App::Part`` is in the
    container's LOCAL frame: everything the Project lists or exports goes through ``cad.health``."""

    def container_doc(self):
        doc = App.newDocument(unique("T3O"))
        doc.addObject("PartDesign::Body", "Body")           # an empty Body: its Origin objects are the trap
        part = doc.addObject("App::Part", "Assembly")
        inside = doc.addObject("Part::Box", "InPart")
        part.addObject(inside)
        part.Placement.Base = App.Vector(0, 0, 50)
        doc.addObject("Part::Box", "Free")
        doc.recompute()
        return doc

    def test_origin_features_and_containers_are_not_geometry(self):
        doc = self.container_doc()
        origin_names = [o.Name for o in doc.Objects if o.TypeId in ("App::Origin", "App::Line", "App::Plane", "App::Point")]
        self.assertGreaterEqual(len(origin_names), 14, "the document must hold the Origin objects of the Body and the Part")
        self.assertEqual(sorted(o.Name for o in gui.geometry_objects(doc)), ["Free", "InPart"])
        for name in origin_names + ["Assembly", "Body"]:
            self.assertFalse(gui.is_geometry(doc.getObject(name)), name)
        self.assertEqual(sorted(o.Name for o in gui.visible_tips(doc)), ["Free", "InPart"])
        self.assertTrue(gui.has_geometry(doc))

    def test_the_decision_is_the_one_of_cad_health(self):
        doc = self.container_doc()
        for obj in doc.Objects:
            self.assertEqual(gui.is_geometry(obj), health.is_geometry(obj), obj.Name)
        inside = doc.getObject("InPart")
        self.assertAlmostEqual(inside.Shape.BoundBox.ZMin, 0.0, places=6, msg="the container-local frame")
        self.assertAlmostEqual(gui.global_shape(inside).BoundBox.ZMin, 50.0, places=6)
        self.assertAlmostEqual(health.global_shape(inside).BoundBox.ZMin, 50.0, places=6)

    def test_a_document_with_only_a_body_has_no_geometry(self):
        doc = App.newDocument(unique("T3B"))
        doc.addObject("PartDesign::Body", "Body")
        doc.recompute()
        self.assertFalse(gui.has_geometry(doc))
        self.assertEqual(gui.geometry_objects(doc), [])
        self.assertEqual(self.proj.snapshot(doc.Name, "x", 1), ("", "empty document"))

    def test_volumes_tree_and_object_names_never_list_the_origin(self):
        doc = self.container_doc()
        volumes = gui.live_volumes(doc)
        self.assertEqual(sorted(volumes), ["Free", "InPart"])
        for value in volumes.values():
            self.assertAlmostEqual(value, 1000.0, places=6)     # not the 4.789e+152 of an Origin feature
        tree = {row["name"]: row for row in gui.model_tree(doc)}
        self.assertEqual(sorted(tree), ["Free", "InPart"])
        self.assertAlmostEqual(tree["InPart"]["bbox"][2], 50.0, places=6)    # ZMin in the GLOBAL frame
        self.assertAlmostEqual(tree["InPart"]["bbox"][5], 60.0, places=6)
        self.assertTrue(all(row["tip"] for row in tree.values()))
        primary, every = gui.object_names(doc)
        self.assertEqual(primary, ["Assembly", "InPart", "Free"])   # the Origin objects and the empty Body are plumbing
        self.assertFalse([n for n in primary if n.startswith(("Origin", "X_Axis", "XY_Plane"))], primary)
        self.assertIn("Origin", every)
        self.assertEqual(sorted(set(primary) - {"Assembly"}), ["Free", "InPart"])

    def test_the_checkpoint_sidecar_and_the_replay_summary_exclude_them_too(self):
        doc = self.container_doc()
        rec = S.record(self.proj, doc=doc.Name)
        self.assertEqual(rec.checkpoint, "001")
        side = json.loads(read_text(self.proj.path("checkpoints", "001.json")))
        self.assertEqual(sorted(side["objects"]), ["Assembly", "Free", "InPart"])
        self.assertIn("Origin", side["all_objects"])
        summary = P.replay_summary(P.ReplayResult(ok=True, doc_name=doc.Name, name_map={doc.Name: doc.Name}))
        self.assertEqual(sorted(summary["docs"][doc.Name]), ["Free", "InPart"])
        for value in summary["docs"][doc.Name].values():
            self.assertAlmostEqual(value, 1000.0, places=6)

    def test_the_export_has_the_global_frame_and_no_origin_file(self):
        doc = self.container_doc()
        result = X.export_tips(self.proj, doc.Name, formats=["step"])
        self.assertEqual(sorted(f.label for f in result.files), ["Free", "InPart"])
        by_label = {f.label: Part.read(f.path) for f in result.files}
        self.assertAlmostEqual(by_label["InPart"].BoundBox.ZMin, 50.0, places=4)
        self.assertAlmostEqual(by_label["Free"].BoundBox.ZMin, 0.0, places=4)
        for shape in by_label.values():
            self.assertLess(shape.BoundBox.DiagonalLength, 1e3)

    def test_the_build_summary_tree_lists_only_real_parts(self):
        doc = self.container_doc()
        S.record(self.proj, doc=doc.Name)
        summary = self.proj.build_summary(None, "answered", None)
        self.assertEqual(sorted(r["name"] for r in summary.data["tree"]), ["Free", "InPart"])
        self.assertNotIn("e+", summary.markdown.split("### Requirements")[0], "a garbage bounding box leaked into the tree")


# ---------------------------------------------------------------------------
# saveCopy against the real FreeCAD

class DocSpy:
    """Wraps a real document and remembers the paths ``saveCopy`` was asked for."""

    def __init__(self, doc):
        object.__setattr__(self, "_doc", doc)
        object.__setattr__(self, "requested", [])

    def saveCopy(self, path):  # noqa: N802 - FreeCAD API name
        self.requested.append(path)
        return self._doc.saveCopy(path)

    def __getattr__(self, name):
        return getattr(self._doc, name)

    def __setattr__(self, name, value):
        setattr(self._doc, name, value)


class RealSaveCopyTest(RealDoc):
    def test_save_copy_creates_the_file_at_exactly_the_requested_path(self):
        doc = build_model(unique("T3S"))
        directory = S.sandbox_dir()
        dest = os.path.join(directory, "model.FCStd")
        self.assertEqual(gui.save_copy(doc, dest), (True, ""))
        self.assertTrue(zipfile.is_zipfile(dest))
        self.assertEqual(os.listdir(directory), ["model.FCStd"], "no temp file, no FCBak, no model.FCStd.tmp-1.FCStd")
        self.assertEqual(gui.save_copy(doc, dest), (True, ""))   # overwrite: still no backup file
        self.assertEqual(os.listdir(directory), ["model.FCStd"])

    def test_every_save_path_asks_freecad_for_an_fcstd_name_and_the_files_exist_there(self):
        doc = build_model(unique("T3S"))
        spy = DocSpy(doc)
        with mock.patch.object(gui, "resolve_doc", lambda name: spy if name in ("", doc.Name) else None):
            self.proj.begin_turn(S.TURN, "work", work=True)
            S.record(self.proj, doc=doc.Name)                       # autosave + checkpoint 001
            self.assertTrue(self.proj.ensure_turn_checkpoint(doc.Name).startswith("T"))
            self.assertEqual(self.proj.checkpoint(doc.Name, "manual"), "manual")
            self.assertEqual(self.proj.autosave(doc.Name), "model.FCStd")
            self.assertTrue(self.proj.restore(doc.Name, "001").ok)  # writes the rescue copy first
            self.assertEqual(self.proj.save_now(doc.Name), ("model.FCStd", ""))
        self.assertGreaterEqual(len(spy.requested), 6)
        for path in spy.requested:
            self.assertTrue(path.endswith(".FCStd"), path)
            self.assertIn(".tmp-", os.path.basename(path))
        for rel in ("model.FCStd", "checkpoints/001.FCStd", "checkpoints/manual.FCStd", "checkpoints/_rescue.FCStd"):
            self.assertTrue(zipfile.is_zipfile(self.proj.path(*rel.split("/"))), rel)
        stray = []
        for folder, _, files in os.walk(self.proj.root):
            stray += [f for f in files if ".tmp-" in f or f.endswith((".FCBak", ".FCStd.FCStd"))]
        self.assertEqual(stray, [])


# ---------------------------------------------------------------------------
# restore re-activation

class RestoreReactivationTest(RealDoc):
    def two_documents(self):
        a = build_model(unique("T3A"))
        b = build_model(unique("T3B"), with_params=False)
        S.record(self.proj, doc=a.Name)                          # step 1 / checkpoint 001
        a.addObject("Part::Box", "Extra")
        a.recompute()
        S.record(self.proj, doc=a.Name, call_id="c2", delta={"added": ["Extra"], "removed": [], "changed": []})
        return a, b

    def test_the_restored_document_is_active_again_and_other_documents_are_untouched(self):
        a, b = self.two_documents()
        App.setActiveDocument(b.Name)
        names_b = [o.Name for o in b.Objects]
        known = set(App.listDocuments())
        result = self.proj.restore(a.Name, "001")
        self.assertTrue(result.ok, result.error)
        self.assertTrue(result.reactivated)
        self.assertEqual(App.ActiveDocument.Name, a.Name)
        self.assertEqual(set(App.listDocuments()), known, "the hidden temp document must be closed again")
        self.assertEqual(a.Name, result.doc)
        self.assertIsNone(a.getObject("Extra"))
        self.assertEqual([o.Name for o in b.Objects], names_b)
        box = App.ActiveDocument.addObject("Part::Box", "AfterRestore")   # an execute_code step would do this
        App.ActiveDocument.recompute()
        self.assertEqual(box.Document.Name, a.Name)

    def test_two_restores_in_a_row_both_leave_a_usable_active_document(self):
        a, _ = self.two_documents()
        for ckpt in ("002", "001", "002"):
            result = self.proj.restore(a.Name, ckpt)
            self.assertTrue(result.ok, result.error)
            self.assertTrue(result.reactivated, ckpt)
            self.assertEqual(App.ActiveDocument.Name, a.Name, ckpt)
        self.assertIsNotNone(a.getObject("Extra"))

    def test_a_failed_restore_re_activates_the_live_document_too(self):
        a, b = self.two_documents()
        App.setActiveDocument(b.Name)
        bad = self.proj.path("checkpoints", "001.FCStd")
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("Document.xml", "<not a document")
        before = [o.Name for o in a.Objects]
        result = self.proj.restore(a.Name, "001")
        self.assertFalse(result.ok)
        self.assertEqual([o.Name for o in a.Objects], before, "a failed restore leaves the live document as it was")
        self.assertIsNotNone(App.ActiveDocument)
        self.assertEqual(App.ActiveDocument.Name, a.Name)


# ---------------------------------------------------------------------------
# the Params mirror

class ParamsMirrorRewindTest(RealDoc):
    def test_a_rewind_brings_the_params_mirror_back_with_the_sheet(self):
        doc = build_model(unique("T3P"))
        S.record(self.proj, doc=doc.Name)
        self.assertEqual(self.proj.read_params()["params"][0]["value"], 2.0)
        sheet = doc.getObject("Params")
        sheet.set("B2", "=3 mm")
        doc.recompute()
        S.record(self.proj, doc=doc.Name, call_id="c2", delta={"added": [], "removed": [],
                                                              "changed": [{"name": "Params", "props": ["B2"]}]})
        mirror = self.proj.read_params()
        self.assertEqual((mirror["step"], {r["name"]: r["value"] for r in mirror["params"]}["wall_t"]), (2, 3.0))
        result = self.proj.rewind_to(1)
        self.assertTrue(result.ok, result.error)
        self.assertAlmostEqual(doc.getObject("Params").get("wall_t").Value, 2.0)   # a NEW sheet object came back with the checkpoint
        mirror = self.proj.read_params()
        self.assertEqual(mirror["step"], 1)
        self.assertEqual({r["name"]: r["value"] for r in mirror["params"]}["wall_t"], 2.0)
        self.assertEqual(mirror["params"], health.read_param_table(doc)["rows"], "the mirror IS the reader's output")

    def test_a_rewind_of_a_document_without_a_sheet_writes_no_mirror(self):
        doc = build_model(unique("T3N"), with_params=False)
        S.record(self.proj, doc=doc.Name)
        S.record(self.proj, doc=doc.Name, call_id="c2")
        self.assertTrue(self.proj.rewind_to(1).ok)
        self.assertFalse(os.path.exists(self.proj.path("params.json")))


# ---------------------------------------------------------------------------
# a real HealthReport in the validation record and the Build Summary

def validation_from(report, *, step, new_errors=True, ran=("health",)):
    """What ``tools_verify.run_validation`` builds from a ``HealthReport``, as a dict."""
    return {"ok": report.valid if new_errors else True, "doc": report.doc, "health": report.to_dict(),
            "gate_errors": [f.to_dict() for f in report.errors] if new_errors else [], "fit_rows": [], "dfm": [],
            "coverage": [], "ran": list(ran), "not_validated": ["physical"], "after_step": step}


class RealHealthValidationTest(RealDoc):
    def test_a_real_report_is_stored_and_summarised(self):
        doc = build_model(unique("T3H"))
        S.record(self.proj, doc=doc.Name)
        report = health.model_health(doc, strict=True)
        self.assertTrue(report.valid, [f.to_dict() for f in report.findings])
        self.proj.record_validation(validation_from(report, step=self.proj.last_geometry_step()))
        stored = self.proj.read_validation()
        self.assertEqual((stored["doc"], stored["ran"], stored["ok"]), (doc.Name, ["health"], True))
        self.assertEqual(stored["health"]["counts"], report.to_dict()["counts"])
        summary = self.proj.build_summary({"intent": "generic_part", "steps": [], "requirements": []}, "validated", None)
        line = next(v["result"] for v in summary.data["validated"] if v["id"] == "health")
        counts = report.to_dict()["counts"]
        self.assertEqual(line, f"valid ({counts['objects']} objects, {counts['solids']} solids)")
        self.assertEqual(sorted(r["name"] for r in summary.data["tree"]), ["Box", "Cyl"])
        json.dumps(stored, allow_nan=False)

    def test_the_tree_comes_from_the_stored_health_rows_when_the_document_is_gone(self):
        doc = build_model(unique("T3H"))
        S.record(self.proj, doc=doc.Name)
        live = {r["name"]: r["volume"] for r in self.proj.build_summary(None, "answered", None).data["tree"]}
        report = health.model_health(doc, strict=True)
        self.proj.record_validation(validation_from(report, step=self.proj.last_geometry_step()))
        App.closeDocument(doc.Name)
        fallback = self.proj.build_summary(None, "answered", None).data["tree"]
        self.assertEqual(sorted(r["name"] for r in fallback), ["Box", "Cyl"])
        for row in fallback:
            self.assertAlmostEqual(row["volume"], live[row["name"]], places=2)
            self.assertTrue(row["tip"])
            self.assertEqual(len(row["bbox"]), 6)

    def broken_doc(self):
        doc = App.newDocument(unique("T3X"))
        cyl = doc.addObject("Part::Cylinder", "Cyl")
        cyl.Radius = 0                                         # a real RECOMPUTE_ERROR
        doc.addObject("Part::Box", "Box")
        doc.recompute()
        return doc

    def test_real_errors_are_gate_errors_when_new_and_pre_existing_items_when_not(self):
        doc = self.broken_doc()
        S.record(self.proj, doc=doc.Name)
        report = health.model_health(doc, strict=True)
        self.assertFalse(report.valid)
        step = self.proj.last_geometry_step()
        self.proj.record_validation(validation_from(report, step=step))
        summary = self.proj.build_summary(None, "needs_help", None)
        self.assertNotIn("health", [v["id"] for v in summary.data["validated"]])
        reason = next(n["reason"] for n in summary.data["not_validated"] if n["id"] == "health")
        self.assertIn("error(s)", reason)
        self.assertIn("RECOMPUTE_ERROR Cyl", reason)
        self.assertTrue(any(i.startswith("health error RECOMPUTE_ERROR Cyl:") for i in summary.data["open_items"]))
        # the same findings, but they existed before the turn (gate_errors empty): reported, never a failure
        self.proj.record_validation(validation_from(report, step=step, new_errors=False))
        again = self.proj.build_summary(None, "answered", None)
        self.assertTrue(any(i.startswith("pre-existing: RECOMPUTE_ERROR Cyl:") for i in again.data["open_items"]))
        line = next(v["result"] for v in again.data["validated"] if v["id"] == "health")
        self.assertRegex(line, r"^no new errors \(\d+ pre-existing\)$")

    def tools_verify(self):
        try:
            from oscar import tools_verify
        except ModuleNotFoundError as e:
            if e.name not in ("oscar.tools_verify", "tools_verify"):
                raise
            self.skipTest("oscar.tools_verify is not available")
        return tools_verify

    def test_t4s_real_validation_result_is_accepted(self):
        verify = self.tools_verify()
        run = getattr(verify, "run_validation", None)
        if run is None or not hasattr(verify, "ValidationResult"):
            self.skipTest("tools_verify has no run_validation / ValidationResult yet")
        doc = build_model(unique("T3V"))
        S.record(self.proj, doc=doc.Name)
        ctx = ToolContext(project=self.proj, turn_id=S.TURN)
        result = run(doc.Name, ctx=ctx, fit=False)
        stored = self.proj.read_validation()
        self.assertIsNotNone(stored, "run_validation must persist through ctx.project.record_validation")
        for key in ("schema", "ts", "after_step", "doc", "ok", "health", "gate_errors", "fit_rows", "dfm", "coverage",
                    "ran", "not_validated"):
            self.assertIn(key, stored)
        self.assertEqual(stored["doc"], doc.Name)
        self.assertEqual(stored["after_step"], self.proj.last_geometry_step())
        self.assertEqual(stored["ran"], list(result.ran))
        recorded = P.Project.open(self.proj.root)
        summary = recorded.build_summary({"intent": "generic_part", "steps": [], "requirements": []}, "validated", result)
        if result.ok and "health" in result.ran:
            self.assertIn("health", [v["id"] for v in summary.data["validated"]])
        self.proj.record_validation(result)   # the dataclass itself, not only its dict
        self.assertEqual(self.proj.read_validation()["ran"], stored["ran"])


# ---------------------------------------------------------------------------
# export: print orientation from the kit spec keys

class OrientationFixture(RealDoc):
    """A kit-like model: a cup-shaped shell (floor at the bottom) and a cap lid (plate on top, skirt hanging down), tagged
    with ``OscarRole`` and deliberately labelled differently so the tag, not the label, decides."""

    def setUp(self):
        super().setUp()
        self.doc = App.newDocument(unique("T3K"))
        outer = Part.makeBox(40, 30, 10, App.Vector(0, 0, -5))
        self.shell_shape = outer.cut(Part.makeBox(36, 26, 8, App.Vector(2, 2, -3)))
        plate = Part.makeBox(40, 30, 2, App.Vector(0, 0, 20))
        skirt = Part.makeBox(40, 30, 6, App.Vector(0, 0, 14)).cut(Part.makeBox(36, 26, 6, App.Vector(2, 2, 14)))
        self.lid_shape = plate.fuse(skirt).removeSplitter()
        for name, label, role, shape in (("Shell_Tip", "Bottom", "shell", self.shell_shape),
                                         ("Lid_Tip", "Cap", "lid", self.lid_shape)):
            obj = self.doc.addObject("Part::Feature", name)
            obj.Label = label
            obj.addProperty("App::PropertyString", "OscarRole", "Oscar")
            obj.OscarRole = role
            obj.Shape = shape
        self.doc.recompute()
        # what the files should hold: the shell as modelled dropped onto z=0, the lid flipped about X then dropped
        self.shell_com = self.shell_shape.Solids[0].CenterOfMass.z
        self.lid_com = self.lid_shape.Solids[0].CenterOfMass.z
        self.expect_shell = self.shell_com - self.shell_shape.BoundBox.ZMin
        self.expect_lid = self.lid_shape.BoundBox.ZMax - self.lid_com
        self.expect_lid_as_modelled = self.lid_com
        self.spec = kit_spec.default_spec().to_dict()

    def exported(self, result, fmt="step"):
        return {f.label: Part.read(f.path) for f in result.files if f.format == fmt}

    def com(self, shape):
        return shape.Solids[0].CenterOfMass.z


class ExportOrientationTest(OrientationFixture):
    def test_the_kit_spec_orients_the_tagged_parts_in_the_files(self):
        self.assertEqual(self.spec["print_orientation"], {"shell": "floor_down", "lid": "top_down"})
        self.proj.save_spec(self.spec)
        result = X.export_tips(self.proj, self.doc.Name, formats=["step", "stl"])
        self.assertEqual(result.errors, [])
        self.assertEqual(sorted((f.label, f.format, f.orientation) for f in result.files),
                         [("Bottom", "step", "floor_down"), ("Bottom", "stl", "floor_down"),
                          ("Cap", "step", "top_down"), ("Cap", "stl", "top_down")])
        parts = self.exported(result)
        self.assertAlmostEqual(parts["Bottom"].BoundBox.ZMin, 0.0, places=4)
        self.assertAlmostEqual(self.com(parts["Bottom"]), self.expect_shell, places=3)
        self.assertAlmostEqual(parts["Cap"].BoundBox.ZMin, 0.0, places=4)
        self.assertAlmostEqual(self.com(parts["Cap"]), self.expect_lid, places=3)
        self.assertLess(self.com(parts["Cap"]), parts["Cap"].BoundBox.ZMax / 2, "the lid plate lies on the bed")
        self.assertGreater(self.expect_lid_as_modelled - self.lid_shape.BoundBox.ZMin, parts["Cap"].BoundBox.ZMax / 2)
        import Mesh

        for item in (f for f in result.files if f.format == "stl"):
            self.assertAlmostEqual(Mesh.Mesh(item.path).BoundBox.ZMin, 0.0, places=3, msg=item.label)
        self.assertEqual(result.notes, [])
        self.assertTrue(os.path.basename(result.files[0].path).startswith("picase-"))   # the spec name

    def test_every_orientation_value_the_kit_accepts_is_honoured(self):
        for shell in kit_spec.SHELL_ORIENTATIONS:
            for lid in kit_spec.LID_ORIENTATIONS:
                self.spec["print_orientation"] = {"shell": shell, "lid": lid}
                self.proj.save_spec(self.spec)
                parts = self.exported(X.export_tips(self.proj, self.doc.Name, formats=["step"]))
                want_shell = self.expect_shell if shell == "floor_down" else self.shell_com
                want_lid = self.expect_lid if lid == "top_down" else self.lid_com
                self.assertAlmostEqual(self.com(parts["Bottom"]), want_shell, places=3, msg=f"{shell}/{lid}")
                self.assertAlmostEqual(self.com(parts["Cap"]), want_lid, places=3, msg=f"{shell}/{lid}")
                if shell == "as_modeled":
                    self.assertAlmostEqual(parts["Bottom"].BoundBox.ZMin, self.shell_shape.BoundBox.ZMin, places=4)

    def test_an_unknown_value_exports_as_modelled_and_says_so(self):
        self.spec["print_orientation"] = {"shell": "floor_down", "lid": "upside_down"}
        self.proj.save_spec(self.spec)
        result = X.export_tips(self.proj, self.doc.Name, formats=["step"])
        parts = self.exported(result)
        self.assertAlmostEqual(self.com(parts["Cap"]), self.lid_com, places=3)
        self.assertAlmostEqual(parts["Cap"].BoundBox.ZMin, self.lid_shape.BoundBox.ZMin, places=4)
        self.assertAlmostEqual(self.com(parts["Bottom"]), self.expect_shell, places=3, msg="the shell value was fine")
        self.assertEqual(len(result.notes), 1)
        self.assertIn("print_orientation.lid 'upside_down'", result.notes[0])
        self.assertIn("top_down", result.notes[0])
        self.assertEqual(sorted((f.label, f.orientation) for f in result.files),
                         [("Bottom", "floor_down"), ("Cap", "as_modeled")])

    def test_no_spec_a_spec_without_the_key_and_a_malformed_key_mean_as_modelled(self):
        cases = {"no spec.json": None, "no key": {"name": "case"}, "string": {"print_orientation": "top_down"},
                 "empty values": {"print_orientation": {"shell": "", "lid": None}}}
        for label, spec in cases.items():
            if spec is not None:
                self.proj.save_spec(spec)
            result = X.export_tips(self.proj, self.doc.Name, formats=["step"])
            parts = self.exported(result)
            self.assertAlmostEqual(self.com(parts["Cap"]), self.lid_com, places=3, msg=label)
            self.assertAlmostEqual(self.com(parts["Bottom"]), self.shell_com, places=3, msg=label)
            self.assertEqual(result.notes, [], label)
            self.assertEqual({f.orientation for f in result.files}, {"as_modeled"}, label)

    def test_print_orientation_false_keeps_the_modelled_pose(self):
        self.proj.save_spec(self.spec)
        result = X.export_tips(self.proj, self.doc.Name, formats=["step"], print_orientation=False)
        parts = self.exported(result)
        self.assertAlmostEqual(self.com(parts["Cap"]), self.lid_com, places=3)
        self.assertAlmostEqual(parts["Bottom"].BoundBox.ZMin, self.shell_shape.BoundBox.ZMin, places=4)
        self.assertEqual({f.orientation for f in result.files}, {"as_modeled"})

    def test_a_combined_file_keeps_the_assembled_pose_and_says_so(self):
        self.proj.save_spec(self.spec)
        result = X.export_tips(self.proj, self.doc.Name, formats=["step"], per_part=False)
        self.assertEqual(len(result.files), 1)
        self.assertEqual(result.files[0].orientation, "as_modeled")
        shape = Part.read(result.files[0].path)
        box = shape.BoundBox
        self.assertAlmostEqual(box.ZMin, -5.0, places=4)       # the shell floor where it was modelled, not dropped
        self.assertAlmostEqual(box.ZMax, 22.0, places=4)       # the lid plate on top: not flipped onto the shell
        self.assertEqual(len(result.notes), 1)
        self.assertIn("not applied to a combined file", result.notes[0])
        quiet = X.export_tips(self.proj, self.doc.Name, formats=["step"], per_part=False, print_orientation=False)
        self.assertEqual(quiet.notes, [])

    def test_the_role_tag_wins_over_the_label(self):
        # the Label fallback ("Shell" / "Lid") is only a fallback: a tagged lid labelled 'Shell' is still a lid
        self.doc.getObject("Lid_Tip").Label = "Shell"
        self.proj.save_spec(self.spec)
        parts = self.exported(X.export_tips(self.proj, self.doc.Name, formats=["step"]))
        self.assertAlmostEqual(self.com(parts["Shell"]), self.expect_lid, places=3)

    def test_the_directory_override_orients_like_the_project_export(self):
        self.proj.save_spec(self.spec)
        target = os.path.join(S.sandbox_dir(), "out")
        result = X.export_tips(self.proj, self.doc.Name, formats=["step"], directory=target)
        self.assertAlmostEqual(self.com(self.exported(result)["Cap"]), self.expect_lid, places=3)
        self.assertNotIn("exported_after_step", self.proj.manifest, "an export elsewhere says nothing about exports/")

    def test_plan_orientation_matches_the_real_kit_constants(self):
        accepted = X.known_orientations()
        self.assertEqual(accepted["shell"], tuple(kit_spec.SHELL_ORIENTATIONS))
        self.assertEqual(accepted["lid"], tuple(kit_spec.LID_ORIENTATIONS))


class ExportToolOrientationTest(OrientationFixture):
    def setUp(self):
        super().setUp()
        self.ctx = ToolContext(project=self.proj, turn_id="T20261001-000001-0a1b")
        self.proj.begin_turn(self.ctx.turn_id, "export", work=True)

    def call(self, **args):
        self.ctx.call_id = f"c{uuid.uuid4().hex[:4]}"
        with use_context(self.ctx):
            return tools.run_tool("export_model", args)

    def test_the_result_lists_the_files_then_the_orientation_that_was_applied(self):
        self.proj.save_spec(self.spec)
        text, _, is_error = self.call(doc_name=self.doc.Name)
        self.assertFalse(is_error, text)
        lines = text.split("\n")
        self.assertEqual(len([l for l in lines if l.startswith("exports/")]), 4)
        for line in lines[:4]:
            self.assertRegex(line, r"^exports/picase-[a-z]+\.(step|stl)  \d+(\.\d)? (B|KB|MB)  sha256 [0-9a-f]{12}  \(.+, 1 solid\)$")
        self.assertIn("Print orientation (print_orientation in spec.json): Bottom floor_down, Cap top_down; "
                      "the other parts are as modelled.", lines)
        self.assertTrue(lines[-1].startswith("Folder: "))
        self.assertLess(lines.index(next(l for l in lines if l.startswith("Print orientation"))), len(lines) - 1)

    def test_no_orientation_line_without_a_spec_and_a_note_for_a_bad_value(self):
        text, _, _ = self.call(doc_name=self.doc.Name)
        self.assertNotIn("Print orientation", text)
        self.assertNotIn("Note:", text)
        self.spec["print_orientation"]["lid"] = "sideways"
        self.proj.save_spec(self.spec)
        text, _, is_error = self.call(doc_name=self.doc.Name, formats=["step"])
        self.assertFalse(is_error, text)
        self.assertIn("Print orientation (print_orientation in spec.json): Bottom floor_down; the other parts are as modelled.", text)
        self.assertIn("Note: print_orientation.lid 'sideways' is not one of top_down, as_modeled; the lid was exported as modelled.", text)

    def test_a_combined_export_carries_the_note_and_no_orientation_line(self):
        self.proj.save_spec(self.spec)
        text, _, is_error = self.call(doc_name=self.doc.Name, per_part=False, formats=["step"])
        self.assertFalse(is_error, text)
        self.assertNotIn("Print orientation (", text)
        self.assertIn("Note: print orientation is not applied to a combined file", text)

    def test_a_successful_export_marks_the_project_and_a_later_step_makes_it_stale(self):
        S.record(self.proj, doc=self.doc.Name)
        self.proj.save_spec(self.spec)
        self.call(doc_name=self.doc.Name, formats=["step"])
        self.assertEqual(self.proj.manifest["exported_after_step"], 1)       # step 1 = the geometry, step 2 = the export itself
        S.record(self.proj, doc=self.doc.Name, call_id="c9")                 # step 3: the model changed after the export
        items = self.proj.build_summary({"intent": "enclosure", "steps": [], "requirements": []}, "answered", None).data["open_items"]
        self.assertTrue(any(i.startswith("exports/ was written after step 1 but the model is at step 3") for i in items), items)
        self.call(doc_name=self.doc.Name, formats=["step"])
        self.assertEqual(self.proj.manifest["exported_after_step"], 3)
        items = self.proj.build_summary({"intent": "enclosure", "steps": [], "requirements": []}, "answered", None).data["open_items"]
        self.assertFalse(any(i.startswith("exports/") for i in items), items)

    def test_finalize_auto_export_is_the_same_code_path(self):
        self.proj.save_spec(self.spec)
        S.record(self.proj, doc=self.doc.Name)
        self.proj.finalize(ledger={"intent": "enclosure", "steps": [], "requirements": []}, outcome="answered",
                           validation=None)
        names = sorted(os.listdir(self.proj.path("exports")))
        self.assertEqual(names, ["picase-bottom.step", "picase-bottom.stl", "picase-cap.step", "picase-cap.stl"])
        auto = Part.read(self.proj.path("exports", "picase-cap.step"))
        self.assertAlmostEqual(self.com(auto), self.expect_lid, places=3, msg="the auto-export orients the lid like the tool")
        self.assertAlmostEqual(self.com(Part.read(self.proj.path("exports", "picase-bottom.step"))), self.expect_shell, places=3)
        other = S.make_project(chat_id=f"20261001-000000-{uuid.uuid4().hex[:6]}")
        other.save_spec(self.spec)
        tool = X.export_tips(other, self.doc.Name, formats=["step"])
        by_label = {f.label: Part.read(f.path) for f in tool.files}
        self.assertAlmostEqual(self.com(by_label["Cap"]), self.com(auto), places=6)
        self.assertAlmostEqual(by_label["Cap"].BoundBox.ZMax, auto.BoundBox.ZMax, places=6)
        self.assertEqual(self.proj.manifest["exported_after_step"], 1)


# ---------------------------------------------------------------------------
# artifacts for the panel

class RealArtifactListTest(RealDoc):
    def test_a_finished_work_turn_lists_everything_the_strip_needs(self):
        from oscar import panel_render

        doc = build_model(unique("T3R"))
        S.record(self.proj, doc=doc.Name)
        self.proj.finalize(ledger={"intent": "generic_part", "steps": [], "requirements": []}, outcome="answered",
                           validation=None)
        entries = self.proj.artifact_list()
        kinds = [e["kind"] for e in entries]
        for kind in ("fcstd", "notes", "rebuild", "export", "snapshot"):
            self.assertIn(kind, kinds)
        self.assertEqual(kinds.count("export"), 4)          # STEP + STL of each visible tip (Box and Cyl)
        for entry in entries:
            self.assertTrue(os.path.isfile(entry["path"]), entry)
            self.assertEqual(sorted(entry), ["kind", "name", "path"])
        html = panel_render.artifacts_strip_html({"project_dir": self.proj.root, "files": entries})
        for action in ("open-doc", "open-notes", "open-rebuild", "open-exports", "open-dir"):
            self.assertIn(f"oscar://artifact/{action}", html)


if __name__ == "__main__":
    unittest.main()
