"""Issue 002 — the imported source STL stays visible after it was rebuilt into an editable Body.

The fix under test: at the end of a work turn ``Project.finalize`` hides every visible ``Mesh::Feature``
of the active document once that document holds at least one visible solid tip (the rebuilt body is the
deliverable; the mesh was only the trace-over source — the enclosure kit hides its reference the same
way). The mesh object stays in the tree, the hiding is recorded as a ``decision`` note, and a failure of
the new step can never break ``finalize``.

Against the real FreeCAD runtime in console mode: no GUI, no network.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

if "FreeCADGui" not in sys.modules:
    try:
        import FreeCADGui  # noqa: F401, E402
    except Exception:
        import types

        sys.modules["FreeCADGui"] = types.ModuleType("FreeCADGui")

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402

import project_support as S  # noqa: E402
from oscar import project_gui as gui  # noqa: E402


def _cube(size=20.0, at=(0.0, 0.0, 0.0)):
    """The 12 triangles of a closed cube, wound outward (same construction as test_import_file)."""
    x, y, z = at
    v = [(x + dx * size, y + dy * size, z + dz * size)
         for dx, dy, dz in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
    quads = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    return [tri for a, b, c, d in quads for tri in ((v[a], v[b], v[c]), (v[a], v[c], v[d]))]


def _add_mesh(doc, name="Base"):
    """A visible closed ``Mesh::Feature`` — what ``import_file`` made of the user's STL."""
    mesh = doc.addObject("Mesh::Feature", name)
    mesh.Mesh = Mesh.Mesh([App.Vector(*p) for tri in _cube() for p in tri])
    return mesh


def _add_solid(doc, name="Body"):
    """A visible solid — the rebuilt deliverable."""
    solid = doc.addObject("Part::Feature", name)
    solid.Shape = Part.makeBox(20, 20, 20)
    return solid


def _new_doc(name):
    doc = App.newDocument(name)
    doc.UndoMode = 1
    return doc


class _DocTestCase(unittest.TestCase):
    """Closes every document a test opened."""

    def setUp(self):
        self._before = set(App.listDocuments())

    def tearDown(self):
        for name in list(App.listDocuments()):
            if name not in self._before:
                try:
                    App.closeDocument(name)
                except Exception:  # noqa: BLE001
                    pass


class HideSourceMeshesTest(_DocTestCase):
    """``project_gui.hide_source_meshes``: the GUI-thread rule itself."""

    def test_a_visible_mesh_is_hidden_once_a_solid_tip_exists(self):
        doc = _new_doc("HideDoc")
        mesh = _add_mesh(doc)
        solid = _add_solid(doc)
        doc.recompute()
        hidden = gui.hide_source_meshes(doc)
        self.assertEqual(hidden, ["Base"])
        self.assertFalse(mesh.Visibility)
        self.assertTrue(solid.Visibility)

    def test_a_mesh_without_a_solid_tip_stays_visible(self):
        doc = _new_doc("MeshOnly")
        mesh = _add_mesh(doc)
        doc.recompute()
        self.assertEqual(gui.hide_source_meshes(doc), [])
        self.assertTrue(mesh.Visibility)

    def test_no_solid_tip_when_the_only_solid_is_hidden(self):
        doc = _new_doc("HiddenTip")
        _add_mesh(doc)
        solid = _add_solid(doc)
        solid.Visibility = False
        doc.recompute()
        self.assertEqual(gui.hide_source_meshes(doc), [])

    def test_an_already_hidden_mesh_is_not_reported_again(self):
        doc = _new_doc("AlreadyHidden")
        mesh = _add_mesh(doc)
        mesh.Visibility = False
        _add_solid(doc)
        doc.recompute()
        self.assertEqual(gui.hide_source_meshes(doc), [])


class FinalizeHidesSourceMeshTest(_DocTestCase):
    """``Project.finalize`` hides the source mesh before the autosave and records a note."""

    def test_finalize_hides_the_mesh_notes_it_and_saves_the_hidden_state(self):
        doc = _new_doc("Rebuild")
        mesh = _add_mesh(doc)
        solid = _add_solid(doc)
        doc.recompute()
        proj = S.make_project(title="Rebuild Base.STL")
        proj.manifest["active_doc"] = "Rebuild"
        summary = proj.finalize(ledger={"intent": "generic_part", "request": "rebuild Base.STL", "steps": [],
                                        "requirements": []}, outcome="validated", validation=None, export=False)
        self.assertTrue(summary.markdown.startswith("## Build Summary"))
        self.assertFalse(mesh.Visibility)          # the fix: the source mesh is no longer visible
        self.assertTrue(solid.Visibility)
        notes = proj.read_notes()
        self.assertEqual(notes[-1]["kind"], "decision")
        self.assertIn("Base", notes[-1]["text"])
        self.assertIn("hidden", notes[-1]["text"])
        saved = App.openDocument(proj.path("model.FCStd"))   # the autosave ran AFTER the hide
        try:
            self.assertFalse(saved.getObject("Base").Visibility)
            self.assertTrue(saved.getObject("Body").Visibility)
        finally:
            App.closeDocument(saved.Name)

    def test_finalize_without_a_solid_tip_hides_nothing(self):
        doc = _new_doc("MeasureOnly")
        mesh = _add_mesh(doc)
        doc.recompute()
        proj = S.make_project(title="Measure Base.STL")
        proj.manifest["active_doc"] = "MeasureOnly"
        proj.finalize(ledger={"intent": "inspect", "request": "measure Base.STL", "steps": [], "requirements": []},
                      outcome="answered", validation=None, export=False)
        self.assertTrue(mesh.Visibility)
        self.assertFalse([n for n in proj.read_notes() if "hidden" in n.get("text", "")])

    def test_a_failure_of_the_hide_step_never_breaks_finalize(self):
        doc = _new_doc("Boom")
        mesh = _add_mesh(doc)
        _add_solid(doc)
        doc.recompute()
        proj = S.make_project(title="Rebuild Base.STL")
        proj.manifest["active_doc"] = "Boom"
        with mock.patch.object(gui, "hide_source_meshes", side_effect=RuntimeError("boom")):
            summary = proj.finalize(ledger={"intent": "generic_part", "request": "rebuild", "steps": [],
                                            "requirements": []}, outcome="validated", validation=None, export=False)
        self.assertTrue(summary.markdown.startswith("## Build Summary"))
        self.assertTrue(mesh.Visibility)           # nothing was changed by the failed step

    def test_finalize_without_an_active_document_writes_no_note(self):
        proj = S.make_project(title="No Doc")
        proj.finalize(ledger={"intent": "other", "request": "hello", "steps": [], "requirements": []},
                      outcome="answered", validation=None, export=False)
        self.assertFalse([n for n in proj.read_notes() if "hidden" in n.get("text", "")])
