# SPDX-License-Identifier: Apache-2.0
"""Issue #5 — the imported source STL stays visible next to the Body it was rebuilt into.

The behaviour under test: at the end of a FINISHED work turn (outcome ``validated`` / ``answered``) that
added a visible solid deliverable to a document, ``Project.finalize`` hides the meshes the project's
``import_file`` steps brought into THAT document (provenance, not type: a mesh nobody imported through
Oscar — a fit-check scan placed next to an existing part, or reference geometry — is never touched). The
mesh object stays in the tree, each hidden name becomes a ``decision`` note, the hidden state lands in
``model.FCStd``, and a capped or unverified build keeps the mesh visible (it is still the reference the
next round needs). Reverting the turn restores the visibility: the engine's turn snapshot records it and
``revert_turn`` puts it back.

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
from oscar.project import Project  # noqa: E402


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


def _add_scripted_mesh(doc, name="ScanPy"):
    """A visible mesh of a DERIVED type (``Mesh::FeaturePython``): counted as geometry, so it must hide too."""
    mesh = doc.addObject("Mesh::FeaturePython", name)
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


LEDGER = {"intent": "generic_part", "request": "rebuild Base.STL", "steps": [], "requirements": []}


def _rebuild_turn(proj, doc, mesh, solid, turn="T20261116-000001-0001", outcome="validated"):
    """The issue #5 flow as the journal sees it: one turn imports the STL, then builds the Body."""
    proj.begin_turn(turn, "Rebuild Base.STL", work=True)
    S.record(proj, "import_file", title="Import Base.STL", args={"path": "/tmp/Base.stl"},
             doc=doc.Name, delta={"added": [mesh.Name], "removed": [], "changed": []}, turn_id=turn)
    S.record(proj, "create_object", title="Build the Body",
             args={"spec": {"Type": "Part::Feature", "Name": solid.Name}}, doc=doc.Name,
             delta={"added": [solid.Name], "removed": [], "changed": []}, turn_id=turn)
    return proj.finalize(ledger=dict(LEDGER), outcome=outcome, validation=None, export=False)


class HideSourceMeshesTest(_DocTestCase):
    """``project_gui.hide_source_meshes``: the GUI-thread rule itself (provenance-filtered)."""

    def test_a_candidate_mesh_is_hidden_once_a_solid_tip_exists(self):
        doc = _new_doc("HideDoc")
        mesh = _add_mesh(doc)
        solid = _add_solid(doc)
        doc.recompute()
        hidden = gui.hide_source_meshes(doc, [mesh.Name])
        self.assertEqual(hidden, [mesh.Name])
        self.assertFalse(mesh.Visibility)
        self.assertTrue(solid.Visibility)

    def test_a_mesh_that_is_not_a_candidate_stays_visible(self):
        doc = _new_doc("FitCheck")
        _add_solid(doc, "Enclosure")          # the user's existing part
        scan = _add_mesh(doc, "PCB_scan")     # imported for a fit check, never a rebuild source
        doc.recompute()
        self.assertEqual(gui.hide_source_meshes(doc, []), [])
        self.assertTrue(scan.Visibility)

    def test_a_mesh_without_a_solid_tip_stays_visible(self):
        doc = _new_doc("MeshOnly")
        mesh = _add_mesh(doc)
        doc.recompute()
        self.assertEqual(gui.hide_source_meshes(doc, [mesh.Name]), [])
        self.assertTrue(mesh.Visibility)

    def test_no_solid_tip_when_the_only_solid_is_hidden(self):
        doc = _new_doc("HiddenTip")
        _add_mesh(doc)
        solid = _add_solid(doc)
        solid.Visibility = False
        doc.recompute()
        self.assertEqual(gui.hide_source_meshes(doc, ["Base"]), [])

    def test_an_already_hidden_mesh_is_not_reported_again(self):
        doc = _new_doc("AlreadyHidden")
        mesh = _add_mesh(doc)
        mesh.Visibility = False
        _add_solid(doc)
        doc.recompute()
        self.assertEqual(gui.hide_source_meshes(doc, ["Base"]), [])

    def test_derived_mesh_types_are_hidden_too(self):
        """``is_geometry`` accepts any type derived from ``Mesh::Feature``; the hide rule must not be narrower."""
        doc = _new_doc("Scripted")
        scan = _add_scripted_mesh(doc)
        _add_solid(doc)
        doc.recompute()
        self.assertEqual(gui.hide_source_meshes(doc, [scan.Name]), [scan.Name])
        self.assertFalse(scan.Visibility)

    def test_a_candidate_that_is_no_longer_a_mesh_is_ignored(self):
        """An import that arrived as a solid (``as_solid``) has nothing to hide: the name resolves, the type says no."""
        doc = _new_doc("Converted")
        solid = _add_solid(doc, "Base")       # used to be the mesh, converted on import
        _add_solid(doc, "Body")
        doc.recompute()
        self.assertEqual(gui.hide_source_meshes(doc, [solid.Name]), [])
        self.assertTrue(solid.Visibility)


class FinalizeHidesSourceMeshTest(_DocTestCase):
    """``Project.finalize`` hides the rebuild's source mesh before the autosave, and only then."""

    def setUp(self):
        super().setUp()
        self.proj = S.make_project(title="Rebuild Base.STL")

    def test_a_finished_rebuild_hides_the_source_mesh_notes_it_and_saves_the_hidden_state(self):
        doc = _new_doc("Rebuild")
        mesh = _add_mesh(doc)
        solid = _add_solid(doc)
        doc.recompute()
        summary = _rebuild_turn(self.proj, doc, mesh, solid)
        self.assertTrue(summary.markdown.startswith("## Build Summary"))
        self.assertFalse(mesh.Visibility)          # the fix: only the rebuilt-from mesh is hidden
        self.assertTrue(solid.Visibility)
        notes = [n for n in self.proj.read_notes() if n["kind"] == "decision"]
        self.assertEqual(len(notes), 1)
        self.assertIn(mesh.Name, notes[-1]["text"])
        self.assertIn("hidden", notes[-1]["text"])
        saved = App.openDocument(self.proj.path("model.FCStd"))   # the autosave ran AFTER the hide
        try:
            self.assertFalse(saved.getObject(mesh.Name).Visibility)
            self.assertTrue(saved.getObject(solid.Name).Visibility)
        finally:
            App.closeDocument(saved.Name)

    def test_an_answered_rebuild_also_hides_the_source(self):
        """The gate passed without a validation record: still a finished build (outcome ``answered``)."""
        doc = _new_doc("Answered")
        mesh = _add_mesh(doc)
        solid = _add_solid(doc)
        doc.recompute()
        _rebuild_turn(self.proj, doc, mesh, solid, outcome="answered")
        self.assertFalse(mesh.Visibility)

    def test_a_capped_build_keeps_the_source_mesh_visible(self):
        """The round cap hit with required steps open: the half-built model needs its reference."""
        doc = _new_doc("Capped")
        mesh = _add_mesh(doc)
        _add_solid(doc, "PartialBody")       # half-built when the cap hit
        doc.recompute()
        self.proj.begin_turn("T20261116-000002-0002", "Rebuild Base.STL", work=True)
        S.record(self.proj, "import_file", title="Import Base.STL", args={"path": "/tmp/Base.stl"},
                 doc=doc.Name, delta={"added": [mesh.Name], "removed": [], "changed": []},
                 turn_id="T20261116-000002-0002")
        S.record(self.proj, "create_object", title="Partial build", doc=doc.Name,
                 delta={"added": ["PartialBody"], "removed": [], "changed": []},
                 turn_id="T20261116-000002-0002")
        self.proj.finalize(ledger=dict(LEDGER), outcome="capped", validation=None, export=False)
        self.assertTrue(mesh.Visibility)
        self.assertFalse([n for n in self.proj.read_notes() if "hidden" in n.get("text", "")])

    def test_an_unverified_build_keeps_the_source_mesh_visible(self):
        """The finish gate failed but some geometry was produced: not a finished reconstruction."""
        doc = _new_doc("Unverified")
        mesh = _add_mesh(doc)
        solid = _add_solid(doc)
        doc.recompute()
        _rebuild_turn(self.proj, doc, mesh, solid, outcome="unverified")
        self.assertTrue(mesh.Visibility)

    def test_an_unrelated_mesh_beside_an_existing_solid_stays_visible(self):
        """A scan imported next to the user's enclosure is never a rebuild source (import-only turn)."""
        doc = _new_doc("FitCheck")
        _add_solid(doc, "Enclosure")            # existed before the turn
        scan = _add_mesh(doc, "PCB_scan")       # imported THIS turn, for a fit check
        doc.recompute()
        self.proj.begin_turn("T20261116-000003-0003", "Fit check", work=True)
        S.record(self.proj, "import_file", title="Import PCB_scan.stl", args={"path": "/tmp/PCB_scan.stl"},
                 doc=doc.Name, delta={"added": [scan.Name], "removed": [], "changed": []},
                 turn_id="T20261116-000003-0003")
        self.proj.finalize(ledger={"intent": "generic_part", "request": "import the scan next to the enclosure",
                                   "steps": [], "requirements": []},
                           outcome="answered", validation=None, export=False)
        self.assertTrue(scan.Visibility)        # no solid was ADDED this turn: nothing was rebuilt
        self.assertFalse([n for n in self.proj.read_notes() if "hidden" in n.get("text", "")])

    def test_a_mesh_only_import_and_measurement_turn_hides_nothing(self):
        doc = _new_doc("MeasureOnly")
        mesh = _add_mesh(doc)
        doc.recompute()
        self.proj.begin_turn("T20261116-000004-0004", "Measure Base.STL", work=True)
        S.record(self.proj, "import_file", title="Import Base.STL", args={"path": "/tmp/Base.stl"},
                 doc=doc.Name, delta={"added": [mesh.Name], "removed": [], "changed": []},
                 turn_id="T20261116-000004-0004")
        self.proj.finalize(ledger=dict(LEDGER), outcome="answered", validation=None, export=False)
        self.assertTrue(mesh.Visibility)

    def test_every_source_of_a_multi_part_rebuild_is_hidden_and_only_those(self):
        """Two imported source meshes, both rebuilt this turn, plus an untouched third mesh: the first two
        hide, the third stays."""
        doc = _new_doc("TwoParts")
        base = _add_mesh(doc, "Base")
        lid = _add_mesh(doc, "Lid")
        spare = _add_mesh(doc, "Spare_scan")       # imported by the user before Oscar met the document
        _add_solid(doc, "BaseBody")
        _add_solid(doc, "LidBody")
        doc.recompute()
        turn = "T20261116-000010-0010"
        self.proj.begin_turn(turn, "Rebuild Base and Lid", work=True)
        S.record(self.proj, "import_file", title="Import Base.stl", args={"path": "/tmp/Base.stl"},
                 doc=doc.Name, delta={"added": [base.Name], "removed": [], "changed": []}, turn_id=turn)
        S.record(self.proj, "import_file", title="Import Lid.stl", args={"path": "/tmp/Lid.stl"},
                 doc=doc.Name, delta={"added": [lid.Name], "removed": [], "changed": []}, turn_id=turn)
        S.record(self.proj, "create_object", title="Build both bodies", doc=doc.Name,
                 delta={"added": ["BaseBody", "LidBody"], "removed": [], "changed": []}, turn_id=turn)
        self.proj.finalize(ledger=dict(LEDGER), outcome="validated", validation=None, export=False)
        self.assertFalse(base.Visibility)
        self.assertFalse(lid.Visibility)
        self.assertTrue(spare.Visibility)
        hidden_notes = [n["text"] for n in self.proj.read_notes() if n["kind"] == "decision"]
        self.assertEqual(sorted(n.split("'")[1] for n in hidden_notes), ["Base", "Lid"])

    def test_only_the_document_the_turn_built_in_is_cleaned_up(self):
        """A rebuild in a second document is tidied there even though ``active_doc`` names the first."""
        first, second = _new_doc("First"), _new_doc("Second")
        keep = _add_mesh(first, "OldScan")      # imported earlier, its doc untouched by this turn
        mesh = _add_mesh(second, "Base")
        solid = _add_solid(second, "Body")
        first.recompute(), second.recompute()
        self.proj.begin_turn("T20261116-000005-0005", "Rebuild in the second document", work=True)
        S.record(self.proj, "import_file", title="Import OldScan", args={"path": "/tmp/OldScan.stl"},
                 doc=first.Name, delta={"added": [keep.Name], "removed": [], "changed": []},
                 turn_id="T20261116-000005-0005")
        S.record(self.proj, "import_file", title="Import Base.STL", args={"path": "/tmp/Base.stl"},
                 doc=second.Name, delta={"added": [mesh.Name], "removed": [], "changed": []},
                 turn_id="T20261116-000005-0005")
        S.record(self.proj, "create_object", title="Build the Body", doc=second.Name,
                 delta={"added": [solid.Name], "removed": [], "changed": []},
                 turn_id="T20261116-000005-0005")
        self.proj.manifest["active_doc"] = first.Name     # what finalize used to rely on alone
        self.proj.finalize(ledger=dict(LEDGER), outcome="validated", validation=None, export=False)
        self.assertTrue(keep.Visibility)        # its document got no deliverable this turn
        self.assertFalse(mesh.Visibility)

    def test_a_reshown_reference_mesh_is_not_hidden_again_on_later_turns(self):
        """The user re-shows the mesh to compare: an unrelated later turn must not hide it again."""
        doc = _new_doc("Reuse")
        mesh = _add_mesh(doc)
        solid = _add_solid(doc)
        doc.recompute()
        _rebuild_turn(self.proj, doc, mesh, solid)
        self.assertFalse(mesh.Visibility)
        mesh.Visibility = True                  # the user shows the reference again
        later = "T20261116-000006-0006"
        self.proj.begin_turn(later, "Make it 2 mm taller", work=True)
        S.record(self.proj, "edit_object", title="Raise the body", doc=doc.Name,
                 delta={"added": [], "removed": [], "changed": [{"name": solid.Name, "props": ["Height"]}]},
                 turn_id=later)
        self.proj.finalize(ledger=dict(LEDGER), outcome="validated", validation=None, export=False)
        self.assertTrue(mesh.Visibility)        # no solid was added: the mesh stays as the working reference
        decisions = [n for n in self.proj.read_notes() if n["kind"] == "decision"]
        self.assertEqual(len(decisions), 1)     # and no second decision note piles up

    def test_a_failure_of_the_hide_step_never_skips_the_autosave(self):
        """A dying ``note()`` (disk full) must not cost the autosave, NOTES.md or the finalized record."""
        doc = _new_doc("Boom")
        mesh = _add_mesh(doc)
        _add_solid(doc)
        doc.recompute()
        self.proj.begin_turn("T20261116-000007-0007", "Rebuild Base.STL", work=True)
        S.record(self.proj, "import_file", title="Import Base.STL", args={"path": "/tmp/Base.stl"},
                 doc=doc.Name, delta={"added": [mesh.Name], "removed": [], "changed": []},
                 turn_id="T20261116-000007-0007")
        S.record(self.proj, "create_object", title="Build the Body", doc=doc.Name,
                 delta={"added": ["Body"], "removed": [], "changed": []},
                 turn_id="T20261116-000007-0007")
        with mock.patch.object(Project, "note", side_effect=OSError("disk full")):
            summary = self.proj.finalize(ledger=dict(LEDGER), outcome="validated", validation=None, export=False)
        self.assertTrue(summary.markdown.startswith("## Build Summary"))
        self.assertFalse(mesh.Visibility)                                # the GUI step had already run
        self.assertTrue(os.path.isfile(self.proj.path("model.FCStd")))   # the autosave was NOT skipped
        self.assertTrue(os.path.isfile(self.proj.path("NOTES.md")))
        self.assertEqual(self.proj.manifest["finalized"]["outcome"], "validated")

    def test_finalize_without_source_meshes_writes_no_note(self):
        self.proj.finalize(ledger=dict(LEDGER), outcome="validated", validation=None, export=False)
        self.assertFalse([n for n in self.proj.read_notes() if "hidden" in n.get("text", "")])

    def test_a_reconstruction_completed_in_a_later_turn_still_hides_the_source(self):
        """The build turn hit the round cap; the NEXT turn only verifies and adds nothing: completing the
        reconstruction must still hide the source mesh (the deliverable was built after the import)."""
        doc = _new_doc("LaterTurn")
        mesh = _add_mesh(doc)
        solid = _add_solid(doc)
        doc.recompute()
        first = "T20261118-000001-0001"
        self.proj.begin_turn(first, "Rebuild Base.STL", work=True)
        S.record(self.proj, "import_file", title="Import Base.STL", args={"path": "/tmp/Base.stl"},
                 doc=doc.Name, delta={"added": [mesh.Name], "removed": [], "changed": []}, turn_id=first)
        S.record(self.proj, "create_object", title="Build the Body", doc=doc.Name,
                 delta={"added": [solid.Name], "removed": [], "changed": []}, turn_id=first)
        self.proj.finalize(ledger=dict(LEDGER), outcome="capped", validation=None, export=False)
        self.assertTrue(mesh.Visibility)           # the capped turn kept the reference visible
        second = "T20261118-000002-0002"
        self.proj.begin_turn(second, "Verify the rebuild", work=True)
        S.record(self.proj, "validate_model", title="Verify", doc=doc.Name,
                 delta={"added": [], "removed": [], "changed": []}, turn_id=second,
                 geometry=True, mutating=True)
        self.proj.finalize(ledger=dict(LEDGER), outcome="validated", validation=None, export=False)
        self.assertFalse(mesh.Visibility)          # completion hides the source, no matter the build's turn
        self.assertIn(mesh.Name, (self.proj.manifest.get("hidden_source_meshes") or {}).get(doc.Name, []))

    def test_the_native_reconstruction_outcomes_hide_their_sources(self):
        """The reconstruction pipeline's own verdicts (ported trees finalize with them) hide too;
        an unverified reconstruction keeps the mesh visible."""
        doc = _new_doc("NativeRecon")
        mesh = _add_mesh(doc)
        solid = _add_solid(doc)
        doc.recompute()
        for outcome, hidden_now in (("verified_parametric", True),
                                    ("partially_verified_parametric", True),
                                    ("reconstruction_unverified", False)):
            with self.subTest(outcome=outcome):
                mesh.Visibility = True
                self.proj.manifest["hidden_source_meshes"] = {}
                turn = f"T20261118-000003-{outcome[:8]}"
                self.proj.begin_turn(turn, "Reconstruct", work=True)
                S.record(self.proj, "import_file", title="Import scan", args={"path": "/tmp/scan.stl"},
                         doc=doc.Name, delta={"added": [mesh.Name], "removed": [], "changed": []}, turn_id=turn)
                S.record(self.proj, "create_object", title="Parametric body", doc=doc.Name,
                         delta={"added": [solid.Name], "removed": [], "changed": []}, turn_id=turn)
                self.proj.finalize(ledger=dict(LEDGER), outcome=outcome, validation=None, export=False)
                self.assertEqual(mesh.Visibility, not hidden_now, outcome)

    def test_a_scan_imported_after_the_deliverable_never_hides(self):
        """A fit-check scan that enters the document AFTER the part was built cannot be that build's source."""
        doc = _new_doc("FitAfter")
        mesh = _add_mesh(doc)
        solid = _add_solid(doc)
        doc.recompute()
        turn = "T20261118-000004-0004"
        self.proj.begin_turn(turn, "Build, then check a scan against it", work=True)
        S.record(self.proj, "create_object", title="Build the Body", doc=doc.Name,
                 delta={"added": [solid.Name], "removed": [], "changed": []}, turn_id=turn)
        S.record(self.proj, "import_file", title="Import the fit-check scan", args={"path": "/tmp/scan.stl"},
                 doc=doc.Name, delta={"added": [mesh.Name], "removed": [], "changed": []}, turn_id=turn)
        self.proj.finalize(ledger=dict(LEDGER), outcome="validated", validation=None, export=False)
        self.assertTrue(mesh.Visibility)          # the deliverable predates the scan: not its source

    def test_after_a_revert_the_next_completed_rebuild_hides_again(self):
        """The hide-once record must not survive a revert: a rebuilt reconstruction hides its source again."""
        doc = _new_doc("RebuildAfterRevert")
        mesh = _add_mesh(doc)
        solid = _add_solid(doc)
        doc.recompute()
        _rebuild_turn(self.proj, doc, mesh, solid)
        self.assertFalse(mesh.Visibility)
        self.proj.note_reverted(1)                # the user reverts the build turn
        mesh.Visibility = True                    # the engine revert restored it; be explicit for the next turn
        again = "T20261118-000005-0005"
        self.proj.begin_turn(again, "Rebuild again", work=True)
        S.record(self.proj, "create_object", title="Rebuild the Body", doc=doc.Name,
                 delta={"added": [solid.Name], "removed": [], "changed": []}, turn_id=again)
        self.proj.finalize(ledger=dict(LEDGER), outcome="validated", validation=None, export=False)
        self.assertFalse(mesh.Visibility)         # hidden again: the record was cleared by the revert

    def test_recompute_and_reopen_keep_the_mesh_hidden_and_its_geometry_untouched(self):
        doc = _new_doc("Stable")
        mesh = _add_mesh(doc)
        _add_solid(doc)
        doc.recompute()
        facets_before = mesh.Mesh.CountFacets
        points_before = mesh.Mesh.CountPoints
        _rebuild_turn(self.proj, doc, mesh, _add_solid(doc, "Body2"))
        self.assertFalse(mesh.Visibility)
        doc.recompute()
        self.assertFalse(mesh.Visibility)         # a recompute never brings the source mesh back
        self.assertEqual(mesh.Mesh.CountFacets, facets_before)
        self.assertEqual(mesh.Mesh.CountPoints, points_before)
        self.assertIsNotNone(doc.getObject(mesh.Name))   # identity kept: never deleted, never replaced


class RevertRestoresVisibilityTest(_DocTestCase):
    """B3: a revert (undo stack or checkpoint) brings the source mesh back — no empty view."""

    def setUp(self):
        super().setUp()
        self.proj = S.make_project(title="Rebuild Base.STL")

    def test_reverting_the_turn_restores_the_source_mesh_visibility(self):
        """The undo-stack path: the engine snapshot records mesh visibility, ``revert_turn`` restores it."""
        from oscar.cad.engine import get_engine, txn_task

        doc = _new_doc("RevertDoc")
        mesh = _add_mesh(doc)
        doc.recompute()
        engine = get_engine()
        snapshot = engine.begin_turn_snapshot()               # turn start: mesh visible

        def build():
            box = doc.addObject("Part::Feature", "Rebuilt")
            box.Shape = Part.makeBox(20, 20, 20)
            return {"success": True}

        txn_task(build, "Oscar: create_object", doc.Name)()   # the turn's tool call, as the engine records it
        doc.recompute()
        _rebuild_turn(self.proj, doc, mesh, doc.getObject("Rebuilt"),
                      turn="T20261116-000008-0008")           # finalize hides the mesh at turn end
        self.assertFalse(mesh.Visibility)
        result = engine.revert_turn(snapshot)                 # the user presses "revert changes"
        self.assertTrue(result.get("success"), result)
        self.assertIsNone(doc.getObject("Rebuilt"), "the revert removed the rebuilt body")
        self.assertTrue(mesh.Visibility, "the source mesh must come back: the view must not be empty")

    def test_the_checkpoint_restore_path_returns_the_pre_turn_visibility(self):
        """The checkpoint fallback (undo-unsafe turns): restore replaces the content, visibility included."""
        doc = _new_doc("CkptDoc")
        mesh = _add_mesh(doc)
        mesh_name = mesh.Name                  # the wrapper dies with the restore's remove/copy cycle
        doc.recompute()
        self.proj.ensure_dir()
        ckpt = self.proj.checkpoint(doc.Name, label="prebuild")   # the turn-start state: mesh visible, no body
        _rebuild_turn(self.proj, doc, mesh, _add_solid(doc, "Body"),
                      turn="T20261116-000009-0009")
        self.assertFalse(mesh.Visibility)
        restored = self.proj.restore(doc.Name, ckpt)
        self.assertTrue(restored.ok, restored.error)
        self.assertIsNone(doc.getObject("Body"))
        self.assertTrue(doc.getObject(mesh_name).Visibility)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
