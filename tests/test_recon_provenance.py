"""K3: ``oscar.recon.provenance`` — the four-category ledger (original / modified /
synthetic / removed), coordinate-keyed so renumbering cannot hide a change, verified
against the mesh (UNCERTAIN when the mapping degrades), consumed by ``recognize.survey``
(evidence tags + synthetic-dimension flags) and persisted as the project's
``analysis/<mesh>.json`` evidence file and the ``OscarProvenance`` object property.

Fixtures are real meshes; save/reopen behaviour is exercised on real FCStd files.
"""

import json
import os
import sys
import tempfile
import unittest
import unittest.mock as mock  # noqa: F401  (hardening failure-injection tests)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402

from oscar.recon import meshops, provenance, recognize  # noqa: E402


def box_mesh(size=20.0, tess=2.0):
    pts, facs = Part.makeBox(size, size, size / 4.0).tessellate(tess)
    return Mesh.Mesh((list(pts), [tuple(f) for f in facs]))


def holed_box():
    # G6: the fill policy fills only tiny flat pinholes, so the shared fixture is a fine
    # grid sheet with one quad missing (the old coarse-box corner hole — unscanned-face
    # class, >= 2% of the area — is now SKIPPED by design, never auto-filled)
    import numpy as np

    n = 21
    xs = np.linspace(0.0, 20.0, n)
    X, Y = np.meshgrid(xs, xs)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n * n)])
    facs = []
    for i in range(n - 1):
        for j in range(n - 1):
            if (i, j) == (10, 10):
                continue
            a, b, c, e = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            facs += [(a, b, e), (a, e, c)]
    return Mesh.Mesh(([App.Vector(*p) for p in pts], facs))


class DiffTest(unittest.TestCase):
    def test_a_fill_creates_synthetic_facets_and_touches_nothing_else(self):
        before = holed_box()
        after = before.copy()
        after.fillupHoles(40)
        diff = provenance.diff_mesh(before, after)
        self.assertGreater(len(diff["added_facets"]), 0)
        self.assertEqual(len(diff["removed_facets"]), 0)
        self.assertEqual(len(diff["moved_vertices"]), 0)

    def test_a_vertex_move_is_detected_as_modified(self):
        before = box_mesh()
        after = before.copy()
        after.setPoint(0, App.Vector(0.01, 0.0, 0.0))
        diff = provenance.diff_mesh(before, after)
        self.assertEqual(len(diff["moved_vertices"]), 1)
        self.assertGreater(len(diff["modified_facets"]), 0,
                           "facets touching the moved vertex are modified")
        self.assertEqual(len(diff["added_facets"]), 0)

    def test_facet_removal_is_detected(self):
        before = box_mesh()
        pts, facs = before.Topology
        after = Mesh.Mesh((list(pts), [tuple(f) for f in facs][:-1]))
        diff = provenance.diff_mesh(before, after)
        self.assertEqual(len(diff["removed_facets"]), 1)
        # the removed triangle's vertices stay referenced by their neighbours here, so
        # removed_vertices counts only UNREFERENCED losses (documented semantics)


class LedgerTest(unittest.TestCase):
    def test_repair_returns_a_ledger_with_all_four_categories(self):
        result = meshops.repair(holed_box(), authorized=["fillupHoles"])
        ledger = result["provenance"]
        for cat in ("modified", "synthetic", "removed", "operations"):
            self.assertIn(cat, ledger)
        self.assertGreater(len(ledger["synthetic"]), 0, "the fill is synthetic")
        self.assertEqual(result["provenance_verification"]["verdict"], "VERIFIED")

    def test_untouched_geometry_is_never_claimed_as_synthetic(self):
        original = holed_box()
        result = meshops.repair(original, authorized=["fillupHoles"])
        synthetic = set(result["provenance"]["synthetic"])
        # facets present in BOTH the input and the repaired copy were not created by any
        # operation — they must never be classified synthetic
        unchanged = set(provenance.facet_keys(original)) & \
            set(provenance.facet_keys(result["mesh"]))
        self.assertTrue(unchanged, "the fixture must keep some untouched facets")
        self.assertFalse(synthetic & unchanged,
                         "untouched original facets must stay original evidence")

    def test_evidence_class_semantics(self):
        ledger = provenance.new_ledger()
        ledger["synthetic"] = [("a",), ("b",)]
        self.assertEqual(provenance.evidence_class([("a",), ("b",)], ledger), "synthetic")
        self.assertEqual(provenance.evidence_class([("a",), ("c",)], ledger), "mixed")
        self.assertEqual(provenance.evidence_class([("c",)], ledger), "original")
        self.assertEqual(provenance.evidence_class([("c",)], None), "original")


class SurveyTaggingTest(unittest.TestCase):
    def test_regions_are_tagged_only_when_provenance_is_supplied(self):
        plain = recognize.survey(box_mesh())
        self.assertTrue(all("evidence" not in r for r in plain["regions"]))

    def test_synthetic_ledger_keys_tag_their_region(self):
        # G6 note: policy fills are TINY (<= 8 edges), so a fill inside one large flat
        # region can never dominate the survey's 20-facet evidence sample — the tagging
        # mechanism is therefore pinned with a hand-built synthetic ledger over a clean
        # sheet (the same contract repair feeds), not through a repair call
        ledger = provenance.new_ledger()
        mesh = holed_box()
        ledger["synthetic"] = provenance.facet_keys(mesh)[:20]
        survey = recognize.survey(mesh, provenance=ledger)
        tagged = [r for r in survey["regions"] if r.get("evidence")]
        self.assertTrue(tagged, "the ledger's synthetic facets must tag their region")
        self.assertTrue(all(r["evidence"] in ("mixed", "synthetic") for r in tagged))
        # every untagged region is original evidence:
        untagged = [r for r in survey["regions"] if not r.get("evidence")]
        self.assertTrue(all(r.get("evidence") is None for r in untagged))


class VerificationTest(unittest.TestCase):
    def test_bit_exact_reopen_keeps_provenance_verified(self):
        result = meshops.repair(holed_box(), authorized=["fillupHoles"])
        ledger = result["provenance"]
        tmp = tempfile.mkdtemp(prefix="k3prov_")
        doc = App.newDocument("k3v")
        obj = doc.addObject("Mesh::Feature", "Repaired")
        obj.Mesh = result["mesh"]
        obj.addProperty("App::PropertyString", "OscarProvenance", "Oscar", "digest")
        obj.OscarProvenance = json.dumps(
            provenance.digest_for_property(ledger, result["provenance_verification"]),
            default=str)
        path = os.path.join(tmp, "prov.FCStd")
        doc.saveAs(path)
        App.closeDocument(doc.Name)
        d2 = App.openDocument(path)
        back = d2.getObject("Repaired")
        # the property survives…
        digest = json.loads(back.OscarProvenance)
        self.assertEqual(digest["counts"]["synthetic"], len(ledger["synthetic"]))
        # …and the recorded keys still identify the same geometry:
        v = provenance.verify_against_mesh(ledger, back.Mesh)
        self.assertEqual(v["verdict"], "VERIFIED")
        self.assertEqual(v["confidence"], 1.0)
        App.closeDocument(d2.Name)

    def test_drifted_geometry_reports_uncertain(self):
        result = meshops.repair(holed_box(), authorized=["fillupHoles"])
        ledger = result["provenance"]
        self.assertTrue(ledger["synthetic"])
        moved = result["mesh"].copy()
        pts, facs = moved.Topology
        # destroy the synthetic facets: rebuild without them
        syn = set(ledger["synthetic"])
        keys = provenance.facet_keys(moved)
        keep = [tuple(f) for f, k in zip(facs, keys) if k not in syn]
        if len(keep) == len(facs):                     # facet-set rebuild dropped none:
            self.skipTest("kernel rebuild kept all facets")  # cannot simulate drift
        stripped = Mesh.Mesh((list(pts), keep))
        v = provenance.verify_against_mesh(ledger, stripped)
        self.assertEqual(v["verdict"], "UNCERTAIN")
        self.assertLess(v["confidence"], provenance.VERIFY_CONFIDENCE)


class AnalysisFileTest(unittest.TestCase):
    def test_project_write_analysis_persists_the_payload(self):
        from oscar.project import Project

        root = os.path.join(tempfile.mkdtemp(prefix="k3proj_"), "proj")
        proj = Project(root, create=True)
        payload = provenance.new_ledger()
        payload["diagnostics"] = {"closed": True}
        path = proj.write_analysis("Base Scan", payload)
        self.assertTrue(path and os.path.isfile(path))
        self.assertIn("analysis", path)
        with open(path, encoding="utf-8") as fh:
            back = json.load(fh)
        self.assertEqual(back["diagnostics"]["closed"], True)

    def test_analysis_payload_covers_identity_quality_and_provenance(self):
        from oscar.tools_recon import _analysis_payload

        doc = App.newDocument("k3payload")
        self.addCleanup(App.closeDocument, "k3payload")
        obj = doc.addObject("Mesh::Feature", "Scan")
        obj.Mesh = box_mesh()
        if "OscarImportSha256" not in obj.PropertiesList:
            obj.addProperty("App::PropertyString", "OscarImportSha256", "Oscar", "src")
        obj.OscarImportSha256 = "abc123"
        from oscar.recon import quality as _q
        survey = recognize.survey(obj.Mesh, quality=_q.estimate_tolerances(obj.Mesh))
        payload = _analysis_payload("analyse", obj, "k3payload", meshops.diagnose(obj.Mesh),
                                    survey, None, None)
        self.assertEqual(payload["source_sha256"], "abc123")
        self.assertEqual(payload["mesh"], "Scan")
        self.assertIn("sigma_mm", payload["quality"])
        self.assertIn("region_digest", payload)
        self.assertEqual(payload["region_digest"]["count"], len(survey["regions"]))


class H1VerifyTest(unittest.TestCase):
    """Complete vs partial verification; removed-reappearance; truncation honesty."""
    def _bare_ledger(self, **kw):
        led = provenance.new_ledger()
        led.update(kw)
        return led

    def test_all_keys_match_without_digest_is_verified_complete(self):
        m = box_mesh()
        led = self._bare_ledger(synthetic=provenance.facet_keys(m)[:3], modified=[],
                                removed=[], operations=[{"operation": "x"}])
        v = provenance.verify_against_mesh(led, m)
        self.assertEqual((v["verdict"], v["scope"]), ("VERIFIED", "complete"))

    def test_a_missing_modified_key_is_uncertain_with_reason(self):
        m = box_mesh()
        keys = provenance.facet_keys(m)
        led = self._bare_ledger(synthetic=[], modified=keys[:2] + [((9e9, 0, 0) * 3)],
                                removed=[], operations=[{"operation": "x"}])
        v = provenance.verify_against_mesh(led, m)
        self.assertEqual(v["verdict"], "UNCERTAIN")
        self.assertTrue(any("no longer present" in r for r in v["reasons"]))

    def test_a_removed_facet_that_reappears_is_uncertain(self):
        m = box_mesh()
        keys = provenance.facet_keys(m)
        led = self._bare_ledger(synthetic=[], modified=[], removed=keys[:1],
                                operations=[{"operation": "x"}])
        v = provenance.verify_against_mesh(led, m)      # key IS present → reappeared
        self.assertEqual(v["verdict"], "UNCERTAIN")
        self.assertTrue(any("reappeared" in r for r in v["reasons"]))

    def test_reordered_facets_verify(self):
        pts, facs = box_mesh().Topology
        a = Mesh.Mesh((list(pts), [tuple(f) for f in facs]))
        b = Mesh.Mesh((list(pts), [tuple(f) for f in reversed(facs)]))
        self.assertEqual(provenance.mesh_digest(a), provenance.mesh_digest(b),
                         "identity is coordinate-based; order is irrelevant")
        led = self._bare_ledger(synthetic=provenance.facet_keys(a)[:2], modified=[],
                                removed=[], operations=[{"operation": "x"}])
        self.assertEqual(provenance.verify_against_mesh(led, b)["verdict"], "VERIFIED")

    def test_duplicated_facets_degrade_identity_to_uncertain(self):
        # two triangles sharing an edge whose third vertices sit 2e-6 apart: the kernel
        # KEEPS both facets, but their coordinate keys collide (one quantum cell)
        pts = [App.Vector(0, 0, 0), App.Vector(10, 0, 0), App.Vector(0, 10, 0),
               App.Vector(2e-6, 0, 0)]
        m = Mesh.Mesh((pts, [(0, 1, 2), (3, 1, 2)]))
        self.assertGreaterEqual(provenance.identity_health(m)["duplicate_facet_keys"], 1)
        led = self._bare_ledger(synthetic=[], modified=[], removed=[],
                                operations=[{"operation": "x"}])
        v = provenance.verify_against_mesh(led, m)
        self.assertEqual(v["verdict"], "UNCERTAIN")
        self.assertTrue(any("duplicate facet key" in r for r in v["reasons"]))

    def test_truncated_key_lists_are_never_silently_verified(self):
        m = box_mesh()
        led = self._bare_ledger(synthetic=provenance.facet_keys(m)[:2], modified=[],
                                removed=[], operations=[{"operation": "x"}],
                                truncated=True)
        v = provenance.verify_against_mesh(led, m)
        self.assertEqual(v["verdict"], "UNCERTAIN")
        self.assertEqual(v["scope"], "partial")
        self.assertTrue(any("truncated" in r for r in v["reasons"]))

    def test_digest_tier_verifies_completely_even_with_truncation(self):
        result = meshops.repair(holed_box(), authorized=["fillupHoles"])
        led = result["provenance"]
        led["truncated"] = True                       # even so: digest matches → complete
        v = provenance.verify_against_mesh(led, result["mesh"])
        self.assertEqual((v["verdict"], v["scope"], v["method"]),
                         ("VERIFIED", "complete", "mesh-digest"))

    def test_post_reopen_edit_reports_uncertain(self):
        result = meshops.repair(holed_box(), authorized=["fillupHoles"])
        led = result["provenance"]
        tmp = tempfile.mkdtemp(prefix="h1_")
        doc = App.newDocument("h1")
        obj = doc.addObject("Mesh::Feature", "M")
        obj.Mesh = result["mesh"]
        path = os.path.join(tmp, "m.FCStd")
        doc.saveAs(path)
        App.closeDocument(doc.Name)
        d2 = App.openDocument(path)
        m2 = d2.getObject("M").Mesh
        self.assertEqual(provenance.verify_against_mesh(led, m2)["verdict"], "VERIFIED")
        m2_copy = m2.copy()
        m2_copy.setPoint(1, m2_copy.Topology[0][1] + App.Vector(0.02, 0, 0))
        v = provenance.verify_against_mesh(led, m2_copy)
        self.assertEqual(v["verdict"], "UNCERTAIN")
        self.assertTrue(any("changed after" in r for r in v["reasons"]))
        App.closeDocument(d2.Name)


class H2MoveSweepTest(unittest.TestCase):
    """Displacement sweep: real moves are modified, never synthetic."""
    def _move(self, delta):
        before = box_mesh()
        after = before.copy()
        after.setPoint(0, before.Topology[0][0] + App.Vector(delta, 0, 0))
        return provenance.diff_mesh(before, after)

    def test_confident_moves(self):
        for delta in (1e-5, 1e-3, 1e-2, 0.05):
            with self.subTest(delta=delta):
                d = self._move(delta)
                self.assertEqual(len(d["moved_vertices"]), 1)
                self.assertEqual(len(d["added_facets"]), 0, "a move is never synthetic")
                self.assertGreater(len(d["modified_facets"]), 0)

    def test_sub_quantum_moves_are_below_identity_resolution(self):
        d = self._move(1e-6)          # < half the 1e-5 quantum → same coordinate key
        self.assertEqual(len(d["moved_vertices"]), 0)
        self.assertEqual(len(d["added_facets"]), 0)   # and never synthetic either

    def test_a_0_1mm_nudge_is_a_low_confidence_move_not_synthetic(self):
        d = self._move(0.1)
        self.assertGreaterEqual(len(d["uncertain_moves"]), 1)
        self.assertEqual(len(d["moved_vertices"]), 1)
        self.assertEqual(len(d["added_facets"]), 0, "must not be classified synthetic")
        self.assertGreater(len(d["modified_facets"]), 0)

    def test_a_merge_onto_an_existing_vertex_is_a_move(self):
        # two triangles sharing an edge; the second's apex sits 0.028 mm from vertex 0 and
        # merges onto it (box corners are mm apart, so a merge needs a close pair)
        pts = [App.Vector(0, 0, 0), App.Vector(10, 0, 0), App.Vector(0, 10, 0),
               App.Vector(0.02, 0, 0.02)]
        before = Mesh.Mesh((pts, [(0, 1, 2), (3, 1, 2)]))
        after = before.copy()
        after.setPoint(3, pts[0])                       # apex merges onto vertex 0
        d = provenance.diff_mesh(before, after)
        self.assertGreaterEqual(len(d["moved_vertices"]), 1)
        self.assertEqual(len(d["added_facets"]), 0, "merge facets are modified, not created")
        self.assertEqual(len(d["removed_facets"]), 0)


class H3CollisionTest(unittest.TestCase):
    """Where coordinate identity degrades, verification says UNCERTAIN."""
    def test_nearly_coincident_vertices_share_a_key(self):
        pts = [App.Vector(0, 0, 0), App.Vector(10, 0, 0), App.Vector(0, 10, 0),
               App.Vector(0, 0, 2e-6)]                    # inside one 1e-5 quantum cell of v0
        m = Mesh.Mesh((pts, [(0, 1, 2), (3, 1, 2)]))
        h = provenance.identity_health(m)
        self.assertGreaterEqual(h["duplicate_vertex_keys"], 1)

    def test_repeated_facets_collapse_identity(self):
        # note: literally repeated facet TUPLES are deduplicated by the kernel at
        # construction (verified) — the constructible collision is two distinct facets
        # whose near-coincident third vertices quantize onto one key:
        pts = [App.Vector(0, 0, 0), App.Vector(10, 0, 0), App.Vector(0, 10, 0),
               App.Vector(0, 0, 2e-6)]
        m = Mesh.Mesh((pts, [(0, 1, 2), (3, 1, 2)]))
        h = provenance.identity_health(m)
        self.assertGreaterEqual(h["duplicate_facet_keys"], 1)
        self.assertGreaterEqual(h["duplicate_vertex_keys"], 1)

    def test_same_coordinates_different_winding_share_a_key(self):
        m1 = Mesh.Mesh([(App.Vector(0, 0, 0), App.Vector(10, 0, 0), App.Vector(0, 10, 0))])
        m2 = Mesh.Mesh([(App.Vector(0, 0, 0), App.Vector(0, 10, 0), App.Vector(10, 0, 0))])
        self.assertEqual(provenance.facet_keys(m1), provenance.facet_keys(m2),
                         "documented: winding/orientation is NOT part of coordinate identity")

    def test_scales_are_distinguished_by_digest(self):
        a = Part.makeBox(20, 20, 5).tessellate(2.0)
        ma = Mesh.Mesh((list(a[0]), [tuple(f) for f in a[1]]))
        b = Part.makeBox(0.02, 0.02, 0.005).tessellate(2.0)
        mb = Mesh.Mesh((list(b[0]), [tuple(f) for f in b[1]]))
        self.assertNotEqual(provenance.mesh_digest(ma), provenance.mesh_digest(mb))


class H4IntegrityTest(unittest.TestCase):
    """The analysis evidence file stays trustworthy."""
    def _project(self):
        from oscar.project import Project

        return Project(os.path.join(tempfile.mkdtemp(prefix="h4_"), "proj"), create=True)

    def test_repeated_runs_reparse_and_latest_wins(self):
        proj = self._project()
        p1 = proj.write_analysis("Scan", {"mode": "analyse", "v": 1})
        p2 = proj.write_analysis("Scan", {"mode": "repair", "v": 2})
        self.assertEqual(p1, p2, "one mesh → one evidence file, latest state")
        with open(p2, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["v"], 2)

    def test_mesh_digest_makes_stale_evidence_detectable(self):
        from oscar.tools_recon import _analysis_payload

        doc = App.newDocument("h4")
        self.addCleanup(App.closeDocument, "h4")
        obj = doc.addObject("Mesh::Feature", "Scan")
        obj.Mesh = box_mesh()
        p1 = _analysis_payload("analyse", obj, "h4", None, None, None, None)
        moved = obj.Mesh.copy()                      # the .Mesh property is an immutable view
        moved.setPoint(0, App.Vector(0.05, 0, 0))
        obj.Mesh = moved
        p2 = _analysis_payload("analyse", obj, "h4", None, None, None, None)
        self.assertNotEqual(p1["mesh_digest"], p2["mesh_digest"],
                            "a consumer can detect the file no longer describes this mesh")

    def test_failed_write_returns_none_and_does_not_raise(self):
        import oscar.project as project_mod

        proj = self._project()
        with mock.patch.object(project_mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            self.assertIsNone(proj.write_analysis("Scan", {"x": 1}))

    def test_same_mesh_name_in_two_projects_never_collides(self):
        pa, pb = self._project(), self._project()
        fa = pa.write_analysis("Scan", {"proj": "a"})
        fb = pb.write_analysis("Scan", {"proj": "b"})
        self.assertNotEqual(fa, fb)
        for path, tag in ((fa, "a"), (fb, "b")):
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["proj"], tag)


class H5ScenariosTest(unittest.TestCase):
    """Original-only / synthetic-only / mixed evidence and conservative uncertainty."""
    def _bracket(self):
        shape = Part.makeBox(60, 40, 10)
        for x, y in ((15, 10), (45, 10), (15, 30), (45, 30)):
            shape = shape.cut(Part.makeCylinder(3.3, 30, App.Vector(x, y, -10)))
        return Mesh.Mesh(shape.tessellate(0.3))

    def test_scenario_a_original_evidence_is_unflagged(self):
        s = recognize.survey(self._bracket())           # no repair → no ledger
        self.assertTrue(s["holes"])
        self.assertTrue(all(not h.get("dimension_basis") for h in s["holes"]))

    def test_scenario_b_synthetic_only_evidence_is_flagged_for_review(self):
        mesh = self._bracket()
        keys = provenance.facet_keys(mesh)
        # hole-wall facets have near-horizontal normals (radial); top/bottom planes are ±Z
        radial = [k for i, k in enumerate(keys)
                  if abs(list(mesh.Facets)[i].Normal.z) < 0.05]
        self.assertTrue(radial)
        led = provenance.new_ledger()
        led["synthetic"] = radial
        s = recognize.survey(mesh, provenance=led)
        flagged = [h for h in s["holes"] if h.get("dimension_basis")]
        self.assertTrue(flagged, "synthetic-supported holes must be flagged for review")

    def test_scenario_c_mixed_evidence_is_flagged(self):
        mesh = self._bracket()
        keys = provenance.facet_keys(mesh)
        radial = [k for i, k in enumerate(keys)
                  if abs(list(mesh.Facets)[i].Normal.z) < 0.05]
        led = provenance.new_ledger()
        led["synthetic"] = radial[:10]              # part of ONE wall → its region is mixed
        s = recognize.survey(mesh, provenance=led)
        self.assertTrue(any(r.get("evidence") == "mixed" for r in s["regions"]),
                        "partly-synthetic regions must read as mixed")

    def test_uncertain_provenance_demotes_the_outcome(self):
        from oscar.recon import meshops as mo, reconstruct as rc

        mesh = self._bracket()
        real_diag = mo.diagnose(mesh)                  # captured BEFORE the diagnose patch

        def fake_repair(m, authorized=()):
            return {"mesh": mesh, "before": {"closed": False}, "after": real_diag,
                    "resolved": {}, "unresolved": {}, "refused": [], "actions": [],
                    "validation": [], "synthetic": [], "quality": {},
                    "provenance": provenance.new_ledger(),
                    "provenance_verification": {"verdict": "UNCERTAIN", "scope": "partial",
                                                "reasons": ["test injection"],
                                                "confidence": 0.5}}

        doc = App.newDocument("h5")
        self.addCleanup(App.closeDocument, "h5")
        obj = doc.addObject("Mesh::Feature", "Scan")
        obj.Mesh = mesh
        # force the repair path so the injected verification is consumed
        diag_patch = {"closed": False, "inconsistent_normals": True}
        with mock.patch.object(rc.meshops, "diagnose", return_value=diag_patch), \
                mock.patch.object(rc.meshops, "repair", side_effect=fake_repair):
            result = rc.reconstruct(doc, mesh)
        self.assertTrue(result.get("supported"), result)
        self.assertNotEqual(result["outcome"], "verified_parametric",
                             "UNCERTAIN provenance must never yield a verified outcome")
        self.assertTrue(result["meta"].get("provenance_uncertain"))


class H6RenderTest(unittest.TestCase):
    def test_plain_language_lines(self):
        from oscar.tools_recon import _provenance_line

        self.assertIn("all original", _provenance_line(None, None, 10))
        led = provenance.new_ledger()
        led["operations"] = [{"operation": "fillupHoles"}]
        led["counts"] = {"synthetic": 2, "modified": 1, "removed": 1, "uncertain": 0}
        line = _provenance_line(led, {"verdict": "VERIFIED", "scope": "complete"}, 12)
        self.assertIn("9 original", line)
        self.assertIn("VERIFIED", line)
        self.assertNotIn("PARTIAL", line)
        line2 = _provenance_line(led, {"verdict": "UNCERTAIN", "scope": "partial",
                                       "reasons": ["key lists were truncated"]}, 12)
        self.assertIn("PARTIAL", line2)
        self.assertIn("truncated", line2)
        self.assertIn("could not", line2)


if __name__ == "__main__":
    unittest.main()
