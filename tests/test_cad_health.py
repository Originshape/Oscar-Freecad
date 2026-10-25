"""Health API: types and renderings (pure), geometry predicates, the per-object checks on real
FreeCAD documents, the hash-keyed snapshot / delta, severity adjustments, the parameter table reader and the
any-thread ``*_via_dispatch`` wrappers.

Runs under ``freecadcmd`` (the dispatcher runs inline there)."""

import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402
import Sketcher  # noqa: E402

from oscar.cad import health, health_eval  # noqa: E402
from oscar.cad.health_types import (FINDING_CODES, TRANSIENT_SUFFIX, Finding, HealthReport, ObjectDelta,  # noqa: E402
                                    ObjRow, StateSnapshot, delta_between)
from oscar.toolctx import ToolContext, use_context  # noqa: E402


# ---------------------------------------------------------------------------------------------- helpers


def _row(name="Box", type_="Part::Box", volume=1000.0, bbox=(0, 0, 0, 10, 10, 10), solids=1, consumed=False,
         state=None, shape_type="Solid", **kw):
    return ObjRow(name=name, label=name, type=type_, state=list(state or ["Up-to-date"]), status="Valid",
                  shape_type=shape_type, valid=True, solids=solids, closed=True, volume=volume,
                  bbox=list(bbox) if bbox else None, visible=True, dof=None, consumed=consumed, **kw)


def _report(findings=(), rows=(), delta=None, doc="PiCase", objects=7, solids=3, **kw):
    f = list(findings)
    counts = {"objects": objects, "solids": solids, "errors": sum(1 for x in f if x.severity == "error"),
              "warnings": sum(1 for x in f if x.severity == "warning")}
    return HealthReport(doc=doc, ts=1.0, rows=list(rows), findings=f, delta=delta, counts=counts, **kw)


class _DocSpy:
    """Delegates to a real document and counts ``recompute`` calls."""

    def __init__(self, doc):
        self._doc = doc
        self.recomputes = 0

    def recompute(self, *args, **kwargs):
        self.recomputes += 1
        return self._doc.recompute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._doc, name)


class DocCase(unittest.TestCase):
    """A fresh FreeCAD document per test; every document the test created is closed afterwards."""

    def setUp(self):
        health_eval.clear_cache()
        self._made: list[str] = []
        self.doc = self.new_doc("HT")
        self.addCleanup(self._close_docs)

    def new_doc(self, name="HT"):
        doc = App.newDocument(name)
        self._made.append(doc.Name)
        return doc

    def _close_docs(self):
        for name in self._made:
            if name in App.listDocuments():
                App.closeDocument(name)
        health_eval.clear_cache()

    def box(self, name="Box", l=10, w=10, h=10, at=(0, 0, 0), doc=None):
        doc = doc or self.doc
        b = doc.addObject("Part::Box", name)
        b.Length, b.Width, b.Height = l, w, h
        b.Placement.Base = App.Vector(*at)
        doc.recompute()
        return b

    def tag(self, obj, role):
        obj.addProperty("App::PropertyString", "OscarRole", "Oscar", "")
        obj.OscarRole = role

    def sheet(self, rows, name="Params", doc=None):
        """rows: (alias, formula, unit, source, note); value in B (aliased), unit C, source D, note E."""
        doc = doc or self.doc
        sh = doc.addObject("Spreadsheet::Sheet", name)
        for col, title in zip("ABCDE", ("name", "value", "unit", "source", "note")):
            sh.set(f"{col}1", title)
        for i, (alias, formula, unit, source, note) in enumerate(rows, start=2):
            sh.set(f"A{i}", alias)
            sh.set(f"B{i}", formula)
            sh.setAlias(f"B{i}", alias)
            for col, text in (("C", unit), ("D", source), ("E", note)):
                if text:
                    sh.set(f"{col}{i}", text)
        doc.recompute()
        return sh

    def health(self, **kw):
        return health_eval.model_health(self.doc, **kw)

    def codes(self, report, severity=None):
        return sorted((f.code, f.object) for f in report.findings if severity in (None, f.severity))


# ---------------------------------------------------------------------------------------------- pure types


class FindingAndSnapshotTypesTest(unittest.TestCase):
    def test_finding_roundtrip_and_key(self):
        f = Finding("error", "FRAGMENTS", "Shell", "3 solids")
        self.assertEqual(f.key(), ("FRAGMENTS", "Shell"))
        self.assertEqual(Finding.from_dict(f.to_dict()), f)

    def test_finding_codes_are_the_contract_codes(self):
        self.assertEqual(set(FINDING_CODES), {
            "RECOMPUTE_ERROR", "TOUCHED", "NULL_SHAPE", "INVALID_SHAPE", "NOT_CLOSED", "NONPOSITIVE_VOLUME",
            "FRAGMENTS", "NOT_SOLID", "NO_GEOMETRIC_CHANGE", "BOOLEAN_INPUT_MISSING", "SKETCH_CONFLICT",
            "SKETCH_REDUNDANT", "SKETCH_MALFORMED", "SKETCH_UNDERCONSTRAINED", "EXPRESSION_ERROR"})
        for code, (severity, trigger) in FINDING_CODES.items():
            self.assertIn(severity, ("error", "warning", "info"), code)
            self.assertTrue(trigger)

    def test_num_is_json_safe(self):
        from oscar.cad.health_types import _num
        self.assertIsNone(_num(float("nan")))
        self.assertIsNone(_num(float("inf")))
        self.assertIsNone(_num("x"))
        self.assertIsNone(_num(None))
        self.assertEqual(_num("1.23456"), 1.235)
        self.assertEqual(str(_num(-5.7e-14)), "0.0")   # rounding noise is never rendered as -0.0

    def test_objrow_roundtrip(self):
        r = _row(role="shell", notes=["skipped (budget)"], consumed=True)
        self.assertEqual(ObjRow.from_dict(r.to_dict()), r)
        self.assertIsNone(ObjRow.from_dict({"name": "x"}).bbox)


class ObjectDeltaTest(unittest.TestCase):
    def test_line_formats_and_limit(self):
        d = ObjectDelta(added=["Lid", "Ref_RPI4B"], removed=[],
                        changed=[{"name": "Shell", "props": ["Shape"], "volume_before": 17020.0, "volume_after": 18432.1}])
        self.assertEqual(d.line(), "+Lid +Ref_RPI4B ~Shell(vol 17020.0->18432.1)")
        self.assertEqual(d.line(max_names=2), "+Lid +Ref_RPI4B (+1 more)")
        self.assertEqual(ObjectDelta(removed=["Old"]).line(), "-Old")

    def test_empty_and_unchanged_volume(self):
        self.assertTrue(ObjectDelta().is_empty())
        self.assertEqual(ObjectDelta().line(), "")
        d = ObjectDelta(changed=[{"name": "Lid", "props": ["Placement"], "volume_before": 5.0, "volume_after": 5.0}])
        self.assertEqual(d.line(), "~Lid")
        self.assertFalse(d.is_empty())

    def test_roundtrip(self):
        d = ObjectDelta(added=["a"], removed=["b"], changed=[{"name": "c", "props": ["Shape"]}])
        self.assertEqual(ObjectDelta.from_dict(d.to_dict()).to_dict(), d.to_dict())
        self.assertTrue(ObjectDelta.from_dict(None).is_empty())


class DeltaBetweenTest(unittest.TestCase):
    @staticmethod
    def _g(h=1, pl=(0, 0, 0, 0, 0, 0, 1), bbox=(0, 0, 0, 1, 1, 1), faces=6, edges=12):
        return ("g", h, pl, tuple(bbox), faces, edges)

    def _snap(self, **objs):
        return StateSnapshot(doc="D", objects={k: dict({"type": "Part::Box", "placement": None, "expr": 0, "volume": None,
                                                        "bbox": None}, **v) for k, v in objs.items()})

    def test_prev_none_is_empty(self):
        self.assertTrue(delta_between(None, self._snap(A={"key": ("p", 1)})).is_empty())

    def test_added_removed_changed(self):
        prev = self._snap(A={"key": self._g(1), "volume": 10.0}, B={"key": ("p", 1)}, C={"key": ("s", 5)})
        now = self._snap(A={"key": self._g(2), "volume": 12.0}, C={"key": ("s", 6)}, D={"key": ("p", 9)})
        d = delta_between(prev, now)
        self.assertEqual((d.added, d.removed), (["D"], ["B"]))
        by_name = {c["name"]: c for c in d.changed}
        self.assertEqual(by_name["A"]["props"], ["Shape"])
        self.assertEqual((by_name["A"]["volume_before"], by_name["A"]["volume_after"]), (10.0, 12.0))
        self.assertEqual(by_name["C"]["props"], ["Properties"])

    def test_placement_only_and_expression(self):
        prev = self._snap(A={"key": self._g(1, pl=(0,) * 7), "placement": (0,) * 7, "expr": 1})
        now = self._snap(A={"key": self._g(1, pl=(5,) + (0,) * 6), "placement": (5,) + (0,) * 6, "expr": 2})
        (change,) = delta_between(prev, now).changed
        self.assertEqual(change["props"], ["Placement", "Expression"])

    def test_visibility_alone_is_not_a_change(self):
        prev = self._snap(A={"key": self._g(1), "visible": True})
        now = self._snap(A={"key": self._g(1), "visible": False})
        self.assertTrue(delta_between(prev, now).is_empty())

    def test_an_unmeasured_volume_is_never_a_change(self):
        prev = self._snap(A={"key": self._g(1), "volume": 10.0})
        now = self._snap(A={"key": self._g(1), "volume": None})
        self.assertTrue(delta_between(prev, now).is_empty())


class HealthReportRenderingTest(unittest.TestCase):
    def test_line_grammar(self):
        self.assertEqual(_report().line(), "[health] doc 'PiCase': 7 objects, 3 solids, VALID")
        delta = ObjectDelta(added=["Lid"], changed=[{"name": "Shell", "props": ["Shape"], "volume_before": 17020.0,
                                                     "volume_after": 18432.1}])
        self.assertEqual(_report(delta=delta).line(),
                         "[health] doc 'PiCase': 7 objects, 3 solids, VALID · Δ +Lid ~Shell(vol 17020.0->18432.1)")
        one = _report(findings=[Finding("error", "FRAGMENTS", "Shell", "m"), Finding("warning", "TOUCHED", "A", "m"),
                                Finding("warning", "TOUCHED", "B", "m")], delta=ObjectDelta(added=["Lid"]))
        self.assertEqual(one.line(), "[health] doc 'PiCase': 7 objects, 3 solids, 1 ERROR, 2 warnings · Δ +Lid")
        two = _report(findings=[Finding("error", "A", "x", "m"), Finding("error", "A", "y", "m"),
                                Finding("warning", "TOUCHED", "A", "m")])
        self.assertIn("2 ERRORS, 1 warning", two.line())
        self.assertNotIn("\n", two.line())

    def test_render_matches_the_contract_sample(self):
        rows = [_row("Shell", "Part::Cut", 18432.1, (0, 0, 0, 94, 65, 22)), _row("Lid", "Part::Cut", 9120.3, (0, 0, 0, 94, 65, 6))]
        rows += [_row(f"Extra{i}", volume=float(10 - i)) for i in range(4)]
        delta = ObjectDelta(added=["Lid"], changed=[{"name": "Shell", "props": ["Shape"], "volume_before": 17020.0,
                                                     "volume_after": 18432.1}])
        fnd = [Finding("error", "FRAGMENTS", "Shell", "3 solids (18000.0, 41.2, 12.9 mm3), expected 1; fuse or delete the floating pieces"),
               Finding("warning", "NO_GEOMETRIC_CHANGE", "Cut001", "volume unchanged by the cut; the tool does not touch the base")]
        text = _report(findings=fnd, rows=rows, delta=delta).render(max_rows=2)
        lines = text.split("\n")
        self.assertEqual(lines[0], "[health] doc 'PiCase': 7 objects, 3 solids, 1 ERROR, 1 warning")
        self.assertEqual(lines[1], "Name         Type            Solids   Volume mm3   BBox mm (x y z)        State")
        self.assertEqual(lines[2], "Shell        Part::Cut            1      18432.1   94.0 x 65.0 x 22.0     ok")
        self.assertEqual(lines[3], "Lid          Part::Cut            1       9120.3   94.0 x 65.0 x 6.0      ok")
        self.assertEqual(lines[4], "(+4 more rows; use get_objects)")
        self.assertEqual(lines[5], "Δ added: Lid · removed: - · changed: Shell(volume 17020.0->18432.1)")
        self.assertTrue(lines[6].startswith("ERROR [FRAGMENTS] Shell: 3 solids"))
        self.assertTrue(lines[7].startswith("WARN  [NO_GEOMETRIC_CHANGE] Cut001: volume unchanged"))

    def test_row_order_findings_then_delta_then_volume_and_consumed_omitted(self):
        rows = [_row("Big", volume=900.0), _row("Small", volume=1.0), _row("Added", volume=5.0), _row("Bad", volume=2.0),
                _row("Input", volume=5000.0, consumed=True)]
        rep = _report(findings=[Finding("warning", "TOUCHED", "Bad", "m")], rows=rows, delta=ObjectDelta(added=["Added"]))
        names = [ln.split()[0] for ln in rep.render().split("\n")[2:6]]
        self.assertEqual(names, ["Bad", "Added", "Big", "Small"])
        self.assertNotIn("Input ", rep.render())
        self.assertIn("Input ", rep.render(mode="full"))   # full lists consumed inputs

    def test_info_findings_only_in_full_mode(self):
        rep = _report(findings=[Finding("info", "SKETCH_UNDERCONSTRAINED", "Sketch", "DoF 2")], rows=[_row()])
        self.assertNotIn("SKETCH_UNDERCONSTRAINED", rep.render())
        self.assertIn("INFO [SKETCH_UNDERCONSTRAINED] Sketch: DoF 2", rep.render(mode="full"))

    def test_full_mode_has_no_row_limit_and_budget_note(self):
        rows = [_row(f"B{i}", volume=float(i)) for i in range(20)]
        rep = _report(rows=rows, budget_exceeded=True)
        self.assertIn("(+8 more rows; use get_objects)", rep.render())
        self.assertNotIn("more rows", rep.render(mode="full"))
        self.assertIn("health budget exceeded", rep.render())

    def test_state_column_shows_first_state_token(self):
        rep = _report(rows=[_row("Cut", state=["Touched", "Invalid"], solids=0, volume=None, bbox=None, shape_type="")],
                      findings=[Finding("error", "RECOMPUTE_ERROR", "Cut", "m")])
        self.assertTrue(any(ln.startswith("Cut") and ln.endswith("Invalid") for ln in rep.render().split("\n")))

    def test_dict_roundtrip_recomputes_valid(self):
        rep = _report(findings=[Finding("error", "FRAGMENTS", "Shell", "m")], rows=[_row()],
                      delta=ObjectDelta(added=["Lid"]), fragments=[{"object": "Shell", "volumes": [1.0, 2.0]}])
        data = rep.to_dict()
        self.assertFalse(data["valid"])
        data["valid"] = True   # a stored flag is ignored
        back = HealthReport.from_dict(data)
        self.assertFalse(back.valid)
        self.assertEqual(back.to_dict()["fragments"], rep.to_dict()["fragments"])
        self.assertEqual(back.line(), rep.line())

    def test_new_errors_against_report_snapshot_and_none(self):
        old = _report(findings=[Finding("error", "FRAGMENTS", "Shell", "m"), Finding("warning", "TOUCHED", "Lid", "m")])
        new = _report(findings=[Finding("error", "FRAGMENTS", "Shell", "m"), Finding("error", "TOUCHED", "Lid", "m"),
                                Finding("error", "NULL_SHAPE", "Cut", "m")])
        # "known" means the key was known at ANY severity: a warning that later became an error is not NEW
        self.assertEqual({f.key() for f in new.new_errors(old)}, {("NULL_SHAPE", "Cut")})
        self.assertEqual(len(new.new_errors(None)), 3)
        snap = StateSnapshot(doc="D", finding_keys=[("FRAGMENTS", "Shell")])
        self.assertEqual({f.key() for f in new.new_errors(snap)}, {("TOUCHED", "Lid"), ("NULL_SHAPE", "Cut")})


# ---------------------------------------------------------------------------------------------- geometry predicates


class GeometryPredicatesTest(DocCase):
    def test_is_geometry_accepts_solids_and_rejects_everything_else(self):
        box = self.box()
        sheet = self.sheet([("wall_t", "=2 mm", "mm", "user_requirement", "")])
        sketch = self.doc.addObject("Sketcher::SketchObject", "Sk")
        empty = self.doc.addObject("Part::Feature", "Empty")
        self.doc.recompute()
        self.assertTrue(health.is_geometry(box))
        for obj in (sheet, sketch, empty):
            self.assertFalse(health.is_geometry(obj), obj.Name)

    def test_origin_features_are_not_geometry(self):
        part = self.doc.addObject("App::Part", "Asm")
        self.doc.recompute()
        features = list(part.Origin.OriginFeatures)
        self.assertTrue(features)
        for o in features + [part.Origin]:
            self.assertFalse(health.is_geometry(o), o.Name)
        # the raw hazard this guards: the feature reports a Shape with a gigantic box
        self.assertTrue(any(hasattr(o, "Shape") for o in features))

    def test_a_part_container_is_not_geometry_even_though_it_reports_a_compound(self):
        part = self.doc.addObject("App::Part", "Asm")
        part.addObject(self.box("InBox"))
        self.doc.recompute()
        self.assertFalse(health.is_geometry(part))
        self.assertTrue(health.is_geometry(self.doc.getObject("InBox")))
        group = self.doc.addObject("App::DocumentObjectGroup", "Grp")
        self.assertFalse(health.is_geometry(group))

    def test_a_non_finite_bounding_box_is_rejected(self):
        shape = mock.Mock()
        shape.isNull.return_value = False
        shape.ShapeType = "Compound"
        shape.BoundBox = types.SimpleNamespace(XMin=-2e100, XMax=2e100, YMin=-2e100, YMax=2e100, ZMin=-2e100, ZMax=2e100,
                                               isValid=lambda: True)
        fake = types.SimpleNamespace(TypeId="Part::Feature", Shape=shape, isDerivedFrom=lambda t: False)
        self.assertFalse(health.is_geometry(fake))

    def test_mesh_object_with_facets_is_geometry_empty_mesh_is_not(self):
        v = App.Vector
        pts = [v(0, 0, 0), v(10, 0, 0), v(0, 10, 0), v(0, 0, 0), v(0, 10, 0), v(0, 0, 10),
               v(0, 0, 0), v(0, 0, 10), v(10, 0, 0), v(10, 0, 0), v(0, 0, 10), v(0, 10, 0)]
        imported = self.doc.addObject("Mesh::Feature", "Imported")
        imported.Mesh = Mesh.Mesh(pts)
        empty = self.doc.addObject("Mesh::Feature", "EmptyMesh")
        self.doc.recompute()
        self.assertTrue(health.is_geometry(imported))
        self.assertFalse(health.is_geometry(empty))
        report = self.health()
        (row,) = [r for r in report.rows if r.name == "Imported"]
        self.assertEqual((row.shape_type, row.solids), ("Mesh", 0))
        self.assertAlmostEqual(row.volume, 166.667, places=2)   # the tetrahedron is a closed mesh
        self.assertEqual(self.codes(report), [])

    def test_never_raises_on_garbage(self):
        self.assertFalse(health.is_geometry(None))
        self.assertFalse(health.is_geometry(object()))

    def test_global_shape_uses_the_container_frame(self):
        part = self.doc.addObject("App::Part", "Asm")
        part.Placement = App.Placement(App.Vector(0, 0, 50), App.Rotation())
        box = self.box("InBox")
        part.addObject(box)
        self.doc.recompute()
        self.assertEqual(box.Shape.BoundBox.ZMin, 0.0)             # container-local frame: the verified trap
        shape = health.global_shape(box)
        self.assertEqual((shape.BoundBox.ZMin, shape.BoundBox.ZMax), (50.0, 60.0))
        shape.translate(App.Vector(0, 0, 1000))                    # a COPY: mutating it never touches the object
        self.assertEqual(box.Shape.BoundBox.ZMin, 0.0)


# ---------------------------------------------------------------------------------------------- model_health


class ModelHealthChecksTest(DocCase):
    def test_a_sound_model_is_valid_and_counts_solids_not_origins(self):
        self.box("A")
        cyl = self.doc.addObject("Part::Cylinder", "Cyl")
        cyl.Placement.Base = App.Vector(30, 0, 0)
        self.doc.addObject("App::Part", "Asm")
        self.doc.recompute()
        report = self.health()
        self.assertTrue(report.valid)
        self.assertEqual(report.findings, [])
        self.assertEqual(report.counts["solids"], 2)
        self.assertEqual({r.name for r in report.rows}, {"A", "Cyl"})   # no Origin features, no container
        self.assertEqual(report.counts["objects"], 3)
        self.assertEqual(report.line(), f"[health] doc '{self.doc.Name}': 3 objects, 2 solids, VALID")

    def test_container_members_report_the_global_bbox_and_the_container_is_not_counted(self):
        part = self.doc.addObject("App::Part", "Asm")
        part.Placement = App.Placement(App.Vector(0, 0, 50), App.Rotation())
        part.addObject(self.box("InBox"))
        self.doc.recompute()
        report = self.health()
        self.assertEqual([r.name for r in report.rows], ["InBox"])
        self.assertEqual(report.counts["solids"], 1)
        self.assertEqual(report.rows[0].bbox[2], 50.0)
        self.assertEqual(report.rows[0].bbox[5], 60.0)
        for r in report.rows:
            self.assertLess(max(abs(x) for x in r.bbox), 1e6)

    def test_recompute_error_is_an_error_with_the_status_text(self):
        bad = self.doc.addObject("Part::Box", "Bad")
        bad.Length = 0
        self.doc.recompute()
        report = self.health()
        (f,) = [f for f in report.findings if f.code == "RECOMPUTE_ERROR"]
        self.assertEqual((f.severity, f.object), ("error", "Bad"))
        self.assertIn("too small", f.message)
        self.assertFalse(report.valid)
        (row,) = [r for r in report.rows if r.name == "Bad"]
        self.assertIn("Invalid", row.state)

    def test_an_invalid_object_is_not_recomputed_again(self):
        bad = self.doc.addObject("Part::Box", "Bad")
        bad.Length = 0
        self.doc.recompute()
        self.assertFalse(health_eval._needs_recompute(self.doc))   # ['Touched', 'Invalid'] forever must not trigger it
        spy = _DocSpy(self.doc)
        report = health_eval.model_health(spy)
        self.assertEqual(spy.recomputes, 0)
        self.assertNotIn("TOUCHED", {f.code for f in report.findings})   # reported once, as RECOMPUTE_ERROR
        self.assertEqual([f.code for f in report.findings if f.object == "Bad"], ["RECOMPUTE_ERROR"])

    def test_a_touched_object_is_recomputed_once_and_not_flagged(self):
        box = self.box()
        box.Length = 30   # touched, not recomputed
        self.assertIn("Touched", box.State)
        self.assertTrue(health_eval._needs_recompute(self.doc))
        report = self.health()
        self.assertNotIn("Touched", box.State)
        self.assertEqual(self.codes(report), [])
        self.assertEqual(report.rows[0].volume, 3000.0)
        box.Width = 20
        spy = _DocSpy(self.doc)
        health_eval.model_health(spy)
        self.assertEqual(spy.recomputes, 1)                        # exactly one recompute
        spy2 = _DocSpy(self.doc)
        health_eval.model_health(spy2)
        self.assertEqual(spy2.recomputes, 0)                       # nothing touched: none

    def test_touched_after_the_recompute_attempt_is_a_warning(self):
        box = self.box()
        box.Length = 30   # touched; pretend the recompute could not clear it
        with mock.patch.object(health_eval, "_needs_recompute", return_value=False):
            report = self.health()
        (f,) = [f for f in report.findings if f.code == "TOUCHED"]
        self.assertEqual((f.severity, f.object), ("warning", "Box"))
        self.assertTrue(report.valid)

    def test_public_names_of_the_facade_exist(self):
        for name in health.__all__:
            self.assertTrue(hasattr(health, name), name)

    def test_invalid_shape_closed_and_volume_checks(self):
        v = App.Vector
        bow = Part.Face(Part.makePolygon([v(0, 0, 0), v(10, 10, 0), v(10, 0, 0), v(0, 10, 0), v(0, 0, 0)])).extrude(v(0, 0, 5))
        open_solid = Part.Solid(Part.Shell(Part.makeBox(10, 10, 10).Faces[:5]))
        inverted = Part.makeBox(10, 10, 10)
        inverted.reverse()
        for name, shape in (("Bow", bow), ("Open", open_solid), ("Rev", inverted)):
            self.doc.addObject("Part::Feature", name).Shape = shape
        self.doc.recompute()
        got = set(self.codes(self.health(), "error"))
        self.assertIn(("INVALID_SHAPE", "Bow"), got)
        self.assertIn(("NOT_CLOSED", "Open"), got)
        self.assertIn(("INVALID_SHAPE", "Open"), got)
        self.assertIn(("NONPOSITIVE_VOLUME", "Rev"), got)
        self.assertNotIn(("INVALID_SHAPE", "Rev"), got)
        negative = [f for f in self.health().findings if f.code == "NONPOSITIVE_VOLUME" and f.object == "Bow"]
        self.assertTrue(all("-0.000" not in f.message for f in negative))   # rounding noise is not printed as a negative volume

    def test_fragments_tagged_part_is_an_error_untagged_is_a_warning_only_below_five_percent(self):
        comp = Part.makeCompound([Part.makeBox(10, 10, 10), Part.makeBox(1, 1, 1, App.Vector(50, 0, 0))])
        shell = self.doc.addObject("Part::Feature", "Shell")
        self.tag(shell, "shell")
        shell.Shape = comp
        self.doc.addObject("Part::Feature", "Loose").Shape = comp
        two_equal = Part.makeCompound([Part.makeBox(10, 10, 10), Part.makeBox(10, 10, 10, App.Vector(50, 0, 0))])
        self.doc.addObject("Part::Feature", "TwoEqual").Shape = two_equal
        self.doc.recompute()
        report = self.health()
        by_obj = {f.object: f for f in report.findings if f.code == "FRAGMENTS"}
        self.assertEqual(by_obj["Shell"].severity, "error")
        self.assertIn("2 solids (1000.0, 1.0 mm3), expected 1", by_obj["Shell"].message)
        self.assertEqual(by_obj["Loose"].severity, "warning")
        self.assertNotIn("TwoEqual", by_obj)
        self.assertEqual([x["object"] for x in report.fragments], ["Shell", "Loose"])
        self.assertEqual(report.fragments[0]["volumes"], [1000.0, 1.0])

    def test_not_solid_tagged_is_an_error_untagged_is_info(self):
        shell = Part.Shell(Part.makeBox(10, 10, 10).Faces)
        lid = self.doc.addObject("Part::Feature", "Lid")
        self.tag(lid, "lid")
        lid.Shape = shell
        self.doc.addObject("Part::Feature", "Surf").Shape = shell
        self.doc.recompute()
        report = self.health()
        sev = {f.object: f.severity for f in report.findings if f.code == "NOT_SOLID"}
        self.assertEqual(sev, {"Lid": "error", "Surf": "info"})

    def test_a_compound_of_one_solid_is_an_accepted_tagged_solid(self):
        part = self.doc.addObject("Part::Feature", "Shell")
        self.tag(part, "shell")
        part.Shape = Part.makeCompound([Part.makeBox(10, 10, 10)])
        self.doc.recompute()
        self.assertEqual(self.codes(self.health()), [])

    def test_boolean_input_missing_and_no_geometric_change(self):
        base = self.box("B")
        cut = self.doc.addObject("Part::Cut", "CutNoTool")
        cut.Base = base
        far = self.box("Far", at=(100, 100, 100))
        useless = self.doc.addObject("Part::Cut", "Useless")
        useless.Base, useless.Tool = base, far
        self.doc.recompute()
        report = self.health(strict=True)
        codes = self.codes(report)
        self.assertIn(("BOOLEAN_INPUT_MISSING", "CutNoTool"), codes)
        self.assertIn(("NO_GEOMETRIC_CHANGE", "Useless"), codes)
        (nochg,) = [f for f in report.findings if f.code == "NO_GEOMETRIC_CHANGE"]
        self.assertEqual(nochg.severity, "warning")
        self.assertIn("tool does not touch the base", nochg.message)

    def test_multifuse_with_empty_shapes_is_missing_input(self):
        self.doc.addObject("Part::MultiFuse", "Fuse")
        self.doc.recompute()
        self.assertIn(("BOOLEAN_INPUT_MISSING", "Fuse"), self.codes(self.health(strict=True)))

    def test_sketch_conflict_redundant_and_underconstrained(self):
        conflict = self.doc.addObject("Sketcher::SketchObject", "Conflict")
        conflict.addGeometry(Part.LineSegment(App.Vector(0, 0, 0), App.Vector(10, 0, 0)))
        conflict.addConstraint(Sketcher.Constraint("Horizontal", 0))
        conflict.addConstraint(Sketcher.Constraint("Distance", 0, 10))
        conflict.addConstraint(Sketcher.Constraint("Distance", 0, 20))
        redundant = self.doc.addObject("Sketcher::SketchObject", "Redundant")
        redundant.addGeometry(Part.LineSegment(App.Vector(0, 0, 0), App.Vector(10, 0, 0)))
        redundant.addConstraint(Sketcher.Constraint("Horizontal", 0))
        redundant.addConstraint(Sketcher.Constraint("Horizontal", 0))
        loose = self.doc.addObject("Sketcher::SketchObject", "Loose")
        loose.addGeometry(Part.LineSegment(App.Vector(0, 0, 0), App.Vector(10, 0, 0)))
        self.doc.recompute()
        report = self.health()
        got = set(self.codes(report))
        self.assertIn(("SKETCH_CONFLICT", "Conflict"), got)
        self.assertIn(("SKETCH_REDUNDANT", "Redundant"), got)
        self.assertIn(("SKETCH_UNDERCONSTRAINED", "Loose"), got)
        sev = {(f.code, f.object): f.severity for f in report.findings}
        self.assertEqual(sev[("SKETCH_CONFLICT", "Conflict")], "error")
        self.assertEqual(sev[("SKETCH_REDUNDANT", "Redundant")], "warning")
        self.assertEqual(sev[("SKETCH_UNDERCONSTRAINED", "Loose")], "info")
        (conflict_f,) = [f for f in report.findings if f.code == "SKETCH_CONFLICT" and f.object == "Conflict"]
        self.assertIn("[2, 3]", conflict_f.message)
        (loose_row,) = [r for r in report.rows if r.name == "Loose"]
        self.assertEqual(loose_row.dof, 4)
        self.assertNotIn("SKETCH_UNDERCONSTRAINED", report.render())        # info only in full mode
        self.assertIn("SKETCH_UNDERCONSTRAINED", report.render(mode="full"))

    def test_expression_errors_missing_alias_text_cell_and_division_by_zero(self):
        sheet = self.sheet([("wall_t", "=2 mm", "mm", "", ""), ("neg", "-1.5 mm", "mm", "", ""), ("bad", "=1 mm/0", "mm", "", "")])
        self.assertIsInstance(sheet.get("neg"), str)   # the stored-as-text trap
        good = self.doc.addObject("Part::Box", "Good")
        good.setExpression("Length", "Params.wall_t * 5")
        missing = self.doc.addObject("Part::Box", "Missing")
        missing.setExpression("Length", "Params.nothere")
        text = self.doc.addObject("Part::Box", "Text")
        text.setExpression("Length", "Params.neg")
        inf = self.doc.addObject("Part::Box", "Inf")
        inf.setExpression("Length", "Params.bad")
        self.doc.recompute()
        report = self.health()
        expr = {f.object: f.message for f in report.findings if f.code == "EXPRESSION_ERROR"}
        self.assertNotIn("Good", expr)
        self.assertIn("nothere", expr["Missing"])
        self.assertIn("is text '-1.5 mm'", expr["Text"])
        self.assertIn("=-1.5 mm", expr["Text"])     # says how to fix it
        self.assertIn("evaluates to inf", expr["Inf"])
        self.assertTrue(all(f.severity == "error" for f in report.findings if f.code == "EXPRESSION_ERROR"))

    def test_objs_limits_the_evaluation_and_names_are_accepted(self):
        self.box("A")
        bad = self.doc.addObject("Part::Box", "Bad")
        bad.Length = 0
        self.doc.recompute()
        only_a = self.health(objs=["A"])
        self.assertEqual((only_a.counts["objects"], only_a.findings), (1, []))
        self.assertFalse(self.health(objs=[bad]).valid)

    def test_tmp_objects_are_ignored(self):
        self.box("A")
        bad = self.doc.addObject("Part::Box", "_oscar_tmp_probe")
        bad.Length = 0
        self.doc.recompute()
        report = self.health()
        self.assertTrue(report.valid)
        self.assertEqual(report.counts["objects"], 1)
        self.assertNotIn("_oscar_tmp_probe", health_eval.snapshot_state(self.doc).objects)

    def test_a_body_with_a_pad_reports_the_body_and_hides_its_features(self):
        body = self.doc.addObject("PartDesign::Body", "Body")
        sk = body.newObject("Sketcher::SketchObject", "Sk")
        sk.AttachmentSupport = [(self.doc.getObject("XY_Plane"), "")]
        sk.MapMode = "FlatFace"
        v = App.Vector
        pts = [v(0, 0, 0), v(10, 0, 0), v(10, 10, 0), v(0, 10, 0)]
        for i in range(4):
            sk.addGeometry(Part.LineSegment(pts[i], pts[(i + 1) % 4]))
        for i in range(4):
            sk.addConstraint(Sketcher.Constraint("Coincident", i, 2, (i + 1) % 4, 1))
        pad = body.newObject("PartDesign::Pad", "Pad")
        pad.Profile = sk
        pad.Length = 5
        self.doc.recompute()
        report = self.health()
        self.assertTrue(report.valid, report.render("full"))
        names = [r.name for r in report.rows]
        self.assertIn("Body", names)
        compact = report.render()
        self.assertNotIn("\nPad ", compact)     # features inside a Body are consumed: omitted from the compact table

    def test_consumed_means_linked_not_merely_referenced_by_an_expression(self):
        a = self.box("Alpha")
        b = self.box("Beta", at=(50, 0, 0))
        b.setExpression("Length", "Alpha.Length")            # an expression dependency: Alpha stays a visible part
        tool = self.box("Tool", at=(5, 5, 5), l=2, w=2, h=2)
        cut = self.doc.addObject("Part::Cut", "Cut")
        cut.Base, cut.Tool = a, tool                      # link properties: Alpha and Tool are consumed
        self.doc.recompute()
        rows = {r.name: r for r in self.health().rows}
        self.assertTrue(rows["Alpha"].consumed)
        self.assertTrue(rows["Tool"].consumed)
        self.assertFalse(rows["Beta"].consumed)
        self.assertFalse(rows["Cut"].consumed)
        only_b = self.doc.addObject("Part::Box", "OnlyRef")
        only_b.setExpression("Length", "Beta.Length")
        self.doc.recompute()
        self.assertFalse({r.name: r for r in self.health().rows}["Beta"].consumed)
        self.assertEqual(self.health().counts["solids"], 3)   # Cut, Beta, OnlyRef (Alpha and Tool are inputs)

    def test_ref_findings_are_warnings(self):
        bad = self.doc.addObject("Part::Box", "Ref_Pi")
        bad.Length = 0
        part = self.doc.addObject("App::Part", "Ref_Holder")
        inner = self.doc.addObject("Part::Feature", "Inner")
        inner.Shape = Part.Face(Part.makePolygon([App.Vector(0, 0, 0), App.Vector(10, 10, 0), App.Vector(10, 0, 0),
                                                  App.Vector(0, 10, 0), App.Vector(0, 0, 0)])).extrude(App.Vector(0, 0, 5))
        part.addObject(inner)
        tagged = self.doc.addObject("Part::Box", "Imported")
        self.tag(tagged, "ref")
        tagged.Length = 0
        self.doc.recompute()
        report = self.health()
        self.assertTrue(report.valid, report.render("full"))
        self.assertTrue(report.findings)
        self.assertEqual({f.severity for f in report.findings if f.severity != "info"}, {"warning"})
        self.assertEqual({f.object for f in report.findings}, {"Ref_Pi", "Inner", "Imported"})

    def test_transient_build_states_are_warnings_only_for_objects_added_by_the_call(self):
        base = self.box("B")
        cut = self.doc.addObject("Part::Cut", "Cut")
        cut.Base = base
        empty = self.doc.addObject("Part::Feature", "Empty")
        body = self.doc.addObject("PartDesign::Body", "Body")
        self.doc.recompute()
        before = StateSnapshot(doc=self.doc.Name, objects={"B": {"type": "Part::Box", "key": ("p", 0)}})
        report = self.health(prev=before)
        by_code = {(f.code, f.object): f for f in report.findings}
        for key in (("BOOLEAN_INPUT_MISSING", "Cut"), ("NULL_SHAPE", "Empty"), ("NULL_SHAPE", "Body")):
            self.assertEqual(by_code[key].severity, "warning", key)
            self.assertTrue(by_code[key].message.endswith(TRANSIENT_SUFFIX), key)
        self.assertNotIn(("RECOMPUTE_ERROR", "Cut"), by_code)       # the Invalid state they cause is not reported on top
        self.assertTrue(report.valid)
        # an object that was ALREADY there, or a strict run: errors
        known = health_eval.snapshot_state(self.doc)
        self.assertFalse(self.health(prev=known).valid)
        strict = self.health(prev=before, strict=True)
        self.assertEqual({f.severity for f in strict.findings if f.code in ("BOOLEAN_INPUT_MISSING", "NULL_SHAPE")}, {"error"})
        del empty, body

    def test_budget_zero_skips_expensive_facts_and_says_so(self):
        self.box()
        report = self.health(budget_s=0)
        self.assertTrue(report.budget_exceeded)
        (row,) = report.rows
        self.assertIn("skipped (budget)", row.notes)
        self.assertIsNone(row.volume)
        self.assertIn("budget exceeded", report.render())
        self.assertFalse(self.health().budget_exceeded)

    def test_default_budget_comes_from_the_environment(self):
        with mock.patch.dict(os.environ, {"OSCAR_HEALTH_BUDGET_S": "2.5"}):
            self.assertEqual(health_eval.default_budget_s(), 2.5)
        for bad in ("abc", "-1", "nan", ""):
            with mock.patch.dict(os.environ, {"OSCAR_HEALTH_BUDGET_S": bad}):
                self.assertEqual(health_eval.default_budget_s(), 4.0, bad)
        with mock.patch.dict(os.environ, {"OSCAR_HEALTH_BUDGET_S": "0"}):
            self.assertEqual(health_eval.default_budget_s(), 0.0)

    def test_isvalid_is_skipped_for_huge_shapes_unless_deep(self):
        self.box()
        with mock.patch.object(health_eval, "FACE_LIMIT", 3):
            shallow = self.health()
            self.assertIsNone(shallow.rows[0].valid)
            self.assertTrue(any("isValid skipped" in n for n in shallow.rows[0].notes))
            health_eval.clear_cache()
            deep = self.health(deep=True)
            self.assertTrue(deep.rows[0].valid)
            self.assertEqual(deep.rows[0].notes, [])


# ---------------------------------------------------------------------------------------------- snapshot, delta, gate


class SnapshotDeltaTest(DocCase):
    def test_snapshot_has_an_entry_for_every_model_object_but_no_origin_or_tmp(self):
        self.box("A")
        self.sheet([("wall_t", "=2 mm", "mm", "", "")])
        self.doc.addObject("App::Part", "Asm")
        self.doc.addObject("Part::Box", "_oscar_tmp_x")
        self.doc.recompute()
        snap = health_eval.snapshot_state(self.doc)
        self.assertEqual(set(snap.objects), {"A", "Params", "Asm"})
        self.assertEqual(snap.objects["A"]["volume"], 1000.0)
        self.assertEqual(snap.objects["A"]["bbox"], [0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
        self.assertIsNone(snap.objects["Params"]["volume"])      # sheets and containers carry no volume
        self.assertEqual(snap.objects["A"]["key"][0], "g")
        self.assertEqual(snap.objects["Params"]["key"][0], "s")
        self.assertEqual(snap.doc, self.doc.Name)

    def test_a_step_that_only_creates_a_sheet_has_a_non_empty_delta(self):
        prev = health_eval.snapshot_state(self.doc)
        self.sheet([("wall_t", "=2 mm", "mm", "", "")])
        report = self.health(prev=prev)
        self.assertEqual(report.delta.added, ["Params"])
        self.assertIn("Δ +Params", report.line())

    def test_delta_after_add_edit_move_expression_sheet_edit_and_remove(self):
        a = self.box("A")
        b = self.box("B", at=(50, 0, 0))
        sheet = self.sheet([("hgt", "=10 mm", "mm", "", "")])
        gone = self.box("Gone", at=(0, 50, 0))
        prev = health_eval.snapshot_state(self.doc)
        cyl = self.doc.addObject("Part::Cylinder", "Cyl")
        a.Length = 20
        b.Placement.Base = App.Vector(60, 0, 0)
        sheet.set("B2", "=12 mm")
        self.doc.removeObject(gone.Name)
        self.box("C", at=(0, 0, 50))
        a.setExpression("Width", "Params.hgt")
        self.doc.recompute()
        delta = self.health(prev=prev).delta
        self.assertEqual(sorted(delta.added), ["C", "Cyl"])
        self.assertEqual(delta.removed, ["Gone"])
        changed = {c["name"]: c for c in delta.changed}
        self.assertEqual(changed["A"]["props"], ["Shape", "Expression"])
        self.assertEqual((changed["A"]["volume_before"], changed["A"]["volume_after"]), (1000.0, 2400.0))   # 20 x 12 x 10
        self.assertIn("Placement", changed["B"]["props"])
        self.assertIn("Properties", changed["Params"]["props"])
        self.assertNotIn("A", delta.added)
        del cyl

    def test_unchanged_document_has_an_empty_delta_and_visibility_is_not_a_change(self):
        a = self.box("A")
        prev = health_eval.snapshot_state(self.doc)
        a.Visibility = False
        self.assertTrue(self.health(prev=prev).delta.is_empty())

    def test_cache_is_hash_keyed_an_unchanged_object_keeps_its_facts(self):
        a = self.box("A")
        self.box("B", at=(50, 0, 0))
        health_eval.snapshot_state(self.doc)
        doc_id = health_eval._doc_id(self.doc)
        rec_a, rec_b = health_eval._RECS[doc_id]["A"], health_eval._RECS[doc_id]["B"]
        health_eval.snapshot_state(self.doc)
        self.assertIs(health_eval._RECS[doc_id]["A"], rec_a)
        a.Length = 20
        self.doc.recompute()
        snap = health_eval.snapshot_state(self.doc)
        self.assertIsNot(health_eval._RECS[doc_id]["A"], rec_a)
        self.assertIs(health_eval._RECS[doc_id]["B"], rec_b)
        self.assertEqual(snap.objects["A"]["volume"], 2000.0)

    def test_snapshot_budget_zero_leaves_volumes_empty(self):
        self.box("A")
        snap = health_eval.snapshot_state(self.doc, budget_s=0)
        self.assertIsNone(snap.objects["A"]["volume"])
        self.assertIsNotNone(snap.objects["A"]["key"])

    def test_cache_is_dropped_for_closed_documents(self):
        other = self.new_doc("HT_other")
        self.box("X", doc=other)
        health_eval.snapshot_state(other)
        other_id = health_eval._doc_id(other)
        self.assertIn(other_id, health_eval._RECS)
        App.closeDocument(other.Name)
        health_eval.snapshot_state(self.doc)
        self.assertNotIn(other_id, health_eval._RECS)

    def test_gate_errors_are_the_errors_new_since_the_snapshot(self):
        old_bad = self.doc.addObject("Part::Box", "OldBad")
        old_bad.Length = 0
        self.doc.recompute()
        prev = health_eval.snapshot_state(self.doc)
        self.assertIn(("RECOMPUTE_ERROR", "OldBad"), prev.finding_key_set())
        again = self.health(prev=prev)
        self.assertFalse(again.valid)                       # still failing ...
        self.assertEqual(again.new_errors(prev), [])        # ... but not NEW
        new_bad = self.doc.addObject("Part::Box", "NewBad")
        new_bad.Length = 0
        self.doc.recompute()
        report = self.health(prev=prev)
        self.assertEqual([(f.code, f.object) for f in report.new_errors(prev)],
                         [("RECOMPUTE_ERROR", "NewBad")])

    def test_explicit_prev_findings_override_the_evaluation(self):
        bad = self.doc.addObject("Part::Box", "Bad")
        bad.Length = 0
        self.doc.recompute()
        snap = health_eval.snapshot_state(self.doc, prev_findings=[("X", "y")])
        self.assertEqual(snap.finding_keys, [("X", "y")])


# ---------------------------------------------------------------------------------------------- parameter table


class ParamTableTest(DocCase):
    def test_no_sheet_gives_an_empty_table(self):
        self.assertEqual(health_eval.read_param_table(self.doc), {"sheet": "", "rows": []})

    def test_rows_values_units_sources_and_formulas(self):
        self.sheet([("wall_t", "=2 mm", "mm", "user_requirement", "wall"),
                    ("screw_count", "=4", "", "derived", ""),
                    ("note_txt", "hello", "", "assumption", "free text"),
                    ("lid_t", "=wall_t * 2", "mm", "derived", "")])
        table = health_eval.read_param_table(self.doc)
        self.assertEqual(table["sheet"], "Params")
        rows = {r["name"]: r for r in table["rows"]}
        self.assertEqual(list(rows), ["wall_t", "screw_count", "note_txt", "lid_t"])
        self.assertEqual((rows["wall_t"]["value"], rows["wall_t"]["unit"], rows["wall_t"]["expr"]), (2.0, "mm", ""))
        self.assertEqual((rows["wall_t"]["source"], rows["wall_t"]["note"], rows["wall_t"]["cell"]),
                         ("user_requirement", "wall", "B2"))
        self.assertEqual(rows["screw_count"]["value"], 4)           # a unitless cell is a plain number: no .Value crash
        self.assertEqual(rows["note_txt"]["value"], "hello")        # text stays text
        self.assertEqual(rows["lid_t"]["value"], 4.0)
        self.assertEqual(rows["lid_t"]["expr"], "=wall_t * 2")
        self.assertEqual(rows["lid_t"]["unit"], "mm")

    def test_bound_counts_and_needs_rebuild(self):
        self.sheet([("wall_t", "=2 mm", "mm", "", ""), ("screw_count", "=4", "", "", ""), ("hook_count", "=2", "", "", "")])
        a = self.doc.addObject("Part::Box", "A")
        a.setExpression("Length", "Params.wall_t * 5")
        a.setExpression("Width", "Params.wall_t * 3")
        b = self.doc.addObject("Part::Box", "B")
        b.setExpression("Height", "Params.hook_count * 2")
        self.doc.recompute()
        rows = {r["name"]: r for r in health_eval.read_param_table(self.doc)["rows"]}
        self.assertEqual((rows["wall_t"]["bound"], rows["hook_count"]["bound"], rows["screw_count"]["bound"]), (2, 1, 0))
        self.assertFalse(rows["wall_t"]["needs_rebuild"])
        self.assertFalse(rows["hook_count"]["needs_rebuild"])     # bound: editing it rebuilds by itself
        self.assertTrue(rows["screw_count"]["needs_rebuild"])     # a count that nothing is bound to

    def test_the_parameters_sheet_name_is_accepted_and_params_wins(self):
        self.sheet([("a_val", "=1 mm", "mm", "", "")], name="Parameters")
        self.assertEqual(health_eval.read_param_table(self.doc)["sheet"], "Parameters")
        self.sheet([("b_val", "=1 mm", "mm", "", "")], name="Params")
        table = health_eval.read_param_table(self.doc)
        self.assertEqual((table["sheet"], [r["name"] for r in table["rows"]]), ("Params", ["b_val"]))
        self.assertEqual(health_eval.read_param_table(self.doc, ("Nope",)), {"sheet": "", "rows": []})

    def test_a_non_finite_cell_value_is_none(self):
        self.sheet([("bad", "=1 mm/0", "mm", "", "")])
        (row,) = health_eval.read_param_table(self.doc)["rows"]
        self.assertIsNone(row["value"])


# ---------------------------------------------------------------------------------------------- any-thread wrappers


class ViaDispatchTest(DocCase):
    def test_wrappers_work_inline_for_the_active_and_for_a_named_document(self):
        self.box("A")
        snap = health.snapshot_via_dispatch(None)
        self.assertIsInstance(snap, StateSnapshot)
        self.assertEqual(snap.doc, self.doc.Name)
        self.assertIsInstance(health.snapshot_via_dispatch(self.doc.Name), StateSnapshot)
        report = health.health_via_dispatch(self.doc.Name, snap)
        self.assertIsInstance(report, HealthReport)
        self.assertTrue(report.valid)
        self.assertTrue(report.delta.is_empty())

    def test_document_is_found_by_label_and_case_insensitively(self):
        self.doc.Label = "PiCase"
        self.assertIsNotNone(health.snapshot_via_dispatch("PiCase"))
        self.assertIsNotNone(health.snapshot_via_dispatch(self.doc.Name.lower()))

    def test_unknown_document_gives_none_and_logs_health_error(self):
        with mock.patch("oscar.turnlog.log_event") as log:
            self.assertIsNone(health.snapshot_via_dispatch("NoSuchDoc"))
            self.assertIsNone(health.health_via_dispatch("NoSuchDoc"))
        self.assertEqual({c.args[0] for c in log.call_args_list}, {"health_error"})
        self.assertIn("not open", log.call_args_list[0].kwargs["error"])
        self.assertEqual(health.param_table_via_dispatch("NoSuchDoc"), {"sheet": "", "rows": []})

    def test_strict_flag_is_passed_through(self):
        base = self.box("B")
        cut = self.doc.addObject("Part::Cut", "Cut")
        cut.Base = base
        self.doc.recompute()
        empty = StateSnapshot(doc=self.doc.Name)
        self.assertTrue(health.health_via_dispatch(self.doc.Name, empty).valid)
        self.assertFalse(health.health_via_dispatch(self.doc.Name, empty, strict=True).valid)

    def test_snapshot_all_puts_the_active_document_first_and_caps_the_count(self):
        d2 = self.new_doc("HT_two")
        d3 = self.new_doc("HT_three")
        self.box("X", doc=d2)
        App.setActiveDocument(d3.Name)
        everything = health.snapshot_all_via_dispatch(max_docs=50)
        self.assertEqual(next(iter(everything)), d3.Name)
        self.assertTrue({self.doc.Name, d2.Name, d3.Name} <= set(everything))
        self.assertEqual(len(health.snapshot_all_via_dispatch(max_docs=1)), 1)
        self.assertEqual(health.snapshot_all_via_dispatch(max_docs=0), {})

    def test_stop_from_the_bound_context_abandons_the_call(self):
        self.box("A")
        with mock.patch("oscar.turnlog.log_event") as log:
            with use_context(ToolContext(should_stop=lambda: True)):
                self.assertIsNone(health.snapshot_via_dispatch(None))
                self.assertIsNone(health.health_via_dispatch(None))
                self.assertEqual(health.snapshot_all_via_dispatch(), {})
        self.assertGreaterEqual(log.call_count, 3)
        self.assertIsNotNone(health.snapshot_via_dispatch(None))   # outside the context it works again

    def test_a_dispatcher_that_raises_never_propagates(self):
        with mock.patch.object(health._dispatch, "dispatch", side_effect=RuntimeError("boom")), \
                mock.patch("oscar.turnlog.log_event") as log:
            self.assertIsNone(health.snapshot_via_dispatch(None))
        self.assertEqual(log.call_args.kwargs["error"], "boom")

    def test_param_table_via_dispatch(self):
        self.sheet([("wall_t", "=2 mm", "mm", "", "")])
        table = health.param_table_via_dispatch(None)
        self.assertEqual([r["name"] for r in table["rows"]], ["wall_t"])


if __name__ == "__main__":
    unittest.main()
