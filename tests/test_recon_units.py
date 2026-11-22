"""K4: ``oscar.recon.units`` — source-unit metadata (3MF re-read from the source file; the
importer drops it — verified), transparent mm/inch plausibility (grid roundness evidence,
never size assumptions), safe ×25.4 conversion on working copies, user confirmation, and the
reconstruction gate: an unconfirmed mm/inch interpretation can never yield a VERIFIED
outcome. Runtime facts asserted here were probed on FreeCAD 1.1 (document units do not touch
mesh coordinates; transformGeometry scales exactly).
"""

import json
import os
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402

from oscar.recon import provenance, units  # noqa: E402


def box_mesh(x, y, z, tess=None):
    tess = tess or max(x, y, z) / 10.0
    pts, facs = Part.makeBox(x, y, z).tessellate(tess)
    return Mesh.Mesh((list(pts), [tuple(f) for f in facs]))


def write_3mf(path, unit, size=1.0):
    s = repr(float(size))
    model = f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="{unit}" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
 <resources><object id="1" type="model"><mesh><vertices>
  <vertex x="0" y="0" z="0"/><vertex x="{s}" y="0" z="0"/><vertex x="{s}" y="{s}" z="0"/>
  <vertex x="0" y="{s}" z="0"/><vertex x="0" y="0" z="{s}"/><vertex x="{s}" y="0" z="{s}"/>
  <vertex x="{s}" y="{s}" z="{s}"/><vertex x="0" y="{s}" z="{s}"/>
 </vertices><triangles>
  <triangle v1="0" v2="2" v3="1"/><triangle v1="0" v2="3" v3="2"/>
  <triangle v1="4" v2="5" v3="6"/><triangle v1="4" v2="6" v3="7"/>
  <triangle v1="0" v2="1" v3="5"/><triangle v1="0" v2="5" v3="4"/>
  <triangle v1="1" v2="2" v3="6"/><triangle v1="1" v2="6" v3="5"/>
  <triangle v1="2" v2="3" v3="7"/><triangle v1="2" v2="7" v3="6"/>
  <triangle v1="3" v2="0" v3="4"/><triangle v1="3" v2="4" v3="7"/>
 </triangles></mesh></object></resources>
 <build><item objectid="1"/></build>
</model>'''
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
                   'package/2006/content-types"><Default Extension="rels" ContentType='
                   '"application/vnd.openxmlformats-package.relationships+xml"/><Default '
                   'Extension="model" ContentType='
                   '"application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0"?><Relationships xmlns="http://schemas.'
                   'openxmlformats.org/package/2006/relationships"><Relationship '
                   'Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.'
                   'com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr("3D/3dmodel.model", model)


class MetadataTest(unittest.TestCase):
    def test_3mf_inch_metadata_is_recovered_from_the_source_file(self):
        path = os.path.join(tempfile.mkdtemp(prefix="k4_"), "inch.3mf")
        write_3mf(path, "inch")
        info = units.read_source_unit(path)
        self.assertTrue(info["explicit"])
        self.assertEqual(info["unit"], "inch")
        report = units.analyse_units(box_mesh(2, 1, 0.5), source_path=path)
        self.assertEqual(report["status"], units.UNIT_KNOWN)
        self.assertEqual(report["source_unit"], "inch")

    def test_3mf_millimeter_metadata(self):
        path = os.path.join(tempfile.mkdtemp(prefix="k4_"), "mm.3mf")
        write_3mf(path, "millimeter")
        report = units.analyse_units(box_mesh(30, 20, 10), source_path=path)
        self.assertEqual((report["status"], report["source_unit"]),
                         (units.UNIT_KNOWN, "mm"))

    def test_stl_has_no_metadata_and_stays_unknown(self):
        report = units.analyse_units(box_mesh(30, 20, 10))          # no source path
        self.assertEqual(report["source_unit"], "unknown")
        self.assertNotEqual(report["status"], units.UNIT_KNOWN,
                            "unknown units must never become known by themselves")

    def test_obj_header_hint_is_a_hint_not_metadata(self):
        path = os.path.join(tempfile.mkdtemp(prefix="k4_"), "hint.obj")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# units: inches\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        info = units.read_source_unit(path)
        self.assertFalse(info["explicit"])
        self.assertEqual(info["unit"], "unknown")
        self.assertTrue(info.get("hint"))


class PlausibilityTest(unittest.TestCase):
    def test_mm_grid_dimensions_are_plausible_not_mismatch(self):
        report = units.analyse_units(box_mesh(30, 20, 10))
        self.assertEqual(report["status"], units.SCALE_PLAUSIBLE)
        self.assertEqual(report["source_unit"], "unknown")
        self.assertTrue(any("mm grid" in e for e in report["evidence"]))

    def test_inch_designed_dimensions_flag_a_possible_mismatch(self):
        report = units.analyse_units(box_mesh(50.8, 25.4, 12.7))
        self.assertEqual(report["status"], units.POSSIBLE_MM_INCH_MISMATCH)
        self.assertTrue(any("inch fraction" in e for e in report["evidence"]))
        self.assertIn("confirm", report["recommended_action"])

    def test_dimensions_exact_in_both_idioms_are_ambiguous(self):
        report = units.analyse_units(box_mesh(127.0, 127.0, 127.0))   # 5 in == 127 mm
        self.assertEqual(report["status"], units.SCALE_AMBIGUOUS)

    def test_very_small_is_a_warning_not_a_mismatch(self):
        report = units.analyse_units(box_mesh(0.5, 0.3, 0.2))
        self.assertNotEqual(report["status"], units.POSSIBLE_MM_INCH_MISMATCH)
        self.assertTrue(any("very small" in w for w in report["warnings"]))

    def test_very_large_is_a_warning_not_a_mismatch(self):
        report = units.analyse_units(box_mesh(20000, 12000, 5000))
        self.assertNotEqual(report["status"], units.POSSIBLE_MM_INCH_MISMATCH)
        self.assertTrue(any("very large" in w for w in report["warnings"]))

    def test_user_confirmation_makes_the_unit_known(self):
        report = units.analyse_units(box_mesh(50.8, 25.4, 12.7), user_unit="inch")
        self.assertEqual(report["status"], units.UNIT_KNOWN)
        self.assertEqual(report["source_unit_origin"], "user confirmation")


class ConversionTest(unittest.TestCase):
    def test_inch_to_mm_on_a_working_copy(self):
        src = box_mesh(2, 1, 0.5)
        out = units.scaled_copy(src, units.INCH_MM)
        self.assertAlmostEqual(out.BoundBox.XLength, 2 * 25.4, places=4)
        self.assertAlmostEqual(out.BoundBox.ZLength, 0.5 * 25.4, places=4)
        self.assertEqual(out.CountFacets, src.CountFacets)
        self.assertAlmostEqual(src.BoundBox.XLength, 2.0, places=6,
                                msg="the ORIGINAL mesh is never modified")

    def test_mm_to_inches(self):
        out = units.scaled_copy(box_mesh(50.8, 25.4, 12.7), 1.0 / units.INCH_MM)
        self.assertAlmostEqual(out.BoundBox.XLength, 2.0, places=4)

    def test_conversion_validation(self):
        src = box_mesh(2, 1, 0.5)
        out = units.scaled_copy(src, units.INCH_MM)
        v = units.conversion_validation(src, out, units.INCH_MM)
        self.assertTrue(v["valid"])
        self.assertTrue(v["facet_count_preserved"])

    def test_conversion_is_recorded_separately_and_provenance_survives(self):
        src = box_mesh(50.8, 25.4, 12.7)
        out = units.scaled_copy(src, units.INCH_MM)
        ledger = provenance.new_ledger()
        units.record_conversion(ledger, units.INCH_MM, src, out)
        self.assertEqual(len(ledger["unit_operations"]), 1)
        self.assertEqual(ledger["operations"], [],
                         "unit operations are NOT geometry repair entries")
        v = provenance.verify_against_mesh(ledger, out)
        self.assertEqual((v["verdict"], v["method"]), ("VERIFIED", "mesh-digest"))

    def test_conversion_and_confirmation_survive_save_reopen(self):
        src = box_mesh(2, 1, 0.5)
        out = units.scaled_copy(src, units.INCH_MM)
        tmp = os.path.join(tempfile.mkdtemp(prefix="k4c_"), "c.FCStd")
        doc = App.newDocument("k4c")
        obj = doc.addObject("Mesh::Feature", "Scan")
        obj.Mesh = out
        obj.addProperty("App::PropertyString", "OscarSourceUnit", "Oscar", "user unit")
        obj.OscarSourceUnit = "inch"
        doc.saveAs(tmp)
        App.closeDocument(doc.Name)
        d2 = App.openDocument(tmp)
        back = d2.getObject("Scan")
        self.assertEqual(back.OscarSourceUnit, "inch")
        self.assertAlmostEqual(back.Mesh.BoundBox.XLength, 50.8, places=3)
        App.closeDocument(d2.Name)


class GateTest(unittest.TestCase):
    """An unconfirmed mm/inch interpretation can never yield a VERIFIED outcome."""

    def _inch_bracket(self):
        shape = Part.makeBox(152.4, 101.6, 25.4)                  # 6 x 4 x 1 inch
        r = 3.3 * 2.54
        for x, y in ((38.1, 25.4), (114.3, 25.4), (38.1, 76.2), (114.3, 76.2)):
            shape = shape.cut(Part.makeCylinder(r, 80, App.Vector(x, y, -20)))
        return Mesh.Mesh(shape.tessellate(0.762))

    def test_unconfirmed_mismatch_caps_the_outcome(self):
        from oscar.recon import reconstruct

        mesh = self._inch_bracket()
        report = units.analyse_units(mesh)
        self.assertEqual(report["status"], units.POSSIBLE_MM_INCH_MISMATCH)
        doc = App.newDocument("k4gate")
        self.addCleanup(App.closeDocument, "k4gate")
        obj = doc.addObject("Mesh::Feature", "Scan")
        obj.Mesh = mesh
        doc.recompute()
        result = reconstruct.reconstruct(doc, mesh, warn_mm=2.0)
        self.assertTrue(result.get("supported"), result.get("note"))
        self.assertNotEqual(result["outcome"], "verified_parametric")
        self.assertEqual(result["meta"]["units_unconfirmed"],
                         units.POSSIBLE_MM_INCH_MISMATCH)

    def test_user_confirmed_mm_allows_full_verification(self):
        from oscar.recon import reconstruct

        mesh = self._inch_bracket()
        doc = App.newDocument("k4gate2")
        self.addCleanup(App.closeDocument, "k4gate2")
        obj = doc.addObject("Mesh::Feature", "Scan")
        obj.Mesh = mesh
        doc.recompute()
        result = reconstruct.reconstruct(doc, mesh, warn_mm=2.0, user_unit="mm")
        self.assertTrue(result.get("supported"), result.get("note"))
        self.assertEqual(result["outcome"], "verified_parametric",
                         "a confirmed unit interpretation removes the cap")
        self.assertNotIn("units_unconfirmed", result["meta"])


class IntegrationTest(unittest.TestCase):
    def test_analysis_payload_carries_the_unit_report(self):
        from oscar.tools_recon import _analysis_payload

        doc = App.newDocument("k4i")
        self.addCleanup(App.closeDocument, "k4i")
        obj = doc.addObject("Mesh::Feature", "Scan")
        obj.Mesh = box_mesh(50.8, 25.4, 12.7)
        report = units.analyse_units(obj.Mesh)
        payload = _analysis_payload("analyse", obj, "k4i", None, None, None, None,
                                    extra={"units": report, "user_unit": None})
        self.assertEqual(payload["extra"]["units"]["status"],
                         units.POSSIBLE_MM_INCH_MISMATCH)
        json.dumps(payload["extra"])                    # fully JSON-serialisable

    def test_the_tool_renders_the_unit_status(self):
        from oscar import tools
        from oscar.toolctx import ToolContext, use_context

        doc = App.newDocument("k4r")
        self.addCleanup(App.closeDocument, "k4r")
        obj = doc.addObject("Mesh::Feature", "Scan")
        obj.Mesh = box_mesh(50.8, 25.4, 12.7)
        doc.recompute()
        with use_context(ToolContext()):
            out = tools.run_tool("reconstruct_mesh", {"doc_name": "k4r",
                                                      "mode": "analyse"})
        text = out[0] if isinstance(out, tuple) else out
        self.assertIn("UNITS: POSSIBLE_MM_INCH_MISMATCH", text)
        self.assertIn("please confirm the intended units", text)

    def test_user_confirmation_is_persisted_on_the_mesh(self):
        from oscar import tools
        from oscar.toolctx import ToolContext, use_context

        doc = App.newDocument("k4u")
        self.addCleanup(App.closeDocument, "k4u")
        obj = doc.addObject("Mesh::Feature", "Scan")
        obj.Mesh = box_mesh(30, 20, 10)
        doc.recompute()
        with use_context(ToolContext()):
            out = tools.run_tool("reconstruct_mesh",
                                 {"doc_name": "k4u", "mode": "analyse",
                                  "confirm_units": "mm"})
        text = out[0] if isinstance(out, tuple) else out
        self.assertIn("user confirmed: mm", text)
        self.assertEqual(doc.getObject("Scan").OscarSourceUnit, "mm")


if __name__ == "__main__":
    unittest.main()
