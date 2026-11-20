"""Phase 6: ``oscar.recon.caps`` (runtime capability probe) and
``oscar.recon.freeform`` (hybrid-route region fitting).

The caps probe must EXERCISE the API (a tiny synthetic approxSurface fit, a real
BSplineCurve interpolation) and record failures as data; a hidden import, a module
present without the API, or an operation failure all downgrade to
``available: False`` with the exact error as evidence — never an exception, never
availability inferred from a name.

The fitter runs REAL approxSurface fits on the assessment's fixtures — the
hemisphere (4,000 facets) and quarter-sphere (~2,000 facets) patches of a
tessellated Part.makeSphere(10) — but NEVER the full sphere: a single default-pole
fit of a closed surface is degenerate (one bbox axis collapses to 0.00 mm; a 20x20
fit folds to 112.78 mm on a 20 mm sphere), which is exactly what the pure
validation tests pin with the assessment's measured numbers. Fits that starve the
fitter raise its informative ValueError, which must surface as a LOUD rejection.
All runtimes here are bounded (every real fit is 0.1-0.6 s at default poles).
"""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
import Part  # noqa: E402
import numpy as np  # noqa: E402

from oscar.recon import caps as recon_caps  # noqa: E402
from oscar.recon import freeform  # noqa: E402

SPHERE_R = 10.0
_fix = {}


def fixtures():
    """The assessment's fixtures, built once: sphere points/facets arrays, the
    hemisphere and quarter-sphere facet selections, their areas/extents, and the
    full-sphere work mesh for the driver tests."""
    if not _fix:
        points, facets = Part.makeSphere(SPHERE_R).tessellate(1.0)   # 8,000 facets / 4,002 pts
        pts = np.array([[p.x, p.y, p.z] for p in points], dtype=float)
        fac = np.array(facets, dtype=np.int64)
        cent = pts[fac].mean(axis=1)
        hemi = np.nonzero(cent[:, 2] >= 0.0)[0]
        quarter = np.nonzero((cent[:, 2] >= 0.0) & (cent[:, 1] >= 0.0))[0]
        a, b, c = pts[fac[:, 0]], pts[fac[:, 1]], pts[fac[:, 2]]
        all_areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        work = Mesh.Mesh()
        for tri in fac:
            work.addFacet(*[App.Vector(*pts[v]) for v in tri])
        _fix.update(pts=pts, fac=fac, hemi=hemi, quarter=quarter,
                    total_area=float(all_areas.sum()),
                    hemi_area=float(all_areas[hemi].sum()),
                    quarter_area=float(all_areas[quarter].sum()),
                    hemi_ext=freeform._region_extents(pts, fac, hemi),
                    quarter_ext=freeform._region_extents(pts, fac, quarter),
                    work=work)
    return _fix


def unavailable_caps(evidence="mock: hidden on this build"):
    return {"version": recon_caps.PROBE_VERSION,
            "approx_surface": {"available": False, "evidence": evidence, "ms": 0.0,
                               "poles": [6, 6]},
            "bspline_interpolate": {"available": True, "evidence": "mock", "ms": 0.0}}


class CapsProbeTest(unittest.TestCase):
    def test_real_probe_is_available_and_versioned_on_supported_builds(self):
        caps = recon_caps.freeform_caps(refresh=True)
        self.assertEqual(caps["version"], recon_caps.PROBE_VERSION)
        for name in ("approx_surface", "bspline_interpolate"):
            cap = caps[name]
            self.assertIsInstance(cap["available"], bool)
            self.assertIsInstance(cap["evidence"], str)
            self.assertGreaterEqual(cap["ms"], 0.0)
        # both capabilities were probed live on 1.1.3 and 1.1.4 (assessment 2.1);
        # if this ever fails the evidence string must say exactly why
        self.assertTrue(caps["approx_surface"]["available"],
                        caps["approx_surface"]["evidence"])
        self.assertTrue(caps["bspline_interpolate"]["available"],
                        caps["bspline_interpolate"]["evidence"])
        self.assertEqual(caps["approx_surface"]["poles"], [6, 6])
        self.assertIn("closure is NOT", caps["approx_surface"]["evidence"])

    def test_the_verdict_is_cached_per_process(self):
        first = recon_caps.freeform_caps(refresh=True)
        self.assertIs(recon_caps.freeform_caps(), first)
        self.assertIsNot(recon_caps.freeform_caps(refresh=True), first)


class CapsDowngradeTest(unittest.TestCase):
    """Failures are data: a hidden import, a name without the API, an operation
    failure — every path returns available=False plus the exact error text."""

    def setUp(self):
        sys.modules.pop("ReverseEngineering", None)

    def tearDown(self):
        sys.modules.pop("ReverseEngineering", None)
        recon_caps.freeform_caps(refresh=True)      # restore the real cached verdict

    def test_hidden_import_downgrades_with_the_error_as_evidence(self):
        sys.modules["ReverseEngineering"] = None    # a blocked/hidden import
        caps = recon_caps.freeform_caps(refresh=True)
        self.assertFalse(caps["approx_surface"]["available"])
        self.assertIn("ReverseEngineering", caps["approx_surface"]["evidence"])
        self.assertIn("failed", caps["approx_surface"]["evidence"])
        # an independent cap is not poisoned by the neighbour's failure
        self.assertTrue(caps["bspline_interpolate"]["available"])

    def test_a_module_name_without_the_api_never_counts(self):
        sys.modules["ReverseEngineering"] = types.ModuleType("ReverseEngineering")
        caps = recon_caps.freeform_caps(refresh=True)
        self.assertFalse(caps["approx_surface"]["available"])
        self.assertIn("approxSurface", caps["approx_surface"]["evidence"])

    def test_probe_operation_failure_is_data_not_an_exception(self):
        fake = types.ModuleType("ReverseEngineering")

        def boom(*args, **kwargs):
            raise RuntimeError("kernel exploded")

        fake.approxSurface = boom
        ok, evidence, _ms = recon_caps._probe_approx_surface(re_mod=fake)
        self.assertFalse(ok)
        self.assertIn("RuntimeError", evidence)
        self.assertIn("kernel exploded", evidence)

    def test_bspline_probe_failure_is_data_not_an_exception(self):
        class _BrokenCurve:
            def interpolate(self, points):
                raise RuntimeError("no interpolation here")

        fake = types.SimpleNamespace(BSplineCurve=_BrokenCurve,
                                     Vector=App.Vector, Vertex=Part.Vertex)
        ok, evidence, _ms = recon_caps._probe_bspline_interpolate(part_mod=fake)
        self.assertFalse(ok)
        self.assertIn("RuntimeError", evidence)
        self.assertIn("no interpolation here", evidence)


class FitValidationNumbersTest(unittest.TestCase):
    """PURE validation logic pinned to the assessment's measured numbers (2.2)."""

    def test_measured_hemisphere_fit_passes(self):
        ok, reason = freeform._check_fit_numbers(735.0, (20.71, 20.71, 15.13),
                                                 627.8, (19.99, 19.99, 10.35))
        self.assertTrue(ok, reason)

    def test_measured_quarter_sphere_fit_passes(self):
        fix = fixtures()
        ok, reason = freeform._check_fit_numbers(371.9, (21.14, 12.52, 12.63),
                                                 fix["quarter_area"], fix["quarter_ext"])
        self.assertTrue(ok, reason)

    def test_full_sphere_collapse_is_rejected(self):
        ok, reason = freeform._check_fit_numbers(362.6, (20.09, 0.00, 20.07),
                                                 1256.6, (20.09, 20.07, 20.09))
        self.assertFalse(ok)
        self.assertIn("collapsed", reason)

    def test_folded_surface_is_rejected(self):
        ok, reason = freeform._check_fit_numbers(1023.7, (20.0, 112.78, 20.0),
                                                 1256.6, (20.09, 20.07, 20.09))
        self.assertFalse(ok)                        # area ratio 0.814 is IN band -
        self.assertIn("overshoots", reason)         # only the bbox guard catches it

    def test_area_ratio_band_boundaries(self):
        reg, ext = 600.0, (20.0, 20.0, 10.0)
        for fitted in (300.0, 1200.0):              # exactly 0.5x / 2.0x: inside
            ok, reason = freeform._check_fit_numbers(fitted, ext, reg, ext)
            self.assertTrue(ok, reason)
        for fitted in (294.0, 1206.0):              # just outside: loud rejection
            ok, reason = freeform._check_fit_numbers(fitted, ext, reg, ext)
            self.assertFalse(ok)
            self.assertIn("area ratio", reason)

    def test_zero_area_fit_is_rejected(self):
        ok, reason = freeform._check_fit_numbers(0.0, (20, 20, 10), 600.0, (20, 20, 10))
        self.assertFalse(ok)
        self.assertIn("no area", reason)


class RegionFitTest(unittest.TestCase):
    """REAL default-pole fits on the assessment's open-patch fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.fix = fixtures()
        cls.docs_before = dict(App.listDocuments())

    def test_hemisphere_patch_fits_at_default_poles(self):
        rec = freeform.fit_region(self.fix["pts"], self.fix["fac"], self.fix["hemi"],
                                  self.fix["hemi_area"], 0,
                                  total_area=self.fix["total_area"])
        self.assertTrue(rec["ok"], rec)
        patch = rec["patch"]
        for key in ("surface", "region_area", "source_region_id", "coverage_share"):
            self.assertIn(key, patch)
        self.assertIsInstance(patch["surface"], Part.BSplineSurface)
        self.assertTrue(patch["surface"].toShape().isValid())
        q = patch["quality"]
        self.assertEqual(q["poles"], [6, 6])        # DEFAULT poles only, ever
        self.assertGreater(q["ms"], 0.0)
        self.assertGreater(q["area_ratio"], 1.0)    # measured overshoot 7-18%
        self.assertLess(q["area_ratio"], 1.35)
        self.assertGreater(q["points"], freeform.MIN_REGION_POINTS)
        self.assertAlmostEqual(patch["coverage_share"], 0.5, delta=0.02)
        self.assertEqual(patch["source_region_id"], 0)
        self.assertAlmostEqual(patch["region_area"], self.fix["hemi_area"], places=1)

    def test_quarter_sphere_patch_fits(self):
        rec = freeform.fit_region(self.fix["pts"], self.fix["fac"], self.fix["quarter"],
                                  self.fix["quarter_area"], 3,
                                  total_area=self.fix["total_area"])
        self.assertTrue(rec["ok"], rec)
        self.assertAlmostEqual(rec["patch"]["coverage_share"], 0.25, delta=0.03)
        self.assertEqual(rec["patch"]["source_region_id"], 3)
        self.assertTrue(1.0 <= rec["patch"]["quality"]["area_ratio"] <= 1.35)

    def test_fitting_never_touches_any_document(self):
        freeform.fit_region(self.fix["pts"], self.fix["fac"], self.fix["hemi"],
                            self.fix["hemi_area"], 0, total_area=self.fix["total_area"])
        self.assertEqual(dict(App.listDocuments()), self.docs_before)

    def test_small_region_is_skipped_before_the_fitter_runs(self):
        rec = freeform.fit_region(self.fix["pts"], self.fix["fac"],
                                  self.fix["hemi"][:30], 5.0, 0)
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["stage"], "skipped")
        self.assertIn("points", rec["reason"])

    def test_over_cap_region_is_skipped_for_the_time_budget(self):
        idx = np.tile(self.fix["hemi"], 4)          # 16,000 declared facets > cap
        rec = freeform.fit_region(self.fix["pts"], self.fix["fac"], idx, 100.0, 0)
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["stage"], "skipped")
        self.assertIn("facet", rec["reason"])

    def test_starved_fit_is_a_loud_rejection_not_a_crash(self):
        empty = Mesh.Mesh()                         # the fitter's informative error
        rec = freeform._fit_and_validate(empty, 0, 10.0, (4.0, 4.0, 4.0))
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["stage"], "fit")
        self.assertIn("Too less data points", rec["reason"])

    def test_area_lie_is_a_validation_rejection(self):
        # a real fit whose declared region area is a lie: the ratio band rejects it
        rec = freeform.fit_region(self.fix["pts"], self.fix["fac"], self.fix["hemi"],
                                  1.0e9, 0, total_area=self.fix["total_area"])
        self.assertFalse(rec["ok"])
        self.assertEqual(rec["stage"], "validation")
        self.assertIn("area ratio", rec["reason"])


class DriverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fix = fixtures()

    def survey(self, regions):
        return {"regions": regions,
                "coverage": {"planes": 0.0, "cylinders": 0.0, "unrecognized": 0.5}}

    def test_driver_fits_the_unrecognized_region(self):
        fix = self.fix
        survey = self.survey([{"type": "unrecognized", "_idx": fix["hemi"].tolist(),
                               "area_mm2": round(fix["hemi_area"], 3),
                               "facets": int(len(fix["hemi"]))}])
        before = dict(App.listDocuments())
        result = freeform.fit_region_patches(survey, fix["work"])
        self.assertEqual(dict(App.listDocuments()), before)     # no document writes
        self.assertNotIn("reason", result)
        self.assertEqual(len(result["fitted"]), 1)
        self.assertEqual(result["rejected"], [])
        self.assertEqual(result["skipped"], [])
        self.assertAlmostEqual(result["fitted"][0]["coverage_share"], 0.5, delta=0.02)
        self.assertEqual(result["caps"]["version"], recon_caps.PROBE_VERSION)

    def test_driver_refuses_up_front_when_caps_say_no(self):
        survey = self.survey([{"type": "unrecognized", "_idx": self.fix["hemi"].tolist(),
                               "area_mm2": 628.0, "facets": 4000}])
        result = freeform.fit_region_patches(survey, self.fix["work"],
                                             caps=unavailable_caps())
        self.assertEqual(result["fitted"], [])
        self.assertEqual(result["rejected"], [])
        self.assertEqual(result["skipped"], [])     # refused BEFORE any fitting
        self.assertIn("approxSurface unavailable", result["reason"])
        self.assertIn("mock: hidden", result["reason"])          # evidence echoed

    def test_driver_skips_regions_without_facet_indices(self):
        survey = self.survey([{"type": "unrecognized", "area_mm2": 628.0, "facets": 4000}])
        result = freeform.fit_region_patches(survey, self.fix["work"],
                                             caps=recon_caps.freeform_caps())
        self.assertEqual(result["fitted"], [])
        self.assertEqual(len(result["skipped"]), 1)
        self.assertIn("_idx", result["skipped"][0]["reason"])

    def test_driver_stops_at_the_staged_time_budget(self):
        survey = self.survey([{"type": "unrecognized", "_idx": self.fix["hemi"].tolist(),
                               "area_mm2": 628.0, "facets": 4000}])
        result = freeform.fit_region_patches(survey, self.fix["work"],
                                             caps=recon_caps.freeform_caps(),
                                             budget_ms=0.0)
        self.assertEqual(result["fitted"], [])
        self.assertEqual(len(result["skipped"]), 1)
        self.assertIn("budget", result["skipped"][0]["reason"])

    def test_failed_fit_stays_unrecognized_with_its_reason(self):
        fix = self.fix
        survey = self.survey([{"type": "unrecognized", "_idx": fix["hemi"].tolist(),
                               "area_mm2": 1.0e9, "facets": int(len(fix["hemi"]))}])
        result = freeform.fit_region_patches(survey, fix["work"],
                                             caps=recon_caps.freeform_caps())
        self.assertEqual(result["fitted"], [])
        self.assertEqual(len(result["rejected"]), 1)
        self.assertEqual(result["rejected"][0]["stage"], "validation")
        self.assertIn("area ratio", result["rejected"][0]["reason"])


if __name__ == "__main__":                          # pragma: no cover
    unittest.main()
