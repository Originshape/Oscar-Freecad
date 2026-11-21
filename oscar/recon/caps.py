"""Runtime capability probe for the fitted-surface (hybrid freeform) route.

Phase 6 (docs/PHASE6_HYBRID_FREEFORM_ASSESSMENT.md §3, spec §14): the hybrid route
may only run on a build whose ReverseEngineering surface fitter and BSpline
interpolation were EXERCISED, not presumed. Availability is therefore probed by
calling the API on a tiny synthetic patch — never inferred from a workbench NAME,
an import success alone, or another build's behavior (PCL-gated functions are
compile-time optional). Failures are DATA (``available: False`` + the exact error
text), never exceptions. No document access; safe on any thread; the result is
cached per process (~0.1 s to re-probe with ``refresh=True``).

Downgrade contract: a False cap means the route refuses (the existing honest
"not implemented" text), it never crashes and never silently degrades.
"""

import time

# Bumped whenever a probe's fixture or acceptance rule changes, so stale cached
# verdicts from an older probe cannot leak through a long-lived process.
PROBE_VERSION = 1

# Default poles ONLY (assessment §2.2: 16x16/20x20 poles are 100-500x slower for
# ~1% area gain, and 20x20 folded on the full sphere — knob-raising is forbidden).
PROBE_POLES = (6, 6)

_cache = None            # per-process verdict (spec: cheap to re-run, but no need)
_fixture = None          # (Mesh, n_points) synthetic patch, built once per process

# 10 analytic points of z = 0.2*x^2 (the interpolation probe input)
_INTERP_XS = (-3.0, -2.33, -1.67, -1.0, -0.33, 0.33, 1.0, 1.67, 2.33, 3.0)


def _synthetic_patch():
    """The approxSurface probe fixture: an 11x11 grid (121 points, 200 facets) on
    z = 0.05*(x^2 + y^2) — big enough that the fitter's "Too less data points"
    ValueError can never fire on it (a 24-facet patch raises it; assessment §2.2)."""
    global _fixture
    if _fixture is None:
        import FreeCAD as App
        import Mesh

        n = 11
        xs = [(-5.0 + 10.0 * i / (n - 1)) for i in range(n)]
        grid = [(x, y, 0.05 * (x * x + y * y)) for x in xs for y in xs]
        mesh = Mesh.Mesh()
        for i in range(n - 1):
            for j in range(n - 1):
                a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
                mesh.addFacet(App.Vector(*grid[a]), App.Vector(*grid[b]), App.Vector(*grid[d]))
                mesh.addFacet(App.Vector(*grid[a]), App.Vector(*grid[d]), App.Vector(*grid[c]))
        _fixture = (mesh, len(grid))
    return _fixture


def _probe_approx_surface(re_mod=None):
    """(available, evidence, ms) for ReverseEngineering.approxSurface. Never raises.
    ``re_mod`` lets tests inject a stub (a hidden import / a name without the API /
    an operation failure) — the production path imports the real module."""
    t0 = time.perf_counter()

    def _done(ok, text):
        return ok, text, (time.perf_counter() - t0) * 1000.0

    try:
        if re_mod is None:
            import ReverseEngineering as re_mod          # try-import: the failure is data
        if not hasattr(re_mod, "approxSurface"):
            return _done(False, "ReverseEngineering imported but has no approxSurface "
                                "attribute - a module/workbench NAME never implies the "
                                "capability (spec 14)")
        mesh, n_points = _synthetic_patch()
        surface = re_mod.approxSurface(mesh, NbUPoles=PROBE_POLES[0], NbVPoles=PROBE_POLES[1])
        shape = surface.toShape()
        if not shape.isValid():
            return _done(False, f"approxSurface returned an invalid shape on a "
                                f"{n_points}-point synthetic patch ({type(surface).__name__})")
        return _done(True, f"ReverseEngineering.approxSurface fitted the {n_points}-point "
                           f"synthetic patch ({PROBE_POLES[0]}x{PROBE_POLES[1]} default poles, "
                           f"valid {type(shape).__name__} face); DEFAULT POLES ONLY - higher "
                           f"pole counts measured 100-500x slower for ~1% gain and can fold; "
                           f"surface closure is NOT a capability of this API (single patch)")
    except Exception as exc:                             # noqa: BLE001 - failures are data
        return _done(False, f"approxSurface probe failed: {type(exc).__name__}: {exc}")


def _probe_bspline_interpolate(part_mod=None):
    """(available, evidence, ms) for Part.BSplineCurve.interpolate: degree and an
    exact pass-through check (< 1e-9 mm) on 10 analytic points. Never raises."""
    t0 = time.perf_counter()

    def _done(ok, text):
        return ok, text, (time.perf_counter() - t0) * 1000.0

    try:
        if part_mod is None:
            import Part as part_mod
        import FreeCAD as App

        curve = part_mod.BSplineCurve()
        points = [App.Vector(x, 0.2 * x * x, 0.0) for x in _INTERP_XS]
        curve.interpolate(points)
        shape = curve.toShape()
        worst = max(float(shape.distToShape(part_mod.Vertex(p))[0]) for p in points)
        degree = int(curve.Degree)
        if worst > 1e-9:
            return _done(False, f"interpolate is not pass-through: worst {worst:.1e} mm "
                                f"> 1e-9 on {len(points)} analytic points")
        return _done(True, f"Part.BSplineCurve.interpolate: {len(points)} analytic points, "
                           f"degree {degree}, pass-through worst {worst:.1e} mm")
    except Exception as exc:                             # noqa: BLE001 - failures are data
        return _done(False, f"bspline interpolate probe failed: {type(exc).__name__}: {exc}")


def freeform_caps(refresh=False):
    """The runtime verdict for the fitted-surface route (spec 14):

    ``{"version": PROBE_VERSION,
        "approx_surface":      {"available", "evidence", "ms", "poles"},
        "bspline_interpolate": {"available", "evidence", "ms"}}``

    Cached per process unless ``refresh=True``. Consumers (reconstruct route
    selection, CI) treat ``available: False`` as refusal, never as an error."""
    global _cache
    if _cache is not None and not refresh:
        return _cache
    ok_a, ev_a, ms_a = _probe_approx_surface()
    ok_b, ev_b, ms_b = _probe_bspline_interpolate()
    _cache = {"version": PROBE_VERSION,
              "approx_surface": {"available": ok_a, "evidence": ev_a, "ms": round(ms_a, 3),
                                 "poles": list(PROBE_POLES)},
              "bspline_interpolate": {"available": ok_b, "evidence": ev_b, "ms": round(ms_b, 3)}}
    return _cache
