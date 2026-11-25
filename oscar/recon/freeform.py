"""Region fitting for the hybrid freeform route (Phase 6, assessment §4/§5).

Fits UNRECOGNIZED survey regions as approximate BSpline patches with
``ReverseEngineering.approxSurface`` at DEFAULT poles only, validates each fit
against the assessment's measured behavior, and returns isolated output
descriptors — ``{surface, region_area, source_region_id, coverage_share}`` — with
NO document writes: the seam (tools_recon, mirroring its mode="solid" pattern)
creates the labeled ``Part::Feature`` OUTSIDE the Body. A fitted region is never
presented as native editable parametric geometry, and any fitted share caps the
reconstruction outcome at ACCEPTED WITH WARNINGS at best (spec §12/§18.3).

Validation bands come from assessment §2.2 (this machine, both builds): good
default-pole fits overshoot area 7-18% (hemisphere 1.170, quarter 1.183) and the
equator somewhat; DEGENERATE fits collapse one bbox axis (full sphere: 0.00 mm on
a 20 mm mesh) or fold far past it (20x20 poles: 112.78 mm on a 20 mm mesh).
Degenerate fits are rejected LOUDLY (reason strings carry the numbers); failed or
skipped regions stay unrecognized — reported, never force-labeled (spec §9.1/§19).

SEAM NOTE (the two edits this module is built against; both are the LEAD's files,
not changed here):
1. ``recognize.py`` (~line 311, ``for r in regions: r.pop("_idx", None)``): keep
   ``_idx`` on ``type == "unrecognized"`` entries only — this module needs the
   facet indices to build region sub-meshes. Recognized regions stay index-free.
   ``reconstruct._SURVEY_CACHE_VERSION`` must bump (2 -> 3) so cached surveys
   predating the change miss.
2. ``reconstruct.py`` (~lines 1353-1359, the ``klass != "prismatic"`` refusal
   branch): when class is ``organic_or_hybrid`` or ``hybrid_mechanical`` and
   ``caps.freeform_caps()["approx_surface"]["available"]`` — call
   ``fit_region_patches(survey, work_mesh)``; on >= 1 patch set
   ``meta["freeform_fitted"] = True``, meta route ``hybrid_fitted_surfaces``;
   coverage vector gains ``"fitted_surfaces"`` (share from the fitted regions'
   ``area_mm2``), ``unrecognized`` shrinks to the still-unfitted share; on 0
   patches keep the existing refusal text byte-identical. The outcome demote
   condition (~lines 1483-1491) gains ``and "freeform_fitted" not in meta`` so a
   fitted region can never reach ``verified_parametric``.

FreeCAD module (Mesh/Part/ReverseEngineering imported lazily; pure geometry, no
document access anywhere in this file).
"""

import time

import numpy as np

from ._arrays import mesh_arrays

# DEFAULT poles only (assessment §2.2: 10x10 is 4-25x slower, 16x16 100x, 20x20
# 500x AND folded; ~1% area gain — knob-raising is a non-goal, §7).
DEFAULT_POLES = (6, 6)

# Region gates (assessment §4.2): point count >= ~4*poles^2 so the fitter is fed,
# facet cap for the time budget (>= 20k facets UNTESTED — capped, never promised;
# measured default-pole fits: 0.1-0.6 s at 2-4k facets).
MIN_REGION_POINTS = 4 * DEFAULT_POLES[0] * DEFAULT_POLES[1]      # 144
MAX_REGION_FACETS = 12000
DEFAULT_BUDGET_MS = 30000.0

# Validation bands (assessment §2.2 measured ranges; sanity band per §4.2)
AREA_RATIO_BAND = (0.5, 2.0)
BBOX_COLLAPSE_RATIO = 0.2    # full-sphere fit: 0.00 mm axis on a 20 mm region
BBOX_FOLD_RATIO = 3.0        # 20x20 folded fit: 112.78 mm axis on a 20 mm region
_BBOX_EPS_MM = 1e-3          # near-flat axes compare against a 1 um floor


def _tri_areas(pts, fac, idx):
    a, b, c = pts[fac[idx, 0]], pts[fac[idx, 1]], pts[fac[idx, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _region_extents(pts, fac, idx):
    """The region's own point-bbox extents (the validation reference)."""
    vertices = pts[np.unique(fac[idx])]
    span = vertices.max(axis=0) - vertices.min(axis=0)
    return tuple(float(s) for s in span)


def build_region_mesh(pts, fac, idx):
    """Assemble a region's facet subset as a standalone ``Mesh.Mesh`` (the input
    approxSurface was probed with). Local re-indexing; ~25 ms per 1,000 facets."""
    import FreeCAD as App
    import Mesh

    local = {}
    tris = []
    for tri in fac[idx]:
        out = []
        for v in tri.tolist():
            j = local.setdefault(v, len(local))
            out.append(j)
        tris.append(out)
    verts = np.zeros((len(local), 3), dtype=float)
    for v, j in local.items():
        verts[j] = pts[v]
    mesh = Mesh.Mesh()
    for a, b, c in tris:
        mesh.addFacet(App.Vector(*verts[a]), App.Vector(*verts[b]), App.Vector(*verts[c]))
    return mesh, len(local)


def _check_fit_numbers(fitted_area, fitted_extents, region_area, region_extents):
    """PURE validation of a fit's numbers against the assessment's bands:
    ``(ok, reason)``. ``fitted_extents``/``region_extents`` are the (x, y, z)
    bbox extents. Collapse and fold guards use the region's own bbox — the two
    decisive negatives of §2.2 (0.00 mm collapsed axis; 112.78 mm folded axis)."""
    if not region_area or region_area <= 0.0:
        return False, f"invalid region area {region_area}"
    if not fitted_area or fitted_area <= 0.0:
        return False, f"fit produced no area ({fitted_area})"
    for axis, (fit_e, reg_e) in enumerate(zip(fitted_extents, region_extents)):
        ref = max(float(reg_e), _BBOX_EPS_MM)
        got = max(float(fit_e), 0.0)
        if got < BBOX_COLLAPSE_RATIO * ref:
            return False, (f"degenerate fit: bbox axis {axis} collapsed to "
                           f"{got:.2f} mm ({got / ref:.3f}x the region's "
                           f"{ref:.2f} mm - single-patch closure artifact)")
        if got > BBOX_FOLD_RATIO * ref:
            return False, (f"degenerate fit: bbox axis {axis} overshoots to "
                           f"{got:.2f} mm ({got / ref:.2f}x the region's "
                           f"{ref:.2f} mm - folded/overshooting surface)")
    ratio = fitted_area / region_area
    lo, hi = AREA_RATIO_BAND
    if ratio < lo or ratio > hi:
        return False, (f"area ratio {ratio:.3f} outside the sanity band "
                       f"[{lo}, {hi}] (fitted {fitted_area:.1f} vs region "
                       f"{region_area:.1f} mm2)")
    return True, ""


def _fit_and_validate(mesh, n_points, region_area, region_extents, poles=DEFAULT_POLES):
    """One approxSurface call + validation. Returns ``{"ok": True, "patch": {...}}``
    or ``{"ok": False, "stage": "fit"|"validation", "reason": str}`` — never raises.
    The fitter's informative small-input ValueError ("Too less data points for
    the specified number of poles") lands here as a loud, catchable rejection."""
    t0 = time.perf_counter()

    def _reject(stage, reason):
        return {"ok": False, "stage": stage, "reason": reason}

    try:
        import ReverseEngineering as RE
        surface = RE.approxSurface(mesh, NbUPoles=poles[0], NbVPoles=poles[1])
        shape = surface.toShape()
        if not shape.isValid():
            return _reject("validation", "approxSurface returned a shape failing isValid()")
        bb = shape.BoundBox
        fitted_extents = (bb.XLength, bb.YLength, bb.ZLength)
        ok, reason = _check_fit_numbers(float(shape.Area), fitted_extents,
                                        float(region_area), region_extents)
        ms = (time.perf_counter() - t0) * 1000.0
        if not ok:
            return _reject("validation", reason)
        return {"ok": True, "patch": {"surface": surface,
                                      "region_area": float(region_area),
                                      "quality": {"area_ratio": round(float(shape.Area) / float(region_area), 3),
                                                  "points": int(n_points),
                                                  "poles": list(poles),
                                                  "ms": round(ms, 1),
                                                  "bbox_mm": [round(e, 3) for e in fitted_extents]}}}
    except Exception as exc:                             # noqa: BLE001 - a failed fit is data
        return _reject("fit", f"approxSurface failed: {type(exc).__name__}: {exc}")


def fit_region(pts, fac, idx, region_area, source_region_id, total_area=1.0):
    """Fit ONE region (``pts``/``fac`` arrays + facet ``idx``). Gates first (skip
    loudly, no fit attempted), then fit + validate. Returns the record above; on
    success ``record["patch"]`` additionally gains ``source_region_id`` and
    ``coverage_share`` and IS the isolated output descriptor (no document
    writes — the seam turns ``surface`` into a labeled Part::Feature)."""
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) == 0:
        return {"ok": False, "stage": "skipped", "reason": "region has no facets"}
    if len(idx) > MAX_REGION_FACETS:
        return {"ok": False, "stage": "skipped",
                "reason": f"{len(idx)} facets exceed the {MAX_REGION_FACETS}-facet "
                          "time-budget cap (default-pole policy; larger regions are "
                          "capped, never promised)"}
    vertices = np.unique(fac[idx])
    if len(vertices) < MIN_REGION_POINTS:
        return {"ok": False, "stage": "skipped",
                "reason": f"{len(vertices)} points < {MIN_REGION_POINTS} "
                          "(~4x poles^2) - the fitter would starve"}
    mesh, n_points = build_region_mesh(pts, fac, idx)
    record = _fit_and_validate(mesh, n_points, region_area,
                               _region_extents(pts, fac, idx))
    if record["ok"]:
        patch = record["patch"]
        patch["source_region_id"] = int(source_region_id)
        patch["coverage_share"] = round(float(region_area) / float(total_area or 1.0), 4)
    return record


def fit_region_patches(survey, work_mesh, caps=None, budget_ms=DEFAULT_BUDGET_MS):
    """The survey-level driver: fit every ``type == "unrecognized"`` region that
    carries ``_idx`` (the recognize.py seam keeps it there). Returns
    ``{"fitted": [descriptors], "rejected": [...], "skipped": [...], "caps": caps}``
    plus a top-level ``reason`` when the route is refused up front (caps say no).
    Regions that fail or are skipped stay unrecognized — reported with reasons,
    never force-labeled; a staged time budget bounds the whole pass."""
    from . import caps as _caps

    if caps is None:
        caps = _caps.freeform_caps()
    out = {"fitted": [], "rejected": [], "skipped": [], "caps": caps}
    if not (caps.get("approx_surface") or {}).get("available"):
        out["reason"] = ("approxSurface unavailable on this build - fitted-surface "
                         "route refused: " + str((caps.get("approx_surface") or {}).get("evidence")))
        return out
    pts, fac = mesh_arrays(work_mesh)
    total_area = float(getattr(work_mesh, "Area", 0.0)) \
        or sum(float(r.get("area_mm2") or 0.0) for r in survey.get("regions", [])) or 1.0
    started = time.perf_counter()
    for region_id, region in enumerate(survey.get("regions", [])):
        if region.get("type") != "unrecognized":
            continue
        idx = region.get("_idx")
        if not idx:
            out["skipped"].append({"source_region_id": region_id,
                                   "reason": "region carries no _idx facet indices "
                                             "(survey predates the recognize.py seam "
                                             "or a stale cache) - nothing fitted"})
            continue
        if budget_ms is not None and (time.perf_counter() - started) * 1000.0 > budget_ms:
            out["skipped"].append({"source_region_id": region_id,
                                   "reason": f"staged time budget ({budget_ms:.0f} ms) "
                                             "exhausted - region left unrecognized"})
            continue
        record = fit_region(pts, fac, idx, float(region.get("area_mm2") or 0.0),
                            region_id, total_area=total_area)
        if record["ok"]:
            out["fitted"].append(record["patch"])
        elif record.get("stage") == "skipped":
            out["skipped"].append({"source_region_id": region_id, "reason": record["reason"]})
        else:
            out["rejected"].append({"source_region_id": region_id,
                                    "stage": record["stage"], "reason": record["reason"]})
    return out
