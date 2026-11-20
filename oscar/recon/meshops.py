"""Mesh diagnostics and repair for the reconstruction workflow (master spec §8, M2; K2).

``diagnose`` reads a mesh and reports every affordable defect class FreeCAD can see — the
duplicate/degenerate counts recovered from throwaway copies (1.1.x has no read-only flags), plus
K2's numpy-on-``Topology`` layer: boundary edges and boundary loops/open chains
(with perimeter/size/position), zero-area / small / sliver / repeated-index facet counts
(scale-aware against the mesh's own median edge and median facet area), per-component detail
(count only, via the read-only ``countComponents``, unless the mesh is small enough to afford
``getSeparateComponents`` copies), tiny isolated components flagged as *candidates* (never
deleted), and, up to ``MAX_SELF_INTERSECTION_FACETS``, the kernel's ``getSelfIntersections``
pairs (confirmed intersections; the kernel already excludes adjacent triangles sharing an edge —
verified by fixture 2026-11-14). Larger scans report that this check was skipped instead of
blocking FreeCAD. A mesh
with boundary loops is not automatically defective: open meshes are legitimate.

``repair`` works on a COPY of the mesh only; the original imported scan is never touched. Safe
operations run by default, geometry-altering ones only when explicitly authorized. K2 adds
per-operation validation: every operation runs on its own trial copy, before/after snapshots
(vertex/facet/component counts, boundary edges and loops, area, volume only when the mesh is
closed both sides, bbox, sampled vertex displacement via the K1 neighbour kernel) are compared
against explicit, scale-aware rules (K1 sigma + mean edge), and the operation is ACCEPTED,
REJECTED (the trial copy is discarded — the mesh keeps the last good state), NEEDS_REVIEW, or
FAILED. The output being closed or having fewer defects is never itself a reason to accept.

Synthetic geometry: facets a fillupHoles added are identified by coordinate-keyed set
difference and recorded per operation (with parameters and the before/after measurements), and
registered as an in-mesh segment (``addSegment``) for THIS session. LIMITATION (verified):
FreeCAD mesh segments do NOT survive save/reopen of the document — the durable record is the
repair history returned here (tools persist it as an object property), so downstream stages
must read provenance from that record, not from the mesh. Honest by construction: synthetic
facet lists are data, never "measured scan evidence".

FreeCAD module (``Mesh``) + numpy; no document access, so any thread may call it.
"""

import math

import numpy as np

from . import provenance
from .quality import _knn, estimate_sigma

SAFE_OPERATIONS = ("harmonizeNormals", "fixIndices", "removeDuplicatedPoints",
                   "removeDuplicatedFacets", "fixDegenerations")
GATED_OPERATIONS = ("fillupHoles", "removeNonManifolds", "fixSelfIntersections",
                    "removeFoldsOnSurface")
ALL_OPERATIONS = SAFE_OPERATIONS + GATED_OPERATIONS

# ---- diagnostic limits (documented, configurable; all scale-aware where they judge geometry)
MAX_INTERSECTION_PAIRS = 50       # pairs listed per report; the COUNT is always complete
MAX_SELF_INTERSECTION_FACETS = 1_500  # laptop profile: bound the OCC pair search on the GUI thread
MAX_COMPONENT_DETAILS = 16        # per-component copies only for few components …
MAX_DETAIL_FACETS = 200_000       # … and small meshes; otherwise the count only (read-only)
MAX_LOOPS_LISTED = 20             # loops listed per report; counts are complete
ZERO_AREA_REL = 1e-10             # area ≤ rel·median_edge² → zero-area (absolute scale floor)
SMALL_AREA_REL = 0.01             # area < rel·median_facet_area → small (relative, evidence-free
                                  # claim about DEFECT, not about legitimate small detail)
SLIVER_ANGLE_DEG = 5.0            # min triangle angle below this → sliver
TINY_COMPONENT_FACETS = 8         # isolated component ≤ this many facets → junk CANDIDATE

# ---- repair validation thresholds (scale-aware; sigma from K1 when available)
AREA_REVIEW_REL = 0.005           # |Δarea| above this share (or 4·σ·√A) → NEEDS_REVIEW …
AREA_REJECT_REL = 0.05            # … above this share → REJECTED
VOLUME_REVIEW_REL = 0.005         # closed meshes only: |Δvol| shares as above
VOLUME_REJECT_REL = 0.05
DISPLACEMENT_SIGMA_MULT = 2.0     # sampled vertex displacement above max(2σ, 0.1·edge) …
DISPLACEMENT_EDGE_MULT = 0.1      # … means geometry MOVED → REJECTED for cleanup-class ops
FILL_SMALL_PERIMETER_EDGES = 10.0  # a fill closing a loop longer than 10·mean_edge → NEEDS_REVIEW
FILL_ARG_EDGES = 1.5              # fillupHoles max-edge argument = 1.5·mean_edge (G6 policy;
                                  # ≥5 floor kept: arg ≤2 was measured to close nothing on a
                                  # 20 mm box — Stage-1 probe)
FILL_MAX_EDGES = 8                # G6: a fillable loop is TINY (<= 8 edges) and flat —
                                  # everything else is never auto-filled (missing_region)


def _flags(mesh, *, check_self_intersections=True) -> dict:
    return {
        "closed": bool(mesh.isSolid()),
        "non_manifold": bool(mesh.hasNonManifolds()),
        "self_intersections": (bool(mesh.hasSelfIntersections())
                                if check_self_intersections else None),
        "inconsistent_normals": bool(mesh.hasNonUniformOrientedFacets()),
    }


def _counts_probe(mesh, operations) -> dict:
    """Facet/point counts after running ``operations`` on a throwaway copy — the only way to
    MEASURE duplicates and degenerates on FreeCAD 1.1.x (no read-only flags exist)."""
    work = mesh.copy()
    removed_facets = 0
    for op in operations:
        try:
            before_facets = int(work.CountFacets)
            getattr(work, op)()
            if op == "removeDuplicatedFacets":
                removed_facets = before_facets - int(work.CountFacets)
        except Exception:  # noqa: BLE001 - a probe that cannot run reports "unknown"
            return {}
    return {"facets": int(work.CountFacets), "points": int(work.CountPoints),
            "duplicate_facets": removed_facets}


def _arrays(mesh):
    """(n,3) float vertices and (m,3) int facets from ``mesh.Topology``."""
    from ._arrays import mesh_arrays

    return mesh_arrays(mesh)   # P4.1: the one shared materializer


def _edge_face_counts(fac):
    """Sorted undirected edge → number of incident facets (vectorized)."""
    e = np.concatenate([fac[:, [0, 1]], fac[:, [1, 2]], fac[:, [2, 0]]])
    e.sort(axis=1)
    edges, counts = np.unique(e, axis=0, return_counts=True)
    return edges, counts


def boundary(mesh) -> dict:
    """Boundary edges, closed loops and open chains (numpy on Topology, read-only).

    Loops/chains carry perimeter, bbox size and centroid so a caller can tell a small bore rim
    from an outer scan boundary. A boundary vertex with more than two boundary edges is a
    junction (non-manifold boundary); chains stop there and are counted, not guessed shut.
    """
    pts, fac = _arrays(mesh)
    return _boundary_arrays(pts, fac)


def _boundary_arrays(pts, fac) -> dict:
    """Boundary report from an already materialized topology array pair."""
    if len(fac) == 0:
        return {"edges": 0, "loops": [], "open_chains": [], "junctions": 0}
    edges, counts = _edge_face_counts(fac)
    bnd = edges[counts == 1]
    adj: dict = {}
    for a, b in bnd:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    junctions = sum(1 for v, ns in adj.items() if len(ns) > 2)

    def _emit(chain):
        ring = chain[:-1] if chain[0] == chain[-1] and len(chain) > 1 else chain
        v = pts[np.unique(ring)]
        size = v.max(axis=0) - v.min(axis=0)
        seg = pts[list(ring)]
        closed = ring[0] == chain[-1] and len(chain) > 1
        pts_ring = np.vstack([seg, seg[0]]) if closed else seg
        return {"vertices": len(set(ring)),
                # G5: the ordered ring (walk order, closing repeat stripped) so callers
                # can compute Newell normals / roundness without re-walking the boundary
                "ring": [int(i) for i in ring],
                "perimeter_mm": round(float(
                    np.linalg.norm(np.diff(pts_ring, axis=0), axis=1).sum()), 3),
                "size_mm": [round(float(s), 3) for s in size],
                "centroid_mm": [round(float(c), 3) for c in v.mean(axis=0)]}

    loops, chains = [], []
    visited = set()
    for a0, b0 in bnd:
        key = (int(a0), int(b0))
        if key in visited:
            continue
        walk, prev, cur = [int(a0)], int(a0), int(b0)
        visited.add((int(a0), int(b0)))
        visited.add((int(b0), int(a0)))
        while True:
            walk.append(cur)
            nxt = [n for n in adj.get(cur, []) if (cur, n) not in visited]
            if not nxt:
                break
            n = nxt[0]
            visited.add((cur, n))
            visited.add((n, cur))
            prev, cur = cur, n
            if cur == walk[0]:
                walk.append(cur)
                break
        if len(walk) > 2 and walk[0] == walk[-1]:
            loops.append(_emit(walk))
        else:
            chains.append(_emit(walk))
    return {"edges": int(len(bnd)), "loops": loops, "open_chains": chains,
            "junctions": junctions}


def _facet_quality_arrays(pts, fac) -> dict:
    """Scale-aware facet defect counts on raw (pts, fac) arrays: zero-area, small, sliver
    (min angle), repeated-index facets. Thresholds are RELATIVE to the mesh's own median
    edge / median facet area, so a legitimately fine tessellation is not misread as
    defective. (Array-level so the math is testable without the kernel's constructor-time
    sanitization, which silently drops hand-built degenerate facets.)"""
    out = {"zero_area_facets": 0, "small_facets": 0, "sliver_facets": 0,
           "repeated_index_facets": 0}
    if len(fac) == 0:
        return out
    p = pts[fac]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    edges = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]], axis=1)
    lengths = np.linalg.norm(edges, axis=2)
    median_edge = float(np.median(lengths))
    median_area = float(np.median(areas))
    repeated = (fac[:, 0] == fac[:, 1]) | (fac[:, 1] == fac[:, 2]) | (fac[:, 0] == fac[:, 2])
    out["repeated_index_facets"] = int(repeated.sum())
    zero = areas <= ZERO_AREA_REL * max(median_edge, 1e-12) ** 2
    out["zero_area_facets"] = int(zero.sum())
    if median_area > 0:
        out["small_facets"] = int(((areas < SMALL_AREA_REL * median_area) & ~zero).sum())
    e = np.maximum(lengths, 1e-15)
    cos_a = np.clip((e[:, 0] ** 2 + e[:, 1] ** 2 - e[:, 2] ** 2) / (2 * e[:, 0] * e[:, 1]),
                    -1.0, 1.0)
    cos_b = np.clip((e[:, 1] ** 2 + e[:, 2] ** 2 - e[:, 0] ** 2) / (2 * e[:, 1] * e[:, 2]),
                    -1.0, 1.0)
    cos_c = np.clip((e[:, 2] ** 2 + e[:, 0] ** 2 - e[:, 1] ** 2) / (2 * e[:, 2] * e[:, 0]),
                    -1.0, 1.0)
    min_ang = np.degrees(np.arccos(np.maximum.reduce([cos_a, cos_b, cos_c])))
    out["sliver_facets"] = int(((min_ang < SLIVER_ANGLE_DEG) & ~zero).sum())
    return out


def facet_quality(mesh) -> dict:
    """Scale-aware facet defect counts for a mesh (see ``_facet_quality_arrays``)."""
    pts, fac = _arrays(mesh)
    return _facet_quality_arrays(pts, fac)


def _components_detail(mesh, facets: int):
    """Read-only component count always; per-component copies only for few/small meshes."""
    try:
        count = int(mesh.countComponents())            # read-only, no copies
    except Exception:  # noqa: BLE001 - older kernels: fall back to the copying path
        try:
            return len(mesh.getSeparateComponents() or [mesh]), []
        except Exception:  # noqa: BLE001
            return 1, []
    if count <= 1:
        return max(count, 1), []
    detail = []
    if count <= MAX_COMPONENT_DETAILS and facets <= MAX_DETAIL_FACETS:
        try:
            for comp in mesh.getSeparateComponents():
                box = comp.BoundBox
                f = int(comp.CountFacets)
                detail.append({"facets": f, "points": int(comp.CountPoints),
                               "size_mm": [round(box.XLength, 3), round(box.YLength, 3),
                                           round(box.ZLength, 3)],
                               "tiny_candidate": f <= TINY_COMPONENT_FACETS})
        except Exception:  # noqa: BLE001 - detail is optional; the count stays
            detail = []
    return count, detail


def diagnose(mesh) -> dict:
    """Everything knowable about ``mesh`` without changing it: topology flags, counts, size,
    area, volume (only meaningful when closed), component count and detail, boundary edges /
    loops / open chains, facet-quality counts, confirmed self-intersection pairs (adjacency
    excluded by the kernel), tiny isolated components as candidates, and explicit warnings and
    limitations. Read-only (the duplicate/degenerate probe runs on a throwaway copy)."""
    facets, points = int(mesh.CountFacets), int(mesh.CountPoints)
    base = _flags(mesh, check_self_intersections=False)
    intersection_pairs = None
    intersection_count = None
    if facets <= MAX_SELF_INTERSECTION_FACETS:
        try:
            # One kernel query returns both the boolean and the localized pairs. Calling
            # hasSelfIntersections() first repeats the same expensive topology traversal.
            intersection_pairs = mesh.getSelfIntersections() or ()
            base["self_intersections"] = bool(intersection_pairs)
            intersection_count = len(intersection_pairs)
        except Exception:  # noqa: BLE001 - retain an honest unavailable result
            pass
    else:
        limitations_note = (f"self-intersection check skipped above the laptop resource limit "
                            f"({MAX_SELF_INTERSECTION_FACETS:,} facets)")
    if facets <= MAX_SELF_INTERSECTION_FACETS and intersection_pairs is None:
        limitations_note = "localized self-intersection pairs unavailable on this kernel"
    box = mesh.BoundBox
    warnings, limitations = [], []
    if facets > MAX_SELF_INTERSECTION_FACETS or intersection_pairs is None:
        limitations.append(limitations_note)
    out = {
        **base,
        "facets": facets,
        "points": points,
        "size_mm": [round(box.XLength, 4), round(box.YLength, 4), round(box.ZLength, 4)],
        "area_mm2": round(float(mesh.Area), 4),
        "components": 0,
    }
    if base["closed"]:
        out["volume_mm3"] = round(float(mesh.Volume), 4)
    count, detail = _components_detail(mesh, facets)
    out["components"] = count
    if detail:
        out["components_detail"] = detail
        tiny = [i for i, c in enumerate(detail) if c["tiny_candidate"]]
        if tiny:
            out["tiny_components"] = tiny
            warnings.append(f"{len(tiny)} tiny isolated component(s) — junk CANDIDATES, "
                            "nothing was deleted; inspect before authorizing removal")
    elif count > 1:
        limitations.append("per-component detail skipped (too many components or a very "
                           "large mesh); the count is complete")
    pts, fac = _arrays(mesh)
    bnd = _boundary_arrays(pts, fac)
    out["boundary_edges"] = bnd["edges"]
    out["boundary_loops"] = len(bnd["loops"])
    out["open_chains"] = len(bnd["open_chains"])
    if bnd["loops"] or bnd["open_chains"]:
        out["loops"] = (bnd["loops"] + bnd["open_chains"])[:MAX_LOOPS_LISTED]
    if bnd["junctions"]:
        out["boundary_junctions"] = bnd["junctions"]
        warnings.append(f"{bnd['junctions']} boundary junction vertex/vertices (>2 boundary "
                        "edges): loop/chain counts there are conservative")
    out.update(_facet_quality_arrays(pts, fac))
    out["self_intersection_pairs_count"] = intersection_count
    if intersection_pairs is not None:
        out["self_intersection_pairs"] = [((int(p[0]), int(p[1]))) for p in
                                          intersection_pairs[:MAX_INTERSECTION_PAIRS]]
        if len(intersection_pairs) > MAX_INTERSECTION_PAIRS:
            limitations.append(f"{len(intersection_pairs) - MAX_INTERSECTION_PAIRS} more intersection "
                               "pairs not listed (count is complete)")
    dedup = _counts_probe(mesh, ("removeDuplicatedPoints", "removeDuplicatedFacets"))
    degen = _counts_probe(mesh, ("fixDegenerations",))
    if dedup:
        out["duplicate_points"] = points - dedup["points"]
        out["duplicate_facets"] = max(0, dedup["duplicate_facets"])
    else:
        out["duplicate_points"] = None
        out["duplicate_facets"] = None
    if degen:
        out["degenerate_facets"] = facets - degen["facets"]
    else:
        out["degenerate_facets"] = None
    if not base["closed"] and bnd["edges"]:
        warnings.append("open mesh: boundary loops are geometry information, not "
                        "automatically defects")
    out["warnings"] = warnings
    out["limitations"] = limitations
    return out


def _duplicate_facet_count(mesh) -> int:
    probe = mesh.copy()
    try:
        probe.removeDuplicatedPoints()
        before = int(probe.CountFacets)
        probe.removeDuplicatedFacets()
        return before - int(probe.CountFacets)
    except Exception:  # noqa: BLE001
        return 0


# ------------------------------------------------------------------ K2-B: repair validation

def _snapshot(mesh) -> dict:
    """The comparable before/after state of a repair step (all read-only)."""
    pts, fac = _arrays(mesh)
    bnd = boundary(mesh)
    snap = {"points": int(mesh.CountPoints), "facets": int(mesh.CountFacets),
            "components": None, "boundary_edges": bnd["edges"],
            "boundary_loops": len(bnd["loops"]), "open_chains": len(bnd["open_chains"]),
            "area_mm2": float(mesh.Area),
            "size_mm": [mesh.BoundBox.XLength, mesh.BoundBox.YLength, mesh.BoundBox.ZLength]}
    try:
        snap["components"] = int(mesh.countComponents())
    except Exception:  # noqa: BLE001
        pass
    if mesh.isSolid():
        snap["volume_mm3"] = float(mesh.Volume)
    return snap


def _displacement(before_pts, after_pts, sample=400):
    """Max/rms nearest-neighbour distance of sampled before-vertices to the after-vertices
    (geometry-based, renumbering-safe). Uses the K1 neighbour kernel (cKDTree + fallback)."""
    step = max(1, len(before_pts) // sample)
    query = np.asarray(before_pts[::step], dtype=float)
    if len(query) == 0 or len(after_pts) == 0:
        return None
    d, _, _ = _knn(query, np.asarray(after_pts, dtype=float), 1, exclude_self=False)
    d = np.asarray(d, dtype=float).reshape(-1)
    return {"max_mm": round(float(d.max()), 6), "rms_mm": round(float(np.sqrt((d ** 2).mean())), 6),
            "sampled": int(len(query))}


def _validate_transition(op, before, after, sigma, mean_edge):
    """(verdict, reasons) for one repair operation. Explicit rules, scale-aware via the K1
    sigma and the mesh's mean edge; when safety cannot be established the answer is
    NEEDS_REVIEW, never a silent accept."""
    reasons = []
    diag = max(max(before["size_mm"]), 1e-9)
    scale = max(sigma or 0.0, 1e-9)
    edge = max(mean_edge or 0.0, 1e-9)
    reject = False
    review = False
    # geometry must not move (all supported ops except none: fills add, they do not slide)
    if before["points"]:
        if after["points"] < 4 or after["facets"] < 4:
            reasons.append("operation left almost no geometry")
            reject = True
    for i, axis in enumerate("xyz"):
        d = abs(after["size_mm"][i] - before["size_mm"][i])
        if d > max(scale, 1e-6 * diag):
            reasons.append(f"bbox {axis} changed by {d:.4f} mm")
            reject = True
    if before.get("components") is not None and after.get("components") is not None:
        if after["components"] > before["components"]:
            reasons.append(f"components split {before['components']}→{after['components']}")
            reject = True
        if after["components"] < before["components"]:
            reasons.append(f"components disappeared {before['components']}→"
                           f"{after['components']} — valid pieces may be gone")
            review = True
    a0, a1 = before["area_mm2"], after["area_mm2"]
    area_review = max(AREA_REVIEW_REL * a0, 4.0 * scale * math.sqrt(max(a0, 1e-12)))
    if op == "fillupHoles":
        # an authorized fill's area INCREASE is its purpose (it restores missing surface).
        # The explained cap is the isoperimetric bound of the loops it closed
        # (P²/4π each — the disk is the max-area curve for a given perimeter), computed by
        # the caller from the actual before/after loops; beyond it → damage.
        cap = before.get("_fill_area_cap")
        if a1 < a0 - area_review:
            reasons.append(f"fill DECREASED area {a0:.3f}→{a1:.3f} mm²")
            reject = True
        elif cap is not None and a1 > a0 + cap + 1e-6:
            reasons.append(f"fill added {a1 - a0:.1f} mm² — more than the closed loops can "
                           f"explain (isoperimetric cap {cap:.1f} mm²)")
            reject = True
    elif abs(a1 - a0) > AREA_REJECT_REL * a0:
        reasons.append(f"area changed {a0:.3f}→{a1:.3f} mm² "
                       f"({100 * (a1 - a0) / a0:+.2f}%)")
        reject = True
    elif abs(a1 - a0) > area_review:
        reasons.append(f"area changed {100 * (a1 - a0) / a0:+.2f}% beyond the noise band")
        review = True
    if op != "fillupHoles" and "volume_mm3" in before and "volume_mm3" in after:
        v0, v1 = before["volume_mm3"], after["volume_mm3"]
        if abs(v1 - v0) > VOLUME_REJECT_REL * abs(v0):
            reasons.append(f"volume changed {v0:.3f}→{v1:.3f} mm³ "
                           f"({100 * (v1 - v0) / max(abs(v0), 1e-9):+.2f}%) on a closed mesh")
            reject = True
        elif abs(v1 - v0) > max(VOLUME_REVIEW_REL * abs(v0), 3.0 * scale * a0):
            reasons.append("volume moved beyond the noise band")
            review = True
    else:
        reasons.append("volume not compared (open mesh before or after — volume would be "
                       "meaningless)")
    if after["boundary_loops"] > before["boundary_loops"] or \
       after["open_chains"] > before["open_chains"]:
        reasons.append("boundary loops/chains increased")
        review = True
    if reject:
        return "REJECTED", reasons
    if op == "fillupHoles":
        closed = before["boundary_loops"] - after["boundary_loops"]
        # G6: judge the loops the fill ACTUALLY closed (the mesh's largest loop may be an
        # outer rim the fill never touched — flagging that was a false review)
        if closed > 0 and before.get("_max_closed_perimeter") is not None and \
                before["_max_closed_perimeter"] > FILL_SMALL_PERIMETER_EDGES * edge:
            reasons.append(f"a large hole was closed (perimeter "
                           f"{before['_max_closed_perimeter']:.2f} mm > "
                           f"{FILL_SMALL_PERIMETER_EDGES:.0f}·mean edge {edge:.2f} mm)")
            review = True
    if review:
        return "NEEDS_REVIEW", reasons
    return "ACCEPTED", reasons


def _synthetic_facets(before_mesh, after_mesh):
    """Facet indices of ``after_mesh`` whose triangle (by rounded vertex coordinates,
    renumbering-safe) did not exist before the operation — the geometry this operation
    CREATED. Facet-level diff: a fill may build new triangles over OLD vertices, so
    new-vertex detection alone would miss it (measured 2026-11-14)."""
    _, fb = _arrays(before_mesh)
    pa, fa = _arrays(after_mesh)
    if len(fa) == 0:
        return []

    def facet_keys(pts, facs):
        tri = np.round(pts[facs], 5)
        order = np.lexsort((tri[:, :, 2], tri[:, :, 1], tri[:, :, 0]))
        tri = np.take_along_axis(tri, order[:, :, None], axis=1)
        return set(map(tuple, tri.reshape(len(facs), -1)))

    before_keys = facet_keys(*_arrays(before_mesh))
    tri = np.round(pa[fa], 5)
    order = np.lexsort((tri[:, :, 2], tri[:, :, 1], tri[:, :, 0]))
    tri = np.take_along_axis(tri, order[:, :, None], axis=1)
    rows = tri.reshape(len(fa), -1)
    mask = np.array([tuple(r) not in before_keys for r in rows], dtype=bool)
    return np.nonzero(mask)[0].tolist()


def _apply_op(mesh, op, mean_edge=None) -> dict:
    """Apply one named repair operation to ``mesh`` (in-place on the trial copy); returns
    the EFFECTIVE kernel arguments actually used (the fillupHoles max-edge integer is
    derived here from FILL_ARG_EDGES × mean_edge — approval binding records this exact
    value, so a changed derivation rule can never reuse an old authorization)."""
    fn = getattr(mesh, op)
    spec: dict = {}
    if op == "fillupHoles":
        arg = max(5, int(FILL_ARG_EDGES * (mean_edge or 1.0)))
        spec = {"max_edge": arg}
        try:
            fn(arg)
        except TypeError:
            fn()
    else:
        fn()
    return spec


def _predicted_spec(op: str, mean_edge) -> dict:
    """The spec a gated op WOULD use on this mesh — shown for refused (unauthorized)
    operations so the approval decision sees the effective parameters before granting."""
    if op == "fillupHoles":
        return {"max_edge": max(5, int(FILL_ARG_EDGES * (mean_edge or 1.0)))}
    return {}


def _component_sizes(mesh):
    """Sorted facet counts of the mesh's components, or None when unavailable."""
    try:
        return sorted(int(c.CountFacets) for c in mesh.getSeparateComponents())
    except Exception:  # noqa: BLE001
        return None


def _disappearances_explained(before_mesh, after_mesh):
    """True when every component the operation lost was tiny (degenerate debris — e.g. the
    pseudo-component a repeated-index facet forms), so the loss is explained, not damage."""
    b, a = _component_sizes(before_mesh), _component_sizes(after_mesh)
    if not b or not a:
        return False
    remaining = list(a)
    for size in b:
        if size in remaining:
            remaining.remove(size)
        elif size > TINY_COMPONENT_FACETS:
            return False           # a real piece vanished: unexplained
    return True


def _fill_policy(mesh, sigma):
    """G6: classify the mesh's boundary loops for the hole-fill policy, using the G5
    classifier. FILLABLE loops are tiny (<= FILL_MAX_EDGES) pinhole-class dropouts in a
    FLAT neighbourhood — a scan dropout small enough that an authorized flat fill is an
    interpolation, not an invention. Every other loop is NEVER auto-filled: dropout-class
    loops become ``missing_region`` records (they exclude from deviation coverage), and
    rims / hole openings / bore ends stay what they are — evidence about the part, not
    gaps to patch. Returns (fillable_perimeters, missing_regions, loops)."""
    from .quality import PLANAR_FRAC, _completeness, _pts_fac

    loops = _completeness(mesh, *_pts_fac(mesh), sigma or 0.0)["loops"]
    fillable = []
    for l in loops:
        tiny_flat = (l["kind"] == "dropout" and l["detail"] == "pinhole"
                     and l["n_edges"] <= FILL_MAX_EDGES
                     and l["planarity_rms_mm"] <= PLANAR_FRAC * 2.0 * l["radius_mm"]
                     + 2.0 * (sigma or 0.0))
        if tiny_flat:
            fillable.append(l["perimeter_mm"])
    missing = [l for l in loops if l["kind"] == "dropout"
               and not any(abs(l["perimeter_mm"] - f) < 1e-3 for f in fillable)]
    return fillable, missing, loops


def repair(mesh, authorized: "tuple | list" = ()) -> dict:
    """Repair a COPY of ``mesh``, one validated operation at a time.

    Safe operations always run; the gated, geometry-altering operations run only when named
    in ``authorized``. Every operation runs on its own TRIAL copy: baseline snapshot → apply
    → re-measure → explicit accept/reject rules (scale-aware: K1 sigma + mean edge) → verdict
    ACCEPTED / REJECTED (trial discarded) / NEEDS_REVIEW (kept, flagged) / FAILED. Facets a
    fill created are recorded as synthetic regions (and registered as a session segment —
    segments do NOT persist across save/reopen; the durable record is the returned history).

    G6 loop-aware fill policy: an authorized ``fillupHoles`` first classifies the boundary
    loops (G5 classifier). With NO fillable tiny flat pinhole it is SKIPPED (never executed);
    a fill that closes any loop the policy forbids (a genuine hole, opening or large
    dropout) is REJECTED wholesale. Dropout-class loops that survive repair are returned as
    ``missing_regions`` (evidence + ring indices into the returned mesh) for the
    measured-only deviation mask.

    Returns ``{"mesh", "actions", "before", "after", "resolved", "unresolved", "refused",
    "validation", "synthetic", "missing_regions", "quality"}`` — the original mesh is never
    modified."""
    before = diagnose(mesh)
    try:
        est = estimate_sigma(mesh)
        sigma, mean_edge = est["sigma_mm"], est["mean_edge_mm"]
    except Exception:  # noqa: BLE001 - validation degrades to edge-scale thresholds
        est, sigma, mean_edge = {}, None, None
    work = mesh.copy()
    actions: list[dict] = []
    validation: list[dict] = []
    synthetic: list[dict] = []
    refused: list[str] = []
    ledger = provenance.new_ledger(mesh_name=getattr(mesh, "Name", None))
    order = list(SAFE_OPERATIONS) + [op for op in GATED_OPERATIONS if op in set(authorized)]
    refused.extend(op for op in GATED_OPERATIONS if op not in set(authorized))
    for op in order:
        start = (int(work.CountFacets), int(work.CountPoints))
        fill_policy = None
        if op == "fillupHoles":
            # G6: decide from the loop classification BEFORE touching anything. The
            # kernel op is all-or-nothing, so with no fillable tiny-flat pinhole it could
            # only close loops the policy forbids — skip it entirely (recorded, honest).
            fillable, _missing_pre, loops_pre = _fill_policy(work, sigma)
            fill_policy = {"fillable": len(fillable),
                           "non_fillable": len(loops_pre) - len(fillable)}
            if not fillable:
                actions.append({"operation": op, "verdict": "SKIPPED",
                                "skipped": "G6 fill policy: no fillable loop (a tiny "
                                           "<= 8-edge flat pinhole); larger openings are "
                                           "recorded as missing_region and are never "
                                           "auto-filled",
                                "facets": start[0], "facets_after": start[0],
                                "spec": _predicted_spec(op, mean_edge)})
                validation.append({"operation": op, "verdict": "SKIPPED",
                                   "reasons": ["G6 fill policy: nothing on this mesh is a "
                                               "fillable tiny flat pinhole; the authorized "
                                               "operation was not executed"]})
                continue
        trial = work.copy()
        try:
            op_spec = _apply_op(trial, op, mean_edge)
        except Exception as exc:  # noqa: BLE001 - a failing operation is reported, not raised
            actions.append({"operation": op, "error": f"{type(exc).__name__}: {exc}"[:160],
                            "spec": _predicted_spec(op, mean_edge) if op in GATED_OPERATIONS else {}})
            validation.append({"operation": op, "verdict": "FAILED",
                               "reasons": [f"{type(exc).__name__}: {exc}"[:160]]})
            continue
        snap_b = _snapshot(work)
        bnd = boundary(work)
        if op == "fillupHoles":
            snap_b["_fillable_perimeters"] = fillable
            after_loops = {l["perimeter_mm"] for l in boundary(trial)["loops"]}
            closed = [l["perimeter_mm"] for l in bnd["loops"]
                      if not any(abs(l["perimeter_mm"] - p) < 1e-3 for p in after_loops)]
            snap_b["_fill_area_cap"] = round(
                sum(p * p for p in closed) / (4.0 * math.pi) * 1.05, 3)
            snap_b["_max_closed_perimeter"] = max(closed, default=0.0)
            snap_b["_policy_violations"] = [
                p for p in closed
                if not any(abs(p - f) < 1e-3 for f in fillable)]
        snap_a = _snapshot(trial)
        pts_b, _ = _arrays(work)
        pts_a, _ = _arrays(trial)
        if op == "fillupHoles":
            # fills may ADD vertices; the originals must still be there (before→after)
            disp = _displacement(pts_b, pts_a)
            moved = bool(disp and disp["max_mm"] > max(
                DISPLACEMENT_SIGMA_MULT * (sigma or 0.0),
                DISPLACEMENT_EDGE_MULT * (mean_edge or 0.0), 1e-6))
        else:
            # cleanup/topology ops may REMOVE vertices (dedup, degenerate) but every
            # SURVIVING vertex must have existed before (after→before), else geometry moved
            disp = _displacement(pts_a, pts_b)
            moved = bool(disp and disp["max_mm"] > max(
                DISPLACEMENT_SIGMA_MULT * (sigma or 0.0),
                DISPLACEMENT_EDGE_MULT * (mean_edge or 0.0), 1e-6))
        verdict, reasons = _validate_transition(op, snap_b, snap_a, sigma, mean_edge)
        if verdict != "REJECTED" and snap_b.get("_policy_violations"):
            # G6: the fill closed a loop that is NOT a fillable tiny flat pinhole — a
            # genuine hole or opening. Never silently kept: the whole trial is discarded.
            verdict = "REJECTED"
            reasons = [*reasons,
                       f"closed {len(snap_b['_policy_violations'])} loop(s) the G6 fill "
                       "policy forbids (only tiny flat pinholes may be filled) — "
                       f"perimeters {[round(p, 2) for p in snap_b['_policy_violations'][:3]]}"
                       " mm"]
        if verdict == "NEEDS_REVIEW" and any("disappeared" in r for r in reasons) and \
                (snap_b.get("components") or 0) > (snap_a.get("components") or 0) and \
                _disappearances_explained(work, trial):
            verdict = "ACCEPTED"
            reasons = [r + " — explained: only tiny degenerate-debris components vanished"
                       if "disappeared" in r else r for r in reasons]
        if moved:
            # vertex movement beyond max(2σ, 0.1·edge) on an operation that must not move
            # geometry is decisive damage evidence — reject regardless of the other rules
            reasons = [*reasons, f"vertices moved up to {disp['max_mm']} mm "
                                 f"(> max(2σ, 0.1·edge))"]
            verdict = "REJECTED"
        record = {"operation": op, "verdict": verdict, "reasons": reasons,
                  "before": {k: v for k, v in snap_b.items() if not k.startswith("_")},
                  "after": snap_a, "displacement": disp}
        if verdict == "REJECTED":
            record["note"] = "trial discarded — the working mesh keeps the last good state"
        else:
            provenance.record_operation(ledger, op, verdict,
                                        provenance.diff_mesh(work, trial))
            if op == "fillupHoles":
                new_facets = _synthetic_facets(work, trial)
                if new_facets:
                    try:
                        trial.addSegment(new_facets)   # session-only; see module docstring
                    except Exception:  # noqa: BLE001
                        record["segment"] = "not registered (kernel refused addSegment)"
                    synthetic.append({"operation": op, "facets": new_facets,
                                      "count": len(new_facets),
                                      "note": "SYNTHETIC fill geometry — NOT measured scan "
                                              "evidence; exclude from deviation coverage"})
            work = trial
        validation.append(record)
        actions.append({"operation": op,
                        "facets": start[0], "facets_after": int(work.CountFacets),
                        "points": start[1], "points_after": int(work.CountPoints),
                        "verdict": verdict, "spec": op_spec,
                        **({"policy": fill_policy} if fill_policy else {})})
    after = diagnose(work)
    # G6: the missing-region record for the OUTPUT mesh — every dropout-class loop that
    # survives repair (the policy never fills them); ring indices address the returned
    # work mesh, so downstream stages can mask these regions without reclassifying.
    _, missing_regions, _ = _fill_policy(work, sigma)
    resolved = {key: bool(before.get(key) is True and after.get(key) is not True)
                for key in ("non_manifold", "self_intersections", "inconsistent_normals")}
    resolved["opened_to_closed"] = bool(not before["closed"] and after["closed"])
    unresolved = {key: after.get(key) for key in ("non_manifold", "self_intersections",
                                                  "inconsistent_normals")
                  if after.get(key)}
    if not after["closed"]:
        unresolved["not_closed"] = True
    ledger["mesh_digest"] = provenance.mesh_digest(work)   # complete-verification tier (H1)
    return {"mesh": work, "actions": actions, "before": before, "after": after,
            "resolved": resolved, "unresolved": unresolved, "refused": refused,
            "refused_specs": {op: _predicted_spec(op, mean_edge) for op in refused},
            "validation": validation, "synthetic": synthetic,
            "missing_regions": missing_regions,
            "provenance": ledger,
            "provenance_verification": provenance.verify_against_mesh(ledger, work),
            "quality": {"sigma_mm": sigma, "mean_edge_mm": mean_edge,
                        **({"note": "sigma estimation failed — edge-scale thresholds only"}
                           if sigma is None else {})}}
