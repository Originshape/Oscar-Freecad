"""Bidirectional sampled deviation metrics between the source mesh and the reconstructed
model (master spec §18 / M6). Point-sample based: scan vertices against model-surface samples
and vice versa — an honest nearest-neighbour approximation of continuous surface distance
(the sample counts are reported with every number). scipy's cKDTree is used when importable
with a chunked numpy fallback so the engine stays one-plugin pure.

Pure-numpy/scipy; no document access, any thread.
"""

import math
import time

KIND = "recon_deviation"


def _subdivide(P, F, spacing, cap=8000):
    """Barycentric 4-way subdivision of triangles with an edge above ``spacing``
    (midpoints kept, centroids returned with the vertices), bounded by ``cap``.
    Vertices alone undersample FLAT regions of tessellated CAD — a coarse box
    has a handful of vertices, and point-sample deviation then reads whole
    edge lengths of phantom distance on a geometrically exact model."""
    import numpy as np

    P = np.asarray(P, dtype=float)
    F = np.asarray(F, dtype=int).reshape(-1, 3)
    if len(F) == 0:
        return P
    while True:
        tri = P[F]
        e01 = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
        e12 = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
        e20 = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
        big = (np.maximum(np.maximum(e01, e12), e20) > spacing)
        n_big = int(big.sum())
        if n_big == 0 or len(F) + 3 * n_big > cap:
            break
        idx = np.nonzero(big)[0]
        mid01 = (tri[idx, 0] + tri[idx, 1]) / 2.0
        mid12 = (tri[idx, 1] + tri[idx, 2]) / 2.0
        mid20 = (tri[idx, 2] + tri[idx, 0]) / 2.0
        base = len(P)
        cols = np.arange(n_big)
        # mid01 rows sit at base+cols, mid12 at base+n_big+cols, mid20 at
        # base+2*n_big+cols — index arithmetic that mixes the blocks builds
        # triangles from OTHER triangles' midpoints (points off the surface)
        P = np.vstack([P, mid01, mid12, mid20])
        a, b, c = F[idx].T
        m01 = base + cols
        m12 = base + n_big + cols
        m20 = base + 2 * n_big + cols
        new = np.vstack([
            np.stack([a, m01, m20], axis=1),
            np.stack([m01, b, m12], axis=1),
            np.stack([m12, c, m20], axis=1),
            np.stack([m20, m01, m12], axis=1),
        ])
        F = np.vstack([F[~big], new])
    return np.vstack([P, P[F].mean(axis=1)])


def _spacing_for(P) -> float:
    """A sampling spacing bounded to [0.3, 2.0] mm from the cloud's diagonal —
    fine enough that nearest-neighbour error stays below the usual warn
    thresholds, coarse enough to stay bounded on big parts."""
    import numpy as np

    P = np.asarray(P, dtype=float)
    if len(P) == 0:
        return 1.0
    dg = float(np.linalg.norm(np.ptp(P, axis=0)))
    return min(max(dg / 200.0, 0.3), 2.0)


def _mesh_points(mesh, dense=True, exclude_vertices=None):
    """Scan cloud. ``dense=False`` returns the RAW mesh vertices only (the
    measured data points — exact-residual refinement wants those, not
    interpolated facet interiors). ``exclude_vertices`` (the G6 mask's
    ``scan_exclude_indices``) drops those vertices, and facets touching them
    are poisoned for the SUBDIVISION (a synthetic fill's region must not
    re-enter through interpolated interiors) while untouched raw vertices
    always stay — the vertex-index mask contract is preserved while the
    cloud grows."""
    import numpy as np

    all_pts = [(p.x, p.y, p.z) for p in mesh.Topology[0]]
    fac = np.array(mesh.Topology[1], dtype=int).reshape(-1, 3)
    drop = {int(i) for i in (exclude_vertices or ())}
    if drop:
        pts = [p for k, p in enumerate(all_pts) if k not in drop]
        fac = fac[~np.isin(fac, sorted(drop)).any(axis=1)]
    else:
        pts = list(all_pts)
    if dense:
        try:
            dense_pts = _subdivide(pts, fac, _spacing_for(pts))
            pts = [tuple(p) for p in dense_pts]
        except Exception:  # noqa: BLE001 - subdivision is an accuracy bonus only
            pass
    return pts


def _shape_points(shape, max_points=30000):
    """Model surface samples: tessellation vertices, subdivision of the
    tessellation triangles, and discretized EDGES.

    P1.2: OCC tessellates FLAT faces as two giant triangles whatever the
    deflection, so a prismatic model's cloud can miss its own corners (a scan
    vertex at a concave corner then reads a full edge-length away). The scan
    cloud is subdivided to match, keeping both directions of the sampled
    nearest-neighbour estimate at the same density."""
    import numpy as np

    verts, facets = shape.tessellate(0.3)
    pts = [(p.x, p.y, p.z) for p in verts]
    try:
        dense = _subdivide(pts, np.array(facets, dtype=int).reshape(-1, 3),
                           _spacing_for(pts))
        pts = [tuple(p) for p in dense]
    except Exception:  # noqa: BLE001 - subdivision is an accuracy bonus only
        pass
    try:
        for e in shape.Edges:
            n = max(2, min(256, int(e.Length / 0.5) + 1))
            try:
                pts.extend((p.x, p.y, p.z) for p in e.discretize(Number=n))
            except Exception:  # noqa: BLE001 - one edge's discretization is optional
                pass
    except Exception:  # noqa: BLE001 - edges are a bonus, never a blocker
        pass
    if len(pts) > max_points:
        step = max(1, len(pts) // max_points)
        pts = pts[::step]
    return pts


def shape_points_and_normals(shape, max_points=30000, min_samples=1024):
    """Sampled model surface as ``(points, unit_normals)`` for point-to-plane ICP
    (registration stage), DENSIFIED beyond the raw triangle centroids: a prismatic solid
    tessellates each planar face into ~2 triangles, so the centroids alone are a dozen
    points that cannot anchor a registration (issue #13 — the aligned diagnostic must
    also exist for simple geometry). Area-weighted barycentric samples fill the faces up
    to ``min_samples`` (still capped by ``max_points``); the fixed seed keeps sampling
    deterministic between runs."""
    import numpy as np

    verts, facets = shape.tessellate(0.3)
    pts = [(p.x, p.y, p.z) for p in verts]
    tri = [(pts[a], pts[b], pts[c]) for a, b, c in facets]
    if len(tri) > max_points:
        step = max(1, len(tri) // max_points)
        tri = tri[::step]
    P = np.array([[[t[0][0], t[0][1], t[0][2]],
                   [t[1][0], t[1][1], t[1][2]],
                   [t[2][0], t[2][1], t[2][2]]] for t in tri], dtype=float)   # (n, 3, 3)
    v0, v1, v2 = P[:, 0], P[:, 1], P[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)
    n = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)
    centroids = (v0 + v1 + v2) / 3.0
    target = min(max(len(tri), min(min_samples, max_points)), max_points)
    extra = target - len(tri)
    if extra <= 0 or not len(tri):
        return centroids, n
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    weights = areas / max(float(areas.sum()), 1e-12)
    rng = np.random.default_rng(20261119)
    # the tessellation VERTICES ride along (an owning-triangle normal each): a coarse or
    # CAD-tessellated scan's own vertices are exactly these corners — corner-to-corner
    # correspondences are what makes a prismatic part's registration converge tightly
    verts_arr, vert_n = P.reshape(-1, 3), np.repeat(n, 3, axis=0)
    if len(centroids) + len(verts_arr) > max_points:
        vstep = max(1, (len(centroids) + len(verts_arr)) // (max_points - len(centroids) or 1))
        verts_arr, vert_n = verts_arr[::vstep], vert_n[::vstep]
    extra = min(extra, max(max_points - len(centroids) - len(verts_arr), 0))
    pick = np.minimum(np.searchsorted(np.cumsum(weights), rng.random(extra), side="right"),
                      len(tri) - 1)
    r1, r2 = rng.random(extra), rng.random(extra)
    s = np.sqrt(r1)                       # uniform barycentric mapping on each triangle
    b0, b1, b2 = (1.0 - s), s * (1.0 - r2), s * r2
    samples = (v0[pick] * b0[:, None] + v1[pick] * b1[:, None] + v2[pick] * b2[:, None])
    return (np.vstack([centroids, verts_arr, samples]),
            np.vstack([n, vert_n, n[pick]]))


def _region_keep_mask(points, regions):
    """Boolean keep-mask over ``points``: False where a point lies inside a missing
    region's planar ring footprint — within ``tol_mm`` of the ring's plane AND inside the
    ring polygon (even-odd ray casting in the ring plane). Regions come from the G5 loop
    classification: dropout-class loops mark surface the scan never measured, so model
    samples there are excluded from the metrics instead of warning as "extra geometry"."""
    import numpy as np

    pts = np.asarray(points, dtype=float)
    keep = np.ones(len(pts), dtype=bool)
    for region in regions or []:
        ring = np.asarray(region["points"], dtype=float)
        if len(ring) < 3:
            continue
        n = np.asarray(region["normal"], dtype=float)
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(n @ ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, ref)
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        v = np.cross(n, u)
        c = ring.mean(axis=0)
        d = pts - c
        near = np.abs(d @ n) <= float(region.get("tol_mm", 0.0))
        x, y = d @ u, d @ v
        rx, ry = (ring - c) @ u, (ring - c) @ v
        crossings = np.zeros(len(pts), dtype=int)
        for i in range(len(rx)):
            j = (i + 1) % len(rx)
            if ry[i] == ry[j]:
                continue
            between = (y >= min(ry[i], ry[j])) & (y < max(ry[i], ry[j]))   # half-open
            xint = rx[i] + (y - ry[i]) * (rx[j] - rx[i]) / (ry[j] - ry[i])
            crossings += (between & (x < xint)).astype(int)
        keep &= ~(near & (crossings % 2 == 1))
    return keep


def _apply_mask(scan_pts, model_pts, mask, excluded_scan=0):
    """G6 measured-only mask: the SCAN exclusion happens at cloud build
    (``_mesh_points(exclude_vertices=...)`` — a poisoned facet leaves the
    dense cloud too); this pass drops model samples inside recorded missing
    regions and reports the summary. ``None`` mask → unchanged."""
    if not mask:
        return scan_pts, model_pts, None
    import numpy as np

    model = model_pts
    excluded_model = 0
    regions = mask.get("model_regions") or []
    if regions:
        keep = _region_keep_mask(model_pts, regions)
        excluded_model = int((~keep).sum())
        model = [tuple(p) for p in np.asarray(model_pts, dtype=float)[keep]]
    return scan_pts, model, {"scan_excluded": int(excluded_scan),
                             "model_excluded": excluded_model, "regions": len(regions),
                             "note": str(mask.get("note") or "")}


def _measured_only_row(summary, a, b):
    return _row("measured_only", "excluded_samples", summary["model_excluded"], 0, "PASS",
                f"measured-only mask: {summary['scan_excluded']} scan vertex/vertices "
                "excluded (synthetic fill geometry is NOT measurement), "
                f"{summary['model_excluded']} model sample(s) excluded across "
                f"{summary['regions']} missing region(s) — all metrics above cover "
                "MEASURED regions only"
                + (f"; {summary['note']}" if summary["note"] else ""), a, b)


# ------------------------------------------------------------- V-1: exact residuals
MAX_EXACT_MODEL_SCAN_FACETS = 1_500  # laptop profile: avoid triangle-BRep sewing on larger scans


def _refine_exact(points, shape, clock=None, deadline=None, batch=8):
    """Exact OCC distances (``Shape.distToShape``) for ``points`` against ``shape``
    **[RT]** — the sampled rows' lower bounds become exact values here. Stops early at
    the deadline (returns fewer), never raises."""
    import FreeCAD as App
    import Part

    out = []
    for i, p in enumerate(points):
        if deadline is not None and clock is not None and i and i % batch == 0 \
                and clock() >= deadline:
            break
        try:
            dist = float(shape.distToShape(Part.Vertex(App.Vector(*p)))[0])
        except Exception:  # noqa: BLE001 - a kernel refusal degrades to skip, not crash
            continue
        out.append((i, dist))
    return out


def exact_residuals(mesh, shape, *, a="scan", b="model", warn_mm=0.3, worst_n=32,
                    sample_cap=250, mask=None, budget_s=6.0, clock=None):
    """V-1: exact worst-point deviation + signed residual clusters.

    The sampled rows are nearest-neighbour approximations (their Hausdorff is a LOWER
    bound). This pass computes EXACT distances (``Shape.distToShape`` **[RT]**: a
    0.500 mm gap measures 0.500) for a capped sample of each side — exact for every
    value, so sparse planar tessellations cannot fake a gap — classifies each
    beyond-warn offender's sign, and clusters them:

    - scan point OUTSIDE the model solid (``Shape.isInside``) → the scan surface extends
      beyond the model → ``missing`` (the model lacks material the scan shows);
    - model sample on the OUTER side of the scan surface ((scan−model)·model_normal
      decisive; meshes have no kernel point-inside test — the model's own normal is the
      honest substitute) → ``excess`` (the model has material the scan does not show).

    Cluster areas are per-point-share ESTIMATES — uniform for uniformly sampled scans,
    crude when a side's planar faces tessellate sparsely (each model-side record carries
    that caveat). Additive to the sampled rows (both stay visible); masked points
    (synthetic fills, missing regions) never take part. Budget-guarded. Returns
    ``{"rows", "clusters", "refined"}``."""
    import time as _time

    import numpy as np

    clock = clock or _time.monotonic
    deadline = clock() + max(1.0, float(budget_s))
    excl = [int(i) for i in (mask.get("scan_exclude_indices") or ())] if mask else []
    scan_pts = _mesh_points(mesh, dense=False, exclude_vertices=excl)
    model_pts = _shape_points(shape)
    scan_pts, model_pts, _masked = _apply_mask(scan_pts, model_pts, mask,
                                               excluded_scan=len(excl))
    rows, clusters = [], []
    refined = {"scan": 0, "model": 0}
    if not scan_pts or not model_pts:
        rows.append(_row("exact_scan_to_model.max", "exact_mm", 0.0, warn_mm, "SKIP",
                         "no sample points on one side — exact refinement unavailable",
                         a, b))
        return {"rows": rows, "clusters": clusters, "refined": refined}

    def _capped(points):
        if len(points) <= sample_cap:
            return list(points)
        rng = np.random.default_rng(20261114)     # unbiased sample (a stride would
        return [points[i] for i in sorted(        # systematically bias clustered regions)
            rng.choice(len(points), sample_cap, replace=False).tolist())]

    # scan side: exact distances for the capped sample, signs from the model solid
    scan_sample = _capped(scan_pts)
    got = _refine_exact(scan_sample, shape, clock, deadline)
    refined["scan"] = len(got)
    if got:
        worst = max(dist for _, dist in got)
        rows.append(_row("exact_scan_to_model.max", "exact_mm", worst, warn_mm,
                         "PASS" if worst <= warn_mm else "WARN",
                         f"exact OCC distance (distToShape) of the worst of "
                         f"{len(got)} sampled scan point(s) — replaces the sampled "
                         "lower bound there", a, b))
        offenders, kinds, off_dists = [], [], []
        for pos, dist in got:
            if dist <= warn_mm:
                continue
            p = scan_sample[pos]
            try:
                kind = "excess" if bool(shape.isInside(_vec(p), 1e-7, True)) \
                    else "missing"
            except Exception:  # noqa: BLE001 - sign is optional honesty, distance is not
                kind = None
            offenders.append(np.asarray(p, dtype=float))
            kinds.append(kind)
            off_dists.append(float(dist))
        scan_per_point = (float(mesh.Area) or 1.0) / max(len(scan_sample), 1)
        clusters += _finish_clusters(offenders, kinds, scan_per_point, dists=off_dists,
                                     direction="s2m")
    else:
        rows.append(_row("exact_scan_to_model.max", "exact_mm", 0.0, warn_mm, "SKIP",
                         "budget exhausted before any scan refinement", a, b))

    # model side: exact distances for capped model samples against the scan converted to
    # a faceted solid (the repo's own converter — capped and closed-mesh-guarded)
    scan_shape, why = (None, "")
    facets = int(getattr(mesh, "CountFacets", 0))
    if facets > MAX_EXACT_MODEL_SCAN_FACETS:
        why = (f"laptop resource limit: model-to-scan exact refinement is limited to "
               f"{MAX_EXACT_MODEL_SCAN_FACETS:,} scan facets; the sampled reconstruction "
               "comparison remains authoritative")
    else:
        try:
            from ..cad.imported import solid_from_mesh

            scan_shape, why = solid_from_mesh(mesh)
        except Exception as exc:  # noqa: BLE001 - OCC raises its own types
            why = f"{type(exc).__name__}: {exc}"[:120]
    centroids, normals = shape_points_and_normals(shape)
    if scan_shape is not None and len(centroids):
        step_m = max(1, (len(centroids) + sample_cap - 1) // sample_cap)
        idx_m = list(range(0, len(centroids), step_m))
        model_sample = [tuple(centroids[i]) for i in idx_m]
        got_m = _refine_exact(model_sample, scan_shape, clock, deadline)
        refined["model"] = len(got_m)
        if got_m:
            worst = max(dist for _, dist in got_m)
            rows.append(_row("exact_model_to_scan.max", "exact_mm", worst, warn_mm,
                             "PASS" if worst <= warn_mm else "WARN",
                             f"exact OCC distance of the worst of {len(got_m)} sampled "
                             "model point(s) against the scan's faceted solid "
                             "(distToShape)", a, b))
            scan_arr = np.asarray(scan_pts, dtype=float)
            offenders_m, kinds_m, off_dists_m = [], [], []
            for pos, dist in got_m:
                if dist <= warn_mm:
                    continue
                q = np.asarray(model_sample[pos], dtype=float)
                n = np.asarray(normals[idx_m[pos]], dtype=float)
                side = 0.0
                if len(scan_arr):
                    near = scan_arr[int(np.argmin(((scan_arr - q) ** 2).sum(axis=1)))]
                    side = float((near - q) @ n)
                offenders_m.append(q)
                kinds_m.append(None if abs(side) <= 0.2 * dist
                               else "missing" if side > 0 else "excess")
                off_dists_m.append(float(dist))
            model_per_point = _shape_area(shape) / max(len(model_sample), 1)
            clusters += _finish_clusters(offenders_m, kinds_m, model_per_point,
                                         caveat="model-side sampling is "
                                                "tessellation-based and non-uniform on "
                                                "planar faces — the area estimate is "
                                                "crude there",
                                         dists=off_dists_m, direction="m2s")
        else:
            rows.append(_row("exact_model_to_scan.max", "exact_mm", 0.0, warn_mm,
                             "SKIP", "budget exhausted before any model refinement",
                             a, b))
    else:
        rows.append(_row("exact_model_to_scan.max", "exact_mm", 0.0, warn_mm, "SKIP",
                         f"model-side exact refinement unavailable (mesh→shape: {why})",
                         a, b))
    n_desc = "; ".join(f"{c['kind']} {c['points']} pts ~{c['area_mm2']} mm² at "
                       f"{c['centroid_mm']}" for c in clusters[:4])
    rows.append(_row("residual_clusters", "clusters", len(clusters), 0, "PASS",
                     ("beyond-warn offenders clustered by sign: " + n_desc)
                     if clusters else "no beyond-warn offenders among the refined "
                     "worst points — no missing/excess clusters", a, b))
    return {"rows": rows, "clusters": clusters, "refined": refined}


def _vec(p):
    import FreeCAD as App

    return App.Vector(p[0], p[1], p[2])


def _shape_area(shape) -> float:
    try:
        return float(shape.Area)
    except Exception:  # noqa: BLE001
        return 0.0


def _finish_clusters(offenders, kinds, per_point_area, caveat="", dists=None,
                     direction=None, member_cap=200):
    """Cluster beyond-warn offenders: merge radius from the offenders' own LOCAL spacing
    (3× their median nearest-neighbour distance — self-consistent even when planar faces
    are sparsely tessellated), area = member count × the side's per-point sampling share,
    kind by majority sign. The caveat lands on every record so a crude estimate is never
    read as exact.

    ``dists`` (optional, one exact distance per offender, same order) adds
    ``mean_dist_mm`` / ``max_dist_mm`` to every record — the magnitude evidence a
    dimension correction needs (area is a per-point-share estimate, never a distance).
    Backwards compatible: new keys only, absent when ``dists`` is not given."""

    import numpy as np

    from .quality import _clusters_labels

    if not offenders:
        return []
    pts = np.asarray(offenders, dtype=float)
    if len(pts) >= 2:
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        spacing = float(np.median(d.min(axis=1)))
    else:
        spacing = 1.0
    labels = _clusters_labels(pts, max(3.0 * spacing, 1e-6))
    # thin-sample regime: a handful of offenders (long strips under-sampled by the refinement
    # cap) still cluster at 2 members — 2 spatially-agreeing points are evidence, 1 is noise
    min_members = 3 if len(pts) >= 30 else 2
    out = []
    for c in np.unique(labels):
        member = [i for i, l in enumerate(labels.tolist()) if l == c]
        if len(member) < min_members:
            continue
        votes = {"missing": 0, "excess": 0}
        for i in member:
            if kinds[i] in votes:
                votes[kinds[i]] += 1
        record = {"kind": max(votes, key=votes.get) if (votes["missing"] or
                                                        votes["excess"]) else "unsigned",
                  "points": len(member),
                  "area_mm2": round(len(member) * per_point_area, 2),
                  "centroid_mm": [round(float(x), 3) for x in pts[member].mean(axis=0)]}
        if direction is not None:
            # localization feed (L-1): the direction the cluster was found IN and a capped
            # set of member coordinates — enough for bounding boxes and feature association
            # without retaining every offender
            sel = member if len(member) <= member_cap else member[::len(member) // member_cap + 1]
            record["direction"] = direction
            record["member_points_mm"] = [[round(float(x), 3) for x in pts[i]] for i in sel]
            if dists is not None and len(dists) == len(offenders):
                record["member_dists_mm"] = [round(float(dists[i]), 4) for i in sel]
        if dists is not None and len(dists) == len(offenders):
            member_d = [float(dists[i]) for i in member]
            record["mean_dist_mm"] = round(sum(member_d) / len(member_d), 4)
            record["max_dist_mm"] = round(max(member_d), 4)
        if caveat:
            record["note"] = caveat
        out.append(record)
    out.sort(key=lambda c: -c["area_mm2"])
    return out


def _distances(from_points, to_points):
    """Nearest-neighbour distances from_points → to_points, cKDTree when available."""
    try:
        from scipy.spatial import cKDTree

        import numpy as np

        tree = cKDTree(np.asarray(to_points, dtype=float))
        d, _ = tree.query(np.asarray(from_points, dtype=float), k=1)
        return [float(x) for x in d], "scipy.cKDTree"
    except Exception:  # noqa: BLE001 - chunked numpy fallback keeps the plugin dependency-free
        import numpy as np

        src = np.asarray(from_points, dtype=float)
        dst = np.asarray(to_points, dtype=float)
        out = []
        chunk = 512
        for start in range(0, len(src), chunk):
            block = src[start:start + chunk]
            d = numpy_linalg_norm(block, dst)
            out.extend(float(x) for x in d)
        return out, "numpy-chunked"


def numpy_linalg_norm(block, dst):
    import numpy as np

    return np.sqrt(((block[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)).min(axis=1)


def _stats(values):
    if not values:
        return {"rms": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    vs = sorted(values)
    n = len(vs)

    def pct(p):
        return vs[min(n - 1, int(math.ceil(p / 100.0 * n)) - 1)]

    return {"rms": math.sqrt(sum(v * v for v in vs) / n), "p95": pct(95), "p99": pct(99),
            "max": vs[-1], "mean": sum(vs) / n}


def _row(rid, metric, value, limit, status, detail, a, b):
    try:
        value = round(float(value), 4)
        if not math.isfinite(value):
            value = 0.0
    except (TypeError, ValueError):
        value = 0.0
    return {"id": rid, "kind": KIND, "a": a, "b": b, "metric": metric, "value": value,
            "limit": limit, "status": status, "detail": detail[:220]}


def _build_rows(scan_pts, model_pts, *, a, b, warn_mm, engine_note="", distances=None):
    """The shared metric core: identical rows for raw AND aligned comparisons (one sampling
    policy, one distance implementation — the registration mandate's parity requirement).
    ``distances`` may carry precomputed ``(d_scan, scan_engine, d_model, model_engine)``
    — the exact path (``_exact_raw_distances``) uses it so the PRIMARY rows stop
    inheriting cloud-spacing error on flat faces."""
    rows = [_row("samples", "sample_counts", 0.0, 0, "PASS",
                 f"scan points {len(scan_pts)}, model samples {len(model_pts)} "
                 f"(sampled nearest-neighbour distances, not exact surface "
                 f"distance){engine_note}", a, b)]
    if len(scan_pts) == 0 or len(model_pts) == 0:
        return rows + [_row("deviation", "unavailable", 0.0, warn_mm, "SKIP",
                            "one side has no sample points", a, b)]
    if distances is not None:
        d_scan, scan_engine, d_model, model_engine = distances
        if d_scan is None:
            d_scan, scan_engine = _distances(scan_pts, model_pts)
        if d_model is None:
            d_model, model_engine = _distances(model_pts, scan_pts)
    else:
        d_scan, scan_engine = _distances(scan_pts, model_pts)
        d_model, model_engine = _distances(model_pts, scan_pts)
    rows[0]["detail"] += f"; engine {scan_engine} / {model_engine}"
    s1, s2 = _stats(d_scan), _stats(d_model)
    hausdorff = max(s1["max"], s2["max"])
    for name, s, d in (("scan_to_model", s1, d_scan), ("model_to_scan", s2, d_model)):
        rows.append(_row(f"{name}.max", "max_mm", s["max"], warn_mm,
                         "PASS" if s["max"] <= warn_mm else "WARN",
                         f"{name}: maximum sampled nearest-neighbour distance", a, b))
        rows.append(_row(f"{name}.rms", "rms_mm", s["rms"], warn_mm,
                         "PASS" if s["rms"] <= warn_mm else "WARN",
                         f"{name}: RMS of sampled nearest-neighbour distances", a, b))
        rows.append(_row(f"{name}.p95", "p95_mm", s["p95"], warn_mm,
                         "PASS" if s["p95"] <= warn_mm else "WARN",
                         f"{name}: 95th percentile of sampled distances", a, b))
        rows.append(_row(f"{name}.p99", "p99_mm", s["p99"], warn_mm,
                         "PASS" if s["p99"] <= warn_mm else "WARN",
                         f"{name}: 99th percentile of sampled distances", a, b))
        inlier_share = sum(1 for v in d if v <= warn_mm) / float(len(d))
        rows.append(_row(f"{name}.coverage", "inlier_share", round(inlier_share, 4), 1.0,
                         "PASS" if inlier_share >= 0.95 else "WARN",
                         f"{name}: share of sampled points within {warn_mm} mm (sampled "
                         "coverage; missing/extra geometry shows up here)", a, b))
    rows.append(_row("hausdorff", "hausdorff_mm", hausdorff, warn_mm,
                     "PASS" if hausdorff <= warn_mm else "WARN",
                     "two-sided estimate from sampled maxima (lower bound of the true "
                     "Hausdorff distance)", a, b))
    return rows


def _mesh_tri(mesh):
    """The mesh's triangle corner arrays ``(A, B, C)`` for `_points_to_mesh_exact` —
    prepared ONCE by callers that probe the SAME mesh repeatedly (hole-rim probes:
    ~45 calls x a 34k-facet re-preparation each was a measured case-07 hotspot).
    Byte-identical to the in-function preparation."""
    import numpy as np

    V = np.array([[p.x, p.y, p.z] for p in mesh.Topology[0]], dtype=float)
    F = np.array(mesh.Topology[1], dtype=int).reshape(-1, 3)
    return V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]


def _points_to_mesh_exact(points, mesh, chunk=48, tri=None):
    """EXACT distance from every point to the mesh's triangle surface (pure
    numpy point-to-triangle, chunked). Replaces triangle-BRep sewing for the
    model→scan direction: sewing a mesh whose walls self-intersect (crossing
    holes) can hang the kernel outright, and the mesh IS the ground truth —
    its facets are exactly the measured surface. ``tri`` is a preprepared
    `_mesh_tri(mesh)` bundle (identical numerics, no re-preparation)."""
    import numpy as np

    P = np.asarray(points, dtype=float)
    if tri is not None:
        A, B, C = tri
        if not len(P) or not len(A):
            return None
    else:
        V = np.array([[p.x, p.y, p.z] for p in mesh.Topology[0]], dtype=float)
        F = np.array(mesh.Topology[1], dtype=int).reshape(-1, 3)
        if not len(P) or not len(F):
            return None
        A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    out = np.full(len(P), np.inf)

    def _seg(p, x, d, l2):
        tt = np.clip(((p - x) * d).sum(-1) / np.maximum(l2, 1e-30), 0.0, 1.0)
        q = x + tt[..., None] * d
        return np.sqrt(((p - q) ** 2).sum(-1))

    for s in range(0, len(P), chunk):
        p = P[s:s + chunk][:, None, :]                     # (c, 1, 3)
        a, b, c = A[None, :, :], B[None, :, :], C[None, :, :]
        ab, ac = b - a, c - a
        ap = p - a
        n = np.cross(ab, ac)
        nn = (n * n).sum(-1)
        safe = np.maximum(nn, 1e-30)
        dplane = np.abs((n * ap).sum(-1) / np.sqrt(safe))
        # barycentric coordinates of the plane projection
        proj = p - ((n * ap).sum(-1) / safe)[..., None] * n
        d00 = (ab * ab).sum(-1)
        d01 = (ab * ac).sum(-1)
        d11 = (ac * ac).sum(-1)
        d20 = (ab * (proj - a)).sum(-1)
        d22 = (ac * (proj - a)).sum(-1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
        vb = (d11 * d20 - d01 * d22) / den
        wc = (d00 * d22 - d01 * d20) / den
        inside = (vb >= 0.0) & (wc >= 0.0) & (vb + wc <= 1.0)
        de = np.minimum(
            np.minimum(_seg(p, a, ab, d00),
                       _seg(p, b, c - b, ((c - b) ** 2).sum(-1))),
            _seg(p, c, a - c, ((a - c) ** 2).sum(-1)))
        dtri = np.where(inside, dplane, de)
        out[s:s + chunk] = dtri.min(axis=1)
    return out


def _drop_phantom_max(model_pts, dists, shape):
    """Exclude PHANTOM model samples from the worst-offender end: OCC's
    BRepMesh can shortcut a planar face's inner wire (a triangle whose three
    vertices sit on a hole rim and whose interior spans the open hole). Such a
    sample lies in mid-air — no material within a small probe on ANY side —
    while every real boundary point has material on one side. Only the worst
    offenders are probed (they are what drives the max); returns
    (dists, dropped_count)."""
    import FreeCAD as App

    if dists is None or not len(dists):
        return dists, 0
    worst = sorted(range(len(dists)), key=lambda k: -dists[k])[:32]
    drop = set()
    probes = ((0.05, 0, 0), (-0.05, 0, 0), (0, 0.05, 0),
              (0, -0.05, 0), (0, 0, 0.05), (0, 0, -0.05))
    for k in worst:
        p = model_pts[k]
        try:
            if not any(shape.isInside(
                    App.Vector(float(p[0] + dx), float(p[1] + dy),
                               float(p[2] + dz)), 1e-6, True)
                    for dx, dy, dz in probes):
                drop.add(k)
        except Exception:  # noqa: BLE001 - an undecidable point is kept
            continue
    if not drop:
        return dists, 0
    return [d for k, d in enumerate(dists) if k not in drop], len(drop)


def _importance_sample(scan_pts, model_pts, cap_fn, seed_top=64):
    """``cap_fn(points)`` (the historic uniform cap) seeded with the ``seed_top`` worst
    sampled-NN offenders first, then uniform-filled to the same size. Bounded: never
    exceeds the cap. Falls back to pure uniform when the NN pass is unavailable."""
    base = cap_fn(scan_pts)
    if len(scan_pts) <= len(base) or len(scan_pts) < 3 * seed_top:
        return base
    try:
        import numpy as np

        d, _ = _distances(scan_pts, model_pts)
        order = np.argsort(d)[::-1]
        chosen = list(dict.fromkeys(int(i) for i in order[:seed_top]))
        rng = np.random.default_rng(0)
        rest = [i for i in rng.permutation(len(scan_pts)) if i not in set(chosen)]
        need = max(0, len(base) - len(chosen))
        idx = chosen + [int(i) for i in rest[:need]]
        return [scan_pts[i] for i in sorted(set(idx))][:len(base)]
    except Exception:  # noqa: BLE001 - sampling strategy, never worth failing for
        return base


def _exact_raw_distances(mesh, shape, scan_pts, model_pts, budget_s=15.0,
                         max_points=1600, max_sew_facets=200_000):
    """EXACT point-to-surface distances where bounded, sampled NN otherwise.

    Sampled nearest-neighbour carries a cloud-spacing error that no budget can
    remove from FLAT faces (OCC tessellates them as two giant triangles, and a
    capped subdivision still leaves grid-vs-grid misalignment of whole
    millimetres on a geometrically exact model). ``distToShape`` has none: a
    point on the surface measures 0 whatever the tessellation. Scan→model is
    exact per point against the real shape; model→scan is exact per point
    against the sewn scan solid (small meshes only — sewing is the expensive
    step); anything beyond the bounds falls back to the sampled clouds."""
    d_scan = d_model = None
    eng_scan = eng_model = "sampled-NN"

    def _subset(points):
        if len(points) <= max_points:
            return list(points)
        stride = max(1, (len(points) + max_points - 1) // max_points)
        return points[::stride]

    try:
        import Part

        sel = _importance_sample(scan_pts, model_pts, _subset)
        out, t0 = [], time.monotonic()
        for p in sel:
            out.append(float(Part.Vertex(float(p[0]), float(p[1]),
                                         float(p[2])).distToShape(shape)[0]))
            if len(out) % 512 == 0 and time.monotonic() - t0 > budget_s:
                out = None
                break
        if out is not None:
            d_scan, eng_scan = out, "exact distToShape"
    except Exception:  # noqa: BLE001 - the sampled fallback always remains
        d_scan = None
    if d_scan is None:
        d_scan, eng_scan = _distances(scan_pts, model_pts)
    try:
        import numpy as np

        # the numpy point-to-mesh pass never SEWS (the triangle-BRep sewing
        # that hung the kernel on self-intersecting meshes is gone); the old
        # 4,000-facet gate was that sewing gate — the sampled-NN fallback it
        # forced on big meshes read whole millimetres of phantom distance on
        # flat faces (case 07: m2s.max 7.5 mm while the exact s2m bound was
        # 0.72). Chunks bound the memory; the budget bounds the time.
        if int(getattr(mesh, "CountFacets", 10 ** 9)) <= max_sew_facets:
            exact = _points_to_mesh_exact(_subset(model_pts), mesh)
            if exact is not None and np.all(np.isfinite(exact)):
                sub = _subset(model_pts)
                exact, dropped = _drop_phantom_max(sub, [float(x) for x in exact],
                                                   shape)
                d_model = list(exact)
                eng_model = "exact point-to-mesh (numpy)"
                if dropped:
                    eng_model += f"; {dropped} tessellation-phantom sample(s) excluded"
    except Exception:  # noqa: BLE001
        d_model = None
    if d_model is None:
        d_model, eng_model = _distances(model_pts, scan_pts)
        d_model, dropped = _drop_phantom_max(model_pts, d_model, shape)
        if dropped:
            eng_model += f"; {dropped} tessellation-phantom sample(s) excluded"
    return d_scan, eng_scan, d_model, eng_model


def deviation_rows(mesh, shape, *, a="scan", b="model", warn_mm=0.3, budget_s=20.0,
                   clock=None, mask=None) -> list:
    """PASS/WARN deviation rows (never FAIL) comparing the source mesh with the reconstructed
    ``shape``: per-direction RMS / P95 / P99 / max and the two-sided Hausdorff estimate.

    G6 measured-only ``mask``: ``{"scan_exclude_indices": [vertex indices into the mesh],
    "model_regions": [{"points": closed 3D ring, "normal": plane normal, "tol_mm": band}],
    "note": str}`` — scan vertices on synthetic (repair-created) facets and model samples
    inside missing regions are excluded, and a ``measured_only`` row reports the counts so
    the masking is never silent. ``None`` (default) keeps the historic behaviour."""
    excl = [int(i) for i in (mask.get("scan_exclude_indices") or ())] if mask else []
    scan_pts = _mesh_points(mesh, exclude_vertices=excl)
    model_pts = _shape_points(shape)
    scan_pts, model_pts, masked = _apply_mask(scan_pts, model_pts, mask,
                                              excluded_scan=len(excl))
    rows = _build_rows(scan_pts, model_pts, a=a, b=b, warn_mm=warn_mm,
                       distances=_exact_raw_distances(mesh, shape, scan_pts, model_pts))
    if masked is not None:
        rows.append(_measured_only_row(masked, a, b))
    return rows


def registered_deviation(mesh, shape, *, a="scan", b="model", warn_mm=0.3, budget_s=20.0,
                   max_icp_points=4000, clock=None, mask=None, register=True):
    """Raw deviation + numpy ICP registration + aligned deviation, one sampling policy for
    both sides (registration mandate 2026-11-14). Returns ``{"raw", "aligned",
    "registration", "mask"}``; ``aligned`` is None whenever the registration is invalid or
    non-converged — it is reported as a lower bound only and NEVER changes the
    reconstruction outcome (the caller keeps using ``raw``). The transform maps scan→model
    (``p' = R @ p + t``). The G6 ``mask`` (see ``deviation_rows``) is applied ONCE, by
    index, before anything else — raw and aligned rows therefore measure the identical
    measured-only point sets (parity preserved; exclusions are index-based so they are
    transform-invariant).

    ``register=False`` skips the ICP block entirely (the refinement re-measure path:
    RAW is what decides, and the aligned rows are not needed to compare two candidate
    states of the same frame) — ``aligned``/``registration`` come back None and the raw
    rows are byte-identical to the registered call's."""
    import numpy as np

    now = clock or time.monotonic
    deadline = now() + max(1.0, float(budget_s))
    excl = [int(i) for i in (mask.get("scan_exclude_indices") or ())] if mask else []
    scan_pts = _mesh_points(mesh, exclude_vertices=excl)
    model_pts = _shape_points(shape)
    scan_pts, model_pts, masked = _apply_mask(scan_pts, model_pts, mask,
                                              excluded_scan=len(excl))
    raw = _build_rows(scan_pts, model_pts, a=a, b=b, warn_mm=warn_mm,
                      distances=_exact_raw_distances(mesh, shape, scan_pts, model_pts))
    if masked is not None:
        raw.append(_measured_only_row(masked, a, b))
    if not register:
        return {"raw": raw, "aligned": None, "registration": None, "mask": masked}
    from . import register

    if len(scan_pts) < 4 or len(model_pts) < 4:
        return {"raw": raw, "aligned": None,
                "registration": None, "mask": masked}
    stride = max(1, (len(scan_pts) + max_icp_points - 1) // max_icp_points)
    scan_sub = np.asarray(scan_pts[::stride], dtype=float)
    centroids, normals = shape_points_and_normals(shape)
    mstride = max(1, (len(centroids) + max_icp_points - 1) // max_icp_points)
    centroids, normals = centroids[::mstride], normals[::mstride]
    if now() >= deadline:
        return {"raw": raw, "aligned": None,
                "registration": {"valid": False, "converged": False, "method": "skipped",
                                 "warnings": ["the time budget ran out before registration"]},
                "mask": masked}
    reg = register.icp_register(scan_sub, centroids, normals, method="auto")
    if not reg.get("valid"):
        # planar-blind rescue: point-to-plane on face centroids cannot see in-plane sliding;
        # the model's tessellation VERTICES (corners/edges) make point-to-point observable
        # planar-blind rescue: point-to-plane on the sampled surface cannot see in-plane
        # sliding (a box's flat faces); the DENSIFIED surface samples (issue #13 — raw
        # tessellation vertices are a prismatic solid's 8 corners, far too few) make
        # point-to-point observable and converge where the normal-only view cannot
        retry = register.icp_register(scan_sub, centroids[::mstride], None,
                                      method="point_to_point")
        if retry.get("valid"):
            retry["method"] = "point_to_plane→point_to_point(surface samples)"
            retry["warnings"] = (["point-to-plane on the surface samples did not converge; "
                                  "sample-based point_to_point fallback: "]
                                 + (reg.get("warnings") or []) + retry.get("warnings", []))
            reg = retry
    if not reg.get("valid"):
        return {"raw": raw, "aligned": None, "registration": reg, "mask": masked}
    # Issue #13: a CONVERGED registration can still be one of several equally good alignments
    # (symmetric or degenerate geometry admits 180°-flip twins). The bounded PCA second
    # opinion marks it — the aligned rows stay (a lower bound of SOME good alignment) but are
    # reported unreliable; they never touch the outcome either way.
    try:
        if now() < deadline:
            _, amb = register._pca_init(scan_sub, centroids)
            if amb.get("ambiguous"):
                reg = dict(reg)
                reg["ambiguous"] = True
                reg["warnings"] = reg.get("warnings", []) + [
                    "alignment AMBIGUOUS (symmetric or degenerate geometry): several rigid "
                    "transforms fit equally well — the aligned numbers are unreliable"]
    except Exception:  # noqa: BLE001 - the probe is advisory and never blocks the rows
        pass
    R, t = np.asarray(reg["R"]), np.asarray(reg["t"])
    aligned_pts = (R @ np.asarray(scan_pts, dtype=float).T).T + t
    aligned = _build_rows(aligned_pts, model_pts, a=a, b=b, warn_mm=warn_mm,
                          engine_note="; AFTER ICP registration (aligned = lower bound; "
                                      "raw remains authoritative)")
    raw_h = next((r["value"] for r in raw if r["id"] == "hausdorff"), math.inf)
    alg_h = next((r["value"] for r in aligned if r["id"] == "hausdorff"), math.inf)
    if alg_h > raw_h:
        # a converged ICP can still be a WRONG local minimum (e.g. partial overlap): a
        # registration that worsens the deviation is harmful and gets discarded
        reg = dict(reg)
        reg["valid"] = False
        reg["warnings"] = reg.get("warnings", []) + [
            f"registration WORSENED the deviation ({raw_h:.3f} → {alg_h:.3f} mm Hausdorff; "
            "wrong local minimum) — aligned result discarded, raw stays authoritative"]
        return {"raw": raw, "aligned": None, "registration": reg, "mask": masked}
    return {"raw": raw, "aligned": aligned, "registration": reg, "mask": masked}


def exact_iou(mesh, shape, *, facet_cap: int = 20000, a: str = "scan", b: str = "model") -> dict:
    """Exact OCC boolean intersection-over-union: ``volume(A∩B) / volume(A∪B)`` between the
    reconstructed ``shape`` and the SEWN scan solid (D-1; the CADBench-comparable acceptance
    metric, DISTINCT from the sampled deviation rows — those stay authoritative for the outcome).

    Honest on every failure mode: a mesh above ``facet_cap`` facets (boolean cost on ordinary
    laptops), a sew failure, a boolean error, an invalid shape or a degenerate union each return
    ``{"iou": None, "note": <the reason>}`` — never a substituted metric, never a guessed value.
    The elapsed time is reported so the caller can tighten the cap from evidence."""
    t0 = time.monotonic()
    out = {"kind": "exact_iou", "a": a, "b": b, "iou": None, "note": "",
           "elapsed_s": 0.0}
    try:
        facets = int(mesh.CountFacets)
    except Exception:  # noqa: BLE001 - no facet count, no honest cost bound
        out["note"] = "facet count unavailable"
        return out
    if facets > facet_cap:
        out["note"] = f"skipped: {facets} facets above the {facet_cap} cap (boolean cost)"
        return out
    from ..cad.imported import solid_from_mesh

    try:
        scan_solid, _sew_note = solid_from_mesh(mesh)
    except Exception as exc:  # noqa: BLE001 - a failed sew is a reported skip
        out["note"] = f"sew failed: {type(exc).__name__}"
        return out
    if scan_solid is None or scan_solid.Volume <= 0:
        out["note"] = "scan solid not sewable (open or zero-volume)"
        return out
    if shape is None or not shape.isValid() or shape.Volume <= 0:
        out["note"] = "reconstructed shape invalid or zero-volume"
        return out
    try:
        inter = shape.common(scan_solid).Volume
        union = shape.Volume + scan_solid.Volume - inter
    except Exception as exc:  # noqa: BLE001 - OCC boolean failures are a reported skip
        out["note"] = f"boolean failed: {type(exc).__name__}"
        return out
    out["elapsed_s"] = round(time.monotonic() - t0, 2)
    if union <= 0:
        out["note"] = "degenerate union (zero volume)"
        return out
    if inter <= 0:
        # physically impossible for solids whose bounding boxes overlap: the sew of a
        # NOISY mesh can yield a degenerate solid against which OCC's common() silently
        # returns empty (observed: iou 0.0 with both volumes positive on the noisy
        # bracket). Reporting 0.0 would claim a measured zero overlap that does not
        # exist — the honest outcome is "cannot compute".
        bi = shape.BoundBox
        bj = scan_solid.BoundBox
        overlap = (min(bi.XMax, bj.XMax) - max(bi.XMin, bj.XMin) > 0
                   and min(bi.YMax, bj.YMax) - max(bi.YMin, bj.YMin) > 0
                   and min(bi.ZMax, bj.ZMax) - max(bi.ZMin, bj.ZMin) > 0)
        out["note"] = ("empty intersection despite overlapping bounding boxes "
                       "(degenerate sewn solid)" if overlap
                       else "the shapes' bounding boxes do not overlap (frame mismatch?)")
        return out
    out["iou"] = round(inter / union, 4)
    return out
