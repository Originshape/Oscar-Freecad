"""Mesh noise estimation, noise-adaptive tolerances (Stage 1 / K1) and the scan-quality
assessment (§2, G5): completeness, integrity and plausibility of the INPUT scan, and the
readiness verdict that gates reconstruction decisions.

A scan's geometric noise sigma is estimated from the mesh itself: for a sample of vertices,
the k-nearest-neighbourhood is fitted a least-squares plane (SVD) and the point-to-plane
residuals are pooled; sigma = 1.4826 x MAD of the pooled residuals, times a fixed calibration
for the k-neighbourhood fit's degrees-of-freedom bias (measured on synthetic noisy-plane
ladders: raw ratio ~0.85 at k=16, stable across sigma 0.01-0.10 mm). Neighbourhoods whose own
fit RMS is far above the median are dropped first (edges and corners inflate residuals and
would otherwise read as "noise" on coarse tessellations — measured on a coarse fused shape).

Recognition gates are then DERIVED, not fixed: plane max(2.5*sigma, 0.02*mean_edge) and
cylinder max(3*sigma, 0.1*mean_edge) mm (curved walls carry tessellation chord error). On a
clean tessellated mesh sigma floors at ~0 and the edge-length floor keeps the gates finite.

§2 (G5) answers "can this scan support a reconstruction at all?" before anything is built:
- completeness: every boundary loop classified (planarity, roundness, rim-normal alignment —
  the classifier thresholds are OriginShape's measured ones) as outer_rim / hole_opening /
  bore_end / dropout, with through-pairing of coaxial round gaps (a hole whose wall was never
  scanned, seen from both faces). bore_end is kept as its own kind so the end of a SCANNED
  bore is never misreported as a defect.
- integrity: doubled surfaces (rays cast both ways along sampled facet normals — Möller-
  Trumbore over a centroid-ball broad-phase, cKDTree with a chunked-numpy fallback — finding
  a same-facing layer 4..40 sigma or 3 mean edges off, capped at 2% of the diagonal) and
  spikes (leave-one-out k-neighbourhood plane residual >= 6 sigma AND >= 1 mean edge, with
  isolation/planarity/in-plane-containment tests, so edges, coarse-tessellation corners and
  displaced PATCHES are never called spikes).
- plausibility: bbox diagonal under both mm/inch readings, the smallest hole radius this
  tessellation could evidence (16 rim points at the mean edge), and the K4 unit status.
- readiness verdict ready / caution / degraded with reasons; DEGRADED caps a reconstruction
  outcome at partially-verified (the decision gate), caution items are reported only.

Pure numpy on mesh points/topology (cKDTree when scipy is importable, chunked numpy
otherwise — the same dependency-free fallback pattern as recon/register.py). Read-only:
never touches the mesh. Honest scope: a single global sigma per mesh; mixed-scale scans and
curvature-induced inflation are mitigated only by the smoothness filter and the edge floor.
"""

import numpy as np

from .units import INCH_MM

K_NEIGHBOURS = 16
SAMPLE_POINTS = 800               # neighbourhoods actually fitted; runtime scales with this
CALIBRATION = 1.18                # k=16 dof bias: raw sigma reads ~0.85x on plane ladders
SMOOTHNESS_TRIM = 3.0             # drop neighbourhoods with fit RMS above 3x the running median
SIGMA_EDGE_FLOOR = 0.005          # sigma floors at 0.5% of the mean edge (fine scans carry
                                  # real sub-0.02 noise; a 2% floor would swamp it — measured)
EDGE_FLOOR = 0.02                 # PLANE-gate floor: 2% of the mean edge length
PLANE_SIGMA_MULT = 2.5            # plane acceptance gate = max(2.5*sigma, EDGE_FLOOR*edge)
CYLINDER_SIGMA_MULT = 3.0         # cylinder gate = max(3*sigma, 0.1*edge)
CYLINDER_EDGE_MULT = 0.1


def _knn(query, points, k, exclude_self=True):
    """k nearest neighbours of each query row; cKDTree when available, chunked numpy else.

    ``exclude_self=False`` is for queries against a DIFFERENT point set (displacement checks):
    the nearest neighbour is real and must not be sliced away."""
    kk = k + 1 if exclude_self else k
    try:
        from scipy.spatial import cKDTree

        d, i = cKDTree(points).query(query, k=kk)
        if exclude_self:
            d, i = d[:, 1:], i[:, 1:]
        return d, i, "scipy.cKDTree"
    except Exception:  # noqa: BLE001 - chunked fallback keeps the module dependency-free
        out_d, out_i = [], []
        for start in range(0, len(query), 512):
            block = query[start:start + 512]
            dd = np.sqrt(((block[:, None, :] - points[None, :, :]) ** 2).sum(2))
            part = np.argpartition(dd, kk, axis=1)[:, :kk]
            rows = np.arange(len(block))[:, None]
            pd = dd[rows, part]
            order = np.argsort(pd, axis=1)
            picked = part[rows, order]
            if exclude_self:
                picked = picked[:, 1:]
            out_d.append(dd[rows, picked])
            out_i.append(picked)
        return (np.concatenate(out_d), np.concatenate(out_i), "numpy-chunked")


def _mean_edge_mm(pts, fac):
    e = np.stack([pts[fac[:, 1]] - pts[fac[:, 0]],
                  pts[fac[:, 2]] - pts[fac[:, 1]],
                  pts[fac[:, 0]] - pts[fac[:, 2]]], axis=1)
    return float(np.linalg.norm(e, axis=2).mean())


def estimate_sigma(mesh, k=K_NEIGHBOURS, sample=SAMPLE_POINTS) -> dict:
    """Estimate the mesh's noise sigma (mm) and the scale the tolerances derive from.

    Read-only on the mesh. Returns sigma_mm (calibrated, edge-floored), sigma_raw_mm,
    mean_edge_mm, floor_mm, sampled (neighbourhood count) and the NN engine used.
    """
    points, facets = mesh.Topology
    pts = np.array([[p.x, p.y, p.z] for p in points], dtype=float)
    fac = np.array(facets, dtype=int).reshape(-1, 3)
    mean_edge = _mean_edge_mm(pts, fac) if len(fac) else 0.0
    floor = SIGMA_EDGE_FLOOR * mean_edge
    base = {"mean_edge_mm": round(mean_edge, 4), "floor_mm": round(floor, 5),
            "k": k, "sampled": 0, "engine": "none", "note": ""}
    if len(pts) < k + 1 or not np.isfinite(pts).all() or mean_edge <= 0.0:
        base.update(sigma_mm=round(floor, 5), sigma_raw_mm=0.0,
                    note="too few/corrupt points for a neighbourhood fit — edge floor only")
        return base
    step = max(1, len(pts) // max(1, sample))
    query = pts[::step]
    dist, nb, engine = _knn(query, pts, k)
    res, rms = [], []
    for row in range(len(nb)):
        P = pts[nb[row]]
        c = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        r = (P - c) @ vt[2]
        res.append(r)
        rms.append(float(np.sqrt((r ** 2).mean())))
    rms = np.asarray(rms)
    kept = np.ones(len(rms), dtype=bool)
    for _ in range(2):                       # drop edge/corner neighbourhoods, re-median once
        med = float(np.median(rms[kept]))
        if med <= 0.0:
            break
        new = kept & (rms <= SMOOTHNESS_TRIM * med)
        if new.sum() < max(4, len(rms) // 10) or (new == kept).all():
            break
        kept = new
    pooled = np.concatenate([r for r, ok in zip(res, kept) if ok]) if kept.any() else np.concatenate(res)
    mad = float(np.median(np.abs(pooled - np.median(pooled))))
    raw = 1.4826 * mad
    sigma = max(CALIBRATION * raw, floor)
    return {"sigma_mm": round(float(sigma), 5), "sigma_raw_mm": round(raw, 5),
            "mean_edge_mm": round(mean_edge, 4), "floor_mm": round(floor, 5),
            "k": k, "sampled": int(kept.sum()), "engine": engine,
            "note": "smoothness filter dropped %d of %d neighbourhoods" % (
                int((~kept).sum()), len(kept)) if (~kept).any() else ""}


def tolerances(sigma_mm: float, mean_edge_mm: float) -> dict:
    """Noise-adaptive recognition gates (mm): plane max(2.5s, 0.02e), cylinder max(3s, 0.1e)."""
    sigma = max(float(sigma_mm), 0.0)
    edge = max(float(mean_edge_mm), 0.0)
    return {"plane_rms_mm": round(max(PLANE_SIGMA_MULT * sigma, EDGE_FLOOR * edge), 5),
            "cylinder_rms_mm": round(max(CYLINDER_SIGMA_MULT * sigma, CYLINDER_EDGE_MULT * edge), 5)}


def estimate_tolerances(mesh, k=K_NEIGHBOURS, sample=SAMPLE_POINTS) -> dict:
    """One call for the recognition pipeline: sigma estimate + derived tolerance gates."""
    est = estimate_sigma(mesh, k=k, sample=sample)
    est.update(tolerances(est["sigma_mm"], est["mean_edge_mm"]))
    return est


# ------------------------------------------------------------------ G5: scan quality (§2)
# Thresholds ported from OriginShape's measured classifier (geom/scan_quality.py), reduced
# to what reconstruction gating needs; every classification carries its evidence numbers.
RIM_AREA_SHARE = 0.30        # the largest loop spanning >= 30% of the mesh area is the
                             # outer rim of a one-sided scan (OriginShape's measured shell
                             # share; below it a big flat gap is reported as what it is —
                             # an unscanned face — which is the honest reading for e.g. a
                             # box whose whole top face is missing)
PINHOLE_EDGES = 6            # loops with <= 6 edges …
PINHOLE_WIDTH_EDGES = 2.5    # … or width < 2.5x their OWN edge length are pinholes (the
                             # loop-local edge, not the mesh mean: long radial edges would
                             # otherwise swallow round openings — measured on an annulus) —
                             # and only when the loop is SMALL (< 2% of the scan area): a
                             # coarse 4-edge rim of a whole missing face is not a pinhole
MIN_ROUND_EDGES = 8          # a roundness claim needs >= 8 rim vertices; a square rim's
                             # 4 corner vertices all sit at one radius and would read as
                             # perfectly round (measured on a box missing its top face)
ROUND_TOL = 0.08             # rim radius spread < 8% = round (a hole's opening)
PLANAR_FRAC = 0.05           # loop planarity rms <= 5% of its diameter (+ 2 sigma)
RIM_ALIGN_COS = 0.6          # median |vertex-normal . loop-normal| above this: the rim sits
                             # in a FLAT face — the hole's wall was NOT scanned; below it
                             # the surrounding faces ARE the wall (bore end)
UNSCANNED_FACE_SHARE = 0.02  # a non-round flat dropout spanning >= 2% = a face never scanned
THROUGH_PAIR_COS = 0.95      # through-pairing: axes parallel …
THROUGH_PAIR_RADIUS = 0.15   # … radii equal within 15% …
THROUGH_PAIR_LATERAL = 0.25  # … centroids laterally within 25% of the larger radius
SPIKE_SIGMA = 6.0            # leave-one-out residual >= 6 sigma AND >= 1 mean edge, with
SPIKE_EDGE_MULT = 1.0        # neighbours quiet (max neighbour residual <= 0.5x own), the
SPIKE_ISOLATION = 0.5        # neighbourhood itself planar (rms <= 2 sigma), and the
SPIKE_NB_RMS_MULT = 2.0      # vertex's projection INSIDE its neighbourhood's in-plane
SPIKE_INPLANE_MULT = 1.5     # extent (<= 1.5x). The edge floor kills the coarse-tessellation
                             # corner artifact: a box corner's distant k-NN plane tilts and
                             # the corner reads 8.3 mm off at sigma 0.05 — but under its
                             # own 9.8 mm mean edge, i.e. below the tessellation's
                             # resolution, while a real spike stands off by whole edges
NOISE_FREE_EDGE = 0.01       # sigma floors at 1% of the mean edge for these tests
DOUBLE_MIN_SIGMA = 4.0       # a second layer >= 4 sigma off (closer is noise) …
DOUBLE_MAX_SIGMA = 40.0      # … and <= 40 sigma …
DOUBLE_EDGE_MULT = 3.0       # … or 3 mean edges (whichever is more), capped at …
DOUBLE_DIAG_CAP = 0.02       # … 2% of the scan diagonal, facing the same way …
DOUBLE_COS = 0.8             # … (normals within ~37 deg)
DOUBLE_MIN_SAMPLES = 5       # a doubled-surface cluster needs >= 5 sampled faces
DOUBLE_SAMPLES = 1200        # faces whose normals cast rays (area-weighted, seeded)
LAPTOP_DOUBLE_SURFACE_MAX_FACETS = 1500
LAPTOP_DOUBLE_SURFACE_SAMPLES = 300  # bound ray work on larger scans in this laptop profile
MIN_HOLE_RIM_POINTS = 16     # a hole radius needs >= 16 rim points to be evidencable


def _pts_fac(mesh):
    from ._arrays import mesh_arrays

    return mesh_arrays(mesh)   # P4.1: the one shared materializer


def _facet_geometry(pts, fac):
    """Per-facet (areas, centroids, unit normals) from raw arrays."""
    p = pts[fac]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    areas = 0.5 * norms[:, 0]
    normals = cross / np.maximum(norms, 1e-15)
    return areas, p.mean(axis=1), normals


def _vertex_normals(pts, fac):
    """Area-weighted vertex normals (incident facet normals summed by area, renormalized)."""
    areas, _, normals = _facet_geometry(pts, fac)
    acc = np.zeros_like(pts)
    for k in range(3):
        np.add.at(acc, fac[:, k], normals * areas[:, None])
    lens = np.linalg.norm(acc, axis=1, keepdims=True)
    return acc / np.maximum(lens, 1e-15)


def _newell(P):
    """Vector area of an ORDERED 3D ring (Newell): direction = the loop's normal,
    magnitude = the spanned area."""
    return np.cross(P, np.roll(P, -1, axis=0)).sum(axis=0) / 2.0


def _completeness(mesh, pts, fac, sigma):
    """G5 completeness: every closed boundary loop with its most likely meaning, and the
    open-chain count. Read-only (loop rings come from meshops.boundary)."""
    from . import meshops                      # lazy: meshops imports this module at load

    vn = _vertex_normals(pts, fac)
    total_area = float(mesh.Area) or 1.0
    bnd = meshops.boundary(mesh)
    entries = []
    for entry in bnd["loops"]:
        ring = np.asarray(entry["ring"], dtype=int)
        P = pts[ring]
        n_vec = _newell(P)
        n_len = float(np.linalg.norm(n_vec))
        normal = n_vec / max(n_len, 1e-12)
        c = P.mean(axis=0)
        h = (P - c) @ normal
        R = np.linalg.norm((P - c) - np.outer(h, normal), axis=1)
        radius = float(R.mean())
        spread = float(R.std() / radius) if radius > 0 else 1.0
        planarity = float(np.sqrt(np.mean(h ** 2)))
        align = float(np.median(np.abs(vn[ring] @ normal)))
        perimeter = float(entry["perimeter_mm"])
        area = n_len
        share = area / total_area
        entries.append({"kind": "dropout", "detail": "", "n_edges": len(ring),
                        "perimeter_mm": round(perimeter, 3),
                        "spanned_area_mm2": round(area, 3), "area_share": round(share, 4),
                        "planarity_rms_mm": round(planarity, 4),
                        "radius_spread": round(spread, 4), "rim_alignment": round(align, 3),
                        "radius_mm": round(radius, 3),
                        "centroid_mm": [round(float(x), 3) for x in c],
                        "normal": [round(float(x), 4) for x in normal],
                        # G6: the ring's vertex indices (walk order) — the fill policy and
                        # the deviation mask need the loop's own polygon, not its summary
                        "ring": [int(i) for i in ring],
                        "_area": area})
    # the outer rim is the LARGEST loop when it spans a big share of the scan (a hole
    # opening can only outweigh it when the bore is nearly the whole scan — documented)
    if entries:
        rim = max(range(len(entries)), key=lambda i: entries[i]["_area"])
        if entries[rim]["area_share"] >= RIM_AREA_SHARE:
            entries[rim]["kind"] = "outer_rim"
    for e in entries:
        if e["kind"] == "outer_rim":
            continue
        diameter = 2.0 * e["radius_mm"]
        planar = e["planarity_rms_mm"] <= PLANAR_FRAC * diameter + 2.0 * sigma
        width = 4.0 * e["spanned_area_mm2"] / e["perimeter_mm"] \
            if e["perimeter_mm"] > 0 else 0.0
        own_edge = e["perimeter_mm"] / max(e["n_edges"], 1)
        if (e["n_edges"] <= PINHOLE_EDGES or width < PINHOLE_WIDTH_EDGES * own_edge) \
                and e["area_share"] < UNSCANNED_FACE_SHARE:
            e["detail"] = "pinhole"
        elif e["n_edges"] >= MIN_ROUND_EDGES and planar and \
                e["radius_spread"] < ROUND_TOL:
            e["kind"] = "hole_opening" if e["rim_alignment"] > RIM_ALIGN_COS else "bore_end"
        elif planar and e["area_share"] >= UNSCANNED_FACE_SHARE:
            e["detail"] = "unscanned_face"
    _pair_through_holes(entries)
    for e in entries:
        e.pop("_area", None)
    return {"boundary_loops": len(entries), "open_chains": len(bnd["open_chains"]),
            "boundary_junctions": int(bnd["junctions"]), "loops": entries}


def _pair_through_holes(entries):
    """Pair two round gaps on one axis as the two ends of one through hole — hole
    openings (wall never scanned) pair into "a hole the reconstruction cannot include";
    bore ends (wall scanned part-way) pair into measured depth for the wall that exists.
    OriginShape's pairing test, evidence-first: both loops keep their own numbers and
    gain the partner's index and the MEASURED depth."""
    round_gaps = [e for e in entries if e["kind"] in ("hole_opening", "bore_end")]
    for a in round_gaps:
        for b in round_gaps:
            if a is b or "through" in a or "through" in b:
                continue
            na, nb = np.asarray(a["normal"]), np.asarray(b["normal"])
            r = max(a["radius_mm"], b["radius_mm"])
            if r <= 0:
                continue
            d = np.asarray(b["centroid_mm"]) - np.asarray(a["centroid_mm"])
            lateral = float(np.linalg.norm(d - float(d @ na) * na))
            if abs(float(na @ nb)) > THROUGH_PAIR_COS and \
                    abs(a["radius_mm"] - b["radius_mm"]) < THROUGH_PAIR_RADIUS * r and \
                    lateral < THROUGH_PAIR_LATERAL * r:
                depth = float(abs(d @ na))
                a["through"] = {"other": entries.index(b), "depth_mm": round(depth, 3)}
                b["through"] = {"other": entries.index(a), "depth_mm": round(depth, 3)}


def _spikes(pts, sigma, mean_edge, sample=SAMPLE_POINTS) -> dict:
    """G5 integrity (a): isolated spike vertices by leave-one-out residual — the vertex's
    distance to the plane fitted through its k nearest NEIGHBOURS (itself excluded) —
    accepted only when it stands off by >= 6 sigma AND >= 1 mean edge, no neighbour is
    half as far off the same plane (isolation: displaced PATCHES fail here), the
    neighbourhood itself is planar (edges fail here), and the vertex's projection stays
    inside the neighbourhood's in-plane extent (coarse corners fail here). Sampled on big
    meshes (count is a scaled estimate, flagged as such)."""
    L = max(mean_edge, 1e-12)
    s = max(sigma, NOISE_FREE_EDGE * L)
    floor = max(SPIKE_SIGMA * s, SPIKE_EDGE_MULT * L)
    out = {"count": 0, "estimated": False, "examples": []}
    if len(pts) < K_NEIGHBOURS + 1:
        return out
    step = max(1, len(pts) // sample)
    rows = np.arange(0, len(pts), step)
    _, nb, _ = _knn(pts[rows], pts, K_NEIGHBOURS)
    h = np.zeros(len(rows))
    nbmax = np.zeros(len(rows))
    nbrms = np.zeros(len(rows))
    inplane = np.zeros(len(rows))
    for row in range(len(rows)):
        P = pts[nb[row]]
        c = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        nrm = vt[2]
        res = (P - c) @ nrm
        h[row] = abs(float((pts[rows[row]] - c) @ nrm))
        nbmax[row] = float(np.abs(res).max())
        nbrms[row] = float(np.sqrt((res ** 2).mean()))
        Q = (P - c) - np.outer(res, nrm)            # neighbours' in-plane offsets
        v = (pts[rows[row]] - c)
        vin = v - float(v @ nrm) * nrm              # the vertex's in-plane offset
        inplane[row] = float(np.linalg.norm(vin) /
                            max(float(np.linalg.norm(Q, axis=1).max()), 1e-12))
    found = (h >= floor) & (nbmax <= SPIKE_ISOLATION * h) & \
            (nbrms <= SPIKE_NB_RMS_MULT * s) & (inplane <= SPIKE_INPLANE_MULT)
    scale = len(pts) / max(len(rows), 1)
    out["count"] = int(round(int(found.sum()) * scale))
    out["estimated"] = scale > 1.0
    if found.any():
        worst = np.argsort(-h[found])[:5]
        out["examples"] = [[round(float(x), 3) for x in p] for p in
                           pts[rows[found]][worst]]
    return out


def _clusters_labels(P, radius):
    """Connected-component labels for points linked within ``radius`` (scipy when
    importable, dict-grid union-find otherwise — the dependency-free pattern)."""
    n = len(P)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        pairs = cKDTree(P).query_pairs(radius, output_type="ndarray")
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        return connected_components(g, directed=False)[1]
    except Exception:  # noqa: BLE001
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        inv = 1.0 / max(radius, 1e-12)
        cells: dict = {}
        keys = [(int(np.floor(p[0] * inv)), int(np.floor(p[1] * inv)),
                 int(np.floor(p[2] * inv))) for p in P]
        for i, key in enumerate(keys):
            cells.setdefault(key, []).append(i)
        for i, key in enumerate(keys):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in cells.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                            if j <= i:
                                continue
                            if float(((P[i] - P[j]) ** 2).sum()) <= radius * radius:
                                ri, rj = find(i), find(j)
                                if ri != rj:
                                    parent[ri] = rj
        return np.asarray([find(i) for i in range(n)], dtype=np.int64)


def _ball_candidates(centroids, origins, radius):
    """Triangle-centroid indices within ``radius`` of each ray origin — the ray broad
phase (a centroid ball of radius hit-band + max triangle reach is conservative: a triangle
whose centroid is farther cannot be hit inside the band)."""
    try:
        from scipy.spatial import cKDTree

        return [np.asarray(ix, dtype=int) for ix in
                cKDTree(centroids).query_ball_point(origins, radius)]
    except Exception:  # noqa: BLE001
        out = []
        for start in range(0, len(origins), 256):
            block = origins[start:start + 256]
            dd = np.sqrt(((block[:, None, :] - centroids[None, :, :]) ** 2).sum(2))
            out.extend(np.nonzero(dd[r] <= radius)[0] for r in range(len(block)))
        return out


def _ray_hits(origins, dirs, v0, v1, v2, t_min, t_max):
    """Vectorized Möller–Trumbore over matched (origin, dir, triangle) rows → the hit
    distance inside [t_min, t_max], or nan."""
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(dirs, e1)
    det = np.einsum("ij,ij->i", p, e2)
    t_vec = origins - v0
    q = np.cross(t_vec, e2)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / det
        u = np.einsum("ij,ij->i", t_vec, p) * inv
        v = np.einsum("ij,ij->i", dirs, q) * inv
        t = np.einsum("ij,ij->i", e1, q) * inv
        ok = (np.abs(det) > 1e-14) & (u >= -1e-9) & (v >= -1e-9) & \
             (u + v <= 1 + 1e-9) & (t >= t_min) & (t <= t_max)
        return np.where(ok, t, np.nan)


def _adjacency_sigma(pts, fac, sample=SAMPLE_POINTS) -> float:
    """A TOPOLOGICALLY local noise estimate for the doubled-surface test: the sampled
    vertices' residuals to planes fitted through their 1-ring adjacency neighbours
    (MAD-scaled). The K1 sigma is metric (k-NN): on a doubled surface the k-NN ball
    crosses the two layers, the neighbourhood fit straddles them, and sigma swallows
    the very offsets being hunted (measured: two 0.8 mm layers read as sigma 0.70,
    emptying the hit band). The 1-ring always stays in one layer, so the offset stays
    VISIBLE next to this sigma. Coarse tessellations (a cube's corner rings span three
    faces) can inflate it — the test takes the smaller of the two sigmas."""
    if len(fac) == 0 or len(pts) < 8:
        return 0.0
    step = max(1, len(pts) // sample)
    rows = np.arange(0, len(pts), step)
    mask = np.zeros(len(pts), dtype=bool)
    mask[rows] = True
    e = np.concatenate([fac[:, [0, 1]], fac[:, [1, 2]], fac[:, [2, 0]]])
    e = e[mask[e[:, 0]] | mask[e[:, 1]]]
    ring: dict = {}
    for a, b in e:
        ring.setdefault(int(a), set()).add(int(b))
        ring.setdefault(int(b), set()).add(int(a))
    res = []
    for v, ns in ring.items():
        if len(ns) < 3:
            continue
        P = pts[sorted(ns)]
        c = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        res.append(abs(float((pts[v] - c) @ vt[2])))
    if len(res) < 8:
        return 0.0
    r = np.asarray(res)
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def _doubled_surfaces(mesh, pts, fac, sigma, mean_edge, samples=DOUBLE_SAMPLES,
                      seed=20261114):
    """G5 integrity (b): two layers of one surface facing the same way a few noise widths
    apart (unaligned scan passes). Rays are cast BOTH ways along the normals of sampled
    faces; a hit inside the band on a same-facing face marks the sample. A thin wall's far
    side and a bore's far wall face the OTHER way and never count. Clustered samples
    become regions with the median offset between the layers."""
    if len(fac) == 0:
        return []
    areas, centroids, normals = _facet_geometry(pts, fac)
    L = max(mean_edge, 1e-12)
    # the smaller of the K1 sigma and the adjacency sigma (see _adjacency_sigma): a
    # doubled surface inflates the k-NN one, and the band must not swallow the offsets
    s = max(min(float(sigma or 0.0), _adjacency_sigma(pts, fac)), NOISE_FREE_EDGE * L)
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    lo = DOUBLE_MIN_SIGMA * s
    hi = min(max(DOUBLE_MAX_SIGMA * s, DOUBLE_EDGE_MULT * L), DOUBLE_DIAG_CAP * diag)
    if hi <= lo:
        return []
    rng = np.random.default_rng(seed)
    k = min(samples, len(fac))
    if k < len(fac) and float(areas.sum()) > 0:
        pick = rng.choice(len(fac), k, replace=False, p=areas / float(areas.sum()))
    else:
        pick = np.arange(k)
    v = pts[fac]
    reach = float(np.linalg.norm(v - centroids[:, None, :], axis=2).max())
    cands = _ball_candidates(centroids, centroids[pick], hi + reach)
    src_n = normals[pick]
    hit = np.zeros(k, dtype=bool)
    off = np.full(k, np.inf)
    for sgn in (1.0, -1.0):
        ray_rows, tri_idx = [], []
        for r, cand in enumerate(cands):
            if len(cand):
                ray_rows.append(np.full(len(cand), r))
                tri_idx.append(cand)
        if not ray_rows:
            continue
        ray_rows = np.concatenate(ray_rows)
        tri_idx = np.concatenate(tri_idx)
        dirs = sgn * src_n[ray_rows]
        t = _ray_hits(centroids[pick[ray_rows]], dirs,
                      v[tri_idx, 0], v[tri_idx, 1], v[tri_idx, 2], lo, hi)
        ok = np.isfinite(t) & (np.einsum("ij,ij->i", normals[tri_idx],
                                         src_n[ray_rows]) >= DOUBLE_COS)
        for r, tt in zip(ray_rows[ok], t[ok]):
            if tt < off[r]:
                off[r] = float(tt)
                hit[r] = True
    idx = np.nonzero(hit)[0]
    if len(idx) == 0:
        return []
    spacing = float(np.sqrt((float(mesh.Area) or 1.0) / k))
    labels = _clusters_labels(centroids[pick[idx]], 3.0 * spacing)
    per_area = (float(mesh.Area) or 1.0) / k
    out = []
    for c in np.unique(labels):
        j = idx[labels == c]
        if len(j) < DOUBLE_MIN_SAMPLES:
            continue
        out.append({"area_mm2": round(len(j) * per_area, 2),
                    "centroid_mm": [round(float(x), 3) for x in
                                    centroids[pick[j]].mean(axis=0)],
                    "offset_mm": round(float(np.median(off[j])), 4),
                    "n_samples": int(len(j))})
    out.sort(key=lambda d: -d["area_mm2"])
    return out


def scan_quality(mesh, est=None, units_status=None) -> dict:
    """G5: the scan-quality report — sigma echo, completeness, integrity, plausibility and
    the readiness verdict that gates reconstruction decisions. Read-only on the mesh.
    ``est`` may carry a precomputed quality estimate (sigma_mm/mean_edge_mm — the K1/K2
    pipelines already have one); ``units_status`` embeds the K4 scale verdict so the report
    is one-stop. DEGRADED readiness = doubled surfaces, missing regions, or hole openings
    without scanned walls; those cap a reconstruction at partially-verified. CAUTION items
    (spikes, open chains, bore ends, an outer rim) are reported but do not cap: the
    deviation math already prices them in."""
    if est is None:
        est = estimate_tolerances(mesh)
    sigma = float(est.get("sigma_mm") or 0.0)
    mean_edge = float(est.get("mean_edge_mm") or 0.0)
    pts, fac = _pts_fac(mesh)
    completeness = _completeness(mesh, pts, fac, sigma)
    double_samples = DOUBLE_SAMPLES
    double_check = {"total_facets": int(len(fac))}
    if len(fac) > LAPTOP_DOUBLE_SURFACE_MAX_FACETS:
        double_samples = min(double_samples, LAPTOP_DOUBLE_SURFACE_SAMPLES)
        double_check["sampling"] = "laptop_reduced"
        double_check["note"] = ("large-scan ray check reduced for this laptop; small doubled "
                                 "regions may be missed")
    elif len(fac) <= double_samples:
        double_check["sampling"] = "all_facets"
    else:
        double_check["sampling"] = "seeded_area_weighted"
    double_check["sampled_facets"] = min(int(len(fac)), int(double_samples))
    double_check["complete"] = double_check["sampled_facets"] == int(len(fac))
    integrity = {"spikes": _spikes(pts, sigma, mean_edge),
                 "doubled_surfaces": _doubled_surfaces(
                     mesh, pts, fac, sigma, mean_edge, samples=double_samples),
                 "doubled_surface_check": double_check}
    box = mesh.BoundBox
    diag = float(np.linalg.norm([box.XLength, box.YLength, box.ZLength]))
    L = max(mean_edge, 1e-12)
    plausibility = {"bbox_diagonal_mm": round(diag, 3),
                    "bbox_diagonal_inch": round(diag / INCH_MM, 4),
                    "min_detectable_hole_radius_mm":
                        round(MIN_HOLE_RIM_POINTS * L / (2.0 * np.pi), 3),
                    "min_detectable_basis":
                        f"{MIN_HOLE_RIM_POINTS} rim points at the {L:.3f} mm mean edge"}
    if units_status:
        plausibility["scale_status"] = units_status

    reasons, notes = [], []
    ds = integrity["doubled_surfaces"]
    if ds:
        reasons.append(f"{len(ds)} doubled-surface region(s) (offsets "
                       f"{[d['offset_mm'] for d in ds[:3]]} mm) — unaligned scan passes; "
                       "fits and deviation would straddle two layers")
    missing = [l for l in completeness["loops"] if l["kind"] == "dropout" and
               l["area_share"] >= UNSCANNED_FACE_SHARE]
    if missing:
        reasons.append(f"{len(missing)} missing region(s) each spanning >= "
                       f"{UNSCANNED_FACE_SHARE:.0%} of the scan area (unscanned face / "
                       "dropout)")
    openings = [l for l in completeness["loops"] if l["kind"] == "hole_opening"]
    if openings:
        paired = sum(1 for l in openings if l.get("through"))
        reasons.append(f"{len(openings)} hole opening(s) with NO scanned wall "
                       f"({paired} through-paired) — real holes exist without wall "
                       "evidence, so a reconstruction cannot include them")
    if integrity["spikes"]["count"]:
        notes.append(f"{integrity['spikes']['count']} isolated spike vertex/vertices "
                     "(reported; the deviation rows already price them in)")
    if completeness["open_chains"]:
        notes.append(f"{completeness['open_chains']} open boundary chain(s) (torn region)")
    bore = sum(1 for l in completeness["loops"] if l["kind"] == "bore_end")
    if bore:
        notes.append(f"{bore} bore end(s) — scanned walls that stop (partial evidence)")
    if any(l["kind"] == "outer_rim" for l in completeness["loops"]):
        notes.append("outer scan rim present (one-sided/open scan)")
    verdict = "degraded" if reasons else ("caution" if notes else "ready")
    return {"sigma_mm": est.get("sigma_mm"), "mean_edge_mm": est.get("mean_edge_mm"),
            "completeness": completeness, "integrity": integrity,
            "plausibility": plausibility,
            "readiness": {"verdict": verdict, "reasons": reasons, "notes": notes}}
