"""The bounded feature survey of a source mesh (or an exact B-rep): what a reconstruction can measure.

Two recognizers, one output schema (``{"face_count", "surface_kinds", "planes", "cylinders",
"valid", "warnings"}``):

* ``survey_solid(shape)`` reads the ANALYTIC faces of a B-rep (a STEP import): every
  ``Part::GeomPlane`` / ``Part::GeomCylinder`` face is exact, cylinders are grouped by axis
  and radius, and concave (a hole) vs convex (a boss) is probed with ``Shape.isInside``;
* ``survey_mesh(mesh)`` recognizes the same vocabulary on the tessellation itself — a scan
  or an STL has no analytic faces, so facets are grown into regions across small dihedral
  turns and each region is classified (plane / cylinder / sphere / freeform) by fits on its
  own points and normals. Concave vs convex reads the facet orientation (an outward-oriented
  hole wall points AT its axis, a pin away from it); an open or inward-wound mesh reports
  ``concave=None`` rather than a guess.

Everything is bounded: facets over ``MAX_FACETS`` skip recognition (bbox facts still exact,
a warning says so), a region under ``MIN_REGION_SHARE`` of the total area is noise unless it
is a cylinder (the wall of a small hole is a sliver of the area on a large plate), and no
method here raises — a mesh that cannot be surveyed reports itself honestly.

What the tessellation survey measures, and where it stops (six fixtures — a plate with four
holes, an L-bracket, a stepped shaft, a four-hole bracket, a rounded plate, a board with
four bosses and four mounting holes — at three seeds per noise level):

* clean tessellation: every hole, bore, boss and step is found, diameter error <= 0.01 mm;
* 0.01 mm of vertex noise: most are still found and nothing false is reported;
* from about 0.02 mm: on a finely tessellated face, noise turns the small facets at a hole's
  rim enough that region growing merges the face with its hole walls, and those holes are
  NOT reported (a face merges with its hole walls through the rim);
* from 0.05 mm: a few small spurious cylinders appear;
* a cylinder tangent to a plane (a rounded plate's ends, a fillet) grows into one region
  with that plane and is never reported as a cylinder.

``unsurveyed_area_pct`` is the share of the surface the vocabulary did not cover; the
model-facing text repeats it, because "no hole listed" must never read as "the part has no
hole" on a noisy scan.

The numbers feed plan parameters as ``step_measurement``-grade evidence: hole / bore / step
diameters and positions, plane normals and offsets, overall extents. Their accuracy is the
chord error of the tessellation (measured on the benchmark suite: <= 0.65 mm at 0.3 mm
deviation for the fixtures), which the tolerances of the tests carry.

Pure geometry on plain arrays (``_jacobi_eigen``, ``_fit_circle``, ``_fit_sphere_center``,
``_solve``) — no numpy, no FreeCAD import at module scope, so the math is testable anywhere.
"""

import math

MAX_FACETS = 20_000          # recognition is skipped above this (the bbox facts stay exact)
MIN_REGION_SHARE = 0.005     # a region under 0.5 % of the total area is noise — unless it is a cylinder
MAX_TURN_DEG = 60.0          # facets joined across a normal turn up to this grow one region (a
                             # smooth tessellation turns at most ~60° per step; a right-angle
                             # edge must keep a margin noise cannot close — at 75° it did not)
PLANE_SPREAD_DEG = 4.0       # a region whose normals stay this close is a plane
PLANE_RESIDUAL_MM = 0.15     # ... or whose points stay this close to one plane: the reconstruction
                             # engine's own fit band (recon.section.MERGE_TOL_DEFAULT), so a scan's
                             # noise does not turn every face into "freeform"
SMOOTH_GATE_DEG = 45.0       # a facet's normal is averaged with neighbours facing its way within
                             # this turn (half a right angle): a sliver at a junction has a normal
                             # its size cannot support, and one facet's noise is one facet's noise
SEW_TOL = 0.05               # mm: mirrors cad.imported.SEW_TOLERANCE (fit residuals absorb it)
AXIS_PERPENDICULAR_SIN = math.sin(math.radians(8.0))   # facet normals of a cylinder are perpendicular to its axis
OUTLIER_AREA_SHARE = 0.05    # ... judged by area: this share of a band may disagree (one noisy facet
                             # rejected the whole hole, which then fell through to the sphere fit)


# ---------------------------------------------------------------------------
# small linear algebra (pure)


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; ``None`` when singular."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            f = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= f * a[col][c]
    out = [0.0] * n
    for r in range(n - 1, -1, -1):
        out[r] = (a[r][n] - sum(a[r][c] * out[c] for c in range(r + 1, n))) / a[r][r]
    return out


def _jacobi_eigen(matrix: list[list[float]]) -> list[tuple[float, list[float]]]:
    """Eigen pairs of a symmetric 3x3 matrix (Jacobi rotations), ascending by value."""
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(30):
        off = sum(a[i][j] ** 2 for i in range(3) for j in range(3) if i != j)
        if off < 1e-18:
            break
        for p in range(2):
            for q in range(p + 1, 3):
                if abs(a[p][q]) < 1e-15:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c, s = 1.0 / math.sqrt(t * t + 1.0), t / math.sqrt(t * t + 1.0)
                for k in range(3):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(3):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(3):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    pairs = [(a[i][i], [v[0][i], v[1][i], v[2][i]]) for i in range(3)]
    return sorted(pairs, key=lambda p: p[0])


def _fit_circle(pts: list[tuple[float, float]], weights: list[float]) -> tuple[float, float, float] | None:
    """Area-weighted algebraic (Kasa) circle fit; ``(cx, cy, r)`` or None when degenerate."""
    sw = sum(weights)
    if sw <= 0 or len(pts) < 3:
        return None
    sx = sum(w * x for (x, _y), w in zip(pts, weights))
    sy = sum(w * y for (_x, y), w in zip(pts, weights))
    sxx = sum(w * x * x for (x, _y), w in zip(pts, weights))
    syy = sum(w * y * y for (_x, y), w in zip(pts, weights))
    sxy = sum(w * x * y for (x, y), w in zip(pts, weights))
    sxz = sum(w * x * (x * x + y * y) for (x, y), w in zip(pts, weights))
    syz = sum(w * y * (x * x + y * y) for (x, y), w in zip(pts, weights))
    szz = sum(w * (x * x + y * y) for (x, y), w in zip(pts, weights))
    # minimize sum w (x^2 + y^2 + D x + E y + F)^2 -> D sxx + E sxy + F sx = -sxz ...
    sol = _solve([[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, sw]], [-sxz, -syz, -szz])
    if sol is None:
        return None
    d, e, _f = sol
    cx, cy = -d / 2.0, -e / 2.0
    r2 = cx * cx + cy * cy - _f
    if r2 <= 1e-12:
        return None
    return cx, cy, math.sqrt(r2)


def _fit_sphere_center(pts: list[tuple[float, float, float]], normals: list[tuple[float, float, float]],
                       weights: list[float]) -> tuple[tuple[float, float, float], float] | None:
    """``(center, radius)`` of a sphere from ``p = c + r * n`` (least squares); None when degenerate."""
    # normal equations for (cx, cy, cz, r): sum w (p - r n) = c ... build via accumulation
    sw = sum(weights)
    if sw <= 0 or len(pts) < 4:
        return None
    acc = [[0.0] * 5 for _ in range(4)]
    for (x, y, z), (nx, ny, nz), w in zip(pts, normals, weights):
        # rows of the design: [1,0,0,-nx]*w -> c - r n = p ; accumulate A^T A and A^T p
        coeffs = ((1.0, 0.0, 0.0, -nx, x), (0.0, 1.0, 0.0, -ny, y), (0.0, 0.0, 1.0, -nz, z))
        for row in coeffs:
            for i in range(4):
                for j in range(4):
                    acc[i][j] += w * row[i] * row[j]
                acc[i][4] += w * row[i] * row[4]
    sol = _solve([row[:4] for row in acc], [row[4] for row in acc])
    if sol is None:
        return None
    return (sol[0], sol[1], sol[2]), sol[3]


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(a) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _norm(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-12 else None


def _canonical(v):
    """The direction with a stable sign (first nonzero component positive)."""
    for c in v:
        if abs(c) > 1e-9:
            return tuple(-x if c < 0 else x for x in v)
    return tuple(v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


# ---------------------------------------------------------------------------
# tessellation recognition (pure: plain arrays in, facts out)


def _facet_frames(points, facets, flip: bool):
    """``[(normal, area, centroid)]`` per facet (``flip`` inverts the winding's normals)."""
    out = []
    for i, j, k in facets:
        a, b, c = points[i], points[j], points[k]
        n = _cross(_sub(b, a), _sub(c, a))
        area = 0.5 * _norm(n)
        if flip:
            n = (-n[0], -n[1], -n[2])
        unit = _unit(n) if area > 0 else (0.0, 0.0, 0.0)
        centroid = tuple((a[d] + b[d] + c[d]) / 3.0 for d in range(3))
        out.append((unit, area, centroid))
    return out


def _adjacency(facets) -> list[list[int]]:
    """Facet neighbour lists through shared edges (a shared edge = two shared vertex indices)."""
    edge_map: dict[tuple, list[int]] = {}
    for idx, (i, j, k) in enumerate(facets):
        for a, b in ((i, j), (j, k), (k, i)):
            edge_map.setdefault((min(a, b), max(a, b)), []).append(idx)
    neighbours: list[list[int]] = [[] for _ in facets]
    for owners in edge_map.values():
        if len(owners) == 2:
            a, b = owners
            neighbours[a].append(b)
            neighbours[b].append(a)
    return neighbours


def _turn_deg(na, nb) -> float:
    c = max(-1.0, min(1.0, _dot(na, nb)))
    return math.degrees(math.acos(c))


def _smoothed_frames(frames, neighbours) -> list[tuple]:
    """Each facet's normal replaced by the area-weighted mean of its own and of the
    neighbours that face its way (within ``SMOOTH_GATE_DEG``). A sliver where two
    surfaces meet has a normal its size cannot support: raw, it lies within
    ``MAX_TURN_DEG`` of BOTH surfaces and lets one region grow from a face into its
    hole walls (measured: 0.01 mm of noise on a bracket merged 73 % of the part into
    one region). Smoothed, it belongs to one side only. A facet on a right-angle edge
    is not pulled across it, and a cylinder band's facets keep their radial spread."""
    gate = math.cos(math.radians(SMOOTH_GATE_DEG))
    out = []
    for f, (_n, area, centroid) in enumerate(frames):
        own = frames[f][0]
        sx = sy = sz = 0.0
        for g in [f] + neighbours[f]:
            n_g, a_g, _c = frames[g]
            if _dot(own, n_g) >= gate:
                sx += a_g * n_g[0]
                sy += a_g * n_g[1]
                sz += a_g * n_g[2]
        out.append((_unit((sx, sy, sz)) or own, area, centroid))
    return out


def _regions(points, facets, flip: bool) -> list[dict]:
    """Grow contiguous facet regions across normal turns up to ``MAX_TURN_DEG``.
    Every region is returned (small ones too): whether a small region is noise is
    decided after it is classified — a hole's wall is a sliver of the area on a large
    plate, and dropping it first is how small holes went unreported."""
    neighbours = _adjacency(facets)
    frames = _smoothed_frames(_facet_frames(points, facets, flip), neighbours)
    seen = [False] * len(facets)
    cos_limit = math.cos(math.radians(MAX_TURN_DEG))
    out = []
    for seed in range(len(facets)):
        if seen[seed]:
            continue
        seen[seed] = True
        stack, members = [seed], [seed]
        while stack:
            f = stack.pop()
            for g in neighbours[f]:
                if not seen[g] and _dot(frames[f][0], frames[g][0]) >= cos_limit:
                    seen[g] = True
                    members.append(g)
                    stack.append(g)
        area = sum(frames[m][1] for m in members)
        out.append({"members": members, "area": area, "frames": frames})
    return out


def _plane_residual(centroids, areas, normal=None) -> float:
    """Area-weighted RMS distance of points from their plane (the given normal, or the
    least-squares one)."""
    total = sum(areas) or 1.0
    c = [sum(p[d] * w for p, w in zip(centroids, areas)) / total for d in range(3)]
    if normal is None:
        acc = [[0.0] * 3 for _ in range(3)]
        for p, w in zip(centroids, areas):
            q = _sub(p, c)
            for i in range(3):
                for j in range(3):
                    acc[i][j] += w * q[i] * q[j]
        normal = _unit(_jacobi_eigen(acc)[0][1])
        if normal is None:
            return 0.0
    return math.sqrt(sum(w * _dot(_sub(p, c), normal) ** 2
                         for p, w in zip(centroids, areas)) / total)


def _classify_region(region: dict, points, facets) -> dict:
    """One surveyed region: plane / cylinder / sphere / freeform with its measured numbers."""
    frames = region["frames"]
    members = region["members"]
    normals = [frames[m][0] for m in members]
    areas = [frames[m][1] for m in members]
    centroids = [frames[m][2] for m in members]

    # area-weighted normal spread
    acc = [[0.0] * 3 for _ in range(3)]
    mean = [0.0, 0.0, 0.0]
    for n, w in zip(normals, areas):
        for i in range(3):
            mean[i] += w * n[i]
            for j in range(3):
                acc[i][j] += w * n[i] * n[j]
    total_w = sum(areas) or 1.0
    mean = [m / total_w for m in mean]
    if _norm(mean) <= 1e-9:
        # the normals cancel (a full band around an axis): the opposite of a plane
        spread_deg = 180.0
    else:
        spread_deg = max(_turn_deg(n, mean) for n in normals)
    kind = {"kind": "freeform", "area": region["area"]}
    n = _unit(mean) if _norm(mean) > 1e-9 else None
    # a face is flat when its normals agree, or when noise is all that scatters them:
    # its points then still lie within the fit band of one plane
    flat = n is not None and _plane_residual(centroids, areas, n) <= PLANE_RESIDUAL_MM
    if spread_deg <= PLANE_SPREAD_DEG or flat:
        if n is None:            # degenerate facets only (all normals zero): nothing to read
            return kind
        offsets = [_dot(p, n) for p in centroids]
        kind.update({"kind": "plane", "normal": list(n),
                     "offset": sum(o * w for o, w in zip(offsets, areas)) / total_w})
        return kind

    # cylinder: the normals live on a great circle -> the smallest-eigenvalue axis of the
    # normal covariance; the points then fit one circle about that axis
    eigen = _jacobi_eigen(acc)
    axis = _unit(eigen[0][1]) if _norm(eigen[0][1]) > 1e-9 else None
    # a cylinder's normals are perpendicular to its axis. Judged by area, not by every
    # facet: on a scan one noisy facet in a band rejected the whole hole, which then fell
    # through to the sphere fit; up to OUTLIER_AREA_SHARE of the area may disagree
    outlier_share = (sum(w for n, w in zip(normals, areas)
                         if abs(_dot(n, axis)) > AXIS_PERPENDICULAR_SIN) / total_w
                     if axis is not None else 1.0)
    if axis is not None and outlier_share <= OUTLIER_AREA_SHARE:
        ref = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0)
        u = _unit(_sub(ref, tuple(axis[d] * _dot(ref, axis) for d in range(3))))
        v = _cross(axis, u)
        pts2 = [(_dot(p, u), _dot(p, v)) for p in centroids]
        fit = _fit_circle(pts2, areas)
        if fit is not None:
            cx, cy, radius = fit
            residuals = [abs(math.hypot(x - cx, y - cy) - radius) for x, y in pts2]
            rms = math.sqrt(sum(w * r * r for r, w in zip(residuals, areas)) / total_w)
            # the circle must explain the points better than their best plane does. A face
            # that is not flat only because of noise "fits" a circle thousands of mm across
            # with a tiny residual (2 % of that radius is generous), and was reported as a
            # giant hole; a real cylinder's points leave any plane far further than that
            # ... and the region must be at least as non-flat as the band that would have
            # made it a plane: noise alone scatters a small patch of a face, and that patch
            # then "fits" a small circle
            if (rms <= max(2.0 * SEW_TOL, 0.02 * radius)
                    and max(2.0 * rms, PLANE_RESIDUAL_MM) <= _plane_residual(centroids, areas)):
                center_on_axis = tuple(u[d] * cx + v[d] * cy for d in range(3))
                ts = [_dot(points[i], axis) for m in members for i in facets[m]]
                t0, t1 = min(ts), max(ts)
                center_mid = tuple(center_on_axis[d] + axis[d] * (t0 + t1) / 2.0 for d in range(3))
                # angular coverage: the fan the region's normals sweep about the axis
                angles = [math.atan2(_dot(_sub(p, center_on_axis), v), _dot(_sub(p, center_on_axis), u))
                          for p in centroids]
                coverage = (max(angles) - min(angles)) / (2.0 * math.pi)
                concave = None
                if region.get("oriented"):
                    radial = [_sub(p, center_on_axis) for p in centroids]
                    dots = [_dot(n, _unit(r) if _norm(r) > 1e-9 else (0, 0, 1))
                            for n, r in zip(normals, radial)]
                    concave = sum(w * d for d, w in zip(dots, areas)) < 0
                kind.update({"kind": "cylinder", "radius": radius, "diameter": 2.0 * radius,
                             "axis": list(_canonical(axis)), "center": list(center_mid),
                             "extent": [t0, t1], "depth": t1 - t0, "concave": concave,
                             "angular_coverage": round(max(0.0, min(1.0, coverage)), 3),
                             "fit_rms_mm": round(rms, 4)})
                return kind

    # sphere: p = c + r n gives a first center; a real sphere then has every centroid at
    # ONE distance from it (the tessellation pulls centroids slightly inside, consistently)
    fit = _fit_sphere_center(centroids, normals, areas)
    if fit is not None:
        center, _r0 = fit
        radii = [_norm(_sub(p, center)) for p in centroids]
        r_bar = sum(w * v for v, w in zip(radii, areas)) / total_w
        rms = math.sqrt(sum(w * (v - r_bar) ** 2 for v, w in zip(radii, areas)) / total_w)
        if r_bar > 1e-6 and rms <= max(0.5, 0.05 * r_bar):
            kind.update({"kind": "sphere", "radius": r_bar, "center": list(center)})
            return kind
    return kind


def _merge_planes(regions: list[dict]) -> list[dict]:
    """Plane regions of (nearly) the same normal and offset are one physical face."""
    out: list[dict] = []
    for r in regions:
        if r["kind"] != "plane":
            continue
        n = _canonical(r["normal"])
        hit = next((p for p in out
                    if _turn_deg(n, _canonical(p["normal"])) <= 1.0
                    and abs(r["offset"] - p["offset"]) <= 4.0 * SEW_TOL), None)
        if hit is None:
            out.append(dict(r))
        else:
            hit["area"] += r["area"]
    return out


def _group_cylinders(regions: list[dict], tol_axis: float = 1e-3, tol_r: float = 1e-3) -> list[dict]:
    """Coaxial same-radius cylinder regions (the strips one surface was split into) become one."""
    out: list[dict] = []
    for r in regions:
        if r["kind"] != "cylinder":
            continue
        axis = _canonical(r["axis"])
        hit = None
        for c in out:
            if abs(r["radius"] - c["radius"]) > max(tol_r, 0.01 * r["radius"]):
                continue
            d = _sub(r["center"], c["center"])
            if _norm(_cross(axis, _canonical(c["axis"]))) <= 1e-2 \
                    and _norm(_cross(axis, d if _norm(d) > 1e-12 else (1.0, 0.0, 0.0))) <= max(tol_axis, 0.05):
                hit = c
                break
        if hit is None:
            out.append(dict(r))
        else:
            hit["area"] += r["area"]
            e0, e1 = hit["extent"], r["extent"]
            hit["extent"] = [min(e0[0], e1[0]), max(e0[1], e1[1])]
            hit["depth"] = hit["extent"][1] - hit["extent"][0]
            hit["angular_coverage"] = max(hit.get("angular_coverage", 0.0), r.get("angular_coverage", 0.0))
    return out


def recognize(points, facets, *, oriented: bool) -> dict:
    """The surface survey of a triangle soup: ``{"face_count", "surface_kinds", "planes",
    "cylinders", "warnings"}`` (pure; ``oriented`` says the winding is outward, which the
    concave / convex verdict needs)."""
    warnings: list[str] = []
    if len(facets) > MAX_FACETS:
        return {"face_count": 0, "surface_kinds": {}, "planes": [], "cylinders": [],
                "warnings": [f"{len(facets):,} facets is beyond the {MAX_FACETS:,} the survey reads; "
                             "only the bounding box and volume are measured"]}
    grown = _regions(points, facets, flip=False)
    total = sum(r["area"] for r in grown) or 1.0
    classified = []
    for region in grown:
        region["oriented"] = oriented
        kind = _classify_region(region, points, facets)
        # a small region is noise — unless it is a cylinder: the wall of a small hole is
        # a sliver of the area on a large plate, and was dropped before it could be read
        if region["area"] >= MIN_REGION_SHARE * total or kind["kind"] == "cylinder":
            classified.append(kind)
    planes = sorted(_merge_planes(classified), key=lambda p: -p["area"])[:16]
    cylinders = sorted(_group_cylinders(classified), key=lambda c: -c["area"])
    kinds: dict[str, int] = {}
    for r in classified:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    # the share of the surface the vocabulary above does not cover (freeform, or a region
    # that noise merged across an edge): whatever holes or bosses lie inside it are NOT
    # reported, so the text must not let "no hole listed" read as "the part has no hole"
    kept = sum(r["area"] for r in classified) or 1.0
    unsurveyed = sum(r["area"] for r in classified if r["kind"] not in ("plane", "cylinder"))
    return {"face_count": len(classified), "surface_kinds": kinds,
            "planes": planes, "cylinders": cylinders, "warnings": warnings,
            "unsurveyed_area_pct": round(100.0 * unsurveyed / kept, 1)}


# ---------------------------------------------------------------------------
# mesh and B-rep entry points


def survey_mesh(mesh) -> dict:
    """Survey a ``Mesh.Mesh``: exact bbox / volume / facet facts plus the recognized
    surface vocabulary. Never raises; sewing is not needed (recognition reads the
    triangles), so an OPEN mesh still gets its planes and cylinders — only its volume
    is withheld (an open mesh encloses nothing) and ``concave`` stays ``None`` when the
    winding cannot be trusted."""
    warnings: list[str] = []
    count = int(mesh.CountFacets)
    bb = mesh.BoundBox
    closed = False
    try:
        closed = count > 0 and bool(mesh.isSolid())   # an empty mesh claims isSolid too
    except Exception:  # noqa: BLE001 — an exotic mesh state is reported, not raised
        closed = False
    volume = None
    if closed:
        try:
            volume = abs(float(mesh.Volume))
        except Exception:  # noqa: BLE001
            volume = None
    points_raw, facets = mesh.Topology
    points = [(p.x, p.y, p.z) for p in points_raw]
    oriented = False
    if closed and volume:
        try:
            oriented = float(mesh.Volume) > 0
        except Exception:  # noqa: BLE001
            oriented = False
    surface = recognize(points, facets, oriented=oriented) if count else \
        {"face_count": 0, "surface_kinds": {}, "planes": [], "cylinders": [], "warnings": []}
    if not oriented:
        for c in surface["cylinders"]:
            c["concave"] = None
        if count and not closed:
            warnings.append("the mesh is not closed (open edges): the volume is not reported and "
                            "concave/convex verdicts need a closed, outward-wound mesh")
    return {"facets": count, "closed": closed, "oriented": oriented,
            "bbox": {"min": [bb.XMin, bb.YMin, bb.ZMin], "max": [bb.XMax, bb.YMax, bb.ZMax],
                     "size": [bb.XLength, bb.YLength, bb.ZLength]},
            "volume": volume, "surface": surface, "warnings": warnings + surface.pop("warnings", [])}


def survey_solid(shape) -> dict:
    """Survey the ANALYTIC faces of a B-rep shape (an imported STEP): exact planes and
    cylinders, concave vs convex probed with ``isInside``. Same schema as the mesh survey's
    ``surface`` block plus ``valid``."""
    warnings: list[str] = []
    if shape is None or getattr(shape, "isNull", lambda: True)():
        return {"valid": False, "face_count": 0, "surface_kinds": {}, "planes": [],
                "cylinders": [], "warnings": ["the shape is empty"]}
    valid = True
    try:
        valid = bool(shape.isValid())
    except Exception:  # noqa: BLE001
        valid = False
    import FreeCAD as App  # local: the module itself imports no FreeCAD

    def axis_key(axis, center, radius):
        """(canonical axis, axial position, radial offset, radius): two cylinders sharing
        these are the same physical surface however the B-rep split it."""
        t = _dot(center, axis)
        perp = _sub(center, tuple(axis[d] * t for d in range(3)))
        return (tuple(round(c, 4) for c in axis),
                round(t, 3), tuple(round(c, 2) for c in perp), round(radius, 4))

    planes: dict[tuple, dict] = {}
    cylinders: dict[tuple, dict] = {}
    kinds: dict[str, int] = {}
    faces = list(getattr(shape, "Faces", []) or [])
    for face in faces:
        surf = face.Surface
        type_id = str(getattr(surf, "TypeId", "") or type(surf).__name__)
        kind = {"Part::GeomPlane": "plane", "Part::GeomCylinder": "cylinder",
                "Part::GeomSphere": "sphere", "Part::GeomCone": "cone",
                "Part::GeomToroid": "torus"}.get(type_id, type_id)
        kinds[kind] = kinds.get(kind, 0) + 1
        area = float(face.Area)
        if kind == "plane":
            pos = surf.Position
            ax = surf.Axis
            n = _unit((ax.x, ax.y, ax.z)) or (0.0, 0.0, 1.0)
            offset = _dot(n, (pos.x, pos.y, pos.z))
            key = (tuple(round(c, 3) for c in _canonical(n)), round(offset, 2))
            entry = planes.setdefault(key, {"kind": "plane", "normal": list(_canonical(n)),
                                            "offset": key[1], "area": 0.0})
            entry["area"] += area
        elif kind == "cylinder":
            axis = _unit((surf.Axis.x, surf.Axis.y, surf.Axis.z)) or (0.0, 0.0, 1.0)
            center = (surf.Center.x, surf.Center.y, surf.Center.z)
            radius = float(surf.Radius)
            concave = None
            verts = [(w.Point.x, w.Point.y, w.Point.z) for w in face.Vertexes]
            ts = [_dot(q, axis) for q in verts] or [_dot(center, axis)]
            try:
                # the centre of curvature tells the side: a hole's axis mid-point sits in
                # the void, a pin's inside the material. The face's mid-height is measured
                # FROM THE CYLINDER'S OWN AXIS POINT: added to it as an absolute height, a
                # step far from the origin was probed outside the part and read as a hole
                raw_axis = (surf.Axis.x, surf.Axis.y, surf.Axis.z)
                mid_height = (min(ts) + max(ts)) / 2.0 - _dot(center, axis)
                mid = tuple(center[d] + raw_axis[d] * mid_height for d in range(3))
                concave = not bool(shape.isInside(App.Vector(*mid), 1e-7, True))
            except Exception:  # noqa: BLE001 — the probe is evidence, not a requirement
                concave = None
            key = axis_key(_canonical(axis), center, radius)
            entry = cylinders.get(key)
            if entry is None:
                entry = cylinders[key] = {
                    "kind": "cylinder", "radius": radius, "diameter": 2.0 * radius,
                    "axis": list(_canonical(axis)), "center": list(center),
                    "extent": [min(ts), max(ts)], "depth": max(ts) - min(ts),
                    "concave": concave, "angular_coverage": None, "area": 0.0}
            else:
                e = entry["extent"]
                entry["extent"] = [min(e[0], min(ts)), max(e[1], max(ts))]
                entry["depth"] = entry["extent"][1] - entry["extent"][0]
                if entry["concave"] is None and concave is not None:
                    entry["concave"] = concave
            entry["area"] += area
    return {"valid": valid, "face_count": len(faces), "surface_kinds": kinds,
            "planes": sorted(planes.values(), key=lambda p: -p["area"])[:16],
            "cylinders": sorted(cylinders.values(), key=lambda c: -c["area"]),
            "warnings": warnings}


# ---------------------------------------------------------------------------
# model-facing text


def _mm(v: float) -> str:
    return f"{v:.3g}"


def render_text(survey: dict) -> str:
    """The bounded report of a mesh survey for the model: extents, then the recognized
    holes / pins and planes with the evidence a plan's parameters need."""
    if not survey or not survey.get("surface"):
        return ""
    lines = []
    surface = survey["surface"]
    holes = [c for c in surface["cylinders"] if c.get("concave")]
    pins = [c for c in surface["cylinders"] if c.get("concave") is False]
    unknown = [c for c in surface["cylinders"] if c.get("concave") is None]
    for label, group in (("hole", holes), ("boss/pin", pins)):
        for c in sorted(group, key=lambda c: -c["diameter"])[:8]:
            where = ", ".join(f"{_mm(v)}" for v in c["center"])
            lines.append(f"{label}: diameter {_mm(c['diameter'])} mm, depth {_mm(c['depth'])} mm, "
                         f"axis {tuple(round(a, 2) for a in c['axis'])}, at ({where}) mm"
                         + ("" if c.get("angular_coverage") in (None, 1.0)
                            else f", sweep {round(100 * c['angular_coverage'])}% of a turn"))
    if unknown:
        diameters = sorted({round(c["diameter"], 2) for c in unknown}, reverse=True)[:6]
        lines.append(f"cylinders of unknown side (open or inward-wound mesh): diameters "
                     f"{', '.join(_mm(d) for d in diameters)} mm")
    for p in surface["planes"][:6]:
        lines.append(f"plane: normal {tuple(round(a, 2) for a in p['normal'])}, offset {_mm(p['offset'])} mm "
                     f"from the origin, area share {round(100 * p['area'] / max(1e-9, sum(x['area'] for x in surface['planes'])))}%")
    not_read = surface.get("unsurveyed_area_pct") or 0
    if not_read:
        lines.append(f"not surveyed: {not_read}% of the surface (freeform, or regions that noise merged "
                     "across their edges). Holes and bosses inside it are not listed above: no hole "
                     "listed is not evidence that the part has none.")
    for w in (survey.get("warnings") or [])[:4]:
        lines.append(f"Warning: {w}.")
    return "\n".join(lines)
