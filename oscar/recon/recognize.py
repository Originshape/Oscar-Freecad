"""Geometry recognition for the reconstruction engine (M3/M4): region growing over mesh
adjacency with an adaptive dihedral threshold (the largest gap in the angle histogram — the
tessellated-CAD signature), then a plane or cylinder least-squares fit per region, hole
detection (full-turn cylinder walls between parallel planes) and an evidence-based part
classification. Pure numpy on mesh topology/normals; no document access, any thread.

Plane/cylinder acceptance gates are NOISE-ADAPTIVE (Stage 1 / K1): derived from the mesh's
estimated noise sigma and mean edge length (recon.quality), so a noisy scan widens its own
gates and a clean tessellation tightens them; the module constants below remain only as the
fallback when a quality estimate is unavailable.

Honest scope (v2, G2): planes, cylinders (Kasa seed + geometric LM refinement — no rim
bias), cones, spheres and tori via recon.fit with anti-fake-primitive gates (nested F-tests
at alpha=1e-4 with effective sample counts, plus >=2-sigma geometric separation). Holes still
require full-turn cylinder walls; rectangle-profile prismatic route evidence as before.
Everything unrecognized is reported as such — never force-labeled.
"""

import math

import numpy as np

from . import fit as _fit
from . import segmerge as _seg
from . import provenance as _provenance
from . import quality as _quality

MIN_REGION_FACETS = 2               # a 2-triangle planar side face is still a plane
PLANE_RMS_MM = 0.05
CYLINDER_RMS_MM = 0.30            # curved walls carry tessellation chord error into the fit
HOLE_RADIUS_MM = (1.5, 20.0)
HOLE_MIN_TURN_DEG = 300.0
DOMINANT_NORMAL_DEG = 20.0
# G7 dressings (fillet / chamfer) — OriginShape's verified thresholds
FILLET_TANGENCY_DEG = 20.0   # junction between the strip and its neighbour <= 20° = tangent
FILLET_MAX_SPAN_DEG = 150.0  # a fillet wraps at most 150° of its arc
DRESSING_AREA_SHARE = 0.05   # a dressing strip is NARROW: <= 5% of the surface each
CHAMFER_ANGLE_RANGE = (25.0, 90.0)  # a chamfer's junction with BOTH neighbours sits at a
                                   # distinct intermediate angle (OriginShape tangency limit ~20 deg)


def _topology(mesh):
    from ._arrays import mesh_arrays

    return mesh_arrays(mesh)


def _facet_normals(mesh, count):
    """FreeCAD's own facet normals. (A vectorized winding-cross version was
    tried and REVERTED: its tiny numeric differences on marginal facets moved
    the adaptive gates enough to flip benchmark outcomes — measured 2026-11-16,
    case 06 chamfer 0.08 → 1.63 mm.)"""
    normals = np.zeros((count, 3), dtype=float)
    for i, facet in enumerate(mesh.Facets):
        n = facet.Normal
        normals[i] = (n.x, n.y, n.z)
    return normals


def _adjacency(fac):
    """Edge-sharing facet pairs, as two index arrays."""
    seen: dict = {}
    a, b = [], []
    for i, tri in enumerate(fac):
        for x, y in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (x, y) if x < y else (y, x)
            if key in seen:
                a.append(seen[key])
                b.append(i)
            else:
                seen[key] = i
    return np.array(a, dtype=int), np.array(b, dtype=int)


def _adaptive_threshold_deg(normals, a, b) -> float:
    """Midpoint of the largest gap in the adjacency dihedral histogram, clamped to
    [10°, 45°] — within-surface strips sit low, feature edges high (MeshToFeatures' idea,
    reimplemented)."""
    if len(a) == 0:
        return 25.0
    dots = np.clip(np.einsum("ij,ij->i", normals[a], normals[b]), -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    angles = angles[(angles >= 3.0) & (angles <= 70.0)]
    if angles.size < 4:
        return 25.0
    angles.sort()
    gaps = np.diff(angles)
    k = int(np.argmax(gaps))
    thr = float((angles[k] + angles[k + 1]) / 2.0)
    return min(45.0, max(10.0, thr))


def _region_adjacency(groups, fac):
    """Region index pairs that share at least one mesh edge (G12 merge candidates)."""
    owner = {}
    for gi, g in enumerate(groups):
        for f in g:
            owner[f] = gi
    pairs = set()
    vgroups: dict = {}
    for gi, g in enumerate(groups):
        for f in g:
            for v in fac[f]:
                vgroups.setdefault(v, set()).add(gi)
    for gi, g in enumerate(groups):
        vs = set()
        for f in g:
            vs.update(int(v) for v in fac[f])
        neigh = set()
        for v in vs:
            for gj in vgroups.get(v, ()):
                if gj != gi:
                    neigh.add(gj)
        for gj in neigh:
            pairs.add((min(gi, gj), max(gi, gj)))
    return sorted(pairs)


def _regions(fac, normals, threshold_deg):
    """Union-find over adjacency pairs below the dihedral threshold."""
    parent = list(range(len(fac)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    thr = math.cos(math.radians(threshold_deg))
    a, b = _adjacency(fac)
    for i, j in zip(a, b):
        if float(np.dot(normals[i], normals[j])) >= thr:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    groups: dict = {}
    for i in range(len(fac)):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= MIN_REGION_FACETS], \
           [g for g in groups.values() if len(g) < MIN_REGION_FACETS]


def _tri_areas(pts, fac, idx):
    p = pts[fac[idx]]
    return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)


def _fit_plane(P):
    center = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - center, full_matrices=False)
    normal = vt[-1]
    rms = float(np.sqrt(((P - center) @ normal) ** 2).mean())
    return normal, center, rms


def _canonical(normal):
    """Normal flipped into a canonical half-space so parallel faces compare equal."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    for k in range(3):
        if abs(n[k]) > 1e-9:
            if n[k] < 0:
                n = -n
            break
    return n


def survey(mesh, diag: "dict | None" = None, quality: "dict | None" = None,
           provenance: "dict | None" = None) -> dict:
    """Recognize regions, holes and the part class of ``mesh``. Returns regions (plane /
    cylinder / unrecognized with areas, fits and rms), hole candidates, classification evidence,
    area coverage and the quality/tolerance basis the gates were derived from. Read-only on the
    mesh. ``quality`` may carry a precomputed recon.quality estimate; without one it is
    estimated here (and falls back to the module constants if that fails). ``provenance`` may
    carry a recon.provenance ledger: every region and hole is then tagged with the evidence
    class of its supporting facets (original / synthetic / mixed) — synthetic-supported
    features are flagged so they can never silently set a reconstructed dimension."""
    pts, fac = _topology(mesh)
    normals = _facet_normals(mesh, len(fac))
    if quality is None:
        try:
            quality = _quality.estimate_tolerances(mesh)
        except Exception:  # noqa: BLE001 - a failed estimate must not stop recognition
            quality = {}
    plane_tol = float(quality.get("plane_rms_mm") or PLANE_RMS_MM)
    cylinder_tol = float(quality.get("cylinder_rms_mm") or CYLINDER_RMS_MM)
    all_keys = _provenance.facet_keys(mesh) if provenance else None
    threshold = _adaptive_threshold_deg(normals, *_adjacency(fac))
    groups, small = _regions(fac, normals, threshold)
    # G12 (a): absorb sub-minimal groups into the best adjacent qualifying region
    groups, small = _seg.absorb_small(groups, small, normals, fac, pts, threshold)
    # G12 (b)+(c): union-fit merge of adjacent compatible regions (incl. cone upgrade).
    # A cheap first-fit per region provides the kind hints the merge needs.
    sigma_val = float(quality.get("sigma_mm") or 0.0)
    hints, region_normals = [], []
    for g in groups:
        Pg = pts[fac[g]].reshape(-1, 3)
        Ng = np.repeat(normals[g], 3, axis=0)[:len(Pg)]
        hints.append(_fit.classify(Pg, Ng, cylinder_tol, sigma_val)["kind"])
        region_normals.append(normals[g].mean(axis=0))
    adjacency_pairs = _region_adjacency(groups, fac)
    groups, hints, merge_log = _seg.merge_adjacent(groups, hints, fac, pts, normals,
                                                   plane_tol, cylinder_tol, sigma_val,
                                                   adjacency_pairs,
                                                   region_normals=region_normals)
    regions, unrecognized_area = [], 0.0
    planes, cylinders = [], []
    for idx in groups:
        P = pts[fac[idx]].reshape(-1, 3)
        area = float(_tri_areas(pts, fac, idx).sum())
        N = normals[idx]
        mean_n = N.mean(axis=0)
        evidence = None
        if all_keys is not None:
            sample = [all_keys[i] for i in idx[:20]]
            evidence = _provenance.evidence_class(sample, provenance)
        normal, center, rms = _fit_plane(P)
        if rms <= plane_tol and abs(float(mean_n @ normal)) > math.cos(math.radians(DOMINANT_NORMAL_DEG)):
            entry = {"_idx": list(idx), "type": "plane", "normal": _canonical(normal).tolist(),
                     "point": center.tolist(), "area_mm2": round(area, 3),
                     "rms_mm": round(rms, 5), "facets": len(idx)}
            if evidence and evidence != "original":
                entry["evidence"] = evidence
            regions.append(entry)
            planes.append(entry)
            continue
        sigma = float(quality.get("sigma_mm") or 0.0)
        Nrep = np.repeat(N, 3, axis=0)[:len(P)]      # one normal per facet corner, as P
        got = _fit.classify(P, Nrep, cylinder_tol, sigma, plane_fit=(normal, center, rms))
        if got["kind"] == "cylinder":
            f = got["fit"]
            entry = {"_idx": list(idx), "type": "cylinder", "axis": _canonical(f["axis"]).tolist(),
                     "point": np.asarray(f["point"]).tolist(), "radius": round(f["radius"], 4),
                     "rms_mm": round(f["rms"], 5), "turn_deg": round(f["turn_deg"], 1),
                     "axial_span": round(f["axial_span"], 3),
                     "axial_mid": round(float((P @ _canonical(f["axis"])).mean()), 4),
                     "area_mm2": round(area, 3), "facets": len(idx)}
            if evidence and evidence != "original":
                entry["evidence"] = evidence
            regions.append(entry)
            cylinders.append(entry)
            continue
        if got["kind"] == "cone":
            f = got["fit"]
            entry = {"_idx": list(idx), "type": "cone", "axis": _canonical(f["axis"]).tolist(),
                     "apex": np.asarray(f["apex"]).tolist(),
                     "half_angle_deg": round(math.degrees(f["half_angle"]), 3),
                     "radius_min": round(f["radius_min"], 4),
                     "radius_max": round(f["radius_max"], 4),
                     "rms_mm": round(f["rms"], 5),
                     "area_mm2": round(area, 3), "facets": len(idx)}
        elif got["kind"] == "sphere":
            f = got["fit"]
            entry = {"_idx": list(idx), "type": "sphere", "center": np.asarray(f["center"]).tolist(),
                     "radius": round(f["radius"], 4), "rms_mm": round(f["rms"], 5),
                     "area_mm2": round(area, 3), "facets": len(idx)}
        elif got["kind"] == "torus":
            f = got["fit"]
            entry = {"_idx": list(idx), "type": "torus", "axis": _canonical(f["axis"]).tolist(),
                     "center": np.asarray(f["center"]).tolist(),
                     "major_radius": round(f["major_radius"], 4),
                     "minor_radius": round(f["minor_radius"], 4),
                     "rms_mm": round(f["rms"], 5),
                     "area_mm2": round(area, 3), "facets": len(idx)}
        else:
            unrecognized_area += area
            entry = {"_idx": list(idx), "type": "unrecognized", "area_mm2": round(area, 3), "facets": len(idx)}
        if evidence and evidence != "original":
            entry["evidence"] = evidence
        regions.append(entry)
    # G7: tangent-junction split — a fillet merges with its planes under ANY dihedral
    # threshold (their junction sits at the tessellation dihedral), so re-split each
    # unrecognized blob by facet-normal direction and refit the pieces through the gates
    split_regions = []
    for r in regions:
        if r.get("type") == "unrecognized" and r.get("_idx"):
            subs = _split_tangent_region(r["_idx"], pts, fac, normals, plane_tol,
                                         sigma_val)
            if subs:
                for sub in subs:
                    if sub["type"] == "plane":
                        planes.append(sub)
                    elif sub["type"] == "cylinder":
                        cylinders.append(sub)
                    else:
                        unrecognized_area += sub["area_mm2"]
                unrecognized_area -= r["area_mm2"]
                split_regions.extend(subs)
                continue
        split_regions.append(r)
    regions = split_regions
    for idx in small:
        unrecognized_area += float(_tri_areas(pts, fac, idx).sum())
    total_area = float(mesh.Area) or 1.0

    region_pts = {id(r): pts[fac[r["_idx"]]].reshape(-1, 3) for r in regions
                  if "_idx" in r}
    holes = _hole_features(cylinders, regions, planes, region_pts, mesh=mesh)
    dressings = _dressings(regions, region_pts, pts, fac, total_area)   # G7

    plane_area = sum(p["area_mm2"] for p in planes)
    cyl_area = sum(c["area_mm2"] for c in cylinders)
    parallel_pairs = _parallel_plane_pairs(planes)
    sphere_area = sum(r["area_mm2"] for r in regions if r["type"] == "sphere")
    classification = _classify(diag or {}, plane_area, cyl_area, unrecognized_area,
                               total_area, parallel_pairs, cylinders, holes,
                               sphere_area=sphere_area)
    size = (diag or {}).get("size_mm") or []
    real_holes = [h for h in holes
                  if not size or h["radius"] < 0.45 * max(min(size), 1e-9)]
    classification["evidence"]["real_holes"] = len(real_holes)
    for r in regions:
        r.pop("_idx", None)
    return {"regions": regions, "holes": holes,
            "parallel_planes": parallel_pairs,
            "classification": classification,
            "coverage": {"planes": round(plane_area / total_area, 3),
                         "cylinders": round(cyl_area / total_area, 3),
                         "unrecognized": round(unrecognized_area / total_area, 3)},
            "quality": {"sigma_mm": quality.get("sigma_mm"), "mean_edge_mm": quality.get("mean_edge_mm"),
                        "plane_rms_mm": plane_tol, "cylinder_rms_mm": cylinder_tol,
                        "engine": quality.get("engine", "")},
            "dressings": dressings,
            "merges": merge_log,
            "threshold_deg": round(threshold, 1)}


def _split_tangent_region(idx, pts, fac, normals, plane_tol, sigma):
    """G7 helper: re-split a region that region-growing merged across TANGENT junctions —
    a fillet meets its planes at roughly the tessellation dihedral (a few degrees), far
    below any dihedral threshold, so 'plane + fillet + plane' grows into one blob. Here
    the blob's facets are clustered by normal DIRECTION (a plane's facet normals are all
    parallel; a fillet's rotate continuously): each cluster that fits a plane becomes a
    plane sub-region, and the connected leftovers are refit as cylinders through the G2
    gates. Returns sub-region dicts with ``_idx`` (never invented geometry — only fits
    that pass their gates survive)."""
    out = []
    if len(idx) < 4:
        return out

    def _area(chosen):
        return round(float(_tri_areas(pts, fac, chosen).sum()), 3)

    # local curvature: a fillet facet's neighbours ALL turn (median adjacent dihedral
    # ~ the tessellation step, measured 1-3° on fine fillets); a plane facet's median
    # neighbour is flush at exactly 0° on a tessellated plane. The 1° gate is for clean
    # tessellations — on a noisy scan the plane fits fail anyway and the piece stays
    # unrecognized (honest) rather than mislabelled.
    a, b = _adjacency(fac[idx])
    turn = {}
    for i, j in zip(a.tolist(), b.tolist()):
        cosij = float(normals[idx[i]] @ normals[idx[j]])
        turn.setdefault(i, []).append(math.degrees(math.acos(min(1.0, max(-1.0, cosij)))))
        turn.setdefault(j, []).append(math.degrees(math.acos(min(1.0, max(-1.0, cosij)))))
    curved = {p for p in range(len(idx))
              if float(np.median(turn.get(p, [0.0]))) > 1.0}
    flat_pos = [p for p in range(len(idx)) if p not in curved]
    nf = normals[idx]
    order = sorted(flat_pos, key=lambda p: -nf[p][2])
    clusters: list[list] = []
    means: list = []
    for pos in order:
        n = nf[pos]
        placed = False
        for ci, m in enumerate(means):
            if float(n @ m) >= math.cos(math.radians(12.0)):
                clusters[ci].append(pos)
                means[ci] = m + (n - m) / len(clusters[ci])   # running mean
                placed = True
                break
        if not placed:
            clusters.append([pos])
            means.append(n.copy())
    leftover = [idx[p] for p in curved]
    plane_normals: list = []
    # second sweep (P1.1): a flat FACE can be mis-flagged wholesale 'curved'
    # when its giant triangles each border MANY tangent fillet chords (the
    # median-neighbour test tips over). Re-cluster the curved facets by normal
    # direction and accept only clusters that are both FLAT (the rms gate) and
    # LARGE (the area floor): a real face qualifies, a ~12-deg slice of a
    # fillet or sphere does not.
    claimed: set = set()
    if curved:
        ccl: list[list] = []
        cmean: list = []
        for p in sorted(curved, key=lambda q: -normals[idx[q]][2]):
            n = normals[idx[p]]
            hit = False
            for ci, m in enumerate(cmean):
                if float(n @ m) >= math.cos(math.radians(12.0)):
                    ccl[ci].append(p)
                    cmean[ci] = m + (n - m) / len(ccl[ci])
                    hit = True
                    break
            if not hit:
                ccl.append([p])
                cmean.append(n.copy())
        for cluster in ccl:
            chosen = [idx[p] for p in cluster]
            if _area(chosen) < 25.0:
                continue
            P = pts[fac[chosen]].reshape(-1, 3)
            normal, center, rms = _fit_plane(P)
            if rms <= plane_tol:
                nrm = _canonical(normal)
                out.append({"_idx": chosen, "type": "plane",
                            "normal": nrm.tolist(), "point": center.tolist(),
                            "rms_mm": rms, "facets": len(chosen),
                            "area_mm2": _area(chosen)})
                plane_normals.append(nrm)
                claimed.update(chosen)
        if claimed:
            leftover = [f for f in leftover if f not in claimed]
    for cluster in clusters:
        chosen = [idx[p] for p in cluster]
        P = pts[fac[chosen]].reshape(-1, 3)
        if len(P) < 6 and _area(chosen) < 25.0:
            leftover.extend(chosen)
            continue
        # >= 25 mm2 with < 6 points is a GIANT tessellation triangle (a flat
        # face often tessellates to 2-4 huge triangles): it is planar by
        # definition — fit it through the same rms gate as everything else
        normal, center, rms = _fit_plane(P)
        if rms <= plane_tol:
            nrm = _canonical(normal)
            out.append({"_idx": chosen, "type": "plane",
                        "normal": nrm.tolist(), "point": center.tolist(),
                        "rms_mm": rms, "facets": len(chosen),
                        "area_mm2": _area(chosen)})
            plane_normals.append(nrm)
        else:
            leftover.extend(chosen)
    if plane_normals:
        # planar faces tessellate with slightly non-coplanar fans (measured 1-5° median
        # turns) that glue themselves onto curved strips and poison the cylinder refits:
        # any curved-flagged facet whose normal matches a RECOVERED plane is fan noise
        pn = np.asarray(plane_normals, dtype=float)
        keep = [f for f in leftover
                if float(np.max(np.abs(normals[f] @ pn.T))) < math.cos(math.radians(10.0))]
        leftover = keep
    if leftover:
        # connected components of the leftover facets (shared vertex)
        parent = list(range(len(leftover)))
        vmap: dict = {}
        for i, f in enumerate(leftover):
            for v in fac[f]:
                vmap.setdefault(int(v), []).append(i)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for members in vmap.values():
            for i in range(len(members) - 1):
                a, b = find(members[i]), find(members[i + 1])
                if a != b:
                    parent[a] = b
        groups: dict = {}
        for i, f in enumerate(leftover):
            groups.setdefault(find(i), []).append(f)
        for chosen in groups.values():
            if len(chosen) < 4:
                # never silently dropped: an unexplained component is honest
                # unrecognized evidence (it feeds the coverage shares)
                out.append({"_idx": chosen, "type": "unrecognized",
                            "facets": len(chosen), "area_mm2": _area(chosen)})
                continue
            out.extend(_extract_cylinders(chosen, pts, fac, normals, plane_tol,
                                          sigma, _area, plane_normals))
    return out


def _split_by_gaps(piece, pts, fac, *, min_facets=30, depth=3):
    """Spatially bisect a facet piece at the biggest EMPTY RUN of its centroid histogram
    along a principal axis. Fillet strips are vertex-GLUED through the doubly-curved
    corner transitions, so connectivity alone cannot separate them; the strips are
    spatially distinct bands with real gaps between them. Returns the leaf pieces."""
    if depth <= 0 or len(piece) < 2 * min_facets:
        return [piece]
    centroids = np.array([pts[fac[f]].mean(axis=0) for f in piece])
    centred = centroids - centroids.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    best = None   # (gap_size, axis_idx, split_position, left, right)
    for axis_idx in range(2):
        t = centred @ vt[axis_idx]
        order = np.argsort(t)
        runs = np.diff(t[order])
        if len(runs) == 0:
            continue
        widest = int(np.argmax(runs))
        gap = float(runs[widest])
        median_span = float(np.ptp(t)) / max(len(t), 1) * 50.0
        if gap <= max(1e-6, median_span * 0.05):
            continue
        cut = float((t[order][widest] + t[order][widest + 1]) / 2.0)
        left = [f for f, tv in zip(piece, t) if tv <= cut]
        right = [f for f, tv in zip(piece, t) if tv > cut]
        if len(left) < min_facets or len(right) < min_facets:
            continue
        if best is None or gap > best[0]:
            best = (gap, axis_idx, cut, left, right)
    if best is None:
        return [piece]
    _gap, _axis_idx, _cut, left, right = best
    return (_split_by_gaps(left, pts, fac, min_facets=min_facets, depth=depth - 1)
            + _split_by_gaps(right, pts, fac, min_facets=min_facets, depth=depth - 1))


def _curved_subset(piece, fac, normals):
    """The CURVED facets of ``piece`` — median adjacent dihedral above the 1-degree clean-
    tessellation gate (the same discriminator the G7 splitter uses at blob level). Fan and
    transition triangles of a tessellated PLANE are flush (0-degree medians): separating
    them from a tangent-glued fillet strip is exactly what connectivity and spatial gaps
    cannot do (the strip touches the plane along its whole tangent line)."""
    a, b = _adjacency(fac[piece])
    turn = {}
    for i, j in zip(a.tolist(), b.tolist()):
        cosij = float(normals[piece[i]] @ normals[piece[j]])
        dihedral = math.degrees(math.acos(min(1.0, max(-1.0, cosij))))
        turn.setdefault(i, []).append(dihedral)
        turn.setdefault(j, []).append(dihedral)
    curved = {piece[p] for p in range(len(piece))
              if float(np.median(turn.get(p, [0.0]))) > 1.0}
    if not curved:
        return []
    # one facet ring of dilation: the fillet's TANGENT boundary rows are flush with the
    # plane (median turn ~0), so without them the extracted strip ends mid-arc and the
    # dressing tangency test measures 40-90 degrees against both neighbours. The ring
    # restores the tangent ends; classify's own gates still reject anything non-cylindrical.
    ring: dict = {}
    for f in piece:
        for v in fac[f]:
            ring.setdefault(int(v), []).append(f)
    out = set(curved)
    for f in curved:
        for v in fac[f]:
            out.update(ring.get(int(v), ()))
    return sorted(out)


def _emit_cylinder(out, member, f, pts, fac, normals, _area):
    axis_c = _canonical(f["axis"])
    P = pts[fac[member]].reshape(-1, 3)
    out.append({"_idx": member, "type": "cylinder",
                "axis": axis_c.tolist(),
                "point": np.asarray(f["point"]).tolist(),
                "radius": round(f["radius"], 4),
                "rms_mm": round(f["rms"], 5),
                "turn_deg": round(f["turn_deg"], 1),
                "axial_span": round(f["axial_span"], 3),
                "axial_mid": round(float((P @ axis_c).mean()), 4),
                "facets": len(member), "area_mm2": _area(member)})


def _facet_components(facet_ids, fac):
    """Connected components of ``facet_ids`` by shared vertex (union-find)."""
    parent = list(range(len(facet_ids)))
    vmap: dict = {}
    for i, f in enumerate(facet_ids):
        for v in fac[f]:
            vmap.setdefault(int(v), []).append(i)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for members in vmap.values():
        for i in range(len(members) - 1):
            a, b = find(members[i]), find(members[i + 1])
            if a != b:
                parent[a] = b
    groups: dict = {}
    for i, f in enumerate(facet_ids):
        groups.setdefault(find(i), []).append(f)
    return list(groups.values())


def _extract_cylinders(chosen, pts, fac, normals, plane_tol, sigma, _area,
                       plane_normals):
    """Pull every CYLINDER out of a mixed leftover component. Transition triangles with
    20-80-degree-inclined normals glue hole walls to their surroundings, so a
    whole-component fit fails; instead each RECOVERED plane's normal is tried as the
    axis (a drilled hole runs perpendicular to a major face), the axis-consistent
    facets (|n.axis| < 0.2) are fitted through the G2 gates, gross radial outliers
    (fillet mid-arc facets sharing the axis) are rejected once and the fit repeated.
    Consumed facets never refit; what remains is unrecognized, never guessed."""
    cyl_tol = max(0.30, 4.0 * plane_tol)
    out = []
    remaining = list(chosen)
    for n_p in plane_normals:
        axis = np.asarray(n_p, dtype=float)
        for _attempt in range(2):
            member = [f for f in remaining if abs(float(normals[f] @ axis)) < 0.2]
            if len(member) < 30:
                break
            P = pts[fac[member]].reshape(-1, 3)
            Nrep = np.repeat(normals[member], 3, axis=0)[:len(P)]
            got = _fit.classify(P, Nrep, cyl_tol, sigma, kinds=("cylinder",))
            if got["kind"] != "cylinder":
                break
            f = got["fit"]
            if _attempt == 0:
                # one outlier-rejection round: facets far off the fitted wall (fillet
                # mid-arc facets share the axis direction) leave, the rest refit
                ax = np.asarray(f["axis"], dtype=float)
                d = P - np.asarray(f["point"], dtype=float)
                radial = np.linalg.norm(d - np.outer(d @ ax, ax), axis=1)
                res = np.abs(radial - float(f["radius"]))
                keep = np.nonzero(res <= max(3.0 * float(f["rms"]),
                                             0.02 * float(f["radius"])))[0]
                if 0 < len(keep) < len(member) and len(keep) >= 30:
                    member = [member[k] for k in keep]
                    P = pts[fac[member]].reshape(-1, 3)
                    Nrep = np.repeat(normals[member], 3, axis=0)[:len(P)]
                    got = _fit.classify(P, Nrep, cyl_tol, sigma, kinds=("cylinder",))
                    if got["kind"] != "cylinder":
                        break
                    f = got["fit"]
            axis_c = _canonical(f["axis"])
            out.append({"_idx": member, "type": "cylinder",
                        "axis": axis_c.tolist(),
                        "point": np.asarray(f["point"]).tolist(),
                        "radius": round(f["radius"], 4),
                        "rms_mm": round(f["rms"], 5),
                        "turn_deg": round(f["turn_deg"], 1),
                        "axial_span": round(f["axial_span"], 3),
                        "axial_mid": round(float((P @ axis_c).mean()), 4),
                        "facets": len(member), "area_mm2": _area(member)})
            consumed = set(member)
            remaining = [f2 for f2 in remaining if f2 not in consumed]
            break
    if remaining:
        # E2 (G7): parallel fillet strips share the plane-normal axis direction, so the
        # axis-consistent member fit above mixes them into one un-fittable cloud. Fit
        # each CONNECTED piece on its own — a single strip is a clean partial cylinder
        # whose axis comes from the normal covariance (fit_cylinder is partial-arc
        # safe). Same classify gates (rms + the flat-impostor and shell checks): a weak
        # or non-cylindrical piece stays unrecognized, never guessed.
        consumed: set = set()
        for piece in _facet_components(remaining, fac):
            if len(piece) < 30:
                continue
            P = pts[fac[piece]].reshape(-1, 3)
            Nrep = np.repeat(normals[piece], 3, axis=0)[:len(P)]
            # a tangent-glued strip+fan piece is not separable by connectivity or spatial
            # gaps (the strip touches the plane along its tangent line): keep only the
            # CURVED facets — the discriminator the G7 splitter itself uses — and fit those
            got = _fit.classify(P, Nrep, cyl_tol, sigma, kinds=("cylinder",))
            curved = _curved_subset(piece, fac, normals)
            if 30 <= len(curved) < len(piece):
                Pc = pts[fac[curved]].reshape(-1, 3)
                Nc = np.repeat(normals[curved], 3, axis=0)[:len(Pc)]
                got = _fit.classify(Pc, Nc, cyl_tol, sigma, kinds=("cylinder",))
            if got["kind"] != "cylinder":
                # the piece is vertex-glued through corner transitions: try the
                # gap-split sub-pieces before leaving it unrecognized
                for sub in _split_by_gaps(piece, pts, fac):
                    if len(sub) < 30:
                        continue
                    Ps = pts[fac[sub]].reshape(-1, 3)
                    Ns = np.repeat(normals[sub], 3, axis=0)[:len(Ps)]
                    got_s = _fit.classify(Ps, Ns, cyl_tol, sigma, kinds=("cylinder",))
                    if got_s["kind"] != "cylinder":
                        continue
                    _emit_cylinder(out, sub, got_s["fit"], pts, fac, normals, _area)
                    consumed.update(sub)
                continue
            _emit_cylinder(out, piece, got["fit"], pts, fac, normals, _area)
            consumed.update(piece)
            continue
            out.append({"_idx": piece, "type": "cylinder",
                        "axis": axis_c.tolist(),
                        "point": np.asarray(f["point"]).tolist(),
                        "radius": round(f["radius"], 4),
                        "rms_mm": round(f["rms"], 5),
                        "turn_deg": round(f["turn_deg"], 1),
                        "axial_span": round(f["axial_span"], 3),
                        "axial_mid": round(float((P @ axis_c).mean()), 4),
                        "facets": len(piece), "area_mm2": _area(piece)})
            consumed.update(piece)
        if consumed:
            remaining = [f2 for f2 in remaining if f2 not in consumed]
    if remaining:
        out.append({"_idx": remaining, "type": "unrecognized",
                    "facets": len(remaining), "area_mm2": _area(remaining)})
    return out


def _dressings(regions, region_pts, pts, fac, total_area):
    """G7: fillet / chamfer recognition from FITTED surfaces (mesh dihedrals lie).

    A FILLET is a narrow cylinder region tangent to two adjacent non-fillet regions
    (junction <= ``FILLET_TANGENCY_DEG`` against the neighbour's fitted surface) with an
    angular span <= ``FILLET_MAX_SPAN_DEG``; its measured size is the G2 cylinder fit
    radius (a single fit — slab-wise variable radius is a documented future refinement).
    A CHAMFER is a narrow plane strip whose junction with BOTH neighbours sits inside
    ``CHAMFER_ANGLE_RANGE`` (a distinct intermediate angle); its size is the strip
    width (area / spine length).
    Junction angles come from the FITTED surfaces' normals along the shared boundary:
    for a cylinder the radial direction at the boundary points, for planes the fitted
    normal. A sharp edge has no strip region, so it can never be classified — the
    specificity guard is structural. Returns evidence-rich records, never a guess."""
    if not regions or total_area <= 0:
        return []
    # facet-index adjacency between the final regions
    vgroups: dict = {}
    for ri, r in enumerate(regions):
        for f in r.get("_idx") or ():
            for v in fac[f]:
                vgroups.setdefault(int(v), set()).add(ri)
    pairs = set()
    for ri in range(len(regions)):
        vs = set()
        for f in regions[ri].get("_idx") or ():
            vs.update(int(v) for v in fac[f])
        neigh = set()
        for v in vs:
            for rj in vgroups.get(v, ()):
                if rj != ri:
                    neigh.add(rj)
        for rj in neigh:
            pairs.add((min(ri, rj), max(ri, rj)))

    def shared_boundary_points(ri, rj):
        """Rough midpoint of the shared boundary: the midpoint of ri's points that lie
        nearest to region rj's points (a boundary midpoint is all the angle test
        needs). P4.5: a cKDTree replaces the O(n·m) pairwise matrix — this was
        the recognition hotspot on multi-region meshes. (Also fixes a latent
        bug: the old mask compared argmin INDICES against a distance
        threshold instead of the nearest distance itself.)"""
        Pi = region_pts.get(id(regions[ri]))
        Pj = region_pts.get(id(regions[rj]))
        if Pi is None or Pj is None or not len(Pi) or not len(Pj):
            return None
        try:
            from scipy.spatial import cKDTree

            k = min(4, len(Pj))
            dist, _ = cKDTree(Pj).query(Pi, k=k)
            dist = np.atleast_2d(dist)
            d1, kth = dist[:, 0], dist[:, -1]
        except Exception:  # noqa: BLE001 - no scipy: same math, just slower
            d = np.sqrt(((Pi[:, None, :] - Pj[None, :, :]) ** 2).sum(2))
            d1 = d.min(axis=1)
            kth = np.sort(d, axis=1)[:, min(3, d.shape[1] - 1)]
        near = Pi[d1 < max(float(np.median(kth)), 1e-9) * 2.0]
        return near if len(near) else None

    def junction_angle_cyl_plane(cyl, plane, boundary=None):
        """Tangency of the FITTED cylinder to the FITTED plane, as an equivalent angle:
        a cylinder is tangent to a plane exactly when the distance from its AXIS LINE to
        the plane equals its radius — |dist - r| mapped onto the arc (asin) so the gate's
        degree threshold keeps its meaning. This is immune to region fragmentation (the
        tangent boundary rows are usually claimed by the plane region, so boundary-point
        methods measure mid-arc angles on genuine fillets — measured 41-90 deg)."""
        if "normal" not in plane or "axis" not in cyl:
            return None                     # no fitted surface to measure against: skip honestly
        a = np.asarray(cyl["axis"], dtype=float)
        point = np.asarray(cyl["point"], dtype=float)
        n = np.asarray(plane["normal"], dtype=float)
        p_plane = np.asarray(plane.get("point") or (0.0, 0.0, 0.0), dtype=float)
        dist = abs(float((point - p_plane) @ n))
        gap = abs(dist - float(cyl["radius"]))
        scale = max(float(cyl["radius"]), 1e-9)
        if gap >= scale:                    # the plane misses the cylinder entirely
            return 90.0
        return float(math.degrees(math.asin(min(1.0, gap / scale))))

    def plane_angle(p1, p2):
        cos = abs(float(np.asarray(p1["normal"], dtype=float) @
                            np.asarray(p2["normal"], dtype=float)))
        return math.degrees(math.acos(min(1.0, max(-1.0, cos))))

    # E2/G7: partial-arc extraction can leave ONE physical fillet as several coaxial
    # equal-radius region FRAGMENTS, each touching a different one of its planes — no
    # fragment alone shows tangency to both. Group coaxial equal-radius cylinders into
    # one physical candidate (union of facets/boundaries, summed area) before the gate.
    group_of: dict = {}
    cylinders: list = []
    for ri, r in enumerate(regions):
        if r["type"] != "cylinder":
            continue
        a = np.asarray(r.get("axis") or (0, 0, 0), dtype=float)
        p = np.asarray(r.get("point") or (0, 0, 0), dtype=float)
        merged = None
        for gi, g in enumerate(cylinders):
            ga = np.asarray(regions[g["members"][0]]["axis"], dtype=float)
            if abs(float(a @ ga)) < 1 - 1e-9:
                continue
            r0 = float(regions[g["members"][0]]["radius"])
            if abs(r0 - float(r["radius"])) > max(0.05 * max(r0, 1e-9), 0.02):
                continue
            gp = np.asarray(regions[g["members"][0]]["point"], dtype=float)
            if abs(float((gp - p) @ a)) > max(0.1 * float(r.get("axial_span") or 1.0), 0.5):
                continue
            merged = gi
            break
        if merged is None:
            cylinders.append({"members": [ri], "area": float(r.get("area_mm2") or 0)})
            group_of[ri] = len(cylinders) - 1
        else:
            cylinders[merged]["members"].append(ri)
            cylinders[merged]["area"] += float(r.get("area_mm2") or 0)
            group_of[ri] = merged

    out = []
    for ri, r in enumerate(regions):
        if r["type"] not in ("cylinder", "plane"):
            continue
        if r["type"] == "cylinder":
            g = cylinders[group_of[ri]]
            if g["area"] > DRESSING_AREA_SHARE * total_area:
                continue                              # the PHYSICAL strip is not narrow
        elif r["area_mm2"] > DRESSING_AREA_SHARE * total_area:
            continue                                  # not a narrow strip
        partners = sorted({other for (x, y) in pairs
                           for other in ((y,) if x == ri else (x,) if y == ri else ())},
                          key=lambda rj: -regions[rj]["area_mm2"])
        if len(partners) < 2:
            continue
        if r["type"] == "cylinder":
            plane_partners = [p for p in partners if regions[p]["type"] == "plane"]
            if len(plane_partners) < 2:
                continue
            a, b = (regions[plane_partners[0]], regions[plane_partners[1]])
        else:
            a, b = (regions[partners[0]], regions[partners[1]])
            if a["type"] != "plane" or b["type"] != "plane":
                continue
        if r["type"] == "cylinder":
            if r.get("turn_deg", 0.0) > FILLET_MAX_SPAN_DEG:
                continue
            # E2: the GEOMETRY selects the fillet's two faces, not partner area — a
            # corner-adjacent END plane can outrank the true tangent side by area while
            # measuring 90 degrees. Tangency to EVERY plane partner is computed from the
            # fitted surfaces; the two smallest win the candidacy, and the gate (both
            # <= FILLET_TANGENCY_DEG) is unchanged.
            measured = []
            for rj in plane_partners:
                ang = junction_angle_cyl_plane(r, regions[rj])
                if ang is not None:
                    measured.append((ang, rj))
            if len(measured) < 2:
                continue
            measured.sort()
            angles = [ang for ang, _rj in measured[:2]]
            a, b = (regions[measured[0][1]], regions[measured[1][1]])
            if max(angles) > FILLET_TANGENCY_DEG:
                continue
            out.append({"kind": "fillet", "radius_mm": r["radius"],
                        "tangency_deg": round(max(angles), 2),
                        "span_deg": r.get("turn_deg"),
                        "between": [a["type"], b["type"]],
                        "area_mm2": r["area_mm2"],
                        # the fitted geometry a planner maps to an edge family:
                        # a PERIMETER fillet's axis lies in the plate plane, a
                        # vertical-corner fillet's runs along the plate normal
                        "axis": r["axis"], "point": r["point"],
                        "variable_radius": False})
        else:                                          # a narrow plane strip
            ang_a, ang_b = plane_angle(r, a), plane_angle(r, b)
            # a REAL chamfer meets both neighbours at a distinct intermediate angle; a
            # fillet fragment split out of a tangent arc meets its neighbours near
            # tangency (<= 25 deg) or at the full corner (>= 80 deg) — neither is one
            if not (25.0 <= ang_a <= 90.0 + 1e-6 and 25.0 <= ang_b <= 90.0 + 1e-6):
                continue
            # corner closure: the strip's two junction angles must ADD UP to the angle
            # between its two neighbours (the chamfer plane splits the corner). A fillet
            # fragment split out of a tangent arc fails this by construction.
            if abs(plane_angle(a, b) - (ang_a + ang_b)) > 5.0:
                continue
            P = region_pts.get(id(r))
            if P is None or len(P) < 6:
                continue
            length = float(np.ptp(P, axis=0).max())
            width = float(r["area_mm2"]) / max(length, 1e-9)
            out.append({"kind": "chamfer", "width_mm": round(width, 4),
                        "angles_deg": [round(ang_a, 2), round(ang_b, 2)],
                        "between": [a["type"], b["type"]],
                        "area_mm2": r["area_mm2"],
                        # the fitted strip geometry a planner maps to an edge
                        # family (a perimeter chamfer's normal tilts out of the
                        # plate plane, a vertical-corner one's stays in it)
                        "normal": r["normal"], "point": r["point"]})
    return out


def _csink_gate(P, cone, sphere, sigma):
    """A cone at a hole entry is a COUNTERSINK only when it beats the sphere fit on
    the same points by the extra-sum-of-squares F-test (the sphere is the nested-ish
    competitor for a conical band; the band often Occam-classifies as sphere)."""
    if cone is None:
        return False, "no cone fit"
    if sphere is None:
        return True, ""
    try:
        from scipy.stats import f as f_dist
    except Exception:  # noqa: BLE001 - without scipy fall back to rms comparison
        return cone["rms"] < sphere["rms"] * 0.8, ""
    n = _fit._effective_n(P)
    rss_c, rss_s = cone["rms"] ** 2 * n, sphere["rms"] ** 2 * n
    q, dof = 6 - 4, max(n - 6, 1)
    if rss_c >= rss_s * 0.999:
        return False, "cone no better than sphere"
    f_stat = ((rss_s - rss_c) / q) / (rss_c / dof)
    if f_stat <= f_dist.isf(_fit.F_ALPHA, q, dof):
        return False, "F-test vs sphere"
    if sigma and sphere["rms"] < cone["rms"] + _fit.SEPARATION_SIGMA * sigma:
        return False, "separation"
    return True, ""


def _extend_wall_to_surface(mesh, a, bore, lo, hi, max_step=1.0, max_reach=25.0):
    """Probe the fitted cylinder surface OUTWARD from each interval end with
    EXACT point-to-mesh distances: the recognized wall REGION can be partial
    (measured on the filleted bracket: a through bore's wall came back
    recognized only over its upper two thirds — the rest sat in
    'unrecognized'). The MESH is the truth: if the tessellated surface lies
    on the fitted cylinder just past the end, the wall extends; the walk
    stops at the first miss (below a blind floor the ring lies inside solid
    material — the nearest mesh surface is the floor itself, well beyond the
    sag tolerance — so real blind holes keep their floor-bounded extent).
    Bounded: ≤ max_reach per end, 4 exact probes per step."""
    try:
        from .deviation import _points_to_mesh_exact
    except Exception:  # noqa: BLE001 - keep the recognized extent on import failure
        return lo, hi
    a = np.asarray(a, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    radius = float(bore["radius"])
    pt = np.asarray(bore["point"], dtype=float)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(a @ ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(a, ref)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(a, u)
    # the tessellation's chord sag is bounded by the mesh deflection, so a
    # probe ON the true cylinder is within ~deflection of the surface; the
    # radius term covers coarser meshes
    tol = max(0.35, 0.1 * radius)
    step = min(max_step, 0.25 * max(hi - lo, 1e-6))

    def _surface_at(x) -> bool:
        probes = [tuple(pt + x * a
                        + radius * (math.cos(ang) * u + math.sin(ang) * v))
                  for ang in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)]
        d = _points_to_mesh_exact(probes, mesh)
        return d is not None and sum(1 for q in d if float(q) <= tol) >= 3

    x = lo - step
    while x > lo - max_reach and _surface_at(x):
        lo = x
        x -= step
    x = hi + step
    while x < hi + max_reach and _surface_at(x):
        hi = x
        x += step
    return lo, hi


def _hole_features(cylinders, regions, planes, region_pts=None, mesh=None):
    """Hole recognition from fitted geometry (R-3): through / blind / counterbore /
    countersink, every dimension MEASURED from the evidence — never invented.

    Through: two planes with normal ∥ the axis intersecting the wall's AXIAL INTERVAL
    (position-checked: the plate's far face beyond a blind bottom is NOT enclosing).
    Blind: one enclosing plane; depth from the enclosing face to a floor plane near the
    far end (high confidence) or the wall's axial extent (medium, basis stated).
    Counterbore: two COAXIAL cylinders with distinct radii (R >= 1.15 r) and axially
    adjacent intervals; cbore diameter/depth from the large cylinder.
    Countersink: a cone coaxial with a cylinder whose radius_max end sits at the entry;
    diameter = 2*radius_max, angle = 2*half_angle (FreeCAD's included angle).
    Anything ambiguous stays separate plain holes — a step without coaxiality or
    significance is NOT a counterbore."""
    cones = [r for r in regions if r["type"] == "cone"]
    holes = []
    used_cone = set()

    def axis_of(entry):
        return np.asarray(entry["axis"], dtype=float)

    def axial_interval(cyl):
        a = axis_of(cyl)
        mid = float(cyl.get("axial_mid", float(np.dot(np.asarray(cyl["point"],
                                                                dtype=float), a))))
        half = float(cyl.get("axial_span", 0.0)) / 2.0
        return a, mid - half, mid + half

    def coaxial(c1, c2):
        a1, _, _ = axial_interval(c1)
        if abs(float(a1 @ axis_of(c2))) < math.cos(math.radians(5.0)):
            return False
        d = np.asarray(c2["point"], dtype=float) - np.asarray(c1["point"], dtype=float)
        lateral = float(np.linalg.norm(d - float(d @ a1) * a1))
        tol = max(0.5, 0.2 * min(c1["radius"], c2["radius"]))
        return lateral <= tol

    def enclosing_planes(a, lo, hi):
        """Planes with normal ∥ the axis whose axial position intersects the
        bore's interval, with their SIGNED facing: (position, plane, sign)
        where sign = n·a — the entry face of a top-drilled hole faces +a, a
        through-exit face faces −a, and a blind floor faces +a (INTO the hole,
        same as the entry). The window widens with the wall's OWN span: the
        fitted cylinder interval under-measures by a facet row at each end
        (measured on the filleted bracket: span 7.988 on a 10 mm bore), and a
        fixed ±1 mm slack then misses a true exit face by hundredths — the
        through/blind decision must not hinge on that."""
        slack = 1.0 + 0.15 * (hi - lo)
        out = []
        for p in planes:
            pn = np.asarray(p["normal"], dtype=float)
            facing = float(pn @ a)
            if abs(facing) < math.cos(math.radians(10.0)):
                continue
            pos = float(np.dot(np.asarray(p["point"], dtype=float), a))
            if lo - slack <= pos <= hi + slack:
                out.append((pos, p, facing))
        return out

    def _covers(p, bore, a):
        """Does the plane's footprint cover the bore cross-section?"""
        d = np.asarray(p["point"], dtype=float) - np.asarray(bore["point"], dtype=float)
        lateral = float(np.linalg.norm(d - float(d @ a) * a))
        return lateral <= 2.0 * bore["radius"]

    def base_hole(cyl, through, extras=None):
        hole = {"center": cyl["point"], "axis": cyl["axis"], "radius": cyl["radius"],
                "through": through, "area_mm2": cyl["area_mm2"],
                "confidence": "high" if cyl["rms_mm"] < 0.1 else "medium"}
        if extras:
            hole.update(extras)
        if cyl.get("evidence") in ("synthetic", "mixed"):
            hole["dimension_basis"] = (
                "synthetic-influenced evidence (%s) — the measured walls were changed or "
                "created by repair; review before trusting this dimension" % cyl["evidence"])
        return hole

    groups = []
    for cyl in cylinders:
        if not (HOLE_RADIUS_MM[0] <= cyl["radius"] <= HOLE_RADIUS_MM[1]):
            continue
        if cyl["turn_deg"] < HOLE_MIN_TURN_DEG:
            continue
        for g in groups:
            if any(coaxial(cyl, other) for other in g):
                g.append(cyl)
                break
        else:
            groups.append([cyl])

    for g in groups:
        g.sort(key=lambda c: -c["radius"])
        bore, extra = g[0], None
        kind = None
        if len(g) >= 2 and g[0]["radius"] >= 1.15 * g[1]["radius"]:
            large, small = g[0], g[1]
            # canonicalization can leave members with OPPOSITE axis signs: re-express
            # each member's axial mid in the GROUP axis frame first
            ga = axis_of(large)

            def mid_in(cyl):
                m = float(cyl.get("axial_mid", 0.0))
                if float(axis_of(cyl) @ ga) < 0:
                    m = -m
                return m

            mid_l, span_l = mid_in(large), float(large.get("axial_span", 0.0))
            mid_s, span_s = mid_in(small), float(small.get("axial_span", 0.0))
            # orientation-free adjacency: the intervals touch or overlap, and the small
            # bore extends beyond the large step (it continues where the step ends)
            gap = abs(mid_s - mid_l) - (span_l + span_s) / 2.0
            deeper = span_s > span_l * 0.6
            if gap <= 1.0 and deeper:
                depth = span_l                    # the step's own measured extent
                if depth > 0.2 * large["radius"]:
                    kind = "counterbore"
                    extra = {"kind": "counterbore", "counterbore": {
                        "diameter": round(2.0 * large["radius"], 3),
                        "depth": round(depth, 3)}}
                    bore = small
        a, lo, hi = axial_interval(bore)
        # the bore's interval is the UNION of every coaxial member's extent:
        # segmentation can SPLIT one wall (measured on the filleted bracket:
        # the bore wall came back as z 3..11 while the physical bore spans the
        # whole 10 mm plate) — the through/blind decision must see the full
        # wall, not the fragment that happened to fit first
        for member in g:
            ma, mlo, mhi = axial_interval(member)
            if abs(float(ma @ a)) < math.cos(math.radians(10.0)):
                continue
            if float(ma @ a) < 0:
                # the member's mid is recorded in ITS OWN (mirrored) frame:
                # negate the interval to the group axis frame
                mlo, mhi = -mhi, -mlo
            lo, hi = min(lo, mlo), max(hi, mhi)
        if mesh is not None and kind is None:
            # a partial wall region must not decide through-vs-blind alone
            lo, hi = _extend_wall_to_surface(mesh, a, bore, lo, hi)
        mirrored = float(axis_of(bore) @ ga) < 0 if kind == "counterbore" else False
        if kind == "counterbore":
            # the bore CONTINUES through the counterbore to the entry face: extend
            # its interval on the step side by the cbore depth (group-frame aware)
            step_high = (mid_l > mid_s) != mirrored
            if step_high:
                hi = max(hi, hi + span_l)
            else:
                lo = min(lo, lo - span_l)
        enc = enclosing_planes(a, lo, hi)
        # plane-region normals are canonicalized (sign lost), so facing alone cannot
        # tell a through-exit from a blind floor. AREA can, honestly: an outer face
        # of the plate is far larger than the bore cross-section; a blind floor IS
        # the bore cross-section (~pi r^2). Exit faces: area > 4 pi r^2.
        cross = math.pi * bore["radius"] ** 2
        exits = [pos for pos, p, _ in enc if p["area_mm2"] > 4.0 * cross]
        floors = [(pos, p) for pos, p, _ in enc
                  if p["area_mm2"] <= 4.0 * cross and _covers(p, bore, a)]
        through = len(exits) >= 2 or (len(exits) >= 1 and False)
        if kind is None:
            kind = "through" if through else "blind"
        extras = dict(extra or {})
        extras["kind"] = kind
        if kind == "counterbore":
            # entry face = the bore interval end on the step side (group-frame aware)
            step_high = (mid_l > mid_s) != mirrored
            entry_ax = hi if step_high else lo
            axis_pt = np.asarray(bore["point"], dtype=float)
            extras["entry_point"] = (axis_pt + float(entry_ax -
                                                     float(axis_pt @ a)) * a).tolist()
        if not through:
            # blind: entry = an exit-sized OUTER face at one interval end (position
            # along a, sign-mirror-free via abs); floor = small covering plane
            if exits:
                entry_pos = exits[0]
                # 3D entry point on the bore axis (plan uses it to pick the drill face)
                axis_pt = np.asarray(bore["point"], dtype=float)
                extras["entry_point"] = (axis_pt + float(entry_pos -
                                                         float(axis_pt @ a)) * a).tolist()
                near_end = min((abs(pos - entry_pos) for pos in (lo, hi)), default=0.0)
                if floors:
                    floor_pos = floors[0][0]
                    extras["depth"] = round(abs(float(entry_pos - floor_pos)), 3)
                    extras["depth_basis"] = "entry face to floor plane"
                else:
                    extras["depth"] = round(float(bore.get("axial_span", hi - lo)), 3)
                    extras["depth_basis"] = "cylinder wall axial extent (no floor plane seen)"
                    extras["confidence"] = "medium"
        if region_pts is not None:
            # countersink evidence: a curved region coaxial at the bore entry — refit
            # a CONE to its points and accept only through the significance gate
            # (region kinds like 'sphere' are the Occam label, not the verdict)
            for region in regions:
                if region["type"] in ("plane", "cylinder") or id(region) in used_cone:
                    continue
                P = region_pts.get(id(region))
                if P is None or len(P) < 12:
                    continue
                c = np.asarray(P, dtype=float).mean(axis=0)
                d = c - np.asarray(bore["point"], dtype=float)
                lateral = float(np.linalg.norm(d - float(d @ a) * a))
                if lateral > 3.0 * bore["radius"]:
                    continue
                cone_f = _fit.fit_cone(P, _fit.fit_plane(P)[0])
                if cone_f is None:
                    continue
                ca = np.asarray(cone_f["axis"], dtype=float)
                if abs(float(ca @ a)) < math.cos(math.radians(10.0)):
                    continue
                if cone_f["radius_max"] < 1.15 * bore["radius"]:
                    continue
                sph_f = _fit.fit_sphere(P)
                ok, _why = _csink_gate(P, cone_f, sph_f, 0.0)
                used_cone.add(id(region))
                if not ok:
                    continue
                Pa_arr = np.asarray(P, dtype=float) @ a
                axis_pt = np.asarray(bore["point"], dtype=float)
                band_mid = float(Pa_arr.mean())
                extras["entry_point"] = (axis_pt + float(band_mid -
                                                         float(axis_pt @ a)) * a).tolist()
                extras["kind"] = "countersink"
                extras["countersink"] = {
                    "diameter": round(2.0 * cone_f["radius_max"], 3),
                    "angle": round(2.0 * math.degrees(cone_f["half_angle"]), 1)}
                # the bore CONTINUES through the conical entry: re-check through with
                # the interval extended by the band height on the entry side
                band = float(np.ptp(np.asarray(P, dtype=float) @ a))
                Pa = np.asarray(P, dtype=float) @ a
                if Pa.mean() > float(np.asarray(bore["point"], dtype=float) @ a):
                    enc2 = enclosing_planes(a, lo, hi + band + 1.0)
                else:
                    enc2 = enclosing_planes(a, lo - band - 1.0, hi)
                cross2 = math.pi * bore["radius"] ** 2
                exits2 = [pos for pos, p2, _ in enc2 if p2["area_mm2"] > 4.0 * cross2]
                if len(exits2) >= 2:
                    through = True
                    extras["through"] = True
                break
        holes.append(base_hole(bore, through, extras))
    return holes


def _parallel_plane_pairs(planes):
    pairs = []
    for i, p in enumerate(planes):
        for q in planes[i + 1:]:
            n1, n2 = np.array(p["normal"]), np.array(q["normal"])
            if abs(float(n1 @ n2)) > math.cos(math.radians(5.0)):
                d = abs(float(np.dot(np.array(q["point"]) - np.array(p["point"]), n1)))
                pairs.append({"normal": p["normal"], "distance_mm": round(d, 3),
                              "area_mm2": round(min(p["area_mm2"], q["area_mm2"]), 3)})
    pairs.sort(key=lambda e: -e["area_mm2"])
    return pairs


def _classify(diag, plane_area, cyl_area, unrecognized, total, pairs, cylinders, holes,
              sphere_area=0.0):
    """Evidence-based route class + the §16 intent fields: ``confidence`` is the share of the
    surface the chosen route's geometry actually explains (never a feeling), and
    ``alternatives`` names the routes whose OWN evidence gate also passed (empty when the
    winner is unambiguous). approve.py's RULE_INTERPRETATION asks only when confidence is
    below 0.5 AND an alternative exists — a heuristic winner with full support stays silent."""
    return _route_confidence(_classify_routes(diag, plane_area, cyl_area, unrecognized, total,
                                              pairs, cylinders, holes, sphere_area))


def _route_confidence(result):
    """Attach the measured intent fields to a classification result (see _classify)."""
    evidence = result.get("evidence") or {}
    winner_share = {"prismatic": "plane_share", "thin_panel": "plane_share",
                    "rotational": "rotational_share", "spherical": "sphere_share"}.get(
                        result.get("class") or "", "")
    result["confidence"] = round(min(max(float(evidence.get(winner_share) or 0.0), 0.0), 1.0), 3)
    alternatives = []
    holes = int(evidence.get("holes") or 0)
    if (result.get("class") not in ("rotational", "spherical")
            and float(evidence.get("rotational_share") or 0.0) >= 0.5 and holes == 0):
        alternatives.append("rotational")
    if (result.get("class") not in ("prismatic", "thin_panel")
            and float(evidence.get("plane_share") or 0.0) >= 0.5
            and int(evidence.get("parallel_pairs") or 0) >= 1):
        alternatives.append("prismatic_extrusion")
    if alternatives:
        result["alternatives"] = alternatives
    return result


def _classify_routes(diag, plane_area, cyl_area, unrecognized, total, pairs, cylinders, holes,
                     sphere_area=0.0):
    """Evidence-based route class (heuristics, honestly labelled as such in the report)."""
    box = (diag or {}).get("size_mm") or [0, 0, 0]
    thin = bool(box and min(box) > 0 and min(box) / max(max(box), 1e-9) <= 0.1)
    notes = {"plane_share": round(plane_area / total, 3),
             "cylinder_share": round(cyl_area / total, 3),
             "rotational_share": round((cyl_area + sphere_area) / total, 3),
             "unrecognized_share": round(unrecognized / total, 3),
             "parallel_pairs": len(pairs), "holes": len(holes), "thin": thin}
    if (diag or {}).get("components", 1) > 1:
        return {"class": "multi_component", "evidence": notes}
    if unrecognized / total >= 0.4:
        return {"class": "organic_or_hybrid", "evidence": notes}
    sphere_share = sphere_area / total if total else 0.0
    if sphere_share >= 0.8:
        # G2: a genuinely spherical scan (a ball) is RECOGNIZED — honestly labelled;
        # a DOME (cylinder + spherical cap, share 0.5-0.8) is a rotational part
        return {"class": "spherical", "evidence": {**notes, "sphere_share":
                round(sphere_share, 3)}}
    box = (diag or {}).get("size_mm") or [0, 0, 0]
    # R-4: coaxial full-turn cylinders are the STEPS of a turned part, not holes — a
    # pseudo-counterbore on the dominant axis must not block the rotational class
    dom_axis = None
    if cylinders:
        big = max(cylinders, key=lambda c: c["area_mm2"])
        dom_axis = np.asarray(big["axis"], dtype=float)
    def _coaxial(h):
        return dom_axis is not None and abs(float(np.asarray(h["axis"]) @ dom_axis))             > math.cos(math.radians(10.0))

    def _transverse(h):
        # H3: a cross-hole (axis ~perpendicular to the rotation axis) is a FEATURE of
        # the rotational part that R-4 now drills — it must not block the class either
        return dom_axis is not None and abs(float(np.asarray(h["axis"]) @ dom_axis))             < math.cos(math.radians(80.0))

    real_holes = [h for h in holes
                  if (not box or h["radius"] < 0.45 * max(min(box), 1e-9))
                  and not _coaxial(h) and not _transverse(h)]
    rotational_share = (cyl_area + sphere_area) / total if total else 0.0
    # a dome cap's wall merges tangentially with its spherical cap (no cylinder region
    # survives), so a sphere share of 0.4-0.8 with no cross holes is rotational too
    if (rotational_share >= 0.5 or 0.4 <= sphere_share < 0.8) and not real_holes:
        evidence = {**notes}
        if sphere_share >= 0.05:
            evidence["dome_share"] = round(sphere_share, 3)
        return {"class": "rotational", "evidence": evidence}
    # prismatic outranks thin_panel: a plate with strong parallel-plane evidence IS an
    # extrusion (the prismatic route handles it); thin_panel stays for sheet-like parts
    # without a dominant parallel-plane pair
    # 0.5 (was 0.6): a filleted bracket's plane share moves ±10 % with the
    # tessellation's facet boundaries and part orientation (measured: the same part
    # at dev 0.3/0.4 vs a 20-30 deg rotation re-mesh gives 0.50-0.64); pairs +
    # not-rotational still gate it, and the organic class (unrecognized >= 0.4)
    # outranks this branch
    if plane_area / total >= 0.5 and pairs:
        return {"class": "prismatic", "evidence": notes}
    if thin:
        return {"class": "thin_panel", "evidence": notes}
    return {"class": "hybrid_mechanical", "evidence": notes}
