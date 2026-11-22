"""G12 post-passes for ``recognize``: absorb, union-fit merge and kind upgrade.

Runs on the region labeling BEFORE per-region fitting (so merged regions are fitted once,
as unions). The three passes are the verified core of OriginShape's segmentation stability
(``segment_mesh.py:593-759``, reimplemented — see Report 2 §G12):

(a) **absorb**: face groups below the minimum region size join the best adjacent
    qualifying region while ``max(corner_dist / dist_tol, normal_angle / ang_tol) <= 1.5``
    (corner distance to the candidate region's mean plane; normal angle against the
    candidate's mean normal — no fitted model exists yet at this stage);
(b) **union-fit merge**: adjacent regions whose union fits ONE primitive kind within the
    K1 tolerance merge, greedily by best union RMS; same-kind merges (split cylinders,
    coplanar strips) and cylinder+cylinder -> cone upgrades (through the G2 significance
    gate) are both admissible;
(c) **kind upgrade**: the cone upgrade is the union fit above — a merged pair only
    becomes a cone when ``fit.cone_significance`` accepts it against BOTH nested fits.

Deterministic (sorted iteration), NumPy-only, provenance keys ride along as facet indices
so evidence tagging after the passes reflects the MERGED regions.
"""

import math

import numpy as np

from . import fit as _fit

ABSORB_SCORE = 1.5            # the report's absorb threshold (× tolerance)


def absorb_small(groups, small, normals, fac, pts, threshold_deg):
    """(a) Sub-minimal face groups join their best adjacent region while the score
    ``max(dist/dist_tol, angle/ang_tol) <= 1.5``. Returns the new groups list."""
    if not small or not groups:
        return [list(g) for g in groups], list(small)
    ang_tol = math.cos(math.radians(threshold_deg))
    groups = [list(g) for g in groups]
    face_group = {}
    for gi, g in enumerate(groups):
        for f in g:
            face_group[f] = gi
    # candidate region geometry: mean normal + mean plane through its face centroids
    cand = []
    for g in groups:
        n = normals[g].mean(axis=0)
        n /= np.linalg.norm(n) + 1e-12
        c = pts[fac[g]].reshape(-1, 3).mean(axis=0)
        cand.append((n, c))
    sample = fac[: min(len(fac), 2000)].reshape(-1, 2) if len(fac) else np.zeros((1, 2))
    edge_len = float(np.median(np.linalg.norm(pts[sample[:, 0]] - pts[sample[:, 1]],
                                              axis=1))) if len(fac) else 1.0
    dist_scale = max(3.0 * edge_len, 1e-6)
    remaining = []
    for g in small:
        g = list(g)
        # adjacency: any shared mesh EDGE (vertex pair) between g and a candidate region
        edges_small = set()
        for f in g:
            a, b, c3 = fac[f]
            for x, y in ((a, b), (b, c3), (c3, a)):
                edges_small.add((min(x, y), max(x, y)))
        best, best_score = None, None
        for gi, g2 in enumerate(groups):
            g2_edges = set()
            for f in g2:
                a, b, c3 = fac[f]
                for x, y in ((a, b), (b, c3), (c3, a)):
                    e = (min(x, y), max(x, y))
                    if e in edges_small:
                        g2_edges.add(e)
            if not g2_edges:
                continue
            n2, c2 = cand[gi]
            n_small = normals[g].mean(axis=0)
            n_small /= np.linalg.norm(n_small) + 1e-12
            ang = 1.0 - float(n_small @ n2)   # signed: anti-parallel faces score 2
            ang_score = ang / max(1.0 - ang_tol, 1e-6)
            centroids = pts[fac[g]].reshape(-1, 3).mean(axis=0)
            dist = float(abs((centroids - c2) @ n2))
            dist_score = dist / dist_scale
            score = max(ang_score, dist_score)
            if score <= ABSORB_SCORE and (best_score is None or score < best_score):
                best, best_score = gi, score
        if best is not None:
            groups[best].extend(g)
            for f in g:
                face_group[f] = best
        else:
            remaining.append(g)
    return [g for g in groups if g], remaining


def _union_fit(idx_a, idx_b, fac, pts, normals, plane_tol, curved_tol, sigma,
               ka=None, kb=None):
    """Fit the UNION of two face sets; return (kind, fit, rms) for an acceptable
    single-kind union — else None. The union must RESPECT the pair's original kinds:
    plane+plane only as a plane (a perpendicular corner never becomes a "cylinder"),
    cylinder+cylinder as a cylinder or a cone (the G2 upgrade), sphere/torus same-kind
    only. A re-classification to a different kind is not admissible: merging must
    simplify, never reinterpret."""
    idx = np.concatenate((idx_a, idx_b))
    P = pts[fac[idx]].reshape(-1, 3)
    Nrep = np.repeat(normals[idx], 3, axis=0)[:len(P)]
    want = {ka, kb} if ka is not None and kb is not None else None
    plane = _fit.fit_plane(P)
    if plane[2] <= plane_tol and (want is None or want == {"plane"}):
        return "plane", {"normal": plane[0], "point": plane[1], "rms": plane[2]}, plane[2]
    got = _fit.classify(P, Nrep, curved_tol, sigma,
                        plane_fit=(plane[0], plane[1], plane[2]))
    kind = got["kind"]
    if kind == "cone" and want == {"cylinder"}:
        # the ONE admissible upgrade: cylinder+cylinder -> cone through the G2 gate
        # (classify already ran cone_significance before accepting the cone)
        return "cone", got["fit"], got["fit"]["rms"]
    if kind in ("cylinder", "sphere", "torus") and (want is None or want == {kind}):
        return kind, got["fit"], got["fit"]["rms"]
    return None


def merge_adjacent(groups, entries, fac, pts, normals, plane_tol, curved_tol, sigma,
                   adjacency_pairs, region_normals=None):
    """(b)+(c) Union-fit merge: greedily merge adjacent regions whose union fits one kind
    within tolerance; ``cylinder+cylinder`` pairs may upgrade to ``cone`` through the G2
    gate (inside ``_union_fit`` via ``fit.classify``). ``entries`` holds per-group kind
    hints ('plane'/'cylinder'/.../None). Returns (groups, entries, merge_log)."""
    groups = [list(g) for g in groups]
    kinds = list(entries)
    merge_log = []
    changed = True
    while changed:
        changed = False
        best = None   # (rms, ia, ib, kind, fit)
        for ia, ib in adjacency_pairs:
            if ia >= len(groups) or ib >= len(groups) or ia == ib:
                continue
            if not groups[ia] or not groups[ib]:
                continue
            # only admissible kind combinations: same kind, or cylinder pair (cone
            # upgrade candidate); never merge a plane with a curved kind
            ka, kb = kinds[ia], kinds[ib]
            if ka != kb and not (ka == "cylinder" and kb == "cylinder"):
                continue
            if ka is None or kb is None:
                continue
            # a PLANE union is only admissible when the two regions' own normals are
            # (near-)parallel: a perpendicular side face can squeak under the rms
            # tolerance on large flat faces (measured: top+side merged at 0.09 mm rms
            # with a 0.5° tilt) — geometrically nonsense
            if ka == "plane" and region_normals is not None:
                na = np.asarray(region_normals[ia], dtype=float)
                nb = np.asarray(region_normals[ib], dtype=float)
                if abs(float(na @ nb)) < math.cos(math.radians(15.0)):
                    continue
            res = _union_fit(np.array(groups[ia]), np.array(groups[ib]),
                             fac, pts, normals, plane_tol, curved_tol, sigma,
                             ka=ka, kb=kb)
            if res is None:
                continue
            kind, fitres, rms = res
            if best is None or rms < best[0]:
                best = (rms, ia, ib, kind, fitres)
        if best is not None:
            rms, ia, ib, kind, fitres = best
            merge_log.append({"merged": [ia, ib], "kind": kind, "rms": round(rms, 6)})
            changed = True
            groups[ia] = groups[ia] + groups[ib]
            groups[ib] = []
            kinds[ia] = kind
            # the merged region inherits its partner's adjacency: remap ib -> ia
            remapped = set()
            for x, y in adjacency_pairs:
                nx = ia if x == ib else x
                ny = ia if y == ib else y
                if nx != ny:
                    remapped.add((min(nx, ny), max(nx, ny)))
            adjacency_pairs = sorted(remapped)
    pairs = [(g, k) for g, k in zip(groups, kinds) if g]
    return [g for g, _ in pairs], [k for _, k in pairs], merge_log
