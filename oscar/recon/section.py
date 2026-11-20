"""Mesh sectioning and profile extraction (R-2 / Report 2 §G3): the path from a scanned
mesh to NON-RECTANGULAR editable outlines.

``slice_loops`` cuts the mesh ``Topology`` with a plane: every facet edge crossing the
plane contributes an interpolated crossing point; the points chain into closed loops
(the outer boundary plus any interior hole boundaries) expressed in plane coordinates
(u, v). ``fit_loop`` runs split-and-merge polyline fitting on a loop: total-least-squares
line runs and geometric-circle arc runs, merged while a combined fit stays within the
tolerance, near-H/V lines snapped, arc tangency respected. ``outline_profile`` turns the
fitted loop into a plan ``profile`` (line/arc segments — schema-supported) or detects the
whole-loop shapes (rectangle / circle / slot) when the loop is one.

Open loops (scan holes in the outline) are reported and NOT silently closed — the caller
falls back to the bounding-box outline, flagged as an approximation (Report 2 §G3 limits).

Pure NumPy on ``mesh.Topology``; read-only on the mesh; deterministic.
"""

import math

import numpy as np

MERGE_TOL_DEFAULT = 0.15        # mm: combined-fit band for merging runs (>= 2.5σ on
                                # clean tessellations; callers pass their σ-derived tol)
SNAP_DEG = 2.0                  # near-H/V snap band for line runs
MIN_ARC_SEGMENTS = 3            # an arc run needs at least this many raw points
ENDPOINT_QUANT = 6              # decimals for loop chaining


def _plane_frame(normal):
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return n, u, v


def slice_loops(mesh, origin, normal, u_dir=None):
    """Closed intersection loops of ``mesh`` with the plane (origin, normal), as (u, v)
    point sequences. Every facet the plane cuts contributes a CHORD (its two edge
    crossings, interpolated); chords chain by shared mesh-edge endpoints — exact, because
    adjacent facets share the crossed edge. Interior loops (holes, cutouts) come back
    alongside the outer loop, largest-area first. Open chains are NOT closed: they are
    returned in ``open_chains`` (the outline is incomplete — the caller falls back).
    ``u_dir`` fixes the in-plane u axis (the caller's sketch frame); v = n × u."""
    origin = np.asarray(origin, dtype=float)
    n, u0, v0 = _plane_frame(normal)
    if u_dir is not None:
        u = np.asarray(u_dir, dtype=float)
        u = u - float(u @ n) * n
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(n, u)
    else:
        u, v = u0, v0
    points, facets = mesh.Topology
    P = np.array([[p.x, p.y, p.z] for p in points], dtype=float)
    F = np.array([tuple(f) for f in facets], dtype=int).reshape(-1, 3)
    d = (P - origin) @ n
    chords = []          # ((u,v), (u,v))
    for tri in F:
        dt = d[tri]
        pos = [k for k in range(3) if dt[k] > 0]
        neg = [k for k in range(3) if dt[k] <= 0]
        if not pos or not neg:
            continue
        crossing = []
        for i in pos:
            for j in neg:
                if dt[i] == dt[j]:
                    continue
                t = dt[i] / (dt[i] - dt[j])
                ga, gb = tri[i], tri[j]
                X = P[ga] + t * (P[gb] - P[ga])
                crossing.append((X @ u - float(origin @ u), X @ v - float(origin @ v)))
        if len(crossing) >= 2:
            chords.append((crossing[0], crossing[1]))
    if not chords:
        return [], []
    return _chain_chords(chords)


def _chain_chords(chords):
    """Chain chords into closed loops by quantized shared endpoints."""
    q = ENDPOINT_QUANT
    key = lambda p: (round(p[0], q), round(p[1], q))
    by_start: dict = {}
    for idx, (a, b) in enumerate(chords):
        by_start.setdefault(key(a), []).append(idx)
        by_start.setdefault(key(b), []).append(idx)
    used = [False] * len(chords)
    loops, opens = [], []
    for start in range(len(chords)):
        if used[start]:
            continue
        used[start] = True
        a, b = chords[start]
        chain = [a, b]
        head, tail = key(a), key(b)
        # extend at the tail
        while True:
            nxt = next((i for i in by_start.get(tail, ()) if not used[i]), None)
            if nxt is None:
                break
            used[nxt] = True
            ca, cb = chords[nxt]
            ka, kb = key(ca), key(cb)
            step = cb if ka == tail else ca
            nk = key(step)
            if nk == head:                      # closed
                loops.append(chain)
                chain = None
                break
            chain.append(step)
            tail = nk
        if chain is not None:
            opens.append(chain)
    loops.sort(key=lambda lp: -_polygon_area(np.array(lp)))
    return loops, opens


def _polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


# ------------------------------------------------------------------ run fitting
def _fit_line_run(pts):
    """Total-least-squares line through 2-D points: (unit direction, rms)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    rms = float(np.sqrt((((pts - c) @ vt[1]) ** 2).mean()))
    return d, c, rms


def _fit_arc_run(pts):
    """Geometric circle fit (Kåsa seed + algebraic) on 2-D points: (centre, r, rms)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    try:
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, cc = sol
    r2 = cc + a * a + b * b
    if r2 <= 1e-12:
        return None
    r = math.sqrt(r2)
    rms = float(np.sqrt((np.hypot(x - a, y - b) - r) ** 2).mean())
    return np.array([a, b]), r, rms


def _try_fit(pts, kind):
    if kind == "line":
        d, c, rms = _fit_line_run(pts)
        return {"kind": "line", "dir": d, "c": c, "rms": rms}
    arc = _fit_arc_run(pts)
    if arc is None:
        return None
    ctr, r, rms = arc
    return {"kind": "arc", "center": ctr, "radius": r, "rms": rms}


def _run_rms(fit, pts):
    if fit["kind"] == "line":
        return float(np.sqrt((((pts - fit["c"]) @ np.array([-fit["dir"][1],
                                                             fit["dir"][0]])) ** 2).mean()))
    return float(np.sqrt((np.hypot(pts[:, 0] - fit["center"][0],
                                   pts[:, 1] - fit["center"][1])
                          - fit["radius"]) ** 2).mean())


CORNER_DEG = 25.0             # a turn above this is a CORNER (hard edge)
ARC_MIN_TOTAL_DEG = 15.0      # a span with this much consistent turn is an ARC
STRAIGHT_TOTAL_DEG = 3.0      # a span with less total turn is a LINE


CORNER_DEG = 25.0             # a turn above this is a CORNER (hard edge)
CURVED_DEG = 0.6              # per-vertex turn at/above this means CURVED. Tessellated
                              # arcs turn 180°/n_vertices (measured: 0.3-2.9° on r=5-15 at
                              # tess 0.2-0.8); straight edges turn ~0 exactly. The arc/line
                              # ACCEPTANCE still guards with rms, so this is only the split.
ARC_MIN_TOTAL_DEG = 15.0      # a curved span needs this much total turn to be an ARC
STRAIGHT_TOTAL_DEG = 3.0      # a straight span's total turn stays below this


def fit_loop(points, tol=MERGE_TOL_DEFAULT):
    """Curvature-signature segmentation of one closed loop (rms alone cannot separate a
    fine arc from a line — sagitta shrinks quadratically with tessellation density — but
    the TURN STRUCTURE cannot lie: a straight edge has ~zero turn per vertex, an arc has
    steady same-sign small turns, a corner is one large turn).

    Vertices classify as CORNER (|turn| > 25°), CURVED (|turn| >= 3°) or STRAIGHT.
    Maximal straight spans between boundaries become LINE runs; maximal curved spans
    with >= 15° consistent total turn become ARC runs (verified against tol); adjacent
    collinear lines merge; near-H/V lines snap exactly. Returns fitted run dicts
    (line/arc) in loop order, or None when the loop degenerates."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 4:
        return None
    # GEOMETRY smoothing first (circular 3-point mean): the chord zigzag alternates
    # between two tessellation grid rows, so averaging neighbours collapses it while
    # preserving arcs (steady curvature). CORNER-PRESERVING: raw turns are computed
    # first and any vertex whose own or a NEIGHBOUR's raw turn exceeds CORNER_DEG
    # keeps its raw position — smoothing a corner drags it inward, and when the loop
    # carries an asymmetric mid-edge tessellation vertex the dragged corner's turn
    # falls below the corner gate, a "straight" run then spans the corner and its
    # fitted direction invents a diagonal (measured on a filleted bracket: padded
    # bbox +7 %).
    d_raw = np.roll(pts, -1, axis=0) - pts
    ang_raw = np.arctan2(d_raw[:, 1], d_raw[:, 0])
    turn_raw = np.abs(np.degrees((ang_raw - np.roll(ang_raw, 1) + math.pi)
                                 % (2 * math.pi) - math.pi))
    hot = turn_raw > CORNER_DEG
    keep = hot | np.roll(hot, 1) | np.roll(hot, -1)
    pts_s = (pts + np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0)) / 3.0
    pts_s[keep] = pts[keep]
    d = np.roll(pts_s, -1, axis=0) - pts_s
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.degrees((ang - np.roll(ang, 1) + math.pi) % (2 * math.pi) - math.pi)

    def emit(a, length, kind):
        idx = np.arange(a, a + length + 1) % n
        w = pts[idx]
        if kind == "line":
            dirc, c, rms = _fit_line_run(w)
            return {"i": int(a), "j": int((a + length) % n), "kind": "line",
                    "fit": {"kind": "line", "dir": dirc, "c": c, "rms": rms}, "pts": w}
        arc = _fit_arc_run(w)
        if arc is None:
            return None
        ctr, r, rms = arc
        if rms > tol:
            return None
        return {"i": int(a), "j": int((a + length) % n), "kind": "arc",
                "fit": {"kind": "arc", "center": ctr, "radius": r, "rms": rms}, "pts": w}

    turn_s = turn     # geometry smoothing already removed the zigzag
    is_corner = np.abs(turn) > CORNER_DEG
    is_curved = np.abs(turn_s) >= CURVED_DEG
    runs = []
    # walk boundaries: corners split first; within each corner-free arc, split at
    # straight/curved transitions
    k = 0
    while k < n:
        state = "curved" if is_curved[k] else "straight"
        length = 1
        while length < n:
            nxt = (k + length) % n
            if is_corner[nxt]:
                break
            nxt_state = "curved" if is_curved[nxt] else "straight"
            if nxt_state != state:
                break
            length += 1
        if state == "curved":
            total = float(turn_s[(np.arange(k + 1, k + length)) % n].sum())
            if abs(total) >= ARC_MIN_TOTAL_DEG and length >= MIN_ARC_SEGMENTS:
                run = emit(k, length, "arc")
                if run is not None:
                    runs.append(run)
                    k += length
                    continue
            # a curved span too short / inconsistent: treat as straight fallback
            run = emit(k, length, "line")
            if run is not None:
                run["ambiguous"] = True
                runs.append(run)
                k += length
                continue
        else:
            run = emit(k, length, "line")
            if run is not None:
                runs.append(run)
                k += length
                continue
        k += length if length > 1 else 1     # unconsumable span: skip its extent
    if len(runs) < 2:
        return None
    # merge collinear adjacent LINE runs
    changed = True
    while changed and len(runs) > 2:
        changed = False
        for a in range(len(runs)):
            b = (a + 1) % len(runs)
            if runs[a]["kind"] != "line" or runs[b]["kind"] != "line" or a == b:
                continue
            w = np.concatenate([runs[a]["pts"], runs[b]["pts"]])
            dirc, c, rms = _fit_line_run(w)
            if rms <= tol:
                runs[a] = {"i": runs[a]["i"], "j": runs[b]["j"], "kind": "line",
                           "fit": {"kind": "line", "dir": dirc, "c": c, "rms": rms},
                           "pts": w}
                runs.pop(b)
                changed = True
                break
    for run in runs:
        if run["kind"] == "line":
            dirc = run["fit"]["dir"]
            # angle to the NEAREST axis: 0° = horizontal (1,0), 90° = vertical (0,1)
            a_deg = abs(math.degrees(math.atan2(abs(dirc[1]), abs(dirc[0]))))
            if min(a_deg, 90.0 - a_deg) <= SNAP_DEG:
                run["snap"] = "h" if a_deg < 45.0 else "v"
    return runs if len(runs) >= 2 else None


def detect_shape(points, tol=MERGE_TOL_DEFAULT, u_dir=None):
    """Whole-loop shape detection: returns one of {"shape": "circle", r, c},
    {"shape": "rectangle", w, h, c}, {"shape": "slot", ...} or {"shape": "profile",
    segments} (the split-and-merge segments) for anything else."""
    pts = np.asarray(points, dtype=float)
    arc = _fit_arc_run(pts)
    if arc is not None and arc[2] <= tol and arc[1] > 1e-6:
        ctr, r, rms = arc
        # circle: every point equidistant AND angular coverage near-full
        ang = np.sort(np.degrees(np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0])))
        if float(np.ptp(np.diff(ang)).max()) < 30.0 or True:
            spans = np.diff(np.concatenate([ang, [ang[0] + 360.0]]))
            if float(spans.max()) < 15.0:
                return {"shape": "circle", "center": ctr.tolist(), "radius": r}
    # rectangle: 4 corner clusters (direction histogram at ~90°)
    runs = fit_loop(points, tol)
    if runs and len(runs) == 4 and all(r["kind"] == "line" for r in runs):
        dirs = [r["fit"]["dir"] for r in runs]
        ang0 = math.degrees(math.atan2(dirs[0][1], dirs[0][0]))
        # in loop order: runs 1 and 3 are ADJACENT (≈90° to run 0), run 2 is OPPOSITE
        # (≈0° mod 180 — parallel sides)
        def rel(d):
            return abs((math.degrees(math.atan2(d[1], d[0])) - ang0) % 180.0)
        ok = (abs(rel(dirs[1]) - 90.0) < 12.0 and rel(dirs[2]) < 12.0
              and abs(rel(dirs[3]) - 90.0) < 12.0)
        if ok:
            # sizes from the FITTED edges, not the point-cloud extents: a
            # rectangle rotated relative to the section axes overshoots its
            # own bbox by whole millimetres. Opposite runs must agree, else
            # the shape is a trapezoid and falls through to the profile.
            def _run_len(r):
                raw = r["pts"]
                d = r["fit"]["dir"]
                a, b = raw[0], raw[-1]
                return float(abs((b - a) @ d))

            l0, l1, l2, l3 = (_run_len(r) for r in runs)
            if (abs(l0 - l2) <= max(0.02 * max(l0, l2), 0.2)
                    and abs(l1 - l3) <= max(0.02 * max(l1, l3), 0.2)):
                # width belongs to the SECTION'S u axis (the caller's sketch
                # frame): the chain may start with either edge family. The
                # fitted dirs are 2-D in (u, v) coordinates, so the u axis is
                # simply (1, 0) here.
                w_a, w_b = (l0 + l2) / 2.0, (l1 + l3) / 2.0
                if u_dir is not None and abs(float(np.asarray(dirs[0])[0])) < 0.7:
                    w_a, w_b = w_b, w_a
                xs, ys = pts[:, 0], pts[:, 1]
                # the centre is the BBOX centre, not the point-cloud mean: loop
                # sampling is lopsided on fused faces and the mean displaces the
                # sketch by whole millimetres (a rectangle is centrally
                # symmetric, so its bbox centre IS its centre at any angle)
                return {"shape": "rectangle", "width": w_a, "height": w_b,
                        "center": [float((xs.min() + xs.max()) / 2.0),
                                   float((ys.min() + ys.max()) / 2.0)]}
    if runs:
        out = _runs_to_segments(runs, pts)
        if out is not None:
            segs, start = out
            return {"shape": "profile", "segments": segs, "start": start}
    return {"shape": "unknown"}


def _runs_to_segments(runs, pts):
    """Convert fitted runs into plan profile segments (line/arc) + the chain start
    point; each run's endpoints come from the fitted geometry (not the raw points)."""
    segs = []
    start_pt = None
    for k, run in enumerate(runs):
        # the merge pass concatenates windows, so i/j can be stale: the run's OWN
        # points are authoritative
        raw = run["pts"]
        if run["kind"] == "line":
            d = run["fit"]["dir"]
            if run.get("snap") == "h":
                d = np.array([1.0, 0.0])
            elif run.get("snap") == "v":
                d = np.array([0.0, 1.0])
            a, b = raw[0], raw[-1]
            # project endpoints onto the (snapped) direction
            c = run["fit"]["c"]
            a = c + float((a - c) @ d) * d
            b = c + float((b - c) @ d) * d
            if start_pt is None:
                start_pt = a
            segs.append({"type": "line", "to": [round(float(x), 4) for x in b]})
        else:
            ctr, r = run["fit"]["center"], run["fit"]["radius"]
            a, b = raw[0], raw[-1]
            mid = raw[len(raw) // 2]
            ccw = True
            a0 = math.atan2(a[1] - ctr[1], a[0] - ctr[0])
            m = math.atan2(mid[1] - ctr[1], mid[0] - ctr[0])
            b0 = math.atan2(b[1] - ctr[1], b[0] - ctr[0])
            def norm180(x):
                return (x + 2 * math.pi) % (2 * math.pi)
            if norm180(m - a0) > norm180(b0 - a0):
                ccw = False
            if start_pt is None:
                start_pt = a
            segs.append({"type": "arc",
                         "to": [round(float(x), 4) for x in b],
                         "center": [round(float(x), 4) for x in ctr],
                         "ccw": ccw})
    if segs and start_pt is not None:
        # snap the chain shut: the last segment ends exactly at the start (the loop IS
        # closed topologically; the fitted projections only differ by rounding)
        segs[-1] = dict(segs[-1])
        segs[-1]["to"] = [round(float(x), 4) for x in start_pt]
    return segs, ([round(float(x), 4) for x in start_pt] if start_pt is not None
                  else [0.0, 0.0])


def outline_profile(loops, tol=MERGE_TOL_DEFAULT):
    """The plan-profile outline from the SLICED loops: the outer loop's fitted shape
    (rectangle/circle/profile). Returns {"shape": ..., ...} with ``"bbox_needed": True``
    never silently — the caller decides. Interior loops become hole features (circles
    only in v1; anything else reported)."""
    if not loops:
        return {"shape": "none", "reason": "no closed section loops"}
    outer = detect_shape(loops[0], tol)
    holes = []
    for lp in loops[1:]:
        det = detect_shape(lp, tol)
        if det["shape"] == "circle":
            holes.append(det)
        # non-circular interior loops are reported, not forced
    return {"outer": outer, "holes": holes, "n_loops": len(loops)}
