"""Parametric reconstruction orchestration (M4/M5): diagnose → (safe-)repair a copy →
recognize → choose a route from evidence → synthesize a FeaturePlan → build it with the
EXISTING builder → measure deviation against the source mesh → report an honest outcome.

v1 routes: ``prismatic`` (base plate + measured through-holes; the bracket family). Every
other class comes back with ``supported: False`` and the survey evidence — never a forced
approximation labelled as a reconstruction. FreeCAD module (builder access); call on the GUI
thread in GUI sessions (headless/console runs inline).
"""

import json
import math
import os

import numpy as np

from . import meshops, recognize, section as _section
from ..plan import builder

EXTRA_VOLUME_TOLERANCE = 0.10

# P3.2: bounded LRU of per-work-mesh stage results (see reconstruct()); the
# version guards against algorithm changes invalidating old entries
_SURVEY_CACHE_VERSION = 2
_SURVEY_CACHE: dict = {}


def _basis(normal):
    n = np.asarray(normal, dtype=float)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(n @ ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return n, u, v


def _region_plane(survey, normal, point):
    """The plane REGION (with its centroid point) matching ``normal``/``point``."""
    n = np.asarray(normal, dtype=float)
    best, best_dot = None, -1.0
    for region in survey["regions"]:
        if region["type"] != "plane":
            continue
        rn = np.asarray(region["normal"], dtype=float)
        dot = abs(float(rn @ n))
        if dot > best_dot:
            best, best_dot = region, dot
    return best if best_dot > math.cos(math.radians(10.0)) else None


def _source(text):
    return {"kind": "scan_fit", "ref": text}


# P1.1: parameter / feature ids per perimeter edge rule. The ids follow the RULE
# (not the far/base faces) so a plate built upside down keeps meaningful names.
_RULE_IDS = {"top_outer": ("fillet_r", "fillet_top", "chamfer_w", "chamfer_top"),
             "bottom_outer": ("fillet_rb", "fillet_bot", "chamfer_wb", "chamfer_bot")}


def _plan_prismatic(survey, diag, mesh=None):
    """Synthesize the FeaturePlan for the prismatic (extrusion) route from the survey."""
    pair = survey["parallel_planes"][0]
    n, u, v = _basis(pair["normal"])
    perp = [r for r in survey["regions"] if r["type"] == "plane"
            and abs(float(np.asarray(r["normal"], dtype=float) @ n)) > math.cos(math.radians(10.0))]
    if not perp:
        return None, {"reason": "the parallel face pair could not be tied to plane regions"}
    base = None
    if mesh is not None:
        # P1.2: anchor the frame at the OUTERMOST perpendicular plane with the
        # normal oriented so the bulk of the mesh sits on the +n side — with
        # stacked levels, anchoring on a middle face (a normal-alignment tie
        # between parallel planes) grows the stack the wrong way and mirrors u
        P = np.array([[q.x, q.y, q.z] for q in mesh.Topology[0]], dtype=float)
        if float(np.dot(P.mean(axis=0)
                        - np.asarray(perp[0]["point"], dtype=float), n)) < 0:
            n = -n
        u, v = _basis(n)[1:]
        base = min(perp, key=lambda r: float(np.dot(
            np.asarray(r["point"], dtype=float), n)))
    else:
        base = _region_plane(survey, pair["normal"], None)
        other = next((r for r in perp if r is not base), None)
        if base is None or other is None:
            return None, {"reason": "the parallel face pair could not be tied to plane regions"}
        if float(np.dot(np.asarray(other["point"], dtype=float)
                        - np.asarray(base["point"], dtype=float), n)) < 0:
            n = -n
            u, v = _basis(n)[1:]
    base_origin = np.asarray(base["point"], dtype=float)
    thickness = pair["distance_mm"]

    box = diag.get("size_mm") or [0, 0, 0]

    def extent_along(direction) -> float:
        """The scan bbox's extent along ``direction`` (exact for box-like footprints)."""
        return sum(abs(float(direction[k])) * box[k] for k in range(3))

    extent_u, extent_v = extent_along(u), extent_along(v)

    # P1.2 LEVEL DECOMPOSITION: every significant plane ~perpendicular to n is
    # a candidate LEVEL; each band between consecutive levels gets its own
    # measured section, so stacked geometry (an L-bracket's wall, a boss, a
    # stepped plate) builds as stacked pads instead of being reported as
    # unexplained extra volume. Single-level parts (a plain plate) produce one
    # band and behave exactly as the old single-section route.
    levels = [0.0, float(thickness)]
    for r in survey["regions"]:
        if r["type"] != "plane":
            continue
        if abs(float(np.asarray(r["normal"], dtype=float) @ n)) < math.cos(math.radians(10.0)):
            continue
        h = float(np.dot(np.asarray(r["point"], dtype=float) - base_origin, n))
        if h <= 0.1 or r["area_mm2"] < max(0.02 * float(base["area_mm2"]), 25.0):
            continue                    # hole floors, slivers and rims are not levels
        if not any(abs(h - h0) <= 0.1 for h0 in levels):
            levels.append(h)
    levels.sort()
    total_h = levels[-1]

    def _loop_area(lp) -> float:
        p = np.asarray(lp, dtype=float)
        if len(p) < 3:
            return 0.0
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    bands = []
    if mesh is not None:
        for lo, hi in zip(levels, levels[1:]):
            if hi - lo < 0.1:
                continue
            try:
                loops, opens = _section.slice_loops(
                    mesh, base_origin + (0.5 * (lo + hi)) * n, n, u_dir=u)
            except Exception:  # noqa: BLE001 - one band's section must never break the route
                loops, opens = [], 0
            ol = _section.detect_shape(loops[0], u_dir=u) if loops else None
            if ol is not None:
                ol["n_open"] = len(opens)
            bands.append({"lo": lo, "hi": hi, "loops": loops, "outline": ol,
                          "area": _loop_area(loops[0]) if loops else None})

    def _same_footprint(b1, b2) -> bool:
        """Adjacent bands with the SAME outer footprint AND the same interior
        loops are one through-region (an internal level split it): rectangles /
        circles compare by size, profiles conservatively never merge. Different
        interiors mean a recess starts — the bands must stay separate so the
        pocket gets its true floor level."""
        o1, o2 = b1["outline"], b2["outline"]
        if not o1 or not o2 or o1.get("shape") != o2.get("shape"):
            return False
        if o1["shape"] == "rectangle":
            same = (abs(o1["width"] - o2["width"])
                    <= max(0.02 * max(o1["width"], o2["width"]), 0.2)
                    and abs(o1["height"] - o2["height"])
                    <= max(0.02 * max(o1["height"], o2["height"]), 0.2))
        elif o1["shape"] == "circle":
            same = abs(o1["radius"] - o2["radius"]) <= max(
                0.02 * max(o1["radius"], o2["radius"]), 0.2)
        else:
            return False

        def _interior_sig(band):
            out = []
            for lp in band["loops"][1:]:
                d = _section.detect_shape(lp)
                if d is None or d.get("shape") == "unknown":
                    out.append(("unknown", round(_loop_area(lp), 0)))
                elif d["shape"] == "circle":
                    out.append(("circle", round(d["radius"], 1),
                                round(d["center"][0], 1), round(d["center"][1], 1)))
                else:
                    out.append((d["shape"], round(_loop_area(lp), 0)))
            return sorted(out)

        return same and _interior_sig(b1) == _interior_sig(b2)

    merged = []
    for b in bands:
        if merged and _same_footprint(merged[-1], b):
            merged[-1]["hi"] = b["hi"]
        else:
            merged.append(dict(b))
    bands = merged or [{"lo": 0.0, "hi": float(thickness), "loops": [],
                        "outline": None, "area": None}]
    outline = bands[0]["outline"]

    # a hole must fit INSIDE the plate: its radius is bounded by the smaller IN-PLANE
    # extent (the old min(bbox) bound wrongly used the thickness for plates)
    in_plane = max(min(extent_u, extent_v), 1e-9)
    holes = [h for h in survey["holes"]
             if abs(float(np.dot(np.asarray(h["axis"]), n))) > math.cos(math.radians(10.0))
             and h["radius"] < 0.45 * in_plane
             and (h["through"] or h.get("depth"))]     # through OR measured blind depth
    holes.sort(key=lambda h: (round(h["center"][0], 3), round(h["center"][1], 3), h["center"][2]))

    params: dict = {}
    features: list = []
    skipped_bands = []
    pad_count = 0
    for bi, band in enumerate(bands, start=1):
        lo, hi, ol = band["lo"], band["hi"], band["outline"]
        h_mm = round(hi - lo, 3)
        at = base_origin + lo * n
        t_src = (_source(f"distance between the two largest parallel planes "
                         f"({pair['area_mm2']} mm2 pair)") if bi == 1
                 else _source("distance between adjacent perpendicular face levels"))
        if ol is not None and ol.get("shape") == "rectangle":
            lp = "plate_len" if bi == 1 else f"pad{bi}_len"
            wp = "plate_wid" if bi == 1 else f"pad{bi}_wid"
            tp = "plate_t" if bi == 1 else f"pad{bi}_t"
            src = _source("section outline through the plate (true measured extent)"
                          if bi == 1 else
                          f"section outline through the level band {lo:.2g}..{hi:.2g} mm")
            params[lp] = {"value": round(float(ol["width"]), 3), "source": src}
            params[wp] = {"value": round(float(ol["height"]), 3), "source": src}
            params[tp] = {"value": h_mm, "source": t_src}
            c = ol["center"]                 # re-centre the sketch on the section
            sk = at + float(c[0]) * u + float(c[1]) * v
            profile = {"shape": "rectangle", "width": lp, "height": wp}
        elif ol is not None and ol.get("shape") == "profile":
            tp = "plate_t" if bi == 1 else f"pad{bi}_t"
            params[tp] = {"value": h_mm, "source": t_src}
            sk = at
            profile = {"shape": "profile", "start": _profile_start(ol),
                       "segments": list(ol["segments"])}
        elif ol is not None and ol.get("shape") == "circle":
            dp = "plate_d" if bi == 1 else f"pad{bi}_d"
            tp = "plate_t" if bi == 1 else f"pad{bi}_t"
            params[dp] = {"value": round(2.0 * float(ol["radius"]), 3),
                          "source": _source("circular section outline (measured)")}
            params[tp] = {"value": h_mm, "source": t_src}
            c = ol["center"]
            sk = at + float(c[0]) * u + float(c[1]) * v
            profile = {"shape": "circle", "diameter": dp}
        elif bi == 1:
            # the old flagged bbox approximation — kept for the FIRST band only
            params["plate_len"] = {"value": round(float(extent_u), 3),
                                   "source": _source("APPROXIMATION: plate length from the scan bounding box (no closed section outline)")}
            params["plate_wid"] = {"value": round(float(extent_v), 3),
                                   "source": _source("APPROXIMATION: plate width from the scan bounding box (no closed section outline)")}
            params["plate_t"] = {"value": h_mm, "source": t_src}
            sk = at
            profile = {"shape": "rectangle", "width": "plate_len", "height": "plate_wid"}
        else:
            skipped_bands.append({"lo": round(lo, 2), "hi": round(hi, 2),
                                  "reason": "no closed section outline in the band"})
            continue
        sid = "base_sk" if bi == 1 else f"pad{bi}_sk"
        fid = "base_pad" if bi == 1 else f"pad{bi}"
        features.append({"id": sid, "op": "sketch",
                         "plane": {"normal": n.tolist(), "origin": np.asarray(sk).tolist(),
                                   "x_dir": u.tolist()},
                         "profile": profile})
        features.append({"id": fid, "op": "pad", "sketch": sid, "length": tp})
        pad_count += 1
    # ---- P1.3: repeated-hole patterns ----
    # Equal-radius groups (same kind/depth) are tested for a COLLINEAR
    # equal-spacing row (>= 3) and for EQUAL ANGULAR STEPS on a circle (>= 4).
    # A patterned group emits ONE Hole plus a native LinearPattern / PolarPattern
    # (both bound to editable count/spacing parameters); irregular arrangements
    # keep their individual holes — nothing is forced onto weak evidence.
    hole_xy = [(float(np.dot(np.asarray(h["center"], dtype=float) - base_origin, u)),
                float(np.dot(np.asarray(h["center"], dtype=float) - base_origin, v)))
               for h in holes]

    def _kind(h):
        return (h.get("kind") or ("through" if h.get("through") else "blind"),
                round(float(h.get("depth") or 0.0), 1))

    def _axis_token(world_vec, signed=True):
        w = np.asarray(world_vec, dtype=float)
        norm = float(np.linalg.norm(w))
        if norm < 1e-9:
            return None
        w = w / norm
        for name, ax in (("x", np.array([1.0, 0, 0])), ("y", np.array([0, 1.0, 0])),
                         ("z", np.array([0, 0, 1.0]))):
            dot = float(w @ ax)
            if abs(dot) >= math.cos(math.radians(10.0)):
                if not signed or dot > 0:
                    return name
                return "-" + name
        return None

    groups: list = []
    for i, h in enumerate(holes):
        placed = False
        for g in groups:
            r_ref = holes[g[0]]["radius"]
            if (_kind(holes[g[0]]) == _kind(h)
                    and abs(h["radius"] - r_ref) <= 0.05 * max(r_ref, 1e-9)):
                g.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])

    def _best_line(idxs):
        """The longest collinear equally-spaced chain (>= 3) in the group."""
        best = None
        pts = [(i, hole_xy[i]) for i in idxs]
        for a in range(len(pts)):
            for b in range(len(pts)):
                if a == b:
                    continue
                pa, pb = np.asarray(pts[a][1]), np.asarray(pts[b][1])
                d = pb - pa
                length = float(np.linalg.norm(d))
                if length < 1e-6:
                    continue
                dh = d / length
                tol = max(0.15, 0.01 * length)
                members = []
                for i, p in pts:
                    q = np.asarray(p) - pa
                    along = float(q @ dh)
                    if abs(float(q[0] * dh[1] - q[1] * dh[0])) <= tol                and -tol <= along <= length + tol:
                        members.append((along, i))
                if len(members) < 3:
                    continue
                members.sort()
                gaps = [members[k + 1][0] - members[k][0] for k in range(len(members) - 1)]
                if max(gaps) - min(gaps) > max(0.2, 0.03 * max(gaps)):
                    continue                        # spacing must be a real pitch
                if best is None or len(members) > len(best[0]):
                    best = (members, dh)
        return best

    def _circle(idxs):
        """Equal angular steps on a common circle (>= 4) — a bolt circle."""
        if len(idxs) < 4:
            return None
        P = np.asarray([hole_xy[i] for i in idxs], dtype=float)
        A = np.hstack([2 * P, np.ones((len(P), 1))])
        try:
            sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(axis=1), rcond=None)
        except Exception:  # noqa: BLE001
            return None
        c = np.asarray(sol[:2], dtype=float)
        radii = np.linalg.norm(P - c, axis=1)
        r_mean = float(radii.mean())
        if r_mean <= 0 or float(radii.max() - radii.min()) > max(0.25, 0.03 * r_mean):
            return None
        ang = np.sort(np.degrees(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])))
        steps = np.diff(np.concatenate([ang, [ang[0] + 360.0]]))
        if float(steps.max() - steps.min()) > max(3.0, 0.06 * float(steps.mean())):
            return None
        order = np.argsort(np.degrees(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])))
        return idxs[order[0]], float(steps.mean()) * len(idxs), c, float(steps.max() - steps.min())

    patterns = []          # {"kind", "members", "orig", "direction"/"axis", ...}
    consumed: set = set()
    for g in groups:
        if len(g) < 3:
            continue
        # a full bolt circle explains the whole group; otherwise try a row
        circ = _circle(g) if len(g) >= 4 else None
        line = None if circ is not None else _best_line(g)
        if circ is not None:
            axis = _axis_token(n, signed=False)
            if axis:
                patterns.append({"kind": "polar", "members": list(g),
                                 "orig": circ[0], "axis": axis,
                                 "center": (float(circ[2][0]), float(circ[2][1])),
                                 "sweep": round(min(circ[1], 360.0), 2),
                                 "step_spread": round(circ[3], 2)})
                consumed.update(g)
        elif line is not None:
            members, dh = line
            world = dh[0] * u + dh[1] * v
            tok = _axis_token(world)
            if tok:
                gaps = [members[k + 1][0] - members[k][0] for k in range(len(members) - 1)]
                patterns.append({"kind": "linear", "members": [i for _, i in members],
                                 "orig": members[0][1], "direction": tok,
                                 "spacing": round(float(np.mean(gaps)), 3),
                                 "count": len(members)})
                consumed.update(i for _, i in members)
    polar_orig = {p["orig"]: (p, pi) for pi, p in enumerate(patterns, start=1)
                  if p["kind"] == "polar"}
    same_d = len({round(h["radius"], 1) for h in holes}) <= 1
    if same_d and holes:
        params["hole_d"] = {"value": round(float(np.mean([h["radius"] for h in holes])) * 2, 3),
                            "source": _source("cylinder fit of the hole walls (least squares, full-turn wall)")}
    for i, hole in enumerate(holes, start=1):
        if i - 1 in consumed and all(p["orig"] != i - 1 for p in patterns):
            continue                       # a pattern member: the pattern builds it
        c = np.asarray(hole["center"], dtype=float)
        x, y = float(np.dot(c - base_origin, u)), float(np.dot(c - base_origin, v))
        dia = "hole_d" if same_d else f"hole_d{i}"

        def _opening_radius(face_pt, r_bore):
            """Measured material boundary on ONE face plane at this hole: ring
            probes outward from the bore (exact point-to-mesh) until material
            appears. Returns (flare_mm, opening_diameter) — (0, 0) when the
            opening matches the bore (no rim blend). Case-07 evidence: the
            benchmark truth filleted the hole RIMS too, and the blend merged
            into the recognized wall region, so it must be measured here."""
            if mesh is None or r_bore <= 0:
                return 0.0, 0.0
            try:
                from .deviation import _points_to_mesh_exact
            except Exception:  # noqa: BLE001 - no probe without the engine
                return 0.0, 0.0
            ref = np.array([1.0, 0.0, 0.0])
            if abs(float(n @ ref)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            pu = np.cross(n, ref)
            pu = pu / (np.linalg.norm(pu) + 1e-12)
            pv = np.cross(n, pu)
            for rr in np.arange(r_bore + 0.25, r_bore + 3.01, 0.25):
                pts = [tuple(face_pt + rr * (math.cos(a) * pu + math.sin(a) * pv))
                       for a in np.arange(0.0, 2.0 * math.pi, math.pi / 4.0)]
                d = _points_to_mesh_exact(pts, mesh)
                if d is None:
                    return 0.0, 0.0
                if sum(1 for q in d if float(q) <= 0.35) >= 5:
                    return float(rr - r_bore), float(2.0 * rr)
            return 0.0, 0.0

        rim = None
        if hole.get("through") and not (hole.get("counterbore")
                                        or hole.get("countersink")):
            f_base, d_base = _opening_radius(base_origin + x * u + y * v,
                                             hole["radius"])
            f_far, d_far = _opening_radius(base_origin + x * u + y * v
                                           + total_h * n, hole["radius"])
            for f, dd, far in ((f_base, d_base, False), (f_far, d_far, True)):
                if 0.4 <= f <= 0.8 * hole["radius"]:
                    rim = {"flare_mm": round(f, 3), "opening_d": round(dd, 3),
                           "far_face": far}
                    break
        if not same_d:
            params[dia] = {"value": round(hole["radius"] * 2, 3),
                           "source": _source("cylinder fit of the hole wall (least squares)")}
        polar = polar_orig.get(i - 1)
        if polar is not None:
            # the ORIGINAL of a bolt circle: its sketch sits at the circle's
            # CENTRE with the hole at radius R along the sketch x axis, so the
            # pattern can turn about the sketch normal (wherever the circle is)
            pat, pi = polar
            cu, cv = pat["center"]
            r_bc = math.hypot(x - cu, y - cv)
            x_dir = (x - cu) * u + (y - cv) * v
            xn = float(np.linalg.norm(x_dir))
            x_dir = x_dir / xn if xn > 1e-12 else u
            params[f"pat{pi}_r"] = {
                "value": round(r_bc, 3),
                "source": _source("bolt-circle radius (circle fit to the hole centres)")}
            params[f"hole_x{i}"] = {"value": round(cu, 3), "source": _source("bolt-circle centre (u) from the circle fit")}
            params[f"hole_y{i}"] = {"value": round(cv, 3), "source": _source("bolt-circle centre (v) from the circle fit")}
            px, py = cu, cv
            use_x_dir = x_dir
            pos_x = f"pat{pi}_r"
        else:
            params[f"hole_x{i}"] = {"value": round(x, 3), "source": _source("hole axis position from the cylinder fit")}
            params[f"hole_y{i}"] = {"value": round(y, 3), "source": _source("hole axis position from the cylinder fit")}
            px, py = x, y
            use_x_dir = u
            pos_x = 0.0
        top = base_origin + px * u + py * v + total_h * n
        # the drill enters from the face the scan's hole actually enters: with an
        # entry_point, compare it to the base plane (through holes are symmetric)
        sk_origin, sk_normal = top, n
        ep = hole.get("entry_point")
        if ep is not None and (not hole.get("through") or hole.get("counterbore")
                                or hole.get("countersink")):
            ep_v = np.asarray(ep, dtype=float)
            if float(np.dot(ep_v - base_origin, n)) < total_h / 2.0:
                # the drill enters through the BASE face: the sketch normal must point
                # OUT of the material (the Hole feature drills along -normal), and at
                # the base face the outward direction is -n
                sk_origin = base_origin + px * u + py * v
                sk_normal = -n
        elif hole.get("through") and not (hole.get("counterbore")
                                          or hole.get("countersink")):
            # a PLAIN through hole always drills from the frame's base face:
            # an exit-face sketch exposes an OCC ThroughAll quirk on downward
            # frames (a material plug below the exit plane)
            sk_origin = base_origin + px * u + py * v
            sk_normal = -n
        feat = {"id": f"hole_{i}", "op": "hole",
                "plane": {"normal": np.asarray(sk_normal).tolist(),
                          "origin": np.asarray(sk_origin).tolist(),
                          "x_dir": np.asarray(use_x_dir).tolist()},
                "position": {"x": pos_x, "y": 0.0}, "diameter": dia}
        if rim is not None:
            # a measured rim blend (the scan's opening exceeds the bore):
            # modelled as a countersink CONE of the measured opening — the
            # honest simplest native representation; the deviation rows judge it
            params[f"cs_d{i}"] = {"value": rim["opening_d"],
                                  "source": _source("hole-rim blend: measured "
                                                    "material opening on the "
                                                    "entry face (ring probes)")}
            params[f"cs_a{i}"] = {"value": 90.0,
                                  "source": _source("rim blend approximated by a "
                                                    "45 deg cone (fillet-torus "
                                                    "approximation)")}
            feat["countersink"] = {"cs_diameter": f"cs_d{i}",
                                   "cs_angle": f"cs_a{i}"}
            if rim["far_face"]:
                # the blend is on the FAR face: the drill must enter there
                feat["plane"] = {"normal": n.tolist(),
                                 "origin": np.asarray(top).tolist(),
                                 "x_dir": np.asarray(use_x_dir).tolist()}
        if hole.get("through"):
            feat["through"] = True
        elif hole.get("depth"):
            params[f"hole_dp{i}"] = {"value": round(float(hole["depth"]), 3),
                                     "source": _source(f"blind depth ({hole.get('depth_basis', 'measured')})")}
            feat["depth"] = f"hole_dp{i}"
        cut = hole.get("counterbore") or hole.get("countersink")
        if cut:
            if "counterbore" in hole:
                params[f"cb_d{i}"] = {"value": round(float(cut["diameter"]), 3),
                                      "source": _source("counterbore step cylinder fit (coaxial, both radii significant)")}
                params[f"cb_dp{i}"] = {"value": round(float(cut["depth"]), 3),
                                      "source": _source("counterbore step axial extent (measured)")}
                feat["counterbore"] = {"cb_diameter": f"cb_d{i}", "cb_depth": f"cb_dp{i}"}
            else:
                params[f"cs_d{i}"] = {"value": round(float(cut["diameter"]), 3),
                                      "source": _source("countersink cone fit at the entry (F-test gated vs sphere)")}
                if cut.get("angle"):
                    params[f"cs_a{i}"] = {"value": round(float(cut["angle"]), 1),
                                          "source": _source("countersink cone half-angle x2 (measured)")}
                    feat["countersink"] = {"cs_diameter": f"cs_d{i}",
                                           "cs_angle": f"cs_a{i}"}
                else:
                    feat["countersink"] = {"cs_diameter": f"cs_d{i}"}
        features.append(feat)
    # the pattern ops follow their original holes (a PartDesign Transformed
    # feature references an earlier feature)
    pattern_meta = []
    for pi, pat in enumerate(patterns, start=1):
        orig_id = f"hole_{pat['orig'] + 1}"
        if pat["kind"] == "linear":
            params[f"pat{pi}_n"] = {
                "value": pat["count"],
                "source": _source("collinear equal-spacing hole row (measured positions)")}
            params[f"pat{pi}_pitch"] = {
                "value": pat["spacing"],
                "source": _source("mean hole-to-hole pitch (measured)")}
            features.append({"id": f"pat{pi}", "op": "linear_pattern",
                             "of": orig_id, "count": f"pat{pi}_n",
                             "spacing": f"pat{pi}_pitch",
                             "direction": pat["direction"]})
            pattern_meta.append({"kind": "linear", "count": pat["count"],
                                 "direction": pat["direction"],
                                 "spacing_mm": pat["spacing"]})
        else:
            params[f"pat{pi}_n"] = {
                "value": len(pat["members"]),
                "source": _source("holes at equal angular steps on a fitted circle")}
            params[f"pat{pi}_sweep"] = {
                "value": pat["sweep"],
                "source": _source("total angular sweep of the bolt circle (measured)")}
            features.append({"id": f"pat{pi}", "op": "polar_pattern",
                             "of": orig_id, "count": f"pat{pi}_n",
                             "angle": f"pat{pi}_sweep", "axis": "sketch_normal"})
            pattern_meta.append({"kind": "polar", "count": len(pat["members"]),
                                 "axis": pat["axis"], "sweep_deg": pat["sweep"]})
    # P1.2 POCKETS: an interior section loop that matches no recognized hole
    # is a MEASURED recess or through-cut — emitted as a native Pocket from the
    # top face (circles and rectangles in v1, the section module's own shapes)
    hole_fp = [(float(np.dot(np.asarray(h["center"], dtype=float) - base_origin, u)),
                float(np.dot(np.asarray(h["center"], dtype=float) - base_origin, v)),
                float(h["radius"]),
                float((h.get("counterbore") or {}).get("diameter", 0.0) / 2.0))
               for h in holes]

    def _matches_hole(cx, cy, r) -> bool:
        for hx, hy, hr, hcb in hole_fp:
            if math.hypot(cx - hx, cy - hy) <= max(1.0, 0.2 * r):
                for rr in (hr, hcb):
                    if rr > 0 and abs(r - rr) <= 0.2 * max(r, rr):
                        return True
        return False

    pocket_count = 0
    seen_pockets: set = set()
    for bi, band in enumerate(bands):
        for lp in band["loops"][1:]:
            det = _section.detect_shape(lp, u_dir=u)
            if det is None or det.get("shape") not in ("circle", "rectangle"):
                continue
            c = det.get("center") or [0.0, 0.0]
            if det["shape"] == "circle":
                cx, cy, r = float(c[0]), float(c[1]), float(det["radius"])
                sig = ("circle", round(r, 1), round(cx, 1), round(cy, 1))
            else:
                cx, cy = float(c[0]), float(c[1])
                r = 0.5 * math.hypot(float(det["width"]), float(det["height"]))
                sig = ("rect", round(det["width"], 1), round(det["height"], 1),
                       round(cx, 1), round(cy, 1))
            if sig in seen_pockets or _matches_hole(cx, cy, r):
                continue
            seen_pockets.add(sig)
            pocket_count += 1
            # the pocket OPENS at the face adjacent to the band WITHOUT the
            # loop (an inverted frame anchors on the far face and the recess
            # sits at the frame's bottom): sketch normal points OUT of the
            # material there, the Pocket cuts along -normal, and the depth is
            # the run of bands the loop spans
            def _band_has(bd):
                for lp2 in bd["loops"][1:]:
                    d2 = _section.detect_shape(lp2)
                    if d2 is None or d2.get("shape") not in ("circle", "rectangle"):
                        continue
                    if d2.get("shape") == "circle":
                        s2 = ("circle", round(float(d2["radius"]), 1),
                              round(float(d2["center"][0]), 1),
                              round(float(d2["center"][1]), 1))
                    else:
                        s2 = ("rect", round(float(d2["width"]), 1),
                              round(float(d2["height"]), 1),
                              round(float(d2["center"][0]), 1),
                              round(float(d2["center"][1]), 1))
                    if s2 == sig:
                        return True
                return False

            run_hi, k = band["hi"], bi + 1
            while k < len(bands) and _band_has(bands[k]):
                run_hi = bands[k]["hi"]
                k += 1
            solid_above = k < len(bands)   # a loop-free band remains above the run
            # the pocket opens AWAY from the loop-free solid: at the run's
            # upper face (+n) unless the solid is ABOVE the run and the pocket
            # starts at the frame's origin face — then it opens downward (-n)
            open_at_lo = bi == 0 and solid_above
            if open_at_lo:
                face_pt = base_origin + band["lo"] * n
                sk_normal = -n
            else:
                face_pt = base_origin + run_hi * n
                sk_normal = n
            depth = round(run_hi - band["lo"], 3)
            sk = face_pt + cx * u + cy * v
            if det["shape"] == "circle":
                params[f"pocket{pocket_count}_d"] = {
                    "value": round(2.0 * r, 3),
                    "source": _source("interior section loop (measured recess / cut-out)")}
                prof = {"shape": "circle", "diameter": f"pocket{pocket_count}_d"}
            else:
                params[f"pocket{pocket_count}_l"] = {
                    "value": round(float(det["width"]), 3),
                    "source": _source("interior section loop (measured recess / cut-out)")}
                params[f"pocket{pocket_count}_w"] = {
                    "value": round(float(det["height"]), 3),
                    "source": _source("interior section loop (measured recess / cut-out)")}
                prof = {"shape": "rectangle", "width": f"pocket{pocket_count}_l",
                        "height": f"pocket{pocket_count}_w"}
            params[f"pocket{pocket_count}_dp"] = {
                "value": depth,
                "source": _source("recess depth: the open face down to the level "
                                  "the section loop disappears at")}
            features.append({"id": f"pocket{pocket_count}_sk", "op": "sketch",
                             "plane": {"normal": np.asarray(sk_normal).tolist(),
                                       "origin": np.asarray(sk).tolist(),
                                       "x_dir": u.tolist()},
                             "profile": prof})
            features.append({"id": f"pocket{pocket_count}", "op": "pocket",
                             "sketch": f"pocket{pocket_count}_sk",
                             "depth": f"pocket{pocket_count}_dp"})
    # ---- dressings (P1.1: multiple fillet / chamfer groups, each mapped to the
    # face its evidence describes) ----
    # Perimeter fillet radii from a VERTICAL section through the plate (G7): the
    # profile's corner ARCS at the far face and at the base face are the measured
    # radii of the two perimeter families (a tangent fillet merges into its planes
    # under any dihedral threshold, so the survey alone cannot isolate it).
    # Chamfers come from the survey's dressing records, classified by the strip's
    # own fitted normal and height into the far / base face group. Vertical-corner
    # dressings are deliberately NOT emitted as ops: a full-height corner round or
    # chamfer is already expressed by the base sketch's outline arcs / segments,
    # and a rule-based op on top of that would round the corners twice.
    # The edge rules are BODY-LOCAL, and the pad extrudes along the sketch
    # normal, so the local top face is ALWAYS the frame's FAR face — whatever
    # the world orientation. (Measured on the filleted bracket: a downward
    # frame anchored at the world-top plane made a world-Z-dependent mapping
    # send the top fillets to the world-BOTTOM edges — the built model came
    # out as the vertical mirror of the scan.)
    far_rule = "top_outer"
    base_rule = "bottom_outer"
    # the PartDesign edge rules are defined on the world Z axis: a frame that
    # extrudes sideways (the dominant pair was a wall's faces) must not emit
    # dressings against rules that describe different physical edges
    rules_valid = abs(float(n[2])) > math.cos(math.radians(30.0))
    sec_far, sec_base = [], []
    if mesh is not None and outline is not None:
        try:
            # the base centroid is already in-plane centred: nudge only along the
            # normal so the vertical plane cuts the middle of the plate (not an edge)
            v_origin = base_origin + 0.5 * total_h * n
            v_loops, _v_opens = _section.slice_loops(mesh, v_origin, u, u_dir=n)
            if v_loops:
                segs = _section.fit_loop(v_loops[0])
                top = max(p[0] for p in v_loops[0])
                bot = min(p[0] for p in v_loops[0])
                for seg in segs or []:
                    if seg.get("kind") != "arc":
                        continue
                    fit = seg.get("fit") if isinstance(seg.get("fit"), dict) else {}
                    center = fit.get("center")
                    if center is None:
                        continue
                    r = float(fit.get("radius") or 0.0)
                    if r <= 0.0 or r > 0.6 * total_h:
                        continue
                    cx = float(center[0])
                    if abs((cx + r) - top) <= max(0.3 * r, 0.05):
                        sec_far.append(r)
                    elif abs((cx - r) - bot) <= max(0.3 * r, 0.05):
                        sec_base.append(r)
        except Exception:  # noqa: BLE001 - dressings must never break the route
            sec_far, sec_base = [], []
    recs = survey.get("dressings") or []
    fillet_recs = [d for d in recs if d.get("kind") == "fillet" and d.get("axis") is not None]
    chamfer_recs = [d for d in recs if d.get("kind") == "chamfer" and d.get("normal") is not None]
    cos60 = math.cos(math.radians(60.0))

    def _height(d):
        return float(np.dot(np.asarray(d["point"], dtype=float) - base_origin, n))

    # a PERIMETER dressing strip faces the plate normal (fillet axis in-plane /
    # chamfer normal tilted out of it); vertical-corner records stay informational
    perim_f = [d for d in fillet_recs
               if abs(float(np.asarray(d["axis"], dtype=float) @ n)) <= cos60]
    perim_c = [d for d in chamfer_recs
               if abs(float(np.asarray(d["normal"], dtype=float) @ n)) > cos60]
    perim_f_far = [d for d in perim_f if _height(d) > 0.5 * total_h]
    perim_f_base = [d for d in perim_f if d not in perim_f_far]
    perim_c_far = [d for d in perim_c if _height(d) > 0.5 * total_h]
    perim_c_base = [d for d in perim_c if d not in perim_c_far]
    # the section's direct arc measurement supersedes survey strips on that face
    if sec_far:
        perim_f_far = []
    if sec_base:
        perim_f_base = []

    dress_ops = []

    def _group(values, spread):
        return (bool(values) and len(values) >= 2
                and max(values) - min(values) <= spread * max(values))

    for radii, rule, src in (
            (sec_far or [d["radius_mm"] for d in perim_f_far], far_rule,
             "vertical-section far-face corner arcs (G7, measured fillet radii)"
             if sec_far else "perimeter fillet cylinder fits (survey dressing records)"),
            (sec_base or [d["radius_mm"] for d in perim_f_base], base_rule,
             "vertical-section base-face corner arcs (G7, measured fillet radii)"
             if sec_base else "perimeter fillet cylinder fits (survey dressing records)")):
        if not rules_valid or not _group(radii, 0.10):
            continue
        pid, fid = _RULE_IDS[rule][0], _RULE_IDS[rule][1]
        params[pid] = {"value": round(float(np.mean(radii)), 3), "source": _source(src)}
        op = {"id": fid, "op": "fillet", "size": pid, "edges": rule}
        features.append(op)
        dress_ops.append(op)
    fillet_rules = {op["edges"] for op in dress_ops}
    for crecs, rule in ((perim_c_far, far_rule), (perim_c_base, base_rule)):
        widths = [d["width_mm"] for d in crecs]
        if not rules_valid or rule in fillet_rules or not _group(widths, 0.15):
            continue                                   # fillet evidence wins the face
        cpid, cfid = _RULE_IDS[rule][2], _RULE_IDS[rule][3]
        params[cpid] = {"value": round(float(np.mean(widths)), 3),
                        "source": _source("chamfer strip width (G7, sharp junctions "
                                          "to both neighbours)")}
        # P1.1b: the strip WIDTH is the measurement, but FreeCAD's chamfer cuts
        # a DISTANCE — for a 45 deg chamfer the strip is d*sqrt(2) wide, so
        # binding the width directly overcuts by ~41 %. Derive the cut
        # distances from the measured width and junction angles.
        d1s, d2s = [], []
        for d in crecs:
            a1 = math.radians(float((d.get("angles_deg") or [45.0, 45.0])[0]))
            d1s.append(d["width_mm"] * math.cos(a1))
            d2s.append(d["width_mm"] * math.sin(a1))
        d1, d2 = float(np.mean(d1s)), float(np.mean(d2s))
        dpid = cpid.replace("chamfer_w", "chamfer_d")
        if abs(d1 - d2) > 0.15 * max(d1, d2, 1e-9):
            params[dpid] = {"value": round(d1, 3),
                            "source": _source("chamfer cut distance (derived: strip "
                                              "width x junction angle)")}
            params[f"{dpid}2"] = {"value": round(d2, 3),
                                  "source": _source("chamfer second cut distance "
                                                    "(derived: strip width x "
                                                    "junction angle)")}
            op = {"id": cfid, "op": "chamfer", "size": dpid, "size2": f"{dpid}2",
                  "edges": rule}
        else:
            params[dpid] = {"value": round((d1 + d2) / 2.0, 3),
                            "source": _source("chamfer cut distance (derived: strip "
                                              "width / sqrt(2) at 45 deg junctions)")}
            op = {"id": cfid, "op": "chamfer", "size": dpid, "edges": rule}
        features.append(op)
        dress_ops.append(op)
    fillet_evidence = sec_far + sec_base + [d["radius_mm"] for d in fillet_recs]
    chamfer_evidence = [d["width_mm"] for d in chamfer_recs]
    plan = {
        "schema_version": 2, "id": "scan_recon", "part_class": "generic_part",
        "units": "mm", "target": "freecad", "parameters": params, "features": features,
        "notes": [f"route prismatic_extrusion; {pad_count} pad level(s) from section "
                  f"outlines; {len(holes)} holes from cylinder fits"
                  + (f"; {pocket_count} pocket(s) from interior section loops"
                     if pocket_count else "")
                  + f"; recognition coverage {survey['coverage']}"],
    }
    meta = {"route": "prismatic_extrusion", "holes": len(holes), "pads": pad_count}
    if pattern_meta:
        meta["patterns"] = pattern_meta
    if pocket_count:
        meta["pockets"] = pocket_count
    if skipped_bands:
        meta["skipped_bands"] = skipped_bands
    if fillet_evidence:
        meta["fillets"] = len(fillet_evidence)
    if chamfer_evidence:
        meta["chamfers"] = len(chamfer_evidence)
    if dress_ops:
        meta["dressings"] = [op["id"] for op in dress_ops]
    if outline is not None and outline.get("shape") == "rectangle":
        meta["outline"] = "section"
    elif outline is not None and outline.get("shape") == "profile":
        meta["outline"] = f"section profile ({len(outline['segments'])} segments)"
    elif outline is not None and outline.get("shape") == "circle":
        meta["outline"] = "section circle"
    else:
        meta["outline"] = "bbox (approximation)"
    band_volume = sum((float(b["area"]) * (b["hi"] - b["lo"]))
                      for b in bands if b.get("area"))
    expected = (band_volume if band_volume > 0
                else extent_u * extent_v * total_h) * (1 + EXTRA_VOLUME_TOLERANCE)
    measured = diag.get("volume_mm3")
    if measured is not None and measured > expected:
        meta["extra_geometry"] = (
            f"the scan holds {measured:.1f} mm3 against {expected:.0f} mm3 for the recognized "
            "base plate: bosses or walls beyond the base plate are NOT reconstructed in v1")
    return plan, meta


def _plan_rotational(survey, diag, mesh, plane_tol_gate=0.05):
    """R-4: the rotational route — the revolution profile built from RECOGNIZED
    evidence (no slicing): coaxial full-turn cylinders give measured radii over axial
    intervals, planes perpendicular to the axis give the ends, and the outer-radius
    staircase r(x) becomes the sketch profile of a full-turn PartDesign Revolution.
    An axial-plane section is unusable here by construction: it passes through the end
    disc centres (the axis), whose degenerate vertex crossings mis-chain the loop."""
    dom = sorted((r for r in survey["regions"] if r["type"] == "cylinder"
                  and r.get("turn_deg", 0) >= 300),
                 key=lambda r: -r["area_mm2"])
    if dom:
        a = np.asarray(dom[0]["axis"], dtype=float)
        a = a / np.linalg.norm(a)
        origin = np.asarray(dom[0]["point"], dtype=float)
        end_planes = sorted(
            (r for r in survey["regions"] if r["type"] == "plane"
             and abs(float(np.asarray(r["normal"], dtype=float) @ a))
             > math.cos(math.radians(10.0))),
            key=lambda r: -r["area_mm2"])
    else:
        planes = sorted((r for r in survey["regions"] if r["type"] == "plane"),
                        key=lambda r: -r["area_mm2"])
        if not planes:
            return None, {"reason": "no full-turn cylinder and no plane to anchor "
                                    "the axis"}
        a = np.asarray(planes[0]["normal"], dtype=float)
        a = a / np.linalg.norm(a)
        origin = np.asarray(planes[0]["point"], dtype=float)
        end_planes = planes[:1]
    # H2: CANONICAL frame — the axis runs from the largest end plane (the base,
    # origin) INTO the material, so axial coordinates are unambiguous (0 at the base,
    # growing toward the far end). Without this the fit's axis sign/point placement
    # flips per mesh and "beyond the wall" picks the wrong side.
    if end_planes:
        base = end_planes[0]
        origin = np.asarray(base["point"], dtype=float)
        nrm = np.asarray(base["normal"], dtype=float)
        if float(nrm @ a) < 0:
            a = -a
        _pa = np.array([[q.x, q.y, q.z] for q in mesh.Topology[0]], dtype=float)
        if float(((_pa - origin) @ a).mean()) < 0:
            a = -a
    axial = lambda p: float((np.asarray(p, dtype=float) - origin) @ a)

    def in_own_frame(r):
        """(lo, hi, radius) of a region, expressed in the anchor axis frame."""
        mid = float(r.get("axial_mid", axial(r["point"])))
        half = float(r.get("axial_span", 0.0)) / 2.0
        if float(np.asarray(r["axis"], dtype=float) @ a) < 0:
            mid = -mid
        return mid - half, mid + half, float(r["radius"])

    bands = [in_own_frame(r) for r in survey["regions"]
             if r["type"] == "cylinder" and r.get("turn_deg", 0) >= 300
             and abs(float(np.asarray(r["axis"], dtype=float) @ a)) > math.cos(math.radians(10.0))]
    ends = [axial(r["point"]) for r in survey["regions"]
            if r["type"] == "plane"
            and abs(float(np.asarray(r["normal"], dtype=float) @ a)) > math.cos(math.radians(10.0))]
    if not bands:
        return None, {"reason": "no coaxial full-turn cylinders to build a profile"}
    lo_all = min([b[0] for b in bands] + ends) if (bands or ends) else 0.0
    hi_all = max([b[1] for b in bands] + ends)
    length = hi_all - lo_all
    if length <= 0:
        return None, {"reason": "degenerate axial extent"}
    bands = [(max(lo, lo_all), min(hi, hi_all), r) for lo, hi, r in bands
             if min(hi, hi_all) - max(lo, lo_all) > 0.01 * length]

    def radius_at(x):
        covering = [r for lo, hi, r in bands if lo - 1e-9 <= x <= hi + 1e-9]
        if covering:
            return max(covering)
        # a gap means no full-turn cylinder was recognized there (a dome's cap top):
        # borrow the NEAREST band's radius — the profile stays solid, the limitation
        # is recorded in the route notes
        dist, best = None, 0.0
        for lo, hi, r in bands:
            d = max(lo - x, 0.0, x - hi)
            if dist is None or d < dist:
                dist, best = d, r
        return best

    # H2 (dome caps): when a SPHERE region's fitted centre lies on the axis, its
    # circle IS the dome's meridian — emit a genuine Sketcher ARC instead of the
    # staircase (which overfills the cap and honestly fails verification). The arc
    # runs from the wall-radius intersection to the apex; everything below stays the
    # measured wall staircase. Residuals are recorded; a sphere that is off-axis or
    # unfitted keeps the honest staircase.
    dome = None
    spheres = sorted((r for r in survey["regions"] if r["type"] == "sphere"),
                     key=lambda r: -r["area_mm2"])
    for sph in spheres:
        R = float(sph["radius"])
        ctr = np.asarray(sph["center"], dtype=float)
        c_ax = float((ctr - origin) @ a)
        lateral = float(np.linalg.norm((ctr - origin) - c_ax * a))
        if lateral <= max(0.05 * R, 0.5):
            wall_r = max((r for _, _, r in bands), default=0.0)
            if wall_r <= R + 1e-9 and R > 0:
                inner = R * R - wall_r * wall_r
                off = math.sqrt(max(inner, 0.0))
                # the wall sits on the low-x side of the part (lo_all end) when the
                # bands' mean is below the sphere centre — the apex points away
                mean_x = sum((lo + hi) / 2.0 for lo, hi, _ in bands) / max(len(bands), 1)
                direction = 1.0 if mean_x <= c_ax else -1.0
                x_t = c_ax - direction * off
                apex = c_ax + direction * R
                if min(x_t, apex) >= lo_all - 1e-6 and max(x_t, apex) <= hi_all + 1e-6                         and abs(x_t - lo_all) < 0.999 * length:
                    dome = {"R": R, "c_ax": c_ax, "x_t": x_t, "apex": apex,
                            "wall_r": wall_r, "rms": sph.get("rms_mm"),
                            "direction": direction}
                    break
    if dome is None and bands:
        # H2: no sphere REGION survived segmentation (the cap fragments into small
        # cylinders at fine tessellation) — fit a sphere DIRECTLY to the measured
        # cap vertices (axially beyond the wall, radially inside the wall + margin),
        # accepted only through the rms gate; anything else keeps the honest staircase
        from . import fit as _fitmod

        # the WALL band is the LARGEST-AREA band (cap rings also reach full turn but
        # are thin — picking by extent would swallow the cap into the "wall")
        wall_band = max(bands, key=lambda b: b[1] - b[0])
        x_wall_hi = wall_band[1]
        P_all = np.array([[q.x, q.y, q.z] for q in mesh.Topology[0]], dtype=float)
        rel = P_all - origin
        axial = rel @ a
        radial = np.linalg.norm(rel - np.outer(axial, a), axis=1)
        wall_r0 = wall_band[2]
        cap = (axial > x_wall_hi - 0.05 * length) & (radial <= wall_r0 * 1.05)
        if int(cap.sum()) >= 12:
            sph = _fitmod.fit_sphere(P_all[cap])
            if sph is not None and sph["rms"] <= max(0.05, 4.0 * plane_tol_gate):
                R = float(sph["radius"])
                ctr = np.asarray(sph["center"], dtype=float)
                c_ax = float((ctr - origin) @ a)
                lateral = float(np.linalg.norm((ctr - origin) - c_ax * a))
                if lateral <= max(0.05 * R, 0.5) and wall_r0 <= R + 1e-9:
                    off = math.sqrt(max(R * R - wall_r0 * wall_r0, 0.0))
                    direction = 1.0                      # canonical frame: base at 0
                    x_t = c_ax - off
                    apex = c_ax + direction * R
                    if min(x_t, apex) >= lo_all - 1e-6 and                             max(x_t, apex) <= hi_all + max(1e-6, 0.2 * R) and                             abs(x_t - lo_all) < 0.999 * length:
                        dome = {"R": R, "c_ax": c_ax, "x_t": x_t, "apex": apex,
                                "wall_r": wall_r0, "rms": round(sph["rms"], 5),
                                "direction": direction,
                                "basis": "direct sphere fit on the measured cap "
                                         "vertices (rms-gated)"}

    if dome is not None:
        hi_all = max(hi_all, dome["apex"])            # the dome reaches the apex
        length = hi_all - lo_all

    breaks = sorted({lo_all, hi_all, *[b[0] for b in bands], *[b[1] for b in bands]})
    pts = [[lo_all, 0.0]]
    for i, x in enumerate(breaks[:-1]):
        r = radius_at((x + breaks[i + 1]) / 2.0)
        if r <= 0:
            continue
        if abs(pts[-1][1] - r) > 1e-6:
            pts.append([x, pts[-1][1]])
        pts.append([x, r])
        pts.append([breaks[i + 1], r])
    pts.append([hi_all, 0.0])
    pts = [p for i, p in enumerate(pts) if i == 0 or
           abs(p[0] - pts[i - 1][0]) > 1e-6 or abs(p[1] - pts[i - 1][1]) > 1e-6]
    if len(pts) < 4:
        return None, {"reason": "the profile staircase degenerated"}

    # sketch plane: contains the axis (x_dir = a), origin on the axis at the start.
    # The plan's revolution axis is a WORLD line (frame axis or a direction through
    # the frame origin): when the fitted axis lies within 1 deg of a frame axis, snap
    # to it (the residual is recorded) so the axis and the sketch's H axis coincide
    # exactly — a free choice, the staircase profile comes from evidence, not from
    # the plane position.
    for vec in (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0])):
        if abs(float(a @ vec)) > math.cos(math.radians(1.0)):
            if float(a @ vec) < 0:
                a = -a                      # profile x runs with the frame axis
            break
    frame = None
    for name, vec in (("z", np.array([0.0, 0.0, 1.0])),
                      ("x", np.array([1.0, 0.0, 0.0])),
                      ("y", np.array([0.0, 1.0, 0.0]))):
        if float(a @ vec) > math.cos(math.radians(1.0)):
            frame, a = (name, vec)
            break
    axis_spec = {"frame_axis": frame} if frame else {"direction": a.tolist()}
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(a @ ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    m = np.cross(a, ref)
    m = m / np.linalg.norm(m)
    if frame:
        sketch_origin = float(origin @ a) * a          # on the world frame-axis line
    else:
        sketch_origin = origin + float(lo_all) * a
    if dome is not None:
        x_t, apex = dome["x_t"], dome["apex"]
        # keep ONLY the wall-side staircase (axial <= x_t) — the old tail descending
        # to the axis at the pre-dome end must go; the arc replaces it
        wall_pts = [[float(p[0]), float(p[1])] for p in pts if p[0] <= x_t + 1e-9]
        if not wall_pts or wall_pts[-1][1] <= 0.0:
            return None, {"reason": "the dome profile's wall side degenerated"}
        if wall_pts[-1][0] < x_t - 1e-6:
            wall_pts.append([x_t, wall_pts[-1][1]])
        if abs(wall_pts[-1][1] - dome["wall_r"]) > 1e-6:
            wall_pts.append([x_t, dome["wall_r"]])
        segs = []
        prev = [0.0, 0.0]
        for p in wall_pts[1:]:
            to = [round(p[0] - lo_all, 4), round(p[1], 4)]
            if to == prev:
                continue
            segs.append({"type": "line", "to": to})
            prev = to
        # the meridian arc: centre on the axis at c_ax, radius R, from the wall
        # point to the apex (ccw chosen so the arc bulges outward, +y)
        c2 = [round(dome["c_ax"] - lo_all, 4), 0.0]
        segs.append({"type": "arc",
                     "to": [round(apex - lo_all, 4), 0.0],
                     "center": c2, "ccw": dome["direction"] < 0})
        segs.append({"type": "line", "to": [0.0, 0.0]})   # close along the axis
        meta_dome = {"dome": {"radius_mm": round(dome["R"], 4),
                              "sphere_rms_mm": dome["rms"],
                              "wall_radius_mm": round(dome["wall_r"], 4),
                              "note": "meridian arc from the fitted sphere (centre on "
                                      "the axis) — measured curved profile, not a "
                                      "staircase"}}
    else:
        meta_dome = None
        segs = [{"type": "line", "to": [0.0, round(pts[1][1], 4)]}]
        for p, q in zip(pts[1:], pts[2:]):
            to = [round(q[0] - lo_all, 4), round(q[1], 4)]
            if to == segs[-1]["to"]:
                continue                  # rounding can collapse a sliver step
            segs.append({"type": "line", "to": to})
        segs.append({"type": "line", "to": [0.0, 0.0]})   # close along the axis
    params = {"full_ang": {"value": 360.0,
                           "source": _source("full-turn revolution (measured wall "
                                             "turn coverage >= 300 deg)")}}
    plan = {
        "schema_version": 2, "id": "scan_recon", "part_class": "generic_part",
        "units": "mm", "target": "freecad", "parameters": params,
        "frame": {"axes": {"z": {"direction": [0.0, 0.0, 1.0],
                                 "point": [0.0, 0.0, 0.0]}}},
        "features": [
            {"id": "rev_sk", "op": "sketch",
             "plane": {"normal": m.tolist(), "origin": sketch_origin.tolist(),
                       "x_dir": a.tolist()},
             "profile": {"shape": "profile", "start": [0.0, 0.0],
                         "segments": segs}},
            {"id": "rev", "op": "revolution", "sketch": "rev_sk",
             "angle": "full_ang", "axis": axis_spec},
        ],
        "notes": [f"route rotational_revolution; profile staircase from {len(bands)} "
                  f"coaxial cylinder fits + end planes; coverage {survey['coverage']}"],
    }
    meta = {"route": "rotational_revolution",
            "outline": (f"wall staircase + dome arc (R={dome['R']:.3f})"
                        if dome else f"cylinder staircase ({len(bands)} bands)")}
    if meta_dome:
        meta.update(meta_dome)
    if frame:
        residual = math.degrees(math.acos(min(1.0, max(-1.0, float(
            np.asarray(dom[0]["axis"], dtype=float) @ a if dom else 1.0)))))
        meta["axis_snapped_to"] = f"frame {frame} (fitted axis within {residual:.4f} deg)"
    # H3 (cross-holes): transverse full-turn cylinder walls (axis ~perpendicular to
    # the rotation axis) drilled as native PartDesign Holes on the revolution — every
    # dimension from the cylinder fit; through/blind decided against the recognized
    # body radius; coaxial duplicates collapse; partial walls (< 300 deg turn) never
    # drill. Anything not evidenced is reported, never invented.
    cross_feats, seen_axes = [], []
    for r in survey["regions"]:
        if r["type"] != "cylinder" or r.get("turn_deg", 0) < 300:
            continue
        n_hat = np.asarray(r["axis"], dtype=float)
        n_hat = n_hat / np.linalg.norm(n_hat)
        if abs(float(n_hat @ a)) > math.cos(math.radians(30.0)):
            continue                              # axial (a step), not transverse
        if not (2.0 <= 2.0 * r["radius"] <= 20.0):
            continue
        # the wall must PENETRATE the body as a MINOR BORE: a dome-cap fragment
        # fitted as a wide near-tangent "cylinder" (radius ~ the body) lies against
        # the surface and cuts nothing — a real cross-hole is well inside both
        p_mid = np.asarray(r["point"], dtype=float)
        d_mid = p_mid - origin - float((p_mid - origin) @ a) * a
        body_here = max(radius_at(float((p_mid - origin) @ a)) if bands else 0.0, 1e-9)
        if float(np.linalg.norm(d_mid)) + 0.5 * r["radius"] >= body_here:
            continue
        if r["radius"] > 0.6 * body_here:
            continue
        p0 = np.asarray(r["point"], dtype=float)
        # distance of this hole's axis line from the rotation axis (through origin)
        d = p0 - origin - float((p0 - origin) @ a) * a
        lateral = float(np.linalg.norm(d))
        mid_a = float((p0 - origin) @ a)
        body_r = radius_at(mid_a) if bands else 0.0
        if body_r <= lateral + 1e-9 or body_r <= 0.0:
            continue                              # the wall does not reach the body
        surf_half = math.sqrt(max(body_r * body_r - lateral * lateral, 0.0))
        span = float(r.get("axial_span", 0.0))
        through = span >= 1.9 * surf_half
        # coaxial duplicates: same direction and line
        dup = False
        for prev_n, prev_p in seen_axes:
            if abs(float(prev_n @ n_hat)) > math.cos(math.radians(5.0)):
                delta = p0 - prev_p
                off = delta - float(delta @ n_hat) * n_hat
                if float(np.linalg.norm(off)) <= max(0.5, 0.2 * r["radius"]):
                    dup = True
                    break
        if dup:
            continue
        seen_axes.append((n_hat, p0))
        # entry: the +n end of the wall; the sketch normal points OUT of the material
        # (the Hole feature drills along -normal)
        entry = p0 + surf_half * n_hat
        i = len(cross_feats) + 1
        params[f"xhole_d{i}"] = {"value": round(2.0 * r["radius"], 3),
                                 "source": _source("cross-hole cylinder fit "
                                                   "(full-turn wall, least squares)")}
        feat = {"id": f"xhole_{i}", "op": "hole",
                "plane": {"normal": n_hat.tolist(),
                          "origin": [round(float(x), 4) for x in entry],
                          "x_dir": np.cross(n_hat, a).tolist()},
                "position": {"x": 0.0, "y": 0.0}, "diameter": f"xhole_d{i}"}
        if through:
            feat["through"] = True
        else:
            # blind depth is ENTRY-TO-FLOOR: the entry plane sits at p0 + surf_half * n (the
            # body surface on the drill axis), the recognized wall ends at p0 - span/2 * n, so
            # the drill must travel surf_half + span/2 to reach the floor. The old
            # span - surf_half drilled only a shallow scratch near the entry (H3c: 2 mm on a
            # 10 mm bore) — valid on lenient OCC builds, an invalid tangential boolean on
            # stricter ones (macOS CI).
            depth = max(surf_half + span / 2.0 + 0.01, 0.2 * r["radius"])   # +0.01 overshoot: macOS OCC isValid flags the exact-tangency boolean
            params[f"xhole_dp{i}"] = {"value": round(float(depth), 3),
                                      "source": _source("cross-hole wall extent "
                                                        "(blind, no far wall seen)")}
            feat["depth"] = f"xhole_dp{i}"
        cross_feats.append(feat)
    if cross_feats:
        plan["features"].extend(cross_feats)
        meta["cross_holes"] = len(cross_feats)
    return plan, meta


def _profile_start(outline):
    """The profile start point: the fitted loop's first point (the chain closes there)."""
    return outline.get("start") or [0.0, 0.0]


def _measured_mask(base, work, scan_report):
    """G6: build the measured-only deviation mask for the working mesh.

    Scan side: vertices of facets repair CREATED — from ``provenance.diff_mesh(base,
    work)``'s ``added_facets`` (complete and coordinate-keyed, so later renumbering cannot
    hide a fill; never the capped ledger key lists). Model side: the planar footprints of
    dropout-class loops from the G5 scan-quality report (surface the scan never measured —
    model samples there are excluded instead of warning as phantom "extra geometry").
    A mask that excludes nothing is still returned: the deviation row then reports zero
    exclusions honestly."""
    from . import provenance, quality as _quality

    scan_exclude = []
    regions = []
    try:
        if base is not work:
            added = set(provenance.diff_mesh(base, work).get("added_facets") or [])
            if added:
                keys = provenance.facet_keys(work)
                _, fac = _quality._pts_fac(work)
                exclude = set()
                for i, key in enumerate(keys):
                    if key in added:
                        exclude.update(int(v) for v in fac[i])
                scan_exclude = sorted(exclude)
        pts_w, _ = _quality._pts_fac(work)
        sigma = float((scan_report or {}).get("sigma_mm") or 0.0)
        for l in ((scan_report or {}).get("completeness") or {}).get("loops") or []:
            if l.get("kind") != "dropout":
                continue
            regions.append({"points": pts_w[np.asarray(l["ring"], dtype=int)],
                            "normal": np.asarray(l["normal"], dtype=float),
                            "tol_mm": float(l["planarity_rms_mm"]) + 2.0 * sigma})
    except Exception:  # noqa: BLE001 - the mask must never break the reconstruction
        return None
    return {"scan_exclude_indices": scan_exclude, "model_regions": regions,
            "note": "synthetic fill vertices and missing-region footprints excluded"
                    if (scan_exclude or regions) else ""}


def reconstruct(doc, mesh, *, warn_mm=0.3, user_unit=None, authorized=(),
                preview=False, plan_override=None, prefer_class=None) -> dict:
    """Diagnose → safe-repair a copy → recognize → plan → build with the existing builder →
    deviation rows → honest outcome. The original mesh is never modified.

    K4: a unit-plausibility report is computed first (cheap: bbox + edges only). An
    unconfirmed POSSIBLE_MM_INCH_MISMATCH / SCALE_AMBIGUOUS never yields a VERIFIED outcome.
    A user-confirmed ``inch`` source converts a working copy ×25.4 (recorded separately in
    the provenance ledger; the original mesh and its source hash are untouched).

    G5: a scan-quality report (completeness / integrity / plausibility, recon.quality §2)
    is computed on the working mesh; DEGRADED readiness (doubled surfaces, missing regions,
    unscanned hole walls) caps the outcome at partially_verified_parametric with the
    reasons in ``meta['scan_degraded']``.

    G6: ``authorized`` names gated repair operations approved through the tool's human
    approval boundary — the loop-aware fill policy inside ``meshops.repair``
    still refuses to close anything but tiny flat pinholes, and larger openings come back
    as ``missing_regions``. Deviation is measured on MEASURED regions only (the
    ``_measured_mask``: synthetic fill vertices and missing-region footprints excluded,
    reported in a ``measured_only`` row and summarized in ``result['measured_mask']``)."""
    from . import provenance, units

    units_report = units.analyse_units(mesh, user_unit=user_unit)
    base = mesh
    converted_from = None
    base_digest = None
    if user_unit == "inch":
        converted_from = mesh
        base = units.scaled_copy(mesh, units.INCH_MM)   # approved conversion, on a copy
        base_digest = provenance.mesh_digest(base)      # the mm working input repair uses
    diag = meshops.diagnose(base)
    partial_meta = None
    work, repair_info = base, None
    if not diag["closed"] or diag["inconsistent_normals"]:
        repair_info = meshops.repair(base, authorized=authorized)
        work = repair_info["mesh"]
        diag = repair_info["after"]
        if not diag["closed"]:
            # H4 (large dropouts): a controlled PARTIAL-EVIDENCE route. The mesh is
            # NOT filled and the original stays untouched; reconstruction proceeds
            # ONLY when every remaining opening is a FLAT dropout-class region and a
            # minority of the surface (< 25 %) — the adjacent measured planes bound
            # what a flat dropout can hide, the missing footprints feed the
            # measured-only deviation mask, and the outcome is capped at
            # partially-verified. Anything else (outer rim = a one-sided scan, a hole
            # opening = an unseen real feature, a curved dropout, too much missing)
            # keeps the honest refusal, now with per-loop reasons.
            from .meshops import _fill_policy

            sigma_val = float((repair_info.get("quality") or {}).get("sigma_mm") or 0.0)
            _fill, missing_regions, loops = _fill_policy(work, sigma_val)
            reasons = []
            missing_share = 0.0
            for l in loops:
                share = float(l.get("area_share") or 0.0)
                if l.get("kind") == "outer_rim":
                    reasons.append("outer scan rim (a one-sided scan: the far side was "
                                   "never measured)")
                elif l.get("kind") in ("hole_opening", "bore_end"):
                    reasons.append(f"a {l['kind']} (a real opening whose wall was never "
                                   "scanned — a reconstruction would guess it)")
                elif l.get("kind") == "dropout":
                    # planarity is judged HERE (the classifier's pinhole branch fires
                    # before its own planarity test, so the detail alone cannot gate)
                    diameter = 2.0 * float(l.get("radius_mm") or 0.0)
                    planarity = l.get("planarity_rms_mm")
                    planar = float(planarity if planarity is not None else 1e9) <=                         0.05 * diameter + 2.0 * sigma_val   # NB: `or` would turn a
                        # truthful 0.0 (a perfectly flat rim) into the refusal default
                    if not planar:
                        reasons.append("a non-flat dropout (its completion is not "
                                       "bounded by adjacent measured planes)")
                    else:
                        missing_share += share
            if not loops:
                reasons.append("open boundary chains that never close (torn scan)")
            if missing_share > 0.25:
                reasons.append(f"missing flat regions span {missing_share:.0%} of the "
                               "surface (> 25 %: too little measured geometry)")
            if reasons:
                missing = repair_info.get("missing_regions") or missing_regions
                note = ("reconstruction refused — the open mesh's openings are not "
                        "safely bypassable: " + "; ".join(reasons[:3]) +
                        ". Measure those regions or supply a patched mesh; authorize "
                        "fillupHoles for tiny flat pinholes only (G6 policy)")
                return {"supported": False, "blocked": "mesh_not_closed",
                        "note": note, "missing_regions": missing,
                        "loop_reasons": reasons,
                        "diagnostics": diag, "repair": repair_info,
                        "units": units_report, "base_digest": base_digest}
            partial_meta = {
                "partial_evidence": {
                    "missing_regions": len(loops),
                    "missing_area_share": round(missing_share, 4),
                    "note": "flat scan dropout(s) bypassed WITHOUT filling: the model "
                            "is built from measured regions only, the missing "
                            "footprints are excluded from deviation coverage, and the "
                            "outcome is capped at partially-verified"}}
    # P3.2 digest-keyed stage cache: the survey of an unchanged work mesh is
    # deterministic, so analyse → plan → parametric sequences (and repeated
    # parametric calls) reuse it instead of re-running region growing + fits.
    # The key is the WORK-mesh content digest plus whether repair evidence
    # tags were applied — a changed mesh or a different evidence basis misses.
    global _SURVEY_CACHE
    _prov_ledger = (repair_info or {}).get("provenance")
    _cache_key = (_SURVEY_CACHE_VERSION, provenance.mesh_digest(work),
                  bool(_prov_ledger))
    _hit = _SURVEY_CACHE.get(_cache_key)
    if _hit is not None:
        diag, survey = _hit["diag"], _hit["survey"]
    else:
        survey = recognize.survey(work, diag, provenance=_prov_ledger)
        if len(_SURVEY_CACHE) >= 6:
            _SURVEY_CACHE.pop(next(iter(_SURVEY_CACHE)))
        _SURVEY_CACHE[_cache_key] = {"diag": diag, "survey": survey}
    # G5: the scan-quality report gates the outcome — DEGRADED readiness (doubled
    # surfaces, missing regions, unscanned hole walls) caps it at partially-verified
    from . import quality as _quality

    scan_report = None
    try:
        scan_report = _quality.scan_quality(
            work, est=(repair_info or {}).get("quality") or survey.get("quality") or None,
            units_status=units_report.get("status"))
    except Exception:  # noqa: BLE001 - the report must never break the reconstruction
        scan_report = None
    measured_mask = _measured_mask(base, work, scan_report)   # G6: measured regions only
    klass = survey["classification"]["class"]
    if prefer_class in ("prismatic", "rotational") and prefer_class != klass:
        # a §16 user decision (askwire.resolve_interpretation) confirmed the ALTERNATIVE
        # interpretation: this run uses it; the measured classification stays in the survey
        klass = prefer_class
    if plan_override is not None:
        # P2.4: the agent's AMENDED plan (from a preview) builds AS GIVEN —
        # through the same builder, the same honesty caps and the same
        # deviation rows; the engine's synthesis is skipped and nominal
        # re-snapping is not applied twice
        plan = plan_override
        meta = {"route": "agent_amended_plan",
                "holes": sum(1 for f in plan.get("features") or []
                             if f.get("op") == "hole"),
                "pads": sum(1 for f in plan.get("features") or []
                            if f.get("op") == "pad")}
        plan.setdefault("parameters", {})
    # P1.6: the datum / symmetry proposal is EVIDENCE the plan can lean on and
    # the agent (P2) reads: the 3-2-1 frame cross-checks the construction
    # frame, and detected symmetry is recorded (a conflicting datum frame is a
    # warning flag, never silently ignored)
    datum_summary = None
    try:
        from . import datums as _datums

        prop = _datums.propose_datums(work, survey)
        sym = prop.get("symmetry") or {}
        datum_summary = {
            "primary_plane_normal": (prop.get("primary_plane") or {}).get("normal"),
            "alignment": prop.get("alignment"),
            "rotational_symmetry": (
                {"order": sym["rotational"]["order"],
                 "reoccupancy_rms_mm": sym["rotational"]["reoccupancy_rms_mm"]}
                if sym.get("rotational") else None),
            "mirror_symmetry": (
                {"reflection_rms_mm": sym["mirror"]["reflection_rms_mm"],
                 "basis": sym["mirror"]["basis"]}
                if sym.get("mirror") else None),
            "ambiguities": prop.get("ambiguities") or None,
        }
    except Exception:  # noqa: BLE001 - datums are advisory, never a blocker
        datum_summary = None
    if plan_override is None and klass == "rotational":
        plan, meta = _plan_rotational(survey, diag, mesh=work, plane_tol_gate=float((survey.get("quality") or {}).get("plane_rms_mm") or 0.05))
        if plan is None:
            return {"supported": False, "note": meta["reason"], "survey": survey,
                    "diagnostics": diag, "scan_quality": scan_report,
                    "classification": survey["classification"],
                    "units": units_report, "base_digest": base_digest,
                    "datums": datum_summary}
    elif plan_override is None and klass == "thin_panel":
        # P1.5: a thin FLAT panel (a tray, a sheet with windows) has the plane
        # evidence the prismatic route needs — stacked pads plus measured
        # pockets build it natively. A CURVED sheet has no dominant parallel
        # pair; the prismatic route refuses below and this note says why.
        plan, meta = _plan_prismatic(survey, diag, mesh=work)
        if plan is None:
            return {"supported": False, "classification": survey["classification"],
                    "survey": survey, "diagnostics": diag, "repair": repair_info,
                    "units": units_report, "base_digest": base_digest,
                    "scan_quality": scan_report, "datums": datum_summary,
                    "note": "a curved thin panel (no dominant parallel face pair) needs "
                            "shell-from-surface modelling — not implemented in v1; the "
                            "survey above is the honest measured evidence"}
        meta["thin_panel"] = True
    elif plan_override is None and klass != "prismatic":
        return {"supported": False, "classification": survey["classification"],
                "survey": survey, "diagnostics": diag, "repair": repair_info,
                "units": units_report, "base_digest": base_digest,
                "scan_quality": scan_report, "datums": datum_summary,
                "note": f"the {klass} route is not implemented in v1; the survey above is the "
                        "honest measured evidence (no approximation is labelled a reconstruction)"}
    elif plan_override is None:
        plan, meta = _plan_prismatic(survey, diag, mesh=work)
    if datum_summary is not None:
        meta["datums"] = datum_summary
    # (the dispatch above is skipped when the agent's amended plan builds)
    if partial_meta:
        meta = {**partial_meta, **meta}
    if plan is None:
        return {"supported": False, "note": meta["reason"], "survey": survey,
                "diagnostics": diag, "scan_quality": scan_report}
    # G8: nominal dimension snapping (design intent) — lengths/angles snap to the
    # coarsest in-band nominal, diameters only get a drill suggestion; the plan is then
    # built from these values and the deviation rows revalidate them against the scan
    from . import intent as _intent

    sigma_val = float((survey.get("quality") or {}).get("sigma_mm") or 0.0)
    nominal_snaps = []
    if plan_override is None:
        plan["parameters"], nominal_snaps = _intent.apply_nominals(plan["parameters"],
                                                                   sigma_val)
    if nominal_snaps:
        meta["nominal_snaps"] = nominal_snaps
        plan.setdefault("notes", []).append(
            f"nominal snapping (G8): {sum(1 for s in nominal_snaps if s['applied'])} "
            f"applied, {sum(1 for s in nominal_snaps if not s['applied'])} suggested "
            "(drill candidates, not forced)")
    flagged = [h for h in survey.get("holes", []) if h.get("dimension_basis")]
    if flagged:
        meta["synthetic_dimension_flags"] = len(flagged)   # NEVER a verified outcome with
        # synthetic-supported dimensions: demote below, and say why in the meta
    if ((repair_info or {}).get("provenance_verification") or {}).get("verdict") == "UNCERTAIN":
        meta["provenance_uncertain"] = True                # conservative: unverifiable
        # provenance can never support a VERIFIED outcome either
    if not user_unit and units_report.get("status") in units.GATE_STATUSES:
        meta["units_unconfirmed"] = units_report["status"]  # an unconfirmed mm/inch
        # interpretation can never silently support a VERIFIED outcome either
    if partial_meta:
        meta["partial_evidence_outcome_cap"] = True   # H4: missing scan surface can
        # never support a VERIFIED outcome
    if scan_report is not None and scan_report["readiness"]["verdict"] == "degraded":
        meta["scan_degraded"] = scan_report["readiness"]["reasons"]   # G5: a scan
        # defect (doubled surfaces / missing regions / unscanned hole walls) can never
        # silently support a VERIFIED outcome either
    if preview:
        # P2.3: the plan WITHOUT the build — schema-validated, dimensioned,
        # nothing created; the agent reads it, amends it and passes it back as
        # plan_override (or lets the engine build its own)
        from ..plan import schema as _schema

        ok, errors, warnings = _schema.validate(plan)
        return {"supported": True, "preview": True, "plan": plan, "meta": meta,
                "validation": {"ok": ok, "errors": errors, "warnings": warnings},
                "survey": survey, "diagnostics": diag, "units": units_report,
                "scan_quality": scan_report,
                "classification": survey["classification"],
                "nominal_snaps": nominal_snaps}
    report = builder.apply_plan(doc, plan)
    if not report.get("success"):
        # P1.1 recovery: a failed DRESSING never discards the successful base
        # build — retry without the failed op, then without any dressing, and
        # record what was dropped (the deviation rows carry the honest cost)
        failed = next((f for f in plan["features"]
                       if f.get("id") == report.get("feature_id")
                       and f.get("op") in ("fillet", "chamfer")), None)
        if failed is not None:
            dress_ids = {f["id"] for f in plan["features"]
                         if f.get("op") in ("fillet", "chamfer")}
            for drop in ({failed["id"]}, dress_ids):
                feats2 = [f for f in plan["features"] if f["id"] not in drop]
                blob = json.dumps(feats2)
                plan2 = {**plan, "features": feats2,
                         "parameters": {k: val for k, val in plan["parameters"].items()
                                        if f'"{k}"' in blob}}
                retry = builder.apply_plan(doc, plan2)
                if retry.get("success"):
                    meta["dressings_dropped"] = {
                        "features": sorted(dress_ids & drop),
                        "failed": failed["id"],
                        "reason": report.get("error") or "dressing build failure"}
                    plan, report = plan2, retry
                    break
    ledger = (repair_info or {}).get("provenance") or provenance.new_ledger()
    if nominal_snaps:
        ledger.setdefault("nominal_snaps", []).extend(nominal_snaps)   # G8: every
        # measured→nominal decision is provenance (band-protected, Δ and σ-ratio kept)
    conversion = None
    if converted_from is not None:
        conversion = units.record_conversion(ledger, units.INCH_MM, converted_from, base)
        conversion["validation"] = units.conversion_validation(converted_from, base,
                                                                units.INCH_MM)
    result = {"supported": True, "plan": plan, "meta": meta, "build": report,
              "survey": survey, "diagnostics": diag, "repair": repair_info,
              "classification": survey["classification"],
              "provenance": ledger,
              "provenance_verification": provenance.verify_against_mesh(ledger, work),
              "units": units_report, "unit_conversion": conversion,
              "base_digest": base_digest, "scan_quality": scan_report,
              "nominal_snaps": nominal_snaps}
    if not report.get("success"):
        result["outcome"] = "reconstruction_failed"
        return result
    from . import deviation

    body = doc.getObject(report["body"])
    measured = deviation.registered_deviation(work, body.Shape, a="scan", b=body.Name,
                                              warn_mm=warn_mm, mask=measured_mask)
    rows = measured["raw"]
    result["deviation"] = rows                       # RAW stays authoritative for the outcome
    result["deviation_aligned"] = measured["aligned"]
    result["registration"] = measured["registration"]
    result["measured_mask"] = measured.get("mask")   # G6: what the rows do NOT cover
    hausdorff = next((r["value"] for r in rows if r["id"] == "hausdorff"), None)
    ok = hausdorff is not None and hausdorff <= warn_mm
    # V-1: exact worst-point refinement + signed missing/excess clusters (additive rows;
    # budget-guarded inside, masked to measured regions, never changes the outcome math)
    residual = None
    try:
        residual = deviation.exact_residuals(work, body.Shape, a="scan", b=body.Name,
                                             warn_mm=warn_mm, mask=measured_mask)
        result["deviation"] = rows + residual["rows"]
    except Exception:  # noqa: BLE001 - refinement is additive honesty, never a blocker
        residual = None
    result["residual_analysis"] = residual
    # D-1: the exact boolean IoU row — additive EVIDENCE beside the sampled rows (which stay
    # authoritative for the outcome); OPT-IN (OSCAR_RECON_IOU=1): on noisy scans the sew itself
    # can cost minutes (measured 418 s on the 2k-facet noisy bracket), which no laptop run
    # should pay by default. Honest note whenever it is not computed.
    result["exact_iou"] = {"kind": "exact_iou", "iou": None,
                           "note": "not computed (opt-in: OSCAR_RECON_IOU=1)"}
    if os.environ.get("OSCAR_RECON_IOU", "") == "1":
        try:
            result["exact_iou"] = deviation.exact_iou(work, body.Shape, a="scan", b=body.Name)
        except Exception:  # noqa: BLE001 - the metric is evidence, never a blocker
            result["exact_iou"] = {"kind": "exact_iou", "iou": None, "note": "not computed"}
    result["outcome"] = ("verified_parametric" if ok and "extra_geometry" not in meta
                                                  and "synthetic_dimension_flags" not in meta
                                                  and "provenance_uncertain" not in meta
                                                  and "units_unconfirmed" not in meta
                                                  and "scan_degraded" not in meta
                                                  and "partial_evidence_outcome_cap"
                                                  not in meta
                         else "partially_verified_parametric" if ok
                         else "reconstruction_unverified")
    # P5.1 coverage vector: what share of the scan the plan explains natively
    # (recognized surfaces the route represents as native features) vs what it
    # reports as unrecognized — never overlapping quantities, straight from
    # the survey's area shares
    _unrec = float(((survey.get("coverage") or {}).get("unrecognized")) or 0.0)
    meta["coverage_vector"] = {"recognized_native": round(1.0 - _unrec, 3),
                               "unrecognized": round(_unrec, 3),
                               "faceted_fallback": 0}
    if meta.get("skipped_bands"):
        meta["coverage_vector"]["skipped_level_bands"] = len(meta["skipped_bands"])
    # P5.2 evidence-vs-tree reconciliation: recognized features vs what the
    # feature tree actually realizes (patterns legitimately represent N
    # recognized holes with 1 original + occurrences; dressings group by rule)
    _feats = plan.get("features") or []
    _hole_ops = [f for f in _feats if f.get("op") == "hole"]
    _pats = meta.get("patterns") or []
    _realized_holes = len(_hole_ops) + sum(int(m.get("count", 1)) - 1
                                           for m in _pats if m.get("kind") == "polar") \
        + sum(int(m.get("count", 1)) - 1 for m in _pats if m.get("kind") == "linear")
    _recognized_holes = int(meta.get("holes") or 0)
    _dress_ops = sum(1 for f in _feats if f.get("op") in ("fillet", "chamfer"))
    meta["reconciliation"] = {
        "holes_recognized": _recognized_holes,
        "holes_native": _realized_holes,
        "dress_evidence_strips": int(meta.get("fillets") or 0) + int(meta.get("chamfers") or 0),
        "dress_ops": _dress_ops,
        "patterns": len(_pats),
        "unexplained": []}
    if _recognized_holes > _realized_holes:
        meta["reconciliation"]["unexplained"].append(
            f"{_recognized_holes - _realized_holes} recognized hole(s) not realized "
            "in the tree")
    return result
