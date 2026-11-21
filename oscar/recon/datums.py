"""Datum and symmetry PROPOSALS from the recognized geometry (G10).

A 3-2-1 frame proposal: primary = the largest planar region's fitted plane with its
normal flipped AWAY from the material (the centroid test); secondary = the dominant
cylinder axis when a rotational part is present, else the longest in-plane direction
of the mesh projected onto the primary plane (farthest hull pair — the rotating-
calipers outcome that matters); tertiary = the largest hole axis intersected with the
primary plane, else the projected hull centroid. Everything carries residuals and the
assumptions are stated.

Symmetry (cheap versions, honest scores): rotational candidates are tested by
re-occupancy — rotate sampled points about the candidate axis by 360/n (n = 2..8) and
measure the nearest-neighbour RMS against the original cloud; a continuous axis (a
turned part) is reported when the radius spread is tight. Mirror candidates are the
principal-frame planes, scored by reflection NN RMS. AMBIGUITY is flagged, never
resolved silently (an equiangular cube has three equally valid primaries).

Pure numpy on the mesh + the survey (quality kernel for NN); a PROPOSAL — this module
never creates document objects and never moves geometry.
"""

import math

import numpy as np

from .quality import _knn

ANGLE_FLAG_DEG = 0.05     # the acceptance residual: a plate reads XY within 0.05 deg
SYMMETRY_SAMPLE = 500     # points scored for the rotational/mirror tests
SYMMETRY_RMS_MM = 0.10    # NN RMS below this (scaled by size) counts as matching


def _points(mesh):
    return np.array([[p.x, p.y, p.z] for p in mesh.Topology[0]], dtype=float)


def _nn_rms(query, cloud):
    if not len(query) or not len(cloud):
        return math.inf
    d, _, _ = _knn(np.asarray(query, dtype=float), np.asarray(cloud, dtype=float), 1,
                   exclude_self=False)
    d = np.asarray(d, dtype=float).reshape(-1)
    return float(np.sqrt((d ** 2).mean()))


def _hull_2d(P):
    """Convex hull of 2-D points (monotone chain)."""
    P = np.unique(np.round(P, 6), axis=0)
    P = P[np.lexsort((P[:, 1], P[:, 0]))]
    if len(P) < 3:
        return P

    def half(pts):
        out = []
        for p in pts:
            while len(out) >= 2 and np.cross(out[-1] - out[-2], p - out[-2]) <= 1e-12:
                out.pop()
            out.append(p)
        return out

    lower = half(P)
    upper = half(P[::-1])
    return np.array(lower[:-1] + upper[:-1])


def propose_datums(mesh, survey=None) -> dict:
    """The 3-2-1 datum proposal + symmetry scores for ``mesh`` (read-only)."""
    pts = _points(mesh)
    if len(pts) < 4:
        return {"error": "too few points to propose datums"}
    centroid = pts.mean(axis=0)
    planes = sorted((r for r in (survey or {}).get("regions", [])
                     if r.get("type") == "plane"),
                    key=lambda r: -r["area_mm2"])
    out = {"ambiguities": [], "symmetry": {}}

    # primary: the largest plane, normal flipped away from the material
    primary = None
    if planes:
        p0 = planes[0]
        n = np.asarray(p0["normal"], dtype=float)
        pt = np.asarray(p0["point"], dtype=float)
        if float(n @ (centroid - pt)) > 0:
            n = -n                                   # outward = away from the material
        primary = {"normal": n.tolist(), "point": pt.tolist(),
                   "area_mm2": p0["area_mm2"], "residual_rms_mm": p0.get("rms_mm"),
                   "basis": "largest planar region (fit normal flipped away from the "
                            "material centroid)"}
        if len(planes) > 1 and planes[1]["area_mm2"] > 0.9 * p0["area_mm2"]:
            primary["ambiguous_with"] = planes[1]["area_mm2"]
            out["ambiguities"].append(
                "primary plane: a second face of nearly equal area exists — the 3-2-1 "
                "choice between them is a proposal, not a measurement")
    out["primary_plane"] = primary
    n = np.asarray(primary["normal"], dtype=float) if primary else \
        np.array([0.0, 0.0, 1.0])

    # secondary: the dominant cylinder axis (a rotational part) else the longest
    # in-plane direction of the projection onto the primary plane (farthest hull pair)
    cylinders = sorted((r for r in (survey or {}).get("regions", [])
                        if r.get("type") == "cylinder" and r.get("turn_deg", 0) > 90),
                       key=lambda r: -r["area_mm2"])
    if cylinders and cylinders[0]["area_mm2"] > 0.25 * float(mesh.Area):
        axis = np.asarray(cylinders[0]["axis"], dtype=float)
        out["secondary_axis"] = {"direction": axis.tolist(),
                                 "basis": "dominant cylinder axis",
                                 "perpendicularity_deg": round(math.degrees(math.asin(
                                     min(1.0, abs(float(axis @ n))))), 4)}
        rot_axis = axis
    else:
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, ref)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        proj = np.column_stack([pts @ u, pts @ v])
        hull = _hull_2d(proj)
        direction = np.array([1.0, 0.0])
        if len(hull) >= 2:
            d2 = ((hull[:, None, :] - hull[None, :, :]) ** 2).sum(2)
            i, j = np.unravel_index(np.argmax(d2), d2.shape)
            direction = hull[j] - hull[i]
            length = float(np.linalg.norm(direction))
            direction = direction / max(length, 1e-12)
        axis_world = direction[0] * u + direction[1] * v
        out["secondary_axis"] = {"direction": axis_world.tolist(),
                                 "basis": "longest in-plane direction of the projected "
                                          "hull (rotating-calipers outcome)",
                                 "perpendicularity_deg": 0.0}
        rot_axis = axis_world

    # tertiary: the largest hole axis intersected with the primary plane, else the
    # projected hull centroid
    holes = (survey or {}).get("holes") or []
    if holes:
        h0 = max(holes, key=lambda h: h.get("area_mm2", 0))
        a = np.asarray(h0["axis"], dtype=float)
        p = np.asarray(h0["center"], dtype=float)
        denom = float(a @ n)
        if abs(denom) > 1e-9:
            t = float((np.asarray(primary["point"], dtype=float) - p) @ n) / denom
            out["tertiary_point"] = {"point": (p + t * a).tolist(),
                                     "basis": "largest hole axis ∩ primary plane"}
        else:
            out["ambiguities"].append("tertiary: the largest hole axis lies in the "
                                      "primary plane — no intersection; using the "
                                      "projected centroid")
    if "tertiary_point" not in out:
        out["tertiary_point"] = {"point": centroid.tolist(),
                                 "basis": "point-cloud centroid (no usable hole axis)"}

    # alignment proposal: rotate the primary normal to +Z and the secondary into the
    # XZ plane; reported as axis+angle, NEVER applied
    z = np.array([0.0, 0.0, 1.0])
    nn = n / np.linalg.norm(n)
    cross = np.cross(nn, z)
    if float(np.linalg.norm(cross)) > 1e-12:
        angle = math.degrees(math.asin(min(1.0, float(np.linalg.norm(cross)))))
        out["alignment"] = {"axis": (cross / np.linalg.norm(cross)).tolist(),
                            "angle_deg": round(angle, 4),
                            "note": "rotation that would sit the primary plane on XY — "
                                    "a PROPOSAL; nothing is moved"}
        if angle > ANGLE_FLAG_DEG:
            out["alignment"]["residual_note"] = (f"the part sits {angle:.4f} deg off "
                                                 "the XY frame")

    # symmetry — rotational: re-occupancy about the best candidate axis under test
    # rotations; mirror: the principal-frame planes scored by reflection NN RMS
    step = max(1, len(pts) // SYMMETRY_SAMPLE)
    sample = pts[::step]
    best_rot = None
    for candidate in ({"axis": rot_axis, "basis": out["secondary_axis"]["basis"]},):
        a = np.asarray(candidate["axis"], dtype=float)
        a = a / np.linalg.norm(a)
        radii = np.linalg.norm(sample - np.outer(sample @ a, a), axis=1)
        if float(np.median(radii)) < 1e-9:
            continue
        for k in (2, 3, 4, 6, 8):
            ang = 2.0 * math.pi / k
            c = math.cos(ang)
            s = math.sin(ang)
            rotated = (sample * c + np.cross(a, sample) * s +
                       np.outer(sample @ a, a) * (1.0 - c))
            rms = _nn_rms(rotated, pts)
            if rms <= SYMMETRY_RMS_MM * max(1.0, float(np.ptp(pts, axis=0).max()) / 50.0):
                best_rot = {"axis": a.tolist(), "order": k,
                            "reoccupancy_rms_mm": round(rms, 5),
                            "basis": candidate["basis"]}
                break
    if best_rot:
        out["symmetry"]["rotational"] = best_rot
    eigvals, eigvecs = np.linalg.eigh(np.cov((pts - centroid).T))
    for idx, plane_n in enumerate(eigvecs.T):
        reflected = sample - 2.0 * np.outer((sample - centroid) @ plane_n, plane_n)
        rms = _nn_rms(reflected, pts)
        if rms <= SYMMETRY_RMS_MM * max(1.0, float(np.ptp(pts, axis=0).max()) / 50.0):
            out["symmetry"]["mirror"] = {"normal": plane_n.tolist(),
                                         "point": centroid.tolist(),
                                         "reflection_rms_mm": round(rms, 5),
                                         "basis": f"principal frame plane {idx + 1}"}
            break
    if not out["symmetry"]:
        out["symmetry"] = {"none": "no rotational or mirror symmetry scored above "
                                   "threshold"}
    return out
