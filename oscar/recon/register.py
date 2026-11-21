"""NumPy ICP registration for the reconstruction validation (mandate 2026-11-14).

Aligns the SOURCE-SCAN point set onto the MODEL point set — the returned transform maps
``p_model_frame = R @ p_scan + t`` (source→target; documented and tested). Nothing here moves a
document object: registration operates on temporary sampled point arrays only.

Two solvers, both pure numpy: point-to-point (Kabsch/SVD — the reliable baseline) and
point-to-plane (linearized least squares with a ridge fallback for ill-conditioned systems, used
only when model normals are supplied). Safeguards: input sanitation (NaN/inf/degenerate),
max-correspondence distance, residual trimming, minimum correspondence count, iteration cap,
convergence check, and strict transform validation (orthogonality, det≈+1, finiteness).
Observability: translation axes barely spanned by correspondence normals are reported as weakly
observed (the planar-blind limitation), and PCA initialization reports symmetry ambiguity.

The outcome rule of the caller: RAW deviation stays authoritative — registration never upgrades
a reconstruction status; aligned numbers are reported as a lower bound only.
"""

import math

import numpy as np

DEFAULT_MAX_ITERATIONS = 50
DEFAULT_MIN_CORRESPONDENCES = 10
DEFAULT_TOLERANCE = 1e-7          # RMS improvement below this (mm) = converged
DEFAULT_TRIM_FRACTION = 0.20      # drop the largest-residual fifth of correspondences
WEAK_AXIS_SHARE = 0.05            # normal-energy share below which a translation axis is weak


def _finite(points) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        raise ValueError("empty point set")
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must be (n, 3)")
    if not np.isfinite(arr).all():
        raise ValueError("points contain NaN or infinite values")
    return arr


def _rigid(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def _valid_rotation(R) -> bool:
    return bool(np.isfinite(R).all()
                and np.allclose(R @ R.T, np.eye(3), atol=1e-6)
                and abs(np.linalg.det(R) - 1.0) < 1e-6)


def _kabsch(source, target):
    """The rigid transform (R, t) minimizing ‖R·source + t − target‖² (Kabsch/SVD)."""
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    H = (source - cs).T @ (target - ct)
    U, _, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:            # forbid reflections
        S[2, 2] = -1.0
    R = Vt.T @ S @ U.T
    return R, ct - R @ cs


def _point_to_plane(source, target, normals):
    """One linearized point-to-plane step (Chen–Medioni): solve for (ω, t) minimizing
    Σ ((R·p + t − q)·n)² with R ≈ I + [ω]×. Ridge fallback keeps singular systems answerable."""
    A = np.hstack([np.cross(source, normals), normals])
    b = np.einsum("ij,ij->i", target - source, normals)
    G = A.T @ A
    try:
        x = np.linalg.solve(G + 1e-9 * np.eye(6), A.T @ b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    om, t = x[:3], x[3:]
    th = float(np.linalg.norm(om))
    Rd = np.eye(3)
    if th > 1e-12:
        k = om / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rd = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    return Rd, t


def _pca_init(scan, model):
    """PCA-axes initialization with 4-fold sign disambiguation; reports ambiguity when the
    candidates are indistinguishable (symmetric geometry) or the eigenvalues are degenerate."""
    def axes(p):
        c = p - p.mean(axis=0)
        _, vals, vecs = np.linalg.svd(c, full_matrices=False)
        return vals, vecs.T                      # columns = principal axes

    s_vals, s_ax = axes(scan)
    m_vals, m_ax = axes(model)
    degenerate = bool(min(s_vals[1] / max(s_vals[0], 1e-12),
                          s_vals[2] / max(s_vals[1], 1e-12)) < 1e-2
                      or min(m_vals[1] / max(m_vals[0], 1e-12),
                             m_vals[2] / max(m_vals[1], 1e-12)) < 1e-2)
    best, best_err, second = None, math.inf, math.inf
    for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        R = m_ax @ np.diag(signs + (1.0,)) @ s_ax.T
        err = float(_nn_distances(scan @ R.T, model)[0].mean())
        if err < best_err:
            best, best_err, second = R, err, best_err
        elif err < second:
            second = err
    extent = float(max(np.abs(scan).max(), np.abs(model).max())) * 2.0
    ambiguous = bool(degenerate or (second - best_err) < 0.05 * max(extent, 1e-9))
    return best, {"method": "pca", "ambiguous": ambiguous,
                  "warning": "PCA initialization ambiguous (symmetric or degenerate geometry)"
                  if ambiguous else ""}


def _nn_distances(points, targets):
    """Nearest-neighbour distances points→targets; cKDTree when available, chunked numpy else."""
    try:
        from scipy.spatial import cKDTree

        d, _ = cKDTree(np.asarray(targets, dtype=float)).query(np.asarray(points, dtype=float), k=1)
        return np.asarray(d, dtype=float), "scipy.cKDTree"
    except Exception:  # noqa: BLE001 - chunked fallback keeps the module dependency-free
        src = np.asarray(points, dtype=float)
        dst = np.asarray(targets, dtype=float)
        out = []
        for start in range(0, len(src), 512):
            block = src[start:start + 512]
            out.extend(np.sqrt(((block[:, None, :] - dst[None, :, :]) ** 2).sum(2)).min(1))
        return np.asarray(out), "numpy-chunked"


def _nn_targets(points, targets):
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(np.asarray(targets, dtype=float)).query(np.asarray(points, dtype=float), k=1)
        return idx, "scipy.cKDTree"
    except Exception:  # noqa: BLE001
        src = np.asarray(points, dtype=float)
        dst = np.asarray(targets, dtype=float)
        idx = []
        for start in range(0, len(src), 512):
            block = src[start:start + 512]
            idx.extend(np.sqrt(((block[:, None, :] - dst[None, :, :]) ** 2).sum(2)).argmin(1))
        return np.asarray(idx), "numpy-chunked"


def icp_register(scan_points, model_points, model_normals=None, *,
                 method="auto", init="identity", max_iterations=DEFAULT_MAX_ITERATIONS,
                 min_correspondences=DEFAULT_MIN_CORRESPONDENCES,
                 tolerance=DEFAULT_TOLERANCE, trim_fraction=DEFAULT_TRIM_FRACTION,
                 max_correspondence=None):
    """Rigid ICP aligning ``scan_points`` onto ``model_points`` (source→target: R·p + t).
    Returns the full report dict; ``valid`` is False whenever the result must not be used."""
    warnings: list[str] = []
    try:
        scan = _finite(scan_points)
        model = _finite(model_points)
        if len(scan) < 4 or len(model) < 4:
            raise ValueError("fewer than 4 points on a side")
        for name, pts in (("scan", scan), ("model", model)):
            centred = pts - pts.mean(axis=0)
            sv = np.linalg.svd(centred, compute_uv=False)
            if sv[1] < 1e-6 * max(sv[0], 1e-12):
                raise ValueError(f"degenerate {name} point set (collinear)")
    except ValueError as exc:
        return _report(None, None, 0, 0, math.inf, False, method_used=method, init_used=init,
                       warnings=[f"input rejected: {exc}"], weak_axes=[], valid=False,
                       engine="")
    normals = None if model_normals is None else np.asarray(model_normals, dtype=float)
    method_used = method
    if method == "auto":
        method_used = "point_to_plane" if normals is not None else "point_to_point"
    if method_used == "point_to_plane" and normals is None:
        method_used = "point_to_point"
        warnings.append("point_to_plane requested without normals; used point_to_point")

    R, t = np.eye(3), np.zeros(3)
    init_used = {"method": "identity", "ambiguous": False, "warning": ""}
    if init == "pca":
        try:
            R, init_used = _pca_init(scan, model)
        except Exception:  # noqa: BLE001 - a failed init falls back to identity, reported
            init_used = {"method": "pca→identity", "ambiguous": True,
                         "warning": "PCA initialization failed; identity used"}
    current = scan @ R.T + t
    cap = max_correspondence
    if cap is None:
        first, _ = _nn_distances(current, model)
        cap = float(max(np.median(first) * 5.0, 1e-6))
        warnings.append(f"max_correspondence auto = median×5 = {cap:.3f} mm")
    # a coarse scan may legitimately hold fewer points than the default minimum: never demand
    # more correspondences than about half the scan (but at least the 6 a rigid solve needs)
    min_correspondences = min(min_correspondences, max(6, len(scan) // 2))

    iterations, converged, residual = 0, False, math.inf
    correspondences = 0
    used_normals_span = np.zeros((3, 3))
    for iterations in range(1, max_iterations + 1):
        dist, engine = _nn_distances(current, model)
        keep = dist <= cap
        if int(keep.sum()) < min_correspondences:
            warnings.append(f"only {int(keep.sum())} correspondences under the cap; registration "
                            "not reliable")
            return _report(np.eye(3), np.zeros(3), iterations - 1, 0, math.inf, False,
                           method_used, init_used, warnings, ["x", "y", "z"], False, engine)
        idx, _ = _nn_targets(current[keep], model)
        src, tgt = current[keep], model[idx]
        if trim_fraction > 0:
            order = np.argsort(np.linalg.norm(tgt - src, axis=1))
            keep_n = max(min_correspondences, int(len(src) * (1.0 - trim_fraction)))
            chosen = order[:keep_n]
            src, tgt, idx = src[chosen], tgt[chosen], idx[chosen]
        correspondences = len(src)
        if method_used == "point_to_plane":
            nrm = normals[idx]
            Rd, td = _point_to_plane(src, tgt, nrm)
            used_normals_span += nrm.T @ nrm / max(len(nrm), 1)
        else:
            Rd, td = _kabsch(src, tgt)
        R, t = Rd @ R, Rd @ t + td
        current = scan @ R.T + t
        new_residual = float(np.sqrt((_nn_distances(current, model)[0] ** 2).mean()))
        if not np.isfinite(new_residual):
            warnings.append("non-finite residual; registration aborted")
            return _report(R, t, iterations, correspondences, math.inf, False,
                           method_used, init_used, warnings, ["x", "y", "z"], False, engine)
        if abs(residual - new_residual) < tolerance:
            residual, converged = new_residual, True
            break
        residual = new_residual

    weak_axes = []
    if method_used == "point_to_plane":
        energy = np.diag(used_normals_span)
        for ax, share in zip("xyz", energy / max(energy.max(), 1e-12)):
            if share < WEAK_AXIS_SHARE:
                weak_axes.append(ax)
        if weak_axes:
            warnings.append("translation weakly observed along " + ", ".join(weak_axes)
                            + " (correspondence normals barely span these axes — e.g. "
                              "planar faces cannot see in-plane sliding)")
    if not converged:
        warnings.append(f"no convergence within {max_iterations} iterations")
    valid = converged and _valid_rotation(R)
    if not _valid_rotation(R):
        warnings.append("invalid rotation matrix rejected (non-orthogonal, reflected or "
                        "non-finite)")
    if not valid and method == "auto" and method_used == "point_to_plane":
        # planar-blind fallback: point-to-plane cannot converge on geometry whose normals do
        # not span all axes — retry with the point-to-point baseline before giving up
        retry = icp_register(scan_points, model_points, None, method="point_to_point",
                             init=init, max_iterations=max_iterations,
                             min_correspondences=min_correspondences, tolerance=tolerance,
                             trim_fraction=trim_fraction, max_correspondence=max_correspondence)
        retry["warnings"] = (["point_to_plane did not converge; point_to_point fallback: "]
                             + warnings + retry["warnings"])
        retry["method"] = "point_to_plane→point_to_point"
        return retry
    return _report(R, t, iterations, correspondences, residual, converged, method_used,
                   init_used, warnings, weak_axes, valid, engine)


def _report(R, t, iterations, correspondences, residual, converged, method_used, init_used,
            warnings, weak_axes, valid, engine):
    if R is None:
        rotation_deg = translation_mm = 0.0
        R_out, t_out, T_out = None, None, None
    else:
        R_out = np.asarray(R, dtype=float)
        t_out = np.asarray(t, dtype=float)
        T_out = _rigid(R_out, t_out)
        rotation_deg = math.degrees(math.acos(
            max(-1.0, min(1.0, (np.trace(R_out) - 1.0) / 2.0))))
        translation_mm = float(np.linalg.norm(t_out))
    return {"valid": valid, "converged": converged, "method": method_used,
            "init": init_used, "iterations": iterations, "correspondences": correspondences,
            "residual_rmse_mm": round(residual, 6) if math.isfinite(residual) else None,
            "rotation_deg": round(rotation_deg, 4), "translation_mm": round(translation_mm, 4),
            "R": R_out.tolist() if R_out is not None else None,
            "t": t_out.tolist() if t_out is not None else None,
            "T": T_out.tolist() if T_out is not None else None,
            "weak_translation_axes": weak_axes, "warnings": warnings,
            "convention": "source->target: p_model_frame = R @ p_scan + t",
            "nn_engine": engine}
