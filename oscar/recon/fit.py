"""Primitive fitting for the reconstruction engine (G2): sphere, cone, torus with seeded
nonlinear refinement, plus an LM polish for the cylinder (removes the Kasa rim bias).

Seeds are closed-form and cheap (Kåsa algebraic fits, normal-covariance axes, total-least-
squares profile lines); refinement is ``scipy.optimize.least_squares`` on the ORTHOGONAL
geometric residual of each primitive (the distance the fitted surface is from each sample
point), so rms values are true millimetres. The solver runs on a capped subsample
(≤ MAX_FIT_POINTS); the FINAL rms is always computed on every region point.

Anti-fake-primitive gates (``classify``): a cone must beat BOTH nested simpler models
(cylinder, plane) by an extra-sum-of-squares F-test at alpha=1e-4 with the EFFECTIVE sample
count (distinct rounded vertices — tessellations repeat each vertex ~6 times) AND by
geometric separation >= 2*sigma (the simpler model's residual evaluated at the cone's
inliers must span at least 2σ beyond noise); a torus must beat the sphere and the cylinder
the same way. This stops tessellation ripple and shallow cones from reading as new kinds.

NumPy + SciPy only (both ship inside FreeCAD 1.1.3/1.1.4); no FreeCAD import, thread-safe.
"""

import math

import numpy as np

try:  # SciPy ships inside the supported FreeCAD builds; degrade to seeds if absent
    from scipy.optimize import least_squares
except Exception:  # noqa: BLE001
    least_squares = None

MAX_FIT_POINTS = 4000          # solver subsample cap; final rms uses every point
F_ALPHA = 1e-4                 # extra-sum-of-squares gate (OriginShape-verified practice)
SEPARATION_SIGMA = 2.0         # geometric separation gate, in noise sigmas
# a fitted radius far beyond the region's OWN extent describes flatness, not
# curvature: a tangent-connected shell (a plate filleted on every edge, one big
# blob) impostor-fits a HUGE sphere/cylinder at a deceptively low rms. A genuine
# curved region (a ball, a dome cap, a hole wall) never exceeds its own extent.
_EXTENT_FACTOR = 1.5
KINDS = ("plane", "cylinder", "cone", "sphere", "torus")
_MIN_HALF_ANGLE = math.radians(0.05)
_MAX_HALF_ANGLE = math.radians(89.95)


def _subsample(P, cap=MAX_FIT_POINTS):
    n = len(P)
    if n <= cap:
        return P
    return P[np.linspace(0, n - 1, cap, dtype=int)]


def _effective_n(P) -> int:
    """Distinct sample positions (tessellations repeat each vertex ~6×)."""
    return max(1, len({tuple(np.round(p, 4)) for p in np.asarray(P, dtype=float)}))


def _frame(axis):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(axis, u)
    return axis, u, v


def _profile(P, axis):
    """Cylindrical coordinates (rho, h) of P about ``axis`` through the centroid."""
    _, u, v = _frame(axis)
    c = P.mean(axis=0)
    d = P - c
    rho = np.hypot(d @ u, d @ v)
    h = d @ axis
    return rho, h, c


# --------------------------------------------------------------------------- plane / cylinder
def fit_plane(P):
    """PCA plane; returns (unit normal, centroid, orthogonal rms)."""
    P = np.asarray(P, dtype=float)
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    rms = float(np.sqrt(((P - c) @ n) ** 2).mean())
    return n, c, rms


def fit_cylinder(P, N=None, refine=True):
    """Axis from the normal covariance (wall normals satisfy n·axis = 0 even on PARTIAL
    arcs), centre+radius by Kåsa in the world frame, then — when SciPy is available —
    least-squares refinement over [tilt×2, axis offset×2, radius] on the radial residual,
    which removes the Kåsa rim bias on partial walls."""
    P = np.asarray(P, dtype=float)
    if len(P) < 6:
        return None
    axis = None
    if N is not None and len(N) == len(P):
        nc = np.asarray(N, dtype=float)
        nc = nc - nc.mean(axis=0)
        _, vals, vecs = np.linalg.svd(nc, full_matrices=False)
        if abs(vals[-1]) > 1e-9 and vals[-1] > 0.25 * vals[0]:
            return None                                  # normals not planar: not a cylinder
        axis = vecs[-1]
    if axis is None:
        _, _, vt = np.linalg.svd(P - P.mean(axis=0), full_matrices=False)
        axis = vt[-1]
    axis, u, v = _frame(axis)
    x, y = P @ u, P @ v
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    try:
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c = sol
    r2 = c + a * a + b * b
    if r2 <= 0:
        return None
    r = math.sqrt(r2)

    def result(axis_, center_uv, r_):
        _, uu, vv = _frame(axis_)
        px, py = P @ uu, P @ vv
        rms = float(np.sqrt((np.hypot(px - center_uv[0], py - center_uv[1]) - r_) ** 2).mean())
        ang = np.degrees(np.arctan2(py - center_uv[1], px - center_uv[0]))
        return {"axis": axis_, "point": np.asarray(center_uv[0] * uu + center_uv[1] * vv,
                                                   dtype=float),
                "radius": float(r_), "rms": rms,
                "turn_deg": float(np.ptp(np.sort(ang))),
                "axial_span": float(((P - P.mean(axis=0)) @ axis_).ptp())}

    if refine and least_squares is not None:
        c0 = P.mean(axis=0)
        cu, cv = a - float(c0 @ u), b - float(c0 @ v)   # Kasa centre, centroid-relative

        def resid(x):
            ax = axis + x[0] * u + x[1] * v
            nrm = np.linalg.norm(ax) + 1e-12
            ax = ax / nrm
            _, uu, vv = _frame(ax)
            base = c0 + x[2] * u + x[3] * v
            d = P - base
            return np.hypot(d @ uu, d @ vv) - x[4]

        try:
            sol = least_squares(resid, [0.0, 0.0, cu, cv, r], method="trf",
                                xtol=1e-12, ftol=1e-12, max_nfev=400)
            ta, tb, ou, ov, rr = sol.x
            if rr > 0:
                ax = axis + ta * u + tb * v
                ax /= np.linalg.norm(ax)
                _, uu, vv = _frame(ax)
                base = c0 + ou * u + ov * v
                # world-frame circle centre about the FINAL axis (a genuine axis point)
                px, py = P @ uu, P @ vv
                Aw = np.column_stack([2 * px, 2 * py, np.ones_like(px)])
                sw, *_ = np.linalg.lstsq(Aw, px * px + py * py, rcond=None)
                return result(ax, (float(sw[0]), float(sw[1])), rr)
        except Exception:  # noqa: BLE001 - fall back to the Kasa result
            pass
    return result(axis, (a, b), r)


# --------------------------------------------------------------------------- sphere
def fit_sphere(P):
    """Kåsa algebraic seed → least_squares on the radial residual ‖p−c‖ − r (analytic
    Jacobian). Returns {center, radius, rms} or None."""
    P = np.asarray(P, dtype=float)
    if len(P) < 4:
        return None
    S = np.column_stack([2 * P, np.ones(len(P))])
    try:
        sol, *_ = np.linalg.lstsq(S, (P * P).sum(axis=1), rcond=None)
    except np.linalg.LinAlgError:
        return None
    c0, r20 = sol[:3], sol[3] + sol[0] ** 2 + sol[1] ** 2 + sol[2] ** 2
    # degenerate cloud (collinear/coincident): every sphere is meaningless
    extent = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
    centred = P - P.mean(axis=0)
    if extent <= 0 or float(np.linalg.svd(centred, compute_uv=False)[-1]) < 1e-6 * extent:
        return None
    if r20 <= 0:
        # Kasa can go negative on partial caps under noise: centroid fallback seed
        c0 = P.mean(axis=0)
        r0 = float(np.linalg.norm(P - c0, axis=1).mean())
    else:
        r0 = math.sqrt(r20)
    Q = _subsample(P)

    def resid(x):
        return np.linalg.norm(Q - x[:3], axis=1) - x[3]

    def jac(x):
        d = Q - x[:3]
        dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        J = np.concatenate([-d / dist, -np.ones((len(Q), 1))], axis=1)
        return J

    c, r = c0, r0
    if least_squares is not None:
        try:
            sol = least_squares(resid, np.concatenate([c0, [r0]]), jac=jac, method="trf",
                                xtol=1e-12, ftol=1e-12, max_nfev=200)
            c, r = sol.x[:3], float(sol.x[3])
        except Exception:  # noqa: BLE001
            pass
    extent = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
    if r <= 0 or r > 50.0 * extent or r < 1e-9 * extent:
        return None                       # degenerate: collinear/coincident inputs
    rms = float(np.sqrt((np.linalg.norm(P - c, axis=1) - r) ** 2).mean())
    return {"center": np.asarray(c, dtype=float), "radius": float(r), "rms": rms}


# --------------------------------------------------------------------------- cone
def fit_cone(P, N=None):
    """Axis seed from the normal covariance, apex/half-angle from a total-least-squares
    profile line in (rho, h), then least_squares over [tilt×2, offset×2, half_angle, r0]
    on the Lukács residual rho·cosβ − h·sinβ − r0·cosβ. Returns cone params or None."""
    P = np.asarray(P, dtype=float)
    if len(P) < 8:
        return None
    _, _, vt = np.linalg.svd(P - P.mean(axis=0), full_matrices=False)
    axis = vt[-1]
    if N is not None and len(N) == len(P):
        nc = np.asarray(N, dtype=float) - np.asarray(N, dtype=float).mean(axis=0)
        _, _, vecs = np.linalg.svd(nc, full_matrices=False)
        if float(np.linalg.norm(vecs[-1])) > 0:  # normal-covariance axis (cone normals
            axis = vecs[-1]                      # spread in azimuth around the axis)
    rho, h, c0 = _profile(P, axis)
    # TLS line rho = m·h + b  →  tanβ = slope of the wall profile
    hm, rm = h.mean(), rho.mean()
    dh, dr = h - hm, rho - rm
    denom = float((dh * dh).sum())
    m = float((dh * dr).sum() / denom) if abs(denom) > 1e-12 else 0.0
    b = rm - m * hm
    beta0 = math.atan(abs(m))
    r00 = b * math.cos(beta0)
    axis, u, v = _frame(axis)
    Q = _subsample(P)

    def unpack(x):
        ax = axis + x[0] * u + x[1] * v
        ax /= np.linalg.norm(ax) + 1e-12
        beta = beta0 + x[4]
        return ax, x[2], x[3], beta, x[5]

    def resid(x):
        ax, ou, ov, beta, r0 = unpack(x)
        _, uu, vv = _frame(ax)
        base = c0 + ou * u + ov * v
        d = Q - base
        rho_q = np.hypot(d @ uu, d @ vv)
        h_q = d @ ax
        return rho_q * math.cos(beta) - h_q * math.sin(beta) - r0 * math.cos(beta)

    x0 = [0.0, 0.0, 0.0, 0.0, 0.0, r00]
    if least_squares is not None:
        try:
            sol = least_squares(resid, x0, method="trf", xtol=1e-12, ftol=1e-12,
                                max_nfev=400)
            ax, ou, ov, beta, r0 = unpack(sol.x)
        except Exception:  # noqa: BLE001
            ax, ou, ov, beta, r0 = axis, 0.0, 0.0, beta0, r00
    else:
        ax, ou, ov, beta, r0 = axis, 0.0, 0.0, beta0, r00
    if not (_MIN_HALF_ANGLE < beta < _MAX_HALF_ANGLE) or r0 < -1e-6:
        return None
    base = c0 + ou * u + ov * v
    _, uu, vv = _frame(ax)
    d = P - base
    rho_p = np.hypot(d @ uu, d @ vv)
    h_p = d @ ax
    r_all = rho_p * math.cos(beta) - h_p * math.sin(beta) - r0 * math.cos(beta)
    # the Lukács form is the first-order orthogonal distance in the profile plane
    rms = float(np.sqrt((r_all ** 2).mean()))
    apex = base + ax * (r0 / math.sin(beta)) if beta > 1e-6 else base
    return {"axis": ax, "apex": np.asarray(apex, dtype=float), "half_angle": beta,
            "rms": rms,
            "radius_min": float(rho_p.min()), "radius_max": float(rho_p.max())}


# --------------------------------------------------------------------------- torus
def fit_torus(P, N=None):
    """Axis seed from the normal covariance, profile-circle Kåsa seed in (rho, h), then
    least_squares over [tilt×2, offset×2, R, r] on sqrt((rho−R)² + h²) − r."""
    P = np.asarray(P, dtype=float)
    if len(P) < 10:
        return None
    if N is not None and len(N) == len(P):
        nc = np.asarray(N, dtype=float) - np.asarray(N, dtype=float).mean(axis=0)
        _, _, vecs = np.linalg.svd(nc, full_matrices=False)
        axis = vecs[-1]
    else:
        _, _, vt = np.linalg.svd(P - P.mean(axis=0), full_matrices=False)
        axis = vt[-1]
    rho, h, c0 = _profile(P, axis)
    A = np.column_stack([2 * rho, 2 * h, np.ones_like(rho)])
    try:
        sol, *_ = np.linalg.lstsq(A, rho * rho + h * h, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, cc = sol
    r2 = cc + a * a + b * b
    if r2 <= 1e-12:
        return None
    R0, r0 = math.sqrt(a * a + b * b), math.sqrt(r2)
    if R0 <= 1e-9 or r0 <= 1e-9:
        return None
    # a LINE profile (a cylinder wall) is not a torus: when the total-least-squares line
    # through (rho, h) explains the profile as well as the circle, reject
    hm2, rm2 = h.mean(), rho.mean()
    dh2, dr2 = h - hm2, rho - rm2
    line_rms = float(np.sqrt((dr2 - (dh2 * dr2).sum() / max((dh2 * dh2).sum(), 1e-12)
                             * dh2) ** 2).mean())
    circ_rms = float(np.sqrt((np.hypot(rho - a, h - b) - r0) ** 2).mean())
    if line_rms <= max(circ_rms * 1.05, 1e-4):
        return None                 # a real tube profile curves far more than 1e-4 mm
    axis, u, v = _frame(axis)
    Q = _subsample(P)

    def unpack(x):
        ax = axis + x[0] * u + x[1] * v
        ax /= np.linalg.norm(ax) + 1e-12
        return ax, x[2], x[3], x[4], x[5]

    def resid(x):
        ax, ou, ov, R, r = unpack(x)
        _, uu, vv = _frame(ax)
        base = c0 + ou * u + ov * v
        d = Q - base
        rho_q = np.hypot(d @ uu, d @ vv)
        h_q = d @ ax
        return np.sqrt((rho_q - R) ** 2 + h_q * h_q) - r

    if least_squares is not None:
        try:
            sol = least_squares(resid, [0.0, 0.0, 0.0, 0.0, R0, r0], method="trf",
                                xtol=1e-12, ftol=1e-12, max_nfev=400)
            ax, ou, ov, R, r = unpack(sol.x)
        except Exception:  # noqa: BLE001
            ax, ou, ov, R, r = axis, 0.0, 0.0, R0, r0
    else:
        ax, ou, ov, R, r = axis, 0.0, 0.0, R0, r0
    if R <= 1e-9 or r <= 1e-9 or r >= R * 1.5:
        return None
    base = c0 + ou * u + ov * v
    _, uu, vv = _frame(ax)
    d = P - base
    rho_p = np.hypot(d @ uu, d @ vv)
    h_p = d @ ax
    rms = float(np.sqrt((np.sqrt((rho_p - R) ** 2 + h_p * h_p) - r) ** 2).mean())
    ang = np.degrees(np.arctan2(rho_p - R, h_p))
    return {"axis": ax, "center": np.asarray(base, dtype=float),
            "major_radius": float(R), "minor_radius": float(r), "rms": rms,
            "tube_span_deg": float(np.ptp(np.sort(ang)))}


# --------------------------------------------------------------------------- gates + classify
def _rss(fit, P):
    return (fit["rms"] ** 2) * len(P) if fit else None


def cone_significance(P, cone, cylinder, plane_fit, sigma):
    """Accept a cone only when it beats BOTH nested simpler models: extra-sum-of-squares
    F-test at F_ALPHA with the effective sample count, geometric separation >= 2σ, and a
    strictly smaller rms. Returns (ok, reason)."""
    if cone is None:
        return False, "no cone fit"
    n = _effective_n(P)
    rss_c = cone["rms"] ** 2 * n
    for name, simpler, dof_s in (("cylinder", cylinder, 5), ("plane", plane_fit, 3)):
        if simpler is None:
            continue
        rss_s = simpler["rms"] ** 2 * n
        q = 6 - dof_s
        dof = max(n - 6, 1)
        if rss_c >= rss_s * 0.999:
            return False, f"cone no better than {name}"
        try:
            from scipy.stats import f as f_dist

            f_stat = ((rss_s - rss_c) / q) / (rss_c / dof)
            f_crit = f_dist.isf(F_ALPHA, q, dof)
            if f_stat <= f_crit:
                return False, f"F-test vs {name} ({f_stat:.2f} <= {f_crit:.2f})"
        except Exception:  # noqa: BLE001 - no scipy.stats: rely on separation only
            pass
    best_simpler = min((f["rms"] for f in (cylinder, plane_fit) if f), default=None)
    if best_simpler is not None and sigma and sigma > 0 \
            and best_simpler < cone["rms"] + SEPARATION_SIGMA * sigma:
        return False, f"separation < {SEPARATION_SIGMA:.0f}σ"
    return True, ""


def torus_significance(P, torus, sphere, cylinder, sigma):
    """Accept a torus only when it beats the sphere and the cylinder the same way."""
    if torus is None:
        return False, "no torus fit"
    n = _effective_n(P)
    rss_t = torus["rms"] ** 2 * n
    for name, simpler, dof_s in (("sphere", sphere, 4), ("cylinder", cylinder, 5)):
        if simpler is None:
            continue
        rss_s = simpler["rms"] ** 2 * n
        q = 7 - dof_s
        dof = max(n - 7, 1)
        if rss_t >= rss_s * 0.999:
            return False, f"torus no better than {name}"
        try:
            from scipy.stats import f as f_dist

            f_stat = ((rss_s - rss_t) / q) / (rss_t / dof)
            f_crit = f_dist.isf(F_ALPHA, q, dof)
            if f_stat <= f_crit:
                return False, f"F-test vs {name} ({f_stat:.2f} <= {f_crit:.2f})"
        except Exception:  # noqa: BLE001
            pass
    best_simpler = min((f["rms"] for f in (sphere, cylinder) if f), default=None)
    if best_simpler is not None and sigma and sigma > 0 \
            and best_simpler < torus["rms"] + SEPARATION_SIGMA * sigma:
        return False, f"separation < {SEPARATION_SIGMA:.0f}σ"
    return True, ""


def classify(P, N, tol, sigma, kinds=KINDS, plane_fit=None, cylinder_fit=None):
    """Fit the requested primitive kinds and choose the SIMPLEST acceptable one (Occam
    order plane < cylinder < cone < sphere < torus; a harder kind replaces a simpler one
    only through its significance gate). ``tol`` is the σ-derived rms acceptance gate.
    Returns {"kind", "fit", "report"} where report carries per-kind rms for evidence."""
    P = np.asarray(P, dtype=float)
    extent = float(np.linalg.norm(np.ptp(P, axis=0))) if len(P) else 0.0
    report = {}
    plane = plane_fit if plane_fit is not None else fit_plane(P)
    if plane is not None and len(plane) == 3:
        plane = {"normal": plane[0], "point": plane[1], "rms": plane[2]}
    report["plane"] = plane["rms"] if plane else None
    if plane and plane["rms"] <= tol and "plane" in kinds:
        # M2 curvature-significance override: a COARSELY tessellated curved wall fits a
        # plane within the gate (the case-13 recognition failure — the wall's sagitta at
        # coarse resolution is smaller than the noise gate), so plane-by-Occam would eat
        # every hole. When the normals SPREAD (a real plane's normals are parallel) and a
        # cylinder fits decisively better (rms <= 60% of the plane rms, inside the same
        # gate, passing the impostor checks below), the cylinder wins on evidence.
        if N is not None and len(N) == len(P) and "cylinder" in kinds and len(P) >= 6:
            n_hat = np.asarray(plane["normal"], dtype=float)
            n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-30)
            normals = np.asarray(N, dtype=float)
            spread = float(np.degrees(np.arccos(np.clip(np.abs(normals @ n_hat), -1.0, 1.0))).max())
            if spread > 10.0:
                # bounded probe: a ≤400-point subsample with the UNREFINED Kåsa fit decides
                # significance; only an accepted hole wall pays for the full refined fit
                if len(P) > 400:
                    sel = np.linspace(0, len(P) - 1, 400).astype(int)
                    P_t, N_t = P[sel], np.asarray(N, dtype=float)[sel]
                else:
                    P_t, N_t = P, np.asarray(N, dtype=float)
                cyl_try = fit_cylinder(P_t, N_t, refine=False)
                if cyl_try and cyl_try["rms"] <= tol and cyl_try["rms"] <= 0.6 * plane["rms"]:
                    extent = float(np.linalg.norm(np.ptp(P, axis=0))) if len(P) else 0.0
                    if cyl_try["radius"] <= _EXTENT_FACTOR * extent:
                        full = cylinder_fit if cylinder_fit is not None else fit_cylinder(P, N)
                        if full and full["rms"] <= tol:
                            report["plane_override"] = (
                                f"normals span {spread:.0f} deg; cylinder rms "
                                f"{full['rms']:.4g} << plane rms {plane['rms']:.4g}")
                            return {"kind": "cylinder", "fit": full, "report": report}
        return {"kind": "plane", "fit": plane, "report": report}
    cyl = cylinder_fit
    if cyl is None and "cylinder" in kinds:
        cyl = fit_cylinder(P, N)
    report["cylinder"] = cyl["rms"] if cyl else None
    if cyl and cyl["rms"] <= tol:
        reason = None
        if cyl["radius"] > _EXTENT_FACTOR * extent:
            reason = (f"radius {cyl['radius']:.3g} exceeds the region's own extent "
                      "(a flat impostor)")
        elif N is not None and len(N) == len(P):
            ax = np.asarray(cyl["axis"], dtype=float)
            axial = float(np.median(np.abs(np.asarray(N, dtype=float) @ ax)))
            if axial > math.sin(math.radians(15.0)):
                # a real cylinder's facet normals are perpendicular to the axis
                reason = (f"normals not perpendicular to the axis (median |n.a| = "
                          f"{axial:.2f}; a shell impostor)")
        if reason is not None:
            report["cylinder_gate"] = reason
        else:
            return {"kind": "cylinder", "fit": cyl, "report": report}
    if "cone" in kinds:
        cone = fit_cone(P, N)
        report["cone"] = cone["rms"] if cone else None
        if cone and cone["rms"] <= tol:
            if cone["radius_max"] > _EXTENT_FACTOR * extent:
                report["cone_gate"] = (f"radius_max {cone['radius_max']:.3g} exceeds "
                                       "the region's own extent (a flat impostor)")
            else:
                ok, why = cone_significance(P, cone, cyl, plane, sigma)
                report["cone_gate"] = why or "accepted"
                if ok:
                    return {"kind": "cone", "fit": cone, "report": report}
    if "sphere" in kinds:
        sph = fit_sphere(P)
        report["sphere"] = sph["rms"] if sph else None
        if sph and sph["rms"] <= tol:
            reason = None
            if sph["radius"] > _EXTENT_FACTOR * extent:
                reason = (f"radius {sph['radius']:.3g} exceeds the region's own "
                          "extent (a flat impostor)")
            elif N is not None and len(N) == len(P):
                rad = P - np.asarray(sph["center"], dtype=float)
                rad_len = np.linalg.norm(rad, axis=1)
                keep = rad_len > 1e-9
                rad_unit = rad[keep] / rad_len[keep, None]
                consist = float(np.median(np.abs(np.einsum(
                    "ij,ij->i", rad_unit, np.asarray(N[keep], dtype=float)))))
                if consist < math.cos(math.radians(25.0)):
                    # a real sphere's facet normals point radially; a closed
                    # tangent-connected shell (a plate filleted on every edge)
                    # fits a deceptively good sphere with axis-aligned normals
                    reason = (f"normals not radial (median |n.r| = {consist:.2f}; "
                              "a shell impostor)")
            if reason is not None:
                report["sphere_gate"] = reason
            else:
                return {"kind": "sphere", "fit": sph, "report": report}
    if "torus" in kinds:
        tor = fit_torus(P, N)
        report["torus"] = tor["rms"] if tor else None
        if tor and tor["rms"] <= tol:
            reason = None
            if tor["major_radius"] > _EXTENT_FACTOR * extent:
                reason = (f"major radius {tor['major_radius']:.3g} exceeds the "
                          "region's own extent (a flat impostor)")
            elif N is not None and len(N) == len(P):
                ax = np.asarray(tor["axis"], dtype=float)
                rel = P - np.asarray(tor["center"], dtype=float)
                ring = rel - np.outer(rel @ ax, ax)          # in the ring plane
                ring_len = np.linalg.norm(ring, axis=1)
                keep = ring_len > 1e-9
                tube = (P[keep]
                        - (np.asarray(tor["center"], dtype=float)
                           + (ring[keep] / ring_len[keep, None])
                           * float(tor["major_radius"])))
                tube_len = np.linalg.norm(tube, axis=1)
                ok2 = tube_len > 1e-9
                consist = float(np.median(np.abs(np.einsum(
                    "ij,ij->i", tube[ok2] / tube_len[ok2, None],
                    np.asarray(N[keep], dtype=float)[ok2]))))
                if consist < math.cos(math.radians(25.0)):
                    # a real torus's facet normals point away from the TUBE
                    # centre; a closed shell impostor does not
                    reason = (f"normals not tube-radial (median |n.r| = "
                              f"{consist:.2f}; a shell impostor)")
            if reason is not None:
                report["torus_gate"] = reason
            else:
                sph_for_gate = sph if "sphere" in kinds else fit_sphere(P)
                ok, why = torus_significance(P, tor, sph_for_gate, cyl, sigma)
                report["torus_gate"] = why or "accepted"
                if ok:
                    return {"kind": "torus", "fit": tor, "report": report}
    return {"kind": None, "fit": None, "report": report}
